#pragma once

#include <tilestream/cpu.h>
#include <tilestream/cuda.h>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace test {

using namespace tilestream;
using Bytes = std::vector<std::uint8_t>;

inline constexpr std::uint8_t sentinel = 0xa5;
inline constexpr std::uint64_t guard_bytes = 64;
inline constexpr TileShape shapes[] = {{128, 32}, {64, 64}, {256, 16}};

struct DeviceFree {
    void operator()(void *data) const noexcept {
        static_cast<void>(cudaFree(data));
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;

inline DevicePtr device_alloc(std::uint64_t bytes, int fill = sentinel) {
    void *data = nullptr;
    EXPECT_EQ(cudaMalloc(&data, static_cast<std::size_t>(bytes)), cudaSuccess);
    if (data != nullptr) {
        EXPECT_EQ(cudaMemset(data, fill, static_cast<std::size_t>(bytes)), cudaSuccess);
    }
    return DevicePtr(data);
}

inline DevicePtr upload(const Bytes &bytes) {
    auto device = device_alloc(bytes.size());
    EXPECT_EQ(cudaMemcpy(device.get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
              cudaSuccess);
    return device;
}

inline Bytes download(const void *data, std::uint64_t size) {
    Bytes bytes(static_cast<std::size_t>(size));
    EXPECT_EQ(cudaMemcpy(bytes.data(), data, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    return bytes;
}

inline Bytes golden(const std::string &name) {
    std::ifstream input(std::string(TILESTREAM_GOLDEN_DIR) + "/" + name + ".hex");
    EXPECT_TRUE(input.is_open()) << name;
    Bytes result;
    unsigned int value = 0;
    while (input >> std::hex >> value) {
        result.push_back(static_cast<std::uint8_t>(value));
    }
    return result;
}

// Host image with row padding filled by a sentinel that must never reach a file.
struct HostImage {
    ImageSpec image;
    std::uint64_t pitch = 0;
    Bytes bytes;

    std::uint64_t sample_bytes() const {
        return image.type == PixelType::uint16 ? 2 : 4;
    }
    ConstImageView view() const {
        return {bytes.data(), image, pitch, bytes.size()};
    }
    ImageView output_view() {
        return {bytes.data(), image, pitch, bytes.size()};
    }
    std::uint32_t get(std::uint64_t x, std::uint64_t y) const {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + y * pitch + x * sample_bytes(), sample_bytes());
        return value;
    }
    void set(std::uint64_t x, std::uint64_t y, std::uint32_t value) {
        std::memcpy(bytes.data() + y * pitch + x * sample_bytes(), &value, sample_bytes());
    }
    // Logical pixels without padding, for bitwise comparisons.
    Bytes pixels() const {
        Bytes result;
        const auto row = image.width * sample_bytes();
        for (std::uint64_t y = 0; y < image.height; ++y) {
            const auto *line = bytes.data() + y * pitch;
            result.insert(result.end(), line, line + row);
        }
        return result;
    }
};

inline HostImage make_image(PixelType type, std::uint32_t width, std::uint32_t height,
                            std::uint32_t padding = 0, std::uint8_t fill = 0x5a) {
    HostImage result;
    result.image = {type, width, height};
    result.pitch = (std::uint64_t{width} + padding) * (type == PixelType::uint16 ? 2 : 4);
    result.bytes.assign(static_cast<std::size_t>(result.pitch * height), fill);
    return result;
}

// Mixes constant, smooth, wrapping, noisy and arbitrary bit patterns.
inline HostImage pattern_image(PixelType type, std::uint32_t width, std::uint32_t height,
                               std::uint32_t padding, std::uint32_t pattern, std::mt19937 &rng) {
    auto result = make_image(type, width, height, padding);
    const auto base = static_cast<std::uint32_t>(rng());
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            std::uint32_t value = 0;
            switch (pattern % 6) {
            case 0:
                value = base;
                break;
            case 1:
                value = base + x * 3 + y * 7;
                break;
            case 2:
                value = 65535 - x - y * width;
                break;
            case 3:
                value = base + x + static_cast<std::uint32_t>(rng() % 5);
                break;
            case 4:
                value = static_cast<std::uint32_t>(rng());
                break;
            default:
                value = (x / 8 + y / 4) % 2 == 0 ? base : base ^ (1U << (rng() % 32));
                break;
            }
            if (type == PixelType::float32 && pattern % 6 == 1) {
                const auto field = 1.5F + 0.001F * static_cast<float>(x) +
                                   0.01F * static_cast<float>(y) +
                                   0.0001F * static_cast<float>(rng() % 7);
                std::memcpy(&value, &field, 4);
            }
            result.set(x, y, type == PixelType::uint16 ? value & 65535U : value);
        }
    }
    return result;
}

inline Bytes cpu_encode(const HostImage &image, TileShape tile = {}) {
    std::uint64_t bound = 0;
    EXPECT_EQ(compress_bound(image.image, tile, bound), Status::success);
    Bytes file(static_cast<std::size_t>(bound));
    std::uint64_t written = 0;
    EXPECT_EQ(cpu::compress(image.view(), tile, {file.data(), file.size()}, written),
              Status::success);
    file.resize(static_cast<std::size_t>(written));
    return file;
}

inline HostImage cpu_decode(const Bytes &file, ImageSpec image) {
    auto result = make_image(image.type, image.width, image.height, 0, 0);
    EXPECT_EQ(cpu::decompress({file.data(), file.size()}, result.output_view()), Status::success);
    return result;
}

struct Workspace {
    DevicePtr data;
    std::uint64_t size = 0;
    MutableBytes bytes() const {
        return {data.get(), size};
    }
};

inline Workspace workspace_for(const cuda::BatchPlan &plan) {
    Workspace result;
    EXPECT_EQ(workspace_size(plan.workspace_spec(), result.size), Status::success);
    result.data = device_alloc(result.size);
    return result;
}

inline std::vector<DeviceResult> download_results(const DevicePtr &results, std::size_t count) {
    std::vector<DeviceResult> records(count);
    EXPECT_EQ(cudaMemcpy(records.data(), results.get(), count * sizeof(DeviceResult),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return records;
}

// Device-resident encoder output, with guard bytes after each compress_bound capacity.
struct GpuArchive {
    std::vector<DevicePtr> inputs;
    std::vector<DevicePtr> outputs;
    std::vector<std::uint64_t> bounds;
    DevicePtr results;
    std::vector<Bytes> files;
};

inline GpuArchive gpu_encode(const std::vector<HostImage> &images, TileShape tile = {},
                             cudaStream_t stream = nullptr) {
    GpuArchive archive;
    std::vector<cuda::BatchItem> items(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        std::uint64_t bound = 0;
        EXPECT_EQ(compress_bound(images[i].image, tile, bound), Status::success);
        archive.inputs.push_back(upload(images[i].bytes));
        archive.outputs.push_back(device_alloc(bound + guard_bytes));
        archive.bounds.push_back(bound);
        items[i].input = {archive.inputs[i].get(), images[i].image, images[i].pitch,
                          images[i].bytes.size()};
        items[i].compressed_output = {archive.outputs[i].get(), bound};
    }
    std::unique_ptr<cuda::BatchPlan> plan;
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::compress, items.data(), items.size(), tile},
                                  plan, stream),
              Status::success);
    if (plan == nullptr) {
        return archive;
    }
    const auto workspace = workspace_for(*plan);
    archive.results = device_alloc(images.size() * sizeof(DeviceResult), 0xff);
    EXPECT_EQ(cuda::compress_async(
                  *plan, workspace.bytes(),
                  {static_cast<DeviceResult *>(archive.results.get()), images.size()}, stream),
              Status::success);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const auto records = download_results(archive.results, images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        EXPECT_EQ(records[i].status, Status::success) << "image " << i;
        EXPECT_EQ(records[i].reserved, 0U);
        EXPECT_LE(records[i].bytes_written, archive.bounds[i]);
        auto file = download(archive.outputs[i].get(), archive.bounds[i] + guard_bytes);
        const auto written = std::min(records[i].bytes_written, archive.bounds[i]);
        EXPECT_TRUE(std::all_of(file.begin() + static_cast<std::ptrdiff_t>(written), file.end(),
                                [](auto byte) { return byte == sentinel; }))
            << "image " << i;
        file.resize(static_cast<std::size_t>(written));
        archive.files.push_back(std::move(file));
    }
    return archive;
}

} // namespace test
