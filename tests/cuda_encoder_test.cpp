#include "cuda_test_support.h"

#include <array>

namespace {

using namespace test;

void expect_cpu_equivalent(const std::vector<HostImage> &images, TileShape tile,
                           const GpuArchive &archive) {
    ASSERT_EQ(archive.files.size(), images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "image " << i << " " << images[i].image.width << "x"
                                          << images[i].image.height);
        EXPECT_EQ(archive.files[i], cpu_encode(images[i], tile));
        EXPECT_EQ(cpu_decode(archive.files[i], images[i].image).pixels(), images[i].pixels());
    }
}

TEST(CudaEncoder, ReproducesGoldenFiles) {
    struct Case {
        const char *name;
        PixelType type;
        std::uint32_t width;
        std::uint32_t height;
        std::vector<std::uint32_t> pixels;
    };
    std::vector<std::uint32_t> zero_width(16, 0x0102);
    std::fill(zero_width.begin() + 8, zero_width.end(), 0x0304);
    std::vector<std::uint32_t> half_range = {0, 32768, 0, 32768, 0, 32768, 0, 32768, 0};
    half_range.insert(half_range.end(), 9, 5);
    const std::vector<Case> cases = {
        {"uint16_raw", PixelType::uint16, 1, 1, {0x1234}},
        {"uint16_constant", PixelType::uint16, 3, 2, std::vector<std::uint32_t>(6, 0xbeef)},
        {"uint16_wrap",
         PixelType::uint16,
         8,
         2,
         {65535, 0, 1, 2, 3, 4, 5, 6, 0, 65535, 65534, 65533, 65532, 65531, 65530, 65529}},
        {"uint16_zero_width", PixelType::uint16, 8, 2, zero_width},
        {"uint16_raw_tie", PixelType::uint16, 6, 1, {0, 1, 2, 3, 4, 5}},
        {"uint16_half_range", PixelType::uint16, 9, 2, half_range},
        {"float32_raw", PixelType::float32, 1, 1, {0x7fc01234}},
        {"float32_negative_zero_constant", PixelType::float32, 3, 2,
         std::vector<std::uint32_t>(6, 0x80000000)},
    };
    std::vector<HostImage> uint16_images;
    std::vector<HostImage> float_images;
    std::vector<Bytes> expected[2];
    for (const auto &test : cases) {
        auto image = make_image(test.type, test.width, test.height, 3);
        for (std::uint32_t i = 0; i < test.pixels.size(); ++i) {
            image.set(i % test.width, i / test.width, test.pixels[i]);
        }
        const auto index = test.type == PixelType::uint16 ? 0 : 1;
        (index == 0 ? uint16_images : float_images).push_back(image);
        expected[index].push_back(golden(test.name));
    }
    EXPECT_EQ(gpu_encode(uint16_images).files, expected[0]);
    EXPECT_EQ(gpu_encode(float_images).files, expected[1]);
}

TEST(CudaEncoder, EveryResidualWidthAndFinalWord) {
    std::mt19937 rng(2026517);
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        const std::uint32_t maximum = type == PixelType::uint16 ? 16 : 32;
        std::vector<HostImage> images;
        for (const auto width : {31U, 32U, 33U, 127U, 128U}) {
            for (std::uint32_t bits = 0; bits <= maximum; ++bits) {
                auto image = make_image(type, width, 2, 1);
                const auto mask = bits == 32 ? UINT32_MAX : (std::uint32_t{1} << bits) - 1;
                image.set(0, 1, type == PixelType::uint16 ? 65535 : 0xa5a5a5a5);
                for (std::uint32_t x = 0; x < width; ++x) {
                    image.set(x, 0, 400);
                }
                for (std::uint32_t x = 1; x < width; ++x) {
                    const auto z = x == 1 ? mask : static_cast<std::uint32_t>(rng()) & mask;
                    const auto previous = image.get(x - 1, 1);
                    if (type == PixelType::uint16) {
                        const auto delta = z % 2 == 0 ? z / 2 : 65535 - z / 2;
                        image.set(x, 1, (previous + delta) & 65535U);
                    } else {
                        image.set(x, 1, previous ^ z);
                    }
                }
                images.push_back(image);
            }
        }
        const auto archive = gpu_encode(images);
        expect_cpu_equivalent(images, {}, archive);
        for (std::size_t i = 0; i < images.size(); ++i) {
            ImageMetadata metadata;
            ASSERT_EQ(
                parse_metadata({archive.files[i].data(), archive.files[i].size()}, {}, metadata),
                Status::success);
            ASSERT_EQ(metadata.segments.size(), 2U);
            EXPECT_EQ(metadata.segments[1].bit_width, i % (maximum + 1));
        }
    }
}

TEST(CudaEncoder, RandomBatchesMatchCpuBitstreams) {
    constexpr std::array<std::uint32_t, 16> sizes = {1,  2,  15,  16,  31,  32,  33,  63,
                                                     64, 65, 127, 128, 129, 255, 256, 257};
    std::array<std::uint64_t, 3> modes{};
    for (std::uint32_t batch = 0; batch < 20; ++batch) {
        const auto seed = 2026517U + batch;
        SCOPED_TRACE(::testing::Message() << "seed=" << seed);
        std::mt19937 rng(seed);
        const auto type = batch % 2 == 0 ? PixelType::uint16 : PixelType::float32;
        const auto tile = shapes[batch % 3];
        std::vector<HostImage> images;
        for (std::uint32_t i = 0; i < 50; ++i) {
            const auto width = sizes[rng() % sizes.size()];
            const auto height = sizes[rng() % sizes.size()];
            images.push_back(
                pattern_image(type, width, height, static_cast<std::uint32_t>(rng() % 3), i, rng));
        }
        const auto first = gpu_encode(images, tile);
        expect_cpu_equivalent(images, tile, first);
        if (batch < 4) {
            EXPECT_EQ(gpu_encode(images, tile).files, first.files);
        }
        for (const auto &file : first.files) {
            ImageMetadata metadata;
            ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
            for (const auto &entry : metadata.tiles) {
                ++modes[static_cast<std::size_t>(entry.mode)];
            }
        }
    }
    EXPECT_GT(modes[0], 0U);
    EXPECT_GT(modes[1], 0U);
    EXPECT_GT(modes[2], 0U);
}

TEST(CudaEncoder, EveryUint16ValueAndSpecialFloatWords) {
    auto values = make_image(PixelType::uint16, 256, 256, 5);
    for (std::uint32_t i = 0; i < 65536; ++i) {
        values.set(i % 256, i / 256, i);
    }
    constexpr std::array<std::uint32_t, 16> special = {
        0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0x7fc01234, 0x7fa12345,
        0xffc0abcd, 0x3f800000, 0xbf800000, 0x00800000,
    };
    auto floats = make_image(PixelType::float32, 129, 33, 2);
    for (std::uint32_t y = 0; y < 33; ++y) {
        for (std::uint32_t x = 0; x < 129; ++x) {
            floats.set(x, y, special[(x + 3 * y) % special.size()]);
        }
    }
    for (const auto tile : shapes) {
        expect_cpu_equivalent({values}, tile, gpu_encode({values}, tile));
        expect_cpu_equivalent({floats}, tile, gpu_encode({floats}, tile));
    }
}

TEST(CudaEncoder, ImagesSplitAcrossWindows) {
    // 14005 tiles: windows of 4096 split the second and third images, and the third image
    // covers a whole middle window that both continues and is continued.
    std::mt19937 rng(7);
    std::vector<HostImage> images;
    images.push_back(make_image(PixelType::uint16, 1, 32 * 3000, 0));
    for (std::uint32_t y = 0; y < 32 * 3000; ++y) {
        images[0].set(0, y, (y / 32) % 3 == 0 ? 9 : y);
    }
    images.push_back(pattern_image(PixelType::uint16, 129, 32 * 1000, 1, 3, rng));
    images.push_back(make_image(PixelType::uint16, 1, 32 * 9000, 0));
    for (std::uint32_t y = 0; y < 32 * 9000; ++y) {
        images[2].set(0, y, (y / 64) % 2 == 0 ? 4 : static_cast<std::uint32_t>(rng()));
    }
    for (std::uint32_t i = 0; i < 5; ++i) {
        images.push_back(pattern_image(PixelType::uint16, 1, 1, 0, 4, rng));
    }
    expect_cpu_equivalent(images, {}, gpu_encode(images));

    const auto floats = pattern_image(PixelType::float32, 1, 32 * 4100, 0, 5, rng);
    expect_cpu_equivalent({floats}, {64, 64}, gpu_encode({floats}, {64, 64}));
}

TEST(CudaEncoder, ManySmallImagesInSharedAllocations) {
    constexpr std::uint32_t count = 5000;
    std::mt19937 rng(11);
    std::vector<HostImage> images;
    Bytes packed;
    for (std::uint32_t i = 0; i < count; ++i) {
        images.push_back(pattern_image(PixelType::float32, 1 + i % 3, 1, 0, i, rng));
        packed.resize(i * 16);
        const auto pixels = images.back().pixels();
        packed.insert(packed.end(), pixels.begin(), pixels.end());
    }
    packed.resize(count * 16);
    const auto input = upload(packed);
    const auto output = device_alloc(std::uint64_t{count} * 112);
    std::vector<cuda::BatchItem> items(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto &image = images[i];
        items[i].input = {static_cast<std::uint8_t *>(input.get()) + i * 16, image.image,
                          image.pitch, 16};
        items[i].compressed_output = {static_cast<std::uint8_t *>(output.get()) + i * 112, 112};
    }
    std::unique_ptr<cuda::BatchPlan> plan;
    ASSERT_EQ(
        cuda::prepare_batch({cuda::Operation::compress, items.data(), count, {}}, plan, nullptr),
        Status::success);
    EXPECT_EQ(plan->workspace_spec().image_count, 4096U);
    EXPECT_EQ(plan->workspace_spec().tile_count, 4096U);
    EXPECT_GT(plan->workspace_spec().scan_temp_bytes, 0U);
    const auto workspace = workspace_for(*plan);
    const auto results = device_alloc(count * sizeof(DeviceResult));
    ASSERT_EQ(cuda::compress_async(*plan, workspace.bytes(),
                                   {static_cast<DeviceResult *>(results.get()), count}, nullptr),
              Status::success);
    const auto records = download_results(results, count);
    const auto files = download(output.get(), std::uint64_t{count} * 112);
    for (std::uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(records[i].status, Status::success);
        const auto begin = files.begin() + i * 112;
        const Bytes file(begin, begin + static_cast<std::ptrdiff_t>(records[i].bytes_written));
        ASSERT_EQ(file, cpu_encode(images[i])) << "image " << i;
        EXPECT_TRUE(std::all_of(begin + static_cast<std::ptrdiff_t>(file.size()), begin + 112,
                                [](auto byte) { return byte == sentinel; }));
    }
}

TEST(CudaEncoder, EmptyBatchIsNoOp) {
    std::unique_ptr<cuda::BatchPlan> plan;
    ASSERT_EQ(cuda::prepare_batch({cuda::Operation::compress, nullptr, 0, {}}, plan, nullptr),
              Status::success);
    std::uint64_t bytes = 1;
    EXPECT_EQ(workspace_size(plan->workspace_spec(), bytes), Status::success);
    EXPECT_EQ(bytes, 0U);
    EXPECT_EQ(cuda::compress_async(*plan, {}, {}, nullptr), Status::success);
}

class CudaEncoderValidation : public ::testing::Test {
  protected:
    void SetUp() override {
        std::mt19937 rng(3);
        source = pattern_image(PixelType::uint16, 129, 33, 2, 3, rng);
        ASSERT_EQ(compress_bound(source.image, {}, bound), Status::success);
        input = upload(source.bytes);
        output = device_alloc(bound + guard_bytes);
        item.input = {input.get(), source.image, source.pitch, source.bytes.size()};
        item.compressed_output = {output.get(), bound};
    }

    Status prepare(const cuda::BatchItem &changed) {
        return cuda::prepare_batch({cuda::Operation::compress, &changed, 1, {}}, plan, nullptr);
    }

    HostImage source;
    std::uint64_t bound = 0;
    DevicePtr input;
    DevicePtr output;
    cuda::BatchItem item;
    std::unique_ptr<cuda::BatchPlan> plan;
};

TEST_F(CudaEncoderValidation, RejectsInvalidRequestsAndKeepsPreviousPlan) {
    ASSERT_EQ(prepare(item), Status::success);
    const auto *previous = plan.get();
    const auto attempt = [&](const cuda::BatchItem &changed, Status status) {
        EXPECT_EQ(prepare(changed), status);
        EXPECT_EQ(plan.get(), previous);
    };
    auto changed = item;
    changed.input.image.type = static_cast<PixelType>(9);
    attempt(changed, Status::unsupported_type);
    changed = item;
    changed.input.image.width = 0;
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.input.pitch_bytes = 2;
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.input.data = static_cast<const std::uint8_t *>(input.get()) + 1;
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.input.capacity_bytes = 10;
    attempt(changed, Status::buffer_too_small);
    changed = item;
    changed.compressed_output = {nullptr, bound};
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.compressed_output = {static_cast<std::uint8_t *>(output.get()) + 2, bound};
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.compressed_output.size = bound - 1;
    attempt(changed, Status::buffer_too_small);
    changed = item;
    changed.compressed_output.size = UINT64_MAX;
    attempt(changed, Status::size_overflow);
    changed = item;
    changed.compressed_output = {input.get(), bound};
    attempt(changed, Status::invalid_argument);

    std::vector<std::uint8_t> host(static_cast<std::size_t>(bound));
    int pageable = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0), cudaSuccess);
    changed = item;
    changed.compressed_output = {host.data(), bound};
    if (reinterpret_cast<std::uintptr_t>(host.data()) % 4 == 0 && pageable == 0) {
        attempt(changed, Status::invalid_argument);
    }

    std::array<cuda::BatchItem, 2> pair = {item, item};
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::compress, pair.data(), 2, {}}, plan, nullptr),
              Status::invalid_argument);
    const auto second = device_alloc(bound);
    pair[1].compressed_output = {second.get(), bound};
    pair[1].input.image.type = PixelType::float32;
    pair[1].input.pitch_bytes = 129 * 4;
    pair[1].input.image.height = 1;
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::compress, pair.data(), 2, {}}, plan, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(plan.get(), previous);

    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::compress, &item, 1, {32, 128}}, plan, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::calibrate, &item, 1, {}}, plan, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::compress, nullptr, 1, {}}, plan, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(
        cuda::prepare_batch({cuda::Operation::compress, &item, std::uint64_t{UINT32_MAX} + 1, {}},
                            plan, nullptr),
        Status::resource_limit);
    EXPECT_EQ(plan.get(), previous);

    // Inputs are read-only and may be shared.
    pair = {item, item};
    pair[1].compressed_output = {second.get(), bound};
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::compress, pair.data(), 2, {}}, plan, nullptr),
              Status::success);
}

TEST_F(CudaEncoderValidation, RejectedExecutionsLeaveBuffersUntouched) {
    ASSERT_EQ(prepare(item), Status::success);
    auto workspace = workspace_for(*plan);
    const auto results = device_alloc(4 * sizeof(DeviceResult));
    auto *records = static_cast<DeviceResult *>(results.get());
    const auto large = device_alloc(workspace.size + 1024);
    const auto attempt = [&](MutableBytes scratch, cuda::ResultBuffer buffer, Status status) {
        EXPECT_EQ(cuda::compress_async(*plan, scratch, buffer, nullptr), status);
    };
    attempt({workspace.data.get(), workspace.size - 1}, {records, 1}, Status::buffer_too_small);
    attempt({nullptr, 0}, {records, 1}, Status::buffer_too_small);
    attempt({static_cast<std::uint8_t *>(large.get()) + 128, workspace.size}, {records, 1},
            Status::invalid_argument);
    attempt(workspace.bytes(), {records, 0}, Status::buffer_too_small);
    attempt(workspace.bytes(), {nullptr, 1}, Status::invalid_argument);
    attempt(workspace.bytes(),
            {reinterpret_cast<DeviceResult *>(static_cast<std::uint8_t *>(results.get()) + 4), 1},
            Status::invalid_argument);
    attempt(workspace.bytes(), {records, UINT64_MAX / 8}, Status::size_overflow);
    attempt({output.get(), workspace.size}, {records, 1}, Status::invalid_argument);
    attempt(workspace.bytes(), {static_cast<DeviceResult *>(output.get()), 1},
            Status::invalid_argument);
    attempt(workspace.bytes(), {static_cast<DeviceResult *>(input.get()), 1},
            Status::invalid_argument);
    attempt(workspace.bytes(), {static_cast<DeviceResult *>(workspace.data.get()), 1},
            Status::invalid_argument);
    cuda::BatchPlan moved = std::move(*plan);
    EXPECT_EQ(cuda::compress_async(*plan, workspace.bytes(), {records, 1}, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(download(output.get(), bound + guard_bytes), Bytes(bound + guard_bytes, sentinel));
    EXPECT_EQ(download(results.get(), 4 * sizeof(DeviceResult)),
              Bytes(4 * sizeof(DeviceResult), sentinel));

    ASSERT_EQ(cuda::compress_async(moved, workspace.bytes(), {records, 1}, nullptr),
              Status::success);
    const auto record = download_results(results, 1)[0];
    ASSERT_EQ(record.status, Status::success);
    auto file = download(output.get(), record.bytes_written);
    EXPECT_EQ(file, cpu_encode(source));
}

} // namespace
