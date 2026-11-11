#include <tilestream/build_config.h>
#include <tilestream/cpu.h>
#include <tilestream/info.h>
#include <tilestream/sizing.h>
#if TILESTREAM_WITH_CUDA
#include <tilestream/cuda.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace ts = tilestream;

namespace {

constexpr ts::ImageSpec image{ts::PixelType::uint16, 1000, 600};
constexpr std::uint64_t pitch = image.width * sizeof(std::uint16_t);
constexpr std::uint64_t frame_bytes = pitch * image.height;
constexpr int frame_count = 3;
constexpr ts::ImageSpec region{ts::PixelType::float32, image.width, image.height};
constexpr std::uint64_t region_bytes = std::uint64_t{image.width} * image.height * 4;
constexpr float gain = 0.5F;

// A smooth background with noise in the low bits, similar to a detector frame.
std::vector<std::uint16_t> make_frame(int index) {
    std::vector<std::uint16_t> frame(std::uint64_t{image.width} * image.height);
    std::uint32_t noise = 12345U + static_cast<std::uint32_t>(index);
    for (std::uint32_t y = 0; y < image.height; ++y) {
        for (std::uint32_t x = 0; x < image.width; ++x) {
            noise = noise * 1664525U + 1013904223U;
            frame[std::uint64_t{y} * image.width + x] =
                static_cast<std::uint16_t>(1000 + 2 * x + y + 100 * index + (noise >> 28));
        }
    }
    return frame;
}

bool ok(ts::Status status, const char *what) {
    if (status != ts::Status::success) {
        std::cerr << what << " failed with status " << static_cast<std::uint32_t>(status) << '\n';
        return false;
    }
    return true;
}

bool cpu_round_trip() {
    std::uint64_t bound = 0;
    if (!ok(ts::compress_bound(image, {}, bound), "compress_bound")) {
        return false;
    }
    std::vector<std::uint8_t> file(bound);
    std::vector<std::uint16_t> decoded(std::uint64_t{image.width} * image.height);
    for (int i = 0; i < frame_count; ++i) {
        const auto frame = make_frame(i);
        std::uint64_t size = 0;
        if (!ok(ts::cpu::compress({frame.data(), image, pitch, frame_bytes}, {},
                                  {file.data(), bound}, size),
                "cpu::compress") ||
            !ok(ts::cpu::decompress({file.data(), size},
                                    {decoded.data(), image, pitch, frame_bytes}),
                "cpu::decompress")) {
            return false;
        }
        if (decoded != frame) {
            std::cerr << "CPU frame " << i << " does not match\n";
            return false;
        }
        std::cout << "CPU frame " << i << ": " << frame_bytes << " -> " << size << " bytes\n";
    }
    return true;
}

#if TILESTREAM_WITH_CUDA
namespace cuda = ts::cuda;

// Dark-frame correction, (frame - dark) * gain, as the operation that follows decoding.
std::vector<float> make_dark() {
    std::vector<float> dark(std::uint64_t{image.width} * image.height);
    for (std::uint64_t i = 0; i < dark.size(); ++i) {
        dark[i] = 990.0F + static_cast<float>(i % 7);
    }
    return dark;
}

// Plans and buffers are prepared once; each frame only enqueues work in the stream.
bool gpu_round_trip(cudaStream_t stream) {
    // Allocations may still be pending in the stream when setup fails.
    const auto fail = [stream] {
        static_cast<void>(cuda::synchronize(stream));
        return false;
    };
    std::uint64_t bound = 0;
    cuda::DeviceBuffer input;
    cuda::DeviceBuffer archive;
    cuda::DeviceBuffer output;
    cuda::DeviceBuffer dark;
    cuda::DeviceBuffer corrected;
    cuda::DeviceBuffer results;
    const auto dark_frame = make_dark();
    if (!ok(ts::compress_bound(image, {}, bound), "compress_bound") ||
        !ok(cuda::DeviceBuffer::allocate(frame_bytes, stream, input), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(bound, stream, archive), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(frame_bytes, stream, output), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(region_bytes, stream, dark), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(region_bytes, stream, corrected), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(3 * sizeof(ts::DeviceResult), stream, results),
            "allocate") ||
        cudaMemcpyAsync(dark.data(), dark_frame.data(), region_bytes, cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
        return fail();
    }
    auto *records = static_cast<ts::DeviceResult *>(results.data());

    cuda::BatchItem encode;
    encode.input = {input.data(), image, pitch, frame_bytes};
    encode.compressed_output = archive.bytes();
    // The decoder reads the archive as the encoder left it, with no host copy of the index.
    cuda::BatchItem decode;
    decode.compressed_input.bytes = {archive.data(), archive.size()};
    decode.compressed_input.encoder_result = &records[0];
    decode.compressed_input.trusted_image = image;
    decode.output = {output.data(), image, pitch, frame_bytes};
    // The same archive decoded straight into calibrated FP32, without the uint16 frame.
    cuda::BatchItem correct;
    correct.compressed_input = decode.compressed_input;
    correct.roi = {0, 0, image.width, image.height};
    correct.output = {corrected.data(), region, region.width * 4ULL, region_bytes};
    correct.calibration.offset = {
        cuda::ParameterKind::map, 0.0F, {dark.data(), region, region.width * 4ULL, region_bytes}};
    correct.calibration.gain.scalar = gain;

    std::unique_ptr<cuda::BatchPlan> compress;
    std::unique_ptr<cuda::BatchPlan> decompress;
    std::unique_ptr<cuda::BatchPlan> calibrate;
    std::uint64_t sizes[3] = {};
    if (!ok(cuda::prepare_batch({cuda::Operation::compress, &encode, 1, {}}, compress, stream),
            "prepare compress") ||
        !ok(cuda::prepare_batch({cuda::Operation::decompress, &decode, 1, {}}, decompress, stream),
            "prepare decompress") ||
        !ok(cuda::prepare_batch({cuda::Operation::decode_calibrate, &correct, 1, {}}, calibrate,
                                stream),
            "prepare decode_calibrate") ||
        !ok(ts::workspace_size(compress->workspace_spec(), sizes[0]), "workspace_size") ||
        !ok(ts::workspace_size(decompress->workspace_spec(), sizes[1]), "workspace_size") ||
        !ok(ts::workspace_size(calibrate->workspace_spec(), sizes[2]), "workspace_size")) {
        return fail();
    }
    // All operations run in order in one stream, so they can share one workspace.
    cuda::DeviceBuffer workspace;
    if (!ok(cuda::DeviceBuffer::allocate(*std::max_element(sizes, sizes + 3), stream, workspace),
            "allocate")) {
        return fail();
    }

    std::vector<std::uint16_t> decoded(std::uint64_t{image.width} * image.height);
    std::vector<float> calibrated(decoded.size());
    for (int i = 0; i < frame_count; ++i) {
        const auto frame = make_frame(i);
        ts::DeviceResult outcome[3];
        const bool queued =
            cudaMemcpyAsync(input.data(), frame.data(), frame_bytes, cudaMemcpyHostToDevice,
                            stream) == cudaSuccess &&
            ok(cuda::compress_async(*compress, workspace.bytes(), {&records[0], 1}, stream),
               "compress_async") &&
            ok(cuda::decompress_async(*decompress, workspace.bytes(), {&records[1], 1}, stream),
               "decompress_async") &&
            ok(cuda::decode_calibrate_async(*calibrate, workspace.bytes(), {&records[2], 1},
                                            stream),
               "decode_calibrate_async") &&
            cudaMemcpyAsync(outcome, records, sizeof(outcome), cudaMemcpyDeviceToHost, stream) ==
                cudaSuccess &&
            cudaMemcpyAsync(decoded.data(), output.data(), frame_bytes, cudaMemcpyDeviceToHost,
                            stream) == cudaSuccess &&
            cudaMemcpyAsync(calibrated.data(), corrected.data(), region_bytes,
                            cudaMemcpyDeviceToHost, stream) == cudaSuccess;
        // A failed enqueue can leave earlier work running, so wait before the buffers are freed.
        if (!ok(cuda::synchronize(stream), "synchronize") || !queued) {
            return false;
        }
        if (!ok(outcome[0].status, "GPU encode") || !ok(outcome[1].status, "GPU decode") ||
            !ok(outcome[2].status, "GPU decode_calibrate")) {
            return false;
        }
        if (decoded != frame) {
            std::cerr << "GPU frame " << i << " does not match\n";
            return false;
        }
        // The library rounds the subtraction and the multiplication separately, like this.
        for (std::uint64_t k = 0; k < frame.size(); ++k) {
            const float difference = static_cast<float>(frame[k]) - dark_frame[k];
            if (calibrated[k] != difference * gain) {
                std::cerr << "GPU frame " << i << " is not calibrated as expected\n";
                return false;
            }
        }
        std::cout << "GPU frame " << i << ": " << frame_bytes << " -> " << outcome[0].bytes_written
                  << " bytes, calibrated pixel 0: " << calibrated[0] << '\n';
    }
    return true;
}

bool gpu_available() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return true;
}
#endif

} // namespace

int main() {
    std::cout << "TileStream " << ts::version() << '\n';
    if (!cpu_round_trip()) {
        return 1;
    }
#if TILESTREAM_WITH_CUDA
    if (!gpu_available()) {
        std::cout << "No CUDA device, GPU round trip skipped\n";
        return 0;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::cerr << "cudaStreamCreateWithFlags failed\n";
        return 1;
    }
    const bool passed = gpu_round_trip(stream);
    // The buffers and plans were destroyed after their work completed.
    if (cudaStreamDestroy(stream) != cudaSuccess || !passed) {
        return 1;
    }
#endif
    return 0;
}
