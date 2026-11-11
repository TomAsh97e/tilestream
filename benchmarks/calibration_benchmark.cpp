// Separate decode and calibration versus the fused operation, on identical work: the same
// archive already on the device, the same region, and the same parameters or device maps.
#include <tilestream/api.h>
#include <tilestream/cpu.h>
#include <tilestream/cuda.h>
#include <tilestream/sizing.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace ts = tilestream;
namespace cuda = tilestream::cuda;

namespace {

constexpr std::uint32_t frame_size = 4096;
constexpr int warmup = 5;

bool ok(ts::Status status, const char *what) {
    if (status != ts::Status::success) {
        std::fprintf(stderr, "%s failed with status %u\n", what, static_cast<unsigned>(status));
    }
    return status == ts::Status::success;
}

bool ok(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(error));
    }
    return error == cudaSuccess;
}

// A smooth background with noise in the low bits, like a detector frame.
std::vector<std::uint8_t> detector_frame(ts::PixelType type) {
    const std::uint64_t sample_bytes = type == ts::PixelType::uint16 ? 2 : 4;
    std::vector<std::uint8_t> bytes(std::uint64_t{frame_size} * frame_size * sample_bytes);
    std::uint32_t noise = 1;
    for (std::uint32_t y = 0; y < frame_size; ++y) {
        for (std::uint32_t x = 0; x < frame_size; ++x) {
            noise = noise * 1664525U + 1013904223U;
            auto *pixel = bytes.data() + (std::uint64_t{y} * frame_size + x) * sample_bytes;
            if (type == ts::PixelType::uint16) {
                const auto value = static_cast<std::uint16_t>(1000 + x / 4 + y / 8 + (noise >> 28));
                std::memcpy(pixel, &value, 2);
            } else {
                const float value = 100.0F + 0.01F * static_cast<float>(x) +
                                    0.02F * static_cast<float>(y) +
                                    1e-4F * static_cast<float>(noise >> 20);
                std::memcpy(pixel, &value, 4);
            }
        }
    }
    return bytes;
}

struct Timing {
    double median = 0;
    double p95 = 0;
};

Timing summarize(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    return {samples[n / 2], samples[(n * 95 + 99) / 100 - 1]};
}

// Payload bytes of the tiles a region touches; the decoder reads each of them once.
std::uint64_t payload_bytes(const ts::ImageMetadata &metadata, const ts::Roi &roi) {
    const std::uint64_t across =
        (metadata.image.width + metadata.tile.width - 1) / metadata.tile.width;
    std::uint64_t total = 0;
    for (std::uint64_t row = roi.y / metadata.tile.height;
         row <= (std::uint64_t{roi.y} + roi.height - 1) / metadata.tile.height; ++row) {
        for (std::uint64_t column = roi.x / metadata.tile.width;
             column <= (std::uint64_t{roi.x} + roi.width - 1) / metadata.tile.width; ++column) {
            total += metadata.tiles[static_cast<std::size_t>(row * across + column)].length;
        }
    }
    return total;
}

bool run_case(ts::PixelType type, bool full, bool maps, int samples, cudaStream_t stream) {
    const ts::ImageSpec image{type, frame_size, frame_size};
    const std::uint64_t sample_bytes = type == ts::PixelType::uint16 ? 2 : 4;
    const auto frame = detector_frame(type);
    const ts::Roi roi =
        full ? ts::Roi{0, 0, frame_size, frame_size} : ts::Roi{1500, 1300, 1024, 1024};
    const std::uint64_t pixels = std::uint64_t{roi.width} * roi.height;
    const ts::ImageSpec decoded_image{type, roi.width, roi.height};
    const ts::ImageSpec region{ts::PixelType::float32, roi.width, roi.height};

    std::uint64_t bound = 0;
    std::uint64_t size = 0;
    if (!ok(ts::compress_bound(image, {}, bound), "compress_bound")) {
        return false;
    }
    std::vector<std::uint8_t> file(bound);
    ts::ImageMetadata metadata;
    if (!ok(ts::cpu::compress({frame.data(), image, frame_size * sample_bytes, frame.size()}, {},
                              {file.data(), bound}, size),
            "cpu::compress") ||
        !ok(ts::parse_metadata({file.data(), size}, {}, metadata), "parse_metadata")) {
        return false;
    }

    // Allocations and copies may still be pending in the stream when setup fails.
    const auto fail = [stream] {
        static_cast<void>(cuda::synchronize(stream));
        return false;
    };
    cuda::DeviceBuffer source;
    cuda::DeviceBuffer decoded;
    cuda::DeviceBuffer separate;
    cuda::DeviceBuffer fused;
    cuda::DeviceBuffer offsets;
    cuda::DeviceBuffer gains;
    cuda::DeviceBuffer results;
    std::vector<float> offset_map(pixels);
    std::vector<float> gain_map(pixels);
    for (std::uint64_t i = 0; i < pixels; ++i) {
        offset_map[i] = 95.0F + static_cast<float>(i % 11);
        gain_map[i] = 0.9F + 0.01F * static_cast<float>(i % 23);
    }
    if (!ok(cuda::DeviceBuffer::allocate(size, stream, source), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(pixels * sample_bytes, stream, decoded), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(pixels * 4, stream, separate), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(pixels * 4, stream, fused), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(maps ? pixels * 4 : 0, stream, offsets), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(maps ? pixels * 4 : 0, stream, gains), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(3 * sizeof(ts::DeviceResult), stream, results),
            "allocate") ||
        !ok(cudaMemcpyAsync(source.data(), file.data(), size, cudaMemcpyHostToDevice, stream),
            "upload") ||
        (maps && (!ok(cudaMemcpyAsync(offsets.data(), offset_map.data(), pixels * 4,
                                      cudaMemcpyHostToDevice, stream),
                      "upload") ||
                  !ok(cudaMemcpyAsync(gains.data(), gain_map.data(), pixels * 4,
                                      cudaMemcpyHostToDevice, stream),
                      "upload")))) {
        return fail();
    }

    cuda::Calibration calibration;
    if (maps) {
        calibration.offset = {
            cuda::ParameterKind::map, 0.0F, {offsets.data(), region, roi.width * 4ULL, pixels * 4}};
        calibration.gain = {
            cuda::ParameterKind::map, 0.0F, {gains.data(), region, roi.width * 4ULL, pixels * 4}};
    } else {
        calibration.offset.scalar = 100.0F;
        calibration.gain.scalar = 1.25F;
    }
    cuda::BatchItem decode;
    decode.compressed_input.bytes = {source.data(), size};
    decode.compressed_input.metadata = &metadata;
    decode.roi = roi;
    decode.output = {decoded.data(), decoded_image, roi.width * sample_bytes,
                     pixels * sample_bytes};
    cuda::BatchItem calibrate;
    calibrate.input = {decoded.data(), decoded_image, roi.width * sample_bytes,
                       pixels * sample_bytes};
    calibrate.output = {separate.data(), region, roi.width * 4ULL, pixels * 4};
    calibrate.calibration = calibration;
    cuda::BatchItem combined = decode;
    combined.output = {fused.data(), region, roi.width * 4ULL, pixels * 4};
    combined.calibration = calibration;

    std::unique_ptr<cuda::BatchPlan> decode_plan;
    std::unique_ptr<cuda::BatchPlan> calibrate_plan;
    std::unique_ptr<cuda::BatchPlan> fused_plan;
    std::uint64_t sizes[3] = {};
    if (!ok(cuda::prepare_batch({cuda::Operation::decode_roi, &decode, 1, {}}, decode_plan, stream),
            "prepare decode") ||
        !ok(cuda::prepare_batch({cuda::Operation::calibrate, &calibrate, 1, {}}, calibrate_plan,
                                stream),
            "prepare calibrate") ||
        !ok(cuda::prepare_batch({cuda::Operation::decode_calibrate, &combined, 1, {}}, fused_plan,
                                stream),
            "prepare fused") ||
        !ok(ts::workspace_size(decode_plan->workspace_spec(), sizes[0]), "workspace_size") ||
        !ok(ts::workspace_size(calibrate_plan->workspace_spec(), sizes[1]), "workspace_size") ||
        !ok(ts::workspace_size(fused_plan->workspace_spec(), sizes[2]), "workspace_size")) {
        return fail();
    }
    // Everything runs in order in one stream, so one workspace serves all three plans.
    cuda::DeviceBuffer workspace;
    if (!ok(cuda::DeviceBuffer::allocate(*std::max_element(sizes, sizes + 3), stream, workspace),
            "allocate")) {
        return fail();
    }
    auto *records = static_cast<ts::DeviceResult *>(results.data());
    const auto run_separate = [&] {
        return ok(cuda::decode_roi_async(*decode_plan, workspace.bytes(), {&records[0], 1}, stream),
                  "decode_roi_async") &&
               ok(cuda::calibrate_async(*calibrate_plan, workspace.bytes(), {&records[1], 1},
                                        stream),
                  "calibrate_async");
    };
    const auto run_fused = [&] {
        return ok(
            cuda::decode_calibrate_async(*fused_plan, workspace.bytes(), {&records[2], 1}, stream),
            "decode_calibrate_async");
    };

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!ok(cudaEventCreate(&start), "cudaEventCreate") ||
        !ok(cudaEventCreate(&stop), "cudaEventCreate")) {
        return fail();
    }
    const auto measure = [&](const auto &enqueue, std::vector<float> &out) {
        float ms = 0;
        const bool done = ok(cudaEventRecord(start, stream), "cudaEventRecord") && enqueue() &&
                          ok(cudaEventRecord(stop, stream), "cudaEventRecord") &&
                          ok(cudaEventSynchronize(stop), "cudaEventSynchronize") &&
                          ok(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        out.push_back(ms);
        return done;
    };
    std::vector<float> ignored;
    std::vector<float> separate_ms;
    std::vector<float> fused_ms;
    bool passed = true;
    for (int i = 0; i < warmup && passed; ++i) {
        passed = measure(run_separate, ignored) && measure(run_fused, ignored);
    }
    // Alternating the order spreads clock and thermal drift over both variants.
    for (int i = 0; i < samples && passed; ++i) {
        passed = i % 2 == 0 ? measure(run_separate, separate_ms) && measure(run_fused, fused_ms)
                            : measure(run_fused, fused_ms) && measure(run_separate, separate_ms);
    }
    std::vector<float> one(pixels);
    std::vector<float> two(pixels);
    ts::DeviceResult status[3];
    passed =
        passed &&
        ok(cudaMemcpyAsync(one.data(), separate.data(), pixels * 4, cudaMemcpyDeviceToHost, stream),
           "download") &&
        ok(cudaMemcpyAsync(two.data(), fused.data(), pixels * 4, cudaMemcpyDeviceToHost, stream),
           "download") &&
        ok(cudaMemcpyAsync(status, records, sizeof(status), cudaMemcpyDeviceToHost, stream),
           "download");
    passed = ok(cuda::synchronize(stream), "synchronize") && passed;
    static_cast<void>(cudaEventDestroy(start));
    static_cast<void>(cudaEventDestroy(stop));
    if (!passed) {
        return false;
    }
    for (const auto &record : status) {
        if (!ok(record.status, "device result")) {
            return false;
        }
    }
    if (std::memcmp(one.data(), two.data(), pixels * 4) != 0) {
        std::fprintf(stderr, "separate and fused outputs differ\n");
        return false;
    }

    // Minimal DRAM traffic: the separate path also writes and reads the decoded region.
    const auto payload = payload_bytes(metadata, roi);
    const auto fused_bytes = static_cast<double>(payload + pixels * 4 * (maps ? 3 : 1));
    const auto separate_bytes = fused_bytes + static_cast<double>(2 * pixels * sample_bytes);
    const auto a = summarize(separate_ms);
    const auto b = summarize(fused_ms);
    std::printf("%-7s %-11s %-6s %8.3f %8.3f %8.3f %8.3f %7.2f %9.1f %9.1f %7.1f %7.1f\n",
                type == ts::PixelType::uint16 ? "uint16" : "float32",
                full ? "full-4096" : "roi-1024", maps ? "maps" : "scalar", a.median, a.p95,
                b.median, b.p95, a.median / b.median, separate_bytes / 1e6, fused_bytes / 1e6,
                separate_bytes / a.median / 1e6, fused_bytes / b.median / 1e6);
    return true;
}

} // namespace

int main(int argc, char **argv) {
    const int samples = argc > 1 ? std::max(1, std::atoi(argv[1])) : 30;
    cudaDeviceProp properties{};
    if (!ok(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) {
        return 1;
    }
    cudaStream_t stream = nullptr;
    if (!ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
        return 1;
    }
    std::printf("device: %s, %d samples after %d warm-up runs, times in ms (CUDA events)\n",
                properties.name, samples, warmup);
    std::printf("%-7s %-11s %-6s %8s %8s %8s %8s %7s %9s %9s %7s %7s\n", "type", "region", "params",
                "sep", "sep-p95", "fused", "fus-p95", "speedup", "sep-MB", "fused-MB", "sep-GB/s",
                "fus-GB/s");
    bool passed = true;
    for (const auto type : {ts::PixelType::uint16, ts::PixelType::float32}) {
        for (const bool full : {true, false}) {
            for (const bool maps : {false, true}) {
                passed = run_case(type, full, maps, samples, stream) && passed;
            }
        }
    }
    passed = ok(cudaStreamDestroy(stream), "cudaStreamDestroy") && passed;
    return passed ? 0 : 1;
}
