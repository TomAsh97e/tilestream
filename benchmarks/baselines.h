#pragma once

// Baselines measured with the same boundaries as the TileStream scenarios. The adapters for zstd,
// libtiff and nvCOMP live here, outside the library. See docs/baselines.md.
#include "runner.h"

namespace bench {

// Raw device pixels of every image, their calibrated FP32 output and a prepared calibrate plan.
struct Target {
    std::vector<cuda::DeviceBuffer> pixels;
    std::vector<cuda::DeviceBuffer> outputs;
    std::unique_ptr<cuda::BatchPlan> plan;
    cuda::DeviceBuffer workspace;
    cuda::DeviceBuffer results;
};

bool prepare_target(const Workload &w, cudaStream_t stream, Target &t, Result &r);
bool calibrate(const Target &t, cudaStream_t stream);
// Compares every output bit for bit with the host calibration of the source pixels.
bool verify(const Workload &w, const std::vector<cuda::DeviceBuffer> &outputs,
            const cuda::DeviceBuffer &results, cudaStream_t stream);

// Host-resident input to calibrated FP32 output on the device.
bool raw_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r);
bool tsc_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r);
bool zstd_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r);
bool lz4_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r);

// Device-resident nvCOMP LZ4, next to gpu_encode and gpu_decode.
bool lz4_gpu(const Options &o, const Workload &w, bool compress, cudaStream_t stream, Result &r);

// A file in the page cache to calibrated FP32 output: raw_file, tsc_file or tiff_file.
bool file_to_result(const Options &o, const Workload &w, cudaStream_t stream, Result &r);

} // namespace bench
