#include "cuda/device_info.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace tilestream::detail {
namespace {

std::string cuda_status(cudaError_t error) {
    const char *text = cudaGetErrorString(error);
    return text != nullptr ? text : "unknown CUDA error";
}

} // namespace

CudaInfo query_cuda_info() {
    CudaInfo info;

    cudaError_t error = cudaDriverGetVersion(&info.driver_version);
    if (error != cudaSuccess) {
        info.status = "driver query failed: " + cuda_status(error);
        return info;
    }

    error = cudaRuntimeGetVersion(&info.runtime_version);
    if (error != cudaSuccess) {
        info.status = "runtime query failed: " + cuda_status(error);
        return info;
    }

    error = cudaGetDeviceCount(&info.device_count);
    if (error != cudaSuccess) {
        info.status = "device query failed: " + cuda_status(error);
        return info;
    }
    if (info.device_count == 0) {
        info.status = "no CUDA device found";
        return info;
    }

    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, 0);
    if (error != cudaSuccess) {
        info.status = "device properties query failed: " + cuda_status(error);
        return info;
    }

    info.available = true;
    info.status = "available";
    info.device_name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.memory_bytes = static_cast<std::uint64_t>(properties.totalGlobalMem);
    return info;
}

} // namespace tilestream::detail
