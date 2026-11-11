#pragma once

#include <cstdint>
#include <string>

namespace tilestream::detail {

struct CudaInfo {
    bool available = false;
    int device_count = 0;
    int driver_version = 0;
    int runtime_version = 0;
    std::string status;
    std::string device_name;
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t memory_bytes = 0;
};

[[nodiscard]] CudaInfo query_cuda_info();

} // namespace tilestream::detail
