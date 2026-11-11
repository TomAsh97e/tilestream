#include "cuda/plan.h"

#include <cstdint>
#include <utility>

namespace tilestream::cuda {

DeviceBuffer::~DeviceBuffer() {
    if (data_ != nullptr) {
        static_cast<void>(detail::cuda_status(cudaFreeAsync(data_, cudaStreamPerThread)));
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
        DeviceBuffer old(std::move(*this));
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

Status DeviceBuffer::allocate(std::uint64_t bytes, cudaStream_t stream,
                              DeviceBuffer &buffer) noexcept {
    if (bytes > max_bytes) {
        return Status::size_overflow;
    }
    DeviceBuffer result;
    if (bytes != 0) {
        if (cudaMallocAsync(&result.data_, static_cast<std::size_t>(bytes), stream) !=
            cudaSuccess) {
            result.data_ = nullptr;
            return detail::cuda_failure();
        }
        result.size_ = bytes;
        if (reinterpret_cast<std::uintptr_t>(result.data_) % workspace_alignment != 0) {
            static_cast<void>(result.release(stream));
            return Status::cuda_error;
        }
    }
    const auto status = buffer.release(stream);
    buffer = std::move(result);
    return status;
}

Status DeviceBuffer::release(cudaStream_t stream) noexcept {
    if (data_ == nullptr) {
        return Status::success;
    }
    const auto error = cudaFreeAsync(data_, stream);
    data_ = nullptr;
    size_ = 0;
    return detail::cuda_status(error);
}

Status synchronize(cudaStream_t stream) noexcept {
    return detail::cuda_status(cudaStreamSynchronize(stream));
}

} // namespace tilestream::cuda
