#include "cuda/device.cuh"

#include <cstdint>

namespace tilestream::cuda {
namespace {

using detail::block_threads;

constexpr std::uint32_t warps = block_threads / 32;

// One warp per row of the concatenated batch, so narrow images do not leave whole blocks idle.
template <typename Sample>
__global__ void __launch_bounds__(block_threads)
    calibrate(const DeviceImage *images, const std::uint64_t *first_rows, std::uint32_t count,
              std::uint64_t total) {
    const auto lane = threadIdx.x % 32;
    const std::uint64_t stride = std::uint64_t{gridDim.x} * warps;
    for (std::uint64_t row = std::uint64_t{blockIdx.x} * warps + threadIdx.x / 32; row < total;
         row += stride) {
        std::uint32_t low = 0;
        std::uint32_t high = count;
        while (high - low > 1) {
            const auto middle = low + (high - low) / 2;
            if (first_rows[middle] <= row) {
                low = middle;
            } else {
                high = middle;
            }
        }
        const auto &image = images[low];
        const auto y = row - first_rows[low];
        const auto *input = reinterpret_cast<const Sample *>(
            static_cast<const std::uint8_t *>(image.input.data) + y * image.input.pitch_bytes);
        auto *output = reinterpret_cast<float *>(static_cast<std::uint8_t *>(image.output.data) +
                                                 y * image.output.pitch_bytes);
        for (std::uint32_t x = lane; x < image.input.image.width; x += 32) {
            output[x] =
                detail::calibrate(detail::sample_value<Sample>(input[x]), image.calibration, x, y);
        }
    }
}

__global__ void report(const DeviceImage *images, std::uint32_t count, DeviceResult *results) {
    const std::uint64_t stride = std::uint64_t{gridDim.x} * blockDim.x;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const auto &image = images[i].output.image;
        results[i] = {Status::success, 0, std::uint64_t{image.width} * image.height * 4};
    }
}

template <typename Sample>
Status run(const detail::Plan &plan, DeviceResult *results, cudaStream_t stream) noexcept {
    const auto count = static_cast<std::uint32_t>(plan.count);
    const auto rows = (plan.total_rows + warps - 1) / warps;
    const auto blocks = static_cast<std::uint32_t>(rows < 65536 ? rows : 65536);
    calibrate<Sample><<<blocks, block_threads, 0, stream>>>(plan.images, plan.first_rows, count,
                                                            plan.total_rows);
    const auto reports = count / block_threads + 1 < 1024 ? count / block_threads + 1 : 1024;
    report<<<reports, block_threads, 0, stream>>>(plan.images, count, results);
    return detail::cuda_status(cudaGetLastError());
}

template <typename Sample> Status load() noexcept {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, calibrate<Sample>) != cudaSuccess ||
        cudaFuncGetAttributes(&attributes, report) != cudaSuccess) {
        return detail::cuda_failure();
    }
    return Status::success;
}

} // namespace

namespace detail {

Status load_calibration(PixelType type) noexcept {
    return type == PixelType::uint16 ? load<std::uint16_t>() : load<std::uint32_t>();
}

} // namespace detail

Status calibrate_async(const BatchPlan &plan, MutableBytes workspace, ResultBuffer results,
                       cudaStream_t stream) noexcept {
    const detail::Plan *state = plan.impl_.get();
    const auto status =
        detail::check_execution(state, Operation::calibrate, workspace, results, stream);
    if (status != Status::success || state->count == 0) {
        return status;
    }
    return state->type == PixelType::uint16 ? run<std::uint16_t>(*state, results.data, stream)
                                            : run<std::uint32_t>(*state, results.data, stream);
}

} // namespace tilestream::cuda
