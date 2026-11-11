#pragma once

#include <tilestream/cuda.h>

#include "checked_size.h"

#include <cstdint>
#include <vector>

namespace tilestream::cuda::detail {

inline constexpr std::uint32_t block_threads = 256;
// Bounds one window's workspace; larger batches are processed as several ordered windows.
inline constexpr std::uint32_t window_tiles = 4096;
inline constexpr std::uint32_t continues = 1;
inline constexpr std::uint32_t finishes = 2;

// The part of one image that falls into a window.
struct WindowImage {
    std::uint32_t image = 0;
    std::uint32_t first = 0; // First window tile of this image.
    std::uint32_t count = 0;
    std::uint32_t flags = 0;
    std::uint64_t start = 0; // Position of that tile in the image's own tile sequence.
};

struct Window {
    const WindowImage *images = nullptr;
    std::uint32_t image_count = 0;
    std::uint32_t tile_count = 0;
};

struct Range {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

// Writable ranges are disjoint; readable ranges may overlap each other but no writable range.
class RangeSet {
  public:
    [[nodiscard]] bool assign(std::vector<Range> writable, std::vector<Range> readable);
    [[nodiscard]] bool disjoint(Range range) const noexcept;

  private:
    std::vector<Range> writable_;
    std::vector<Range> readable_;
    std::vector<std::uintptr_t> reach_;
};

struct Plan {
    Operation operation = Operation::compress;
    PixelType type = PixelType::uint16;
    int device = 0;
    bool pageable = false;
    std::uint64_t count = 0;
    WorkspaceSpec spec;
    tilestream::detail::WorkspaceLayout layout;
    std::vector<Window> windows;
    RangeSet ranges;
    void *storage = nullptr;
    const DeviceImage *images = nullptr;
    // Standalone calibration: first row of each image in the concatenated rows of the batch.
    const std::uint64_t *first_rows = nullptr;
    std::uint64_t total_rows = 0;

    Plan() = default;
    Plan(const Plan &) = delete;
    Plan &operator=(const Plan &) = delete;
    ~Plan();
};

[[nodiscard]] Status scan_temp_bytes(std::uint32_t items, std::uint64_t &bytes) noexcept;
// A kernel loaded lazily at its first launch can wait for the whole device, so preparation
// loads every kernel the plan launches.
[[nodiscard]] Status load_encoder(PixelType type, cudaStream_t stream) noexcept;
[[nodiscard]] Status load_decoder(PixelType type, bool calibrated) noexcept;
[[nodiscard]] Status load_calibration(PixelType type) noexcept;
[[nodiscard]] Status accessible(const void *data, int device, bool pageable) noexcept;
// Host-side checks shared by all execution functions, before any work is enqueued.
[[nodiscard]] Status check_execution(const Plan *plan, Operation operation, MutableBytes workspace,
                                     ResultBuffer results, cudaStream_t stream) noexcept;

// The error is reported through Status, so it is cleared and cannot be blamed on a later launch.
inline Status cuda_failure() noexcept {
    static_cast<void>(cudaGetLastError());
    return Status::cuda_error;
}

inline Status cuda_status(cudaError_t error) noexcept {
    return error == cudaSuccess ? Status::success : cuda_failure();
}

} // namespace tilestream::cuda::detail

namespace tilestream::cuda {

struct BatchPlan::Impl final : detail::Plan {};

} // namespace tilestream::cuda
