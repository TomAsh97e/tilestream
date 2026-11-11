#include <tilestream/format.h>

#include "cuda/device.cuh"

#include <cstdint>
#include <type_traits>

namespace tilestream::cuda {
namespace {

using detail::block_threads;
using detail::load_le;
using detail::max_rows;
using detail::tile_pixels;
using detail::WindowImage;

constexpr std::uint32_t full_mask = 0xffffffffU;

__device__ void fail(DeviceResult *results, std::uint32_t image, Status status) {
    atomicCAS(reinterpret_cast<unsigned int *>(&results[image].status),
              static_cast<unsigned int>(Status::success), static_cast<unsigned int>(status));
}

__device__ std::uint32_t unpack(const std::uint32_t *words, std::uint32_t index,
                                std::uint32_t bits) {
    if (bits == 0) {
        return 0;
    }
    if (bits == 32) {
        return words[index];
    }
    const auto position = index * bits;
    const auto shift = position % 32;
    auto value = words[position / 32] >> shift;
    if (shift + bits > 32) {
        value |= words[position / 32 + 1] << (32 - shift);
    }
    return value & ((1U << bits) - 1);
}

// Residual turned into the term of an inclusive scan whose first term is the seed.
template <typename Sample> __device__ std::uint32_t term(std::uint32_t residual) {
    if constexpr (sizeof(Sample) == 2) {
        // Inverse zigzag as an unsigned delta modulo 65536.
        return (residual & 1U) != 0 ? 65535 - (residual >> 1) : residual >> 1;
    } else {
        return residual;
    }
}

template <typename Sample> __device__ std::uint32_t combine(std::uint32_t a, std::uint32_t b) {
    if constexpr (sizeof(Sample) == 2) {
        return a + b;
    } else {
        return a ^ b;
    }
}

template <typename Sample> __device__ std::uint32_t uncombine(std::uint32_t a, std::uint32_t b) {
    if constexpr (sizeof(Sample) == 2) {
        return a - b;
    } else {
        return a ^ b;
    }
}

// Every lane of the warp must call this, including lanes past the end of a row.
template <typename Sample> __device__ std::uint32_t warp_scan(std::uint32_t value) {
    const auto lane = threadIdx.x % 32;
    for (std::uint32_t offset = 1; offset < 32; offset *= 2) {
        const auto other = __shfl_up_sync(full_mask, value, offset);
        if (lane >= offset) {
            value = combine<Sample>(other, value);
        }
    }
    return value;
}

struct Rows {
    std::uint32_t seeds[max_rows];
    std::uint32_t widths[max_rows];
    std::uint32_t starts[max_rows]; // Packed residuals, as word indices in the staged payload.
};

// Reads the row descriptors of an encoder-produced tile, which has no host metadata.
template <typename Sample>
__device__ bool walk_rows(const std::uint32_t *payload, std::uint32_t words, std::uint32_t width,
                          std::uint32_t height, Rows &rows) {
    constexpr std::uint32_t maximum_bits = sizeof(Sample) * 8;
    std::uint32_t cursor = 0;
    for (std::uint32_t row = 0; row < height; ++row) {
        if (words - cursor < 2) {
            return false;
        }
        const auto seed = payload[cursor];
        const auto bits = payload[cursor + 1];
        if (bits > maximum_bits || (sizeof(Sample) == 2 && seed > 65535)) {
            return false;
        }
        const auto packed = ((width - 1) * bits + 31) / 32;
        if (words - cursor - 2 < packed) {
            return false;
        }
        rows.seeds[row] = seed;
        rows.widths[row] = bits;
        rows.starts[row] = cursor + 2;
        cursor += 2 + packed;
    }
    return cursor == words;
}

template <typename Output, std::uint32_t Size> struct alignas(Size * sizeof(Output)) Group {
    Output values[Size];
};

// One bitpacked row: every lane scans a group of Size consecutive columns in registers, so the
// warp scans one total per lane, and an aligned group is written with one vector store. x and y
// place column 0 of the row in the output region; line points at column left of the row.
template <std::uint32_t Size, typename Sample, bool Calibrated, typename Output>
__device__ void decode_row(const std::uint32_t *words, std::uint32_t seed, std::uint32_t bits,
                           std::uint32_t width, std::uint32_t left, std::uint32_t right,
                           Output *line, const Calibration &calibration, std::uint64_t x,
                           std::uint64_t y) {
    const auto lane = threadIdx.x % 32;
    std::uint32_t carry = 0;
    for (std::uint32_t base = 0; base < right; base += 32 * Size) {
        const auto first = base + lane * Size;
        std::uint32_t prefix[Size];
        std::uint32_t total = 0;
#pragma unroll
        for (std::uint32_t k = 0; k < Size; ++k) {
            const auto col = first + k;
            std::uint32_t item = 0;
            if (col == 0) {
                item = seed;
            } else if (col < width) {
                item = term<Sample>(unpack(words, col - 1, bits));
            }
            total = combine<Sample>(total, item);
            prefix[k] = total;
        }
        const auto inclusive = warp_scan<Sample>(total);
        const auto before = combine<Sample>(carry, uncombine<Sample>(inclusive, total));
        carry = combine<Sample>(carry, __shfl_sync(full_mask, inclusive, 31));
        Group<Output, Size> values;
#pragma unroll
        for (std::uint32_t k = 0; k < Size; ++k) {
            // The uint16 scan is modulo 65536 only after this narrowing.
            const auto sample = static_cast<Sample>(combine<Sample>(before, prefix[k]));
            if constexpr (Calibrated) {
                values.values[k] = detail::calibrate(detail::sample_value<Sample>(sample),
                                                     calibration, x + first + k, y);
            } else {
                values.values[k] = sample;
            }
        }
        if (first >= left && first + Size <= right &&
            reinterpret_cast<std::uintptr_t>(line + (first - left)) % sizeof(Group<Output, Size>) ==
                0) {
            *reinterpret_cast<Group<Output, Size> *>(line + (first - left)) = values;
        } else {
#pragma unroll
            for (std::uint32_t k = 0; k < Size; ++k) {
                if (first + k >= left && first + k < right) {
                    line[first + k - left] = values.values[k];
                }
            }
        }
    }
}

// The calibrated variant writes FP32 output computed from each decoded sample.
template <typename Sample, bool Calibrated>
__global__ void __launch_bounds__(block_threads)
    decode(const DeviceImage *images, const WindowImage *entries, std::uint32_t entry_count,
           DeviceResult *results) {
    __shared__ std::uint32_t payload[tile_pixels];
    __shared__ Rows rows;
    __shared__ bool valid;

    const auto &entry = detail::owner(entries, entry_count, blockIdx.x);
    const auto &image = images[entry.image];
    auto tile = image.tiles[entry.start + (blockIdx.x - entry.first)];
    const auto g = detail::tile_geometry(image.archive_image, image.archive_tile, tile.tile_index);
    const auto *bytes = static_cast<const std::uint8_t *>(image.compressed_input.data);
    const bool trusted = image.encoder_result != nullptr;

    if (trusted) {
        // The producer's result and index are read only after it completed; every thread
        // sees the same values, so the early returns are uniform for the block.
        const auto produced = *image.encoder_result;
        if (produced.status != Status::success) {
            return;
        }
        const auto *slot =
            bytes + format::header_bytes + tile.tile_index * format::index_entry_bytes;
        const auto mode = slot[format::entry::mode];
        const auto offset = load_le(slot + format::entry::offset, 8);
        const auto length = load_le(slot + format::entry::length, 8);
        const auto end = produced.bytes_written < image.compressed_input.size
                             ? produced.bytes_written
                             : image.compressed_input.size;
        const auto raw = (std::uint64_t{g.width} * g.height * sizeof(Sample) + 3) / 4 * 4;
        if (mode > static_cast<std::uint8_t>(TileMode::predict_bitpack) ||
            load_le(slot + format::entry::width, 2) != g.width ||
            load_le(slot + format::entry::height, 2) != g.height || offset % 4 != 0 ||
            length % 4 != 0 || length == 0 || length > raw || offset > end ||
            length > end - offset) {
            if (threadIdx.x == 0) {
                fail(results, entry.image, Status::malformed_data);
            }
            return;
        }
        tile.mode = static_cast<TileMode>(mode);
        tile.input_offset = offset;
        tile.length = length;
    }

    const auto words = static_cast<std::uint32_t>(tile.length / 4);
    const auto *source = reinterpret_cast<const std::uint32_t *>(bytes + tile.input_offset);
    for (auto k = threadIdx.x; k < words; k += block_threads) {
        payload[k] = source[k];
    }
    if (!trusted && tile.mode == TileMode::predict_bitpack && threadIdx.x < g.height) {
        const auto segment = image.segments[tile.first_segment + threadIdx.x];
        rows.seeds[threadIdx.x] = segment.seed;
        rows.widths[threadIdx.x] = segment.bit_width;
        rows.starts[threadIdx.x] =
            static_cast<std::uint32_t>((segment.input_offset - tile.input_offset) / 4);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        valid = !trusted || tile.mode != TileMode::predict_bitpack ||
                walk_rows<Sample>(payload, words, g.width, g.height, rows);
    }
    __syncthreads();
    if (!valid) {
        if (threadIdx.x == 0) {
            fail(results, entry.image, Status::malformed_data);
        }
        return;
    }

    // Only rows and columns inside the region are written; prediction still starts at column 0.
    const auto &roi = image.roi;
    const auto top = roi.y > g.y ? roi.y - g.y : 0;
    const auto bottom = roi.y + roi.height < g.y + g.height ? roi.y + roi.height - g.y : g.height;
    const auto left = roi.x > g.x ? static_cast<std::uint32_t>(roi.x - g.x) : 0U;
    const auto right = roi.x + roi.width < g.x + g.width
                           ? static_cast<std::uint32_t>(roi.x + roi.width - g.x)
                           : g.width;
    using Output = std::conditional_t<Calibrated, float, Sample>;
    auto *output = static_cast<std::uint8_t *>(image.output.data);
    const auto lane = threadIdx.x % 32;
    for (auto row = static_cast<std::uint32_t>(top) + threadIdx.x / 32; row < bottom;
         row += block_threads / 32) {
        auto *line =
            reinterpret_cast<Output *>(output + (g.y + row - roi.y) * image.output.pitch_bytes +
                                       (g.x + left - roi.x) * sizeof(Output));
        if (tile.mode == TileMode::predict_bitpack) {
            // The group size keeps every lane busy on narrow tiles.
            const auto *words = payload + rows.starts[row];
            const auto x = g.x - roi.x;
            const auto y = g.y + row - roi.y;
            if (g.width > 64) {
                decode_row<4, Sample, Calibrated>(words, rows.seeds[row], rows.widths[row], g.width,
                                                  left, right, line, image.calibration, x, y);
            } else if (g.width > 32) {
                decode_row<2, Sample, Calibrated>(words, rows.seeds[row], rows.widths[row], g.width,
                                                  left, right, line, image.calibration, x, y);
            } else {
                decode_row<1, Sample, Calibrated>(words, rows.seeds[row], rows.widths[row], g.width,
                                                  left, right, line, image.calibration, x, y);
            }
            continue;
        }
        for (auto base = left / 32 * 32; base < right; base += 32) {
            const auto col = base + lane;
            std::uint32_t value = 0;
            if (col < right && tile.mode == TileMode::raw) {
                const auto index = row * g.width + col;
                if constexpr (sizeof(Sample) == 2) {
                    value = payload[index / 2] >> (index % 2 * 16) & 65535U;
                } else {
                    value = payload[index];
                }
            } else if (col < right) {
                value = payload[0];
            }
            if (col >= left && col < right) {
                const auto sample = static_cast<Sample>(value);
                if constexpr (Calibrated) {
                    line[col - left] =
                        detail::calibrate(detail::sample_value<Sample>(sample), image.calibration,
                                          g.x + col - roi.x, g.y + row - roi.y);
                } else {
                    line[col - left] = sample;
                }
            }
        }
    }
}

__global__ void begin(std::uint32_t count, DeviceResult *results) {
    const std::uint64_t stride = std::uint64_t{gridDim.x} * blockDim.x;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        results[i] = {Status::success, 0, 0};
    }
}

__global__ void complete(const DeviceImage *images, std::uint32_t count, DeviceResult *results) {
    const std::uint64_t stride = std::uint64_t{gridDim.x} * blockDim.x;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const auto &image = images[i];
        auto status = results[i].status;
        if (image.encoder_result != nullptr && image.encoder_result->status != Status::success) {
            status = image.encoder_result->status;
        }
        const std::uint64_t sample_bytes = image.output.image.type == PixelType::uint16 ? 2 : 4;
        const auto bytes = std::uint64_t{image.roi.width} * image.roi.height * sample_bytes;
        results[i] = {status, 0, status == Status::success ? bytes : 0};
    }
}

template <typename Sample, bool Calibrated>
Status run(const detail::Plan &plan, DeviceResult *results, cudaStream_t stream) noexcept {
    const auto count = static_cast<std::uint32_t>(plan.count);
    const auto blocks = count / block_threads + 1 < 1024 ? count / block_threads + 1 : 1024;
    begin<<<blocks, block_threads, 0, stream>>>(count, results);
    for (const auto &window : plan.windows) {
        decode<Sample, Calibrated><<<window.tile_count, block_threads, 0, stream>>>(
            plan.images, window.images, window.image_count, results);
    }
    complete<<<blocks, block_threads, 0, stream>>>(plan.images, count, results);
    return detail::cuda_status(cudaGetLastError());
}

template <typename Sample, bool Calibrated> Status load() noexcept {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, begin) != cudaSuccess ||
        cudaFuncGetAttributes(&attributes, decode<Sample, Calibrated>) != cudaSuccess ||
        cudaFuncGetAttributes(&attributes, complete) != cudaSuccess) {
        return detail::cuda_failure();
    }
    return Status::success;
}

Status enqueue(const detail::Plan *plan, Operation operation, MutableBytes workspace,
               ResultBuffer results, cudaStream_t stream) noexcept {
    const auto status = detail::check_execution(plan, operation, workspace, results, stream);
    if (status != Status::success || plan->count == 0) {
        return status;
    }
    if (operation == Operation::decode_calibrate) {
        return plan->type == PixelType::uint16
                   ? run<std::uint16_t, true>(*plan, results.data, stream)
                   : run<std::uint32_t, true>(*plan, results.data, stream);
    }
    return plan->type == PixelType::uint16 ? run<std::uint16_t, false>(*plan, results.data, stream)
                                           : run<std::uint32_t, false>(*plan, results.data, stream);
}

} // namespace

namespace detail {

Status load_decoder(PixelType type, bool calibrated) noexcept {
    if (calibrated) {
        return type == PixelType::uint16 ? load<std::uint16_t, true>()
                                         : load<std::uint32_t, true>();
    }
    return type == PixelType::uint16 ? load<std::uint16_t, false>() : load<std::uint32_t, false>();
}

} // namespace detail

Status decompress_async(const BatchPlan &plan, MutableBytes workspace, ResultBuffer results,
                        cudaStream_t stream) noexcept {
    return enqueue(plan.impl_.get(), Operation::decompress, workspace, results, stream);
}

Status decode_roi_async(const BatchPlan &plan, MutableBytes workspace, ResultBuffer results,
                        cudaStream_t stream) noexcept {
    return enqueue(plan.impl_.get(), Operation::decode_roi, workspace, results, stream);
}

Status decode_calibrate_async(const BatchPlan &plan, MutableBytes workspace, ResultBuffer results,
                              cudaStream_t stream) noexcept {
    return enqueue(plan.impl_.get(), Operation::decode_calibrate, workspace, results, stream);
}

} // namespace tilestream::cuda
