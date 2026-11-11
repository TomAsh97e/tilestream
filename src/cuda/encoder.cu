#include <tilestream/format.h>

#include "cuda/device.cuh"

#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

#include <cstdint>

namespace tilestream::cuda {
namespace {

using detail::block_threads;
using detail::Geometry;
using detail::max_rows;
using detail::store_le;
using detail::tile_pixels;
using detail::WindowImage;

struct TileRecord {
    std::uint64_t tile;
    std::uint64_t length;
    TileMode mode;
    std::uint16_t width;
    std::uint16_t height;
    std::uint64_t reserved;
};
static_assert(sizeof(TileRecord) == 32);

struct RowRecord {
    std::uint32_t offset; // Segment start in the tile payload.
    std::uint32_t bits;
};
static_assert(sizeof(RowRecord) == format::segment_header_bytes);

struct Workspace {
    std::uint32_t *residuals;
    RowRecord *rows;
    TileRecord *tiles;
    std::uint64_t *lengths;
    std::uint64_t *offsets; // offsets[t] is the window prefix before tile t.
    std::uint64_t *carry;   // Output position of an image continued by the next window.
    void *temp;
    std::size_t temp_bytes;
    std::uint32_t row_stride;
};

__device__ std::uint64_t payload_offset(const DeviceImage &image) {
    return format::header_bytes + image.tile_count * format::index_entry_bytes;
}

template <typename Sample>
__device__ std::uint32_t residual(std::uint32_t current, std::uint32_t previous) {
    if constexpr (sizeof(Sample) == 2) {
        const auto delta = (current - previous) & 65535U;
        return delta < 32768 ? delta * 2 : (65536 - delta) * 2 - 1;
    } else {
        return current ^ previous;
    }
}

template <typename Sample>
__global__ void __launch_bounds__(block_threads)
    analyze(const DeviceImage *images, const WindowImage *entries, std::uint32_t entry_count,
            Workspace work) {
    using Scan = cub::BlockScan<std::uint32_t, block_threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ std::uint32_t widths[max_rows];

    const auto tile = blockIdx.x;
    const auto &entry = detail::owner(entries, entry_count, tile);
    const auto &image = images[entry.image];
    const auto index = entry.start + (tile - entry.first);
    const auto g = detail::tile_geometry(image.archive_image, image.archive_tile, index);
    const auto pitch = image.input.pitch_bytes;
    const auto *origin =
        static_cast<const std::uint8_t *>(image.input.data) + g.y * pitch + g.x * sizeof(Sample);
    const std::uint32_t first = *reinterpret_cast<const Sample *>(origin);
    auto *residuals = work.residuals + std::uint64_t{tile} * tile_pixels;
    const auto warp = threadIdx.x / 32;
    const auto lane = threadIdx.x % 32;

    // One warp per row; column zero keeps the seed in the residual slot.
    bool constant = true;
    for (auto row = warp; row < g.height; row += block_threads / 32) {
        const auto *line = reinterpret_cast<const Sample *>(origin + row * pitch);
        std::uint32_t combined = 0;
        for (auto col = lane; col < g.width; col += 32) {
            const std::uint32_t current = line[col];
            auto value = current;
            constant = constant && current == first;
            if (col != 0) {
                value = residual<Sample>(current, line[col - 1]);
                combined |= value;
            }
            residuals[row * g.width + col] = value;
        }
        for (auto offset = 16; offset > 0; offset /= 2) {
            combined |= __shfl_xor_sync(0xffffffffU, combined, offset);
        }
        if (lane == 0) {
            widths[row] = static_cast<std::uint32_t>(32 - __clz(static_cast<int>(combined)));
        }
    }
    const bool uniform = __syncthreads_and(constant) != 0;

    std::uint32_t cost = 0;
    if (threadIdx.x < g.height) {
        cost = 8 + ((g.width - 1) * widths[threadIdx.x] + 31) / 32 * 4;
    }
    std::uint32_t offset = 0;
    std::uint32_t predictive = 0;
    Scan(scan).ExclusiveSum(cost, offset, predictive);
    if (threadIdx.x < g.height) {
        work.rows[std::uint64_t{tile} * work.row_stride + threadIdx.x] = {offset,
                                                                          widths[threadIdx.x]};
    }
    if (threadIdx.x == 0) {
        // Same candidate costs and tie order as the CPU encoder: RAW, CONSTANT, PREDICT_BITPACK.
        auto mode = TileMode::raw;
        std::uint64_t length = (std::uint64_t{g.width} * g.height * sizeof(Sample) + 3) / 4 * 4;
        if (uniform && 4 < length) {
            mode = TileMode::constant;
            length = 4;
        }
        if (predictive < length) {
            mode = TileMode::predict_bitpack;
            length = predictive;
        }
        work.tiles[tile] = {index,
                            length,
                            mode,
                            static_cast<std::uint16_t>(g.width),
                            static_cast<std::uint16_t>(g.height),
                            0};
        work.lengths[tile] = length;
    }
}

template <typename Sample>
__device__ std::uint32_t raw_word(const DeviceImage &image, const Geometry &g, std::uint32_t word) {
    const auto pitch = image.input.pitch_bytes;
    const auto *origin =
        static_cast<const std::uint8_t *>(image.input.data) + g.y * pitch + g.x * sizeof(Sample);
    std::uint32_t value = 0;
    constexpr std::uint32_t per_word = 4 / sizeof(Sample);
    for (std::uint32_t i = 0; i < per_word; ++i) {
        const auto sample = word * per_word + i;
        if (sample < g.width * g.height) {
            const auto *line = reinterpret_cast<const Sample *>(origin + sample / g.width * pitch);
            value |= std::uint32_t{line[sample % g.width]} << (16 * i);
        }
    }
    return value;
}

// Gathers every residual overlapping one output word, so each word has a single writer.
__device__ std::uint32_t packed_word(const std::uint32_t *residuals, std::uint32_t count,
                                     std::uint32_t bits, std::uint32_t word) {
    if (bits == 32) {
        return residuals[word];
    }
    const auto first_bit = word * 32;
    const auto end = (first_bit + 31) / bits;
    const auto last = end < count - 1 ? end : count - 1;
    std::uint32_t value = 0;
    for (auto j = first_bit / bits; j <= last; ++j) {
        const auto position = j * bits;
        value |= position >= first_bit ? residuals[j] << (position - first_bit)
                                       : residuals[j] >> (first_bit - position);
    }
    return value;
}

template <typename Sample>
__global__ void __launch_bounds__(block_threads)
    pack(const DeviceImage *images, const WindowImage *entries, std::uint32_t entry_count,
         Workspace work, std::uint32_t parity) {
    __shared__ RowRecord rows[max_rows];

    const auto tile = blockIdx.x;
    const auto &entry = detail::owner(entries, entry_count, tile);
    const auto &image = images[entry.image];
    const auto record = work.tiles[tile];
    const auto g = detail::tile_geometry(image.archive_image, image.archive_tile, record.tile);
    const auto base =
        (entry.flags & detail::continues) != 0 ? work.carry[parity] : payload_offset(image);
    const auto offset = base + work.offsets[tile] - work.offsets[entry.first];
    auto *file = static_cast<std::uint8_t *>(image.compressed_output.data);

    if (threadIdx.x == 0) {
        auto *slot = file + format::header_bytes + record.tile * format::index_entry_bytes;
        store_le(slot + format::entry::mode, static_cast<std::uint8_t>(record.mode), 1);
        store_le(slot + format::entry::reserved, 0, 3);
        store_le(slot + format::entry::width, record.width, 2);
        store_le(slot + format::entry::height, record.height, 2);
        store_le(slot + format::entry::offset, offset, 8);
        store_le(slot + format::entry::length, record.length, 8);
        store_le(slot + format::entry::reserved_tail, 0, 8);
    }
    if (threadIdx.x < g.height) {
        rows[threadIdx.x] = work.rows[std::uint64_t{tile} * work.row_stride + threadIdx.x];
    }
    __syncthreads();

    auto *words = reinterpret_cast<std::uint32_t *>(file + offset);
    const auto *residuals = work.residuals + std::uint64_t{tile} * tile_pixels;
    const auto count = static_cast<std::uint32_t>(record.length / 4);
    for (auto k = threadIdx.x; k < count; k += block_threads) {
        std::uint32_t value = 0;
        if (record.mode == TileMode::raw) {
            value = raw_word<Sample>(image, g, k);
        } else if (record.mode == TileMode::constant) {
            value = residuals[0];
        } else {
            std::uint32_t low = 0;
            std::uint32_t high = g.height;
            while (high - low > 1) {
                const auto middle = low + (high - low) / 2;
                if (rows[middle].offset <= k * 4) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            const auto local = k * 4 - rows[low].offset;
            const auto *line = residuals + low * g.width;
            if (local == 0) {
                value = line[0];
            } else if (local == 4) {
                value = rows[low].bits;
            } else {
                value = packed_word(line + 1, g.width - 1, rows[low].bits, (local - 8) / 4);
            }
        }
        words[k] = value;
    }
}

__global__ void finish(const DeviceImage *images, const WindowImage *entries,
                       std::uint32_t entry_count, Workspace work, std::uint32_t parity,
                       DeviceResult *results) {
    const auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= entry_count) {
        return;
    }
    const auto entry = entries[j];
    const auto &image = images[entry.image];
    // Only the first entry can continue and only the last can be continued, so the
    // parity slots keep the previous and the next carry apart.
    const auto base =
        (entry.flags & detail::continues) != 0 ? work.carry[parity] : payload_offset(image);
    const auto end = base + work.offsets[entry.first + entry.count] - work.offsets[entry.first];
    if ((entry.flags & detail::finishes) == 0) {
        work.carry[parity ^ 1] = end;
        return;
    }
    auto *file = static_cast<std::uint8_t *>(image.compressed_output.data);
    constexpr std::uint8_t magic[] = {'T', 'S', 'C', 0, 13, 10, 26, 10};
    for (std::uint32_t i = 0; i < sizeof(magic); ++i) {
        file[i] = magic[i];
    }
    store_le(file + format::header::version_major, format::major_version, 2);
    store_le(file + format::header::version_minor, format::minor_version, 2);
    store_le(file + format::header::size, format::header_bytes, 2);
    store_le(file + format::header::pixel_type, static_cast<std::uint8_t>(image.archive_image.type),
             1);
    store_le(file + format::header::flags, 0, 1);
    store_le(file + format::header::width, image.archive_image.width, 4);
    store_le(file + format::header::height, image.archive_image.height, 4);
    store_le(file + format::header::tile_width, image.archive_tile.width, 2);
    store_le(file + format::header::tile_height, image.archive_tile.height, 2);
    store_le(file + format::header::entry_size, format::index_entry_bytes, 2);
    store_le(file + format::header::reserved, 0, 2);
    store_le(file + format::header::tile_count, image.tile_count, 8);
    store_le(file + format::header::index_offset, format::header_bytes, 8);
    store_le(file + format::header::payload_offset, payload_offset(image), 8);
    store_le(file + format::header::file_size, end, 8);
    results[entry.image] = {Status::success, 0, end};
}

// Every scan goes through here, so the size query, the kernel preload and execution all use
// the same CUB instantiation.
cudaError_t scan(void *temp, std::size_t &bytes, const std::uint64_t *lengths,
                 std::uint64_t *offsets, std::uint32_t items, cudaStream_t stream) {
    return cub::DeviceScan::InclusiveSum(temp, bytes, lengths, offsets, static_cast<int>(items),
                                         stream);
}

template <typename Sample>
Status run(const detail::Plan &plan, void *workspace, DeviceResult *results,
           cudaStream_t stream) noexcept {
    auto *base = static_cast<std::uint8_t *>(workspace);
    const auto &layout = plan.layout;
    const Workspace work = {reinterpret_cast<std::uint32_t *>(base + layout.residuals),
                            reinterpret_cast<RowRecord *>(base + layout.rows),
                            reinterpret_cast<TileRecord *>(base + layout.tiles),
                            reinterpret_cast<std::uint64_t *>(base + layout.lengths),
                            reinterpret_cast<std::uint64_t *>(base + layout.offsets),
                            reinterpret_cast<std::uint64_t *>(base + layout.images),
                            base + layout.temp,
                            static_cast<std::size_t>(plan.spec.scan_temp_bytes),
                            plan.spec.tile.height};
    if (cudaMemsetAsync(work.offsets, 0, sizeof(std::uint64_t), stream) != cudaSuccess) {
        return detail::cuda_failure();
    }
    for (std::size_t k = 0; k < plan.windows.size(); ++k) {
        const auto &window = plan.windows[k];
        const auto parity = static_cast<std::uint32_t>(k % 2);
        analyze<Sample><<<window.tile_count, block_threads, 0, stream>>>(plan.images, window.images,
                                                                         window.image_count, work);
        auto temp_bytes = work.temp_bytes;
        if (cudaGetLastError() != cudaSuccess ||
            scan(work.temp, temp_bytes, work.lengths, work.offsets + 1, window.tile_count,
                 stream) != cudaSuccess) {
            return detail::cuda_failure();
        }
        pack<Sample><<<window.tile_count, block_threads, 0, stream>>>(
            plan.images, window.images, window.image_count, work, parity);
        finish<<<(window.image_count + block_threads - 1) / block_threads, block_threads, 0,
                 stream>>>(plan.images, window.images, window.image_count, work, parity, results);
        if (cudaGetLastError() != cudaSuccess) {
            return detail::cuda_failure();
        }
    }
    return Status::success;
}

template <typename Sample> Status load(cudaStream_t stream) noexcept {
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, analyze<Sample>) != cudaSuccess ||
        cudaFuncGetAttributes(&attributes, pack<Sample>) != cudaSuccess ||
        cudaFuncGetAttributes(&attributes, finish) != cudaSuccess) {
        return detail::cuda_failure();
    }
    // CUB does not expose its scan kernels, so a one-item scan loads them.
    std::size_t bytes = 0;
    void *memory = nullptr;
    if (scan(nullptr, bytes, nullptr, nullptr, 1, stream) != cudaSuccess ||
        cudaMallocAsync(&memory, 256 + bytes, stream) != cudaSuccess) {
        return detail::cuda_failure();
    }
    auto *words = static_cast<std::uint64_t *>(memory);
    const bool done =
        cudaMemsetAsync(memory, 0, 2 * sizeof(std::uint64_t), stream) == cudaSuccess &&
        scan(words + 32, bytes, words, words + 1, 1, stream) == cudaSuccess;
    const bool freed = cudaFreeAsync(memory, stream) == cudaSuccess;
    return done && freed ? Status::success : detail::cuda_failure();
}

} // namespace

namespace detail {

Status scan_temp_bytes(std::uint32_t items, std::uint64_t &bytes) noexcept {
    std::size_t size = 0;
    if (scan(nullptr, size, nullptr, nullptr, items, nullptr) != cudaSuccess) {
        return cuda_failure();
    }
    bytes = size;
    return Status::success;
}

Status load_encoder(PixelType type, cudaStream_t stream) noexcept {
    return type == PixelType::uint16 ? load<std::uint16_t>(stream) : load<std::uint32_t>(stream);
}

} // namespace detail

Status compress_async(const BatchPlan &plan, MutableBytes workspace, ResultBuffer results,
                      cudaStream_t stream) noexcept {
    const detail::Plan *state = plan.impl_.get();
    const auto status =
        detail::check_execution(state, Operation::compress, workspace, results, stream);
    if (status != Status::success || state->count == 0) {
        return status;
    }
    return state->type == PixelType::uint16
               ? run<std::uint16_t>(*state, workspace.data, results.data, stream)
               : run<std::uint32_t>(*state, workspace.data, results.data, stream);
}

} // namespace tilestream::cuda
