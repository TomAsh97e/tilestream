#include "cuda/plan.h"

#include <tilestream/format.h>

#include "cpu/views.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace tilestream::cuda {
namespace detail {

using cpu::detail::byte_range;
using tilestream::detail::add_size;
using tilestream::detail::align_size;
using tilestream::detail::multiply_size;

bool RangeSet::assign(std::vector<Range> writable, std::vector<Range> readable) {
    const auto empty = [](const Range &range) { return range.begin == range.end; };
    const auto by_begin = [](const Range &a, const Range &b) { return a.begin < b.begin; };
    writable.erase(std::remove_if(writable.begin(), writable.end(), empty), writable.end());
    readable.erase(std::remove_if(readable.begin(), readable.end(), empty), readable.end());
    std::sort(writable.begin(), writable.end(), by_begin);
    std::sort(readable.begin(), readable.end(), by_begin);
    for (std::size_t i = 1; i < writable.size(); ++i) {
        if (writable[i].begin < writable[i - 1].end) {
            return false;
        }
    }
    std::vector<std::uintptr_t> reach(readable.size());
    for (std::size_t i = 0; i < readable.size(); ++i) {
        reach[i] = i == 0 ? readable[i].end : std::max(reach[i - 1], readable[i].end);
    }
    writable_.clear();
    readable_ = std::move(readable);
    reach_ = std::move(reach);
    for (const auto &range : writable) {
        if (!disjoint(range)) {
            return false;
        }
    }
    writable_ = std::move(writable);
    return true;
}

bool RangeSet::disjoint(Range range) const noexcept {
    if (range.begin == range.end) {
        return true;
    }
    const auto before = [](const Range &item, std::uintptr_t end) { return item.begin < end; };
    const auto writable = std::lower_bound(writable_.begin(), writable_.end(), range.end, before);
    if (writable != writable_.begin() && std::prev(writable)->end > range.begin) {
        return false;
    }
    const auto readable = std::lower_bound(readable_.begin(), readable_.end(), range.end, before);
    const auto count = static_cast<std::size_t>(readable - readable_.begin());
    return count == 0 || reach_[count - 1] <= range.begin;
}

Plan::~Plan() {
    // Callers must have finished every execution, so the free needs no ordering.
    if (storage != nullptr) {
        static_cast<void>(cuda_status(cudaFreeAsync(storage, cudaStreamPerThread)));
    }
}

Status accessible(const void *data, int device, bool pageable) noexcept {
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, data) != cudaSuccess) {
        return cuda_failure();
    }
    switch (attributes.type) {
    case cudaMemoryTypeDevice:
        return attributes.device == device ? Status::success : Status::invalid_argument;
    case cudaMemoryTypeManaged:
        return Status::success;
    case cudaMemoryTypeHost:
        return attributes.devicePointer == data ? Status::success : Status::invalid_argument;
    default:
        return pageable ? Status::success : Status::invalid_argument;
    }
}

namespace {

Range range_of(const void *data, std::uint64_t size) noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    return {begin, begin + static_cast<std::uintptr_t>(size)};
}

Status stream_device(cudaStream_t stream, int device) noexcept {
    int current = 0;
    int owner = 0;
    if (cudaGetDevice(&current) != cudaSuccess ||
        cudaStreamGetDevice(stream, &owner) != cudaSuccess) {
        return cuda_failure();
    }
    return current == device && owner == device ? Status::success : Status::invalid_argument;
}

std::uint64_t columns(const ImageSpec &image, TileShape tile) noexcept {
    return image.width / tile.width + (image.width % tile.width != 0);
}

std::uint64_t rows(const ImageSpec &image, TileShape tile) noexcept {
    return image.height / tile.height + (image.height % tile.height != 0);
}

struct Draft {
    PixelType type = PixelType::uint16;
    TileShape shape;
    std::vector<DeviceImage> images;
    std::vector<std::uint64_t> tiles;
    std::vector<DeviceTile> tile_records;
    std::vector<DeviceSegment> segment_records;
    std::vector<std::uint64_t> tile_begin;
    std::vector<std::uint64_t> segment_begin;
    std::vector<std::uint64_t> first_rows;
    std::uint64_t total_rows = 0;
    std::vector<Range> writable;
    std::vector<Range> readable;
    std::vector<const void *> pointers;
};

Status draft_compress(const BatchRequest &request, Draft &draft) {
    draft.type = request.items[0].input.image.type;
    draft.shape = request.tile;
    for (std::uint64_t i = 0; i < request.count; ++i) {
        const auto &item = request.items[i];
        auto status = cpu::detail::codec_view(item.input);
        if (status != Status::success) {
            return status;
        }
        if (item.input.image.type != draft.type) {
            return Status::invalid_argument;
        }
        std::uint64_t bound = 0;
        status = compress_bound(item.input.image, request.tile, bound);
        if (status != Status::success) {
            return status;
        }
        const auto &output = item.compressed_output;
        status = byte_range(output.data, output.size);
        if (status != Status::success) {
            return status;
        }
        if (output.data == nullptr || reinterpret_cast<std::uintptr_t>(output.data) % 4 != 0) {
            return Status::invalid_argument;
        }
        if (output.size < bound) {
            return Status::buffer_too_small;
        }
        DeviceImage record;
        record.input = item.input;
        record.compressed_output = output;
        record.tile_count =
            columns(item.input.image, request.tile) * rows(item.input.image, request.tile);
        record.archive_image = item.input.image;
        record.archive_tile = request.tile;
        draft.images.push_back(record);
        draft.tiles.push_back(record.tile_count);
        draft.readable.push_back(range_of(item.input.data, item.input.capacity_bytes));
        draft.writable.push_back(range_of(output.data, output.size));
        draft.pointers.push_back(item.input.data);
        draft.pointers.push_back(output.data);
    }
    return Status::success;
}

// A map matches the output region exactly and is read with output coordinates.
Status check_parameter(const Parameter &parameter, const ImageSpec &region, Draft &draft) {
    if (parameter.kind == ParameterKind::scalar) {
        return Status::success;
    }
    if (parameter.kind != ParameterKind::map) {
        return Status::invalid_argument;
    }
    const auto &map = parameter.map;
    const auto status = cpu::detail::codec_view(map);
    if (status != Status::success) {
        return status;
    }
    if (map.image.type != PixelType::float32 || map.image.width != region.width ||
        map.image.height != region.height) {
        return Status::invalid_argument;
    }
    draft.readable.push_back(range_of(map.data, map.capacity_bytes));
    draft.pointers.push_back(map.data);
    return Status::success;
}

Status check_calibration(const Calibration &calibration, const ImageSpec &region, Draft &draft) {
    const auto status = check_parameter(calibration.offset, region, draft);
    return status != Status::success ? status : check_parameter(calibration.gain, region, draft);
}

Status draft_calibrate(const BatchRequest &request, Draft &draft) {
    draft.type = request.items[0].input.image.type;
    for (std::uint64_t i = 0; i < request.count; ++i) {
        const auto &item = request.items[i];
        const auto &input = item.input;
        const auto &output = item.output;
        auto status = cpu::detail::codec_view(input);
        if (status == Status::success) {
            status = cpu::detail::codec_view(
                {output.data, output.image, output.pitch_bytes, output.capacity_bytes});
        }
        if (status != Status::success) {
            return status;
        }
        if (input.image.type != draft.type || output.image.type != PixelType::float32 ||
            output.image.width != input.image.width || output.image.height != input.image.height) {
            return Status::invalid_argument;
        }
        status = check_calibration(item.calibration, output.image, draft);
        if (status != Status::success) {
            return status;
        }
        DeviceImage record;
        record.input = input;
        record.output = output;
        record.calibration = item.calibration;
        draft.images.push_back(record);
        draft.first_rows.push_back(draft.total_rows);
        draft.total_rows += input.image.height;
        draft.readable.push_back(range_of(input.data, input.capacity_bytes));
        draft.writable.push_back(range_of(output.data, output.capacity_bytes));
        draft.pointers.push_back(input.data);
        draft.pointers.push_back(output.data);
    }
    return Status::success;
}

// Repeats the parser's structural checks that keep device reads inside the tile payload.
Status check_tile(const ImageMetadata &metadata, std::uint64_t index) noexcept {
    const auto &image = metadata.image;
    const auto &tile = metadata.tiles[static_cast<std::size_t>(index)];
    const auto across = columns(image, metadata.tile);
    const auto x = index % across * metadata.tile.width;
    const auto y = index / across * metadata.tile.height;
    const auto width = std::min<std::uint64_t>(metadata.tile.width, image.width - x);
    const auto height = std::min<std::uint64_t>(metadata.tile.height, image.height - y);
    const std::uint64_t sample_bytes = image.type == PixelType::uint16 ? 2 : 4;
    const std::uint32_t maximum_bits = image.type == PixelType::uint16 ? 16 : 32;
    const auto raw = (width * height * sample_bytes + 3) / 4 * 4;
    if (tile.width != width || tile.height != height || tile.offset % 4 != 0 ||
        tile.offset > metadata.file_bytes || tile.length > metadata.file_bytes - tile.offset) {
        return Status::malformed_data;
    }
    switch (tile.mode) {
    case TileMode::raw:
        return tile.length == raw ? Status::success : Status::malformed_data;
    case TileMode::constant:
        return tile.length == 4 ? Status::success : Status::malformed_data;
    case TileMode::predict_bitpack:
        break;
    default:
        return Status::malformed_data;
    }
    if (tile.length > raw || tile.first_segment > metadata.segments.size() ||
        height > metadata.segments.size() - tile.first_segment) {
        return Status::malformed_data;
    }
    for (std::uint64_t row = 0; row < height; ++row) {
        const auto &segment = metadata.segments[static_cast<std::size_t>(tile.first_segment + row)];
        if (segment.bit_width > maximum_bits ||
            (image.type == PixelType::uint16 && segment.seed > UINT16_MAX) ||
            segment.payload_offset < format::segment_header_bytes ||
            segment.payload_offset % 4 != 0 || segment.payload_offset > tile.length ||
            segment.payload_bytes > tile.length - segment.payload_offset ||
            segment.payload_bytes != ((width - 1) * segment.bit_width + 31) / 32 * 4) {
            return Status::malformed_data;
        }
    }
    return Status::success;
}

// Selects the tiles that intersect the region and rebases them into the supplied bytes.
Status stage_tiles(const CompressedInput &source, const Roi &roi, Draft &draft) {
    const auto &metadata = *source.metadata;
    const auto across = columns(metadata.image, metadata.tile);
    if (metadata.tiles.size() != across * rows(metadata.image, metadata.tile)) {
        return Status::malformed_data;
    }
    const std::uint64_t left = roi.x / metadata.tile.width;
    const std::uint64_t right = (std::uint64_t{roi.x} + roi.width - 1) / metadata.tile.width;
    const std::uint64_t top = roi.y / metadata.tile.height;
    const std::uint64_t bottom = (std::uint64_t{roi.y} + roi.height - 1) / metadata.tile.height;
    const auto selected = (right - left + 1) * (bottom - top + 1);
    const bool mapped = source.mappings != nullptr || source.mapping_count != 0;
    if (mapped ? source.mappings == nullptr || source.mapping_count != selected
               : metadata.file_bytes > source.bytes.size) {
        return mapped ? Status::invalid_argument : Status::buffer_too_small;
    }
    std::vector<Range> staged;
    const auto first_segment = draft.segment_records.size();
    std::uint64_t k = 0;
    for (auto row = top; row <= bottom; ++row) {
        for (auto column = left; column <= right; ++column, ++k) {
            const auto index = row * across + column;
            const auto status = check_tile(metadata, index);
            if (status != Status::success) {
                return status;
            }
            const auto &tile = metadata.tiles[static_cast<std::size_t>(index)];
            auto offset = tile.offset;
            if (mapped) {
                const auto &mapping = source.mappings[k];
                if (mapping.tile_index != index || mapping.input_offset % 4 != 0 ||
                    mapping.input_offset > source.bytes.size ||
                    tile.length > source.bytes.size - mapping.input_offset) {
                    return Status::invalid_argument;
                }
                offset = mapping.input_offset;
                staged.push_back({static_cast<std::uintptr_t>(offset),
                                  static_cast<std::uintptr_t>(offset + tile.length)});
            }
            DeviceTile record;
            record.tile_index = index;
            record.input_offset = offset;
            record.length = tile.length;
            record.first_segment = draft.segment_records.size() - first_segment;
            record.mode = tile.mode;
            record.width = tile.width;
            record.height = tile.height;
            draft.tile_records.push_back(record);
            if (tile.mode == TileMode::predict_bitpack) {
                for (std::uint64_t line = 0; line < tile.height; ++line) {
                    const auto &segment =
                        metadata.segments[static_cast<std::size_t>(tile.first_segment + line)];
                    draft.segment_records.push_back(
                        {segment.seed, segment.bit_width, offset + segment.payload_offset});
                }
            }
        }
    }
    std::sort(staged.begin(), staged.end(),
              [](const Range &a, const Range &b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < staged.size(); ++i) {
        if (staged[i].begin < staged[i - 1].end) {
            return Status::invalid_argument;
        }
    }
    draft.tiles.push_back(selected);
    return Status::success;
}

Status draft_decode(const BatchRequest &request, Draft &draft) {
    bool mixed = false;
    for (std::uint64_t i = 0; i < request.count; ++i) {
        const auto &item = request.items[i];
        const auto &source = item.compressed_input;
        const bool trusted = source.encoder_result != nullptr;
        if (trusted ? source.metadata != nullptr || source.mappings != nullptr ||
                          source.mapping_count != 0
                    : source.metadata == nullptr) {
            return Status::invalid_argument;
        }
        const auto image = trusted ? source.trusted_image : source.metadata->image;
        const auto shape = trusted ? source.trusted_tile : source.metadata->tile;
        std::uint64_t bound = 0;
        auto status = compress_bound(image, shape, bound);
        if (status != Status::success) {
            return status;
        }
        if (i == 0) {
            draft.type = image.type;
            draft.shape = shape;
        }
        if (image.type != draft.type) {
            return Status::invalid_argument;
        }
        mixed = mixed || shape.width != draft.shape.width || shape.height != draft.shape.height;

        const auto &output = item.output;
        status = cpu::detail::codec_view(
            {output.data, output.image, output.pitch_bytes, output.capacity_bytes});
        if (status != Status::success) {
            return status;
        }
        Roi roi{0, 0, image.width, image.height};
        if (request.operation != Operation::decompress) {
            roi = item.roi;
            if (roi.width == 0 || roi.height == 0 ||
                std::uint64_t{roi.x} + roi.width > image.width ||
                std::uint64_t{roi.y} + roi.height > image.height) {
                return Status::invalid_roi;
            }
        }
        const bool calibrated = request.operation == Operation::decode_calibrate;
        if (output.image.type != (calibrated ? PixelType::float32 : image.type) ||
            output.image.width != roi.width || output.image.height != roi.height) {
            return Status::invalid_argument;
        }
        if (calibrated) {
            status = check_calibration(item.calibration, output.image, draft);
            if (status != Status::success) {
                return status;
            }
        }
        status = byte_range(source.bytes.data, source.bytes.size);
        if (status != Status::success) {
            return status;
        }
        if (source.bytes.data == nullptr ||
            reinterpret_cast<std::uintptr_t>(source.bytes.data) % 4 != 0) {
            return Status::invalid_argument;
        }

        draft.tile_begin.push_back(draft.tile_records.size());
        draft.segment_begin.push_back(draft.segment_records.size());
        if (trusted) {
            // Modes and offsets are only known on the device, after the encoder has finished.
            if (reinterpret_cast<std::uintptr_t>(source.encoder_result) % alignof(DeviceResult) !=
                0) {
                return Status::invalid_argument;
            }
            if (source.bytes.size < bound) {
                return Status::buffer_too_small;
            }
            const auto across = columns(image, shape);
            for (std::uint64_t row = roi.y / shape.height;
                 row <= (std::uint64_t{roi.y} + roi.height - 1) / shape.height; ++row) {
                for (std::uint64_t column = roi.x / shape.width;
                     column <= (std::uint64_t{roi.x} + roi.width - 1) / shape.width; ++column) {
                    DeviceTile record;
                    record.tile_index = row * across + column;
                    draft.tile_records.push_back(record);
                }
            }
            draft.tiles.push_back(draft.tile_records.size() - draft.tile_begin.back());
            draft.readable.push_back(range_of(source.encoder_result, sizeof(DeviceResult)));
            draft.pointers.push_back(source.encoder_result);
        } else {
            status = stage_tiles(source, roi, draft);
            if (status != Status::success) {
                return status;
            }
        }

        DeviceImage record;
        record.output = output;
        record.compressed_input = source.bytes;
        record.tile_count = draft.tiles.back();
        record.encoder_result = source.encoder_result;
        record.archive_image = image;
        record.archive_tile = shape;
        record.roi = roi;
        if (calibrated) {
            record.calibration = item.calibration;
        }
        draft.images.push_back(record);
        draft.readable.push_back(range_of(source.bytes.data, source.bytes.size));
        draft.writable.push_back(range_of(output.data, output.capacity_bytes));
        draft.pointers.push_back(source.bytes.data);
        draft.pointers.push_back(output.data);
    }
    if (mixed) {
        draft.shape = {64, 64};
    }
    return Status::success;
}

// Splits the concatenated tiles of all images into windows of at most window_tiles tiles.
void split_windows(const std::vector<std::uint64_t> &tiles, std::vector<WindowImage> &entries,
                   std::vector<std::pair<std::size_t, Window>> &windows) {
    std::uint32_t used = 0;
    for (std::size_t image = 0; image < tiles.size(); ++image) {
        for (std::uint64_t start = 0; start < tiles[image];) {
            if (windows.empty() || used == window_tiles) {
                windows.push_back({entries.size(), {}});
                used = 0;
            }
            const auto take = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(tiles[image] - start, window_tiles - used));
            std::uint32_t flags = start == 0 ? 0 : continues;
            if (start + take == tiles[image]) {
                flags |= finishes;
            }
            entries.push_back({static_cast<std::uint32_t>(image), used, take, flags, start});
            auto &window = windows.back().second;
            ++window.image_count;
            window.tile_count += take;
            used += take;
            start += take;
        }
    }
}

template <typename T>
bool upload_table(void *storage, std::uint64_t offset, const std::vector<T> &table,
                  cudaStream_t stream) {
    return table.empty() ||
           cudaMemcpyAsync(static_cast<std::uint8_t *>(storage) + offset, table.data(),
                           table.size() * sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

Status build(const BatchRequest &request, Plan &plan, cudaStream_t stream) {
    plan.operation = request.operation;
    plan.count = request.count;
    if (request.count == 0) {
        return Status::success;
    }

    Draft draft;
    draft.images.reserve(static_cast<std::size_t>(request.count));
    auto status = request.operation == Operation::compress    ? draft_compress(request, draft)
                  : request.operation == Operation::calibrate ? draft_calibrate(request, draft)
                                                              : draft_decode(request, draft);
    if (status != Status::success) {
        return status;
    }
    plan.type = draft.type;
    if (!plan.ranges.assign(std::move(draft.writable), std::move(draft.readable))) {
        return Status::invalid_argument;
    }

    int pageable = 0;
    if (cudaGetDevice(&plan.device) != cudaSuccess ||
        cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, plan.device) !=
            cudaSuccess) {
        return cuda_failure();
    }
    plan.pageable = pageable != 0;
    status = stream_device(stream, plan.device);
    if (status != Status::success) {
        return status;
    }
    for (const auto *pointer : draft.pointers) {
        status = accessible(pointer, plan.device, plan.pageable);
        if (status != Status::success) {
            return status;
        }
    }

    std::vector<WindowImage> entries;
    std::vector<std::pair<std::size_t, Window>> windows;
    split_windows(draft.tiles, entries, windows);
    plan.spec = {0, 0, draft.shape, 0};
    for (const auto &[first, window] : windows) {
        // Decoding uses only block and warp primitives, so it needs no scan storage.
        std::uint64_t temp = 0;
        if (request.operation == Operation::compress) {
            status = scan_temp_bytes(window.tile_count, temp);
            if (status != Status::success) {
                return status;
            }
        }
        plan.spec.image_count = std::max<std::uint64_t>(plan.spec.image_count, window.image_count);
        plan.spec.tile_count = std::max<std::uint64_t>(plan.spec.tile_count, window.tile_count);
        plan.spec.scan_temp_bytes = std::max(plan.spec.scan_temp_bytes, temp);
    }
    status = tilestream::detail::workspace_layout(plan.spec, plan.layout);
    if (status != Status::success) {
        return status;
    }

    std::uint64_t total = 0;
    const auto place = [&total](std::uint64_t count, std::uint64_t size, std::uint64_t &offset) {
        std::uint64_t bytes = 0;
        return align_size(total, 16, offset) && multiply_size(count, size, bytes) &&
               add_size(offset, bytes, total);
    };
    std::uint64_t images_at = 0;
    std::uint64_t tiles_at = 0;
    std::uint64_t segments_at = 0;
    std::uint64_t entries_at = 0;
    std::uint64_t rows_at = 0;
    if (!place(draft.images.size(), sizeof(DeviceImage), images_at) ||
        !place(draft.tile_records.size(), sizeof(DeviceTile), tiles_at) ||
        !place(draft.segment_records.size(), sizeof(DeviceSegment), segments_at) ||
        !place(entries.size(), sizeof(WindowImage), entries_at) ||
        !place(draft.first_rows.size(), sizeof(std::uint64_t), rows_at)) {
        return Status::size_overflow;
    }
    if (cudaMallocAsync(&plan.storage, static_cast<std::size_t>(total), stream) != cudaSuccess) {
        plan.storage = nullptr;
        return cuda_failure();
    }
    auto *device = static_cast<std::uint8_t *>(plan.storage);
    const auto *tiles = reinterpret_cast<const DeviceTile *>(device + tiles_at);
    const auto *segments = reinterpret_cast<const DeviceSegment *>(device + segments_at);
    // Only decoding drafts have tile tables.
    for (std::size_t i = 0; i < draft.tile_begin.size(); ++i) {
        draft.images[i].tiles = tiles + draft.tile_begin[i];
        draft.images[i].segments = segments + draft.segment_begin[i];
    }
    plan.images = reinterpret_cast<const DeviceImage *>(device + images_at);
    plan.first_rows = reinterpret_cast<const std::uint64_t *>(device + rows_at);
    plan.total_rows = draft.total_rows;
    for (const auto &[first, window] : windows) {
        auto placed = window;
        placed.images = reinterpret_cast<const WindowImage *>(device + entries_at) + first;
        plan.windows.push_back(placed);
    }
    switch (request.operation) {
    case Operation::compress:
        status = load_encoder(plan.type, stream);
        break;
    case Operation::calibrate:
        status = load_calibration(plan.type);
        break;
    default:
        status = load_decoder(plan.type, request.operation == Operation::decode_calibrate);
        break;
    }
    if (status != Status::success) {
        return status;
    }
    // The tables are copied straight from the draft, which outlives the copies.
    if (!upload_table(plan.storage, images_at, draft.images, stream) ||
        !upload_table(plan.storage, tiles_at, draft.tile_records, stream) ||
        !upload_table(plan.storage, segments_at, draft.segment_records, stream) ||
        !upload_table(plan.storage, entries_at, entries, stream) ||
        !upload_table(plan.storage, rows_at, draft.first_rows, stream) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return cuda_failure();
    }
    return Status::success;
}

} // namespace

Status check_execution(const Plan *plan, Operation operation, MutableBytes workspace,
                       ResultBuffer results, cudaStream_t stream) noexcept {
    if (plan == nullptr || plan->operation != operation) {
        return Status::invalid_argument;
    }
    if (plan->count == 0) {
        return Status::success;
    }
    std::uint64_t result_bytes = 0;
    if (!multiply_size(results.count, sizeof(DeviceResult), result_bytes)) {
        return Status::size_overflow;
    }
    auto status = byte_range(workspace.data, workspace.size);
    if (status == Status::success) {
        status = byte_range(results.data, result_bytes);
    }
    if (status != Status::success) {
        return status;
    }
    // Both sizes are nonzero after this check, so byte_range has already rejected null pointers.
    if (workspace.size < plan->layout.total || results.count < plan->count) {
        return Status::buffer_too_small;
    }
    if (reinterpret_cast<std::uintptr_t>(workspace.data) % workspace_alignment != 0 ||
        reinterpret_cast<std::uintptr_t>(results.data) % alignof(DeviceResult) != 0) {
        return Status::invalid_argument;
    }
    const auto scratch = range_of(workspace.data, workspace.size);
    const auto records = range_of(results.data, result_bytes);
    if (!plan->ranges.disjoint(scratch) || !plan->ranges.disjoint(records) ||
        cpu::detail::overlaps(workspace.data, workspace.size, results.data, result_bytes)) {
        return Status::invalid_argument;
    }
    status = stream_device(stream, plan->device);
    // An empty workspace is never accessed, so its pointer need not be device memory.
    if (status == Status::success && workspace.size != 0) {
        status = accessible(workspace.data, plan->device, plan->pageable);
    }
    if (status == Status::success) {
        status = accessible(results.data, plan->device, plan->pageable);
    }
    // Launch failures are seen only through the last error, so a pending one from the caller
    // would be reported as ours. It is left for the caller to read.
    if (status == Status::success && cudaPeekAtLastError() != cudaSuccess) {
        status = Status::cuda_error;
    }
    return status;
}

} // namespace detail

BatchPlan::BatchPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
BatchPlan::~BatchPlan() = default;
BatchPlan::BatchPlan(BatchPlan &&) noexcept = default;
BatchPlan &BatchPlan::operator=(BatchPlan &&) noexcept = default;

WorkspaceSpec BatchPlan::workspace_spec() const noexcept {
    return impl_ != nullptr ? impl_->spec : WorkspaceSpec{};
}

Status prepare_batch(const BatchRequest &request, std::unique_ptr<BatchPlan> &plan,
                     cudaStream_t stream) noexcept {
    if (request.count > UINT32_MAX) {
        return Status::resource_limit;
    }
    if (request.count != 0 && request.items == nullptr) {
        return Status::invalid_argument;
    }
    if (request.operation > Operation::decode_calibrate) {
        return Status::invalid_argument;
    }
    try {
        auto impl = std::make_unique<BatchPlan::Impl>();
        const auto status = detail::build(request, *impl, stream);
        if (status != Status::success) {
            return status;
        }
        plan.reset(new BatchPlan(std::move(impl)));
        return Status::success;
    } catch (const std::bad_alloc &) {
        return Status::resource_limit;
    } catch (const std::length_error &) {
        return Status::resource_limit;
    }
}

} // namespace tilestream::cuda
