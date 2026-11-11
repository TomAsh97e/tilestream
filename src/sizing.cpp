#include <tilestream/sizing.h>

#include <tilestream/format.h>

#include "checked_size.h"

namespace tilestream {

Status image_span(const ImageSpec &image, std::uint64_t pitch_bytes,
                  std::uint64_t &bytes) noexcept {
    std::uint64_t sample_bytes = 0;
    const auto status = detail::pixel_size(image, sample_bytes);
    if (status != Status::success) {
        return status;
    }
    std::uint64_t row_bytes = 0;
    if (!detail::multiply_size(image.width, sample_bytes, row_bytes) || pitch_bytes > max_bytes) {
        return Status::size_overflow;
    }
    if (pitch_bytes < row_bytes || pitch_bytes % sample_bytes != 0) {
        return Status::invalid_argument;
    }
    std::uint64_t span = 0;
    if (!detail::multiply_size(pitch_bytes, image.height - 1, span) ||
        !detail::add_size(span, row_bytes, span)) {
        return Status::size_overflow;
    }
    bytes = span;
    return Status::success;
}

Status compress_bound(const ImageSpec &image, TileShape tile, std::uint64_t &bytes) noexcept {
    std::uint64_t sample_bytes = 0;
    const auto status = detail::pixel_size(image, sample_bytes);
    if (status != Status::success) {
        return status;
    }
    if (!detail::supported_tile(tile)) {
        return Status::invalid_argument;
    }
    const std::uint64_t columns = image.width / tile.width + (image.width % tile.width != 0);
    const std::uint64_t rows = image.height / tile.height + (image.height % tile.height != 0);
    std::uint64_t tiles = 0;
    std::uint64_t index_bytes = 0;
    std::uint64_t raw_bytes = 0;
    std::uint64_t bound = 0;
    const std::uint64_t padding =
        sample_bytes == 2 && image.width % 2 != 0 && image.height % 2 != 0 ? 2 : 0;
    if (!detail::multiply_size(columns, rows, tiles) ||
        !detail::multiply_size(tiles, format::index_entry_bytes, index_bytes) ||
        !detail::multiply_size(image.width, image.height, raw_bytes) ||
        !detail::multiply_size(raw_bytes, sample_bytes, raw_bytes) ||
        !detail::add_size(format::header_bytes, index_bytes, bound) ||
        !detail::add_size(bound, raw_bytes, bound) || !detail::add_size(bound, padding, bound)) {
        return Status::size_overflow;
    }
    bytes = bound;
    return Status::success;
}

Status workspace_size(const WorkspaceSpec &spec, std::uint64_t &bytes) noexcept {
    detail::WorkspaceLayout layout;
    const auto status = detail::workspace_layout(spec, layout);
    if (status == Status::success) {
        bytes = layout.total;
    }
    return status;
}

namespace detail {

Status workspace_layout(const WorkspaceSpec &spec, WorkspaceLayout &layout) noexcept {
    if (spec.image_count == 0) {
        if (spec.tile_count != 0 || spec.scan_temp_bytes != 0) {
            return Status::invalid_argument;
        }
        layout = {};
        return Status::success;
    }
    if (!supported_tile(spec.tile) || spec.tile_count < spec.image_count) {
        return Status::invalid_argument;
    }
    if (spec.image_count > UINT32_MAX || spec.tile_count > max_window_tiles) {
        return Status::resource_limit;
    }

    WorkspaceLayout result;
    std::uint64_t total = 0;
    const auto append = [&total](std::uint64_t count, std::uint64_t size, std::uint64_t &offset) {
        std::uint64_t region = 0;
        offset = total;
        return multiply_size(count, size, region) &&
               align_size(region, workspace_alignment, region) && add_size(total, region, total);
    };

    // Regions are independent and may not borrow another region's padding.
    if (!append(spec.tile_count, 4096 * 4, result.residuals) ||
        !append(spec.tile_count, std::uint64_t{spec.tile.height} * format::segment_header_bytes,
                result.rows) ||
        !append(spec.tile_count, 32, result.tiles) || !append(spec.tile_count, 8, result.lengths) ||
        !append(spec.tile_count + 1, 8, result.offsets) ||
        !append(spec.image_count, 16, result.images) ||
        !append(spec.scan_temp_bytes, 1, result.temp)) {
        return Status::size_overflow;
    }
    result.total = total;
    layout = result;
    return Status::success;
}

} // namespace detail

} // namespace tilestream
