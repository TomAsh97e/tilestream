#include <tilestream/api.h>
#include <tilestream/format.h>

#include "checked_size.h"
#include "views.h"
#include "wire.h"

#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

namespace tilestream {
namespace {

using cpu::detail::read_le;
using cpu::detail::zero_bytes;

struct Layout {
    ImageSpec image;
    TileShape tile;
    std::uint64_t columns = 0;
    std::uint64_t count = 0;
    std::uint64_t payload = 0;
};

Status read_header(ConstBytes source, Layout &layout) noexcept {
    const auto *data = static_cast<const std::uint8_t *>(source.data);
    if (source.size < format::header_bytes) {
        return Status::malformed_data;
    }
    if (!std::equal(format::magic.begin(), format::magic.end(), data)) {
        return Status::malformed_data;
    }
    if (read_le<std::uint16_t>(data + format::header::version_major) != format::major_version ||
        read_le<std::uint16_t>(data + format::header::version_minor) != format::minor_version ||
        data[format::header::flags] != 0) {
        return Status::unsupported_format;
    }
    if (read_le<std::uint16_t>(data + format::header::size) != format::header_bytes ||
        read_le<std::uint16_t>(data + format::header::entry_size) != format::index_entry_bytes ||
        !zero_bytes(data + format::header::reserved, 2)) {
        return Status::malformed_data;
    }
    if (data[format::header::pixel_type] != static_cast<std::uint8_t>(PixelType::uint16)) {
        return Status::unsupported_type;
    }
    layout.image = {PixelType::uint16, read_le<std::uint32_t>(data + format::header::width),
                    read_le<std::uint32_t>(data + format::header::height)};
    layout.tile = {read_le<std::uint16_t>(data + format::header::tile_width),
                   read_le<std::uint16_t>(data + format::header::tile_height)};
    if (layout.image.width == 0 || layout.image.height == 0) {
        return Status::malformed_data;
    }
    if (!detail::supported_tile(layout.tile)) {
        return Status::unsupported_format;
    }
    std::uint64_t bound = 0;
    const auto status = compress_bound(layout.image, layout.tile, bound);
    if (status != Status::success) {
        return status;
    }
    layout.columns = layout.image.width / layout.tile.width +
                     (layout.image.width % layout.tile.width != 0);
    const std::uint64_t rows = layout.image.height / layout.tile.height +
                               (layout.image.height % layout.tile.height != 0);
    std::uint64_t index_bytes = 0;
    if (!detail::multiply_size(layout.columns, rows, layout.count) ||
        !detail::multiply_size(layout.count, format::index_entry_bytes, index_bytes) ||
        !detail::add_size(format::header_bytes, index_bytes, layout.payload)) {
        return Status::size_overflow;
    }
    if (read_le<std::uint64_t>(data + format::header::tile_count) != layout.count ||
        read_le<std::uint64_t>(data + format::header::index_offset) != format::header_bytes ||
        read_le<std::uint64_t>(data + format::header::payload_offset) != layout.payload ||
        read_le<std::uint64_t>(data + format::header::file_size) != source.size ||
        source.size > bound || layout.payload > source.size) {
        return Status::malformed_data;
    }
    return Status::success;
}

Status walk_tiles(ConstBytes source, const Layout &layout, std::uint64_t &segment_count,
                  ImageMetadata *metadata) noexcept {
    const auto *data = static_cast<const std::uint8_t *>(source.data);
    std::uint64_t expected_offset = layout.payload;
    segment_count = 0;
    for (std::uint64_t i = 0; i < layout.count; ++i) {
        const auto *entry = data + format::header_bytes + i * format::index_entry_bytes;
        const auto x = (i % layout.columns) * layout.tile.width;
        const auto y = (i / layout.columns) * layout.tile.height;
        const auto width = std::min<std::uint64_t>(layout.tile.width, layout.image.width - x);
        const auto height = std::min<std::uint64_t>(layout.tile.height, layout.image.height - y);
        const auto offset = read_le<std::uint64_t>(entry + format::entry::offset);
        const auto length = read_le<std::uint64_t>(entry + format::entry::length);
        const auto raw_bytes = width * height * 2;
        const auto raw_padded = (raw_bytes + 3) / 4 * 4;
        if (entry[format::entry::mode] > static_cast<std::uint8_t>(TileMode::predict_bitpack) ||
            !zero_bytes(entry + format::entry::reserved, 3) ||
            !zero_bytes(entry + format::entry::reserved_tail, 8) ||
            read_le<std::uint16_t>(entry + format::entry::width) != width ||
            read_le<std::uint16_t>(entry + format::entry::height) != height ||
            offset != expected_offset || offset > source.size || length > source.size - offset ||
            length == 0 || length % 4 != 0 || length > raw_padded) {
            return Status::malformed_data;
        }
        const auto mode = static_cast<TileMode>(entry[format::entry::mode]);
        const auto *payload = data + offset;
        const auto first_segment = mode == TileMode::predict_bitpack ? segment_count : 0;
        if (mode == TileMode::raw) {
            if (length != raw_padded || !zero_bytes(payload + raw_bytes, length - raw_bytes)) {
                return Status::malformed_data;
            }
        } else if (mode == TileMode::constant) {
            if (length != 4 || !zero_bytes(payload + 2, 2)) {
                return Status::malformed_data;
            }
        } else {
            std::uint64_t cursor = 0;
            for (std::uint64_t row = 0; row < height; ++row) {
                if (length - cursor < format::segment_header_bytes) {
                    return Status::malformed_data;
                }
                const auto *descriptor = payload + cursor;
                const auto seed = read_le<std::uint32_t>(descriptor + format::segment::seed);
                const auto bits = descriptor[format::segment::bit_width];
                if (seed > UINT16_MAX || bits > 16 || (width == 1 && bits != 0) ||
                    !zero_bytes(descriptor + format::segment::reserved, 3)) {
                    return Status::malformed_data;
                }
                cursor += format::segment_header_bytes;
                const auto bit_count = (width - 1) * bits;
                const auto packed_bytes = ((bit_count + 31) / 32) * 4;
                if (packed_bytes > length - cursor) {
                    return Status::malformed_data;
                }
                if (bit_count % 32 != 0) {
                    const auto last = read_le<std::uint32_t>(payload + cursor + packed_bytes - 4);
                    if ((last >> (bit_count % 32)) != 0) {
                        return Status::malformed_data;
                    }
                }
                if (metadata != nullptr) {
                    metadata->segments[segment_count] = {seed, bits, cursor, packed_bytes};
                }
                ++segment_count;
                cursor += packed_bytes;
            }
            if (cursor != length) {
                return Status::malformed_data;
            }
        }
        if (metadata != nullptr) {
            metadata->tiles[i] = {mode, static_cast<std::uint16_t>(width),
                                  static_cast<std::uint16_t>(height), offset, length, first_segment};
        }
        expected_offset += length;
    }
    return expected_offset == source.size ? Status::success : Status::malformed_data;
}

Status metadata_size(std::uint64_t tiles, std::uint64_t segments, std::uint64_t &bytes) noexcept {
    std::uint64_t tile_bytes = 0;
    std::uint64_t segment_bytes = 0;
    if (!detail::multiply_size(tiles, sizeof(TileMetadata), tile_bytes) ||
        !detail::multiply_size(segments, sizeof(SegmentMetadata), segment_bytes) ||
        !detail::add_size(sizeof(ImageMetadata), tile_bytes, bytes) ||
        !detail::add_size(bytes, segment_bytes, bytes)) {
        return Status::size_overflow;
    }
    return Status::success;
}

} // namespace

Status parse_metadata(ConstBytes source, const ParseLimits &limits, ImageMetadata &metadata) noexcept {
    const auto range_status = cpu::detail::byte_range(source.data, source.size);
    if (range_status != Status::success) {
        return range_status;
    }
    if (source.size > limits.file_bytes) {
        return Status::resource_limit;
    }
    Layout layout;
    auto status = read_header(source, layout);
    if (status != Status::success) {
        return status;
    }
    if (layout.count > limits.tile_count) {
        return Status::resource_limit;
    }
    std::uint64_t required = 0;
    status = metadata_size(layout.count, 0, required);
    if (status != Status::success) {
        return status;
    }
    if (required > limits.metadata_bytes) {
        return Status::resource_limit;
    }
    std::uint64_t segments = 0;
    status = walk_tiles(source, layout, segments, nullptr);
    if (status != Status::success) {
        return status;
    }
    status = metadata_size(layout.count, segments, required);
    if (status != Status::success) {
        return status;
    }
    if (required > limits.metadata_bytes) {
        return Status::resource_limit;
    }
    try {
        ImageMetadata parsed;
        parsed.image = layout.image;
        parsed.tile = layout.tile;
        parsed.file_bytes = source.size;
        parsed.tiles = std::vector<TileMetadata>(static_cast<std::size_t>(layout.count));
        parsed.segments = std::vector<SegmentMetadata>(static_cast<std::size_t>(segments));
        status = walk_tiles(source, layout, segments, &parsed);
        if (status != Status::success) {
            return status;
        }
        metadata = std::move(parsed);
        return Status::success;
    } catch (const std::bad_alloc &) {
        return Status::resource_limit;
    } catch (const std::length_error &) {
        return Status::resource_limit;
    }
}

} // namespace tilestream
