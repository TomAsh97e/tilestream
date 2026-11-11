#pragma once

#include <tilestream/sizing.h>
#include <tilestream/types.h>

#include <cstdint>
#include <vector>

namespace tilestream {

struct SegmentMetadata {
    std::uint32_t seed = 0;
    std::uint32_t bit_width = 0;
    std::uint64_t payload_offset = 0; // Relative to this tile's payload.
    std::uint64_t payload_bytes = 0;
};

struct TileMetadata {
    TileMode mode = TileMode::raw;
    std::uint16_t width = 0;
    std::uint16_t height = 0;
    std::uint64_t offset = 0; // Absolute file offset.
    std::uint64_t length = 0;
    std::uint64_t first_segment = 0;
};

struct ImageMetadata {
    ImageSpec image;
    TileShape tile;
    std::uint64_t file_bytes = 0;
    std::vector<TileMetadata> tiles;
    std::vector<SegmentMetadata> segments;
};

struct ParseLimits {
    std::uint64_t file_bytes = max_bytes;
    std::uint64_t tile_count = max_bytes;
    std::uint64_t metadata_bytes = max_bytes;
};

// T04 declaration; parser implementation belongs to the CPU codec stages.
[[nodiscard]] Status parse_metadata(ConstBytes source, const ParseLimits &limits,
                                    ImageMetadata &metadata) noexcept;

} // namespace tilestream
