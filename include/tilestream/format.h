#pragma once

#include <array>
#include <cstdint>

namespace tilestream::format {

inline constexpr std::array<std::uint8_t, 8> magic = {'T', 'S', 'C', 0, 13, 10, 26, 10};
inline constexpr std::uint16_t major_version = 1;
inline constexpr std::uint16_t minor_version = 0;
inline constexpr std::uint64_t header_bytes = 64;
inline constexpr std::uint64_t index_entry_bytes = 32;
inline constexpr std::uint64_t segment_header_bytes = 8;
inline constexpr std::uint64_t payload_alignment = 4;

namespace header {
inline constexpr std::uint64_t version_major = 8;
inline constexpr std::uint64_t version_minor = 10;
inline constexpr std::uint64_t size = 12;
inline constexpr std::uint64_t pixel_type = 14;
inline constexpr std::uint64_t flags = 15;
inline constexpr std::uint64_t width = 16;
inline constexpr std::uint64_t height = 20;
inline constexpr std::uint64_t tile_width = 24;
inline constexpr std::uint64_t tile_height = 26;
inline constexpr std::uint64_t entry_size = 28;
inline constexpr std::uint64_t reserved = 30;
inline constexpr std::uint64_t tile_count = 32;
inline constexpr std::uint64_t index_offset = 40;
inline constexpr std::uint64_t payload_offset = 48;
inline constexpr std::uint64_t file_size = 56;
} // namespace header

namespace entry {
inline constexpr std::uint64_t mode = 0;
inline constexpr std::uint64_t reserved = 1;
inline constexpr std::uint64_t width = 4;
inline constexpr std::uint64_t height = 6;
inline constexpr std::uint64_t offset = 8;
inline constexpr std::uint64_t length = 16;
inline constexpr std::uint64_t reserved_tail = 24;
} // namespace entry

namespace segment {
inline constexpr std::uint64_t seed = 0;
inline constexpr std::uint64_t bit_width = 4;
inline constexpr std::uint64_t reserved = 5;
inline constexpr std::uint64_t payload = 8;
} // namespace segment

} // namespace tilestream::format
