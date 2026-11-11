#pragma once

#include <tilestream/sizing.h>

#include <cstdint>

namespace tilestream::detail {

inline bool add_size(std::uint64_t a, std::uint64_t b, std::uint64_t &result) noexcept {
    if (a > max_bytes || b > max_bytes - a) {
        return false;
    }
    result = a + b;
    return true;
}

inline bool multiply_size(std::uint64_t a, std::uint64_t b, std::uint64_t &result) noexcept {
    if (a > max_bytes || b > max_bytes || (b != 0 && a > max_bytes / b)) {
        return false;
    }
    result = a * b;
    return true;
}

inline bool align_size(std::uint64_t value, std::uint64_t alignment,
                       std::uint64_t &result) noexcept {
    if (alignment == 0 || alignment > max_bytes || value > max_bytes) {
        return false;
    }
    const auto remainder = value % alignment;
    return add_size(value, remainder == 0 ? 0 : alignment - remainder, result);
}

inline bool supported_tile(TileShape tile) noexcept {
    return (tile.width == 128 && tile.height == 32) || (tile.width == 64 && tile.height == 64) ||
           (tile.width == 256 && tile.height == 16);
}

inline Status pixel_size(const ImageSpec &image, std::uint64_t &bytes) noexcept {
    if (image.type != PixelType::uint16 && image.type != PixelType::float32) {
        return Status::unsupported_type;
    }
    if (image.width == 0 || image.height == 0) {
        return Status::invalid_argument;
    }
    bytes = image.type == PixelType::uint16 ? 2 : 4;
    return Status::success;
}

} // namespace tilestream::detail
