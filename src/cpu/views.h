#pragma once

#include <tilestream/api.h>

#include <cstdint>
#include <limits>

namespace tilestream::cpu::detail {

inline Status byte_range(const void *data, std::uint64_t capacity) noexcept {
    if (capacity > max_bytes) {
        return Status::size_overflow;
    }
    if (data == nullptr && capacity != 0) {
        return Status::invalid_argument;
    }
    const auto start = reinterpret_cast<std::uintptr_t>(data);
    if (capacity > std::numeric_limits<std::uintptr_t>::max() - start) {
        return Status::size_overflow;
    }
    return Status::success;
}

// Both ranges must already have passed byte_range.
inline bool overlaps(const void *a, std::uint64_t a_size, const void *b,
                     std::uint64_t b_size) noexcept {
    const auto first = reinterpret_cast<std::uintptr_t>(a);
    const auto second = reinterpret_cast<std::uintptr_t>(b);
    return a_size != 0 && b_size != 0 && first < second + b_size && second < first + a_size;
}

inline Status uint16_view(ConstImageView view) noexcept {
    if (view.image.type != PixelType::uint16) {
        return Status::unsupported_type;
    }
    std::uint64_t span = 0;
    const auto status = image_span(view.image, view.pitch_bytes, span);
    if (status != Status::success) {
        return status;
    }
    const auto range_status = byte_range(view.data, view.capacity_bytes);
    if (range_status != Status::success) {
        return range_status;
    }
    if (view.data == nullptr || reinterpret_cast<std::uintptr_t>(view.data) % 2 != 0) {
        return Status::invalid_argument;
    }
    return span <= view.capacity_bytes ? Status::success : Status::buffer_too_small;
}

} // namespace tilestream::cpu::detail
