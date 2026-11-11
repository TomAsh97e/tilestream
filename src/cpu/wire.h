#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tilestream::cpu::detail {

template <typename T> T read_le(const std::uint8_t *data) noexcept {
    static_assert(std::is_unsigned_v<T> && sizeof(T) <= 8);
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        value |= std::uint64_t{data[i]} << (8 * i);
    }
    return static_cast<T>(value);
}

template <typename T> void write_le(std::uint8_t *data, T value) noexcept {
    static_assert(std::is_unsigned_v<T> && sizeof(T) <= 8);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        data[i] = static_cast<std::uint8_t>(std::uint64_t{value} >> (8 * i));
    }
}

inline bool zero_bytes(const std::uint8_t *data, std::uint64_t count) noexcept {
    for (std::uint64_t i = 0; i < count; ++i) {
        if (data[i] != 0) {
            return false;
        }
    }
    return true;
}

} // namespace tilestream::cpu::detail
