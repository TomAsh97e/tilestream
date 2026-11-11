#pragma once

#include <cstdint>

namespace tilestream {

enum class Status : std::uint32_t {
    success = 0,
    invalid_argument = 1,
    invalid_roi = 2,
    unsupported_type = 3,
    unsupported_format = 4,
    malformed_data = 5,
    size_overflow = 6,
    buffer_too_small = 7,
    resource_limit = 8,
    cuda_error = 9,
    io_error = 10,
};

enum class PixelType : std::uint32_t { uint16 = 1, float32 = 2 };
enum class TileMode : std::uint32_t { raw = 0, constant = 1, predict_bitpack = 2 };

struct ImageSpec {
    PixelType type = PixelType::uint16;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct TileShape {
    std::uint16_t width = 128;
    std::uint16_t height = 32;
};

struct Roi {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct ConstBytes {
    const void *data = nullptr;
    std::uint64_t size = 0;
};

struct MutableBytes {
    void *data = nullptr;
    std::uint64_t size = 0;
};

struct ConstImageView {
    const void *data = nullptr;
    ImageSpec image;
    std::uint64_t pitch_bytes = 0;
    std::uint64_t capacity_bytes = 0;
};

struct ImageView {
    void *data = nullptr;
    ImageSpec image;
    std::uint64_t pitch_bytes = 0;
    std::uint64_t capacity_bytes = 0;
};

// Native host/device records, never serialized as C++ objects.
struct DeviceResult {
    Status status = Status::success;
    std::uint32_t reserved = 0;
    std::uint64_t bytes_written = 0;
};

} // namespace tilestream
