#pragma once

#include "cuda/plan.h"

#include <cstdint>

namespace tilestream::cuda::detail {

inline constexpr std::uint32_t tile_pixels = 4096;
inline constexpr std::uint32_t max_rows = 64;

struct Geometry {
    std::uint64_t x = 0;
    std::uint64_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

__device__ inline Geometry tile_geometry(ImageSpec image, TileShape shape, std::uint64_t tile) {
    const std::uint64_t columns = image.width / shape.width + (image.width % shape.width != 0);
    Geometry result;
    result.x = tile % columns * shape.width;
    result.y = tile / columns * shape.height;
    const auto width = image.width - result.x;
    const auto height = image.height - result.y;
    result.width = static_cast<std::uint32_t>(width < shape.width ? width : shape.width);
    result.height = static_cast<std::uint32_t>(height < shape.height ? height : shape.height);
    return result;
}

// Entries are ordered by their first window tile.
__device__ inline const WindowImage &owner(const WindowImage *entries, std::uint32_t count,
                                           std::uint32_t tile) {
    std::uint32_t low = 0;
    std::uint32_t high = count;
    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        if (entries[middle].first <= tile) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return entries[low];
}

__device__ inline void store_le(std::uint8_t *data, std::uint64_t value, std::uint32_t bytes) {
    for (std::uint32_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

__device__ inline std::uint64_t load_le(const std::uint8_t *data, std::uint32_t bytes) {
    std::uint64_t value = 0;
    for (std::uint32_t i = 0; i < bytes; ++i) {
        value |= std::uint64_t{data[i]} << (8 * i);
    }
    return value;
}

// Exact for every uint16 value; FP32 samples keep their bits, including NaN payloads.
template <typename Sample> __device__ inline float sample_value(std::uint32_t bits) {
    if constexpr (sizeof(Sample) == 2) {
        return static_cast<float>(bits);
    } else {
        return __uint_as_float(bits);
    }
}

// Map coordinates are relative to the output region.
__device__ inline float parameter(const Parameter &value, std::uint64_t x, std::uint64_t y) {
    if (value.kind == ParameterKind::scalar) {
        return value.scalar;
    }
    const auto *row = static_cast<const std::uint8_t *>(value.map.data) + y * value.map.pitch_bytes;
    return reinterpret_cast<const float *>(row)[x];
}

// Two roundings to nearest, subtraction then multiplication; the intrinsics are never
// contracted into an FMA, so the separate and fused paths give the same bits.
__device__ inline float calibrate(float value, const Calibration &calibration, std::uint64_t x,
                                  std::uint64_t y) {
    return __fmul_rn(__fsub_rn(value, parameter(calibration.offset, x, y)),
                     parameter(calibration.gain, x, y));
}

} // namespace tilestream::cuda::detail
