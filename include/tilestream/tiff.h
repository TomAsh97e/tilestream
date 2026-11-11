#pragma once

#include <tilestream/sizing.h>
#include <tilestream/types.h>

#include <cstdint>
#include <vector>

namespace tilestream::io {

enum class TiffCompression : std::uint32_t { none, deflate };

struct TiffReadLimits {
    std::uint32_t max_width = UINT32_MAX;
    std::uint32_t max_height = UINT32_MAX;
    std::uint64_t max_decoded_bytes = max_bytes;
};

struct TiffImage {
    ImageSpec image;
    std::uint64_t pitch_bytes = 0;
    std::vector<std::uint8_t> pixels;
};

struct TiffWriteOptions {
    bool bigtiff = false;
    TiffCompression compression = TiffCompression::none;
    bool horizontal_predictor = false;
    std::uint32_t tile_width = 128;
    std::uint32_t tile_height = 32;
    std::uint32_t deflate_level = 6;
};

[[nodiscard]] Status read_tiff(const char *path, TiffImage &image,
                               const TiffReadLimits &limits = {}) noexcept;
[[nodiscard]] Status write_tiff(const char *path, ConstImageView image,
                                const TiffWriteOptions &options = {}) noexcept;

} // namespace tilestream::io
