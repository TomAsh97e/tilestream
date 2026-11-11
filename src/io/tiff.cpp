#include <tilestream/tiff.h>

#include "checked_size.h"
#include "cpu/views.h"

#include <tiffio.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace tilestream::io {
namespace {

using TiffPtr = std::unique_ptr<TIFF, decltype(&TIFFClose)>;

int discard_message(TIFF *, void *, const char *, const char *, va_list) {
    return 1;
}

TiffPtr open_tiff(const char *path, const char *mode, std::uint64_t allocation_limit) noexcept {
    TIFFOpenOptions *options = TIFFOpenOptionsAlloc();
    if (options == nullptr) {
        return {nullptr, TIFFClose};
    }
    const auto max_tmsize = static_cast<std::uint64_t>(std::numeric_limits<tmsize_t>::max());
    TIFFOpenOptionsSetMaxSingleMemAlloc(
        options, static_cast<tmsize_t>(std::min(allocation_limit, max_tmsize)));
    TIFFOpenOptionsSetErrorHandlerExtR(options, discard_message, nullptr);
    TIFFOpenOptionsSetWarningHandlerExtR(options, discard_message, nullptr);
    TIFF *handle = TIFFOpenExt(path, mode, options);
    TIFFOpenOptionsFree(options);
    return {handle, TIFFClose};
}

Status image_layout(std::uint32_t width, std::uint32_t height, PixelType type,
                    const TiffReadLimits &limits, std::uint64_t &row_bytes,
                    std::uint64_t &total_bytes) noexcept {
    if (width == 0 || height == 0) {
        return Status::unsupported_format;
    }
    if (width > limits.max_width || height > limits.max_height) {
        return Status::resource_limit;
    }
    const std::uint64_t sample_bytes = type == PixelType::uint16 ? 2 : 4;
    if (!detail::multiply_size(width, sample_bytes, row_bytes) ||
        !detail::multiply_size(row_bytes, height, total_bytes)) {
        return Status::size_overflow;
    }
    return total_bytes <= limits.max_decoded_bytes ? Status::success : Status::resource_limit;
}

Status read_tags(TIFF *tiff, const TiffReadLimits &limits, ImageSpec &image,
                 std::uint64_t &row_bytes, std::uint64_t &total_bytes) noexcept {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t samples = 0;
    std::uint16_t bits = 0;
    std::uint16_t sample_format = 0;
    std::uint16_t planar = 0;
    std::uint16_t orientation = 0;
    std::uint16_t photometric = 0;
    if (TIFFNumberOfDirectories(tiff) != 1 || TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) != 1 ||
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) != 1 ||
        TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples) != 1 ||
        TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits) != 1 ||
        TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format) != 1 ||
        TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar) != 1 ||
        TIFFGetFieldDefaulted(tiff, TIFFTAG_ORIENTATION, &orientation) != 1 ||
        TIFFGetFieldDefaulted(tiff, TIFFTAG_PHOTOMETRIC, &photometric) != 1) {
        return Status::unsupported_format;
    }
    if (samples != 1 || planar != PLANARCONFIG_CONTIG || orientation != ORIENTATION_TOPLEFT ||
        photometric != PHOTOMETRIC_MINISBLACK) {
        return Status::unsupported_format;
    }
    if (bits == 16 && sample_format == SAMPLEFORMAT_UINT) {
        image = {PixelType::uint16, width, height};
    } else if (bits == 32 && sample_format == SAMPLEFORMAT_IEEEFP) {
        image = {PixelType::float32, width, height};
    } else {
        return Status::unsupported_format;
    }
    return image_layout(width, height, image.type, limits, row_bytes, total_bytes);
}

Status read_tiles(TIFF *tiff, TiffImage &image, std::uint64_t row_bytes) {
    std::uint32_t tile_width = 0;
    std::uint32_t tile_height = 0;
    if (TIFFGetField(tiff, TIFFTAG_TILEWIDTH, &tile_width) != 1 ||
        TIFFGetField(tiff, TIFFTAG_TILELENGTH, &tile_height) != 1 || tile_width == 0 ||
        tile_height == 0) {
        return Status::unsupported_format;
    }
    const std::uint64_t sample_bytes = image.image.type == PixelType::uint16 ? 2 : 4;
    std::uint64_t expected = 0;
    if (!detail::multiply_size(tile_width, tile_height, expected) ||
        !detail::multiply_size(expected, sample_bytes, expected) ||
        expected > static_cast<std::uint64_t>(std::numeric_limits<tmsize_t>::max())) {
        return Status::size_overflow;
    }
    if (TIFFTileSize64(tiff) != expected) {
        return Status::unsupported_format;
    }
    const auto tiles_x = (std::uint64_t{image.image.width} + tile_width - 1) / tile_width;
    const auto tiles_y = (std::uint64_t{image.image.height} + tile_height - 1) / tile_height;
    std::uint64_t tile_count = 0;
    if (!detail::multiply_size(tiles_x, tiles_y, tile_count) || tile_count > UINT32_MAX) {
        return Status::size_overflow;
    }
    if (TIFFNumberOfTiles(tiff) != tile_count) {
        return Status::unsupported_format;
    }
    std::vector<std::uint8_t> tile(static_cast<std::size_t>(expected));
    for (std::uint64_t y = 0; y < image.image.height; y += tile_height) {
        for (std::uint64_t x = 0; x < image.image.width; x += tile_width) {
            const auto tile_index = TIFFComputeTile(tiff, static_cast<std::uint32_t>(x),
                                                    static_cast<std::uint32_t>(y), 0, 0);
            const auto read =
                TIFFReadEncodedTile(tiff, tile_index, tile.data(), static_cast<tmsize_t>(expected));
            if (read < 0 || static_cast<std::uint64_t>(read) != expected) {
                return Status::io_error;
            }
            const auto copy_width = std::min<std::uint64_t>(tile_width, image.image.width - x);
            const auto copy_height = std::min<std::uint64_t>(tile_height, image.image.height - y);
            const auto bytes = copy_width * sample_bytes;
            for (std::uint64_t row = 0; row < copy_height; ++row) {
                std::memcpy(image.pixels.data() + (y + row) * row_bytes + x * sample_bytes,
                            tile.data() + row * tile_width * sample_bytes,
                            static_cast<std::size_t>(bytes));
            }
        }
    }
    return Status::success;
}

Status read_strips(TIFF *tiff, TiffImage &image, std::uint64_t row_bytes) {
    std::uint32_t rows_per_strip = 0;
    if (TIFFGetFieldDefaulted(tiff, TIFFTAG_ROWSPERSTRIP, &rows_per_strip) != 1 ||
        rows_per_strip == 0) {
        return Status::unsupported_format;
    }
    const auto max_rows = std::min<std::uint64_t>(rows_per_strip, image.image.height);
    std::uint64_t regular_bytes = 0;
    if (!detail::multiply_size(row_bytes, max_rows, regular_bytes) ||
        regular_bytes > static_cast<std::uint64_t>(std::numeric_limits<tmsize_t>::max())) {
        return Status::size_overflow;
    }
    if (TIFFStripSize64(tiff) != regular_bytes) {
        return Status::unsupported_format;
    }
    const auto strip_count =
        (std::uint64_t{image.image.height} + rows_per_strip - 1) / rows_per_strip;
    if (TIFFNumberOfStrips(tiff) != strip_count) {
        return Status::unsupported_format;
    }
    std::vector<std::uint8_t> strip(static_cast<std::size_t>(regular_bytes));
    for (std::uint64_t y = 0; y < image.image.height; y += rows_per_strip) {
        const auto rows = std::min<std::uint64_t>(rows_per_strip, image.image.height - y);
        const auto expected = rows * row_bytes;
        const auto strip_index = TIFFComputeStrip(tiff, static_cast<std::uint32_t>(y), 0);
        const auto read =
            TIFFReadEncodedStrip(tiff, strip_index, strip.data(), static_cast<tmsize_t>(expected));
        if (read < 0 || static_cast<std::uint64_t>(read) != expected) {
            return Status::io_error;
        }
        std::memcpy(image.pixels.data() + y * row_bytes, strip.data(),
                    static_cast<std::size_t>(expected));
    }
    return Status::success;
}

bool set_common_tags(TIFF *tiff, ConstImageView image, const TiffWriteOptions &options) noexcept {
    const auto bits = static_cast<std::uint16_t>(image.image.type == PixelType::uint16 ? 16 : 32);
    const auto sample_format = static_cast<std::uint16_t>(
        image.image.type == PixelType::uint16 ? SAMPLEFORMAT_UINT : SAMPLEFORMAT_IEEEFP);
    const auto compression = static_cast<std::uint16_t>(options.compression == TiffCompression::none
                                                            ? COMPRESSION_NONE
                                                            : COMPRESSION_ADOBE_DEFLATE);
    return TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, image.image.width) == 1 &&
           TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, image.image.height) == 1 &&
           TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits) == 1 &&
           TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1) == 1 &&
           TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, sample_format) == 1 &&
           TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) == 1 &&
           TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) == 1 &&
           TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT) == 1 &&
           TIFFSetField(tiff, TIFFTAG_TILEWIDTH, options.tile_width) == 1 &&
           TIFFSetField(tiff, TIFFTAG_TILELENGTH, options.tile_height) == 1 &&
           TIFFSetField(tiff, TIFFTAG_COMPRESSION, compression) == 1;
}

Status validate_write_options(ConstImageView image, const TiffWriteOptions &options) noexcept {
    const auto view_status = cpu::detail::codec_view(image);
    if (view_status != Status::success) {
        return view_status;
    }
    if (options.compression != TiffCompression::none &&
        options.compression != TiffCompression::deflate) {
        return Status::invalid_argument;
    }
    if (options.tile_width == 0 || options.tile_height == 0 || options.tile_width % 16 != 0 ||
        options.tile_height % 16 != 0) {
        return Status::invalid_argument;
    }
    if (options.horizontal_predictor && (image.image.type != PixelType::uint16 ||
                                         options.compression != TiffCompression::deflate)) {
        return Status::invalid_argument;
    }
    if (options.compression == TiffCompression::deflate &&
        (options.deflate_level < 1 || options.deflate_level > 9)) {
        return Status::invalid_argument;
    }
    return Status::success;
}

} // namespace

Status read_tiff(const char *path, TiffImage &image, const TiffReadLimits &limits) noexcept {
    if (path == nullptr || limits.max_width == 0 || limits.max_height == 0 ||
        limits.max_decoded_bytes > max_bytes) {
        return Status::invalid_argument;
    }
    auto tiff = open_tiff(path, "r", max_bytes);
    if (!tiff) {
        return Status::io_error;
    }
    try {
        TiffImage decoded;
        std::uint64_t total_bytes = 0;
        auto status =
            read_tags(tiff.get(), limits, decoded.image, decoded.pitch_bytes, total_bytes);
        if (status != Status::success) {
            return status;
        }
        decoded.pixels.resize(static_cast<std::size_t>(total_bytes));
        status = TIFFIsTiled(tiff.get()) != 0
                     ? read_tiles(tiff.get(), decoded, decoded.pitch_bytes)
                     : read_strips(tiff.get(), decoded, decoded.pitch_bytes);
        if (status != Status::success) {
            return status;
        }
        image = std::move(decoded);
        return Status::success;
    } catch (const std::bad_alloc &) {
        return Status::resource_limit;
    } catch (const std::length_error &) {
        return Status::resource_limit;
    }
}

Status write_tiff(const char *path, ConstImageView image,
                  const TiffWriteOptions &options) noexcept {
    if (path == nullptr) {
        return Status::invalid_argument;
    }
    const auto status = validate_write_options(image, options);
    if (status != Status::success) {
        return status;
    }
    const std::uint64_t sample_bytes = image.image.type == PixelType::uint16 ? 2 : 4;
    std::uint64_t tile_bytes = 0;
    if (!detail::multiply_size(options.tile_width, options.tile_height, tile_bytes) ||
        !detail::multiply_size(tile_bytes, sample_bytes, tile_bytes) ||
        tile_bytes > static_cast<std::uint64_t>(std::numeric_limits<tmsize_t>::max())) {
        return Status::size_overflow;
    }
    const auto tiles_x =
        (std::uint64_t{image.image.width} + options.tile_width - 1) / options.tile_width;
    const auto tiles_y =
        (std::uint64_t{image.image.height} + options.tile_height - 1) / options.tile_height;
    std::uint64_t tile_count = 0;
    if (!detail::multiply_size(tiles_x, tiles_y, tile_count) || tile_count > UINT32_MAX) {
        return Status::size_overflow;
    }
    auto tiff = open_tiff(path, options.bigtiff ? "w8" : "w", max_bytes);
    if (!tiff) {
        return Status::io_error;
    }
    try {
        if (!set_common_tags(tiff.get(), image, options)) {
            tiff.reset();
            std::remove(path);
            return Status::io_error;
        }
        if (options.compression == TiffCompression::deflate) {
            if (TIFFSetField(tiff.get(), TIFFTAG_ZIPQUALITY, options.deflate_level) != 1 ||
                TIFFSetField(tiff.get(), TIFFTAG_PREDICTOR,
                             options.horizontal_predictor ? PREDICTOR_HORIZONTAL
                                                          : PREDICTOR_NONE) != 1) {
                tiff.reset();
                std::remove(path);
                return Status::io_error;
            }
        }
        std::vector<std::uint8_t> tile(static_cast<std::size_t>(tile_bytes));
        const auto *source = static_cast<const std::uint8_t *>(image.data);
        for (std::uint64_t y = 0; y < image.image.height; y += options.tile_height) {
            for (std::uint64_t x = 0; x < image.image.width; x += options.tile_width) {
                std::fill(tile.begin(), tile.end(), 0);
                const auto copy_width =
                    std::min<std::uint64_t>(options.tile_width, image.image.width - x);
                const auto copy_height =
                    std::min<std::uint64_t>(options.tile_height, image.image.height - y);
                const auto bytes = copy_width * sample_bytes;
                for (std::uint64_t row = 0; row < copy_height; ++row) {
                    std::memcpy(tile.data() + row * options.tile_width * sample_bytes,
                                source + (y + row) * image.pitch_bytes + x * sample_bytes,
                                static_cast<std::size_t>(bytes));
                }
                const auto index = TIFFComputeTile(tiff.get(), static_cast<std::uint32_t>(x),
                                                   static_cast<std::uint32_t>(y), 0, 0);
                const auto written = TIFFWriteEncodedTile(tiff.get(), index, tile.data(),
                                                          static_cast<tmsize_t>(tile_bytes));
                if (written < 0 || static_cast<std::uint64_t>(written) != tile_bytes) {
                    tiff.reset();
                    std::remove(path);
                    return Status::io_error;
                }
            }
        }
        if (TIFFFlush(tiff.get()) != 1) {
            tiff.reset();
            std::remove(path);
            return Status::io_error;
        }
        return Status::success;
    } catch (const std::bad_alloc &) {
        tiff.reset();
        std::remove(path);
        return Status::resource_limit;
    } catch (const std::length_error &) {
        tiff.reset();
        std::remove(path);
        return Status::resource_limit;
    }
}

} // namespace tilestream::io
