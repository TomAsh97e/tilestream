#include <tilestream/cpu.h>
#include <tilestream/format.h>

#include "views.h"
#include "wire.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace tilestream::cpu {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

struct TileAnalysis {
    std::array<std::uint32_t, 4096> samples{};
    std::array<std::uint32_t, 4096> residuals{};
    std::array<std::uint8_t, 64> widths{};
    TileMode mode = TileMode::raw;
    std::uint64_t length = 0;
};

std::uint32_t load_sample(const std::uint8_t *source, PixelType type) noexcept {
    std::uint32_t sample = 0;
    const std::size_t size = type == PixelType::uint16 ? 2 : 4;
    std::memcpy(&sample, source, size);
    return sample;
}

std::uint32_t uint16_residual(std::uint32_t current, std::uint32_t previous) noexcept {
    const auto delta = (current - previous) & std::uint32_t{65535};
    const auto signed_delta =
        delta < 32768 ? static_cast<std::int32_t>(delta) : static_cast<std::int32_t>(delta) - 65536;
    return static_cast<std::uint32_t>(signed_delta >= 0 ? signed_delta * 2 : -signed_delta * 2 - 1);
}

void analyze(ConstImageView input, std::uint64_t x, std::uint64_t y, std::uint16_t width,
             std::uint16_t height, TileAnalysis &tile) noexcept {
    const auto *data = static_cast<const std::uint8_t *>(input.data);
    const std::uint64_t sample_bytes = input.image.type == PixelType::uint16 ? 2 : 4;
    bool constant = true;
    std::uint64_t predictive_bytes = 0;
    for (std::uint32_t row = 0; row < height; ++row) {
        std::uint32_t combined = 0;
        for (std::uint32_t col = 0; col < width; ++col) {
            const auto index = row * width + col;
            tile.samples[index] = load_sample(
                data + (y + row) * input.pitch_bytes + (x + col) * sample_bytes, input.image.type);
            constant = constant && tile.samples[index] == tile.samples[0];
            if (col != 0) {
                const auto residual =
                    input.image.type == PixelType::uint16
                        ? uint16_residual(tile.samples[index], tile.samples[index - 1])
                        : tile.samples[index] ^ tile.samples[index - 1];
                tile.residuals[index] = residual;
                combined |= residual;
            }
        }
        std::uint8_t bits = 0;
        while (combined != 0) {
            ++bits;
            combined >>= 1;
        }
        tile.widths[row] = bits;
        predictive_bytes += 8 + ((std::uint64_t{width - 1U} * bits + 31) / 32) * 4;
    }
    tile.mode = TileMode::raw;
    tile.length = ((std::uint64_t{width} * height * sample_bytes + 3) / 4) * 4;
    if (constant && 4 < tile.length) {
        tile.mode = TileMode::constant;
        tile.length = 4;
    }
    if (predictive_bytes < tile.length) {
        tile.mode = TileMode::predict_bitpack;
        tile.length = predictive_bytes;
    }
}

void write_payload(std::uint8_t *output, const TileAnalysis &tile, PixelType type,
                   std::uint16_t width, std::uint16_t height) noexcept {
    std::memset(output, 0, static_cast<std::size_t>(tile.length));
    if (tile.mode == TileMode::raw) {
        const std::uint64_t sample_bytes = type == PixelType::uint16 ? 2 : 4;
        for (std::uint32_t i = 0; i < std::uint32_t{width} * height; ++i) {
            if (type == PixelType::uint16) {
                detail::write_le(output + i * sample_bytes,
                                 static_cast<std::uint16_t>(tile.samples[i]));
            } else {
                detail::write_le(output + i * sample_bytes, tile.samples[i]);
            }
        }
    } else if (tile.mode == TileMode::constant) {
        if (type == PixelType::uint16) {
            detail::write_le(output, static_cast<std::uint16_t>(tile.samples[0]));
        } else {
            detail::write_le(output, tile.samples[0]);
        }
    } else {
        for (std::uint32_t row = 0; row < height; ++row) {
            detail::write_le(output, tile.samples[row * width]);
            const auto bits = tile.widths[row];
            output[format::segment::bit_width] = bits;
            output += format::segment_header_bytes;
            // The encoder writes bits directly; the scalar decoder reads words independently.
            for (std::uint32_t col = 1; col < width; ++col) {
                const auto residual = tile.residuals[row * width + col];
                for (std::uint32_t bit = 0; bit < bits; ++bit) {
                    const auto position = (col - 1) * bits + bit;
                    const auto value = (residual >> bit) & 1U;
                    output[position / 8] |= static_cast<std::uint8_t>(value << (position % 8));
                }
            }
            output += ((std::uint64_t{width - 1U} * bits + 31) / 32) * 4;
        }
    }
}

void write_header(std::uint8_t *output, ImageSpec image, TileShape tile, std::uint64_t count,
                  std::uint64_t length) noexcept {
    std::memset(output, 0, format::header_bytes);
    std::copy(format::magic.begin(), format::magic.end(), output);
    detail::write_le(output + format::header::version_major, format::major_version);
    detail::write_le(output + format::header::version_minor, format::minor_version);
    detail::write_le(output + format::header::size,
                     static_cast<std::uint16_t>(format::header_bytes));
    output[format::header::pixel_type] = static_cast<std::uint8_t>(image.type);
    detail::write_le(output + format::header::width, image.width);
    detail::write_le(output + format::header::height, image.height);
    detail::write_le(output + format::header::tile_width, tile.width);
    detail::write_le(output + format::header::tile_height, tile.height);
    detail::write_le(output + format::header::entry_size,
                     static_cast<std::uint16_t>(format::index_entry_bytes));
    detail::write_le(output + format::header::tile_count, count);
    detail::write_le(output + format::header::index_offset, format::header_bytes);
    detail::write_le(output + format::header::payload_offset,
                     format::header_bytes + count * format::index_entry_bytes);
    detail::write_le(output + format::header::file_size, length);
}

} // namespace

Status compress(ConstImageView input, TileShape tile, MutableBytes output,
                std::uint64_t &bytes_written) noexcept {
    auto status = detail::codec_view(input);
    if (status != Status::success) {
        return status;
    }
    std::uint64_t bound = 0;
    status = compress_bound(input.image, tile, bound);
    if (status != Status::success) {
        return status;
    }
    status = detail::byte_range(output.data, output.size);
    if (status != Status::success) {
        return status;
    }
    if (output.size < bound) {
        return Status::buffer_too_small;
    }
    if (detail::overlaps(input.data, input.capacity_bytes, output.data, output.size) ||
        detail::overlaps(&bytes_written, sizeof(bytes_written), output.data, output.size) ||
        detail::overlaps(&bytes_written, sizeof(bytes_written), input.data, input.capacity_bytes)) {
        return Status::invalid_argument;
    }

    const std::uint64_t columns =
        input.image.width / tile.width + (input.image.width % tile.width != 0);
    const std::uint64_t rows =
        input.image.height / tile.height + (input.image.height % tile.height != 0);
    const auto count = columns * rows;
    auto cursor = format::header_bytes + count * format::index_entry_bytes;
    auto *data = static_cast<std::uint8_t *>(output.data);
    TileAnalysis analysis;
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto x = (i % columns) * tile.width;
        const auto y = (i / columns) * tile.height;
        const auto width =
            static_cast<std::uint16_t>(std::min<std::uint64_t>(tile.width, input.image.width - x));
        const auto height = static_cast<std::uint16_t>(
            std::min<std::uint64_t>(tile.height, input.image.height - y));
        analyze(input, x, y, width, height, analysis);
        auto *entry = data + format::header_bytes + i * format::index_entry_bytes;
        std::memset(entry, 0, format::index_entry_bytes);
        entry[format::entry::mode] = static_cast<std::uint8_t>(analysis.mode);
        detail::write_le(entry + format::entry::width, width);
        detail::write_le(entry + format::entry::height, height);
        detail::write_le(entry + format::entry::offset, cursor);
        detail::write_le(entry + format::entry::length, analysis.length);
        write_payload(data + cursor, analysis, input.image.type, width, height);
        cursor += analysis.length;
    }
    write_header(data, input.image, tile, count, cursor);
    bytes_written = cursor;
    return Status::success;
}

} // namespace tilestream::cpu
