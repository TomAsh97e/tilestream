#include <tilestream/cpu.h>

#include "views.h"
#include "wire.h"

#include <cstring>

namespace tilestream::cpu {
namespace {

std::uint32_t unpack(const std::uint8_t *payload, std::uint32_t index,
                     std::uint32_t bits) noexcept {
    if (bits == 0) {
        return 0;
    }
    const auto position = index * bits;
    const auto shift = position % 32;
    const auto *word = payload + (position / 32) * 4;
    if (bits == 32) {
        return detail::read_le<std::uint32_t>(word);
    }
    auto value = detail::read_le<std::uint32_t>(word) >> shift;
    if (shift + bits > 32) {
        value |= detail::read_le<std::uint32_t>(word + 4) << (32 - shift);
    }
    return value & ((std::uint32_t{1} << bits) - 1);
}

void reconstruct(ConstBytes input, ImageView output, const ImageMetadata &metadata) noexcept {
    const auto *source = static_cast<const std::uint8_t *>(input.data);
    auto *destination = static_cast<std::uint8_t *>(output.data);
    const std::uint64_t sample_bytes = metadata.image.type == PixelType::uint16 ? 2 : 4;
    const std::uint64_t columns = metadata.image.width / metadata.tile.width +
                                  (metadata.image.width % metadata.tile.width != 0);
    for (std::size_t i = 0; i < metadata.tiles.size(); ++i) {
        const auto &tile = metadata.tiles[i];
        const auto x = (i % columns) * metadata.tile.width;
        const auto y = (i / columns) * metadata.tile.height;
        const auto *payload = source + tile.offset;
        for (std::uint32_t row = 0; row < tile.height; ++row) {
            auto *line = destination + (y + row) * output.pitch_bytes + x * sample_bytes;
            if (tile.mode == TileMode::raw) {
                for (std::uint32_t col = 0; col < tile.width; ++col) {
                    if (metadata.image.type == PixelType::uint16) {
                        const auto sample = detail::read_le<std::uint16_t>(
                            payload + (std::uint64_t{row} * tile.width + col) * sample_bytes);
                        std::memcpy(line + col * sample_bytes, &sample, 2);
                    } else {
                        const auto sample = detail::read_le<std::uint32_t>(
                            payload + (std::uint64_t{row} * tile.width + col) * sample_bytes);
                        std::memcpy(line + col * sample_bytes, &sample, 4);
                    }
                }
            } else if (tile.mode == TileMode::constant) {
                const auto sample = detail::read_le<std::uint32_t>(payload);
                for (std::uint32_t col = 0; col < tile.width; ++col) {
                    std::memcpy(line + col * sample_bytes, &sample,
                                static_cast<std::size_t>(sample_bytes));
                }
            } else {
                const auto &segment = metadata.segments[tile.first_segment + row];
                auto previous = segment.seed;
                std::memcpy(line, &previous, static_cast<std::size_t>(sample_bytes));
                for (std::uint32_t col = 1; col < tile.width; ++col) {
                    const auto residual =
                        unpack(payload + segment.payload_offset, col - 1, segment.bit_width);
                    if (metadata.image.type == PixelType::uint16) {
                        const auto delta = (residual & 1U) == 0
                                               ? static_cast<std::int32_t>(residual / 2)
                                               : -static_cast<std::int32_t>(residual / 2) - 1;
                        const auto sum = static_cast<std::int32_t>(previous) + delta;
                        previous = static_cast<std::uint32_t>(sum) & 65535U;
                    } else {
                        previous ^= residual;
                    }
                    std::memcpy(line + col * sample_bytes, &previous,
                                static_cast<std::size_t>(sample_bytes));
                }
            }
        }
    }
}

} // namespace

Status decompress(ConstBytes input, ImageView output, const ParseLimits &limits) noexcept {
    auto status =
        detail::codec_view({output.data, output.image, output.pitch_bytes, output.capacity_bytes});
    if (status != Status::success) {
        return status;
    }
    status = detail::byte_range(input.data, input.size);
    if (status != Status::success) {
        return status;
    }
    if (detail::overlaps(input.data, input.size, output.data, output.capacity_bytes)) {
        return Status::invalid_argument;
    }
    ImageMetadata metadata;
    status = parse_metadata(input, limits, metadata);
    if (status != Status::success) {
        return status;
    }
    if (output.image.width != metadata.image.width ||
        output.image.height != metadata.image.height || output.image.type != metadata.image.type) {
        return Status::invalid_argument;
    }
    reconstruct(input, output, metadata);
    return Status::success;
}

} // namespace tilestream::cpu
