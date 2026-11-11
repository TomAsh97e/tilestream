#pragma once

// Deterministic random round-trip cases shared by the CPU and CUDA robustness suites and the
// fuzzers. Every case comes from its own seed, so one failure replays with TILESTREAM_CASE=seed.
#include <tilestream/types.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace robustness {

using tilestream::ImageSpec;
using tilestream::PixelType;
using tilestream::Roi;
using tilestream::TileShape;

inline constexpr std::uint64_t first_seed = 13000000;
inline constexpr std::uint64_t default_count = 10000;
inline constexpr TileShape shapes[] = {{128, 32}, {64, 64}, {256, 16}};
inline constexpr std::uint32_t pattern_count = 8;

// Words of every IEEE class, used by the special pattern.
inline constexpr std::uint32_t special_words[] = {0x00000000, 0x80000000, 0x00000001, 0x807fffff,
                                                  0x00800000, 0x7f7fffff, 0xff800000, 0x7f800000,
                                                  0x7fc00000, 0xffc12345, 0x7f800001, 0x3f800000};

inline std::uint64_t sample_bytes(PixelType type) {
    return type == PixelType::uint16 ? 2 : 4;
}

// Fills the logical pixels of a pitched buffer. Patterns cover constant tiles, ramps, wrapping
// differences, rows with a chosen residual width, noise, smooth fields, IEEE special words and
// checkerboards with flipped bits.
inline void fill(ImageSpec image, std::uint64_t pitch, std::uint32_t pattern, std::uint64_t seed,
                 std::uint8_t *bytes) {
    std::mt19937_64 rng(seed);
    const auto s = sample_bytes(image.type);
    const std::uint32_t mask = image.type == PixelType::uint16 ? 0xffffU : 0xffffffffU;
    const auto base = static_cast<std::uint32_t>(rng());
    const auto bits =
        static_cast<std::uint32_t>(rng() % (image.type == PixelType::uint16 ? 17 : 33));
    const std::uint32_t spread = bits == 32 ? 0xffffffffU : (1U << bits) - 1;
    for (std::uint32_t y = 0; y < image.height; ++y) {
        std::uint32_t previous = base;
        for (std::uint32_t x = 0; x < image.width; ++x) {
            std::uint32_t value = 0;
            switch (pattern % pattern_count) {
            case 0:
                value = base;
                break;
            case 1:
                value = base + x * 3 + y * 7;
                break;
            case 2:
                value = 65535U - x - y * image.width;
                break;
            case 3:
                // Residuals of at most `bits` bits, in the transform of each type.
                value = image.type == PixelType::uint16
                            ? previous + (static_cast<std::uint32_t>(rng()) & spread)
                            : previous ^ (static_cast<std::uint32_t>(rng()) & spread);
                break;
            case 4:
                value = static_cast<std::uint32_t>(rng());
                break;
            case 5: {
                const auto field = 1.5F + 0.001F * static_cast<float>(x) +
                                   0.01F * static_cast<float>(y) +
                                   0.0001F * static_cast<float>(rng() % 7);
                if (image.type == PixelType::uint16) {
                    value = 1000 + x / 4 + y / 8 + static_cast<std::uint32_t>(rng() % 16);
                } else {
                    std::memcpy(&value, &field, 4);
                }
                break;
            }
            case 6:
                value = image.type == PixelType::uint16
                            ? (rng() % 2 == 0 ? 0U : 65535U)
                            : special_words[rng() % std::size(special_words)];
                break;
            default:
                value = (x / 8 + y / 4) % 2 == 0 ? base : base ^ (1U << (rng() % 32));
                break;
            }
            previous = value & mask;
            std::memcpy(bytes + y * pitch + x * s, &previous, static_cast<std::size_t>(s));
        }
    }
}

struct Case {
    std::uint64_t seed = 0;
    ImageSpec image;
    TileShape tile;
    std::uint64_t pitch = 0;
    std::uint32_t pattern = 0;
    Roi roi;
    std::vector<std::uint8_t> bytes;

    std::uint32_t get(std::uint64_t x, std::uint64_t y) const {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + y * pitch + x * sample_bytes(image.type),
                    static_cast<std::size_t>(sample_bytes(image.type)));
        return value;
    }
    // Logical pixels of a region, without padding.
    std::vector<std::uint8_t> region(const Roi &r) const {
        const auto s = sample_bytes(image.type);
        std::vector<std::uint8_t> result;
        for (std::uint64_t y = r.y; y < std::uint64_t{r.y} + r.height; ++y) {
            const auto *line = bytes.data() + y * pitch + r.x * s;
            result.insert(result.end(), line, line + r.width * s);
        }
        return result;
    }
    std::vector<std::uint8_t> pixels() const {
        return region({0, 0, image.width, image.height});
    }
    std::string describe() const {
        return "case seed " + std::to_string(seed) + " (" +
               (image.type == PixelType::uint16 ? "uint16 " : "float32 ") +
               std::to_string(image.width) + "x" + std::to_string(image.height) + ", tile " +
               std::to_string(tile.width) + "x" + std::to_string(tile.height) + ", pattern " +
               std::to_string(pattern) + "); replay with TILESTREAM_CASE=" + std::to_string(seed);
    }
};

// Sizes around warp, word and tile boundaries half of the time.
inline std::uint32_t pick_size(std::mt19937_64 &rng, std::uint32_t limit) {
    static constexpr std::uint32_t edges[] = {1,  2,   3,   31,  32,  33,  63, 64,
                                              65, 127, 128, 129, 255, 256, 257};
    if (rng() % 2 == 0) {
        const auto value = edges[rng() % std::size(edges)];
        if (value <= limit) {
            return value;
        }
    }
    return 1 + static_cast<std::uint32_t>(rng() % limit);
}

// Type and tile geometry follow the case number so that both are evenly represented.
inline Case make_case(std::uint64_t seed) {
    const auto index = seed - first_seed;
    std::mt19937_64 rng(seed);
    Case result;
    result.seed = seed;
    result.image.type = index % 2 == 0 ? PixelType::uint16 : PixelType::float32;
    result.tile = shapes[index / 2 % std::size(shapes)];
    result.image.width = pick_size(rng, 300);
    result.image.height = pick_size(rng, 160);
    const auto padding = rng() % 4 == 0 ? rng() % 5 : 0;
    result.pitch = (result.image.width + padding) * sample_bytes(result.image.type);
    result.pattern = static_cast<std::uint32_t>(rng() % pattern_count);
    if (rng() % 4 == 0) {
        result.roi = {0, 0, result.image.width, result.image.height};
    } else {
        const auto x = static_cast<std::uint32_t>(rng() % result.image.width);
        const auto y = static_cast<std::uint32_t>(rng() % result.image.height);
        result.roi = {x, y, 1 + static_cast<std::uint32_t>(rng() % (result.image.width - x)),
                      1 + static_cast<std::uint32_t>(rng() % (result.image.height - y))};
    }
    result.bytes.assign(static_cast<std::size_t>(result.pitch * result.image.height), 0x5a);
    fill(result.image, result.pitch, result.pattern, rng(), result.bytes.data());
    return result;
}

// TILESTREAM_CASE selects one seed, TILESTREAM_CASES changes the number of cases.
inline std::vector<std::uint64_t> selected_seeds() {
    if (const char *one = std::getenv("TILESTREAM_CASE")) {
        return {std::strtoull(one, nullptr, 10)};
    }
    auto count = default_count;
    if (const char *many = std::getenv("TILESTREAM_CASES")) {
        count = std::strtoull(many, nullptr, 10);
    }
    std::vector<std::uint64_t> seeds;
    for (std::uint64_t i = 0; i < count; ++i) {
        seeds.push_back(first_seed + i);
    }
    return seeds;
}

} // namespace robustness
