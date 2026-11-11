#include <tilestream/cpu.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace tilestream;
using Bytes = std::vector<std::uint8_t>;
using Words = std::vector<std::uint32_t>;

Bytes golden(const std::string &name) {
    std::ifstream input(std::string(TILESTREAM_GOLDEN_DIR) + "/" + name + ".hex");
    EXPECT_TRUE(input.is_open()) << name;
    Bytes result;
    unsigned int value = 0;
    while (input >> std::hex >> value) {
        EXPECT_LE(value, 255U);
        result.push_back(static_cast<std::uint8_t>(value));
    }
    EXPECT_TRUE(input.eof());
    return result;
}

Bytes encode(const void *data, std::uint64_t capacity, std::uint32_t width, std::uint32_t height,
             std::uint64_t pitch = 0, TileShape tile = {}) {
    const ImageSpec image{PixelType::float32, width, height};
    if (pitch == 0) {
        pitch = std::uint64_t{width} * 4;
    }
    std::uint64_t bound = 0;
    EXPECT_EQ(compress_bound(image, tile, bound), Status::success);
    Bytes file(static_cast<std::size_t>(bound + 16), 0xa5);
    std::uint64_t written = 0;
    EXPECT_EQ(cpu::compress({data, image, pitch, capacity}, tile, {file.data(), bound}, written),
              Status::success);
    EXPECT_LE(written, bound);
    if (written > bound) {
        return {};
    }
    EXPECT_TRUE(std::all_of(file.begin() + static_cast<std::ptrdiff_t>(written), file.end(),
                            [](auto byte) { return byte == 0xa5; }));
    file.resize(static_cast<std::size_t>(written));
    return file;
}

Bytes encode(const Words &words, std::uint32_t width, std::uint32_t height, TileShape tile = {}) {
    return encode(words.data(), words.size() * 4, width, height, std::uint64_t{width} * 4, tile);
}

Words decode(const Bytes &file, std::uint32_t width, std::uint32_t height) {
    auto storage = std::make_unique<std::uint8_t[]>(file.size() + 1);
    std::copy(file.begin(), file.end(), storage.get() + 1);
    Words output(static_cast<std::size_t>(width) * height, 0xdeadbeef);
    EXPECT_EQ(
        cpu::decompress({storage.get() + 1, file.size()}, {output.data(),
                                                           {PixelType::float32, width, height},
                                                           std::uint64_t{width} * 4,
                                                           output.size() * 4}),
        Status::success);
    return output;
}

TEST(Fp32Codec, HandCalculatedRawAndNegativeZeroConstant) {
    struct Case {
        const char *name;
        std::uint32_t width;
        std::uint32_t height;
        Words words;
        TileMode mode;
    };
    const std::vector<Case> cases = {
        {"float32_raw", 1, 1, {0x7fc01234}, TileMode::raw},
        {"float32_negative_zero_constant", 3, 2, Words(6, 0x80000000), TileMode::constant},
    };
    for (const auto &test : cases) {
        SCOPED_TRACE(test.name);
        const auto file = golden(test.name);
        EXPECT_EQ(encode(test.words, test.width, test.height), file);
        EXPECT_EQ(decode(file, test.width, test.height), test.words);
        ImageMetadata metadata;
        ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        EXPECT_EQ(metadata.image.type, PixelType::float32);
        ASSERT_EQ(metadata.tiles.size(), 1U);
        EXPECT_EQ(metadata.tiles[0].mode, test.mode);
    }
}

TEST(Fp32Codec, EveryXorWidthAndFinalPackedWords) {
    std::mt19937 rng(2026517);
    for (const auto width : {31U, 32U, 33U, 127U, 128U}) {
        for (std::uint32_t bits = 0; bits <= 32; ++bits) {
            SCOPED_TRACE(::testing::Message() << "width=" << width << " bits=" << bits);
            Words words(width * 2, 0x3f800000);
            words[width] = 0xa5a5a5a5;
            const auto mask = bits == 32 ? UINT32_MAX : (std::uint32_t{1} << bits) - 1;
            std::vector<std::uint32_t> residuals;
            for (std::uint32_t x = 1; x < width; ++x) {
                const auto residual = x == 1 ? mask : static_cast<std::uint32_t>(rng()) & mask;
                words[width + x] = words[width + x - 1] ^ residual;
                residuals.push_back(residual);
            }
            const auto file = encode(words, width, 2);
            ImageMetadata metadata;
            ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
            ASSERT_EQ(metadata.tiles.size(), 1U);
            ASSERT_EQ(metadata.tiles[0].mode, TileMode::predict_bitpack);
            ASSERT_EQ(metadata.segments.size(), 2U);
            EXPECT_EQ(metadata.segments[0].bit_width, 0U);
            const auto &row = metadata.segments[1];
            EXPECT_EQ(row.bit_width, bits);
            EXPECT_EQ(row.seed, 0xa5a5a5a5U);

            Bytes expected;
            std::uint64_t reservoir = 0;
            std::uint32_t occupied = 0;
            for (const auto residual : residuals) {
                reservoir |= std::uint64_t{residual} << occupied;
                occupied += bits;
                if (occupied >= 32) {
                    for (unsigned int byte = 0; byte < 4; ++byte) {
                        expected.push_back(static_cast<std::uint8_t>(reservoir >> (8 * byte)));
                    }
                    reservoir >>= 32;
                    occupied -= 32;
                }
            }
            if (occupied != 0) {
                for (unsigned int byte = 0; byte < 4; ++byte) {
                    expected.push_back(static_cast<std::uint8_t>(reservoir >> (8 * byte)));
                }
            }
            ASSERT_EQ(row.payload_bytes, expected.size());
            const auto start = metadata.tiles[0].offset + row.payload_offset;
            EXPECT_TRUE(std::equal(expected.begin(), expected.end(),
                                   file.begin() + static_cast<std::ptrdiff_t>(start)));
            EXPECT_EQ(decode(file, width, 2), words);
        }
    }
}

TEST(Fp32Codec, SpecialRepresentationsSurviveAsBytes) {
    const Words patterns = {
        0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0x7fc01234, 0x7fa12345,
        0xffc0abcd, 0x3f800000, 0xbf800000, 0x00800000,
    };
    std::vector<float> input(patterns.size());
    std::memcpy(input.data(), patterns.data(), patterns.size() * sizeof(std::uint32_t));
    const auto width = static_cast<std::uint32_t>(input.size());
    const auto file = encode(input.data(), input.size() * sizeof(float), width, 1);
    std::vector<float> output(input.size());
    ASSERT_EQ(cpu::decompress({file.data(), file.size()},
                              {output.data(),
                               {PixelType::float32, static_cast<std::uint32_t>(input.size()), 1},
                               input.size() * sizeof(float),
                               output.size() * sizeof(float)}),
              Status::success);
    EXPECT_EQ(std::memcmp(input.data(), output.data(), input.size() * sizeof(float)), 0);

    Words recovered(patterns.size());
    std::memcpy(recovered.data(), output.data(), output.size() * sizeof(float));
    EXPECT_EQ(recovered, patterns);
}

TEST(Fp32Codec, ConstantsUseBitEquality) {
    for (const auto word : {0x00000000U, 0x80000000U, 0x7fc01234U, 0x7fa12345U}) {
        const Words words(16, word);
        const auto file = encode(words, 16, 1);
        ImageMetadata metadata;
        ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        ASSERT_EQ(metadata.tiles.size(), 1U);
        EXPECT_EQ(metadata.tiles[0].mode, TileMode::constant);
        EXPECT_EQ(decode(file, 16, 1), words);
    }
    for (const Words &words : {Words{0x00000000, 0x80000000, 0x00000000, 0x80000000},
                               Words{0x7fc01234, 0x7fc01235, 0x7fc01234, 0x7fc01235}}) {
        const auto file = encode(words, 4, 1);
        ImageMetadata metadata;
        ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        ASSERT_EQ(metadata.tiles.size(), 1U);
        EXPECT_NE(metadata.tiles[0].mode, TileMode::constant);
        EXPECT_EQ(decode(file, 4, 1), words);
    }
}

TEST(Fp32Codec, OrdinaryFieldsWithPitchAndEdges) {
    constexpr std::uint32_t width = 129;
    constexpr std::uint32_t height = 33;
    constexpr std::uint32_t stride = 136;
    std::vector<float> packed(width * height);
    std::vector<float> padded(stride * height, -999.0F);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const float value = static_cast<float>(x) * 0.25F - static_cast<float>(y) * 2.0F;
            packed[y * width + x] = padded[y * stride + x] = value;
        }
    }
    const auto original = padded;
    const auto packed_file = encode(packed.data(), packed.size() * sizeof(float), width, height);
    const auto padded_file =
        encode(padded.data(), padded.size() * sizeof(float), width, height, stride * sizeof(float));
    EXPECT_EQ(padded_file, packed_file);
    EXPECT_EQ(padded, original);

    std::vector<float> output(stride * height, -999.0F);
    ASSERT_EQ(cpu::decompress({packed_file.data(), packed_file.size()},
                              {output.data(),
                               {PixelType::float32, width, height},
                               stride * sizeof(float),
                               output.size() * sizeof(float)}),
              Status::success);
    EXPECT_EQ(std::memcmp(output.data(), original.data(), output.size() * sizeof(float)), 0);
}

TEST(Fp32Codec, SeededRandomBitPatternsSelectRaw) {
    for (std::uint32_t trial = 0; trial < 100; ++trial) {
        const auto seed = 2026517U + trial;
        SCOPED_TRACE(::testing::Message() << "seed=" << seed);
        std::mt19937 rng(seed);
        Words words(128 * 32);
        std::generate(words.begin(), words.end(), [&rng] { return rng(); });
        const auto file = encode(words, 128, 32);
        ImageMetadata metadata;
        ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        ASSERT_EQ(metadata.tiles.size(), 1U);
        EXPECT_EQ(metadata.tiles[0].mode, TileMode::raw);
        EXPECT_EQ(decode(file, 128, 32), words);
    }
}

TEST(Fp32Parser, RejectsInvalidWidthAndPadding) {
    Words words(33 * 2, 0x3f800000);
    words[33] = 0x01234567;
    for (std::uint32_t x = 1; x < 33; ++x) {
        words[33 + x] = words[33 + x - 1] ^ (x == 1 ? UINT32_MAX : x);
    }
    const auto file = encode(words, 33, 2);
    ImageMetadata metadata;
    ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
    ASSERT_EQ(metadata.tiles[0].mode, TileMode::predict_bitpack);
    const auto second = metadata.tiles[0].offset + 8;
    auto bad_width = file;
    bad_width[second + metadata.segments[0].payload_bytes + 4] = 33;
    EXPECT_EQ(parse_metadata({bad_width.data(), bad_width.size()}, {}, metadata),
              Status::malformed_data);

    Words narrow(32 * 2, 0x3f800000);
    narrow[32] = 0x40000000;
    for (std::uint32_t x = 1; x < 32; ++x) {
        narrow[32 + x] = narrow[32 + x - 1] ^ (x == 1 ? 7U : 1U);
    }
    auto padding = encode(narrow, 32, 2);
    ASSERT_EQ(parse_metadata({padding.data(), padding.size()}, {}, metadata), Status::success);
    const auto &segment = metadata.segments[1];
    ASSERT_EQ(segment.bit_width, 3U);
    const auto last = metadata.tiles[0].offset + segment.payload_offset + segment.payload_bytes - 1;
    padding[last] |= 0x80;
    EXPECT_EQ(parse_metadata({padding.data(), padding.size()}, {}, metadata),
              Status::malformed_data);
}

TEST(Fp32Codec, InvalidAlignmentAndTypeMismatchDoNotWrite) {
    alignas(4) std::array<std::uint8_t, 32> unaligned{};
    Bytes output(256, 0xa5);
    const auto before = output;
    std::uint64_t written = 999;
    EXPECT_EQ(cpu::compress({unaligned.data() + 1, {PixelType::float32, 1, 1}, 4, 4}, {},
                            {output.data(), output.size()}, written),
              Status::invalid_argument);
    EXPECT_EQ(output, before);
    EXPECT_EQ(written, 999U);

    const auto file = golden("float32_raw");
    std::array<std::uint16_t, 2> destination = {0xdead, 0xbeef};
    const auto original = destination;
    EXPECT_EQ(
        cpu::decompress({file.data(), file.size()},
                        {destination.data(), {PixelType::uint16, 1, 1}, 2, sizeof(destination)}),
        Status::invalid_argument);
    EXPECT_EQ(destination, original);

    auto bad_type = file;
    bad_type[14] = 99;
    ImageMetadata metadata;
    EXPECT_EQ(parse_metadata({bad_type.data(), bad_type.size()}, {}, metadata),
              Status::unsupported_type);
}

} // namespace
