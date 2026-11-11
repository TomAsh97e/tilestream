#include <tilestream/cpu.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace tilestream;
using Bytes = std::vector<std::uint8_t>;
using Pixels = std::vector<std::uint16_t>;

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

ConstImageView input_view(const Pixels &pixels, std::uint32_t width, std::uint32_t height) {
    return {pixels.data(),
            {PixelType::uint16, width, height},
            std::uint64_t{width} * 2,
            pixels.size() * 2};
}

ImageView output_view(Pixels &pixels, std::uint32_t width, std::uint32_t height) {
    return {pixels.data(),
            {PixelType::uint16, width, height},
            std::uint64_t{width} * 2,
            pixels.size() * 2};
}

Bytes encode(const Pixels &pixels, std::uint32_t width, std::uint32_t height, TileShape tile = {}) {
    std::uint64_t bound = 0;
    EXPECT_EQ(compress_bound({PixelType::uint16, width, height}, tile, bound), Status::success);
    Bytes bytes(static_cast<std::size_t>(bound + 16), 0xa5);
    std::uint64_t written = 0;
    EXPECT_EQ(
        cpu::compress(input_view(pixels, width, height), tile, {bytes.data(), bound}, written),
        Status::success);
    EXPECT_LE(written, bound);
    if (written > bound) {
        return {};
    }
    EXPECT_TRUE(std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(written), bytes.end(),
                            [](auto byte) { return byte == 0xa5; }));
    bytes.resize(static_cast<std::size_t>(written));
    return bytes;
}

void decode_matches(const Bytes &file, const Pixels &expected, std::uint32_t width,
                    std::uint32_t height) {
    // The logical file ends at the allocation boundary, including for unaligned input.
    auto storage = std::make_unique<std::uint8_t[]>(file.size() + 1);
    std::copy(file.begin(), file.end(), storage.get() + 1);
    Pixels decoded(expected.size(), 0xdead);
    ASSERT_EQ(
        cpu::decompress({storage.get() + 1, file.size()}, output_view(decoded, width, height)),
        Status::success);
    EXPECT_EQ(decoded, expected);
}

void put(Bytes &file, std::size_t offset, std::uint64_t value, std::size_t size) {
    ASSERT_LE(offset + size, file.size());
    for (std::size_t i = 0; i < size; ++i) {
        file[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
    }
}

TEST(CpuCodec, HandCalculatedGoldenFiles) {
    struct Case {
        const char *name;
        std::uint32_t width;
        std::uint32_t height;
        Pixels pixels;
        TileMode mode;
    };
    Pixels zero_width(16, 0x0102);
    std::fill(zero_width.begin() + 8, zero_width.end(), 0x0304);
    // Delta -32768 maps to 65535, including the reverse transition modulo 65536.
    Pixels half_range = {0, 32768, 0, 32768, 0, 32768, 0, 32768, 0};
    half_range.insert(half_range.end(), 9, 5);
    const std::vector<Case> cases = {
        {"uint16_raw", 1, 1, {0x1234}, TileMode::raw},
        {"uint16_constant", 3, 2, Pixels(6, 0xbeef), TileMode::constant},
        {"uint16_wrap",
         8,
         2,
         {65535, 0, 1, 2, 3, 4, 5, 6, 0, 65535, 65534, 65533, 65532, 65531, 65530, 65529},
         TileMode::predict_bitpack},
        {"uint16_zero_width", 8, 2, zero_width, TileMode::predict_bitpack},
        {"uint16_raw_tie", 6, 1, {0, 1, 2, 3, 4, 5}, TileMode::raw},
        {"uint16_half_range", 9, 2, half_range, TileMode::predict_bitpack},
    };
    for (const auto &test : cases) {
        SCOPED_TRACE(test.name);
        const auto file = golden(test.name);
        ASSERT_FALSE(file.empty());
        EXPECT_EQ(encode(test.pixels, test.width, test.height), file);
        decode_matches(file, test.pixels, test.width, test.height);
        ImageMetadata metadata;
        ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        ASSERT_EQ(metadata.tiles.size(), 1U);
        EXPECT_EQ(metadata.tiles[0].mode, test.mode);
    }
}

TEST(CpuCodec, AllWidthsAndFinalPackedWords) {
    std::mt19937 rng(2026517);
    for (const auto width : {31U, 32U, 33U, 127U, 128U}) {
        for (std::uint32_t bits = 0; bits <= 16; ++bits) {
            SCOPED_TRACE(::testing::Message() << "width=" << width << " bits=" << bits);
            Pixels pixels(width * 2, 400);
            pixels[width] = 65535;
            const auto mask = (1U << bits) - 1;
            std::vector<std::uint32_t> residuals;
            for (std::uint32_t x = 1; x < width; ++x) {
                const auto z = x == 1 ? mask : static_cast<std::uint32_t>(rng()) & mask;
                const auto delta = z % 2 == 0 ? static_cast<std::int32_t>(z / 2)
                                              : -static_cast<std::int32_t>(z / 2) - 1;
                const auto value = static_cast<std::int32_t>(pixels[width + x - 1]) + delta;
                pixels[width + x] = static_cast<std::uint16_t>(value);
                residuals.push_back(z);
            }
            const auto file = encode(pixels, width, 2);
            ImageMetadata metadata;
            ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
            ASSERT_EQ(metadata.tiles.size(), 1U);
            ASSERT_EQ(metadata.tiles[0].mode, TileMode::predict_bitpack);
            ASSERT_EQ(metadata.segments.size(), 2U);
            EXPECT_EQ(metadata.segments[0].bit_width, 0U);
            const auto &row = metadata.segments[1];
            EXPECT_EQ(row.bit_width, bits);
            EXPECT_EQ(row.seed, 65535U);

            // Independent 64-bit reservoir, unlike the encoder's per-bit writer.
            Bytes expected;
            std::uint64_t reservoir = 0;
            std::uint32_t occupied = 0;
            for (const auto z : residuals) {
                reservoir |= std::uint64_t{z} << occupied;
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
            decode_matches(file, pixels, width, 2);
        }
    }
}

TEST(CpuCodec, EveryUint16ValueAndSingleSampleRows) {
    Pixels pixels(65536);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<std::uint16_t>(i);
    }
    for (const auto tile : {TileShape{128, 32}, TileShape{64, 64}, TileShape{256, 16}}) {
        decode_matches(encode(pixels, 256, 256, tile), pixels, 256, 256);
        const Pixels column = {65535, 0, 32768, 1, 65534, 42, 0};
        decode_matches(encode(column, 1, 7, tile), column, 1, 7);
        decode_matches(encode(Pixels(65, 17), 1, 65, tile), Pixels(65, 17), 1, 65);
    }
}

TEST(CpuCodec, DeterministicRandomImagesAndTileEdges) {
    constexpr std::array<std::uint32_t, 10> sizes = {1, 2, 15, 16, 31, 32, 33, 127, 128, 129};
    constexpr std::array<TileShape, 3> shapes = {{{128, 32}, {64, 64}, {256, 16}}};
    for (std::uint32_t trial = 0; trial < 1000; ++trial) {
        const auto seed = 2026517U + trial;
        SCOPED_TRACE(::testing::Message() << "seed=" << seed);
        std::mt19937 rng(seed);
        const auto width = sizes[rng() % sizes.size()];
        const auto height = sizes[rng() % sizes.size()];
        Pixels pixels(static_cast<std::size_t>(width) * height);
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            switch (trial % 4) {
            case 0:
                pixels[i] = 1234;
                break;
            case 1:
                pixels[i] = static_cast<std::uint16_t>(65535 - i);
                break;
            case 2:
                pixels[i] = static_cast<std::uint16_t>(i + rng() % 3);
                break;
            default:
                pixels[i] = static_cast<std::uint16_t>(rng());
                break;
            }
        }
        const auto tile = shapes[trial % shapes.size()];
        const auto first = encode(pixels, width, height, tile);
        ASSERT_FALSE(first.empty());
        EXPECT_EQ(first, encode(pixels, width, height, tile));
        decode_matches(first, pixels, width, height);
    }
}

TEST(CpuCodec, PitchPaddingAndPartialTiles) {
    constexpr std::uint32_t width = 129;
    constexpr std::uint32_t height = 33;
    constexpr std::uint32_t stride = 136;
    Pixels packed(width * height);
    Pixels padded(stride * height, 0xdead);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto value = static_cast<std::uint16_t>(y * 37 + x);
            packed[y * width + x] = padded[y * stride + x] = value;
        }
    }
    const auto original = padded;
    const auto expected = encode(packed, width, height);
    std::uint64_t bound = 0;
    ASSERT_EQ(compress_bound({PixelType::uint16, width, height}, {}, bound), Status::success);
    Bytes file(static_cast<std::size_t>(bound));
    std::uint64_t size = 0;
    ASSERT_EQ(
        cpu::compress(
            {padded.data(), {PixelType::uint16, width, height}, stride * 2, padded.size() * 2}, {},
            {file.data(), file.size()}, size),
        Status::success);
    file.resize(static_cast<std::size_t>(size));
    EXPECT_EQ(file, expected);
    EXPECT_EQ(padded, original);
    Pixels decoded(stride * height, 0xdead);
    ASSERT_EQ(
        cpu::decompress(
            {file.data(), file.size()},
            {decoded.data(), {PixelType::uint16, width, height}, stride * 2, decoded.size() * 2}),
        Status::success);
    EXPECT_EQ(decoded, original);
    ImageMetadata metadata;
    ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
    ASSERT_EQ(metadata.tiles.size(), 4U);
    EXPECT_EQ(metadata.tiles.back().width, 1);
    EXPECT_EQ(metadata.tiles.back().height, 1);
    EXPECT_EQ(metadata.tiles.back().mode, TileMode::raw);
}

TEST(CpuCodec, MixedModesInOneImage) {
    Pixels pixels(384 * 32);
    std::mt19937 rng(2026517);
    for (std::uint32_t row = 0; row < 32; ++row) {
        for (std::uint32_t col = 0; col < 384; ++col) {
            if (col < 128) {
                pixels[row * 384 + col] = 7;
            } else if (col < 256) {
                pixels[row * 384 + col] = static_cast<std::uint16_t>(col);
            } else {
                pixels[row * 384 + col] = static_cast<std::uint16_t>(rng());
            }
        }
    }
    const auto file = encode(pixels, 384, 32);
    ImageMetadata metadata;
    ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
    ASSERT_EQ(metadata.tiles.size(), 3U);
    EXPECT_EQ(metadata.tiles[0].mode, TileMode::constant);
    EXPECT_EQ(metadata.tiles[1].mode, TileMode::predict_bitpack);
    EXPECT_EQ(metadata.tiles[2].mode, TileMode::raw);
    EXPECT_EQ(metadata.segments.size(), 32U);
    decode_matches(file, pixels, 384, 32);

    auto overlap = file;
    put(overlap, 104, metadata.tiles[0].offset, 8);
    EXPECT_EQ(parse_metadata({overlap.data(), overlap.size()}, {}, metadata),
              Status::malformed_data);
}

TEST(CpuCodec, InvalidViewsDoNotModifyOutputs) {
    Pixels pixels = {1, 2, 3};
    auto view = input_view(pixels, 3, 1);
    Bytes output(128, 0xa5);
    const auto original = output;
    std::uint64_t written = 999;
    const auto attempt = [&](ConstImageView input, TileShape tile, MutableBytes target,
                             Status status) {
        EXPECT_EQ(cpu::compress(input, tile, target, written), status);
        EXPECT_EQ(output, original);
        EXPECT_EQ(written, 999U);
    };
    auto invalid = view;
    invalid.image.type = static_cast<PixelType>(99);
    attempt(invalid, {}, {output.data(), output.size()}, Status::unsupported_type);
    invalid = view;
    invalid.data = nullptr;
    attempt(invalid, {}, {output.data(), output.size()}, Status::invalid_argument);
    invalid = view;
    invalid.pitch_bytes = 5;
    attempt(invalid, {}, {output.data(), output.size()}, Status::invalid_argument);
    invalid = view;
    invalid.data = reinterpret_cast<const std::uint8_t *>(pixels.data()) + 1;
    attempt(invalid, {}, {output.data(), output.size()}, Status::invalid_argument);
    invalid = view;
    invalid.capacity_bytes = 5;
    attempt(invalid, {}, {output.data(), output.size()}, Status::buffer_too_small);
    invalid = view;
    invalid.image.width = 0;
    attempt(invalid, {}, {output.data(), output.size()}, Status::invalid_argument);
    attempt(view, {0, 0}, {output.data(), output.size()}, Status::invalid_argument);
    attempt(view, {}, {output.data(), 1}, Status::buffer_too_small);
    attempt(view, {}, {nullptr, output.size()}, Status::invalid_argument);
    attempt(view, {}, {output.data(), UINT64_MAX}, Status::size_overflow);

    alignas(8) std::array<std::uint8_t, 256> shared{};
    attempt({shared.data(), {PixelType::uint16, 1, 1}, 2, 256}, {}, {shared.data() + 128, 128},
            Status::invalid_argument);
    std::array<std::uint64_t, 16> result_storage{};
    result_storage[0] = 999;
    const auto result_before = result_storage;
    EXPECT_EQ(
        cpu::compress(view, {}, {result_storage.data(), sizeof(result_storage)}, result_storage[0]),
        Status::invalid_argument);
    EXPECT_EQ(result_storage, result_before);

    const auto file = golden("uint16_raw");
    Pixels decoded(8, 0xdead);
    const auto before = decoded;
    auto destination = output_view(decoded, 1, 1);
    destination.image.type = static_cast<PixelType>(99);
    EXPECT_EQ(cpu::decompress({file.data(), file.size()}, destination), Status::unsupported_type);
    destination = output_view(decoded, 2, 1);
    EXPECT_EQ(cpu::decompress({file.data(), file.size()}, destination), Status::invalid_argument);
    destination = output_view(decoded, 1, 1);
    destination.capacity_bytes = 1;
    EXPECT_EQ(cpu::decompress({file.data(), file.size()}, destination), Status::buffer_too_small);
    EXPECT_EQ(decoded, before);

    auto alias_file = golden("uint16_raw");
    const auto alias_before = alias_file;
    EXPECT_EQ(cpu::decompress({alias_file.data(), alias_file.size()},
                              {alias_file.data() + 96, {PixelType::uint16, 1, 1}, 2, 4}),
              Status::invalid_argument);
    EXPECT_EQ(alias_file, alias_before);
    EXPECT_EQ(cpu::decompress({file.data(), file.size()}, output_view(decoded, 1, 1),
                              {max_bytes, 0, max_bytes}),
              Status::resource_limit);
    EXPECT_EQ(decoded, before);
}

TEST(CpuParser, AllTruncationsAreRejectedBeforeDecode) {
    for (const auto name : {"uint16_raw", "uint16_constant", "uint16_wrap", "uint16_half_range"}) {
        const auto full = golden(name);
        ImageMetadata valid;
        ASSERT_EQ(parse_metadata({full.data(), full.size()}, {}, valid), Status::success);
        for (std::size_t size = 0; size < full.size(); ++size) {
            SCOPED_TRACE(::testing::Message() << name << " truncated=" << size);
            auto truncated = std::make_unique<std::uint8_t[]>(size);
            std::copy_n(full.data(), size, truncated.get());
            ImageMetadata previous = valid;
            ASSERT_NE(parse_metadata({truncated.get(), size}, {}, previous), Status::success);
            EXPECT_EQ(previous.file_bytes, valid.file_bytes);
            EXPECT_EQ(previous.tiles.size(), valid.tiles.size());
            Pixels pixels(static_cast<std::size_t>(valid.image.width) * valid.image.height, 0xabcd);
            const auto before = pixels;
            EXPECT_NE(cpu::decompress({truncated.get(), size},
                                      output_view(pixels, valid.image.width, valid.image.height)),
                      Status::success);
            EXPECT_EQ(pixels, before);
        }
    }
}

TEST(CpuParser, MalformedFieldsAndPadding) {
    struct Mutation {
        std::size_t offset;
        std::size_t size;
        std::uint64_t value;
        Status expected = Status::malformed_data;
    };
    const auto file = golden("uint16_wrap");
    const std::vector<Mutation> mutations = {
        {0, 1, 0},
        {8, 2, 2, Status::unsupported_format},
        {10, 2, 1, Status::unsupported_format},
        {12, 2, 63},
        {14, 1, 99, Status::unsupported_type},
        {15, 1, 1, Status::unsupported_format},
        {16, 4, 0},
        {20, 4, 0},
        {24, 2, 0, Status::unsupported_format},
        {28, 2, 31},
        {30, 1, 1},
        {32, 8, UINT64_MAX},
        {40, 8, 0},
        {48, 8, UINT64_MAX},
        {56, 8, 119},
        {64, 1, 3},
        {65, 1, 1},
        {68, 2, 7},
        {70, 2, 3},
        {72, 8, 95},
        {72, 8, UINT64_MAX},
        {80, 8, 0},
        {80, 8, 23},
        {80, 8, UINT64_MAX},
        {88, 1, 1},
        {98, 1, 1},
        {100, 1, 17},
        {101, 1, 1},
        {107, 1, 0x80},
        {119, 1, 1},
    };
    for (const auto &mutation : mutations) {
        SCOPED_TRACE(::testing::Message()
                     << "offset=" << mutation.offset << " value=" << mutation.value);
        auto changed = file;
        put(changed, mutation.offset, mutation.value, mutation.size);
        ImageMetadata metadata;
        metadata.file_bytes = 777;
        EXPECT_EQ(parse_metadata({changed.data(), changed.size()}, {}, metadata),
                  mutation.expected);
        EXPECT_EQ(metadata.file_bytes, 777U);
        Pixels output(16, 0xdead);
        const auto before = output;
        EXPECT_EQ(cpu::decompress({changed.data(), changed.size()}, output_view(output, 8, 2)),
                  mutation.expected);
        EXPECT_EQ(output, before);
    }
    for (const auto name : {"uint16_raw", "uint16_constant"}) {
        auto changed = golden(name);
        changed[98] = 1;
        ImageMetadata metadata;
        EXPECT_EQ(parse_metadata({changed.data(), changed.size()}, {}, metadata),
                  Status::malformed_data);
    }
    auto overflow = file;
    put(overflow, 16, UINT32_MAX, 4);
    put(overflow, 20, UINT32_MAX, 4);
    ImageMetadata metadata;
    EXPECT_EQ(parse_metadata({overflow.data(), overflow.size()}, {}, metadata),
              Status::size_overflow);
    auto trailing = file;
    trailing.push_back(0);
    put(trailing, 56, trailing.size(), 8);
    EXPECT_EQ(parse_metadata({trailing.data(), trailing.size()}, {}, metadata),
              Status::malformed_data);
}

TEST(CpuParser, SegmentLengthsMustConsumeExactlyThePayload) {
    auto shortened = golden("uint16_wrap");
    shortened.resize(shortened.size() - 4);
    put(shortened, 56, shortened.size(), 8);
    put(shortened, 80, 20, 8);
    ImageMetadata metadata;
    EXPECT_EQ(parse_metadata({shortened.data(), shortened.size()}, {}, metadata),
              Status::malformed_data);

    auto extended = golden("uint16_wrap");
    extended.insert(extended.end(), 4, 0);
    put(extended, 56, extended.size(), 8);
    put(extended, 80, 28, 8);
    EXPECT_EQ(parse_metadata({extended.data(), extended.size()}, {}, metadata),
              Status::malformed_data);
}

TEST(CpuParser, LimitsAndEmptyInput) {
    const auto file = golden("uint16_wrap");
    ImageMetadata metadata;
    metadata.file_bytes = 777;
    EXPECT_EQ(parse_metadata({}, {}, metadata), Status::malformed_data);
    EXPECT_EQ(parse_metadata({nullptr, 1}, {}, metadata), Status::invalid_argument);
    EXPECT_EQ(parse_metadata({file.data(), UINT64_MAX}, {}, metadata), Status::size_overflow);
    EXPECT_EQ(parse_metadata({file.data(), file.size()}, {file.size() - 1, max_bytes, max_bytes},
                             metadata),
              Status::resource_limit);
    EXPECT_EQ(parse_metadata({file.data(), file.size()}, {max_bytes, 0, max_bytes}, metadata),
              Status::resource_limit);
    const auto exact = sizeof(ImageMetadata) + sizeof(TileMetadata) + 2 * sizeof(SegmentMetadata);
    EXPECT_EQ(
        parse_metadata({file.data(), file.size()}, {max_bytes, max_bytes, exact - 1}, metadata),
        Status::resource_limit);
    EXPECT_EQ(metadata.file_bytes, 777U);
    ASSERT_EQ(parse_metadata({file.data(), file.size()}, {file.size(), 1, exact}, metadata),
              Status::success);
    EXPECT_EQ(metadata.file_bytes, file.size());
    ASSERT_EQ(metadata.segments.size(), 2U);
    EXPECT_EQ(metadata.tiles[0].first_segment, 0U);
    EXPECT_EQ(metadata.segments[0].payload_offset, 8U);
    EXPECT_EQ(metadata.segments[1].payload_offset, 20U);
    EXPECT_EQ(metadata.segments[0].payload_bytes, 4U);
}

TEST(CpuParser, AcceptsNoncanonicalButStructurallyValidRows) {
    auto file = golden("uint16_wrap");
    // Seven ones can use width 2 instead of the canonical width 1, with the same length.
    file[112] = 2;
    put(file, 116, 0x1555, 4);
    const Pixels expected = {65535, 0,     1,     2,     3,     4,     5,     6,
                             0,     65535, 65534, 65533, 65532, 65531, 65530, 65529};
    decode_matches(file, expected, 8, 2);
}

} // namespace
