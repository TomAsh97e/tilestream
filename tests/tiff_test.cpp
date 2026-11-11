#include <tilestream/cpu.h>
#include <tilestream/tiff.h>

#include <gtest/gtest.h>
#include <tiffio.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using namespace tilestream;
using Bytes = std::vector<std::uint8_t>;

class TempDirectory {
  public:
    TempDirectory() {
        static std::uint32_t serial = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("tilestream-tiff-" + std::to_string(getpid()) + "-" + std::to_string(serial++));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::filesystem::remove_all(path_);
    }

    std::string file(const std::string &name) const {
        return (path_ / name).string();
    }

  private:
    std::filesystem::path path_;
};

Bytes uint16_bytes(std::uint32_t width, std::uint32_t height) {
    std::vector<std::uint16_t> values(static_cast<std::size_t>(width) * height);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            values[static_cast<std::size_t>(y) * width + x] =
                static_cast<std::uint16_t>(y * 257 + x * 3);
        }
    }
    Bytes result(values.size() * sizeof(std::uint16_t));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

Bytes float32_bytes(std::uint32_t width, std::uint32_t height) {
    constexpr std::array<std::uint32_t, 16> special = {
        0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0x7fc01234, 0x7fa12345,
        0xffc0abcd, 0x3f800000, 0xbf800000, 0x00800000,
    };
    std::vector<std::uint32_t> words(static_cast<std::size_t>(width) * height);
    for (std::size_t i = 0; i < words.size(); ++i) {
        words[i] = i < special.size() ? special[i]
                                      : std::uint32_t{0x3f000000} + static_cast<std::uint32_t>(i);
    }
    Bytes result(words.size() * sizeof(std::uint32_t));
    std::memcpy(result.data(), words.data(), result.size());
    return result;
}

ConstImageView view(const Bytes &pixels, PixelType type, std::uint32_t width,
                    std::uint32_t height) {
    const std::uint64_t sample = type == PixelType::uint16 ? 2 : 4;
    return {pixels.data(), {type, width, height}, width * sample, pixels.size()};
}

void expect_image(const io::TiffImage &image, PixelType type, std::uint32_t width,
                  std::uint32_t height, const Bytes &pixels) {
    EXPECT_EQ(image.image.type, type);
    EXPECT_EQ(image.image.width, width);
    EXPECT_EQ(image.image.height, height);
    EXPECT_EQ(image.pitch_bytes, std::uint64_t{width} * (type == PixelType::uint16 ? 2 : 4));
    EXPECT_EQ(image.pixels, pixels);
}

bool write_stripped(const std::string &path, ConstImageView image, std::uint32_t rows_per_strip,
                    std::uint16_t compression) {
    TIFF *tiff = TIFFOpen(path.c_str(), "w");
    if (tiff == nullptr) {
        return false;
    }
    const auto bits = image.image.type == PixelType::uint16 ? 16 : 32;
    const auto sample_format =
        image.image.type == PixelType::uint16 ? SAMPLEFORMAT_UINT : SAMPLEFORMAT_IEEEFP;
    bool ok = TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, image.image.width) == 1 &&
              TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, image.image.height) == 1 &&
              TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits) == 1 &&
              TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1) == 1 &&
              TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, sample_format) == 1 &&
              TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) == 1 &&
              TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) == 1 &&
              TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT) == 1 &&
              TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, rows_per_strip) == 1 &&
              TIFFSetField(tiff, TIFFTAG_COMPRESSION, compression) == 1;
    const auto *source = static_cast<const std::uint8_t *>(image.data);
    for (std::uint32_t y = 0; ok && y < image.image.height; y += rows_per_strip) {
        const auto rows = std::min(rows_per_strip, image.image.height - y);
        const auto bytes = static_cast<tmsize_t>(rows * image.pitch_bytes);
        const auto written = TIFFWriteEncodedStrip(
            tiff, TIFFComputeStrip(tiff, y, 0),
            const_cast<std::uint8_t *>(source + y * image.pitch_bytes), bytes);
        ok = written == bytes;
    }
    ok = TIFFFlush(tiff) == 1 && ok;
    TIFFClose(tiff);
    return ok;
}

bool write_short_payload(const std::string &path, bool tiled) {
    TIFF *tiff = TIFFOpen(path.c_str(), "w");
    if (tiff == nullptr) {
        return false;
    }
    bool ok = TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, 16) == 1 &&
              TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, 16) == 1 &&
              TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16) == 1 &&
              TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1) == 1 &&
              TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT) == 1 &&
              TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) == 1 &&
              TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) == 1 &&
              TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT) == 1 &&
              TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE) == 1;
    std::array<std::uint8_t, 256> partial{};
    if (tiled) {
        ok = TIFFSetField(tiff, TIFFTAG_TILEWIDTH, 16) == 1 &&
             TIFFSetField(tiff, TIFFTAG_TILELENGTH, 16) == 1 &&
             TIFFWriteRawTile(tiff, 0, partial.data(), partial.size()) ==
                 static_cast<tmsize_t>(partial.size()) &&
             ok;
    } else {
        ok = TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 16) == 1 &&
             TIFFWriteRawStrip(tiff, 0, partial.data(), partial.size()) ==
                 static_cast<tmsize_t>(partial.size()) &&
             ok;
    }
    TIFFClose(tiff);
    return ok;
}

bool write_unsupported(const std::string &path, std::uint16_t bits, std::uint16_t sample_format,
                       std::uint16_t samples, std::uint16_t planar, std::uint16_t orientation,
                       std::uint16_t photometric, bool second_directory = false) {
    TIFF *tiff = TIFFOpen(path.c_str(), "w");
    if (tiff == nullptr) {
        return false;
    }
    const auto directory = [&] {
        const auto bytes = static_cast<tmsize_t>(2 * 2 * samples * ((bits + 7) / 8));
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(bytes));
        return TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, 2) == 1 &&
               TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, 2) == 1 &&
               TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits) == 1 &&
               TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, samples) == 1 &&
               TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, sample_format) == 1 &&
               TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, planar) == 1 &&
               TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, photometric) == 1 &&
               TIFFSetField(tiff, TIFFTAG_ORIENTATION, orientation) == 1 &&
               TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 2) == 1 &&
               TIFFWriteEncodedStrip(tiff, 0, pixels.data(), bytes) == bytes;
    };
    bool ok = directory();
    if (second_directory) {
        ok = TIFFWriteDirectory(tiff) == 1 && directory() && ok;
    }
    TIFFClose(tiff);
    return ok;
}

TEST(TiffAdapter, TiledClassicAndBigTiffRoundTrip) {
    TempDirectory temporary;
    constexpr std::uint32_t width = 33;
    constexpr std::uint32_t height = 17;
    const auto integers = uint16_bytes(width, height);
    const auto floats = float32_bytes(width, height);

    struct Case {
        const char *name;
        PixelType type;
        const Bytes *pixels;
        io::TiffWriteOptions options;
    };
    const std::vector<Case> cases = {
        {"u16-none.tif", PixelType::uint16, &integers, {}},
        {"u16-deflate.tif",
         PixelType::uint16,
         &integers,
         {false, io::TiffCompression::deflate, false, 128, 32, 6}},
        {"u16-predictor.tif",
         PixelType::uint16,
         &integers,
         {false, io::TiffCompression::deflate, true, 64, 64, 6}},
        {"u16-big.tif",
         PixelType::uint16,
         &integers,
         {true, io::TiffCompression::none, false, 256, 16, 6}},
        {"f32-none.tif", PixelType::float32, &floats, {}},
        {"f32-deflate-big.tif",
         PixelType::float32,
         &floats,
         {true, io::TiffCompression::deflate, false, 64, 64, 6}},
    };
    for (const auto &test : cases) {
        SCOPED_TRACE(test.name);
        const auto path = temporary.file(test.name);
        ASSERT_EQ(io::write_tiff(path.c_str(), view(*test.pixels, test.type, width, height),
                                 test.options),
                  Status::success);
        io::TiffImage decoded;
        ASSERT_EQ(io::read_tiff(path.c_str(), decoded), Status::success);
        expect_image(decoded, test.type, width, height, *test.pixels);

        TIFF *tiff = TIFFOpen(path.c_str(), "r");
        ASSERT_NE(tiff, nullptr);
        EXPECT_NE(TIFFIsTiled(tiff), 0);
        std::uint16_t compression = 0;
        ASSERT_EQ(TIFFGetFieldDefaulted(tiff, TIFFTAG_COMPRESSION, &compression), 1);
        EXPECT_EQ(compression, test.options.compression == io::TiffCompression::none
                                   ? COMPRESSION_NONE
                                   : COMPRESSION_ADOBE_DEFLATE);
        if (test.options.compression == io::TiffCompression::deflate) {
            std::uint16_t predictor = 0;
            ASSERT_EQ(TIFFGetFieldDefaulted(tiff, TIFFTAG_PREDICTOR, &predictor), 1);
            EXPECT_EQ(predictor,
                      test.options.horizontal_predictor ? PREDICTOR_HORIZONTAL : PREDICTOR_NONE);
        }
        TIFFClose(tiff);

        std::ifstream raw(path, std::ios::binary);
        std::array<std::uint8_t, 4> signature{};
        raw.read(reinterpret_cast<char *>(signature.data()), signature.size());
        ASSERT_TRUE(raw.good());
        EXPECT_EQ(signature[2], test.options.bigtiff ? 43 : 42);
    }
}

TEST(TiffAdapter, ImportsStripsForBothTypes) {
    TempDirectory temporary;
    constexpr std::uint32_t width = 35;
    constexpr std::uint32_t height = 19;
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        const auto pixels =
            type == PixelType::uint16 ? uint16_bytes(width, height) : float32_bytes(width, height);
        const auto path =
            temporary.file(type == PixelType::uint16 ? "u16-strips.tif" : "f32-strips.tif");
        const auto rows_per_strip = type == PixelType::uint16 ? 7U : UINT32_MAX;
        ASSERT_TRUE(write_stripped(path, view(pixels, type, width, height), rows_per_strip,
                                   COMPRESSION_ADOBE_DEFLATE));
        io::TiffImage decoded;
        ASSERT_EQ(io::read_tiff(path.c_str(), decoded), Status::success);
        expect_image(decoded, type, width, height, pixels);
    }
}

TEST(TiffAdapter, TscTiffTscPreservesPixels) {
    TempDirectory temporary;
    constexpr std::uint32_t width = 129;
    constexpr std::uint32_t height = 33;
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        const auto pixels =
            type == PixelType::uint16 ? uint16_bytes(width, height) : float32_bytes(width, height);
        const auto source_view = view(pixels, type, width, height);
        std::uint64_t bound = 0;
        ASSERT_EQ(compress_bound(source_view.image, {}, bound), Status::success);
        Bytes first(static_cast<std::size_t>(bound));
        std::uint64_t first_size = 0;
        ASSERT_EQ(cpu::compress(source_view, {}, {first.data(), first.size()}, first_size),
                  Status::success);
        Bytes decoded(pixels.size());
        ASSERT_EQ(
            cpu::decompress({first.data(), first_size}, {decoded.data(), source_view.image,
                                                         source_view.pitch_bytes, decoded.size()}),
            Status::success);
        EXPECT_EQ(decoded, pixels);

        const auto path = temporary.file(type == PixelType::uint16 ? "u16-tsc.tif" : "f32-tsc.tif");
        const io::TiffWriteOptions options{
            false, io::TiffCompression::deflate, type == PixelType::uint16, 128, 32, 6};
        ASSERT_EQ(io::write_tiff(path.c_str(), view(decoded, type, width, height), options),
                  Status::success);
        io::TiffImage imported;
        ASSERT_EQ(io::read_tiff(path.c_str(), imported), Status::success);
        expect_image(imported, type, width, height, pixels);

        Bytes second(static_cast<std::size_t>(bound));
        std::uint64_t second_size = 0;
        ASSERT_EQ(cpu::compress({imported.pixels.data(), imported.image, imported.pitch_bytes,
                                 imported.pixels.size()},
                                {}, {second.data(), second.size()}, second_size),
                  Status::success);
        EXPECT_EQ(second_size, first_size);
        EXPECT_TRUE(std::equal(first.begin(),
                               first.begin() + static_cast<std::ptrdiff_t>(first_size),
                               second.begin()));
    }
}

TEST(TiffAdapter, RejectsUnsupportedTagsWithoutConversion) {
    TempDirectory temporary;
    struct Case {
        const char *name;
        std::uint16_t bits;
        std::uint16_t format;
        std::uint16_t samples;
        std::uint16_t planar;
        std::uint16_t orientation;
        std::uint16_t photometric;
        bool multi = false;
    };
    const std::vector<Case> cases = {
        {"uint8.tif", 8, SAMPLEFORMAT_UINT, 1, PLANARCONFIG_CONTIG, ORIENTATION_TOPLEFT,
         PHOTOMETRIC_MINISBLACK},
        {"int16.tif", 16, SAMPLEFORMAT_INT, 1, PLANARCONFIG_CONTIG, ORIENTATION_TOPLEFT,
         PHOTOMETRIC_MINISBLACK},
        {"float64.tif", 64, SAMPLEFORMAT_IEEEFP, 1, PLANARCONFIG_CONTIG, ORIENTATION_TOPLEFT,
         PHOTOMETRIC_MINISBLACK},
        {"rgb.tif", 8, SAMPLEFORMAT_UINT, 3, PLANARCONFIG_CONTIG, ORIENTATION_TOPLEFT,
         PHOTOMETRIC_RGB},
        {"separate.tif", 16, SAMPLEFORMAT_UINT, 1, PLANARCONFIG_SEPARATE, ORIENTATION_TOPLEFT,
         PHOTOMETRIC_MINISBLACK},
        {"bottom.tif", 16, SAMPLEFORMAT_UINT, 1, PLANARCONFIG_CONTIG, ORIENTATION_BOTLEFT,
         PHOTOMETRIC_MINISBLACK},
        {"white.tif", 16, SAMPLEFORMAT_UINT, 1, PLANARCONFIG_CONTIG, ORIENTATION_TOPLEFT,
         PHOTOMETRIC_MINISWHITE},
        {"multi.tif", 16, SAMPLEFORMAT_UINT, 1, PLANARCONFIG_CONTIG, ORIENTATION_TOPLEFT,
         PHOTOMETRIC_MINISBLACK, true},
    };
    io::TiffImage unchanged;
    unchanged.image = {PixelType::uint16, 7, 9};
    unchanged.pixels = {1, 2, 3};
    for (const auto &test : cases) {
        SCOPED_TRACE(test.name);
        const auto path = temporary.file(test.name);
        ASSERT_TRUE(write_unsupported(path, test.bits, test.format, test.samples, test.planar,
                                      test.orientation, test.photometric, test.multi));
        auto output = unchanged;
        EXPECT_EQ(io::read_tiff(path.c_str(), output), Status::unsupported_format);
        EXPECT_EQ(output.image.width, unchanged.image.width);
        EXPECT_EQ(output.pixels, unchanged.pixels);
    }
}

TEST(TiffAdapter, LimitsAndInvalidArguments) {
    TempDirectory temporary;
    const auto pixels = uint16_bytes(33, 17);
    const auto path = temporary.file("valid.tif");
    ASSERT_EQ(io::write_tiff(path.c_str(), view(pixels, PixelType::uint16, 33, 17)),
              Status::success);
    io::TiffImage unchanged;
    unchanged.image.width = 7;
    unchanged.pixels = {1, 2, 3};
    for (const auto limits : {io::TiffReadLimits{32, UINT32_MAX, max_bytes},
                              io::TiffReadLimits{UINT32_MAX, 16, max_bytes},
                              io::TiffReadLimits{UINT32_MAX, UINT32_MAX, pixels.size() - 1}}) {
        auto output = unchanged;
        EXPECT_EQ(io::read_tiff(path.c_str(), output, limits), Status::resource_limit);
        EXPECT_EQ(output.image.width, 7U);
        EXPECT_EQ(output.pixels, unchanged.pixels);
    }
    EXPECT_EQ(io::read_tiff(nullptr, unchanged), Status::invalid_argument);
    EXPECT_EQ(io::read_tiff(temporary.file("missing.tif").c_str(), unchanged), Status::io_error);

    auto invalid = view(pixels, PixelType::uint16, 33, 17);
    invalid.pitch_bytes = 65;
    EXPECT_EQ(io::write_tiff(temporary.file("bad-pitch.tif").c_str(), invalid),
              Status::invalid_argument);
    auto options = io::TiffWriteOptions{};
    options.tile_width = 15;
    EXPECT_EQ(io::write_tiff(temporary.file("bad-tile.tif").c_str(),
                             view(pixels, PixelType::uint16, 33, 17), options),
              Status::invalid_argument);
    options = {};
    options.horizontal_predictor = true;
    EXPECT_EQ(io::write_tiff(temporary.file("bad-predictor.tif").c_str(),
                             view(pixels, PixelType::uint16, 33, 17), options),
              Status::invalid_argument);
    options = {false, io::TiffCompression::deflate, false, 128, 32, 0};
    EXPECT_EQ(io::write_tiff(temporary.file("bad-level.tif").c_str(),
                             view(pixels, PixelType::uint16, 33, 17), options),
              Status::invalid_argument);

    const auto float_pixels = float32_bytes(33, 17);
    options = {false, io::TiffCompression::deflate, true, 128, 32, 6};
    EXPECT_EQ(io::write_tiff(temporary.file("float-predictor.tif").c_str(),
                             view(float_pixels, PixelType::float32, 33, 17), options),
              Status::invalid_argument);

    constexpr std::uint32_t large_side = 1U << 20;
    constexpr std::uint64_t large_pitch = std::uint64_t{large_side} * 2;
    const ConstImageView too_many_tiles{reinterpret_cast<const void *>(4096),
                                        {PixelType::uint16, large_side, large_side},
                                        large_pitch,
                                        large_pitch * large_side};
    options = {true, io::TiffCompression::none, false, 16, 16, 6};
    EXPECT_EQ(io::write_tiff(temporary.file("too-many-tiles.tif").c_str(), too_many_tiles, options),
              Status::size_overflow);
}

TEST(TiffAdapter, ShortTileAndStripAreIoErrorsAndDoNotChangeOutput) {
    TempDirectory temporary;
    for (const bool tiled : {false, true}) {
        const auto path = temporary.file(tiled ? "short-tile.tif" : "short-strip.tif");
        ASSERT_TRUE(write_short_payload(path, tiled));
        io::TiffImage output;
        output.image.width = 7;
        output.pixels = {1, 2, 3};
        EXPECT_EQ(io::read_tiff(path.c_str(), output), Status::io_error);
        EXPECT_EQ(output.image.width, 7U);
        EXPECT_EQ(output.pixels, (Bytes{1, 2, 3}));
    }
}

} // namespace
