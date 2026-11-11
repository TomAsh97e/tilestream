#include <tilestream/cpu.h>
#include <tilestream/tiff.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace tilestream;
using Bytes = std::vector<std::uint8_t>;

Bytes make_uint16(std::uint32_t width, std::uint32_t height) {
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

Bytes make_float32(std::uint32_t width, std::uint32_t height) {
    static constexpr std::uint32_t special[] = {
        0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0x7fc01234, 0x7fa12345,
        0xffc0abcd, 0x3f800000, 0xbf800000, 0x00800000,
    };
    std::vector<std::uint32_t> words(static_cast<std::size_t>(width) * height);
    for (std::size_t i = 0; i < words.size(); ++i) {
        words[i] = i < std::size(special)
                       ? special[i]
                       : std::uint32_t{0x3f000000} + static_cast<std::uint32_t>(i);
    }
    Bytes result(words.size() * sizeof(std::uint32_t));
    std::memcpy(result.data(), words.data(), result.size());
    return result;
}

bool tsc_round_trip(const Bytes &source, PixelType type, std::uint32_t width, std::uint32_t height,
                    Bytes &decoded) {
    const ImageSpec image{type, width, height};
    const std::uint64_t pitch = std::uint64_t{width} * (type == PixelType::uint16 ? 2 : 4);
    std::uint64_t bound = 0;
    if (compress_bound(image, {}, bound) != Status::success) {
        return false;
    }
    Bytes tsc(static_cast<std::size_t>(bound));
    std::uint64_t written = 0;
    if (cpu::compress({source.data(), image, pitch, source.size()}, {}, {tsc.data(), tsc.size()},
                      written) != Status::success) {
        return false;
    }
    decoded.resize(source.size());
    return cpu::decompress({tsc.data(), written}, {decoded.data(), image, pitch, decoded.size()}) ==
               Status::success &&
           decoded == source;
}

bool write_files(const std::filesystem::path &directory) {
    constexpr std::uint32_t width = 35;
    constexpr std::uint32_t height = 19;
    std::filesystem::create_directories(directory);
    const auto uint16 = make_uint16(width, height);
    const auto float32 = make_float32(width, height);
    Bytes uint16_decoded;
    Bytes float32_decoded;
    if (!tsc_round_trip(uint16, PixelType::uint16, width, height, uint16_decoded) ||
        !tsc_round_trip(float32, PixelType::float32, width, height, float32_decoded)) {
        return false;
    }
    const ConstImageView uint16_view{uint16_decoded.data(),
                                     {PixelType::uint16, width, height},
                                     width * 2,
                                     uint16_decoded.size()};
    const ConstImageView float32_view{float32_decoded.data(),
                                      {PixelType::float32, width, height},
                                      width * 4,
                                      float32_decoded.size()};
    return io::write_tiff((directory / "cpp-u16-none.tif").c_str(), uint16_view) ==
               Status::success &&
           io::write_tiff((directory / "cpp-u16-deflate-predictor.tif").c_str(), uint16_view,
                          {false, io::TiffCompression::deflate, true, 64, 64, 6}) ==
               Status::success &&
           io::write_tiff((directory / "cpp-u16-big.tif").c_str(), uint16_view,
                          {true, io::TiffCompression::none, false, 256, 16, 6}) ==
               Status::success &&
           io::write_tiff((directory / "cpp-f32-deflate.tif").c_str(), float32_view,
                          {false, io::TiffCompression::deflate, false, 128, 32, 6}) ==
               Status::success;
}

bool verify_file(const std::filesystem::path &path, const Bytes &expected, PixelType type,
                 std::uint32_t width, std::uint32_t height) {
    io::TiffImage image;
    if (io::read_tiff(path.c_str(), image) != Status::success || image.image.type != type ||
        image.image.width != width || image.image.height != height || image.pixels != expected) {
        return false;
    }
    Bytes decoded;
    return tsc_round_trip(image.pixels, type, width, height, decoded);
}

bool verify_python_files(const std::filesystem::path &directory) {
    constexpr std::uint32_t width = 35;
    constexpr std::uint32_t height = 19;
    return verify_file(directory / "python-u16-strips.tif", make_uint16(width, height),
                       PixelType::uint16, width, height) &&
           verify_file(directory / "python-f32-tiles.tif", make_float32(width, height),
                       PixelType::float32, width, height);
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        return 2;
    }
    const std::filesystem::path directory = argv[2];
    if (std::string{argv[1]} == "write") {
        return write_files(directory) ? 0 : 1;
    }
    if (std::string{argv[1]} == "verify") {
        return verify_python_files(directory) ? 0 : 1;
    }
    return 2;
}
