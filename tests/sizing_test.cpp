#include <tilestream/api.h>
#include <tilestream/cuda.h>
#include <tilestream/format.h>

#include "checked_size.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

int failures = 0;

void check(bool condition, const char *expression, int line) {
    if (!condition) {
        std::cerr << "line " << line << ": " << expression << '\n';
        ++failures;
    }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

using namespace tilestream;

static_assert(sizeof(DeviceResult) == 16);
static_assert(alignof(DeviceResult) == 8);
static_assert(std::is_trivially_copyable_v<DeviceResult>);
static_assert(std::is_standard_layout_v<ImageView>);

constexpr std::array<TileShape, 3> shapes = {{{128, 32}, {64, 64}, {256, 16}}};

std::uint64_t tiled_raw_bound(const ImageSpec &image, TileShape tile) {
    const std::uint64_t sample = image.type == PixelType::uint16 ? 2 : 4;
    std::uint64_t result = 64;
    for (std::uint64_t y = 0; y < image.height; y += tile.height) {
        for (std::uint64_t x = 0; x < image.width; x += tile.width) {
            const auto width = std::min<std::uint64_t>(tile.width, image.width - x);
            const auto height = std::min<std::uint64_t>(tile.height, image.height - y);
            const auto raw = width * height * sample;
            result += 32 + ((raw + 3) / 4) * 4;
        }
    }
    return result;
}

void test_bound() {
    std::uint64_t bytes = 123;
    CHECK(compress_bound({PixelType::uint16, 1, 1}, {}, bytes) == Status::success);
    CHECK(bytes == 100);
    CHECK(compress_bound({PixelType::float32, 1, 1}, {}, bytes) == Status::success);
    CHECK(bytes == 100);
    CHECK(compress_bound({PixelType::uint16, 128, 32}, {}, bytes) == Status::success);
    CHECK(bytes == 8288);
    CHECK(compress_bound({PixelType::uint16, 129, 33}, {}, bytes) == Status::success);
    CHECK(bytes == 8708);
    CHECK(compress_bound({PixelType::float32, 128, 32}, {}, bytes) == Status::success);
    CHECK(bytes == 16480);
    CHECK(compress_bound({PixelType::uint16, 65536, 65536}, {}, bytes) == Status::success);
    CHECK(bytes == 8623489088ULL);
    CHECK(compress_bound({PixelType::float32, UINT32_MAX, 1}, {}, bytes) == Status::success);
    CHECK(bytes > UINT32_MAX);

    for (const auto shape : shapes) {
        for (const auto type : {PixelType::uint16, PixelType::float32}) {
            for (std::uint32_t height = 1; height <= 129; ++height) {
                for (std::uint32_t width = 1; width <= 129; ++width) {
                    const ImageSpec image{type, width, height};
                    CHECK(compress_bound(image, shape, bytes) == Status::success);
                    CHECK(bytes == tiled_raw_bound(image, shape));
                }
            }
            for (const auto width : {255U, 256U, 257U, 513U}) {
                const ImageSpec image{type, width, 65};
                CHECK(compress_bound(image, shape, bytes) == Status::success);
                CHECK(bytes == tiled_raw_bound(image, shape));
            }
        }
    }

    bytes = 123;
    CHECK(compress_bound({PixelType::uint16, 0, 1}, {}, bytes) == Status::invalid_argument);
    CHECK(compress_bound({PixelType::uint16, 1, 0}, {}, bytes) == Status::invalid_argument);
    CHECK(compress_bound({static_cast<PixelType>(99), 1, 1}, {}, bytes) ==
          Status::unsupported_type);
    CHECK(compress_bound({PixelType::uint16, 1, 1}, {0, 0}, bytes) == Status::invalid_argument);
    CHECK(compress_bound({PixelType::uint16, 1, 1}, {32, 128}, bytes) == Status::invalid_argument);
    CHECK(compress_bound({PixelType::uint16, 1, 1}, {128, 64}, bytes) == Status::invalid_argument);
    CHECK(compress_bound({PixelType::float32, UINT32_MAX, UINT32_MAX}, {}, bytes) ==
          Status::size_overflow);
    CHECK(compress_bound({PixelType::uint16, 0x80000000U, 0x80000000U}, {}, bytes) ==
          Status::size_overflow);
    // Raw pixels fit INT64_MAX here; adding the index must still be rejected.
    CHECK(compress_bound({PixelType::uint16, INT32_MAX, INT32_MAX}, {}, bytes) ==
          Status::size_overflow);
    CHECK(bytes == 123);
}

void test_span() {
    std::uint64_t bytes = 123;
    CHECK(image_span({PixelType::uint16, 3, 2}, 8, bytes) == Status::success);
    CHECK(bytes == 14);
    CHECK(image_span({PixelType::float32, 3, 2}, 16, bytes) == Status::success);
    CHECK(bytes == 28);
    CHECK(image_span({PixelType::uint16, 1, 1}, max_bytes - 1, bytes) == Status::success);
    CHECK(bytes == 2);
    bytes = 123;
    CHECK(image_span({PixelType::uint16, 3, 2}, 5, bytes) == Status::invalid_argument);
    CHECK(image_span({PixelType::uint16, 3, 2}, 7, bytes) == Status::invalid_argument);
    CHECK(image_span({PixelType::float32, 1, 1}, 6, bytes) == Status::invalid_argument);
    CHECK(image_span({PixelType::uint16, 0, 1}, 0, bytes) == Status::invalid_argument);
    CHECK(image_span({PixelType::uint16, 1, 0}, 2, bytes) == Status::invalid_argument);
    CHECK(image_span({static_cast<PixelType>(0), 1, 1}, 4, bytes) == Status::unsupported_type);
    CHECK(image_span({PixelType::uint16, 1, 1}, UINT64_MAX, bytes) == Status::size_overflow);
    CHECK(image_span({PixelType::uint16, 2, 2}, max_bytes - 1, bytes) == Status::size_overflow);
    CHECK(image_span({PixelType::uint16, 1, 3}, max_bytes - 1, bytes) == Status::size_overflow);
    CHECK(bytes == 123);
}

void test_checked_arithmetic() {
    std::uint64_t result = 123;
    CHECK(detail::add_size(max_bytes, 0, result));
    CHECK(result == max_bytes);
    CHECK(detail::multiply_size(max_bytes, 1, result));
    CHECK(result == max_bytes);
    CHECK(detail::multiply_size(max_bytes, 0, result));
    CHECK(result == 0);
    CHECK(detail::align_size(256, 256, result));
    CHECK(result == 256);
    CHECK(detail::align_size(257, 256, result));
    CHECK(result == 512);
    CHECK(detail::align_size(0, 256, result));
    CHECK(result == 0);
    CHECK(detail::align_size(max_bytes, 1, result));
    CHECK(result == max_bytes);

    result = 123;
    CHECK(!detail::add_size(max_bytes, 1, result));
    CHECK(!detail::add_size(UINT64_MAX, 0, result));
    CHECK(!detail::add_size(0, UINT64_MAX, result));
    CHECK(!detail::multiply_size(max_bytes, 2, result));
    CHECK(!detail::multiply_size(UINT64_MAX, 0, result));
    CHECK(!detail::multiply_size(0, UINT64_MAX, result));
    CHECK(!detail::align_size(max_bytes, 4, result));
    CHECK(!detail::align_size(UINT64_MAX, 256, result));
    CHECK(!detail::align_size(1, 0, result));
    CHECK(result == 123);
}

void test_workspace() {
    std::uint64_t bytes = 123;
    CHECK(workspace_size({}, bytes) == Status::success);
    CHECK(bytes == 0);
    CHECK(workspace_size({0, 0, {0, 0}, 0}, bytes) == Status::success);
    CHECK(bytes == 0);
    CHECK(workspace_size({1, 1, {128, 32}, 0}, bytes) == Status::success);
    CHECK(bytes == 17664);
    CHECK(workspace_size({1, 1, {64, 64}, 0}, bytes) == Status::success);
    CHECK(bytes == 17920);
    CHECK(workspace_size({1, 1, {256, 16}, 0}, bytes) == Status::success);
    CHECK(bytes == 17664);
    CHECK(workspace_size({1, 1, {128, 32}, 1}, bytes) == Status::success);
    CHECK(bytes == 17920);
    CHECK(workspace_size({1, 1, {128, 32}, 257}, bytes) == Status::success);
    CHECK(bytes == 18176);
    CHECK(workspace_size({2, 2, {128, 32}, 0}, bytes) == Status::success);
    CHECK(bytes == 34304);
    CHECK(workspace_size({1, max_window_tiles, {64, 64}, 0}, bytes) == Status::success);
    CHECK(bytes > UINT32_MAX && bytes % workspace_alignment == 0);

    const auto largest_scratch = ((max_bytes - 17664) / workspace_alignment) * workspace_alignment;
    CHECK(workspace_size({1, 1, {128, 32}, largest_scratch}, bytes) == Status::success);
    CHECK(bytes == largest_scratch + 17664);
    bytes = 123;
    CHECK(workspace_size({1, 1, {128, 32}, largest_scratch + 1}, bytes) == Status::size_overflow);
    CHECK(workspace_size({1, 1, {128, 32}, max_bytes}, bytes) == Status::size_overflow);
    CHECK(workspace_size({1, 1, {128, 32}, UINT64_MAX}, bytes) == Status::size_overflow);
    CHECK(workspace_size({0, 1, {}, 0}, bytes) == Status::invalid_argument);
    CHECK(workspace_size({0, 0, {}, 1}, bytes) == Status::invalid_argument);
    CHECK(workspace_size({1, 0, {}, 0}, bytes) == Status::invalid_argument);
    CHECK(workspace_size({2, 1, {}, 0}, bytes) == Status::invalid_argument);
    CHECK(workspace_size({1, 1, {128, 64}, 0}, bytes) == Status::invalid_argument);
    CHECK(workspace_size({1, max_window_tiles + 1, {}, 0}, bytes) == Status::resource_limit);
    CHECK(workspace_size({std::uint64_t{UINT32_MAX} + 1, std::uint64_t{UINT32_MAX} + 1, {}, 0},
                         bytes) == Status::resource_limit);
    CHECK(bytes == 123);
}

void test_layout_constants() {
    CHECK(format::magic == (std::array<std::uint8_t, 8>{0x54, 0x53, 0x43, 0, 13, 10, 26, 10}));
    CHECK(format::header::file_size + 8 == format::header_bytes);
    CHECK(format::entry::reserved_tail + 8 == format::index_entry_bytes);
    CHECK(format::segment::reserved + 3 == format::segment_header_bytes);
    CHECK(format::header_bytes % format::payload_alignment == 0);
    CHECK(static_cast<std::uint32_t>(PixelType::uint16) == 1);
    CHECK(static_cast<std::uint32_t>(PixelType::float32) == 2);
    CHECK(static_cast<std::uint32_t>(TileMode::raw) == 0);
    CHECK(static_cast<std::uint32_t>(TileMode::constant) == 1);
    CHECK(static_cast<std::uint32_t>(TileMode::predict_bitpack) == 2);
}

} // namespace

int main() {
    test_bound();
    test_span();
    test_checked_arithmetic();
    test_workspace();
    test_layout_constants();
    if (failures != 0) {
        std::cerr << failures << " failed checks\n";
        return 1;
    }
    std::cout << "Size and header contracts passed\n";
    return 0;
}
