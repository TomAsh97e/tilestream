#pragma once

// Statistics, JSON output and input images shared by the benchmark programs.
#include <tilestream/types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace bench {

using tilestream::ImageSpec;
using tilestream::PixelType;

inline double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

inline std::uint64_t sample_bytes(PixelType type) {
    return type == PixelType::uint16 ? 2 : 4;
}

// Nearest-rank percentile of sorted samples.
inline double percentile(const std::vector<double> &sorted, double q) {
    const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    return sorted[std::max<std::size_t>(rank, 1) - 1];
}

// One timed quantity: every sample in milliseconds per operation.
struct Series {
    std::string name;
    std::string clock;
    std::vector<double> ms;

    double median() const {
        auto sorted = ms;
        std::sort(sorted.begin(), sorted.end());
        return percentile(sorted, 0.5);
    }
};

class Json {
  public:
    Json &key(const std::string &name) {
        separate();
        out_ << quote(name) << ':';
        fresh_ = true;
        return *this;
    }
    Json &open(char bracket) {
        separate();
        out_ << bracket;
        first_ = true;
        return *this;
    }
    Json &close(char bracket) {
        out_ << bracket;
        first_ = false;
        return *this;
    }
    Json &value(const std::string &text) {
        separate();
        out_ << quote(text);
        return *this;
    }
    Json &value(const char *text) {
        return value(std::string(text));
    }
    Json &value(double number) {
        separate();
        char buffer[32];
        std::snprintf(buffer, sizeof buffer, "%.9g", number);
        out_ << (std::isfinite(number) ? buffer : "null");
        return *this;
    }
    Json &value(std::uint64_t number) {
        separate();
        out_ << number;
        return *this;
    }
    Json &value(bool flag) {
        separate();
        out_ << (flag ? "true" : "false");
        return *this;
    }
    // Inserts an already serialized JSON value.
    Json &raw(const std::string &text) {
        separate();
        out_ << text;
        return *this;
    }
    template <class T> Json &field(const std::string &name, const T &v) {
        return key(name).value(v);
    }
    std::string str() const {
        return out_.str();
    }

  private:
    void separate() {
        if (fresh_) {
            fresh_ = false;
            first_ = false;
            return;
        }
        if (!first_) {
            out_ << ',';
        }
        first_ = false;
    }
    static std::string quote(const std::string &text) {
        std::string result = "\"";
        for (const char c : text) {
            if (c == '"' || c == '\\') {
                result += '\\';
            }
            result += c;
        }
        return result + '"';
    }

    std::ostringstream out_;
    bool first_ = true;
    bool fresh_ = false;
};

inline void write(Json &json, const Series &series) {
    auto sorted = series.ms;
    std::sort(sorted.begin(), sorted.end());
    const auto mean =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    double squares = 0;
    for (const auto x : sorted) {
        squares += (x - mean) * (x - mean);
    }
    json.open('{').field("name", series.name).field("clock", series.clock);
    json.field("unit", "ms").field("count", std::uint64_t{sorted.size()});
    json.field("median", percentile(sorted, 0.5)).field("p05", percentile(sorted, 0.05));
    json.field("p95", percentile(sorted, 0.95)).field("min", sorted.front());
    json.field("max", sorted.back()).field("mean", mean);
    json.field("stdev", sorted.size() > 1
                            ? std::sqrt(squares / static_cast<double>(sorted.size() - 1))
                            : 0.0);
    json.key("samples").open('[');
    for (const auto x : series.ms) {
        json.value(x);
    }
    json.close(']').close('}');
}

// A dense image in host memory.
struct Image {
    ImageSpec spec;
    std::vector<std::uint8_t> bytes;

    std::uint64_t row() const {
        return spec.width * sample_bytes(spec.type);
    }
};

// A smooth background with noise in the low bits, like a detector frame.
inline Image detector_frame(PixelType type, std::uint32_t width, std::uint32_t height,
                            std::uint64_t seed) {
    Image image{{type, width, height}, {}};
    image.bytes.resize(image.row() * height);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto noise = rng();
            auto *pixel = image.bytes.data() + y * image.row() + x * sample_bytes(type);
            if (type == PixelType::uint16) {
                const auto value = static_cast<std::uint16_t>(1000 + x / 4 + y / 8 + (noise >> 28));
                std::memcpy(pixel, &value, 2);
            } else {
                const float value = 100.0F + 0.01F * static_cast<float>(x) +
                                    0.02F * static_cast<float>(y) +
                                    1e-4F * static_cast<float>(noise >> 20);
                std::memcpy(pixel, &value, 4);
            }
        }
    }
    return image;
}

// Reads a little-endian, C-ordered two-dimensional uint16 or float32 .npy file.
inline bool load_npy(const std::string &path, Image &image, std::string &error) {
    std::ifstream in(path, std::ios::binary);
    char magic[8] = {};
    if (!in.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        error = "not a .npy file";
        return false;
    }
    std::uint32_t length = 0;
    if (magic[6] == 1) {
        unsigned char size[2] = {};
        in.read(reinterpret_cast<char *>(size), 2);
        length = size[0] | std::uint32_t{size[1]} << 8;
    } else {
        unsigned char size[4] = {};
        in.read(reinterpret_cast<char *>(size), 4);
        length = size[0] | std::uint32_t{size[1]} << 8 | std::uint32_t{size[2]} << 16 |
                 std::uint32_t{size[3]} << 24;
    }
    std::string header(length, '\0');
    in.read(header.data(), length);
    const bool u16 = header.find("'<u2'") != std::string::npos;
    const bool f32 = header.find("'<f4'") != std::string::npos;
    unsigned height = 0;
    unsigned width = 0;
    const auto shape = header.find("'shape': (");
    if ((!u16 && !f32) || header.find("'fortran_order': False") == std::string::npos ||
        shape == std::string::npos ||
        std::sscanf(header.c_str() + shape, "'shape': (%u, %u)", &height, &width) != 2) {
        error = "expected a C-ordered 2-D <u2 or <f4 array";
        return false;
    }
    image.spec = {u16 ? PixelType::uint16 : PixelType::float32, width, height};
    image.bytes.resize(image.row() * height);
    if (!in.read(reinterpret_cast<char *>(image.bytes.data()),
                 static_cast<std::streamsize>(image.bytes.size()))) {
        error = "truncated data";
        return false;
    }
    return true;
}

} // namespace bench
