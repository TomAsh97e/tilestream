#pragma once

// Checks shared by the libFuzzer targets and the deterministic robustness tests. A check returns
// an empty string when the input is rejected with a documented status, or when it parses, decodes
// and survives a canonical round trip consistently.
#include <tilestream/api.h>
#include <tilestream/cpu.h>
#include <tilestream/format.h>
#include <tilestream/sizing.h>

#include "robustness/cases.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace fuzz {

using namespace tilestream;
using Bytes = std::vector<std::uint8_t>;

// Allocation limits of every fuzzed parse; images above decode_limit are parsed but not decoded.
inline constexpr ParseLimits limits{1 << 20, 1 << 15, 8 << 20};
inline constexpr std::uint64_t decode_limit = 4 << 20;

inline std::uint64_t read(const std::uint8_t *data, std::uint64_t bytes) {
    std::uint64_t value = 0;
    for (std::uint64_t i = 0; i < bytes; ++i) {
        value |= std::uint64_t{data[i]} << (8 * i);
    }
    return value;
}

inline void write(std::uint8_t *data, std::uint64_t bytes, std::uint64_t value) {
    for (std::uint64_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

inline bool documented_rejection(Status status) {
    return status == Status::malformed_data || status == Status::unsupported_format ||
           status == Status::unsupported_type || status == Status::size_overflow ||
           status == Status::resource_limit;
}

// Structure of accepted metadata, recomputed independently of the parser.
inline std::string check_structure(const ImageMetadata &m, std::uint64_t size) {
    const auto &image = m.image;
    const std::uint64_t across = (image.width + m.tile.width - 1) / m.tile.width;
    const std::uint64_t down = (image.height + m.tile.height - 1) / m.tile.height;
    if (m.file_bytes != size || m.tiles.size() != across * down) {
        return "accepted metadata has the wrong size or tile count";
    }
    const std::uint64_t s = robustness::sample_bytes(image.type);
    std::uint64_t offset = format::header_bytes + format::index_entry_bytes * m.tiles.size();
    std::uint64_t segment = 0;
    for (std::size_t i = 0; i < m.tiles.size(); ++i) {
        const auto &t = m.tiles[i];
        const auto x = i % across * m.tile.width;
        const auto y = i / across * m.tile.height;
        const auto width = std::min<std::uint64_t>(m.tile.width, image.width - x);
        const auto height = std::min<std::uint64_t>(m.tile.height, image.height - y);
        const auto raw = (width * height * s + 3) / 4 * 4;
        if (t.width != width || t.height != height || t.offset != offset || t.length == 0 ||
            t.length % 4 != 0 || t.length > raw) {
            return "accepted tile " + std::to_string(i) + " is inconsistent";
        }
        if (t.mode == TileMode::predict_bitpack) {
            if (t.first_segment != segment || m.segments.size() - segment < height) {
                return "accepted tile " + std::to_string(i) + " has wrong segments";
            }
            std::uint64_t cursor = 0;
            for (std::uint64_t row = 0; row < height; ++row, ++segment) {
                const auto &g = m.segments[static_cast<std::size_t>(segment)];
                cursor += format::segment_header_bytes;
                if (g.payload_offset != cursor ||
                    g.payload_bytes != ((width - 1) * g.bit_width + 31) / 32 * 4) {
                    return "accepted segment " + std::to_string(segment) + " is inconsistent";
                }
                cursor += g.payload_bytes;
            }
            if (cursor != t.length) {
                return "accepted tile " + std::to_string(i) + " has a wrong length";
            }
        }
        offset += t.length;
    }
    if (offset != size || segment != m.segments.size()) {
        return "accepted payloads do not cover the file";
    }
    return "";
}

inline std::string check_file(const std::uint8_t *data, std::size_t size,
                              Status *result = nullptr) {
    ImageMetadata metadata;
    const auto status = parse_metadata({data, size}, limits, metadata);
    if (result != nullptr) {
        *result = status;
    }
    if (status != Status::success) {
        if (!documented_rejection(status)) {
            return "undocumented parser status " + std::to_string(static_cast<int>(status));
        }
        return metadata.tiles.empty() && metadata.file_bytes == 0 ? ""
                                                                  : "rejection changed metadata";
    }
    auto problem = check_structure(metadata, size);
    if (!problem.empty()) {
        return problem;
    }
    const auto image = metadata.image;
    const auto s = robustness::sample_bytes(image.type);
    const std::uint64_t row = image.width * s;
    if (row * image.height > decode_limit) {
        return "";
    }

    // Decode with one padding sample per row, which must stay untouched.
    const auto pitch = row + s;
    Bytes output(static_cast<std::size_t>(pitch * image.height), 0xa5);
    if (cpu::decompress({data, size}, {output.data(), image, pitch, output.size()}, limits) !=
        Status::success) {
        return "accepted file failed to decode";
    }
    for (std::uint64_t y = 0; y < image.height; ++y) {
        const auto *pad = output.data() + y * pitch + row;
        if (!std::all_of(pad, pad + s, [](auto byte) { return byte == 0xa5; })) {
            return "decode wrote into row padding";
        }
    }

    // A canonical encoding of the decoded pixels is never larger than any accepted encoding,
    // and decodes to the same pixels.
    std::uint64_t bound = 0;
    std::uint64_t written = 0;
    if (compress_bound(image, metadata.tile, bound) != Status::success) {
        return "accepted image has no bound";
    }
    Bytes again(static_cast<std::size_t>(bound));
    if (cpu::compress({output.data(), image, pitch, output.size()}, metadata.tile,
                      {again.data(), bound}, written) != Status::success) {
        return "decoded pixels failed to encode";
    }
    if (written > size) {
        return "canonical encoding is larger than the accepted file";
    }
    Bytes repeat(static_cast<std::size_t>(row * image.height));
    if (cpu::decompress({again.data(), written}, {repeat.data(), image, row, repeat.size()}) !=
        Status::success) {
        return "canonical encoding failed to decode";
    }
    for (std::uint64_t y = 0; y < image.height; ++y) {
        if (!std::equal(repeat.begin() + static_cast<std::ptrdiff_t>(y * row),
                        repeat.begin() + static_cast<std::ptrdiff_t>((y + 1) * row),
                        output.begin() + static_cast<std::ptrdiff_t>(y * pitch))) {
            return "canonical round trip changed pixels";
        }
    }
    return "";
}

class Reader {
  public:
    Reader(const std::uint8_t *data, std::size_t size) : data_(data), size_(size) {}

    bool empty() const {
        return position_ >= size_;
    }
    std::uint64_t take(std::uint64_t bytes) {
        std::uint64_t value = 0;
        for (std::uint64_t i = 0; i < bytes && position_ < size_; ++i) {
            value |= std::uint64_t{data_[position_++]} << (8 * i);
        }
        return value;
    }

  private:
    const std::uint8_t *data_;
    std::size_t size_;
    std::size_t position_ = 0;
};

struct Field {
    std::uint64_t offset;
    std::uint64_t bytes;
};

inline constexpr Field header_fields[] = {{0, 8},  {8, 2},  {10, 2}, {12, 2}, {14, 1}, {15, 1},
                                          {16, 4}, {20, 4}, {24, 2}, {26, 2}, {28, 2}, {30, 2},
                                          {32, 8}, {40, 8}, {48, 8}, {56, 8}};
inline constexpr Field entry_fields[] = {{0, 1}, {1, 3}, {4, 2}, {6, 2}, {8, 8}, {16, 8}, {24, 8}};
inline constexpr Field segment_fields[] = {{0, 4}, {4, 1}, {5, 3}};

enum class Mutation { header, entry, segment, payload, splice, resize };

inline const char *name(Mutation kind) {
    constexpr const char *names[] = {"header", "entry", "segment", "payload", "splice", "resize"};
    return names[static_cast<int>(kind)];
}

// Replaces a field with an arbitrary value, or moves it by a small signed delta.
inline void change(Bytes &file, std::uint64_t at, Field field, std::uint64_t value) {
    if (at + field.bytes > file.size()) {
        return;
    }
    const auto old = read(file.data() + at, field.bytes);
    const auto delta = static_cast<std::uint64_t>(static_cast<std::int8_t>(value & 0xff));
    write(file.data() + at, field.bytes, (value >> 63) != 0 ? old + delta : value >> 8);
}

// Grows or shrinks the end of one tile payload by whole words and shifts the later offsets and
// the file size, so that only the contents of that tile become inconsistent.
inline void splice(Bytes &file, std::uint64_t tiles, std::uint64_t index, std::uint64_t field,
                   std::uint64_t value) {
    if (file.size() < format::header_bytes + format::index_entry_bytes * tiles) {
        return;
    }
    const auto entry = format::header_bytes + format::index_entry_bytes * index;
    const auto offset = read(file.data() + entry + format::entry::offset, 8);
    const auto length = read(file.data() + entry + format::entry::length, 8);
    const auto bytes = 4 * (1 + field % 4);
    const bool grow = value % 2 == 0;
    if (offset > file.size() || length > file.size() - offset || (!grow && length < bytes)) {
        return;
    }
    const auto end = file.begin() + static_cast<std::ptrdiff_t>(offset + length);
    if (grow) {
        file.insert(end, bytes, static_cast<std::uint8_t>(value >> 8));
    } else {
        file.erase(end - static_cast<std::ptrdiff_t>(bytes), end);
    }
    const auto delta = grow ? bytes : 0 - bytes;
    write(file.data() + entry + format::entry::length, 8, length + delta);
    for (auto i = index + 1; i < tiles; ++i) {
        auto *later = file.data() + format::header_bytes + format::index_entry_bytes * i +
                      format::entry::offset;
        write(later, 8, read(later, 8) + delta);
    }
    auto *size = file.data() + format::header::file_size;
    write(size, 8, read(size, 8) + delta);
}

// The leading bytes describe a valid image. Each following 12-byte command mutates its canonical
// file: a header field, an index entry, a row descriptor, a payload byte, the length of one tile
// payload, or the file size.
inline std::string check_mutation(const std::uint8_t *data, std::size_t size,
                                  Status *result = nullptr, Mutation *last = nullptr) {
    Reader in(data, size);
    const auto config = in.take(1);
    const ImageSpec image{config % 2 == 0 ? PixelType::uint16 : PixelType::float32,
                          1 + static_cast<std::uint32_t>(in.take(2) % 300),
                          1 + static_cast<std::uint32_t>(in.take(2) % 160)};
    const auto tile = robustness::shapes[config / 2 % std::size(robustness::shapes)];
    const auto pattern = static_cast<std::uint32_t>(in.take(1));
    const auto seed = in.take(4);

    const auto row = image.width * robustness::sample_bytes(image.type);
    Bytes pixels(static_cast<std::size_t>(row * image.height));
    robustness::fill(image, row, pattern, seed, pixels.data());
    std::uint64_t bound = 0;
    std::uint64_t written = 0;
    if (compress_bound(image, tile, bound) != Status::success) {
        return "valid image has no bound";
    }
    Bytes file(static_cast<std::size_t>(bound));
    if (cpu::compress({pixels.data(), image, row, pixels.size()}, tile, {file.data(), bound},
                      written) != Status::success) {
        return "valid image failed to encode";
    }
    file.resize(static_cast<std::size_t>(written));
    ImageMetadata metadata;
    if (parse_metadata({file.data(), file.size()}, {}, metadata) != Status::success) {
        return "canonical file was rejected";
    }

    bool mutated = false;
    const auto payload = metadata.tiles[0].offset;
    for (int command = 0; command < 8 && !in.empty(); ++command) {
        const auto kind = static_cast<Mutation>(in.take(1) % 6);
        const auto field = in.take(1);
        const auto target = in.take(2);
        const auto value = in.take(8);
        mutated = true;
        if (last != nullptr) {
            *last = kind;
        }
        const auto index = target % metadata.tiles.size();
        const auto &t = metadata.tiles[static_cast<std::size_t>(index)];
        switch (kind) {
        case Mutation::header:
            change(file, 0, header_fields[field % std::size(header_fields)], value);
            break;
        case Mutation::entry:
            change(file, format::header_bytes + format::index_entry_bytes * index,
                   entry_fields[field % std::size(entry_fields)], value);
            break;
        case Mutation::segment:
            if (t.mode == TileMode::predict_bitpack) {
                const auto &g =
                    metadata.segments[static_cast<std::size_t>(t.first_segment + field % t.height)];
                change(file, t.offset + g.payload_offset - format::segment_header_bytes,
                       segment_fields[target % std::size(segment_fields)], value);
            }
            break;
        case Mutation::payload:
            if (file.size() > payload) {
                file[static_cast<std::size_t>(payload + value % (file.size() - payload))] ^=
                    static_cast<std::uint8_t>(field | 1);
            }
            break;
        case Mutation::splice:
            splice(file, metadata.tiles.size(), index, field, value);
            break;
        default:
            if (field % 2 == 0) {
                file.resize(static_cast<std::size_t>(value % (file.size() + 1)));
            } else {
                file.resize(file.size() + 1 + field % 8, static_cast<std::uint8_t>(value));
            }
            break;
        }
    }
    if (!mutated) {
        Bytes decoded(pixels.size());
        if (cpu::decompress({file.data(), file.size()},
                            {decoded.data(), image, row, decoded.size()},
                            limits) != Status::success ||
            decoded != pixels) {
            return "canonical file did not reproduce its pixels";
        }
    }
    return check_file(file.data(), file.size(), result);
}

} // namespace fuzz
