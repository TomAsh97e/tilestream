// Robustness of the CPU codec and parser: deterministic random round trips with recorded seeds,
// and structured mutations of valid files through the checks used by the fuzzers.
#include "harness.h"
#include "robustness/cases.h"

#include <gtest/gtest.h>

#include <array>
#include <bitset>
#include <map>

namespace {

using namespace tilestream;
using robustness::Case;

// Every case: canonical encoding within the bound and deterministic, accepted by the parser, and
// decoded to the same pixels without touching the output padding.
TEST(Robustness, RandomRoundTrips) {
    const auto seeds = robustness::selected_seeds();
    std::array<std::uint64_t, 3> modes = {};
    std::bitset<17> uint16_widths;
    std::bitset<33> float_widths;
    std::uint64_t single_sample_rows = 0;
    std::uint64_t edge_tiles = 0;
    std::uint64_t padded = 0;
    std::uint64_t failures = 0;
    for (const auto seed : seeds) {
        const auto c = robustness::make_case(seed);
        const auto fail = [&](const std::string &what) {
            if (failures++ < 10) {
                ADD_FAILURE() << what << ", " << c.describe();
            }
        };
        const auto s = robustness::sample_bytes(c.image.type);
        std::uint64_t bound = 0;
        ASSERT_EQ(compress_bound(c.image, c.tile, bound), Status::success) << c.describe();
        fuzz::Bytes file(static_cast<std::size_t>(bound + 8), 0xa5);
        fuzz::Bytes second(file.size(), 0x5a);
        std::uint64_t written = 0;
        std::uint64_t again = 0;
        const ConstImageView input{c.bytes.data(), c.image, c.pitch, c.bytes.size()};
        if (cpu::compress(input, c.tile, {file.data(), bound}, written) != Status::success ||
            cpu::compress(input, c.tile, {second.data(), bound}, again) != Status::success) {
            fail("compression failed");
            continue;
        }
        if (written > bound || again != written ||
            !std::equal(file.begin(), file.begin() + static_cast<std::ptrdiff_t>(written),
                        second.begin()) ||
            !std::all_of(file.begin() + static_cast<std::ptrdiff_t>(bound), file.end(),
                         [](auto byte) { return byte == 0xa5; })) {
            fail("encoding is not deterministic or exceeds its bound");
            continue;
        }
        ImageMetadata metadata;
        if (parse_metadata({file.data(), written}, {}, metadata) != Status::success) {
            fail("canonical file rejected");
            continue;
        }
        const auto structure = fuzz::check_structure(metadata, written);
        if (!structure.empty()) {
            fail(structure);
            continue;
        }
        const auto pitch = c.image.width * s + 2 * s;
        fuzz::Bytes output(static_cast<std::size_t>(pitch * c.image.height), 0xa5);
        if (cpu::decompress({file.data(), written},
                            {output.data(), c.image, pitch, output.size()}) != Status::success) {
            fail("decode failed");
            continue;
        }
        bool same = true;
        for (std::uint64_t y = 0; y < c.image.height && same; ++y) {
            const auto *line = output.data() + y * pitch;
            const auto *source = c.bytes.data() + y * c.pitch;
            same = std::equal(line, line + c.image.width * s, source) &&
                   std::all_of(line + c.image.width * s, line + pitch,
                               [](auto byte) { return byte == 0xa5; });
        }
        if (!same) {
            fail("decoded pixels or padding differ");
            continue;
        }

        padded += c.pitch > c.image.width * s ? 1 : 0;
        for (const auto &tile : metadata.tiles) {
            ++modes[static_cast<std::size_t>(tile.mode)];
            edge_tiles += tile.width != c.tile.width || tile.height != c.tile.height ? 1 : 0;
            // The encoder never packs one-sample rows, RAW is always smaller.
            single_sample_rows += tile.width == 1 ? tile.height : 0;
            if (tile.mode == TileMode::predict_bitpack) {
                for (std::uint64_t row = 0; row < tile.height; ++row) {
                    const auto width =
                        metadata.segments[static_cast<std::size_t>(tile.first_segment + row)]
                            .bit_width;
                    if (c.image.type == PixelType::uint16) {
                        uint16_widths.set(width);
                    } else {
                        float_widths.set(width);
                    }
                }
            }
        }
    }
    EXPECT_EQ(failures, 0U);
    RecordProperty("cases", static_cast<int>(seeds.size()));
    RecordProperty("tiles", "raw " + std::to_string(modes[0]) + ", constant " +
                                std::to_string(modes[1]) + ", predict " + std::to_string(modes[2]));
    RecordProperty("edges", "edge tiles " + std::to_string(edge_tiles) + ", one-sample rows " +
                                std::to_string(single_sample_rows) + ", padded images " +
                                std::to_string(padded));
    if (seeds.size() >= robustness::default_count) {
        // The generator reaches every mode, width and edge that the tests require.
        EXPECT_TRUE(modes[0] > 0 && modes[1] > 0 && modes[2] > 0);
        EXPECT_TRUE(uint16_widths.all()) << uint16_widths;
        EXPECT_TRUE(float_widths.all()) << float_widths;
        EXPECT_GT(single_sample_rows, 0U);
        EXPECT_GT(edge_tiles, 0U);
        EXPECT_GT(padded, 0U);
    }
}

// Valid files of every type and geometry with up to eight field mutations each. Every outcome
// must be a documented rejection or a consistent decode.
TEST(Robustness, MutatedFilesAreRejectedOrDecodedConsistently) {
    std::mt19937_64 rng(1313);
    std::map<std::string, std::map<int, std::uint64_t>> outcomes;
    std::uint64_t failures = 0;
    for (int i = 0; i < 20000; ++i) {
        fuzz::Bytes input(10 + 12 * (1 + rng() % 3));
        for (auto &byte : input) {
            byte = static_cast<std::uint8_t>(rng());
        }
        // Mostly small deltas, which keep the rest of the file valid.
        for (std::size_t at = 10 + 4; at + 8 <= input.size(); at += 12) {
            if (rng() % 4 != 0) {
                input[at + 7] |= 0x80;
            }
        }
        Status status = Status::success;
        fuzz::Mutation kind = fuzz::Mutation::header;
        const auto problem = fuzz::check_mutation(input.data(), input.size(), &status, &kind);
        if (!problem.empty() && failures++ < 10) {
            ADD_FAILURE() << problem << ", input " << i;
        }
        ++outcomes[fuzz::name(kind)][static_cast<int>(status)];
    }
    EXPECT_EQ(failures, 0U);
    for (const auto &[kind, statuses] : outcomes) {
        std::string line;
        for (const auto &[status, count] : statuses) {
            line += " " + std::to_string(status) + ":" + std::to_string(count);
        }
        RecordProperty(kind, line);
        // Each kind of mutation reaches the parser's rejections.
        EXPECT_GT(statuses.size(), 1U) << kind;
    }
    EXPECT_EQ(outcomes.size(), 6U);
}

// Inputs without mutation commands are canonical files that decode to their own pixels.
TEST(Robustness, HarnessAcceptsValidFilesAndUnmutatedInputs) {
    std::mt19937_64 rng(7);
    for (int i = 0; i < 200; ++i) {
        fuzz::Bytes input(10);
        for (auto &byte : input) {
            byte = static_cast<std::uint8_t>(rng());
        }
        Status status = Status::invalid_argument;
        EXPECT_EQ(fuzz::check_mutation(input.data(), input.size(), &status), "");
        EXPECT_EQ(status, Status::success);
    }
}

} // namespace
