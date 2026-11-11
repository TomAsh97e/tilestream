// GPU paths of the deterministic random cases, and fabricated host metadata that preparation must
// either reject or keep within the payload.
#include "cuda_test_support.h"
#include "robustness/cases.h"

#include <map>

namespace {

using namespace test;
using robustness::Case;

using Execute = Status (*)(const cuda::BatchPlan &, MutableBytes, cuda::ResultBuffer,
                           cudaStream_t) noexcept;

HostImage host(const Case &c) {
    HostImage result;
    result.image = c.image;
    result.pitch = c.pitch;
    result.bytes = c.bytes;
    return result;
}

// Device output with one padding sample per row, filled with the sentinel.
struct Output {
    ImageSpec image;
    std::uint64_t pitch = 0;
    DevicePtr data;

    explicit Output(ImageSpec spec) : image(spec) {
        const auto s = robustness::sample_bytes(spec.type);
        pitch = (std::uint64_t{spec.width} + 1) * s;
        data = device_alloc(pitch * spec.height);
    }
    ImageView view() const {
        return {data.get(), image, pitch, pitch * image.height};
    }
    // Logical pixels match and the padding still holds the sentinel.
    bool holds(const std::vector<std::uint8_t> &expected) const {
        const auto bytes = download(data.get(), pitch * image.height);
        const auto row = image.width * robustness::sample_bytes(image.type);
        for (std::uint64_t y = 0; y < image.height; ++y) {
            const auto *line = bytes.data() + y * pitch;
            if (!std::equal(line, line + row,
                            expected.begin() + static_cast<std::ptrdiff_t>(y * row)) ||
                !std::all_of(line + row, line + pitch,
                             [](auto byte) { return byte == sentinel; })) {
                return false;
            }
        }
        return true;
    }
};

std::vector<DeviceResult> run(cuda::Operation operation, const std::vector<cuda::BatchItem> &items,
                              Execute execute) {
    std::unique_ptr<cuda::BatchPlan> plan;
    EXPECT_EQ(cuda::prepare_batch({operation, items.data(), items.size(), {}}, plan, nullptr),
              Status::success);
    if (plan == nullptr) {
        return {};
    }
    const auto workspace = workspace_for(*plan);
    const auto results = device_alloc(items.size() * sizeof(DeviceResult), 0xff);
    EXPECT_EQ(execute(*plan, workspace.bytes(),
                      {static_cast<DeviceResult *>(results.get()), items.size()}, nullptr),
              Status::success);
    EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    return download_results(results, items.size());
}

class Failures {
  public:
    void add(const std::string &what, const Case &c) {
        if (count_++ < 10) {
            ADD_FAILURE() << what << ", " << c.describe();
        }
    }
    std::uint64_t count() const {
        return count_;
    }

  private:
    std::uint64_t count_ = 0;
};

// One batch of a single pixel type with all three geometries: GPU encoding against the CPU
// bitstream, full decode of the CPU files, and ROI decode of the GPU output on the trusted path.
void check_batch(const std::vector<Case> &cases, Failures &failures) {
    std::vector<Bytes> files;
    for (const auto &c : cases) {
        files.push_back(cpu_encode(host(c), c.tile));
    }
    std::vector<GpuArchive> archives(std::size(robustness::shapes));
    std::vector<std::pair<std::size_t, std::size_t>> placed(cases.size());
    for (std::size_t g = 0; g < archives.size(); ++g) {
        const auto shape = robustness::shapes[g];
        std::vector<HostImage> images;
        for (std::size_t i = 0; i < cases.size(); ++i) {
            if (cases[i].tile.width == shape.width && cases[i].tile.height == shape.height) {
                placed[i] = {g, images.size()};
                images.push_back(host(cases[i]));
            }
        }
        if (!images.empty()) {
            archives[g] = gpu_encode(images, shape);
        }
    }

    std::vector<ImageMetadata> metadata(cases.size());
    std::vector<DevicePtr> sources;
    std::vector<Output> full;
    std::vector<Output> regions;
    std::vector<cuda::BatchItem> decode(cases.size());
    std::vector<cuda::BatchItem> roi(cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto &c = cases[i];
        const auto &archive = archives[placed[i].first];
        const auto j = placed[i].second;
        if (archive.files.size() <= j || archive.files[j] != files[i]) {
            failures.add("GPU bitstream differs from the CPU bitstream", c);
        }
        EXPECT_EQ(parse_metadata({files[i].data(), files[i].size()}, {}, metadata[i]),
                  Status::success);
        sources.push_back(upload(files[i]));
        full.emplace_back(c.image);
        regions.emplace_back(ImageSpec{c.image.type, c.roi.width, c.roi.height});
        decode[i].compressed_input.bytes = {sources.back().get(), files[i].size()};
        decode[i].compressed_input.metadata = &metadata[i];
        decode[i].output = full.back().view();

        auto &trusted = roi[i].compressed_input;
        trusted.bytes = {archive.outputs[j].get(), archive.bounds[j]};
        trusted.encoder_result = static_cast<const DeviceResult *>(archive.results.get()) + j;
        trusted.trusted_image = c.image;
        trusted.trusted_tile = c.tile;
        roi[i].roi = c.roi;
        roi[i].output = regions.back().view();
    }
    const auto decoded = run(cuda::Operation::decompress, decode, cuda::decompress_async);
    const auto cropped = run(cuda::Operation::decode_roi, roi, cuda::decode_roi_async);
    ASSERT_EQ(decoded.size(), cases.size());
    ASSERT_EQ(cropped.size(), cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto &c = cases[i];
        if (decoded[i].status != Status::success || !full[i].holds(c.pixels())) {
            failures.add("GPU decode of the CPU file differs", c);
        }
        if (cropped[i].status != Status::success || !regions[i].holds(c.region(c.roi))) {
            failures.add("GPU ROI decode of the GPU file differs", c);
        }
    }
}

TEST(CudaRobustness, RandomRoundTripsOnEveryPath) {
    const auto seeds = robustness::selected_seeds();
    Failures failures;
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        std::vector<Case> batch;
        for (const auto seed : seeds) {
            auto c = robustness::make_case(seed);
            if (c.image.type == type) {
                batch.push_back(std::move(c));
            }
            if (batch.size() == 240 || (seed == seeds.back() && !batch.empty())) {
                check_batch(batch, failures);
                batch.clear();
            }
        }
    }
    EXPECT_EQ(failures.count(), 0U);
    RecordProperty("cases", static_cast<int>(seeds.size()));
}

// Constant, smooth and noisy bands aligned to every tile height, so the archives hold all three
// tile modes.
HostImage mixed_image(PixelType type) {
    auto image = make_image(type, 300, 192);
    const std::uint32_t patterns[] = {0, 5, 4};
    for (std::uint32_t band = 0; band < 3; ++band) {
        robustness::fill({type, 300, 64}, image.pitch, patterns[band], band,
                         image.bytes.data() + band * 64 * image.pitch);
    }
    return image;
}

std::uint64_t nudge(std::uint64_t value, std::mt19937_64 &rng, std::uint64_t step = 1) {
    if (rng() % 3 == 0) {
        return rng() % 3 == 0 ? rng() : rng() % 65536;
    }
    return value + (rng() % 17 - 8) * step;
}

// Changes one field of parsed metadata. Preparation must reject the result, or every device read
// must stay inside the source bytes; compute-sanitizer memcheck checks the second part.
void fabricate(ImageMetadata &m, std::mt19937_64 &rng) {
    auto &tile = m.tiles[static_cast<std::size_t>(rng() % m.tiles.size())];
    auto &segment = m.segments.empty()
                        ? m.segments.emplace_back()
                        : m.segments[static_cast<std::size_t>(rng() % m.segments.size())];
    switch (rng() % 12) {
    case 0:
        tile.mode = static_cast<TileMode>(rng() % 4);
        break;
    case 1:
        tile.width = static_cast<std::uint16_t>(nudge(tile.width, rng));
        break;
    case 2:
        tile.height = static_cast<std::uint16_t>(nudge(tile.height, rng));
        break;
    case 3:
        tile.offset = nudge(tile.offset, rng, 4);
        break;
    case 4:
        tile.length = nudge(tile.length, rng, 4);
        break;
    case 5:
        tile.first_segment = nudge(tile.first_segment, rng);
        break;
    case 6:
        segment.seed = static_cast<std::uint32_t>(nudge(segment.seed, rng));
        break;
    case 7:
        segment.bit_width = static_cast<std::uint32_t>(rng() % 40);
        break;
    case 8:
        segment.payload_offset = nudge(segment.payload_offset, rng, 4);
        break;
    case 9:
        segment.payload_bytes = nudge(segment.payload_bytes, rng, 4);
        break;
    case 10:
        m.file_bytes = nudge(m.file_bytes, rng, 4);
        break;
    default:
        if (rng() % 2 == 0) {
            m.tiles.pop_back();
        } else if (!m.segments.empty()) {
            m.segments.pop_back();
        }
        break;
    }
}

TEST(CudaRobustness, FabricatedMetadataIsRejectedOrStaysInBounds) {
    std::mt19937_64 rng(4242);
    std::map<int, std::uint64_t> statuses;
    std::uint64_t accepted = 0;
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        const auto image = mixed_image(type);
        for (const auto shape : robustness::shapes) {
            const auto file = cpu_encode(image, shape);
            ImageMetadata metadata;
            ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
            std::array<std::uint64_t, 3> modes = {};
            for (const auto &tile : metadata.tiles) {
                ++modes[static_cast<std::size_t>(tile.mode)];
            }
            EXPECT_TRUE(modes[0] > 0 && modes[1] > 0 && modes[2] > 0)
                << modes[0] << " " << modes[1] << " " << modes[2];
            const auto source = upload(file);
            const Output output(image.image);
            const auto results = device_alloc(sizeof(DeviceResult));
            for (int trial = 0; trial < 500; ++trial) {
                auto fabricated = metadata;
                fabricate(fabricated, rng);
                cuda::BatchItem item;
                item.compressed_input.bytes = {source.get(), file.size()};
                item.compressed_input.metadata = &fabricated;
                const auto x = static_cast<std::uint32_t>(rng() % 300);
                const auto y = static_cast<std::uint32_t>(rng() % 192);
                item.roi = {x, y, 1 + static_cast<std::uint32_t>(rng() % (300 - x)),
                            1 + static_cast<std::uint32_t>(rng() % (192 - y))};
                item.output = {output.data.get(),
                               {type, item.roi.width, item.roi.height},
                               output.pitch,
                               output.pitch * 192};
                std::unique_ptr<cuda::BatchPlan> plan;
                const auto status =
                    cuda::prepare_batch({cuda::Operation::decode_roi, &item, 1, {}}, plan, nullptr);
                ++statuses[static_cast<int>(status)];
                if (status != Status::success) {
                    EXPECT_TRUE(status == Status::malformed_data ||
                                status == Status::unsupported_format ||
                                status == Status::buffer_too_small)
                        << static_cast<int>(status);
                    continue;
                }
                ++accepted;
                const auto workspace = workspace_for(*plan);
                ASSERT_EQ(cuda::decode_roi_async(*plan, workspace.bytes(),
                                                 {static_cast<DeviceResult *>(results.get()), 1},
                                                 nullptr),
                          Status::success);
                ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            }
        }
    }
    std::string summary;
    for (const auto &[status, count] : statuses) {
        summary += " " + std::to_string(status) + ":" + std::to_string(count);
    }
    RecordProperty("statuses", summary);
    EXPECT_GT(accepted, 0U);
    EXPECT_GT(statuses[static_cast<int>(Status::malformed_data)], 0U);
}

} // namespace
