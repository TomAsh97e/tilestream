#include "cuda_test_support.h"

#include <array>

namespace {

using namespace test;

HostImage crop(const HostImage &full, Roi roi) {
    auto result = make_image(full.image.type, roi.width, roi.height);
    for (std::uint32_t y = 0; y < roi.height; ++y) {
        for (std::uint32_t x = 0; x < roi.width; ++x) {
            result.set(x, y, full.get(roi.x + x, roi.y + y));
        }
    }
    return result;
}

std::vector<std::uint64_t> intersecting(const ImageMetadata &metadata, Roi roi) {
    const std::uint64_t columns =
        (metadata.image.width + metadata.tile.width - 1) / metadata.tile.width;
    std::vector<std::uint64_t> tiles;
    for (std::uint64_t row = roi.y / metadata.tile.height;
         row <= (std::uint64_t{roi.y} + roi.height - 1) / metadata.tile.height; ++row) {
        for (std::uint64_t column = roi.x / metadata.tile.width;
             column <= (std::uint64_t{roi.x} + roi.width - 1) / metadata.tile.width; ++column) {
            tiles.push_back(row * columns + column);
        }
    }
    return tiles;
}

cuda::CompressedInput parsed(ConstBytes bytes, const ImageMetadata &metadata) {
    cuda::CompressedInput input;
    input.bytes = bytes;
    input.metadata = &metadata;
    return input;
}

enum class Source { full, staged, poisoned };

// Device copies of parsed archives: the complete file, only the selected payloads in reverse
// order, or the complete file with every unselected payload overwritten.
struct Uploaded {
    ImageMetadata metadata;
    DevicePtr bytes;
    std::uint64_t size = 0;
    std::vector<cuda::TileMapping> mappings;
};

Uploaded upload_archive(const Bytes &file, Roi roi, Source source) {
    Uploaded result;
    EXPECT_EQ(parse_metadata({file.data(), file.size()}, {}, result.metadata), Status::success);
    const auto tiles = intersecting(result.metadata, roi);
    if (source == Source::staged) {
        Bytes staging(8, 0xee);
        for (auto it = tiles.rbegin(); it != tiles.rend(); ++it) {
            const auto &tile = result.metadata.tiles[*it];
            result.mappings.push_back({*it, staging.size()});
            const auto begin = file.begin() + static_cast<std::ptrdiff_t>(tile.offset);
            staging.insert(staging.end(), begin, begin + static_cast<std::ptrdiff_t>(tile.length));
            staging.insert(staging.end(), 4, 0xee);
        }
        std::reverse(result.mappings.begin(), result.mappings.end());
        result.bytes = upload(staging);
        result.size = staging.size();
        return result;
    }
    auto copy = file;
    if (source == Source::poisoned) {
        for (std::uint64_t i = 0; i < result.metadata.tiles.size(); ++i) {
            if (!std::binary_search(tiles.begin(), tiles.end(), i)) {
                const auto &tile = result.metadata.tiles[i];
                std::fill_n(copy.begin() + static_cast<std::ptrdiff_t>(tile.offset), tile.length,
                            0xff);
            }
        }
    }
    result.bytes = upload(copy);
    result.size = copy.size();
    return result;
}

struct Decoded {
    std::vector<HostImage> images;
    std::vector<DeviceResult> results;
};

Decoded decode_on_gpu(const std::vector<cuda::BatchItem> &items, cuda::Operation operation,
                      std::vector<DevicePtr> &outputs, cudaStream_t stream = nullptr) {
    Decoded decoded;
    std::unique_ptr<cuda::BatchPlan> plan;
    EXPECT_EQ(cuda::prepare_batch({operation, items.data(), items.size(), {}}, plan, stream),
              Status::success);
    if (plan == nullptr) {
        return decoded;
    }
    const auto workspace = workspace_for(*plan);
    const auto results = device_alloc(items.size() * sizeof(DeviceResult), 0xff);
    const cuda::ResultBuffer buffer{static_cast<DeviceResult *>(results.get()), items.size()};
    const auto status = operation == cuda::Operation::decompress
                            ? cuda::decompress_async(*plan, workspace.bytes(), buffer, stream)
                            : cuda::decode_roi_async(*plan, workspace.bytes(), buffer, stream);
    EXPECT_EQ(status, Status::success);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    decoded.results = download_results(results, items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        HostImage image;
        image.image = items[i].output.image;
        image.pitch = items[i].output.pitch_bytes;
        image.bytes = download(outputs[i].get(), items[i].output.capacity_bytes);
        decoded.images.push_back(std::move(image));
    }
    return decoded;
}

ImageView device_output(std::vector<DevicePtr> &outputs, ImageSpec image, std::uint32_t padding) {
    const std::uint64_t sample = image.type == PixelType::uint16 ? 2 : 4;
    const auto pitch = (std::uint64_t{image.width} + padding) * sample;
    outputs.push_back(device_alloc(pitch * image.height));
    return {outputs.back().get(), image, pitch, pitch * image.height};
}

void expect_pixels(const HostImage &decoded, const HostImage &expected) {
    EXPECT_EQ(decoded.pixels(), expected.pixels());
    const auto row = decoded.image.width * decoded.sample_bytes();
    for (std::uint64_t y = 0; y < decoded.image.height; ++y) {
        const auto begin = decoded.bytes.begin() + static_cast<std::ptrdiff_t>(y * decoded.pitch);
        ASSERT_TRUE(std::all_of(begin + static_cast<std::ptrdiff_t>(row),
                                begin + static_cast<std::ptrdiff_t>(decoded.pitch),
                                [](auto byte) { return byte == sentinel; }))
            << "row padding " << y;
    }
}

// Decodes CPU archives from parsed metadata and compares with cropped references.
void check_parsed(const std::vector<HostImage> &images, const std::vector<Bytes> &files,
                  const std::vector<Roi> &rois, cuda::Operation operation, Source source) {
    std::vector<Uploaded> archives;
    archives.reserve(images.size());
    std::vector<DevicePtr> outputs;
    std::vector<cuda::BatchItem> items(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        archives.push_back(upload_archive(files[i], rois[i], source));
        auto &archive = archives.back();
        items[i].compressed_input = parsed({archive.bytes.get(), archive.size}, archive.metadata);
        items[i].compressed_input.mappings =
            archive.mappings.empty() ? nullptr : archive.mappings.data();
        items[i].compressed_input.mapping_count = archive.mappings.size();
        items[i].roi = rois[i];
        items[i].output =
            device_output(outputs, {images[i].image.type, rois[i].width, rois[i].height},
                          static_cast<std::uint32_t>(i % 3 * 2));
    }
    const auto decoded = decode_on_gpu(items, operation, outputs);
    ASSERT_EQ(decoded.images.size(), images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        SCOPED_TRACE(::testing::Message()
                     << "image " << i << " roi " << rois[i].x << "," << rois[i].y << " "
                     << rois[i].width << "x" << rois[i].height);
        EXPECT_EQ(decoded.results[i].status, Status::success);
        EXPECT_EQ(decoded.results[i].bytes_written,
                  std::uint64_t{rois[i].width} * rois[i].height * images[i].sample_bytes());
        expect_pixels(decoded.images[i], crop(images[i], rois[i]));
    }
}

Roi full(const HostImage &image) {
    return {0, 0, image.image.width, image.image.height};
}

Roi random_roi(const HostImage &image, std::mt19937 &rng) {
    const auto width = image.image.width;
    const auto height = image.image.height;
    switch (rng() % 5) {
    case 0:
        return full(image);
    case 1:
        return {width - 1, height - 1, 1, 1};
    default: {
        const auto x = static_cast<std::uint32_t>(rng() % width);
        const auto y = static_cast<std::uint32_t>(rng() % height);
        return {x, y, 1 + static_cast<std::uint32_t>(rng() % (width - x)),
                1 + static_cast<std::uint32_t>(rng() % (height - y))};
    }
    }
}

std::vector<HostImage> random_images(PixelType type, std::uint32_t count, std::mt19937 &rng) {
    constexpr std::array<std::uint32_t, 14> sizes = {1,  2,   31,  32,  33,  63,  64,
                                                     65, 127, 128, 129, 255, 256, 257};
    std::vector<HostImage> images;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto width = sizes[rng() % sizes.size()];
        const auto height = sizes[rng() % sizes.size()];
        images.push_back(
            pattern_image(type, width, height, static_cast<std::uint32_t>(rng() % 3), i, rng));
    }
    return images;
}

TEST(CudaDecoder, FullDecodeOfCpuArchives) {
    for (std::uint32_t batch = 0; batch < 12; ++batch) {
        std::mt19937 rng(2026517U + batch);
        const auto type = batch % 2 == 0 ? PixelType::uint16 : PixelType::float32;
        const auto tile = shapes[batch % 3];
        const auto images = random_images(type, 40, rng);
        std::vector<Bytes> files;
        std::vector<Roi> rois;
        for (const auto &image : images) {
            files.push_back(cpu_encode(image, tile));
            rois.push_back(full(image));
        }
        check_parsed(images, files, rois, cuda::Operation::decompress, Source::full);
    }
}

TEST(CudaDecoder, RandomRoisMatchCroppedFullDecode) {
    for (std::uint32_t batch = 0; batch < 24; ++batch) {
        SCOPED_TRACE(::testing::Message() << "batch " << batch);
        std::mt19937 rng(77U + batch);
        const auto type = batch % 2 == 0 ? PixelType::uint16 : PixelType::float32;
        const auto images = random_images(type, 30, rng);
        std::vector<Bytes> files;
        std::vector<Roi> rois;
        for (std::size_t i = 0; i < images.size(); ++i) {
            files.push_back(cpu_encode(images[i], shapes[(batch + i) % 3]));
            rois.push_back(random_roi(images[i], rng));
        }
        check_parsed(images, files, rois, cuda::Operation::decode_roi,
                     static_cast<Source>(batch % 3));
    }
}

TEST(CudaDecoder, RoiReadsOnlyIntersectingTiles) {
    std::mt19937 rng(5);
    const auto image = pattern_image(PixelType::uint16, 1000, 300, 0, 3, rng);
    const auto file = cpu_encode(image);
    const Roi roi{250, 40, 100, 30};
    const auto staged = upload_archive(file, roi, Source::staged);
    EXPECT_EQ(staged.mappings.size(), 4U);
    EXPECT_LT(staged.size, file.size() / 10);
    check_parsed({image}, {file}, {roi}, cuda::Operation::decode_roi, Source::staged);
    check_parsed({image}, {file}, {roi}, cuda::Operation::decode_roi, Source::poisoned);
}

TEST(CudaDecoder, EveryWidthPartialWarpsAndSingleSampleRows) {
    std::mt19937 rng(2026517);
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        const std::uint32_t maximum = type == PixelType::uint16 ? 16 : 32;
        std::vector<HostImage> images;
        for (const auto width : {1U, 2U, 31U, 32U, 33U, 127U, 128U, 129U, 255U, 256U}) {
            for (std::uint32_t bits = 0; bits <= maximum; ++bits) {
                auto image = make_image(type, width, 3, 1);
                const auto mask = bits == 32 ? UINT32_MAX : (std::uint32_t{1} << bits) - 1;
                for (std::uint32_t y = 0; y < 3; ++y) {
                    image.set(0, y,
                              static_cast<std::uint32_t>(rng()) &
                                  (maximum == 16 ? 65535U : UINT32_MAX));
                    for (std::uint32_t x = 1; x < width; ++x) {
                        const auto z = x == 1 ? mask : static_cast<std::uint32_t>(rng()) & mask;
                        const auto previous = image.get(x - 1, y);
                        image.set(x, y,
                                  type == PixelType::uint16
                                      ? (previous + (z % 2 == 0 ? z / 2 : 65535 - z / 2)) & 65535U
                                      : previous ^ z);
                    }
                }
                images.push_back(image);
            }
        }
        for (const auto tile : shapes) {
            std::vector<Bytes> files;
            std::vector<Roi> rois;
            for (const auto &image : images) {
                files.push_back(cpu_encode(image, tile));
                rois.push_back(full(image));
            }
            check_parsed(images, files, rois, cuda::Operation::decompress, Source::full);
        }
    }
}

TEST(CudaDecoder, SpecialFloatWordsAndEveryUint16Value) {
    auto values = make_image(PixelType::uint16, 256, 256);
    for (std::uint32_t i = 0; i < 65536; ++i) {
        values.set(i % 256, i / 256, i);
    }
    constexpr std::array<std::uint32_t, 16> special = {
        0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0x7fc01234, 0x7fa12345,
        0xffc0abcd, 0x3f800000, 0xbf800000, 0x00800000,
    };
    auto floats = make_image(PixelType::float32, 129, 33);
    auto constant = make_image(PixelType::float32, 64, 64);
    for (std::uint32_t y = 0; y < 64; ++y) {
        for (std::uint32_t x = 0; x < 129; ++x) {
            if (y < 33) {
                floats.set(x, y, special[(x / (y % 4 + 1) + y) % special.size()]);
            }
            if (x < 64) {
                constant.set(x, y, 0x7fa12345);
            }
        }
    }
    for (const auto tile : shapes) {
        check_parsed({values}, {cpu_encode(values, tile)}, {full(values)},
                     cuda::Operation::decompress, Source::full);
        check_parsed({floats, constant}, {cpu_encode(floats, tile), cpu_encode(constant, tile)},
                     {full(floats), full(constant)}, cuda::Operation::decompress, Source::full);
    }
}

TEST(CudaDecoder, MixedGeometriesAndWindows) {
    std::mt19937 rng(19);
    std::vector<HostImage> images;
    images.push_back(pattern_image(PixelType::uint16, 1, 32 * 5000, 0, 4, rng));
    images.push_back(pattern_image(PixelType::uint16, 300, 200, 2, 3, rng));
    images.push_back(pattern_image(PixelType::uint16, 129, 32 * 40, 1, 1, rng));
    images.push_back(pattern_image(PixelType::uint16, 1, 64 * 3000, 0, 5, rng));
    std::vector<Bytes> files;
    std::vector<Roi> rois;
    for (std::size_t i = 0; i < images.size(); ++i) {
        files.push_back(cpu_encode(images[i], shapes[i % 3]));
        rois.push_back(full(images[i]));
    }
    check_parsed(images, files, rois, cuda::Operation::decompress, Source::full);
    rois = {
        {0, 100, 1, 32 * 4900}, {17, 3, 280, 190}, {0, 0, 129, 1}, {0, 64 * 1000, 1, 64 * 1999}};
    check_parsed(images, files, rois, cuda::Operation::decode_roi, Source::staged);
}

TEST(CudaDecoder, FourEncoderDecoderCombinations) {
    std::mt19937 rng(31);
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        const auto images = random_images(type, 60, rng);
        const auto archive = gpu_encode(images);
        std::vector<Bytes> cpu_files;
        std::vector<Roi> rois;
        for (std::size_t i = 0; i < images.size(); ++i) {
            cpu_files.push_back(cpu_encode(images[i]));
            rois.push_back(random_roi(images[i], rng));
            // CPU -> CPU and GPU -> CPU.
            EXPECT_EQ(cpu_decode(cpu_files[i], images[i].image).pixels(), images[i].pixels());
            EXPECT_EQ(cpu_decode(archive.files[i], images[i].image).pixels(), images[i].pixels());
        }
        // CPU -> GPU.
        check_parsed(images, cpu_files, rois, cuda::Operation::decode_roi, Source::full);

        // GPU -> GPU without copying the index to the host.
        for (const auto operation : {cuda::Operation::decompress, cuda::Operation::decode_roi}) {
            std::vector<DevicePtr> outputs;
            std::vector<cuda::BatchItem> items(images.size());
            for (std::size_t i = 0; i < images.size(); ++i) {
                const auto roi =
                    operation == cuda::Operation::decompress ? full(images[i]) : rois[i];
                items[i].compressed_input.bytes = {archive.outputs[i].get(), archive.bounds[i]};
                items[i].compressed_input.encoder_result =
                    static_cast<const DeviceResult *>(archive.results.get()) + i;
                items[i].compressed_input.trusted_image = images[i].image;
                items[i].compressed_input.trusted_tile = {};
                items[i].roi = roi;
                items[i].output = device_output(outputs, {type, roi.width, roi.height}, 1);
            }
            const auto decoded = decode_on_gpu(items, operation, outputs);
            ASSERT_EQ(decoded.images.size(), images.size());
            for (std::size_t i = 0; i < images.size(); ++i) {
                SCOPED_TRACE(::testing::Message() << "image " << i);
                EXPECT_EQ(decoded.results[i].status, Status::success);
                expect_pixels(decoded.images[i], crop(images[i], items[i].roi));
            }
        }
    }
}

class TrustedPath : public ::testing::Test {
  protected:
    void SetUp() override {
        std::mt19937 rng(41);
        images = {pattern_image(PixelType::float32, 300, 70, 0, 3, rng),
                  pattern_image(PixelType::float32, 129, 33, 0, 1, rng),
                  pattern_image(PixelType::float32, 64, 64, 0, 1, rng)};
        archive = gpu_encode(images);
        results = device_alloc(3 * sizeof(DeviceResult));
        ASSERT_EQ(cudaMemcpy(results.get(), archive.results.get(), 3 * sizeof(DeviceResult),
                             cudaMemcpyDeviceToDevice),
                  cudaSuccess);
    }

    Decoded run() {
        outputs.clear();
        std::vector<cuda::BatchItem> items(images.size());
        for (std::size_t i = 0; i < images.size(); ++i) {
            items[i].compressed_input.bytes = {archive.outputs[i].get(), archive.bounds[i]};
            items[i].compressed_input.encoder_result =
                static_cast<const DeviceResult *>(results.get()) + i;
            items[i].compressed_input.trusted_image = images[i].image;
            items[i].output = device_output(outputs, images[i].image, 0);
        }
        return decode_on_gpu(items, cuda::Operation::decompress, outputs);
    }

    void corrupt(std::size_t image, std::uint64_t offset, std::uint32_t value) {
        ASSERT_EQ(cudaMemcpy(static_cast<std::uint8_t *>(archive.outputs[image].get()) + offset,
                             &value, 4, cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    std::vector<HostImage> images;
    GpuArchive archive;
    DevicePtr results;
    std::vector<DevicePtr> outputs;
};

TEST_F(TrustedPath, ProducerErrorsArePropagatedPerImage) {
    const DeviceResult failed{Status::buffer_too_small, 0, 0};
    ASSERT_EQ(cudaMemcpy(static_cast<DeviceResult *>(results.get()) + 1, &failed, sizeof(failed),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    const auto decoded = run();
    ASSERT_EQ(decoded.results.size(), 3U);
    EXPECT_EQ(decoded.results[0].status, Status::success);
    EXPECT_EQ(decoded.results[1].status, Status::buffer_too_small);
    EXPECT_EQ(decoded.results[1].bytes_written, 0U);
    EXPECT_EQ(decoded.results[2].status, Status::success);
    expect_pixels(decoded.images[0], images[0]);
    expect_pixels(decoded.images[2], images[2]);
    EXPECT_EQ(decoded.images[1].bytes, Bytes(decoded.images[1].bytes.size(), sentinel));
}

TEST_F(TrustedPath, CorruptIndexAndRowsAreReportedNotRead) {
    ImageMetadata metadata;
    ASSERT_EQ(parse_metadata({archive.files[1].data(), archive.files[1].size()}, {}, metadata),
              Status::success);
    ASSERT_EQ(metadata.tiles[0].mode, TileMode::predict_bitpack);
    const auto entry = [](std::uint64_t tile) { return 64 + tile * 32; };
    // Mode, oversized length, offset past the file, and an impossible row width.
    corrupt(0, entry(1), 7);
    corrupt(1, entry(0) + 16, 1 << 20);
    corrupt(2, entry(0) + 8, 0x7ffffff0);
    auto decoded = run();
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(decoded.results[i].status, Status::malformed_data) << i;
        EXPECT_EQ(decoded.results[i].bytes_written, 0U);
    }
    SetUp();
    corrupt(1, metadata.tiles[0].offset + 4, 33);
    decoded = run();
    EXPECT_EQ(decoded.results[0].status, Status::success);
    EXPECT_EQ(decoded.results[1].status, Status::malformed_data);
    EXPECT_EQ(decoded.results[2].status, Status::success);
}

TEST(CudaDecoder, EmptyBatchIsNoOp) {
    for (const auto operation : {cuda::Operation::decompress, cuda::Operation::decode_roi}) {
        std::unique_ptr<cuda::BatchPlan> plan;
        ASSERT_EQ(cuda::prepare_batch({operation, nullptr, 0, {}}, plan, nullptr), Status::success);
        EXPECT_EQ(operation == cuda::Operation::decompress
                      ? cuda::decompress_async(*plan, {}, {}, nullptr)
                      : cuda::decode_roi_async(*plan, {}, {}, nullptr),
                  Status::success);
        EXPECT_EQ(cuda::compress_async(*plan, {}, {}, nullptr), Status::invalid_argument);
    }
}

class CudaDecoderValidation : public ::testing::Test {
  protected:
    void SetUp() override {
        std::mt19937 rng(23);
        source = pattern_image(PixelType::uint16, 200, 70, 0, 3, rng);
        file = cpu_encode(source);
        ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        bytes = upload(file);
        output = device_alloc(200 * 70 * 2);
        item.compressed_input = parsed({bytes.get(), file.size()}, metadata);
        item.output = {output.get(), source.image, 400, 200 * 70 * 2};
        item.roi = {0, 0, 200, 70};
    }

    Status prepare(const cuda::BatchItem &changed,
                   cuda::Operation operation = cuda::Operation::decode_roi) {
        return cuda::prepare_batch({operation, &changed, 1, {}}, plan, nullptr);
    }

    HostImage source;
    Bytes file;
    ImageMetadata metadata;
    DevicePtr bytes;
    DevicePtr output;
    cuda::BatchItem item;
    std::unique_ptr<cuda::BatchPlan> plan;
};

TEST_F(CudaDecoderValidation, RejectsInvalidRegionsAndViews) {
    ASSERT_EQ(prepare(item), Status::success);
    const auto *previous = plan.get();
    const auto attempt = [&](const cuda::BatchItem &changed, Status status) {
        EXPECT_EQ(prepare(changed), status);
        EXPECT_EQ(plan.get(), previous);
    };
    auto changed = item;
    for (const auto roi :
         {Roi{0, 0, 0, 70}, Roi{0, 0, 200, 0}, Roi{1, 0, 200, 70}, Roi{0, 70, 200, 1},
          Roi{200, 0, 1, 1}, Roi{UINT32_MAX, 0, 2, 1}, Roi{0, UINT32_MAX, 1, 2}}) {
        changed.roi = roi;
        changed.output.image.width = roi.width == 0 ? 1 : roi.width;
        changed.output.image.height = roi.height == 0 ? 1 : roi.height;
        changed.output.pitch_bytes = std::uint64_t{changed.output.image.width} * 2;
        changed.output.capacity_bytes = 200 * 70 * 2;
        attempt(changed, Status::invalid_roi);
    }
    changed = item;
    changed.roi = {10, 10, 20, 20};
    attempt(changed, Status::invalid_argument);
    changed.output.image = {PixelType::uint16, 20, 20};
    changed.output.pitch_bytes = 40;
    EXPECT_EQ(prepare(changed), Status::success);
    previous = plan.get();
    changed.output.image.type = PixelType::float32;
    changed.output.pitch_bytes = 80;
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.output.capacity_bytes = 100;
    attempt(changed, Status::buffer_too_small);
    changed = item;
    changed.output.data = static_cast<std::uint8_t *>(output.get()) + 1;
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.output.data = bytes.get();
    changed.output.capacity_bytes = 200 * 70 * 2;
    attempt(changed, Status::invalid_argument);
    changed = item;
    changed.output.image.width = 199;
    changed.output.pitch_bytes = 398;
    EXPECT_EQ(prepare(changed, cuda::Operation::decompress), Status::invalid_argument);
}

TEST_F(CudaDecoderValidation, RejectsInconsistentSources) {
    ASSERT_EQ(prepare(item), Status::success);
    auto changed = item;
    changed.compressed_input.bytes.size = file.size() - 1;
    EXPECT_EQ(prepare(changed), Status::buffer_too_small);
    changed = item;
    changed.compressed_input.bytes.data = static_cast<std::uint8_t *>(bytes.get()) + 2;
    EXPECT_EQ(prepare(changed), Status::invalid_argument);
    changed = item;
    changed.compressed_input.metadata = nullptr;
    EXPECT_EQ(prepare(changed), Status::invalid_argument);
    const DeviceResult *fake = static_cast<const DeviceResult *>(output.get());
    changed = item;
    changed.compressed_input.encoder_result = fake;
    EXPECT_EQ(prepare(changed), Status::invalid_argument);

    // Mappings must list exactly the intersecting tiles, aligned and without overlap.
    changed = item;
    changed.roi = {130, 0, 10, 40};
    changed.output.image = {PixelType::uint16, 10, 40};
    changed.output.pitch_bytes = 20;
    std::vector<cuda::TileMapping> mappings = {{1, metadata.tiles[1].offset},
                                               {3, metadata.tiles[3].offset}};
    changed.compressed_input.mappings = mappings.data();
    changed.compressed_input.mapping_count = 2;
    EXPECT_EQ(prepare(changed), Status::success);
    for (const auto &bad : {std::vector<cuda::TileMapping>{{1, metadata.tiles[1].offset}},
                            std::vector<cuda::TileMapping>{{3, 0}, {1, 0}},
                            std::vector<cuda::TileMapping>{{1, 2}, {3, 1024}},
                            std::vector<cuda::TileMapping>{{1, 0}, {3, 4}},
                            std::vector<cuda::TileMapping>{{1, 0}, {3, file.size()}}}) {
        changed.compressed_input.mappings = bad.data();
        changed.compressed_input.mapping_count = bad.size();
        EXPECT_EQ(prepare(changed), Status::invalid_argument);
    }
    changed.compressed_input.mappings = nullptr;
    EXPECT_EQ(prepare(changed), Status::invalid_argument);

    // Fabricated metadata that would send device reads outside a payload.
    for (int mutation = 0; mutation < 5; ++mutation) {
        auto fabricated = metadata;
        auto &tile = fabricated.tiles[0];
        ASSERT_EQ(tile.mode, TileMode::predict_bitpack);
        switch (mutation) {
        case 0:
            tile.length = 1 << 16;
            break;
        case 1:
            tile.first_segment = fabricated.segments.size();
            break;
        case 2:
            fabricated.segments[3].payload_bytes += 4;
            break;
        case 3:
            fabricated.segments[0].bit_width = 17;
            break;
        default:
            fabricated.tiles.pop_back();
            break;
        }
        changed = item;
        changed.compressed_input.metadata = &fabricated;
        EXPECT_EQ(prepare(changed), Status::malformed_data) << mutation;
    }

    std::array<cuda::BatchItem, 2> pair = {item, item};
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::decode_roi, pair.data(), 2, {}}, plan, nullptr),
              Status::invalid_argument);
    const auto second = device_alloc(200 * 70 * 2);
    pair[1].output.data = second.get();
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::decode_roi, pair.data(), 2, {}}, plan, nullptr),
              Status::success);
    std::mt19937 rng(1);
    const auto floats = cpu_encode(pattern_image(PixelType::float32, 5, 5, 0, 1, rng));
    ImageMetadata float_metadata;
    ASSERT_EQ(parse_metadata({floats.data(), floats.size()}, {}, float_metadata), Status::success);
    const auto float_bytes = upload(floats);
    pair[1].compressed_input = parsed({float_bytes.get(), floats.size()}, float_metadata);
    pair[1].output = {second.get(), {PixelType::float32, 5, 5}, 20, 100};
    pair[1].roi = {0, 0, 5, 5};
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::decode_roi, pair.data(), 2, {}}, plan, nullptr),
              Status::invalid_argument);
}

TEST_F(CudaDecoderValidation, TrustedSourcesNeedCompleteShapes) {
    const auto encoded = device_alloc(file.size());
    const auto result = device_alloc(sizeof(DeviceResult));
    auto changed = item;
    changed.compressed_input = {};
    changed.compressed_input.bytes = {encoded.get(), file.size()};
    changed.compressed_input.encoder_result = static_cast<const DeviceResult *>(result.get());
    changed.compressed_input.trusted_image = source.image;
    std::uint64_t bound = 0;
    ASSERT_EQ(compress_bound(source.image, {}, bound), Status::success);
    EXPECT_EQ(prepare(changed), Status::buffer_too_small);
    const auto large = device_alloc(bound);
    changed.compressed_input.bytes = {large.get(), bound};
    EXPECT_EQ(prepare(changed), Status::success);
    changed.compressed_input.trusted_tile = {16, 16};
    EXPECT_EQ(prepare(changed), Status::invalid_argument);
    changed.compressed_input.trusted_tile = {};
    changed.compressed_input.trusted_image.type = static_cast<PixelType>(5);
    EXPECT_EQ(prepare(changed), Status::unsupported_type);
    changed.compressed_input.trusted_image = source.image;
    changed.compressed_input.encoder_result =
        reinterpret_cast<const DeviceResult *>(static_cast<std::uint8_t *>(result.get()) + 4);
    EXPECT_EQ(prepare(changed), Status::invalid_argument);
    changed.compressed_input.encoder_result = static_cast<const DeviceResult *>(result.get());
    changed.compressed_input.metadata = &metadata;
    EXPECT_EQ(prepare(changed), Status::invalid_argument);

    // The consuming result buffer must not alias the producer's result.
    changed.compressed_input.metadata = nullptr;
    ASSERT_EQ(prepare(changed), Status::success);
    const auto workspace = workspace_for(*plan);
    EXPECT_EQ(cuda::decode_roi_async(*plan, workspace.bytes(),
                                     {static_cast<DeviceResult *>(result.get()), 1}, nullptr),
              Status::invalid_argument);
}

} // namespace
