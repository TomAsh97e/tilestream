#include "cuda_test_support.h"

#include <cmath>

namespace {

using namespace test;

using Execute = Status (*)(const cuda::BatchPlan &, MutableBytes, cuda::ResultBuffer,
                           cudaStream_t) noexcept;

float as_float(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, 4);
    return value;
}

std::uint32_t bits_of(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return bits;
}

// Two roundings, like the device; (x - offset) * gain has no form that could become an FMA.
float reference(float x, float offset, float gain) {
    const float difference = x - offset;
    return difference * gain;
}

float sample(const HostImage &image, std::uint64_t x, std::uint64_t y) {
    const auto bits = image.get(x, y);
    return image.image.type == PixelType::uint16 ? static_cast<float>(bits) : as_float(bits);
}

struct HostParameter {
    float scalar = 0.0F;
    bool mapped = false;
    HostImage map;
    DevicePtr device;

    float at(std::uint64_t x, std::uint64_t y) const {
        return mapped ? as_float(map.get(x, y)) : scalar;
    }
    cuda::Parameter parameter() const {
        if (!mapped) {
            return {cuda::ParameterKind::scalar, scalar, {}};
        }
        return {
            cuda::ParameterKind::map, 0.0F, {device.get(), map.image, map.pitch, map.bytes.size()}};
    }
};

struct HostCalibration {
    HostParameter offset;
    HostParameter gain;
    cuda::Calibration view() const {
        return {offset.parameter(), gain.parameter()};
    }
};

// Detector-like values with occasional zeros and arbitrary words, which include specials.
float random_value(std::mt19937 &rng, float low, float high) {
    switch (rng() % 16) {
    case 0:
        return as_float(static_cast<std::uint32_t>(rng()));
    case 1:
        return 0.0F;
    case 2:
        return -0.0F;
    default:
        return low + (high - low) * static_cast<float>(rng() % 100000) / 100000.0F;
    }
}

HostImage float_map(ImageSpec region, std::uint32_t padding) {
    return make_image(PixelType::float32, region.width, region.height, padding);
}

HostParameter map_parameter(HostImage map) {
    HostParameter result;
    result.mapped = true;
    result.device = upload(map.bytes);
    result.map = std::move(map);
    return result;
}

HostParameter random_parameter(bool mapped, ImageSpec region, float low, float high,
                               std::mt19937 &rng) {
    if (!mapped) {
        HostParameter result;
        result.scalar = low + (high - low) * static_cast<float>(rng() % 1000) / 1000.0F;
        return result;
    }
    auto map = float_map(region, static_cast<std::uint32_t>(rng() % 3));
    for (std::uint32_t y = 0; y < region.height; ++y) {
        for (std::uint32_t x = 0; x < region.width; ++x) {
            map.set(x, y, bits_of(random_value(rng, low, high)));
        }
    }
    return map_parameter(std::move(map));
}

// Bit 0 selects an offset map and bit 1 a gain map.
HostCalibration random_calibration(std::uint32_t kinds, ImageSpec region, std::mt19937 &rng) {
    return {random_parameter((kinds & 1) != 0, region, -100.0F, 2000.0F, rng),
            random_parameter((kinds & 2) != 0, region, 0.25F, 4.0F, rng)};
}

HostImage expected(const HostImage &source, const Roi &roi, const HostCalibration &calibration) {
    auto result = make_image(PixelType::float32, roi.width, roi.height, 0, 0);
    for (std::uint32_t y = 0; y < roi.height; ++y) {
        for (std::uint32_t x = 0; x < roi.width; ++x) {
            const auto value = reference(sample(source, roi.x + x, roi.y + y),
                                         calibration.offset.at(x, y), calibration.gain.at(x, y));
            result.set(x, y, bits_of(value));
        }
    }
    return result;
}

// Other results match bit for bit; NaN payloads are not part of the semantics.
void expect_calibrated(const HostImage &actual, const HostImage &wanted, const std::string &label) {
    std::uint64_t mismatches = 0;
    for (std::uint32_t y = 0; y < wanted.image.height; ++y) {
        for (std::uint32_t x = 0; x < wanted.image.width; ++x) {
            const auto a = actual.get(x, y);
            const auto w = wanted.get(x, y);
            if (a != w && !(std::isnan(as_float(a)) && std::isnan(as_float(w))) &&
                mismatches++ == 0) {
                ADD_FAILURE() << label << ": first mismatch at " << x << ", " << y << ": "
                              << std::hex << a << " instead of " << w;
            }
        }
    }
    EXPECT_EQ(mismatches, 0U) << label;
}

struct Output {
    HostImage image;
    DevicePtr device;

    ImageView view() const {
        return {device.get(), image.image, image.pitch, image.bytes.size()};
    }
    // Checks that the padding still holds the sentinel.
    HostImage fetch() const {
        auto result = image;
        result.bytes = download(device.get(), image.bytes.size());
        const auto row = result.image.width * result.sample_bytes();
        for (std::uint64_t y = 0; y < result.image.height; ++y) {
            const auto *line = result.bytes.data() + y * result.pitch;
            EXPECT_TRUE(std::all_of(line + row, line + result.pitch,
                                    [](auto byte) { return byte == sentinel; }))
                << "padding of row " << y;
        }
        return result;
    }
};

Output make_output(ImageSpec region, std::uint32_t padding) {
    Output result;
    result.image = make_image(region.type, region.width, region.height, padding, sentinel);
    result.device = upload(result.image.bytes);
    return result;
}

std::vector<DeviceResult> execute(cuda::Operation operation,
                                  const std::vector<cuda::BatchItem> &items, Execute run) {
    std::unique_ptr<cuda::BatchPlan> plan;
    EXPECT_EQ(cuda::prepare_batch({operation, items.data(), items.size(), {}}, plan, nullptr),
              Status::success);
    if (plan == nullptr) {
        return {};
    }
    const auto workspace = workspace_for(*plan);
    const auto results = device_alloc(items.size() * sizeof(DeviceResult), 0xff);
    EXPECT_EQ(run(*plan, workspace.bytes(),
                  {static_cast<DeviceResult *>(results.get()), items.size()}, nullptr),
              Status::success);
    EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    return download_results(results, items.size());
}

void expect_results(const std::vector<DeviceResult> &records, const std::vector<Roi> &regions) {
    ASSERT_EQ(records.size(), regions.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        EXPECT_EQ(records[i].status, Status::success) << "image " << i;
        EXPECT_EQ(records[i].reserved, 0U);
        EXPECT_EQ(records[i].bytes_written,
                  std::uint64_t{regions[i].width} * regions[i].height * 4);
    }
}

Roi random_roi(const ImageSpec &image, std::mt19937 &rng) {
    if (rng() % 4 == 0) {
        return {0, 0, image.width, image.height};
    }
    const auto x = static_cast<std::uint32_t>(rng() % image.width);
    const auto y = static_cast<std::uint32_t>(rng() % image.height);
    return {x, y, 1 + static_cast<std::uint32_t>(rng() % (image.width - x)),
            1 + static_cast<std::uint32_t>(rng() % (image.height - y))};
}

ConstImageView input_view(const HostImage &image, const DevicePtr &device) {
    return {device.get(), image.image, image.pitch, image.bytes.size()};
}

TEST(CudaCalibration, StandaloneMatchesCpuReference) {
    std::mt19937 rng(11);
    const std::uint32_t sizes[] = {1, 2, 31, 32, 33, 127, 300};
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        for (std::uint32_t kinds = 0; kinds < 4; ++kinds) {
            std::vector<HostImage> images;
            std::vector<DevicePtr> inputs;
            std::vector<HostCalibration> calibrations;
            std::vector<Output> outputs;
            std::vector<Roi> regions;
            std::vector<cuda::BatchItem> items(12);
            for (std::uint32_t i = 0; i < items.size(); ++i) {
                const auto width = sizes[rng() % std::size(sizes)];
                const auto height = sizes[rng() % std::size(sizes)];
                const auto padding = static_cast<std::uint32_t>(rng() % 3);
                images.push_back(pattern_image(type, width, height, padding, i, rng));
                const ImageSpec region{PixelType::float32, width, height};
                inputs.push_back(upload(images[i].bytes));
                calibrations.push_back(random_calibration(kinds, region, rng));
                outputs.push_back(make_output(region, static_cast<std::uint32_t>(rng() % 3)));
                regions.push_back({0, 0, width, height});
                items[i].input = input_view(images[i], inputs[i]);
                items[i].output = outputs[i].view();
                items[i].calibration = calibrations[i].view();
            }
            expect_results(execute(cuda::Operation::calibrate, items, cuda::calibrate_async),
                           regions);
            for (std::size_t i = 0; i < items.size(); ++i) {
                expect_calibrated(outputs[i].fetch(),
                                  expected(images[i], regions[i], calibrations[i]),
                                  "kinds " + std::to_string(kinds) + " image " + std::to_string(i));
            }
        }
    }
}

// Compares fused decode and calibration with a decoded ROI that is calibrated afterwards.
void expect_fused_matches_separate(PixelType type, bool trusted, std::mt19937 &rng) {
    const std::uint32_t sizes[] = {1, 7, 33, 128, 129, 300, 513};
    const auto shared = shapes[rng() % std::size(shapes)];
    std::vector<HostImage> images;
    for (std::uint32_t i = 0; i < 10; ++i) {
        const auto width = sizes[rng() % std::size(sizes)];
        const auto height = sizes[rng() % std::size(sizes)];
        images.push_back(pattern_image(type, width, height, 0, i, rng));
    }
    GpuArchive archive;
    if (trusted) {
        archive = gpu_encode(images, shared);
    }
    std::vector<Bytes> files;
    std::vector<ImageMetadata> metadata(images.size());
    std::vector<DevicePtr> sources;
    std::vector<HostCalibration> calibrations;
    std::vector<Roi> regions;
    std::vector<Output> decoded;
    std::vector<Output> separate;
    std::vector<Output> fused;
    std::vector<cuda::BatchItem> decode(images.size());
    std::vector<cuda::BatchItem> calibrate(images.size());
    std::vector<cuda::BatchItem> combined(images.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
        auto &source = decode[i].compressed_input;
        if (trusted) {
            files.push_back(archive.files[i]);
            source.bytes = {archive.outputs[i].get(), archive.bounds[i]};
            source.encoder_result = static_cast<const DeviceResult *>(archive.results.get()) + i;
            source.trusted_image = images[i].image;
            source.trusted_tile = shared;
        } else {
            files.push_back(cpu_encode(images[i], shapes[rng() % std::size(shapes)]));
            ASSERT_EQ(parse_metadata({files[i].data(), files[i].size()}, {}, metadata[i]),
                      Status::success);
            sources.push_back(upload(files[i]));
            source.bytes = {sources.back().get(), files[i].size()};
            source.metadata = &metadata[i];
        }
        const auto roi = random_roi(images[i].image, rng);
        const ImageSpec region{PixelType::float32, roi.width, roi.height};
        regions.push_back(roi);
        calibrations.push_back(
            random_calibration(static_cast<std::uint32_t>(rng() % 4), region, rng));
        decoded.push_back(make_output({type, roi.width, roi.height}, 1));
        separate.push_back(make_output(region, 0));
        fused.push_back(make_output(region, 2));

        decode[i].roi = roi;
        decode[i].output = decoded[i].view();
        const auto &ready = decoded[i].image;
        calibrate[i].input = {decoded[i].device.get(), ready.image, ready.pitch,
                              ready.bytes.size()};
        calibrate[i].output = separate[i].view();
        calibrate[i].calibration = calibrations[i].view();
        combined[i].compressed_input = source;
        combined[i].roi = roi;
        combined[i].output = fused[i].view();
        combined[i].calibration = calibrations[i].view();
    }

    const auto label = std::string(type == PixelType::uint16 ? "uint16" : "float32") +
                       (trusted ? " trusted" : " parsed");
    for (const auto &record :
         execute(cuda::Operation::decode_roi, decode, cuda::decode_roi_async)) {
        EXPECT_EQ(record.status, Status::success) << label;
    }
    expect_results(execute(cuda::Operation::calibrate, calibrate, cuda::calibrate_async), regions);
    expect_results(
        execute(cuda::Operation::decode_calibrate, combined, cuda::decode_calibrate_async),
        regions);
    for (std::size_t i = 0; i < images.size(); ++i) {
        const auto one = separate[i].fetch();
        const auto two = fused[i].fetch();
        // Both paths run the same device operations on the same bits, NaNs included.
        EXPECT_EQ(one.pixels(), two.pixels()) << label << " image " << i;
        expect_calibrated(
            two, expected(cpu_decode(files[i], images[i].image), regions[i], calibrations[i]),
            label + " image " + std::to_string(i));
    }
}

TEST(CudaCalibration, FusedMatchesSeparateDecodeAndCalibrate) {
    std::mt19937 rng(12);
    for (const auto type : {PixelType::uint16, PixelType::float32}) {
        for (int round = 0; round < 3; ++round) {
            expect_fused_matches_separate(type, false, rng);
        }
        expect_fused_matches_separate(type, true, rng);
    }
}

// Every pairing of special input, offset and gain words, through both paths.
TEST(CudaCalibration, SpecialWordsMatchCpu) {
    const std::uint32_t words[] = {
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000, 0x80800000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7f800001,
        0x3f800000, 0x3f800001, 0xbf800000, 0x4b800000, 0x33800000, 0x0c000000, 0x72000000};
    const auto count = static_cast<std::uint32_t>(std::size(words));
    const ImageSpec region{PixelType::float32, count, count * count};
    auto image = make_image(PixelType::float32, region.width, region.height, 1);
    auto offsets = float_map(region, 0);
    auto gains = float_map(region, 3);
    for (std::uint32_t y = 0; y < region.height; ++y) {
        for (std::uint32_t x = 0; x < count; ++x) {
            image.set(x, y, words[x]);
            offsets.set(x, y, words[y % count]);
            gains.set(x, y, words[y / count]);
        }
    }
    const HostCalibration calibration{map_parameter(std::move(offsets)),
                                      map_parameter(std::move(gains))};
    const Roi full{0, 0, region.width, region.height};
    const auto wanted = expected(image, full, calibration);

    const auto input = upload(image.bytes);
    const auto separate = make_output(region, 1);
    std::vector<cuda::BatchItem> items(1);
    items[0].input = input_view(image, input);
    items[0].output = separate.view();
    items[0].calibration = calibration.view();
    expect_results(execute(cuda::Operation::calibrate, items, cuda::calibrate_async), {full});
    const auto one = separate.fetch();
    expect_calibrated(one, wanted, "standalone");

    const auto file = cpu_encode(image);
    ImageMetadata metadata;
    ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
    const auto source = upload(file);
    const auto fused = make_output(region, 0);
    items[0] = {};
    items[0].compressed_input.bytes = {source.get(), file.size()};
    items[0].compressed_input.metadata = &metadata;
    items[0].roi = full;
    items[0].output = fused.view();
    items[0].calibration = calibration.view();
    expect_results(execute(cuda::Operation::decode_calibrate, items, cuda::decode_calibrate_async),
                   {full});
    const auto two = fused.fetch();
    expect_calibrated(two, wanted, "fused");
    EXPECT_EQ(one.pixels(), two.pixels());
}

TEST(CudaCalibration, EveryUint16Value) {
    std::mt19937 rng(13);
    auto image = make_image(PixelType::uint16, 256, 256, 3);
    for (std::uint32_t value = 0; value < 65536; ++value) {
        image.set(value % 256, value / 256, value);
    }
    const ImageSpec region{PixelType::float32, 256, 256};
    const auto calibration = random_calibration(3, region, rng);
    const auto input = upload(image.bytes);
    const auto output = make_output(region, 0);
    std::vector<cuda::BatchItem> items(1);
    items[0].input = input_view(image, input);
    items[0].output = output.view();
    items[0].calibration = calibration.view();
    const Roi full{0, 0, 256, 256};
    expect_results(execute(cuda::Operation::calibrate, items, cuda::calibrate_async), {full});
    expect_calibrated(output.fetch(), expected(image, full, calibration), "uint16");
}

TEST(CudaCalibration, EmptyBatchIsNoOp) {
    for (const auto operation : {cuda::Operation::calibrate, cuda::Operation::decode_calibrate}) {
        std::unique_ptr<cuda::BatchPlan> plan;
        ASSERT_EQ(cuda::prepare_batch({operation, nullptr, 0, {}}, plan, nullptr), Status::success);
        const auto run = operation == cuda::Operation::calibrate ? cuda::calibrate_async
                                                                 : cuda::decode_calibrate_async;
        EXPECT_EQ(run(*plan, {}, {}, nullptr), Status::success);
    }
}

class CudaCalibrationValidation : public ::testing::Test {
  protected:
    void SetUp() override {
        std::mt19937 rng(14);
        image = pattern_image(PixelType::uint16, 64, 40, 2, 1, rng);
        input = upload(image.bytes);
        output = make_output(region, 1);
        offset = random_parameter(true, region, 0.0F, 10.0F, rng);
        item.input = input_view(image, input);
        item.output = output.view();
        item.calibration.offset = offset.parameter();

        file = cpu_encode(image);
        ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        source = upload(file);
        roi_output = make_output(roi_region, 0);
        roi_offset = random_parameter(true, roi_region, 0.0F, 10.0F, rng);
        fused.compressed_input.bytes = {source.get(), file.size()};
        fused.compressed_input.metadata = &metadata;
        fused.roi = {5, 6, 30, 20};
        fused.output = roi_output.view();
        fused.calibration.offset = roi_offset.parameter();
    }

    static void attempt(cuda::Operation operation, const cuda::BatchItem &changed, Status status) {
        std::unique_ptr<cuda::BatchPlan> plan;
        EXPECT_EQ(cuda::prepare_batch({operation, &changed, 1, {}}, plan, nullptr), status);
        EXPECT_EQ(plan == nullptr, status != Status::success);
    }

    const ImageSpec region{PixelType::float32, 64, 40};
    const ImageSpec roi_region{PixelType::float32, 30, 20};
    HostImage image;
    DevicePtr input;
    Output output;
    HostParameter offset;
    cuda::BatchItem item;
    Bytes file;
    ImageMetadata metadata;
    DevicePtr source;
    Output roi_output;
    HostParameter roi_offset;
    cuda::BatchItem fused;
};

TEST_F(CudaCalibrationValidation, RejectsInvalidOutputsAndMaps) {
    const auto calibrate = cuda::Operation::calibrate;
    attempt(calibrate, item, Status::success);

    auto changed = item;
    changed.output.image.type = PixelType::uint16;
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    changed.output.image.width = 63;
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    changed.output.data = nullptr;
    attempt(calibrate, changed, Status::invalid_argument);

    auto &map = changed.calibration.offset.map;
    changed = item;
    map.image.width = 63;
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    map.image.type = PixelType::uint16;
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    map.image.type = static_cast<PixelType>(7);
    attempt(calibrate, changed, Status::unsupported_type);
    changed = item;
    map.pitch_bytes = 64 * 4 - 4;
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    map.pitch_bytes = 64 * 4 + 2;
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    map.capacity_bytes = 64 * 4 * 39;
    attempt(calibrate, changed, Status::buffer_too_small);
    changed = item;
    map.data = nullptr;
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    changed.calibration.gain.kind = static_cast<cuda::ParameterKind>(5);
    attempt(calibrate, changed, Status::invalid_argument);
    changed = item;
    changed.calibration.gain.kind = cuda::ParameterKind::map;
    attempt(calibrate, changed, Status::invalid_argument);

    // Calibration is never in place, and outputs cannot overlap maps or each other.
    changed = item;
    changed.output.data = const_cast<void *>(map.data);
    changed.output.pitch_bytes = map.pitch_bytes;
    changed.output.capacity_bytes = map.capacity_bytes;
    attempt(calibrate, changed, Status::invalid_argument);
    std::vector<float> host(64 * 40);
    int pageable = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0), cudaSuccess);
    if (pageable == 0) {
        changed = item;
        map.data = host.data();
        map.pitch_bytes = 64 * 4;
        map.capacity_bytes = host.size() * 4;
        attempt(calibrate, changed, Status::invalid_argument);
    }
}

TEST_F(CudaCalibrationValidation, RejectsInconsistentBatches) {
    std::mt19937 rng(15);
    const auto other = pattern_image(PixelType::float32, 64, 40, 0, 1, rng);
    const auto other_input = upload(other.bytes);
    const auto other_output = make_output(region, 0);
    std::vector<cuda::BatchItem> pair = {item, item};
    pair[1].input = input_view(other, other_input);
    pair[1].output = other_output.view();
    std::unique_ptr<cuda::BatchPlan> plan;
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::calibrate, pair.data(), 2, {}}, plan, nullptr),
              Status::invalid_argument);
    pair[1] = item;
    EXPECT_EQ(cuda::prepare_batch({cuda::Operation::calibrate, pair.data(), 2, {}}, plan, nullptr),
              Status::invalid_argument);

    // A float32 image may not be calibrated into its own buffer either.
    auto in_place = item;
    in_place.input = input_view(other, other_input);
    in_place.output = {const_cast<void *>(other_input.get()), region, other.pitch,
                       other.bytes.size()};
    attempt(cuda::Operation::calibrate, in_place, Status::invalid_argument);
    in_place.input.image.type = static_cast<PixelType>(9);
    attempt(cuda::Operation::calibrate, in_place, Status::unsupported_type);
    EXPECT_EQ(plan, nullptr);
}

TEST_F(CudaCalibrationValidation, FusedRegionsAndOutputs) {
    const auto operation = cuda::Operation::decode_calibrate;
    attempt(operation, fused, Status::success);
    auto changed = fused;
    changed.roi = {5, 6, 0, 20};
    attempt(operation, changed, Status::invalid_roi);
    changed.roi = {40, 6, 30, 20};
    attempt(operation, changed, Status::invalid_roi);
    changed = fused;
    changed.output.image.type = PixelType::uint16;
    attempt(operation, changed, Status::invalid_argument);
    // Maps follow the region, not the full image.
    changed = fused;
    changed.calibration.offset = offset.parameter();
    attempt(operation, changed, Status::invalid_argument);
    changed = fused;
    changed.output = output.view();
    changed.calibration = {};
    attempt(operation, changed, Status::invalid_argument);
}

TEST_F(CudaCalibrationValidation, ExecutionNeedsMatchingPlanAndNoWorkspace) {
    std::unique_ptr<cuda::BatchPlan> standalone;
    std::unique_ptr<cuda::BatchPlan> combined;
    ASSERT_EQ(cuda::prepare_batch({cuda::Operation::calibrate, &item, 1, {}}, standalone, nullptr),
              Status::success);
    ASSERT_EQ(
        cuda::prepare_batch({cuda::Operation::decode_calibrate, &fused, 1, {}}, combined, nullptr),
        Status::success);
    std::uint64_t bytes = 1;
    ASSERT_EQ(workspace_size(standalone->workspace_spec(), bytes), Status::success);
    EXPECT_EQ(bytes, 0U);

    const auto results = device_alloc(sizeof(DeviceResult), 0xff);
    const cuda::ResultBuffer records{static_cast<DeviceResult *>(results.get()), 1};
    const auto workspace = workspace_for(*combined);
    EXPECT_EQ(cuda::decode_calibrate_async(*standalone, {}, records, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(cuda::calibrate_async(*combined, workspace.bytes(), records, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(cuda::decode_roi_async(*combined, workspace.bytes(), records, nullptr),
              Status::invalid_argument);
    EXPECT_EQ(cuda::decode_calibrate_async(*combined, {workspace.data.get(), workspace.size - 1},
                                           records, nullptr),
              Status::buffer_too_small);
    EXPECT_EQ(cuda::calibrate_async(*standalone, {}, records, nullptr), Status::success);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    expect_results(download_results(results, 1), {{0, 0, 64, 40}});
}

} // namespace
