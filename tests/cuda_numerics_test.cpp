// Calibration error suite on the device: the library output against the host semantics, the
// special-value contract and the bounds fixed in docs/numerical-analysis.md.
#include "cuda_test_support.h"
#include "numerics/support.h"

namespace {

using namespace test;
using numerics::as_float;
using numerics::bits_of;
using numerics::Case;

struct Maps {
    HostImage offset;
    HostImage gain;
    DevicePtr offset_data;
    DevicePtr gain_data;

    cuda::Calibration calibration() const {
        const auto view = [](const HostImage &map, const DevicePtr &data) {
            return ConstImageView{data.get(), map.image, map.pitch, map.bytes.size()};
        };
        return {{cuda::ParameterKind::map, 0.0F, view(offset, offset_data)},
                {cuda::ParameterKind::map, 0.0F, view(gain, gain_data)}};
    }
};

Maps upload_maps(const Case &c) {
    Maps result;
    result.offset = make_image(PixelType::float32, c.width, c.height, 1);
    result.gain = make_image(PixelType::float32, c.width, c.height, 2);
    for (std::uint64_t i = 0; i < c.size(); ++i) {
        result.offset.set(i % c.width, i / c.width, bits_of(c.offset[i]));
        result.gain.set(i % c.width, i / c.width, bits_of(c.gain[i]));
    }
    result.offset_data = upload(result.offset.bytes);
    result.gain_data = upload(result.gain.bytes);
    return result;
}

HostImage input_image(const Case &c) {
    auto image =
        make_image(c.integer ? PixelType::uint16 : PixelType::float32, c.width, c.height, 3);
    for (std::uint64_t i = 0; i < c.size(); ++i) {
        image.set(i % c.width, i / c.width, c.integer ? c.samples[i] : bits_of(c.x[i]));
    }
    return image;
}

std::vector<float> execute(const std::vector<cuda::BatchItem> &items, cuda::Operation operation,
                           const std::vector<DevicePtr> &outputs,
                           const std::vector<std::uint64_t> &sizes) {
    std::unique_ptr<cuda::BatchPlan> plan;
    EXPECT_EQ(cuda::prepare_batch({operation, items.data(), items.size(), {}}, plan, nullptr),
              Status::success);
    if (plan == nullptr) {
        return {};
    }
    const auto workspace = workspace_for(*plan);
    const auto results = device_alloc(items.size() * sizeof(DeviceResult), 0xff);
    const auto run = operation == cuda::Operation::calibrate ? cuda::calibrate_async
                                                             : cuda::decode_calibrate_async;
    EXPECT_EQ(run(*plan, workspace.bytes(),
                  {static_cast<DeviceResult *>(results.get()), items.size()}, nullptr),
              Status::success);
    EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    for (const auto &record : download_results(results, items.size())) {
        EXPECT_EQ(record.status, Status::success);
    }
    std::vector<float> values;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto bytes = download(outputs[i].get(), sizes[i]);
        const auto offset = values.size();
        values.resize(offset + bytes.size() / 4);
        std::memcpy(values.data() + offset, bytes.data(), bytes.size());
    }
    return values;
}

// Calibrates one case with offset and gain maps through the standalone or the fused operation.
std::vector<float> library(const Case &c, bool fused) {
    const auto image = input_image(c);
    const auto maps = upload_maps(c);
    const ImageSpec region{PixelType::float32, c.width, c.height};
    std::vector<DevicePtr> outputs;
    outputs.push_back(device_alloc(c.size() * 4));
    cuda::BatchItem item;
    item.output = {outputs[0].get(), region, c.width * 4ULL, c.size() * 4};
    item.calibration = maps.calibration();

    DevicePtr data;
    Bytes file;
    ImageMetadata metadata;
    if (fused) {
        file = cpu_encode(image);
        EXPECT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
        data = upload(file);
        item.compressed_input.bytes = {data.get(), file.size()};
        item.compressed_input.metadata = &metadata;
        item.roi = {0, 0, c.width, c.height};
    } else {
        data = upload(image.bytes);
        item.input = {data.get(), image.image, image.pitch, image.bytes.size()};
    }
    return execute({item}, fused ? cuda::Operation::decode_calibrate : cuda::Operation::calibrate,
                   outputs, {c.size() * 4});
}

std::vector<Case> cases() {
    auto result = numerics::synthetic_cases(1 << 16, 2026);
    result.push_back(numerics::special_case());
    return result;
}

TEST(CudaNumerics, LibraryMatchesHostSemanticsBitwise) {
    RecordProperty("cxx_flags", TILESTREAM_NUMERICS_CXX_FLAGS);
    RecordProperty("cuda_flags", TILESTREAM_NUMERICS_CUDA_FLAGS);
    for (const auto &c : cases()) {
        const auto separate = library(c, false);
        const auto fused = library(c, true);
        ASSERT_EQ(separate.size(), c.size());
        ASSERT_EQ(fused.size(), c.size());
        // One device runs the same operations on the same bits in both paths, NaNs included.
        EXPECT_EQ(std::memcmp(separate.data(), fused.data(), c.size() * 4), 0) << c.name;
        std::uint64_t mismatches = 0;
        for (std::uint64_t i = 0; i < c.size(); ++i) {
            const auto host = numerics::calibrate(c.x[i], c.offset[i], c.gain[i]);
            if (!numerics::same(separate[i], host) && mismatches++ < 5) {
                ADD_FAILURE() << c.name << " sample " << i << std::hex << ": device "
                              << bits_of(separate[i]) << ", host " << bits_of(host);
            }
        }
        EXPECT_EQ(mismatches, 0U) << c.name;
    }
}

TEST(CudaNumerics, DeviceResultsWithinBounds) {
    for (const auto &c : cases()) {
        const auto values = library(c, false);
        ASSERT_EQ(values.size(), c.size());
        std::uint64_t failures = 0;
        for (std::uint64_t i = 0; i < c.size(); ++i) {
            const auto problem = numerics::violation(c.x[i], c.offset[i], c.gain[i], values[i]);
            if (!problem.empty() && failures++ < 5) {
                ADD_FAILURE() << c.name << " sample " << i << ": " << problem;
            }
        }
        EXPECT_EQ(failures, 0U) << c.name;
    }
}

void expect_rows(const std::vector<float> &values, const std::vector<numerics::Row> &rows,
                 const std::string &path) {
    ASSERT_EQ(values.size(), rows.size()) << path;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        EXPECT_TRUE(numerics::same(values[i], as_float(rows[i].expected)))
            << path << ", " << rows[i].note << ": " << std::hex << bits_of(values[i]);
    }
}

// Every contract row through maps, scalars, uint16 samples and the fused operation.
TEST(CudaNumerics, SpecialValueContract) {
    const std::vector<numerics::Row> rows(std::begin(numerics::contract_rows),
                                          std::end(numerics::contract_rows));
    Case all;
    all.name = "contract";
    all.width = static_cast<std::uint32_t>(rows.size());
    all.height = 1;
    Case integers = all;
    std::vector<numerics::Row> integer_rows;
    for (const auto &row : rows) {
        all.add(as_float(row.x), as_float(row.offset), as_float(row.gain));
        const auto x = as_float(row.x);
        if (x >= 0 && x <= 65535 && x == std::floor(x) && !std::signbit(x)) {
            integers.add(static_cast<std::uint16_t>(x), as_float(row.offset), as_float(row.gain));
            integer_rows.push_back(row);
        }
    }
    integers.integer = true;
    integers.width = static_cast<std::uint32_t>(integer_rows.size());
    ASSERT_GE(integer_rows.size(), 10U);

    expect_rows(library(all, false), rows, "maps");
    expect_rows(library(all, true), rows, "fused");
    expect_rows(library(integers, false), integer_rows, "uint16");
    expect_rows(library(integers, true), integer_rows, "uint16 fused");

    // One 1x1 image per row, with scalar parameters.
    std::vector<HostImage> images;
    std::vector<DevicePtr> inputs;
    std::vector<DevicePtr> outputs;
    std::vector<cuda::BatchItem> items(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        images.push_back(make_image(PixelType::float32, 1, 1));
        images[i].set(0, 0, rows[i].x);
        inputs.push_back(upload(images[i].bytes));
        outputs.push_back(device_alloc(4));
        items[i].input = {inputs[i].get(), images[i].image, 4, 4};
        items[i].output = {outputs[i].get(), {PixelType::float32, 1, 1}, 4, 4};
        items[i].calibration.offset.scalar = as_float(rows[i].offset);
        items[i].calibration.gain.scalar = as_float(rows[i].gain);
    }
    expect_rows(execute(items, cuda::Operation::calibrate, outputs,
                        std::vector<std::uint64_t>(rows.size(), 4)),
                rows, "scalars");
}

} // namespace
