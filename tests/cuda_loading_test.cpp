#include "cuda_test_support.h"

#include <atomic>
#include <chrono>
#include <thread>

// Kernels are loaded once per process, so this suite has an executable of its own and every
// test is the first use of the kernels it runs. Setup never launches a kernel, not even a memset.
namespace {

using namespace test;

using Execute = Status (*)(const cuda::BatchPlan &, MutableBytes, cuda::ResultBuffer,
                           cudaStream_t) noexcept;

struct Stream {
    cudaStream_t handle = nullptr;
    Stream() {
        EXPECT_EQ(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), cudaSuccess);
    }
    ~Stream() {
        static_cast<void>(cudaStreamDestroy(handle));
    }
};

DevicePtr allocate(std::uint64_t bytes) {
    void *data = nullptr;
    EXPECT_EQ(cudaMalloc(&data, static_cast<std::size_t>(bytes)), cudaSuccess);
    return DevicePtr(data);
}

DevicePtr copy(const Bytes &bytes) {
    auto device = allocate(bytes.size());
    EXPECT_EQ(cudaMemcpy(device.get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
              cudaSuccess);
    return device;
}

void CUDART_CB hold(void *flag) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!static_cast<std::atomic<bool> *>(flag)->load() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// The first execution after preparation must not wait for another, held stream.
void expect_independent(const cuda::BatchRequest &request, Execute execute) {
    Stream work;
    Stream blocked;
    std::unique_ptr<cuda::BatchPlan> plan;
    ASSERT_EQ(cuda::prepare_batch(request, plan, work.handle), Status::success);
    std::uint64_t bytes = 0;
    ASSERT_EQ(workspace_size(plan->workspace_spec(), bytes), Status::success);
    const auto workspace = allocate(std::max<std::uint64_t>(bytes, 1));
    const auto results = allocate(request.count * sizeof(DeviceResult));

    std::atomic<bool> released{false};
    ASSERT_EQ(cudaLaunchHostFunc(blocked.handle, hold, &released), cudaSuccess);
    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(execute(*plan, {workspace.get(), bytes},
                      {static_cast<DeviceResult *>(results.get()), request.count}, work.handle),
              Status::success);
    EXPECT_EQ(cuda::synchronize(work.handle), Status::success);
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
    EXPECT_EQ(cudaStreamQuery(blocked.handle), cudaErrorNotReady);
    released = true;
    EXPECT_EQ(cuda::synchronize(blocked.handle), Status::success);

    std::vector<DeviceResult> records(static_cast<std::size_t>(request.count));
    ASSERT_EQ(cudaMemcpy(records.data(), results.get(), records.size() * sizeof(DeviceResult),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (const auto &record : records) {
        EXPECT_EQ(record.status, Status::success);
    }
}

HostImage sample(PixelType type) {
    std::mt19937 rng(static_cast<std::uint32_t>(type));
    return pattern_image(type, 300, 200, 1, 1, rng);
}

void expect_compress(PixelType type) {
    const auto image = sample(type);
    const auto input = copy(image.bytes);
    std::uint64_t bound = 0;
    ASSERT_EQ(compress_bound(image.image, {}, bound), Status::success);
    const auto output = allocate(bound);
    cuda::BatchItem item;
    item.input = {input.get(), image.image, image.pitch, image.bytes.size()};
    item.compressed_output = {output.get(), bound};
    expect_independent({cuda::Operation::compress, &item, 1, {}}, cuda::compress_async);
}

void expect_decode(PixelType type, cuda::Operation operation, Execute execute) {
    const auto image = sample(type);
    const auto file = cpu_encode(image);
    ImageMetadata metadata;
    ASSERT_EQ(parse_metadata({file.data(), file.size()}, {}, metadata), Status::success);
    const auto source = copy(file);
    const Roi roi{10, 20, 150, 100};
    const bool full = operation == cuda::Operation::decompress;
    const bool calibrated = operation == cuda::Operation::decode_calibrate;
    const ImageSpec region{calibrated ? PixelType::float32 : type,
                           full ? image.image.width : roi.width,
                           full ? image.image.height : roi.height};
    const auto pitch = std::uint64_t{region.width} * (calibrated ? 4 : image.sample_bytes());
    const auto output = allocate(pitch * region.height);
    cuda::BatchItem item;
    item.compressed_input.bytes = {source.get(), file.size()};
    item.compressed_input.metadata = &metadata;
    item.roi = roi;
    item.output = {output.get(), region, pitch, pitch * region.height};
    expect_independent({operation, &item, 1, {}}, execute);
}

void expect_calibrate(PixelType type) {
    const auto image = sample(type);
    const auto input = copy(image.bytes);
    const ImageSpec region{PixelType::float32, image.image.width, image.image.height};
    const auto pitch = std::uint64_t{region.width} * 4;
    const auto output = allocate(pitch * region.height);
    cuda::BatchItem item;
    item.input = {input.get(), image.image, image.pitch, image.bytes.size()};
    item.output = {output.get(), region, pitch, pitch * region.height};
    expect_independent({cuda::Operation::calibrate, &item, 1, {}}, cuda::calibrate_async);
}

TEST(CudaLoading, CompressUint16) {
    expect_compress(PixelType::uint16);
}

TEST(CudaLoading, CompressFloat32) {
    expect_compress(PixelType::float32);
}

TEST(CudaLoading, DecompressUint16) {
    expect_decode(PixelType::uint16, cuda::Operation::decompress, cuda::decompress_async);
}

TEST(CudaLoading, DecodeRoiFloat32) {
    expect_decode(PixelType::float32, cuda::Operation::decode_roi, cuda::decode_roi_async);
}

TEST(CudaLoading, CalibrateUint16) {
    expect_calibrate(PixelType::uint16);
}

TEST(CudaLoading, CalibrateFloat32) {
    expect_calibrate(PixelType::float32);
}

TEST(CudaLoading, DecodeCalibrateUint16) {
    expect_decode(PixelType::uint16, cuda::Operation::decode_calibrate,
                  cuda::decode_calibrate_async);
}

TEST(CudaLoading, DecodeCalibrateFloat32) {
    expect_decode(PixelType::float32, cuda::Operation::decode_calibrate,
                  cuda::decode_calibrate_async);
}

} // namespace
