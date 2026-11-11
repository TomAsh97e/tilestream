#include "cuda_test_support.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace {

using namespace test;
using cuda::DeviceBuffer;

static_assert(static_cast<std::uint32_t>(Status::success) == 0);
static_assert(static_cast<std::uint32_t>(Status::invalid_argument) == 1);
static_assert(static_cast<std::uint32_t>(Status::invalid_roi) == 2);
static_assert(static_cast<std::uint32_t>(Status::unsupported_type) == 3);
static_assert(static_cast<std::uint32_t>(Status::unsupported_format) == 4);
static_assert(static_cast<std::uint32_t>(Status::malformed_data) == 5);
static_assert(static_cast<std::uint32_t>(Status::size_overflow) == 6);
static_assert(static_cast<std::uint32_t>(Status::buffer_too_small) == 7);
static_assert(static_cast<std::uint32_t>(Status::resource_limit) == 8);
static_assert(static_cast<std::uint32_t>(Status::cuda_error) == 9);
static_assert(static_cast<std::uint32_t>(Status::io_error) == 10);

struct Stream {
    cudaStream_t handle = nullptr;
    Stream() {
        EXPECT_EQ(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), cudaSuccess);
    }
    ~Stream() {
        static_cast<void>(cudaStreamDestroy(handle));
    }
};

// Everything a GPU round trip needs, with memory owned by library buffers on one stream.
class RoundTrip {
  public:
    RoundTrip(const std::vector<HostImage> &images, cudaStream_t stream)
        : images_(images), stream_(stream), count_(images.size()) {}

    Status prepare() {
        std::vector<cuda::BatchItem> encode(count_);
        std::vector<cuda::BatchItem> decode(count_);
        inputs_.resize(count_);
        archives_.resize(count_);
        outputs_.resize(count_);
        auto status = DeviceBuffer::allocate(count_ * sizeof(DeviceResult), stream_, encoded_);
        for (std::size_t i = 0; i < count_ && status == Status::success; ++i) {
            const auto &image = images_[i];
            std::uint64_t bound = 0;
            status = compress_bound(image.image, {}, bound);
            if (status == Status::success) {
                status = DeviceBuffer::allocate(image.bytes.size(), stream_, inputs_[i]);
            }
            if (status == Status::success) {
                status = DeviceBuffer::allocate(bound, stream_, archives_[i]);
            }
            if (status == Status::success) {
                status = DeviceBuffer::allocate(image.bytes.size(), stream_, outputs_[i]);
            }
            // The decoder leaves pitch padding alone, and verify() copies whole buffers.
            if (status == Status::success &&
                (cudaMemcpyAsync(inputs_[i].data(), image.bytes.data(), image.bytes.size(),
                                 cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                 cudaMemsetAsync(outputs_[i].data(), 0, image.bytes.size(), stream_) !=
                     cudaSuccess)) {
                status = Status::cuda_error;
            }
            encode[i].input = {inputs_[i].data(), image.image, image.pitch, image.bytes.size()};
            encode[i].compressed_output = archives_[i].bytes();
            decode[i].compressed_input.bytes = {archives_[i].data(), bound};
            decode[i].compressed_input.encoder_result =
                static_cast<const DeviceResult *>(encoded_.data()) + i;
            decode[i].compressed_input.trusted_image = image.image;
            decode[i].output = {outputs_[i].data(), image.image, image.pitch, image.bytes.size()};
        }
        if (status == Status::success) {
            status = cuda::prepare_batch({cuda::Operation::compress, encode.data(), count_, {}},
                                         compress_, stream_);
        }
        if (status == Status::success) {
            status = cuda::prepare_batch({cuda::Operation::decompress, decode.data(), count_, {}},
                                         decompress_, stream_);
        }
        for (auto *plan : {compress_.get(), decompress_.get()}) {
            std::uint64_t bytes = 0;
            if (status == Status::success) {
                status = workspace_size(plan->workspace_spec(), bytes);
            }
            if (status == Status::success) {
                status = DeviceBuffer::allocate(bytes, stream_,
                                                plan == compress_.get() ? compress_workspace_
                                                                        : decompress_workspace_);
            }
        }
        if (status == Status::success) {
            status = DeviceBuffer::allocate(count_ * sizeof(DeviceResult), stream_, decoded_);
        }
        return status;
    }

    Status enqueue() {
        auto status =
            cuda::compress_async(*compress_, compress_workspace_.bytes(),
                                 {static_cast<DeviceResult *>(encoded_.data()), count_}, stream_);
        if (status == Status::success) {
            status = cuda::decompress_async(*decompress_, decompress_workspace_.bytes(),
                                            {static_cast<DeviceResult *>(decoded_.data()), count_},
                                            stream_);
        }
        return status;
    }

    // Waits for the stream and checks every image bit for bit.
    bool verify() {
        std::vector<DeviceResult> results(count_);
        std::vector<Bytes> pixels(count_);
        bool queued =
            cudaMemcpyAsync(results.data(), decoded_.data(), count_ * sizeof(DeviceResult),
                            cudaMemcpyDeviceToHost, stream_) == cudaSuccess;
        for (std::size_t i = 0; i < count_; ++i) {
            pixels[i].resize(images_[i].bytes.size());
            queued =
                queued && cudaMemcpyAsync(pixels[i].data(), outputs_[i].data(), pixels[i].size(),
                                          cudaMemcpyDeviceToHost, stream_) == cudaSuccess;
        }
        if (!queued || cuda::synchronize(stream_) != Status::success) {
            return false;
        }
        for (std::size_t i = 0; i < count_; ++i) {
            HostImage decoded = images_[i];
            decoded.bytes = pixels[i];
            if (results[i].status != Status::success || decoded.pixels() != images_[i].pixels()) {
                return false;
            }
        }
        return true;
    }

    Status release() {
        compress_.reset();
        decompress_.reset();
        auto status = Status::success;
        for (auto *group : {&inputs_, &archives_, &outputs_}) {
            for (auto &buffer : *group) {
                if (buffer.release(stream_) != Status::success) {
                    status = Status::cuda_error;
                }
            }
        }
        for (auto *buffer : {&encoded_, &decoded_, &compress_workspace_, &decompress_workspace_}) {
            if (buffer->release(stream_) != Status::success) {
                status = Status::cuda_error;
            }
        }
        return status;
    }

  private:
    const std::vector<HostImage> &images_;
    cudaStream_t stream_;
    std::size_t count_;
    std::vector<DeviceBuffer> inputs_;
    std::vector<DeviceBuffer> archives_;
    std::vector<DeviceBuffer> outputs_;
    DeviceBuffer encoded_;
    DeviceBuffer decoded_;
    DeviceBuffer compress_workspace_;
    DeviceBuffer decompress_workspace_;
    std::unique_ptr<cuda::BatchPlan> compress_;
    std::unique_ptr<cuda::BatchPlan> decompress_;
};

bool cycle(const std::vector<HostImage> &images, cudaStream_t stream) {
    RoundTrip trip(images, stream);
    const bool ok =
        trip.prepare() == Status::success && trip.enqueue() == Status::success && trip.verify();
    return trip.release() == Status::success && ok;
}

std::vector<HostImage> sample_images(PixelType type, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<HostImage> images;
    for (std::uint32_t i = 0; i < 8; ++i) {
        const auto width = 1 + static_cast<std::uint32_t>(rng() % 300);
        const auto height = 1 + static_cast<std::uint32_t>(rng() % 200);
        const auto padding = static_cast<std::uint32_t>(rng() % 3);
        images.push_back(pattern_image(type, width, height, padding, i, rng));
    }
    return images;
}

std::uint64_t pool_in_use() {
    cudaMemPool_t pool = nullptr;
    std::uint64_t used = 0;
    EXPECT_EQ(cudaDeviceGetDefaultMemPool(&pool, 0), cudaSuccess);
    EXPECT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used), cudaSuccess);
    return used;
}

void CUDART_CB hold(void *flag) {
    // The timeout keeps a hidden device synchronization from hanging the test forever.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!static_cast<std::atomic<bool> *>(flag)->load() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

TEST(CudaApi, IndependentPlansOnTwoStreams) {
    const auto first = sample_images(PixelType::uint16, 1);
    const auto second = sample_images(PixelType::float32, 2);
    Stream a;
    Stream b;
    RoundTrip one(first, a.handle);
    RoundTrip two(second, b.handle);
    ASSERT_EQ(one.prepare(), Status::success);
    ASSERT_EQ(two.prepare(), Status::success);
    // Both plans are in flight at the same time and reused for several executions.
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(one.enqueue(), Status::success);
        ASSERT_EQ(two.enqueue(), Status::success);
    }
    EXPECT_TRUE(one.verify());
    EXPECT_TRUE(two.verify());
    EXPECT_EQ(one.release(), Status::success);
    EXPECT_EQ(two.release(), Status::success);
}

TEST(CudaApi, IndependentPlansFromTwoHostThreads) {
    const std::vector<HostImage> sets[] = {sample_images(PixelType::uint16, 3),
                                           sample_images(PixelType::float32, 4)};
    std::atomic<int> failures{0};
    const auto worker = [&](int k) {
        Stream stream;
        for (int i = 0; i < 10; ++i) {
            if (!cycle(sets[k], stream.handle)) {
                ++failures;
            }
        }
    };
    std::thread a(worker, 0);
    std::thread b(worker, 1);
    a.join();
    b.join();
    EXPECT_EQ(failures.load(), 0);
}

TEST(CudaApi, RepeatedCyclesReturnAllMemory) {
    const auto images = sample_images(PixelType::uint16, 5);
    Stream stream;
    ASSERT_TRUE(cycle(images, stream.handle));
    ASSERT_EQ(cudaStreamSynchronize(stream.handle), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
    const auto before = pool_in_use();
    for (int i = 0; i < 50; ++i) {
        ASSERT_TRUE(cycle(images, stream.handle)) << "cycle " << i;
    }
    ASSERT_EQ(cudaStreamSynchronize(stream.handle), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
    EXPECT_EQ(pool_in_use(), before);
}

TEST(CudaApi, NoHiddenDeviceSynchronization) {
    const auto images = sample_images(PixelType::float32, 6);
    Stream blocked;
    Stream worker;
    // The first preparation in a process loads the kernels, which may wait for the device.
    ASSERT_TRUE(cycle(images, worker.handle));
    std::atomic<bool> released{false};
    ASSERT_EQ(cudaLaunchHostFunc(blocked.handle, hold, &released), cudaSuccess);
    // Planning, allocation, execution, completion and release on another stream all finish
    // while the first stream is still held.
    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(cycle(images, worker.handle));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(10));
    EXPECT_EQ(cudaStreamQuery(blocked.handle), cudaErrorNotReady);
    released = true;
    EXPECT_EQ(cuda::synchronize(blocked.handle), Status::success);
}

TEST(CudaApi, SuccessfulEnqueueIsNotCompletion) {
    const auto images = sample_images(PixelType::uint16, 7);
    Stream setup;
    Stream blocked;
    RoundTrip trip(images, setup.handle);
    ASSERT_EQ(trip.prepare(), Status::success);
    ASSERT_EQ(cuda::synchronize(setup.handle), Status::success);
    std::atomic<bool> released{false};
    ASSERT_EQ(cudaLaunchHostFunc(blocked.handle, hold, &released), cudaSuccess);

    std::unique_ptr<cuda::BatchPlan> plan;
    const auto input = upload(images[0].bytes);
    std::uint64_t bound = 0;
    ASSERT_EQ(compress_bound(images[0].image, {}, bound), Status::success);
    const auto output = device_alloc(bound);
    cuda::BatchItem item;
    item.input = {input.get(), images[0].image, images[0].pitch, images[0].bytes.size()};
    item.compressed_output = {output.get(), bound};
    ASSERT_EQ(cuda::prepare_batch({cuda::Operation::compress, &item, 1, {}}, plan, setup.handle),
              Status::success);
    DeviceBuffer workspace;
    DeviceBuffer results;
    std::uint64_t bytes = 0;
    ASSERT_EQ(workspace_size(plan->workspace_spec(), bytes), Status::success);
    ASSERT_EQ(DeviceBuffer::allocate(bytes, setup.handle, workspace), Status::success);
    ASSERT_EQ(DeviceBuffer::allocate(sizeof(DeviceResult), setup.handle, results), Status::success);
    ASSERT_EQ(cudaMemsetAsync(results.data(), 0xff, sizeof(DeviceResult), setup.handle),
              cudaSuccess);
    ASSERT_EQ(cuda::synchronize(setup.handle), Status::success);

    // Argument errors are immediate; accepted work completes only after the stream runs.
    EXPECT_EQ(cuda::compress_async(*plan, {workspace.data(), bytes - 1},
                                   {static_cast<DeviceResult *>(results.data()), 1},
                                   blocked.handle),
              Status::buffer_too_small);
    ASSERT_EQ(cuda::compress_async(*plan, workspace.bytes(),
                                   {static_cast<DeviceResult *>(results.data()), 1},
                                   blocked.handle),
              Status::success);
    DeviceResult record;
    ASSERT_EQ(cudaMemcpyAsync(&record, results.data(), sizeof(record), cudaMemcpyDeviceToHost,
                              setup.handle),
              cudaSuccess);
    ASSERT_EQ(cuda::synchronize(setup.handle), Status::success);
    EXPECT_EQ(static_cast<std::uint32_t>(record.status), UINT32_MAX);
    released = true;
    ASSERT_EQ(cuda::synchronize(blocked.handle), Status::success);
    ASSERT_EQ(cudaMemcpy(&record, results.data(), sizeof(record), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(record.status, Status::success);
    EXPECT_EQ(download(output.get(), record.bytes_written), cpu_encode(images[0]));
    EXPECT_EQ(trip.release(), Status::success);
}

TEST(CudaApi, DeviceBufferOwnership) {
    Stream stream;
    DeviceBuffer buffer;
    ASSERT_EQ(DeviceBuffer::allocate(0, stream.handle, buffer), Status::success);
    EXPECT_EQ(buffer.data(), nullptr);
    ASSERT_EQ(DeviceBuffer::allocate(1000, stream.handle, buffer), Status::success);
    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % workspace_alignment, 0U);
    EXPECT_EQ(buffer.size(), 1000U);
    const auto *first = buffer.data();
    EXPECT_EQ(DeviceBuffer::allocate(UINT64_MAX, stream.handle, buffer), Status::size_overflow);
    EXPECT_EQ(DeviceBuffer::allocate(std::uint64_t{1} << 46, stream.handle, buffer),
              Status::cuda_error);
    EXPECT_EQ(buffer.data(), first);
    EXPECT_EQ(buffer.size(), 1000U);

    DeviceBuffer moved(std::move(buffer));
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(moved.data(), first);
    buffer = std::move(moved);
    EXPECT_EQ(buffer.data(), first);
    EXPECT_EQ(buffer.release(stream.handle), Status::success);
    EXPECT_EQ(buffer.release(stream.handle), Status::success);
    EXPECT_EQ(buffer.size(), 0U);
    ASSERT_EQ(cuda::synchronize(stream.handle), Status::success);

    // A failed allocation must not be reported again by the next execution.
    EXPECT_TRUE(cycle(sample_images(PixelType::uint16, 8), stream.handle));
}

TEST(CudaApi, PendingCallerErrorBlocksEnqueue) {
    const auto images = sample_images(PixelType::float32, 9);
    Stream stream;
    RoundTrip trip(images, stream.handle);
    ASSERT_EQ(trip.prepare(), Status::success);
    void *huge = nullptr;
    ASSERT_EQ(cudaMalloc(&huge, std::size_t{1} << 46), cudaErrorMemoryAllocation);
    EXPECT_EQ(trip.enqueue(), Status::cuda_error);
    // The error belongs to the caller, so it is still there to be read.
    EXPECT_EQ(cudaGetLastError(), cudaErrorMemoryAllocation);
    ASSERT_EQ(trip.enqueue(), Status::success);
    EXPECT_TRUE(trip.verify());
    EXPECT_EQ(trip.release(), Status::success);
}

} // namespace
