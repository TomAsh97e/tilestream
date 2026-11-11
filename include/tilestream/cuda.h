#pragma once

#include <tilestream/build_config.h>

#if TILESTREAM_WITH_CUDA

#include <tilestream/api.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace tilestream::cuda {

enum class Operation : std::uint32_t {
    compress,
    decompress,
    decode_roi,
    calibrate,
    decode_calibrate,
};

enum class ParameterKind : std::uint32_t { scalar, map };

struct Parameter {
    ParameterKind kind = ParameterKind::scalar;
    float scalar = 0.0F;
    ConstImageView map;
};

struct Calibration {
    Parameter offset;
    Parameter gain{ParameterKind::scalar, 1.0F, {}};
};

struct TileMapping {
    std::uint64_t tile_index = 0;
    std::uint64_t input_offset = 0;
};

struct CompressedInput {
    ConstBytes bytes;
    const ImageMetadata *metadata = nullptr;
    const TileMapping *mappings = nullptr;
    std::uint64_t mapping_count = 0;
    // The alternative trusted path accepts only a matching TileStream encoder output.
    const DeviceResult *encoder_result = nullptr;
    ImageSpec trusted_image;
    TileShape trusted_tile;
};

struct BatchItem {
    ConstImageView input;
    ImageView output;
    MutableBytes compressed_output;
    CompressedInput compressed_input;
    Roi roi;
    Calibration calibration;
};

struct BatchRequest {
    Operation operation = Operation::compress;
    const BatchItem *items = nullptr;
    std::uint64_t count = 0;
    TileShape tile;
};

// Device-only, validated execution records. No host vector or metadata pointer is uploaded.
struct DeviceSegment {
    std::uint32_t seed = 0;
    std::uint32_t bit_width = 0;
    std::uint64_t input_offset = 0;
};

struct DeviceTile {
    std::uint64_t tile_index = 0;
    std::uint64_t input_offset = 0;
    std::uint64_t length = 0;
    std::uint64_t first_segment = 0;
    TileMode mode = TileMode::raw;
    std::uint16_t width = 0;
    std::uint16_t height = 0;
};

struct DeviceImage {
    ConstImageView input;
    ImageView output;
    ConstBytes compressed_input;
    MutableBytes compressed_output;
    const DeviceTile *tiles = nullptr;
    const DeviceSegment *segments = nullptr;
    std::uint64_t tile_count = 0;
    const DeviceResult *encoder_result = nullptr;
    ImageSpec archive_image;
    TileShape archive_tile;
    Roi roi;
    Calibration calibration;
};

struct DeviceBatch {
    const DeviceImage *images = nullptr;
    std::uint32_t count = 0;
};

struct ResultBuffer {
    DeviceResult *data = nullptr;
    std::uint64_t count = 0;
};

// T04 contracts only. Construction and execution are implemented in T08-T11.
class BatchPlan {
  public:
    ~BatchPlan();
    BatchPlan(const BatchPlan &) = delete;
    BatchPlan &operator=(const BatchPlan &) = delete;
    BatchPlan(BatchPlan &&) noexcept;
    BatchPlan &operator=(BatchPlan &&) noexcept;

    [[nodiscard]] WorkspaceSpec workspace_spec() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit BatchPlan(std::unique_ptr<Impl> impl) noexcept;
    friend Status prepare_batch(const BatchRequest &, std::unique_ptr<BatchPlan> &,
                                cudaStream_t) noexcept;
    friend Status compress_async(const BatchPlan &, MutableBytes, ResultBuffer,
                                 cudaStream_t) noexcept;
    friend Status decompress_async(const BatchPlan &, MutableBytes, ResultBuffer,
                                   cudaStream_t) noexcept;
    friend Status decode_roi_async(const BatchPlan &, MutableBytes, ResultBuffer,
                                   cudaStream_t) noexcept;
    friend Status calibrate_async(const BatchPlan &, MutableBytes, ResultBuffer,
                                  cudaStream_t) noexcept;
    friend Status decode_calibrate_async(const BatchPlan &, MutableBytes, ResultBuffer,
                                         cudaStream_t) noexcept;
};

[[nodiscard]] Status prepare_batch(const BatchRequest &request, std::unique_ptr<BatchPlan> &plan,
                                   cudaStream_t stream) noexcept;
[[nodiscard]] Status compress_async(const BatchPlan &plan, MutableBytes workspace,
                                    ResultBuffer results, cudaStream_t stream) noexcept;
[[nodiscard]] Status decompress_async(const BatchPlan &plan, MutableBytes workspace,
                                      ResultBuffer results, cudaStream_t stream) noexcept;
[[nodiscard]] Status decode_roi_async(const BatchPlan &plan, MutableBytes workspace,
                                      ResultBuffer results, cudaStream_t stream) noexcept;
[[nodiscard]] Status calibrate_async(const BatchPlan &plan, MutableBytes workspace,
                                     ResultBuffer results, cudaStream_t stream) noexcept;
[[nodiscard]] Status decode_calibrate_async(const BatchPlan &plan, MutableBytes workspace,
                                            ResultBuffer results, cudaStream_t stream) noexcept;

} // namespace tilestream::cuda

#endif
