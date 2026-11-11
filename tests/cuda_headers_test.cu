#include <tilestream/cuda.h>

#include <type_traits>

using namespace tilestream;
using namespace tilestream::cuda;

using AsyncOperation = Status (*)(const BatchPlan &, MutableBytes, ResultBuffer,
                                  cudaStream_t) noexcept;

static_assert(std::is_same_v<decltype(&compress_async), AsyncOperation>);
static_assert(std::is_same_v<decltype(&decompress_async), AsyncOperation>);
static_assert(std::is_same_v<decltype(&decode_roi_async), AsyncOperation>);
static_assert(std::is_same_v<decltype(&calibrate_async), AsyncOperation>);
static_assert(std::is_same_v<decltype(&decode_calibrate_async), AsyncOperation>);
static_assert(!std::is_copy_constructible_v<BatchPlan>);
static_assert(std::is_nothrow_move_constructible_v<BatchPlan>);
static_assert(std::is_standard_layout_v<DeviceImage>);
static_assert(std::is_trivially_copyable_v<DeviceImage>);
static_assert(std::is_trivially_copyable_v<DeviceBatch>);
static_assert(std::is_trivially_copyable_v<DeviceTile>);
static_assert(std::is_trivially_copyable_v<DeviceSegment>);
static_assert(sizeof(DeviceResult) == 16);
static_assert(sizeof(DeviceSegment) == 16);
static_assert(sizeof(DeviceTile) == 40);

int main() {
    return 0;
}
