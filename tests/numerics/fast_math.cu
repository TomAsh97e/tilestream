// Built with --use_fast_math (see CMakeLists.txt); the rest of the project never is.
#include "numerics/variants.h"

namespace numerics {
namespace {

struct Intrinsics {
    __device__ float operator()(float x, float offset, float gain, float) const {
        return __fmul_rn(__fsub_rn(x, offset), gain);
    }
};

struct Plain {
    __device__ float operator()(float x, float offset, float gain, float) const {
        return (x - offset) * gain;
    }
};

struct Affine {
    __device__ float operator()(float x, float, float gain, float b) const {
        return gain * x + b;
    }
};

template <typename Formula> __global__ void apply(Inputs in, Formula formula) {
    const std::uint64_t stride = std::uint64_t{gridDim.x} * blockDim.x;
    for (std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < in.count;
         i += stride) {
        in.output[i] = formula(in.x[i], in.offset[i], in.gain[i], in.b[i]);
    }
}

template <typename Formula> cudaError_t run(const Inputs &inputs, cudaStream_t stream) {
    apply<<<256, 256, 0, stream>>>(inputs, Formula{});
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_fast(Variant variant, const Inputs &inputs, cudaStream_t stream) {
    switch (variant) {
    case Variant::fast_intrinsics:
        return run<Intrinsics>(inputs, stream);
    case Variant::fast_plain:
        return run<Plain>(inputs, stream);
    case Variant::fast_affine:
        return run<Affine>(inputs, stream);
    default:
        return cudaErrorInvalidValue;
    }
}

} // namespace numerics
