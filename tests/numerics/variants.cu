#include "numerics/variants.h"

namespace numerics {
namespace {

struct Plain {
    __device__ float operator()(float x, float offset, float gain, float) const {
        return (x - offset) * gain;
    }
};

struct AffineFma {
    __device__ float operator()(float x, float, float gain, float b) const {
        return __fmaf_rn(gain, x, b);
    }
};

struct AffineSeparate {
    __device__ float operator()(float x, float, float gain, float b) const {
        return __fadd_rn(__fmul_rn(gain, x), b);
    }
};

struct AffinePlain {
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

cudaError_t launch(Variant variant, const Inputs &inputs, cudaStream_t stream) {
    switch (variant) {
    case Variant::plain:
        return run<Plain>(inputs, stream);
    case Variant::affine_fma:
        return run<AffineFma>(inputs, stream);
    case Variant::affine_separate:
        return run<AffineSeparate>(inputs, stream);
    case Variant::affine_plain:
        return run<AffinePlain>(inputs, stream);
    default:
        return launch_fast(variant, inputs, stream);
    }
}

} // namespace numerics
