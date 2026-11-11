#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

// Formulas studied next to the library calibration. They are experiments, not library code.
namespace numerics {

enum class Variant {
    plain,           // (x - offset) * gain with ordinary operators
    affine_fma,      // fmaf(gain, x, b)
    affine_separate, // gain * x + b, rounded twice
    affine_plain,    // gain * x + b with ordinary operators, contractible under -fmad=true
    fast_intrinsics, // the library intrinsics, compiled with --use_fast_math
    fast_plain,      // plain, compiled with --use_fast_math
    fast_affine,     // affine_plain, compiled with --use_fast_math
};

// b holds the rounded products -offset * gain of the affine form.
struct Inputs {
    const float *x = nullptr;
    const float *offset = nullptr;
    const float *gain = nullptr;
    const float *b = nullptr;
    float *output = nullptr;
    std::uint64_t count = 0;
};

cudaError_t launch(Variant variant, const Inputs &inputs, cudaStream_t stream);
cudaError_t launch_fast(Variant variant, const Inputs &inputs, cudaStream_t stream);

} // namespace numerics
