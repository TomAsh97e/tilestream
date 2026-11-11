# Numerical analysis of calibration

This document defines what TileStream promises about calibrated FP32 output, derives the error bounds of the implemented arithmetic, and reports measured errors. It is separate from codec correctness: the codec reproduces 32-bit representations exactly, while calibration computes new values and has a tolerance.

## Two kinds of correctness

| | Codec (`decompress`, `decode_roi`) | Calibration (`calibrate`, `decode_calibrate`) |
|---|---|---|
| Output | the stored uint16 or FP32 words | new FP32 values `(x - offset) * gain` |
| Requirement | identical bits, including NaN payloads, signed zeros and subnormals | the contract and bounds below |
| Arithmetic | none on FP32 samples | two binary32 operations per pixel |
| Tests | CTest label `codec` | CTest labels `numerics` and `calibration` |

`ctest -L codec` runs the bit-exact codec suites (`tilestream.cpu.codec`, `tilestream.cpu.fp32`, `tilestream.cuda.encoder`, `tilestream.cuda.decoder`). `ctest -L numerics` runs the calibration error suite (`tilestream.cpu.numerics`, `tilestream.cuda.numerics`). `tilestream.cuda.calibration` (label `calibration`) checks the operations themselves: validation, batches, regions and fused against separate execution. A passing codec test says nothing about calibration error, and calibration tolerance never applies to the codec.

## Implemented semantics

For every output pixel, with binary32 round to nearest even (RN):

```text
r = RN( RN(x - offset) * gain )
```

- A uint16 sample converts to binary32 exactly. An FP32 sample is used with its stored bits.
- The subtraction is rounded, then the multiplication. The two are never fused into an FMA.
- Subnormal operands are used as they are and subnormal results are produced (gradual underflow). Nothing is flushed to zero.
- Standalone and fused calibration run the same device operations on the same sample bits, so on one device they return identical words, NaNs included.

On the device this is `__fmul_rn(__fsub_rn(x, offset), gain)`. These intrinsics are never contracted into an FMA. The library is compiled with the default `-ftz=false` and never with `--use_fast_math`; the measurement below shows that `--use_fast_math` changes even these intrinsics.

The host form used by the tests is `float d = x - offset; return d * gain;`, compiled with `-ffp-contract=off` and checked by `static_assert(FLT_EVAL_METHOD == 0)`. `tilestream.cpu.numerics` checks at run time that the host rounds to nearest and that MXCSR has neither flush-to-zero nor denormals-are-zero set. Under these settings host and device must return the same word for every input, except for NaN payloads.

## Special-value contract

| Condition | Result |
|---|---|
| any operand is NaN (quiet or signaling) | NaN; sign and payload unspecified |
| `inf - inf` in the subtraction, or `0 * inf` in the multiplication | NaN |
| an infinite operand otherwise | the IEEE 754 result, an infinity of the corresponding sign |
| `x == offset` (finite) | `+0` before the multiplication; the result is `+0` or `-0` by the sign of `gain` |
| `(-0) - (+0)` | `-0`; every other exact zero difference is `+0` |
| `gain` is `+0` or `-0` and the difference is finite | a zero with the sign of the product |
| `|RN(x - offset) * gain|` reaches `2^128 - 2^103` | `+inf` or `-inf` (overflow) |
| `|x - offset|` reaches `2^128 - 2^103` | the difference is infinite, so the result is an infinity, or NaN when `gain` is zero, even when the exact `(x - offset) * gain` is finite (intermediate overflow) |
| a nonzero result below `2^-126` in magnitude | a subnormal or zero of the same sign, absolute error below `2^-149` |

NaN payloads are not part of the contract because the two supported implementations disagree on them. On the measured GPU every NaN result is the word `0x7fffffff`. On the x86-64 host the result carries the payload of the first NaN operand with the quiet bit set. Neither is wrong under IEEE 754; the codec, unlike calibration, preserves payloads exactly.

Intermediate overflow is a property of the specified two-rounding formula, not an implementation error. It is classified separately and never counted as an error against the FP64 reference.

`tilestream.cpu.numerics` and `tilestream.cuda.numerics` check 42 contract rows with their exact expected words: every case above, overflow at the `FLT_MAX` boundary, ties to even in the subnormal range, and exact subtractions. The device runs them through offset and gain maps, scalar parameters, uint16 samples where the row allows it, and the fused operation.

## Error metrics

These definitions were fixed before any measurement.

**Reference.** `ref = (double(x) - double(offset)) * double(gain)` in binary64. It cannot overflow or underflow for binary32 inputs, but each operation may round, so it is a higher-precision reference, not always the exact mathematical value. Its relative error is at most about `2^-52`, which is below `2^-27` units in the last place (ulp) of binary32.

**Absolute error.** `|r - ref|`, evaluated in binary64.

**Relative error.** `|r - ref| / max(|ref|, 2^-126)`. The threshold `FLT_MIN = 2^-126` keeps the value finite for zero and subnormal references, where relative error is not a useful measure.

**ULP error.** `|r - ref| / ulp32(ref)`, where `ulp32(v) = 2^(e - 23)` for `2^e <= |v| < 2^(e+1)` and `e >= -126`, and `ulp32(v) = 2^-149` below `2^-126`. It is measured against the binary64 reference, so it is a real number rather than a count of representable values.

**Statistics.** Maximum, median and 99th percentile, with the nearest-rank definition: the q-quantile of n sorted values is the value at position `ceil(q * n)`. They cover samples whose reference is finite and inside the binary32 range (classes `zero`, `subnormal`, `normal` below) and whose result is finite. A non-finite result for such a sample is counted separately, never mixed into the statistics. "ULP (normal)" restricts the statistics to class `normal`, where the ULP bound applies.

**Classes.** Each sample is classified once from its inputs and the reference, in this order: `nan_input` (an operand is NaN), `invalid` (the reference is NaN), `infinite` (the reference is infinite), `intermediate_overflow` (`|x - offset| >= 2^128 - 2^103`), `overflow` (`|ref| >= 2^128 - 2^103`), `zero` (`ref == 0`), `subnormal` (`|ref| < 2^-126`), and `normal`.

## Error bounds

Let `y = (x - offset) * gain` exactly and `u = 2^-24`.

1. The subtraction of two binary32 values is either exact or a normal-range result, because a difference in the subnormal range is always representable. So `RN(x - offset) = (x - offset)(1 + d)` with `|d| <= u`, and the product before the final rounding is `p = y(1 + d)`.
2. **Normal results.** `|r - y| <= |r - p| + |p - y| <= ulp32(p)/2 + u|y|`. If `p` and `y` lie in the same binade, this is below `0.5 + 1 = 1.5` ulp of `y`. If `p` crosses into the next binade up, the final rounding returns that power of two and the error is below 1 ulp; if it falls into the binade below, the error is below 1.25 ulp. The bound is therefore **`|r - y| < 1.5 ulp32(y)`**, and the relative error is at most `2u(1 + u/2)`, about **`2^-23`**.
3. **Subnormal results.** `|r - p| <= 2^-150` and `|p - y| <= u|y| < 2^-150`, so **`|r - y| < 2^-149`**. The relative error is unbounded in this range: `2^-150` may round to zero.
4. **Exact subtraction.** When `offset/2 <= x <= 2*offset` (Sterbenz lemma), `x - offset` is exact and only the multiplication rounds: **`|r - y| <= 0.5 ulp32(y)`**. Dark-frame subtraction and normalization around the mean of real data mostly fall here. The bound does not apply when the difference needs more than 24 significant bits, for example a uint16 sample near 65535 minus an offset with a fractional part.
5. **Overflow band.** The first rounding can move `p` across the overflow threshold, so a reference within a factor `1 + u` of `2^128 - 2^103` may give either `inf` or `FLT_MAX`. Both are accepted only inside that band.

The tests use these bounds with slack only for the reference error: 1.5 + 2^-20 ulp and a relative error of `2^-23 (1 + 2^-20)` for normal results, `2^-149 (1 + 2^-20)` absolute for subnormal results, and 0.5 + 2^-20 ulp for the exact-subtraction case. Tightening the normal bound to 1.0 ulp makes both numerical suites fail, so the bound is not vacuous.

## Variants compared

The library computes the controlled form above. Other formulas are studied in a separate experiment, `tests/numerics/variants.cu` and `tests/numerics/fast_math.cu`, which are not part of the library:

| Variant | Formula | Build |
|---|---|---|
| `controlled` | library `calibrate_async` with offset and gain maps | default |
| `fused` | library `decode_calibrate_async` on a CPU-encoded archive | default |
| `plain` | `(x - offset) * gain` with ordinary operators | default |
| `affine_fma` | `__fmaf_rn(gain, x, b)` | default |
| `affine_separate` | `__fadd_rn(__fmul_rn(gain, x), b)` | default |
| `affine_plain` | `gain * x + b` with ordinary operators | default (`-fmad=true`) |
| `fast_intrinsics` | `__fmul_rn(__fsub_rn(x, offset), gain)` | `--use_fast_math` |
| `fast_plain` | `(x - offset) * gain` | `--use_fast_math` |
| `fast_affine` | `gain * x + b` | `--use_fast_math` |

The affine form precomputes `b = -(offset * gain)` once in binary32, like a stored map, so `gain * x + b` is algebraically equal to `(x - offset) * gain` but not numerically. The experiment also recomputes `affine_fma` with the host `fmaf` and `affine_separate` with host binary32 operations; both agree with the device on every sample, so the differences below come from the formulas, not from the device.

## Method

`tilestream_numerics_report` (built with the tests in a CUDA configuration) generates each case, runs every variant on the GPU, and writes `results/numerics/summary.json`. With `--dump DIR` it also writes the inputs and every output as `.npy` files. `tests/numerics/check.py` then recomputes, independently with NumPy, the binary64 reference, the classes, the NumPy binary32 calibration, and every statistic, compares them with the report, and prints the tables below. All 11 cases agreed exactly.

```sh
cmake --preset cuda-release && cmake --build --preset cuda-release
build/cuda-release/tilestream_numerics_report --output results/numerics/summary.json --dump build/numerics-dump
python -I tests/numerics/check.py results/numerics/summary.json build/numerics-dump
```

The report stops with an error if the library differs from the host form, fused from standalone calibration, or a host recomputation of an affine variant from its device result.

Cases, all with per-pixel offset and gain maps:

| Case | Samples | Inputs |
|---|---:|---|
| `uint16_detector` | 131072 | uint16 samples uniform in [0, 65535], offset in [90, 110), gain in [0.5, 2) |
| `float32_field` | 131072 | smooth field 100 to 115 with noise, offset in [95, 105), gain in [0.9, 1.1) |
| `cancellation` | 131072 | offset in [1, 2^21) with random significand, x within 64 representable steps of it, gain of either sign |
| `dynamic_range` | 131072 | x and offset with random sign and exponent in [-126, 127], gain exponent in [-40, 40] |
| `underflow` | 131072 | x and offset exponents in [-126, -100], gain exponent in [-50, -1] |
| `overflow` | 131072 | x and offset exponents in [100, 127], gain exponent in [-30, 30]; every eighth sample has opposite signs near `FLT_MAX` |
| `gain_near_zero` | 131072 | x in [0, 65535), offset in [90, 110), gain zero, subnormal, or exponent in [-126, -90] |
| `random_bits` | 131072 | x, offset and gain as random 32-bit words |
| `special_grid` | 9261 | all triples of 21 words: zeros, subnormals, normal extremes, infinities, NaNs, values near 1, 2^24 and 2^-24 |
| `real_ct_small` | 16384 | CT image, uint16, 128 x 128 (`datasets/manifest.json`, tuning split) |
| `real_jacksboro_dem` | 138632 | SRTM elevation field, float32, 344 x 403 (evaluation split) |

Synthetic cases use `std::mt19937_64` with seed 2026517 and integer-to-float mappings defined in `tests/numerics/support.h`, not standard library distributions. The real images are normalized with parameters fixed before measuring: `offset` is the mean, `gain = 1 / (max - min)`, both rounded to binary32 (CT: 904.926147, 0.000484730; elevation: 531.031189, 0.00119047624). Nothing was tuned on either split. Synthetic distributions are stress tests; they do not describe how often each class occurs in real data.

Environment: NVIDIA GeForce RTX 4060 (compute capability 8.9), driver 580.178.04, CUDA 13.0.88, GCC 13.3.0, Release build. Host flags `-O3 -DNDEBUG -ffp-contract=off`; device flags `-O3 -DNDEBUG` with the nvcc defaults `-ftz=false -prec-div=true -prec-sqrt=true -fmad=true`. Host MXCSR `0x1fa0`: round to nearest, all exceptions masked, no flush to zero. The report records these in its `environment` block. A second run produced an identical report.

## Results

### Classes per case

| case | samples | nan_input | invalid | infinite | intermediate_overflow | overflow | zero | subnormal | normal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uint16_detector | 131072 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| float32_field | 131072 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 131072 |
| cancellation | 131072 | 0 | 0 | 0 | 0 | 0 | 1047 | 0 | 130025 |
| dynamic_range | 131072 | 0 | 0 | 0 | 6 | 10202 | 0 | 560 | 120304 |
| underflow | 131072 | 0 | 0 | 0 | 0 | 0 | 0 | 83645 | 47427 |
| overflow | 131072 | 0 | 0 | 0 | 16696 | 40267 | 0 | 0 | 74109 |
| gain_near_zero | 131072 | 0 | 0 | 0 | 0 | 0 | 32611 | 7 | 98454 |
| random_bits | 131072 | 1553 | 0 | 0 | 6 | 27174 | 0 | 5530 | 96809 |
| special_grid | 9261 | 3429 | 204 | 1532 | 32 | 454 | 760 | 724 | 2126 |
| real_ct_small | 16384 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16384 |
| real_jacksboro_dem | 138632 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 138632 |

### Library calibration

| case | ULP max / p99 / median (normal) | relative max | absolute max (subnormal) | violations |
|---|---|---:|---:|---:|
| uint16_detector | 1.37 / 0.99 / 0.295 | 1.11e-07 | - | 0 |
| float32_field | 0.5 / 0.495 / 0.25 | 5.93e-08 | - | 0 |
| cancellation | 0.5 / 0.5 / 0.25 | 5.96e-08 | - | 0 |
| dynamic_range | 1.39 / 0.799 / 0.257 | 1.08e-07 | 8.76e-46 | 0 |
| underflow | 1.39 / 0.998 / 0.297 | 1.12e-07 | 1.27e-45 | 0 |
| overflow | 1.39 / 1 / 0.293 | 1.14e-07 | - | 0 |
| gain_near_zero | 1.41 / 0.983 / 0.295 | 1.14e-07 | 5.54e-46 | 0 |
| random_bits | 1.41 / 0.801 / 0.258 | 1.16e-07 | 9.95e-46 | 0 |
| special_grid | 1 / 0.5 / 0 | 5.96e-08 | 7.01e-46 | 0 |
| real_ct_small | 0.5 / 0.497 / 0.25 | 5.76e-08 | - | 0 |
| real_jacksboro_dem | 0.5 / 0.496 / 0.251 | 5.63e-08 | - | 0 |

### Variants against the library

Cells: samples differing from the library / contract violations / ULP max, p99 for normal results.

| case | plain | affine_fma | affine_separate | affine_plain | fast_intrinsics | fast_plain | fast_affine |
|---|---|---|---|---|---|---|---|
| uint16_detector | 0 / 0 / 1.37, 0.99 | 32797 / 78 / 59.5, 0.496 | 44757 / 114 / 86.6, 0.919 | 32797 / 78 / 59.5, 0.496 | 0 / 0 / 1.37, 0.99 | 0 / 0 / 1.37, 0.99 | 32797 / 78 / 59.5, 0.496 |
| float32_field | 0 / 0 / 0.5, 0.495 | 122715 / 109688 / 9.24e+04, 210 | 123108 / 110918 / 3.36e+05, 275 | 122715 / 109688 / 9.24e+04, 210 | 0 / 0 / 0.5, 0.495 | 0 / 0 / 0.5, 0.495 | 122715 / 109688 / 9.24e+04, 210 |
| cancellation | 0 / 0 / 0.5, 0.5 | 131072 / 131071 / 8.38e+06, 3.19e+06 | 130514 / 130514 / 1.66e+07, 3.62e+06 | 131072 / 131071 / 8.38e+06, 3.19e+06 | 0 / 0 / 0.5, 0.5 | 0 / 0 / 0.5, 0.5 | 131072 / 131071 / 8.38e+06, 3.19e+06 |
| dynamic_range | 0 / 0 / 1.39, 0.799 | 7456 / 303 / 64.1, 0.721 | 8384 / 497 / 107, 0.813 | 7456 / 303 / 64.1, 0.721 | 531 / 524 / 1.39, 0.799 | 531 / 524 / 1.39, 0.799 | 8033 / 1033 / 2.68e+06, 0.758 |
| underflow | 0 / 0 / 1.39, 0.998 | 32066 / 5697 / 556, 1.13 | 34227 / 5794 / 469, 1.35 | 32066 / 5697 / 556, 1.13 | 62845 / 60369 / 1.39, 0.998 | 62845 / 60369 / 1.39, 0.998 | 91528 / 77897 / 1.55e+07, 1.35e+06 |
| overflow | 0 / 0 / 1.39, 1 | 45529 / 23346 / 822, 1.54 | 54649 / 29391 / 482, 1.97 | 45529 / 23346 / 822, 1.54 | 0 / 0 / 1.39, 1 | 0 / 0 / 1.39, 1 | 45529 / 23346 / 822, 1.54 |
| gain_near_zero | 0 / 0 / 1.41, 0.983 | 41080 / 16363 / 378, 0.497 | 50084 / 16394 / 669, 0.921 | 41080 / 16363 / 378, 0.497 | 33040 / 33040 / 1.68e+07, 1.64e+07 | 33040 / 33040 / 1.68e+07, 1.64e+07 | 65786 / 49381 / 7.62e+08, 1.66e+07 |
| random_bits | 0 / 0 / 1.41, 0.801 | 8035 / 2133 / 241, 0.725 | 10045 / 3553 / 199, 0.82 | 8035 / 2133 / 241, 0.725 | 3044 / 2948 / 1.68e+07, 0.875 | 3044 / 2948 / 1.68e+07, 0.875 | 10882 / 5147 / 2.71e+07, 0.837 |
| special_grid | 0 / 0 / 1, 0.5 | 793 / 759 / 1, 0.5 | 844 / 806 / 1, 0.5 | 793 / 759 / 1, 0.5 | 1040 / 858 / 1.41e+14, 1.68e+07 | 1040 / 858 / 1.41e+14, 1.68e+07 | 1490 / 1314 / 1.41e+14, 1.68e+07 |
| real_ct_small | 0 / 0 / 0.5, 0.497 | 2520 / 555 / 176, 5.26 | 11509 / 7831 / 1.66e+03, 46.5 | 2520 / 555 / 176, 5.26 | 0 / 0 / 0.5, 0.497 | 0 / 0 / 0.5, 0.497 | 2520 / 555 / 176, 5.26 |
| real_jacksboro_dem | 0 / 0 / 0.5, 0.496 | 133907 / 103011 / 6.63e+03, 104 | 122620 / 87244 / 1.09e+03, 103 | 133907 / 103011 / 6.63e+03, 104 | 0 / 0 / 0.5, 0.496 | 0 / 0 / 0.5, 0.496 | 133907 / 103011 / 6.63e+03, 104 |

### Examples (results printed with 9 significant digits)

| x | offset | gain | FP64 reference | controlled | fused | plain | affine_fma | affine_separate | affine_plain | fast_intrinsics | fast_plain | fast_affine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.00000012 | 1 | 100000000 | 11.920929 | 11.920929 | 11.920929 | 11.920929 | 11.920929 | 8 | 11.920929 | 11.920929 | 11.920929 | 11.920929 |
| 1234.56787 | 1234.5 | 0.00100000005 | 6.7871097e-05 | 6.78711003e-05 | 6.78711003e-05 | 6.78711003e-05 | 6.78791912e-05 | 6.78300858e-05 | 6.78791912e-05 | 6.78711003e-05 | 6.78711003e-05 | 6.78791912e-05 |
| 1001 | 1000.37 | 0.800000012 | 0.504003914 | 0.504003942 | 0.504003942 | 0.504003942 | 0.503991425 | 0.503967285 | 0.503991425 | 0.504003942 | 0.504003942 | 0.503991425 |
| 531.25 | 531.031189 | 0.00119047624 | 0.000260489338 | 0.000260489323 | 0.000260489323 | 0.000260489323 | 0.000260465225 | 0.000260472298 | 0.000260465225 | 0.000260489323 | 0.000260489323 | 0.000260465225 |
| 3.00000001e+38 | 2.90000002e+38 | 2 | 1.99999973e+37 | 1.99999973e+37 | 1.99999973e+37 | 1.99999973e+37 | -inf | nan | -inf | 1.99999973e+37 | 1.99999973e+37 | -inf |
| 3.40282347e+38 | -3.40282347e+38 | 0.25 | 1.70141173e+38 | inf | inf | inf | 1.70141173e+38 | 1.70141173e+38 | 1.70141173e+38 | inf | inf | 1.70141173e+38 |
| 7.34683969e-40 | 0 | 1 | 7.34683969e-40 | 7.34683969e-40 | 7.34683969e-40 | 7.34683969e-40 | 7.34683969e-40 | 7.34683969e-40 | 7.34683969e-40 | 0 | 0 | 0 |
| 1000 | 0 | 7.17464814e-43 | 7.17464814e-40 | 7.17464814e-40 | 7.17464814e-40 | 7.17464814e-40 | 7.17464814e-40 | 7.17464814e-40 | 7.17464814e-40 | 0 | 0 | 0 |
| 7.88860905e-31 | 5.91645679e-31 | 9.31322575e-10 | 1.83670992e-40 | 1.83670992e-40 | 1.83670992e-40 | 1.83670992e-40 | 1.83670992e-40 | 1.83670992e-40 | 1.83670992e-40 | 0 | 0 | 0 |

## Findings

1. **The library stays within its bound.** The largest error of a normal result was 1.41 ulp, below the derived 1.5 ulp; the 99th percentile was at most 1.001 ulp. Subnormal results stayed below `2^-149` absolute error. No sample of any case violated the contract. The device matched the host binary32 form and the NumPy binary32 computation on all 1,212,853 samples (NaN payloads aside), and fused output matched standalone output bit for bit.
2. **Cancellation is harmless in this form.** Where `x` is within a factor of two of `offset`, the subtraction is exact and the error is at most 0.5 ulp: in `cancellation`, `float32_field` and both real images, the maximum is 0.5 ulp. The larger errors in `uint16_detector` (up to 1.37 ulp) come from differences such as 65535 minus 97.3, which need more than 24 significant bits.
3. **The affine form is not a numerically equivalent replacement.** Precomputing `b = -offset * gain` rounds a term much larger than the result, and `gain * x + b` then cancels it. With samples a few ulps from the offset, the FMA affine form has a median error of 9.68e4 ulp and a maximum of 8.38e6 ulp; for exact zeros (`x == offset`) it returns the rounding error of `b` instead of zero. On the real elevation field, normalized around its mean, the FMA affine form violates the library's error bound on 103011 of 138632 pixels, with a 99th percentile of 104 ulp and a maximum of 6630 ulp, where the library form stays at 0.5 ulp. The CT image shows the same effect, less strongly (maximum 176 ulp).
4. **FMA versus separate rounding.** FMA removes the rounding of `gain * x` and is usually more accurate than the separate affine form (99th percentile 5.26 against 46.5 ulp on CT, 210 against 275 ulp on the smooth field), but not always: on the elevation field the FMA maximum (6630 ulp) is higher than the separate one (1090 ulp). Neither reaches the controlled form. The first example row shows the extreme: with `x = 1 + 2^-23`, `offset = 1`, `gain = 1e8`, the controlled and FMA forms both return the exact 11.920929, and the separate affine form returns 8.
5. **Source code does not fix the rounding.** With the default `-fmad=true`, nvcc compiled `gain * x + b` into an FMA: `affine_plain` matched `affine_fma` bit for bit in every case. Only the explicit intrinsics, or `-fmad=false`, fix the number of roundings. `(x - offset) * gain` has no contractible form, so `plain` matched the library everywhere.
6. **The forms overflow in different places.** For `x = 3e38`, `offset = 2.9e38`, `gain = 2`, the exact result 2e37 is finite and the library returns it, but `b = -5.8e38` overflows, so the FMA affine form returns `-inf` and the separate one NaN. Conversely, for `FLT_MAX - (-FLT_MAX)` with `gain = 0.25`, the library's difference overflows and it returns `inf`, while the FMA affine form returns the finite 1.7e38. In the `overflow` case the FMA affine form produced 176 infinite results for finite references and returned finite values for many samples the library reports as intermediate overflow; neither form is free of overflow artifacts.
7. **`--use_fast_math` breaks the subnormal contract, even for the intrinsics.** It implies `-ftz=true`, and `__fsub_rn` and `__fmul_rn` then flush subnormal operands and results: `fast_intrinsics` matched `fast_plain` in every case. In `underflow`, 60369 samples violated the contract; in `gain_near_zero`, every subnormal gain produced zero (33040 violations, relative error 1). On data without subnormals (`uint16_detector`, `float32_field`, `cancellation`, `overflow`, both real images), the fast-math forms matched their default-flag counterparts bit for bit. This is why fast math remains a per-file experiment and is never a global or library flag.
8. **Measured NaN behavior.** The GPU returned `0x7fffffff` for every NaN result. The host returned the first NaN operand's payload with the quiet bit set. Calibration therefore compares NaNs by class only.

## Limitations

- One GPU architecture (sm_89), one CUDA toolkit, one host compiler and one host instruction set were measured. Other configurations must rerun the suites rather than assume these results.
- The binary64 reference is not exact. Its error is bounded and far below the tolerances, but ULP values are measured against it, not against the real number.
- The statistics describe calibration only. Codec correctness is the bit-exact suite.
- Scalar parameters share the arithmetic with maps and are covered by the contract rows; the statistics use maps.
- `results/numerics/summary.json` holds the full statistics, including absolute and relative error summaries and result classes per variant, that the tables abbreviate.
