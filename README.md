# TileStream

TileStream is a lossless, tiled image codec for single-channel `uint16` and `float32` scientific images (detector frames, CT slices, elevation models), with a C++17 CPU reference implementation and a CUDA implementation that encodes, decodes, decodes regions of interest and applies calibration on the GPU.

The goal is to move fewer bytes over PCIe: archives stay compressed until they reach the GPU, where whole images or only the tiles a region needs are decoded, optionally fused with an affine calibration to FP32.

## Results

RTX 4060 (sm_89, 272 GB/s, PCIe 4.0 x8 at about 12 GB/s), CUDA 13.0, synthetic 4096x4096 detector frames. Every benchmark verifies its output bit for bit before reporting a time.

| | uint16 | float32 |
|---|---:|---:|
| Compression ratio (whole `.tsc` file) | 2.82 | 1.67 |
| GPU decode, device to device | 0.228 ms | 0.454 ms |
| DRAM traffic of the decode kernel | 199 GB/s | 237 GB/s |
| Host memory to calibrated FP32 on the GPU, TileStream | 1.45 ms | 3.93 ms |
| Same, uncompressed copy and calibration | 3.32 ms | 6.21 ms |

On this PCIe link the end-to-end time follows the transferred bytes, so decoding on the GPU is 2.3x (uint16) and 1.6x (float32) faster than copying the raw pixels. Against nvCOMP LZ4, zstd and TIFF Deflate on the same data and boundaries, TileStream has the highest ratio for uint16 and the fastest path from host memory to a calibrated image for both types; LZ4 with bitshuffle has a higher ratio for float32. Starting from a file in the page cache, host-side parsing and planning currently cost about as much as the transfer saves, so TileStream only matches reading raw pixels there. The full comparison, including the configurations that lose, is in [docs/baselines.md](docs/baselines.md).

## How it works

- **Format.** A `.tsc` file holds one image split into 128x32, 64x64 or 256x16 tiles with an index of tile offsets. Each tile is stored as RAW, CONSTANT or PREDICT_BITPACK, whichever is smallest. Predicted rows store a seed and the residuals against the left neighbor (zigzag modulo 2^16 for `uint16`, XOR of the bit patterns for `float32`), bit-packed with a per-row width. The encoding is canonical: the CPU and GPU encoders produce identical bytes. Specification: [docs/format.md](docs/format.md).
- **GPU decoder.** One block decodes one tile; a warp reconstructs a row with a prefix scan. Each lane decodes a group of consecutive columns sequentially in registers and the warp scans only the group totals, which reduced the shuffles per pixel and made the kernel memory-bound. Aligned groups are written with vector stores.
- **Regions of interest.** Only the tiles that intersect a region are copied and decoded; plans are prepared on the host once and executed asynchronously in a caller's stream with a caller-owned workspace.
- **Calibration.** `(x - offset) * gain` with scalar or per-pixel maps, standalone or fused into the decoder, with a documented error bound and a special-value contract: [docs/numerical-analysis.md](docs/numerical-analysis.md).

## Correctness

- Bit-exact round trips for every `uint16` value, every residual width, special `float32` words (signed zeros, infinities, NaN payloads, subnormals), partial tiles and pitched images; hand-calculated golden files; CPU and GPU bitstreams compared byte for byte.
- Seeded random round trips, structured mutations of valid archives and libFuzzer targets for the parser, with the minimized corpora replayed in every build.
- ASan and UBSan builds, and Compute Sanitizer memcheck, racecheck, synccheck and initcheck on all GPU suites with 0 errors. Compute Sanitizer also caught a real out-of-bounds read in an optimization of the decoder, described with its fix in [docs/performance-investigations.md](docs/performance-investigations.md).

The map from requirements to tests is in [docs/testing.md](docs/testing.md).

## Performance work

The benchmark harness separates device time, stage times and end-to-end host time, interleaves A/B comparisons and repeats runs in fresh processes, because the GPU boost clock changes between processes ([docs/benchmarking.md](docs/benchmarking.md)). Profiling with Nsight Systems led to two kept changes: the decoder described above (uint16 decode 21% faster) and copying plan tables without a staging buffer (plan preparation for a whole archive 2.5 to 1.1 ms). The write-up also covers a failed first variant and a ±30% CPU timing artifact caused by code placement alone.

## Building

Requirements: CMake 3.25+, Ninja, a C++17 compiler, CUDA 13 (for the GPU build), libtiff 4.5, GoogleTest 1.14 for the tests, libzstd for the benchmarks and pybind11 2.11 for the small Python module (`-DTILESTREAM_BUILD_PYTHON=OFF` skips it). nvCOMP is optional and only used by the benchmarks.

```sh
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release
```

Other presets: `cpu-only`, `cuda-debug` (Compute Sanitizer build), `asan`, `cuda-asan` and `fuzz`. The CUDA presets target `sm_89`; set `CMAKE_CUDA_ARCHITECTURES` for another GPU. `examples/cpp_client` builds against the installed package and runs a CPU round trip, then GPU compression, decompression and a fused decode with dark-frame correction.

Benchmarks:

```sh
benchmarks/run.py run --suite default --output results/benchmarks/run.json
benchmarks/run.py table results/benchmarks/run.json
```

## Repository layout

| Path | Contents |
|---|---|
| `include/tilestream/` | public API: sizing, CPU codec, CUDA plans and operations, TIFF I/O |
| `src/cpu/`, `src/cuda/`, `src/io/` | CPU reference codec and parser, CUDA kernels and planning, libtiff adapter |
| `tests/`, `fuzz/` | GoogleTest suites, golden files, numerical reports, fuzz targets and corpora |
| `benchmarks/` | benchmark harness, baselines (zstd, TIFF, nvCOMP LZ4), suite runner, Nsight Compute script |
| `datasets/` | deterministic generators and small real-data fixtures with licenses |
| `docs/` | requirements, format, API, testing, numerics, benchmarking, baselines, performance |
| `results/` | recorded benchmark, sanitizer and profiling runs referenced by the documents |

## Status and limitations

- Measured on one machine and one GPU (RTX 4060). Numbers on other GPUs and PCIe links will differ.
- Copies and decodes of consecutive requests are not yet overlapped, and there is no streaming pipeline for data larger than a device budget.
- The format and API are version 1 and may still change.
