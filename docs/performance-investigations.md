# Performance investigations

This document records how the GPU path was profiled, which bottlenecks were found and which changes were kept. Final numbers come from the benchmark harness ([benchmarking.md](benchmarking.md)); profilers were used for diagnosis only. Hardware: RTX 4060 (sm_89, 24 SMs, 24 MiB L2, 272 GB/s, PCIe 4.0 x8), driver 580.178.04, CUDA 13.0.88.

## Tools and their availability

- **Nsight Systems 2025.3.2** works without privileges (`nsys profile -t cuda,osrt`). The reports are not committed, because they record the whole process environment of the machine; the numbers quoted below come from `nsys stats` on them. Running a scenario of `tilestream_codec_benchmark` under `nsys profile -t cuda,osrt` regenerates them.
- **Nsight Compute 2025.3.1** cannot read the GPU performance counters on this machine: the driver runs with `RmProfilingAdminOnly=1` and `ncu` fails with `ERR_NVGPUCTRPERM`. Without counters there are no measured memory throughput, achieved occupancy or stall reasons. `benchmarks/ncu.sh` collects the full section set for the decoder and encoder kernels and has to be run with administrator rights (`sudo benchmarks/ncu.sh`); its reports are not part of this document.
- Instead, the kernels were examined with what needs no counters: register and shared memory use from `cuobjdump --dump-resource-usage`, theoretical occupancy from those limits, the instruction mix from `cuobjdump -sass`, and the effective bandwidth from the measured time and the bytes moved.
- **CPU profiling** is not available either (`perf_event_paranoid=4`, which also disables CPU sampling in Nsight Systems). Host code was timed with temporary `steady_clock` instrumentation in a scratch copy of `src/cuda/plan.cpp`, which is not part of the repository.

## Findings

### Kernels

Nsight Systems on `gpu_decode` and `gpu_encode`, 4096x4096, tile 128x32:

| Operation | Kernels | Time per operation |
|---|---|---|
| decode uint16 | `begin` 1.1 us, `decode` 309 us, `complete` 1.5 us | 0.31 ms |
| decode float32 | `decode` 479 us | 0.48 ms |
| decode + calibrate uint16 | `decode<.., true>` 444 us | 0.45 ms |
| encode uint16 | `analyze` 402 us, `pack` 356 us, scan 3.7 us, `finish` 2.0 us | 0.77 ms |
| encode float32 | `analyze` 531 us, `pack` 462 us | 1.0 ms |

- The decoder is one kernel per window of up to 4096 tiles; the bookkeeping kernels and the launch gaps between them (0.6-1.2 us each) add about 5 us per operation, 2% of a 4096x4096 decode. They matter only for small images.
- Decoding moved 11.9 MB in and 33.5 MB out for uint16 in 0.309 ms, 147 GB/s, about half of the DRAM bandwidth. For float32 (40.3 MB in, 67.1 MB out in 0.479 ms) it is 224 GB/s, closer to the limit. The uint16 decoder was therefore limited by instructions rather than memory.
- The encoder reads its input twice (`analyze` chooses the mode and bit width of every row, `pack` writes the bits). `analyze` reaches only 83 GB/s on its input. It is the largest kernel by time, but the encoder is not on the critical path of the read scenarios, so it was not changed in this stage.
- `decode`: 256 threads, 46-48 registers, 17.2 KB of shared memory (the tile payload and row tables). Registers and shared memory each allow 5 blocks per SM, so the theoretical occupancy is 40 of 48 warps (83%).

### Synchronizations and transfers

- `prepare_batch` copies its tables from pageable memory and waits for its stream, as documented. For a whole 4096x4096 archive the tables are 2.26 MB (131 072 row descriptors of 16 bytes) and the copy and wait take 0.25 ms.
- Host-to-device copies run at about 12 GB/s on this PCIe 4.0 x8 link. Copying the uint16 archive takes 0.99 ms, more than twice its decode time, so on this machine compression pays off mainly by reducing the transfer.
- `archive_roi` issues copies in one stream and decodes in another, but every decode starts only after the last copy. With `CUDA_DEVICE_MAX_CONNECTIONS=1` they overlap. Overlapping copies and decodes needs a pipeline with its own scheduling and was not changed here.

### Host planning

The `tsc_file` baseline showed that reading a 4096x4096 archive from the page cache costs more in `parse_metadata` and `prepare_batch` than in reading the file. Timed separately (`uint16`, 4096 tiles, 131 072 rows):

| Step | Time |
|---|---|
| `parse_metadata` | 1.5-1.7 ms (float32: 4.9 ms) |
| `prepare_batch`: draft and checks | 1.55-1.84 ms |
| `prepare_batch`: zero-filled 2.26 MB staging buffer | 0.53-0.61 ms |
| `prepare_batch`: table fill and `cudaMallocAsync` | 0.10-0.13 ms |
| `prepare_batch`: copy and stream wait | 0.25 ms |

The staging buffer costs about 4 GB/s, the speed of first-touch page faults rather than of `memset`: glibc returns such large blocks to the system when they are freed, so every plan pays again. The parser walks the file twice (count, then fill) and touches a row descriptor and the last packed word of every row; the float32 file (20 MB) no longer fits the CPU caches as well as the uint16 file (12 MB), which explains its 3 times longer parse.

## Changes

Each change was measured against the state just before it: a second build tree that differs only in the changed file, built with the same flags, and both binaries interleaved in three rounds of 30 samples per process. The tables show the median of the round medians (`results/performance/opt*-ab.txt`).

### 1. Several columns per lane in bitpacked rows (kept)

Before, a warp decoded 32 columns of a row at a time: one unpacked residual per lane, a 5-step warp scan and one more shuffle for the carry, 6 shuffles per 32 pixels, and one 2-byte store per lane. Now every lane decodes a group of consecutive columns, scans them sequentially in registers, and the warp scans only the group totals. An aligned group is written with one vector store (`ST.E.64` for four uint16 samples, `ST.E.128` for four float values, confirmed in the SASS); unaligned groups at region edges fall back to scalar stores. The output bits are unchanged: the uint16 scan still narrows to 16 bits only at the end, and the float32 scan still uses XOR.

The first version used groups of 4 columns for every tile. It was faster for 128x32 and 256x16 tiles, but slower for 64x64 tiles (`gpu_decode` uint16 +8.1%, `gpu_decode_roi` +12.5%), because a 64-pixel row left half of the lanes idle (`opt1-fixed-group-ab.txt`). The kept version picks the group size from the tile width: 4 above 64 pixels, 2 up to 64, 1 up to 32.

| Case (4096x4096) | 128x32 | 64x64 | 256x16 |
|---|---:|---:|---:|
| `gpu_decode` uint16 | 0.289 -> 0.228 ms (-21.1%) | 0.330 -> 0.261 (-20.8%) | 0.282 -> 0.224 (-20.7%) |
| `gpu_decode_calibrate` uint16 | 0.448 -> 0.409 (-8.8%) | 0.429 -> 0.400 (-6.7%) | 0.415 -> 0.366 (-11.9%) |
| `gpu_decode_roi` uint16, 8 x 1024² | 0.173 -> 0.126 (-26.9%) | 0.182 -> 0.154 (-15.3%) | 0.172 -> 0.123 (-28.0%) |
| `gpu_decode_calibrate` uint16, 8 x 1024² | 0.244 -> 0.216 (-11.5%) | 0.255 -> 0.241 (-5.6%) | 0.260 -> 0.217 (-16.5%) |
| `gpu_decode` float32 | 0.472 -> 0.453 (-4.1%) | 0.490 -> 0.465 (-5.1%) | 0.485 -> 0.456 (-6.0%) |
| `gpu_decode_calibrate` float32 | 0.524 -> 0.515 (-1.8%) | 0.524 -> 0.537 (+2.5%) | 0.532 -> 0.512 (-3.8%) |
| `gpu_decode_roi` float32, 8 x 1024² | 0.268 -> 0.254 (-5.2%) | 0.271 -> 0.259 (-4.5%) | 0.285 -> 0.268 (-5.9%) |
| `gpu_decode_calibrate` float32, 8 x 1024² | 0.297 -> 0.296 (-0.3%) | 0.306 -> 0.302 (-1.3%) | 0.311 -> 0.307 (-1.2%) |

uint16 decoding now moves 45.4 MB in 0.228 ms (199 GB/s) and float32 decoding 107 MB in 0.453 ms (237 GB/s), both close to the DRAM bandwidth, which is why float32 gained less.

The calibrated rows come from a separate A/B run after a fix (`opt1-calibrated-fix-ab.txt`). The full Compute Sanitizer run found an out-of-bounds read in the first kept version: a group that starts before the left edge of the region, or ends after its right edge, calibrated all of its columns, and with a calibration map the columns outside the region read outside the map. The test suites with maps covering the whole image did not notice it; memcheck failed in the calibration and numerics suites, which use maps of the region only. The fix calibrates only the columns inside the region. It costs part of the calibrated gain: uint16 is 6-17% faster instead of 13-21%, and float32 with 64x64 tiles is now 2.5% slower than before the change, while the other float32 cases are within -4% to 0%. The calibrated kernels use 64 registers instead of 48, so their theoretical occupancy dropped from 5 to 4 blocks per SM (83% to 67%).

### 2. Plan tables copied without a staging buffer (kept, with a cost for tiny plans)

`prepare_batch` assembled its tables in a zero-filled host vector and copied that vector to the device. Now each table is copied directly from the vector it was built in (up to five copies instead of one); the stream wait stays.

| Case | Before | After | Change |
|---|---:|---:|---:|
| `prepare_batch`, whole uint16 4096x4096 archive (probe) | 2.5-2.6 ms | 1.09-1.10 ms | -57% |
| `archive_roi` uint16, whole image, `prepare` stage | 2.488 ms | 1.167 ms | -53.1% |
| `archive_roi` uint16, whole image, end to end | 4.810 ms | 3.479 ms | -27.7% |
| `archive_roi` float32, whole image, end to end | 9.842 ms | 8.238 ms | -16.3% |
| `tsc_file` uint16, `parse_prepare` stage | 3.776 ms | 2.971 ms | -21.3% |
| `tsc_file` uint16, end to end | 6.459 ms | 5.593 ms | -13.4% |
| `tsc_file` float32, end to end | 15.445 ms | 14.843 ms | -3.9% |
| `archive_roi` uint16, 4 x 1024², end to end | 1.067 ms | 0.996 ms | -6.7% |
| `archive_roi` uint16, 1 x 256², `prepare` stage | 14.9 us | 16.4 us | +9.9% |
| `archive_roi` uint16, 1 x 256², end to end | 53.6 us | 54.9 us | +2.5% |

Small plans pay about 1.4 us for the additional copy calls, 2-2.5% of a 256x256 request. Large plans save more than a millisecond. The change was kept for the large plans; a size threshold that keeps one copy for small plans would remove the cost, but adds a second code path and was not tried.

## Before and after

The default suite and the TileStream cases of the baseline suite were run before the first change and after the second one with `run.py` (5 rounds of 30 samples each; `results/performance/before-*.json`, `after-*.json`, `compare-*.txt`).

| Case | Before ms | After ms | Change |
|---|---:|---:|---:|
| `gpu_decode` uint16 4096² | 0.291 | 0.228 | -21.7% |
| `gpu_decode_roi` uint16, 8 x 1024² | 0.161 | 0.127 | -21.3% |
| `gpu_decode_calibrate` uint16, 8 x 1024² (before the fix) | 0.249 | 0.195 | -21.8% |
| `archive_roi` uint16, 4 x 1024² | 1.116 | 0.985 | -11.8% |
| `gpu_decode` float32 4096² | 0.472 | 0.454 | -3.9% |
| `archive_roi` float32, 4 x 1024² | 2.395 | 2.228 | -7.0% |
| `gpu_decode` CT slice x64 | 0.0281 | 0.0199 | -29.1% |
| `gpu_decode` DEM x16 | 0.0549 | 0.0429 | -21.8% |
| `tsc_calibrate` uint16 4096² (before the fix) | 1.450 | 1.327 | -8.4% |
| `tsc_file` uint16 4096² | 6.623 | 5.537 | -16.4% |
| `tsc_file` float32 4096² | 15.634 | 14.652 | -6.3% |
| `gpu_encode`, `archive_roi` 256² | | | within ±2% |

The two calibrated rows were measured before the fix of the calibration map read described above; the A/B numbers in that section replace them.

`cpu_decode/float32/2048` became 32% slower (9.66 to 12.80 ms) although the CPU decoder did not change and no code of the plan runs in that scenario. The cause is code placement: the change of `plan.cpp` moved the CPU decoder by 0x9230 bytes, a multiple of 16 but not of 32. Adding 16, 32 or 48 bytes of padding in front of the CPU decoder, with no other change, gives 10.6, 9.6-9.9 and 12.9-13.5 ms. The CPU numbers of this harness therefore carry an uncertainty of about ±30% from the binary layout alone (an effect of how loops fall on 32-byte boundaries on this Intel CPU), and differences of that size between builds must not be read as code changes. The GPU kernels are not affected.

## Not changed

- **Parser.** A single pass with an upper bound for the row count would avoid the second walk over the file, but it changes when the metadata limit is checked. It is the next candidate for the file path.
- **Encoder.** Fusing `analyze` and `pack` would read the input once; the encoder is not on the critical path of the measured read scenarios.
- **Bookkeeping kernels.** `begin` and `complete` cost about 5 us per operation, which matters only for small images.

## Checks

After each change: CTest (all suites, `cuda-release`), Compute Sanitizer memcheck on the decoder and robustness suites (0 errors), and the full sanitizer script after both changes. Its first run found the calibration map read described above; after the fix all four tools pass on all six suites with 0 errors (`compute-sanitizer.txt`). The benchmark scenarios compare their outputs with the source pixels and the calibrated reference, so every number above comes from verified runs.
