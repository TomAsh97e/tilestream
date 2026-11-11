# Benchmark harness

This document describes how TileStream performance is measured: the programs, the timing rules, the byte accounting and the result schema. Baselines, profiling, pipelining and the ROI experiment build on this harness in later stages; the numbers below only show that the harness itself is repeatable.

## Programs

`tilestream_codec_benchmark` (built when both CUDA and `TILESTREAM_BUILD_BENCHMARKS` are enabled) measures one case and prints one JSON object with every sample. `benchmarks/run.py` runs a suite of cases, records the environment, validates the result against `benchmarks/schema.json` and writes one file.

```sh
cmake --preset cuda-release && cmake --build --preset cuda-release
benchmarks/run.py run --build build/cuda-release --output results/benchmarks/run.json
benchmarks/run.py summary results/benchmarks/run.json
benchmarks/run.py compare results/benchmarks/run.json results/benchmarks/rerun.json
benchmarks/run.py validate results/benchmarks/run.json
benchmarks/run.py table results/benchmarks/run.json
```

`run.py` uses only the Python standard library. `--suite quick` runs a short smoke suite, `--suite baselines` the comparison with the baselines described in [baselines.md](baselines.md), and `--filter TEXT` runs the cases whose id contains `TEXT`.

## Scenarios

| Scenario | Boundary | Requirement |
|---|---|---|
| `gpu_encode` | device-resident pixels to device-resident archives, `compress_async` | 10.2 |
| `gpu_decode` | device-resident archive with parsed host metadata to device-resident pixels, `decompress_async` | 10.2 |
| `gpu_decode_roi` | as `gpu_decode`, one region per image, `decode_roi_async` | 10.2 |
| `gpu_decode_calibrate` | as `gpu_decode_roi` with fused calibration to FP32, `decode_calibrate_async` | 10.2 |
| `archive_roi` | archive in pinned host memory to calibrated FP32 output on the device: gather the intersecting tiles, `prepare_batch`, one host-to-device copy per image, `decode_calibrate_async` | 10.1, pinned RAM tier |
| `cpu_encode`, `cpu_decode` | host pixels to a host archive and back, one thread | reference |

Archives are created and parsed before measurement and are never part of a timing. Plan preparation and workspace allocation happen once before the GPU-resident scenarios; `archive_roi` prepares a new plan for every request and reports that time as its own stage. Each scenario verifies its output after the timed runs: GPU archives must equal the CPU archives byte for byte, decoded pixels must equal the source pixels, and every device result must report success. Calibrated output is checked by the numerical test suites, not here.

Input is either a synthetic detector frame (a smooth field with noise in the low bits, generated from `--seed`) or a two-dimensional `.npy` file. A batch of copies of one file is labeled `file_repeated`; it measures a batched operation and does not count as new data.

## Timing rules

- **Warm-up and samples.** Every process runs `--warmup` (default 5) untimed executions, then `--samples` (default 30, `run.py` warns below 30) timed samples.
- **Rounds.** `run.py` runs every case in `--rounds` (default 5) separate processes, interleaved with the other cases and in alternating order, and merges their samples. All samples are stored, together with the median, p05 and p95 (nearest rank), minimum, maximum, mean and standard deviation over all rounds, and the median and GPU state of every round.
- **Short operations.** If the fastest warm-up execution is shorter than 2 ms, one sample enqueues `repeat` back-to-back executions, with `repeat` chosen so that the sample lasts at least 2 ms (at most 1000). Stored samples are per operation. `repeat` is chosen per round and stored per round; the case records the smallest.
- **Device time.** CUDA events recorded in the stream of the measured work, before the first and after the last execution of a sample.
- **Host end-to-end time.** `std::chrono::steady_clock` from before the first enqueue until `cudaEventSynchronize` on the stop event has returned, so it ends only after the GPU has finished. With `repeat > 1` it is divided by `repeat` like the device time.
- **Stages and overlap.** `archive_roi` copies in one stream and decodes in another, so copies of later images overlap decodes of earlier ones. It reports the host stages `gather` and `prepare`, the event sums `h2d` and `decode_calibrate`, the device span `device_total` (first copy start to last decode end) and `host_end_to_end`. `stage_sum` adds all four stages and `device_overlap` is `h2d + decode_calibrate - device_total`. Totals are always measured, never computed from stages.
- **Separation from instrumentation.** `run.py` refuses a build that is not `Release`, a build with `-fsanitize` or `-pg`, and a process environment with profiler or sanitizer variables (`CUDA_INJECTION64_PATH`, `NSYS_PROFILING_SESSION_ID`, `LD_PRELOAD`, `ASAN_OPTIONS`, `COMPUTE_SANITIZER`). `--allow-instrumented` overrides this and is recorded in the result. Profilers are for diagnosis only.
- **Clocks are not locked.** The harness needs no privileges. It records the GPU clocks, performance state, temperature, power limit and clock event reasons before and after every case instead.

## Bytes, throughput and ratio

- **GB/s** are decimal (10^9 bytes per second) and always name their counter: `uncompressed input bytes` for encoding, `uncompressed decoded bytes` for decoding. For region scenarios the counter is the bytes of the requested regions in the stored pixel type, not the FP32 output and not the internally decoded tiles. Throughput is given for the whole-operation timings only (`device`, `host`, `device_total`, `host_end_to_end`), from the median.
- **Ratio** is uncompressed bytes divided by compressed bytes of the whole images. `ratio_payload` divides by the tile payloads only (including row descriptors); `ratio_container` divides by the complete `.tsc` files, including header and index.
- `transferred` is the compressed payload of the tiles the regions intersect, and `output` the bytes written by the operation.

## Memory

`workspace_bytes` is the size returned by `workspace_size` for the plan. `peak_device_bytes` is the high watermark of the default stream-ordered memory pool during the case. Every library allocation (plans, workspaces, `DeviceBuffer`) and every harness device buffer comes from that pool, so the watermark is the peak device allocation of the case. It does not include the CUDA context or loaded modules. `pinned_host_bytes` is the pinned host memory the scenario allocated.

## Recorded configuration

Each case records the scenario, the library build information (`tilestream::info_json`: version, build type, compiler, CUDA toolkit, driver and runtime versions, device), the dataset (source, path and SHA-256 for files, type, size, seed), tile geometry, batch size, region size, region seed and the regions themselves, samples, warm-up, rounds, `repeat`, CPU threads and CUDA streams. The run records the git commit and whether the tree had changes, the CMake cache (build type, compilers, flags, CUDA architectures), the host CPU, logical CPU count, frequency governor, operating system, the power supplies under `/sys/class/power_supply`, the GPU processes and display state before the run, the GPU state before and after every round of every case, and all `CUDA_*` environment variables. `run.py` warns when another process uses the GPU, because time-slicing with another context added about 2 ms to every synchronizing call in an early test run. A display attached to the measured GPU, or another process listed under `processes`, means the GPU was shared.

## Result schema

`benchmarks/schema.json` (JSON Schema 2020-12) defines a run file:

```text
schema        "tilestream-benchmark-run/1"
created       UTC time
command       run.py arguments
environment   git, build, host, power, gpu {before, after, processes}, instrumentation,
              cuda_variables
cases[]       id, args, rounds[] {gpu_state {before, after}, repeat, medians}, result
  result      schema "tilestream-benchmark-case/1", scenario, library, dataset, config, rois,
              bytes, memory, throughput_gbps, timings[]
    timings[] name, clock (cuda_event | cuda_event_sum | steady_clock | derived), unit "ms",
              count, median, p05, p95, min, max, mean, stdev, samples[]
```

The first timing of a case is its primary timing: `device` for GPU-resident scenarios, `host_end_to_end` for `archive_roi`, `host` for the CPU codec.

## Repeatability

`run.py compare FIRST SECOND` checks every case of the second run: its primary median must lie within the first run's p05-p95 range. It prints the relative change of each median and exits with status 1 if any median falls outside.

## Harness validation runs

Three full runs of the default suite (`results/benchmarks/t14-run1.json` to `t14-run3.json`, 5 rounds of 30 samples per case, 150 samples in total) on an RTX 4060 desktop GPU (driver 580.178.04, CUDA 13.0.88, GCC 13.3.0, Release, `sm_89`), with the display attached to the same GPU, no other compute process, clocks not locked and the CPU governor at `powersave`. Every pair of runs passed `compare`: 0 of 20 medians outside the other run's p05-p95 range, and no median changed by more than 2.2%. The tree had uncommitted changes, which the files record.

Rounds are needed for this. Within one process the samples are tight, often within 1%, but each new process can land on a different boost clock (2505 or 2730 MHz) and memory state (8251 or 8501 MHz), which moved the median by up to 8%. With a single round per case, 6 of 20 medians of a second run fell outside the first run's range; with three rounds 2 of 20 did. Five rounds per case is the default.

First run, primary timing per case (median and p05-p95 in ms, throughput from the median):

| Case | Timing | Median | p05 | p95 | GB/s | Ratio | Repeat | Peak MB |
|---|---|---|---|---|---|---|---|---|
| `gpu_encode/uint16/4096` | device | 0.7660 | 0.7618 | 0.7851 | 43.8 | 2.82 | 3 | 135.6 |
| `gpu_decode/uint16/4096` | device | 0.3043 | 0.2889 | 0.3120 | 110.3 | 2.82 | 7 | 116.1 |
| `gpu_decode_roi/uint16/4096/roi1024x8` | device | 0.1742 | 0.1740 | 0.1750 | 96.3 | 2.82 | 12 | 153.0 |
| `gpu_decode_calibrate/uint16/4096/roi1024x8` | device | 0.2487 | 0.2447 | 0.2492 | 67.5 | 2.82 | 8 | 169.8 |
| `archive_roi/uint16/4096/roi1024x4` | host_end_to_end | 1.0914 | 1.0824 | 1.1365 | 7.7 | 2.82 | 1 | 25.8 |
| `archive_roi/uint16/4096/roi256x1` | host_end_to_end | 0.0553 | 0.0513 | 0.0581 | 2.4 | 2.82 | 1 | 0.8 |
| `cpu_encode/uint16/2048` | host | 28.9089 | 28.5428 | 29.8876 | 0.3 | 2.82 | 1 | 0.0 |
| `cpu_decode/uint16/2048` | host | 7.7535 | 7.6502 | 7.9973 | 1.1 | 2.82 | 1 | 0.0 |
| `gpu_encode/float32/4096` | device | 1.0022 | 0.9919 | 1.2418 | 67.0 | 1.67 | 3 | 202.7 |
| `gpu_decode/float32/4096` | device | 0.4766 | 0.4711 | 0.5585 | 140.8 | 1.67 | 5 | 178.0 |
| `gpu_decode_roi/float32/4096/roi1024x8` | device | 0.2682 | 0.2675 | 0.2722 | 125.1 | 1.67 | 8 | 396.8 |
| `gpu_decode_calibrate/float32/4096/roi1024x8` | device | 0.2945 | 0.2936 | 0.2962 | 113.9 | 1.67 | 7 | 396.8 |
| `archive_roi/float32/4096/roi1024x4` | host_end_to_end | 2.3216 | 2.2924 | 2.3823 | 7.2 | 1.67 | 1 | 34.0 |
| `archive_roi/float32/4096/roi256x1` | host_end_to_end | 0.0784 | 0.0761 | 0.0809 | 3.3 | 1.67 | 1 | 1.0 |
| `cpu_encode/float32/2048` | host | 52.3221 | 51.9608 | 53.5674 | 0.3 | 1.64 | 1 | 0.0 |
| `cpu_decode/float32/2048` | host | 10.1307 | 9.9549 | 10.7520 | 1.7 | 1.64 | 1 | 0.0 |
| `gpu_encode/ct_small/x64` | device | 0.0414 | 0.0400 | 0.0416 | 50.6 | 1.71 | 47 | 8.5 |
| `gpu_decode/ct_small/x64` | device | 0.0281 | 0.0271 | 0.0281 | 74.7 | 1.71 | 68 | 7.8 |
| `gpu_encode/jacksboro_dem/x16` | device | 0.1208 | 0.1174 | 0.1216 | 73.5 | 1.19 | 17 | 29.5 |
| `gpu_decode/jacksboro_dem/x16` | device | 0.0550 | 0.0529 | 0.0552 | 161.2 | 1.19 | 36 | 28.5 |

`Ratio` is `ratio_container`. `Peak MB` is the device pool high watermark in decimal megabytes. The CPU codec allocates no device memory.

These numbers check the harness; they are not the performance analysis of TileStream. The small real images are batches of copies (`file_repeated`) whose working set fits in the 24 MiB L2 cache of this GPU and stays there between repeated executions, so their throughput is not comparable with the 4096x4096 cases.

`archive_roi` shows why stages are not summed. For four uint16 1024x1024 regions the medians were `gather` 0.259 ms, `prepare` 0.367 ms, `h2d` 0.294 ms and `decode_calibrate` 0.156 ms (stage sum 1.078 ms), `device_total` 0.460 ms and `host_end_to_end` 1.091 ms. `device_overlap` was -0.009 ms: although each decode waits only for its own copy, the trace showed every decode starting after the last copy had finished. With `CUDA_DEVICE_MAX_CONNECTIONS=1` in a separate check, `device_overlap` rose to 0.127 ms and `device_total` fell to 0.342 ms. The cause is not investigated here; `run.py` records all `CUDA_*` environment variables so that such settings are visible in a result.
