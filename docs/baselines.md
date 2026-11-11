# Baselines

TileStream is compared with uncompressed transfer, zstd on the CPU, TIFF Deflate through libtiff and nvCOMP LZ4 on the GPU. All of them run in `tilestream_codec_benchmark` with the timing rules of [benchmarking.md](benchmarking.md). The zstd, libtiff and nvCOMP adapters are benchmark code (`benchmarks/baselines.cpp`, `benchmarks/nvcomp_baseline.cpp`); the library does not link zstd or nvCOMP.

## Boundaries

Every comparison has one boundary, and results are only compared within a group.

| Group | Start | End | Scenarios |
|---|---|---|---|
| Host to result | pixels or compressed data in host memory, metadata and plans prepared | calibrated FP32 image in device memory | `raw_calibrate`, `tsc_calibrate`, `zstd_calibrate`, `lz4_calibrate` |
| Device codec | input in device memory | output in device memory | `gpu_encode`, `gpu_decode`, `lz4_gpu_compress`, `lz4_gpu_decompress` |
| File to result | file in the page cache | calibrated FP32 image in device memory | `raw_file`, `tsc_file`, `tiff_file` |

- **`raw_calibrate`**: pixels in pinned memory, one copy per image, `calibrate_async` on the batch.
- **`tsc_calibrate`**: `.tsc` files in pinned memory, one copy of every whole file, `decode_calibrate_async` on the batch. The files were parsed and the plan prepared before the measurement, as for every other scenario of this group.
- **`zstd_calibrate`**: the same pixels compressed into independent zstd frames of 1 MiB of raw data. `--threads` persistent threads decompress the frames straight into one pinned staging buffer, then the pixels are copied and calibrated as in `raw_calibrate`. Decompression and transfer do not overlap.
- **`lz4_calibrate`**: the pixels of every image split into fixed-size chunks and compressed once with nvCOMP batched LZ4. The compressed chunks are packed into one aligned stream in pinned memory; one copy of the stream, `nvcompBatchedLZ4DecompressAsync` into the raw pixel buffers, then `calibrate_async`.
- **`lz4_gpu_compress`, `lz4_gpu_decompress`**: the nvCOMP calls alone, with chunk pointer and size arrays already on the device, next to `gpu_encode` and `gpu_decode`.
- **`raw_file`, `tsc_file`, `tiff_file`**: one file per image, written before the measurement and read back from the page cache in every operation (`cache: warm` in the result). `raw_file` and `tsc_file` read with `fread` into pinned memory; `tsc_file` then parses the metadata and calls `prepare_batch` in every operation. `tiff_file` reads and decodes with `tilestream::io::read_tiff` (libtiff), which returns pageable memory, and copies from there. Cold-cache reads are not measured here.

All four host-to-result scenarios end in the same calibration (`offset = 100`, `gain = 1.25`) and use the same batch. Each operation runs the stages in order in one stream and waits for it, so the timings contain no overlap between operations. The stages are reported next to the total: host stages with `steady_clock`, device stages with events. `device_total` is reported only when every stage is device work.

## Same values

After the timed runs every scenario of the host-to-result and file-to-result groups downloads the FP32 output and compares every sample bit for bit with the host calibration of the source pixels, `(x - offset) * gain` in binary32 without contraction, which is the semantics of the library. The device codec scenarios compare decoded pixels with the source pixels, and `lz4_gpu_compress` must reproduce the chunk sizes of the first compression. A changed calibrated sample (tested by altering the host reference for one input value) fails `raw_calibrate`, `zstd_calibrate`, `lz4_calibrate` and `tiff_file`. CTest runs every baseline on a batch of two 1000x777 images of both types (`ctest -L baseline`).

## Byte accounting

The ratio is always uncompressed bytes divided by the stored bytes of the method.

| Method | `payload` | `container` |
|---|---|---|
| raw | pixels | pixels |
| TileStream | tile payloads and row descriptors | `.tsc` files |
| zstd | zstd frames | frames plus an 8-byte offset per frame |
| nvCOMP LZ4 | compressed chunks | aligned packed stream plus an 8-byte size per chunk (the batched API has no container) |
| TIFF Deflate | `TileByteCounts` | TIFF files |

`transferred` is what crosses PCIe: pixels for raw, zstd and TIFF, whole files for TileStream, the packed stream for LZ4. `pinned_host_bytes` is the pinned memory the scenario allocated; `peak_device_bytes` includes nvCOMP temporary memory, which comes from the same stream-ordered pool.

## Installation check

Before measuring, the adapters compare the run-time libraries with the headers they were built with and stop on a mismatch: `ZSTD_versionNumber()`, `TIFFGetVersion()`, and `nvcompGetProperties()` for the nvCOMP version and the CUDA major version it was built for. The versions are written to the `method` object of each result. nvCOMP comes from the PyPI wheel `nvidia-libnvcomp-cu13` 6.0.0.12 (see `config/dependencies.json`):

```sh
python3 -m pip download --no-deps nvidia-libnvcomp-cu13==6.0.0.12 -d build/deps
unzip -q build/deps/nvidia_libnvcomp_cu13-6.0.0.12-*.whl -d build/deps/nvcomp-6.0.0.12
cmake --preset cuda-release -DTILESTREAM_ENABLE_NVCOMP=ON \
  -Dnvcomp_DIR=$PWD/build/deps/nvcomp-6.0.0.12/nvidia/libnvcomp/lib64/cmake/nvcomp
cmake --build --preset cuda-release
benchmarks/run.py run --suite baselines --output results/baselines/run.json
benchmarks/run.py table results/baselines/run.json
```

The wheel was built against the CUDA 13.4 runtime (`cudart_version` 13040) and runs on the 13.0 runtime and driver 580 through CUDA minor version compatibility.

## nvCOMP configurations

The batched LZ4 API leaves the chunk size, the data type and bitshuffle to the caller. A sweep on the 4096x4096 frames (`results/baselines/lz4-sweep.txt`, one process and 10 samples per point, exploratory only) chose three configurations, all with the CUDA backend:

| Name | Chunk | Data type | Bitshuffle | Why |
|---|---|---|---|---|
| `controlled` | one TileStream tile (8 KiB uint16, 16 KiB float32) | `char` | none | same unit of work as TileStream with the API defaults |
| `default64k` | 64 KiB | `char` | none | the chunk size the nvCOMP documentation recommends |
| `tuned` | one tile | `USHORT` / `UINT` | MSB first | the best ratio and, for uint16, the best speed of the sweep |

Larger chunks compress slightly better without bitshuffle but leave too few chunks for the GPU: a 32 MiB frame in 1 MiB chunks is 32 chunks and decompresses at 1.4 GB/s. The chunks are linear ranges of the row-major image; TileStream tiles are 2-D, so the controlled configuration matches the amount of data per unit, not the geometry. The `tuned` configuration was chosen on the same frames it is measured on, which favors nvCOMP.

## Decompression engine

nvCOMP can decompress on a dedicated hardware engine where the GPU has one. On the RTX 4060 `CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK` is 0, and `--backend hardware` fails with `nvcompErrorCannotDecompress` (status 12). Every LZ4 result therefore uses `NVCOMP_DECOMPRESS_BACKEND_CUDA`, that is CUDA kernels, recorded as `backend` in the result. `--backend default` gives the same CUDA path. A result with the hardware engine would have to be reported separately and would not be a comparison of CUDA kernels.

## Results

RTX 4060 (PCIe 4.0 x8, driver 580.178.04, CUDA 13.0.88), i7-12700KF, synthetic 4096x4096 detector frames, batch 1, 5 rounds of 30 samples (`results/baselines/t15-run1.json`). Ratio is `ratio_container`; GB/s counts the uncompressed bytes of the image. Peak device memory includes all buffers of the case, also the raw pixel buffers that nvCOMP and zstd decompress into.

Host to calibrated FP32 on the device (`host_end_to_end`):

| Method | uint16 ratio | uint16 ms | uint16 GB/s | float32 ratio | float32 ms | float32 GB/s | Peak device MiB (u16 / f32) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1.000 | 3.32 | 10.1 | 1.000 | 6.21 | 10.8 | 96 / 128 |
| TileStream | 2.818 | 1.45 | 23.1 | 1.666 | 3.93 | 17.1 | 143 / 170 |
| zstd -1, 1 thread | 1.344 | 27.40 | 1.2 | 1.205 | 41.37 | 1.6 | 96 / 128 |
| zstd -1, 8 threads | 1.344 | 7.46 | 4.5 | 1.205 | 11.75 | 5.7 | 96 / 128 |
| zstd -3, 1 thread | 1.783 | 34.03 | 1.0 | 1.210 | 43.66 | 1.5 | 96 / 128 |
| zstd -3, 8 threads | 1.783 | 9.36 | 3.6 | 1.210 | 12.12 | 5.5 | 96 / 128 |
| LZ4 controlled | 1.044 | 4.50 | 7.5 | 0.997 | 7.47 | 9.0 | 287 / 385 |
| LZ4 default64k | 1.166 | 4.79 | 7.0 | 0.999 | 7.19 | 9.3 | 172 / 288 |
| LZ4 tuned | 2.633 | 2.23 | 15.1 | 1.809 | 5.12 | 13.1 | 301 / 420 |

The host-to-device copy runs at about 12 GB/s in every scenario, so on this PCIe link the time follows the transferred bytes: TileStream copies 11.9 MB of uint16 archive in 0.99 ms and decodes with calibration in 0.46 ms, against 2.77 ms of copy and 0.54 ms of calibration for raw pixels. The zstd times are dominated by CPU decompression (24 ms on one thread for uint16, 3.9 ms on eight). zstd and LZ4 without bitshuffle compress these frames poorly because the noise sits in the low byte of every sample; bitshuffle with the element type recovers most of it.

Device-resident codec (`device` time):

| Method | uint16 ratio | encode GB/s | decode GB/s | float32 ratio | encode GB/s | decode GB/s | Peak device MiB (u16 / f32) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TileStream | 2.818 | 43.3 | 116.2 | 1.666 | 64.0 | 141.5 | 111-129 / 170-193 |
| LZ4 controlled | 1.044 | 8.9 | 28.8 | 0.997 | 4.6 | 57.2 | 287 / 385 |
| LZ4 default64k | 1.166 | 3.2 | 19.3 | 0.999 | 3.2 | 74.2 | 172 / 288 |
| LZ4 tuned | 2.633 | 17.6 | 49.9 | 1.809 | 16.6 | 50.3 | 301 / 420 |

LZ4 on float32 frames without bitshuffle stores the data almost uncompressed, which explains its high decode rate at a ratio below 1. The nvCOMP temporary memory grows with the number of chunks: 128 MiB for 4096 chunks without bitshuffle and 160 MiB with it.

File in the page cache to calibrated FP32 on the device (`host_end_to_end`):

| Method | uint16 ratio | uint16 ms | float32 ratio | float32 ms | Largest stage |
|---|---:|---:|---:|---:|---|
| raw file | 1.000 | 6.63 | 1.000 | 12.97 | `read` 3.26 / 6.69 ms |
| TileStream file | 2.818 | 6.73 | 1.666 | 15.32 | `parse_prepare` 3.99 / 7.34 ms |
| TIFF Deflate | 2.344 | 90.84 | 1.446 | 149.54 | `read_decode` 87.0 / 142.3 ms |

TIFF uses Deflate level 6 with the horizontal predictor for uint16 and no predictor for float32 (the adapter supports the predictor for integer samples only), tiles of 128x32. Its time is almost all libtiff decoding on one CPU thread and must not be compared with kernel-only numbers. For TileStream files the parser and `prepare_batch` (4096 tiles with 32 row descriptors each) take longer than reading the file, so the file path is no faster than raw files although it reads and transfers 2.8 times fewer bytes. This is the first target of the profiling stage.

## Limitations

- One GPU with a display attached, unlocked clocks and a desktop CPU with P- and E-cores; zstd thread counts above the 8 P-cores were not measured.
- Synthetic frames only.
- Each operation is synchronous: no overlap between decompression, transfer and calibration of different chunks or images. A pipelined zstd or LZ4 path could hide part of the copy; the stage times show the upper bound of that gain.
- Warm page cache only; the `read` stages measure memory copies from the cache, not the disk.
- zstd frames of 1 MiB were fixed before measuring; larger frames compress slightly better and parallelize worse.
- nvCOMP 6.0.0.12 from the PyPI wheel, built for CUDA 13.4 and running on the 13.0 runtime.

## Repeatability

A second full run (`t15-run2.json`, `t15-compare-1-2.txt`) put 38 of 40 primary medians inside the p05-p95 range of the first run. The two outside were `lz4_gpu_compress/float32/4096/controlled` (+5.7%) and `raw_file/float32/4096` (+4.1%). The largest change of a median, -17.8% for zstd level 3 on eight threads, stayed inside the range: the multi-threaded zstd cases have a wide spread (p95 up to 1.8 times the median), most likely from the scheduler placing threads on P- or E-cores. Differences within a few percent between methods are not significant; the conclusions above rest on differences of 1.5 times or more.
