# Testing, fuzzing and sanitizers

This document maps the TileStream V1 requirements to tests, explains how to run every suite, and records the robustness runs: deterministic random round trips, structured mutations, libFuzzer, ASan/UBSan and Compute Sanitizer. The runs report how many inputs were checked and what they covered. They do not prove the absence of bugs.

## Running the tests

| Preset | Compiler and build | Purpose |
|---|---|---|
| `cpu-only` | default host compiler, Release, no CUDA | portable CPU codec, parser and TIFF suites |
| `cuda-debug`, `cuda-release` | GCC and nvcc, `sm_89` | all CPU and GPU suites; `cuda-debug` is the Compute Sanitizer build |
| `asan` | Clang, Debug, ASan and UBSan, no CUDA | host memory and undefined-behavior checks |
| `cuda-asan` | GCC, Debug, ASan and UBSan, with CUDA | host checks of the CUDA host code; the test preset sets `ASAN_OPTIONS=protect_shadow_gap=0`, which the CUDA driver needs under ASan |
| `fuzz` | Clang, RelWithDebInfo, libFuzzer instrumentation with ASan and UBSan | builds `tilestream_parse_fuzzer` and `tilestream_mutation_fuzzer` |

The presets find GoogleTest 1.14.0 through CMake; pass `-DGTest_DIR=...` when it is not installed system-wide.

```sh
cmake --preset cuda-release && cmake --build --preset cuda-release
ctest --preset cuda-release                  # everything
ctest --preset cuda-release -L codec         # bit-exact codec suites
ctest --preset cuda-release -L robustness    # random, mutation and fuzz-replay suites
tests/compute_sanitizer.sh build/cuda-debug results/robustness/compute-sanitizer
```

CTest labels: `cpu`, `cuda`, `codec`, `numerics`, `calibration`, `robustness`, `tiff`, `headers`. GPU suites return 77 and are reported as skipped when no CUDA device is available.

The random suites accept two environment variables. `TILESTREAM_CASES=N` changes the number of cases, and `TILESTREAM_CASE=seed` runs one case. Every failure message names its seed and the command that replays it, for example `TILESTREAM_CASE=13000042 build/cuda-release/tilestream_robustness_tests`.

## Test matrix

Section numbers refer to [requirements.md](requirements.md).

| Requirement | Tests | Scope |
|---|---|---|
| uint16 bit-exact round trip (4.1) | `CpuCodec.EveryUint16ValueAndSingleSampleRows`, `CpuCodec.HandCalculatedGoldenFiles`, `CudaEncoder.EveryUint16ValueAndSpecialFloatWords`, `CudaDecoder.SpecialFloatWordsAndEveryUint16Value`, `Robustness.RandomRoundTrips` | all 65,536 values, hand-calculated golden files, 5,000 random uint16 images |
| FP32 bit-exact round trip (4.1) | `Fp32Codec.SpecialRepresentationsSurviveAsBytes`, `Fp32Codec.ConstantsUseBitEquality`, `Fp32Codec.SeededRandomBitPatternsSelectRaw`, `CudaDecoder.SpecialFloatWordsAndEveryUint16Value`, `Robustness.RandomRoundTrips` | signed zeros, infinities, subnormals, quiet and signaling NaN payloads, arbitrary words, 5,000 random FP32 images |
| every residual width and final packed word | `CpuCodec.AllWidthsAndFinalPackedWords`, `Fp32Codec.EveryXorWidthAndFinalPackedWords`, `CudaEncoder.EveryResidualWidthAndFinalWord`, `CudaDecoder.EveryWidthPartialWarpsAndSingleSampleRows` | widths 0-16 (uint16) and 0-32 (FP32); the random suite also asserts that it reaches every width |
| one-sample rows, partial warps, word ends | `CudaDecoder.EveryWidthPartialWarpsAndSingleSampleRows`, `CpuCodec.EveryUint16ValueAndSingleSampleRows`, `Robustness.RandomRoundTrips` | widths 1, 2, 31-33, 63-65, 127-129, 255-257 |
| logical pixels and padding (4.2) | `CpuCodec.PitchPaddingAndPartialTiles`, `Fp32Codec.OrdinaryFieldsWithPitchAndEdges`, `Robustness.RandomRoundTrips`, `CudaRobustness.RandomRoundTripsOnEveryPath` | padded input pitch, sentinel output padding that must stay untouched, partial edge tiles |
| identical CPU and GPU bitstreams (4.2) | `CudaEncoder.ReproducesGoldenFiles`, `CudaEncoder.RandomBatchesMatchCpuBitstreams`, `CudaDecoder.FourEncoderDecoderCombinations`, `CudaRobustness.RandomRoundTripsOnEveryPath` | 10,000 random images encoded on both, compared byte for byte |
| host validation before device decode (4.3) | `CpuParser.*`, `Fp32Parser.RejectsInvalidWidthAndPadding`, `Robustness.MutatedFilesAreRejectedOrDecodedConsistently`, fuzz replays, `CudaDecoderValidation.RejectsInconsistentSources`, `CudaRobustness.FabricatedMetadataIsRejectedOrStaysInBounds` | every truncation, malformed fields, segment lengths, limits, 20,000 structured mutations, fabricated host metadata |
| trusted encoder output | `TrustedPath.ProducerErrorsArePropagatedPerImage`, `TrustedPath.CorruptIndexAndRowsAreReportedNotRead`, `CudaDecoderValidation.TrustedSourcesNeedCompleteShapes`, `CudaRobustness.RandomRoundTripsOnEveryPath` | producer errors, corrupt device index, random ROIs decoded from GPU encoder output |
| ROI (5) | `CudaDecoder.RandomRoisMatchCroppedFullDecode`, `CudaDecoder.RoiReadsOnlyIntersectingTiles`, `CudaDecoderValidation.RejectsInvalidRegionsAndViews`, `CudaRobustness.RandomRoundTripsOnEveryPath` | random and full-image ROIs, tile selection, invalid regions |
| calibration (6) | `CudaCalibration.*`, `CudaCalibrationValidation.*`, `Numerics.*`, `CudaNumerics.*` | see [numerical-analysis.md](numerical-analysis.md) |
| TIFF and BigTIFF (7) | `TiffAdapter.*`, `tests/tiff_interop.py` | tiled and strip files, unsupported tags, short reads |
| error semantics, no partial work (8) | `CpuCodec.InvalidViewsDoNotModifyOutputs`, `Fp32Codec.InvalidAlignmentAndTypeMismatchDoNotWrite`, `CudaEncoderValidation.*`, `CudaApi.*` | statuses before enqueue, untouched outputs, asynchronous completion, no hidden device synchronization |
| empty and mixed batches (9) | `CudaEncoder.EmptyBatchIsNoOp`, `CudaDecoder.EmptyBatchIsNoOp`, `CudaCalibration.EmptyBatchIsNoOp`, `CudaDecoder.MixedGeometriesAndWindows`, `CudaEncoder.ManySmallImagesInSharedAllocations`, `CudaRobustness.RandomRoundTripsOnEveryPath` | empty batches with null items, decode batches mixing all three tile geometries |
| sizes and overflow (3) | `tilestream.cpu.sizing` | metadata-only bounds and checked arithmetic |
| memory safety | `asan`, `cuda-asan` presets, `tests/compute_sanitizer.sh` | all CTest suites under ASan/UBSan, six GPU suites under four Compute Sanitizer tools |

## Deterministic random round trips

`tests/robustness/cases.h` derives each case from its seed alone. The default run uses seeds 13000000-13009999. Even case numbers are uint16, odd ones FP32, and the tile geometry cycles through 128x32, 64x64 and 256x16. Width (1-300) and height (1-160) fall on a warp, word or tile boundary half of the time. About a fifth of the images have row padding. Pixels follow one of eight patterns: constant, ramp, wrapping differences, rows with a chosen residual width, noise, a smooth field, IEEE special words, and a checkerboard with flipped bits. Each case also carries a random or full-image ROI.

`tilestream.cpu.robustness` checks, for every case, that the encoding fits `compress_bound` and is deterministic, that the parser accepts it and its structure matches an independent recomputation, and that decoding reproduces the pixels without writing into output padding. Over 10,000 cases the suite produced 17,341 RAW, 5,966 CONSTANT and 21,940 PREDICT_BITPACK tiles, 36,069 partial edge tiles, 73,621 rows of one-sample-wide tiles and 2,017 padded images, and asserts that it reached every residual width.

`tilestream.cuda.robustness` runs the same 10,000 cases in batches of up to 240 images of one type. It encodes them on the GPU and compares the bytes with the CPU files, fully decodes the CPU files with batches mixing all three geometries, and decodes each case's ROI from the GPU encoder output on the trusted path. Output rows have a padding sample filled with a sentinel that must survive.

## Structured mutations

The mutation harness in `fuzz/harness.h` is shared by `tilestream.cpu.robustness` and `tilestream_mutation_fuzzer`. Its first ten input bytes describe a valid image (type, geometry, size, pattern and seed). Each following 12-byte command mutates the canonical file of that image: a header field, an index entry field, a row descriptor field, a payload byte, the length of one tile payload with consistent shifts of all later offsets and of the file size, or the file size itself. A field is either replaced or moved by a small signed delta, so most mutations keep the rest of the file valid and reach the deeper checks.

`fuzz::check_file` then requires one of two outcomes. A rejection must use a documented status (`malformed_data`, `unsupported_format`, `unsupported_type`, `size_overflow` or `resource_limit`) and leave the metadata empty. An accepted file must have an independently consistent structure, decode without writing into row padding, and re-encode canonically to at most its own size with the same pixels. Inputs without commands must reproduce their original pixels exactly.

The suite runs 20,000 inputs with seed 1313. Every mutation kind produced both accepted and rejected files. All rejections were `malformed_data`; the accepted share ranged from 0.03% of inputs whose last command changed the file size to 41% of those whose last command flipped a payload byte, which inside a RAW tile only changes pixel values.

## Fuzzing

| Target | Input | Checks |
|---|---|---|
| `tilestream_parse_fuzzer` | arbitrary bytes as a `.tsc` file | `fuzz::check_file` |
| `tilestream_mutation_fuzzer` | an image description and up to eight mutation commands | `fuzz::check_mutation` |

Parsing uses `ParseLimits{1 MiB file, 32,768 tiles, 8 MiB metadata}`, and only accepted images up to 4 MiB are decoded, so every allocation is bounded.

```sh
cmake --preset fuzz && cmake --build --preset fuzz
build/fuzz/tilestream_parse_fuzzer -max_total_time=600 -max_len=65536 corpus-dir fuzz/corpus/parse
build/fuzz/tilestream_mutation_fuzzer -max_total_time=600 -max_len=106 corpus-dir fuzz/corpus/mutation
```

`fuzz/corpus/<target>` holds the minimized corpora (`-merge=1`). Minimal reproducers of fixed bugs belong in `fuzz/regressions/<target>`; the recorded runs found none, so the directory does not exist yet. Every build replays these directories through `fuzz/replay.cpp` as `tilestream.cpu.fuzz_parse` and `tilestream.cpu.fuzz_mutation`, without libFuzzer.

Recorded runs, on one CPU core each with ASan and UBSan, Clang 18.1.3:

| Target | Duration | Executions | Coverage edges | Corpus after merge | Crashes |
|---|---|---|---|---|---|
| parse | 601 s | 29,408,249 | 274 | 141 files | 0 |
| mutation | 601 s | 741,431 | 402 | 427 files | 0 |

The parse run started from the eight golden files and 36 canonical files of random cases. The mutation run started from 32 random image descriptions and from the corpus of an earlier 5-minute run with five mutation kinds (336,970 executions, 0 crashes), which was stopped when the splice mutation was added. Neither run found a crash, leak, sanitizer report or failed check. Coverage-guided fuzzing for ten minutes explores a small part of the input space; it raises confidence in the parser and decoder, it does not prove them correct.

## Sanitizers

ASan and UBSan run every CTest suite with `-fno-sanitize-recover=undefined`, so any report fails the test. `asan` (Clang 18, CPU-only) passed 9 of 9 tests and `cuda-asan` (GCC 13 with CUDA 13.0) passed 18 of 18.

`tests/compute_sanitizer.sh` runs memcheck (with `--leak-check full`), racecheck, synccheck and initcheck on the `encoder`, `decoder`, `api`, `calibration`, `numerics` and `robustness` suites of the `cuda-debug` build, with 1,000 random cases. All 24 runs reported 0 errors, 0 race hazards and 0 leaked bytes. Three exclusions are deliberate. `--report-api-errors no` hides the failed allocations that the API suite triggers on purpose. The `loading` suite and `CudaApi.NoHiddenDeviceSynchronization` are not run, because the sanitizer serializes streams, which breaks their overlap and synchronization assertions; both run in every ordinary build. Racecheck on the `api` suite skips the kernels of CUB's device scan (`--kernel-name-exclude kns=DeviceScan`): with them, `CudaApi.IndependentPlansFromTwoHostThreads`, which runs the encoder from two host threads on two streams, never finished in 3 of 20 racecheck runs, while it passed 20 of 20 runs without them, 10 of 10 runs under each other tool and 1,000 of 1,000 runs without a sanitizer. The likely cause is that the scan's blocks wait for earlier blocks, a forward-progress assumption that racecheck does not appear to honor for two concurrent scans; this was not confirmed further. The `encoder` suite still runs the scan under racecheck on one stream. Every run has a time limit (`SANITIZER_TIMEOUT`, default 1,800 s), so a hang fails the script instead of blocking it.

`CudaRobustness.FabricatedMetadataIsRejectedOrStaysInBounds` targets memcheck. It parses six valid files with all three tile modes, then changes one field of the parsed metadata 3,000 times (tile mode, size, offset, length, first segment, segment seed, width, offset and length, file size, or the number of tiles or segments) and prepares a random ROI decode. In the recorded run 550 attempts were rejected as `malformed_data` and 107 as `buffer_too_small`; 2,343 were accepted, mostly because the changed field belonged to a tile outside the ROI or was a value the decoder accepts, such as a seed. Every accepted plan executed without a CUDA error and without a memcheck report.

The checks were tested against deliberate defects that were then reverted. Without the parser's check that segments consume the whole tile, the mutation suite fails with "accepted tile 1 has a wrong length". Without the segment payload range checks in `prepare_batch`, memcheck reports out-of-bounds reads in the fabricated-metadata test.

Logs of these runs are in `results/robustness/`.

## Responsibility boundary

Untrusted `.tsc` bytes are validated on the host by `parse_metadata` before any device decode. It checks the header, the index, every tile offset, length, mode and padding, every row descriptor, bit width and seed range, the packed payload length and its zero padding bits, and that the payloads exactly cover the file. The CPU decoder parses the file itself.

The device API receives metadata produced by a successful parse, unmodified, together with the exact validated payload bytes, as stated in [api.md](api.md). `prepare_batch` checks again, on the host, every tile it uploads: shape, offset and length within the file, mode, and the offset, length and bit width of every row segment. These checks keep device reads inside the supplied bytes even for fabricated metadata, but they do not authenticate the payload. Bytes changed after validation, for example nonzero padding bits or different residuals, may decode to wrong pixels without an error.

The trusted encoder-output path accepts only the output of a TileStream encoder under the stated stream dependencies. Execution checks the device index and rows it reads and reports inconsistencies as `malformed_data` in the result record, but the path is not hardened against arbitrary modification of that output.
