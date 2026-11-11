# TileStream V1 Algorithm Review and Comparison Plan

## 1. Purpose

TileStream is a tiled, lossless codec for single-channel `uint16` and IEEE 754 binary32 images. Its main hypothesis is narrower than general compression: independent tiles may reduce transfer and memory costs when an application requests only a region of a large image and then calibrates that region on the GPU.

The project does not claim a new universal compression method. Its original work is the combination of a versioned tile container, deterministic mode selection, row-local reversible transforms, CPU and CUDA implementations, ROI decoding, and an image-processing API with explicit memory and stream contracts.

This review fixes the V1 algorithm scope before the format and codec are implemented. It also defines how external libraries will be compared without presenting their algorithms as TileStream work.

## 2. Building blocks

### 2.1 Modular delta and zigzag for `uint16`

For each real row segment in a tile, TileStream stores the first sample as a seed. Every later sample is replaced by its difference from the previous sample modulo 2^16. The unsigned modular difference is then interpreted in the range `[-32768, 32767]` and mapped to an unsigned value by zigzag.

Zigzag maps small positive and negative deltas to small unsigned integers. A smooth ramp with deltas near zero can therefore use a small bit width even when its absolute pixel values are large. Delta does not compress by itself. Its purpose is to reduce the range presented to bit-packing, matching the role described for delta in nvCOMP Cascaded [S2].

The transform is defined with unsigned and wider arithmetic. Signed overflow and implementation-defined right shifts of negative values are not part of its correctness argument.

Decode has a prefix dependency: each reconstructed value depends on the seed and all earlier deltas in the segment. The CPU oracle may perform a simple serial accumulation. CUDA decode can use a segmented prefix sum. The dependency is intentionally limited to one real image row inside one tile. It never crosses a row, tile, or image. Consequently:

- decoding a sample in the middle of a predictive row cannot begin from that sample alone;
- decoding an ROI may still decode the intersecting part of a complete tile row;
- different rows and tiles remain independent and can be scheduled in parallel;
- corruption or invalid metadata must be rejected before a prefix scan is launched.

### 2.2 XOR prediction for FP32

FP32 samples are treated only as 32-bit words. The first word of each row segment is a seed. Each later residual is the bitwise XOR of the current and previous words. Reconstruction is a prefix XOR beginning with the seed.

This operation is bit-exact for every binary32 representation, including infinities, subnormal values, signed zero, and all NaN payloads. It performs no floating-point arithmetic and does not canonicalize NaNs.

XOR can reduce the highest set bit when adjacent representations share sign and exponent bits. The V1 bit-packer only removes unused high bits from the residual range. It does not encode separate leading-zero and trailing-zero windows. One residual with a high set bit can therefore force a wide encoding for its complete row segment. Sign changes, exponent changes, arbitrary NaN payloads, and random bit patterns often lead to width 32 and should select `RAW`.

Like modular delta, XOR reconstruction has a prefix dependency. It uses an associative XOR scan, but random access remains row-segment granular rather than sample granular.

### 2.3 Bit-packing

For each predictive segment, TileStream finds the maximum residual and stores the minimum number of low bits required to represent every residual in that segment. Width zero represents all-zero residuals. Width 16 for `uint16` and width 32 for FP32 are valid full-width cases and must not use a shift by the word width.

Bit-packing saves space when `b`, the stored width, is materially smaller than the original width `B`. The approximate payload ratio `B / b` is only a first estimate. Seeds, segment descriptors, tile headers, 32-bit payload alignment, edge dimensions, and the tile index are included in actual mode selection and published ratios.

Symbols can cross 32-bit word boundaries. Every output word needs a single owner or a proven synchronization scheme. The decoder must not read beyond the final allocated word, even when the last symbol straddles a logical boundary. nvCOMP documents the same general word-straddling issue for Cascaded bit-packing [S2], but TileStream defines and implements its own bitstream.

### 2.4 Run-length encoding

RLE replaces each consecutive run by a value and a run length. It is effective for long repeated runs. It can expand a sequence with no duplicates because every input value becomes a value-length pair. NVIDIA documents a worst case expansion by a factor of two for this basic representation [S2].

V1 does not contain a general RLE tile mode. `CONSTANT` is the useful degenerate case in which an entire tile contains the same bit pattern and only one value is needed. Adding general RLE would require another descriptor grammar, output bound, GPU compaction path, malformed-stream validation, and comparison policy. Cascaded already supplies an external baseline for combinations involving RLE.

### 2.5 LZ4

LZ4 is a general byte-oriented LZ scheme. It emits literals and references to earlier byte sequences. It can exploit repeated byte strings even when they are not aligned to pixel or row boundaries. NVIDIA describes nvCOMP LZ4 as a general-purpose, no-entropy byte compressor and notes that its incompressible expansion is bounded by its token design [S1, S5].

TileStream prediction instead uses known image order and fixed-width numeric residuals. It has no backward-match search or dictionary. LZ4 can win on repeated textures, recurring byte sequences, metadata-like streams, or data whose local numeric differences are poor. TileStream can be simpler to decode per independent image tile and can expose direct tile indexing without wrapping one large LZ stream.

### 2.6 Deflate

Deflate combines LZ77 matching with Huffman entropy coding [S1]. It can exploit repeated byte strings and non-uniform symbol frequencies that V1 bit-packing does not model. This usually brings more serial and irregular work than fixed-width unpacking, and its performance depends on implementation and compression level.

TileStream does not implement Deflate. libtiff owns TIFF Deflate compression and decompression. `TIFFReadEncodedTile` writes decompressed bytes into a caller buffer and returns the actual byte count or `-1` [S7]. `TIFFWriteEncodedTile` accepts raw bytes, compresses them according to the configured TIFF codec, appends the encoded tile data, and returns the input size or `-1` [S8]. The word "Encoded" in these APIs does not mean that the application receives or supplies compressed bytes.

TIFF horizontal prediction creates a row-order dependency before Deflate, conceptually similar to a row-local delta transform. libtiff owns that transform and its reversal. TileStream must check returned byte counts, tile sizes, byte swapping, and all relevant TIFF tags rather than treating a TIFF file as a raw byte container.

## 3. V1 tile modes

| Mode | Stored information | Expected strengths | Expected losses |
|---|---|---|---|
| `RAW` | Logical tile pixels without synthetic edge padding | Noise, random bits, high residual widths, very small tiles where headers dominate | No payload compression; complete file still has container overhead |
| `CONSTANT` | One exact pixel bit pattern and tile dimensions | Blank regions, masks, saturated areas, repeated FP32 bit patterns | Any differing bit disqualifies the mode |
| `PREDICT_BITPACK` for `uint16` | Per-row seed, width, aligned modular-delta payload | Smooth fields, ramps, slowly varying sensor images | Noise, alternating extremes, abrupt changes, a single wide residual per segment |
| `PREDICT_BITPACK` for FP32 | Per-row bit seed, width, aligned XOR payload | Adjacent values with stable high representation bits | Sign or exponent changes, random mantissas, NaN payload variation, unrelated bits |

The encoder compares complete candidate sizes, including mode headers, row seeds, descriptors, and alignment. `RAW` wins an exact tie. The remaining tie rule will be fixed in `docs/format.md`. Tile payload selection limits tile-level expansion, but the `.tsc` header and index can still make the complete file larger than the uncompressed pixels.

## 4. How the methods differ

| Property | TileStream predictive mode | Basic RLE | LZ4 | Deflate | nvCOMP Cascaded |
|---|---|---|---|---|---|
| Primary redundancy | Small row-local numeric or bitwise changes | Consecutive equal values | Repeated byte strings | Repeated byte strings and symbol frequency | Structured integer values, configurable RLE, delta, FOR, bit-packing |
| Entropy coding | No | No | No | Huffman | Terminal bit-packing in the documented modes; nvCOMP also offers separate entropy codecs |
| Decode dependency | Prefix scan within one row segment | Run prefix positions and run expansion | Back-references within an independent block | LZ77 history plus Huffman decode | Depends on selected stages; delta decode uses a prefix sum |
| Random-access unit in this project | Independently indexed tile | Not a V1 mode | Controlled external chunks | TIFF tile or strip | Controlled external chunks |
| FP32 interpretation | Exact 32-bit XOR transform | Exact equality if applied to words | Opaque bytes | Opaque bytes | Logical integer types; FP32 bytes may be reinterpreted only as an explicitly reported baseline |
| Incompressible behavior | Select `RAW` per tile | Can expand substantially | Small bounded format expansion as described by NVIDIA | Implementation and block dependent | Stage and chunk dependent |

Cascaded is related conceptually but is not TileStream. Current nvCOMP documentation describes configurable combinations of RLE, Delta, Frame of Reference, and terminal bit-packing. Its asymmetric mode retains optional stages per internal chunk when they reduce encoded size, while symmetric mode balances compression and decompression throughput [S2]. TileStream uses fixed image-aware row segments, modular 16-bit delta or FP32 XOR, three tile modes, and its own container and index.

## 5. Comparison with NVIDIA libraries

### 5.1 nvCOMP

nvCOMP supplies GPU compression interfaces and implementations including LZ4, Deflate-compatible formats, Zstandard, Cascaded, Bitcomp, and separate entropy-oriented formats [S1]. TileStream will use nvCOMP LZ4 as the required GPU baseline. A second nvCOMP algorithm is optional and must be justified by the data and supported environment rather than selected only because it produces a favorable comparison.

A fair nvCOMP comparison requires two chunking views:

1. controlled chunks corresponding as closely as practical to TileStream tile payloads;
2. a documented nvCOMP-oriented chunk size chosen on tuning data.

The same canonical pixel bytes, batch composition, device residency, and correctness checks apply to both. Reinterpreting FP32 words as integers for Cascaded is allowed only as an explicitly labeled experiment. It does not make Cascaded an FP32 image codec.

The `.tsc` format is not an nvCOMP bitstream. TileStream will not call nvCOMP inside its own `RAW`, `CONSTANT`, or `PREDICT_BITPACK` implementation. nvCOMP code, metadata, workspace, and results remain in a separate benchmark adapter.

The current nvCOMP installation page requires Volta or newer and gives different CUDA requirements for dynamic and static CUDA 12 and CUDA 13 packages [S3]. The development machine has an RTX 4060 and CUDA 13.0.88, but nvCOMP is not installed. At baseline integration time, the exact nvCOMP package, dynamic or static variant, operating-system support, driver, and tested version must be recorded. Compatibility must not be inferred only from the GPU architecture.

### 5.2 nvImageCodec

nvImageCodec is an accelerated image-codec framework with a unified API, extension modules, format parsers, batch support, Python bindings, fallback selection, and integrations with libraries such as nvJPEG, nvJPEG2000, and nvTIFF [S4]. It addresses established image formats and codec plugin dispatch.

TileStream V1 addresses a custom independently tiled scientific-data container, direct ROI selection, exact FP32 bit preservation inside that container, user-owned CUDA buffers, and fused calibration. It is not an nvImageCodec plugin, and `.tsc` is not an nvImageCodec, nvTIFF, JPEG, or JPEG 2000 format. TIFF interoperability is provided through a separate libtiff adapter. A future plugin would require a separately approved scope and is not necessary to validate the codec hypothesis.

## 6. Data-dependent expectations

| Input characteristic | Likely TileStream outcome | Likely competing outcome |
|---|---|---|
| Constant tile | `CONSTANT` should be smallest | RLE and general codecs can also do well, with different headers |
| Smooth `uint16` rows | Small zigzag values and narrow bit widths | Cascaded delta plus bit-packing may also do well; LZ4 depends on byte repetition |
| Repeating texture with large local jumps | Predictive widths may be large | LZ4 or Deflate can exploit repeated byte strings |
| Alternating `uint16` extremes | Width near 16, usually `RAW` | Dictionary codecs may still compress if the byte pattern repeats |
| Slowly changing positive FP32 values | XOR may remove stable high bits | General codecs may or may not find byte repetition |
| FP32 values crossing signs or exponents | One residual can force width 32 | Byte-oriented codecs can win if longer patterns repeat |
| Random `uint16` or random 32-bit words | `RAW`, with container overhead | Every lossless codec should approach raw size and add processing cost |
| Small edge tiles | Headers and 32-bit alignment dominate | Larger external chunks can have lower relative overhead |
| Small, unaligned ROI | Only intersecting tiles are read, but read amplification may be high | Whole-stream formats can require much more decoding; independently chunked baselines reduce this difference |

A historical NVIDIA transfer study found both substantial wins on suitable structured columns and slowdowns on poorly compressible floating-point columns because compression and decompression added work without enough transfer savings [S5]. Those measurements used different data and hardware. They motivate measuring the complete TileStream path but are not performance targets.

## 7. Benchmark comparison plan

### 7.1 Frozen inputs and configurations

- Use canonical logical pixels from `datasets/manifest.json`.
- Select tile geometry, mode rules, nvCOMP chunk settings, zstd levels, and TIFF settings using only the tuning split.
- Run the frozen evaluation split without selecting parameters per image.
- Report synthetic and real data separately. Repetition used only to create a larger workload is labeled `synthetic_stress`.

### 7.2 Correctness gate

Before timing, every method must reconstruct the same `uint16` values or FP32 bits within its declared contract. TIFF limitations for special FP32 representations are tested and reported separately. A method that fails correctness is not assigned a throughput result.

### 7.3 Required baselines

| Baseline | Scope | Required controls |
|---|---|---|
| Raw | Pinned host-to-device transfer followed by the same calibration | Same batches, output, stream completion, and map residency |
| TileStream | Own CPU and CUDA paths | Complete container and payload sizes; selected tile modes |
| zstd CPU | CPU decompression, transfer, calibration | Same canonical bytes; fixed levels 1 and 3 unless tuning justifies a change; thread count reported |
| TIFF Deflate | File-to-result | Parser, libtiff decode, transfer, and calibration included; TIFF settings reported |
| nvCOMP LZ4 | GPU-resident codec and applicable transfer path | Same bytes; controlled tile-like chunks and tuned chunks reported separately |
| Optional second nvCOMP codec | Only with a written rationale | Same boundaries and no intentionally weak competitor configuration |

### 7.4 Timing boundaries

Three scenarios remain separate:

1. Archive/ROI begins with an existing compressed archive and ends with calibrated FP32 output ready on the GPU. Archive creation is reported separately.
2. GPU-resident codec begins with uncompressed input already on the GPU and measures encode and decode without file I/O.
3. Cold file-to-result begins with a file and includes parsing, I/O, decompression, transfer, and calibration. Cold and warm cache states are reported separately.

Device intervals use CUDA events with explicit dependencies. Host end-to-end time stops only after actual completion. Overlapping kernels and copies are not summed as if serialized. If an exact compressed size requires host synchronization, that synchronization is included or clearly separated. NVIDIA's transfer example similarly notes that compressed size availability can introduce a stream synchronization unless later work remains device-driven [S5].

### 7.5 Reported quantities

- full-container ratio and payload-only ratio;
- encode and decode throughput using uncompressed input bytes and output bytes respectively;
- median, p95, all raw samples, warm-up policy, and batch size;
- host-to-device bytes, tile read amplification, and ROI dimensions;
- workspace and peak device allocation;
- kernel-only, staged device, and end-to-end times as distinct values;
- exact hardware, software, source commit, and cache state;
- failures, unsupported configurations, and cases where raw transfer wins.

The central break-even test is:

`compressed transfer + decode + process < raw transfer + process`

When data is compressed on demand, encode and all additional copies are included.

## 8. Why V1 excludes entropy coding and full JPEG

### 8.1 Entropy coding

An entropy coder could exploit skewed residual frequencies after prediction, but it would add codebooks or probability state, variable-rate symbol boundaries, more complex output bounds, more branches and synchronization, additional malformed-input cases, and less direct parallel random access. V1 first tests whether fixed-width packing plus tile independence is sufficient to improve the measured ROI pipeline.

Deflate, zstd, and nvCOMP provide entropy-aware or general-purpose baselines without hiding that complexity inside TileStream. A later entropy stage is justified only by measured residual distributions and end-to-end results, not by ratio alone.

### 8.2 JPEG and JPEG 2000

Conventional JPEG is primarily a lossy transform codec and does not satisfy the required exact `uint16` and arbitrary FP32 bit-pattern contract. JPEG and JPEG 2000 also bring substantial format, marker, transform, entropy, metadata, color, precision, and interoperability scope unrelated to the narrow V1 hypothesis. Implementing an incomplete version would weaken both correctness and comparison quality.

Existing libraries already provide established image codecs. nvImageCodec explicitly exposes accelerated JPEG, JPEG 2000, and TIFF-related codec integrations [S4]. TileStream therefore uses libtiff for the required TIFF subset and compares external compressors rather than claiming to replace standard image formats.

## 9. Decisions fixed for subsequent stages

- V1 tile modes remain `RAW`, `CONSTANT`, and `PREDICT_BITPACK`.
- General RLE, entropy coding, dictionary matching, and lossy transforms are not added to the core codec.
- Prediction state resets at every real tile row.
- CPU and CUDA implementations must produce the same deterministic mode decisions and byte stream.
- Complete candidate size, not residual width alone, determines the selected mode.
- nvCOMP and libtiff remain external adapters and are identified as dependency-provided work.
- Negative results on FP32, noisy data, small tiles, or ROI read amplification are reportable outcomes, not reasons to alter the evaluation split.

## 10. Sources

Accessed 2026-10-19.

- **[S1] NVIDIA nvCOMP documentation, overview and algorithms.** https://docs.nvidia.com/cuda/nvcomp/
- **[S2] NVIDIA nvCOMP, Cascaded Compression.** https://docs.nvidia.com/cuda/nvcomp/cascaded.html
- **[S3] NVIDIA nvCOMP, Installation.** https://docs.nvidia.com/cuda/nvcomp/installation.html
- **[S4] NVIDIA nvImageCodec developer page.** https://developer.nvidia.com/nvimagecodec
- **[S5] NVIDIA Technical Blog, Optimizing Data Transfer Using Lossless Compression with NVIDIA nvCOMP.** https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/
- **[S6] NVIDIA CUDA C++ Best Practices Guide.** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **[S7] LibTIFF, TIFFReadEncodedTile.** https://libtiff.gitlab.io/libtiff/functions/TIFFReadEncodedTile.html
- **[S8] LibTIFF, TIFFWriteEncodedTile.** https://libtiff.gitlab.io/libtiff/functions/TIFFWriteEncodedTile.html

S1 through S5 support the scope and comparison decisions. S7 and S8 fix the semantics expected from the future TIFF adapter. S6 is included now as the required reference for the later measurement and optimization implementation; no performance conclusion in this review is inferred without a benchmark.
