# TileStream V1 Requirements

## 1. Purpose and status

This document defines the functional and correctness contract for TileStream V1. It is normative for implementation and tests. Format layout and public API details will be specified separately, but those specifications must preserve the behavior defined here.

The terms MUST, MUST NOT, SHOULD, and MAY describe requirements in descending order of strength.

## 2. Supported image model

TileStream V1 supports:

- two-dimensional, single-channel images;
- `uint16`, interpreted as an unsigned 16-bit integer;
- `float32`, interpreted as an IEEE 754 binary32 bit pattern;
- one image or a batch of images;
- images with different dimensions and pitches in one batch;
- homogeneous pixel type within one batch operation.

RGB, multi-channel images, dimensions other than 2D, and pixel types other than `uint16` and `float32` are unsupported.

An image has the following logical properties:

- `width`: number of pixels in a row;
- `height`: number of rows;
- `pitch_bytes`: byte distance between the starts of adjacent rows;
- `data`: pointer to the first pixel of the top row;
- `type`: `uint16` or `float32`.

Pixels are ordered from left to right, then from top to bottom. Coordinates use an origin at the top-left corner. Padding bytes between rows are not image data and are not encoded, decoded, compared, or calibrated.

The in-memory representation uses the native byte order of the supported host and CUDA platform. V1 supports little-endian, 64-bit Linux systems. The `.tsc` representation is explicitly little-endian and does not serialize native C++ structures.

## 3. Dimensions, pitch, and size limits

The following limits apply before any allocation, copy, or kernel launch:

| Property | V1 requirement |
|---|---|
| Width | 1 through `UINT32_MAX` pixels |
| Height | 1 through `UINT32_MAX` pixels |
| Batch size | 0 through `UINT32_MAX` images |
| Pixel size | 2 bytes for `uint16`, 4 bytes for `float32` |
| Row bytes | `width * pixel_size`, computed with checked 64-bit arithmetic |
| Pitch | At least `row_bytes`, no greater than `INT64_MAX` |
| Addressed image span | `pitch_bytes * (height - 1) + row_bytes`, at most `INT64_MAX` and representable by `size_t` |
| File, payload, and workspace sizes | At most `INT64_MAX` bytes and representable by `size_t` for a single allocation |

All additions, multiplications, alignments, and conversions used to derive sizes MUST be checked. A mathematically valid image that cannot be represented by an implementation dependency MAY be processed in bounded tile batches. It MUST NOT be silently truncated to the dependency's narrower index type.

A non-empty image requires a non-null data pointer. A data pointer and pitch MUST satisfy the alignment required by the declared pixel type: 2 bytes for `uint16` and 4 bytes for `float32`. Negative pitch and bottom-up memory views are unsupported.

Zero width or zero height is invalid. A batch containing an invalid image is invalid. An empty batch is a successful no-op, requires no image pointers, enqueues no GPU work, and requires zero per-image workspace.

Physical memory, configured workspace, and configured VRAM pool limits can impose lower operational limits. Exceeding a configured resource limit returns an error or triggers documented bounded batching. It never causes an unchecked allocation.

## 4. Codec correctness

### 4.1 Bit-exact contract

Encoding followed by decoding MUST reproduce every logical input pixel bit for bit.

For `uint16`, all 65,536 values are preserved exactly. Arithmetic used by predictive transforms is defined independently of signed overflow and implementation-defined shifts.

For `float32`, all 32-bit representations are data, including:

- positive and negative zero;
- positive and negative infinity;
- normal and subnormal values;
- quiet and signaling NaN representations and their payload bits;
- arbitrary 32-bit patterns.

The codec MUST NOT perform floating-point arithmetic on FP32 pixels or canonicalize NaNs. Codec equality for FP32 means equality of the 32-bit representation, not floating-point comparison.

The bit-exact contract covers the `.tsc` codec only. It does not imply that every external TIFF implementation preserves every special FP32 representation.

### 4.2 Logical data and padding

Only logical pixels are encoded. Input row padding MUST NOT affect the encoded result. Decoding writes logical output pixels and MUST NOT write into output row padding.

For a deterministic format configuration, CPU and CUDA encoders MUST select the same tile modes and produce the same byte stream. CPU and CUDA decoders MUST accept streams produced by either encoder.

### 4.3 Trust boundary

Untrusted `.tsc` data is parsed and validated on the host before device decoding. Validation includes all offsets, lengths, tile descriptors, segment descriptors, bit widths, counts, alignments, and derived ranges needed by the decoder. Device decoding accepts only validated metadata and payload under this contract.

No decoder may read outside a validated input range or write outside a declared output view, including the final packed word and partial edge tiles.

## 5. ROI contract

An ROI is `(x, y, width, height)` in pixels and denotes the half-open rectangle:

`[x, x + width) x [y, y + height)`

All ROI components are unsigned. A valid ROI MUST satisfy:

- `width > 0` and `height > 0`;
- `x < image_width` and `y < image_height`;
- `x + width <= image_width` and `y + height <= image_height`, evaluated with checked arithmetic.

TileStream does not silently clip an invalid ROI. Zero-area, overflowing, or out-of-bounds ROIs return an invalid ROI error.

ROI output is top-left oriented and densely represents exactly the requested pixels, subject to the caller-provided output pitch. The output pitch follows the same rules as an image pitch. Pixels outside the ROI MUST NOT be written. Decoding an ROI MUST decode only the tiles intersecting it, although complete intersecting tiles or tile rows may be read and decoded internally.

An ROI equal to the full image is valid and produces the same logical pixels as full-image decoding.

## 6. Calibration contract

Calibration is image processing separate from bit-exact codec reconstruction. For each logical pixel it computes FP32 output with the operation:

`y = (x - offset) * gain`

`offset` and `gain` are either scalar FP32 values or FP32 per-pixel maps matching the processed image or ROI. Map coordinates are relative to the requested output region. A mismatched map size, invalid map pitch, or missing required map is an invalid argument.

V1 provides:

1. a separate bit-exact decode operation;
2. a separate calibration operation;
3. a fused decode and calibration operation.

The separate and fused calibration paths MUST implement the numerical semantics defined by the numerical analysis specification and MUST be tested against each other under that contract. They are not required to preserve the input representation or NaN payloads. Visualization MUST NOT be used as a correctness oracle for calibration.

The endpoint for processing benchmarks is calibrated FP32 output ready for subsequent GPU work in a caller-provided device buffer. For host end-to-end timing, the endpoint is reached only after the stream operation producing that output has completed.

## 7. TIFF and BigTIFF contract

V1 imports and exports a deliberately limited TIFF subset:

- one sample per pixel;
- 16-bit unsigned integer or 32-bit IEEE floating-point samples;
- top-left orientation;
- tiled or stripped organization for import;
- tiled organization for export;
- uncompressed or Deflate-compressed export, subject to the format specification.

Unsupported sample types, channel counts, orientations, planar organizations, or ambiguous layouts are rejected with an unsupported format error. They are not silently converted.

TileStream preserves pixel values within the supported path. It does not guarantee preservation or interpretation of arbitrary TIFF, BigTIFF, OME-TIFF, GeoTIFF, EXIF, color, acquisition, or application-specific metadata. Export writes only the tags explicitly documented by the TIFF adapter specification.

TIFF interoperability and `.tsc` bit-exact correctness are separate test concerns. Exported files must be readable by an independent TIFF stack, and supported imported files must reproduce their declared pixel values.

## 8. Error semantics

Operations report stable error categories. Exact names and language bindings are defined by the API specification.

| Category | Meaning |
|---|---|
| Success | Operation completed or was successfully enqueued |
| Invalid argument | Null required pointer, invalid dimensions, pitch, alignment, aliasing, or incompatible views |
| Invalid ROI | Empty, overflowing, or out-of-bounds region |
| Unsupported type | Pixel type is not `uint16` or `float32` |
| Unsupported format | External file or format feature is outside the supported subset |
| Malformed data | Invalid or inconsistent `.tsc` metadata or payload |
| Size overflow | A required arithmetic result or conversion is not representable |
| Buffer too small | Caller-provided output or workspace does not meet the queried requirement |
| Resource limit | Operation exceeds a configured memory, file, batch, or workspace limit |
| CUDA error | CUDA launch, execution, copy, event, or stream failure |
| I/O error | File read, write, seek, or flush failure |

Argument, descriptor, capacity, and host-validation errors are returned before GPU work for that call is enqueued. Batch-level structural validation is all-or-nothing. A rejected call MUST NOT partially process the batch.

Asynchronous GPU calls return an immediate status for host validation and launch. A successful immediate status does not guarantee successful completion. Execution errors are observed through the documented completion mechanism or stream synchronization. Library operations MUST NOT hide a device-wide synchronization.

Input and output memory aliasing is forbidden by default. On an error detected before execution, logical output remains unmodified. If an asynchronous execution error occurs after work was enqueued, output contents are unspecified and MUST NOT be consumed.

C++ convenience wrappers MAY translate stable status codes into documented exceptions. The core API and CLI must retain the original error category. CLI failures return a nonzero exit status and a diagnostic that identifies the category and affected input.

## 9. Required edge-case behavior

| Case | Required result |
|---|---|
| Empty batch | Success, no-op, no pointer access, no GPU work |
| Zero-width or zero-height image | Invalid argument |
| Null data pointer for a non-empty image | Invalid argument |
| Pitch smaller than logical row bytes | Invalid argument |
| Misaligned pointer or pitch for the declared pixel type | Invalid argument |
| Overflow in dimensions, pitch, span, offsets, or allocation size | Size overflow |
| Unsupported pixel type | Unsupported type |
| Zero-area ROI | Invalid ROI |
| ROI outside the image or with overflowing endpoint | Invalid ROI |
| Full-image ROI | Valid, same logical result as full decode |
| Non-contiguous valid input or output | Valid when pitch satisfies the contract |
| Aliased input and output | Invalid argument |
| Unsupported TIFF tags or layout | Unsupported format, no silent conversion |
| Invalid `.tsc` descriptor or truncated payload | Malformed data before device decode |

These cases require direct tests. Size-overflow tests may use metadata-only inputs and do not require allocating the represented size.

## 10. Benchmark scenarios

Every published benchmark identifies the scenario, dataset and hash, pixel type, dimensions, ROI distribution, batch size, tile geometry, hardware, software versions, cache state, and timing boundary.

### 10.1 Archive/ROI

The `.tsc` archive exists before measurement. The primary measured path includes:

1. selecting intersecting tiles;
2. reading or copying the selected compressed bytes from the declared source tier;
3. transfer to the GPU when required;
4. decoding the requested ROI;
5. calibration;
6. completion of calibrated FP32 output on the GPU.

Archive creation time and archive size are reported separately and are not included in the primary ROI latency. Full-image and ROI cases use the same endpoint.

### 10.2 GPU-resident codec

Uncompressed input and required codec buffers already reside on the GPU. Encode and decode are measured separately and together where useful. Measurements include the kernels and required device-side staging for the declared operation, but exclude file I/O and host-to-device input transfer.

Compressed size includes all bytes required by the chosen representation. Payload-only and complete-container ratios are reported separately.

### 10.3 Cold file-to-result

Input begins in a file. The measured path includes file open and metadata parsing, required reads, decompression or decoding, transfers, and calibration through completed FP32 output on the GPU.

Cold or cache-disabled runs and warm-cache runs are reported separately. A run is called cold only when the method used to control or bypass the operating-system cache is recorded. Data already present in RAM or the page cache is not described as disk throughput.

### 10.4 Common timing rules

CUDA events measure device intervals with explicit stream dependencies. Host end-to-end timing uses a monotonic host clock and stops only after actual completion. Overlapping stages are not naively summed. Codec preparation, plan construction, allocation, and archive creation are either included in the stated boundary or reported separately.

The primary comparison tests whether:

`compressed transfer + decode + process < raw transfer + process`

For on-the-fly compression, encode and every additional copy are included. A result where compression loses is valid and must be reported.

## 11. Visualization contract

Visualization is a consumer of decoded or calibrated data, not part of codec correctness. A full-image preview MAY be downsampled on the CPU and cached. Such a preview must be labeled separately from a full-resolution ROI request.

Displayed colors, contrast, normalization, and NaN rendering do not alter source data and are not used to claim bit-exact or numerical correctness. UI latency and backend processing latency are measured and presented separately.

## 12. Acceptance criteria for this contract

This requirements stage is complete when subsequent tests can be written without guessing the required result for:

- empty images and batches;
- unsupported types and formats;
- invalid dimensions, pitch, pointers, and ROI;
- overflow and resource-limit conditions;
- bit-exact `uint16` and FP32 codec behavior;
- calibration versus codec correctness;
- TIFF metadata limitations;
- timing boundaries and the GPU-ready processing endpoint.

Later format and API documents may add stricter representational limits. They must not silently weaken these correctness, error, or measurement requirements.
