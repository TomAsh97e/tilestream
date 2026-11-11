# TileStream V1 API and memory contracts

## Implementation status

The library provides pure host size queries (`image_span`, `compress_bound`, `workspace_size`), a host parser and a scalar CPU codec for uint16 and bit-exact FP32, and TIFF/BigTIFF interoperability through libtiff. On the GPU, `prepare_batch` plans batches, `compress_async` produces the same canonical bitstream as the CPU encoder, and `decompress_async` and `decode_roi_async` decode parsed archives, staged tile subsets and encoder output. `calibrate_async` and the fused `decode_calibrate_async` apply calibration under a defined [numerical contract](numerical-analysis.md). `DeviceBuffer` and `synchronize` are conveniences, and `examples/cpp_client` uses only installed headers. Testing is described in [testing.md](testing.md), the benchmark harness in [benchmarking.md](benchmarking.md), the comparison with other codecs in [baselines.md](baselines.md), and profiling and optimizations in [performance-investigations.md](performance-investigations.md).

The headers are:

- `types.h`: fixed-value status/type enums, image and byte views, ROI, device results;
- `format.h`: wire offsets and constants, not serialized C++ structures;
- `sizing.h`: checked, allocation-free size queries;
- `api.h`: owned host metadata and the host-parser declaration;
- `cpu.h`: scalar host compression and decompression;
- `tiff.h`: checked TIFF/BigTIFF import and tiled export;
- `cuda.h`: native device records, the RAII `DeviceBuffer` and `BatchPlan`, and CUDA entry points.

The [TIFF interoperability contract](tiff.md) defines the accepted tags, compression and predictor options, resource limits, and metadata limitations. TIFF I/O is synchronous host file I/O and is separate from the CUDA execution interface.

Host headers require C++17 but no CUDA toolkit. In a CPU-only installation `cuda.h` exposes no CUDA entry points and includes no CUDA headers. In a CUDA installation it uses the real `cudaStream_t` from CUDA Runtime. The exported CMake target supplies that public dependency.

## Images and views

`ImageSpec` contains pixel type, width, and height. `ConstImageView` and `ImageView` add a pointer, byte pitch, and accessible byte capacity. In CPU APIs these pointers refer to host memory; in CUDA requests they refer to device-accessible memory on the plan's device. V1 does not infer a memory kind from the pointer or silently copy a host image to a GPU.

Coordinates are top-left, row-major. Pointer and pitch alignment is at least 2 bytes for uint16 or 4 for FP32. Let `r = width * sample_bytes`. A view must satisfy:

```text
width > 0, height > 0
pitch_bytes >= r
pitch_bytes % sample_bytes == 0
span = pitch_bytes * (height - 1) + r
span <= capacity_bytes <= L
```

`L = min(INT64_MAX, SIZE_MAX)`. A required pointer is non-null. The caller guarantees the declared capacity is actually allocated and device-accessible. Integer address-range addition must also be checked before overlap testing. Neither a size query nor a declared capacity can prove the validity of an arbitrary pointer.

Only logical pixels are read or written. Input padding does not affect compression; output padding is untouched. `image_span` validates metadata/pitch and computes the addressed span without dereferencing any pointer. Unsupported type is checked before dimensions; invalid dimensions or pitch yield `invalid_argument`, and unrepresentable sizes yield `size_overflow`.

ROI uses unsigned `(x, y, width, height)` and a nonempty half-open rectangle. Validate endpoints in 64-bit arithmetic, never by wrapping a 32-bit sum. No clipping is implicit. ROI output dimensions are exactly the requested dimensions, with its own pitch and capacity.

## Host metadata versus device metadata

`parse_metadata(ConstBytes, ParseLimits, ImageMetadata&)` accepts a complete host-visible `.tsc` byte span, including a mapped file. It validates the [format](format.md), all predictive segment lengths and padding, and caller resource limits before returning success. It does not decode image pixels. Invalid input leaves the previous output metadata unchanged. Allocation failure maps to `resource_limit`; exceptions must not escape the status API.

Parser error mapping is explicit: a nonempty null source is `invalid_argument`; truncation, bad magic, zero dimensions, inconsistent fields, invalid modes/widths, and nonzero reserved/padding bytes are `malformed_data`; an unknown version, flags, or geometry is `unsupported_format`; an unknown pixel type is `unsupported_type`; unrepresentable arithmetic is `size_overflow`; and exceeding a caller budget is `resource_limit`. An empty byte span is a truncated file, not an empty-batch no-op. The parser never launches GPU work.

`ImageMetadata` owns native vectors of `TileMetadata` and `SegmentMetadata`. Tile offsets are absolute file offsets; segment payload offsets are relative to the tile payload. `first_segment` indexes the segment vector. RAW and CONSTANT use no segment records and set it to zero. A predictive tile has exactly its actual height in consecutive segment records. Metadata allocation limits count the native tables and object storage with checked arithmetic, not just their serialized counterparts. The parser must not allocate from unchecked file counts.

These public records are not an authenticity token. Metadata used for untrusted data must come from successful parsing, remain unmodified, and describe the same immutable payload. Changing metadata or source bytes invalidates the validation result. The low-level device path is not hardened against callers fabricating pointers or falsely declaring payload provenance.

`DeviceImage`, `DeviceTile`, and `DeviceSegment` contain only native values and device pointers. No host vector or pointer to `ImageMetadata` is copied as a device descriptor. These records have an internal native layout, not an on-disk ABI. Their memory is accounted separately from workspace.

### Selected-tile uploads

For a parsed archive, `CompressedInput.metadata` points to host metadata during preparation. `bytes` points to device-resident compressed bytes. If `mappings` is null and `mapping_count` is zero, the buffer contains the complete file at offset zero. Otherwise, the host `TileMapping` array lists `(tile_index, input_offset)` for a staged subset of payloads.

For an ROI, preparation requires exactly the intersecting tiles in increasing tile-index order. For a full decode it requires all tiles. Each mapping covers the corresponding padded payload length within the supplied byte buffer, is four-byte aligned, and does not overlap another mapping. The bytes must be copies of the validated payloads, not newly generated data. Preparation rebases segment offsets into this staging buffer and uploads only the selected descriptors. CPU metadata can cover the entire file while GPU memory contains only selected tiles.

This supports selective transfers without decoding the whole image. The header/index may remain on the host. It does not allow a caller to substitute unvalidated bytes after validation.

### Trusted encoder-output path

A GPU-resident round-trip need not copy its index to the CPU just to validate bytes generated by its own trusted encoder. For this path, `metadata` and `mappings` are null, `mapping_count` is zero, and `encoder_result` points to the successful producing call's device result. `trusted_image` and `trusted_tile` describe that encoder's fixed plan. `bytes` points to the same full output allocation, with capacity at least `compress_bound` for that image.

Preparation validates the host-known shapes, capacities, and device pointers. Execution reads the actual size and index only after the producing result is ready, in the same stream or behind an explicit event dependency. If the producer reports a per-image error, no compressed bytes for that image are consumed; the consumer propagates the error with zero bytes written. This is not an option for imported files or arbitrary data. Mutating the encoded output, its result, or its expected shape breaches the trusted-device precondition. The producing result must not alias the consuming result.

## Batch plans and ownership

`prepare_batch(request, plan, stream)` is an explicit, non-hot-path convenience operation. It first validates all host-known arguments, descriptors, capacities, ROI/map shapes, and non-aliasing rules for the complete batch. It then copies request metadata into a new plan, creates device descriptor tables, and completes their upload in the supplied stream before returning. It may allocate and wait for that stream. It also loads every kernel the plan launches, so that no execution has to. Under CUDA's default lazy module loading, the first preparation of an operation and pixel type in a process can therefore synchronize the device; later preparations do not. The previous `plan` remains intact on failure.

The RAII `BatchPlan` owns its host metadata copies and device descriptor tables, not image, payload, map, workspace, or result buffers. Borrowed request arrays and host metadata may be released after successful preparation. Device data buffers and the plan itself must remain alive until all executions using them complete. Destruction must not be relied upon as an execution-completion mechanism. A destroyed plan returns its descriptor tables with `cudaFreeAsync` on `cudaStreamPerThread` and never waits.

A plan fixes its operation, image layouts, buffer addresses/capacities, ROI, calibration parameter values, and CUDA device. Reuse may change input pixel values, or map pixel values, only after preceding work has finished. A parsed compressed source must remain byte-identical to the validated source; otherwise revalidate and prepare again. An encoder-output source may be regenerated by the matching encoder under proper dependencies. Changing bound addresses/layouts or scalar parameters requires a new plan. Later wrappers may expose explicit rebinding, but V1 has no implicit rebinding.

One nonempty batch has one input pixel type, although dimensions and pitch may differ. Compression uses `request.tile` for every image. Decode uses each archive's geometry. Calibration always produces FP32, so the output type may differ from the common input type.

A plan is not reentrant: one execution at a time per plan. Executions of one plan write the same bound outputs, so they must be ordered, for example by issuing them in one stream. Concurrent calls require independent plans, workspaces, results, and nonoverlapping writable buffers, typically on different streams. Immutable inputs may be shared. There is no global mutable execution scratch.

TileStream keeps no mutable global state of its own. Different host threads may prepare, execute, and destroy independent plans at the same time. A plan's host side is immutable after preparation, so `workspace_spec` may be called from any thread. Preparation and execution require the calling thread's current device to be the plan's device.

A logical batch may have up to `UINT32_MAX` images. Larger file tile counts and batches are traversed in bounded execution windows, with 64-bit global offsets. Each active window has at most `INT32_MAX` tiles before using a primitive with a signed 32-bit count. The CUDA implementation uses windows of at most 4096 tiles, so workspace stays bounded for any image or batch size. Resource budgeting and the orchestration of overlapping copies and decodes are not part of this contract.

## Workspace derivation

The initial multipass design reserves separate global regions for residuals, row analysis, tile analysis, lengths, offsets, per-image carry state, and CUB temporary storage. It avoids speculative writes into the final compact output.

For a window with `T` active tiles, `I` distinct active images, and maximum tile height `h`, all geometries have 4096 pixels per full tile. Let `A256(n)` be checked rounding to a multiple of 256. A conservative workspace bound is:

```text
W = A256(4 * 4096 * T)       residual words, including seed slots
  + A256(8 * h * T)          row seed/width records
  + A256(32 * T)             internal tile-analysis records
  + A256(8 * T)              selected padded lengths
  + A256(8 * (T + 1))        64-bit scanned offsets and terminal offset
  + A256(16 * I)             per-image output base/carry state
  + A256(S)                 primitive temporary storage
```

The 32-byte tile-analysis record is scratch, not the public `DeviceTile` descriptor. Full slots are reserved for partial tiles, without counting those unused slots as image data. Block-local scans/reductions use kernel local/shared resources, not an undeclared global allocation. Each region starts at a 256-byte boundary; regions cannot borrow each other's padding.

`WorkspaceSpec` describes maxima sufficient for every sequential window in a plan. `image_count` and `tile_count` are active-window counts, not necessarily full file/batch counts. Require `1 <= I <= UINT32_MAX`, `I <= T <= INT32_MAX`, and an allowed tile geometry. For mixed decode geometries, use the `(64,64)` shape to bound the maximum row count. Decode and fused decode with calibration may use less scratch, but their query deliberately returns this conservative bound. Standalone calibration has no tiles and uses no scratch, so its plan reports the valid empty spec and needs no workspace.

`S` is not guessed from the number of tiles. Preparation must query the actual CUB primitive, data types, item count, toolkit version, and device used by the implementation. If several primitive operations or windows run sequentially, reserve the maximum of their queried temporary sizes. The region is reusable only after earlier work in the stream completes. Concurrent operations use separate workspaces. If a backend later requires simultaneously live temporary areas, it must account for their sum instead of reusing this maximum.

The implemented `workspace_size(WorkspaceSpec, bytes)` is a pure arithmetic query. It does not call CUB and cannot verify a caller-supplied `scan_temp_bytes`. Executable plans supply the authoritative spec through `BatchPlan::workspace_spec()` after actual CUB queries. Zero scratch is valid only if the chosen primitive genuinely requires none. The zero-scratch test cases verify the formula, not a device allocation requirement.

For one default tile and one image, the fixed regions total 17664 bytes; append `A256(S)`. Every product, rounding, and sum is checked against `L`. A valid empty spec (`I=T=S=0`) returns zero, without inspecting geometry; inconsistent empty counts are invalid. Excess active-window counts return `resource_limit`. Arithmetic overflow returns `size_overflow`. On any failure, the output argument is unchanged.

Workspace excludes image/output buffers, result arrays, host metadata, and persistent device descriptors owned by the plan. Peak device allocation must report all these components, not just `W`. Larger images do not imply a single workspace proportional to all their tiles: the same workspace is reused for bounded windows. Output files and full-image destination allocations still have their own limits; out-of-core transport is not implemented by this size query.

## Asynchronous entry points

All five execution functions have the common signature:

```cpp
Status operation(const BatchPlan& plan, MutableBytes workspace,
                 ResultBuffer results, cudaStream_t stream) noexcept;
```

| Function | Required plan operation | Inputs and outputs |
|---|---|---|
| `compress_async` | `compress` | `BatchItem.input` to `compressed_output`, one complete `.tsc` per image |
| `decompress_async` | `decompress` | `compressed_input` to full-size `output`; bit-exact |
| `decode_roi_async` | `decode_roi` | `compressed_input`, nonempty `roi`, and ROI-sized `output`; bit-exact |
| `calibrate_async` | `calibrate` | `input`, `calibration`, and same-sized FP32 `output` |
| `decode_calibrate_async` | `decode_calibrate` | `compressed_input`, nonempty `roi`, `calibration`, and ROI-sized FP32 `output` |

For full-image fused decode, supply an ROI covering the full image. `roi` is ignored for full decode and standalone calibration. Fields not used by the selected operation are ignored and should remain default-initialized. No execution function allocates, frees memory, performs file I/O, or waits for host-visible compressed sizes.

`Parameter.kind` selects a scalar or FP32 map independently for gain and offset. Maps exactly match the output region dimensions, with independent pitches. Scalar parameters default to offset 0 and gain 1. Maps use output-local coordinates, never full-image coordinates implicitly. A map must be a valid FP32 view of device-accessible memory: a wrong size, type, or pitch, a missing map, or an unknown kind is `invalid_argument`, an unknown pixel type value is `unsupported_type`, and too little capacity is `buffer_too_small`. The output must be FP32 and may not overlap the input, a map, or another output.

Calibration computes `y = (x - offset) * gain` for each output pixel in binary32, with two roundings to nearest even: the subtraction is rounded, then the multiplication. The device uses `__fsub_rn` and `__fmul_rn`, which are never contracted into an FMA, and does not flush subnormal operands or results. A uint16 sample converts to FP32 exactly; an FP32 sample is used with its stored bits. Fused and separate calibration run the same operations on the same sample bits, so on one device they produce identical output bits, NaNs included. NaN payloads are otherwise not part of the contract. Calibrated output is processed data, not a lossless reconstruction of the input. The special-value contract, error bounds, and measured errors are defined in [numerical-analysis.md](numerical-analysis.md) and tested separately from the bit-exact codec.

### Alignment and capacity

- Image/map data: natural pixel alignment, valid pitch/span as above.
- Compressed device input/output buffers: four-byte alignment; upload offsets also four-byte aligned. The host parser may read an unaligned byte span through explicit byte loads.
- Workspace: 256-byte aligned, capacity at least the plan's checked query. Null is permitted only for a zero-size requirement.
- Results: alignment of `DeviceResult` (8 on the supported platform), at least one 16-byte record per logical image, with checked `count * sizeof(DeviceResult)`.
- Device descriptor tables: natural alignment of their native record type, allocated by preparation.

Preparation and execution check the base address of every device pointer with `cudaPointerGetAttributes`. Device memory must belong to the plan's device. Managed memory and mapped pinned host memory are accepted. Ordinary pageable host memory is accepted only on a device that reports pageable memory access. Other pointers return `invalid_argument`. This check cannot prove that the declared capacity is allocated.

For compression, capacity must be at least `compress_bound` before launch, irrespective of how compressible the data appears. Exact output size is not known on the host until completion. Whole file buffers larger than the actual result remain allocations, not valid file lengths.

### Aliasing

Within a batch, all writable output ranges, workspace, result records, and plan-owned writable storage must be pairwise disjoint. None may overlap any input image, compressed source, map, or producing encoder result. Immutable read-only inputs may overlap one another. Use entire declared byte capacities for conservative range checks, including padding. There is no in-place codec or calibration exception. Other streams must not read or write these outputs without explicit completion dependencies.

### Status and completion

1. Host argument/plan/capacity validation occurs for the entire call before enqueueing any work. An invalid call returns the corresponding stable `Status` and leaves outputs and results unchanged.
2. The immediate return reports host validation and CUDA enqueue errors, not completion. `success` means work was accepted. If an enqueue fails after earlier work was submitted, return `cuda_error`; buffers may already be in use and must be retained until that stream has been safely drained or the failed context handled.
3. Successful device execution writes one `DeviceResult` per image. `bytes_written` is the exact file size for compression, or logical output pixel bytes excluding pitch for decode/calibration. A per-image failure writes its status and zero bytes; that image's output is not consumable. The reserved field is zero.
4. Record an event after execution or use `cudaStreamSynchronize` to observe completion and CUDA execution errors. If CUDA execution fails, neither output nor device status records are trustworthy, even if a record still contains an earlier `success` value.
5. To read results on the CPU, enqueue a device-to-pinned-host copy in the same stream after the operation and wait for an event after that copy. A host file wrapper may explicitly wait, read `bytes_written`, then copy exactly that many bytes. Those waits/copies are measured as host-wrapper cost.

Errors are separated by when they are detected. Argument, plan, capacity, alignment, aliasing, and pointer checks run first and return their own status before any work is enqueued. `cuda_error` from an execution function means a CUDA runtime call failed while checking or enqueueing. Per-image codec failures, such as `malformed_data` on the trusted path, appear only in the `DeviceResult` records. CUDA execution failures appear when the caller waits on the stream or an event.

CUDA reports launch failures only through the runtime's per-thread last error. Every library call that returns `cuda_error` after a failed runtime call clears that error, so a later launch check cannot report it again. An execution function does not enqueue while an error from the caller's earlier work is pending: it returns `cuda_error` and leaves that error for the caller to read with `cudaGetLastError`.

An event or status buffer is not implicitly synchronized by the library. The caller supplies the stream explicitly; a null/default CUDA stream is accepted but carries CUDA's default-stream ordering semantics. The stream and all device allocations must belong to the plan's device. Independent streams require explicit event ordering when sharing producer/consumer data.

No execution function synchronizes the entire device. Executions never load kernels either, because preparation has already loaded them; see [Batch plans and ownership](#batch-plans-and-ownership).

An empty batch is a no-op: no image/result pointer access, no stream work, no CUDA availability requirement, and zero workspace. Preparing it needs only an empty host plan. Execution still requires the matching operation, but ignores otherwise unused buffers and stream. No fabricated successful CUDA test should be used as evidence that a device executed work.

### Device buffers

`DeviceBuffer` is an optional RAII owner for device memory from the stream-ordered allocator of the stream's device. Execution functions never allocate; callers may use it, or any other allocator, for inputs, outputs, workspace, and results.

- `DeviceBuffer::allocate(bytes, stream, buffer)` allocates with `cudaMallocAsync` in `stream`. The memory is aligned to `workspace_alignment`. A zero size gives an empty buffer. On success the previous memory of `buffer` is freed in stream order; if the allocation fails, `buffer` is unchanged. Sizes above `L` return `size_overflow`, and an allocation failure returns `cuda_error`.
- `release(stream)` frees with `cudaFreeAsync` after the work already enqueued in `stream`. Releasing an empty buffer succeeds.
- The destructor frees on `cudaStreamPerThread` without waiting, so it requires all work using the memory to have completed. Use `release` while work may still be pending.
- `synchronize(stream)` waits for `stream` and returns `cuda_error` for any CUDA launch or execution failure seen there.

## Verification of the size queries

`tilestream.cpu.sizing` tests metadata-only bounds, all supported tile geometries, edge padding against explicit per-tile sums, invalid types/dimensions, image span, checked add/multiply/alignment, workspace limits and overflow, and output-argument preservation. These tests allocate no enormous image or workspace buffers.

`tilestream.cuda.headers` compiles the CUDA contracts and asserts that device records are standard-layout/trivially copyable, without linking unimplemented operations or pretending to test a codec. CPU-only builds compile host API headers without CUDA. Execution, parser, golden-bitstream, numerical and sanitizer tests are described in [testing.md](testing.md).
