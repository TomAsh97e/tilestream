# TileStream container format 1.0

## Scope and conventions

One `.tsc` file stores one nonempty, single-channel image. A batch consists of independent files, not concatenated images in one container. V1 accepts only tile geometries `(width, height)` of `(128, 32)`, `(64, 64)`, and `(256, 16)`. The default is `(128, 32)`. These choices were approved before T04 implementation.

All integers are unsigned, fixed-width, little-endian unless an intermediate mathematical quantity is explicitly signed. FP32 pixels are serialized as their exact IEEE 754 binary32 words, never converted through a floating-point computation. Byte offsets are relative to the start of the file unless stated otherwise.

The tables below define the wire layout. C++ structures, pointers, compiler padding, and native enums MUST NOT be serialized directly. `include/tilestream/format.h` provides the corresponding constants. The format version is independent of the library release version.

## File header: 64 bytes

| Byte offset | Size | Field | Required value or meaning |
|---|---:|---|---|
| 0 | 8 | Magic | Hex `54 53 43 00 0d 0a 1a 0a` |
| 8 | 2 | Major version | 1 |
| 10 | 2 | Minor version | 0 |
| 12 | 2 | Header size | 64 |
| 14 | 1 | Pixel type | 1 = uint16, 2 = float32 |
| 15 | 1 | Flags | 0 |
| 16 | 4 | Width | Nonzero logical pixel count |
| 20 | 4 | Height | Nonzero logical row count |
| 24 | 2 | Tile width | Part of an allowed geometry |
| 26 | 2 | Tile height | Part of an allowed geometry |
| 28 | 2 | Index entry size | 32 |
| 30 | 2 | Reserved | 0 |
| 32 | 8 | Tile count | `ceil(width / tile_width) * ceil(height / tile_height)` |
| 40 | 8 | Index offset | 64 |
| 48 | 8 | Payload offset | `64 + 32 * tile_count` |
| 56 | 8 | File size | Exact number of bytes, including all padding |

Unknown versions, flags, or tile geometries are unsupported, not guessed. A truncated header, inconsistent counts, or nonzero reserved bytes are malformed. There are no extension records or checksums in 1.0. Structural validation is not a guarantee against undetected pixel corruption.

## Tile index: 32 bytes per tile

Tiles are ordered left-to-right, then top-to-bottom. For `columns = ceil(width / tile_width)`, tile `i` begins at `(i % columns * tile_width, i / columns * tile_height)`. Products and endpoints are computed in checked 64-bit arithmetic.

| Relative byte offset | Size | Field | Meaning |
|---|---:|---|---|
| 0 | 1 | Mode | 0 = RAW, 1 = CONSTANT, 2 = PREDICT_BITPACK |
| 1 | 3 | Reserved | All zero |
| 4 | 2 | Actual width | `min(tile_width, width - tile_x)` |
| 6 | 2 | Actual height | `min(tile_height, height - tile_y)` |
| 8 | 8 | Payload offset | Absolute file offset, multiple of 4 |
| 16 | 8 | Payload length | Includes mode-specific padding, multiple of 4 |
| 24 | 8 | Reserved | All zero |

Actual dimensions MUST agree with the tile position. They are not independent image dimensions. There is no separate tile header in the payload: the index carries the common mode and dimensions.

The first tile starts at the header's payload offset. Every subsequent tile immediately follows the preceding tile's padded payload. No gaps, overlaps, reordered payloads, extra index entries, or trailing bytes are allowed. The final payload end equals file size. This makes the byte stream deterministic and eliminates ambiguity about which bytes belong to a tile.

## Payload modes

Let `w` and `h` be the actual tile dimensions, `s` be 2 or 4 bytes per pixel, and `A4(n)` mean rounding `n` up to a multiple of four using checked arithmetic.

### RAW: mode 0

Store exactly `w * h` pixels in row-major order, without input pitch or synthetic edge pixels. Append zero bytes to reach `A4(s * w * h)` bytes. There is no padding between rows. For uint16, the only possible trailing padding is two zero bytes; for float32 there is none.

### CONSTANT: mode 1

Store one 32-bit word. For uint16 the low 16 bits are the sample and the high 16 bits are zero. For float32 all 32 bits are the sample representation. The payload length is always four bytes. Repeat this value over all `w * h` output pixels. FP32 constant detection uses bit equality, so positive and negative zero are different values.

### PREDICT_BITPACK: mode 2

A tile contains exactly `h` sequential row segments. Each segment describes exactly `w` pixels. The first row begins at the tile payload offset. Each row contains an eight-byte descriptor immediately followed by its packed residual words:

| Relative byte offset | Size | Field |
|---|---:|---|
| 0 | 4 | Seed word; uint16 uses the low half and requires a zero high half |
| 4 | 1 | Residual width `b`, 0 through 16 or 32 depending on pixel type |
| 5 | 3 | Reserved zeros |
| 8 | Variable | Packed residuals, in little-endian 32-bit words |

For `n = w - 1` residuals, the packed length is `4 * ceil(n * b / 32)` bytes. The segment length is this quantity plus eight. The next row starts immediately after it. A row of one pixel has no residual words and requires `b = 0`. Width zero stores no residual words and reconstructs a constant row from the seed. No extra terminator, segment offset table, or final word is present.

Residual `j` starts at bit `j * b`, with `j = 0` corresponding to the second pixel. Bit zero is the least significant bit of the first 32-bit word. Residual bits are stored least significant first, continuing into the next word if necessary. Unused high bits of the last word are zero. For example, three 3-bit residuals `[1, 2, 7]` form the word `0x000001d1`, stored as `d1 01 00 00`.

An unpacker reads a second word only when the symbol actually crosses a word boundary and the validated segment length contains that word. `b = 0` and `b = 32` use explicit cases: neither a mask computation nor a merge may shift a 32-bit value by 32.

### uint16 transform

The first sample is the seed. For each subsequent `(previous, current)` pair:

```text
d = (current - previous) mod 65536
s = d                  if d < 32768
s = d - 65536          otherwise
z = 2*s                if s >= 0
z = -2*s - 1           otherwise
```

Perform subtraction modulo 65536 with unsigned arithmetic and the signed interpretation/zigzag in a wider representable type. `z` lies in `[0, 65535]`. Inverse zigzag is `s = z/2` for even `z`, and `s = -(z/2) - 1` for odd `z`. Reconstruction adds `s` modulo 65536 to the preceding value. CUDA may use unsigned prefix sums followed by masking; no signed overflow is permitted.

### float32 transform

The seed and all samples are bit words. Residuals are `current_bits XOR previous_bits`. Reconstruction uses a prefix XOR combined with the seed. CPU C++17 uses `memcpy` for representation changes; CUDA code must use alias-safe loads or bit-cast intrinsics. NaNs, signed zeros, infinities, and subnormals are unchanged by the codec.

## Canonical encoder and decoder acceptance

For each row, choose the minimum width that holds all its residuals (`0` for no residuals or all zeros). Reset prediction at every tile row. Evaluate these complete candidate costs:

```text
RAW:        32 + A4(s*w*h)
CONSTANT:   32 + 4, only if all sample bits are equal
PREDICT:    32 + sum over rows [8 + 4*ceil((w-1)*b_row/32)]
```

The common 32-byte index entry is included explicitly. The file header is common to all candidates. Select the smallest cost; ties use the order RAW, CONSTANT, PREDICT_BITPACK. No floating-point comparisons or CPU/GPU-specific heuristics participate. Alignment and padding are part of the stored size.

A decoder need not recompress a tile to prove that the encoder chose the optimal mode or minimal widths. It accepts structurally valid noncanonical choices provided their payload length does not exceed the RAW candidate length. It still checks widths, exact lengths, dimensions, reserved bytes, and zero padding. Consequently the output bound below applies to all accepted 1.0 files as well as canonical encoders. Full-width residuals can occur in a predictive tile when savings in other rows compensate their descriptor overhead.

## Checked output bound

For image dimensions `W,H`, tile geometry `tw,th`, and sample size `s`:

```text
Nx = W/tw + (W%tw != 0)
Ny = H/th + (H%th != 0)
N = Nx * Ny
U = W * H * s
B = 64 + 32*N + sum over tiles A4(s*w_tile*h_tile)
```

All allowed tile dimensions are even. For uint16, an odd pixel count can occur only in the bottom-right tile when both image dimensions are odd. Float32 needs no RAW padding. Thus an O(1), tight RAW bound is:

```text
B = 64 + 32*N + U + pad
pad = 2 if s == 2 and W is odd and H is odd, otherwise 0
```

Every selected payload is at most its RAW candidate size, proving `file_size <= B`. This is not a promise of compression: the header and index remain overhead. For a 1x1 image of either type, `B = 100` and the canonical mode is RAW. For default tiles, a 128x32 uint16 image has `B = 8288`; 129x33 has four tiles and `B = 8708`.

`compress_bound` implements this formula without inspecting image memory. Each multiplication and addition is checked against `L = min(INT64_MAX, SIZE_MAX)` before it is performed. Ceilings use quotient/remainder, not potentially overflowing addition. On failure the result argument is unchanged.

## Limits and validation order

- Image width and height: 1 through `UINT32_MAX`, subject to raw span and file bound fitting `L`.
- File, payload offsets/lengths, workspace, and a single allocation: at most `L` bytes.
- Tile indices and counts: 64-bit. A file is not restricted to a 32-bit primitive's item count.
- Logical API batch count: at most `UINT32_MAX`; no batch is serialized inside one file.
- Execution windows: at most `INT32_MAX` tiles, often much smaller under a configured memory budget. Never narrow a file tile count to a CUDA/CUB index without splitting it.
- Host parsing also obeys the caller's file-size, tile-count, and metadata-byte budgets before allocating tables.

Host validation checks the fixed header first, derives counts and the bound with checked arithmetic, checks the entire index range, then validates every tile and predictive segment before creating device metadata. Read fields only after their byte range is proven available. Check the mode-specific length before reading seeds, widths, residual words, or padding. Reject illegal widths before multiplication or shifting. Check sums as `length <= limit - offset` after verifying `offset <= limit`.

Successful validation binds metadata to the exact immutable bytes validated. If any source bytes change, validation must be repeated. Selecting an ROI does not permit skipping validation of malformed descriptors supplied to the GPU. The device API is not an untrusted-file parser. See [API and workspace contracts](api.md).
