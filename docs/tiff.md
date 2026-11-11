# TIFF and BigTIFF interoperability

## Scope

The TIFF adapter provides file I/O through libtiff 4.5.1 or newer. It accepts one monochrome image per file and does not silently convert pixel values, orientation, color interpretation, or planar organization.

The public declarations are in `tilestream/tiff.h`.

## Import

`tilestream::io::read_tiff` accepts tiled or stripped TIFF and BigTIFF files with this effective tag set:

| Tag | Accepted value |
|---|---|
| `ImageWidth` | 1 through the configured width limit |
| `ImageLength` | 1 through the configured height limit |
| `BitsPerSample` | 16 for unsigned integer, 32 for IEEE floating point |
| `SampleFormat` | `UnsignedInteger` or `IEEEFP`, matching `BitsPerSample` |
| `SamplesPerPixel` | 1 |
| `PlanarConfiguration` | contiguous |
| `PhotometricInterpretation` | `MinIsBlack` |
| `Orientation` | top-left |
| `TileWidth`, `TileLength` | valid nonzero dimensions for tiled input |
| `RowsPerStrip` | valid nonzero row count for stripped input |

Only a single TIFF directory is accepted. libtiff performs decompression before the adapter receives bytes. A compression method is usable on import only when the installed libtiff build has its codec and returns the complete decoded tile or strip.

`TiffReadLimits` bounds width, height, and decoded pixel bytes before the destination allocation. Every decoded tile or strip must return exactly the expected byte count. Arithmetic is checked before allocations and libtiff calls.

## Export

`tilestream::io::write_tiff` always emits tiled data. `TiffWriteOptions` selects classic TIFF or BigTIFF, tile dimensions, and either:

- no compression and predictor `None`,
- Deflate with level 1 through 9 and predictor `None`,
- Deflate for `uint16` with horizontal differencing.

FP32 export always uses predictor `None`. The adapter rejects a horizontal predictor for FP32 and for uncompressed output. Edge tiles are zero-padded outside the logical image dimensions.

The output sets `SamplesPerPixel=1`, contiguous planar configuration, `MinIsBlack`, top-left orientation, and the sample format matching the TileStream pixel type.

## Metadata and FP32 limitations

The adapter transfers the pixel array, dimensions, and pixel type. It does not preserve arbitrary TIFF tags, image descriptions, geospatial keys, resolution, acquisition metadata, ICC profiles, private tags, multiple pages, masks, or pyramids. Applications that require those semantics must manage them separately.

TileStream reads and writes FP32 TIFF samples as 32-bit storage. Tests with libtiff 4.5.1 and tifffile 2026.10.15 verify the exact bit patterns of signed zero, infinities, subnormal values, and several NaN payloads for the supported pipeline. TIFF interoperability outside the tested stack does not guarantee preservation of NaN payloads or signaling state. The `.tsc` format remains the normative bit-exact representation for all FP32 words.
