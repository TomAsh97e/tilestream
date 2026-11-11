import argparse
from pathlib import Path

import numpy as np
import tifffile


WIDTH = 35
HEIGHT = 19
SPECIAL = np.array(
    [
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x00000001,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7FC01234,
        0x7FA12345,
        0xFFC0ABCD,
        0x3F800000,
        0xBF800000,
        0x00800000,
    ],
    dtype=np.uint32,
)


def uint16_values() -> np.ndarray:
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.uint16)
    return (y * np.uint16(257) + x * np.uint16(3)).astype(np.uint16)


def float32_values() -> np.ndarray:
    words = np.arange(WIDTH * HEIGHT, dtype=np.uint32) + np.uint32(0x3F000000)
    words[: SPECIAL.size] = SPECIAL
    return words.reshape(HEIGHT, WIDTH).view(np.float32)


def verify_cpp(path: Path, expected: np.ndarray, *, tiled: bool, compression: int,
               predictor: int, bigtiff: bool) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.is_bigtiff == bigtiff
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.is_tiled == tiled
        assert int(page.compression) == compression
        assert int(page.predictor) == predictor
        assert page.samplesperpixel == 1
        assert int(page.sampleformat) in (1, 3)
        assert page.shape == (HEIGHT, WIDTH)
        actual = page.asarray()
    assert actual.dtype == expected.dtype
    if expected.dtype == np.float32:
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    else:
        np.testing.assert_array_equal(actual, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory

    integers = uint16_values()
    floats = float32_values()
    verify_cpp(directory / "cpp-u16-none.tif", integers, tiled=True, compression=1,
               predictor=1, bigtiff=False)
    verify_cpp(directory / "cpp-u16-deflate-predictor.tif", integers, tiled=True,
               compression=8, predictor=2, bigtiff=False)
    verify_cpp(directory / "cpp-u16-big.tif", integers, tiled=True, compression=1,
               predictor=1, bigtiff=True)
    verify_cpp(directory / "cpp-f32-deflate.tif", floats, tiled=True, compression=8,
               predictor=1, bigtiff=False)

    tifffile.imwrite(directory / "python-u16-strips.tif", integers, photometric="minisblack",
                     rowsperstrip=7, metadata=None)
    tifffile.imwrite(directory / "python-f32-tiles.tif", floats, photometric="minisblack",
                     tile=(16, 16), metadata=None)


if __name__ == "__main__":
    main()
