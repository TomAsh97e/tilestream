from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import array_sha256, file_sha256, save_array

GLOBAL_SEED = 2_026_517
PATTERNS = (
    "constant",
    "ramp_up",
    "ramp_down",
    "edges",
    "checkerboard",
    "noise",
    "smooth_noise",
)
SHAPES = (
    (1, 1),
    (31, 31),
    (32, 32),
    (33, 33),
    (127, 127),
    (128, 128),
    (129, 129),
    (31, 127),
    (31, 128),
    (31, 129),
    (32, 127),
    (32, 128),
    (32, 129),
    (33, 127),
    (33, 128),
    (33, 129),
    (35, 130),
)
DTYPES = ("uint16", "float32")
TUNING_PATTERNS = frozenset(("constant", "ramp_up", "edges", "smooth_noise"))
FIXTURES = (
    ("uint16", "smooth_noise", 33, 129),
    ("float32", "checkerboard", 35, 130),
)


def case_seed(dtype: str, pattern: str, height: int, width: int) -> int:
    label = f"{GLOBAL_SEED}:{dtype}:{pattern}:{height}:{width}".encode("ascii")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "little")


def _coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(height, dtype=np.uint64)[:, None]
    x = np.arange(width, dtype=np.uint64)[None, :]
    return y, x


def generate_uint16(pattern: str, height: int, width: int, seed: int) -> np.ndarray:
    y, x = _coordinates(height, width)
    index = y * np.uint64(width) + x
    rng = np.random.Generator(np.random.PCG64(seed))

    if pattern == "constant":
        result = np.full((height, width), seed & 0xFFFF, dtype=np.uint16)
    elif pattern == "ramp_up":
        result = (index & np.uint64(0xFFFF)).astype("<u2")
    elif pattern == "ramp_down":
        result = (np.uint64(0xFFFF) - (index & np.uint64(0xFFFF))).astype("<u2")
    elif pattern == "edges":
        result = np.zeros((height, width), dtype="<u2")
        result[:, width // 2 :] = np.uint16(65535)
        result[height // 2 :, :] ^= np.uint16(0x5555)
    elif pattern == "checkerboard":
        result = (((x + y) & np.uint64(1)) * np.uint64(0xFFFF)).astype("<u2")
    elif pattern == "noise":
        result = rng.integers(0, 65536, size=(height, width), dtype=np.uint16)
    elif pattern == "smooth_noise":
        noise = rng.integers(-8, 9, size=(height, width), dtype=np.int32)
        base = x.astype(np.int64) * 97 + y.astype(np.int64) * 53 + noise
        result = np.clip(base, 0, 65535).astype("<u2")
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return np.ascontiguousarray(result, dtype="<u2")


def generate_float32(pattern: str, height: int, width: int, seed: int) -> np.ndarray:
    y, x = _coordinates(height, width)
    index = y * np.uint64(width) + x
    rng = np.random.Generator(np.random.PCG64(seed))

    if pattern == "constant":
        value = np.float32((seed & 0xFF) / 16.0)
        result = np.full((height, width), value, dtype="<f4")
    elif pattern == "ramp_up":
        result = index.astype("<f4") * np.float32(0.25)
    elif pattern == "ramp_down":
        result = -index.astype("<f4") * np.float32(0.25)
    elif pattern == "edges":
        result = np.full((height, width), np.float32(-3.0), dtype="<f4")
        result[:, width // 2 :] = np.float32(7.0)
        result[height // 2 :, :] += np.float32(0.5)
    elif pattern == "checkerboard":
        result = np.where(((x + y) & np.uint64(1)) == 0, -1.0, 1.0).astype("<f4")
    elif pattern == "noise":
        result = rng.standard_normal(size=(height, width), dtype=np.float32).astype("<f4")
    elif pattern == "smooth_noise":
        noise = rng.integers(-8, 9, size=(height, width), dtype=np.int32)
        fixed_point = x.astype(np.int64) * 97 + y.astype(np.int64) * 53 + noise
        result = fixed_point.astype("<f4") * np.float32(1.0 / 256.0)
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return np.ascontiguousarray(result, dtype="<f4")


def generate_case(dtype: str, pattern: str, height: int, width: int) -> tuple[np.ndarray, int]:
    seed = case_seed(dtype, pattern, height, width)
    if dtype == "uint16":
        return generate_uint16(pattern, height, width, seed), seed
    if dtype == "float32":
        return generate_float32(pattern, height, width, seed), seed
    raise ValueError(f"unknown dtype: {dtype}")


def fixture_name(dtype: str, pattern: str, height: int, width: int) -> str:
    return f"synthetic_{dtype}_{pattern}_{height}x{width}.npy"


def build_index(output: Path) -> dict[str, object]:
    cases = []
    for dtype in DTYPES:
        for pattern in PATTERNS:
            for height, width in SHAPES:
                array, seed = generate_case(dtype, pattern, height, width)
                cases.append(
                    {
                        "id": f"{dtype}-{pattern}-{height}x{width}",
                        "dtype": dtype,
                        "pattern": pattern,
                        "height": height,
                        "width": width,
                        "seed": seed,
                        "split": "tuning" if pattern in TUNING_PATTERNS else "evaluation",
                        "content_sha256": array_sha256(array),
                    }
                )

    fixtures = []
    for dtype, pattern, height, width in FIXTURES:
        array, seed = generate_case(dtype, pattern, height, width)
        path = output / fixture_name(dtype, pattern, height, width)
        save_array(path, array)
        fixtures.append(
            {
                "path": path.as_posix(),
                "seed": seed,
                "file_sha256": file_sha256(path),
                "content_sha256": array_sha256(array),
            }
        )

    return {
        "schema_version": 1,
        "generator_seed": GLOBAL_SEED,
        "bit_generator": "NumPy PCG64",
        "numpy_version": np.__version__,
        "shapes": [list(shape) for shape in SHAPES],
        "tile_geometry_under_test": {"width": 128, "height": 32},
        "cases": cases,
        "fixtures": fixtures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic TileStream fixtures")
    parser.add_argument("--output", type=Path, default=Path("datasets/fixtures"))
    parser.add_argument("--index", type=Path, default=Path("datasets/generated-cases.json"))
    args = parser.parse_args()

    index = build_index(args.output)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
