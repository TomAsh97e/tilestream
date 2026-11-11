from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pydicom

from common import file_sha256, save_array

SOURCES = {
    "ct": {
        "url": "https://raw.githubusercontent.com/pydicom/pydicom/v3.0.1/src/pydicom/data/test_files/CT_small.dcm",
        "filename": "CT_small.dcm",
        "sha256": "3dd31e5cc835b3f2cdd46c9da1982f59251e78518fefa8163d914631c66437d6",
    },
    "dem": {
        "url": "https://raw.githubusercontent.com/matplotlib/matplotlib/v3.9.2/lib/matplotlib/mpl-data/sample_data/jacksboro_fault_dem.npz",
        "filename": "jacksboro_fault_dem.npz",
        "sha256": "d493f50a33e82a4420494c54d1fca1539d177bdc27ab190bc5fe6e92f62fb637",
    },
}


def download(source: dict[str, str], cache: Path) -> Path:
    destination = cache / source["filename"]
    cache.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_sha256(destination) == source["sha256"]:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    for attempt in range(5):
        try:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "TileStream/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            break
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)

    actual = file_sha256(temporary)
    if actual != source["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {source['filename']}: {actual}")
    temporary.replace(destination)
    return destination


def convert_ct(source: Path) -> np.ndarray:
    dataset = pydicom.dcmread(source)
    pixels = dataset.pixel_array
    if pixels.shape != (128, 128) or pixels.ndim != 2:
        raise RuntimeError(f"unexpected CT dimensions: {pixels.shape}")
    if int(pixels.min()) < 0 or int(pixels.max()) > 65535:
        raise RuntimeError("CT stored values cannot be represented as uint16")
    return np.ascontiguousarray(pixels, dtype="<u2")


def convert_dem(source: Path) -> np.ndarray:
    with np.load(source, allow_pickle=False) as archive:
        if "elevation" not in archive:
            raise RuntimeError("DEM source has no elevation array")
        elevation = archive["elevation"]
    if elevation.shape != (344, 403) or elevation.ndim != 2:
        raise RuntimeError(f"unexpected DEM dimensions: {elevation.shape}")
    return np.ascontiguousarray(elevation, dtype="<f4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and convert real TileStream datasets")
    parser.add_argument("--cache", type=Path, default=Path("datasets/downloads"))
    parser.add_argument("--output", type=Path, default=Path("datasets/fixtures"))
    args = parser.parse_args()

    ct_source = download(SOURCES["ct"], args.cache)
    dem_source = download(SOURCES["dem"], args.cache)
    save_array(args.output / "ct_small_uint16.npy", convert_ct(ct_source))
    save_array(args.output / "jacksboro_dem_float32.npy", convert_dem(dem_source))

    for name in ("ct_small_uint16.npy", "jacksboro_dem_float32.npy"):
        path = args.output / name
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")


if __name__ == "__main__":
    main()
