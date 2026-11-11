from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    shape = ",".join(str(value) for value in contiguous.shape)
    header = f"{contiguous.dtype.str};{shape}\n".encode("ascii")
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.save(output, np.ascontiguousarray(array), allow_pickle=False)
