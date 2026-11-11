from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DATASETS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATASETS_DIR.parent
sys.path.insert(0, str(DATASETS_DIR))

from common import array_sha256, file_sha256  # noqa: E402
from generate import DTYPES, PATTERNS, SHAPES, generate_case  # noqa: E402


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_and_generated_index_hashes() -> None:
    manifest = load_json(DATASETS_DIR / "manifest.json")
    suite = manifest["synthetic_suite"]
    index_path = PROJECT_DIR / suite["index"]

    assert file_sha256(index_path) == suite["index_sha256"]
    assert suite["case_count"] == len(DTYPES) * len(PATTERNS) * len(SHAPES)
    assert suite["numpy_version"] == np.__version__


def test_all_synthetic_cases_reproduce() -> None:
    index = load_json(DATASETS_DIR / "generated-cases.json")
    indexed_cases = {case["id"]: case for case in index["cases"]}

    assert len(indexed_cases) == len(DTYPES) * len(PATTERNS) * len(SHAPES)
    for dtype in DTYPES:
        for pattern in PATTERNS:
            for height, width in SHAPES:
                case_id = f"{dtype}-{pattern}-{height}x{width}"
                expected = indexed_cases[case_id]
                first, seed = generate_case(dtype, pattern, height, width)
                second, second_seed = generate_case(dtype, pattern, height, width)

                assert seed == expected["seed"] == second_seed
                assert first.shape == (height, width)
                assert first.dtype == np.dtype(dtype)
                assert first.flags.c_contiguous
                assert np.array_equal(first, second, equal_nan=True)
                assert array_sha256(first) == expected["content_sha256"]


def test_required_patterns_and_boundary_shapes_are_present() -> None:
    assert set(PATTERNS) == {
        "constant",
        "ramp_up",
        "ramp_down",
        "edges",
        "checkerboard",
        "noise",
        "smooth_noise",
    }
    square_sizes = {height for height, width in SHAPES if height == width}
    assert {1, 31, 32, 33, 127, 128, 129} <= square_sizes
    tile_boundary_shapes = {(height, width) for height in (31, 32, 33) for width in (127, 128, 129)}
    assert tile_boundary_shapes <= set(SHAPES)
    assert any(width % 128 != 0 and height % 32 != 0 for height, width in SHAPES)


def test_committed_fixtures_match_manifest() -> None:
    manifest = load_json(DATASETS_DIR / "manifest.json")
    for dataset in manifest["datasets"]:
        path = PROJECT_DIR / dataset["path"]
        array = np.load(path, allow_pickle=False)

        assert path.stat().st_size < 1024 * 1024
        assert file_sha256(path) == dataset["file_sha256"]
        assert array_sha256(array) == dataset["content_sha256"]
        assert array.dtype == np.dtype(dataset["dtype"])
        assert array.shape == (dataset["height"], dataset["width"])
        assert array.ndim == 2
        assert array.flags.c_contiguous


def test_real_sources_and_frozen_splits_are_distinct() -> None:
    manifest = load_json(DATASETS_DIR / "manifest.json")
    assert manifest["split_policy"]["frozen_before_optimization"] is True

    datasets = manifest["datasets"]
    real = [dataset for dataset in datasets if dataset["kind"].startswith("real_")]
    assert {(dataset["kind"], dataset["dtype"]) for dataset in real} == {
        ("real_image", "uint16"),
        ("real_field", "float32"),
    }
    assert len({dataset["source"]["url"] for dataset in real}) == 2
    assert {dataset["split"] for dataset in real} == {"tuning", "evaluation"}

    for dataset in real:
        source = dataset["source"]
        assert len(source["sha256"]) == 64
        assert (PROJECT_DIR / source["license_path"]).is_file()
        assert source["license_url"].startswith("https://")


def test_synthetic_pattern_split_is_disjoint() -> None:
    index = load_json(DATASETS_DIR / "generated-cases.json")
    tuning = {case["pattern"] for case in index["cases"] if case["split"] == "tuning"}
    evaluation = {case["pattern"] for case in index["cases"] if case["split"] == "evaluation"}

    assert tuning
    assert evaluation
    assert tuning.isdisjoint(evaluation)
