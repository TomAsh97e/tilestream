# TileStream datasets

This directory contains deterministic generators, small committed fixtures, and a frozen tuning/evaluation split. Large source downloads are cached under `datasets/downloads/` and are not committed.

## Environment

Create an isolated environment and install the pinned data dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-data.txt
```

## Reproduce synthetic data

```sh
.venv/bin/python datasets/generate.py
```

The command regenerates the two committed synthetic fixtures and `datasets/generated-cases.json`. The index records the seed and canonical content SHA-256 for all 238 combinations of type, pattern, and boundary shape.

The generated shapes cover 1, 31/32/33, 127/128/129, and non-divisible dimensions for the default 128x32 tile. Patterns include constant fields, increasing and decreasing ramps, edges, checkerboards, uniform integer or normal FP32 noise, and smooth integer-derived fields with added seeded noise.

## Reproduce real-data fixtures

```sh
.venv/bin/python datasets/fetch_real.py
```

The command downloads versioned source files into the ignored cache, verifies their SHA-256 hashes, and creates:

- `ct_small_uint16.npy`, a deidentified CT image represented as `uint16`;
- `jacksboro_dem_float32.npy`, an SRTM elevation field represented as `float32`.

Conversions are specified in `datasets/manifest.json`. The committed fixtures contain only pixel arrays, not DICOM patient or acquisition metadata.

## Verify

```sh
.venv/bin/python -m pytest datasets/test_datasets.py
```

Tests verify every committed hash, regenerate every synthetic case, enforce the frozen split, and check fixture types and dimensions.

## Split policy

The split in `datasets/manifest.json` was frozen before codec optimization. Tuning data may guide implementation choices. Evaluation data must not be used to select parameters per image.

Repeating or tiling an image for a larger workload does not create a new real dataset. Such data must be marked `synthetic_stress` in future manifest entries.
