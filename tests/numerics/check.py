"""Independent NumPy check of tilestream_numerics_report.

    python -I tests/numerics/check.py results/numerics/summary.json build/numerics-dump

Recomputes the FP64 reference, the classes, the host calibration and every error statistic from
the dumped arrays, compares them with the report, and prints the Markdown tables used in
docs/numerical-analysis.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

MIN_NORMAL = 2.0**-126
MIN_SUBNORMAL = 2.0**-149
THRESHOLD = 2.0**128 - 2.0**103
KINDS = ["nan_input", "invalid", "infinite", "intermediate_overflow", "overflow", "zero",
         "subnormal", "normal"]
SHAPES = ["nan", "infinite", "zero", "subnormal", "normal"]


def classify(x, offset, gain):
    with np.errstate(invalid="ignore", over="ignore"):
        difference = x.astype(np.float64) - offset.astype(np.float64)
        ref = difference * gain.astype(np.float64)
        magnitude = np.abs(ref)
        conditions = [
            np.isnan(x) | np.isnan(offset) | np.isnan(gain),
            np.isnan(ref),
            np.isinf(ref),
            np.abs(difference) >= THRESHOLD,
            magnitude >= THRESHOLD,
            ref == 0,
            magnitude < MIN_NORMAL,
        ]
    return np.select(conditions, range(7), default=7), ref


def shapes(values):
    magnitude = np.abs(values)
    with np.errstate(invalid="ignore"):
        codes = np.select([np.isnan(values), np.isinf(values), values == 0,
                           magnitude < np.float32(MIN_NORMAL)], range(4), default=4)
    return {name: int((codes == i).sum()) for i, name in enumerate(SHAPES)}


def same(a, b):
    return (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))


def summary(values):
    if values.size == 0:
        return {"count": 0, "max": 0.0, "median": 0.0, "p99": 0.0}
    return {
        "count": int(values.size),
        "max": float(values.max()),
        "median": float(np.quantile(values, 0.5, method="inverted_cdf")),
        "p99": float(np.quantile(values, 0.99, method="inverted_cdf")),
    }


def variant_metrics(values, controlled, kinds, ref):
    finite_ref = (kinds == 5) | (kinds == 6) | (kinds == 7)
    usable = finite_ref & np.isfinite(values)
    absolute = np.abs(values[usable].astype(np.float64) - ref[usable])
    magnitude = np.abs(ref[usable])
    _, exponent = np.frexp(magnitude)
    ulp = np.where(magnitude < MIN_NORMAL, MIN_SUBNORMAL, np.ldexp(1.0, exponent - 24))
    ulps = absolute / ulp
    used = kinds[usable]
    return {
        "shapes": shapes(values),
        "differs_from_controlled": int((~same(values, controlled)).sum()),
        "nonfinite_for_finite_reference": int((finite_ref & ~np.isfinite(values)).sum()),
        "absolute": summary(absolute),
        "relative": summary(absolute / np.maximum(magnitude, MIN_NORMAL)),
        "ulps": summary(ulps),
        "ulps_normal": summary(ulps[used == 7]),
        "absolute_subnormal": summary(absolute[used == 6]),
    }


def compare(expected, actual, path, problems):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(value, actual[key], f"{path}.{key}", problems)
    elif isinstance(expected, float):
        if not np.isclose(expected, actual, rtol=1e-12, atol=0.0):
            problems.append(f"{path}: numpy {expected!r}, report {actual!r}")
    elif expected != actual:
        problems.append(f"{path}: numpy {expected!r}, report {actual!r}")


def check_case(case, directory, problems):
    base = Path(directory) / case["name"]
    load = lambda suffix: np.load(f"{base}.{suffix}.npy")
    x, offset, gain = load("x"), load("offset"), load("gain")
    if case["integer"]:
        if not np.array_equal(load("samples").astype(np.float32), x):
            problems.append(f"{case['name']}: uint16 samples do not convert to x")
    kinds, ref = classify(x, offset, gain)
    counts = {name: int((kinds == i).sum()) for i, name in enumerate(KINDS)}
    compare(counts, case["kinds"], f"{case['name']}.kinds", problems)

    controlled = load("controlled")
    with np.errstate(all="ignore"):
        host = (x - offset) * gain
    if host.dtype != np.float32 or not same(host, controlled).all():
        problems.append(f"{case['name']}: NumPy float32 calibration differs from the device")
    for name, reported in case["variants"].items():
        metrics = variant_metrics(load(name), controlled, kinds, ref)
        compare(metrics, reported, f"{case['name']}.{name}", problems)


def ulp_cell(stats):
    if stats["count"] == 0:
        return "-"
    return f"{stats['max']:.3g} / {stats['p99']:.3g} / {stats['median']:.3g}"


def tables(report):
    print("### Classes\n")
    print("| case | samples | " + " | ".join(KINDS) + " |")
    print("|---|---:|" + "---:|" * len(KINDS))
    for case in report["cases"]:
        cells = " | ".join(str(case["kinds"][k]) for k in KINDS)
        print(f"| {case['name']} | {case['samples']} | {cells} |")

    print("\n### Controlled calibration\n")
    print("| case | ULP max / p99 / median (normal) | relative max | absolute max (subnormal)"
          " | violations |")
    print("|---|---|---:|---:|---:|")
    for case in report["cases"]:
        m = case["variants"]["controlled"]
        sub = m["absolute_subnormal"]
        print(f"| {case['name']} | {ulp_cell(m['ulps_normal'])} | {m['relative']['max']:.3g} | "
              f"{sub['max']:.3g} | {m['violations']} |" if sub["count"] else
              f"| {case['name']} | {ulp_cell(m['ulps_normal'])} | {m['relative']['max']:.3g} | - | "
              f"{m['violations']} |")

    variants = [v for v in report["cases"][0]["variants"] if v not in ("controlled", "fused")]
    print("\n### Variants\n")
    print("Cells: samples differing from the library / contract violations / "
          "ULP max, p99 for normal results.\n")
    print("| case | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    for case in report["cases"]:
        cells = []
        for v in variants:
            m = case["variants"][v]
            u = m["ulps_normal"]
            cells.append(f"{m['differs_from_controlled']} / {m['violations']} / "
                         f"{u['max']:.3g}, {u['p99']:.3g}")
        print(f"| {case['name']} | " + " | ".join(cells) + " |")

    print("\n### Examples\n")
    names = list(report["examples"][0]["results"])
    print("| x | offset | gain | FP64 reference | " + " | ".join(names) + " |")
    print("|---|---|---|---|" + "---|" * len(names))
    word = lambda h: f"{np.uint32(int(h, 16)).view(np.float32):.9g}"
    for e in report["examples"]:
        cells = " | ".join(word(e["results"][n]) for n in names)
        print(f"| {word(e['x'])} | {word(e['offset'])} | {word(e['gain'])} | "
              f"{e['reference']:.9g} | {cells} |")


def main():
    report = json.loads(Path(sys.argv[1]).read_text())
    problems = []
    for case in report["cases"]:
        check_case(case, sys.argv[2], problems)
    if problems:
        print("\n".join(problems[:40]), file=sys.stderr)
        print(f"numpy check FAILED: {len(problems)} differences", file=sys.stderr)
        return 1
    tables(report)
    print(f"\nnumpy check passed for {len(report['cases'])} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
