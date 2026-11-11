#!/usr/bin/env python3
"""Runs the TileStream benchmark cases and records every sample with its environment.

    benchmarks/run.py run --build build/cuda-release --output results/benchmarks/run.json
    benchmarks/run.py summary results/benchmarks/run.json
    benchmarks/run.py compare first.json second.json
    benchmarks/run.py validate results/benchmarks/run.json

Only the standard library is used, so the script runs without the data environment.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = Path(__file__).resolve().parent / "schema.json"
FIXTURES = ROOT / "datasets" / "fixtures"

GPU_FIELDS = [
    "name", "driver_version", "pstate", "temperature.gpu", "power.draw", "power.limit",
    "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks_event_reasons.active",
    "utilization.gpu", "memory.used", "display_active", "persistence_mode",
]

# Environment variables that mean a profiler or sanitizer is attached.
INSTRUMENTATION = ["CUDA_INJECTION64_PATH", "NSYS_PROFILING_SESSION_ID", "LD_PRELOAD",
                   "ASAN_OPTIONS", "COMPUTE_SANITIZER"]


def suite(name):
    """Benchmark cases as (id, arguments)."""
    cases = []
    if name == "quick":
        for kind in ("uint16", "float32"):
            frame = ["--type", kind, "--width", "1024", "--height", "1024"]
            cases += [
                (f"gpu_encode/{kind}/1024", ["--scenario", "gpu_encode"] + frame),
                (f"gpu_decode_roi/{kind}/1024/roi256x4",
                 ["--scenario", "gpu_decode_roi", "--roi", "256", "--batch", "4"] + frame),
                (f"archive_roi/{kind}/1024/roi256x2",
                 ["--scenario", "archive_roi", "--roi", "256", "--batch", "2"] + frame),
            ]
        return cases
    if name == "baselines":
        return baselines()
    for kind in ("uint16", "float32"):
        frame = ["--type", kind, "--width", "4096", "--height", "4096"]
        cases += [
            (f"gpu_encode/{kind}/4096", ["--scenario", "gpu_encode"] + frame),
            (f"gpu_decode/{kind}/4096", ["--scenario", "gpu_decode"] + frame),
            (f"gpu_decode_roi/{kind}/4096/roi1024x8",
             ["--scenario", "gpu_decode_roi", "--roi", "1024", "--batch", "8"] + frame),
            (f"gpu_decode_calibrate/{kind}/4096/roi1024x8",
             ["--scenario", "gpu_decode_calibrate", "--roi", "1024", "--batch", "8"] + frame),
            (f"archive_roi/{kind}/4096/roi1024x4",
             ["--scenario", "archive_roi", "--roi", "1024", "--batch", "4"] + frame),
            (f"archive_roi/{kind}/4096/roi256x1",
             ["--scenario", "archive_roi", "--roi", "256", "--batch", "1"] + frame),
            (f"cpu_encode/{kind}/2048",
             ["--scenario", "cpu_encode", "--type", kind, "--width", "2048", "--height", "2048"]),
            (f"cpu_decode/{kind}/2048",
             ["--scenario", "cpu_decode", "--type", kind, "--width", "2048", "--height", "2048"]),
        ]
    # Small real images are short kernels; a batch of copies is one operation, not new data.
    ct = str(FIXTURES / "ct_small_uint16.npy")
    dem = str(FIXTURES / "jacksboro_dem_float32.npy")
    cases += [
        ("gpu_encode/ct_small/x64", ["--scenario", "gpu_encode", "--input", ct, "--batch", "64"]),
        ("gpu_decode/ct_small/x64", ["--scenario", "gpu_decode", "--input", ct, "--batch", "64"]),
        ("gpu_encode/jacksboro_dem/x16",
         ["--scenario", "gpu_encode", "--input", dem, "--batch", "16"]),
        ("gpu_decode/jacksboro_dem/x16",
         ["--scenario", "gpu_decode", "--input", dem, "--batch", "16"]),
    ]
    return cases


# nvCOMP LZ4: chunks of one TileStream tile, the vendor's 64 KiB default and the best
# configuration of the chunk, data type and bitshuffle sweep in docs/baselines.md.
LZ4 = {
    "controlled": ["--lz4-type", "char", "--bitshuffle", "0"],
    "default64k": ["--chunk", "65536", "--lz4-type", "char", "--bitshuffle", "0"],
    "tuned": ["--lz4-type", "typed", "--bitshuffle", "1"],
}


def baselines():
    """The same frames through TileStream and every baseline, grouped by boundary."""
    cases = []
    for kind in ("uint16", "float32"):
        frame = ["--type", kind, "--width", "4096", "--height", "4096"]
        cases += [
            (f"raw_calibrate/{kind}/4096", ["--scenario", "raw_calibrate"] + frame),
            (f"tsc_calibrate/{kind}/4096", ["--scenario", "tsc_calibrate"] + frame),
        ]
        for level in ("1", "3"):
            for threads in ("1", "8"):
                cases.append((f"zstd_calibrate/{kind}/4096/l{level}t{threads}",
                              ["--scenario", "zstd_calibrate", "--level", level,
                               "--threads", threads] + frame))
        cases += [(f"lz4_calibrate/{kind}/4096/{name}",
                   ["--scenario", "lz4_calibrate"] + args + frame) for name, args in LZ4.items()]
        cases += [
            (f"gpu_encode/{kind}/4096", ["--scenario", "gpu_encode"] + frame),
            (f"gpu_decode/{kind}/4096", ["--scenario", "gpu_decode"] + frame),
        ]
        for scenario in ("lz4_gpu_compress", "lz4_gpu_decompress"):
            cases += [(f"{scenario}/{kind}/4096/{name}", ["--scenario", scenario] + args + frame)
                      for name, args in LZ4.items()]
        cases += [(f"{scenario}/{kind}/4096", ["--scenario", scenario] + frame)
                  for scenario in ("raw_file", "tsc_file", "tiff_file")]
    return cases


def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def gpu_state():
    output = command(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
                      "--format=csv,noheader,nounits"])
    values = [v.strip() for v in output.split(",")] if output else []
    return dict(zip(GPU_FIELDS, values))


def gpu_processes():
    output = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"])
    processes = []
    for line in output.splitlines():
        pid, name, memory = [v.strip() for v in line.split(",")]
        processes.append({"pid": int(pid), "name": name, "used_memory_mib": memory})
    return processes


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def power_supplies():
    supplies = []
    for supply in sorted(Path("/sys/class/power_supply").glob("*")):
        entry = {"name": supply.name}
        for field in ("type", "scope", "online", "status", "capacity"):
            value = read(supply / field)
            if value is not None:
                entry[field] = value
        supplies.append(entry)
    return supplies


def cpu_model():
    for line in (read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def build_info(directory):
    cache = {}
    for line in (read(Path(directory) / "CMakeCache.txt") or "").splitlines():
        if ":" in line and "=" in line and not line.startswith(("//", "#")):
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    keys = ["CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CUDA_COMPILER",
            "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_ARCHITECTURES", "CMAKE_EXE_LINKER_FLAGS"]
    flags = " ".join(cache.get(k, "") for k in keys)
    return {
        "directory": str(directory),
        "build_type": cache.get("CMAKE_BUILD_TYPE", ""),
        "cache": {k: cache.get(k, "") for k in keys},
        "instrumented": cache.get("CMAKE_BUILD_TYPE") != "Release" or "-fsanitize" in flags
        or "-pg" in flags.split(),
    }


def environment(build):
    status = command(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {
        "git": {
            "commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "dirty": bool(status),
            "changed_files": len(status.splitlines()),
        },
        "build": build,
        "host": {
            "cpu": cpu_model(),
            "logical_cpus": os.cpu_count() or 1,
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "os": " ".join(platform.uname()[:3]),
            "python": platform.python_version(),
        },
        "power": {"supplies": power_supplies()},
        "gpu": {"before": gpu_state(), "after": None, "processes": gpu_processes()},
        "instrumentation": {k: os.environ[k] for k in INSTRUMENTATION if k in os.environ},
        # For example CUDA_DEVICE_MAX_CONNECTIONS changes whether copies and kernels overlap.
        "cuda_variables": {k: v for k, v in os.environ.items() if k.startswith("CUDA_")},
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(options):
    build = build_info(options.build)
    env = environment(build)
    if (build["instrumented"] or env["instrumentation"]) and not options.allow_instrumented:
        sys.exit("refusing to measure an instrumented build or process; "
                 "use a Release build without sanitizers or pass --allow-instrumented")
    # Another context on the GPU is time-sliced with ours and can add milliseconds to any wait.
    for process in env["gpu"]["processes"]:
        print(f"warning: GPU shared with pid {process['pid']} {process['name']}", file=sys.stderr)
    binary = Path(options.build) / "tilestream_codec_benchmark"
    selected = [c for c in suite(options.suite) if not options.filter or options.filter in c[0]]
    rounds = {case_id: [] for case_id, _ in selected}
    # Each round is a new process per case, and the order alternates, so that boost and memory
    # clock states, which change between processes, show up in the reported spread.
    for number in range(options.rounds):
        order = selected if number % 2 == 0 else selected[::-1]
        for case_id, args in order:
            args = args + ["--samples", str(options.samples), "--warmup", str(options.warmup)]
            before = gpu_state()
            process = subprocess.run([str(binary)] + args, capture_output=True, text=True,
                                     check=False)
            after = gpu_state()
            if process.returncode != 0:
                sys.exit(f"{case_id} failed:\n{process.stderr}")
            rounds[case_id].append((args, before, after, json.loads(process.stdout)))
            print(f"round {number + 1} {case_id:45} {describe(rounds[case_id][-1][3])}", flush=True)
    cases = []
    for case_id, _ in selected:
        results = [r[3] for r in rounds[case_id]]
        result = merge(results)
        if result["dataset"]["path"]:
            result["dataset"]["file_sha256"] = sha256(result["dataset"]["path"])
            result["dataset"]["path"] = os.path.relpath(result["dataset"]["path"], ROOT)
        cases.append({
            "id": case_id,
            "args": rounds[case_id][0][0],
            "rounds": [{"gpu_state": {"before": before, "after": after},
                        "repeat": r["config"]["repeat"],
                        "medians": {t["name"]: t["median"] for t in r["timings"]}}
                       for _, before, after, r in rounds[case_id]],
            "result": result,
        })
    env["gpu"]["after"] = gpu_state()
    document = {
        "schema": "tilestream-benchmark-run/1",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "environment": env,
        "cases": cases,
    }
    problems = validate(document, json.loads(SCHEMA.read_text()))
    if problems:
        sys.exit("result does not match the schema:\n" + "\n".join(problems))
    Path(options.output).parent.mkdir(parents=True, exist_ok=True)
    Path(options.output).write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {options.output}")


def percentile(sorted_samples, q):
    """Nearest rank, as in the benchmark program."""
    rank = math.ceil(q * len(sorted_samples))
    return sorted_samples[max(rank, 1) - 1]


def statistics(samples):
    ordered = sorted(samples)
    mean = sum(ordered) / len(ordered)
    variance = sum((x - mean) ** 2 for x in ordered) / max(len(ordered) - 1, 1)
    return {"count": len(ordered), "median": percentile(ordered, 0.5),
            "p05": percentile(ordered, 0.05), "p95": percentile(ordered, 0.95),
            "min": ordered[0], "max": ordered[-1], "mean": mean, "stdev": math.sqrt(variance)}


def merge(results):
    """All samples of all rounds of one case, with statistics over all of them."""
    merged = json.loads(json.dumps(results[0]))
    for i, timing in enumerate(merged["timings"]):
        samples = [x for r in results for x in r["timings"][i]["samples"]]
        timing.update(statistics(samples))
        timing["samples"] = samples
    medians = {t["name"]: t["median"] for t in merged["timings"]}
    for name in merged["throughput_gbps"]:
        merged["throughput_gbps"][name] = merged["bytes"]["counted"] / medians[name] / 1e6
    merged["config"]["rounds"] = len(results)
    merged["config"]["repeat"] = min(r["config"]["repeat"] for r in results)
    return merged


def primary(result):
    return result["timings"][0]


def describe(result):
    timing = primary(result)
    rate = result["throughput_gbps"].get(timing["name"], float("nan"))
    return (f"{timing['name']:16} median {timing['median']:9.4f} ms  p95 {timing['p95']:9.4f} ms"
            f"  {rate:7.2f} GB/s  ratio {result['bytes']['ratio_container']:5.2f}"
            f"  x{result['config']['repeat']}")


def summary(options):
    document = json.loads(Path(options.file).read_text())
    for case in document["cases"]:
        result = case["result"]
        print(f"{case['id']:45} {describe(result)}")
        for timing in result["timings"][1:]:
            print(f"{'':47}{timing['name']:16} median {timing['median']:9.4f} ms"
                  f"  p95 {timing['p95']:9.4f} ms  ({timing['clock']})")


def table(options):
    """A Markdown table of ratio, primary time, throughput and memory per case."""
    document = json.loads(Path(options.file).read_text())
    print("| case | ratio | median ms | p95 ms | GB/s | peak device MiB | pinned MiB |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for case in document["cases"]:
        result = case["result"]
        timing = primary(result)
        rate = result["throughput_gbps"].get(timing["name"], float("nan"))
        memory = result["memory"]
        print(f"| {case['id']} | {result['bytes']['ratio_container']:.3f} "
              f"| {timing['median']:.3f} | {timing['p95']:.3f} | {rate:.1f} "
              f"| {memory['peak_device_bytes'] / 2**20:.0f} "
              f"| {memory.get('pinned_host_bytes', 0) / 2**20:.0f} |")


def compare(options):
    """The second run reproduces the first when every primary median lies in [p05, p95]."""
    first = {c["id"]: c["result"] for c in json.loads(Path(options.first).read_text())["cases"]}
    second = json.loads(Path(options.second).read_text())["cases"]
    outside = 0
    for case in second:
        if case["id"] not in first:
            continue
        a = primary(first[case["id"]])
        b = primary(case["result"])
        inside = a["p05"] <= b["median"] <= a["p95"]
        outside += not inside
        print(f"{case['id']:45} {a['median']:9.4f} -> {b['median']:9.4f} ms "
              f"({b['median'] / a['median'] - 1:+6.1%}), first run [{a['p05']:.4f}, "
              f"{a['p95']:.4f}] {'ok' if inside else 'OUTSIDE'}")
    print(f"{outside} of {len(second)} medians outside the first run's p05-p95 range")
    sys.exit(1 if outside else 0)


def validate(value, schema, root=None, path="$"):
    """The subset of JSON Schema used by schema.json."""
    root = root or schema
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            target = target[part]
        return validate(value, target, root, path)
    problems = []
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float)}
    kind = schema.get("type")
    if kind and (not isinstance(value, kinds[kind]) or
                 (kind in ("integer", "number") and isinstance(value, bool))):
        return [f"{path}: expected {kind}"]
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not in {schema['enum']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        problems.append(f"{path}: below {schema['minimum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}: missing {key}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                problems += validate(value[key], sub, root, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            problems.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                problems += validate(item, schema["items"], root, f"{path}[{i}]")
    return problems


def check(options):
    problems = validate(json.loads(Path(options.file).read_text()), json.loads(SCHEMA.read_text()))
    print("\n".join(problems) or "valid")
    sys.exit(1 if problems else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="run a suite and write one result file")
    p.add_argument("--build", default=str(ROOT / "build" / "cuda-release"))
    p.add_argument("--suite", choices=["default", "quick", "baselines"], default="default")
    p.add_argument("--filter", default="", help="run only cases whose id contains this text")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5, help="separate processes per case")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-instrumented", action="store_true",
                   help="measure a Debug or sanitizer build; the result records it")
    p.set_defaults(handler=run)
    p = commands.add_parser("summary", help="print the timings of a result file")
    p.add_argument("file")
    p.set_defaults(handler=summary)
    p = commands.add_parser("table", help="print a Markdown table of a result file")
    p.add_argument("file")
    p.set_defaults(handler=table)
    p = commands.add_parser("compare", help="check that a second run reproduces the first")
    p.add_argument("first")
    p.add_argument("second")
    p.set_defaults(handler=compare)
    p = commands.add_parser("validate", help="check a result file against schema.json")
    p.add_argument("file")
    p.set_defaults(handler=check)
    options = parser.parse_args()
    if options.command == "run" and options.samples < 30:
        print("warning: fewer than 30 samples per case", file=sys.stderr)
    options.handler(options)


if __name__ == "__main__":
    main()
