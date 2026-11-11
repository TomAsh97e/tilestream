#!/usr/bin/env bash
# Nsight Compute reports of the dominant kernels. GPU performance counters need administrator
# rights unless the driver allows them for all users (RmProfilingAdminOnly=0):
#   sudo benchmarks/ncu.sh [build-dir] [output-dir]
set -euo pipefail

build=${1:-build/cuda-release}
out=${2:-results/performance/profiles}
bench="$build/tilestream_codec_benchmark"
ncu=${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}
mkdir -p "$out"

profile() {
    local name=$1 kernel=$2
    shift 2
    # Skip the warm-up launches and profile one steady-state kernel.
    "$ncu" --set full --kernel-name "regex:$kernel" --launch-skip 3 --launch-count 1 \
        --force-overwrite --export "$out/$name" \
        "$bench" "$@" --samples 1 --warmup 3 --min-sample-ms 0 --output /dev/null
}

for type in uint16 float32; do
    frame=(--type "$type" --width 4096 --height 4096)
    profile "decode-$type" '^decode<' --scenario gpu_decode "${frame[@]}"
    profile "decode-calibrate-$type" '^decode<' --scenario gpu_decode_calibrate "${frame[@]}"
    profile "analyze-$type" '^analyze<' --scenario gpu_encode "${frame[@]}"
    profile "pack-$type" '^pack<' --scenario gpu_encode "${frame[@]}"
done
