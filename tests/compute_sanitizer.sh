#!/usr/bin/env bash
# Runs the four Compute Sanitizer tools on the representative GPU suites.
# Usage: tests/compute_sanitizer.sh BUILD_DIR LOG_DIR
set -euo pipefail

build=${1:?build directory}
logs=${2:?log directory}
sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
mkdir -p "$logs"

# The API suite allocates past the device limit on purpose, so failed API calls are not
# reported. The sanitizer serializes streams, which defeats the overlap and synchronization
# checks of the loading suite and of NoHiddenDeviceSynchronization.
suites=(encoder decoder api calibration numerics robustness)
filter_api='-CudaApi.NoHiddenDeviceSynchronization'
export TILESTREAM_CASES=${TILESTREAM_CASES:-1000}
# Under racecheck, CUB's device scan sometimes never finishes when two host threads run it on two
# streams at once (3 of 20 runs); without its kernels, or under the other tools, it always does.
# The encoder suite still runs the scan under racecheck on a single stream.
racecheck_api=(--kernel-name-exclude kns=DeviceScan)
limit=${SANITIZER_TIMEOUT:-1800}

status=0
for tool in memcheck racecheck synccheck initcheck; do
    options=(--tool "$tool" --error-exitcode 1 --report-api-errors no)
    if [[ $tool == memcheck ]]; then
        options+=(--leak-check full)
    fi
    for suite in "${suites[@]}"; do
        extra=()
        args=()
        if [[ $suite == api ]]; then
            args=(--gtest_filter="$filter_api")
            if [[ $tool == racecheck ]]; then
                extra=("${racecheck_api[@]}")
            fi
        fi
        log="$logs/$tool-$suite.log"
        if timeout "$limit" "$sanitizer" "${options[@]}" "${extra[@]}" \
            "$build/tilestream_cuda_${suite}_tests" "${args[@]}" >"$log" 2>&1; then
            echo "$tool $suite: ok ($(grep -hE '(ERROR|RACECHECK) SUMMARY' "$log" | tail -n 1))"
        else
            echo "$tool $suite: FAILED, see $log"
            status=1
        fi
    done
done
exit "$status"
