#!/usr/bin/env bash
# Measure resident memory per configuration on a unified-memory device.
#
# Getting this right on Jetson took three attempts; see docs/METHODOLOGY.md.
#   - nvidia-smi reports "Not Supported" for memory usage entirely.
#   - host MemTotal-MemAvailable measures the whole machine, not the model
#     (it reported ~28 GiB for both a 7.2 GB and a 3.8 GB model).
#   - cudaMemGetInfo does not track the weights either: GPU memory *is*
#     system memory here, and mmap'd weights sit in reclaimable page cache,
#     so a 7.2 GB model moved the counter by only ~450 MiB.
#
# What works is the server process's peak RSS (VmHWM), which captures every
# weight page actually touched. It reads 6.99 GiB for the 7.2 GB ternary
# build, matching the file. VmRSS alone is NOT sufficient -- with mmap the
# kernel evicts pages, so instantaneous RSS reads far below the true
# footprint.
#
# Measured at two points:
#   loaded  -- server healthy, before any request (weights + context)
#   active  -- after one short decode (adds compute buffers / graphs)
#
# Usage: ./measure_memory.sh > ../results/memory.txt
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PORT="${PORT:-8080}"
GPU_MEM="$HERE/gpu_mem"

[ -x "$GPU_MEM" ] || { echo "build it first: nvcc -O2 -o $GPU_MEM $HERE/gpu_mem.cu" >&2; exit 1; }

# Peak RSS of the server process, in MiB.
peak_mib() { awk '/VmHWM/ {printf "%.0f", $2/1024}' "/proc/$1/status" 2>/dev/null; }

printf "%-16s %12s %12s\n" "config" "loaded_MiB" "active_MiB"

for cell in ternary:native ternary:dspark onebit:native onebit:dspark; do
    variant="${cell%%:*}"; mode="${cell##*:}"

    sync
    sleep 3

    PORT="$PORT" "$REPO/reference/serve_reference.sh" "$variant" "$mode" \
        > /tmp/memmeas-$variant-$mode.log 2>&1 &
    pid=$!

    ok=0
    for _ in $(seq 1 150); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    if [ "$ok" -ne 1 ]; then
        printf "%-16s %12s %12s\n" "$variant-$mode" "FAILED" "-"
        kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
        continue
    fi

    # serve_reference.sh execs the server, so $pid is the server itself.
    sleep 2
    loaded=$(peak_mib "$pid")

    curl -sf "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"def f(x):","n_predict":16,"temperature":0,"cache_prompt":false}' \
        >/dev/null 2>&1 || true
    sleep 2
    active=$(peak_mib "$pid")

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 3

    printf "%-16s %12s %12s\n" "$variant-$mode" "${loaded:-?}" "${active:-?}"
done

echo
echo "Peak RSS (VmHWM) of the llama-server process, MiB."
echo "On Jetson the GPU shares the system memory pool, so this is the"
echo "resident footprint; see docs/METHODOLOGY.md for why the obvious"
echo "measurements (nvidia-smi, cudaMemGetInfo, VmRSS) do not work here."
