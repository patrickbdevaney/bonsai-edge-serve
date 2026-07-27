#!/usr/bin/env bash
# Milestone 0: first Jetson Thor (sm_110) numbers for the Bonsai-27B family,
# measured from the PrismML reference fork alone -- no custom engine involved.
#
# Sweeps variant x mode x workload, capturing for each cell:
#   decode tok/s, TTFT, prefill tok/s, DSpark acceptance %, resident memory
# plus the per-token logprob traces that gate every later backend.
#
# Usage:
#   ./run_milestone0.sh                 # full sweep
#   ./run_milestone0.sh ternary dspark  # one cell
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
RESULTS="${RESULTS:-$REPO/results}"
PORT="${PORT:-8080}"
N_PREDICT="${N_PREDICT:-128}"
DEVICE="${DEVICE:-thor}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"

mkdir -p "$RESULTS/bench" "$RESULTS/traces" "$RESULTS/raw"

if [ $# -eq 2 ]; then
    CELLS=("$1:$2")
else
    CELLS=(ternary:native ternary:dspark onebit:native onebit:dspark)
fi

wait_for_server() {
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$1" 2>/dev/null; then
            echo "server process died during startup" >&2
            return 1
        fi
        sleep 2
    done
    echo "server did not become healthy within ${STARTUP_TIMEOUT}s" >&2
    return 1
}

for cell in "${CELLS[@]}"; do
    variant="${cell%%:*}"
    mode="${cell##*:}"
    tag="$DEVICE-$variant-$mode"
    echo
    echo "================ $tag ================"

    logfile="$RESULTS/raw/$tag.server.log"
    PORT="$PORT" "$HERE/serve_reference.sh" "$variant" "$mode" > "$logfile" 2>&1 &
    server_pid=$!

    # serve_reference.sh execs the server, so $! is the server itself.
    if ! wait_for_server "$server_pid"; then
        echo "--- last 40 lines of $logfile ---" >&2
        tail -40 "$logfile" >&2
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        continue
    fi
    echo "server up (pid $server_pid)"

    python3 "$REPO/bench/bench.py" \
        --url "http://127.0.0.1:$PORT" \
        --label "$tag" \
        --workload all \
        --n-predict "$N_PREDICT" \
        --json "$RESULTS/bench/$tag.json" || echo "bench failed for $tag" >&2

    python3 "$HERE/capture_traces.py" \
        --url "http://127.0.0.1:$PORT" \
        --label "$tag" \
        --workload all \
        --n-predict "$N_PREDICT" \
        --out "$RESULTS/traces/$tag.json" || echo "trace capture failed for $tag" >&2

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "server stopped"
    sleep 3
done

echo
echo "==> results in $RESULTS"
