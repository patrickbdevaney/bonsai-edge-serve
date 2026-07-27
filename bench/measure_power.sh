#!/usr/bin/env bash
# Energy per token, from Jetson's power rails.
#
# PrismML publish 0.275 mWh/token on an M5 Pro and use energy-per-token as
# part of their pitch, so extending it to Jetson is the comparable number.
# tegrastats exposes per-rail power; VIN is whole-board draw, which is the
# honest figure to compare against another whole device.
#
# Protocol per configuration:
#   1. sample idle power for IDLE_S with the server loaded but not decoding
#   2. run one long sustained generation, sampling throughout
#   3. energy/token = (mean VIN during decode) / (tok/s) / 3600, in mWh
#
# Idle is reported alongside because on an edge device the interesting
# quantity is often marginal energy (decode minus idle), not total.
#
# Usage: ./measure_power.sh [variant mode] > ../results/power.txt
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PORT="${PORT:-8080}"
IDLE_S="${IDLE_S:-10}"
N_PREDICT="${N_PREDICT:-256}"
BACKEND="${BACKEND:-cuda}"

command -v tegrastats >/dev/null || { echo "tegrastats not found (not a Jetson?)" >&2; exit 1; }

if [ $# -eq 2 ]; then CELLS=("$1:$2"); else
    CELLS=(ternary:native ternary:dspark onebit:native onebit:dspark); fi

# Mean of the VIN rail (whole board, mW) over a tegrastats capture.
mean_vin() { grep -oE "VIN [0-9]+mW" "$1" | grep -oE "[0-9]+" \
             | awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n; else print "0"}'; }

printf "%-16s %10s %10s %10s %12s\n" \
    "config" "idle_mW" "busy_mW" "tok/s" "mWh/token"

for cell in "${CELLS[@]}"; do
    variant="${cell%%:*}"; mode="${cell##*:}"
    pkill -x llama-server 2>/dev/null || true
    sleep 3

    PORT="$PORT" BACKEND="$BACKEND" "$REPO/reference/serve_reference.sh" \
        "$variant" "$mode" > "/tmp/power-$variant-$mode.log" 2>&1 &
    pid=$!
    ok=0
    for _ in $(seq 1 200); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    [ "$ok" -eq 1 ] || { printf "%-16s %10s\n" "$variant-$mode" "FAILED"; continue; }

    sleep 3
    timeout $((IDLE_S + 2)) tegrastats --interval 500 > "/tmp/idle-$variant-$mode.txt" 2>/dev/null &
    tg=$!
    sleep "$IDLE_S"
    kill $tg 2>/dev/null || true; wait $tg 2>/dev/null || true
    idle=$(mean_vin "/tmp/idle-$variant-$mode.txt")

    tegrastats --interval 500 > "/tmp/busy-$variant-$mode.txt" 2>/dev/null &
    tg=$!
    # A code prompt is used deliberately: several prose prompts make this
    # model emit EOS immediately, which yields zero decode time and a
    # sentinel rate. read() below rejects such a run rather than
    # reporting a nonsense energy figure.
    read -r tps ntok < <(curl -sf "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"# Python: implement an LRU cache with an OrderedDict.\\nfrom collections import OrderedDict\\n\\nclass LRUCache:\\n    def __init__(self, capacity):\\n\",\"n_predict\":$N_PREDICT,\"temperature\":0,\"cache_prompt\":false}" \
        | python3 -c "import json,sys; t=json.load(sys.stdin)['timings']; print(t['predicted_per_second'], t['predicted_n'])")
    kill $tg 2>/dev/null || true; wait $tg 2>/dev/null || true
    busy=$(mean_vin "/tmp/busy-$variant-$mode.txt")

    # Guard: too few tokens means the decode was too short to sample power.
    if [ "${ntok:-0}" -lt 32 ] || [ "$busy" = "0" ]; then
        printf "%-16s %10s %10s %10s %12s\n" "$variant-$mode" \
            "$idle" "$busy" "n=${ntok:-0}" "UNRELIABLE"
        kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
        continue
    fi

    # mWh per token = mW / (tokens/s) / 3600 s/h
    mwh=$(python3 -c "print(f'{$busy/$tps/3600:.4f}')")
    printf "%-16s %10s %10s %10.2f %12s\n" "$variant-$mode" "$idle" "$busy" "$tps" "$mwh"

    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    sleep 2
done

echo
echo "VIN is whole-board draw in mW. mWh/token uses busy power and the"
echo "measured decode rate. Compare against whole-device figures only."
