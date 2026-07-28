#!/usr/bin/env bash
# Re-run the full Vulkan cell of the results matrix against the MMVQ build:
# {ternary, onebit} x {native, dspark}, both workload classes each.
#
# Both workload classes always run. Publishing only the high-accept code
# numbers is the selective-benchmark pattern this repo exists to avoid.
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8080}"
OUT="${OUT:-results/bench}"
SUF="${SUF:-.mmvq}"

for variant in ternary onebit; do
  for mode in native dspark; do
    label="thor.vulkan-${variant}-${mode}${SUF}"
    echo "=============== $label ==============="
    BACKEND=vulkan PORT="$PORT" ./reference/serve_reference.sh "$variant" "$mode" \
        > "/tmp/srv-$label.log" 2>&1 &
    srv=$!
    ok=0
    for _ in $(seq 1 200); do
        grep -q "server is listening" "/tmp/srv-$label.log" 2>/dev/null && { ok=1; break; }
        kill -0 $srv 2>/dev/null || break
        sleep 2
    done
    if [ "$ok" -ne 1 ]; then
        echo "  server did not start; see /tmp/srv-$label.log"
        tail -5 "/tmp/srv-$label.log"
        pkill -P $srv 2>/dev/null; kill $srv 2>/dev/null; wait $srv 2>/dev/null
        continue
    fi
    python3 bench/bench.py --url "http://127.0.0.1:$PORT" \
        --label "$label" --workload all --json "$OUT/$label.json" 2>&1 | tail -14
    pkill -P $srv 2>/dev/null; kill $srv 2>/dev/null; wait $srv 2>/dev/null
    sleep 3
  done
done
echo "done"
