#!/usr/bin/env bash
# Context-length sweep: what does long context actually COST on this model?
#
# Bonsai is 48/64 gated-delta-net. A GDN layer's recurrent state summarises
# the sequence and is FIXED SIZE regardless of length; only the 16
# full-attention layers grow a KV cache. So the usual "KV eats your edge
# device" arithmetic should be 4x milder here -- worth measuring rather than
# asserting, since n_ctx_train is 262144 and everything in this repo so far
# has run at 16K.
#
# Memory comes from ggml's own buffer accounting, not from RSS: on unified
# memory RSS is a trap (WIKI L1).
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-$HOME/models/bonsai/ternary/Ternary-Bonsai-27B-Q2_0.gguf}"
SRV="${SRV:-$HOME/prism-llama.cpp/build-vulkan2/bin/llama-server}"
CTXS="${CTXS:-4096 16384 65536 131072 262144}"
KVTYPES="${KVTYPES:-f16 q8_0}"
PORT="${PORT:-18099}"

parse_mem() {   # parse_mem <logfile>
    python3 - "$1" <<'PYEOF'
import re, sys
kv = rs = comp = 0.0
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r'llama_kv_cache: size\s*=\s*([\d.]+)\s*MiB', line)
    if m: kv = max(kv, float(m.group(1)))
    # The GDN state: fixed size by construction.
    m = re.search(r'llama_memory_recurrent: size\s*=\s*([\d.]+)\s*MiB', line)
    if m: rs = max(rs, float(m.group(1)))
    m = re.search(r'compute buffer size\s*=\s*([\d.]+)\s*MiB', line)
    if m: comp += float(m.group(1))
print(f"{kv:.0f} {rs:.0f} {comp:.0f} {kv+rs+comp:.0f}")
PYEOF
}

get_tps() {     # get_tps <port>
    curl -s --max-time 600 "http://127.0.0.1:$1/completion" \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"def binary_search(arr, target):","n_predict":48,"temperature":0,"cache_prompt":false}' \
    | python3 -c "import json,sys
try: print('%.2f' % json.load(sys.stdin)['timings']['predicted_per_second'])
except Exception: print('-')" 2>/dev/null
}

printf '%-8s %-6s %9s %9s %9s %10s %9s\n' ctx kv "KV MiB" "GDN MiB" "compute" "total MiB" "tok/s"

for kv in $KVTYPES; do
  for ctx in $CTXS; do
    log="/tmp/ctxsweep-$kv-$ctx.log"
    timeout 900 "$SRV" -m "$MODEL" -ngl 999 -fa on -c "$ctx" \
        -ctk "$kv" -ctv "$kv" --port "$PORT" --host 127.0.0.1 -lv 10 \
        > "$log" 2>&1 &
    pid=$!
    ok=0
    for _ in $(seq 1 220); do
        grep -q "server is listening" "$log" 2>/dev/null && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 3
    done
    if [ "$ok" -ne 1 ]; then
        why=$(grep -oiE "out of memory|failed to allocate|not supported|error" "$log" | head -1)
        printf '%-8s %-6s %9s %9s %9s %10s %9s\n' "$ctx" "$kv" "-" "-" "-" "-" "${why:-fail}"
        kill $pid 2>/dev/null; wait $pid 2>/dev/null
        sleep 2
        continue
    fi

    tps=$(get_tps "$PORT")
    read -r kvm rsm cmp tot <<<"$(parse_mem "$log")"
    printf '%-8s %-6s %9s %9s %9s %10s %9s\n' "$ctx" "$kv" "$kvm" "$rsm" "$cmp" "$tot" "$tps"

    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    sleep 2
  done
done
