#!/usr/bin/env bash
# Numerical correctness gate for the Vulkan q2_0 MMVQ integration.
#
# Runs the SAME binary twice on the same prompts at temperature 0, once with
# the integer-dot kernel dispatching and once with GGML_VK_DISABLE_MMVQ=1
# forcing the f32 dequant path, and compares the generated token streams.
#
# Why this and not a kernel microbenchmark: a wrong extraction (e.g. Q2_K's
# `(w >> s) & 0x03030303`, which selects weights strided by 4 rather than
# consecutive ones) COMPILES, RUNS, and produces plausible-looking numbers.
# It is only visible against a reference. The two paths here share
# everything except the kernel under test, so any divergence beyond
# float-reassociation noise is that kernel.
#
# Divergence is scored as a PREFIX MATCH (tokens agreeing from the start),
# not as an exact string compare -- two runs that agree for 200 tokens and
# then differ on a near-tie are not the same failure as two runs that
# diverge immediately, and a length difference alone is not a divergence.
set -uo pipefail

MODEL="${MODEL:-$HOME/models/bonsai/ternary/Ternary-Bonsai-27B-Q2_0.gguf}"
BIN="${BIN:-$HOME/prism-llama.cpp/build-vulkan2/bin/llama-server}"
PORT="${PORT:-8099}"
NTOK="${NTOK:-160}"
OUT="${OUT:-/tmp/mmvq-gate}"
mkdir -p "$OUT"

# Both workload classes, per the repo rule: code (high accept) and prose
# (low accept) always travel together.
PROMPTS=(
  "Write a Python function that does binary search on a sorted list. Code only, no explanation."
  "Implement a thread-safe LRU cache in C++. Code only."
  "Explain in one paragraph why unified memory changes how you budget a KV cache on an edge device."
  "Describe the tradeoff between draft depth and acceptance rate in speculative decoding."
)

run_set() {  # run_set <label> <env-assignment-or-empty>
    local label="$1" envvar="$2"
    if [ -n "$envvar" ]; then export "${envvar?}"; fi
    "$BIN" -m "$MODEL" -ngl 999 -c 4096 --port "$PORT" --host 127.0.0.1 \
        > "$OUT/server-$label.log" 2>&1 &
    local pid=$!
    local ready=0
    for _ in $(seq 1 150); do
        if grep -q "server is listening" "$OUT/server-$label.log" 2>/dev/null; then ready=1; break; fi
        if ! kill -0 $pid 2>/dev/null; then break; fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        echo "  server failed to start for $label; see $OUT/server-$label.log" >&2
        kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
    fi

    local i=0
    for p in "${PROMPTS[@]}"; do
        python3 - "$PORT" "$NTOK" "$OUT/$label.$i.json" "$p" <<'PY'
import json, sys, urllib.request
port, ntok, path, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
body = json.dumps({"prompt": prompt, "n_predict": ntok, "temperature": 0,
                   "top_k": 1, "seed": 1234, "cache_prompt": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                             {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
json.dump({"content": r.get("content", ""),
           "tokens_predicted": r.get("tokens_predicted"),
           "tps": (r.get("timings") or {}).get("predicted_per_second")},
          open(path, "w"))
PY
        i=$((i+1))
    done
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    if [ -n "$envvar" ]; then unset "${envvar%%=*}"; fi
    return 0
}

echo "Vulkan q2_0 MMVQ -- numerical gate (temperature 0, greedy)"
echo "model: $MODEL"
echo
echo "[1/2] reference: MMVQ disabled (f32 dequant path)"
run_set ref "GGML_VK_DISABLE_MMVQ=1" || exit 1
echo "[2/2] under test: MMVQ enabled (integer-dot path)"
run_set mmvq "" || exit 1

echo
python3 - "$OUT" "${#PROMPTS[@]}" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2])
worst, fail = 1.0, 0
print(f"{'#':>2}  {'ref tok':>7} {'mmvq tok':>8}  {'prefix':>6}  {'match':>7}  verdict")
for i in range(n):
    a = json.load(open(f"{out}/ref.{i}.json"))["content"]
    b = json.load(open(f"{out}/mmvq.{i}.json"))["content"]
    k = 0
    for x, y in zip(a, b):
        if x != y: break
        k += 1
    denom = min(len(a), len(b)) or 1
    frac = k / denom
    worst = min(worst, frac)
    ok = frac >= 0.98
    fail |= (not ok)
    print(f"{i:>2}  {len(a):>7} {len(b):>8}  {k:>6}  {frac:>6.1%}  {'PASS' if ok else 'DIVERGENT'}")
print()
print(f"worst prefix match: {worst:.1%}")
if fail:
    print("RESULT: FAIL -- the integer-dot kernel disagrees with the reference.")
    print("A wrong extraction still produces plausible text, so treat any")
    print("early divergence as a kernel bug until proven to be a near-tie.")
    sys.exit(1)
print("RESULT: PASS -- integer-dot path agrees with the f32 reference.")
PY
