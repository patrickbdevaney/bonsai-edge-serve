#!/usr/bin/env bash
# Gate for in-server DSpark speculative decoding.
#
# Speculative decoding is unusually easy to ship broken, because every failure
# mode still produces fluent text:
#
#   * a drafter that never fires  -> correct output, no speedup
#   * an undersized rollback ring -> correct-LOOKING output, corrupted state
#   * an off-by-one in accept-n   -> plausible output, wrong tokens
#
# So this gate asserts on the three things that actually distinguish those:
# that speculation is ENGAGED (acceptance > 0), that greedy speculative output
# is CHARACTER-IDENTICAL to greedy autoregressive output, and that the guard
# REFUSES to start when the rollback ring is too small.
#
# The identity check is the strong one. Speculative decoding is only a
# performance technique -- it is defined to produce exactly what the target
# model would have produced on its own. Any difference is either a bug or, on
# this model, a near-tie flip (see results/vulkan-divergence-explained.txt),
# which is why the threshold is "most prompts", not "all".
#
#   ./bench/gate_dspark.sh [SPEC_PORT] [AR_PORT]
#
# Expects two already-running servers: one with -md (speculating) and one
# without, on the SAME model.
set -uo pipefail
cd "$(dirname "$0")/.."

SPEC_PORT="${1:-8091}"
AR_PORT="${2:-8092}"
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }

echo "== DSpark gate: spec=:$SPEC_PORT  ar=:$AR_PORT"

# ---- 1. speculation is actually engaged
echo "-- engagement"
python3 - "$SPEC_PORT" <<'PY' && ok "drafter fires (alpha > 0, rounds > 0)" || bad "drafter never fired"
import json,sys,urllib.request
port=sys.argv[1]
b=json.dumps({"prompt":"def binary_search(arr, target):","n_predict":80,
              "temperature":0,"cache_prompt":False}).encode()
d=json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/completion",b,{"Content-Type":"application/json"}),timeout=600))
sp=d["bonsai"]["spec"]
# A null result here is the classic silent failure: everything "works", the
# output is right, and nothing was ever drafted.
assert sp.get("enabled"), "spec not enabled"
assert sp["rounds"]>0,   "no draft rounds"
assert sp["alpha"]>0,    "zero acceptance"
print(f"     alpha={sp['alpha']:.3f} rounds={sp['rounds']} tok/round={sp['tokens_per_round']:.2f}")
PY

# ---- 2. greedy speculative output == greedy autoregressive output
echo "-- greedy identity vs non-speculative"
python3 - "$SPEC_PORT" "$AR_PORT" <<'PY' && ok "greedy identity on >=4/5 prompts" || bad "greedy identity below threshold"
import json,sys,urllib.request
sp_port,ar_port=sys.argv[1],sys.argv[2]
def ask(port,p,n=150):
    b=json.dumps({"prompt":p,"n_predict":n,"temperature":0,"cache_prompt":False}).encode()
    d=json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",b,{"Content-Type":"application/json"}),timeout=900))
    t=d.get("content") or d.get("choices",[{}])[0].get("text","")
    # An empty body would make the comparison vacuously true -- the exact
    # shape of "a 200 that looks like success".
    assert t, "empty completion"
    return t
P=["def binary_search(arr, target):",
   "class LRUCache:\n    def __init__(self, capacity: int):",
   "static int levenshtein(const char *a, const char *b) {",
   "// CUDA kernel: parallel reduction sum\n__global__ void reduce(",
   "She had not expected the letter to arrive so late in the year."]
same=0
for p in P:
    a,b=ask(sp_port,p),ask(ar_port,p)
    same += a==b
    if a!=b:
        i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
        print(f"     diverges at char {i}: {p[:30]!r}")
print(f"     identical on {same}/{len(P)}")
assert same>=4, f"only {same}/5 identical"
PY

# ---- 3. the guard: refuse to run with an undersized rollback ring
#
# This is the one that matters most and the only one that cannot be checked by
# looking at output. With n_rs_seq=0 the post-verify crop silently no-ops and
# the gated-delta-net state absorbs every rejected draft tail -- the server
# must refuse to start rather than serve that.
echo "-- guard: undersized recurrent rollback ring"
MODEL="${MODEL:-$HOME/models/bonsai/onebit/Bonsai-27B-Q1_0.gguf}"
DRAFT="${DRAFT:-$HOME/models/bonsai/onebit/Bonsai-27B-dspark-Q4_1.gguf}"
if [ -f "$MODEL" ] && [ -f "$DRAFT" ] && [ -x ./build/bonsai-server ]; then
    out=$(BONSAI_FORCE_RS_SEQ0=1 timeout 300 ./build/bonsai-server \
            -m "$MODEL" -md "$DRAFT" --backend cuda --speculate on --slots 1 \
            -c 2048 --port 18097 --no-preflight --checkpoint-mb 0 2>&1)
    rc=$?
    if [ $rc -ne 0 ] && grep -q "rollback ring is 0" <<<"$out"; then
        ok "refuses to start when the ring cannot undo a draft block (exit $rc)"
    else
        bad "started (or wrong error) with a zero rollback ring -- exit $rc"
    fi
else
    echo "  SKIP guard check (model/drafter/binary not found)"
fi

echo
[ "$FAIL" -eq 0 ] && echo "gate_dspark: ALL PASS" || echo "gate_dspark: $FAIL FAILED"
exit "$FAIL"
