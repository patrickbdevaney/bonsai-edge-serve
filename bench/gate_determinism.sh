#!/usr/bin/env bash
# Determinism gate: N identical greedy requests must produce ONE distinct
# output. Anything else is a race, and a racy backend cannot be gated for
# correctness at all -- a divergence-vs-oracle number measured on top of a
# race is not reproducible and therefore means nothing.
#
# This exists because the repo previously recorded "Vulkan diverges from
# the CUDA oracle on 15/16 traces, 5 at token 0" and attributed it to
# `n_probs`. Both parts were wrong. n_probs is irrelevant; the backend was
# nondeterministic, so the trace comparison was measuring a different roll
# of the dice each time.
#
# cache_prompt is forced OFF. With it on, prefill runs once and every later
# request replays the cached KV, which HIDES a prefill race behind a cache
# hit -- the failure looks fixed while nothing changed.
#
# Usage: gate_determinism.sh <port> [label] [reps]
set -uo pipefail

PORT="${1:?usage: gate_determinism.sh <port> [label] [reps]}"
LABEL="${2:-backend}"
REPS="${3:-10}"
cd "$(dirname "$0")/.."

python3 - "$PORT" "$LABEL" "$REPS" <<'PY'
import json, sys, urllib.request

port, label, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])

prompts = {}
for w in ("code", "prose"):
    for line in open(f"bench/workloads/{w}.jsonl"):
        line = line.strip()
        if line and not line.startswith("//"):
            o = json.loads(line)
            prompts[o["id"]] = o["prompt"]

# Both workload classes. Prose is where near-ties cluster, so a code-only
# determinism check is the easy half of the test.
SAMPLE = ["code/py-binsearch", "code/cuda-reduce",
          "prose/review-restaurant", "prose/coastal-town"]

def run(pid):
    body = json.dumps({
        "prompt": prompts[pid], "n_predict": 30, "temperature": 0,
        "top_k": 1, "seed": 1234,
        "cache_prompt": False,        # see header -- caching hides the race
    }).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", body,
        {"Content-Type": "application/json"}), timeout=900))
    return r["content"]

print(f"=== {label}: {reps} identical greedy requests each, cache_prompt=False ===")
bad = 0
for pid in SAMPLE:
    outs = [run(pid) for _ in range(reps)]
    n = len(set(outs))
    bad += n > 1
    print(f"  {pid:26} distinct={n:2}  {'NONDETERMINISTIC' if n > 1 else 'stable'}")

print()
if bad:
    print(f"RESULT: FAIL -- {bad}/{len(SAMPLE)} prompts nondeterministic.")
    print("Do not run a correctness gate against this backend until it is")
    print("deterministic; the result would not be reproducible.")
    print("Known cause on ggml-vulkan: the graph_optimize reorder pass.")
    print("Workaround: GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 (costs ~2.2% prefill,")
    print("~1.0% decode on Thor, measured interleaved).")
    sys.exit(1)
print(f"RESULT: PASS -- all {len(SAMPLE)} prompts stable over {reps} runs.")
PY
