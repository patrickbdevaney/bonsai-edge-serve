#!/usr/bin/env python3
"""
Capture the reference (oracle) traces from the PrismML fork server.

Every custom backend in this repo -- CUDA, Vulkan, CPU -- gates against
these traces before any performance claim is made. Two things are captured
per workload prompt at temperature 0:

  1. Per-token top-k logprobs from the target model. This is the numerical
     oracle: a candidate backend must reproduce these within quant noise.
  2. DSpark accept/reject counts per request. At temperature 0 the accept
     decision is deterministic given identical logits, so a matching
     backend must make the same decisions.

Timing is also recorded, but timings are NOT part of the correctness gate --
they are the Milestone 0 measurement.

IMPORTANT -- gate the SAME mode against the same mode: native captures
against native captures, DSpark against DSpark. Native-vs-DSpark is not a
valid gate on this fork. Measured on Thor, DSpark output diverges from
native greedy output at temperature 0, and DSpark captures carry
placeholder logprobs (0.0) for every drafted token. See
docs/METHODOLOGY.md.

Usage:
    # server must already be running (see serve_reference.sh)
    python3 capture_traces.py --label ternary-native --out ../results/traces/thor-ternary-native.json
    python3 capture_traces.py --label ternary-dspark --out ../results/traces/thor-ternary-dspark.json

Compare two captures:
    python3 capture_traces.py --compare a.json b.json
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

WORKLOAD_DIR = Path(__file__).parent.parent / "bench" / "workloads"


def load_workload(name):
    path = WORKLOAD_DIR / f"{name}.jsonl"
    if not path.exists():
        sys.exit(f"workload not found: {path}")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip() and not l.startswith("//")]


def capture_one(url, prompt, n_predict, n_probs, timeout):
    """Non-streaming request; returns the full completion object.

    temperature 0 with top_k 1 makes decoding deterministic, which is what
    makes cross-backend trace comparison meaningful at all.
    """
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "n_probs": n_probs,
        "stream": False,
        "cache_prompt": False,
        "post_sampling_probs": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read())
    obj["_wall_s"] = time.perf_counter() - t0
    return obj


def extract_trace(obj, n_probs):
    """Reduce a completion response to the comparable trace form."""
    tokens = []
    for entry in obj.get("completion_probabilities", []) or []:
        top = []
        for p in (entry.get("top_logprobs") or entry.get("probs") or [])[:n_probs]:
            top.append({
                "tok": p.get("token", p.get("tok_str")),
                "logprob": p.get("logprob"),
                "prob": p.get("prob"),
            })
        tokens.append({
            "tok": entry.get("token", entry.get("content")),
            "logprob": entry.get("logprob"),
            "top": top,
        })

    t = obj.get("timings", {}) or {}
    drafted = t.get("draft_n", 0)
    accepted = t.get("draft_n_accepted", 0)
    return {
        "content": obj.get("content", ""),
        "tokens": tokens,
        "timings": {
            "prompt_n": t.get("prompt_n"),
            "predicted_n": t.get("predicted_n"),
            "decode_tok_s": t.get("predicted_per_second"),
            "prefill_tok_s": t.get("prompt_per_second"),
            "wall_s": obj.get("_wall_s"),
        },
        "dspark": {
            "draft_n": drafted,
            "draft_n_accepted": accepted,
            "acceptance": (accepted / drafted) if drafted else None,
        },
    }


def cmd_capture(args):
    names = ["code", "prose"] if args.workload == "all" else [args.workload]
    out = {
        "label": args.label,
        "url": args.url,
        "n_predict": args.n_predict,
        "n_probs": args.n_probs,
        "unix_time": time.time(),
        "traces": {},
    }
    for name in names:
        for item in load_workload(name):
            print(f"  capturing {item['id']} ...", flush=True)
            try:
                obj = capture_one(args.url, item["prompt"], args.n_predict,
                                  args.n_probs, args.timeout)
            except Exception as e:
                print(f"    FAILED: {e}", file=sys.stderr)
                continue
            tr = extract_trace(obj, args.n_probs)
            out["traces"][item["id"]] = tr
            d = tr["dspark"]
            acc = f"{d['acceptance']*100:.1f}%" if d["acceptance"] is not None else "n/a"
            print(f"    {len(tr['tokens'])} tokens, "
                  f"{tr['timings']['decode_tok_s']:.2f} tok/s, accept {acc}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


def cmd_compare(args):
    """Gate check: does a candidate backend reproduce the oracle?

    Two criteria, both reported:
      - token match: identical greedy token sequence (the strict gate)
      - logprob drift: max abs difference on the top-1 logprob, which is
        allowed to be nonzero within quant/reduction-order noise
    """
    a = json.loads(Path(args.compare[0]).read_text())
    b = json.loads(Path(args.compare[1]).read_text())
    print(f"A = {a['label']}  ({args.compare[0]})")
    print(f"B = {b['label']}  ({args.compare[1]})")
    print()

    shared = sorted(set(a["traces"]) & set(b["traces"]))
    if not shared:
        sys.exit("no overlapping trace ids")

    # The server reports logprob 0.0 as a placeholder for tokens that came
    # back through the DSpark draft path -- only the first, non-drafted
    # token of a request carries a real value. Comparing drift against such
    # a capture measures the placeholder, not numerics, so detect it and
    # fall back to a token-only comparison.
    # Not every token is a placeholder: where a draft was rejected, the
    # target produces the token and a real logprob comes back. So test the
    # fraction that are exactly 0.0. A genuine logprob of exactly 0.0 means
    # probability exactly 1.0, which does not occur in practice, so even a
    # modest fraction of them indicates the placeholder path.
    def is_placeholder(cap, frac=0.2):
        lps = [t.get("logprob") for tr in cap["traces"].values()
               for t in tr["tokens"][1:]]
        real = [x for x in lps if x is not None]
        if not real:
            return False
        return (sum(1 for x in real if x == 0.0) / len(real)) > frac

    ph_a, ph_b = is_placeholder(a), is_placeholder(b)
    drift_valid = not (ph_a or ph_b)
    if not drift_valid:
        which = " and ".join(n for n, p in ((a["label"], ph_a), (b["label"], ph_b)) if p)
        print(f"note: {which} reports placeholder logprobs for drafted tokens;")
        print("      comparing token sequences only, drift is not meaningful here.")
        print()

    worst_drift = 0.0
    n_divergent = 0
    for tid in shared:
        ta, tb = a["traces"][tid], b["traces"][tid]
        toks_a = [t["tok"] for t in ta["tokens"]]
        toks_b = [t["tok"] for t in tb["tokens"]]
        n = min(len(toks_a), len(toks_b))

        first_div = None
        for i in range(n):
            if toks_a[i] != toks_b[i]:
                first_div = i
                break

        drift = 0.0
        for i in range(n):
            la = ta["tokens"][i].get("logprob")
            lb = tb["tokens"][i].get("logprob")
            if la is not None and lb is not None:
                drift = max(drift, abs(la - lb))
        worst_drift = max(worst_drift, drift)

        if first_div is None and len(toks_a) == len(toks_b):
            status = "MATCH"
        else:
            status = f"DIVERGE@{first_div if first_div is not None else n}"
            n_divergent += 1
        print(f"  {tid:<24} {status:<16} max|dlogprob|={drift:.5f}  "
              f"({len(toks_a)} vs {len(toks_b)} tok)")

    print()
    print(f"divergent traces: {n_divergent}/{len(shared)}")
    if drift_valid:
        print(f"worst top-1 logprob drift: {worst_drift:.5f}")
        ok = n_divergent == 0 and worst_drift <= args.tol
        print(f"GATE {'PASS' if ok else 'FAIL'} (tolerance {args.tol})")
        return 0 if ok else 1

    ok = n_divergent == 0
    print(f"GATE {'PASS' if ok else 'FAIL'} (token comparison only)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--workload", default="all", help="code | prose | all")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--out", default="traces.json")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two capture files instead of capturing")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed top-1 logprob drift for a gate pass")
    args = ap.parse_args()

    if args.compare:
        sys.exit(cmd_compare(args))
    cmd_capture(args)


if __name__ == "__main__":
    main()
