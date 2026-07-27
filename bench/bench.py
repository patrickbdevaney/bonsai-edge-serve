#!/usr/bin/env python3
"""
Backend-agnostic benchmark harness for bonsai-edge-serve.

Talks to any llama.cpp-compatible /completion endpoint (the PrismML fork
reference server, or this repo's own engines once they expose the same
surface) and reports the metrics the cross-device scaling table needs:

    tok/s (decode), TTFT, DSpark acceptance %, resident memory

Acceptance comes from the server's own timings object: `draft_n` (tokens
drafted) and `draft_n_accepted` (tokens the target accepted). Acceptance %
is reported as accepted/drafted, which is the quantity the model card
quotes (~69.2% ternary, 74.6-78.6% 1-bit on code, ~51% on prose).

Both workload classes are run by default. Publishing only the high-accept
code numbers is exactly the selective-benchmark pattern this repo exists
to avoid -- `--workload` narrows it only for debugging.

Usage:
    python3 bench.py --label "thor/ternary/dspark" --workload all
    python3 bench.py --url http://127.0.0.1:8080 --n-predict 128 --json out.json
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKLOAD_DIR = Path(__file__).parent / "workloads"


def load_workload(name):
    path = WORKLOAD_DIR / f"{name}.jsonl"
    if not path.exists():
        sys.exit(f"workload not found: {path}")
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                prompts.append(json.loads(line))
    return prompts


def post_stream(url, payload, timeout):
    """POST to /completion with streaming; return (ttft_s, total_s, text, timings)."""
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    timings = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            raw = raw.strip()
            if not raw or not raw.startswith(b"data: "):
                continue
            obj = json.loads(raw[6:])
            piece = obj.get("content", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            if piece:
                chunks.append(piece)
            if obj.get("timings"):
                timings = obj["timings"]
    total = time.perf_counter() - t0
    return ttft, total, "".join(chunks), timings


def gpu_memory_mib():
    """Resident GPU memory, best effort. Returns None where unavailable
    (notably Jetson, where nvidia-smi reports 'Not Supported' for the
    unified-memory pool -- see docs/METHODOLOGY.md)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        val = out.stdout.strip().splitlines()[0].strip()
        return int(val)
    except Exception:
        return None


def host_memory_mib():
    """Total system RAM in use, from /proc/meminfo. On Jetson this is the
    meaningful footprint number, since GPU and CPU share one pool."""
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.strip().split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) // 1024
    except Exception:
        return None


def run_workload(url, prompts, n_predict, temperature, timeout, warmup):
    if warmup:
        try:
            post_stream(url, {"prompt": "warmup", "n_predict": 8, "temperature": 0,
                              "stream": True, "cache_prompt": False}, timeout)
        except Exception as e:
            print(f"  warmup failed: {e}", file=sys.stderr)

    rows = []
    for i, item in enumerate(prompts):
        payload = {
            "prompt": item["prompt"],
            "n_predict": n_predict,
            "temperature": temperature,
            "stream": True,
            # No prefix reuse between prompts: every row measures a cold
            # decode, so tok/s is not inflated by a warm cache.
            "cache_prompt": False,
        }
        if temperature == 0:
            payload["top_k"] = 1
        try:
            ttft, total, text, timings = post_stream(url, payload, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  [{item['id']}] request failed: {e}", file=sys.stderr)
            continue

        drafted = timings.get("draft_n", 0)
        accepted = timings.get("draft_n_accepted", 0)
        row = {
            "id": item["id"],
            "ttft_s": ttft,
            "total_s": total,
            "prompt_n": timings.get("prompt_n"),
            "predicted_n": timings.get("predicted_n"),
            "decode_tok_s": timings.get("predicted_per_second"),
            "prefill_tok_s": timings.get("prompt_per_second"),
            "draft_n": drafted,
            "draft_n_accepted": accepted,
            "acceptance": (accepted / drafted) if drafted else None,
            "output_chars": len(text),
        }
        rows.append(row)
        acc = f"{row['acceptance']*100:.1f}%" if row["acceptance"] is not None else "n/a"
        tps = row["decode_tok_s"]
        if tps is None:
            print(f"  [{item['id']:<18}] no timings returned by server")
        else:
            # ttft is None when the response carried no content chunk at
            # all (e.g. the model emitted EOS immediately).
            ttft_s = f"{ttft:6.3f}s" if ttft is not None else "    n/a"
            print(f"  [{item['id']:<18}] {tps:6.2f} tok/s  "
                  f"ttft {ttft_s}  accept {acc}")
    return rows


def summarize(rows):
    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    drafted = sum(r["draft_n"] or 0 for r in rows)
    accepted = sum(r["draft_n_accepted"] or 0 for r in rows)
    return {
        "n_requests": len(rows),
        "decode_tok_s_median": med("decode_tok_s"),
        "decode_tok_s_mean": (statistics.mean([r["decode_tok_s"] for r in rows
                                               if r.get("decode_tok_s")])
                              if any(r.get("decode_tok_s") for r in rows) else None),
        "ttft_s_median": med("ttft_s"),
        "prefill_tok_s_median": med("prefill_tok_s"),
        "draft_n_total": drafted,
        "draft_n_accepted_total": accepted,
        # Pooled over all tokens, not a mean of per-request rates: long
        # requests should weigh more than short ones.
        "acceptance_pooled": (accepted / drafted) if drafted else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--workload", default="all",
                    help="code | prose | all (default: all -- both classes are required)")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="unlabeled",
                    help="e.g. thor/ternary/dspark -- recorded in the JSON output")
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    names = ["code", "prose"] if args.workload == "all" else [args.workload]

    result = {
        "label": args.label,
        "url": args.url,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
        "unix_time": time.time(),
        "host_mem_mib_before": host_memory_mib(),
        "gpu_mem_mib": gpu_memory_mib(),
        "workloads": {},
    }

    for name in names:
        prompts = load_workload(name)
        print(f"== workload: {name} ({len(prompts)} prompts) ==")
        rows = run_workload(args.url, prompts, args.n_predict, args.temperature,
                            args.timeout, not args.no_warmup)
        summ = summarize(rows)
        result["workloads"][name] = {"summary": summ, "rows": rows}
        if not rows:
            print("  -> no successful requests\n")
            continue
        acc = summ["acceptance_pooled"]
        acc_s = f"{acc*100:.1f}%" if acc is not None else "n/a (no drafter)"
        tps_s = (f"{summ['decode_tok_s_median']:.2f}"
                 if summ["decode_tok_s_median"] is not None else "n/a")
        ttft_s = (f"{summ['ttft_s_median']:.3f}s"
                  if summ["ttft_s_median"] is not None else "n/a")
        print(f"  -> median {tps_s} tok/s, TTFT {ttft_s}, acceptance {acc_s}\n")

    result["host_mem_mib_after"] = host_memory_mib()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
