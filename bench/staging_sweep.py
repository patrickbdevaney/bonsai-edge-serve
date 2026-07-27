#!/usr/bin/env python3
"""
Sweep the DSpark staging-buffer cap against context length.

Answers the project's central question: can the 1-bit 27B build plus a
DSpark drafter plus usable KV fit an 8 GB Jetson Orin Nano Super?

Background. The reference sizes the drafter's staging batch to
`n_ctx + block_size`, because a round stages every context row since the
drafter's cache position in one batch, and the worst case is a long
history ingested in a single round. In steady-state decode the demand is
far smaller. Capping the batch trades that worst case for memory.

The tradeoff is real and sticky: a round needing more than the cap is
skipped, and a skipped round does NOT advance the drafter's cache
position, so a cap below the prompt length disables speculation for that
request entirely. This sweep measures where that boundary bites by
counting the fallbacks the server logs.

Requires the local patch adding BONSAI_DSPARK_MAX_BATCH (see
patches/ in this repo).

Usage:
    python3 staging_sweep.py                       # default grid
    python3 staging_sweep.py --variant ternary
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from measure_memory import parse  # noqa: E402

PORT = 8080


def start(variant, mode, ctx, cap, log_path):
    env = dict(os.environ)
    env["PORT"] = str(PORT)
    env["CTX"] = str(ctx)
    env["EXTRA_ARGS"] = "-lv 10"
    if cap:
        env["BONSAI_DSPARK_MAX_BATCH"] = str(cap)
    else:
        env.pop("BONSAI_DSPARK_MAX_BATCH", None)

    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)
    time.sleep(2)
    f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(REPO / "reference" / "serve_reference.sh"), variant, mode],
        stdout=f, stderr=subprocess.STDOUT, env=env)
    for _ in range(200):
        if proc.poll() is not None:
            return None
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return proc
        except Exception:
            time.sleep(2)
    return None


def run_workload(name, n_predict):
    """Return (median tok/s, pooled acceptance) over a workload file."""
    rates, drafted, accepted = [], 0, 0
    path = HERE / "workloads" / f"{name}.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        payload = {"prompt": item["prompt"], "n_predict": n_predict,
                   "temperature": 0, "top_k": 1, "cache_prompt": False}
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                t = json.loads(r.read())["timings"]
        except Exception as e:
            print(f"    request failed: {e}", file=sys.stderr)
            continue
        if t.get("predicted_n", 0) >= 8:
            rates.append(t["predicted_per_second"])
        drafted += t.get("draft_n", 0)
        accepted += t.get("draft_n_accepted", 0)
    rates.sort()
    med = rates[len(rates) // 2] if rates else None
    acc = (accepted / drafted) if drafted else None
    return med, acc, drafted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="onebit")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--cap", type=int, nargs="+", default=[256, 512, 1024, 0],
                    help="0 means uncapped (reference behaviour)")
    args = ap.parse_args()

    print(f"variant={args.variant}  n_predict={args.n_predict}")
    print(f"{'ctx':>6} {'cap':>7} {'total_MiB':>10} {'code_tps':>9} "
          f"{'prose_tps':>10} {'accept':>7} {'fallbacks':>10}")

    rows = []
    for ctx in args.ctx:
        for cap in args.cap:
            log = Path(f"/tmp/staging-{args.variant}-{ctx}-{cap}.log")
            proc = start(args.variant, "dspark", ctx, cap, log)
            if proc is None:
                print(f"{ctx:>6} {cap or 'none':>7} {'FAILED':>10}")
                continue
            time.sleep(2)
            tgt, dft = parse(log.read_text())
            total = sum(tgt.values()) + sum(dft.values())

            code_tps, acc, drafted = run_workload("code", args.n_predict)
            prose_tps, _, _ = run_workload("prose", args.n_predict)

            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(2)

            # Each skipped round logs "round needs N tokens > n_batch".
            fallbacks = len(re.findall(r"round needs \d+ tokens", log.read_text()))
            rows.append({"ctx": ctx, "cap": cap, "total_mib": round(total, 1),
                         "code_tps": code_tps, "prose_tps": prose_tps,
                         "acceptance": acc, "draft_n": drafted,
                         "fallbacks": fallbacks})
            print(f"{ctx:>6} {cap or 'none':>7} {total:>10.0f} "
                  f"{(code_tps or 0):>9.2f} {(prose_tps or 0):>10.2f} "
                  f"{(acc*100 if acc else 0):>6.1f}% {fallbacks:>10}")

    out = REPO / "results" / f"staging-sweep-{args.variant}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    print("fallbacks>0 means some rounds skipped speculation and that")
    print("sequence fell back to autoregressive decode.")


if __name__ == "__main__":
    main()
