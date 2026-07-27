#!/usr/bin/env python3
"""
Render the results table from bench JSON files.

Keeps the README honest: the published table is generated from measured
run artifacts in results/bench/, never hand-typed. Filenames are expected
to be <device>-<variant>-<mode>.json, matching run_milestone0.sh.

Usage:
    python3 render_table.py results/bench/*.json
    python3 render_table.py --speedup results/bench/*.json
"""

import argparse
import json
import sys
from pathlib import Path

VARIANT_LABEL = {"ternary": "ternary Q2_0", "onebit": "1-bit Q1_0"}
MODE_LABEL = {"native": "native", "dspark": "DSpark"}


def fmt(x, spec, dash="--"):
    return format(x, spec) if x is not None else dash


def load(paths):
    runs = {}
    for p in paths:
        obj = json.loads(Path(p).read_text())
        stem = Path(p).stem
        parts = stem.split("-")
        if len(parts) < 3:
            print(f"skipping {p}: filename is not <device>-<variant>-<mode>",
                  file=sys.stderr)
            continue
        device, variant, mode = parts[0], parts[1], parts[2]
        runs[(device, variant, mode)] = obj
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--speedup", action="store_true",
                    help="also emit the native-vs-DSpark speedup table")
    args = ap.parse_args()

    runs = load(args.paths)
    if not runs:
        sys.exit("no usable result files")

    print("| Variant | Mode | Workload | tok/s | TTFT (s) | Prefill tok/s | "
          "Acceptance | Resident |")
    print("| :-- | :-- | :-- | --: | --: | --: | --: | --: |")

    order = [("ternary", "native"), ("ternary", "dspark"),
             ("onebit", "native"), ("onebit", "dspark")]
    for device, variant, mode in sorted(
            runs, key=lambda k: (k[0], order.index((k[1], k[2]))
                                 if (k[1], k[2]) in order else 99)):
        obj = runs[(device, variant, mode)]
        # Resident memory is the host-side delta; on Jetson, CPU and GPU
        # share one pool and nvidia-smi reports nothing usable.
        mem = obj.get("host_mem_mib_before")
        for wl in ("code", "prose"):
            w = obj.get("workloads", {}).get(wl)
            if not w:
                continue
            s = w["summary"]
            acc = s.get("acceptance_pooled")
            acc_s = f"{acc*100:.1f}%" if acc is not None else "n/a"
            print(f"| {VARIANT_LABEL.get(variant, variant)} "
                  f"| {MODE_LABEL.get(mode, mode)} | {wl} "
                  f"| {fmt(s.get('decode_tok_s_median'), '.2f')} "
                  f"| {fmt(s.get('ttft_s_median'), '.3f')} "
                  f"| {fmt(s.get('prefill_tok_s_median'), '.1f')} "
                  f"| {acc_s} "
                  f"| {fmt(mem, '.0f') if mem is None else f'{mem/1024:.1f} GiB'} |")

    if not args.speedup:
        return

    print()
    print("| Variant | Workload | Native tok/s | DSpark tok/s | Speedup | Acceptance |")
    print("| :-- | :-- | --: | --: | --: | --: |")
    devices = sorted({k[0] for k in runs})
    for device in devices:
        for variant in ("ternary", "onebit"):
            nat = runs.get((device, variant, "native"))
            spec = runs.get((device, variant, "dspark"))
            if not nat or not spec:
                continue
            for wl in ("code", "prose"):
                nw = nat.get("workloads", {}).get(wl)
                sw = spec.get("workloads", {}).get(wl)
                if not nw or not sw:
                    continue
                n = nw["summary"].get("decode_tok_s_median")
                d = sw["summary"].get("decode_tok_s_median")
                acc = sw["summary"].get("acceptance_pooled")
                speed = (d / n) if (n and d) else None
                # A speedup below 1.00x is a real, reportable outcome for
                # this model family, not an error -- see README.
                print(f"| {VARIANT_LABEL.get(variant, variant)} | {wl} "
                      f"| {fmt(n, '.2f')} | {fmt(d, '.2f')} "
                      f"| {fmt(speed, '.2f')}x "
                      f"| {fmt(acc*100 if acc is not None else None, '.1f')}% |")


if __name__ == "__main__":
    main()
