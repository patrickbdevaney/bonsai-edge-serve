#!/usr/bin/env python3
"""
Regenerate the README's cross-backend table from results/bench/*.json.

Hand-transcribing these numbers is how a stale row survives a rerun, and
the Vulkan rows went stale exactly that way once the MMVQ path landed. This
reads the JSON the harness wrote and prints the markdown, so the table
cannot disagree with the measurements it claims to summarise.

Both workload classes are always emitted. Reporting only the high-accept
code column is the selective-benchmark pattern the repo exists to avoid.

Usage:
    python3 bench/cross_backend_table.py                  # current results
    python3 bench/cross_backend_table.py --suffix .mmvq   # prefer a variant
"""

import argparse
import glob
import json
import os
import sys

BACKENDS = [("", "CUDA"), (".vulkan", "Vulkan"), (".cpu", "CPU (NEON)")]
VARIANTS = [("ternary", "ternary"), ("onebit", "1-bit")]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def cell(d, workload):
    if not d:
        return None
    w = d.get("workloads", {}).get(workload)
    if not w:
        return None
    return w["summary"]["decode_tok_s_median"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bench")
    ap.add_argument("--suffix", default="",
                    help="prefer files with this suffix (e.g. .mmvq); falls "
                         "back to the unsuffixed file when absent")
    args = ap.parse_args()

    def pick(backend, variant, mode):
        base = f"thor{backend}-{variant}-{mode}"
        for suf in ([args.suffix] if args.suffix else []) + [""]:
            p = os.path.join(args.dir, f"{base}{suf}.json")
            if os.path.exists(p):
                return load(p), suf
        return None, None

    print("| Backend | Variant | native code | native prose | "
          "DSpark code | DSpark prose |")
    print("| :-- | :-- | --: | --: | --: | --: |")
    notes = []
    for bk, bname in BACKENDS:
        for vk, vname in VARIANTS:
            nat, nsuf = pick(bk, vk, "native")
            spec, ssuf = pick(bk, vk, "dspark")
            row = [bname, vname]
            for wl in ("code", "prose"):
                v = cell(nat, wl)
                row.append(f"{v:.2f}" if v is not None else "--")
            for wl in ("code", "prose"):
                v, base = cell(spec, wl), cell(nat, wl)
                if v is None:
                    row.append("--")
                elif base:
                    r = v / base
                    # Bold only what actually beats not speculating, and by
                    # a margin that survives rounding: a raw ratio of
                    # 1.0001 displays as "1.00x", so bolding on r > 1.0
                    # marks a wash as a win.
                    txt = f"{v:.2f} ({r:.2f}x)"
                    row.append(f"**{txt}**" if r > 1.005 else txt)
                else:
                    row.append(f"{v:.2f}")
            print("| " + " | ".join(row) + " |")
            if nsuf:
                notes.append(f"{bname}/{vname} native from '{nsuf}' results")
            if ssuf:
                notes.append(f"{bname}/{vname} dspark from '{ssuf}' results")

    if notes:
        print()
        print("<!-- sources: " + "; ".join(sorted(set(notes))) + " -->")
    return 0


if __name__ == "__main__":
    sys.exit(main())
