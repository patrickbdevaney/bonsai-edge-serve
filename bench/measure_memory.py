#!/usr/bin/env python3
"""
Measure resident memory per configuration, from ggml's own allocator.

Observing the process from outside does not work on unified memory --
nvidia-smi, MemTotal-MemAvailable, cudaMemGetInfo, VmRSS and even peak
RSS all give wrong or non-additive answers on Jetson (see
docs/METHODOLOGY.md and results/memory.txt). ggml, however, logs exactly
what it allocates and on which backend. At -lv 10 the server emits:

    load_tensors:        CUDA0 model buffer size =  6500.64 MiB
    load_tensors:   CPU_Mapped model buffer size =   322.07 MiB
    llama_kv_cache:      CUDA0 KV buffer size =  1024.00 MiB
    llama_memory_recurrent:  CUDA0 RS buffer size =   149.62 MiB
    sched_reserve:       CUDA0 compute buffer size =   138.02 MiB

That is exact, additive, and attributable to target vs drafter. This
script drives a server per configuration, parses those lines, and prints
the breakdown.

Two parsing subtleties:
  - The fork runs a parameter-fitting dry pass before the real load. Its
    buffers are logged too, with `mmap = false`, and mostly read 0.00 MiB.
    Only lines after the first `mmap = true` are real allocations.
  - `loading draft model` separates target buffers from drafter buffers.

Usage:
    python3 measure_memory.py                 # all four configurations
    python3 measure_memory.py ternary dspark  # one
"""

import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
PORT = 8080

BUF_RE = re.compile(
    r"(?:load_tensors|llama_kv_cache|llama_memory_recurrent|sched_reserve|llama_context):\s+"
    r"(\S+)\s+(model|KV|RS|compute|output)\s+buffer size\s*=\s*([0-9.]+)\s*MiB"
)


def parse(log_text):
    """Return (target_buffers, draft_buffers) as {(backend, kind): MiB}."""
    target, draft = {}, {}
    real_load = False
    cur = target
    for line in log_text.splitlines():
        if "mmap = true" in line:
            real_load = True
        if "loading draft model" in line:
            cur = draft
        if not real_load:
            continue
        m = BUF_RE.search(line)
        if not m:
            continue
        backend, kind, mib = m.group(1), m.group(2), float(m.group(3))
        if mib == 0.0:
            continue
        # sched_reserve can log the same buffer more than once; keep the
        # largest, which is the reservation actually held.
        key = (backend, kind)
        cur[key] = max(cur.get(key, 0.0), mib)
    return target, draft


def run_config(variant, mode):
    log = Path(f"/tmp/memparse-{variant}-{mode}.log")
    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)
    time.sleep(2)

    env_cmd = [str(REPO / "reference" / "serve_reference.sh"), variant, mode]
    with open(log, "w") as f:
        proc = subprocess.Popen(env_cmd, stdout=f, stderr=subprocess.STDOUT,
                                env={**__import__("os").environ,
                                     "PORT": str(PORT),
                                     "EXTRA_ARGS": "-lv 10"})

    ok = False
    for _ in range(200):
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            ok = True
            break
        except Exception:
            time.sleep(2)

    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(2)

    if not ok:
        print(f"  {variant}-{mode}: server failed to start", file=sys.stderr)
        return None
    return parse(log.read_text())


def report(variant, mode, target, draft):
    def total(bufs, backend_pred):
        return sum(v for (b, _), v in bufs.items() if backend_pred(b))

    is_dev = lambda b: b.startswith("CUDA") and b != "CUDA_Host"
    is_host = lambda b: not is_dev(b)

    print(f"=== {variant} / {mode} ===")
    for label, bufs in (("target", target), ("drafter", draft)):
        if not bufs:
            continue
        print(f"  {label}:")
        for (backend, kind), mib in sorted(bufs.items(),
                                           key=lambda kv: -kv[1]):
            print(f"      {mib:9.2f} MiB  {backend:<12} {kind}")
    dev = total(target, is_dev) + total(draft, is_dev)
    host = total(target, is_host) + total(draft, is_host)
    print(f"  ---- device (CUDA0): {dev:9.2f} MiB")
    print(f"  ---- host:           {host:9.2f} MiB")
    print(f"  ---- TOTAL:          {dev + host:9.2f} MiB")
    print()
    return dev, host


def main():
    if len(sys.argv) == 3:
        cells = [(sys.argv[1], sys.argv[2])]
    else:
        cells = [("ternary", "native"), ("ternary", "dspark"),
                 ("onebit", "native"), ("onebit", "dspark")]

    summary = []
    for variant, mode in cells:
        res = run_config(variant, mode)
        if res is None:
            continue
        dev, host = report(variant, mode, *res)
        summary.append((variant, mode, dev, host))

    print("| Variant | Mode | Device MiB | Host MiB | Total MiB |")
    print("| :-- | :-- | --: | --: | --: |")
    for variant, mode, dev, host in summary:
        print(f"| {variant} | {mode} | {dev:.0f} | {host:.0f} | {dev + host:.0f} |")


if __name__ == "__main__":
    main()
