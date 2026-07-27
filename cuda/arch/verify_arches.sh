#!/usr/bin/env bash
# Phase 3: compile-verify the CUDA backend for every target architecture.
#
# Only sm_110 (Thor) can be RUN here. The rest are compile-verified so a
# codegen break is caught locally, and are explicitly marked unvalidated
# in the results table until someone runs them on the real part.
#
#   sm_87   Jetson Orin / Orin Nano Super   -- deploy target, 8 GB
#   sm_86   RTX 3090
#   sm_110  Jetson Thor                     -- this box, RUN + validated
#   sm_120a RTX 5090                        -- fork's own validation target
#   sm_121a DGX Spark
#
# Builds only the ggml-cuda target: that is where every architecture-
# dependent path lives, and it keeps a 5-arch sweep affordable.
#
# Usage: ./verify_arches.sh [arch ...]
set -euo pipefail

FORK_DIR="${FORK_DIR:-$HOME/prism-llama.cpp}"
JOBS="${JOBS:-$(nproc)}"
OUT="${OUT:-/tmp/arch-verify}"
mkdir -p "$OUT"

ARCHES=("$@")
if [ ${#ARCHES[@]} -eq 0 ]; then
    ARCHES=(87-real 86-real 110-real 120a-real 121a-real)
fi

cd "$FORK_DIR"
printf "%-12s %-10s %s\n" "arch" "result" "notes"

for arch in "${ARCHES[@]}"; do
    bdir="build-arch-$arch"
    log="$OUT/$arch.log"
    if cmake -B "$bdir" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="$arch" \
             > "$log" 2>&1 \
       && cmake --build "$bdir" --target ggml-cuda -j "$JOBS" >> "$log" 2>&1; then
        note=""
        [ "$arch" = "110-real" ] && note="RUN + validated on this box"
        printf "%-12s %-10s %s\n" "$arch" "PASS" "$note"
    else
        printf "%-12s %-10s %s\n" "$arch" "FAIL" "see $log"
        grep -iE "error" "$log" | head -3 | sed 's/^/    /'
    fi
    rm -rf "$bdir"
done

echo
echo "PASS means the CUDA backend compiles for that architecture."
echo "It is NOT a correctness or performance claim: only sm_110 was run."
