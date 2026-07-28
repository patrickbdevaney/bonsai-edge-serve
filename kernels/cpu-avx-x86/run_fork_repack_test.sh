#!/usr/bin/env bash
# Validate the x86 AVX2/AVX-VNNI q1_0/q2_0 repack kernels against ggml's own
# scalar reference (ggml_gem{v,m}_q{1,2}_0_4x8_q8_0_generic), which is the
# function they replace. Both symbols are exported from libggml-cpu.so, so
# this links against the real build -- not a copy of the kernel that could
# drift from it. Mirror of ../cpu-neon-arm/run_q2_0_repack_test.sh.
#
#   ./run_fork_repack_test.sh
#
# Requires patches/0005-cpu-x86-avx2-repack.patch applied to the fork and
# `cmake --build <build> --target ggml-cpu`.
set -euo pipefail
cd "$(dirname "$0")"

FORK_DIR="${FORK_DIR:-$HOME/prism-llama.cpp}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
LIB="$FORK_DIR/$BUILD_DIR/bin"

[ -f "$LIB/libggml-cpu.so" ] || { echo "no libggml-cpu.so in $LIB" >&2; exit 2; }

# Count, don't grep -q: -q SIGPIPEs nm under pipefail (see the ARM script).
SYMS=$(nm -D --defined-only "$LIB/libggml-cpu.so" | grep -cE "ggml_gem[vm]_q[12]_0_4x8_q8_0$" || true)
if [ "$SYMS" -lt 4 ]; then
    echo "error: q1_0/q2_0 4x8 kernels not exported -- patch 0005 not applied?" >&2
    exit 2
fi

OUT="${TMPDIR:-/tmp}/test_fork_repack"
gcc -O2 -o "$OUT" test_fork_repack.c -L "$LIB" -lggml-cpu -Wl,-rpath,"$LIB" -lm
echo "q1_0/q2_0 x86 AVX2 repack kernels vs ggml generic reference:"
"$OUT"
