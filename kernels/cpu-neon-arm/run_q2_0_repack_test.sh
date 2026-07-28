#!/usr/bin/env bash
# Validate the ARM NEON q2_0 repack kernels against ggml's own scalar
# reference (ggml_gem{v,m}_q2_0_4x8_q8_0_generic), which is the function they
# replace. Both symbols are exported from libggml-cpu.so, so this links
# against the real build -- not a copy of the kernel that could drift from it.
#
#   ./run_q2_0_repack_test.sh
#
# Requires patches/0004-cpu-arm-q2_0-repack.patch applied to the fork and
# `cmake --build <build> --target ggml-cpu`.
set -euo pipefail
cd "$(dirname "$0")"

FORK_DIR="${FORK_DIR:-$HOME/prism-llama.cpp}"
BUILD_DIR="${BUILD_DIR:-build}"
LIB="$FORK_DIR/$BUILD_DIR/bin"

[ -f "$LIB/libggml-cpu.so" ] || { echo "no libggml-cpu.so in $LIB" >&2; exit 2; }

# If the kernels are not in the library, the test would link against the
# generic under both names and pass while comparing a function to itself.
# Note: no `grep -q` here. It exits on the first match, SIGPIPEs nm, and with
# `set -o pipefail` the pipeline reports failure -- so a correct build gets
# rejected by its own guard. Count instead.
SYMS=$(nm -D --defined-only "$LIB/libggml-cpu.so" | grep -c "ggml_gemv_q2_0_4x8_q8_0$" || true)
if [ "$SYMS" -eq 0 ]; then
    echo "error: ggml_gemv_q2_0_4x8_q8_0 not exported -- patch 0004 not applied?" >&2
    exit 2
fi

OUT="${TMPDIR:-/tmp}/test_q2_0_repack"
gcc -O2 -o "$OUT" test_q2_0_repack.c -L "$LIB" -lggml-cpu -Wl,-rpath,"$LIB" -lm
echo "q2_0 ARM repack kernels vs ggml generic reference:"
"$OUT"
