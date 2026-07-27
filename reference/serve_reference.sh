#!/usr/bin/env bash
# Canonical llama-server invocations for the Bonsai-27B family, via the
# PrismML fork. These are the reference (oracle) configurations -- every
# custom backend in this repo is gated against traces captured from these.
#
# Usage:
#   ./serve_reference.sh ternary native        # Q2_0, no drafter
#   ./serve_reference.sh ternary dspark        # Q2_0 + DSpark drafter
#   ./serve_reference.sh onebit  native        # Q1_0, no drafter
#   ./serve_reference.sh onebit  dspark        # Q1_0 + DSpark drafter
#   ./serve_reference.sh onebit  native upstream   # stock llama.cpp (2nd oracle)
#
# Env overrides: MODEL_DIR, FORK_DIR, UPSTREAM_DIR, PORT, CTX, DRAFT_N_MAX
set -euo pipefail

VARIANT="${1:-ternary}"
MODE="${2:-native}"
ENGINE="${3:-fork}"

MODEL_DIR="${MODEL_DIR:-$HOME/models/bonsai}"
FORK_DIR="${FORK_DIR:-$HOME/prism-llama.cpp}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$HOME/llama.cpp}"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
DRAFT_N_MAX="${DRAFT_N_MAX:-4}"

case "$VARIANT" in
    ternary)
        MODEL="$MODEL_DIR/ternary/Ternary-Bonsai-27B-Q2_0.gguf"
        DRAFT="$MODEL_DIR/ternary/Ternary-Bonsai-27B-dspark-Q4_1.gguf"
        ;;
    onebit)
        MODEL="$MODEL_DIR/onebit/Bonsai-27B-Q1_0.gguf"
        DRAFT="$MODEL_DIR/onebit/Bonsai-27B-dspark-Q4_1.gguf"
        ;;
    *) echo "unknown variant: $VARIANT (want: ternary|onebit)" >&2; exit 2 ;;
esac

# BACKEND selects which build of the fork to serve from, so the same
# harness drives CUDA, Vulkan and CPU runs. NGL is the offload count:
# the CPU backend must run with 0 layers offloaded.
BACKEND="${BACKEND:-cuda}"
NGL=999
NGLD=999

case "$ENGINE" in
    fork)
        case "$BACKEND" in
            cuda)
                BIN="$FORK_DIR/build-cuda/bin/llama-server"
                [ -x "$BIN" ] || BIN="$FORK_DIR/build/bin/llama-server"
                ;;
            vulkan)
                BIN="$FORK_DIR/build-vulkan/bin/llama-server"
                ;;
            cpu)
                # Reuses any build; the CPU backend is always present.
                # -ngl 0 keeps every layer on the ARM NEON path.
                BIN="$FORK_DIR/build-cpu/bin/llama-server"
                [ -x "$BIN" ] || BIN="$FORK_DIR/build/bin/llama-server"
                NGL=0
                NGLD=0
                ;;
            *) echo "unknown backend: $BACKEND (want: cuda|vulkan|cpu)" >&2; exit 2 ;;
        esac
        ;;
    upstream)
        # Q1_0 is merged upstream; ternary Q2_0 is fork-only.
        if [ "$VARIANT" = "ternary" ]; then
            echo "ternary Q2_0 requires the PrismML fork; upstream cannot load it" >&2
            exit 2
        fi
        BIN="$UPSTREAM_DIR/build/bin/llama-server"
        ;;
    *) echo "unknown engine: $ENGINE (want: fork|upstream)" >&2; exit 2 ;;
esac

[ -x "$BIN" ] || { echo "server binary not found/executable: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

# Sampling defaults are the model card's reported settings (thinking mode).
# Benchmarks and trace capture override temperature per request.
ARGS=(
    -m "$MODEL"
    -ngl "$NGL"
    -fa on
    -c "$CTX"
    -np 1
    --host 0.0.0.0 --port "$PORT"
)

if [ "$MODE" = "dspark" ]; then
    [ -f "$DRAFT" ] || { echo "drafter not found: $DRAFT" >&2; exit 1; }
    ARGS+=(
        -md "$DRAFT"
        --spec-type draft-dspark
        --spec-draft-n-max "$DRAFT_N_MAX"
        -ngld "$NGLD"
    )
elif [ "$MODE" != "native" ]; then
    echo "unknown mode: $MODE (want: native|dspark)" >&2; exit 2
fi

# EXTRA_ARGS passes through anything else, e.g. EXTRA_ARGS="-lv 10" to get
# ggml's buffer-size accounting for bench/measure_memory.py.
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_ARGS)
fi

echo "==> $BIN"
echo "==> variant=$VARIANT mode=$MODE engine=$ENGINE backend=$BACKEND ngl=$NGL port=$PORT ctx=$CTX"
exec "$BIN" "${ARGS[@]}"
