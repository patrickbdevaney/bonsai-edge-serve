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
                # build-vulkan2 carries the q1_0/q2_0 integer-dot MMVQ path
                # (patches/0003, +34.1% / +23.7% decode). Prefer it, and fall
                # back to the pre-MMVQ build if it has not been built.
                BIN="$FORK_DIR/build-vulkan2/bin/llama-server"
                [ -x "$BIN" ] || BIN="$FORK_DIR/build-vulkan/bin/llama-server"

                # ggml-vulkan's graph_optimize reorder pass makes this model
                # NONDETERMINISTIC -- up to 8 distinct outputs from 10
                # identical greedy requests, because it reorders around
                # hazards that is_src_of cannot see (recurrent state across
                # 48 gated-delta-net layers). Determinism costs 2.2% prefill
                # and 1.0% decode, measured interleaved, and takes the trace
                # gate from 1/16 reproducing the CUDA oracle to 11/16.
                # See results/vulkan-nondeterminism.txt.
                # Set BONSAI_VK_GRAPH_OPTIMIZE=1 to opt back into the fast,
                # racy behaviour (e.g. to reproduce the old throughput rows).
                if [ "${BONSAI_VK_GRAPH_OPTIMIZE:-0}" != "1" ]; then
                    export GGML_VK_DISABLE_GRAPH_OPTIMIZE=1
                fi
                ;;
            cpu)
                # Reuses any build; the CPU backend is always present.
                # -ngl 0 keeps every layer on the ARM NEON path.
                BIN="$FORK_DIR/build-cpu/bin/llama-server"
                [ -x "$BIN" ] || BIN="$FORK_DIR/build/bin/llama-server"
                NGL=0
                # -ngl 0 bounds where the WEIGHTS live, not where the OPS run,
                # and it does NOT give you the CPU backend's real behaviour if
                # the process can see a GPU:
                #
                #   * make_cpu_buft_list orders bufts as ACCEL, HOST, EXTRA,
                #     CPU. On a CUDA build the CUDA pinned-host buffer outranks
                #     the EXTRA (repack) buffers, so the ARM repack kernels are
                #     unreachable -- Q1_0 has them and never used them.
                #   * with weights in a host buffer, ggml offloads large prefill
                #     matmuls to the GPU, so part of the "CPU" prefill number is
                #     a GPU number.
                #
                # Hiding the GPU is what makes this a CPU measurement. Measured:
                # 1-bit decode 4.85 -> 6.80 tok/s and prefill 11.89 -> 20.16
                # once repack can actually be selected.
                # See results/cpu-repack.txt.
                export CUDA_VISIBLE_DEVICES=""
                NGLD=0
                # Leave cores for the OS. Measured on Thor (14 cores),
                # decode collapses if every core is taken:
                #   ternary  t=12 3.18 tok/s  vs  t=14 1.84  (1.73x)
                #   1-bit    t=12 4.85 tok/s  vs  t=14 2.14  (2.27x)
                # The default (nproc) is therefore the worst setting.
                CPU_THREADS="${CPU_THREADS:-12}"
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

if [ -n "${CPU_THREADS:-}" ]; then
    ARGS+=(-t "$CPU_THREADS")
fi

# N-gram self-speculation: no drafter model, no extra weights, and
# need_n_rs_seq() returns 0 for these types so the target's recurrent-state
# buffer does NOT get the 5x block-verification blowup a drafter causes.
# That makes them the only speculation that is plausibly free on 8 GB.
case "$MODE" in
    ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache)
        ARGS+=(--spec-type "$MODE" --spec-draft-n-max "$DRAFT_N_MAX")
        echo "==> $BIN"
        echo "==> variant=$VARIANT mode=$MODE engine=$ENGINE backend=$BACKEND ngl=$NGL port=$PORT ctx=$CTX"
        if [ -n "${EXTRA_ARGS:-}" ]; then
            # shellcheck disable=SC2206
            ARGS+=($EXTRA_ARGS)
        fi
        exec "$BIN" "${ARGS[@]}"
        ;;
esac

if [ "$MODE" = "dspark" ]; then
    [ -f "$DRAFT" ] || { echo "drafter not found: $DRAFT" >&2; exit 1; }
    ARGS+=(
        -md "$DRAFT"
        --spec-type draft-dspark
        --spec-draft-n-max "$DRAFT_N_MAX"
        -ngld "$NGLD"
    )
elif [ "$MODE" != "native" ]; then
    echo "unknown mode: $MODE (want: native|dspark|ngram-*)" >&2; exit 2
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
