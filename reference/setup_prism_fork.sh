#!/usr/bin/env bash
# Build the PrismML llama.cpp fork (prism branch) -- the correctness oracle
# for every backend in this repo.
#
# The fork is required for ternary Q2_0 (custom hybrid-attention kernels).
# Q1_0 (1-bit) is merged upstream, so upstream llama.cpp is a second,
# independent reference for the 1-bit variant only.
#
# Usage:
#   ./setup_prism_fork.sh cuda      # default
#   ./setup_prism_fork.sh vulkan
#   ./setup_prism_fork.sh cpu
#   ./setup_prism_fork.sh cuda /path/to/checkout
set -euo pipefail

BACKEND="${1:-cuda}"
FORK_DIR="${2:-${BONSAI_FORK_DIR:-$HOME/prism-llama.cpp}}"
FORK_URL="https://github.com/PrismML-Eng/llama.cpp"
FORK_BRANCH="prism"
JOBS="${JOBS:-$(nproc)}"

if [ ! -d "$FORK_DIR/.git" ]; then
    echo "==> cloning $FORK_URL ($FORK_BRANCH) into $FORK_DIR"
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$FORK_DIR"
else
    echo "==> reusing existing checkout at $FORK_DIR"
fi

cd "$FORK_DIR"
echo "==> fork HEAD: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

case "$BACKEND" in
    cuda)
        CMAKE_ARGS=(-DGGML_CUDA=ON)
        # Architecture notes (see docs/METHODOLOGY.md):
        #   Jetson Thor  = sm_110  (CC 1100)  -- CMake native detection yields 110-real
        #   Orin / Orin Nano Super = sm_87
        #   RTX 3090 = sm_86, RTX 5090 = sm_120a, DGX Spark = sm_121a
        # The fork's CMakeLists rewrites plain 12X -> 12Xa for Blackwell's
        # arch-specific FP4 tensor-core instructions. It does NOT rewrite 11X,
        # and it does not need to: ggml gates its Blackwell tensor-core paths on
        # CC >= 1200, so a CC 1100 part never reaches them and 110-real is the
        # correct, complete build. Leaving CMAKE_CUDA_ARCHITECTURES unset lets
        # native detection do the right thing on Thor.
        if [ -n "${CUDA_ARCHS:-}" ]; then
            CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS")
        fi
        ;;
    vulkan) CMAKE_ARGS=(-DGGML_VULKAN=ON) ;;
    cpu)    CMAKE_ARGS=() ;;
    *) echo "unknown backend: $BACKEND (want: cuda|vulkan|cpu)" >&2; exit 2 ;;
esac

BUILD_DIR="build-$BACKEND"
echo "==> configuring $BUILD_DIR ${CMAKE_ARGS[*]:-(cpu defaults)}"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON "${CMAKE_ARGS[@]}"

echo "==> building with -j$JOBS"
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

echo
echo "==> done. binaries in $FORK_DIR/$BUILD_DIR/bin"
"$FORK_DIR/$BUILD_DIR/bin/llama-server" --version 2>&1 | head -5 || true
