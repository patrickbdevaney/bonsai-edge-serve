#!/usr/bin/env bash
# Build bonsai-server against a chosen PrismML-fork build.
#
# The fork build you link against IS the backend selection: libggml loads
# whichever ggml backend shared objects sit next to it, so linking
# build-vulkan2 gives you Vulkan and build (CUDA) gives you CUDA. There is
# no runtime switch, which is why --backend only selects the POLICY table.
#
#   BUILD_DIR=build-vulkan2 ./server/build.sh     # Vulkan (with q1_0/q2_0 MMVQ)
#   BUILD_DIR=build          ./server/build.sh     # CUDA
#
# Output: build/bonsai-server
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FORK_DIR="${FORK_DIR:-$HOME/prism-llama.cpp}"
BUILD_DIR="${BUILD_DIR:-build-vulkan2}"
OUT="${OUT:-$REPO/build/bonsai-server}"

LIB="$FORK_DIR/$BUILD_DIR/bin"
[ -f "$LIB/libllama.so" ] || {
    echo "error: no libllama.so in $LIB" >&2
    echo "       build the fork first, or set BUILD_DIR / FORK_DIR" >&2
    exit 2
}

mkdir -p "$(dirname "$OUT")"

echo "fork:    $FORK_DIR/$BUILD_DIR"
echo "output:  $OUT"

# httplib.h needs -DCPPHTTPLIB_OPENSSL_SUPPORT only for https; plain HTTP
# on localhost needs nothing beyond pthread.
g++ -O2 -std=c++17 -DNDEBUG \
    -I "$REPO/server" \
    -I "$FORK_DIR/include" \
    -I "$FORK_DIR/ggml/include" \
    -I "$FORK_DIR/vendor" \
    -I "$FORK_DIR/vendor/cpp-httplib" \
    -I "$FORK_DIR/vendor/nlohmann" \
    "$REPO/server/bonsai-server.cpp" \
    "$FORK_DIR/vendor/cpp-httplib/httplib.cpp" \
    -o "$OUT" \
    -L "$LIB" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LIB" \
    -lpthread

echo "built:   $OUT"
"$OUT" --help >/dev/null && echo "smoke:   --help ok"

# Terminal client. Talks plain OpenAI, so it also works against the fork's
# own llama-server or any other compatible endpoint.
CHAT="$(dirname "$OUT")/chat"
g++ -O2 -std=c++17 -DNDEBUG \
    -I "$FORK_DIR/vendor" \
    -I "$FORK_DIR/vendor/cpp-httplib" \
    -I "$FORK_DIR/vendor/nlohmann" \
    "$REPO/server/chat.cpp" \
    "$FORK_DIR/vendor/cpp-httplib/httplib.cpp" \
    -o "$CHAT" -lpthread
echo "built:   $CHAT"
