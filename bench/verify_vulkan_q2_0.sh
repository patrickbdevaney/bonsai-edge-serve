#!/usr/bin/env bash
# Assert that every edit of the Vulkan q2_0 MMVQ integration is PRESENT in
# the fork before any measurement is believed.
#
# This exists because of WIKI L7: a "fix" was applied by a script whose
# replace calls silently matched nothing (the tree had been reverted
# underneath it). It built, ran, produced correct output and normal
# throughput -- all four signals said success, and all four were measuring
# the baseline. A clean build and plausible numbers are EXACTLY what a
# reverted tree produces, so they cannot be the evidence that an edit
# landed. Grep can be.
#
# Usage: verify_vulkan_q2_0.sh [fork-dir]   (exit 0 = all edits present)
set -u

FORK="${1:-/home/patrickd/prism-llama.cpp}"
VK="$FORK/ggml/src/ggml-vulkan"
SH="$VK/vulkan-shaders"
fail=0

check() {  # check <label> <count-expected-at-least> <file> <pattern>
    local label="$1" want="$2" file="$3" pat="$4"
    if [ ! -f "$file" ]; then
        printf '  MISSING FILE  %-42s %s\n' "$label" "$file"; fail=1; return
    fi
    local n
    n=$(grep -c -- "$pat" "$file" 2>/dev/null || true)
    if [ "${n:-0}" -ge "$want" ]; then
        printf '  ok    (%s)  %s\n' "$n" "$label"
    else
        printf '  ABSENT (%s<%s)  %s\n      in %s\n      want /%s/\n' \
               "${n:-0}" "$want" "$label" "${file#$FORK/}" "$pat"
        fail=1
    fi
}

echo "Vulkan q2_0 MMVQ integration -- edit presence gate"
echo "fork: $FORK"
echo

# Gate 1: toolchain. Not a source edit; the build must have compiled the
# integer-dot support in at all, or every shader silently vanishes.
check "toolchain: INTEGER_DOT_GLSLC_SUPPORT referenced" 1 \
      "$SH/vulkan-shaders-gen.cpp" "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"

# Gate 2: the packed16 struct. A Q2_0 block is 34 bytes and only 2-byte
# aligned, so a uint32_t member pads the std430 stride to 36 and misreads
# every block after the first.
check "types: block_q2_0_packed16 declared" 1 \
      "$SH/types.glsl" "struct block_q2_0_packed16"
check "types: A_TYPE_PACKED16 defined for q2_0" 1 \
      "$SH/types.glsl" "define A_TYPE_PACKED16 block_q2_0_packed16"

# Gate 3: the shader functions.
check "funcs: DATA_A_Q2_0 block" 1 \
      "$SH/mul_mat_vecq_funcs.glsl" "defined(DATA_A_Q2_0)"
check "funcs: q2_0_codes GGUF-order extraction" 1 \
      "$SH/mul_mat_vecq_funcs.glsl" "q2_0_codes"
check "funcs: reads via data_a_packed16 (NOT packed32)" 1 \
      "$SH/mul_mat_vecq_funcs.glsl" "data_a_packed16\[ib_a / 4\]"

# The old, wrong form must be gone, not merely outnumbered.
if grep -q "data_a_packed32\[ib_a / 4\]" "$SH/mul_mat_vecq_funcs.glsl" 2>/dev/null; then
    echo "  REGRESSION  packed32 q2_0 access still present (36-byte stride bug)"
    fail=1
else
    echo "  ok    (0)  no packed32 q2_0 access remains"
fi

# Gate 4: K_PER_ITER size class. Without this the shader hits
# "#error unimplemented" and never compiles.
check "comp: q2_0 joins the K_PER_ITER 16 class" 1 \
      "$SH/mul_mat_vecq.comp" "defined(DATA_A_QUANT_K) || defined(DATA_A_Q2_0)"

# Gate 5: shader generation lists.
check "gen: q2_0 in the MMVQ generate list" 1 \
      "$SH/vulkan-shaders-gen.cpp" 'iq1_m" || tname == "q2_0"'
check "gen: q2_0 in the q8_1 header-emit list" 1 \
      "$SH/vulkan-shaders-gen.cpp" 'iq1_m" \&\& tname != "q2_0"'

# Gate 6: runtime pipeline registration. Generating the shader is not
# enough -- without this the runtime falls back to dequant-MMV and the
# tok/s is simply unchanged, which reads as "no win" rather than "not run".
check "runtime: ggml_vk_create_pipeline for Q2_0" 1 \
      "$VK/ggml-vulkan.cpp" "pipeline_dequant_mul_mat_vec_q8_1_f32\[w\]\[GGML_TYPE_Q2_0\]"

# Gate 7: the b_type == Q8_1 allow-list in ggml_vk_get_dequantize_mul_mat_vec.
# This one is the nastiest of the set because its failure is DESIGNED to be
# silent: a type missing from the switch returns nullptr, and the caller
# treats nullptr as "use the f32 path instead". Everything builds, the
# pipeline is created, the shader is in the .so, generation is correct --
# and the kernel never runs. It was found only by A/B against
# GGML_VK_DISABLE_MMVQ showing 11.58 vs 11.61 tok/s, i.e. no difference.
if awk '/^static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec\(/,/^    switch \(a_type\)/' \
       "$VK/ggml-vulkan.cpp" | grep -q "case GGML_TYPE_Q2_0:"; then
    echo "  ok    (1)  runtime: Q2_0 in the b_type==Q8_1 allow-list"
else
    echo "  ABSENT      runtime: Q2_0 missing from the b_type==Q8_1 allow-list"
    echo "      ggml_vk_get_dequantize_mul_mat_vec returns nullptr -> SILENT"
    echo "      fallback to the f32 dequant path. tok/s will look unchanged."
    fail=1
fi

echo
if [ "$fail" -ne 0 ]; then
    echo "RESULT: FAIL -- edits are not all present. Any measurement taken"
    echo "now describes the BASELINE, not the integration. Do not report it."
    exit 1
fi
echo "RESULT: PASS -- all edits present in the tree."
echo "This gate proves the code is THERE, not that it is CORRECT."
echo "Correctness is gated separately, on real generation output."
exit 0
