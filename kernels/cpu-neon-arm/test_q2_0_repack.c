// Validate the ARM NEON q2_0 repack kernels against ggml's own scalar
// reference implementations.
//
// This is the check that matters for this kernel, and it is deliberately NOT
// a model run. A model run compares text, which stays fluent through a wide
// range of numerical wrongness -- the repo has already been burned twice by
// output that looked fine while the computation underneath was broken. Here
// the reference is `ggml_gemv_q2_0_4x8_q8_0_generic`, the same function the
// kernel replaces, fed byte-identical inputs. Any disagreement beyond float
// reassociation noise is a bug, with nowhere for it to hide.
//
// Build: see run_q2_0_repack_test.sh
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK2_0 128
#define QK8_0 32

typedef uint16_t ggml_half;

// block_q2_0x4: 4 interleaved columns, 128 values each, 2 bits per value.
typedef struct { ggml_half d[4]; int8_t qs[128]; } block_q2_0x4;
// block_q8_0:   32 activations
typedef struct { ggml_half d;    int8_t qs[32];  } block_q8_0;
// block_q8_0x4: 4 interleaved activation rows, 32 values each, 8-byte groups
typedef struct { ggml_half d[4]; int8_t qs[128]; } block_q8_0x4;

void ggml_gemv_q2_0_4x8_q8_0        (int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_0_4x8_q8_0_generic(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_0_4x8_q8_0        (int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_0_4x8_q8_0_generic(int, float *, size_t, const void *, const void *, int, int);

static uint32_t rng = 12345;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

// fp32 -> fp16 (round-to-nearest-even not required; values are small and exact enough)
static ggml_half f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man  = (x >> 13) & 0x3ff;
    if (exp <= 0)   return (ggml_half)sign;
    if (exp >= 31)  return (ggml_half)(sign | 0x7c00);
    return (ggml_half)(sign | (exp << 10) | man);
}

// Error relative to the SCALE OF THE RESULT, not to each element.
//
// Elementwise relative error is the wrong metric for a dot product over
// signed data: at large K the terms cancel, and an output can land near zero
// while the individual products are O(1). Two implementations that agree to
// 1e-9 absolute then "disagree" by 1e-2 relative on a value of 1e-5, and the
// check fails on arithmetic that is fine. Normalising by max|ref| over the
// output is the standard GEMM criterion and measures what matters: error
// against the magnitude the computation actually produces.
//
// Both are reported, because the elementwise figure is still informative --
// it just is not the pass/fail gate.
static float maxrel(const float * a, const float * b, int n, int * where, float * elemwise) {
    float worst_abs = 0.0f, scale = 0.0f, worst_elem = 0.0f;
    *where = -1;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        scale = fmaxf(scale, fabsf(b[i]));
        if (d > worst_abs) { worst_abs = d; *where = i; }
        const float s = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (s > 1e-6f) worst_elem = fmaxf(worst_elem, d / s);
    }
    if (elemwise) *elemwise = worst_elem;
    return scale > 0.0f ? worst_abs / scale : worst_abs;
}

static int run_shape(int K, int NC, int NR, int quiet);

int main(void) {
    int fail = 0;
    // Shape sweep. The kernels index weights, activations and output with
    // three different strides, and a single shape can satisfy all three by
    // coincidence -- K=NC=NR=power-of-two especially. Vary each independently,
    // including the minimum legal values (K=128 is one q2_0 block, NC=4 is one
    // interleaved group, NR=4 is one i8mm tile), where an off-by-one in a loop
    // bound has nowhere to hide.
    const int Ks[]  = { 128, 256, 512, 1024, 1280 };
    const int NCs[] = { 4, 8, 12, 32, 64 };
    const int NRs[] = { 4, 8, 12, 32 };

    printf("shape sweep (K x NC x NR):\n");
    for (size_t a = 0; a < sizeof(Ks)/sizeof(*Ks); a++)
        for (size_t b = 0; b < sizeof(NCs)/sizeof(*NCs); b++)
            for (size_t c = 0; c < sizeof(NRs)/sizeof(*NRs); c++)
                fail += run_shape(Ks[a], NCs[b], NRs[c], 1);
    printf("  %d shapes, %d failed\n\n", (int)(sizeof(Ks)/sizeof(*Ks) *
           sizeof(NCs)/sizeof(*NCs) * sizeof(NRs)/sizeof(*NRs)), fail);

    printf("detail at K=512 nc=32 nr=8:\n");
    fail += run_shape(512, 32, 8, 0);

    printf(fail ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fail);
    return fail != 0;
}

static int run_shape(int K, int NC, int NR, int quiet) {
    const int nb = K / QK2_0;

    block_q2_0x4 * B = calloc((size_t)(NC / 4) * nb, sizeof(block_q2_0x4));
    block_q8_0   * A = calloc((size_t) nb * 4,       sizeof(block_q8_0));
    block_q8_0x4 * Am = calloc((size_t) (NR / 4) * 4 * nb, sizeof(block_q8_0x4));

    for (size_t i = 0; i < (size_t)(NC / 4) * nb; i++) {
        for (int j = 0; j < 4; j++) B[i].d[j] = f2h(0.01f + 0.003f * (float)(xr() % 7));
        for (int j = 0; j < 128; j++) B[i].qs[j] = (int8_t)(xr() & 0xff);   // all 2-bit codes
    }
    for (size_t i = 0; i < (size_t) nb * 4; i++) {
        A[i].d = f2h(0.02f + 0.004f * (float)(xr() % 5));
        for (int j = 0; j < 32; j++) A[i].qs[j] = (int8_t)((int)(xr() % 255) - 127);
    }
    for (size_t i = 0; i < (size_t)(NR / 4) * 4 * nb; i++) {
        for (int j = 0; j < 4; j++) Am[i].d[j] = f2h(0.02f + 0.004f * (float)(xr() % 5));
        for (int j = 0; j < 128; j++) Am[i].qs[j] = (int8_t)((int)(xr() % 255) - 127);
    }

    int fail = 0;

    // ---- gemv
    {
        float * got = calloc(NC, sizeof(float));
        float * ref = calloc(NC, sizeof(float));
        ggml_gemv_q2_0_4x8_q8_0        (K, got, 0, B, A, 1, NC);
        ggml_gemv_q2_0_4x8_q8_0_generic(K, ref, 0, B, A, 1, NC);
        int w; float ew; const float e = maxrel(got, ref, NC, &w, &ew);
        if (!quiet) printf("  gemv q2_0 4x8   K=%d nc=%d   err/|ref|max %.3e (elemwise %.3e)", K, NC, e, ew);
        if (e < 1e-4f) { if (!quiet) printf("   OK\n"); }
        else { printf("  gemv FAIL K=%d nc=%d at %d (got %.6f ref %.6f, rel %.3e)\n",
                      K, NC, w, got[w], ref[w], e); fail++; }
        free(got); free(ref);
    }

    // ---- gemm
    {
        const size_t bs = NC;
        float * got = calloc((size_t) NR * NC, sizeof(float));
        float * ref = calloc((size_t) NR * NC, sizeof(float));
        ggml_gemm_q2_0_4x8_q8_0        (K, got, bs, B, Am, NR, NC);
        ggml_gemm_q2_0_4x8_q8_0_generic(K, ref, bs, B, Am, NR, NC);
        int w; float ew; const float e = maxrel(got, ref, (int)((size_t) NR * NC), &w, &ew);
        if (!quiet) printf("  gemm q2_0 4x8   K=%d nr=%d nc=%d   err/|ref|max %.3e (elemwise %.3e)", K, NR, NC, e, ew);
        if (e < 1e-4f) { if (!quiet) printf("   OK\n"); }
        else { printf("  gemm FAIL K=%d nr=%d nc=%d at %d (got %.6f ref %.6f, rel %.3e)\n",
                      K, NR, NC, w, got[w], ref[w], e); fail++; }
        free(got); free(ref);
    }

    // ---- degenerate weights: all codes equal. Catches a bias/offset error
    // that random data can average away.
    for (int code = 0; code < 4; code++) {
        const uint8_t byte = (uint8_t)(code | (code << 2) | (code << 4) | (code << 6));
        for (size_t i = 0; i < (size_t)(NC / 4) * nb; i++)
            memset(B[i].qs, byte, 128);
        float * got = calloc(NC, sizeof(float));
        float * ref = calloc(NC, sizeof(float));
        ggml_gemv_q2_0_4x8_q8_0        (K, got, 0, B, A, 1, NC);
        ggml_gemv_q2_0_4x8_q8_0_generic(K, ref, 0, B, A, 1, NC);
        int w; float ew; const float e = maxrel(got, ref, NC, &w, &ew);
        if (!quiet) printf("  gemv all-code-%d (w=%+d)      err/|ref|max %.3e", code, code - 1, e);
        if (e < 1e-4f) { if (!quiet) printf("   OK\n"); }
        else { printf("  gemv all-code-%d FAIL K=%d at %d (rel %.3e)\n", code, K, w, e); fail++; }
        free(got); free(ref);
    }

    free(B); free(A); free(Am);
    return fail;
}
