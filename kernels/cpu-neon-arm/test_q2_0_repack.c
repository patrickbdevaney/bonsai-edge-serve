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

static float maxrel(const float * a, const float * b, int n, int * where) {
    float worst = 0.0f; *where = -1;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        const float s = fmaxf(fabsf(a[i]), fabsf(b[i]));
        const float r = s > 1e-6f ? d / s : d;
        if (r > worst) { worst = r; *where = i; }
    }
    return worst;
}

int main(void) {
    // K must be a multiple of QK2_0; NC a multiple of 4; NR a multiple of 4.
    const int K  = 512;
    const int NC = 32;
    const int NR = 8;
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
        int w; const float e = maxrel(got, ref, NC, &w);
        printf("  gemv q2_0 4x8   K=%d nc=%d   max rel err %.3e", K, NC, e);
        if (e < 1e-4f) printf("   OK\n");
        else { printf("   FAIL at %d (got %.6f ref %.6f)\n", w, got[w], ref[w]); fail++; }
        free(got); free(ref);
    }

    // ---- gemm
    {
        const size_t bs = NC;
        float * got = calloc((size_t) NR * NC, sizeof(float));
        float * ref = calloc((size_t) NR * NC, sizeof(float));
        ggml_gemm_q2_0_4x8_q8_0        (K, got, bs, B, Am, NR, NC);
        ggml_gemm_q2_0_4x8_q8_0_generic(K, ref, bs, B, Am, NR, NC);
        int w; const float e = maxrel(got, ref, (int)((size_t) NR * NC), &w);
        printf("  gemm q2_0 4x8   K=%d nr=%d nc=%d   max rel err %.3e", K, NR, NC, e);
        if (e < 1e-4f) printf("   OK\n");
        else { printf("   FAIL at %d (got %.6f ref %.6f)\n", w, got[w], ref[w]); fail++; }
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
        int w; const float e = maxrel(got, ref, NC, &w);
        printf("  gemv all-code-%d (w=%+d)      max rel err %.3e", code, code - 1, e);
        if (e < 1e-4f) printf("   OK\n");
        else { printf("   FAIL at %d (got %.6f ref %.6f)\n", w, got[w], ref[w]); fail++; }
        free(got); free(ref);
    }

    printf(fail ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fail);
    return fail != 0;
}
