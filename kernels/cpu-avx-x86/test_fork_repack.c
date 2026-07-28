// Validate the x86 AVX2/AVX-VNNI q1_0/q2_0 repack kernels against
// ggml's own scalar reference implementations -- the x86 mirror of
// ../cpu-neon-arm/test_q2_0_repack.c, and the same philosophy: this is
// deliberately NOT a model run. A model run compares text, which stays
// fluent through a wide range of numerical wrongness. Here the
// reference is ggml_gem{v,m}_q{1,2}_0_4x8_q8_0_generic, the same
// functions the kernels replace, fed byte-identical inputs.
//
// Build & run: ./run_fork_repack_test.sh (links the fork's real
// libggml-cpu.so, not a copy of the kernels that could drift).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK2_0 128
#define QK1_0 128
#define QK8_0 32

typedef uint16_t ggml_half;

typedef struct { ggml_half d[4]; int8_t qs[128]; } block_q2_0x4;
typedef struct { ggml_half d[4]; int8_t qs[64];  } block_q1_0x4;
typedef struct { ggml_half d;    int8_t qs[32];  } block_q8_0;
typedef struct { ggml_half d[4]; int8_t qs[128]; } block_q8_0x4;

void ggml_gemv_q2_0_4x8_q8_0        (int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q2_0_4x8_q8_0_generic(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_0_4x8_q8_0        (int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q2_0_4x8_q8_0_generic(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q1_0_4x8_q8_0        (int, float *, size_t, const void *, const void *, int, int);
void ggml_gemv_q1_0_4x8_q8_0_generic(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q1_0_4x8_q8_0        (int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_q1_0_4x8_q8_0_generic(int, float *, size_t, const void *, const void *, int, int);

static uint32_t rng = 12345;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static ggml_half f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man  = (x >> 13) & 0x3ff;
    if (exp <= 0)   return (ggml_half)sign;
    if (exp >= 31)  return (ggml_half)(sign | 0x7c00);
    return (ggml_half)(sign | (exp << 10) | man);
}

// error normalised by max|ref| over the output -- the L16 metric.
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

typedef void gemfn(int, float *, size_t, const void *, const void *, int, int);

static int check(const char * name, gemfn * got_fn, gemfn * ref_fn,
                 int K, const void * B, const void * A, int NR, int NC,
                 size_t bs, int quiet) {
    float * got = calloc((size_t) NR * NC, sizeof(float));
    float * ref = calloc((size_t) NR * NC, sizeof(float));
    got_fn(K, got, bs, B, A, NR, NC);
    ref_fn(K, ref, bs, B, A, NR, NC);
    int w; float ew;
    const float e = maxrel(got, ref, (int)((size_t) NR * NC), &w, &ew);
    int fail = !(e < 1e-4f);
    if (!quiet || fail)
        printf("  %-14s K=%-5d nr=%-3d nc=%-3d  err/|ref|max %.3e (elemwise %.3e)  %s\n",
               name, K, NR, NC, e, ew, fail ? "FAIL" : "OK");
    free(got); free(ref);
    return fail;
}

static int run_shape(int K, int NC, int NR, int quiet) {
    const int nb = K / QK2_0;

    block_q2_0x4 * B2 = calloc((size_t)(NC / 4) * nb, sizeof(block_q2_0x4));
    block_q1_0x4 * B1 = calloc((size_t)(NC / 4) * nb, sizeof(block_q1_0x4));
    block_q8_0   * A  = calloc((size_t) nb * 4,       sizeof(block_q8_0));
    block_q8_0x4 * Am = calloc((size_t)(NR / 4) * 4 * nb, sizeof(block_q8_0x4));

    for (size_t i = 0; i < (size_t)(NC / 4) * nb; i++) {
        for (int j = 0; j < 4; j++)  B2[i].d[j] = f2h(0.01f + 0.003f * (float)(xr() % 7));
        for (int j = 0; j < 128; j++) B2[i].qs[j] = (int8_t)(xr() & 0xff);
        for (int j = 0; j < 4; j++)  B1[i].d[j] = f2h(0.01f + 0.003f * (float)(xr() % 7));
        for (int j = 0; j < 64; j++)  B1[i].qs[j] = (int8_t)(xr() & 0xff);
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
    fail += check("gemv q2_0 4x8", ggml_gemv_q2_0_4x8_q8_0, ggml_gemv_q2_0_4x8_q8_0_generic,
                  K, B2, A, 1, NC, 0, quiet);
    fail += check("gemm q2_0 4x8", ggml_gemm_q2_0_4x8_q8_0, ggml_gemm_q2_0_4x8_q8_0_generic,
                  K, B2, Am, NR, NC, NC, quiet);
    fail += check("gemv q1_0 4x8", ggml_gemv_q1_0_4x8_q8_0, ggml_gemv_q1_0_4x8_q8_0_generic,
                  K, B1, A, 1, NC, 0, quiet);
    fail += check("gemm q1_0 4x8", ggml_gemm_q1_0_4x8_q8_0, ggml_gemm_q1_0_4x8_q8_0_generic,
                  K, B1, Am, NR, NC, NC, quiet);

    // degenerate weights: all codes equal, catches bias/offset errors that
    // random data averages away. For q1_0: all-zero and all-one bits.
    for (int code = 0; code < 4; code++) {
        const uint8_t byte2 = (uint8_t)(code | (code << 2) | (code << 4) | (code << 6));
        const uint8_t byte1 = (code & 1) ? 0xff : 0x00;
        for (size_t i = 0; i < (size_t)(NC / 4) * nb; i++) {
            memset(B2[i].qs, byte2, 128);
            memset(B1[i].qs, byte1, 64);
        }
        fail += check("gemv q2_0 deg", ggml_gemv_q2_0_4x8_q8_0, ggml_gemv_q2_0_4x8_q8_0_generic,
                      K, B2, A, 1, NC, 0, 1);
        fail += check("gemm q2_0 deg", ggml_gemm_q2_0_4x8_q8_0, ggml_gemm_q2_0_4x8_q8_0_generic,
                      K, B2, Am, NR, NC, NC, 1);
        fail += check("gemv q1_0 deg", ggml_gemv_q1_0_4x8_q8_0, ggml_gemv_q1_0_4x8_q8_0_generic,
                      K, B1, A, 1, NC, 0, 1);
        fail += check("gemm q1_0 deg", ggml_gemm_q1_0_4x8_q8_0, ggml_gemm_q1_0_4x8_q8_0_generic,
                      K, B1, Am, NR, NC, NC, 1);
    }

    free(B2); free(B1); free(A); free(Am);
    return fail;
}

int main(void) {
    int fail = 0;
    const int Ks[]  = { 128, 256, 512, 1024, 1280 };
    const int NCs[] = { 4, 8, 12, 32, 64 };
    const int NRs[] = { 4, 8, 12, 32 };

    printf("shape sweep (K x NC x NR), 4 kernels + 16 degenerate cases each:\n");
    for (size_t a = 0; a < sizeof(Ks)/sizeof(*Ks); a++)
        for (size_t b = 0; b < sizeof(NCs)/sizeof(*NCs); b++)
            for (size_t c = 0; c < sizeof(NRs)/sizeof(*NRs); c++)
                fail += run_shape(Ks[a], NCs[b], NRs[c], 1);
    printf("  %d shapes, %d kernel checks failed\n\n",
           (int)(sizeof(Ks)/sizeof(*Ks) * sizeof(NCs)/sizeof(*NCs) * sizeof(NRs)/sizeof(*NRs)), fail);

    printf("detail at K=512 nc=32 nr=8:\n");
    fail += run_shape(512, 32, 8, 0);

    printf(fail ? "\nRESULT: %d FAILED\n" : "\nRESULT: all passed\n", fail);
    return fail != 0;
}
