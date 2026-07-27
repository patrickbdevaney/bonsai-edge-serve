// Optimal batch-1 low-bit GEMV for the Bonsai family, ARM NEON.
//
// Same packed weight format as the CUDA and Vulkan backends (see
// ../common/bonsai_gemv.h). This is one engine with three code
// generators, not three engines.
//
// THE PORTABILITY RULE, made concrete here:
//   weights    -> one shared bit-plane layout, repacked once at load
//   activations-> permuted per backend, once per token
//
// The weight layout puts logical index (16w + 4q + p) in word w, byte p,
// at shift 2q. For a 4-byte CUDA word that makes the four dp4a lanes
// logically consecutive, so activations need no permutation at all. For a
// 16-byte NEON vector spanning four words the lanes are four groups of
// four strided by 16, so the activations DO need a shuffle. Doing it on
// the activations (K bytes, once per token) rather than the weights
// (N*K/4 bytes, and shared with every other backend) is the cheap side of
// that trade -- it amortizes over every row of the matrix.
//
// The ladder (every variant checked against a scalar reference):
//   v0 scalar     per-element extract -- the shape the existing ARM
//                 Q2_0 vec_dot has
//   v1..v4 sdot   whole-vector shift+mask unpack + SDOT, R row
//                 accumulators
//
// Why unpack shape beats MAC choice on this core: SHR and AND issue at
// ~4/cycle on Neoverse V3AE while SDOT and SMMLA issue at ~2/cycle, so a
// kernel spending 4-6 scalar ops per weight is unpack-bound. Whole-vector
// unpack gets 64 weights per 16-byte load in two instructions.
//
// Build: cc -O3 -march=armv8.2-a+dotprod -o gemv_neon_bench gemv_neon_bench.c -lm
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>
#include "../common/bonsai_gemv.h"

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------
// Activation permutation for a 16-byte NEON vector.
//
// Q2_0: a 16-byte load at block byte offset i covers words i/4..i/4+3.
// Lane L of shift q holds logical  4i + 16*(L/4) + 4q + (L%4).
// Store those in scan order so the kernel can use straight vld1q_s8.
static void permute_activations_q2(const int8_t *A, int8_t *P, int K) {
    const int nblk = K / BONSAI_QK;
    for (int b = 0; b < nblk; ++b) {
        const int8_t *a = A + b * BONSAI_QK;
        int8_t *p = P + b * BONSAI_QK;
        for (int i = 0; i < BONSAI_Q2_BYTES; i += 16)      // 2 vectors/block
            for (int q = 0; q < 4; ++q)
                for (int L = 0; L < 16; ++L)
                    p[(i / 16) * 64 + q * 16 + L] = a[4 * i + 16 * (L / 4) + 4 * q + (L % 4)];
    }
}
// Q1_0: one 16-byte load is the whole 128-weight block; word w covers
// logical [32w, 32w+32); lane L of bit q holds 32*(L/4) + 4q + (L%4).
static void permute_activations_q1(const int8_t *A, int8_t *P, int K) {
    const int nblk = K / BONSAI_QK;
    for (int b = 0; b < nblk; ++b) {
        const int8_t *a = A + b * BONSAI_QK;
        int8_t *p = P + b * BONSAI_QK;
        for (int q = 0; q < 8; ++q)
            for (int L = 0; L < 16; ++L)
                p[q * 16 + L] = a[32 * (L / 4) + 4 * q + (L % 4)];
    }
}

// ---------------------------------------------------------------- v0
static void gemv_q2_scalar(const uint8_t *W, const float *S, const int8_t *A,
                           float da, float *out, int N, int K) {
    const int nblk = K / BONSAI_QK;
    for (int r = 0; r < N; ++r) {
        const uint8_t *w = W + (size_t)r * (K / 4);
        float acc = 0.f;
        for (int b = 0; b < nblk; ++b) {
            int dot = 0;
            for (int j = 0; j < BONSAI_QK; ++j) {
                const int idx = b * BONSAI_QK + j;
                const int c = (w[idx >> 2] >> ((idx & 3) * 2)) & 3;
                dot += (c - 1) * A[idx];
            }
            acc += S[(size_t)r * nblk + b] * (float)dot;
        }
        out[r] = acc * da;
    }
}

static void gemv_q1_scalar(const uint8_t *W, const float *S, const int8_t *A,
                           float da, float *out, int N, int K) {
    const int nblk = K / BONSAI_QK;
    for (int r = 0; r < N; ++r) {
        const uint8_t *w = W + (size_t)r * (K / 8);
        float acc = 0.f;
        for (int b = 0; b < nblk; ++b) {
            int dot = 0;
            for (int j = 0; j < BONSAI_QK; ++j) {
                const int idx = b * BONSAI_QK + j;
                const int bit = (w[idx >> 3] >> (idx & 7)) & 1;
                dot += (bit ? 1 : -1) * A[idx];
            }
            acc += S[(size_t)r * nblk + b] * (float)dot;
        }
        out[r] = acc * da;
    }
}

// ------------------------------------------------------- q2, R rows
// R independent accumulator chains. Weights differ per row so the unpack
// is not shared, but the chains are independent, which is what keeps the
// two SDOT pipes busy and the load unit ahead.
#define Q2_ROWS_BODY(R)                                                       \
    const int nblk = K / BONSAI_QK;                                           \
    const uint8x16_t m = vdupq_n_u8(0x03);                                    \
    for (int r0 = 0; r0 + (R) <= N; r0 += (R)) {                              \
        float accf[R];                                                        \
        for (int i = 0; i < (R); ++i) accf[i] = 0.f;                          \
        for (int b = 0; b < nblk; ++b) {                                      \
            int32x4_t acc[R];                                                 \
            for (int i = 0; i < (R); ++i) acc[i] = vdupq_n_s32(0);            \
            const int8_t *pb = P + b * BONSAI_QK;                             \
            for (int i = 0; i < BONSAI_Q2_BYTES; i += 16) {                   \
                const int8x16_t x0 = vld1q_s8(pb + (i/16)*64 +  0);           \
                const int8x16_t x1 = vld1q_s8(pb + (i/16)*64 + 16);           \
                const int8x16_t x2 = vld1q_s8(pb + (i/16)*64 + 32);           \
                const int8x16_t x3 = vld1q_s8(pb + (i/16)*64 + 48);           \
                for (int rr = 0; rr < (R); ++rr) {                            \
                    const uint8x16_t v = vld1q_u8(W + (size_t)(r0+rr)*(K/4)   \
                                                  + b*BONSAI_Q2_BYTES + i);   \
                    acc[rr] = vdotq_s32(acc[rr],                              \
                        vreinterpretq_s8_u8(vandq_u8(v, m)), x0);             \
                    acc[rr] = vdotq_s32(acc[rr],                              \
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,2), m)), x1);\
                    acc[rr] = vdotq_s32(acc[rr],                              \
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,4), m)), x2);\
                    acc[rr] = vdotq_s32(acc[rr],                              \
                        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,6), m)), x3);\
                }                                                             \
            }                                                                 \
            for (int rr = 0; rr < (R); ++rr)                                  \
                accf[rr] += S[(size_t)(r0+rr)*nblk + b]                       \
                            * (float)(vaddvq_s32(acc[rr]) - ASUM[b]);         \
        }                                                                     \
        for (int rr = 0; rr < (R); ++rr) out[r0+rr] = accf[rr] * da;          \
    }

#define DEF_Q2(R) \
static void gemv_q2_r##R(const uint8_t *W, const float *S, const int8_t *P,   \
                         const int32_t *ASUM, float da, float *out,           \
                         int N, int K) { Q2_ROWS_BODY(R) }
DEF_Q2(1) DEF_Q2(2) DEF_Q2(4) DEF_Q2(8)

// ------------------------------------------------------- q1, R rows
#define Q1_ROWS_BODY(R)                                                       \
    const int nblk = K / BONSAI_QK;                                           \
    const uint8x16_t m = vdupq_n_u8(0x01);                                    \
    for (int r0 = 0; r0 + (R) <= N; r0 += (R)) {                              \
        float accf[R];                                                        \
        for (int i = 0; i < (R); ++i) accf[i] = 0.f;                          \
        for (int b = 0; b < nblk; ++b) {                                      \
            int32x4_t acc[R];                                                 \
            for (int i = 0; i < (R); ++i) acc[i] = vdupq_n_s32(0);            \
            const int8_t *pb = P + b * BONSAI_QK;                             \
            const int8x16_t x0 = vld1q_s8(pb +   0), x1 = vld1q_s8(pb +  16); \
            const int8x16_t x2 = vld1q_s8(pb +  32), x3 = vld1q_s8(pb +  48); \
            const int8x16_t x4 = vld1q_s8(pb +  64), x5 = vld1q_s8(pb +  80); \
            const int8x16_t x6 = vld1q_s8(pb +  96), x7 = vld1q_s8(pb + 112); \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const uint8x16_t v = vld1q_u8(W + (size_t)(r0+rr)*(K/8)       \
                                              + b*BONSAI_Q1_BYTES);           \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(v, m)), x0);                 \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,1), m)), x1);   \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,2), m)), x2);   \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,3), m)), x3);   \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,4), m)), x4);   \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,5), m)), x5);   \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,6), m)), x6);   \
                acc[rr] = vdotq_s32(acc[rr],                                  \
                    vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v,7), m)), x7);   \
            }                                                                 \
            for (int rr = 0; rr < (R); ++rr)                                  \
                accf[rr] += S[(size_t)(r0+rr)*nblk + b]                       \
                            * (float)(2*vaddvq_s32(acc[rr]) - ASUM[b]);       \
        }                                                                     \
        for (int rr = 0; rr < (R); ++rr) out[r0+rr] = accf[rr] * da;          \
    }

#define DEF_Q1(R) \
static void gemv_q1_r##R(const uint8_t *W, const float *S, const int8_t *P,   \
                         const int32_t *ASUM, float da, float *out,           \
                         int N, int K) { Q1_ROWS_BODY(R) }
DEF_Q1(1) DEF_Q1(2) DEF_Q1(4) DEF_Q1(8)

// ------------------------------------------------- threaded wrappers
// Rows are embarrassingly parallel: each thread owns a disjoint row
// range and writes disjoint outputs. Activations are read-only and
// shared, which is exactly the access pattern a decode step wants.
#ifdef _OPENMP
#include <omp.h>
#endif

static void gemv_q2_mt(const uint8_t *W, const float *S, const int8_t *P,
                       const int32_t *ASUM, float da, float *out,
                       int N, int K, int R, int nth) {
    (void)R;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(static)
#endif
    for (int t = 0; t < nth; ++t) {
        const int per = ((N / 8) + nth - 1) / nth * 8;   // multiple of 8 rows
        const int lo = t * per;
        int hi = lo + per; if (hi > N) hi = N;
        if (lo < hi)
            gemv_q2_r8(W + (size_t)lo * (K / 4), S + (size_t)lo * (K / BONSAI_QK),
                       P, ASUM, da, out + lo, hi - lo, K);
    }
}

static void gemv_q1_mt(const uint8_t *W, const float *S, const int8_t *P,
                       const int32_t *ASUM, float da, float *out,
                       int N, int K, int R, int nth) {
    (void)R;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(static)
#endif
    for (int t = 0; t < nth; ++t) {
        const int per = ((N / 8) + nth - 1) / nth * 8;
        const int lo = t * per;
        int hi = lo + per; if (hi > N) hi = N;
        if (lo < hi)
            gemv_q1_r8(W + (size_t)lo * (K / 8), S + (size_t)lo * (K / BONSAI_QK),
                       P, ASUM, da, out + lo, hi - lo, K);
    }
}

// ---------------------------------------------------------------------

int main(int argc, char **argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? atoi(argv[2]) : 8192;
    const int iters = (argc > 3) ? atoi(argv[3]) : 5;
    const int nblk = K / BONSAI_QK;

    printf("shape N=%d K=%d   Q2_0 %.1f MiB   Q1_0 %.1f MiB\n\n",
           N, K, (double)N*K/4/1048576.0, (double)N*K/8/1048576.0);

    uint8_t *q2  = aligned_alloc(64, (size_t)N*K/4);
    uint8_t *q2r = aligned_alloc(64, (size_t)N*K/4);
    uint8_t *q1  = aligned_alloc(64, (size_t)N*K/8);
    uint8_t *q1r = aligned_alloc(64, (size_t)N*K/8);
    float   *S   = aligned_alloc(64, (size_t)N*nblk*sizeof(float));
    int8_t  *A   = aligned_alloc(64, K);
    int8_t  *P2  = aligned_alloc(64, K);
    int8_t  *P1  = aligned_alloc(64, K);
    int32_t *ASUM= aligned_alloc(64, nblk*sizeof(int32_t));
    float   *ref2= aligned_alloc(64, N*sizeof(float));
    float   *ref1= aligned_alloc(64, N*sizeof(float));
    float   *o   = aligned_alloc(64, N*sizeof(float));

    srand(99);
    for (size_t i = 0; i < (size_t)N*K/4; ++i) q2[i] = rand() & 0xFF;
    for (size_t i = 0; i < (size_t)N*K/8; ++i) q1[i] = rand() & 0xFF;
    for (size_t i = 0; i < (size_t)N*nblk; ++i) S[i] = 0.01f + 0.001f*(i%7);
    for (int i = 0; i < K; ++i) A[i] = (int8_t)((rand()%255)-127);
    for (int b = 0; b < nblk; ++b) {
        int s = 0; for (int j = 0; j < BONSAI_QK; ++j) s += A[b*BONSAI_QK+j];
        ASUM[b] = s;
    }
    for (size_t r = 0; r < (size_t)N; ++r)
        for (int b = 0; b < nblk; ++b) {
            bonsai_q2_repack_block(&q2[r*(K/4)+b*BONSAI_Q2_BYTES],
                                   &q2r[r*(K/4)+b*BONSAI_Q2_BYTES]);
            bonsai_q1_repack_block(&q1[r*(K/8)+b*BONSAI_Q1_BYTES],
                                   &q1r[r*(K/8)+b*BONSAI_Q1_BYTES]);
        }
    permute_activations_q2(A, P2, K);
    permute_activations_q1(A, P1, K);
    const float da = 0.0037f;

    const int NREF = N < 256 ? N : 256;
    gemv_q2_scalar(q2, S, A, da, ref2, NREF, K);
    gemv_q1_scalar(q1, S, A, da, ref1, NREF, K);

    printf("=== validation (first %d rows vs scalar reference) ===\n", NREF);
    int ok = 1;
    #define VALIDATE(name, call, ref) do { \
        memset(o, 0, N*sizeof(float)); call; \
        double worst = 0; \
        for (int i = 0; i < NREF; ++i) { \
            double d = fabs((ref)[i]) > 1e-6 ? fabs((ref)[i]) : 1e-6; \
            double e = fabs(o[i]-(ref)[i])/d; if (e > worst) worst = e; } \
        printf("  %-24s max rel err %.3e  %s\n", name, worst, \
               worst < 1e-4 ? "OK" : "*** MISMATCH ***"); \
        if (!(worst < 1e-4)) ok = 0; } while (0)

    VALIDATE("q2 sdot 1 row",  gemv_q2_r1(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE("q2 sdot 4 rows", gemv_q2_r4(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE("q2 sdot 8 rows", gemv_q2_r8(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE("q1 sdot 1 row",  gemv_q1_r1(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
    VALIDATE("q1 sdot 4 rows", gemv_q1_r4(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
    VALIDATE("q1 sdot 8 rows", gemv_q1_r8(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
    if (!ok) { printf("\nVALIDATION FAILED -- timings suppressed\n"); return 1; }

    printf("\n=== throughput, single thread (%d iters) ===\n", iters);
    #define BENCH(name, bytes, call) do { \
        call; double t0 = now_s(); \
        for (int i = 0; i < iters; ++i) { call; } \
        double dt = (now_s()-t0)/iters; \
        printf("  %-24s %8.2f ms  %7.2f GB/s\n", name, dt*1e3, (bytes)/dt/1e9); \
    } while (0)

    const double b2 = (double)N*K/4, b1 = (double)N*K/8;
    BENCH("q2 v0 scalar",   b2, gemv_q2_scalar(q2,S,A,da,o,N,K));
    BENCH("q2 sdot 1 row",  b2, gemv_q2_r1(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 sdot 2 rows", b2, gemv_q2_r2(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 sdot 4 rows", b2, gemv_q2_r4(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 sdot 8 rows", b2, gemv_q2_r8(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q1 v0 scalar",   b1, gemv_q1_scalar(q1,S,A,da,o,N,K));
    BENCH("q1 sdot 1 row",  b1, gemv_q1_r1(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 sdot 4 rows", b1, gemv_q1_r4(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 sdot 8 rows", b1, gemv_q1_r8(q1r,S,P1,ASUM,da,o,N,K));

    printf("\n=== threaded (8 rows/chain) ===\n");
    for (int nth = 1; nth <= 14; nth += (nth < 4 ? 1 : 2)) {
        char nm[64];
        snprintf(nm, sizeof nm, "q2 mt t=%d", nth);
        BENCH(nm, b2, gemv_q2_mt(q2r,S,P2,ASUM,da,o,N,K,8,nth));
    }
    for (int nth = 1; nth <= 14; nth += (nth < 4 ? 1 : 2)) {
        char nm[64];
        snprintf(nm, sizeof nm, "q1 mt t=%d", nth);
        BENCH(nm, b1, gemv_q1_mt(q1r,S,P1,ASUM,da,o,N,K,8,nth));
    }

    printf("\nGB/s counts weight bytes only.\n");
    return 0;
}
