// Optimal batch-1 low-bit GEMV for the Bonsai family, x86 AVX2 / AVX-VNNI.
//
// Same packed weight format as the CUDA, Vulkan and NEON backends (see
// ../common/bonsai_gemv.h). One engine, four code generators.
//
// THE PORTABILITY RULE, made concrete here:
//   weights    -> one shared bit-plane layout, repacked once at load
//   activations-> permuted per backend, once per token
//
// The weight layout puts logical index (16w + 4q + p) in word w, byte p,
// at shift 2q. A 32-byte AVX2 vector spans eight words, so shift q's
// lanes are eight groups of four strided by 16 -- like NEON (but twice
// as wide), the activations need a per-backend shuffle, done once per
// token on K bytes rather than on the N*K/4 weight bytes.
//
// The dot instruction is the x86 twist, and it is a better fit than
// SDOT: VPDPBUSD is u8 x s8 -> s32, which is EXACTLY the biased
// encoding -- raw unsigned codes dotted with signed int8 activations,
// no sign handling at all. On AVX2-only parts the same step is
// VPMADDUBSW (u8 x s8 -> s16 pairs) + VPMADDWD against ones -- 3 uops
// instead of 1, and both are exact: codes <= 3 and |a| <= 127 keep the
// s16 intermediate at most 762, far from saturation.
//
// The ladder (every variant checked against the scalar reference):
//   v0 scalar         per-element extract
//   avx2  R rows      shift+mask unpack + MADDUBS/MADDWD, R chains
//   vnni  R rows      shift+mask unpack + VPDPBUSD,       R chains
//   fp32 sgemv (BLAS) the same matrix dequantized to fp32, cblas_sgemv
//
// The BLAS row is the honest control, not a contender: at batch 1 the
// GEMV streams every weight byte once, so fp32 moves 16x the bytes of
// Q2_0 (32x for Q1_0) through the same DRAM. BLAS's AVX kernels are
// fine; the format is what loses. That is the measurement that justifies
// this directory's existence.
//
// Build (see Makefile):
//   cc -O3 -mavx2 -mavxvnni -mfma -ffp-contract=off -fopenmp
//      -o gemv_avx_bench gemv_avx_bench.c -lm
// -ffp-contract=off keeps the per-block float accumulate in the same
// rounding order as the scalar reference, which is what makes the
// kernels validate bit-exact rather than to ~7e-06. Add -DUSE_BLAS plus
// a CBLAS to link for the fp32 control row.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include "../common/bonsai_gemv.h"

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------
// Activation permutation for a 32-byte AVX2 vector.
//
// Q2_0: one 32-byte load is the whole 128-weight block (8 words). Lane
// L of shift q holds logical 16*(L/4) + 4q + (L%4). Four planes of 32.
static void permute_activations_q2(const int8_t *A, int8_t *P, int K) {
    const int nblk = K / BONSAI_QK;
    for (int b = 0; b < nblk; ++b) {
        const int8_t *a = A + b * BONSAI_QK;
        int8_t *p = P + b * BONSAI_QK;
        for (int q = 0; q < 4; ++q)
            for (int L = 0; L < 32; ++L)
                p[q * 32 + L] = a[16 * (L / 4) + 4 * q + (L % 4)];
    }
}
// Q1_0: a 16-byte block is only half a vector, so blocks are processed
// in PAIRS: one 32-byte weight load covers blocks b and b+1 (their code
// streams are adjacent on disk already). Shift q's lower 16 lanes come
// from block b, upper 16 from block b+1, so the permuted plane is
// [16 bytes of b's plane q][16 bytes of b+1's plane q], 8 planes per
// pair. Within a block, lane L of bit q holds logical 32*(L/4)+4q+(L%4).
// Requires an even block count, i.e. K % 256 == 0.
static void permute_activations_q1(const int8_t *A, int8_t *P, int K) {
    const int nblk = K / BONSAI_QK;
    for (int b = 0; b < nblk; b += 2) {
        int8_t *p = P + b * BONSAI_QK;             // 256 bytes per pair
        for (int q = 0; q < 8; ++q)
            for (int h = 0; h < 2; ++h) {
                const int8_t *a = A + (b + h) * BONSAI_QK;
                for (int L = 0; L < 16; ++L)
                    p[q * 32 + h * 16 + L] = a[32 * (L / 4) + 4 * q + (L % 4)];
            }
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

// ---------------------------------------------------------------------
// Dot step, two spellings of the same u8 x s8 -> s32 contraction.
//
// GCC names the AVX-VNNI (VEX, non-EVEX) intrinsic _mm256_dpbusd_avx_
// epi32; the plain name needs AVX512VNNI+VL. Handle both.
#if defined(__AVXVNNI__)
#  define BONSAI_DPBUSD(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#  define BONSAI_HAVE_VNNI 1
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#  define BONSAI_DPBUSD(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#  define BONSAI_HAVE_VNNI 1
#endif

#define DOT_AVX2(acc, u, s) \
    _mm256_add_epi32((acc), _mm256_madd_epi16(_mm256_maddubs_epi16((u), (s)), ones))
#define DOT_VNNI(acc, u, s) BONSAI_DPBUSD((acc), (u), (s))

static inline int hsum128_epi32(__m128i v) {
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
}
static inline int hsum256_epi32(__m256i v) {
    return hsum128_epi32(_mm_add_epi32(_mm256_castsi256_si128(v),
                                       _mm256_extracti128_si256(v, 1)));
}
static inline float hsum256_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                          _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// ------------------------------------------------------- q2, R rows
// One 32-byte weight load is a whole block. Per-byte shifts have no
// epi8 form on x86; VPSRLW + 0x03 mask is exact because the bits that
// bleed across byte boundaries land above the mask.
#define Q2_ROWS_BODY(R, DOT)                                                  \
    const int nblk = K / BONSAI_QK;                                           \
    const __m256i m = _mm256_set1_epi8(0x03);                                 \
    const __m256i ones = _mm256_set1_epi16(1); (void)ones;                    \
    for (int r0 = 0; r0 + (R) <= N; r0 += (R)) {                              \
        float accf[R];                                                        \
        for (int i = 0; i < (R); ++i) accf[i] = 0.f;                          \
        for (int b = 0; b < nblk; ++b) {                                      \
            const int8_t *pb = P + b * BONSAI_QK;                             \
            const __m256i x0 = _mm256_loadu_si256((const __m256i*)(pb +  0)); \
            const __m256i x1 = _mm256_loadu_si256((const __m256i*)(pb + 32)); \
            const __m256i x2 = _mm256_loadu_si256((const __m256i*)(pb + 64)); \
            const __m256i x3 = _mm256_loadu_si256((const __m256i*)(pb + 96)); \
            __m256i acc[R];                                                   \
            for (int i = 0; i < (R); ++i) acc[i] = _mm256_setzero_si256();    \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const __m256i v = _mm256_loadu_si256((const __m256i*)         \
                    (W + (size_t)(r0+rr)*(K/4) + b*BONSAI_Q2_BYTES));         \
                acc[rr] = DOT(acc[rr], _mm256_and_si256(v, m), x0);           \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 2), m), x1);        \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 4), m), x2);        \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 6), m), x3);        \
            }                                                                 \
            for (int rr = 0; rr < (R); ++rr)                                  \
                accf[rr] += S[(size_t)(r0+rr)*nblk + b]                       \
                            * (float)(hsum256_epi32(acc[rr]) - ASUM[b]);      \
        }                                                                     \
        for (int rr = 0; rr < (R); ++rr) out[r0+rr] = accf[rr] * da;          \
    }

#define DEF_Q2(R, SUFFIX, DOT)                                                \
static void gemv_q2_##SUFFIX##_r##R(const uint8_t *W, const float *S,         \
                         const int8_t *P, const int32_t *ASUM, float da,      \
                         float *out, int N, int K) { Q2_ROWS_BODY(R, DOT) }
DEF_Q2(1, avx2, DOT_AVX2) DEF_Q2(2, avx2, DOT_AVX2)
DEF_Q2(4, avx2, DOT_AVX2) DEF_Q2(8, avx2, DOT_AVX2)
#ifdef BONSAI_HAVE_VNNI
DEF_Q2(1, vnni, DOT_VNNI) DEF_Q2(2, vnni, DOT_VNNI)
DEF_Q2(4, vnni, DOT_VNNI) DEF_Q2(8, vnni, DOT_VNNI)
#endif

// ------------------------------------------------------- q1, R rows
// Blocks in pairs: one 32-byte load covers two adjacent 16-byte blocks.
// The accumulator's low 128 bits belong to block b, high to b+1, so the
// per-block reduction falls out of the halves.
#define Q1_ROWS_BODY(R, DOT)                                                  \
    const int nblk = K / BONSAI_QK;                                           \
    const __m256i m = _mm256_set1_epi8(0x01);                                 \
    const __m256i ones = _mm256_set1_epi16(1); (void)ones;                    \
    for (int r0 = 0; r0 + (R) <= N; r0 += (R)) {                              \
        float accf[R];                                                        \
        for (int i = 0; i < (R); ++i) accf[i] = 0.f;                          \
        for (int b = 0; b < nblk; b += 2) {                                   \
            const int8_t *pb = P + b * BONSAI_QK;                             \
            __m256i x[8];                                                     \
            for (int q = 0; q < 8; ++q)                                       \
                x[q] = _mm256_loadu_si256((const __m256i*)(pb + q*32));       \
            __m256i acc[R];                                                   \
            for (int i = 0; i < (R); ++i) acc[i] = _mm256_setzero_si256();    \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const __m256i v = _mm256_loadu_si256((const __m256i*)         \
                    (W + (size_t)(r0+rr)*(K/8) + b*BONSAI_Q1_BYTES));         \
                acc[rr] = DOT(acc[rr], _mm256_and_si256(v, m), x[0]);         \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 1), m), x[1]);      \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 2), m), x[2]);      \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 3), m), x[3]);      \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 4), m), x[4]);      \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 5), m), x[5]);      \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 6), m), x[6]);      \
                acc[rr] = DOT(acc[rr],                                        \
                    _mm256_and_si256(_mm256_srli_epi16(v, 7), m), x[7]);      \
            }                                                                 \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const int lo = hsum128_epi32(_mm256_castsi256_si128(acc[rr]));\
                const int hi = hsum128_epi32(                                 \
                                   _mm256_extracti128_si256(acc[rr], 1));     \
                accf[rr] += S[(size_t)(r0+rr)*nblk + b]                       \
                            * (float)(2*lo - ASUM[b]);                        \
                accf[rr] += S[(size_t)(r0+rr)*nblk + b+1]                     \
                            * (float)(2*hi - ASUM[b+1]);                      \
            }                                                                 \
        }                                                                     \
        for (int rr = 0; rr < (R); ++rr) out[r0+rr] = accf[rr] * da;          \
    }

#define DEF_Q1(R, SUFFIX, DOT)                                                \
static void gemv_q1_##SUFFIX##_r##R(const uint8_t *W, const float *S,         \
                         const int8_t *P, const int32_t *ASUM, float da,      \
                         float *out, int N, int K) { Q1_ROWS_BODY(R, DOT) }
DEF_Q1(1, avx2, DOT_AVX2) DEF_Q1(2, avx2, DOT_AVX2)
DEF_Q1(4, avx2, DOT_AVX2) DEF_Q1(8, avx2, DOT_AVX2)
#ifdef BONSAI_HAVE_VNNI
DEF_Q1(1, vnni, DOT_VNNI) DEF_Q1(2, vnni, DOT_VNNI)
DEF_Q1(4, vnni, DOT_VNNI) DEF_Q1(8, vnni, DOT_VNNI)
#endif

// ---------------------------------------------- v2: vectorized scale
// The v1 kernels pay hsum256 (5 uops, ~10 cycles of dependent latency)
// plus a scalar convert+fma PER BLOCK PER ROW to apply the per-block
// scale. But the scale is uniform across the accumulator's 8 lanes, so
// the reduction can be deferred: keep a float accumulator per row,
// fma(broadcast(scale), cvtepi32_ps(acc)) per block, and hsum ONCE per
// row. The asum correction stays scalar (one fma per block per row).
// Costs bit-exactness -- lanes now sum in a different order than the
// scalar reference -- for ~5 fewer uops per 32 weight bytes on a loop
// the v1 ladder showed to be issue-bound. PF > 0 adds a software
// prefetch that far ahead on the weight stream.
#ifdef BONSAI_HAVE_VNNI
#define Q2_VS_BODY(R, PF)                                                     \
    const int nblk = K / BONSAI_QK;                                           \
    const __m256i m = _mm256_set1_epi8(0x03);                                 \
    for (int r0 = 0; r0 + (R) <= N; r0 += (R)) {                              \
        __m256 accv[R]; float acs[R];                                         \
        for (int i = 0; i < (R); ++i) { accv[i] = _mm256_setzero_ps();        \
                                        acs[i] = 0.f; }                       \
        for (int b = 0; b < nblk; ++b) {                                      \
            const int8_t *pb = P + b * BONSAI_QK;                             \
            const __m256i x0 = _mm256_loadu_si256((const __m256i*)(pb +  0)); \
            const __m256i x1 = _mm256_loadu_si256((const __m256i*)(pb + 32)); \
            const __m256i x2 = _mm256_loadu_si256((const __m256i*)(pb + 64)); \
            const __m256i x3 = _mm256_loadu_si256((const __m256i*)(pb + 96)); \
            __m256i acc[R];                                                   \
            for (int i = 0; i < (R); ++i) acc[i] = _mm256_setzero_si256();    \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const uint8_t *wp = W + (size_t)(r0+rr)*(K/4)                 \
                                    + b*BONSAI_Q2_BYTES;                      \
                if ((PF) > 0) _mm_prefetch((const char*)wp + (PF),            \
                                           _MM_HINT_T0);                      \
                const __m256i v = _mm256_loadu_si256((const __m256i*)wp);     \
                acc[rr] = DOT_VNNI(acc[rr], _mm256_and_si256(v, m), x0);      \
                acc[rr] = DOT_VNNI(acc[rr],                                   \
                    _mm256_and_si256(_mm256_srli_epi16(v, 2), m), x1);        \
                acc[rr] = DOT_VNNI(acc[rr],                                   \
                    _mm256_and_si256(_mm256_srli_epi16(v, 4), m), x2);        \
                acc[rr] = DOT_VNNI(acc[rr],                                   \
                    _mm256_and_si256(_mm256_srli_epi16(v, 6), m), x3);        \
            }                                                                 \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const float s = S[(size_t)(r0+rr)*nblk + b];                  \
                accv[rr] = _mm256_fmadd_ps(_mm256_set1_ps(s),                 \
                    _mm256_cvtepi32_ps(acc[rr]), accv[rr]);                   \
                acs[rr] += s * (float)ASUM[b];                                \
            }                                                                 \
        }                                                                     \
        for (int rr = 0; rr < (R); ++rr)                                      \
            out[r0+rr] = (hsum256_ps(accv[rr]) - acs[rr]) * da;               \
    }

#define Q1_VS_BODY(R, PF)                                                     \
    const int nblk = K / BONSAI_QK;                                           \
    const __m256i m = _mm256_set1_epi8(0x01);                                 \
    for (int r0 = 0; r0 + (R) <= N; r0 += (R)) {                              \
        __m256 accv[R]; float acs[R];                                         \
        for (int i = 0; i < (R); ++i) { accv[i] = _mm256_setzero_ps();        \
                                        acs[i] = 0.f; }                       \
        for (int b = 0; b < nblk; b += 2) {                                   \
            const int8_t *pb = P + b * BONSAI_QK;                             \
            __m256i x[8];                                                     \
            for (int q = 0; q < 8; ++q)                                       \
                x[q] = _mm256_loadu_si256((const __m256i*)(pb + q*32));       \
            __m256i acc[R];                                                   \
            for (int i = 0; i < (R); ++i) acc[i] = _mm256_setzero_si256();    \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const uint8_t *wp = W + (size_t)(r0+rr)*(K/8)                 \
                                    + b*BONSAI_Q1_BYTES;                      \
                if ((PF) > 0) _mm_prefetch((const char*)wp + (PF),            \
                                           _MM_HINT_T0);                      \
                const __m256i v = _mm256_loadu_si256((const __m256i*)wp);     \
                for (int q = 0; q < 8; ++q)                                   \
                    acc[rr] = DOT_VNNI(acc[rr], _mm256_and_si256(             \
                        _mm256_srli_epi16(v, q), m), x[q]);                   \
            }                                                                 \
            for (int rr = 0; rr < (R); ++rr) {                                \
                const float s0 = S[(size_t)(r0+rr)*nblk + b];                 \
                const float s1 = S[(size_t)(r0+rr)*nblk + b+1];               \
                const __m256 sv = _mm256_insertf128_ps(                       \
                    _mm256_castps128_ps256(_mm_set1_ps(s0)),                  \
                    _mm_set1_ps(s1), 1);                                      \
                accv[rr] = _mm256_fmadd_ps(sv,                                \
                    _mm256_cvtepi32_ps(acc[rr]), accv[rr]);                   \
                acs[rr] += s0 * (float)ASUM[b] + s1 * (float)ASUM[b+1];       \
            }                                                                 \
        }                                                                     \
        for (int rr = 0; rr < (R); ++rr)                                      \
            out[r0+rr] = (2.f * hsum256_ps(accv[rr]) - acs[rr]) * da;         \
    }

#define DEF_Q2VS(R, SUF, PF)                                                  \
static void gemv_q2_vs##SUF##_r##R(const uint8_t *W, const float *S,          \
                         const int8_t *P, const int32_t *ASUM, float da,      \
                         float *out, int N, int K) { Q2_VS_BODY(R, PF) }
#define DEF_Q1VS(R, SUF, PF)                                                  \
static void gemv_q1_vs##SUF##_r##R(const uint8_t *W, const float *S,          \
                         const int8_t *P, const int32_t *ASUM, float da,      \
                         float *out, int N, int K) { Q1_VS_BODY(R, PF) }
DEF_Q2VS(1, , 0) DEF_Q2VS(4, , 0) DEF_Q2VS(1, pf, 1024) DEF_Q2VS(4, pf, 1024)
DEF_Q1VS(1, , 0) DEF_Q1VS(4, , 0) DEF_Q1VS(4, pf, 1024)
#endif // BONSAI_HAVE_VNNI

// ------------------------------------------------- threaded wrappers
// Rows are embarrassingly parallel: each thread owns a disjoint row
// range and writes disjoint outputs. Activations are read-only and
// shared. schedule(static) keeps the P-core/E-core split stable across
// iterations rather than optimal -- hybrid-aware balancing is future
// work, and the sweep below shows where the E-cores stop paying.
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef BONSAI_HAVE_VNNI
#define MT_KERN_Q2 gemv_q2_vnni_r4
#define MT_KERN_Q1 gemv_q1_vnni_r4
#else
#define MT_KERN_Q2 gemv_q2_avx2_r4
#define MT_KERN_Q1 gemv_q1_avx2_r4
#endif

static void gemv_q2_mt(const uint8_t *W, const float *S, const int8_t *P,
                       const int32_t *ASUM, float da, float *out,
                       int N, int K, int nth) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(static)
#endif
    for (int t = 0; t < nth; ++t) {
        const int per = ((N / 8) + nth - 1) / nth * 8;
        const int lo = t * per;
        int hi = lo + per; if (hi > N) hi = N;
        if (lo < hi)
            MT_KERN_Q2(W + (size_t)lo * (K / 4), S + (size_t)lo * (K / BONSAI_QK),
                       P, ASUM, da, out + lo, hi - lo, K);
    }
}

static void gemv_q1_mt(const uint8_t *W, const float *S, const int8_t *P,
                       const int32_t *ASUM, float da, float *out,
                       int N, int K, int nth) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(static)
#endif
    for (int t = 0; t < nth; ++t) {
        const int per = ((N / 8) + nth - 1) / nth * 8;
        const int lo = t * per;
        int hi = lo + per; if (hi > N) hi = N;
        if (lo < hi)
            MT_KERN_Q1(W + (size_t)lo * (K / 8), S + (size_t)lo * (K / BONSAI_QK),
                       P, ASUM, da, out + lo, hi - lo, K);
    }
}

// Dynamic-scheduled wrappers: fixed-size row chunks pulled from a
// shared queue. On a hybrid part a static split hands P- and E-cores
// equal work and the E-cores straggle; work-stealing lets both core
// types contribute what they actually can. Chunk of 256 rows = 512 KiB
// of Q2_0 at K=8192, big enough that queue traffic is noise.
#ifdef BONSAI_HAVE_VNNI
#define MTD_CH 256
static void gemv_q2_mtd(const uint8_t *W, const float *S, const int8_t *P,
                        const int32_t *ASUM, float da, float *out,
                        int N, int K, int nth) {
    const int nch = (N + MTD_CH - 1) / MTD_CH;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
#endif
    for (int c = 0; c < nch; ++c) {
        const int lo = c * MTD_CH;
        const int hi = lo + MTD_CH > N ? N : lo + MTD_CH;
        gemv_q2_vspf_r4(W + (size_t)lo * (K / 4), S + (size_t)lo * (K / BONSAI_QK),
                        P, ASUM, da, out + lo, hi - lo, K);
    }
}
static void gemv_q1_mtd(const uint8_t *W, const float *S, const int8_t *P,
                        const int32_t *ASUM, float da, float *out,
                        int N, int K, int nth) {
    const int nch = (N + MTD_CH - 1) / MTD_CH;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(dynamic, 1)
#endif
    for (int c = 0; c < nch; ++c) {
        const int lo = c * MTD_CH;
        const int hi = lo + MTD_CH > N ? N : lo + MTD_CH;
        gemv_q1_vspf_r4(W + (size_t)lo * (K / 8), S + (size_t)lo * (K / BONSAI_QK),
                        P, ASUM, da, out + lo, hi - lo, K);
    }
}
#endif // BONSAI_HAVE_VNNI

// ------------------------------------------------- BLAS control row
// The same matrix, dequantized to fp32, through cblas_sgemv. CBLAS ABI
// declared inline because the conda OpenBLAS ships no cblas.h; any
// CBLAS links (libopenblas-dev provides the header route on apt hosts).
#ifdef USE_BLAS
enum { CblasRowMajor = 101 };
enum { CblasNoTrans = 111 };
extern void cblas_sgemv(int layout, int trans, int m, int n, float alpha,
                        const float *a, int lda, const float *x, int incx,
                        float beta, float *y, int incy);
extern void openblas_set_num_threads(int);

static void dequant_q2_fp32(const uint8_t *W, const float *S, float *Wf,
                            int N, int K) {
    const int nblk = K / BONSAI_QK;
    for (size_t r = 0; r < (size_t)N; ++r)
        for (int b = 0; b < nblk; ++b) {
            int8_t dec[BONSAI_QK];
            bonsai_q2_ref_decode(W + r*(K/4) + b*BONSAI_Q2_BYTES, dec);
            for (int j = 0; j < BONSAI_QK; ++j)
                Wf[r*K + b*BONSAI_QK + j] = S[r*nblk + b] * (float)dec[j];
        }
}
#endif

// ---------------------------------------------------------------------

int main(int argc, char **argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? atoi(argv[2]) : 8192;
    const int iters = (argc > 3) ? atoi(argv[3]) : 5;
    // "mt": validation + threaded sweep only. For pinning/affinity
    // experiments, where re-running the scalar baselines would dominate
    // the wall clock without adding information.
    const int mt_only = (argc > 4) && !strcmp(argv[4], "mt");
    const int nblk = K / BONSAI_QK;
    if (K % 256 != 0) {
        fprintf(stderr, "K must be a multiple of 256 (q1 pairs blocks)\n");
        return 1;
    }

    printf("shape N=%d K=%d   Q2_0 %.1f MiB   Q1_0 %.1f MiB\n",
           N, K, (double)N*K/4/1048576.0, (double)N*K/8/1048576.0);
#ifdef BONSAI_HAVE_VNNI
    printf("dot step: VPDPBUSD (AVX-VNNI) + VPMADDUBSW ladder\n\n");
#else
    printf("dot step: VPMADDUBSW only (no VNNI at compile time)\n\n");
#endif

    // C11 aligned_alloc requires size to be a multiple of the
    // alignment; round up so small shapes (e.g. K=1024 -> 32-byte ASUM)
    // are not UB.
    #define XALLOC(sz) aligned_alloc(64, ((size_t)(sz) + 63) & ~(size_t)63)
    uint8_t *q2  = XALLOC((size_t)N*K/4);
    uint8_t *q2r = XALLOC((size_t)N*K/4);
    uint8_t *q1  = XALLOC((size_t)N*K/8);
    uint8_t *q1r = XALLOC((size_t)N*K/8);
    float   *S   = XALLOC((size_t)N*nblk*sizeof(float));
    int8_t  *A   = XALLOC(K);
    int8_t  *P2  = XALLOC(K);
    int8_t  *P1  = XALLOC(K);
    int32_t *ASUM= XALLOC(nblk*sizeof(int32_t));
    float   *ref2= XALLOC((size_t)N*sizeof(float));
    float   *ref1= XALLOC((size_t)N*sizeof(float));
    float   *o   = XALLOC((size_t)N*sizeof(float));

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

    VALIDATE("q2 avx2 1 row",  gemv_q2_avx2_r1(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE("q2 avx2 8 rows", gemv_q2_avx2_r8(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE("q1 avx2 1 row",  gemv_q1_avx2_r1(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
    VALIDATE("q1 avx2 8 rows", gemv_q1_avx2_r8(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
#ifdef BONSAI_HAVE_VNNI
    VALIDATE("q2 vnni 1 row",  gemv_q2_vnni_r1(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE("q2 vnni 8 rows", gemv_q2_vnni_r8(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE("q1 vnni 1 row",  gemv_q1_vnni_r1(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
    VALIDATE("q1 vnni 8 rows", gemv_q1_vnni_r8(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
    // v2 kernels reduce lanes in a different order than the scalar
    // reference: float-accurate, not bit-exact (like CUDA and Vulkan).
    // The tolerance is against the FLOAT reference, which is itself a
    // rounding away from truth: at K=8192 a double-precision oracle
    // puts the scalar reference at 4.7e-05 error and vscale at 3.5e-05
    // -- vscale is the more accurate of the two -- yet they sit up to
    // ~1.1e-04 apart from each other. So the reordered kernels get a
    // 1e-3 gate: loose enough for two valid orderings to disagree,
    // tight enough that any real indexing bug (typically > 1e-2) still
    // fails.
    #define VALIDATE_F(name, call, ref) do { \
        memset(o, 0, N*sizeof(float)); call; \
        double worst = 0; \
        for (int vi = 0; vi < NREF; ++vi) { \
            double d = fabs((ref)[vi]) > 1e-6 ? fabs((ref)[vi]) : 1e-6; \
            double e = fabs(o[vi]-(ref)[vi])/d; if (e > worst) worst = e; } \
        printf("  %-24s max rel err %.3e  %s\n", name, worst, \
               worst < 1e-3 ? "OK (float)" : "*** MISMATCH ***"); \
        if (!(worst < 1e-3)) ok = 0; } while (0)
    VALIDATE_F("q2 vscale 1 row", gemv_q2_vs_r1(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE_F("q2 vscale 4 rows",gemv_q2_vs_r4(q2r,S,P2,ASUM,da,o,NREF,K), ref2);
    VALIDATE_F("q1 vscale 1 row", gemv_q1_vs_r1(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
    VALIDATE_F("q1 vscale 4 rows",gemv_q1_vs_r4(q1r,S,P1,ASUM,da,o,NREF,K), ref1);
#endif

#ifdef USE_BLAS
    float *Wf = NULL;
    float *Af = XALLOC(K*sizeof(float));
    if (!mt_only) {
    Wf = XALLOC((size_t)N*K*sizeof(float));
    if (!Wf) { fprintf(stderr, "fp32 matrix alloc failed\n"); return 1; }
    dequant_q2_fp32(q2, S, Wf, N, K);
    for (int j = 0; j < K; ++j) Af[j] = da * (float)A[j];
    openblas_set_num_threads(1);
    memset(o, 0, N*sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, NREF, K, 1.f, Wf, K, Af, 1, 0.f, o, 1);
    {   // fp32 accumulates in a different order; loose tolerance
        double worst = 0;
        for (int i = 0; i < NREF; ++i) {
            double d = fabs(ref2[i]) > 1e-6 ? fabs(ref2[i]) : 1e-6;
            double e = fabs(o[i]-ref2[i])/d; if (e > worst) worst = e;
        }
        printf("  %-24s max rel err %.3e  %s\n", "fp32 sgemv (BLAS)", worst,
               worst < 1e-2 ? "OK" : "*** MISMATCH ***");
        if (!(worst < 1e-2)) ok = 0;
    }
    } // !mt_only
#endif
    if (!ok) { printf("\nVALIDATION FAILED -- timings suppressed\n"); return 1; }

    // The loop counter is deliberately NOT named `i`: `call` is pasted
    // into this loop body, and the MT sweep's call sites index sweep[i]
    // with the *enclosing* i. A counter named i here shadows it, so
    // every timed call would read sweep[0..iters-1] -- a mixture of
    // thread counts reported under one label, and out-of-bounds reads
    // (crashing in libgomp) once iters exceeds the sweep length. This
    // happened; the flat MT scaling it produced looked real for a day.
    #define BENCH(name, bytes, call) do { \
        call; double t0 = now_s(); \
        for (int bench_it_ = 0; bench_it_ < iters; ++bench_it_) { call; } \
        double dt = (now_s()-t0)/iters; \
        printf("  %-24s %8.2f ms  %7.2f GB/s\n", name, dt*1e3, (bytes)/dt/1e9); \
    } while (0)

    const double b2 = (double)N*K/4, b1 = (double)N*K/8;
    if (!mt_only) {
    printf("\n=== throughput, single thread (%d iters) ===\n", iters);
    BENCH("q2 v0 scalar",    b2, gemv_q2_scalar(q2,S,A,da,o,N,K));
    BENCH("q2 avx2 1 row",   b2, gemv_q2_avx2_r1(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 avx2 2 rows",  b2, gemv_q2_avx2_r2(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 avx2 4 rows",  b2, gemv_q2_avx2_r4(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 avx2 8 rows",  b2, gemv_q2_avx2_r8(q2r,S,P2,ASUM,da,o,N,K));
#ifdef BONSAI_HAVE_VNNI
    BENCH("q2 vnni 1 row",   b2, gemv_q2_vnni_r1(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 vnni 2 rows",  b2, gemv_q2_vnni_r2(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 vnni 4 rows",  b2, gemv_q2_vnni_r4(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 vnni 8 rows",  b2, gemv_q2_vnni_r8(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 vscale 1 row", b2, gemv_q2_vs_r1(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 vscale 4 rows",b2, gemv_q2_vs_r4(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 vscale+pf 1 row", b2, gemv_q2_vspf_r1(q2r,S,P2,ASUM,da,o,N,K));
    BENCH("q2 vscale+pf 4 rows",b2, gemv_q2_vspf_r4(q2r,S,P2,ASUM,da,o,N,K));
#endif
    BENCH("q1 v0 scalar",    b1, gemv_q1_scalar(q1,S,A,da,o,N,K));
    BENCH("q1 avx2 1 row",   b1, gemv_q1_avx2_r1(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 avx2 4 rows",  b1, gemv_q1_avx2_r4(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 avx2 8 rows",  b1, gemv_q1_avx2_r8(q1r,S,P1,ASUM,da,o,N,K));
#ifdef BONSAI_HAVE_VNNI
    BENCH("q1 vnni 1 row",   b1, gemv_q1_vnni_r1(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 vnni 4 rows",  b1, gemv_q1_vnni_r4(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 vnni 8 rows",  b1, gemv_q1_vnni_r8(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 vscale 1 row", b1, gemv_q1_vs_r1(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 vscale 4 rows",b1, gemv_q1_vs_r4(q1r,S,P1,ASUM,da,o,N,K));
    BENCH("q1 vscale+pf 4 rows",b1, gemv_q1_vspf_r4(q1r,S,P1,ASUM,da,o,N,K));
#endif

#ifdef USE_BLAS
    // fp32 control: same matrix, 16x the bytes. GB/s below counts the
    // fp32 bytes it actually streams -- compare the ms column, which is
    // what a token costs.
    BENCH("fp32 sgemv 1T (BLAS)", (double)N*K*4,
          cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.f, Wf, K, Af, 1, 0.f, o, 1));
#endif
    } // !mt_only

    printf("\n=== threaded (best kernel, R=4) ===\n");
    const int sweep[] = {1, 2, 4, 6, 8, 12, 16, 20, 24, 32};
    for (size_t i = 0; i < sizeof(sweep)/sizeof(*sweep); ++i) {
        char nm[64];
        snprintf(nm, sizeof nm, "q2 mt t=%d", sweep[i]);
        BENCH(nm, b2, gemv_q2_mt(q2r,S,P2,ASUM,da,o,N,K,sweep[i]));
    }
    for (size_t i = 0; i < sizeof(sweep)/sizeof(*sweep); ++i) {
        char nm[64];
        snprintf(nm, sizeof nm, "q1 mt t=%d", sweep[i]);
        BENCH(nm, b1, gemv_q1_mt(q1r,S,P1,ASUM,da,o,N,K,sweep[i]));
    }
#ifdef BONSAI_HAVE_VNNI
    for (size_t i = 0; i < sizeof(sweep)/sizeof(*sweep); ++i) {
        char nm[64];
        snprintf(nm, sizeof nm, "q2 mtd t=%d", sweep[i]);
        BENCH(nm, b2, gemv_q2_mtd(q2r,S,P2,ASUM,da,o,N,K,sweep[i]));
    }
    for (size_t i = 0; i < sizeof(sweep)/sizeof(*sweep); ++i) {
        char nm[64];
        snprintf(nm, sizeof nm, "q1 mtd t=%d", sweep[i]);
        BENCH(nm, b1, gemv_q1_mtd(q1r,S,P1,ASUM,da,o,N,K,sweep[i]));
    }
#endif
#ifdef USE_BLAS
    if (!mt_only)
    for (size_t i = 0; i < sizeof(sweep)/sizeof(*sweep); ++i) {
        char nm[64];
        snprintf(nm, sizeof nm, "fp32 sgemv t=%d", sweep[i]);
        openblas_set_num_threads(sweep[i]);
        BENCH(nm, (double)N*K*4,
              cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.f, Wf, K, Af, 1, 0.f, o, 1));
    }
    openblas_set_num_threads(1);
#endif

    printf("\nGB/s counts each variant's own streamed weight bytes; the\n"
           "fp32 rows stream 16x Q2_0's bytes, so compare ms per GEMV.\n");
    return 0;
}
