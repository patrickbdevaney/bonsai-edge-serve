// Shared low-bit GEMV format definitions for the Bonsai family.
//
// One packed layout, consumed by all three backends (CUDA, Vulkan, CPU).
// This file is the single source of truth for what the bits mean; each
// backend implements the same math against the same bytes.
//
// ---------------------------------------------------------------------
// On-disk GGUF formats (read from ggml-common.h / ggml-quants.c)
// ---------------------------------------------------------------------
//
//   Q2_0: 128 weights/block, fp16 scale + 32 bytes.
//         weight j -> byte j/4, bits (j%4)*2, code c in {0,1,2,3}
//         value = (c - 1) * d, i.e. {-1, 0, +1, +2}
//         NOTE: this is not strictly ternary -- code 3 encodes +2.
//
//   Q1_0: 128 weights/block, fp16 scale + 16 bytes.
//         weight j -> byte j/8, bit j%8
//         value = bit ? +d : -d
//
// ---------------------------------------------------------------------
// The biased-encoding identity (this is what makes it fast everywhere)
// ---------------------------------------------------------------------
//
// Quantize activations to int8 per 128-group: x_j = da * a_j.
//
//   Q2_0:  sum_j w_j x_j = d*da * ( sum_j c_j a_j  -  sum_j a_j )
//   Q1_0:  sum_j w_j x_j = d*da * ( 2*sum_j b_j a_j - sum_j a_j )
//
// So the inner loop is a dot product over the RAW unsigned codes -- no
// sign handling, no per-element subtract -- plus one correction term
// `asum = sum_j a_j` computed once when the activations are quantized.
// The same identity maps onto CUDA __dp4a, Vulkan dotPacked4x8AccSatEXT,
// and ARM SDOT/SMMLA. Measured worth ~+17% on CUDA by itself, because
// the per-element subtract otherwise expands into PRMT+LOP3 chains.
//
// ---------------------------------------------------------------------
// Bit-plane interleaved repack (done once at load)
// ---------------------------------------------------------------------
//
// Goal: extract 4 dp4a-ready lanes with ONE shift + ONE mask.
//
// Take a 32-bit word covering 16 logical weights [base, base+16).
// We want `(w >> 2q) & 0x03030303` to yield the codes for logical
// indices base+4q+0..3 in bytes 0..3 -- i.e. four CONSECUTIVE logical
// weights, so the matching activations are also consecutive and need no
// permutation at all.
//
// That requires byte p of the word to hold logical {p, 4+p, 8+p, 12+p}
// at shifts {0,2,4,6}. The GGUF layout instead has byte p holding
// {4p, 4p+1, 4p+2, 4p+3}. So the repack is a transpose within each group
// of 16 -- a pure permutation, applied once at load time, and it changes
// nothing about the values.
//
// Q1_0 gets the analogous treatment with 32 logical weights per word.
//
// Weights are repacked; ACTIVATIONS STAY IN NATURAL ORDER. That property
// is what keeps the format portable -- every backend can consume the
// same repacked weights with its own native dot instruction.

#ifndef BONSAI_GEMV_H
#define BONSAI_GEMV_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BONSAI_QK 128                 // weights per block, both formats
#define BONSAI_Q2_BYTES (BONSAI_QK/4) // 32
#define BONSAI_Q1_BYTES (BONSAI_QK/8) // 16

// Packed weight blocks, post-repack. Scales are kept in a separate plane
// so the code stream stays 16-byte aligned and densely streamable.
typedef struct { uint8_t qs[BONSAI_Q2_BYTES]; } bonsai_q2_block;
typedef struct { uint8_t qs[BONSAI_Q1_BYTES]; } bonsai_q1_block;

// ---- reference (slow, correctness oracle) ----------------------------

// Decode a GGUF-order Q2_0 block to {-1,0,1,2} codes-minus-one.
static inline void bonsai_q2_ref_decode(const uint8_t *qs, int8_t *out) {
    for (int j = 0; j < BONSAI_QK; ++j) {
        const int c = (qs[j >> 2] >> ((j & 3) * 2)) & 0x3;
        out[j] = (int8_t)(c - 1);
    }
}

static inline void bonsai_q1_ref_decode(const uint8_t *qs, int8_t *out) {
    for (int j = 0; j < BONSAI_QK; ++j) {
        const int b = (qs[j >> 3] >> (j & 7)) & 1;
        out[j] = (int8_t)(b ? 1 : -1);
    }
}

// ---- repack: GGUF order -> bit-plane interleaved ---------------------

// Q2_0. Word w (4 bytes) covers logical [base, base+16).
// Destination: byte p of word holds logical {p,4+p,8+p,12+p} at shifts
// {0,2,4,6}.
static inline void bonsai_q2_repack_block(const uint8_t *src, uint8_t *dst) {
    for (int i = 0; i < BONSAI_Q2_BYTES; ++i) dst[i] = 0;
    for (int j = 0; j < BONSAI_QK; ++j) {
        const int c = (src[j >> 2] >> ((j & 3) * 2)) & 0x3;
        const int word = j >> 4;         // which 16-weight group
        const int r    = j & 15;         // position within the group
        const int p    = r & 3;          // -> byte lane
        const int q    = r >> 2;         // -> shift 2q
        dst[word * 4 + p] |= (uint8_t)(c << (2 * q));
    }
}

// Q1_0. Word w (4 bytes) covers logical [base, base+32).
// Byte p holds logical {p, 4+p, ..., 28+p} at bits {0..7}.
static inline void bonsai_q1_repack_block(const uint8_t *src, uint8_t *dst) {
    for (int i = 0; i < BONSAI_Q1_BYTES; ++i) dst[i] = 0;
    for (int j = 0; j < BONSAI_QK; ++j) {
        const int b = (src[j >> 3] >> (j & 7)) & 1;
        const int word = j >> 5;         // which 32-weight group
        const int r    = j & 31;
        const int p    = r & 3;
        const int q    = r >> 2;         // 0..7 -> bit q
        dst[word * 4 + p] |= (uint8_t)(b << q);
    }
}

// ---- activation quantization ----------------------------------------
//
// Per 128-group symmetric int8. Returns the scale and writes the group
// sum, which is the correction term the biased identity needs.
static inline float bonsai_quantize_activations(const float *x, int n,
                                                int8_t *a, float *asum_out) {
    float amax = 0.0f;
    for (int j = 0; j < n; ++j) {
        const float v = x[j] < 0 ? -x[j] : x[j];
        if (v > amax) amax = v;
    }
    const float d  = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int sum = 0;
    for (int j = 0; j < n; ++j) {
        int q = (int)lrintf(x[j] * id);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        a[j] = (int8_t)q;
        sum += q;
    }
    *asum_out = (float)sum;
    return d;
}

#ifdef __cplusplus
}
#endif
#endif // BONSAI_GEMV_H
