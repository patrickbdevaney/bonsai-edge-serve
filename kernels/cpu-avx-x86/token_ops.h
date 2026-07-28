// Per-token activation ops for the x86 backend: per-128-group int8
// quantization and plane permutation, scalar reference + AVX2. AVX2
// quant is bit-exact vs scalar (same nearest-even rounding; pack
// saturation cannot fire since |x*id| <= 127 by construction);
// permutes are byte-exact. Measured in token_overhead_bench.c.
// Requires gemv_avx_bench.c (or equivalent) included first for
// bonsai_gemv.h types and hsum256_epi32.
#ifndef BONSAI_TOKEN_OPS_H
#define BONSAI_TOKEN_OPS_H

// ---- per-128-group quantization, scalar reference -------------------
// The real decode op: each group gets its own scale and group sum
// (the biased identity's correction term is per group).
static void quant_groups_scalar(const float *x, int K, int8_t *a,
                                float *d, int32_t *asum) {
    for (int b = 0; b < K / BONSAI_QK; ++b) {
        float s;
        d[b] = bonsai_quantize_activations(x + b*BONSAI_QK, BONSAI_QK,
                                           a + b*BONSAI_QK, &s);
        asum[b] = (int32_t)s;
    }
}

// ---- per-128-group quantization, AVX2 -------------------------------
static void quant_groups_avx2(const float *x, int K, int8_t *a,
                              float *d, int32_t *asum) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int b = 0; b < K / BONSAI_QK; ++b) {
        const float *xb = x + b*BONSAI_QK;
        __m256 mx = _mm256_setzero_ps();
        for (int j = 0; j < BONSAI_QK; j += 8)
            mx = _mm256_max_ps(mx, _mm256_and_ps(absmask,
                                       _mm256_loadu_ps(xb + j)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx),
                               _mm256_extractf128_ps(mx, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        const float dd  = amax / 127.0f;
        const float id  = (dd != 0.0f) ? 1.0f / dd : 0.0f;
        const __m256 vid = _mm256_set1_ps(id);
        __m256i vsum = _mm256_setzero_si256();
        int8_t *ab = a + b*BONSAI_QK;
        for (int j = 0; j < BONSAI_QK; j += 32) {
            const __m256i i0 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(xb + j +  0), vid));
            const __m256i i1 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(xb + j +  8), vid));
            const __m256i i2 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(xb + j + 16), vid));
            const __m256i i3 = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(xb + j + 24), vid));
            vsum = _mm256_add_epi32(vsum, _mm256_add_epi32(
                       _mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
            // packs interleave 128-bit lanes; vpermd restores order
            const __m256i p16a = _mm256_packs_epi32(i0, i1);
            const __m256i p16b = _mm256_packs_epi32(i2, i3);
            const __m256i p8   = _mm256_packs_epi16(p16a, p16b);
            const __m256i fix  = _mm256_permutevar8x32_epi32(p8,
                _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256((__m256i*)(ab + j), fix);
        }
        d[b] = dd;
        asum[b] = hsum256_epi32(vsum);
    }
}

// ---- q2 plane permutation, AVX2: 4x8 dword transpose per block ------
static void permute_q2_avx2(const int8_t *A, int8_t *P, int K) {
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int b = 0; b < K / BONSAI_QK; ++b) {
        const __m256i *src = (const __m256i*)(A + b*BONSAI_QK);
        __m256i *dst = (__m256i*)(P + b*BONSAI_QK);
        const __m256i a0 = _mm256_loadu_si256(src + 0);
        const __m256i a1 = _mm256_loadu_si256(src + 1);
        const __m256i a2 = _mm256_loadu_si256(src + 2);
        const __m256i a3 = _mm256_loadu_si256(src + 3);
        const __m256i u0 = _mm256_unpacklo_epi32(a0, a1);
        const __m256i u1 = _mm256_unpackhi_epi32(a0, a1);
        const __m256i u2 = _mm256_unpacklo_epi32(a2, a3);
        const __m256i u3 = _mm256_unpackhi_epi32(a2, a3);
        _mm256_storeu_si256(dst + 0, _mm256_permutevar8x32_epi32(
            _mm256_unpacklo_epi64(u0, u2), order));
        _mm256_storeu_si256(dst + 1, _mm256_permutevar8x32_epi32(
            _mm256_unpackhi_epi64(u0, u2), order));
        _mm256_storeu_si256(dst + 2, _mm256_permutevar8x32_epi32(
            _mm256_unpacklo_epi64(u1, u3), order));
        _mm256_storeu_si256(dst + 3, _mm256_permutevar8x32_epi32(
            _mm256_unpackhi_epi64(u1, u3), order));
    }
}

// ---- q1 plane permutation, AVX2 -------------------------------------
// Pair layout: plane q's low 16 bytes come from block b, high 16 from
// b+1, one dword from each of four source vectors -- so build sources
// with block b in the low lane and b+1 in the high lane, and the q2
// network's per-lane unpacks do the rest. Even planes use sources
// 0,2,4,6; odd planes 1,3,5,7.
static void permute_q1_avx2(const int8_t *A, int8_t *P, int K) {
    // No vpermd here, unlike q2: each 128-bit lane is one block's data
    // end to end, so the per-lane unpacks land every dword in place.
    for (int b = 0; b < K / BONSAI_QK; b += 2) {
        const int8_t *b0 = A + b*BONSAI_QK, *b1 = b0 + BONSAI_QK;
        __m256i *dst = (__m256i*)(P + b*BONSAI_QK);
        __m256i v[8];
        for (int i = 0; i < 8; ++i)
            v[i] = _mm256_loadu2_m128i((const __m128i*)(b1 + 16*i),
                                       (const __m128i*)(b0 + 16*i));
        for (int half = 0; half < 2; ++half) {
            const __m256i s0 = v[half], s1 = v[half+2];
            const __m256i s2 = v[half+4], s3 = v[half+6];
            const __m256i u0 = _mm256_unpacklo_epi32(s0, s1);
            const __m256i u1 = _mm256_unpackhi_epi32(s0, s1);
            const __m256i u2 = _mm256_unpacklo_epi32(s2, s3);
            const __m256i u3 = _mm256_unpackhi_epi32(s2, s3);
            _mm256_storeu_si256(dst + 0 + half*4, _mm256_unpacklo_epi64(u0, u2));
            _mm256_storeu_si256(dst + 1 + half*4, _mm256_unpackhi_epi64(u0, u2));
            _mm256_storeu_si256(dst + 2 + half*4, _mm256_unpacklo_epi64(u1, u3));
            _mm256_storeu_si256(dst + 3 + half*4, _mm256_unpackhi_epi64(u1, u3));
        }
    }
}


#endif // BONSAI_TOKEN_OPS_H
