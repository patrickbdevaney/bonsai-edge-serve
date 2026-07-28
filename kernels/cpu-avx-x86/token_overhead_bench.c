// Per-token overhead bench: the DRAM-independent side of a decode step.
//
// A decode step is weight streaming (DRAM-bound, measured elsewhere)
// plus per-token compute that touches only K-sized vectors: quantizing
// the activations to int8 per 128-group and permuting them into the
// x86 plane layout. Both are scalar in the harness today, and both run
// once per projection per layer per token -- 64 layers deep, that is
// real time that no amount of memory bandwidth helps.
//
// What is measured, per K in {5120, 17408} (the recorded model dims:
// hidden and FFN intermediate):
//   quant   scalar per-128-group int8 quantization vs AVX2
//   perm q2 scalar plane permutation vs AVX2 -- the permutation is a
//           4x8 dword transpose per 128-byte block: 4 unpacklo/hi_epi32,
//           4 unpack_epi64, 4 vpermd. 12 shuffles replace 128 scalar
//           stores.
//   perm q1 same idea on block pairs: the pair layout makes each output
//           plane one dword per source vector in both 128-bit lanes, so
//           the identical network runs once for even planes, once for
//           odd.
//   fork-join   cost of an OpenMP parallel region at decode thread
//           counts, since a real decode step opens one per projection.
//
// AVX2 quant matches scalar BIT-EXACT (same round-to-nearest-even, and
// |x*id| <= 127 by construction so pack saturation never fires);
// permutes are byte-exact.
#define BONSAI_NO_MAIN
#include "gemv_avx_bench.c"

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

// ---------------------------------------------------------------------
int main(void) {
    const int Ks[] = {5120, 17408};
    printf("per-token overhead ops, i9-14900HX (single thread)\n\n");

    for (size_t ki = 0; ki < sizeof(Ks)/sizeof(*Ks); ++ki) {
        const int K = Ks[ki], nblk = K / BONSAI_QK;
        float  *x    = XALLOC(K*sizeof(float));
        int8_t *a_s  = XALLOC(K), *a_v = XALLOC(K);
        int8_t *p_s  = XALLOC(K), *p_v = XALLOC(K);
        float  *d_s  = XALLOC(nblk*sizeof(float));
        float  *d_v  = XALLOC(nblk*sizeof(float));
        int32_t*as_s = XALLOC(nblk*sizeof(int32_t));
        int32_t*as_v = XALLOC(nblk*sizeof(int32_t));
        srand(31 + K);
        for (int i = 0; i < K; ++i)
            x[i] = (float)(rand()%2001 - 1000) * 0.001f;

        // correctness first
        quant_groups_scalar(x, K, a_s, d_s, as_s);
        quant_groups_avx2  (x, K, a_v, d_v, as_v);
        const int bq  = memcmp(a_s, a_v, K) != 0;
        const int bas = memcmp(as_s, as_v, nblk*4) != 0;
        const int bd  = memcmp(d_s, d_v, nblk*4) != 0;
        permute_activations_q2(a_s, p_s, K);
        permute_q2_avx2      (a_s, p_v, K);
        const int bp2 = memcmp(p_s, p_v, K) != 0;
        int q1ok = (K % 256) == 0, bp1 = 0;
        if (q1ok) {
            permute_activations_q1(a_s, p_s, K);
            permute_q1_avx2      (a_s, p_v, K);
            bp1 = memcmp(p_s, p_v, K) != 0;
        }
        int bad = bq || bas || bd || bp2 || bp1;
        if (bad)
            printf("  mismatch in:%s%s%s%s%s\n", bq?" quant-codes":"",
                   bas?" asum":"", bd?" scales":"", bp2?" perm-q2":"",
                   bp1?" perm-q1":"");
        printf("K=%d  correctness: %s%s\n", K,
               bad ? "*** MISMATCH ***" : "quant bit-exact, permutes byte-exact",
               q1ok ? "" : " (K%256!=0: q1 n/a)");
        if (bad) return 1;

        #define US(call, iters) ({ \
            call; double t0 = now_s(); \
            for (int it_ = 0; it_ < (iters); ++it_) { call; } \
            (now_s()-t0)/(iters)*1e6; })
        const int IT = 20000;
        double q_s = US(quant_groups_scalar(x,K,a_s,d_s,as_s), IT);
        double q_v = US(quant_groups_avx2  (x,K,a_v,d_v,as_v), IT);
        double p2s = US(permute_activations_q2(a_s,p_s,K), IT);
        double p2v = US(permute_q2_avx2      (a_s,p_v,K), IT);
        printf("  quant    %8.2f us scalar  %8.2f us avx2   %5.1fx\n",
               q_s, q_v, q_s/q_v);
        printf("  perm q2  %8.2f us scalar  %8.2f us avx2   %5.1fx\n",
               p2s, p2v, p2s/p2v);
        if (q1ok) {
            double p1s = US(permute_activations_q1(a_s,p_s,K), IT);
            double p1v = US(permute_q1_avx2      (a_s,p_v,K), IT);
            printf("  perm q1  %8.2f us scalar  %8.2f us avx2   %5.1fx\n",
                   p1s, p1v, p1s/p1v);
        }
        printf("\n");
        free(x); free(a_s); free(a_v); free(p_s); free(p_v);
        free(d_s); free(d_v); free(as_s); free(as_v);
    }

    // fork-join: what one parallel region costs at decode thread counts.
    // A decode step opens one per projection (~7/layer x 64 layers).
#ifdef _OPENMP
    printf("OpenMP fork-join, empty region:\n");
    volatile int sink = 0;
    for (int nth = 2; nth <= 16; nth *= 2) {
        double t0 = now_s();
        const int IT2 = 2000;
        for (int i = 0; i < IT2; ++i) {
            #pragma omp parallel num_threads(nth)
            { sink += omp_get_thread_num(); }
        }
        printf("  t=%-2d  %6.2f us/region\n", nth, (now_s()-t0)/IT2*1e6);
    }
#endif
    return 0;
}
