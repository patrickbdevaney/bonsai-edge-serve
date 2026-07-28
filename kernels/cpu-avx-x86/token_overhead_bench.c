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

#include "token_ops.h"

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
