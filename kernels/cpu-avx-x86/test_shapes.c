// Shape sweep for the x86 AVX kernels -- the mirror of the ARM
// 100-shape sweep. The bench always runs friendly shapes; real model
// layers do not, and the R>1 kernels' tail handling (N % R rows
// finished through the R=1 variant) plus the MT wrappers' last-chunk
// clamping are exactly the code the friendly shapes never touch. Every
// kernel that ships is run over the full (K, N) grid, including N
// values chosen to hit each tail residue and K values matching the
// model's real projection dims, and checked against the scalar
// reference.
//
// The gate is error normalised by max|ref| over the output -- the
// standard GEMM criterion, and the same metric the ARM sweep switched
// to (L16): over signed data a long dot product cancels, so individual
// outputs land near zero and an elementwise-relative metric turns
// ~1e-8 of absolute agreement into a "failure". The first version of
// this sweep used the elementwise gate and reproduced the ARM lesson
// verbatim: 7 "failures", every one a near-cancelled row (|ref| down
// to 2e-8 against typical outputs of ~1e-3), got and ref agreeing to
// ~5e-05 of the double-precision oracle. Bit-exact kernels gate at
// 1e-7, reordered (vscale/mtd) at 1e-4.
//
// Build & run: make test
#define BONSAI_NO_MAIN
#include "gemv_avx_bench.c"

static int fails = 0, runs = 0;

static void check(const char *kind, int K, int N, const float *got,
                  const float *ref, double gate) {
    double worst = 0, amax = 0;
    for (int i = 0; i < N; ++i) {
        const double e = fabs(got[i] - ref[i]);
        const double a = fabs(ref[i]);
        if (e > worst) worst = e;
        if (a > amax)  amax = a;
    }
    const double err = worst / (amax > 0 ? amax : 1.0);
    ++runs;
    if (!(err < gate)) {
        ++fails;
        printf("FAIL %-16s K=%-5d N=%-5d  err %.3e vs max|ref| %.3e (gate %.0e)\n",
               kind, K, N, err, amax, gate);
    }
}

int main(void) {
    // K: multiples of 256 exercise both formats (q1 pairs blocks);
    // 2560/5120/12288-class dims are the model-shaped ones. K values
    // that are multiples of 128 but NOT 256 (384, 1152) exercise the
    // q2 path alone and the q1 guard.
    const int Ks[] = {256, 384, 512, 1024, 1152, 2560, 4096, 5120, 8192};
    // N: every residue mod 8 plus 1 and model-ish sizes.
    const int Ns[] = {1, 2, 3, 5, 7, 8, 13, 100, 127, 256, 1000, 2049};

    for (size_t ki = 0; ki < sizeof(Ks)/sizeof(*Ks); ++ki)
    for (size_t ni = 0; ni < sizeof(Ns)/sizeof(*Ns); ++ni) {
        const int K = Ks[ki], N = Ns[ni];
        const int nblk = K / BONSAI_QK;
        const int q1ok = (K % 256) == 0;

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

        srand(1000003 * K + N);
        for (size_t i = 0; i < (size_t)N*K/4; ++i) q2[i] = rand() & 0xFF;
        for (size_t i = 0; i < (size_t)N*K/8; ++i) q1[i] = rand() & 0xFF;
        for (size_t i = 0; i < (size_t)N*nblk; ++i)
            S[i] = 0.005f + 0.002f*(i%11);
        for (int i = 0; i < K; ++i) A[i] = (int8_t)((rand()%255)-127);
        for (int b = 0; b < nblk; ++b) {
            int s = 0;
            for (int j = 0; j < BONSAI_QK; ++j) s += A[b*BONSAI_QK+j];
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
        if (q1ok) permute_activations_q1(A, P1, K);
        const float da = 0.0037f;

        gemv_q2_scalar(q2, S, A, da, ref2, N, K);
        if (q1ok) gemv_q1_scalar(q1, S, A, da, ref1, N, K);

        #define T(kind, call, ref, gate) do { \
            memset(o, 0, (size_t)N*sizeof(float)); call; \
            check(kind, K, N, o, ref, gate); } while (0)

        T("q2 avx2 r8",   gemv_q2_avx2_r8(q2r,S,P2,ASUM,da,o,N,K), ref2, 1e-7);
        T("q2 avx2 r4",   gemv_q2_avx2_r4(q2r,S,P2,ASUM,da,o,N,K), ref2, 1e-7);
#ifdef BONSAI_HAVE_VNNI
        T("q2 vnni r8",   gemv_q2_vnni_r8(q2r,S,P2,ASUM,da,o,N,K), ref2, 1e-7);
        T("q2 vscale r4", gemv_q2_vs_r4(q2r,S,P2,ASUM,da,o,N,K),   ref2, 1e-4);
        T("q2 vscale+pf", gemv_q2_vspf_r4(q2r,S,P2,ASUM,da,o,N,K), ref2, 1e-4);
        T("q2 mt t=4",    gemv_q2_mt(q2r,S,P2,ASUM,da,o,N,K,4),    ref2, 1e-7);
        T("q2 mtd t=4",   gemv_q2_mtd(q2r,S,P2,ASUM,da,o,N,K,4),   ref2, 1e-4);
#endif
        if (q1ok) {
            T("q1 avx2 r8",   gemv_q1_avx2_r8(q1r,S,P1,ASUM,da,o,N,K), ref1, 1e-7);
#ifdef BONSAI_HAVE_VNNI
            T("q1 vnni r8",   gemv_q1_vnni_r8(q1r,S,P1,ASUM,da,o,N,K), ref1, 1e-7);
            T("q1 vscale r4", gemv_q1_vs_r4(q1r,S,P1,ASUM,da,o,N,K),   ref1, 1e-4);
            T("q1 mt t=4",    gemv_q1_mt(q1r,S,P1,ASUM,da,o,N,K,4),    ref1, 1e-7);
            T("q1 mtd t=4",   gemv_q1_mtd(q1r,S,P1,ASUM,da,o,N,K,4),   ref1, 1e-4);
#endif
        }

        free(q2); free(q2r); free(q1); free(q1r); free(S); free(A);
        free(P2); free(P1); free(ASUM); free(ref2); free(ref1); free(o);
    }

    printf("%d kernel/shape combinations, %d failures\n", runs, fails);
    return fails ? 1 : 0;
}
