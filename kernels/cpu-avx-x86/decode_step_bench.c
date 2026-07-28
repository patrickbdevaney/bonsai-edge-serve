// Decode-step proxy: the full per-token projection schedule of
// Bonsai-27B on the x86 backend, measured in tokens/second.
//
// WHAT THIS IS: every GEMV a token costs, at the recorded shapes
// (RESEARCH_FINDINGS RQ5: hidden 5120, FFN 17408, vocab 248320, 64
// layers = 48 GDN + 16 full attention), with the real per-projection
// pipeline -- quantize activations per 128-group, permute to plane
// layout, low-bit GEMV -- run under the measured integration rules
// (one parallel region per token, work-stealing row chunks).
//
// WHAT THIS IS NOT: a working model. Weights are random bytes (any
// byte string is valid packed data), and the elementwise glue between
// projections (norms, gates, GDN recurrence, attention over KV) is not
// computed -- each projection quantizes its own persistent input
// buffer. Those omissions are compute the real decode adds ON TOP;
// per RESEARCH_FINDINGS the GDN state math is ~4-8% of per-token
// traffic. So the number here is a weights-and-overhead tok/s CEILING
// for a real decode step, not a claim of one.
//
// Projection schedule per layer (proxy, from the recorded head
// geometry):
//   GDN layer (48x):  in-proj  5120 -> 10240   (48v+16k+16q heads x128)
//                     out-proj 6144 -> 5120
//   Attn layer (16x): qkv      5120 -> 8192    (24q x256 + 2x4kv x256)
//                     out      6144 -> 5120
//   FFN (all 64):     gate,up  5120 -> 17408 (x2)
//                     down     17408 -> 5120
//   lm_head (1x):     5120 -> 248320
//
// Run: decode_step_bench q2|q1 [threads] [tokens] [naive]
//   "naive" opens a parallel region per projection instead of one per
//   token -- the anti-pattern, kept measurable on purpose.
#define BONSAI_NO_MAIN
#include "gemv_avx_bench.c"
#include "token_ops.h"

typedef struct { int K, N; } proj_t;

typedef struct {
    int K, N, nblk;
    uint8_t *W;          // packed codes, repacked layout, random
    float   *S;          // per-row per-block scales
    float   *x;          // persistent input activations (proxy)
    int8_t  *a, *p;      // quantized + permuted activations
    float   *dscale;     // per-group activation scales (computed, unused
                         // by the single-da kernels; priced anyway)
    int32_t *asum;
    float   *out;
} layer_buf;

static uint64_t rng = 0x243F6A8885A308D3ull;
static inline uint64_t xs64(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng;
}

static size_t build(layer_buf *L, int K, int N, int isq1) {
    L->K = K; L->N = N; L->nblk = K / BONSAI_QK;
    const size_t wb = (size_t)N * K / (isq1 ? 8 : 4);
    L->W = XALLOC(wb);
    L->S = XALLOC((size_t)N * L->nblk * sizeof(float));
    L->x = XALLOC(K * sizeof(float));
    L->a = XALLOC(K); L->p = XALLOC(K);
    L->dscale = XALLOC(L->nblk * sizeof(float));
    L->asum = XALLOC(L->nblk * sizeof(int32_t));
    L->out = XALLOC((size_t)N * sizeof(float));
    if (!L->W || !L->S || !L->out) {
        fprintf(stderr, "alloc failed at K=%d N=%d\n", K, N); exit(1);
    }
    uint64_t *w8 = (uint64_t*)L->W;
    for (size_t i = 0; i < wb / 8; ++i) w8[i] = xs64();
    for (size_t i = 0; i < (size_t)N * L->nblk; ++i)
        L->S[i] = 0.004f + 0.002f * (float)(xs64() & 7) / 7.0f;
    for (int i = 0; i < K; ++i)
        L->x[i] = (float)((int)(xs64() % 2001) - 1000) * 0.001f;
    return wb;
}

int main(int argc, char **argv) {
    const int isq1  = (argc > 1) && !strcmp(argv[1], "q1");
    const int nth   = (argc > 2) ? atoi(argv[2]) : 8;
    const int toks  = (argc > 3) ? atoi(argv[3]) : 10;
    const int naive = (argc > 4) && !strcmp(argv[4], "naive");

    const proj_t gdn[]  = {{5120, 10240}, {6144, 5120}};
    const proj_t attn[] = {{5120, 8192},  {6144, 5120}};
    const proj_t ffn[]  = {{5120, 17408}, {5120, 17408}, {17408, 5120}};
    const proj_t head   = {5120, 248320};

    // flatten the 64-layer schedule
    enum { MAXP = 64 * 5 + 1 };
    layer_buf *P = calloc(MAXP, sizeof(layer_buf));
    int np = 0;
    size_t total_wb = 0;
    for (int l = 0; l < 64; ++l) {
        const proj_t *blk = (l % 4 == 3) ? attn : gdn;   // every 4th: full attn
        for (int i = 0; i < 2; ++i) total_wb += build(&P[np++], blk[i].K, blk[i].N, isq1);
        for (int i = 0; i < 3; ++i) total_wb += build(&P[np++], ffn[i].K, ffn[i].N, isq1);
    }
    total_wb += build(&P[np++], head.K, head.N, isq1);

    printf("decode-step proxy, %s, %d projections, %.2f GiB weights/token"
           " (recorded model: %s)\n",
           isq1 ? "Q1_0" : "Q2_0", np, (double)total_wb / (1u << 30),
           isq1 ? "3.80 GB" : "6.67 GiB");
    printf("threads=%d  region=%s\n\n", nth,
           naive ? "per-projection (anti-pattern)" : "per-token");

    const float da = 0.0037f;   // kernels take one da; per-group scales
                                // are computed and priced but not wired
    #define RUN_PROJ(L) do {                                              \
        if (naive) {                                                      \
            quant_groups_avx2((L)->x, (L)->K, (L)->a, (L)->dscale, (L)->asum); \
            if (isq1) permute_q1_avx2((L)->a, (L)->p, (L)->K);            \
            else      permute_q2_avx2((L)->a, (L)->p, (L)->K);            \
            if (isq1) gemv_q1_mtd((L)->W, (L)->S, (L)->p, (L)->asum, da,  \
                                  (L)->out, (L)->N, (L)->K, nth);         \
            else      gemv_q2_mtd((L)->W, (L)->S, (L)->p, (L)->asum, da,  \
                                  (L)->out, (L)->N, (L)->K, nth);         \
        }                                                                 \
    } while (0)

    double best = 1e30, sum = 0;
    for (int t = 0; t < toks + 2; ++t) {
        const double t0 = now_s();
        if (naive) {
            for (int i = 0; i < np; ++i) RUN_PROJ(&P[i]);
        } else {
            // one region per token: quant/permute on one thread (0.8 ms
            // total, measured), GEMV rows work-stolen by everyone, the
            // omp-for barrier is the inter-projection dependency
            #pragma omp parallel num_threads(nth)
            for (int i = 0; i < np; ++i) {
                layer_buf *L = &P[i];
                #pragma omp single
                {
                    quant_groups_avx2(L->x, L->K, L->a, L->dscale, L->asum);
                    if (isq1) permute_q1_avx2(L->a, L->p, L->K);
                    else      permute_q2_avx2(L->a, L->p, L->K);
                }
                const int CH2 = 128;
                const int nch = (L->N + CH2 - 1) / CH2;
                #pragma omp for schedule(dynamic, 1)
                for (int c = 0; c < nch; ++c) {
                    const int lo = c * CH2;
                    const int hi = lo + CH2 > L->N ? L->N : lo + CH2;
                    if (isq1)
                        gemv_q1_vspf_r4(L->W + (size_t)lo * (L->K / 8),
                            L->S + (size_t)lo * L->nblk, L->p, L->asum, da,
                            L->out + lo, hi - lo, L->K);
                    else
                        gemv_q2_vspf_r4(L->W + (size_t)lo * (L->K / 4),
                            L->S + (size_t)lo * L->nblk, L->p, L->asum, da,
                            L->out + lo, hi - lo, L->K);
                }
            }
        }
        const double dt = now_s() - t0;
        if (t < 2) continue;               // warmup tokens
        sum += dt; if (dt < best) best = dt;
        printf("  token %2d  %7.2f ms\n", t - 2, dt * 1e3);
    }
    const double avg = sum / toks;
    printf("\n%s decode-step proxy: best %.2f ms/token (%.2f tok/s), "
           "avg %.2f ms (%.2f tok/s), %.1f GB/s of weight bytes\n",
           isq1 ? "Q1_0" : "Q2_0", best * 1e3, 1.0 / best,
           avg * 1e3, 1.0 / avg, (double)total_wb / best / 1e9);
    return 0;
}
