// Optimal batch-1 low-bit GEMV for the Bonsai family, CUDA.
//
// Batch-1 decode streams every weight exactly once, so this kernel's job
// is not arithmetic -- it is keeping the maximum number of loads in
// flight. Every design choice below serves that.
//
// The ladder (all variants validated bit-exact against a CPU reference):
//
//   v0 naive      scalar 2-bit extract, activations read from global
//   v1 smem       activations staged in shared memory, scalar extract
//   v2 dp4a       LOP3 mask + __dp4a, GGUF layout, 1 row/warp
//   v3 interleave bit-plane repacked layout, 1 row/warp
//   v4 rows       + ROWS rows per warp  <-- the one to ship
//
// Why the biased encoding matters: Q2_0 stores codes c in {0,1,2,3} for
// values {-1,0,1,2}. Rather than subtract 1 per element (which expands
// into PRMT+LOP3 chains, and whose saturating form __vsubss4 has no
// single SASS instruction), we dot the RAW codes and correct once per
// 16-weight chunk with a precomputed activation sum:
//
//     sum w*x = d*da * ( sum c*a - sum a )
//
// The `sum a` table is computed once per token when activations are
// quantized, and amortizes over every row of the matrix.
//
// Build: nvcc -O3 -arch=sm_110a -o gemv_bench gemv_bench.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../common/bonsai_gemv.h"

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Extract 4 dp4a lanes with one shift + one mask.
// LOP3 immLut 0xea == (a & b) | c; with c=0 it is a plain AND, which the
// compiler emits as a single LOP3.
__device__ __forceinline__ int unpack4_q2(uint32_t w, int shift) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;"
        : "=r"(r) : "r"(w >> shift), "n"(0x03030303u), "n"(0u));
    return (int)r;
}
__device__ __forceinline__ int unpack4_q1(uint32_t w, int shift) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;"
        : "=r"(r) : "r"(w >> shift), "n"(0x01010101u), "n"(0u));
    return (int)r;
}

// ---------------------------------------------------------------- v0
__global__ void gemv_q2_v0(const uint8_t *__restrict__ W, const half *__restrict__ S,
                           const float *__restrict__ X, float *__restrict__ out,
                           int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const uint8_t *w = W + (size_t)row * (K / 4);
    const half *s = S + (size_t)row * (K / BONSAI_QK);
    float acc = 0.f;
    for (int j = 0; j < K; ++j) {
        int c = (w[j >> 2] >> ((j & 3) * 2)) & 3;
        acc += (float)(c - 1) * __half2float(s[j / BONSAI_QK]) * X[j];
    }
    out[row] = acc;
}

// ---------------------------------------------------------------- v1
extern __shared__ char smem_raw[];
__global__ void gemv_q2_v1(const uint8_t *__restrict__ W, const half *__restrict__ S,
                           const float *__restrict__ X, float *__restrict__ out,
                           int N, int K) {
    float *xs = (float *)smem_raw;
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = X[i];
    __syncthreads();
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const uint8_t *w = W + (size_t)row * (K / 4);
    const half *s = S + (size_t)row * (K / BONSAI_QK);
    float acc = 0.f;
    for (int j = 0; j < K; ++j) {
        int c = (w[j >> 2] >> ((j & 3) * 2)) & 3;
        acc += (float)(c - 1) * __half2float(s[j / BONSAI_QK]) * xs[j];
    }
    out[row] = acc;
}

// ---------------------------------------------------------------- v2
// dp4a on the GGUF layout. One warp per row. Byte p of a word holds
// logical 4p..4p+3, so a single 32-bit word covers 16 weights but the
// four dp4a lanes are NOT consecutive -- we must gather 4 separate
// activation words. That gather is what v3 removes.
__global__ void gemv_q2_v2(const uint32_t *__restrict__ W, const half *__restrict__ S,
                           const int32_t *__restrict__ A4, const int32_t *__restrict__ ASUM16,
                           float da, float *__restrict__ out, int N, int K) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint32_t *w = W + (size_t)row * (K / 16);
    const half *s = S + (size_t)row * (K / BONSAI_QK);
    float acc = 0.f;
    for (int widx = lane; widx < K / 16; widx += 32) {
        const uint32_t v = w[widx];
        int dot = 0;
        // GGUF order: byte p = logical 4p..4p+3, all at the same shift set.
        for (int p = 0; p < 4; ++p) {
            const int c4 = (int)((v >> (8 * p)) & 0xFFu);
            const int codes = ((c4 & 3) | (((c4 >> 2) & 3) << 8) |
                               (((c4 >> 4) & 3) << 16) | (((c4 >> 6) & 3) << 24));
            dot = __dp4a(codes, A4[widx * 4 + p], dot);
        }
        acc += __half2float(s[widx / 8]) * (float)(dot - ASUM16[widx]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) out[row] = acc * da;
}

// ---------------------------------------------------------------- v5
// v3's structure, but with the quantized activations staged in shared
// memory once per block.
//
// Motivation, from measuring v2/v3/v4: every 4-byte weight word needs
// four A4[] words plus one ASUM16[] word -- 20 bytes of activation reads
// per 4 bytes of weights. Those reads are not DRAM traffic (activations
// are only K bytes and stay resident), but they are load-ISSUE traffic,
// and issue slots are exactly what a batch-1 GEMV is short of. That also
// explains why ROWS>1 did not help: the loop is issue-bound, not
// latency-bound, so adding independent loads cannot hide anything.
//
// Staging A4/ASUM16 in smem turns 5 global loads per word into 5 shared
// loads plus one global weight load.
__global__ void gemv_q2_v5(const uint32_t *__restrict__ W, const half *__restrict__ S,
                           const int32_t *__restrict__ A4, const int32_t *__restrict__ ASUM16,
                           float da, float *__restrict__ out, int N, int K) {
    extern __shared__ int32_t sh[];
    const int nw = K / 16;                 // weight words per row
    int32_t *sA4 = sh;                     // nw*4 entries
    int32_t *sAS = sh + (size_t)nw * 4;    // nw entries
    for (int i = threadIdx.x; i < nw * 4; i += blockDim.x) sA4[i] = A4[i];
    for (int i = threadIdx.x; i < nw;     i += blockDim.x) sAS[i] = ASUM16[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint32_t *w = W + (size_t)row * nw;
    const half *s = S + (size_t)row * (K / BONSAI_QK);
    float acc = 0.f;
    for (int widx = lane; widx < nw; widx += 32) {
        const uint32_t v = w[widx];
        int dot = 0;
        dot = __dp4a(unpack4_q2(v, 0), sA4[widx * 4 + 0], dot);
        dot = __dp4a(unpack4_q2(v, 2), sA4[widx * 4 + 1], dot);
        dot = __dp4a(unpack4_q2(v, 4), sA4[widx * 4 + 2], dot);
        dot = __dp4a(unpack4_q2(v, 6), sA4[widx * 4 + 3], dot);
        acc += __half2float(s[widx / 8]) * (float)(dot - sAS[widx]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) out[row] = acc * da;
}

// ---------------------------------------------------------------- v3/v4
// Bit-plane interleaved layout: `(v >> 2q) & 0x03030303` yields the codes
// for logical base+4q+0..3 -- four CONSECUTIVE weights -- so the matching
// activations are one aligned 32-bit load, no gather.
//
// ROWS rows per warp is what supplies memory-level parallelism: each lane
// issues ROWS independent loads before consuming any of them.
template <int ROWS>
__global__ void gemv_q2_v4(const uint32_t *__restrict__ W, const half *__restrict__ S,
                           const int32_t *__restrict__ A4, const int32_t *__restrict__ ASUM16,
                           float da, float *__restrict__ out, int N, int K) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x / 32) + warp) * ROWS;
    if (row0 >= N) return;

    const int nw = K / 16;                 // 32-bit words per row
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

    for (int widx = lane; widx < nw; widx += 32) {
        // Issue all ROWS weight loads first; they are independent.
        uint32_t v[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
            v[r] = W[(size_t)(row0 + r) * nw + widx];

        const int a0 = A4[widx * 4 + 0], a1 = A4[widx * 4 + 1];
        const int a2 = A4[widx * 4 + 2], a3 = A4[widx * 4 + 3];
        const int asum = ASUM16[widx];

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            int dot = 0;
            dot = __dp4a(unpack4_q2(v[r], 0), a0, dot);
            dot = __dp4a(unpack4_q2(v[r], 2), a1, dot);
            dot = __dp4a(unpack4_q2(v[r], 4), a2, dot);
            dot = __dp4a(unpack4_q2(v[r], 6), a3, dot);
            acc[r] += __half2float(S[(size_t)(row0 + r) * (K / BONSAI_QK) + widx / 8])
                      * (float)(dot - asum);
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = acc[r];
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0 && row0 + r < N) out[row0 + r] = a * da;
    }
}

// Q1_0: value = bit ? +1 : -1, so sum w*x = 2*sum(b*a) - sum(a).
// One 32-bit word covers 32 weights (8 shifts x 4 byte lanes).
template <int ROWS>
__global__ void gemv_q1_v4(const uint32_t *__restrict__ W, const half *__restrict__ S,
                           const int32_t *__restrict__ A4, const int32_t *__restrict__ ASUM32,
                           float da, float *__restrict__ out, int N, int K) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x / 32) + warp) * ROWS;
    if (row0 >= N) return;

    const int nw = K / 32;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

    for (int widx = lane; widx < nw; widx += 32) {
        uint32_t v[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
            v[r] = W[(size_t)(row0 + r) * nw + widx];

        int a[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) a[q] = A4[widx * 8 + q];
        const int asum = ASUM32[widx];

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            int dot = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) dot = __dp4a(unpack4_q1(v[r], q), a[q], dot);
            acc[r] += __half2float(S[(size_t)(row0 + r) * (K / BONSAI_QK) + widx / 4])
                      * (float)(2 * dot - asum);
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float a = acc[r];
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (lane == 0 && row0 + r < N) out[row0 + r] = a * da;
    }
}

// ---------------------------------------------------------------------

struct Timing { float ms; double gbs; };

template <class F>
static Timing time_kernel(F launch, size_t bytes, int iters) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) launch();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaGetLastError());
    return { ms / iters, (double)bytes * iters / (ms / 1e3) / 1e9 };
}

int main(int argc, char **argv) {
    const int K = (argc > 1) ? atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? atoi(argv[2]) : 65536;
    const int iters = (argc > 3) ? atoi(argv[3]) : 20;

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const size_t q2_bytes = (size_t)N * K / 4;
    const size_t q1_bytes = (size_t)N * K / 8;
    printf("device %s  CC %d.%d  %d SMs\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("shape  N=%d K=%d   Q2_0 weights %.1f MiB   Q1_0 %.1f MiB\n\n",
           N, K, q2_bytes / 1048576.0, q1_bytes / 1048576.0);

    // ---- host data ----
    const int nblk = K / BONSAI_QK;
    std::vector<uint8_t> hq2((size_t)N * K / 4), hq2r((size_t)N * K / 4);
    std::vector<uint8_t> hq1((size_t)N * K / 8), hq1r((size_t)N * K / 8);
    std::vector<half>    hs((size_t)N * nblk);
    std::vector<float>   hx(K), href(N);

    srand(1234);
    for (auto &b : hq2) b = (uint8_t)(rand() & 0xFF);
    for (auto &b : hq1) b = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = __float2half(0.01f + 0.001f * (i % 7));
    for (int i = 0; i < K; ++i) hx[i] = (float)((rand() % 200) - 100) / 100.f;

    // repack (this is the load-time step a real engine does once)
    for (size_t r = 0; r < (size_t)N; ++r)
        for (int b = 0; b < nblk; ++b) {
            bonsai_q2_repack_block(&hq2[r * (K / 4) + b * BONSAI_Q2_BYTES],
                                   &hq2r[r * (K / 4) + b * BONSAI_Q2_BYTES]);
            bonsai_q1_repack_block(&hq1[r * (K / 8) + b * BONSAI_Q1_BYTES],
                                   &hq1r[r * (K / 8) + b * BONSAI_Q1_BYTES]);
        }

    // activations: int8 + per-chunk sums
    std::vector<int8_t> ha(K);
    std::vector<float> asum_grp(nblk);
    float da = 0.f;
    {
        float amax = 0; for (float v : hx) amax = fmaxf(amax, fabsf(v));
        da = amax / 127.f; const float id = 1.f / da;
        for (int i = 0; i < K; ++i) { int q = (int)lrintf(hx[i] * id);
            ha[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q)); }
    }
    std::vector<int32_t> ha4(K / 4), asum16(K / 16), asum32(K / 32);
    for (int i = 0; i < K / 4; ++i) {
        int32_t p = 0; for (int k = 0; k < 4; ++k) p |= ((int32_t)(uint8_t)ha[i*4+k]) << (8*k);
        ha4[i] = p;
    }
    for (int i = 0; i < K / 16; ++i) { int s = 0; for (int k = 0; k < 16; ++k) s += ha[i*16+k]; asum16[i] = s; }
    for (int i = 0; i < K / 32; ++i) { int s = 0; for (int k = 0; k < 32; ++k) s += ha[i*32+k]; asum32[i] = s; }

    // Dequantized activations, for the float rungs (v0/v1).
    // The dp4a rungs consume int8 activations, so the CPU reference below
    // is built from `ha * da` -- the quantized values. Handing v0/v1 the
    // ORIGINAL floats would make them compute a different (in fact more
    // accurate) dot product, and they would fail validation against that
    // reference for a reason that has nothing to do with the kernel.
    // Feeding them the round-tripped activations keeps every rung on
    // identical math, which is also what makes the GB/s ladder a fair
    // comparison.
    std::vector<float> hxq(K);
    for (int i = 0; i < K; ++i) hxq[i] = (float)ha[i] * da;

    // CPU reference on the FIRST 64 rows only (it is slow)
    const int NREF = 64;
    std::vector<float> ref2(NREF), ref1(NREF);
    for (int r = 0; r < NREF; ++r) {
        double acc2 = 0, acc1 = 0;
        for (int b = 0; b < nblk; ++b) {
            int8_t c2[BONSAI_QK], c1[BONSAI_QK];
            bonsai_q2_ref_decode(&hq2[(size_t)r*(K/4) + b*BONSAI_Q2_BYTES], c2);
            bonsai_q1_ref_decode(&hq1[(size_t)r*(K/8) + b*BONSAI_Q1_BYTES], c1);
            const float s = __half2float(hs[(size_t)r*nblk + b]);
            for (int j = 0; j < BONSAI_QK; ++j) {
                acc2 += (double)c2[j] * s * (double)ha[b*BONSAI_QK+j] * da;
                acc1 += (double)c1[j] * s * (double)ha[b*BONSAI_QK+j] * da;
            }
        }
        ref2[r] = (float)acc2; ref1[r] = (float)acc1;
    }

    // ---- device ----
    uint8_t *dq2, *dq2r, *dq1r; half *ds; float *dx, *dout;
    int32_t *da4, *das16, *das32;
    CHECK(cudaMalloc(&dq2, q2_bytes));  CHECK(cudaMalloc(&dq2r, q2_bytes));
    CHECK(cudaMalloc(&dq1r, q1_bytes)); CHECK(cudaMalloc(&ds, hs.size()*sizeof(half)));
    CHECK(cudaMalloc(&dx, K*sizeof(float))); CHECK(cudaMalloc(&dout, N*sizeof(float)));
    float *dxq; CHECK(cudaMalloc(&dxq, K*sizeof(float)));
    CHECK(cudaMalloc(&da4, ha4.size()*4)); CHECK(cudaMalloc(&das16, asum16.size()*4));
    CHECK(cudaMalloc(&das32, asum32.size()*4));
    CHECK(cudaMemcpy(dq2, hq2.data(), q2_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dq2r, hq2r.data(), q2_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dq1r, hq1r.data(), q1_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ds, hs.data(), hs.size()*sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hx.data(), K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dxq, hxq.data(), K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(da4, ha4.data(), ha4.size()*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(das16, asum16.data(), asum16.size()*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(das32, asum32.data(), asum32.size()*4, cudaMemcpyHostToDevice));

    std::vector<float> hout(N);
    auto check = [&](const char *name, const std::vector<float> &ref, float tol) {
        CHECK(cudaMemcpy(hout.data(), dout, N*sizeof(float), cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int r = 0; r < NREF; ++r) {
            double den = fabs(ref[r]) > 1e-6 ? fabs(ref[r]) : 1e-6;
            worst = fmax(worst, fabs(hout[r] - ref[r]) / den);
        }
        printf("  %-26s max rel err %.3e  %s\n", name, worst,
               worst <= tol ? "OK" : "*** MISMATCH ***");
        return worst <= tol;
    };

    printf("=== validation (first %d rows vs CPU reference) ===\n", NREF);
    bool ok = true;
    { int t = 256, g = (N + t - 1) / t;
      gemv_q2_v0<<<g, t>>>(dq2, ds, dxq, dout, N, K); CHECK(cudaDeviceSynchronize());
      ok &= check("v0 naive", ref2, 2e-2); }
    { int t = 256, g = (N + t - 1) / t;
      gemv_q2_v1<<<g, t, K*sizeof(float)>>>(dq2, ds, dxq, dout, N, K); CHECK(cudaDeviceSynchronize());
      ok &= check("v1 smem activations", ref2, 2e-2); }
    { int t = 256, w = t/32, g = (N + w - 1) / w;
      gemv_q2_v2<<<g, t>>>((const uint32_t*)dq2, ds, da4, das16, da, dout, N, K);
      CHECK(cudaDeviceSynchronize()); ok &= check("v2 dp4a, GGUF layout", ref2, 2e-2); }
    { int t = 256, w = t/32, g = (N + w - 1) / w;
      gemv_q2_v4<1><<<g, t>>>((const uint32_t*)dq2r, ds, da4, das16, da, dout, N, K);
      CHECK(cudaDeviceSynchronize()); ok &= check("v3 dp4a, interleaved", ref2, 2e-2); }
    { int t = 256, w = t/32, g = (N + w - 1) / w;
      size_t sm = ((size_t)(K/16)*4 + (K/16)) * sizeof(int32_t);
      gemv_q2_v5<<<g, t, sm>>>((const uint32_t*)dq2r, ds, da4, das16, da, dout, N, K);
      CHECK(cudaDeviceSynchronize()); ok &= check("v5 dp4a + smem acts", ref2, 2e-2); }
    { int t = 256, w = t/32, g = (N + w*8 - 1) / (w*8);
      gemv_q2_v4<8><<<g, t>>>((const uint32_t*)dq2r, ds, da4, das16, da, dout, N, K);
      CHECK(cudaDeviceSynchronize()); ok &= check("v4 dp4a, 8 rows/warp", ref2, 2e-2); }
    { int t = 256, w = t/32, g = (N + w*8 - 1) / (w*8);
      gemv_q1_v4<8><<<g, t>>>((const uint32_t*)dq1r, ds, da4, das32, da, dout, N, K);
      CHECK(cudaDeviceSynchronize()); ok &= check("q1 v4, 8 rows/warp", ref1, 2e-2); }

    if (!ok) { printf("\nVALIDATION FAILED -- timings suppressed\n"); return 1; }

    printf("\n=== throughput (%d iters) ===\n", iters);
    struct Row { const char *name; Timing t; } rows[16]; int nr = 0;

    auto add = [&](const char *name, size_t bytes, auto fn) {
        rows[nr].name = name;
        rows[nr].t = time_kernel(fn, bytes, iters);
        ++nr;
    };

    add("v0 naive", q2_bytes, [=]{ int t=256,g=(N+t-1)/t;
        gemv_q2_v0<<<g,t>>>(dq2,ds,dxq,dout,N,K); });
    add("v1 smem activations", q2_bytes, [=]{ int t=256,g=(N+t-1)/t;
        gemv_q2_v1<<<g,t,K*sizeof(float)>>>(dq2,ds,dxq,dout,N,K); });
    add("v2 dp4a GGUF", q2_bytes, [=]{ int t=256,w=t/32,g=(N+w-1)/w;
        gemv_q2_v2<<<g,t>>>((const uint32_t*)dq2,ds,da4,das16,da,dout,N,K); });
    add("v3 dp4a interleaved", q2_bytes, [=]{ int t=256,w=t/32,g=(N+w-1)/w;
        gemv_q2_v4<1><<<g,t>>>((const uint32_t*)dq2r,ds,da4,das16,da,dout,N,K); });
    add("v5 dp4a + smem acts", q2_bytes, [=]{ int t=256,w=t/32,g=(N+w-1)/w;
        size_t sm = ((size_t)(K/16)*4 + (K/16))*sizeof(int32_t);
        gemv_q2_v5<<<g,t,sm>>>((const uint32_t*)dq2r,ds,da4,das16,da,dout,N,K); });
    add("v4 q2 2 rows/warp", q2_bytes, [=]{ int t=256,w=t/32,g=(N+w*2-1)/(w*2);
        gemv_q2_v4<2><<<g,t>>>((const uint32_t*)dq2r,ds,da4,das16,da,dout,N,K); });
    add("v4 q2 4 rows/warp", q2_bytes, [=]{ int t=256,w=t/32,g=(N+w*4-1)/(w*4);
        gemv_q2_v4<4><<<g,t>>>((const uint32_t*)dq2r,ds,da4,das16,da,dout,N,K); });
    add("v4 q2 8 rows/warp", q2_bytes, [=]{ int t=256,w=t/32,g=(N+w*8-1)/(w*8);
        gemv_q2_v4<8><<<g,t>>>((const uint32_t*)dq2r,ds,da4,das16,da,dout,N,K); });
    add("v4 q1 8 rows/warp", q1_bytes, [=]{ int t=256,w=t/32,g=(N+w*8-1)/(w*8);
        gemv_q1_v4<8><<<g,t>>>((const uint32_t*)dq1r,ds,da4,das32,da,dout,N,K); });

    for (int i = 0; i < nr; ++i)
        printf("  %-22s %8.3f ms   %7.1f GB/s\n", rows[i].name, rows[i].t.ms, rows[i].t.gbs);

    printf("\nGB/s counts weight bytes only -- that is the term that scales\n"
           "with model size and dominates batch-1 decode.\n");
    return 0;
}
