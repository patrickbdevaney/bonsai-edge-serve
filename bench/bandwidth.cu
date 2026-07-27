// Measured achievable memory bandwidth, for the decode roofline.
//
// Batch-1 low-bit decode is memory-bandwidth bound: each step streams the
// whole weight set once. So the ceiling is
//
//     tok/s_max = achievable_bandwidth / bytes_per_step
//
// and every measured tok/s should be read as a fraction of that. Vendor
// peak bandwidth is not the right denominator -- what a real kernel can
// sustain is. This measures the latter with a pure streaming read, which
// is the access pattern a weight-streaming GEMV actually has.
//
// Reported separately for float4 (128-bit) and uint (32-bit) loads,
// because low-bit unpacking kernels sometimes cannot use the widest load.
//
// Build:  nvcc -O3 -arch=native -o bandwidth bandwidth.cu
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return 1; } } while (0)

__global__ void read_f4(const float4 *__restrict__ in, float *__restrict__ sink, size_t n4) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n4; i += stride) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // Never true; keeps the loads from being optimized away.
    if (acc == 1234.5678f) sink[0] = acc;
}

__global__ void read_u32(const unsigned int *__restrict__ in, float *__restrict__ sink, size_t n) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    size_t stride = (size_t) gridDim.x * blockDim.x;
    unsigned int acc = 0;
    for (; i < n; i += stride) acc ^= in[i];
    if (acc == 0xDEADBEEFu) sink[0] = (float) acc;
}

int main(int argc, char **argv) {
    size_t mb = (argc > 1) ? (size_t) atoll(argv[1]) : 2048;
    size_t bytes = mb * 1024ull * 1024ull;
    int iters = (argc > 2) ? atoi(argv[2]) : 20;

    void *buf = nullptr;
    float *sink = nullptr;
    CHECK(cudaMalloc(&buf, bytes));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(buf, 1, bytes));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount * 32;

    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));

    printf("device            : %s (CC %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("buffer            : %zu MiB, %d iters\n", mb, iters);

    for (int mode = 0; mode < 2; ++mode) {
        // Warm up, then time.
        for (int i = 0; i < 3; ++i) {
            if (mode == 0) read_f4<<<blocks, 256>>>((const float4 *) buf, sink, bytes / 16);
            else           read_u32<<<blocks, 256>>>((const unsigned int *) buf, sink, bytes / 4);
        }
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) {
            if (mode == 0) read_f4<<<blocks, 256>>>((const float4 *) buf, sink, bytes / 16);
            else           read_u32<<<blocks, 256>>>((const unsigned int *) buf, sink, bytes / 4);
        }
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));

        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        double gb = (double) bytes * iters / 1e9;
        printf("read %-12s : %7.1f GB/s\n",
               mode == 0 ? "(float4)" : "(uint32)", gb / (ms / 1e3));
    }

    cudaFree(buf);
    cudaFree(sink);
    return 0;
}
