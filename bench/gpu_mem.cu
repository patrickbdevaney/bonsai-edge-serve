// Print CUDA device memory as "free_bytes total_bytes".
//
// Needed because nvidia-smi reports "Not Supported" for memory usage on
// Jetson (Thor/Orin): CPU and GPU share one physical pool, so there is no
// separate GPU memory figure for it to report. cudaMemGetInfo does work
// there, and is what the bench harness uses to get a real resident number.
//
// Build:  nvcc -o gpu_mem gpu_mem.cu
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    size_t free_b = 0, total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("%zu %zu\n", free_b, total_b);
    return 0;
}
