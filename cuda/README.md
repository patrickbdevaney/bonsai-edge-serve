CUDA backend.

The batch-1 GEMV ladder (v0-v7, best v5 at 178.5 GB/s on Thor) lives in ../kernels/cuda/;
cross-architecture compile verification in arch/. The fused single-launch
engine with DSpark (Phase 2) is not started. See ../README.md and ../kernels/README.md.
