x86 AVX CPU backend.

Named for the instruction set deliberately, and precisely: this box is an
Intel i9-14900HX (Raptor Lake, 8P+16E), which has **AVX2 + AVX-VNNI and
no AVX-512** -- AVX-512 is fused off on consumer Raptor Lake, so the
fork's AVX512-VNNI repack kernels cannot even be run here, let alone
validated. What CAN be developed and measured here is the VEX-encoded
path: 256-bit unpack plus `VPDPBUSD` (AVX-VNNI), with a pure-AVX2
`VPMADDUBSW` ladder for parts without VNNI. That is also the wider
target: every x86 CPU since Haswell runs the AVX2 path, and AVX-VNNI
covers Alder Lake onward, which is a strictly larger set than the
AVX-512 machines.

VPDPBUSD is u8 x s8, which matches the biased encoding exactly: raw
unsigned codes dotted with signed int8 activations, the correction
subtracted once per block. Same shared packed format as CUDA, Vulkan
and NEON; activations get the x86-shaped permutation, once per token.

A BLAS control row (fp32 `cblas_sgemv` over the dequantized matrix,
OpenBLAS) is built into the bench: at batch 1 the fp32 matrix streams
16x the bytes of Q2_0 through the same DRAM, and the measurement shows
exactly what that costs. BLAS is the control, not the contender.

Kernel work lives in ../kernels/cpu-avx-x86/. See ../kernels/README.md
for measured numbers and ../results/gemv-avx-x86.txt for the raw log.
