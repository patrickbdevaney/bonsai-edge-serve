ARM NEON CPU backend.

Named for the instruction set deliberately: this box is ARM (14x Neoverse
V3AE), so NEON/SVE is the path that can be developed and measured here.
x86 AVX-512 is a separate backend and needs a borrowed x86 host -- it is
explicitly NOT covered by this directory, and the fork's existing
AVX512-VNNI repack kernels for Q1_0/Q2_0 are unvalidated by us.

Kernel work lives in ../kernels/cpu-neon-arm/. See ../kernels/README.md
for measured numbers and ../docs/WIKI.md for the learnings.
