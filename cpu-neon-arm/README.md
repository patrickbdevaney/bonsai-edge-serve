ARM NEON CPU backend.

Named for the instruction set deliberately: this box is ARM (14x Neoverse
V3AE), so NEON/SVE is the path that can be developed and measured here.
x86 is a separate backend and is explicitly NOT covered by this
directory -- see ../cpu-avx-x86/ (AVX2 + AVX-VNNI, measured on an
i9-14900HX). The fork's existing AVX512-VNNI repack kernels for
Q1_0/Q2_0 remain unvalidated by us: the x86 host we measured on is
consumer Raptor Lake, which has no AVX-512.

Kernel work lives in ../kernels/cpu-neon-arm/. See ../kernels/README.md
for measured numbers and ../docs/WIKI.md for the learnings.
