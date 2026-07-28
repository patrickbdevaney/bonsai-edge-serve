# Decode kernels

Batch-1 low-bit GEMV for the Bonsai family, across CUDA, Vulkan, ARM
NEON and x86 AVX. This is the core of Phase 2: at batch 1 the model streams every
weight exactly once per token, so decode speed is this kernel.

## The design in one page

**One weight format, three code generators.** `common/bonsai_gemv.h` is
the single source of truth. Backends differ in their dot instruction and
their vector width, not in what the bytes mean.

**Biased encoding.** Q2_0 stores codes `c` in `{0,1,2,3}` for values
`{-1,0,+1,+2}` (note: not strictly ternary -- code 3 is +2). Rather than
subtract 1 per element, dot the raw codes and correct once per chunk:

```
Q2_0:  sum w*x = d*da * ( sum c*a  -  sum a )
Q1_0:  sum w*x = d*da * ( 2*sum b*a - sum a )
```

The `sum a` table is built once per token when activations are quantized
and amortizes over every row. No sign handling, no branches, and it maps
identically onto CUDA `__dp4a`, Vulkan `dotPacked4x8AccSatEXT`, and ARM
`SDOT`. Measured worth ~+17% on CUDA on its own, because the per-element
subtract otherwise expands into PRMT+LOP3 chains (`__vsubss4` has no
single SASS instruction).

**Bit-plane interleaved repack.** The weight word is rearranged so that
`(w >> 2q) & 0x03030303` yields four *consecutive* logical weights. Then
the matching activations are one aligned load rather than a gather.
Convergent with llama.cpp's TQ2_0, microsoft/BitNet, and bonsai-turbo --
treat as settled.

**Where the permutation lands.** The repack makes lanes consecutive for a
*4-byte* word. CUDA's `__dp4a` and Vulkan's `dotPacked4x8` both consume
exactly 4 bytes, so both use the same weights *and* the same activation
layout, unpermuted. NEON's vector is 16 bytes and spans four weight
words, so its lanes are four groups of four strided by 16 -- it needs an
activation shuffle. Doing that on the activations (K bytes, once per
token) rather than on the weights (N*K/4 bytes, shared with every other
backend) is the cheap side of the trade.

## Status

| Backend | Correctness | Performance |
| :-- | :-- | :-- |
| CPU (NEON) | **bit-exact vs scalar reference** | **measured, below** |
| CPU (x86 AVX2/VNNI) | **bit-exact vs scalar reference** | **measured, below** |
| CUDA | **validated, all 7 rungs** | **measured, below** |
| Vulkan | **validated, both shaders** | **measured, below** |

All four backends now decode the same shared packed format to the same
answer, which is the property the format exists to provide.

## All three backends, one format, one box

The point of the shared format is that the same packed weights decode
correctly and fast everywhere. That is now measured rather than asserted.
Best kernel per backend, Q2_0, N=131072 K=8192, GB/s of weight bytes:

| Backend | GB/s | % of the 273 GB/s SoC spec | note |
| :-- | --: | --: | :-- |
| CUDA (v5, dp4a + smem acts) | **177.2** | 65% | best overall |
| Vulkan (int-dot Tier 3) | 103.5 | 38% | needs glslang >= 16 |
| CPU (whole-vector + SDOT, 1T) | 5.8 | 2% | single thread |

Correctness: every backend matches the same CPU reference -- CUDA
1.2e-07, Vulkan 4.8e-06, NEON bit-exact.

The Vulkan row needed a glslang built from source to reach it; both
system compilers lack the GLSL front end for the hardware dot-product
instruction (see below). The CPU column is single-threaded and is
included for shape, not as a backend comparison.

## Measured: CUDA, Jetson Thor (sm_110a)

Shape N=131072 K=8192 (256 MiB of Q2_0 weights, 8x the 32 MB L2, so this
is a genuine streaming test). Full log in `../results/gemv-cuda.txt`.

| Kernel | GB/s | % of 273 GB/s spec |
| :-- | --: | --: |
| v0 naive, scalar extract | 16.0 | 6% |
| v1 + activations in smem, scalar extract | 12.6 | 5% |
| v2 dp4a, GGUF layout | 168.8 | 62% |
| v3 dp4a, bit-plane interleaved | 170.3 | 62% |
| **v5 dp4a + activations in smem** | **178.5** | **65%** |
| v6 + 128-bit (`uint4`) weight loads | 158.1 | 58% |
| v7 smem acts + 2 rows/warp | 168.1 | 62% |
| v7 smem acts + 4 rows/warp | 138.0 | 51% |
| v7 smem acts + 8 rows/warp | 133.6 | 49% |
| v4 2 rows/warp (global acts) | 142.5 | 52% |
| v4 4 rows/warp (global acts) | 117.8 | 43% |
| v4 8 rows/warp (global acts) | 124.4 | 46% |
| v4 Q1_0, 8 rows/warp | 102.5 | 38% |

**11x from naive to best.** The dp4a step is the whole cliff (16 -> 169);
the bit-plane repack adds ~1%, and staging activations in shared memory
another ~5%. **v5 -- one row per warp, activations in shared memory --
is the kernel to ship.**

### Three ways of adding memory-level parallelism, all slower

Everything tried to widen or deepen the loop lost, and the pattern is
consistent enough to be a design rule for this kernel:

| Attempt | Idea | Result |
| :-- | :-- | --: |
| v4 | rows/warp, activations in global | 142.5 -> 117.8 |
| v6 | 128-bit `uint4` weight loads | 158.1 |
| v7 | rows/warp, activations in **shared** | 168.1 -> 133.6 |

v6 is the one that pins the diagnosis. Wide loads cut *weight* load
instructions 4:1 -- and cost 12%. That only makes sense if weight loads
were never the constraint: each 4-byte weight word needs four `A4[]`
words plus one `ASUM16[]`, so **20 of every 21 loads in the inner loop
are activations**. Widening the 1 does nothing and costs register
pressure.

v7 was the fair test of row-blocking, since v4's version was amortizing
*global* activation reads and could have been losing for that reason
alone. With the activations already in shared memory it still loses
monotonically. So rows/warp is not being defeated by the cost of the read
it shares -- it is defeated by what it costs in occupancy and per-row
scale gathers.

Conclusion: this kernel is **issue-bound and occupancy-sensitive**, not
latency-bound. The lever that worked (v5) reduced the *number* of
instructions per weight byte; every lever that added parallelism to hide
latency made it worse.

### Two results that contradict the research brief

**Row-blocking makes this kernel slower, not faster.** The literature
pass predicted 8 rows/warp would close the last 25% (158 -> 229 GB/s). It
does the opposite here: 171 -> 135 -> 117 -> 123 as ROWS goes 1/2/4/8.

The reason is visible in the inner loop. Each 4-byte weight word needs
four `A4[]` words plus one `ASUM16[]` word -- 20 bytes of activation
reads per 4 bytes of weights. Those are not DRAM traffic (activations are
only K bytes and stay resident), but they are load-*issue* traffic, and
issue slots are what a batch-1 GEMV is short of. So the loop is
issue-bound, not latency-bound, and adding independent loads cannot hide
anything -- it only adds register pressure and, for ROWS>1, an
uncoalesced per-row scale gather.

That diagnosis is what v5 tests directly, and it holds: moving the
activation reads to shared memory is the only change that improved on
v3.

**We could not reproduce 229 GB/s (84% of spec).** Our best is 177 GB/s
(65%), at the same shape the claim was made for. Since the ladder below
177 matches the predicted shape closely (naive is catastrophic, dp4a is
the cliff, interleaving helps a little), the gap is most likely in the
part of the design we could not reconstruct from the summary. **Until it
is reproduced, 177 GB/s is what this repo claims.**

### What that does to the roofline

At 177 GB/s achieved, ternary decode (6.67 GiB of weights per token) tops
out near **26 tok/s**, not the ~32 implied by 229 GB/s. The reference
fork measures 16.77, so it is at **~64% of what our own best kernel would
allow** -- real headroom, about 1.55x, but materially less than the ~2x
the research pass suggested. The honest version of the Phase 2 claim is
"~1.5x from kernel work", not "~2x".

## Measured: Vulkan, Jetson Thor

Same shape, same shared format. Full log in `../results/gemv-vulkan.txt`.

| Shader | GB/s | vs portable | vs CUDA best |
| :-- | --: | --: | --: |
| Q2_0 portable (Tier 1) | 101.1 | -- | 57% |
| **Q2_0 int-dot (Tier 3)** | **103.5** | **+2%** | 58% |
| Q1_0 portable (Tier 1) | 71.2 | -- | 40% |
| **Q1_0 int-dot (Tier 3)** | **99.4** | **+40%** | 56% |

All four validate against the same CPU reference, and the int-dot
variants return *identical* error figures to the portable ones (4.8e-06
Q2_0, 9.6e-07 Q1_0), which is the check that the accelerated path is
computing the same thing rather than merely computing something faster.

**Q1_0 gains 40%, Q2_0 only 2%**, and the asymmetry is structural rather
than surprising: a Q1_0 word packs 32 weights and needs eight `dot4`
calls, against four for Q2_0's 16 weights, so the scalar fallback's
per-call cost weighs twice as heavily. The lower the bit width, the more
the hardware dot instruction is worth.

### Getting the int-dot path to build at all

This was the blocker, and it was a toolchain problem, not a device one.
Thor's driver reports `integerDotProduct4x8BitPackedSignedAccelerated =
true`, but neither system compiler has the GLSL front end:

| Compiler | glslang | `GL_EXT_integer_dot_product` |
| :-- | :-- | :-- |
| `glslc` 2023.8 (Ubuntu 24.04) | 14.0.0 | not supported |
| `glslang-tools` 15.1.0 (apt) | 15.1.0 | not supported |
| **built from source** | **16.4.0** | **supported** |

It is not a `--target-env` issue -- `vulkan1.1/1.2/1.3` all fail
identically. Build recipe and the exact failure text are in
`../results/gemv-vulkan.txt`.

One shader fix was needed too, and it is a trap worth naming. The
overloads are:

```glsl
uint dotPacked4x8AccSatEXT(uint, uint, uint)
int  dotPacked4x8AccSatEXT(uint, int,  int)   // the one we want
```

Activations are packed **signed** int8 while codes are unsigned, so the
activation word must be cast: `dotPacked4x8AccSatEXT(c, int(a), acc)`.
Passing both as `uint` selects the all-unsigned overload, which treats
activations as 0..255. Here that fails to compile against an `int`
accumulator, so it surfaces loudly -- but with a `uint` accumulator it
would have been a silent wrong-answer bug.

### This affects the whole Vulkan backend, not just these kernels

`ggml-vulkan` runs the same feature test
(`ggml/src/ggml-vulkan/CMakeLists.txt`, `test_shader_extension_support`).
With the system `glslc` it resolves to *"GL_EXT_integer_dot_product not
supported by glslc"* and compiles the **entire Vulkan backend with no
integer-dot path**. So the llama.cpp Vulkan numbers elsewhere in this
repo (11.79 / 15.98 tok/s) were measured with that path disabled, and
rebuilding the fork with a newer glslc should lift them independently of
anything we write.

## Measured: CPU (ARM NEON), Jetson Thor, 14x Neoverse V3AE

Shape N=32768 K=4096. All variants bit-exact (max rel err 0.000e+00).
Full log in `../results/gemv-neon.txt`.

Single thread, GB/s of weight bytes:

| Kernel | Q2_0 | Q1_0 |
| :-- | --: | --: |
| scalar per-element extract (shape of the current ARM path) | 0.37 | 0.03 |
| whole-vector unpack + SDOT | **5.78** | **2.78** |

**~16x for ternary, ~90x for 1-bit** over per-element extract. Row
blocking (1/2/4/8 accumulator chains) made no difference single-threaded
-- this kernel is unpack-ALU bound, not latency bound, which matches the
core's issue rates (SHR/AND ~4/cycle, SDOT ~2/cycle).

Threaded, 8 rows per chain:

| threads | Q2_0 GB/s | Q1_0 GB/s |
| --: | --: | --: |
| 1 | 5.72 | 2.75 |
| 4 | 14.90 | 7.86 |
| 8 | 26.41 | 14.46 |
| 10 | 26.82 | **16.91** |
| **12** | **38.75** | 15.43 |
| 14 | 19.01 | 14.39 |

**14 threads collapses** -- independently reproducing the earlier
end-to-end finding that leaving cores for the OS is worth more than using
them. Optimum is 12 for Q2_0 and 10 for Q1_0.

### What that means end to end, and the honest split

Projecting weights-only bandwidth onto the real model
(ternary 7.17 GB, 1-bit 3.80 GB of weights per token):

| Variant | This kernel | llama.cpp today (t=12) | Ratio |
| :-- | --: | --: | --: |
| ternary Q2_0 | **5.40 tok/s** | 3.18 | **1.70x** |
| 1-bit Q1_0 | 4.45 tok/s | 4.85 | **0.92x** |

**Ternary is a clear win; 1-bit is a loss.** That is not a bug, it is the
predicted result: SDOT retires 16 weights per instruction regardless of
bit width, so a 1-bit kernel shaped like a 2-bit one gets the same
weights/s while moving half the bytes -- it is purely ALU-bound and gains
nothing from being smaller. llama.cpp's Q1_0 path uses a lookup table,
which retires more weights per instruction.

The fix is a T-MAC style **register-resident LUT** (`vqtbl1q`, g=4),
published and independently measured at ~4.4x for 1-bit on this class of
core. It is deliberately not half-implemented here, and the reason is
structural rather than a matter of effort:

1. **Table range.** For g=4 an entry is a signed sum of 4 activations,
   range +/-508, so entries need int16. `vqtbl1q` returns bytes, so an
   exact version needs a hi/lo split (two lookups, recombined as
   `lo + 256*hi`) or a scaled table that gives up bit-exactness. The
   split is fine -- 4 TBL per 128 weights is still 32 weights/TBL against
   SDOT's 16 weights/instruction.

2. **The blocker: a LUT indexes one table across 16 lanes, but each
   activation group needs a *different* table.** Group `t` uses
   activations `a[4t..4t+3]`, so a 128-weight block needs 32 distinct
   tables. `vqtbl1q` applies one table to all 16 lanes. T-MAC resolves
   this by making the 16 lanes **16 different output rows** at a fixed
   activation group, not 16 different groups of one row -- which requires
   the weights to be **row-interleaved**, a different packing from the
   one CUDA and Vulkan use.

So the 1-bit CPU win costs a CPU-specific weight layout, and therefore a
decision the rest of the design has so far avoided: either the CPU
backend repacks again at load (paying memory for a second copy, or CPU
time to transform), or the shared format gains a row-interleaved variant
that the GPU backends would have to tolerate. That is a real
architectural choice and it should be made deliberately with the
measurement in hand, not slipped in. Recommended next step: prototype the
row-interleaved 1-bit LUT kernel standalone, measure it against the 2.78
GB/s SDOT baseline, and only then decide whether the win justifies the
second layout.

Note this also reframes the "one format, three backends" claim honestly:
it holds for the dot-product-shaped kernels (CUDA, Vulkan, CPU 2-bit),
and the CPU 1-bit path is where it may have to bend.

## Measured: CPU (x86 AVX2 / AVX-VNNI), i9-14900HX

The x86 leg of the cross-device story (GAP 3). Host: Intel i9-14900HX
(Raptor Lake, 8 P + 16 E cores), gcc 13, laptop DDR5. This part has
**no AVX-512** -- it is fused off on consumer Raptor Lake -- so the
fork's AVX512-VNNI repack kernels are not runnable here; what is
measured is the VEX path, which also happens to be the wider target
(AVX2 reaches every x86 since Haswell, AVX-VNNI reaches Alder Lake+).

Same shared packed format, same activation-side permutation trade as
NEON (a 32-byte vector spans eight weight words). The dot step is where
x86 is actually a *better* fit than ARM: `VPDPBUSD` is u8 x s8 -> s32,
which is the biased encoding verbatim -- raw unsigned codes against
signed int8 activations, correction once per block. The AVX2-only
fallback spells the same contraction as `VPMADDUBSW` + `VPMADDWD`
(3 uops instead of 1), exact because codes <= 3 keep the s16
intermediate at 762 max.

All v1 variants validate **bit-exact** (max rel err 0.000e+00) against
the same scalar reference the other backends use; `-ffp-contract=off`
keeps the per-block float accumulate in the reference's rounding order.
Beyond the bench's friendly shapes, `make test` sweeps 108 (K, N)
combinations -- every tail residue, N=1, and the model-shaped dims --
through every shipped kernel and both MT wrappers: 1176 combinations
pass. The sweep already paid for itself twice: it caught the static MT
wrapper dropping N%8 rows (the bench's round shapes could never see
it), and it reproduced the ARM sweep's error-metric lesson (L16)
verbatim -- 7 "failures" that were near-cancelled rows under an
elementwise-relative gate, all within 5e-05 of a double-precision
oracle, now judged by error over max|ref| like the ARM tests. Full
log in `../results/gemv-avx-x86.txt`.

Streaming shape N=131072 K=8192 (256 MiB Q2_0, 7x the 36 MiB L3),
single thread, GB/s of weight bytes:

| Kernel | Q2_0 | Q1_0 |
| :-- | --: | --: |
| scalar per-element extract | 0.20 | 0.02 |
| AVX2 maddubs, best R | 6.87 | 3.13 |
| AVX-VNNI vpdpbusd, best R | 8.64 | 4.12 |
| + vscale (deferred reduction) | 10.96 | **5.46** |
| **+ software prefetch (PF=1024)** | **12.22** | 5.21 |

VNNI is worth +26-32% over the maddubs spelling of the same math.
**vscale** is the v2 restructure: the v1 kernels paid an `hsum256` plus
a scalar convert+fma per block per row to apply the per-block scale;
since the scale is uniform across lanes, v2 defers the reduction --
`fma(broadcast(scale), cvtepi32_ps(acc))` per block, one horizontal sum
per row. ~5 fewer uops per 32 weight bytes on an issue-bound loop:
+27% Q2_0, +39% Q1_0. It costs bit-exactness (lanes sum in a different
order); a double-precision oracle puts vscale at 3.5e-05 vs the scalar
reference's own 4.7e-05, so it is the *more* accurate ordering, and the
bench validates it at a documented 1e-3 float gate. Prefetch buys Q2_0
another +11% and does nothing for Q1_0, whose stream runs at half the
byte rate. Row blocking stays nearly flat, as on ARM.

Threaded, same streaming shape:

| Config | Q2_0 | Q1_0 |
| :-- | --: | --: |
| v1 static split, unpinned best | 24.7 | 18.8 |
| v1 static split, pinned P-cores | 30.3 | 24.8 |
| **v2 + work-stealing, unpinned** | **28.2** (t=6) | **29.8** (t=8) |
| v2 + work-stealing, pinned P | 28.0 | 28.9 |
| DRAM ceiling (fp32 sgemv, threaded) | 31.0 | -- |

**Both formats now saturate the measured DRAM ceiling, and neither
needs pinning.** The v1 static row split handed P- and E-cores equal
work, so on a hybrid part the E-cores straggled: unpinned Q1_0 lost 20%
and the fix was manual pinning. v2 pulls 256-row chunks from a shared
queue (`schedule(dynamic)`), so each core type contributes what it can:
unpinned Q1_0 goes 18.8 -> 29.8 GB/s -- the ALU-bound format is the one
work-stealing rescues, because recruiting E-core ALU is exactly what it
was missing -- and it reaches the ceiling at 8 threads where v1 needed
32 pinned to get to 80%. Q2_0's ceiling was already reachable; the win
there is needing 6 unpinned threads instead of 32 pinned ones.

The BLAS control row: the same matrix dequantized to fp32 through
OpenBLAS `cblas_sgemv` takes **238 ms single-threaded against 31 ms**
for the Q2_0 VNNI kernel at the same shape -- 7.6x -- and its best
threaded time, 139 ms, is still 15.6x slower than our best threaded
kernel (8.9 ms). BLAS's AVX kernels are fine; streaming 16x the bytes
is what loses. That measurement, not an assertion, is why the low-bit
format exists.

### Post-mortem: the flat-scaling result that wasn't

The first published version of this section reported multithread
scaling as flat at ~9 GB/s against the 30 GB/s ceiling and called it an
open item. That number was wrong, and the way it was wrong is worth a
paragraph, because the bug is a landmine anyone extending the bench
could re-arm.

The `BENCH` macro times `call` inside `for (int i = 0; i < iters; ++i)`.
The MT sweep's call sites said `gemv_q2_mt(..., sweep[i])` -- and the
macro's `i` **shadowed the sweep index**, so every timed row ran thread
counts `sweep[0..iters-1]` instead of the one in its label: ten rows,
each reporting the same mixture's average. Flat by construction. Worse,
any `iters` beyond the sweep length walked off the array and handed
libgomp garbage `num_threads` (a crash that only fired at `iters=40`,
which is how it was caught). Three things made the diagnosis: the
flatness contradicted the sgemv ceiling measured in the same run; a
standalone driver with the same kernel scaled to the ceiling; and the
crash pinned it. The macro's counter is now named `bench_it_`, and the
comment on it says why. Lesson restated: a benchmark whose result
contradicts a control measured in the same process is broken until
proven otherwise -- the contradiction was visible in the first
published table.

### The Q1_0 LUT, scoped for x86 and deprioritized

The ARM section flags a T-MAC register-LUT kernel as the fix for 1-bit
being ALU-bound, worth ~4.4x there. The x86 op-count model says the
same trick does not transfer to this ISA tier, and it is worth showing
why rather than porting it on faith. The ARM win has two ingredients:
`SDOT` retires only 16 weights per instruction, and `TBL` retires 32
lookup lanes with cheap byte adds. On x86, `VPDPBUSD` already retires
32 weights per instruction -- the dot side starts twice as wide -- and
AVX2's `VPSHUFB` offers the same 32 lanes but no `vpermb`, so the
exact int16 table must be split into lo/hi byte planes and re-widened
(2 shuffles + ~10 support uops per 128 MACs against ~12.5 for the VNNI
path). Modeled, not measured: **~1.0-1.4x**, against ARM's measured
4.4x -- not worth the row-interleaved second weight layout it would
cost the shared format. And since work-stealing already puts threaded
Q1_0 at the DRAM ceiling, a LUT could only help the few-thread case.
Revisit on AVX-512 VBMI hosts (`vpermb`: 64 lanes, no plane split),
where the model tips the other way.

### What that means end to end

Projecting weights-only bandwidth onto the real model (ternary 7.17 GB,
1-bit 3.80 GB of weights per token), at the measured unpinned 28-30 GB/s
for both formats: **~4.2 tok/s ternary, ~7.8 tok/s 1-bit** as this
machine's weights-bandwidth roofline for a decode step built on these
kernels -- no pinning, 6-8 threads. Unlike the Thor CPU result, 1-bit
wins end to end here: the DRAM is slow enough relative to 24 cores of
unpack ALU that halving the bytes pays in full. For comparison, fp32 at
the same ceiling would be ~0.3 tok/s: the format is the difference
between "unusable" and "usable-slow" on commodity x86.

## Building

```bash
# CPU, ARM
cc -O3 -march=armv8.2-a+dotprod -fopenmp \
   -o cpu-neon-arm/gemv_neon_bench cpu-neon-arm/gemv_neon_bench.c -lm
./cpu-neon-arm/gemv_neon_bench 4096 32768 3

# CPU, x86 (see cpu-avx-x86/Makefile; `make blas` adds the fp32 control
# row -- point BLAS_LIB/BLAS_SO at any CBLAS, incl. a conda OpenBLAS)
cc -O3 -mavx2 -mavxvnni -mfma -ffp-contract=off -fopenmp \
   -o cpu-avx-x86/gemv_avx_bench cpu-avx-x86/gemv_avx_bench.c -lm
./cpu-avx-x86/gemv_avx_bench 8192 131072 3

# CUDA (needs a working GPU to run)
nvcc -O3 -arch=sm_110a --extended-lambda -o cuda/gemv_bench cuda/gemv_bench.cu
./cuda/gemv_bench 4096 65536 20

# Vulkan. GLSLANG must be >= 16; the system glslc (2023.8 / glslang 14)
# and apt glslang-tools 15.1 both lack GL_EXT_integer_dot_product.
#   git clone --depth 1 https://github.com/KhronosGroup/glslang.git
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=0 \
#         -DGLSLANG_TESTS=OFF -DENABLE_GLSLANG_BINARIES=ON
#   cmake --build build -j 12      # -> build/StandAlone/glslang
GLSLANG=/path/to/glslang-src/build/StandAlone/glslang
cd vulkan
for q in q2 q1; do
  $GLSLANG -V --target-env vulkan1.3            gemv_$q.comp -o gemv_$q.spv
  $GLSLANG -V --target-env vulkan1.3 -DUSE_INT_DOT gemv_$q.comp -o gemv_${q}_dot.spv
done
g++ -O2 -o gemv_vk_bench gemv_vk_bench.cpp -lvulkan
./gemv_vk_bench 8192 131072 30      # run from this directory: it loads *.spv by relative path
```

The Vulkan shaders compile two ways. The default is the **portable**
path: plain integer arithmetic, any Vulkan 1.1 device, and it is what
runs on the ~30% of devices without the extension. `-DUSE_INT_DOT`
selects `dotPacked4x8AccSatEXT`, which needs
`VK_KHR_shader_integer_dot_product` (~70% of devices). Measured here:
+40% on Q1_0, +2% on Q2_0. Note the *packed* form is the accelerated
one -- Thor reports
`integerDotProduct4x8BitPackedSignedAccelerated = true` but
`integerDotProduct8BitSignedAccelerated = false`, so the `i8vec4` form is
not the fast path anywhere it matters.

## Next, in order

Rewritten after measuring; the earlier list had two items that are now
done and two that turned out to be wrong.

1. **Add `q1_0`/`q2_0` to ggml-vulkan's MMVQ type lists.** Highest-value
   item, and the risky part is now **done**: the GGUF-order extraction
   ggml must use is implemented, gated against the CPU reference
   (4.783e-06, identical to the repacked path) and measured 2-8% FASTER
   than repacking. So no repack and no format change are needed -- copy
   `unpack_gguf()` from `vulkan/gemv_q2.comp` verbatim. The toolchain gate
   is open (glslc 2026.4-dev builds `GL_EXT_integer_dot_product`), but
   `is_legacy_quant()` at `vulkan-shaders-gen.cpp:226` is
   `{q4_0,q4_1,q5_0,q5_1,q8_0}` and the MMVQ gate at line 710 admits that
   set plus mxfp4/k-quants/iq1_s/iq1_m -- so our two formats take the
   float path regardless. Needs: the type added to both gates, plus
   `repack()` and `mul_q8_1()` in `mul_mat_vecq_funcs.glsl`. The bias
   correction is already derived -- Q2_0 codes are `{0,1,2,3}` for
   `{-1,0,1,2}` so `mul_q8_1 = da*(q_sum*dsb.x - (1/div)*dsb.y)`; Q1_0 is
   `da*(2*q_sum*dsb.x - (1/div)*dsb.y)`. The catch is that both formats
   are `QUANT_K = 128` while the legacy quants the framework was built
   around are 32. Precedent: PR #16536 did this for the k-quants and got
   Q2_K +78%, Q4_K_S +131%.

2. **Q1_0 register LUT on CPU** -- the one measured kernel regression.
   Note the structural requirement discovered while scoping it: a T-MAC
   g=4 table is indexed by the weight nibble but *built from the
   activations*, so each of the 32 nibble positions in a 128-weight block
   needs its own table. It only works with the loop order transposed --
   activation group outer, rows inner -- so the table build amortizes
   across all N rows. A naive drop-in keeping the current row-outer order
   will be slower, not faster.

3. **Q2_0 SMMLA 4-row on CPU.** Measured issue rates on this core:
   `SMMLA` 1.97/cycle at 2x the MACs of `SDOT`, so it is a free 2x
   wherever M >= 2. Predicted 8.3x kernel, ~2.4x end to end.

4. **Vulkan barrier scoping** -- ~1.4us narrow vs ~3.9us wide per
   dispatch, ~2293 dispatches per token, a ~3.9 ms/token floor and the
   reason speculation loses on that backend. Note the entire narrow/wide
   gap is `VK_ACCESS_INDIRECT_COMMAND_READ_BIT` in `dstAccessMask`;
   naming the *stage* is free.

5. **Wire the kernels into a decode step** (GDN recurrent layers, GQA
   attention) rather than benchmarking them standalone. This is what
   turns a GB/s number into a tok/s number.

### Closed, with results

- ~~Run the CUDA ladder~~ -- done, 178.5 GB/s best (v5).
- ~~Get a toolchain that can build the Vulkan int-dot path~~ -- done,
  glslang 16.4 / shaderc 2026.4-dev from source. Worth +40% on Q1_0 in
  our own kernels.
- ~~Rebuild the fork with a newer glslc to lift its Vulkan tok/s~~ --
  done and **it did not help** (11.62 -> 11.80). Superseded by item 1;
  see `../results/vulkan-toolchain-rebuild.txt`.
- ~~128-bit weight loads~~ -- tried, **12% slower** (v6). Weight loads
  were never the constraint.
- ~~Row-blocking with shared-memory activations~~ -- tried, **slower**
  (v7). Row-blocking is wrong for this kernel, not merely mis-tuned.

Still untried from the original list: `cp.async` staging of activations,
an L2 persisting window for the activation block, and splitting K across
CTAs so the 20 SMs are not wave-quantized.
