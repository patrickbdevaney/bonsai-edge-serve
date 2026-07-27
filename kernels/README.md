# Decode kernels

Batch-1 low-bit GEMV for the Bonsai family, across CUDA, Vulkan and ARM
NEON. This is the core of Phase 2: at batch 1 the model streams every
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
| CUDA | **validated, all 7 rungs** | **measured, below** |
| Vulkan | **validated, both shaders** | **measured, below** |

All three backends now decode the same shared packed format to the same
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
| v1 + activations in smem | 12.5 | 5% |
| v2 dp4a, GGUF layout | 167.9 | 61% |
| v3 dp4a, bit-plane interleaved | 171.2 | 63% |
| **v5 dp4a + activations in smem** | **177.2** | **65%** |
| v4 2 rows/warp | 134.5 | 49% |
| v4 4 rows/warp | 117.0 | 43% |
| v4 8 rows/warp | 122.5 | 45% |
| v4 Q1_0, 8 rows/warp | 100.1 | 37% |

**11x from naive to best.** The dp4a step is the whole cliff (16 -> 168);
the bit-plane repack adds 2%, and staging activations in shared memory
adds another 3.5%.

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

## Building

```bash
# CPU
cc -O3 -march=armv8.2-a+dotprod -fopenmp \
   -o cpu/gemv_neon_bench cpu/gemv_neon_bench.c -lm
./cpu/gemv_neon_bench 4096 32768 3

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

1. **Q1_0 register LUT on CPU** -- the one measured regression above.
2. **Close the CUDA gap from 177 GB/s toward the reported 229.** The
   ladder is built, validated and instrumented, and the diagnosis is
   specific: the loop is issue-bound on activation reads, not
   latency-bound. Things not yet tried -- wider (128-bit) weight loads
   per lane, `cp.async` staging of activations, an L2 persisting window
   for the activation block, and splitting K across CTAs so the 20 SMs
   are not wave-quantized at this shape.
3. **Rebuild the llama.cpp fork with glslang >= 16** so ggml-vulkan's own
   integer-dot path is enabled. Our kernels gained 40% on Q1_0 from it;
   the backend's shaders should gain independently, and the current
   Vulkan tok/s figures were all measured with it compiled out.
4. **Then attack Vulkan barrier scoping** -- ~1.4us narrow vs ~3.9us wide
   per dispatch, ~2293 dispatches per token, a ~3.9 ms/token floor and
   the reason speculation loses on that backend.
5. **Wire the kernels into a decode step** (GDN recurrent layers, GQA
   attention) rather than benchmarking them standalone.
