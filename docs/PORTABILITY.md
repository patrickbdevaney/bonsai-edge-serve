# Portability: what each kernel needs, and what happens without it

Everything in this repo was developed on one box (Jetson Thor, sm_110,
Neoverse V3AE). The kernels are **not** Thor-specific, but "not
Thor-specific" is a claim that needs stating precisely, because the failure
mode for a wrongly-gated kernel is silence: the model loads, the output is
correct, and the optimisation simply never runs. That has already happened
three times in this repo (WIKI L8, L14, L15).

So for each path: the feature it requires, what a device *without* that
feature gets instead, and whether that fallback has actually been exercised
here or is merely believed to work.

**Tested here** means the code path ran on this box and was validated.
**Gated, untested** means the gate is believed correct but no device was
available to run the other side of it. The distinction matters and is not
smoothed over.

---

## CPU — ARM (`patches/0004`)

| Feature | Gate | Path taken | Status |
| :-- | :-- | :-- | :-- |
| NEON + i8mm (Armv8.6+) | `ggml_cpu_has_dotprod()` selects repack; `__ARM_FEATURE_MATMUL_INT8` selects the gemm | q2_0 repack, `vmmlaq_s32` gemm | **tested here** (Neoverse V3AE) |
| NEON + DOTPROD only (Armv8.2+) | same selector; gemm falls to the `vdotq_s32` path | q2_0 repack, DOTPROD gemm | **tested here** via `BONSAI_Q2_0_NO_I8MM=1` |
| NEON without DOTPROD | selector returns `nullptr` | no repack; ggml's per-row NEON `vec_dot` | **path exercised here** via `BONSAI_NO_ARM_REPACK=1`; greedy output character-identical |

The DOTPROD fallback is the reason the selector gates on `dotprod` and not
`matmul_int8`. **i8mm is Armv8.6; Cortex-A76/A78 are Armv8.2 and do not have
it — which includes Jetson Orin (A78AE), a device this repo explicitly
targets.** Gating on i8mm would have meant Orin silently getting the
non-repack path and none of the win.

Measured on this box, ternary Q2_0, CPU-only, `-t 12`:

| gemm path | pp64 | tg16 | vs no repack |
| :-- | --: | --: | --: |
| no repack (before this work) | 4.04 | 3.28 | — |
| DOTPROD (Armv8.2 class, e.g. Orin) | **8.94** | 5.03 | **2.21x** / 1.53x |
| i8mm (Armv8.6+, e.g. Thor) | **16.05** | 5.06 | **3.97x** / 1.54x |

i8mm is worth another 1.8x on prefill over DOTPROD, but DOTPROD is worth
2.21x over nothing, which is the point. Decode is identical across the two
(5.03 vs 5.06) because decode uses the gemv, which only ever needed DOTPROD.

Both gemm paths are validated against ggml's own scalar reference over the
same 100-shape sweep (`kernels/cpu-neon-arm/run_q2_0_repack_test.sh`, and
with `BONSAI_Q2_0_NO_I8MM=1`).

Note the DOTPROD number is measured by *forcing* the fallback on hardware
that has i8mm. That validates the code path and prices the instruction-set
difference; it does not measure an actual A78. Cache sizes, core counts and
memory bandwidth all differ, so treat 8.94 as "this path works and is worth
~2.2x", not as an Orin prediction.

## CPU — x86

| Feature | Path | Status |
| :-- | :-- | :-- |
| AVX512 + AVX512-VNNI | q2_0/q1_0 repack (upstream fork code, not ours) | **untested by us** — no x86 host |
| AVX2 only | no repack; scalar/AVX2 `vec_dot` | gap |

We have not written or validated any x86 kernel and do not claim one. The
fork's AVX512-VNNI repack path for these types exists and we have never
executed it. An AVX2 path does not exist for q2_0 at all, which is the same
shape of gap the ARM DOTPROD fallback just closed — worth doing, but not
worth writing blind. It needs a machine.

## Vulkan (`patches/0003`)

| Feature | Gate | Path taken | Status |
| :-- | :-- | :-- | :-- |
| `VK_KHR_shader_integer_dot_product` + glslc with `GL_EXT_integer_dot_product` | runtime extension check + `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` | q1_0/q2_0 MMVQ (**+44% / +31% decode**) | **tested here** — 56/56 op shapes |
| no integer dot product | gate fails | dequant MMV — correct, slower | **path exercised here** via `GGML_VK_DISABLE_MMVQ=1`; 56/56 op shapes, 0 fail |
| KHR_coopmat present | shader-gen `!coopmat && !coopmat2` | MMQ **not** built; dequant + coopmat prefill | **tested here** |
| no coopmat | non-coopmat branch | scalar int-dot MMQ | **path exercised here** via `GGML_VK_FORCE_MMQ_COOPMAT=1` — see below |

Two things worth flagging for anyone porting this:

**The MMQ path is deliberately off on this class of device, and that is a
measured decision, not an oversight.** Forcing it on Thor gives 0.36x/0.45x
of the dequant+coopmat path, because these are scalar `dotPacked4x8AccSat`
shaders that cannot reach the tensor cores. On a device with integer dot
product but *without* cooperative matrix, MMQ is the right path.

**Those shaders are now validated, which they were not when first shipped.**
The original commit measured MMQ's throughput and never checked its output --
meaning correctness-unverified shaders were being shipped for other people's
hardware, since a non-coopmat device selects them automatically. Forcing the
path here (`GGML_VK_FORCE_MMQ_COOPMAT=1`):

* 38 q2_0/q1_0 `MUL_MAT` shapes pass, **0 fail**
* greedy model output is **character-identical** to the default
  dequant+coopmat path

(62-65 `type_a=f32` failures appear in these runs, but they are present in
the *baseline* too -- a pre-existing permuted-f32 matmul issue in the fork,
not related to this work. Counting them per-type is what separates the two.)

What remains untested is MMQ's **performance** on a genuine non-coopmat
device. Correctness is no longer in that category. See
`results/vulkan-mmq-attempt.txt`.

**The MMVQ change moves q1_0/q2_0 into the 32-quant size class**, alongside
IQ1_S/IQ1_M. That is a shared code path, so it is not q1_0/q2_0-specific
behaviour, but it does mean a device where the wider load hurts would see it
on these types. Not observed here; unmeasured elsewhere.

The `packed16` block types use `float16_t` members, so 16-bit storage is
required for the MMVQ path. Ubiquitous on Vulkan 1.3 hardware, untested
without.

## CUDA

Phase 3 in the README is honest about this: the CUDA targets are
**compile-verified for multiple architectures, and only sm_110 has been
run.** Nothing here claims otherwise. The q1_0/q2_0 CUDA support is the
fork's, not ours; our CUDA work is the standalone GEMV ladder in
`kernels/cuda/`, which is a study, not a shipped kernel.

## Speculative decoding is the one thing that does NOT generalise

Worth stating plainly, because it is the repo's headline feature. The round
cost `R` — decode steps consumed per speculative round — is a property of
the (backend, target, drafter) triple, and it decides everything:

| | ternary `alpha*` | 1-bit `alpha*` |
| :-- | --: | --: |
| CUDA | 0.296 | 0.295 |
| CPU | 0.97 | **1.33** |
| Vulkan | **1.54** | **1.96** |

`alpha*` above 1.0 means no acceptance rate can make speculation profitable.
That holds on four of the six configurations measured. **So a port of this
repo to new hardware must re-measure R before enabling speculation**, and
should not assume the CUDA result transfers. `results/spec-round-cost.txt`
has the method; it needs one native run and one speculative run.

The server does this for you to the extent it can: `--speculate auto`
consults a measured table and refuses where speculation is known to lose,
and `/v1/policy` prints the constant and its consequence rather than just
asserting a default.

## Exercising every path on your own hardware

The point of the hooks below is that a porter does not have to trust any of
the tables above. Each forces a code path that the hardware would otherwise
never select, so every branch can be validated on whatever device you have:

| Hook | Forces | Use |
| :-- | :-- | :-- |
| `BONSAI_Q2_0_NO_I8MM=1` | q2_0 gemm onto DOTPROD | validate/cost the Armv8.2 path on Armv8.6 hardware |
| `BONSAI_NO_ARM_REPACK=1` | selector returns `nullptr` | the no-repack fallback, and the A/B control |
| `BONSAI_REPACK_DEBUG=1` | prints every type offered to the repack selector | confirm the path is reached *at all* |
| `GGML_VK_FORCE_MMQ_COOPMAT=1` | MMQ registration on a coopmat device | validate the non-coopmat shaders |
| `GGML_VK_DISABLE_MMVQ=1` | dequant MMV | the no-integer-dot fallback |
| `GGML_VK_FORCE_MMVQ=1` | MMVQ below its k threshold | required for op tests to reach MMVQ at all |
| `BONSAI_FORCE_RS_SEQ0=1` | recurrent rollback ring to 0 | check the server refuses to speculate |

Two of these were added specifically because a gated path could not otherwise
be reached here, and both immediately paid for themselves: the MMQ hook
revealed that shaders shipped for other hardware had never been checked for
correctness, and the repack-debug hook revealed that an entire A/B had been
measuring nothing (WIKI L15).

## Summary of real gaps

Now genuinely untested:

1. **x86 AVX2 q2_0** — no path exists. Needs an x86 host.
2. **x86 AVX512-VNNI q2_0/q1_0** — exists upstream, never executed by us.
3. **Any CUDA arch other than sm_110** — compiles, never run.
4. **Performance** (not correctness) of Vulkan MMQ and the ARM DOTPROD gemm
   on hardware that actually needs them — the code paths are validated here,
   but a real A78 or a non-coopmat GPU differs in cache, bandwidth and core
   count.
5. **R for speculation on any device but this one** — unmeasured by
   construction, and it must be re-measured before enabling speculation.

Closed since the first version of this document:

* ~~Vulkan MMQ on a non-coopmat device~~ — correctness now validated here
  (38 shapes, 0 fail; output character-identical).
* ~~ARM without DOTPROD~~ — fallback exercised, output character-identical.
* ~~Vulkan without integer dot product~~ — exercised, 56/56 op shapes.

None of the remaining items are believed broken. All are untested, which is a
different statement, and the reason they are listed separately.
