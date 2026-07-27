# Engineering Wiki

Running ledger of what we learned, what we won, and -- most usefully --
what we got wrong and how we caught it. Every entry is tied to a commit
and to the measurement that settled it.

Read this before re-deriving anything. Several entries exist specifically
because a plausible belief survived for a while before measurement killed
it, and the same belief will look plausible again next time.

**House rule:** an entry only goes in the WINS table if a number on this
hardware backs it. Everything else lives under Open Questions.

---

## Wins ledger

| # | Win | Measured | Commit |
| :-- | :-- | :-- | :-- |
| W1 | First Bonsai-27B numbers on Jetson Thor, any variant | ternary 16.77 / 1-bit 19.03 tok/s native | `b91df7c` |
| W2 | DSpark on Thor never goes negative | 1.37-1.75x across all 4 cells | `b91df7c` |
| W3 | Trustworthy memory accounting on unified memory | validated to 0.00 MiB against the fork's own projection | `18fa5f4` |
| W4 | Drafter staging cap | -1035 MiB at ctx 4096, zero throughput loss, zero fallbacks | `80537fb` |
| W5 | CPU thread count | +1.73x ternary, +2.27x 1-bit, one flag | `80537fb` |
| W6 | Energy per token nearly halved by DSpark | 1.133 -> 0.603 mWh/tok ternary | `53057d8` |
| W7 | Shared low-bit format decodes identically on 3 backends | CUDA 1.2e-07, Vulkan 4.8e-06, NEON bit-exact | `124ed60` |
| W8 | CUDA GEMV ladder | 16.0 -> 177.2 GB/s, 11x | `124ed60` |
| W9 | Vulkan integer-dot path enabled | Q1_0 +40%, Q2_0 +2% | `c8f642f` |
| W10 | Killed a bad direction before building on it | `n_rs_seq=0` dominated by simply not speculating | `c7eb705` |
| W11 | CUDA GEMV is near-optimal | 212-220 GB/s vs 244.7 achievable = ~90% | `94df1be` |
| W12 | Shape-adaptive kernel selection rule | v7 <32MB (L2), v5 >32MB (streaming) | `3f8c4e2` |
| W13 | Four candidate optimizations tested and rejected on evidence | uint4 loads, row-blocking (streaming), split-K, `n_rs_seq=0` | various |

---

## The measurement lessons (retroactive)

This section is the most valuable thing in the repo. Five separate times,
a measurement looked right and was wrong. The pattern is always the same:
**a number that agrees with expectation gets believed without a
cross-check.**

### L1. Peak RSS looked exactly right and was a coincidence

`VmHWM` read 7158 MiB for a 7.17 GB model and 3952 MiB for a 3.80 GB
model. Both matched their GGUF to within a percent, which felt like
confirmation.

It was measuring page-cache traffic during load, not residency. `smaps`
showed only 322 MiB of the mapping actually resident. The tell was that
it **was not additive**: attaching an identical drafter moved it +22 MiB
in one configuration and +2767 MiB in another.

**Generalizable:** when a metric agrees with expectation, test it on a
case where you know the answer *changes*. Agreement on one point is not
validation.

Five methods failed before one worked (`nvidia-smi` reports "Not
Supported"; `MemTotal-MemAvailable` gave ~28 GiB for both a 7.2 GB and a
3.8 GB model; `cudaMemGetInfo` moved 450 MiB for a 7.2 GB model;
`--no-mmap`+RSS gave 796 MiB). What works is asking ggml itself at
`-lv 10` and summing its buffer log, which independently reproduces the
fork's own "projected to use 7812 MiB" to the megabyte.

### L2. A correctness gate that cried wolf

The trace gate reported **16/16 divergent** for the CPU backend. Real
figure: 3/16. The CPU sweep ran 64 tokens against a 128-token oracle and
the comparison counted the length difference as divergence.

**Generalizable:** a gate that can report total failure for a benign
configuration difference will not be believed when it reports a real one.
The Vulkan verdict was only trustworthy because this was caught first.
Fixed by reporting `PREFIX-MATCH` when one run is a strict prefix.

### L3. A losslessness check that passed by stopping too early

A 64-token check of DSpark-vs-native output matched exactly. The
divergence is at token 96.

**Generalizable:** negative results from bounded checks are only as
strong as the bound. State the bound.

### L4. Power measured while a build was running

First energy reading showed 33 W "idle". A compile was running. Also, the
first decode prompt was prose, and this model emits EOS immediately on
several prose prompts, giving zero decode time and a sentinel 1e6 tok/s.

**Generalizable:** for energy, assert the machine is quiet *and* assert
the workload actually ran. `measure_power.sh` now rejects any run
producing under 32 tokens.

### L6. A launch failure reported as a PASS

At K=17408 the harness printed `v1 smem activations ... OK`. It never
ran: v1 asks for `K*sizeof(float)` = 68 KB of dynamic shared memory,
above the 48 KB default, so the launch failed. `check()` then compared
whatever was still in the output buffer -- v0's result, from the previous
rung -- and it matched.

Worse, the failure then killed the entire throughput section silently,
because `time_kernel` ended in `CHECK(cudaGetLastError())` and `CHECK`
calls `exit(1)`. The run printed an empty "=== throughput ===" header and
exited 0.

Fixed three ways: `check()` calls `cudaGetLastError()` before trusting
the buffer, oversized-smem kernels are opted in via
`cudaFuncAttributeMaxDynamicSharedMemorySize` or explicitly skipped with
a printed reason, and a failed launch in timing reports `LAUNCH FAILED`
for that row instead of aborting the sweep.

**Generalizable:** comparing an output buffer proves nothing unless you
know the kernel wrote it. Any harness that reuses a buffer across
variants needs an explicit did-it-run check, or a failure will present as
agreement with the previous variant.

### L5. A validation failure that was the test's fault, not the kernel's

The CUDA ladder's v0/v1 rungs failed at 13.85% relative error. They were
not broken -- they read the original float activations while the CPU
reference was built from the quantized ones the dp4a rungs consume. The
naive kernels were computing the *more accurate* answer and failing for
it.

**Generalizable:** when the simplest implementation fails and the complex
ones pass, suspect the harness. Fixed by putting every rung on identical
inputs, which is also what makes the GB/s ladder a fair comparison.

---

## Hardware notes: Jetson Thor (sm_110)

Measured on this box, not from spec sheets.

| Property | Value | Why it matters |
| :-- | :-- | :-- |
| Compute capability | **11.0** | Sits between ggml's HOPPER (900) and BLACKWELL (1200) |
| SMs | **20** | Wave quantization is real; split-K matters |
| Max threads/SM | **1536** | NOT 2048 -- every H100 occupancy heuristic is wrong here |
| L2 | 32 MB, 24 MB persisting window | Large enough to pin a KV block |
| DRAM (GPU stream) | 240 GB/s w/ 128-bit loads, 158 w/ 32-bit | 16 bytes per thread per load is mandatory |
| DRAM (CPU cluster) | 126.4 GB/s at 14 threads | ~46% of the 273 GB/s SoC figure |
| CPU | 14x Neoverse V3AE, SVE VL=**128 bits**, no SME | SVE buys nothing at VL=128 |

**Thor is Blackwell-generation but CC 11.0**, so ggml gates it out of the
Blackwell FP4 tensor-core paths (`CC >= 1200`) and it runs Ampere/Turing
MMA. That is correct behaviour, but it means **Thor numbers are not
representative of Blackwell tensor cores.**

**Build `110a-real`, not `110-real`.** Baseline `sm_110` silently forgoes
tcgen05, `setmaxnreg`, FP4/E8M0 conversions and TMA multicast. Our
Milestone 0 numbers used `110-real`; correctness is unaffected because
ggml never emits those, but a custom engine wants the `a` variant.

**No CMake patch is needed for the fork to build on sm_110.** The
`12X -> 12Xa` rewrite does not cover `11X` and does not need to, because
every Blackwell path is gated on `CC >= 1200`.

**Sub-byte tensor cores are emulated here, and will mislead you.** They
still compile:

| Instruction | T-MAC/s | vs INT8 |
| :-- | --: | --: |
| `mma.s8` | 56.5 | 1.00x |
| `mma.b1.and.popc` | 11.1 | **0.20x** |
| `__dp4a` (CUDA core) | 7.07 | 0.13x |
| `mma.s4` | 1.83 | **0.03x** |

On sm_75/sm_80 `b1` ran 4-8x *faster* than INT8. Here it is 5x slower.
**This kills the popcount/XNOR two-binary-plane family of ternary
kernels on this hardware.**

`cudaMallocManaged` is not GPU-cached on Thor under CUDA 13.0 (233 vs 211
GB/s), so `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is the wrong path.

---

## Kernel learnings

### K1. The dp4a step is the entire cliff

CUDA ladder, 256 MiB streaming shape: naive 16.0 -> dp4a 167.9 GB/s. The
bit-plane repack adds 2% on top, shared-memory activations another 3.5%.
**Everything else is rounding compared to using the integer dot
instruction at all.**

The same holds on Vulkan (portable 71.2 -> int-dot 99.4 GB/s on Q1_0) and
on CPU (per-element extract 0.03 -> whole-vector+SDOT 2.78 Gw/s on Q1_0,
~90x). Three backends, same conclusion.

### K2. Row-blocking made it slower, against prediction

Predicted: 8 rows/warp closes the last 25%. Measured: 171 -> 135 -> 117
-> 123 GB/s as ROWS goes 1/2/4/8.

Diagnosis: each 4-byte weight word needs four `A4[]` words plus one
`ASUM16[]` -- **20 bytes of activation reads per 4 bytes of weights.**
Those are not DRAM traffic (activations are small and stay resident) but
they are load-*issue* traffic, and issue slots are what a batch-1 GEMV is
short of. The loop is issue-bound, not latency-bound, so extra
independent loads cannot hide anything; they only add register pressure
and, for ROWS>1, an uncoalesced per-row scale gather.

Testing that diagnosis directly is what produced v5 (activations in
shared memory), the only variant that beat v3.

### K2b. Three ways of adding parallelism, all slower -- a design rule

After K2, two more attempts were made to widen or deepen the loop. Both
lost, and together they settle the question:

| Attempt | Idea | GB/s |
| :-- | :-- | --: |
| **v5** | 1 row/warp, activations in shared memory | **178.5** |
| v6 | 128-bit `uint4` weight loads | 158.1 |
| v7 | rows/warp *with* shared-memory activations | 168.1 / 138.0 / 133.6 |
| v4 | rows/warp with global activations | 142.5 / 117.8 / 124.4 |

**v6 is the one that pins the diagnosis.** Wide loads cut *weight* load
instructions 4:1 and still cost 12%. That only makes sense if weight
loads were never the constraint -- and they are not: 20 of every 21 loads
in the inner loop are activations. Widening the 1 does nothing and costs
register pressure.

**v7 was the fair retest of row-blocking.** v4's version amortized
*global* activation reads, so it could have been losing for that reason
alone. With activations already in shared memory it still loses
monotonically. Row-blocking is not defeated by the cost of the read it
shares; it is defeated by occupancy and per-row scale gathers.

**Design rule for this kernel: it is issue-bound and occupancy-sensitive,
not latency-bound.** The one lever that worked reduced the *number of
instructions per weight byte*. Every lever that added parallelism to hide
latency made it worse. Standard GPU intuition ("more in flight is
better") is backwards here.

**...but only in the streaming regime. See K2c -- this rule was
over-generalized from a single shape.**

### K2c. The rule above is shape-dependent, and we had only tested one shape

Every conclusion in K2/K2b was drawn at N=131072 K=8192 (256 MiB). Across
shapes, the winner flips at the L2 boundary:

| weights MiB | v5 (1 row) | v7 (2 rows) | winner |
| --: | --: | --: | :-- |
| 5 | 212.4 | **254.8** | v7 |
| 20 | 237.9 | **308.6** | v7 |
| 40 | **212.6** | 204.5 | v5 |
| 160 | **220.5** | 211.3 | v5 |
| 320 | **177.6** | 169.9 | v5 |

The flip is between 20 and 40 MiB -- exactly Thor's 32 MB L2. On the real
Bonsai down-projection shape (K=17408 N=5120, 21 MiB) v7 reaches **387.8
GB/s**, well past the 273 GB/s DRAM spec, which is the tell that L2 is
serving it.

Physical reading: **rows-per-warp buys ACTIVATION reuse, which is only
worth having once WEIGHT bandwidth has stopped being the bottleneck.**
From DRAM, extra rows cost registers and buy nothing. From L2 -- ~4x
faster here -- bandwidth is no longer the limit and the reuse pays.

**Which regime is real: the streaming one.** Every Bonsai matrix (6-21
MiB) fits L2, but a decode step reads all 64 layers (~3.8 GB) and touches
each matrix exactly once, so the intervening traffic evicts it long
before the next token. The benchmark's repeated iterations manufacture a
locality decode does not have. **v5 remains the kernel to ship**; the
L2-resident numbers describe a microbenchmark-only regime.

**Generalizable:** a kernel conclusion drawn at one problem size is a
conclusion about that size. Sweep the size until you cross a cache
boundary, or you are characterizing your benchmark rather than your
kernel.

**And the kernel is nearly done.** Measured achievable streaming read is
**244.7 GB/s** (float4) / 161.4 (scalar uint32). v5 reaches 212-220 GB/s
in the streaming regime, i.e. **~90% of achievable bandwidth**. Two
consequences: there is very little left to win here, and v6's failure is
explained -- v5's 32-bit loads are already coalesced across the warp into
wide transactions, so explicit `uint4` loads had nothing to add. The
161.4 figure characterizes a non-coalescing pattern, not a ceiling for
well-formed 32-bit access.

This also gives the first plausible answer to Q2: v5 alone hits 237.9
GB/s at 20 MiB and v7 hits 308.6, so a reported 229 GB/s is entirely
consistent with a measurement that had L2 reuse. Hypothesis, not
confirmation -- but it predicts something checkable, namely that any such
figure must state its weight footprint relative to L2 to be
interpretable.

### K3. The biased-encoding identity is the one thing that ports everywhere

Store `u = t+1 in {0,1,2}` and use `sum w*x = sum u*x - sum x`. Identical
on CUDA `__dp4a`, Vulkan `dotPacked4x8AccSatEXT`, and ARM `SDOT`/`SMMLA`.
Worth +17% on CUDA alone, and it removes saturating-subtract instructions
(`__vsubss4` has no single SASS instruction) on every backend.

### K4. LUT is a CPU-only win, and the split is sharp

- **CPU:** T-MAC-style register LUT wins decisively at 1-bit (4.4x),
  because ARM `TBL` does a 16-entry lookup in one instruction on a
  register.
- **GPU:** every direct batch-1 measurement in the literature shows LUT
  *losing* to arithmetic dequant. Microsoft's own GPU library is
  dequant-based, and their GPU LUT paper proposes new silicon.

"Memory-bound" does not mean everything else is free -- it means keeping
loads in flight, and a shared-memory LUT costs registers and serializes
on bank conflicts.

### K5. Keep 2 bits/weight; base-3 is a trap on GPU

Base-3 (5 trits/8 bits) saves 0.375 bpw but costs ~4 ALU ops per weight
against ~0.5 for the 2-bit LOP3 path. K1 shows the total ALU budget is
about 4 op-equivalents per weight. Since the weights are shared across
all three backends, this settles it for all three.

### K6. The Vulkan overload trap

```glsl
uint dotPacked4x8AccSatEXT(uint, uint, uint)
int  dotPacked4x8AccSatEXT(uint, int,  int)   // the one you want
```

Activations are packed **signed** int8, codes are unsigned. Passing both
as `uint` selects the all-unsigned overload and treats activations as
0..255. It failed to compile against an `int` accumulator so it surfaced
loudly -- but with a `uint` accumulator it is a silent wrong-answer bug.

---

## Speculation learnings

### S1. Breakeven is `alpha > c`, not an acceptance threshold

Leviathan et al.: improvement requires acceptance `alpha` to exceed the
**cost ratio** `c = draft-step / target-step`.

Our measurement is a clean demonstration: on CPU, 1-bit code **loses** at
56.6% acceptance while ternary code **wins** at 57.7%. Any heuristic
keyed on acceptance alone gets this backwards. The control rule must
measure realised `t_draft` / `t_verify`.

Add a per-round overhead term `o`: on CUDA `o ~= 0.03-0.12`; **on Vulkan
`o` is plausibly > 1**, which forces speedup < 1 regardless of
acceptance. That is the 0.39-0.59x.

### S2. Speculation pays on exactly one of three backends

Same box, same weights, same drafter: CUDA 1.37-1.75x, CPU 0.79-1.07x,
Vulkan 0.39-0.59x.

**Acceptance is backend-independent** (Vulkan within ~1 point of CUDA in
every cell), which is the diagnostic: the drafter predicts equally well,
so the loss is entirely execution structure. Degraded acceptance would
have implied a drafter problem.

### S3. The drafter's cost is buffers, not weights

Server estimates 1802 MiB for the drafter. Real cost at ctx 16384 is
**+8558 MiB**. Dominant term is the drafter's compute buffers (2432 dev +
2752 host), because the fork sizes the draft context's batch to the
*full context* for staging. ~1.3 GB more lands on the **target**, whose
recurrent-state buffer grows 149.62 -> 748.12 MiB for block verification.

Capping the staging batch recovers ~1 GB at ctx 4096 with zero
throughput loss and zero fallbacks. The cap is **sticky** -- a skipped
round does not advance the drafter's cache -- so it bounds the ingestible
*prompt*, not the context window.

### S4. Zero-memory speculation is free but is not a speedup here

N-gram self-speculation costs **byte-identical memory to native** (4173
MiB, every variant), confirming zero overhead. Throughput is also
identical (~1.00x), because n-gram pays on *input-grounded* generation
and both our suites are open-ended. Right free fallback; not a drafter
replacement.

### S6. `n_rs_seq = 0` is strictly dominated -- a killed direction

This was listed in our own findings as *"the highest-leverage remaining
experiment, requires no new code"* (~1.3 GB). Measured, it is a trap.

| ctx | staging cap | `rs_seq0` | total MiB | code tok/s |
| --: | --: | --: | --: | --: |
| 4096 | none | no | 8355 | 32.71 |
| 4096 | 256 | no | **7320** | **32.74** |
| 4096 | 256 | yes | 5996 | 13.55 |
| 2048 | 256 | yes | 5838 | 13.50 |

It saves exactly the predicted 1324 MiB, and costs **59% of throughput**
plus an acceptance drop from 61.6% to 41.8%.

The comparison that kills it is against not speculating at all:

```
1-bit native,     ctx 4096:  4173 MiB   18.61 tok/s
1-bit dspark+rs0, ctx 4096:  5996 MiB   13.55 tok/s
```

**Native is smaller AND faster.** So no memory budget exists at which
this knob is correct: any budget that cannot afford normal DSpark is
better served by turning speculation off.

**Why acceptance moved at all** -- this is the interesting part, because
a memory knob should not change an algorithmic property. With
`n_rs_seq = 0`, `use_ckpt_tgt` is true for every partial acceptance, and
that branch ends in `continue`, which **bypasses
`common_speculative_accept()`**. The drafter is never told what was
accepted, and the rollback goes to the *checkpoint* position rather than
the accepted position -- discarding accepted tokens along with rejected
ones. The cost is structural, not bandwidth; a faster memcpy would not
fix it.

**Generalizable:** "trades memory for a cheap memcpy on unified memory"
was a reasonable-sounding model derived from reading the buffer
arithmetic. It was wrong because the fallback path differs from the fast
path in *control flow*, not just in where bytes live. Read the whole
branch, not just the allocation it changes.

### S5. DSpark is not token-identical to native at temperature 0

Speculative decoding is supposed to be lossless. Measured on a freshly
started server's first request, ternary Q2_0, `code/cuda-reduce`:

```
native: for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
dspark: for (int offset = warpSize / 2;   offset > 0; offset /= 2)
```

Reproducible, same character offset, 5/16 ternary and 6/16 1-bit trace
pairs. Not a capture artifact: native output is byte-identical with and
without `n_probs`, and both modes are individually deterministic.

Consequence: **native output cannot be the oracle for a DSpark run**, so
all gates compare like mode against like mode. Unresolved whether this is
Thor-specific numerics or general to the fork -- needs a second device.

---

## Toolchain learnings

### T1. Device capability != toolchain capability

Thor's driver reports
`integerDotProduct4x8BitPackedSignedAccelerated = true`. Neither system
compiler could emit it:

| Compiler | glslang | supports `GL_EXT_integer_dot_product` |
| :-- | :-- | :-- |
| `glslc` 2023.8 (Ubuntu 24.04) | 14.0.0 | no |
| `glslang-tools` 15.1.0 (apt) | 15.1.0 | no |
| built from source | 16.4.0 | **yes** |

Not a `--target-env` problem; 1.1/1.2/1.3 fail identically. Cost: a
source build with `ENABLE_OPT=0` (drops the SPIRV-Tools dependency).

**And it reaches past our kernels.** `ggml-vulkan` runs the same feature
test and, failing it, compiles the **entire Vulkan backend with no
integer-dot path**. Every llama.cpp Vulkan number in this repo was
measured with it disabled.

### T1b. ...but fixing the toolchain gate alone changed nothing (correction)

We predicted in `c8f642f` that rebuilding the fork with a newer glslc
"should lift them independently of anything we write." **Measured, it
does not:** ternary 11.62 -> 11.80, 1-bit 15.99 -> 16.02 tok/s, both
within noise.

The rebuild worked, and enabled *five* previously-disabled extensions
(coopmat, coopmat2, decode_vector, bfloat16, integer_dot). It changed
nothing because there is a **second, independent gate** in the shader
generator:

```cpp
bool is_legacy_quant(const std::string& t) {   // vulkan-shaders-gen.cpp:226
    return t=="q4_0"||t=="q4_1"||t=="q5_0"||t=="q5_1"||t=="q8_0";
}
```

Neither `q1_0` nor `q2_0` is in that list, nor in the MMVQ gate at line
710, so Bonsai's formats take the float path regardless of compiler
support -- while IQ1_S, a harder format, gets integer dot.

**Generalizable:** a capability needs every gate on its path open.
Finding and opening one gate proves nothing about the others, and the
satisfying feeling of having fixed "the" blocker is exactly when to go
looking for the next one.

### T2. Jetson breaks the obvious tools

`nvidia-smi` reports "Not Supported" for memory. `tegrastats` is the
supported path and also gives per-rail power (`VIN` = whole board).
`cudaDeviceProp::clockRate` was removed in CUDA 13.0 and returns garbage
on Tegra anyway -- never compute a roofline from it.

### T3. The GPU can wedge across suspend/resume

After a suspend/resume, 201 processes (`rustdesk`, not ours) were stuck
in `D` state inside the NVIDIA UVM fault handler with `nvidia_uvm`
refcount at 1272; any new CUDA or Vulkan context hung, while
`nvidia-smi` still reported the device healthy at 0% util. Only a reboot
cleared it. **A healthy-looking `nvidia-smi` does not mean the GPU is
usable.**

---

### T2b. "The framework only handles 32-quant blocks" was wrong

We recorded `QUANT_K = 128` as the blocker for putting `q1_0`/`q2_0` on
ggml-vulkan's integer-dot path. It is not: IQ1_S and Q2_K are both 256
quants and both supported, via a `DATA_A_QUANT_K` path that takes `ib` as
a **virtual block index in units of 32 quants**. Block size is a division,
not an obstacle.

The real trap is subtler and would have shipped silently wrong numbers.
Q2_K extracts codes with `(word >> shift) & 0x03030303`. On a Q2_0 word
that selects the same bit-field from each of four bytes, which is weights
strided by 4 -- *not* four consecutive weights -- so every
`dotPacked4x8EXT` would pair codes with the wrong activations. It
compiles, runs, and produces plausible output. Only a numerical gate
catches it.

This is the same distinction our own CUDA ladder drew between v2 (GGUF
order, gather activations) and v3 (repacked, one aligned load). Vulkan
gets the weights in GGUF order and cannot repack, so it must use the v2
shape.

**Generalizable:** when reusing a sibling type's extraction code, check
what its bit layout means, not just that the types match. Two 2-bit
formats can need opposite extraction.

Full contract in `docs/INTEGRATION_NOTES.md`.

## Scope corrections

### C1. The smaller Bonsai models are a different architecture

8B/4B/1.7B exist and are small (0.23-2.15 GiB), but they are **plain
dense Qwen3**: no GDN, no recurrent state, no DSpark drafter, no vision,
32-64K context vs 262K. They are a simpler engine path and give **zero**
GDN coverage, so they cannot be used to iterate on the 27B's kernels.

### C2. Neither DSpark configuration fits an 8 GB Orin as originally planned

Measured totals contradicted the assumption that 1-bit + drafter fits.
Post-cap the best is 7162 MiB, still tight on a device holding an OS.
Only 1-bit native (4041-4437 MiB) fits comfortably.

### C3. Ratios flatter slow hardware

Our DSpark speedups (1.48-1.75x) exceed H100's published 1.34x. That is
**not** Thor doing better -- raw throughput is ~6x lower. H100 at batch 1
is launch/sync-latency-bound, so there is less per-step cost for
speculation to amortize. Always lead with absolute tok/s.

---

## Open questions

| # | Question | Why it matters |
| :-- | :-- | :-- |
| Q1 | Is the temp-0 DSpark divergence Thor-specific or general? | Needs a second device. If general, it is a finding about the reference implementation. |
| Q2 | Why can't we reproduce 229 GB/s? We get 178.5. | Ladder shape matches prediction. The three obvious levers (wide loads, row-blocking with global and with shared activations) are now all tested and all lose, so the gap is not any of them. |
| Q3 | Does rebuilding the fork with glslang >= 16 lift its Vulkan tok/s? | Directly testable; the backend currently has no integer-dot path. |
| ~~Q2~~ | ~~Why can't we reproduce 229 GB/s?~~ | **LIKELY ANSWERED: L2 residency. v5 hits 237.9 and v7 308.6 GB/s on L2-resident shapes; see K2c.** |
| ~~Q4~~ | ~~What does `n_rs_seq = 0` cost in throughput?~~ | **ANSWERED: 59% of throughput. Strictly dominated -- see S6.** |
| Q5 | Can factor buffering replace the 748 MiB rollback ring? | Literature reports 78% DRAM reduction, numerically exact. |
