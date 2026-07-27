# Research Findings

Answers to `RESEARCH_PROMPT.md`, from a five-agent deep research pass
(literature + code reading + **new measurements taken on this Thor**).

Provenance is marked throughout: **[M]** measured on this box during the
research pass, **[L]** read from the local `prism-llama.cpp` checkout,
otherwise cited literature. Uncertain claims are flagged.

---

## 0. The five findings that change the plan

1. **We are at ~64% of an achievable roofline, and the gap is kernel
   quality, not hardware.** The research pass reported a 2-bit ternary
   GEMV reaching **229 GB/s = 84% of the 273 GB/s spec**. **We
   subsequently built that ladder ourselves and could not reproduce it:
   our best is 177 GB/s (65%)**, an 11x span from naive rather than 33x,
   and row-blocking -- the change credited with closing the last 25% --
   made it *slower*. See `kernels/README.md` for the measured ladder.
   Corrected implication: ~26 tok/s ternary against the fork's 16.77, so
   roughly **1.55x of headroom, not ~2x**. The rest of this section is
   left as the research pass reported it, with our corrections marked.
2. **The 8.5 GB DSpark overhead is three lines of worst-case sizing, and
   capping one of them is free.** [M, ours] Capping the drafter staging
   batch saves ~1 GB at ctx 4096 with **zero throughput loss, zero
   fallbacks, and identical acceptance**.
3. **Vulkan's loss is the pipeline barrier, not submits or fences.** [M]
   A barrier-separated dispatch costs ~1.4 us narrow / ~3.9 us wide,
   against ~0.17 us with no barrier. A token graph is ~2293 real
   dispatches, so barriers alone are a **~3.9 ms/token floor** regardless
   of model size. Submits and host round-trips together are <5% of it.
4. **Both our low-bit formats have no integer-dot path on Vulkan at
   all.** [L] `is_legacy_quant()` gates the MMQ/MMVQ dp4a lists and
   includes neither `q1_0` nor `q2_0`, so they run the float dequant
   path while IQ1_S -- a harder format -- gets dp4a. Precedent for the
   fix is +78-131%.
5. **The smaller Bonsai models are a different architecture.** 8B/4B/1.7B
   exist and are tiny (0.23-2.15 GiB), but they are plain dense Qwen3:
   **no GDN, no recurrent state, no DSpark drafter, no vision**, 32-64K
   context. They are a simpler engine path and give **zero** GDN
   coverage.

---

## 1. RQ1 -- Fitting 1-bit + speculation in 8 GB

### 1.1 The staging cap works, measured

Our own experiment (`bench/staging_sweep.py`, patch in `patches/`),
1-bit, CUDA, 128 tokens, 16 prompts:

| ctx | cap | total MiB | code tok/s | prose tok/s | acceptance | fallbacks |
| --: | --: | --: | --: | --: | --: | --: |
| 2048 | 256 | **7162** | 31.89 | 25.93 | 61.6% | 0 |
| 2048 | 512 | 7229 | 33.95 | 26.46 | 61.6% | 0 |
| 2048 | none | 7631 | 33.68 | 26.64 | 61.6% | 0 |
| 4096 | 256 | **7320** | 33.23 | 26.81 | 61.6% | 0 |
| 4096 | 512 | 7389 | 33.81 | 26.66 | 61.6% | 0 |
| 4096 | 1024 | 7527 | 34.23 | 26.72 | 61.6% | 0 |
| 4096 | none | 8355 | 33.87 | 26.78 | 61.6% | 0 |

**~1 GB saved at ctx 4096 for nothing.** Acceptance is identical in every
row, confirming the cap does not touch draft quality. Zero fallbacks
because our prompts are short; the cap bounds the *prompt* the drafter
can ingest in one round, and the fallback is sticky (a skipped round does
not advance the drafter's cache), so a cap below the prompt length
disables speculation for that request. Size the cap to the longest
expected prompt, not to the context window.

This is a real improvement but **not sufficient alone**: 7162 MiB still
leaves little room on an 8 GB device that must hold an OS.

### 1.2 The root cause is `n_ubatch`, and there is a second knob we have not pulled

[L] `tools/server/server-context.cpp:1046-1055` forces
`n_batch = n_ctx + block_size` and then `n_ubatch = n_batch`. **The
compute buffer is sized by `graph_reserve(n_ubatch, ...)`, so `n_ubatch`
is the term that matters.** The 16K sizing exists solely to survive the
first round after `begin()`, where `ctx_len == whole prompt`; steady-state
rounds need only `n_accepted + 1` rows [L, `common/speculative.cpp:1147`].

The principled fix is **chunked drafter prefill** -- ingest the prompt in
`n_ubatch`-sized context-only decodes, then one block-only decode --
which is exactly what TensorRT-LLM and vLLM do for draft models. One
correctness caveat to test rather than assume: [L]
`common/speculative.cpp:997` sets `llama_set_causal_attn(ctx_dft, false)`,
so chunking breaks intra-chunk bidirectionality. But the implementation
*already* accepts that approximation on every steady-state round, so this
is the same approximation class applied once more, not a new one.

**The second, unpulled knob:** [L] `common/common.h:379-383` sets
`n_rs_seq = draft.n_max` (= 4) for MTP/DSPARK, and
`llama-memory-recurrent.cpp:100` does `n_rows = mem_size * (1 + n_rs_seq)`
-- exactly our measured 149.62 -> 748.12 MiB (5.000x). The same factor is
replicated in the compute graph (`llama-graph.cpp:2850`, `n_g = n_rs_seq + 1`),
which is the 138 -> 863 MiB growth. **Setting `n_rs_seq = 0` falls back to
host-side state checkpoints that are already implemented** [L,
`server-context.cpp:2738`, `3620`], recovering ~1.3 GB. On Jetson's
unified memory that "host" copy is a DRAM memcpy, not a PCIe transfer, so
the tradeoff is far better here than on a discrete GPU. ~~**This is the highest-leverage untried experiment and requires no new
code.**~~ **TESTED AND REJECTED.** It saves exactly the predicted 1324
MiB and costs 59% of throughput, because the fallback branch `continue`s
past `common_speculative_accept()` and rewinds to the checkpoint rather
than the accepted position. 1-bit *native* is both smaller (4173 MiB) and
faster (18.61 tok/s) than 1-bit DSpark with this knob (5996 MiB, 13.55
tok/s), so the knob is dominated everywhere. See `results/memory-knobs.txt`
and WIKI S6.

### 1.3 Zero-memory speculation already ships in this fork

[L] `common/common.h:161-171` defines five n-gram self-speculation types
(`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`,
`ngram-cache`), and critically `need_n_rs_seq()` returns **0** for all of
them -- so on our hybrid GDN target they do not trigger the 5x
recurrent-state blowup at all. No drafter weights, no drafter KV, no
drafter compute buffer.

Published envelope for n-gram: 0% weight overhead, 0% extra KV, ~2.0-2.4x
on input-grounded workloads, ~1.0x on open-ended
(arXiv 2601.11580, MLSys 2026). llama.cpp PR #6828 measured **1.67x
CPU-only on a 70B with no draft model at all**, 62.5% acceptance.

**Measured here** (1-bit, ctx 4096, CUDA), `results/ngram-and-threads.txt`:

| Mode | total MiB | code tok/s | prose tok/s |
| :-- | --: | --: | --: |
| native | 4173 | 18.61 | 18.62 |
| ngram-simple | **4173** | 18.51 | 18.85 |
| ngram-map-k | **4173** | 18.85 | 18.85 |
| ngram-cache | **4173** | 18.83 | 18.84 |
| DSpark (capped 256) | 7320 | 33.23 | 26.81 |
| DSpark (uncapped) | 8355 | 33.87 | 26.78 |

**Memory is byte-identical to native in every n-gram variant** --
confirming zero overhead exactly as predicted. But **throughput is also
identical (~1.00x)**. That is the correct result for these workloads, not
a failure: n-gram speculation pays on *input-grounded* generation
(summarize, edit, rewrite, code-completion-in-context) where the output
copies from the prompt, and both our suites are open-ended generation.

So the honest 8 GB picture: n-gram is free and cannot hurt, making it the
right default fallback, but **it does not substitute for a drafter on
open-ended work**. The 8 GB options in order are (a) 1-bit native at
4173 MiB, (b) 1-bit + n-gram at 4173 MiB for grounded workloads, (c)
1-bit + capped DSpark at 7162-7320 MiB if the deployment can spare it.

### 1.4 If a separate drafter is needed, the literature has a better shape

- **CATS** (arXiv 2605.11186) is the most relevant paper found: cascaded
  shallow-draft -> shallow-verify -> full verify, **peak memory equal to
  the target alone**, explicitly benchmarked at **2/6/8 GB on Jetson AGX
  Orin** (2.82x / 2.96x / 3.16x). Its key negative result matters to us:
  deepening the draft raised accepted length 2.27 -> 2.98 but *dropped*
  speedup 1.93 -> 1.53x. **More accepted tokens is not more speed under a
  memory budget.**
- **MemSpec** (LCTES 2026) reports +40.7% on a **Jetson Orin Nano**
  specifically, diagnosing draft selection decoupled from draft
  availability under tight memory. Paywalled, unread.
- **Vocabulary size disqualifies Medusa-style heads here.** Medusa
  carries a full `d x V` projection per head; with V = 248,320 that is
  ~1.4B params per head. EAGLE-style heads reuse the target's embedding
  and LM head and are ~12x smaller.
- **Keep verification narrow.** arXiv 2505.22179 measured that on 4-bit
  weights the verify/decode ratio is ~1.8 at a 60-token tree vs ~1.2 for
  FP16, and *shrinking* the tree 60 -> 30 **raised** speedup 2.1 -> 2.3x
  -- the opposite of the FP16 trend. Our own DFlash work found linear
  drafts beat trees. Both point the same way: ternary targets want narrow,
  sequential verification.

### 1.5 Recurrent-state rollback: the 748 MiB is avoidable

The state math is exactly reproducible from the real config
(`prism-ml/Ternary-Bonsai-27B-unpacked`): 48 GDN layers x
(48 x 128 x 128 x 4 B state + 120 KiB conv) = **149.625 MiB**, x5 planes
= **748.125 MiB**. Confirmed `block_size = 4` in the drafter GGUF header,
matching `n_rs_seq = 4`.

Three cheaper mechanisms exist, in increasing order of merit:

1. **One checkpoint + recompute** (*The Mamba in the Llama*,
   arXiv 2408.15237): 6 tokens for 1.47x the cost of 1.
2. **Factor buffering** -- store the factors verification already
   produced (`k`, corrected value `d`, gate `g`, `beta`) and apply only
   the accepted subset, exploiting the delta rule's path-independence.
   **SpecLA** (arXiv 2607.16673) reports **128x less than per-candidate
   state**; vLLM PR #49847 measures per-step DRAM **5.35 -> 1.20 GB
   (-78%)** with **mean acceptance length unchanged** -- it is
   numerically exact. For our shapes a factor is ~8,384 floats vs
   786,432 for a state: **94x smaller per token**. Estimated 748 -> ~157
   MiB.
3. **STree** (arXiv 2505.14969) for tree verification with one state
   array.

**Reject state inversion.** Sherman-Morrison formally inverts the delta
update, but the gain diverges as beta -> 1 (exactly the hard-replace
regime the delta rule exists for), and you would still need the per-step
factors -- i.e. the factor buffer, which goes forward stably instead.

### 1.6 The 8 GB budget, arithmetic

1-bit native at ctx 2048 is 4041 MiB measured. KV for the 16 full-
attention layers is `16 x 2 x 4 x 256 x S x bpe` = 64 KiB/token at fp16,
~16 KiB at 4-bit; the GDN recurrent state is a fixed 149.6 MiB. So 1-bit
native leaves several GB of headroom -- **the model was never the
problem; the drafter's buffers were.**

---

## 2. RQ2 -- The roofline, and where the reference loses

[M] Measured on this Thor, not spec sheets:

| Quantity | Value |
| :-- | --: |
| SMs | **20** |
| L2 | **32 MB** (24 MB persisting window) |
| Max threads/SM | **1536** (not 2048 -- H100 heuristics are wrong here) |
| Streaming read, 128-bit loads, >=2 blocks/SM | **240 GB/s (88%)** |
| Streaming read, 32-bit scalar loads | 158 GB/s (58%) |
| L2-resident | ~1060 GB/s |
| `cudaMalloc` vs `cudaMallocManaged` vs pageable | **233 / 211 / 155 GB/s** |
| Empty kernel launch, streamed | 2.2-4.1 us |
| Empty kernel, **CUDA-graph replay** | **0.50-0.68 us** |
| Grid-wide `cg::sync()` | 1.24-1.49 us |

**Roofline at 229 GB/s achieved: ternary ~32 tok/s, 1-bit ~49 tok/s.**
*(Superseded: at our reproduced 177 GB/s this is ~26 tok/s ternary.)*
Against our measured 16.77 / 19.03 the reference fork is at **~52% / ~39%
of achievable**. Published Thor llama.cpp numbers for a dense Q8_0 7B sit
at ~70% of spec, so ~52% for the low-bit path is specifically a low-bit
kernel-quality gap.

### 2.1 The kernel ladder, measured on this box

Same problem, same hardware, five successive designs [M]:

| Design | GB/s | % spec |
| :-- | --: | --: |
| Naive scalar 2-bit extract, activations from global | 6.9 | 3% |
| Activations in shared memory, scalar extract + `fmaf` | 45 | 16% |
| fp16 magic-number dequant + `__hfma2`, sequential layout | 55 | 20% |
| fp16 + `__hfma2`, **bit-plane-interleaved layout** | 104 | 38% |
| `dp4a` + bit-plane layout, 1 row/warp | 158-191 | 58-70% |
| **`dp4a` + bit-plane, 8 rows/warp x 8 warps, grid 160** | **229** | **84%** |

The winning inner loop is 4 instructions per 4 weights (LOP3 mask +
non-saturating `__vsub4` + `__dp4a`). The ALU budget explains why:
sustaining 229 GB/s at 2 bits/weight needs 916 G-MAC/s, and [M] `dp4a`
delivers **7.07 T-MAC/s** -- 7.7x headroom -- while `fmaf` gives only
3.45 and the scalar unpack burns 4-6 extra ops per weight on the same
pipes.

Two load-bearing details: **bit-plane interleaving** must be baked into
the weight layout offline (convergent across llama.cpp TQ2_0,
microsoft/BitNet, and bonsai-turbo -- treat as settled), and **multiple
rows per warp** is what closed the last 25% by providing memory-level
parallelism.

Free extra: **hoist the ternary bias.** `Sum(c-1)x = Sum(c·x) - Sum(x)`
lets you dp4a the raw biased codes and subtract a per-group `Sum(x)`
computed once. **[M] +17%**, and it removes `__vsubss4` (which has no
single SASS instruction) entirely.

### 2.2 Do not build a LUT kernel on GPU

Every direct measurement at batch 1 shows LUT losing to arithmetic
dequant on GPU. any4 (arXiv 2507.04610, ICML 2025) measures LUT ~2x vs
int4 dequant ~3x at M=1, and states the LUT costs ~33% of the achieved
speedup; Meta evaluated and *rejected* a 256-entry shared-memory LUT over
bank conflicts. Microsoft's own GPU library (BitBLAS) is dequant-based,
and their GPU LUT paper (LUT Tensor Core, ISCA 2025) proposes **new
silicon** because LUT kernels underperform dequant on GPUs.

"Memory-bound" does not mean everything else is free -- it means keeping
the maximum number of loads in flight, and a shared-memory LUT costs
registers and serializes on bank conflicts.

**LUT is a CPU-only win.** See §4.

### 2.3 Sub-byte tensor cores are emulated on sm_110 -- avoid

[M] Throughput, measured:

| Instruction | T-MAC/s | vs INT8 |
| :-- | --: | --: |
| `mma.sync.m16n8k32.s8` | 56.5 | 1.00x |
| `mma.sync.m16n8k16.f16` | 28.2 | 0.50x |
| **`mma.m8n8k128.b1.and.popc`** | **11.1** | **0.20x** |
| `__dp4a` (CUDA core) | 7.07 | 0.13x |
| **`mma.sync.m8n8k32.s4`** | **1.83** | **0.03x** |

On sm_75/sm_80 `b1` ran at 4-8x the INT8 rate; on sm_110 it runs at
0.20x. SASS disassembly showed sm_110a emulates BMMA as 4x
`IMMA.16832.U8.U8` + `LOP3`. **This kills the entire popcount/XNOR
two-binary-plane family of ternary approaches on this hardware.** They
still compile, which will silently mislead.

Correction to our earlier note and to a prior memory: **Thor does have
tcgen05 / 5th-gen tensor cores with Tensor Memory** (requires
`-arch=sm_110a`), but it does **not** have warp-level FP4 `mma.sync` --
that is sm_120a/121a only. Neither matters at batch 1, but it explains
why ggml correctly gates Thor out of the Blackwell FP4 path.

### 2.4 Build and allocator issues in our current setup

- **We build `110-real`; we should build `110a-real`.** Baseline
  `sm_110` silently forgoes tcgen05, `setmaxnreg`, FP4/E8M0 conversions
  and TMA multicast. (Correctness of our published numbers is unaffected
  -- ggml never emits those -- but a custom engine would want them.)
- **`cudaMallocManaged` is not GPU-cached on Thor in CUDA 13.0** (NVIDIA
  Jetson Thor blog). [M] 211 vs 233 GB/s. `GGML_CUDA_ENABLE_UNIFIED_MEMORY`
  is the wrong path here.
- [L] `ggml-cuda.cu:249` hard-codes `integrated = false`, so ggml's iGPU
  zero-copy paths are disabled on Thor even though `prop.integrated == 1`.
- [M] **L2 persisting window is unexploited and free-ish**: +8.6% in a
  synthetic weight-stream + hot-buffer test using a 16 MiB persisting
  window with the weight stream marked streaming.
- **CUDA graphs are worth more on Thor than on x86+H100** [M] (0.5 us/node
  vs 2.2-4.1 us streamed) because the ARM CPU is slow at driver dispatch
  -- but against a ~31 ms roofline it is a single-digit-percent effect for
  ternary.

### 2.5 Megakernels: later, and smaller than the hype

[M] A grid-wide barrier (~1.3 us) costs roughly **2x a CUDA-graph node
boundary (~0.6 us)** on Thor, so the megakernel's central trade is not
automatically a win. Arithmetic: graphed ~0.3-0.6 ms/token vs a
megakernel's ~0.7 ms of grid syncs, against a 31 ms roofline -- **~2%
either way for ternary.**

The cross-GPU pattern predicts this: achieved fraction of peak bandwidth
at batch 1 falls as peak bandwidth rises (L4 ~81%, H100 ~27%), and CUDA
graphs gave 1.259x on H100 but 1.028x on L4. **Thor sits at the L4 end --
the favourable end for kernel work, the unfavourable end for
launch-overhead elimination.** `bonsai-turbo`'s own H100 numbers agree:
for ternary, CUDA-graph mode (151.1) ~= megakernel mode (149.6).

Try **Programmatic Dependent Launch** (CC >= 9.0, composes with CUDA
graphs) before a megakernel.

---

## 3. RQ3 -- Vulkan and CPU speculation

### 3.1 Vulkan: the barrier is the whole story

[M] on this Thor:

| Test | Cost |
| :-- | --: |
| dispatch, **no barrier** | 0.13-0.17 us |
| dispatch, **narrow** barrier (COMPUTE->COMPUTE) | **1.36-1.52 us** |
| dispatch, **wide** barrier (+`INDIRECT_COMMAND_READ`) | **3.94-4.11 us** |
| `vkCmdDispatchIndirect` vs `vkCmdDispatch`, masks equal | **+1 to +52 ns (zero)** |
| `VkBufferMemoryBarrier` vs global | +15 ns (no difference) |
| empty submit + fence | 3.5 us |
| submit 1 cmdbuf + fence wait | 49.5-55.6 us |
| **1 submit, 16 `VkSubmitInfo`s, 1 fence** | **15.6 us (6.1x)** |
| spin on GPU-written mapped flag vs fence | **20.0 us (2.8x)** |

A token graph is **3703 nodes / ~2293 real dispatches** [M, via
`export-graph-ops` on our Bonsai-27B-Q1_0], giving a **~3.9 ms/token
barrier floor**. Linear model validated to 0.3%:
`GPU us = N x 0.168 + B x 1.36`.

**This reverses our earlier framing.** We attributed the Vulkan loss to
submit/synchronize cost; submits and host round-trips together are <5% of
the tax. It is barriers. And an important refinement: the entire
narrow->wide gap is `VK_ACCESS_INDIRECT_COMMAND_READ_BIT` in
`dstAccessMask` (naming the *stage* is free), so **indirect dispatch is
free if you scope the access mask** -- naively wide-masking ~2000
barriers costs ~5.2 ms/token of pure avoidable time.

[L] `ggml_vk_sync_buffers` (`ggml-vulkan.cpp:3069`) issues a **global**
pipeline barrier with full read/write masks at **49 call sites**, with
only three narrow elision flags. That is exactly the benchmarked pattern.

**Ranked Vulkan actions:** (1) add `q1_0`/`q2_0` to the integer-dot MMQ
and MMVQ lists -- precedent +78-131%; (2) barrier scoping and elision
plus op fusion -- ~5.3x less barrier time, and CUDA-side fusion precedent
is +27-43%; (3) one `vkQueueSubmit` per step and spin on mapped memory
rather than a fence; (4) device-side sampling and accept/reject.

**Do not build a persistent megakernel in Vulkan.** Khronos Vulkan-Docs
#2233: there are **no forward-progress guarantees** for compute
workgroups, so inter-workgroup spin-wait is undefined behaviour (UE5
Nanite disabled its equivalent on PC for this reason). [M] confirmed
empirically here: grid-wide rendezvous **deadlocked above 96 workgroups**,
and mid-kernel CPU<->GPU coherence does not exist in either direction --
the submission boundary is the only coherence point.

**Check first:** llama.cpp issue #17957 / PR #17974 suggest Vulkan
speculative decoding may be functionally broken rather than merely slow.
Our 0.39-0.59x could be partly a bug. Note also our independent finding
that `n_probs` corrupts Vulkan generation.

Closest published analogue: llama.cpp #23752, MTP speculative decoding is
a **net loss on Metal at every configuration**, -11% even at 100%
acceptance. Acceptance fine, per-step dispatch tax fatal -- the same
shape as ours.

### 3.2 The breakeven rule, and why acceptance is the wrong trigger

Leviathan et al. (arXiv 2211.17192, ICML 2023):

```
E[tokens/iter] = (1 - a^(g+1)) / (1 - a)
speedup        = (1 - a^(g+1)) / ((1 - a)(g*c + 1))
improvement iff   a > c
```

where `a` is acceptance and **`c` is the ratio of one draft step to one
target step**. **`a > c` is the whole answer to our measured anomaly**:
1-bit code loses at 56.6% acceptance while ternary code wins at 57.7%
because `c` differs between them. Independently confirmed by "Decoding
Speculative Decoding" (arXiv 2402.01528, 350+ experiments): *"the draft
model's capability in language modeling does not correlate strongly with
its performance in speculative decoding"* -- they got 111% higher
throughput purely by optimizing draft latency.

Two corrections the standard model needs for our setting:

- `g*c + 1` assumes verifying `g+1` tokens costs the same as 1. False on
  CPU, and false on 4-bit weights (arXiv 2505.22179). Replace with
  `g*c + v(g)` using **measured** verify cost at width `g`.
- Add a per-round overhead term `o`: `(1 - a^(g+1)) / ((1-a)(g*c + v(g) + o))`.
  On CUDA `o ~= 0.03-0.12`; **on Vulkan `o` is plausibly > 1**, which
  forces speedup < 1 regardless of acceptance. That is our 0.39-0.59x.

**Recommended control rule** -- Cascade / utility-driven
(arXiv 2506.20675): `U = ETR / (t_spec / t_base)`, disable speculation
when `U < 1`. Reported 7-14% over the best static choice with **worst-case
slowdown capped at 5% vs 54% for static schemes**. One mechanism handles
the Vulkan loss, the 1-bit/ternary inversion, and long-context drift.
`BanditSpec` (ICML 2025) is the training-free drop-in alternative.

---

## 4. RQ4 -- Kernels, portability, and the CPU backend

### 4.1 ARM: our CPU numbers are ~6x off the achievable, and one fix is free

[M] on this Thor's Neoverse V3AE cores (no SME; SVE VL = **128 bits**):

| | quoted | **measured best** | threads | roofline | headroom |
| :-- | --: | --: | --: | --: | --: |
| Q2_0 CPU | 2.22 | **2.84** | **12** | 17.7 | 6.2x |
| Q1_0 CPU | 3.41 | **5.79** | **10** | 33.3 | 5.8x |

CPU-cluster achievable DRAM bandwidth is **126.4 GB/s at 14 threads**,
but both baselines run at ~20 GB/s -- **one core's worth**. The binding
constraint is unpack ALU, not DRAM.

**Free win: never use all 14 cores.** Confirmed by our own sweep
(`results/ngram-and-threads.txt`, 48-token requests):

| threads | ternary tok/s | 1-bit tok/s |
| --: | --: | --: |
| 8 | 2.33 | 3.64 |
| 10 | 2.79 | 4.28 |
| **12** | **3.18** | **4.85** |
| 14 (default) | 1.84 | 2.14 |

**The default is the worst setting**: `-t 12` is worth **1.73x on ternary
and 2.27x on 1-bit** over `nproc`. Our published CPU numbers used the
default, so they understate the backend. `serve_reference.sh` now
defaults the CPU path to 12 threads.

[M] Issue rates that overturn LLVM's scheduling model for this core:
`SDOT` 1.96/cycle, **`SMMLA` 1.97/cycle (exactly 2x the MACs)**, `TBL`
1.96, `SHR+AND` **3.94**. So **SMMLA is a free 2x wherever M >= 2**, and
pure-TBL LUT kernels are pipe-starved while TBL mixed with SDOT co-issues
nearly free.

[M] Measured kernel bake-off, single core:

- **2-bit:** current Q2_0 `vec_dot` 11.4 Gw/s -> shr/and + **SMMLA 4-row
  94.6 (8.3x)**
- **1-bit:** current memory-LUT 19.8 -> **T-MAC register LUT g=4 with
  fast aggregation 87.8 (4.4x)**

**LUT wins at 1-bit on CPU (4.4x); SMMLA wins at 2-bit (8.3x).** Note this
is the exact opposite of the GPU verdict in §2.2 -- because ARM `TBL` does
a 16-entry lookup in one instruction on a register, and GPUs have no such
instruction.

[L] Root cause in the tree: `repack.cpp:5352` has **no NEON branch for
Q2_0** (only avx512+vnni), so Q2_0 never repacks and falls back to
`vec_dot` with `nrows=1` -- prompt processing degenerates to independent
GEMVs with zero weight reuse. That is why Q2_0's pp/tg is 1.37 while
Q1_0's is ~2.9-3.5.

[M] Things measured to be **worthless or negative on this core**: SVE2 at
VL=128 (bit-identical to NEON), SVE2 BitPerm `BDEP`/`BEXT` (0.5
instr/cycle, byte-local only), SVE gather, 4-register `TBL`.

### 4.2 Portability: what generalizes across the three backends

**Generalizes cleanly, belongs in shared templates:**

1. **Biased encoding + sum correction.** Store `u = t + 1 in {0,1,2}`, so
   `Sum(w*b) = Sum(u*b) - Sum(b)`. Identical on CUDA `__dp4a`, Vulkan
   `dotPacked4x8AccSatEXT`, and ARM `SDOT`/`SMMLA`. [M] +17% on CUDA
   alone, and it removes saturating-subtract instructions everywhere.
2. **Row-interleaved layouts feeding a 4-row accumulator** -- works on
   SMMLA, MMA, and dp4a alike.
3. **Batch-1 is dot-shaped; batch >= 2 is matrix-shaped.** The same split
   KleidiAI makes for SME2, and the same one our CPU verify-batch curve
   shows.

**Does NOT generalize:** LUT/TBL is a CPU-only win (§2.2 vs §4.1);
popcount/XNOR ternary is dead on sm_110 (§2.3); persistent megakernels are
UB in Vulkan (§3.1).

**Packing layout decision: keep 2 bits/weight with group scales.** Base-3
(5 trits/8 bits, llama.cpp TQ1_0) saves 0.375 bpw but costs ~4 ALU ops per
weight versus ~0.5 for the 2-bit LOP3 path -- a non-starter on GPU where
§2.1 shows a budget of ~4 op-equivalents per weight total. Since the
weights are shared across all three backends, this settles it for all
three.

---

## 5. RQ5 -- GDN and attention specifics

Verified architecture (from the real config, correcting assumptions):
`hidden 5120`, `vocab 248320`, **dense not MoE**; GDN layers 48 value
heads / 16 key heads / head_dim 128, conv kernel 4; full-attention layers
**24 heads / 4 KV heads / head_dim 256** (not 128), `attn_output_gate`
with **swish**, `partial_rotary_factor 0.25`, `rope_theta 1e7`, ctx
262144.

Three actionable consequences:

1. **`gqa_ratio = 24/4 = 6` is not a power of two.** [L] `fattn.cu:483`
   computes the effective ratio by doubling while `gqa_ratio % (2*eff) == 0`,
   giving **eff = 2, not 6** -- 3x of the available GQA folding is lost to
   the divisibility test alone.
2. **Quantized KV silently loses the tensor-core path.** [L] `fattn.cu:459`
   has an MMA escape hatch in the f16 branch with no equivalent in the
   quantized branch, so 4-bit KV at long context drops to the scalar vec
   kernel. This likely explains why the vendor measured 4-bit KV *slower*
   than fp16 (82.7 vs 85.5 tok/s). BitDecoding (arXiv 2503.18773) is the
   blueprint.
3. **`partial_rotary_factor = 0.25` means 192 of 256 key channels are
   RoPE-free**, so KVQuant-style pre-RoPE per-channel key quantization
   works unconditionally on 75% of the cache. Nobody has published this.

At decode a GDN layer is a pure recurrence with **arithmetic intensity
0.75 FLOP/byte (fp32)** -- decisively state-bandwidth bound. But in plain
decode GDN state is only **4.1% (ternary) / 7.5% (1-bit)** of per-token
traffic; weights dominate. It rises to ~17% during block verification,
which is what §1.5 attacks.

For prefix caching, prefer SGLang's design (node-local mamba locks,
copy-on-write state forking, **fixed** global checkpoint intervals) over
vLLM's geometry-determined single checkpoint -- vLLM issue #45238 shows a
100-token prompt shift flipping 52/64 cache hits to 0/64. At 149.6 MiB per
checkpoint, budget 2-8 checkpoints with LRU plus recompute, not interval
coverage.

---

## 6. RQ6 -- The model family: a scope correction

Verified against the HF API (33 repos in `prism-ml`):

| Family | Architecture | Layers | GDN | ctx | Vision | DSpark |
| :-- | :-- | --: | :-- | --: | :-- | :-- |
| **27B** (both builds) | `Qwen3_5ForConditionalGeneration` | 64 | **48/16 hybrid** | 262144 | yes | **yes** |
| 8B / 4B / 1.7B | `Qwen3ForCausalLM` | 36/36/28 | **none** | 65536/32768/32768 | no | **no** |

Sizes: `Bonsai-8B-Q1_0` 1.08 GiB, `Ternary-8B-Q2_0` 2.03 GiB,
`Bonsai-4B-Q1_0` 0.53 GiB, `Ternary-4B-Q2_0` 1.00 GiB,
`Bonsai-1.7B-Q1_0` 0.23 GiB, `Ternary-1.7B-Q2_0` 0.43 GiB. No 0.6B. Also
published: MLX (8 repos), unpacked safetensors (8), AWQ-4bit (3), and a
separate **image-generation** family (6, diffusion pipelines, not VLMs).

**Consequence for the suite plan:** the small variants are a *simpler*
engine path -- no recurrent state at all, no drafter, shorter context --
and they give **zero GDN test coverage**. Supporting them is worthwhile
for the "runs everywhere" story, but they cannot be used to iterate on the
GDN kernels that the 27B needs. A small hybrid for that would have to come
from outside the Bonsai family (Qwen3-Next / Qwen3.5-9B class).

DSpark drafter internals, read from the GGUF header: `block_count 6`,
`block_size 4`, `target_layers [1,16,31,46,61]`, `markov_rank 256`,
confidence head with Markov, log-SNR conditioning, `mask_token_id 248319`,
shares the target's tokenizer. **DSpark is an industry format, not
PrismML-specific** -- `deepseek-ai/dspark_qwen3_*`, `nvidia/MiniMax-M3-DSpark`,
`RedHatAI/GLM-5.2-speculator.dspark` all ship it, which means drafter
kernel work is reusable well beyond Bonsai.

---

## 7. Revised priority list

Ordered by measured value per unit of work.

**Free or near-free, do first:**

1. **CPU thread count** `-t 12` / `-t 10`. [M] up to 2.3x on CPU tg.
2. **Drafter staging cap.** [M, ours] ~1 GB saved, zero cost. Done;
   `patches/0001-dspark-staging-batch-cap.patch`.
3. **`n_rs_seq = 0`** with the existing host-checkpoint fallback. ~1.3 GB,
   no new code. Untested -- highest-leverage remaining experiment.
4. **N-gram self-speculation** as the 8 GB story. Zero weights, zero KV,
   no recurrent-state blowup, already in the fork.

**Cheap and high-value:**

5. **Add `q1_0`/`q2_0` to Vulkan's integer-dot lists.** Precedent
   +78-131%; both formats currently take the float path.
6. **NEON repack + SMMLA 4-row for Q2_0.** [M] 8.3x kernel, ~2.4x
   end-to-end.
7. **T-MAC register LUT for Q1_0 on CPU.** [M] 4.4x kernel; would put CPU
   1-bit at 17-25 tok/s, **beating our current CUDA 19.03**.
8. **Barrier scoping/elision + fusion on Vulkan.** ~5.3x less barrier
   time.
9. **Utility-based adaptive draft disable** (`U < 1`), keyed on measured
   `t_draft`/`t_verify`, not acceptance.

**The engine itself:**

10. **The GEMV recipe of §2.1** -- this is ~95% of the available win and
    is where Phase 2 should start.
11. Chunked drafter prefill; factor buffering for GDN rollback.
12. CUDA graphs, then PDL. **Megakernel last, expect ~5% on Thor.**

**Explicitly rejected:** GPU LUT kernels; popcount/XNOR ternary on
sm_110; base-3 packing; Vulkan persistent megakernels; state inversion
for rollback; SVE2/BitPerm/gather on this core.
