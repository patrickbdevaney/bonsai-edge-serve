# Deep Research Prompt: Optimal Low-Bit Decode Engines for the Bonsai Family

A meta-critique of this repo as it stands, and the research brief for what
it should become. Written to be handed to a research effort (human or
model) verbatim.

Everything stated as measured here was measured on this box. Everything
else is marked as assumption or open question. The point of the document
is to be honest about which is which, because the most expensive mistake
available to this project is optimizing against a number we believe but
have not established.

---

## 0. What this project is actually trying to do

Build the **fastest possible batch-1 decode** for one model family --
PrismML's Bonsai (ternary `Q2_0_g128` and 1-bit `Q1_0_g128` builds of
Qwen3.6-27B hybrid) -- across **three backends that share a design**:
CUDA, Vulkan, and CPU (ARM NEON now; x86 AVX-512 later, elsewhere).

Two properties are in tension and the whole engineering problem lives in
that tension:

- **Model-specific optimization.** We are allowed to hard-code this
  architecture: 64 layers, 48 gated-delta-net linear-attention layers,
  16 full-attention layers with GQA + QK-norm + gated Q, packed low-bit
  weights consumed directly and never expanded to FP16.
- **Hardware-agnostic structure.** The three backends must be *templates
  of one engine*, not three engines. Arch-specific fast paths are
  allowed; arch-specific *core paths without a portable fallback* are
  not.

The deploy targets are small: Jetson Orin Nano Super at 8 GB, phones.
Jetson Thor (122 GB unified) is only the development box. **Thor's
headroom must never be spent on anything that will not exist on the
deploy target.**

---

## 1. Meta-critique: what is wrong with this repo today

### 1.1 We have measured the reference, not built an engine

Every number published is the PrismML llama.cpp fork. Phase 2 -- the
custom engine -- has not started. That was deliberate (build the oracle
and the gates first) and it produced real findings, but it means **this
repo currently has no engine to defend**. The research must serve the
build, not more measurement.

### 1.2 Our headline result is a ratio, and ratios flatter slow hardware

DSpark gives 1.48-1.75x on Thor's CUDA backend versus H100's published
1.34x. That is **not** Thor doing better. Raw throughput is ~6x lower
(16.8 vs 98.0 tok/s ternary native). H100 at batch 1 is launch- and
sync-latency-bound, so there is less per-step cost for speculation to
amortize; Thor's slower bandwidth-bound decode gives drafting more to
hide behind. Any future claim must lead with absolute tok/s.

### 1.3 The measurements that were easy to get wrong, we got wrong first

Recorded because the pattern will repeat:

- **Memory.** `nvidia-smi`, `MemTotal-MemAvailable`, `cudaMemGetInfo`,
  `--no-mmap`+RSS, and instantaneous `VmRSS` all gave wrong answers on
  unified memory. Peak RSS (`VmHWM`) *looked* right -- 7158 MiB for a
  7.17 GB model -- but `smaps` showed only 322 MiB resident: it was
  measuring load-time page-cache traffic. Only ggml's own buffer
  accounting is trustworthy.
- **The correctness gate.** It reported 16/16 divergent for the CPU
  backend purely because that run generated 64 tokens against a 128-token
  oracle. Real figure: 3/16.
- **Power.** First reading showed 33 W "idle" because a build was running.
- **Losslessness.** A 64-token check passed; the divergence was at 96.

**Research implication:** every performance or memory claim in the
literature we are about to read deserves the same suspicion. Prefer
sources that state their measurement method.

### 1.4 We do not know if our biggest finding generalizes

Measured: DSpark output diverges from native greedy output at temperature
0 (5/16 ternary, 6/16 1-bit trace pairs, reproducible on a fresh
server's first request, not a capture artifact). Speculative decoding is
supposed to be lossless. We cannot tell whether this is Thor-specific
numerics or general to the fork without a second device. **If it is
general, it is a significant correctness finding about the reference
implementation. If it is Thor-specific, it is a warning about sm_110
numerics that our own kernels will inherit.** Either way it must be
resolved before we trust our own engine's gates.

### 1.5 The Orin plan was wrong and we only found out by measuring

The directive assumed 1-bit + drafter fits 8 GB and ternary + drafter is
tight. Measured totals (MiB):

| config | ctx 2048 | 4096 | 8192 | 16384 |
| :-- | --: | --: | --: | --: |
| ternary native | 7247 | 7379 | 7643 | 8171 |
| ternary DSpark | 10988 | 11713 | 13257 | 16729 |
| 1-bit native | 4041 | 4173 | 4437 | 4965 |
| 1-bit DSpark | 7631 | 8355 | 9899 | 13372 |

Neither DSpark configuration fits 8 GB with an OS resident. **This is now
the project's central engineering target.**

---

## 2. The central problem, stated precisely

The drafter's cost is dominated by **compute buffers, not weights**. At
ctx 16384 the drafter adds 2432 MiB device + 2752 MiB host of compute
buffer against only ~1534 MiB of weights, and ~1.3 GB more lands on the
*target* (recurrent state 149.62 -> 748.12 MiB, compute 138.02 -> 863.64
MiB).

Root cause, read from the reference implementation
(`tools/server/server-context.cpp`):

```
n_batch_dspark = n_ctx + block_size;   n_ubatch = n_batch;
```

with the comment that the drafter "stages all context rows since its
cache position PLUS a full block in ONE batch -- worst case
ctx_len == n_ctx right after begin() (e.g. a follow-up request with a
long history). If the drafter's batch cannot fit ctx_len + block_size,
the round is skipped and speculation silently degrades to plain AR."

So the buffer is sized for a **worst case that is rare in steady-state
decode**, and the failure mode when it does not fit is already graceful
(skip the round, fall back to autoregressive). That asymmetry is the
opening.

**RQ-CORE:** What is the smallest drafter staging buffer that preserves
most of the speculative speedup, and what is the actual distribution of
per-round staging demand in steady-state decode versus at session
resume? Is a capped buffer plus graceful per-round fallback strictly
better than the current worst-case sizing, on a memory-constrained
device?

---

## 3. Research questions, in priority order

### RQ1 -- Fit 1-bit 27B + speculation + usable KV in 8 GB

The flagship target. Sub-questions:

1. **Conservative drafter buffers.** Per RQ-CORE. What does the
   throughput/memory curve look like as the cap shrinks? Where is the
   knee? How often does the fallback trigger at each cap?
2. **Zero-extra-memory speculation.** If a separate drafter cannot fit,
   what self-speculation methods have near-zero memory cost --
   n-gram/prompt-lookup, layer-skip/early-exit self-drafting, Medusa-style
   heads (but those add weights), lookahead decoding? What acceptance and
   speedup do they achieve on code and prose, and what do they cost in
   memory? **A method that gives 1.15x for 0 MB may beat one that gives
   1.5x for 3 GB on this device.**
3. **KV and state budget.** With 4-bit KV cache (parity with the
   reference) plus a fixed 149.62 MiB recurrent state, what context
   length is actually reachable at 8 GB for the 1-bit build? Give the
   budget arithmetic explicitly.
4. **Must-work-without-speculation.** The engine has to be excellent with
   drafting entirely off, because that may be the only thing that fits.
   What is the maximum achievable native decode rate, and what is the
   roofline?

### RQ2 -- Roofline and the honest performance ceiling

Batch-1 decode of a 27B model at ~2 bits/weight is memory-bandwidth
bound. Establish, with method stated:

1. Thor's achievable memory bandwidth (measured, not spec-sheet).
2. Bytes moved per decode step for each build (weights + KV + recurrent
   state + activations).
3. Therefore the theoretical tok/s ceiling, and what fraction the
   reference fork achieves today (16.77 ternary / 19.03 1-bit measured).
4. **The gap is the entire opportunity.** Prior work (bonsai-turbo)
   claims 1.76x over the fork on H100 without speculation, which implies
   the fork leaves a lot on the table. Where does it go: kernel launch
   overhead, poor unpack throughput, insufficient memory-level
   parallelism, synchronization?

### RQ3 -- Make speculation pay off on Vulkan and CPU

Measured, same box, same weights, same drafter:

| Backend | ternary code | ternary prose | 1-bit code | 1-bit prose |
| :-- | --: | --: | --: | --: |
| CUDA | 1.48x | 1.37x | 1.75x | 1.38x |
| Vulkan | 0.59x | 0.54x | 0.47x | 0.39x |
| CPU NEON | 1.07x | 0.79x | 0.88x | 1.00x |

Acceptance is backend-independent (Vulkan within ~1 point of CUDA
everywhere), so **the drafter predicts equally well and the loss is
purely execution structure**: each speculation round is a small dependent
dispatch, and Vulkan's submit/synchronize cost dominates a step.

1. Can pre-recorded/reusable command buffers, `vkCmdDispatchIndirect`,
   timeline semaphores, device-side control flow, or persistent compute
   shaders collapse the per-round overhead enough to flip the sign?
2. Is there ANY published case of speculative decoding being net-positive
   in a graphics-API dispatch model? If not, is the honest answer "run
   the whole draft-verify round as one pre-recorded submission" or "do
   not speculate on Vulkan"?
3. On CPU, breakeven is workload-dependent and close to 1.0. What
   structure makes drafting cheap on CPU -- thread-pool reuse, avoiding
   re-entering the scheduler per round, drafting on a subset of cores
   while the target uses the rest?
4. **The adaptive-disable decision rule.** We measured that keying on
   acceptance is wrong: 1-bit code *loses* at 56.6% acceptance while
   ternary code *wins* at 57.7%. The rule must key on realised cost
   ratio. What is the correct online estimator, and what do published
   dynamic-speculation systems actually use?

### RQ4 -- Low-bit kernel state of the art, and what is portable

1. Best-known ternary/1-bit GEMV formulations: lookup-table methods
   (T-MAC, LUT-GEMM, BitBLAS), bit-slicing, popcount, dequant-fused
   matmul. Which win at **batch 1** specifically, where the operation is
   a GEMV and bandwidth-bound, not a GEMM?
2. Packing layouts: 2-bit slots vs base-3 (5 trits/8 bits). What does
   each cost to unpack per backend, and does a layout that is optimal on
   CUDA remain reasonable in GLSL and NEON? **We must not adopt a layout
   that is fast on one backend and pathological on another** -- the
   weights are shared across all three.
3. Which techniques are portable and which are hardware-specific? For
   each, name the portable fallback.
4. sm_110 precisely: what a CC 11.0 part has and lacks versus sm_120
   (tcgen05, FP4 tensor cores, cp.async variants, wgmma, clusters). Note
   ggml gates Blackwell paths on CC >= 1200, so Thor takes Ampere/Turing
   MMA paths -- confirm whether that is leaving performance available.

### RQ5 -- GDN recurrent state: kernels, speculation, and prefix reuse

1. Optimal batch-1 decode kernel for a gated delta net layer. At decode
   it is a pure recurrence: what bounds it, state bandwidth or compute?
2. **Speculative verification against a destructively-updated recurrent
   state.** With a KV cache, rejecting tokens is trivial. With a
   recurrent state it is not. How do published systems handle rollback
   for Mamba/RWKV/linear-attention models? This likely explains the
   measured 149.62 -> 748.12 MiB growth: block verification needs
   multiple state slots. **What is the minimum number of state copies
   required for a block of size k, and can rollback be done by
   recomputation instead of buffering?**
3. Prefix reuse with recurrent state: resuming at position N requires the
   state at N. The reference already checkpoints (~149.6 MiB per
   checkpoint, up to 32 checkpoints, min spacing 256). What is the right
   checkpoint interval / recompute tradeoff for an 8 GB device where 32
   checkpoints is 4.8 GB and therefore impossible?

### RQ6 -- The model family, and engines below 27B

The repo should become a suite for the Bonsai family, not one model.

1. Exhaustive inventory of what actually exists on Hugging Face: every
   parameter count, quantization, format, and any DSpark drafter
   published per variant, with file sizes.
2. For each, the smallest device it plausibly serves, using the same
   budget arithmetic as RQ1.3.
3. **Does the optimal engine design change with scale?** A 1.7B ternary
   model on a phone is a different regime: weights may fit in cache-ish
   budgets, dispatch overhead dominates more, and speculation economics
   invert again. Where are the regime boundaries?

### RQ7 -- Single-launch decode with speculation fused

The differentiating claim. Prior art (bonsai-turbo) proved single-launch
batch-1 decode works and is fast **without** speculation. We want
draft + verify + accept/reject inside one graph replay.

1. Survey megakernel / persistent-kernel / CUDA-graph decode designs and
   their pitfalls (occupancy, register pressure, dynamic shapes, control
   flow).
2. Speculation needs data-dependent control flow (how many tokens were
   accepted). How do you keep that inside a graph -- device-side
   branching, fixed-shape verify with masking, indirect dispatch?
3. What is the portable analogue in Vulkan and on CPU? If the single
   launch is CUDA-only, the "one engine, three backends" claim weakens;
   is there a shared abstraction (a recorded schedule replayed per step)
   that all three can implement?

---

## 4. Design constraints any proposal must respect

1. **Weights are shared across backends.** One packed layout, or an
   explicitly justified per-backend repack with the conversion cost
   stated.
2. **No hardware-specific core path without a portable fallback.**
   Low-bit dequant+matmul is integer/ALU work, so this should be
   tractable; if a tensor-core path is used it is the exception and needs
   an arch guard.
3. **Correctness gates before performance claims.** Same-mode against
   same-mode. Cross-backend tolerance ~0.10 on top-1 logprob drift
   (measured: matching CPU traces drift 0.01-0.06); same-backend
   regression can stay tight.
4. **Both workload classes always.** Code (high acceptance) and prose
   (low acceptance). Publishing only code numbers is the failure mode
   this repo exists to avoid.
5. **Deploy-target realism.** No design that only works with Thor's
   122 GB. State the 8 GB budget for every proposal.
6. **Graceful degradation is a feature.** The reference already skips a
   speculation round rather than failing when the batch does not fit.
   Any memory reduction should preserve that property.

---

## 5. What a good answer looks like

Not a literature survey. A **prioritized, costed engineering plan**:

- For each technique: expected gain, implementation cost, portability
  across the three backends, and what measurement would confirm or kill
  it.
- Explicit kill-criteria. If self-speculation at 1.15x for 0 MB beats a
  drafter at 1.5x for 3 GB on 8 GB devices, say so and drop the drafter
  there.
- Cite sources with enough specificity to check them, and state which
  claims are measured, which are reported by authors, and which are
  inferred.
- Where the literature does not answer a question, say so and propose the
  experiment. Several questions here (Vulkan speculation, recurrent-state
  rollback cost, staging-buffer knee) may simply be unpublished, and
  measuring them is then itself the contribution.

---

## 6. Known-unknowns to resolve first

Cheap experiments that unblock large decisions:

1. Cap the drafter staging buffer and measure the throughput/memory knee
   plus fallback frequency. **Directly decides whether 1-bit + drafter
   fits 8 GB.**
2. Measure Thor's achievable memory bandwidth, to convert every tok/s
   into a fraction of roofline.
3. Determine whether the temperature-0 DSpark divergence is Thor-specific
   or general.
4. Count the recurrent-state copies block verification actually needs, to
   see whether the 149.62 -> 748.12 MiB growth is necessary or
   incidental.
5. Inventory the Bonsai family so the suite's scope is based on models
   that exist.
