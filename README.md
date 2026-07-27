# bonsai-edge-serve

Multi-backend speculative serving for PrismML's **Bonsai-27B** family --
the ternary (`Q2_0`) and 1-bit (`Q1_0`) builds of Qwen3.6-27B -- with
**DSpark** speculative decoding, developed on **Jetson Thor (sm_110)**
and targeting the same model across Jetson Orin Nano Super, Thor, RTX
cards, and plain CPUs.

This is **not** a general GGUF engine. That is llama.cpp's job, and
llama.cpp does it well. This is model-specific optimization for one
architecture -- Qwen3.6-27B hybrid: 64 layers, gated-delta-net linear
attention on 48 of them, full GQA attention with QK norm and gated Q on
every 4th layer -- in Bonsai's packed low-bit formats, with the DSpark
drafter fused into the decode step.

## Why this repo exists

Three things are not published anywhere, by anyone:

1. **No Jetson Thor (sm_110) numbers** exist for any Bonsai variant,
   native or speculative.
2. **No bespoke engine has DSpark fused in.** `bonsai-turbo` is the
   fastest published Bonsai decode engine and says so explicitly:
   speculative decoding is its missing piece.
3. **No cross-device scaling story.** Same repo, same model weights,
   Orin Nano Super to Thor to RTX to x86 CPU, measured end to end.

This repo owns those three gaps.

## Status

| Phase | What | State |
| :-- | :-- | :-- |
| 0 | Environment + models on Thor | done |
| 1 | Reference stack + captured oracle traces | done |
| 1 | **Milestone 0** -- first Thor numbers, reference fork | **done, below** |
| 2 | CUDA engine, DSpark fused into a single launch | not started |
| 3 | Cross-arch CUDA targets | compile-verified; only sm_110 run |
| 4 | Vulkan backend | measured via fork; correctness gap found |
| 5 | CPU backend | ARM NEON measured; AVX-512 needs an x86 host |
| 6 | Cross-backend table on one device | done, below |
| 6 | Cross-*device* table (Orin, RTX) | blocked: no second device here |

No performance numbers are published in this README until they have been
measured on the hardware named in the row. Nothing here is estimated.

## Milestone 0 -- Jetson Thor reference numbers

**First published Bonsai-27B numbers on Jetson Thor (sm_110), closing
GAP 1.** Measured from the PrismML reference fork alone -- no custom
engine involved.

Device: NVIDIA Thor, CC 11.0, driver 580.00, CUDA 13.0, 122 GiB unified
memory. Engine: PrismML `llama.cpp` fork `b9597-7529fdaaf`, CUDA build
(`ARCHS = 1100`), `-fa on -c 16384 -np 1`, draft depth 4. Decode is the
median over 8 prompts per workload class, 128 tokens each, temperature 0,
no prefix reuse. Reproduce with `reference/run_milestone0.sh`; raw
artifacts in `results/`.

| Variant | Mode | Workload | tok/s | TTFT (s) | Prefill tok/s | Acceptance | Resident |
| :-- | :-- | :-- | --: | --: | --: | --: | --: |
| ternary Q2_0 | native | code | 16.77 | 0.491 | 120.2 | n/a | 7158 MiB |
| ternary Q2_0 | native | prose | 16.97 | 0.439 | 79.7 | n/a | 7158 MiB |
| ternary Q2_0 | DSpark | code | **24.85** | 0.523 | 107.7 | 54.0% | +1802 MiB* |
| ternary Q2_0 | DSpark | prose | 23.23 | 0.477 | 69.1 | 43.3% | +1802 MiB* |
| 1-bit Q1_0 | native | code | 19.03 | 0.424 | 149.1 | n/a | 3952 MiB |
| 1-bit Q1_0 | native | prose | 18.98 | 0.380 | 100.3 | n/a | 3952 MiB |
| 1-bit Q1_0 | DSpark | code | **33.24** | 0.464 | 128.3 | 61.6% | +1802 MiB* |
| 1-bit Q1_0 | DSpark | prose | 26.25 | 0.418 | 82.7 | 51.4% | +1802 MiB* |

\* Native footprints are measured (peak RSS, matching each GGUF's size).
The drafter figure is the **server's own estimate**, not a measurement:
peak RSS is not additive across two mmap'd models, so no trustworthy
measured two-model number exists yet. See `results/memory.txt`.

### DSpark speedup on Thor

| Variant | Workload | Native | DSpark | Speedup | Acceptance |
| :-- | :-- | --: | --: | --: | --: |
| ternary Q2_0 | code | 16.77 | 24.85 | **1.48x** | 54.0% |
| ternary Q2_0 | prose | 16.97 | 23.23 | 1.37x | 43.3% |
| 1-bit Q1_0 | code | 19.03 | 33.24 | **1.75x** | 61.6% |
| 1-bit Q1_0 | prose | 18.98 | 26.25 | 1.38x | 51.4% |

### What these numbers say

**DSpark never goes negative on Thor.** The published concern is that on
prose, where acceptance falls to ~51%, the 1-bit target's weights are
small enough that drafting overhead can make speculation a net loss. On
Thor it does not: the worst case measured is still 1.37x, and 1-bit on
prose -- the exact regime flagged as risky -- returns 1.38x at 51.4%
acceptance. The adaptive draft-disable heuristic planned for Phase 2 has
no regime to trigger in on this device, though it may still on Orin.

**The speedup ratios are larger than H100's published 1.34x, and that is
a statement about Thor being slower, not better.** Raw throughput is not
close: 16.8 tok/s here against H100's published 98.0 tok/s on ternary
native, roughly 6x apart. PrismML's own model card notes H100 at batch 1
is bound by kernel-launch and synchronization latency rather than weight
bandwidth -- there is simply less per-step cost for speculation to
amortize. Thor's slower, bandwidth-bound decode gives drafting more to
hide behind, so the *ratio* is higher while the absolute number is far
lower.

**Acceptance is below the published figures** (54.0% vs ~69.2% ternary on
code) but this is not a device comparison. Acceptance is a property of
the drafter and the token distribution, not the hardware; our workloads
are raw completions rather than PrismML's chat-templated thinking-mode
benchmark, and prompt content moves acceptance substantially. The
code-vs-prose *gap* reproduces clearly in both variants, which is the
behaviour that matters.

## Cross-backend on one device (GAP 3, in part)

Same Thor box, same weights, three backends. This isolates the backend
from the hardware -- every row below ran on the same GPU or its own CPU
cores, so differences are the backend's, not the device's.

| Backend | Variant | native code | native prose | DSpark code | DSpark prose |
| :-- | :-- | --: | --: | --: | --: |
| CUDA | ternary | 16.77 | 16.97 | **24.85** (1.48x) | **23.23** (1.37x) |
| CUDA | 1-bit | 19.03 | 18.98 | **33.24** (1.75x) | **26.25** (1.38x) |
| Vulkan | ternary | 11.79 | 11.79 | 7.01 (0.59x) | 6.38 (0.54x) |
| Vulkan | 1-bit | 15.98 | 16.01 | 7.56 (0.47x) | 6.28 (0.39x) |
| CPU (NEON) | ternary | 2.22 | 2.15 | 2.37 (1.07x) | 1.71 (0.79x) |
| CPU (NEON) | 1-bit | 3.41 | 3.35 | 3.00 (0.88x) | 3.35 (1.00x) |

tok/s, median over 8 prompts per class. CPU rows use 64 generated tokens
rather than 128 to keep a ~3 tok/s backend tractable; rates and
acceptance stay comparable. Full table with TTFT, prefill and acceptance:
`python3 bench/render_table.py --speedup results/bench/*.json`.

**Speculation pays off on exactly one of the three backends.** CUDA
1.37-1.75x, Vulkan 0.39-0.59x, CPU 0.79-1.07x. Same weights, same
drafter, same box.

### DSpark is a net loss on Vulkan -- on every configuration

On CUDA the drafter is worth 1.37-1.75x. On Vulkan it costs more than it
saves in all four cells, down to 0.39x.

The cause is not draft quality. **Acceptance on Vulkan matches CUDA to
within about a point everywhere** (54.5 vs 54.0, 62.4 vs 61.6, 44.3 vs
43.3, 51.8 vs 51.4). The drafter makes the same predictions; the backend
cannot execute draft-and-verify cheaply. Each speculation round is a
small dependent dispatch, and Vulkan's submit/synchronise cost is a far
larger share of a step than a CUDA kernel launch. Degraded acceptance
would have implied a drafter problem -- identical acceptance points at
dispatch overhead.

This is the measurement the directive anticipated might be the Vulkan
deliverable, and it is: speculative decoding as currently structured does
not pay for itself in Vulkan's dispatch model. It also means acceptance
is backend-independent, which is a useful invariant for gating.

### CPU (ARM NEON) straddles break-even -- the adaptive-disable case

On Thor's own ARM cores, DSpark lands almost exactly at break-even and
which side it falls on depends on the workload:

| Variant | Workload | native | DSpark | speedup | acceptance |
| :-- | :-- | --: | --: | --: | --: |
| ternary | code | 2.22 | 2.37 | 1.07x | 57.7% |
| ternary | prose | 2.15 | 1.71 | **0.79x** | 40.2% |
| 1-bit | code | 3.41 | 3.00 | **0.88x** | 56.6% |
| 1-bit | prose | 3.35 | 3.35 | 1.00x | 52.6% |

This is the regime the planned adaptive draft-disable heuristic exists
for -- and notably it is *not* the regime the published data predicted.
The documented risk was 1-bit on prose at ~51% acceptance; measured here,
that cell is exactly 1.00x while ternary *prose* (0.79x) and 1-bit *code*
(0.88x) are the losses. A heuristic keyed to a fixed acceptance threshold
would get this wrong: 1-bit code loses at 56.6% acceptance while ternary
code wins at 57.7%. Breakeven depends on the ratio of draft cost to
target cost on that backend, not on acceptance alone.

x86 AVX-512 is **not** tested -- Thor is ARM. The fork already carries
AVX512-VNNI repack kernels for Q1_0 and Q2_0, so the code exists, but
validating it needs a borrowed x86 host.

### CPU reproduces the CUDA oracle; Vulkan does not

Gated against the CUDA traces, the two non-CUDA backends fail in
qualitatively different ways, and the difference matters:

| Backend | traces reproducing oracle | worst drift on those | character |
| :-- | :-- | --: | :-- |
| CPU (NEON) | 13/16 | 0.058 | numerical noise |
| Vulkan | 1/16 | 3.586 | broken output |

CPU matches the oracle token-for-token on every code prompt and most
prose prompts, with drift at the 0.01-0.06 level -- ordinary quant and
reduction-order noise. Its 3 prose divergences are greedy tie-flips where
accumulated noise crosses a near-equal choice, which is expected across
different kernels. Vulkan, by contrast, diverges at token 0 on 5 traces
and emits degenerate text. Different backends, different verdicts:
**CPU is a credible backend, Vulkan is not yet.**

(An earlier version of this gate reported CPU as 16/16 divergent. That was
a bug in the comparison tool, which counted the CPU run's shorter output
as divergence rather than as a prefix. Fixed; the tool now reports
`PREFIX-MATCH`.)

### Vulkan correctness is NOT established

Gated against the CUDA oracle, 15/16 traces diverge, 5 of them at token 0.
Two distinct problems:

- **With `n_probs` requested, Vulkan degenerates** -- it emits
  `{\n{\n{\n{...` where CUDA emits correct code. This is what fails the
  gate, and it affects the trace-capture path.
- **Without `n_probs` it is broadly sane** -- output lengths track CUDA
  across nearly every prompt, so the throughput numbers above stand. But
  one prompt still terminates after a single empty token on ternary where
  CUDA generates normally.

So: Vulkan performance is reportable here, Vulkan correctness is not. The
fork ships a `test-vulkan-q2_0-shader-sim`, so Q2_0 support is intended,
which makes this look like a real gap in the fork's Vulkan path for
Bonsai's low-bit formats rather than an unsupported setup. Worth
reporting upstream -- by a human, since the fork does not accept
AI-generated contributions.

## Energy per token (Phase 6)

From Jetson's own power rails via `tegrastats`. `VIN` is whole-board
draw, so this is comparable only against other whole-device figures.
CUDA backend, 256-token sustained decode.

| Config | idle W | busy W | tok/s | mWh/token | marginal mWh/token |
| :-- | --: | --: | --: | --: | --: |
| ternary native | 20.2 | 67.7 | 16.60 | 1.133 | 0.794 |
| ternary DSpark | 20.5 | 63.6 | 29.29 | **0.603** | 0.409 |
| 1-bit native | 20.7 | 61.9 | 18.66 | 0.921 | 0.613 |
| 1-bit DSpark | 21.0 | 64.7 | 32.83 | **0.548** | 0.370 |

**DSpark nearly halves energy per token** (1.133 -> 0.603 ternary, 0.921
-> 0.548 for 1-bit). It raises throughput at essentially unchanged board
power, so the speedup converts almost directly into efficiency. That is a
better argument for speculation on an edge device than the tok/s number
alone.

Marginal energy subtracts the ~20 W idle floor, which is a large share of
Thor's ~65 W busy draw. Which column is right depends on the question:
total for "what does this box cost to run", marginal for "what does
serving one more token cost".

For scale, PrismML publish 0.275 mWh/token on an M5 Pro. Thor's best here
is 0.548 total / 0.370 marginal -- roughly 1.3-2x that figure. A dev kit
with a 20 W idle floor is not built for the same efficiency point as a
laptop SoC, so this is context rather than a defeat.

## Cross-architecture compile verification (Phase 3)

| Arch | Target | Result |
| :-- | :-- | :-- |
| sm_87 | Jetson Orin / Orin Nano Super | compiles |
| sm_86 | RTX 3090 | compiles |
| sm_110 | Jetson Thor | compiles, **run and validated** |
| sm_120a | RTX 5090 | compiles |
| sm_121a | DGX Spark | compiles |

Reproduce with `cuda/arch/verify_arches.sh`. Compiling is not a
correctness or performance claim -- only sm_110 was executed. Everything
else is unvalidated until run on the real part.

### Published baselines for comparison

From the model cards, measured by PrismML on other hardware. Context, not
this repo's results:

| Platform | Variant | Native tok/s | + DSpark | Speedup |
| :-- | :-- | --: | --: | --: |
| H100 | ternary | 98.0 | 131.8 | 1.34x |
| H100 | 1-bit | 104.8 | 143.8 | 1.37x |
| Apple M5 Max | ternary | 44.0 | not enabled | -- |
| Apple M5 Pro | ternary | 26.2 | not enabled | -- |

### Correctness finding: DSpark is not token-identical to native

Speculative decoding is expected to be lossless, so at temperature 0 a
DSpark run should emit exactly the tokens a native run emits. **On Thor
it does not.** The divergence is reproducible, occurs on a freshly
started server's first request, and is not an artifact of trace capture:

```
native: for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
dspark: for (int offset = warpSize / 2;   offset > 0; offset /= 2)
```

It affected 5/16 ternary and 6/16 1-bit trace pairs. This does not
invalidate the throughput above and does not indicate worse output --
both continuations are reasonable. It does mean native output cannot
serve as the oracle for a DSpark run, so every backend gate here compares
like mode against like mode. Whether the cause is Thor-specific numerics
or general to the fork needs a second device to separate.
Detail in `docs/METHODOLOGY.md`.

## What is not done, and why

Being explicit so the tables above are not read as more than they are.

**Phase 2 -- the custom CUDA engine with DSpark fused -- has not been
started.** It is the core build of this repo and it is a large one: a
weight loader for the packed Q2_0/Q1_0 formats, GDN linear-attention
kernels for 48 layers, GQA-with-QK-norm kernels for the other 16, and a
single-launch decode step with draft-and-verify inside one graph replay.
Everything published here so far measures the *reference fork*, not any
engine of ours. The work below it is now in place -- oracle traces,
gates, benchmark harness, memory accounting -- so the engine has
something to be judged against, which was the point of doing it first.

Three findings from this phase feed directly into that build:

1. The drafter's full-context staging buffer is what blocks 8 GB devices.
   Sizing it to the draft block is a concrete, measurable win.
2. Speculation's benefit is entirely a function of per-step dispatch
   cost. Fusing draft+verify into one launch attacks exactly the overhead
   that makes Vulkan lose and CPU break even.
3. An adaptive draft-disable heuristic cannot key on acceptance; it has
   to measure realised throughput both ways.

**Blocked on hardware not present here:**

- Orin Nano Super (sm_87) and RTX (sm_86/120a/121a) are compile-verified
  only. No run, no correctness claim, no performance claim.
- x86 AVX-512 is untested; Thor is ARM.
- The cross-*device* scaling table (the full GAP 3) needs a second
  device. What exists today is a cross-*backend* table on one device.

**Open:** whether the temperature-0 DSpark divergence is Thor-specific
numerics or general to the fork. Separating those needs a second device.

## Layout

```
reference/     wrappers around the PrismML fork -- the correctness oracle
  setup_prism_fork.sh    build the fork per backend
  serve_reference.sh     canonical llama-server invocations
  capture_traces.py      capture + compare per-token logprob traces
  run_milestone0.sh      full variant x mode x workload sweep
bench/         backend-agnostic harness
  bench.py               tok/s, TTFT, acceptance %
  measure_memory.sh      resident footprint (peak RSS; see methodology)
  gpu_mem.cu             cudaMemGetInfo helper, since nvidia-smi is blind on Jetson
  render_table.py        generates the results tables from run artifacts
  workloads/             code (high-accept) AND prose (low-accept)
cuda/          primary backend, developed on Thor      [not started]
vulkan/        hardware-agnostic GPU backend           [not started]
cpu/           x86 AVX-512 + ARM NEON                  [not started]
docs/METHODOLOGY.md      portable lessons, per backend
```

## Quickstart

```bash
# 1. Build the reference fork (clones it if absent)
./reference/setup_prism_fork.sh cuda

# 2. Fetch the weights (~15 GB for both variants + both drafters)
hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf \
    --local-dir ~/models/bonsai/ternary
hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-dspark-Q4_1.gguf \
    --local-dir ~/models/bonsai/ternary
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf \
    --local-dir ~/models/bonsai/onebit
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-dspark-Q4_1.gguf \
    --local-dir ~/models/bonsai/onebit

# 3. Run the full reference sweep
./reference/run_milestone0.sh
```

## Prior art and credit

- **[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)**,
  `prism` branch -- the reference implementation, the source of the
  ternary `Q2_0_g128` hybrid-attention CUDA kernels, and the correctness
  oracle for everything in this repo. `Q1_0` is merged upstream, so stock
  llama.cpp serves as a second independent reference for the 1-bit
  variant.
- **[RightNow-AI/bonsai-turbo](https://github.com/RightNow-AI/bonsai-turbo)**
  -- prior art for single-launch batch-1 Bonsai decode (Apache 2.0,
  reports 1.76x over the vendor fork on H100). This repo studies and
  credits that design rather than silently reimplementing it; the
  differentiation is DSpark fused into the same launch, Thor validation,
  and the multi-backend story.

Model weights, the Q2_0/Q1_0 formats, and the DSpark drafters are
PrismML's work, under their published licenses. See `NOTICE` in each
model repo.
