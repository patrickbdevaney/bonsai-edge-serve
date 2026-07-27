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
| 3 | Cross-arch CUDA targets (sm_87 Orin, sm_86, sm_120/121) | not started |
| 4 | Vulkan backend | not started |
| 5 | CPU backend (AVX-512, NEON) | not started |
| 6 | Cross-device scaling table | not started |

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
