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
| 1 | Reference stack + captured oracle traces | in progress |
| 1 | **Milestone 0** -- first Thor numbers, reference fork | pending measurement |
| 2 | CUDA engine, DSpark fused into a single launch | not started |
| 3 | Cross-arch CUDA targets (sm_87 Orin, sm_86, sm_120/121) | not started |
| 4 | Vulkan backend | not started |
| 5 | CPU backend (AVX-512, NEON) | not started |
| 6 | Cross-device scaling table | not started |

No performance numbers are published in this README until they have been
measured on the hardware named in the row. Nothing here is estimated.

## Milestone 0 -- Jetson Thor reference numbers

*Pending measurement. The harness and oracle capture are in place; the
table below is populated by `reference/run_milestone0.sh` and will be
filled from `results/bench/*.json`.*

Device: NVIDIA Thor, CC 11.0, driver 580.00, CUDA 13.0, 122 GiB unified
memory. Engine: PrismML `llama.cpp` fork, `prism` branch, built
`110-real`. No custom engine involved -- this measures the reference
implementation only.

| Variant | Mode | Workload | tok/s | TTFT (s) | Acceptance | Resident |
| :-- | :-- | :-- | --: | --: | --: | --: |
| ternary Q2_0 | native | code | | | n/a | |
| ternary Q2_0 | native | prose | | | n/a | |
| ternary Q2_0 | DSpark | code | | | | |
| ternary Q2_0 | DSpark | prose | | | | |
| 1-bit Q1_0 | native | code | | | n/a | |
| 1-bit Q1_0 | native | prose | | | n/a | |
| 1-bit Q1_0 | DSpark | code | | | | |
| 1-bit Q1_0 | DSpark | prose | | | | |

### Published baselines for comparison

From the model cards, measured by PrismML on other hardware. These are
context, not this repo's results:

| Platform | Variant | Native tok/s | + DSpark | Speedup |
| :-- | :-- | --: | --: | --: |
| H100 | ternary | 98.0 | 131.8 | 1.34x |
| H100 | 1-bit | 104.8 | 143.8 | 1.37x |
| Apple M5 Max | ternary | 44.0 | not enabled | -- |
| Apple M5 Pro | ternary | 26.2 | not enabled | -- |

Reported DSpark acceptance: ~69.2% ternary, 74.6-78.6% 1-bit on code,
falling to ~51% on prose -- where the 1-bit target's weights are small
enough that drafting overhead can make DSpark a net *loss*. Both regimes
are measured here; see `docs/METHODOLOGY.md`.

## Layout

```
reference/     wrappers around the PrismML fork -- the correctness oracle
  setup_prism_fork.sh    build the fork per backend
  serve_reference.sh     canonical llama-server invocations
  capture_traces.py      capture + compare per-token logprob traces
  run_milestone0.sh      full variant x mode x workload sweep
bench/         backend-agnostic harness
  bench.py               tok/s, TTFT, acceptance %, memory
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
