# Methodology

Portable lessons, per backend. This file records what was actually
measured or observed on hardware, and what remains unverified. Anything
not yet measured is marked as such rather than estimated.

## Correctness gating

The PrismML fork (`prism` branch) is the correctness oracle for every
backend here. The gate is a two-part comparison, implemented in
`reference/capture_traces.py --compare`:

1. **Token match.** At `temperature 0, top_k 1` decoding is deterministic,
   so a candidate backend must emit the identical greedy token sequence
   for every workload prompt. This is the strict gate.
2. **Logprob drift.** Max absolute difference on the top-1 logprob per
   position. This is allowed to be nonzero -- different reduction orders
   and dequant paths produce small numerical differences -- but is
   reported alongside the token match so drift can be tracked rather than
   hidden. Default tolerance `0.05`.

Speculative decoding is lossless by construction: verification preserves
the target distribution exactly. So a DSpark run and a native run of the
same target model must produce the *same* tokens at temperature 0. That
makes native-vs-DSpark trace comparison a self-check on the drafting
implementation, independent of any cross-backend comparison.

## Benchmark protocol

- Both workload classes are always run. `bench/workloads/code.jsonl` is
  the high-acceptance regime; `bench/workloads/prose.jsonl` is the
  low-acceptance regime where the model card reports acceptance falling
  to ~51% and where 1-bit + DSpark can be *slower* than native. Reporting
  only code numbers would misrepresent the drafter.
- Raw `/completion` is used rather than the chat endpoint. Bonsai is a
  thinking model; going through the chat template makes it emit reasoning
  prose even for code prompts, which blurs the very distinction the two
  workload classes exist to measure. Raw completion keeps the two regimes
  clean.
- `cache_prompt: false` on every request, so each row is a cold decode
  and tok/s is not inflated by prefix reuse.
- Acceptance is pooled over all tokens in a workload
  (`sum(accepted) / sum(drafted)`), not averaged over per-request rates,
  so long requests weigh proportionally.
- Decode tok/s, prefill tok/s and token counts are taken from the
  server's own `timings` object. TTFT is measured client-side as
  wall-clock to the first streamed content chunk, which includes
  request and queueing overhead a server-side number would miss.

## CUDA / NVIDIA architecture notes

### Jetson Thor is compute capability 11.0, and that has consequences

Thor reports CC 1100 (`sm_110`). ggml's architecture constants place it
between Hopper and Blackwell:

```
GGML_CUDA_CC_HOPPER      900
GGML_CUDA_CC_BLACKWELL  1200   <- Thor (1100) is BELOW this
GGML_CUDA_CC_DGX_SPARK  1210
```

The fork's `common.cuh` states this deliberately: Blackwell spans CC
1000, 1100 and 1200, but the tensor-core instructions integrated are the
sm_120-family ones. So although Thor is a Blackwell-generation part, it
does not take the Blackwell tensor-core paths -- it falls through to the
Ampere/Turing MMA paths. This is intended behaviour, not a bug, but it
means **Thor numbers should not be read as representative of Blackwell
tensor-core performance.**

### `110-real` is the correct build, and no CMake patch is needed

The fork's `ggml/src/ggml-cuda/CMakeLists.txt` rewrites plain `12X`
architectures to `12Xa` ("architecture-specific"), because Blackwell's
FP4 tensor-core instructions are not forward-compatible. The rewrite
regex matches `^12[0-9]` only, so it does not touch `11X`.

On Thor, CMake native detection yields `CMAKE_CUDA_ARCHITECTURES=110-real`.
The initial expectation was that this would need patching to `110a-real`.
It does not: because ggml gates every Blackwell-specific code path on
`CC >= 1200`, a CC 1100 part never reaches an instruction that requires
the `a` (arch-specific) variant. `110-real` is therefore a complete and
correct build, and the fork needs no change to build on sm_110.

`nvcc` 13.0 does accept `-arch=sm_110a` if an arch-specific path is ever
needed; that was verified directly. It simply is not needed today.

### Toolkit / driver actually used

| Component | Version |
| :-- | :-- |
| Device | NVIDIA Thor, CC 11.0 (`sm_110`) |
| Driver | 580.00 |
| CUDA (driver) | 13.0 |
| `nvcc` | 13.0.48 |
| CMake | 3.28.3 |
| gcc | 13.3.0 |
| Host | 14 cores, 122 GiB unified memory |

### Memory reporting on Jetson is not `nvidia-smi`

`nvidia-smi` reports `Not Supported` for memory usage on Thor: CPU and
GPU share one physical pool, so there is no separate GPU memory figure to
report. The bench harness records host memory from `/proc/meminfo`
(`MemTotal - MemAvailable`) as the meaningful resident-footprint number,
and leaves the GPU field null. Any cross-device memory column must
compare like with like -- a discrete-GPU row's "GPU memory" and a
Jetson row's "resident memory" are different measurements.

## Vulkan

Not yet started.

## CPU

Not yet started. Note that Thor's CPU is ARM, so the AVX-512 path cannot
be self-tested here; it needs a borrowed x86 host. The fork already
carries AVX512-VNNI repack GEMV/GEMM for Q1_0 and Q2_0 (fork commit
`9fcaed763`), so the x86 reference exists even though this repo cannot
run it locally.

## Prior art

- **RightNow-AI/bonsai-turbo** -- single-launch batch-1 CUDA decode
  engine, Apache 2.0, reports 1.76x over the vendor fork on H100 with
  matching outputs. It does **not** integrate DSpark, and publishes no
  Thor/sm_110 numbers. Its single-launch design is the prior art this
  repo's CUDA engine builds on; the differentiation is DSpark fused into
  the same launch, Thor validation, and the multi-backend story.
- **PrismML-Eng/llama.cpp `prism` branch** -- the reference
  implementation and correctness oracle. Note its `AGENTS.md` prohibits
  AI-generated pull requests; any patch this work produces for the fork
  must be authored and submitted by a human contributor who understands
  it.
