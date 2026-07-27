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

**Gate the same mode against the same mode**: a candidate backend's
native capture against the fork's native capture, its DSpark capture
against the fork's DSpark capture. Native-vs-DSpark is *not* a valid
gate here, for two measured reasons.

### DSpark output diverges from native greedy output at temperature 0

The expectation going in was that speculative decoding is lossless --
verification preserves the target distribution -- so at `temperature 0,
top_k 1` a DSpark run and a native run of the same target should emit
identical tokens. On Thor they do not.

Measured, ternary Q2_0, workload prompt `code/cuda-reduce`, 128 tokens,
each request the *first* on a freshly started server:

```
native: for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
dspark: for (int offset = warpSize / 2;   offset > 0; offset /= 2)
```

The divergence is reproducible, lands at the same character offset (317),
and is not an artifact of trace capture: native output is byte-identical
with and without `n_probs` requested, and both native and DSpark are
individually deterministic across repeated runs. Divergence appeared in
5/16 ternary and 6/16 1-bit trace pairs.

Note this was nearly missed. A first check at 64 tokens on a shortened
prompt showed an exact match -- the divergence simply had not occurred
yet by that point. Losslessness checks need to run long enough to reach
one.

What this does *not* mean: it does not invalidate the throughput numbers,
and it does not show the DSpark output is worse -- both continuations
above are reasonable code. What it means is that on this fork, at this
architecture, DSpark is not token-identical to native decoding, so
native output cannot be used as the oracle for a DSpark run. Whether this
is Thor-specific numerics or general to the fork is untested; it needs a
second device to separate those.

### DSpark captures report placeholder logprobs

In a DSpark capture every token after the first carries `logprob = 0.0`.
Only the first token of a request -- the one not produced through the
draft path -- has a real value. Any logprob-drift comparison against a
DSpark capture is therefore measuring the placeholder. `--compare`
detects this and falls back to a token-only comparison rather than
reporting a meaningless drift figure.

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
- Acceptance uses the fork's own definition, so it is directly comparable
  to the published figures. In `server-context.cpp` the accepted counter
  is incremented as `n_draft_accepted += ids.size() - 1`: the `- 1` drops
  the free bonus token that verification always yields, leaving only
  genuinely accepted *draft* tokens. The server logs
  `n_draft_accepted / n_draft_total` itself, which is the same ratio this
  harness reports. This also reconciles the model card's two numbers: at
  draft depth `k = 4` with ~69% acceptance, ~2.8 draft tokens are accepted
  per step, plus the bonus token gives the quoted accepted length
  `tau ~ 3.7`.
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

### Measuring resident memory on unified memory is its own problem

Four obvious methods all give wrong answers on Thor, because the GPU
shares the system memory pool and weights are mmap'd:

| Method | Result for the 7.2 GB ternary build | Why it fails |
| :-- | :-- | :-- |
| `nvidia-smi` | "Not Supported" | no separate GPU pool to report |
| host `MemTotal - MemAvailable` | ~28 GiB | measures the whole machine; read the same for the 3.8 GB model |
| `cudaMemGetInfo` delta | ~450 MiB | weights are unified-memory pages, not a device allocation |
| `--no-mmap` + RSS | 796 MiB | weights go into CUDA allocations absent from RSS |
| instantaneous `VmRSS` | ~800 MiB | kernel evicts untouched mmap pages |

What works for a single model is the server process's **peak** RSS
(`VmHWM`), which captures every weight page actually touched: 7158 MiB
for the 7.17 GB ternary build and 3952 MiB for the 3.80 GB 1-bit build,
both matching their GGUF. That agreement is the validation.

It stops working once the drafter is attached. With two mmap'd models the
kernel reclaims pages from one mapping while faulting in the other, so
peak RSS is no longer a sum: the ternary DSpark config measured only
+22 MiB over native while the 1-bit config measured +2767 MiB, despite
both servers logging the same `estimated memory usage of draft model is
1802.06 MiB` and both demonstrably running the drafter. The README
reports the server's own estimate for DSpark rows rather than a measured
number known to be wrong. Full detail in `results/memory.txt`.

A trustworthy two-model footprint on unified memory remains an open item,
and it matters: whether ternary + drafter fits the 8 GB Orin Nano Super
is exactly the question Phase 3 has to answer.

### Power is measurable via tegrastats

`tegrastats` exposes per-rail power on Jetson (`VDD_GPU`,
`VDD_CPU_SOC_MSS`, `VIN_SYS_5V0`, `VIN`), which is what the Phase 6
energy-per-token column will be built from. Idle GPU draw observed around
2.4 W, whole-board `VIN` around 23 W under light load.

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
