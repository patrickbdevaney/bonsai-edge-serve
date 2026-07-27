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

Peak RSS (`VmHWM`) initially looked like the answer -- 7158 MiB for the
7.17 GB ternary build, 3952 MiB for the 3.80 GB 1-bit build, each
matching its GGUF. **That agreement was a coincidence.** `smaps` shows
only 322 MiB of the model mapping actually resident: `VmHWM` was
measuring transient page-cache traffic while weights were copied into
backend buffers, not steady-state residency. The tell was that it is not
additive -- attaching an identical drafter moved it +22 MiB in one
configuration and +2767 MiB in another.

**What works is asking ggml.** Every backend buffer is logged at `-lv 10`:

```
load_tensors:            CUDA0 model buffer size   = 6500.64 MiB
load_tensors:       CPU_Mapped model buffer size   =  322.07 MiB
llama_kv_cache:          CUDA0 KV buffer size      = 1024.00 MiB
llama_memory_recurrent:  CUDA0 RS buffer size      =  149.62 MiB
sched_reserve:           CUDA0 compute buffer size =  138.02 MiB
```

Exact, additive, attributable to target vs drafter.
`bench/measure_memory.py` parses it. Two subtleties: the fork runs a
parameter-fitting dry pass whose buffers are also logged (real
allocations are those after `mmap = true`), and `loading draft model`
separates target from drafter.

Independently validated: ternary native device buffers sum to 7812.28
MiB, exactly the fork's own `projected to use 7812 MiB`.

### DSpark costs about 4.7x what the server estimates

The server logs `estimated memory usage of draft model is 1802.06 MiB`.
Measured at ctx 16384 it adds ~8.5 GB:

| Variant | native | DSpark | delta |
| :-- | --: | --: | --: |
| ternary | 8171 MiB | 16729 MiB | +8558 (2.05x) |
| 1-bit | 4965 MiB | 13372 MiB | +8407 (2.69x) |

The dominant term is not weights. It is the drafter's compute buffers
(2432 MiB device + 2752 MiB host), because the fork raises the draft
context's batch to the full context -- it logs `draft-dspark: raising
draft ctx n_batch 2048 -> 16388 (full-context staging + block)`. That is
why the overhead scales with `-c`.

Two non-obvious contributions land on the *target*, not the drafter: its
recurrent-state buffer grows 149.62 -> 748.12 MiB and its compute buffer
138.02 -> 863.64 MiB, together ~1.3 GB, because block verification needs
more GDN state slots.

### This changes the Phase 3 plan for Orin Nano Super

Measured totals by context (MiB):

| config | 2048 | 4096 | 8192 | 16384 |
| :-- | --: | --: | --: | --: |
| ternary native | 7247 | 7379 | 7643 | 8171 |
| ternary DSpark | 10988 | 11713 | 13257 | 16729 |
| 1-bit native | 4041 | 4173 | 4437 | 4965 |
| 1-bit DSpark | 7631 | 8355 | 9899 | 13372 |

The directive assumed 1-bit + drafter fits an 8 GB Orin Nano Super and
ternary + drafter is tight. Measured, **neither DSpark configuration
fits**: the cheapest wants 7631 MiB on a device that must also hold the
OS. Ternary native is marginal at best even at ctx 2048. Only 1-bit
native fits comfortably (4041-4437 MiB).

The blocker is the full-context drafter staging buffer, not the weights.
Sizing that buffer to the draft block rather than the whole context is
what would make DSpark viable on 8 GB, and it is a concrete target for
this repo's own engine. Full detail in `results/memory.txt`.

### Power via tegrastats, and the pitfall in measuring it

`tegrastats` exposes per-rail power on Jetson (`VDD_GPU`,
`VDD_CPU_SOC_MSS`, `VIN_SYS_5V0`, `VIN`). `VIN` is whole-board draw and
is the only figure comparable against another whole device.
`bench/measure_power.sh` samples idle and busy separately and divides by
the measured decode rate.

Measured, CUDA backend, 256-token sustained decode:

| Config | idle W | busy W | tok/s | mWh/token | marginal |
| :-- | --: | --: | --: | --: | --: |
| ternary native | 20.2 | 67.7 | 16.60 | 1.133 | 0.794 |
| ternary DSpark | 20.5 | 63.6 | 29.29 | 0.603 | 0.409 |
| 1-bit native | 20.7 | 61.9 | 18.66 | 0.921 | 0.613 |
| 1-bit DSpark | 21.0 | 64.7 | 32.83 | 0.548 | 0.370 |

DSpark nearly halves energy per token because it raises throughput at
essentially unchanged board power. Report both columns: Thor's ~20 W idle
floor is a third of its busy draw, so total and marginal energy answer
different questions.

Two measurement pitfalls, both hit here:

1. **Do not sample power while anything else runs.** An early reading
   showed 33 W "idle" because a compile was running in the background.
   Every figure above comes from a run with nothing else scheduled.
2. **Check the decode actually produced tokens.** The first version of
   the script used a prose instruction as its prompt; this model emits
   EOS immediately on several such prompts, giving zero decode time, a
   sentinel rate of 1e6 tok/s, and an empty power capture. The script now
   uses a code prompt and rejects any run producing under 32 tokens
   rather than reporting a nonsense energy figure.

## Cross-architecture compile verification

`cuda/arch/verify_arches.sh` builds the `ggml-cuda` target -- where all
architecture-dependent code lives -- for each target arch.

| Arch | Target | Result |
| :-- | :-- | :-- |
| sm_87 | Jetson Orin / Orin Nano Super | compiles |
| sm_86 | RTX 3090 | compiles |
| sm_110 | Jetson Thor | compiles, run, validated |
| sm_120a | RTX 5090 | compiles |
| sm_121a | DGX Spark | compiles |

Compiling is not correctness and not performance. Only sm_110 was
executed. The `a` suffix is required for 120/121 because Blackwell's FP4
tensor-core instructions are not forward-compatible; 87, 86 and 110 need
no suffix, for the reason given in the sm_110 section above.

## Vulkan

Built from the same fork (`-DGGML_VULKAN=ON`) and run on Thor's own GPU
through the Vulkan driver (1.4.315), so this is a backend comparison on
identical hardware -- not a device comparison. Build needs `glslc`
(`apt install glslc`) and `spirv-headers`; neither is pulled in by the
CUDA build.

### DSpark is a net loss on Vulkan, on every configuration

This is the headline. On CUDA the drafter is worth 1.37-1.75x. On Vulkan
it costs more than it saves, every time:

| Variant | Workload | native | DSpark | speedup | acceptance |
| :-- | :-- | --: | --: | --: | --: |
| ternary | code | 11.79 | 7.01 | **0.59x** | 54.5% |
| ternary | prose | 11.79 | 6.38 | **0.54x** | 44.3% |
| 1-bit | code | 15.98 | 7.56 | **0.47x** | 62.4% |
| 1-bit | prose | 16.01 | 6.28 | **0.39x** | 51.8% |

The cause is not draft quality. Acceptance on Vulkan matches CUDA to
within about a point in every cell (54.5 vs 54.0, 62.4 vs 61.6, 44.3 vs
43.3, 51.8 vs 51.4). The drafter is making the same predictions; the
backend simply cannot execute the draft-and-verify pattern cheaply. Each
speculation round is a small dependent dispatch, and Vulkan's
submit/synchronise overhead is a much larger fraction of a step than
CUDA's kernel launch. Worse acceptance would have meant a drafter
problem; identical acceptance points squarely at dispatch cost.

That acceptance is backend-independent is itself worth recording: it
confirms acceptance is a property of the drafter and token distribution,
not of the hardware or backend.

### Vulkan native is 63-84% of CUDA

| Variant | CUDA | Vulkan | ratio |
| :-- | --: | --: | --: |
| ternary code | 16.77 | 11.79 | 0.70x |
| 1-bit code | 19.03 | 15.98 | 0.84x |

Expected -- this backend's claim is reach, not peak. Note the 1-bit build
closes most of the gap, consistent with a bandwidth-bound decode where
the smaller weights matter more than kernel quality.

### Vulkan does NOT match CUDA output on these models

Gated against the CUDA oracle, same variant and mode, 15/16 traces
diverge, with 5 diverging at token 0 -- the very first forward pass. That
is not accumulated rounding.

Two separate problems, and they need separating because they have
different consequences:

1. **With `n_probs` requested, Vulkan degenerates.** The trace capture
   asks for top-5 logprobs, and under that request Vulkan emits
   repetitive garbage where CUDA emits correct code:

   ```
   CUDA  : "    extern __shared__ float sdata[];\n    int tid = ..."
   Vulkan: "{\n{\n{\n{\n{\n{\n{\n{\n{\n{\n{\n..."
   ```

   This is what fails the gate. It affects trace capture only.

2. **Without `n_probs`, Vulkan is broadly sane but not identical.** The
   benchmark path requests no probabilities, and there output lengths
   track CUDA closely (350-490 chars, full 128 tokens, across nearly
   every prompt). The throughput numbers above are therefore
   trustworthy. But one prompt (`prose/coastal-town`, ternary only)
   terminates after a single token with empty output on Vulkan while
   CUDA produces a normal 128-token continuation -- so a real behavioural
   difference remains even without `n_probs`.

The consequence for this repo: **Vulkan performance numbers are
reportable, Vulkan correctness is not established.** The fork ships a
`test-vulkan-q2_0-shader-sim` binary, so Q2_0 support is intended rather
than accidental, which makes this look like a genuine gap in the fork's
Vulkan path for Bonsai's low-bit formats rather than an unsupported
configuration. It is worth reporting upstream, with the caveat that the
fork does not accept AI-generated contributions -- a human needs to
confirm and file it.

## CPU (ARM NEON)

Thor's CPU is ARM, so the NEON path is testable here and the AVX-512 path
is not; that one needs a borrowed x86 host and is left explicitly
unvalidated. The fork already carries AVX512-VNNI repack GEMV/GEMM for
Q1_0 and Q2_0 (commit `9fcaed763`), so the x86 code exists even though
this repo cannot exercise it.

Run with `-ngl 0` against the existing build -- the CPU backend is always
compiled in, so no separate build is needed. `system_info` confirms the
relevant paths are active: `NEON = 1, ARM_FMA = 1, MATMUL_INT8 = 1,
SVE = 1, SVE_CNT = 16, DOTPROD = 1, REPACK = 1`.

CPU decode is slow enough (~3.6 tok/s on 1-bit) that the CPU sweep uses
`N_PREDICT=64` rather than 128 to keep the run tractable. tok/s, TTFT and
acceptance remain directly comparable across backends -- only the number
of tokens generated per request differs.

### That shorter run exposed a bug in the gate

With the CPU traces at 64 tokens and the oracle at 128, the comparison
tool reported **16/16 divergent** for both variants. That was wrong: it
was treating the length difference itself as divergence. Every trace
actually agreed on all 64 tokens they had in common, with drift of
0.01-0.06. Fixed -- `--compare` now reports `PREFIX-MATCH` when one run
is a strict prefix of the other, and the real figure is 3/16 divergent.

Worth recording as a gate-design lesson: a correctness gate that can
report total failure for a benign configuration difference will be
disbelieved when it reports real failures. The Vulkan result was only
trustworthy because it was checked against this one.

### CPU reproduces the oracle; the divergences are tie-flips

| Backend | reproducing oracle | worst drift on those | character |
| :-- | :-- | --: | :-- |
| CPU (NEON) | 13/16 | 0.058 | numerical noise |
| Vulkan | 1/16 | 3.586 | broken output |

All 8 code prompts prefix-match exactly. The 3 prose divergences
(`coastal-town@41`, `history-canal@29`, `philosophy-memory@6`) are what
accumulated numerical noise looks like under greedy decoding: prose is
higher-entropy, so a near-tie between two candidate tokens eventually
flips. Note the tolerance: the worst drift among *matching* traces is
0.058, just above the default 0.05, so cross-backend gating wants a
tolerance around 0.10 while same-backend regression gating can stay
tight.

### DSpark on CPU straddles break-even, and not where predicted

| Variant | Workload | native | DSpark | speedup | acceptance |
| :-- | :-- | --: | --: | --: | --: |
| ternary | code | 2.22 | 2.37 | 1.07x | 57.7% |
| ternary | prose | 2.15 | 1.71 | 0.79x | 40.2% |
| 1-bit | code | 3.41 | 3.00 | 0.88x | 56.6% |
| 1-bit | prose | 3.35 | 3.35 | 1.00x | 52.6% |

This is the regime an adaptive draft-disable heuristic is for. It also
shows a threshold on acceptance alone would be the wrong trigger: 1-bit
code *loses* at 56.6% acceptance while ternary code *wins* at 57.7%.
Breakeven is set by the ratio of draft-step cost to target-step cost on
that backend, so the heuristic must measure realised tok/s with drafting
on versus off, not infer it from acceptance.

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
