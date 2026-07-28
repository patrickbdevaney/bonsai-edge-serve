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

Those are the reference `llama-server` numbers. `bonsai-server` now runs
DSpark **in-process** and measures 1.74x/1.80x on code and 1.49x/1.60x on
prose (ternary/1-bit), with greedy output character-identical to
non-speculative decode on 14 of 20 runs.

It also yields the constant worth serving by: a speculative round costs
**2.18 plain decode steps**, flat across a 2.3x range of measured speedups
and across both quantisations, so breakeven is `1 + 4*alpha = 2.18`, i.e.
**alpha* = 0.296**. Estimated from all ten runs, that constant then predicts
the single losing case (alpha 0.238 -> predicted 0.90x, measured 0.91x).
See `results/dspark-server.txt` and `server/README.md`.

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
| CUDA | ternary | 16.77 | 16.97 | **24.85 (1.48x)** | **23.23 (1.37x)** |
| CUDA | 1-bit | 19.03 | 18.98 | **33.24 (1.75x)** | **26.25 (1.38x)** |
| Vulkan | ternary | 15.74 | 16.14 | 6.81 (0.43x) | 6.24 (0.39x) |
| Vulkan | 1-bit | 19.57 | 19.60 | 8.20 (0.42x) | 6.43 (0.33x) |
| CPU (NEON) | ternary | 2.22 | 2.15 | **2.37 (1.07x)** | 1.71 (0.79x) |
| CPU (NEON) | 1-bit | 3.41 | 3.35 | 3.00 (0.88x) | 3.35 (1.00x) |

<!-- sources: Vulkan/1-bit dspark from '.mmvq' results; Vulkan/1-bit native from '.mmvq' results; Vulkan/ternary dspark from '.mmvq' results; Vulkan/ternary native from '.mmvq' results -->

The Vulkan rows were measured with the fork's default `graph_optimize`
enabled. That pass is a correctness bug (see below) and
`serve_reference.sh` now disables it by default, which costs 2.2% prefill
and 1.0% decode -- so the Vulkan figures above are optimistic by about a
point, and reproducing them needs `BONSAI_VK_GRAPH_OPTIMIZE=1`.

tok/s, median over 8 prompts per class. CPU rows use 64 generated tokens
rather than 128 to keep a ~3 tok/s backend tractable; rates and
acceptance stay comparable. Full table with TTFT, prefill and acceptance:
`python3 bench/render_table.py --speedup results/bench/*.json`.

**Speculation pays off on exactly one of the three backends.** CUDA
1.37-1.75x, Vulkan 0.33-0.43x, CPU 0.79-1.07x. Same weights, same
drafter, same box.

**Vulkan native has closed the gap to CUDA, and the 1-bit build passes
it.** The Vulkan rows moved after the integer-dot MMVQ path was
implemented for `q1_0`/`q2_0` (ternary 11.79 -> 15.74, 1-bit 15.98 ->
19.57; `results/vulkan-mmvq-q2_0.txt`). Vulkan was at 70% of CUDA on
ternary and is now at 94%; on 1-bit it is ahead, 19.57 against 19.03.
Sampling and dispatch are unchanged -- this is the weight kernel alone.

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

**The MMVQ result narrows this further.** That change lifted Vulkan native
decode by 33% and left every DSpark cell flat (ternary 7.01 -> 6.81, 1-bit
7.56 -> 8.20, acceptance unmoved). It could not have helped: verification
submits a block of drafted tokens, so `n > 1` and the work routes through
`mul_mm`, never touching the `mul_mat_vec` path that was optimised. So the
speculative path is provably not weight-GEMV-bound on this backend, and the
ratios got *worse* precisely because the denominator improved -- a cheaper
target step raises the breakeven ratio `c` in `alpha > c`.

That leaves two live candidates, and one of them is now concrete: Vulkan
has no integer-dot **MMQ** path for `q1_0`/`q2_0` (the `mul_mmq.comp` gate
at `vulkan-shaders-gen.cpp:594` excludes them), while CUDA's `mmq.cu`
supports both. CUDA therefore verifies draft blocks with an integer-dot
batch kernel and Vulkan dequantises to float. Whether closing that gap is
enough to reach break-even is untested -- it would have to be worth ~2.4x
on the verify step alone -- but it is a specific, checkable lever rather
than an attribution to dispatch cost.

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

### The Vulkan correctness gap was a RACE, and it is now root-caused

An earlier version of this section reported that Vulkan diverged from the
CUDA oracle on 15/16 traces, 5 at token 0, and blamed `n_probs`. **That
attribution was wrong in every part.** `n_probs` is irrelevant. The Vulkan
backend was *nondeterministic*, so the gate was scoring a different roll of
the dice on every run.

Ten identical requests -- temperature 0, `top_k 1`, fixed seed,
`cache_prompt=false`, one server process:

| Backend | py-binsearch | cuda-reduce | review-restaurant | coastal-town |
| :-- | --: | --: | --: | --: |
| CUDA | 1 | 1 | 1 | 1 |
| Vulkan | 2 | 7 | 8 | 3 |

Distinct outputs from identical greedy requests. Vulkan returned **8
different answers to the same question**.

Bisected to one subsystem by toggling them one at a time on the same
binary: coopmat, async and fusion all still fail 4/4;
`GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` passes 4/4. `ggml_vk_graph_optimize`
reorders nodes to expose parallelism and judges independence with
`is_src_of`, which sees only a direct `src[]` edge plus one level of
`view_src`. It cannot see a write-after-read hazard on a shared buffer
where neither node is the other's source -- and Bonsai carries recurrent
state across 48 gated-delta-net layers, which is exactly that shape. The
reorder itself is deterministic; what varies is what it lets run
concurrently without a barrier.

It is localised to **prefill**: with `cache_prompt=true` everything is
stable, because prefill runs once and later requests replay cached KV. That
is also why the bug is invisible at the server's defaults, and why
`bench/gate_determinism.sh` forces caching off.

**Fixing it costs 1-2%** -- interleaved A/B, three reps: prefill 371.2 ->
362.9 (-2.2%), decode 16.2 -> 16.0 (-1.0%).

With determinism enforced, re-capturing and re-gating against the same
oracle:

| Backend | traces reproducing oracle | worst drift on those | character |
| :-- | --: | --: | :-- |
| CPU (NEON) | 13/16 | 0.058 | numerical noise |
| Vulkan, racy (old) | 1/16 | 3.586 | broken output |
| Vulkan, deterministic | 11/16 | 0.131 | numerical noise |

No trace diverges at token 0 any more, all 16 produce a full 128 tokens,
and the 5 remaining divergences sit at tokens 29-96 -- mid-generation
greedy tie-flips, the same character as the CPU backend's 3. **Vulkan is
now the same kind of backend as CPU rather than a broken one**, which
retires the earlier verdict that "CPU is a credible backend, Vulkan is
not yet."

The gate still reports FAIL at its 0.05 tolerance, but the 5 remaining
divergences are now **explained, and they are not errors**. Every one lands
on a token where the model is essentially undecided -- top1-top2 gaps of
0.0024 to 0.0651 nats, i.e. probability ratios between 1.002 and 1.067.

Across the oracle's 2048 scored tokens the median gap is 4.712 nats, and
only 1.17% are that close. But at 1.17%, a 128-token trace has a **78%**
chance of containing one, so 12.5 of 16 traces would diverge if every tie
flipped independently. We observe 5 -- fewer than chance would give.

The decisive evidence is that **CPU and Vulkan diverge on the same two
prompts, at the same token, with the same pre-divergence drift**
(coastal-town tok 41: 0.0567 vs 0.0573; history-canal tok 29: 0.0423 vs
0.0423). Those points are properties of the model, not of a backend.

Measuring drift *before* the first divergence -- after it, the two runs are
generating different text and their logprobs are not comparable -- gives
Vulkan 0.131 worst-case against CPU's 0.058. Same order, both consistent
with quantisation and reduction-order noise.
`results/vulkan-divergence-explained.txt`.

Worth stating plainly: a correctness gate run against a nondeterministic
backend does not measure correctness. "15/16" was never a property of the
Vulkan kernels -- it was one sample, and re-running it would have given a
different number. **Establish determinism before running any comparison
gate.** Full write-up: `results/vulkan-nondeterminism.txt`.

## Long context is nearly free on this architecture

Bonsai is 48/64 gated-delta-net, and a GDN layer's recurrent state is fixed
size regardless of sequence length -- only the 16 full-attention layers grow
a KV cache. Measured across the model's full trained context:

| ctx | KV (f16) | KV (q8_0) | GDN state | total (q8_0) | tok/s |
| --: | --: | --: | --: | --: | --: |
| 4096 | 256 MiB | 136 MiB | 598 MiB | 1056 MiB | 15.79 |
| 65536 | 4096 | 2176 | 598 | 3336 | 15.77 |
| 131072 | 8192 | 4352 | 598 | 5768 | 15.77 |
| 262144 | 16384 | 8704 | 598 | 10632 | 15.83 |

**Decode throughput is flat across a 64x context increase** (15.33 -> 15.83
tok/s). On a dense transformer decode slows as the KV grows because every
attention layer rescans it; here only 16 of 64 do. The full 262144 context
runs in ~25 GiB including weights, against 122 GiB of unified memory.

`q8_0` KV halves the cache for no measurable cost -- unchanged throughput
and character-identical greedy output on a code prompt.

Two honest limits: **prefill is not free** (filling 256K still costs 256K
tokens at ~345 tok/s), and **quality at long context is untested** -- these
runs show it fits and stays fast, not that it reasons well over 256K.
`results/context-scaling.txt`.

## Serving: `bonsai-server` (pure C++, single binary)

A lean OpenAI-compatible server that links `libllama` directly -- no Python
on the hot path, no proxy hop. `server/README.md` has the detail.

```bash
BUILD_DIR=build-vulkan2 ./server/build.sh
./build/bonsai-server -m ~/models/bonsai/ternary/Ternary-Bonsai-27B-Q2_0.gguf \
    --backend vulkan --webui server/webui.html
./build/chat                     # streaming terminal client
./bench/gate_server.sh 8085      # 20-check smoke gate
```

`/v1/chat/completions` (SSE + non-streaming), `/v1/completions`,
`/v1/models`, `/metrics`, `/v1/policy`, self-contained web UI at `/`.
`<think>` blocks are routed to `reasoning_content`; `enable_thinking:false`
suppresses them for short answers.

**The policy layer is the point.** Running `llama-server` gets you an
OpenAI API too. What this binary adds is that this repo's measurements are
applied by default rather than living in a wiki you have to read first:
speculation OFF on Vulkan (measured 0.33-0.43x), determinism enforced on
Vulkan, 12 CPU threads and not 14. `GET /v1/policy` returns every decision
with the number behind it.

**Prefix caching is nearly inert here, and that is a finding.** Bonsai is
hybrid -- 48 of 64 layers are gated-delta-net, whose recurrent state cannot
be truncated by `llama_memory_seq_rm`. An ordinary LCP cache does not error;
it silently continues the previous answer. Only a pure append can reuse the
KV, which means append-only continuation reuses (21 of 23 tokens measured)
and multi-turn chat does not. An earlier draft of this README claimed a 7.7x
TTFT win from that cache -- it was measuring the broken path. See WIKI L10.

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

## Research pass: where the performance actually is

`docs/RESEARCH_PROMPT.md` is the meta-critique and research brief;
`docs/RESEARCH_FINDINGS.md` is the answer, from a five-agent literature
pass plus new measurements on this box. The five results that changed the
plan:

**1. We are at ~56% of an achievable roofline, and our kernel is already
at 90% of the machine.** We built the GEMV ladder and measured it across
shapes. Best streaming rate is **212-220 GB/s**, against a measured
achievable read bandwidth of **244.7 GB/s** -- so the kernel itself has
little left to give. That puts ternary decode near **30 tok/s** against
the reference fork's 16.77, i.e. **~1.8x of headroom** from kernel work,
with an absolute machine ceiling around 34 tok/s.

The research pass reported 229 GB/s (84% of spec) and we could not
reproduce it at the shape claimed. The likely explanation is now
measured: the winner and the rate both depend on whether the weights fit
Thor's 32 MB L2, and L2-resident shapes reach 238-388 GB/s. A decode step
touches each matrix once across ~3.8 GB, so the streaming regime is the
real one. Full ladder and shape sweep in `kernels/README.md` and
`results/gemv-cuda-shapes.txt`.

**2. The DSpark memory blowup is fixable, measured.** Capping the
drafter's staging batch saves ~1 GB at ctx 4096 with **zero throughput
loss, zero fallbacks, identical acceptance**:

| ctx | cap | total MiB | code tok/s | acceptance | fallbacks |
| --: | --: | --: | --: | --: | --: |
| 4096 | 256 | **7320** | 33.23 | 61.6% | 0 |
| 4096 | none | 8355 | 33.87 | 61.6% | 0 |
| 2048 | 256 | **7162** | 31.89 | 61.6% | 0 |

A second ~1.3 GB is available by setting `n_rs_seq = 0` and using the
host-side state checkpoints already implemented in the fork -- untested,
and the highest-leverage remaining experiment.

**3. Zero-memory speculation is free but not a speedup here.** N-gram
self-speculation costs **byte-identical memory to native** (4173 MiB, all
variants) -- no drafter weights, no drafter KV, and no recurrent-state
blowup. But throughput is also identical (~1.00x), because n-gram pays on
input-grounded generation and both our suites are open-ended. It is the
right free fallback; it does not replace a drafter.

**4. Vulkan's loss is the pipeline barrier, not submits.** Measured here:
a barrier-separated dispatch costs ~1.4 us narrow / ~3.9 us wide vs
~0.17 us with none, and a token graph is ~2293 real dispatches -- a
**~3.9 ms/token floor**. Submits and host round-trips together are <5%.
Separately, **neither `q1_0` nor `q2_0` is in ggml-vulkan's integer-dot
lists at all**, so both run the float path while IQ1_S gets dp4a;
precedent for that fix is +78-131%.

**5. The CPU default is the worst setting.** Leaving 2 of 14 cores for the
OS is worth **1.73x (ternary) and 2.27x (1-bit)**:

| threads | ternary | 1-bit |
| --: | --: | --: |
| **12** | **3.18** | **4.85** |
| 14 (default) | 1.84 | 2.14 |

> **Superseded: Q2_0 ARM repack kernels now exist** (`patches/0004`), taking
> CPU-only ternary prefill from 4.04 to **15.76 tok/s (3.90x)** and decode
> from 3.28 to **5.08 (1.55x)**. See `results/cpu-repack.txt`.
>
> **These CPU figures are the non-repack path and understate 1-bit.** They
> were taken with `-ngl 0` on the CUDA build, where the CUDA pinned-host
> buffer type outranks the repack buffer types in `make_cpu_buft_list`, so
> the ARM repack kernels Q1_0 *does* have were never selected. Hiding the
> GPU (`CUDA_VISIBLE_DEVICES=""`) takes 1-bit decode to **6.80 tok/s** and
> prefill from 11.89 to **20.16**. `-ngl 0` bounds where the weights live,
> not where the ops run. The thread-count ratio above is unaffected.
> See `results/cpu-repack.txt`.

Fixed: the CPU path now defaults to 12 threads. The Milestone 0 CPU rows
above used the default and therefore understate that backend.

**Scope correction:** the smaller Bonsai models (8B/4B/1.7B, 0.23-2.15
GiB) exist but are **plain dense Qwen3** -- no GDN, no recurrent state, no
DSpark, 32-64K context. They are a simpler engine path and give zero GDN
coverage, so they cannot be used to iterate on the 27B's kernels.

## Phase 2 started: the decode kernels

`kernels/` holds the batch-1 low-bit GEMV for all three backends, sharing
one packed weight format (`kernels/common/bonsai_gemv.h`). See
`kernels/README.md` for the design and full results.

**CPU (ARM NEON) is measured and bit-exact** against a scalar reference.
Whole-vector unpack + SDOT versus the per-element extract the current ARM
path uses: **0.37 -> 5.78 GB/s on ternary (~16x)** and **0.03 -> 2.78
GB/s on 1-bit (~90x)** single-threaded. Threaded best is 38.75 GB/s at
12 threads, and 14 threads collapses to 19.01 -- independently
reproducing the leave-cores-for-the-OS finding.

Projected onto the real model that is **5.40 tok/s ternary against
llama.cpp's tuned 3.18 (1.70x)** -- but **4.45 vs 4.85 (0.92x) on 1-bit,
a real regression**. That is the predicted outcome, not a bug: SDOT
retires 16 weights per instruction regardless of bit width, so a 1-bit
kernel shaped like a 2-bit one moves half the bytes at the same
weights/s. The fix is a lookup-table kernel, and it turns out to require
a row-interleaved weight layout -- i.e. the one place the shared-format
design may have to bend. Documented rather than rushed.

**CUDA and Vulkan compile but are unvalidated on device.** The GPU is
wedged (see below), so no performance claim is made for either.

## Blocked: the GPU needs a reboot

After the suspend/resume cycle, 201 processes are stuck in `D` state
inside the NVIDIA UVM replayable-fault handler and `nvidia_uvm`'s
refcount is pinned at 1272, so **any new CUDA or Vulkan context hangs** --
including a trivial `cudaMemGetInfo` that worked before the suspend. The
stuck processes are `rustdesk`, not this project's, and `nvidia-smi`
still reports the device healthy at 0% util. Stopping the rustdesk
service did not release them; D-state processes cannot be killed and the
module cannot be unloaded at that refcount. A reboot clears it.

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
cpu-neon-arm/  ARM NEON backend (x86 AVX-512 is separate)
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
