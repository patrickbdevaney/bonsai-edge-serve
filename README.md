# bonsai-edge-serve

**Model-specific, hardware-portable serving for PrismML's Bonsai-27B (ternary `Q2_0` and 1-bit `Q1_0`) with DSpark
speculative decoding, across CUDA, Vulkan, ARM NEON and x86 AVX**

| | |
|---|---|
| **Models** | [`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) (`Q2_0`, 7.0 GiB) and [`prism-ml/Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf) (`Q1_0`, 3.9 GiB), each with a DSpark drafter (`Q4_1`, ~1.8 GB). Low-bit builds of Qwen3.6-27B |
| **Architecture** | 64-layer hybrid: gated-delta-net (GDN) linear attention on 48 layers; GQA with QK norm and gated Q on every 4th layer; 262,144-token context |
| **Backends** | CUDA · Vulkan (vendor-agnostic SIMT) · CPU ARM NEON (DOTPROD / i8mm) · CPU x86 AVX2 / AVX-VNNI |
| **Reference engine** | PrismML [`llama.cpp` fork](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, `b9597-7529fdaaf`), plus five patches in [`patches/`](patches/) |
| **Server** | `bonsai-server`: one C++ binary linking `libllama`, OpenAI-compatible, with a measured per-backend policy layer |
| **Measured on** | Jetson AGX Thor (`sm_110`, 14× Neoverse V3AE, 122 GiB) for all backends; Intel i9-14900HX for x86 |
| **Status** | Reference-fork measurements, four-backend kernel ladder, two upstreamable CPU patches, server shipped. The fused custom CUDA engine (Phase 2) is not built |

---

## Abstract

llama.cpp is a general GGUF engine, and this repository does not try to replace it. It takes one architecture,
Bonsai-27B, and asks what it costs to serve that model well on every common class of edge hardware: NVIDIA GPUs
through CUDA, any GPU through Vulkan, Arm SoCs through NEON, and commodity x86 through AVX. The weights stay the same
throughout; only the backend changes.

Four results stand out.

- **Thor numbers for Bonsai.** No Jetson Thor figures for Bonsai had been published before these. DSpark speculation reaches
  24.85 tok/s on ternary and 33.24 tok/s on 1-bit, 1.37–1.75× over native decode, at about half the energy per token.
- **One constant predicts speculation.** A backend's round cost `R`, measured in plain decode steps, sets the
  break-even acceptance `α* = (R−1)/4`. On four of six backend–format pairs α* exceeds 1. There, no drafter, workload
  or tuning can make speculation pay, and CPU 1-bit still lost (0.79×) at 100% acceptance.
- **Two CPU backends were gated off by too-narrow ISA checks.** The fork's repack kernels required i8mm on Arm or
  AVX-512 VNNI on x86, which excludes Jetson Orin and every consumer Intel part. Patches 0004 and 0005 widen the gates
  and are validated bit-exact over 100+ shapes:
  - Arm, ternary: 3.90× prefill, 1.55× decode;
  - x86: 3.4× prefill, 1.8× decode.
- **A Vulkan "correctness gap" was a race.** It was traced to `ggml_vk_graph_optimize` reordering writes to the GDN
  recurrent state. Once determinism was enforced, Vulkan matches the CPU backend's numerical-noise profile.

Nothing in this README is estimated unless it is labeled as an estimate.

---

## Results at a glance

### Decode throughput, Jetson AGX Thor, one device, three backends

Same box, same weights. Median over 8 prompts per class at temperature 0, `-c 16384 -np 1`, draft depth 4, no prefix
reuse. Speedups are against the same backend's native decode.

| Backend | Variant | Native, code | Native, prose | DSpark, code | DSpark, prose |
|---|---|--:|--:|--:|--:|
| CUDA | ternary `Q2_0` | 16.77 | 16.97 | **24.85 (1.48×)** | **23.23 (1.37×)** |
| CUDA | 1-bit `Q1_0` | 19.03 | 18.98 | **33.24 (1.75×)** | **26.25 (1.38×)** |
| Vulkan | ternary `Q2_0` | 15.74 | 16.14 | 6.81 (0.43×) | 6.24 (0.39×) |
| Vulkan | 1-bit `Q1_0` | 19.57 | 19.60 | 8.20 (0.42×) | 6.43 (0.33×) |
| CPU, NEON + i8mm, 12 threads | ternary `Q2_0` | 5.19 | 5.14 | 4.95 (0.95×) | 2.49 (0.48×) |

Units are tok/s.

- **Vulkan** rows include the integer-dot MMVQ path (patch 0003). They were taken with the fork's default
  `graph_optimize` on, which is racy. The deterministic default that `bonsai-server` uses costs 1.0% decode and 2.2%
  prefill.
- **CPU** rows are CPU-only with the patch-0004 repack engaged. 1-bit CPU-only decode measures 6.80 tok/s in
  llama-bench (`results/cpu-repack.txt`).
- **Sources:** `results/bench/`, `results/vulkan-mmvq-q2_0.txt`, `results/cpu-repack.txt`. Regenerate the table with
  `python3 bench/render_table.py --speedup results/bench/*.json`.

**Native Vulkan is now within reach of CUDA.** It reaches 94% of CUDA on ternary and is ahead on 1-bit
(19.57 vs 19.03), after the integer-dot MMVQ path was added for `q1_0` and `q2_0`. The speculative path did not move
with it, and the round-cost analysis below explains why.

### Thor, reference fork in detail (Milestone 0)

PrismML fork, CUDA build (`ARCHS=1100`), driver 580.00, CUDA 13.0. Reproduce with
[`reference/run_milestone0.sh`](reference/run_milestone0.sh).

| Variant | Mode | Workload | tok/s | TTFT (s) | Prefill tok/s | Acceptance | Resident |
|---|---|---|--:|--:|--:|--:|--:|
| ternary `Q2_0` | native | code | 16.77 | 0.491 | 120.2 | — | 7,158 MiB |
| ternary `Q2_0` | native | prose | 16.97 | 0.439 | 79.7 | — | 7,158 MiB |
| ternary `Q2_0` | DSpark | code | **24.85** | 0.523 | 107.7 | 54.0% | +1,802 MiB* |
| ternary `Q2_0` | DSpark | prose | 23.23 | 0.477 | 69.1 | 43.3% | +1,802 MiB* |
| 1-bit `Q1_0` | native | code | 19.03 | 0.424 | 149.1 | — | 3,952 MiB |
| 1-bit `Q1_0` | native | prose | 18.98 | 0.380 | 100.3 | — | 3,952 MiB |
| 1-bit `Q1_0` | DSpark | code | **33.24** | 0.464 | 128.3 | 61.6% | +1,802 MiB* |
| 1-bit `Q1_0` | DSpark | prose | 26.25 | 0.418 | 82.7 | 51.4% | +1,802 MiB* |

\* Native footprints are measured peak RSS. The drafter figure is the server's own estimate, because peak RSS is not
additive across two memory-mapped models (`results/memory.txt`).

**`bonsai-server`, which runs DSpark in-process,** measures 1.74× / 1.80× on code and 1.49× / 1.60× on prose
(ternary / 1-bit). Its greedy output is character-identical to non-speculative decode on 14 of 20 runs
(`results/dspark-server.txt`).

### x86 (Intel i9-14900HX, AVX2 + AVX-VNNI, no AVX-512)

x86 was measured at three levels. **Bonsai-27B itself has not been run end to end on x86.**

| Level | Measurement | `Q2_0` | `Q1_0` |
|---|---|--:|--:|
| Fork kernels (patch 0005), llama-bench, 8 threads | Prefill pp512, repack vs generic | **23.8 vs 7.1 (3.4×)** | parity; not selected |
| | Decode tg128, repack vs generic | **8.6 vs 4.8 (1.8×)** | parity; not selected |
| Decode-step proxy, real 27B shape schedule | Best tok/s (321 GEMVs, 5.49 GiB per token) | **4.23** | **7.50** |
| Standalone GEMV | Unpinned DRAM saturation | 28–30 GB/s | 28–30 GB/s |

- **The fork runs used a requantized Qwen3.5-4B.** It validates the pipeline and kernel speed only; 2-bit without QAT
  produces unusable text.
- **The decode-step proxy runs random weights** and skips elementwise work. It is a weights-and-overhead ceiling,
  not a served rate.
- Sources: `results/fork-x86-repack.txt`, `results/decode-proxy-avx-x86.txt`, `results/gemv-avx-x86.txt`.

---

## Findings

### 1. Speculation is governed by one constant per backend

A DSpark round yields `1 + 4α` tokens and costs `R` plain decode steps. Break-even is therefore `α* = (R−1)/4`.
`R` was measured independently on prompts whose acceptance differs 3×. It agrees to 4–13% across workloads, which is
what makes it predictive.

| Backend | Ternary R (α*) | 1-bit R (α*) | Verdict |
|---|--:|--:|---|
| CUDA | 2.18 (**0.296**) | 2.18 (**0.295**) | Wins; measured 1.37–1.80× |
| CPU (NEON) | ~4.9 (0.97) | ~6.0 (**1.33**) | Break-even at best; impossible on 1-bit |
| Vulkan | ~7.3 (**1.54**) | ~8.85 (**1.96**) | Cannot win at any acceptance rate |

The CUDA constant predicted the one losing CUDA-server case before it was measured: α 0.238 gives a predicted 0.90×,
and 0.91× was measured. Because the drafter is a fixed ~1.8 GB model, a faster target raises `R`. **Making the target
faster makes speculation less attractive.** That is why the Vulkan MMVQ change, which lifted native decode 33%, made
every DSpark ratio worse (`results/spec-round-cost.txt`).

**Implication for porting:** measure `R` on new hardware before enabling speculation. The server's
`--speculate auto` refuses where it is known to lose.

### 2. Vulkan's loss is per-dispatch pipeline barriers

Acceptance on Vulkan matches CUDA to within about a point in every cell. The drafter predicts the same tokens; the
backend cannot execute draft-and-verify cheaply.

- A barrier-separated dispatch costs ~1.4 µs (narrow scope) to ~3.9 µs (wide scope). With no barrier it costs
  ~0.17 µs.
- One token graph is ~2,293 dispatches, a **~3.9 ms/token floor**. Submits and host round-trips are under 5%.
- The whole narrow-to-wide gap is `VK_ACCESS_INDIRECT_COMMAND_READ_BIT` in `dstAccessMask`.

### 3. The Vulkan "correctness gap" was a race

Ten identical greedy requests (fixed seed, `cache_prompt=false`) returned up to **8 distinct outputs** on Vulkan and
exactly 1 on CUDA. Bisected to `ggml_vk_graph_optimize`:
- It judges node independence with `is_src_of`, which sees only direct `src[]` edges.
- It misses write-after-read hazards on shared buffers. Bonsai's recurrent state across 48 GDN layers has exactly
  that shape.

With the pass disabled:

| Backend | Traces reproducing the oracle | Worst pre-divergence drift | Character |
|---|--:|--:|---|
| CPU (NEON) | 13/16 | 0.058 | Numerical noise |
| Vulkan, racy | 1/16 | 3.586 | Broken output |
| Vulkan, deterministic | 11/16 | 0.131 | Numerical noise |

The remaining divergences all fall on near-ties: top-1/top-2 gaps of 0.0024–0.0651 nats. CPU and Vulkan diverge on
the same two prompts, at the same token, with the same drift, so those points are properties of the model.
**Establish determinism before running any comparison gate.** (`results/vulkan-nondeterminism.txt`,
`results/vulkan-divergence-explained.txt`.)

### 4. Narrow ISA gates silently disable optimizations, three times over

A wrongly gated kernel fails silently: the model loads, the output is correct, and the fast path never runs.

| Instance | Gate | Who was excluded | Fix |
|---|---|---|---|
| Arm `Q2_0` repack | none existed (AVX-512 only) | every Arm CPU | Patch 0004: NEON repack. DOTPROD tier for Armv8.2 (Orin), i8mm `SMMLA` gemm for Armv8.6+ |
| x86 `Q1_0`/`Q2_0` repack | AVX-512 VNNI | consumer Intel (fused off), every AVX2 part | Patch 0005: AVX2 / AVX-VNNI tier |
| CPU runs on a CUDA build | CUDA pinned-host buffer outranks repack buffers, even at `-ngl 0` | every "CPU" number taken on a CUDA build | Measure on a CPU-only build, or hide the GPU |

Arm, ternary, CPU-only, 12 threads:

| gemm path | pp64 | tg16 | vs no repack |
|---|--:|--:|--:|
| no repack (before) | 4.04 | 3.28 | — |
| DOTPROD (Armv8.2 class, forced) | 8.94 | 5.03 | 2.21× / 1.53× |
| i8mm (Armv8.6+, Thor) | **16.05** | 5.06 | **3.97×** / 1.54× |

In the same spirit, x86 `Q1_0` measured at parity with its existing `vec_dot`, so patch 0005 does not select it.
**Gate on what was measured to help, not on what the instruction set makes possible.**

### 5. Long context is nearly free on this architecture

Only the 16 full-attention layers grow a KV cache. The GDN state is a fixed 598 MiB.

| Context | KV f16 | KV q8_0 | GDN state | Total (q8_0) | Decode tok/s |
|--:|--:|--:|--:|--:|--:|
| 4,096 | 256 MiB | 136 MiB | 598 MiB | 1,056 MiB | 15.79 |
| 65,536 | 4,096 | 2,176 | 598 | 3,336 | 15.77 |
| 131,072 | 8,192 | 4,352 | 598 | 5,768 | 15.77 |
| 262,144 | 16,384 | 8,704 | 598 | 10,632 | 15.83 |

- **Decode is flat across a 64× context increase.** The full 262K context fits in ~25 GiB including weights.
- **`q8_0` KV cache is free:** same throughput, character-identical greedy output on a code prompt.
- **Not shown:** prefill at 256K still costs ~345 tok/s, and quality at long context is untested
  (`results/context-scaling.txt`).

### 6. DSpark roughly halves energy per token

Whole-board `VIN` from `tegrastats`, CUDA, 256-token sustained decode.

| Config | Idle W | Busy W | tok/s | mWh/token | Marginal mWh/token |
|---|--:|--:|--:|--:|--:|
| ternary native | 20.2 | 67.7 | 16.60 | 1.133 | 0.794 |
| ternary DSpark | 20.5 | 63.6 | 29.29 | **0.603** | 0.409 |
| 1-bit native | 20.7 | 61.9 | 18.66 | 0.921 | 0.613 |
| 1-bit DSpark | 21.0 | 64.7 | 32.83 | **0.548** | 0.370 |

Speculation raises throughput at unchanged board power. For an edge device, that is a stronger argument than tok/s
alone (`results/power.txt`).

### 7. What hybrid recurrence does to serving

- **Prefix caching is nearly inert.** GDN recurrent state cannot be truncated by `llama_memory_seq_rm`. An ordinary
  longest-common-prefix cache does not error; it silently continues the previous answer. Only pure appends reuse
  state (21 of 23 tokens measured), so multi-turn chat does not. An earlier claim of a 7.7× TTFT win was measuring
  the broken path (WIKI L10).
- **Continuous batching buys latency, not throughput.** The recurrent path splits batches by sequence, so aggregate
  decode stays flat at 14.3–14.7 tok/s for 1, 2 and 4 streams (`results/concurrency.txt`).
- **DSpark is not token-identical to native at temperature 0.** 5/16 ternary and 6/16 1-bit trace pairs diverge
  reproducibly, and both continuations are reasonable. Gates therefore compare like mode against like mode.

### 8. Kernel ceilings: one packed format, four backends

`kernels/` holds batch-1 GEMVs for all four backends, sharing one packed format
([`kernels/common/bonsai_gemv.h`](kernels/common/bonsai_gemv.h)):
- a biased encoding, so raw codes are dotted directly with one correction per block;
- a bit-plane interleave, so four consecutive weights unpack from one word.

All backends match the same CPU reference: CUDA 1.2e-07, Vulkan 4.8e-06, NEON and AVX bit-exact.

| Backend | Best kernel | `Q2_0` GB/s, best measured |
|---|---|--:|
| CUDA, Thor | v5: `dp4a`, activations in shared memory | **178.5** (65% of 273 GB/s spec) |
| Vulkan, Thor | integer-dot (Tier 3); needs glslang ≥ 16 | 103.5 |
| CPU NEON, Thor, 12 threads | whole-vector unpack + `SDOT` | 38.75 |
| CPU AVX2/VNNI, i9-14900HX | `VPDPBUSD`, work-stealing threads | 28–30 (DRAM-saturated) |

GPU rows use N=131072 × K=8192 (8× the 32 MB L2); the NEON row uses N=32768 × K=4096; the AVX row is DRAM
saturation across shapes. At 178 GB/s, ternary decode on Thor is bounded near **26 tok/s** from weight traffic. The reference fork's 16.77
leaves about **1.5×** for a fused engine.

**Rejected on CUDA, with measurements:**
- 128-bit weight loads: 12% slower;
- row-blocking: 6–25% slower;
- multi-row-per-warp variants: all slower.

See [`kernels/README.md`](kernels/README.md).

---

## Backend support matrix

**Tested here** means the path ran and was validated on available hardware. **Exercised** means it was forced on
hardware that would not otherwise select it. **Untested** means no device was available. Detail and every force-hook
are in [`docs/PORTABILITY.md`](docs/PORTABILITY.md).

| Backend | Feature tier | Path | Status |
|---|---|---|---|
| CPU Arm | NEON + i8mm (Armv8.6+) | `Q2_0` repack, `SMMLA` gemm | **Tested** (Neoverse V3AE) |
| CPU Arm | NEON + DOTPROD (Armv8.2+, e.g. Orin) | `Q2_0` repack, `SDOT` gemm | **Exercised** (`BONSAI_Q2_0_NO_I8MM=1`) |
| CPU Arm | NEON without DOTPROD | per-row `vec_dot` | **Exercised**; character-identical output |
| CPU x86 | AVX2 / AVX-VNNI | `Q2_0` repack (patch 0005) | **Tested** (i9-14900HX) |
| CPU x86 | AVX-512 VNNI | fork's upstream repack | Untested |
| CPU x86 | pre-AVX2 | `vec_dot` | Untested |
| Vulkan | integer dot product + coopmat | MMVQ decode, dequant + coopmat prefill | **Tested**; 56/56 op shapes |
| Vulkan | no integer dot product | dequant MMV | **Exercised**; 56/56 op shapes |
| Vulkan | integer dot, no coopmat | scalar integer-dot MMQ | **Exercised for correctness** (38 shapes, 0 fail); performance untested |
| CUDA | `sm_110` (Thor) | fork kernels | **Run and validated** |
| CUDA | `sm_87`, `sm_86`, `sm_120a`, `sm_121a` | fork kernels | Compiles only (`cuda/arch/verify_arches.sh`) |

**Speculation does not transfer between devices.** Re-measure `R` on every new device.

---

## Getting started

### 1. Build the reference fork and apply the patches

```bash
./reference/setup_prism_fork.sh cuda        # clones the fork to ~/prism-llama.cpp if absent and builds build-cuda
git -C ~/prism-llama.cpp checkout 7529fdaaf # the validated base; the prism branch tip moves
git -C ~/prism-llama.cpp apply \
    "$PWD"/patches/0003-vulkan-q1_0-q2_0-mmvq.patch \
    "$PWD"/patches/0004-cpu-arm-q2_0-repack.patch \
    "$PWD"/patches/0005-cpu-x86-avx2-repack.patch
./reference/setup_prism_fork.sh vulkan      # rebuild per backend after patching: cuda | vulkan | cpu
```

The 0003 → 0004 → 0005 chain applies cleanly to `7529fdaaf` and builds on aarch64.

| Patch | Scope | Effect |
|---|---|---|
| `0001-dspark-staging-batch-cap` | DSpark drafter staging batch | Experiment. Saves ~1 GB at ctx 4,096 with no throughput or acceptance change |
| `0002-force-rs-seq0` | Recurrent rollback ring | Experiment. `BONSAI_FORCE_RS_SEQ0=1` tests that the server refuses to speculate without rollback state |
| `0003-vulkan-q1_0-q2_0-mmvq` | ggml-vulkan | Integer-dot MMVQ for `q1_0`/`q2_0` (native decode: ternary +34%, 1-bit +22%). Includes the 0002 hook |
| `0004-cpu-arm-q2_0-repack` | ggml-cpu, Arm | `Q2_0` NEON repack with DOTPROD and i8mm tiers |
| `0005-cpu-x86-avx2-repack` | ggml-cpu, x86 | AVX2 / AVX-VNNI tier for the `q1_0`/`q2_0` repack kernels |

### 2. Fetch the weights (~15 GB for both variants and both drafters)

```bash
hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf Ternary-Bonsai-27B-dspark-Q4_1.gguf \
    --local-dir ~/models/bonsai/ternary
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf Bonsai-27B-dspark-Q4_1.gguf \
    --local-dir ~/models/bonsai/onebit
```

### 3. Serve with `bonsai-server`

```bash
BUILD_DIR=build-vulkan ./server/build.sh              # links that fork build's libllama; the build dir selects the backend
./build/bonsai-server -m ~/models/bonsai/ternary/Ternary-Bonsai-27B-Q2_0.gguf \
    --backend vulkan --webui server/webui.html
./build/chat                                          # streaming terminal client
./bench/gate_server.sh 8085                           # 20-check smoke gate
curl -s localhost:8085/v1/policy                      # every default, with the measurement behind it
```

| Interface | Detail |
|---|---|
| API | `/v1/chat/completions` (SSE and non-streaming), `/v1/completions`, `/completion`, `/v1/models`, `/metrics`, `/v1/policy` |
| Reasoning | `<think>` routed to `reasoning_content`; `"enable_thinking": false` suppresses it |
| Startup | Preflight determinism probe: two identical greedy requests |
| UI | Self-contained web UI at `/` (no CDN, works offline) |

**The policy layer applies this repository's measurements by default:**
- speculation off on Vulkan (measured 0.33–0.43×) and `auto` elsewhere;
- `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` on Vulkan (determinism);
- 12 CPU threads, not 14. Using all 14 cores was 1.73× / 2.27× slower.

See [`server/README.md`](server/README.md).

### 4. Reproduce the measurements

```bash
./reference/run_milestone0.sh                         # variant x mode x workload sweep, CUDA reference
python3 bench/bench.py ...                            # tok/s, TTFT, acceptance against any OpenAI endpoint
./bench/gate_determinism.sh                           # determinism gate (prefix caching off)
python3 reference/capture_traces.py ...               # per-token logprob traces against the oracle
```

Measurement rules are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). They include pre-divergence drift only,
error over max |ref| for dot products, and "check what the device selects before checking what the feature needs".

---

## Limitations

- **The fused custom CUDA engine (Phase 2) is not built.** Every end-to-end number measures the patched reference
  fork and `bonsai-server` on top of it. The kernels in `kernels/` are standalone and not yet wired into a decode
  step.
- **One device per architecture.** All GPU and Arm numbers are from one Jetson Thor; x86 is from one laptop. Orin and
  RTX targets compile but have not run.
- **Bonsai-27B has not run end to end on x86.** See the x86 table.
- **Accuracy is not evaluated beyond oracle agreement.** Gates compare against the reference fork's logprob traces.
  No task benchmarks were run, and long-context quality is untested.
- **DSpark is not token-identical to native decode at temperature 0** on this fork. Whether that is specific to
  Thor's numerics is open and needs a second device to settle.
- **Workloads are raw completions,** not PrismML's chat-templated thinking-mode benchmark. Acceptance is therefore
  below their published figures (54.0% vs ~69.2% on ternary code); this is not a device effect.

---

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 0–1 | Environment, reference stack, oracle traces, Milestone 0 | Done |
| 2 | Fused CUDA engine: packed-weight loader, GDN and GQA kernels, draft-and-verify in one graph replay | Kernels measured; engine not started |
| 3 | Cross-architecture CUDA | Compile-verified; only `sm_110` run |
| 4 | Vulkan | MMVQ shipped (0003); race root-caused; barrier scoping next |
| 5 | CPU | Arm (0004) and x86 AVX2 (0005) shipped; `Q1_0` register LUT and `Q2_0` `SMMLA` 4-row next |
| 6 | Cross-backend table on one device | Done |
| 6 | Cross-*device* table (Orin, RTX) | Blocked on hardware |

Phase 2 inherits three requirements from this work:
- size the drafter staging buffer to the draft block, which unblocks 8 GB devices;
- fuse draft and verify into one launch, which attacks the dispatch cost behind `R`;
- key adaptive draft-disable on realized throughput, not acceptance.

---

## Repository map

| Path | Contents |
|---|---|
| [`reference/`](reference/) | Fork build per backend, canonical server invocations, trace capture, Milestone 0 sweep. This is the correctness oracle |
| [`patches/`](patches/) | Five patches against the PrismML fork (table above) |
| [`server/`](server/) | `bonsai-server` (C++), policy table (`policy.h`), in-process DSpark (`dspark.h`), terminal client, web UI |
| [`kernels/`](kernels/) | Shared packed format and batch-1 GEMV ladders for CUDA, Vulkan, NEON and AVX, with tests |
| [`bench/`](bench/) | Backend-agnostic harness, determinism and server gates, memory measurement, table renderer, workloads |
| [`results/`](results/) | Raw logs and write-ups behind every number in this README |
| [`cpu-neon-arm/`](cpu-neon-arm/), [`cpu-avx-x86/`](cpu-avx-x86/) | Backend notes for the two CPU targets |
| [`cuda/arch/`](cuda/arch/) | Cross-architecture compile verification |
| [`docs/WIKI.md`](docs/WIKI.md) | Measurement lessons (L1–L17), hardware notes, kernel, speculation and toolchain learnings |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Gating protocol and portable lessons per backend |
| [`docs/PORTABILITY.md`](docs/PORTABILITY.md) | Every gated path, its fallback, and its test status |
| [`docs/INTEGRATION_NOTES.md`](docs/INTEGRATION_NOTES.md) | Fork-integration contracts: Vulkan MMVQ, CPU `Q1_0` LUT |
| [`docs/RESEARCH_PROMPT.md`](docs/RESEARCH_PROMPT.md), [`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md) | Research brief and literature pass, with the measurements that confirmed or refuted it |

---

## Prior art and credit

- **[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)**, `prism` branch. It is the reference
  implementation, the source of the `Q2_0`/`Q1_0` hybrid-attention kernels, and the oracle for everything here.
  `Q1_0` is merged upstream, so stock llama.cpp is a second independent reference for the 1-bit variant.
- **[RightNow-AI/bonsai-turbo](https://github.com/RightNow-AI/bonsai-turbo)** (Apache-2.0) is prior art for
  single-launch batch-1 Bonsai decode, reporting 1.76× over the vendor fork on H100. This repository's planned
  differentiation is DSpark fused into the same launch, Thor validation, and the multi-backend story.
- **PrismML's published H100 figures** are 98.0 → 131.8 tok/s ternary and 104.8 → 143.8 tok/s 1-bit with DSpark
  (1.34× / 1.37×). Thor's higher ratio reflects slower, bandwidth-bound decode, which gives drafting more to hide
  behind. It does not mean better hardware.

## License

- **Code and patches in this repository:** MIT ([`LICENSE`](LICENSE)). The patches modify llama.cpp and the PrismML
  fork, both MIT.
- **Model weights, the `Q2_0`/`Q1_0` formats and the DSpark drafters** are PrismML's work, released under
  Apache-2.0. They are derived from Qwen3.6-27B. No weights are distributed here.

## Citation

```bibtex
@misc{devaney2026bonsaiedgeserve,
  title  = {bonsai-edge-serve: model-specific, hardware-portable serving of ternary and 1-bit Bonsai-27B with DSpark
            speculative decoding across CUDA, Vulkan, ARM NEON and x86 AVX},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/bonsai-edge-serve}
}
```
