# bonsai-server

A lean, single-binary, **pure C++** OpenAI-compatible server for the
Bonsai-27B family on Jetson Thor. No Python on the hot path, no proxy hop —
it links `libllama` directly, so the model lives in-process.

```bash
BUILD_DIR=build-vulkan2 ./server/build.sh          # -> build/bonsai-server, build/chat
./build/bonsai-server -m ~/models/bonsai/ternary/Ternary-Bonsai-27B-Q2_0.gguf \
    --backend vulkan --webui server/webui.html
./build/chat                                        # terminal client
```

## What it has

**API** — `POST /v1/chat/completions` (SSE streaming + non-streaming),
`POST /v1/completions`, `POST /completion` (llama.cpp alias, so
`bench/bench.py` works unchanged), `GET /v1/models`.

**Reasoning delineation** — this model emits `<think>…</think>`. Those go to
`reasoning_content`, never into `content`, both non-streaming and as
streaming deltas. The streaming splitter holds back a short tail so a tag
split across chunks is never emitted and then retracted.

**`enable_thinking`** — the model thinks by default and its chat template
has no switch for it, so a short question can burn its entire token budget
on `<think>` and return empty content. `"enable_thinking": false` pre-fills
a closed empty think block, which is the Qwen3-family way to suppress it.

**Policy that explains itself** — `GET /v1/policy` returns, in plain text,
every decision the server made and the measurement behind it. The server
prints the same thing at startup. See `policy.h`.

**Preflight determinism probe** — two identical greedy requests at startup.
A backend that answers the same question differently cannot be gated for
correctness, and this repo has already been burned by measuring a gate on
top of a race (WIKI L9).

**Telemetry** — `GET /metrics`: decode tok/s, prefill tok/s, TTFT, cached
tokens, DSpark acceptance. Per-request the same numbers come back in a
`bonsai` object.

**Web UI** — self-contained single file, no CDN or npm, works offline.
Streaming, collapsible thinking blocks, code blocks, light/dark.

**Terminal client** — `build/chat`, streaming multi-turn REPL with dimmed
thinking and a tok/s readout. Plain OpenAI, so it also works against the
fork's own `llama-server`.

## The policy layer is the point

Running `llama-server` directly gets you an OpenAI API too. What this binary
adds is that the repo's measurements are **applied by default** instead of
living in a wiki you have to read first:

| Default | Why |
| :-- | :-- |
| speculation OFF on Vulkan | measured **0.33–0.43×** — DSpark there is ~2.3× slower than not speculating. `--speculate auto` consults the table; `on` overrides and says so |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | the reorder pass makes this model nondeterministic (up to 8 distinct outputs from 10 identical greedy requests). Costs 2.2% prefill / 1.0% decode |
| `-t 12` on CPU, not 14 | taking every core is the *worst* setting: 12 threads measured 1.73–2.27× faster than 14 |

`--backend` selects the **policy table**, not the compute backend. The
compute backend is whichever fork build you linked against, because
`libggml` loads the backend shared objects sitting next to it.

## Prefix caching: truncation cannot work here, checkpoints can

Bonsai is **hybrid**: 48 of 64 layers are gated-delta-net. Their recurrent
state is a single summary of every token decoded, with no per-position
representation, so `llama_memory_seq_rm` **cannot roll it back**. It drops
attention KV entries while the recurrent state still reflects the tokens it
just "removed".

This does not error — it silently continues the previous answer. We shipped
that bug and the gate caught it: four identical greedy requests returned
four different continuations, each resuming where the last stopped. An
earlier version of this file claimed a 7.7× TTFT win from that cache; **that
was the broken path**, fast because it reused a state it had no right to.

Two mechanisms now, in order of preference:

**1. State checkpoints (the real fix).** After prefilling, the server
snapshots the whole sequence state with `llama_state_seq_get_data` — KV
*and* recurrent, ~150 MiB. A later request whose prompt begins with those
exact tokens restores it and prefills only the remainder. Because the state
is saved rather than truncated, the recurrent part is correct by
construction.

| | turn 0 | turn 1 | turn 2 |
| :-- | --: | --: | --: |
| restored tokens | 0 | 440 | 455 |
| TTFT | 1.557s | **0.360s** | **0.379s** |

4.3× on a 440-token shared system prompt, and the gate asserts the restored
output is **character-identical** to a fresh prefill — the check the earlier
claim lacked.

The checkpoint is taken at the **template boundary**, not at the end of the
prompt. With `enable_thinking:false` the server appends `<think></think>`
after the template output, and that suffix does not appear at the same
position in the next turn's prompt — so a checkpoint at the full prompt
length could never be a prefix of the follow-up. Landing the prefill exactly
on the boundary makes reuse work in both modes.

**Interop note:** reuse only fires if the history the client sends
re-renders to the same tokens. A client that echoes `reasoning_content`
back as `content` inserts text the original prompt never contained, the
prefix breaks, and the checkpoint is correctly refused. Real OpenAI clients
echo `content` only, which is the case that works.

`--checkpoint-mb N` caps the per-checkpoint size (default 2048, `0`
disables); two checkpoints per slot are kept.

**2. Append-only KV reuse (the fallback).** If no checkpoint matches, the
resident tokens may still be reused when they are a strict prefix of the new
prompt — measured 21 of 23 tokens on a continuation. Any divergence,
including stepping back a single token, forces a full reset.

## Concurrency: 4 slots, and what they are actually for

`--slots N` (default 4) runs N independent sequences through one scheduler
thread that builds a single batch per step. Each slot is its own
`llama_seq_id` with its own recurrent state, allocated up front by
`n_seq_max`.

**It does not increase throughput, and the reason is structural.** Decode is
weight-bandwidth-bound, so batching *should* be nearly free -- but
`src/llama-memory-recurrent.cpp:440` splits a batch by sequence whenever
every token is an output, which is exactly the decode case. Four sequences
become four single-token ubatches and four full weight reads:

| concurrency | aggregate tok/s | per stream |
| --: | --: | --: |
| 1 | 14.33 | 14.33 |
| 2 | 14.70 | 7.35 |
| 4 | 14.70 | 3.67 |

**What it buys is latency.** Tokens interleave, so a short request no longer
waits behind a long one -- with a 200-token request and three short ones in
flight, the short ones completed at 1.27s instead of ~15s. Single-stream
throughput is unaffected (14.45 with 1 slot, 14.52 with 4), so the default
costs nothing. `results/concurrency.txt`.

## Structured output

`response_format: {"type": "json_object"}` constrains sampling to a JSON
grammar; `"grammar": "<GBNF>"` takes an arbitrary one. The grammar sampler
runs first in the chain, before temperature and truncation, so the formal
language masks the candidate set rather than being applied to an
already-truncated one.

```bash
curl localhost:8085/v1/chat/completions -d '{"messages":[...],
     "response_format":{"type":"json_object"}}'      # always valid JSON
curl localhost:8085/v1/chat/completions -d '{"messages":[...],
     "grammar":"root ::= (\"yes\" | \"no\")"}'        # yes or no, nothing else
```

## KV cache

`-ctk q8_0` is the default. On this model it halves the KV cache for no
measurable cost -- at 262144 context, 16384 -> 8704 MiB with unchanged
throughput and character-identical greedy output. `-ctk f16` restores the
uncompressed cache. See `results/context-scaling.txt`; decode speed is flat
from 4K to 262144.

## Speculative decoding (DSpark)

In-process. Pass `-md` and the server drafts, verifies and rolls back itself.

```bash
./build/bonsai-server -m .../Ternary-Bonsai-27B-Q2_0.gguf \
                      -md .../Ternary-Bonsai-27B-dspark-Q4_1.gguf \
                      --backend cuda --speculate on
```

| | ternary Q2_0 | 1-bit Q1_0 |
| :-- | --: | --: |
| code (5 prompts) | **1.74×** (α 0.707) | **1.80×** (α 0.737) |
| prose (5 prompts) | **1.49×** (α 0.563) | **1.60×** (α 0.611) |

Greedy speculative output is character-identical to greedy AR output on 14 of
20 runs; the other 6 are the near-tie flips this repo already characterised,
not rollback bugs — `results/dspark-server.txt` separates the two hypotheses
four ways rather than assuming.

**The number to serve by.** A speculative round costs **2.18** plain decode
steps regardless of how well the drafter does — flat across a 2.3× range of
measured speedups *and* across both quantisations. So the breakeven is
arithmetic:

```
1 + block_size·α = 2.18   →   α* = 0.296
```

Below α≈0.3 speculation loses. One prose prompt sits at α 0.238 and measures
0.91×, which the constant predicts as 0.90× — estimated from all ten runs,
then reproducing the single losing case. Acceptance is a **per-prompt**
property, not a per-workload-class one: the spread inside prose (0.238–0.913)
is wider than the gap between prose and code.

**It refuses to run unsafely.** On a hybrid target the post-verify crop only
works if the recurrent rollback ring was sized at *context creation*
(`n_rs_seq`); at zero it silently no-ops and the gated-delta-net state absorbs
every rejected draft tail, with fluent text throughout and no symptom to
observe. The server checks the ring it actually got and exits:

```
error: DSpark: recurrent rollback ring is 0 but a draft block needs 4 ...
```

`--draft-max` also sizes that ring, while the real draft length is the
drafter's `block_size`, so the server reads `block_size` from the GGUF and
raises `--draft-max` itself rather than trusting the two to have been matched.

**It costs prefix caching.** DSpark conditions on the target's activations at
every position, so a position restored from a checkpoint — never decoded,
never captured — is a hole. Speculating requests therefore bypass the prompt
cache, which makes checkpoints (4.3× TTFT) and DSpark (1.5–1.8× decode)
mutually exclusive. DSpark wins unless the answer is much shorter than the
shared prefix; see `results/dspark-server.txt` for the crossover.

On Vulkan the policy still disables speculation (measured 0.33–0.43× there).
`reference/serve_reference.sh ternary dspark` remains available as the oracle.

```bash
./bench/gate_dspark.sh 8091 8092    # speculating server, AR server
```

## Gate it

```bash
./bench/gate_server.sh 8085
```

20 checks: every endpoint, streaming, stop strings, reasoning separation,
the three prefix-cache invariants above, API-level determinism, and that a
malformed request returns a structured error rather than a crash. Each one
asserts on **content**, because a 200 with an empty body is the failure that
looks like success.
