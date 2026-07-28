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

## Prefix caching, and why it is nearly inert here

Bonsai is **hybrid**: 48 of 64 layers are gated-delta-net. Their recurrent
state is a single summary of every token decoded, with no per-position
representation, so `llama_memory_seq_rm` **cannot roll it back**. It can
drop attention KV entries while the recurrent state still reflects the
tokens it just "removed".

This does not error. It silently continues the previous answer.

We shipped that bug and the gate caught it: four identical greedy requests
returned four different continuations, each picking up where the last left
off. The rule now is that on a recurrent or hybrid model **only a pure
append may reuse the KV** — if the resident tokens are not a strict prefix
of the new prompt, the state is reset and the prompt is re-prefilled.

Consequence, stated plainly: **append-only continuation reuses the cache
(measured 21 of 23 tokens), and multi-turn chat does not.** A chat template
emits `<|im_end|>` and a new header *after* the assistant's text, so the
resident tokens stop being a prefix and the reuse is correctly refused.

An earlier version of this file claimed a 7.7× TTFT win (2.416s → 0.313s,
97.6% reuse) for a shared system prompt. **That measurement was of the
broken path** — fast because it was reusing a state it had no right to
reuse. With the correct rule the same test reuses 0 tokens and TTFT is flat
at ~1.42s, matching the uncached path exactly.

Getting the agentic multi-turn win on this architecture requires explicit
state checkpoints (`llama_state_seq_save_data` / `restore`), which is what
the fork's own server does — its logs show `created context checkpoint …
149.626 MiB`. That is real future work, not something to fake.

## Speculative decoding

Not implemented in this binary yet. DSpark needs the draft/verify/rollback
loop with target-layer tap capture (`common_speculative_*`), and on a
recurrent model the rollback needs the same checkpoint machinery as above.
Reimplementing it half-correctly would produce plausible text and wrong
acceptance numbers, which is precisely the failure mode this repo exists to
avoid.

Until then: on Vulkan the policy disables speculation anyway (it loses), and
on CUDA use `reference/serve_reference.sh ternary dspark`, which is worth
1.37–1.75×.

## Gate it

```bash
./bench/gate_server.sh 8085
```

20 checks: every endpoint, streaming, stop strings, reasoning separation,
the three prefix-cache invariants above, API-level determinism, and that a
malformed request returns a structured error rather than a crash. Each one
asserts on **content**, because a 200 with an empty body is the failure that
looks like success.
