#!/usr/bin/env bash
# Smoke gate for bonsai-server: every endpoint and every serving feature we
# claim, checked against a running instance.
#
# A server that returns 200 with empty content is the failure mode that
# looks like success, so each check asserts on the CONTENT, not the status.
#
# Usage: gate_server.sh [port]
set -uo pipefail
PORT="${1:-8085}"
cd "$(dirname "$0")/.."

python3 - "$PORT" <<'PY'
import json, sys, urllib.error, urllib.request

port = sys.argv[1]
BASE = f"http://127.0.0.1:{port}"
fails = []

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return r.status, r.read()

def post(path, obj, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)

print("bonsai-server smoke gate")

# --- discovery endpoints
try:
    st, body = get("/health")
    j = json.loads(body)
    check("/health reports ok", st == 200 and j.get("status") == "ok", str(j))
except Exception as e:
    check("/health reachable", False, str(e)); print("\nserver not reachable; aborting"); sys.exit(1)

st, body = get("/v1/models")
j = json.loads(body)
check("/v1/models lists a model", st == 200 and len(j.get("data", [])) >= 1)

st, body = get("/v1/policy")
pol = json.loads(body)
check("/v1/policy explains itself", bool(pol.get("reasons")), str(pol))

# The policy must never silently enable speculation where it is measured
# to lose -- that regression is the whole reason policy.h exists.
if pol.get("backend") == "vulkan":
    check("speculation OFF on vulkan (measured 0.33-0.43x)",
          pol.get("speculation") is False, "speculation enabled on a backend where it loses")

st, body = get("/")
check("web UI served", st == 200 and len(body) > 2000, f"{len(body)} bytes")

# --- non-streaming chat, with thinking suppressed
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
    "max_tokens": 24, "temperature": 0, "enable_thinking": False})
j = json.load(r)
content = j["choices"][0]["message"]["content"]
check("chat returns non-empty content", bool(content.strip()), repr(content))
check("usage accounting present",
      j["usage"]["completion_tokens"] > 0 and j["usage"]["prompt_tokens"] > 0)
check("bonsai telemetry present", j.get("bonsai", {}).get("decode_tok_s", 0) > 0)

# --- reasoning goes to reasoning_content, never into content
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "What is 17 * 23? Think it through."}],
    "max_tokens": 200, "temperature": 0, "enable_thinking": True})
j = json.load(r)
msg = j["choices"][0]["message"]
check("<think> never leaks into content", "<think>" not in msg.get("content", ""),
      repr(msg.get("content", "")[:80]))
check("reasoning_content populated when thinking",
      bool(msg.get("reasoning_content")), "no reasoning_content field")

# --- streaming
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Count: 1 2 3"}],
    "max_tokens": 60, "temperature": 0, "stream": True, "enable_thinking": False})
deltas, saw_done, finish, streamed = 0, False, None, ""
for raw in r:
    line = raw.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        saw_done = True
        continue
    o = json.loads(payload)
    for ch in o.get("choices", []):
        d = ch.get("delta") or {}
        if d.get("content"):
            deltas += 1
            streamed += d["content"]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
check("stream emits content deltas", deltas > 0, f"{deltas} deltas")
check("stream terminates with [DONE]", saw_done)
check("stream reports finish_reason", finish is not None, str(finish))

# --- stop strings must not appear in the output
r = post("/v1/completions", {"prompt": "Count to three: 1, 2,", "max_tokens": 40,
                             "temperature": 0, "stop": ["4"]})
j = json.load(r)
check("stop string is not echoed", "4" not in j["choices"][0]["text"],
      repr(j["choices"][0]["text"][:60]))

# --- prefix caching, on a HYBRID model
# Bonsai is 48/64 gated-delta-net. The recurrent state summarises every
# token decoded and has no per-position form, so llama_memory_seq_rm cannot
# roll it back. Only a PURE APPEND may reuse the KV; any divergence must
# force a full reset. Getting this wrong does not error -- it silently
# continues the previous answer. So the gate asserts CORRECTNESS first and
# reuse second.
def comp(prompt, cache=True, n=16):
    r = post("/v1/completions", {"prompt": prompt, "max_tokens": n,
                                 "temperature": 0, "cache_prompt": cache})
    return json.load(r)

base = "The capital of France is"
r1 = comp(base)
cont = base + r1["choices"][0]["text"]

# Repeating an identical request must NOT continue the previous answer.
again = comp(base)["choices"][0]["text"]
check("repeated request does not continue the previous answer",
      again == r1["choices"][0]["text"],
      "cached run drifted -- recurrent state was not reset")

# Pure append is the one safe reuse, and it must not change the result.
r2 = comp(cont + " Also,")
r3 = comp(cont + " Also,", cache=False)
check("append-only reuse fires", r2["bonsai"]["cached_tokens"] > 0,
      f"reused {r2['bonsai']['cached_tokens']} of {r2['usage']['prompt_tokens']}")
check("append-only reuse does not change output",
      r2["choices"][0]["text"] == r3["choices"][0]["text"],
      "cached and uncached disagree")

# A diverging prefix must refuse reuse rather than truncate the state.
r4 = comp("An entirely different opening sentence about turbines")
check("diverging prefix refuses reuse", r4["bonsai"]["cached_tokens"] == 0,
      f"reused {r4['bonsai']['cached_tokens']} after divergence")

# --- determinism, at the API level
outs = set()
for _ in range(3):
    r = post("/v1/completions", {"prompt": "def binary_search(arr, target):",
                                 "max_tokens": 24, "temperature": 0,
                                 "cache_prompt": False})
    outs.add(json.load(r)["choices"][0]["text"])
check("greedy decoding is deterministic", len(outs) == 1, f"{len(outs)} distinct outputs")

# --- errors are structured, not crashes
try:
    post("/v1/chat/completions", {"max_tokens": 8})
    check("missing 'messages' rejected", False, "accepted a request with no messages")
except urllib.error.HTTPError as e:
    body = json.loads(e.read())
    check("missing 'messages' rejected with an error object",
          e.code == 400 and "error" in body, str(body))

print()
if fails:
    print(f"RESULT: FAIL -- {len(fails)} check(s): {', '.join(fails)}")
    sys.exit(1)
print("RESULT: PASS -- all checks green")
PY
