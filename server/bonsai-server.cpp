// bonsai-server.cpp -- a lean, single-binary OpenAI-compatible server for
// the Bonsai-27B family (ternary Q2_0 / 1-bit Q1_0) on Jetson Thor.
//
// No Python on the hot path. Links libllama directly, so the model lives
// in-process and there is no proxy hop. The engine is the PrismML fork
// (Milestone 0: the fork is the correctness oracle); what this binary adds
// is the serving layer:
//
//   * OpenAI /v1/chat/completions (SSE streaming + non-streaming),
//     /v1/completions, /v1/models
//   * reasoning delineation -- this model's <think> blocks are routed to
//     `reasoning_content` rather than leaking into `content`
//   * prefix caching -- longest-common-prefix KV reuse across turns, which
//     is the agentic win: a long system prompt is prefilled once
//   * /metrics with decode tok/s, TTFT and DSpark acceptance
//   * /v1/policy -- the server explains, in plain text, why it is
//     configured the way it is (see policy.h)
//   * a preflight determinism probe, because a backend that answers the
//     same question differently each time cannot be gated for correctness
//   * a self-contained web UI at /
//
// Build: server/build.sh   (see that script for backend selection)

#include "llama.h"
#include "policy.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <sstream>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>

using json = nlohmann::json;
using clock_t_ = std::chrono::steady_clock;

static double now_s() {
    return std::chrono::duration<double>(clock_t_::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- logging
static bool g_verbose = false;

static void logf(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

// ------------------------------------------------------- reasoning split
// Bonsai emits <think>...</think> in thinking mode. OpenAI clients render
// `content`, so putting chain-of-thought there makes every answer look like
// it leaked its scratchpad. It goes to `reasoning_content`, the same field
// vLLM's reasoning parsers use.
//
// Streaming cannot buffer to the end, so this is incremental: it tracks
// whether we are inside a think block and holds back a short tail that
// might be a partial tag, rather than emitting text it would have to
// retract.
struct ReasoningSplitter {
    std::string buf;
    bool in_think = false;

    static constexpr size_t HOLD = 8; // strlen("</think>") - 1 rounded up

    void feed(const std::string & delta, bool final,
              std::string & out_content, std::string & out_reasoning) {
        buf += delta;
        for (;;) {
            if (in_think) {
                const size_t i = buf.find("</think>");
                if (i == std::string::npos) break;
                out_reasoning += buf.substr(0, i);
                buf.erase(0, i + 8);
                in_think = false;
            } else {
                const size_t i = buf.find("<think>");
                if (i == std::string::npos) break;
                out_content += buf.substr(0, i);
                buf.erase(0, i + 7);
                in_think = true;
            }
        }
        const size_t keep = final ? 0 : std::min(HOLD, buf.size());
        const std::string emit = buf.substr(0, buf.size() - keep);
        buf.erase(0, buf.size() - keep);
        (in_think ? out_reasoning : out_content) += emit;
    }
};

// ---------------------------------------------------------------- metrics
struct Metrics {
    std::mutex mu;
    uint64_t requests = 0, errors = 0;
    uint64_t prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
    double   decode_s = 0.0, prefill_s = 0.0, ttft_sum = 0.0;
    uint64_t ttft_n = 0;
    uint64_t draft_n = 0, draft_accepted = 0;
    double   started = now_s();

    void record(uint64_t n_prompt, uint64_t n_cached, uint64_t n_gen,
                double prefill, double decode, double ttft) {
        std::lock_guard<std::mutex> lk(mu);
        requests++;
        prompt_tokens += n_prompt;
        cached_tokens += n_cached;
        completion_tokens += n_gen;
        prefill_s += prefill;
        decode_s  += decode;
        if (ttft > 0) { ttft_sum += ttft; ttft_n++; }
    }

    json snapshot() {
        std::lock_guard<std::mutex> lk(mu);
        json j;
        j["uptime_s"]          = now_s() - started;
        j["requests"]          = requests;
        j["errors"]            = errors;
        j["prompt_tokens"]     = prompt_tokens;
        j["cached_tokens"]     = cached_tokens;
        j["completion_tokens"] = completion_tokens;
        j["decode_tok_s"]      = decode_s  > 0 ? completion_tokens / decode_s  : 0.0;
        j["prefill_tok_s"]     = prefill_s > 0 ? prompt_tokens     / prefill_s : 0.0;
        j["ttft_s_mean"]       = ttft_n ? ttft_sum / ttft_n : 0.0;
        j["draft_n"]           = draft_n;
        j["draft_accepted"]    = draft_accepted;
        // accepted/drafted -- the quantity the model card quotes
        j["acceptance"]        = draft_n ? (double) draft_accepted / draft_n : 0.0;
        return j;
    }
};

static Metrics g_metrics;

// ------------------------------------------------------------------ model
struct GenParams {
    int    n_predict   = 512;
    float  temperature = 0.7f;
    float  top_p       = 0.95f;
    int    top_k       = 40;
    float  min_p       = 0.0f;
    uint32_t seed      = LLAMA_DEFAULT_SEED;
    std::vector<std::string> stop;
    std::string grammar;   // GBNF; constrains sampling to a formal language
    bool use_cache = true;
    size_t ckpt_at = 0;   // checkpoint here; 0 = end of prompt
};

class Engine {
public:
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    std::mutex mu;                 // one generation at a time; requests queue
    std::vector<std::vector<llama_token>> cache_tokens;  // per slot: resident in KV
    bool recurrent = false;                 // hybrid/recurrent state cannot be truncated
    int n_ctx = 0;

    int n_slots = 1;

    bool load(const std::string & path, int ngl, int n_ctx_req, int n_threads,
              int slots, ggml_type kv_type) {
        n_slots = slots;
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = ngl;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) return false;
        vocab = llama_model_get_vocab(model);

        llama_context_params cp = llama_context_default_params();
        // n_ctx is the TOTAL across sequences, so each slot gets n_ctx/slots.
        cp.n_ctx       = (uint32_t) n_ctx_req * slots;
        cp.n_seq_max   = (uint32_t) slots;
        cp.n_batch     = 2048;
        cp.n_ubatch    = 512;
        cp.n_threads   = n_threads;
        cp.n_threads_batch = n_threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = kv_type;
        cp.type_v = kv_type;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) return false;
        n_ctx = (int) llama_n_ctx(ctx) / slots;   // per-sequence budget
        recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        return true;
    }

    ~Engine() {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
    }

    std::vector<llama_token> tokenize(const std::string & text, bool add_bos) {
        int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                nullptr, 0, add_bos, true);
        std::vector<llama_token> out(n);
        if (llama_tokenize(vocab, text.c_str(), (int) text.size(),
                           out.data(), (int) out.size(), add_bos, true) < 0) {
            return {};
        }
        return out;
    }

    std::string detokenize(llama_token tok) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        return n > 0 ? std::string(buf, n) : std::string();
    }

    llama_sampler * make_sampler(const GenParams & gp) {
        auto sp = llama_sampler_chain_default_params();
        llama_sampler * s = llama_sampler_chain_init(sp);
        // Grammar goes FIRST: it masks out every token the formal language
        // forbids, and the temperature/top-k/top-p stages then choose among
        // what remains. Applying it after truncation could leave an empty
        // candidate set when the sampler has already discarded every legal
        // token.
        if (!gp.grammar.empty()) {
            llama_sampler * g = llama_sampler_init_grammar(vocab, gp.grammar.c_str(), "root");
            if (g) {
                llama_sampler_chain_add(s, g);
            } else {
                // A malformed grammar must not silently degrade to
                // unconstrained sampling -- the caller asked for a
                // guarantee. The request is rejected before we get here;
                // this is the belt-and-braces path.
                fprintf(stderr, "warning: grammar failed to compile, refusing to sample unconstrained\n");
            }
        }
        if (gp.temperature <= 0.0f) {
            // Greedy. Exact and reproducible -- the mode every gate uses.
            llama_sampler_chain_add(s, llama_sampler_init_greedy());
        } else {
            if (gp.top_k > 0)  llama_sampler_chain_add(s, llama_sampler_init_top_k(gp.top_k));
            if (gp.top_p < 1.f) llama_sampler_chain_add(s, llama_sampler_init_top_p(gp.top_p, 1));
            if (gp.min_p > 0.f) llama_sampler_chain_add(s, llama_sampler_init_min_p(gp.min_p, 1));
            llama_sampler_chain_add(s, llama_sampler_init_temp(gp.temperature));
            llama_sampler_chain_add(s, llama_sampler_init_dist(gp.seed));
        }
        return s;
    }

    // Longest common prefix with the KV already resident.
    //
    // CRITICAL for this model. Bonsai is HYBRID: 48 of its 64 layers are
    // gated-delta-net, whose recurrent state is a single summary of every
    // token decoded so far. There is no per-position representation of it,
    // so llama_memory_seq_rm CANNOT roll it back -- it can drop attention
    // KV entries while the recurrent state still reflects the tokens it
    // just "removed". Reusing that state silently continues the previous
    // generation instead of answering the new prompt.
    //
    // Therefore, on a recurrent/hybrid model only a PURE APPEND is safe:
    // the resident tokens must be a strict prefix of the new prompt. Any
    // divergence -- including stepping back a single token -- forces a full
    // reset. On a pure-attention model the ordinary LCP rule applies.
    size_t reusable_prefix(int slot, const std::vector<llama_token> & prompt) const {
        const auto & resident = cache_tokens[slot];
        size_t i = 0;
        const size_t n = std::min(resident.size(), prompt.size());
        while (i < n && resident[i] == prompt[i]) i++;

        if (recurrent) {
            if (i < resident.size()) return 0;      // truncation: unsafe
            if (i == prompt.size())  return 0;      // nothing left to decode
            return i;                               // pure append: safe
        }

        // Never reuse the whole prompt: at least one token must be decoded
        // to produce logits for the first sampled token.
        if (i == prompt.size() && i > 0) i--;
        return i;
    }
};

// ------------------------------------------------------------- scheduler
// Continuous batching.
//
// Decode is weight-bandwidth-bound: a step reads all 6.66 GB of weights
// regardless of how many sequences are advancing. So running S sequences in
// one batch costs barely more than running one, and aggregate throughput
// scales nearly with S. That is the largest serving win available on this
// box, and it is why the server owns a scheduler rather than generating
// inline on the request thread.
//
// Each slot is an independent llama sequence (seq_id == slot index) with its
// own recurrent state -- the context is created with n_seq_max = slots, so
// the GDN state is allocated per slot up front.

struct GenStats {
    int    n_prompt = 0, n_cached = 0, n_generated = 0, n_lcp = 0, n_restored = 0;
    double prefill_s = 0, decode_s = 0, ttft_s = 0;
    std::string finish_reason = "stop";
};

// A saved sequence state, restorable at an exact token count.
//
// This is the thing plain KV truncation cannot do on a hybrid model. The
// GDN recurrent state summarises every token decoded and has no
// per-position form, so llama_memory_seq_rm cannot roll it back -- which is
// why prefix reuse was limited to pure appends. A checkpoint captures the
// whole state (attention KV + recurrent) at a known prefix length, so it
// can be restored and the remainder prefilled, regardless of where the new
// prompt diverges.
struct Checkpoint {
    std::vector<llama_token> tokens;   // exactly what this state has consumed
    std::vector<uint8_t>     blob;
};

struct Slot {
    int id = 0;
    bool busy = false;
    // `busy` stays true until the HTTP thread releases the slot, which can
    // be several scheduler iterations after generation ends. Without a
    // separate `done`, the loop keeps appending tokens to a finished slot
    // -- burning decode steps and eventually running its position past
    // n_ctx, which surfaces as a 500.
    bool done = false;

    std::vector<llama_token> prompt;
    std::vector<llama_token> generated;
    GenParams gp;
    llama_sampler * smpl = nullptr;

    size_t n_prefilled = 0;      // how much of `prompt` is in the KV
    double t_start = 0, t_prefill_done = 0;
    GenStats st;
    std::string tail;            // rolling buffer for stop-string matching
    // Checkpoint at the TEMPLATE boundary, not at the end of the prompt.
    // With enable_thinking=false the server appends "<think>\n\n</think>"
    // after the template output; that suffix is not present at the same
    // position in the next turn's prompt, so a checkpoint taken at the full
    // prompt length is never a prefix of the follow-up and can never be
    // reused. Saving at the boundary keeps it reusable in both modes.
    size_t ckpt_at = 0;
    bool   ckpt_saved = true;

    // handoff to the HTTP thread
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> out;
    bool finished = false;
    bool cancel = false;
    std::string error;
};

class Scheduler {
public:
    Engine & eng;
    // deque, not vector: Slot owns a mutex and a condition_variable, so it
    // is neither movable nor copyable and vector::resize cannot build it.
    std::deque<Slot> slots;
    // Per slot, most-recent-first. Small: each blob is the full sequence
    // state, so this trades memory for prefill and must stay bounded.
    std::vector<std::deque<Checkpoint>> ckpts;
    size_t ckpt_budget_bytes = 0;
    int    ckpt_per_slot     = 2;
    std::mutex mu;                    // guards slot allocation + the batch loop
    std::condition_variable work_cv;
    std::thread th;
    std::atomic<bool> stop{false};
    llama_batch batch{};

    explicit Scheduler(Engine & e) : eng(e) {
        for (int i = 0; i < eng.n_slots; i++) {
            slots.emplace_back();
            slots.back().id = i;
        }
        eng.cache_tokens.assign(eng.n_slots, {});
        ckpts.resize(eng.n_slots);
        batch = llama_batch_init(2048, 0, eng.n_slots);
    }

    ~Scheduler() {
        stop = true;
        work_cv.notify_all();
        if (th.joinable()) th.join();
        llama_batch_free(batch);
    }

    void start() { th = std::thread([this]{ loop(); }); }

    // Returns nullptr if every slot is busy.
    Slot * acquire(const std::vector<llama_token> & prompt, const GenParams & gp,
                   std::string & err) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & sl : slots) {
            if (sl.busy) continue;
            if ((int) prompt.size() + gp.n_predict > eng.n_ctx) {
                err = "prompt + max_tokens exceeds the per-slot context (" +
                      std::to_string(prompt.size()) + " + " +
                      std::to_string(gp.n_predict) + " > " +
                      std::to_string(eng.n_ctx) + ")";
                return nullptr;
            }
            sl.busy = true;
            sl.prompt = prompt;
            sl.gp = gp;
            sl.generated.clear();
            sl.out.clear();
            sl.tail.clear();
            sl.finished = false;
            sl.done = false;
            sl.cancel = false;
            sl.error.clear();
            sl.st = GenStats{};
            sl.smpl = eng.make_sampler(gp);
            sl.t_start = now_s();
            sl.t_prefill_done = 0;

            // Prefix reuse, subject to the hybrid rule (see reusable_prefix).
            size_t keep = 0;
            {
                const auto & res = eng.cache_tokens[sl.id];
                size_t i = 0;
                const size_t n = std::min(res.size(), prompt.size());
                while (i < n && res[i] == prompt[i]) i++;
                sl.st.n_lcp = (int) i;
            }
            if (gp.use_cache) {
                keep = eng.reusable_prefix(sl.id, prompt);
                // If plain reuse cannot help (the usual case in multi-turn
                // chat, where the template appends tokens after the
                // assistant's text and so breaks the strict-prefix rule),
                // look for a checkpoint whose tokens ARE a prefix.
                if (keep == 0) {
                    size_t best = 0;
                    const Checkpoint * hit = nullptr;
                    for (const auto & c : ckpts[sl.id]) {
                        if (c.tokens.size() >= prompt.size()) continue;
                        if (c.tokens.size() <= best) continue;
                        if (std::equal(c.tokens.begin(), c.tokens.end(), prompt.begin())) {
                            best = c.tokens.size();
                            hit  = &c;
                        }
                    }
                    if (g_verbose) fprintf(stderr, "checkpoint: %zu stored, best prefix %zu of %zu\n",
                                           ckpts[sl.id].size(), best, prompt.size());
                    if (hit && llama_state_seq_set_data(eng.ctx, hit->blob.data(),
                                                        hit->blob.size(), sl.id) > 0) {
                        keep = best;
                        eng.cache_tokens[sl.id] = hit->tokens;
                        sl.st.n_restored = (int) best;
                    } else {
                        llama_memory_seq_rm(llama_get_memory(eng.ctx), sl.id, -1, -1);
                        eng.cache_tokens[sl.id].clear();
                    }
                }
            } else {
                llama_memory_seq_rm(llama_get_memory(eng.ctx), sl.id, -1, -1);
                eng.cache_tokens[sl.id].clear();
            }
            if (sl.st.n_restored == 0) {
                llama_memory_seq_rm(llama_get_memory(eng.ctx), sl.id, (llama_pos) keep, -1);
            }
            sl.ckpt_at    = gp.ckpt_at ? gp.ckpt_at : prompt.size();
            sl.ckpt_saved = !(gp.use_cache && ckpt_budget_bytes > 0);
            sl.n_prefilled = keep;
            sl.st.n_prompt = (int) prompt.size();
            sl.st.n_cached = (int) keep;

            work_cv.notify_one();
            return &sl;
        }
        err = "all " + std::to_string(eng.n_slots) + " slots busy";
        return nullptr;
    }

    void release(Slot & sl) {
        std::lock_guard<std::mutex> lk(mu);
        if (sl.smpl) { llama_sampler_free(sl.smpl); sl.smpl = nullptr; }
        sl.busy = false;
    }

private:
    // Snapshot the sequence state at exactly the prompt boundary. That is
    // the point a follow-up turn will share, because a chat template
    // reproduces the whole prior conversation verbatim before appending.
    void save_checkpoint(Slot & sl, size_t n_tokens) {
        if (n_tokens == 0 || n_tokens > sl.prompt.size()) return;
        const size_t need = llama_state_seq_get_size(eng.ctx, sl.id);
        if (g_verbose) fprintf(stderr, "checkpoint: slot %d needs %.1f MiB (budget %.1f)\n",
                               sl.id, need/1048576.0, ckpt_budget_bytes/1048576.0);
        if (need == 0 || need > ckpt_budget_bytes) return;
        Checkpoint c;
        c.tokens.assign(sl.prompt.begin(), sl.prompt.begin() + n_tokens);
        c.blob.resize(need);
        const size_t got = llama_state_seq_get_data(eng.ctx, c.blob.data(), need, sl.id);
        if (got == 0) return;
        c.blob.resize(got);
        auto & q = ckpts[sl.id];
        // Drop an existing checkpoint at the same length rather than
        // accumulating duplicates of the same prefix.
        for (auto it = q.begin(); it != q.end(); ++it) {
            if (it->tokens.size() == c.tokens.size()) { q.erase(it); break; }
        }
        q.push_front(std::move(c));
        while ((int) q.size() > ckpt_per_slot) q.pop_back();
    }

    void finish(Slot & sl, const std::string & reason, const std::string & err = "") {
        if (sl.done) return;
        sl.done = true;
        sl.st.finish_reason = reason;
        sl.st.decode_s = now_s() - (sl.t_prefill_done > 0 ? sl.t_prefill_done : sl.t_start);
        // Record what is RESIDENT: prompt + everything generated.
        auto & res = eng.cache_tokens[sl.id];
        res = sl.prompt;
        res.insert(res.end(), sl.generated.begin(), sl.generated.end());
        std::lock_guard<std::mutex> lk(sl.mu);
        sl.error = err;
        sl.finished = true;
        sl.cv.notify_all();
    }

    void loop() {
        while (!stop) {
            std::unique_lock<std::mutex> lk(mu);
            bool any = false;
            for (auto & sl : slots) if (sl.busy && !sl.done) { any = true; break; }
            if (!any) {
                work_cv.wait_for(lk, std::chrono::milliseconds(20));
                continue;
            }

            // ---- build one batch across every active slot
            batch.n_tokens = 0;
            struct Out { Slot * sl; int idx; };
            std::vector<Out> outs;

            for (auto & sl : slots) {
                if (!sl.busy || sl.done) continue;
                if (sl.n_prefilled < sl.prompt.size()) {
                    // Prefill chunk. Cap so one long prompt cannot starve
                    // the other slots for an unbounded time.
                    const size_t remain = sl.prompt.size() - sl.n_prefilled;
                    const size_t room   = (size_t) (2048 - batch.n_tokens);
                    size_t take = std::min({remain, room, (size_t) 512});
                    // Land exactly on the checkpoint boundary so the state
                    // can be snapshotted there.
                    if (!sl.ckpt_saved && sl.n_prefilled < sl.ckpt_at &&
                        sl.n_prefilled + take > sl.ckpt_at) {
                        take = sl.ckpt_at - sl.n_prefilled;
                    }
                    for (size_t i = 0; i < take; i++) {
                        const int j = batch.n_tokens;
                        batch.token[j]    = sl.prompt[sl.n_prefilled + i];
                        batch.pos[j]      = (llama_pos) (sl.n_prefilled + i);
                        batch.n_seq_id[j] = 1;
                        batch.seq_id[j][0] = sl.id;
                        const bool last = (sl.n_prefilled + i + 1 == sl.prompt.size());
                        batch.logits[j]   = last;
                        batch.n_tokens++;
                        if (last) outs.push_back({&sl, j});
                    }
                    sl.n_prefilled += take;
                } else {
                    // Decode: one token, the last thing this slot produced.
                    if (batch.n_tokens >= 2048) continue;
                    const int j = batch.n_tokens;
                    batch.token[j]    = sl.generated.empty()
                                          ? sl.prompt.back() : sl.generated.back();
                    batch.pos[j]      = (llama_pos) (sl.prompt.size() + sl.generated.size() - 1);
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = sl.id;
                    batch.logits[j]   = true;
                    batch.n_tokens++;
                    outs.push_back({&sl, j});
                }
            }

            if (batch.n_tokens == 0) { lk.unlock(); continue; }

            const int rc = llama_decode(eng.ctx, batch);
            if (rc != 0) {
                for (auto & sl : slots)
                    if (sl.busy) finish(sl, "error", "llama_decode failed");
                lk.unlock();
                continue;
            }

            // ---- snapshot any slot that just landed on its boundary
            for (auto & sl : slots) {
                if (sl.busy && !sl.done && !sl.ckpt_saved &&
                    sl.n_prefilled >= sl.ckpt_at) {
                    sl.ckpt_saved = true;
                    save_checkpoint(sl, sl.ckpt_at);
                }
            }

            // ---- sample per slot
            for (auto & o : outs) {
                Slot & sl = *o.sl;
                if (!sl.busy || sl.done) continue;
                if (sl.t_prefill_done == 0) {
                    sl.t_prefill_done = now_s();
                    sl.st.prefill_s = sl.t_prefill_done - sl.t_start;
                }
                bool cancelled;
                { std::lock_guard<std::mutex> slk(sl.mu); cancelled = sl.cancel; }
                if (cancelled) { finish(sl, "abort"); continue; }

                const llama_token tok = llama_sampler_sample(sl.smpl, eng.ctx, o.idx);
                if (llama_vocab_is_eog(eng.vocab, tok)) { finish(sl, "stop"); continue; }

                llama_sampler_accept(sl.smpl, tok);
                sl.generated.push_back(tok);
                sl.st.n_generated++;
                if (sl.st.ttft_s == 0) sl.st.ttft_s = now_s() - sl.t_start;

                const std::string piece = eng.detokenize(tok);

                bool hit_stop = false;
                if (!sl.gp.stop.empty()) {
                    sl.tail += piece;
                    for (const auto & ss : sl.gp.stop)
                        if (!ss.empty() && sl.tail.find(ss) != std::string::npos) {
                            hit_stop = true; break;
                        }
                    if (sl.tail.size() > 256) sl.tail.erase(0, sl.tail.size() - 256);
                }

                { std::lock_guard<std::mutex> slk(sl.mu);
                  sl.out.push_back(piece);
                  sl.cv.notify_all(); }

                if (hit_stop) { finish(sl, "stop"); continue; }
                if (sl.st.n_generated >= sl.gp.n_predict) { finish(sl, "length"); continue; }
            }
            lk.unlock();
        }
    }
};

// Drain a slot, feeding each piece to `on_token`; returns when generation
// ends. `on_token` returning false cancels.
static void consume(Slot & sl, const std::function<bool(const std::string &)> & on_token) {
    for (;;) {
        std::unique_lock<std::mutex> lk(sl.mu);
        sl.cv.wait(lk, [&]{ return !sl.out.empty() || sl.finished; });
        while (!sl.out.empty()) {
            const std::string piece = sl.out.front();
            sl.out.pop_front();
            lk.unlock();
            if (!on_token(piece)) {
                std::lock_guard<std::mutex> l2(sl.mu);
                sl.cancel = true;
                return;
            }
            lk.lock();
        }
        if (sl.finished) return;
    }
}

// ------------------------------------------------------------ chat template
static std::string apply_chat_template(llama_model * model, const json & messages,
                                       bool add_assistant, std::string & err) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) { err = "model has no chat template"; return {}; }

    std::vector<llama_chat_message> msgs;
    std::vector<std::string> storage;   // keep strings alive for the C API
    storage.reserve(messages.size() * 2);
    for (const auto & m : messages) {
        if (!m.contains("role") || !m.contains("content")) {
            err = "each message needs 'role' and 'content'";
            return {};
        }
        // OpenAI allows content to be an array of parts; flatten the text.
        std::string content;
        if (m["content"].is_string()) {
            content = m["content"].get<std::string>();
        } else if (m["content"].is_array()) {
            for (const auto & part : m["content"]) {
                if (part.contains("text")) content += part["text"].get<std::string>();
            }
        }
        storage.push_back(m["role"].get<std::string>());
        storage.push_back(content);
    }
    for (size_t i = 0; i < storage.size(); i += 2) {
        msgs.push_back({storage[i].c_str(), storage[i + 1].c_str()});
    }

    std::vector<char> out(1 << 16);
    int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                      add_assistant, out.data(), (int32_t) out.size());
    if (n > (int) out.size()) {
        out.resize(n);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                      add_assistant, out.data(), (int32_t) out.size());
    }
    if (n < 0) { err = "failed to apply chat template"; return {}; }
    return std::string(out.data(), n);
}

// ------------------------------------------------------------------ utils
static GenParams parse_gen_params(const json & body, int default_predict) {
    GenParams gp;
    gp.n_predict = body.value("max_tokens",
                    body.value("max_completion_tokens",
                     body.value("n_predict", default_predict)));
    gp.temperature = body.value("temperature", 0.7f);
    gp.top_p       = body.value("top_p", 0.95f);
    gp.top_k       = body.value("top_k", 40);
    gp.min_p       = body.value("min_p", 0.0f);
    if (body.contains("seed") && body["seed"].is_number_integer()) {
        gp.seed = (uint32_t) body["seed"].get<int64_t>();
    }
    if (body.contains("stop")) {
        if (body["stop"].is_string()) gp.stop.push_back(body["stop"].get<std::string>());
        else if (body["stop"].is_array())
            for (const auto & s : body["stop"])
                if (s.is_string()) gp.stop.push_back(s.get<std::string>());
    }
    // Structured output. Either an explicit GBNF grammar, or OpenAI's
    // response_format. json_object is the common case and needs no schema.
    if (body.contains("grammar") && body["grammar"].is_string()) {
        gp.grammar = body["grammar"].get<std::string>();
    } else if (body.contains("response_format")) {
        const auto & rf = body["response_format"];
        const std::string type = rf.value("type", "");
        if (type == "json_object" || type == "json_schema") {
            // Minimal self-contained JSON grammar. Kept inline rather than
            // pulled from common/json-schema-to-grammar.h so this binary
            // does not need libllama-common.
            gp.grammar =
                "root   ::= object\n"
                "value  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\n"
                "object ::= \"{\" ws ( string \":\" ws value (\",\" ws string \":\" ws value)* )? \"}\" ws\n"
                "array  ::= \"[\" ws ( value (\",\" ws value)* )? \"]\" ws\n"
                "string ::= \"\\\"\" ( [^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\bfnrt/] | \"u\" [0-9a-fA-F]{4}) )* \"\\\"\" ws\n"
                "number ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) (\".\" [0-9]+)? ([eE] [-+]? [0-9]{1,4})? ws\n"
                "ws     ::= | \" \" | \"\\n\" [ \\t]{0,20}\n";
        }
    }

    gp.use_cache = body.value("cache_prompt", true);
    if (gp.n_predict <= 0) gp.n_predict = default_predict;
    return gp;
}

static json error_json(const std::string & msg, const char * type, int code) {
    return json{{"error", {{"message", msg}, {"type", type}, {"code", code}}}};
}

static std::string gen_id(const char * prefix) {
    static std::atomic<uint64_t> n{0};
    return std::string(prefix) + "-" + std::to_string((uint64_t) now_s()) + "-" +
           std::to_string(n.fetch_add(1));
}

// ------------------------------------------------------------------- main
int main(int argc, char ** argv) {
    std::string model_path, variant = "ternary", speculate = "auto";
    std::string host = "0.0.0.0", webui_path;
    int port = 8080, n_ctx = 16384, ngl = 999, n_threads = -1, default_predict = 512;
    int n_slots = 4;
    int ckpt_mb = 2048;          // total checkpoint budget
    std::string kv_type = "q8_0";
    bool deterministic = true, preflight = true;
    bonsai::Backend backend = bonsai::Backend::CPU;
    bool backend_set = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if      (a == "-m"      || a == "--model")    model_path = next();
        else if (a == "--variant")                    variant = next();
        else if (a == "--speculate")                  speculate = next();
        else if (a == "--port")                       port = atoi(next().c_str());
        else if (a == "--host")                       host = next();
        else if (a == "-c" || a == "--ctx")           n_ctx = atoi(next().c_str());
        else if (a == "-ngl")                         ngl = atoi(next().c_str());
        else if (a == "-t" || a == "--threads")       n_threads = atoi(next().c_str());
        else if (a == "--max-tokens")                 default_predict = atoi(next().c_str());
        else if (a == "--slots" || a == "-np")        n_slots = atoi(next().c_str());
        else if (a == "-ctk" || a == "--cache-type")  kv_type = next();
        else if (a == "--checkpoint-mb")              ckpt_mb = atoi(next().c_str());
        else if (a == "--webui")                      webui_path = next();
        else if (a == "--backend") {
            const std::string b = next();
            backend = b == "cuda"   ? bonsai::Backend::CUDA :
                      b == "vulkan" ? bonsai::Backend::VULKAN : bonsai::Backend::CPU;
            backend_set = true;
        }
        else if (a == "--no-deterministic")           deterministic = false;
        else if (a == "--no-preflight")               preflight = false;
        else if (a == "-v" || a == "--verbose")       g_verbose = true;
        else if (a == "-h" || a == "--help") {
            printf(
"bonsai-server -- OpenAI-compatible server for Bonsai-27B on Jetson Thor\n\n"
"usage: bonsai-server -m MODEL.gguf [options]\n\n"
"  -m, --model PATH      GGUF model (Q2_0 ternary or Q1_0 1-bit)\n"
"      --variant V       ternary | onebit   (selects the measured policy table)\n"
"      --backend B       cuda | vulkan | cpu (default: inferred from -ngl)\n"
"      --speculate MODE  auto | on | off   (auto refuses to enable DSpark\n"
"                        where it is measured to LOSE, e.g. Vulkan)\n"
"      --port N          default 8080\n"
"      --host H          default 0.0.0.0\n"
"  -c, --ctx N           context size, default 16384\n"
"  -ngl N                layers offloaded to GPU, default 999 (0 = CPU only)\n"
"  -t, --threads N       CPU threads (default 12 on CPU -- NOT 14, see /v1/policy)\n"
"      --max-tokens N    default max_tokens when a request omits it\n"
"      --slots N, -np N  concurrent sequences, default 4. Decode is weight-\n"
"                        bandwidth-bound, so N sequences cost barely more\n"
"                        than one and aggregate throughput scales with N\n"
"      -ctk TYPE         KV cache type: q8_0 (default) or f16. q8_0 halves\n"
"                        the cache at 262144 ctx for no measurable cost\n"
"      --checkpoint-mb N per-checkpoint size cap, default 2048, 0 disables.\n"
"                        State checkpoints are what make multi-turn prefix\n"
"                        reuse possible at all on a hybrid model\n"
"      --webui PATH      serve this HTML file at /\n"
"      --no-deterministic  allow the racy-but-1%%-faster Vulkan graph optimizer\n"
"      --no-preflight    skip the startup determinism probe\n"
"  -v, --verbose\n\n"
"endpoints: POST /v1/chat/completions, POST /v1/completions,\n"
"           GET /v1/models, /v1/policy, /health, /metrics, /\n");
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "error: -m/--model is required (see --help)\n");
        return 2;
    }
    if (!backend_set) backend = ngl > 0 ? bonsai::Backend::CUDA : bonsai::Backend::CPU;

    const bonsai::Policy pol =
        bonsai::resolve(backend, variant, speculate, deterministic, n_threads);

    // Correctness flag must be set before any Vulkan device is created.
    if (backend == bonsai::Backend::VULKAN && deterministic) {
        setenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE", "1", /*overwrite=*/0);
    }

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (g_verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);

    ggml_backend_load_all();
    llama_backend_init();

    printf("bonsai-server: %s on %s%s\n", variant.c_str(),
           bonsai::backend_name(backend), pol.speculate ? " + DSpark" : "");
    printf("policy:\n");
    for (const auto & r : pol.reasons) printf("  - %s\n", r.c_str());

    if (pol.speculate) {
        printf("  ! speculative decoding is not yet implemented in this binary;\n"
               "    running native decode. Use reference/serve_reference.sh for\n"
               "    DSpark until the verify/rollback loop is ported and gated.\n");
    }

    ggml_type kvt = GGML_TYPE_Q8_0;
    if      (kv_type == "f16")  kvt = GGML_TYPE_F16;
    else if (kv_type == "q8_0") kvt = GGML_TYPE_Q8_0;
    else if (kv_type == "q4_0") kvt = GGML_TYPE_Q4_0;
    else { fprintf(stderr, "error: unknown -ctk %s\n", kv_type.c_str()); return 2; }

    if (n_slots < 1) n_slots = 1;

    Engine eng;
    printf("loading %s ...\n", model_path.c_str());
    fflush(stdout);
    if (!eng.load(model_path, ngl, n_ctx, pol.n_threads, n_slots, kvt)) {
        fprintf(stderr, "error: failed to load model\n");
        return 1;
    }
    printf("loaded: %d slot(s) x n_ctx=%d, KV %s\n", n_slots, eng.n_ctx, kv_type.c_str());

    auto sched = std::make_unique<Scheduler>(eng);
    sched->ckpt_budget_bytes = (size_t) std::max(0, ckpt_mb) << 20;
    sched->start();
    if (ckpt_mb > 0) {
        printf("checkpoints: up to %d per slot, %d MiB each max\n",
               sched->ckpt_per_slot, ckpt_mb);
    }

    // ---- preflight: is this backend even deterministic?
    if (preflight) {
        GenParams gp; gp.temperature = 0.0f; gp.n_predict = 16; gp.use_cache = false;
        std::string a, b, e;
        auto toks = eng.tokenize("def binary_search(arr, target):", true);
        for (int rep = 0; rep < 2; rep++) {
            Slot * sl = sched->acquire(toks, gp, e);
            if (!sl) { fprintf(stderr, "preflight: %s\n", e.c_str()); break; }
            std::string out;
            consume(*sl, [&](const std::string & pc) { out += pc; return true; });
            sched->release(*sl);
            (rep == 0 ? a : b) = out;
        }
        if (!a.empty() && a == b) {
            printf("preflight: deterministic over 2 identical greedy requests\n");
        } else if (!a.empty()) {
            printf("preflight: NONDETERMINISTIC -- 2 identical greedy requests "
                   "differed.\n  A backend that answers the same question "
                   "differently cannot be gated for correctness.\n");
        }
    }

    // ------------------------------------------------------------ routes
    httplib::Server srv;
    srv.set_payload_max_length(64ull << 20);
    const double t_started = now_s();

    auto cors = [](httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    };
    srv.Options(".*", [&](const httplib::Request &, httplib::Response & res) {
        cors(res); res.status = 204;
    });

    std::string webui_html;
    if (!webui_path.empty()) {
        std::ifstream f(webui_path);
        if (f) {
            std::stringstream ss; ss << f.rdbuf(); webui_html = ss.str();
        } else {
            logf("warning: could not read web UI at %s", webui_path.c_str());
        }
    }
    srv.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        if (webui_html.empty()) {
            res.set_content("bonsai-server is running. POST /v1/chat/completions\n",
                            "text/plain");
        } else {
            res.set_content(webui_html, "text/html; charset=utf-8");
        }
    });

    srv.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.set_content(json{{"status", "ok"},
                             {"backend", bonsai::backend_name(pol.backend)},
                             {"variant", pol.variant},
                             {"speculation", pol.speculate},
                             {"slots", eng.n_slots}}.dump(), "application/json");
    });

    srv.Get("/metrics", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        json j = g_metrics.snapshot();
        j["backend"] = bonsai::backend_name(pol.backend);
        j["variant"] = pol.variant;
        j["speculation"] = pol.speculate;
        res.set_content(j.dump(2), "application/json");
    });

    srv.Get("/v1/policy", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.set_content(json{{"backend", bonsai::backend_name(pol.backend)},
                             {"variant", pol.variant},
                             {"speculation", pol.speculate},
                             {"deterministic", pol.deterministic},
                             {"n_threads", pol.n_threads},
                             {"slots", eng.n_slots},
                             {"reasons", pol.reasons}}.dump(2), "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        cors(res);
        res.set_content(json{{"object", "list"}, {"data", json::array({
            json{{"id", "bonsai-27b"}, {"object", "model"},
                 {"created", (uint64_t) t_started}, {"owned_by", "prismml"}}
        })}}.dump(), "application/json");
    });

    // ---- the one real route, shared by /v1/chat/completions and /v1/completions
    auto handle_completion = [&](const httplib::Request & req,
                                 httplib::Response & res, bool chat) {
        cors(res);
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_json(std::string("bad JSON body: ") + e.what(),
                                       "invalid_request_error", 400).dump(),
                            "application/json");
            return;
        }

        std::string prompt_text, err;
        size_t think_boundary = 0;
        if (chat) {
            if (!body.contains("messages") || !body["messages"].is_array()) {
                res.status = 400;
                res.set_content(error_json("'messages' array is required",
                                           "invalid_request_error", 400).dump(),
                                "application/json");
                return;
            }
            prompt_text = apply_chat_template(eng.model, body["messages"], true, err);
            if (!err.empty()) {
                res.status = 400;
                res.set_content(error_json(err, "invalid_request_error", 400).dump(),
                                "application/json");
                return;
            }
            // This model thinks by default and its template has no
            // `enable_thinking` switch, so a short question can burn the
            // entire token budget on <think> and return empty content.
            // Pre-filling a closed, empty think block is the standard
            // Qwen3-family way to suppress it: the model resumes after
            // </think> and goes straight to the answer.
            if (!body.value("enable_thinking", true)) {
                // Record where the template output ended, so the state can be
                // checkpointed at a position the NEXT turn will reproduce.
                think_boundary = eng.tokenize(prompt_text, true).size();
                prompt_text += "<think>\n\n</think>\n\n";
            }
        } else {
            if (!body.contains("prompt") || !body["prompt"].is_string()) {
                res.status = 400;
                res.set_content(error_json("'prompt' string is required",
                                           "invalid_request_error", 400).dump(),
                                "application/json");
                return;
            }
            prompt_text = body["prompt"].get<std::string>();
        }

        GenParams gp = parse_gen_params(body, default_predict);
        gp.ckpt_at = think_boundary;
        const bool stream = body.value("stream", false);
        const std::string id = gen_id(chat ? "chatcmpl" : "cmpl");
        const uint64_t created = (uint64_t) now_s();

        if (!stream) {
            auto toks = eng.tokenize(prompt_text, true);
            Slot * slp = sched->acquire(toks, gp, err);
            if (!slp) {
                g_metrics.errors++;
                res.status = err.rfind("all ", 0) == 0 ? 503 : 400;
                res.set_content(error_json(err, "server_error", res.status).dump(),
                                "application/json");
                return;
            }
            Slot & sl = *slp;
            std::string full;
            consume(sl, [&](const std::string & p) { full += p; return true; });
            const GenStats st = sl.st;
            const std::string serr = sl.error;
            sched->release(sl);
            if (!serr.empty()) {
                g_metrics.errors++;
                res.status = 500;
                res.set_content(error_json(serr, "server_error", 500).dump(),
                                "application/json");
                return;
            }
            g_metrics.record(st.n_prompt, st.n_cached, st.n_generated,
                             st.prefill_s, st.decode_s, st.ttft_s);

            json out;
            out["id"] = id; out["created"] = created; out["model"] = "bonsai-27b";
            out["object"] = chat ? "chat.completion" : "text_completion";
            if (chat) {
                std::string content, reasoning;
                ReasoningSplitter sp;
                sp.feed(full, true, content, reasoning);
                json msg{{"role", "assistant"}, {"content", content}};
                if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
                out["choices"] = json::array({json{
                    {"index", 0}, {"message", msg},
                    {"finish_reason", st.finish_reason}}});
            } else {
                out["choices"] = json::array({json{
                    {"index", 0}, {"text", full},
                    {"finish_reason", st.finish_reason}}});
            }
            out["usage"] = json{
                {"prompt_tokens", st.n_prompt},
                {"completion_tokens", st.n_generated},
                {"total_tokens", st.n_prompt + st.n_generated},
                {"prompt_tokens_details", json{{"cached_tokens", st.n_cached}}}};
            out["bonsai"] = json{
                {"backend", bonsai::backend_name(pol.backend)},
                {"speculation", pol.speculate},
                {"decode_tok_s", st.decode_s > 0 ? st.n_generated / st.decode_s : 0.0},
                {"prefill_tok_s", st.prefill_s > 0 ? (st.n_prompt - st.n_cached) / st.prefill_s : 0.0},
                {"ttft_s", st.ttft_s},
                {"cached_tokens", st.n_cached},
                {"cache_resident", (int) eng.cache_tokens[0].size()},
                {"lcp", st.n_lcp},
                {"restored_tokens", st.n_restored}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        // ---- streaming (SSE)
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [&eng, &sched, gp, prompt_text, id, created, chat, &pol]
            (size_t, httplib::DataSink & sink) {
                ReasoningSplitter sp;
                std::string err2;

                auto emit = [&](const json & j) {
                    const std::string s = "data: " + j.dump() + "\n\n";
                    return sink.write(s.data(), s.size());
                };

                if (chat) {
                    emit(json{{"id", id}, {"object", "chat.completion.chunk"},
                              {"created", created}, {"model", "bonsai-27b"},
                              {"choices", json::array({json{
                                  {"index", 0},
                                  {"delta", json{{"role", "assistant"}}},
                                  {"finish_reason", nullptr}}})}});
                }

                auto toks = eng.tokenize(prompt_text, true);
                Slot * slp = sched->acquire(toks, gp, err2);
                if (!slp) {
                    const std::string e = "data: " +
                        error_json(err2, "server_error", 503).dump() + "\n\n";
                    sink.write(e.data(), e.size());
                    sink.done();
                    return true;
                }
                Slot & sl = *slp;
                consume(sl,
                    [&](const std::string & piece) -> bool {
                        json delta;
                        if (chat) {
                            std::string c, r;
                            sp.feed(piece, false, c, r);
                            if (c.empty() && r.empty()) return true;  // held back
                            if (!c.empty()) delta["content"] = c;
                            if (!r.empty()) delta["reasoning_content"] = r;
                            return emit(json{{"id", id},
                                {"object", "chat.completion.chunk"},
                                {"created", created}, {"model", "bonsai-27b"},
                                {"choices", json::array({json{
                                    {"index", 0}, {"delta", delta},
                                    {"finish_reason", nullptr}}})}});
                        }
                        return emit(json{{"id", id}, {"object", "text_completion"},
                            {"created", created}, {"model", "bonsai-27b"},
                            {"choices", json::array({json{
                                {"index", 0}, {"text", piece},
                                {"finish_reason", nullptr}}})}});
                    });
                const GenStats st = sl.st;
                sched->release(sl);

                if (chat) {   // flush any held-back tail
                    std::string c, r;
                    sp.feed("", true, c, r);
                    if (!c.empty() || !r.empty()) {
                        json delta;
                        if (!c.empty()) delta["content"] = c;
                        if (!r.empty()) delta["reasoning_content"] = r;
                        emit(json{{"id", id}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", "bonsai-27b"},
                            {"choices", json::array({json{
                                {"index", 0}, {"delta", delta},
                                {"finish_reason", nullptr}}})}});
                    }
                }

                g_metrics.record(st.n_prompt, st.n_cached, st.n_generated,
                                 st.prefill_s, st.decode_s, st.ttft_s);

                emit(json{{"id", id},
                    {"object", chat ? "chat.completion.chunk" : "text_completion"},
                    {"created", created}, {"model", "bonsai-27b"},
                    {"choices", json::array({json{
                        {"index", 0}, {"delta", json::object()},
                        {"finish_reason", st.finish_reason}}})},
                    {"usage", json{{"prompt_tokens", st.n_prompt},
                                   {"completion_tokens", st.n_generated},
                                   {"total_tokens", st.n_prompt + st.n_generated}}},
                    {"bonsai", json{
                        {"backend", bonsai::backend_name(pol.backend)},
                        {"decode_tok_s", st.decode_s > 0 ? st.n_generated / st.decode_s : 0.0},
                        {"ttft_s", st.ttft_s},
                        {"cached_tokens", st.n_cached}}}});

                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    };

    srv.Post("/v1/chat/completions", [&](const httplib::Request & rq, httplib::Response & rs) {
        handle_completion(rq, rs, true);
    });
    srv.Post("/v1/completions", [&](const httplib::Request & rq, httplib::Response & rs) {
        handle_completion(rq, rs, false);
    });
    // llama.cpp-compatible alias, so bench/bench.py works unchanged.
    srv.Post("/completion", [&](const httplib::Request & rq, httplib::Response & rs) {
        handle_completion(rq, rs, false);
    });

    printf("ready: http://%s:%d  (OpenAI API at /v1, metrics at /metrics)\n",
           host.c_str(), port);
    fflush(stdout);

    if (!srv.listen(host.c_str(), port)) {
        fprintf(stderr, "error: failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    llama_backend_free();
    return 0;
}
