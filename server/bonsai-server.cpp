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
#include <mutex>
#include <sstream>
#include <string>
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
};

class Engine {
public:
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    std::mutex mu;                 // one generation at a time; requests queue
    std::vector<llama_token> cache_tokens;  // prompt + generated, resident in KV
    bool recurrent = false;                 // hybrid/recurrent state cannot be truncated
    int n_ctx = 0;

    bool load(const std::string & path, int ngl, int n_ctx_req, int n_threads) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = ngl;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) return false;
        vocab = llama_model_get_vocab(model);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx       = n_ctx_req;
        cp.n_batch     = 2048;
        cp.n_ubatch    = 512;
        cp.n_threads   = n_threads;
        cp.n_threads_batch = n_threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) return false;
        n_ctx = (int) llama_n_ctx(ctx);
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
    size_t reusable_prefix(const std::vector<llama_token> & prompt) const {
        size_t i = 0;
        const size_t n = std::min(cache_tokens.size(), prompt.size());
        while (i < n && cache_tokens[i] == prompt[i]) i++;

        if (recurrent) {
            if (i < cache_tokens.size()) return 0;  // truncation: unsafe
            if (i == prompt.size())      return 0;  // nothing left to decode
            return i;                               // pure append: safe
        }

        // Never reuse the whole prompt: at least one token must be decoded
        // to produce logits for the first sampled token.
        if (i == prompt.size() && i > 0) i--;
        return i;
    }
};

// The callback returns false to abort generation (client disconnected).
using TokenCb = std::function<bool(const std::string & piece)>;

struct GenStats {
    int    n_prompt = 0, n_cached = 0, n_generated = 0, n_lcp = 0;
    double prefill_s = 0, decode_s = 0, ttft_s = 0;
    std::string finish_reason = "stop";
};

static bool generate(Engine & eng, const std::vector<llama_token> & prompt,
                     const GenParams & gp, const TokenCb & on_token,
                     GenStats & st, std::string & err, bool use_cache = true) {
    if ((int) prompt.size() + gp.n_predict > eng.n_ctx) {
        err = "prompt + max_tokens exceeds context (" +
              std::to_string(prompt.size()) + " + " +
              std::to_string(gp.n_predict) + " > " + std::to_string(eng.n_ctx) + ")";
        return false;
    }

    // NOTE ON REPRODUCIBILITY: a cached prefill and a fresh prefill are not
    // numerically identical. Reusing N tokens means the tail is decoded in a
    // differently-shaped batch, which changes reduction order and therefore
    // the low bits of the logits. Greedy output is reproducible for a FIXED
    // cache state, not across cache states. Gates that need bit-stability
    // must pass cache_prompt=false, exactly as bench/gate_determinism.sh does.
    llama_memory_t mem = llama_get_memory(eng.ctx);
    size_t keep = 0;
    {   // raw LCP, for telemetry -- distinguishes "prefix diverged" from
        // "prefix matched but reuse was refused as unsafe"
        size_t i = 0;
        const size_t n = std::min(eng.cache_tokens.size(), prompt.size());
        while (i < n && eng.cache_tokens[i] == prompt[i]) i++;
        st.n_lcp = (int) i;
    }
    if (use_cache) {
        keep = eng.reusable_prefix(prompt);
    } else {
        llama_memory_clear(mem, true);
        eng.cache_tokens.clear();
    }
    st.n_prompt = (int) prompt.size();
    st.n_cached = (int) keep;

    // Drop everything after the shared prefix, then decode only the tail.
    llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1);

    const double t_start = now_s();

    std::vector<llama_token> todo(prompt.begin() + keep, prompt.end());
    llama_batch batch = llama_batch_get_one(todo.data(), (int32_t) todo.size());
    if (llama_decode(eng.ctx, batch) != 0) {
        err = "llama_decode failed on the prompt";
        return false;
    }
    const double t_prefill = now_s();
    st.prefill_s = t_prefill - t_start;

    llama_sampler * smpl = eng.make_sampler(gp);
    std::string tail;                 // rolling buffer for stop-string match
    size_t longest_stop = 0;
    for (const auto & s : gp.stop) longest_stop = std::max(longest_stop, s.size());

    bool aborted = false;
    std::vector<llama_token> generated;
    for (int i = 0; i < gp.n_predict; i++) {
        llama_token tok = llama_sampler_sample(smpl, eng.ctx, -1);
        if (llama_vocab_is_eog(eng.vocab, tok)) { st.finish_reason = "stop"; break; }

        const std::string piece = eng.detokenize(tok);
        generated.push_back(tok);
        st.n_generated++;
        if (st.ttft_s == 0) st.ttft_s = now_s() - t_start;

        // Stop strings can straddle token boundaries, so match on a rolling
        // tail rather than on the individual piece.
        bool hit_stop = false;
        if (!gp.stop.empty()) {
            tail += piece;
            for (const auto & s : gp.stop) {
                if (!s.empty() && tail.find(s) != std::string::npos) { hit_stop = true; break; }
            }
            if (tail.size() > 2 * longest_stop + 16) {
                tail.erase(0, tail.size() - (2 * longest_stop + 16));
            }
        }

        if (!on_token(piece)) { aborted = true; st.finish_reason = "abort"; break; }
        if (hit_stop) { st.finish_reason = "stop"; break; }

        if (i + 1 == gp.n_predict) { st.finish_reason = "length"; break; }

        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(eng.ctx, batch) != 0) {
            err = "llama_decode failed during generation";
            llama_sampler_free(smpl);
            return false;
        }
    }
    llama_sampler_free(smpl);
    st.decode_s = now_s() - t_prefill;

    // Record what is now RESIDENT, which is prompt + everything generated --
    // not just the prompt. Recording only the prompt is what made a repeated
    // request continue the previous answer: the next call saw a full prefix
    // match, kept the KV, and the recurrent state still held the generated
    // tokens.
    eng.cache_tokens = prompt;
    eng.cache_tokens.insert(eng.cache_tokens.end(), generated.begin(), generated.end());
    (void) aborted;
    return true;
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

    Engine eng;
    printf("loading %s ...\n", model_path.c_str());
    fflush(stdout);
    if (!eng.load(model_path, ngl, n_ctx, pol.n_threads)) {
        fprintf(stderr, "error: failed to load model\n");
        return 1;
    }
    printf("loaded: n_ctx=%d\n", eng.n_ctx);

    // ---- preflight: is this backend even deterministic?
    if (preflight) {
        GenParams gp; gp.temperature = 0.0f; gp.n_predict = 16;
        std::string a, b, err;
        for (int rep = 0; rep < 2; rep++) {
            std::string out;
            eng.cache_tokens.clear();   // force a real prefill each time --
                                        // caching would replay the first one
                                        // and hide exactly this class of bug
            llama_memory_clear(llama_get_memory(eng.ctx), true);
            auto toks = eng.tokenize("def binary_search(arr, target):", true);
            GenStats st;
            if (!generate(eng, toks, gp, [&](const std::string & p) {
                    out += p; return true; }, st, err)) {
                fprintf(stderr, "preflight: generation failed: %s\n", err.c_str());
                break;
            }
            (rep == 0 ? a : b) = out;
        }
        eng.cache_tokens.clear();
        llama_memory_clear(llama_get_memory(eng.ctx), true);
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
                             {"speculation", pol.speculate}}.dump(), "application/json");
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

        const GenParams gp = parse_gen_params(body, default_predict);
        const bool stream = body.value("stream", false);
        // llama.cpp-compatible knob. Default on (the agentic win), but a
        // correctness gate needs it off -- see the note in generate().
        const bool use_cache = body.value("cache_prompt", true);
        const std::string id = gen_id(chat ? "chatcmpl" : "cmpl");
        const uint64_t created = (uint64_t) now_s();

        if (!stream) {
            std::lock_guard<std::mutex> lk(eng.mu);
            auto toks = eng.tokenize(prompt_text, true);
            std::string full;
            GenStats st;
            if (!generate(eng, toks, gp,
                          [&](const std::string & p) { full += p; return true; },
                          st, err, use_cache)) {
                g_metrics.errors++;
                res.status = 500;
                res.set_content(error_json(err, "server_error", 500).dump(),
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
                {"cache_resident", (int) eng.cache_tokens.size()},
                {"lcp", st.n_lcp}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        // ---- streaming (SSE)
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [&eng, gp, prompt_text, id, created, chat, use_cache, &pol]
            (size_t, httplib::DataSink & sink) {
                std::lock_guard<std::mutex> lk(eng.mu);
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
                GenStats st;
                const bool ok = generate(eng, toks, gp,
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
                    }, st, err2, use_cache);

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

                if (!ok) g_metrics.errors++;
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
