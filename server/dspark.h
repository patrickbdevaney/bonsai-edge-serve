#pragma once
//
// DSpark speculative decoding for bonsai-server.
// ---------------------------------------------------------------------------
// DSpark is a *capture-type* drafter: instead of conditioning on a single
// embedding, it reads the target model's intermediate activations at several
// layers (for Bonsai: layers 1/16/31/46/61) concatenated per position, and
// emits a whole block of draft tokens per round. That has three consequences
// which drive every design decision in this file.
//
//   1. The target must run with `llama_set_capture_layers(..., masked=false)`,
//      and EVERY decoded position must be fed to common_speculative_process().
//      A position the target decodes without staging a capture row leaves a
//      hole the drafter cannot condition on.
//
//   2. Because of (1), prompt-cache reuse is incompatible with DSpark: a
//      restored checkpoint skips decoding (and therefore capturing) the reused
//      positions. The fork's own server disables cache reuse whenever capture
//      is engaged; we do the same, and say so out loud rather than silently
//      producing a degraded drafter.
//
//   3. Verification decodes [id_last, draft...] in one llama_decode and then
//      crops the target cache back to the accepted length. On this HYBRID
//      target (48/64 gated-delta-net) a partial crop only works if the
//      recurrent-state rollback ring was sized UP FRONT, at context creation,
//      via llama_context_params::n_rs_seq. It cannot be enabled afterwards.
//
// (3) is the sharp edge. From the fork's own common.h:
//
//     "Omitting a block-verify draft type here silently leaves n_rs_seq=0, so
//      the post-verify crop on ctx_tgt no-ops instead of failing loudly
//      (llama_memory_hybrid::seq_rm short-circuits to `return false` without
//      mutating either sub-cache) -- the target's GDN state then keeps
//      absorbing every future round's rejected draft tail."
//
// An undersized ring is therefore not a crash but a slow corruption: the
// recurrent state accumulates tokens the model never actually emitted, and the
// output drifts in a way that still reads as fluent text. This repo has been
// burned by exactly this shape before (the prefix-cache bug that "worked" at
// 7.7x TTFT while continuing the previous answer). So `guard()` below refuses
// to enable speculation unless the ring is confirmed large enough AFTER
// context creation, and the failure is a startup error, not a warning.
//
// The ring is also why draft.n_max must be >= the drafter's block_size: the
// GGUF's block_size (4 here) is the real per-round draft length, while n_max
// is what sizes the ring. The fork notes this in common_speculative_n_max():
// "callers enabling dspark should set --draft-max to at least the checkpoint's
// block_size." We read block_size from the GGUF and raise n_max ourselves
// rather than trusting the operator to have matched them.

#include "llama.h"
#include "gguf.h"

#include <atomic>
#include <string>
#include <vector>
#include <algorithm>

#if BONSAI_HAVE_DSPARK
#include "common.h"        // fork: common/common.h
#include "speculative.h"   // fork: common/speculative.h
#include "llama-ext.h"     // fork: src/llama-ext.h  (llama_set_capture_layers)
#endif

namespace bonsai {

// Read an int-array GGUF key written under "<arch>.<suffix>". The drafter
// carries its target-layer list as an array-typed KV, which llama_model's
// string-KV cache skips -- so it has to come from the file directly.
inline std::vector<int32_t> gguf_int_array(const std::string & path,
                                           const std::string & suffix) {
    gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return {};

    std::vector<int32_t> out;
    const int64_t arch_kid = gguf_find_key(g, "general.architecture");
    if (arch_kid >= 0) {
        const std::string key = std::string(gguf_get_val_str(g, arch_kid)) + "." + suffix;
        const int64_t kid = gguf_find_key(g, key.c_str());
        if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(g, kid) == GGUF_TYPE_INT32) {
            const size_t n = gguf_get_arr_n(g, kid);
            const auto * d = (const int32_t *) gguf_get_arr_data(g, kid);
            out.assign(d, d + n);
        }
    }
    gguf_free(g);
    return out;
}

inline uint32_t gguf_u32(const std::string & path, const std::string & suffix) {
    gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return 0;

    uint32_t out = 0;
    const int64_t arch_kid = gguf_find_key(g, "general.architecture");
    if (arch_kid >= 0) {
        const std::string key = std::string(gguf_get_val_str(g, arch_kid)) + "." + suffix;
        const int64_t kid = gguf_find_key(g, key.c_str());
        if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_UINT32) {
            out = gguf_get_val_u32(g, kid);
        }
    }
    gguf_free(g);
    return out;
}

struct DSparkStats {
    std::atomic<long> n_rounds  {0};   // verify batches issued
    std::atomic<long> n_drafted {0};   // draft tokens proposed
    std::atomic<long> n_accepted{0};   // draft tokens the target agreed with

    // Acceptance rate alpha: the fraction of proposed draft tokens that
    // survive verification. This is THE number for speculative decoding --
    // the breakeven condition is alpha > cost_ratio, and reporting tok/s
    // without it makes a regression indistinguishable from a hard prompt.
    double alpha() const {
        const long d = n_drafted.load();
        return d > 0 ? (double) n_accepted.load() / (double) d : 0.0;
    }
    // Mean tokens committed per target forward pass, including the free one.
    double tokens_per_round() const {
        const long r = n_rounds.load();
        return r > 0 ? (double) (n_accepted.load() + r) / (double) r : 0.0;
    }
};

#if BONSAI_HAVE_DSPARK

class DSpark {
public:
    common_params_speculative params;
    common_speculative *      spec      = nullptr;
    llama_model *             model_dft = nullptr;
    llama_context *           ctx_dft   = nullptr;

    std::vector<int32_t> capture_layers;
    uint32_t block_size = 0;
    int32_t  n_max      = 0;

    bool        enabled = false;
    std::string status;          // human-readable, surfaced on /v1/policy

    DSparkStats stats;

    // Phase 1 -- metadata only. MUST run before the target context is created,
    // because n_rs_seq is a context-creation parameter and the ring cannot be
    // resized later. Returns the ring size the target context must request.
    bool probe(const std::string & drafter_path, int32_t n_max_req, std::string & err) {
        block_size     = gguf_u32(drafter_path, "dspark.block_size");
        capture_layers = gguf_int_array(drafter_path, "dspark.target_layers");

        if (capture_layers.empty()) {
            err = "drafter has no <arch>.dspark.target_layers array -- not a dspark checkpoint";
            return false;
        }
        if (block_size == 0) {
            err = "drafter has no <arch>.dspark.block_size";
            return false;
        }

        // The GGUF's block_size is the real per-round draft length; n_max only
        // sizes the rollback ring. If n_max < block_size the ring is too small
        // to undo a fully-rejected block, so raise it rather than let the crop
        // silently no-op.
        n_max = std::max(n_max_req, (int32_t) block_size);

        params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        params.draft.mparams.path = drafter_path;
        params.draft.n_max        = n_max;
        params.draft.n_min        = 0;
        return true;
    }

    uint32_t need_n_rs_seq() const { return params.need_n_rs_seq(); }

    // Outputs the TARGET context must be able to produce per step: every row of
    // a verify batch carries logits, for every slot.
    uint32_t need_n_outputs(int n_slots) const {
        return (uint32_t) n_slots * (1u + (uint32_t) n_max);
    }

    // Phase 2 -- after the target context exists. Verifies the ring actually
    // came out big enough, engages capture, builds the drafter context and the
    // speculative context.
    bool init(llama_context * ctx_tgt, int n_slots, int n_ctx_per_seq,
              int ngl, std::string & err) {
        // The guard. An undersized ring does not fail loudly at rollback time
        // (llama_memory_hybrid::seq_rm returns false without mutating), so the
        // only place to catch it is here, against the context we actually got.
        const uint32_t ring = llama_n_rs_seq(ctx_tgt);
        if (ring < (uint32_t) n_max) {
            err = "recurrent rollback ring is " + std::to_string(ring) +
                  " but a draft block needs " + std::to_string(n_max) +
                  "; the target context was not created with n_rs_seq >= draft-max. "
                  "Refusing to speculate: partial rollback on a hybrid model would "
                  "silently corrupt the gated-delta-net state instead of failing.";
            return false;
        }

        llama_set_capture_layers(ctx_tgt, capture_layers.data(),
                                 capture_layers.size(), /* masked = */ false);

        // Drafter context. Two sizing rules, both from the fork:
        //  - n_batch must cover n_ctx + block_size, because dspark stages every
        //    context row since its cache position PLUS a full block in ONE
        //    batch. If it does not fit, the round is skipped and speculation
        //    silently degrades to plain autoregressive decode.
        //  - n_outputs_max must cover (1 + block_size) rows per sequence.
        //  - n_rs_seq stays 0: the drafter's own cache is managed inside
        //    common_speculative_draft(), never rolled back by us.
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = ngl;
        model_dft = llama_model_load_from_file(params.draft.mparams.path.c_str(), mp);
        if (!model_dft) { err = "failed to load drafter"; return false; }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx        = (uint32_t) n_ctx_per_seq * n_slots;
        cp.n_seq_max    = (uint32_t) n_slots;
        cp.n_batch      = (uint32_t) n_ctx_per_seq + block_size;
        cp.n_ubatch     = cp.n_batch;
        cp.n_outputs_max = (uint32_t) n_slots * (1u + block_size);
        cp.n_rs_seq     = 0;
        cp.ctx_other    = ctx_tgt;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        ctx_dft = llama_init_from_model(model_dft, cp);
        if (!ctx_dft) { err = "failed to create drafter context"; return false; }

        params.draft.ctx_tgt = ctx_tgt;
        params.draft.ctx_dft = ctx_dft;

        spec = common_speculative_init(params, (uint32_t) n_slots);
        if (!spec) { err = "common_speculative_init failed"; return false; }

        if (!common_speculative_need_embd_capture(spec)) {
            err = "speculative context did not request tap capture -- drafter is not dspark";
            return false;
        }

        enabled = true;
        status  = "dspark, block_size=" + std::to_string(block_size) +
                  ", taps=" + std::to_string(capture_layers.size()) +
                  ", rollback ring=" + std::to_string(ring);
        return true;
    }

    // ---- the per-step interface the scheduler drives.
    // Wrapped here so bonsai-server.cpp never names a common_speculative
    // symbol directly, and so the whole feature can compile out against a
    // fork build that does not ship libllama-common.

    // Clears the capture staging window; must precede the prompt decode.
    void begin(int seq, const std::vector<llama_token> & prompt) {
        common_speculative_begin(spec, seq, prompt);
    }

    // Mark one sequence for drafting this round.
    void want_draft(int seq, int n_past, llama_token id_last,
                    const std::vector<llama_token> * prompt,
                    std::vector<llama_token> * result) {
        auto & dp = common_speculative_get_draft_params(spec, seq);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = n_past;
        dp.id_last  = id_last;
        dp.prompt   = prompt;
        dp.result   = result;
    }

    // One batched drafter pass covering every sequence marked above; resets
    // all drafting flags on return.
    void draft_all() { common_speculative_draft(spec); }

    // Stage a decoded batch's captured tap features. Must follow EVERY decode.
    bool process(const llama_batch & b) { return common_speculative_process(spec, b); }

    void accept(int seq, int n) {
        common_speculative_accept(spec, seq, (uint16_t) n);
    }

    ~DSpark() {
        if (spec)      common_speculative_free(spec);
        if (ctx_dft)   llama_free(ctx_dft);
        if (model_dft) llama_model_free(model_dft);
    }
};

#else  // !BONSAI_HAVE_DSPARK

// Stub: the fork build linked against does not ship libllama-common, so the
// speculative API is unavailable. Everything reports disabled and the server
// runs plain autoregressive decode. probe() fails loudly rather than letting
// a --model-draft flag look like it took effect.
class DSpark {
public:
    std::vector<int32_t> capture_layers;
    uint32_t block_size = 0;
    int32_t  n_max      = 0;
    bool        enabled = false;
    std::string status  = "unavailable (built without libllama-common)";
    DSparkStats stats;

    bool probe(const std::string &, int32_t, std::string & err) {
        err = "this binary was built without speculative support "
              "(no libllama-common.so in the fork build directory)";
        return false;
    }
    uint32_t need_n_rs_seq() const { return 0; }
    uint32_t need_n_outputs(int) const { return 0; }
    bool init(llama_context *, int, int, int, std::string & err) {
        err = status; return false;
    }
    void begin(int, const std::vector<llama_token> &) {}
    void want_draft(int, int, llama_token, const std::vector<llama_token> *,
                    std::vector<llama_token> *) {}
    void draft_all() {}
    bool process(const llama_batch &) { return true; }
    void accept(int, int) {}
};

#endif // BONSAI_HAVE_DSPARK

} // namespace bonsai
