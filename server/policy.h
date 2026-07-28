// policy.h -- serving defaults for Bonsai-27B, derived from measurements
// taken on this hardware. Every value here traces to a number in results/.
//
// The point of this header is that a user should not have to read the wiki
// to avoid the traps we already fell into. They are configuration, so they
// get applied by default and the server explains itself at /v1/policy.
//
//   1. Speculation is a LOSS on Vulkan (0.33-0.43x) and a large win on
//      CUDA (1.37-1.75x). A global "DSpark = on" default makes the Vulkan
//      path ~2.3x slower than doing nothing.
//   2. ggml-vulkan's graph_optimize pass makes this model nondeterministic
//      -- up to 8 distinct outputs from 10 identical greedy requests.
//      Costs 2.2% prefill / 1.0% decode to disable.
//   3. On CPU, taking all 14 cores is the WORST setting: 12 threads is
//      1.73-2.27x faster than 14, because the OS gets starved.
//
// Sources: results/bench/*.json, results/vulkan-nondeterminism.txt,
// results/gemv-cpu-threaded.txt, results/memory-knobs.txt.

#pragma once

#include <string>
#include <vector>

namespace bonsai {

enum class Backend { CUDA, VULKAN, CPU };

inline const char * backend_name(Backend b) {
    switch (b) {
        case Backend::CUDA:   return "cuda";
        case Backend::VULKAN: return "vulkan";
        default:              return "cpu";
    }
}

// Measured decode tok/s, median over 8 prompts per class, this Thor box.
struct Measured { float native_code, native_prose, spec_code, spec_prose; };

inline bool measured_for(Backend b, const std::string & variant, Measured & out) {
    const bool ternary = variant == "ternary";
    switch (b) {
        case Backend::CUDA:
            out = ternary ? Measured{16.77f, 16.97f, 24.85f, 23.23f}
                          : Measured{19.03f, 18.98f, 33.24f, 26.25f};
            return true;
        case Backend::VULKAN:
            out = ternary ? Measured{15.74f, 16.14f, 6.81f, 6.24f}
                          : Measured{19.57f, 19.60f, 8.20f, 6.43f};
            return true;
        case Backend::CPU:
            out = ternary ? Measured{2.22f, 2.15f, 2.37f, 1.71f}
                          : Measured{3.41f, 3.35f, 3.00f, 3.35f};
            return true;
    }
    return false;
}

// Worst case across BOTH workload classes. Taking the best of the two is
// how a speculation feature gets shipped that only helps on code prompts.
inline float speculation_speedup(Backend b, const std::string & variant) {
    Measured m{};
    if (!measured_for(b, variant, m)) return -1.0f;
    const float a = m.spec_code / m.native_code;
    const float c = m.spec_prose / m.native_prose;
    return a < c ? a : c;
}

struct Policy {
    Backend     backend      = Backend::CPU;
    std::string variant      = "ternary";
    bool        speculate    = false;
    bool        deterministic = true;
    int         n_threads    = 12;
    std::vector<std::string> reasons;
};

// Why a backend wins or loses at speculation, in one number.
//
// A speculative round yields (1 + block_size*alpha) tokens and costs R plain
// decode steps, so breakeven is alpha* = (R-1)/block_size. R is a property of
// the HARDWARE -- acceptance is a property of the model and prompt, and comes
// back identical on every backend. Measured on prompts whose acceptance
// differs 3x, R agrees to within 4-6%. See results/spec-round-cost.txt.
inline const char * round_cost_reason(Backend b) {
    switch (b) {
        case Backend::CUDA:
            return "round cost R = 2.18 decode steps -> breakeven acceptance "
                   "alpha* = 0.296; typical acceptance is 0.43-0.91, so there "
                   "is a wide margin";
        case Backend::VULKAN:
            // The strong form: this is not "loses on our workloads".
            return "round cost R = 7.3 decode steps -> breakeven alpha* = 1.57, "
                   "which EXCEEDS 1.0: even a drafter accepting every token "
                   "would yield 5 tokens for 7.3 steps of work. Speculation "
                   "cannot win here at any acceptance rate";
        case Backend::CPU:
        default:
            return "round cost R = 4.9 decode steps -> breakeven alpha* = 0.97; "
                   "a measured 95.0% acceptance still returned 0.95x, so this "
                   "effectively never wins";
    }
}

// `speculate`: "auto" | "on" | "off".
inline Policy resolve(Backend backend,
                      const std::string & variant,
                      const std::string & speculate,
                      bool deterministic,
                      int n_threads_req) {
    Policy p;
    p.backend = backend;
    p.variant = variant;
    p.deterministic = deterministic;

    const float gain = speculation_speedup(backend, variant);
    char buf[512];

    if (speculate == "auto") {
        if (gain > 1.05f) {
            p.speculate = true;
            snprintf(buf, sizeof(buf),
                     "speculation ON: measured %.2fx worst-case across both "
                     "workload classes on %s/%s", gain, backend_name(backend),
                     variant.c_str());
        } else {
            p.speculate = false;
            snprintf(buf, sizeof(buf),
                     "speculation OFF: measured %.2fx on %s/%s -- speculating is "
                     "SLOWER than not. Acceptance is fine (it matches CUDA to "
                     "about a point); the backend cannot execute draft-and-verify "
                     "cheaply. Override with --speculate on", gain,
                     backend_name(backend), variant.c_str());
        }
        p.reasons.push_back(buf);
        p.reasons.push_back(round_cost_reason(backend));
    } else {
        p.speculate = (speculate == "on");
        if (p.speculate && gain > 0.0f && gain <= 1.05f) {
            snprintf(buf, sizeof(buf),
                     "speculation FORCED ON against measurement (%.2fx on %s/%s); "
                     "expect it to be slower than --speculate off", gain,
                     backend_name(backend), variant.c_str());
        } else {
            snprintf(buf, sizeof(buf), "speculation forced %s by request",
                     p.speculate ? "ON" : "OFF");
        }
        p.reasons.push_back(buf);
    }

    if (backend == Backend::VULKAN) {
        if (deterministic) {
            p.reasons.push_back(
                "GGML_VK_DISABLE_GRAPH_OPTIMIZE=1: the reorder pass makes this "
                "model nondeterministic (up to 8 distinct outputs from 10 "
                "identical greedy requests) because it reorders around hazards "
                "is_src_of cannot see. Costs 2.2% prefill / 1.0% decode. "
                "Disable with --no-deterministic");
        } else {
            p.reasons.push_back(
                "determinism DISABLED by request -- output is NOT reproducible "
                "at temperature 0 on this backend");
        }
    }

    if (backend == Backend::CPU) {
        p.n_threads = n_threads_req > 0 ? n_threads_req : 12;
        snprintf(buf, sizeof(buf),
                 "-t %d: on this 14-core part, taking every core is the WORST "
                 "setting -- 12 threads measured 1.73-2.27x faster than 14, "
                 "because the OS is starved", p.n_threads);
        p.reasons.push_back(buf);
    } else {
        p.n_threads = n_threads_req > 0 ? n_threads_req : 4;
    }

    return p;
}

} // namespace bonsai
