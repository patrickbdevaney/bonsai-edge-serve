// chat.cpp -- terminal chat client for bonsai-server.
//
// Streaming multi-turn REPL with dimmed thinking blocks and a tok/s
// readout. Talks plain OpenAI, so it works against any compatible endpoint,
// including the fork's own llama-server.
//
// build: see server/build.sh (it builds this too)
// run:   ./build/chat [host] [port]

#include "httplib.h"
#include "json.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    const std::string host = argc > 1 ? argv[1] : "localhost";
    const int port = argc > 2 ? atoi(argv[2]) : 8085;

    // THINK=1 keeps the model's reasoning on; default off, because this
    // model will otherwise spend a short answer's whole budget thinking.
    const bool think = getenv("THINK") != nullptr;
    const float temp = getenv("TEMP") ? atof(getenv("TEMP")) : 0.7f;
    const int   maxt = getenv("MAXTOK") ? atoi(getenv("MAXTOK")) : 768;

    httplib::Client cli(host, port);
    cli.set_read_timeout(3600, 0);

    // Show what we are talking to, and why it is configured that way.
    std::string banner = host + ":" + std::to_string(port);
    if (auto r = cli.Get("/v1/policy"); r && r->status == 200) {
        try {
            auto p = json::parse(r->body);
            banner = p.value("variant", "?") + " · " + p.value("backend", "?") +
                     (p.value("speculation", false) ? " · DSpark" : "") +
                     (p.value("deterministic", true) ? " · deterministic" : "");
        } catch (...) {}
    }

    printf("\033[1;32m▌ Bonsai-27B · Thor\033[0m  %s\n", banner.c_str());
    printf("\033[2m  /new resets · /policy shows why · Ctrl-D quits · "
           "THINK=1 TEMP=.. MAXTOK=..\033[0m\n");

    json msgs = json::array();
    std::string line;
    for (;;) {
        printf("\n\033[1;36myou ›\033[0m ");
        fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/new") {
            msgs.clear();
            printf("\033[2m[history cleared]\033[0m\n");
            continue;
        }
        if (line == "/policy") {
            if (auto r = cli.Get("/v1/policy"); r && r->status == 200) {
                try {
                    for (const auto & s : json::parse(r->body)["reasons"])
                        printf("\033[2m  - %s\033[0m\n", s.get<std::string>().c_str());
                } catch (...) {}
            }
            continue;
        }

        msgs.push_back({{"role", "user"}, {"content", line}});
        const json body = {
            {"messages", msgs}, {"stream", true}, {"max_tokens", maxt},
            {"temperature", temp}, {"enable_thinking", think},
        };

        printf("\033[1;32mbonsai ›\033[0m ");
        fflush(stdout);

        std::string buf, answer, stats;
        bool in_reasoning = false;
        int ntok = 0;
        const auto t0 = std::chrono::steady_clock::now();

        cli.Post("/v1/chat/completions", httplib::Headers{}, body.dump(),
                 "application/json",
                 [&](const char * data, size_t len) -> bool {
            buf.append(data, len);
            size_t i;
            while ((i = buf.find("\n\n")) != std::string::npos) {
                const std::string ln = buf.substr(0, i);
                buf.erase(0, i + 2);
                if (ln.rfind("data: ", 0) != 0) continue;
                const std::string payload = ln.substr(6);
                if (payload == "[DONE]") continue;
                try {
                    auto j = json::parse(payload);
                    if (j.contains("choices") && !j["choices"].empty()) {
                        const auto & d = j["choices"][0]["delta"];
                        if (d.contains("reasoning_content")) {
                            if (!in_reasoning) { printf("\033[2m🤔 "); in_reasoning = true; }
                            printf("%s", d["reasoning_content"].get<std::string>().c_str());
                        }
                        if (d.contains("content")) {
                            if (in_reasoning) { printf("\033[0m\n"); in_reasoning = false; }
                            const auto c = d["content"].get<std::string>();
                            printf("%s", c.c_str());
                            answer += c;
                            ntok++;
                        }
                    }
                    if (j.contains("bonsai") && j["bonsai"].contains("decode_tok_s")) {
                        char s[192];
                        snprintf(s, sizeof(s), "%.1f tok/s · TTFT %.2fs · %d cached",
                                 j["bonsai"].value("decode_tok_s", 0.0),
                                 j["bonsai"].value("ttft_s", 0.0),
                                 j["bonsai"].value("cached_tokens", 0));
                        stats = s;
                    }
                } catch (...) {}
                fflush(stdout);
            }
            return true;
        });

        if (in_reasoning) printf("\033[0m");
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (stats.empty() && dt > 0) {
            char s[96];
            snprintf(s, sizeof(s), "%.1f tok/s", ntok / dt);
            stats = s;
        }
        printf("\n\033[2m   [%s]\033[0m\n", stats.c_str());

        msgs.push_back({{"role", "assistant"}, {"content", answer}});
    }
    printf("\n");
    return 0;
}
