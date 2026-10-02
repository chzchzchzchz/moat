/*
 * Contract tests for the unified C bridge (src/antigravity_engine_c.cpp).
 *
 * The bridge is pure C++ — the Metal/Objective-C++ engine sits behind the opaque
 * antigravity_c_api.h handle — so it can be exercised on any platform by linking
 * against stub implementations of that API. Nothing here touches the GPU.
 *
 * This covers the buffer layout the bridge assumes between generation and
 * verification, which tests/test_cpp_sdk.cpp does not: that test calls
 * AntigravityEngineNativeGenerate directly and never goes through this bridge.
 * A mis-built AntigravityMCTSConfig ({ n_channels, 3, 4, ... } positionally sets
 * chunk_tokens = n_channels) once capped output at n_channels * 3 tokens and left
 * every channel but the first decoding a zero-filled tail, undetected.
 *
 * Build and run:
 *   c++ -std=c++17 -Wall -Wextra -Isrc \
 *       src/antigravity_engine_c.cpp tests/test_bridge_contract.cpp \
 *       -o bin/test_bridge_contract && ./bin/test_bridge_contract
 */

#include "antigravity_engine_c.h"
#include "antigravity_c_api.h"
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Stub engine: stands in for the Metal C API.
// ---------------------------------------------------------------------------
// Stubs for the Objective-C++/Metal C API so the pure-C++ bridge can be exercised on Linux.

struct AntigravityEngineContext { int n_channels; };
static AntigravityEngineContext g_ctx;

// Records what the bridge asked for, so the test can assert on it.
int  g_last_max_new_tokens = -1;
int  g_last_prompt_len = -1;

AntigravityEngineContext* AntigravityEngineCreate(const AntigravityConfig* cfg) {
    g_ctx.n_channels = cfg ? cfg->n_channels : 0;
    return &g_ctx;
}
void AntigravityEngineDestroy(AntigravityEngineContext*) {}
int32_t AntigravityEngineLoadModel(AntigravityEngineContext*, const char*) { return 0; }
void AntigravityEngineSanitizeBuffers(AntigravityEngineContext*) {}

// Emits a distinct, recognisable token stream per channel, with differing lengths,
// exactly as the real NativeGenerate would fill [n_channels * max_new_tokens].
int32_t AntigravityEngineNativeGenerate(
    AntigravityEngineContext* ctx, const int32_t*, int32_t prompt_len,
    int32_t max_new_tokens, float, float,
    int32_t* out_tokens, float* out_logprobs, int32_t* out_counts,
    double* ttft, double* total
) {
    g_last_max_new_tokens = max_new_tokens;
    g_last_prompt_len = prompt_len;
    int N = ctx->n_channels;
    for (int c = 0; c < N; c++) {
        int n = max_new_tokens - c;             // channel c is one token shorter
        if (n < 1) n = 1;
        out_counts[c] = n;
        out_logprobs[c] = -1.0f * (float)(N - c);   // channel N-1 scores highest
        for (int t = 0; t < max_new_tokens; t++) {
            out_tokens[c * max_new_tokens + t] = (t < n) ? (1000 * (c + 1) + t) : 0;
        }
    }
    if (ttft) *ttft = 1.0;
    if (total) *total = 42.0;
    return 0;
}

// Echoes back the tokens it was asked to score so the test can inspect them.
std::vector<int32_t> g_scored;
int32_t g_scored_seq_len = 0;
int32_t AntigravityEngineVerifyCandidates(
    AntigravityEngineContext* ctx, const int32_t* candidate_tokens,
    int32_t seq_len, float* out_scores
) {
    int N = ctx->n_channels;
    g_scored_seq_len = seq_len;
    g_scored.assign(candidate_tokens, candidate_tokens + (size_t)N * seq_len);
    for (int c = 0; c < N; c++) out_scores[c] = 0.5f + 0.1f * c;
    return N - 1;
}

// ---------------------------------------------------------------------------
// Contract checks
// ---------------------------------------------------------------------------

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

int main() {
    antigravity_config_t cfg{};
    cfg.memory_limit_bytes = 4500ll * 1024 * 1024;
    cfg.parallel_channels = 4;
    cfg.reflection_threshold = 0.75f;
    cfg.enable_lut = true;

    antigravity_engine_t eng = antigravity_engine_create(&cfg, "/nonexistent/model.safetensors");
    if (!eng) { printf("[FAIL]  engine_create returned NULL\n"); return 1; }

    const uint32_t MAXTOK = 16;
    antigravity_rollout_result_t* res =
        antigravity_generate_rollouts(eng, "hello world", MAXTOK, 0.7f);
    if (!res) { printf("[FAIL]  generate_rollouts returned NULL\n"); return 1; }

    check(g_last_max_new_tokens == (int)MAXTOK,
          "max_tokens is forwarded to the engine verbatim (not capped by a mis-built config)");
    check(res->candidate_count == 4, "one candidate per channel");

    // Every channel must carry its own real tokens, not a zero-filled tail.
    bool all_distinct = true, none_empty = true, counts_ok = true;
    for (uint32_t c = 0; c < res->candidate_count; c++) {
        if (res->candidates[c].token_count == 0) none_empty = false;
        if (res->candidates[c].token_count != MAXTOK - c) counts_ok = false;
    }
    for (uint32_t c = 1; c < res->candidate_count; c++) {
        if (res->candidates[c].logprob == res->candidates[0].logprob) all_distinct = false;
    }
    check(none_empty, "no channel decodes an empty/zero sequence");
    check(counts_ok, "each candidate reports its own real token count");
    check(all_distinct, "each candidate carries its own per-channel logprob");
    check(res->best_candidate_index == 3, "best index follows the highest per-channel logprob");
    check(res->total_latency_ms == 42.0, "latency comes from the generate call");

    antigravity_verification_result_t* v = antigravity_verify_candidates(eng, res);
    if (!v) { printf("[FAIL]  verify_candidates returned NULL\n"); return 1; }

    check(g_scored_seq_len == (int32_t)MAXTOK, "verification spans the longest candidate");

    // Channel c holds 1000*(c+1)+t for t < count, then pads with its own final token.
    bool scored_real = true, pad_ok = true;
    for (uint32_t c = 0; c < 4; c++) {
        uint32_t n = MAXTOK - c;
        for (uint32_t t = 0; t < n; t++) {
            if (g_scored[c * MAXTOK + t] != (int32_t)(1000 * (c + 1) + t)) scored_real = false;
        }
        for (uint32_t t = n; t < MAXTOK; t++) {
            if (g_scored[c * MAXTOK + t] != (int32_t)(1000 * (c + 1) + n - 1)) pad_ok = false;
        }
    }
    check(scored_real, "verifier scores the engine's own token IDs");
    check(pad_ok, "short candidates pad with their own final token, never noise or zeros");

    antigravity_free_verification_result(v);
    antigravity_free_rollout_result(res);
    antigravity_engine_destroy(eng);

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
