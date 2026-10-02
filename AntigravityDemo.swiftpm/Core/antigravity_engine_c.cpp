/* Project Antigravity — Unified C API Engine Bridge */
#include "antigravity_engine_c.h"
#include "antigravity_c_api.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>

struct AntigravityEngineInternal {
    antigravity_config_t config;
    std::string model_path;
    AntigravityEngineContext* ctx;
    bool weights_loaded;
};

extern "C" {

antigravity_engine_t antigravity_engine_create(const antigravity_config_t* config, const char* model_path) {
    if (!config || !model_path) return NULL;

    AntigravityEngineInternal* engine = new AntigravityEngineInternal();
    engine->config = *config;
    engine->model_path = std::string(model_path);

    AntigravityConfig api_config;
    api_config.n_channels = config->parallel_channels > 0 ? (int32_t)config->parallel_channels : 8;
    api_config.vocab_size = 32000;
    api_config.hidden_dim = 2048;
    api_config.max_seq_len = 2048;
    api_config.use_metal_gpu = true;

    engine->ctx = AntigravityEngineCreate(&api_config);
    if (!engine->ctx) {
        delete engine;
        return NULL;
    }

    engine->weights_loaded = false;
    if (model_path && strlen(model_path) > 0) {
        int32_t load_rc = AntigravityEngineLoadModel(engine->ctx, model_path);
        if (load_rc != 0) {
            // Weight loading is not optional: without it there is nothing to infer from.
            // Fail construction rather than handing back an engine that cannot generate.
            std::cerr << "[AntigravityEngine] Failed to load model weights from '"
                      << model_path << "' (rc=" << load_rc << ")." << std::endl;
            AntigravityEngineDestroy(engine->ctx);
            delete engine;
            return NULL;
        }
        engine->weights_loaded = true;
    } else {
        std::cerr << "[AntigravityEngine] No model path supplied; generation will be refused."
                  << std::endl;
    }
    return engine;
}

void antigravity_engine_destroy(antigravity_engine_t engine) {
    if (engine) {
        if (engine->ctx) {
            AntigravityEngineDestroy(engine->ctx);
            engine->ctx = NULL;
        }
        delete engine;
    }
}

antigravity_rollout_result_t* antigravity_generate_rollouts_tokens(
    antigravity_engine_t engine,
    const int32_t* prompt_tokens,
    uint32_t prompt_len,
    uint32_t max_tokens,
    float temperature,
    float top_p
) {
    if (!engine || !engine->ctx || !prompt_tokens || prompt_len == 0 || max_tokens == 0) return NULL;

    if (!engine->weights_loaded) {
        std::cerr << "[AntigravityEngine] Refusing to generate: no model weights are loaded."
                  << std::endl;
        return NULL;
    }

    uint32_t n_channels = engine->config.parallel_channels > 0 ? engine->config.parallel_channels : 8;

    // Real inference: full multi-layer Metal transformer forward pass.
    std::vector<int32_t> out_tokens(n_channels * max_tokens, 0);
    std::vector<float>   out_logprobs(n_channels, 0.0f);
    std::vector<int32_t> out_counts(n_channels, 0);
    double ttft_ms = 0.0;
    double total_ms = 0.0;

    int ret = AntigravityEngineNativeGenerate(
        engine->ctx,
        prompt_tokens,
        (int32_t)prompt_len,
        (int32_t)max_tokens,
        temperature,
        top_p,
        out_tokens.data(),
        out_logprobs.data(),
        out_counts.data(),
        &ttft_ms,
        &total_ms
    );

    if (ret != 0) {
        std::cerr << "[AntigravityEngine] AntigravityEngineNativeGenerate failed (rc=" << ret << ")."
                  << std::endl;
        return NULL;
    }

    antigravity_rollout_result_t* res = new antigravity_rollout_result_t();
    res->candidate_count = n_channels;
    res->candidates = new antigravity_candidate_t[n_channels];

    uint32_t best_idx = 0;
    float max_logprob = -std::numeric_limits<float>::infinity();

    for (uint32_t c = 0; c < n_channels; c++) {
        uint32_t n_toks = (uint32_t)std::max(0, out_counts[c]);
        if (n_toks > max_tokens) n_toks = max_tokens;

        // Carry the engine's real token IDs. This target links no tokenizer, so there is
        // no text to report: trace_text stays NULL rather than holding invented text.
        int32_t* ids = new int32_t[n_toks > 0 ? n_toks : 1];
        for (uint32_t s = 0; s < n_toks; s++) {
            ids[s] = out_tokens[c * max_tokens + s];
        }

        res->candidates[c].trace_text = NULL;
        res->candidates[c].token_ids  = ids;
        res->candidates[c].logprob    = out_logprobs[c];
        res->candidates[c].token_count = n_toks;

        if (out_logprobs[c] > max_logprob) {
            max_logprob = out_logprobs[c];
            best_idx = c;
        }
    }

    res->best_candidate_index = best_idx;
    res->total_latency_ms = total_ms;
    res->token_savings_pct = 0.0f;
    res->reflection_triggered = false;

    return res;
}

antigravity_rollout_result_t* antigravity_generate_rollouts(
    antigravity_engine_t engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature
) {
    (void)engine; (void)prompt; (void)max_tokens; (void)temperature;

    // This target links no tokenizer, so `prompt` cannot be turned into token IDs. Earlier
    // revisions papered over that by decoding PRNG-seeded activations into "tok_<id>" strings
    // and reporting them as reasoning traces. Refusing is the honest behaviour: tokenize in
    // the host app and call antigravity_generate_rollouts_tokens().
    std::cerr << "[AntigravityEngine] antigravity_generate_rollouts() requires a tokenizer, "
              << "which this target does not link. Tokenize the prompt in the host app and "
              << "call antigravity_generate_rollouts_tokens() instead." << std::endl;
    return NULL;
}

antigravity_verification_result_t* antigravity_verify_candidates(
    antigravity_engine_t engine,
    const antigravity_rollout_result_t* rollouts
) {
    if (!engine || !engine->ctx || !rollouts || rollouts->candidate_count == 0) return NULL;

    uint32_t n_channels = rollouts->candidate_count;
    uint32_t seq_len = rollouts->candidates[0].token_count;
    if (seq_len == 0) return NULL;

    // Score the engine's own token IDs. Earlier revisions recovered tokens by scanning the
    // trace string for "tok_" and substituted PRNG values when that failed, which scored
    // invented data. Missing token IDs are now an error instead.
    std::vector<int32_t> candidate_tokens(n_channels * seq_len, 0);
    for (uint32_t c = 0; c < n_channels; c++) {
        const antigravity_candidate_t& cand = rollouts->candidates[c];
        if (!cand.token_ids || cand.token_count == 0) {
            std::cerr << "[AntigravityEngine] Candidate " << c << " carries no token IDs; "
                      << "refusing to verify fabricated token data." << std::endl;
            return NULL;
        }
        for (uint32_t s = 0; s < seq_len; s++) {
            // Shorter candidates are padded with their own final token, never with noise.
            uint32_t idx = (s < cand.token_count) ? s : (cand.token_count - 1);
            candidate_tokens[c * seq_len + s] = cand.token_ids[idx];
        }
    }

    std::vector<float> scores(n_channels, 0.0f);
    int32_t best_channel = AntigravityEngineVerifyCandidates(
        engine->ctx,
        candidate_tokens.data(),
        (int32_t)seq_len,
        scores.data()
    );

    antigravity_verification_result_t* vres = new antigravity_verification_result_t();
    vres->selected_index = (uint32_t)(best_channel >= 0 ? best_channel : 0);
    vres->confidence_score = scores[vres->selected_index];

    std::string reasoning = "Verified candidate channel " + std::to_string(vres->selected_index) +
                            " with confidence score " + std::to_string(vres->confidence_score);
    vres->verifier_reasoning = strdup(reasoning.c_str());

    return vres;
}

void antigravity_free_rollout_result(antigravity_rollout_result_t* result) {
    if (!result) return;
    if (result->candidates) {
        for (uint32_t i = 0; i < result->candidate_count; i++) {
            if (result->candidates[i].trace_text) {
                free(result->candidates[i].trace_text);
            }
            delete[] result->candidates[i].token_ids;
        }
        delete[] result->candidates;
    }
    delete result;
}

void antigravity_free_verification_result(antigravity_verification_result_t* result) {
    if (!result) return;
    if (result->verifier_reasoning) {
        free(result->verifier_reasoning);
    }
    delete result;
}

void antigravity_sanitize_buffers(antigravity_engine_t engine) {
    if (engine && engine->ctx) {
        AntigravityEngineSanitizeBuffers(engine->ctx);
    }
}

} // extern "C"

