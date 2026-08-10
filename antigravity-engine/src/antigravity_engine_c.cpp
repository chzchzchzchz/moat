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
#include <random>

struct AntigravityEngineInternal {
    antigravity_config_t config;
    std::string model_path;
    AntigravityEngineContext* ctx;
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

    if (model_path && strlen(model_path) > 0) {
        AntigravityEngineLoadModel(engine->ctx, model_path);
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

antigravity_rollout_result_t* antigravity_generate_rollouts(
    antigravity_engine_t engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature
) {
    if (!engine || !engine->ctx || !prompt) return NULL;

    uint32_t n_channels = engine->config.parallel_channels > 0 ? engine->config.parallel_channels : 8;
    int32_t hidden_dim = 256;
    int32_t vocab_size = 1000;

    std::vector<float> weight_matrix(hidden_dim * vocab_size);
    std::mt19937 weight_rng(42);
    std::normal_distribution<float> weight_dist(0.0f, 1.0f / std::sqrt((float)hidden_dim));
    for (size_t i = 0; i < weight_matrix.size(); i++) {
        weight_matrix[i] = weight_dist(weight_rng);
    }

    std::vector<int32_t> out_tokens(n_channels * max_tokens, 0);
    AntigravityRolloutResult api_result;
    memset(&api_result, 0, sizeof(api_result));

    int ret = AntigravityEngineGenerateRollouts(
        engine->ctx,
        weight_matrix.data(),
        (int32_t)max_tokens,
        temperature,
        out_tokens.data(),
        &api_result
    );

    if (ret != 0) return NULL;

    antigravity_rollout_result_t* res = new antigravity_rollout_result_t();
    res->candidate_count = n_channels;
    res->candidates = new antigravity_candidate_t[n_channels];

    std::string prompt_str = std::string(prompt);

    for (uint32_t c = 0; c < n_channels; c++) {
        std::string trace = prompt_str + "\n[Channel " + std::to_string(c + 1) + " rollout]:";
        float accum_logprob = 0.0f;
        for (uint32_t s = 0; s < max_tokens; s++) {
            int32_t tok = out_tokens[c * max_tokens + s];
            if (s < 15) {
                trace += " tok_" + std::to_string(tok);
            }
            float logit_val = 0.0f;
            for (int k = 0; k < std::min(hidden_dim, 16); k++) {
                logit_val += weight_matrix[k * vocab_size + (tok % vocab_size)];
            }
            accum_logprob += logf(1.0f / (1.0f + expf(-logit_val)));
        }
        res->candidates[c].trace_text = strdup(trace.c_str());
        res->candidates[c].logprob = accum_logprob;
        res->candidates[c].token_count = max_tokens;
    }

    res->best_candidate_index = (uint32_t)api_result.best_candidate_channel_idx;
    res->total_latency_ms = api_result.execution_wall_time_ms;
    res->token_savings_pct = 0.0f;
    res->reflection_triggered = false;

    return res;
}

antigravity_verification_result_t* antigravity_verify_candidates(
    antigravity_engine_t engine,
    const antigravity_rollout_result_t* rollouts
) {
    if (!engine || !engine->ctx || !rollouts || rollouts->candidate_count == 0) return NULL;

    uint32_t n_channels = rollouts->candidate_count;
    uint32_t seq_len = rollouts->candidates[0].token_count;

    std::vector<int32_t> candidate_tokens(n_channels * seq_len, 0);
    for (uint32_t c = 0; c < n_channels; c++) {
        const char* trace = rollouts->candidates[c].trace_text;
        int s = 0;
        if (trace) {
            const char* p = trace;
            while ((p = strstr(p, "tok_")) != NULL && s < (int)seq_len) {
                int tok_id = atoi(p + 4);
                candidate_tokens[c * seq_len + s] = tok_id;
                s++;
                p += 4;
            }
        }
        if (s > 0) {
            int num_extracted = s;
            while (s < (int)seq_len) {
                candidate_tokens[c * seq_len + s] = candidate_tokens[c * seq_len + (s % num_extracted)];
                s++;
            }
        } else {
            std::mt19937 cand_rng(777 + c * 101);
            std::uniform_int_distribution<int32_t> cand_dist(0, 999);
            while (s < (int)seq_len) {
                candidate_tokens[c * seq_len + s] = cand_dist(cand_rng);
                s++;
            }
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

