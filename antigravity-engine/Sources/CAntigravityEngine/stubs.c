/*
 * Project Antigravity — Stub C API implementations for SPM swift test linking.
 * In production, these symbols are provided by libantigravity_engine.dylib
 * or libAntigravityEngine.a (built by scripts/build_xcframework.sh).
 *
 * These stubs allow `swift test` to link and run Swift-only unit tests
 * (config, weight manager, vision encoder, error types) without requiring
 * the full Metal C++ engine to be compiled through SPM.
 */

#include "antigravity_c_api.h"
#include "antigravity_engine_c.h"
#include <stdlib.h>
#include <string.h>

/* ============================== */
/* antigravity_c_api.h stubs      */
/* ============================== */

AntigravityEngineContext* AntigravityEngineCreate(const AntigravityConfig* config) {
    (void)config;
    /* Return a non-NULL sentinel — tests won't actually invoke Metal GPU */
    return (AntigravityEngineContext*)(void*)0x1;
}

void AntigravityEngineDestroy(AntigravityEngineContext* ctx) {
    (void)ctx;
}

int32_t AntigravityEngineLoadModel(AntigravityEngineContext* ctx, const char* model_path) {
    (void)ctx; (void)model_path;
    return 0;
}

void AntigravityEngineUnloadWeights(AntigravityEngineContext* ctx) {
    (void)ctx;
}

bool AntigravityEngineHasWeights(const AntigravityEngineContext* ctx) {
    (void)ctx;
    return false;
}

uint64_t AntigravityEngineGetAllocatedMemoryBytes(const AntigravityEngineContext* ctx) {
    (void)ctx;
    return 0;
}

int32_t AntigravityEngineGenerateRollouts(
    AntigravityEngineContext* ctx,
    const float* weight_matrix,
    int32_t max_steps,
    float temperature,
    int32_t* out_tokens,
    AntigravityRolloutResult* out_result
) {
    (void)ctx; (void)weight_matrix; (void)max_steps;
    (void)temperature; (void)out_tokens; (void)out_result;
    return 0;
}

int32_t AntigravityEngineVerifyCandidates(
    AntigravityEngineContext* ctx,
    const int32_t* candidate_tokens,
    int32_t seq_len,
    float* out_scores
) {
    (void)ctx; (void)candidate_tokens; (void)seq_len; (void)out_scores;
    return 0;
}

void AntigravityEngineSanitizeBuffers(AntigravityEngineContext* ctx) {
    (void)ctx;
}

int32_t AntigravityEngineNativeGenerate(
    AntigravityEngineContext* ctx,
    const int32_t* prompt_tokens,
    int32_t prompt_len,
    int32_t max_new_tokens,
    float temperature,
    float top_p,
    int32_t* out_tokens,
    float* out_logprobs,
    int32_t* out_token_counts,
    double* out_ttft_ms,
    double* out_total_ms
) {
    (void)ctx; (void)prompt_tokens; (void)prompt_len;
    (void)max_new_tokens; (void)temperature; (void)top_p;
    (void)out_tokens; (void)out_logprobs; (void)out_token_counts;
    (void)out_ttft_ms; (void)out_total_ms;
    return 0;
}

int32_t AntigravityEngineNativeGenerateMultimodal(
    AntigravityEngineContext* ctx,
    const int32_t* text_tokens,
    int32_t text_len,
    const float* image_embeddings,
    int32_t n_image_patches,
    int32_t max_new_tokens,
    float temperature,
    float top_p,
    int32_t* out_tokens,
    float* out_logprobs,
    int32_t* out_token_counts,
    double* out_ttft_ms,
    double* out_total_ms
) {
    (void)ctx; (void)text_tokens; (void)text_len;
    (void)image_embeddings; (void)n_image_patches;
    (void)max_new_tokens; (void)temperature; (void)top_p;
    (void)out_tokens; (void)out_logprobs; (void)out_token_counts;
    (void)out_ttft_ms; (void)out_total_ms;
    return 0;
}

/* ============================== */
/* antigravity_engine_c.h stubs   */
/* ============================== */

antigravity_engine_t antigravity_engine_create(const antigravity_config_t* config, const char* model_path) {
    (void)config; (void)model_path;
    return (antigravity_engine_t)(void*)0x1;
}

void antigravity_engine_destroy(antigravity_engine_t engine) {
    (void)engine;
}

antigravity_rollout_result_t* antigravity_generate_rollouts(
    antigravity_engine_t engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature
) {
    (void)engine; (void)prompt; (void)max_tokens; (void)temperature;
    return NULL;
}

antigravity_verification_result_t* antigravity_verify_candidates(
    antigravity_engine_t engine,
    const antigravity_rollout_result_t* rollouts
) {
    (void)engine; (void)rollouts;
    return NULL;
}

void antigravity_free_rollout_result(antigravity_rollout_result_t* result) {
    (void)result;
}

void antigravity_free_verification_result(antigravity_verification_result_t* result) {
    (void)result;
}

void antigravity_sanitize_buffers(antigravity_engine_t engine) {
    (void)engine;
}
