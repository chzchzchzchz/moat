#ifndef ANTIGRAVITY_ENGINE_C_H
#define ANTIGRAVITY_ENGINE_C_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version Identification
#define ANTIGRAVITY_API_VERSION_MAJOR 1
#define ANTIGRAVITY_API_VERSION_MINOR 0
#define ANTIGRAVITY_API_VERSION_PATCH 0

// Opaque Handle Types
typedef struct AntigravityEngineInternal OpaqueEngine;
typedef OpaqueEngine* antigravity_engine_t;

// Configuration Struct
typedef struct {
    int64_t  memory_limit_bytes;   // Physical RAM limit (default: 4500MB)
    uint32_t parallel_channels;    // Rollout channels N (4, 8, 16)
    float    reflection_threshold; // Tau threshold (default: 0.75)
    bool     enable_lut;           // Enable precomputed softmax LUT
} antigravity_config_t;

// Rollout Candidate Output Struct
typedef struct {
    char*    trace_text;           // Null-terminated trace string
    float    logprob;              // Cumulative log probability
    uint32_t token_count;          // Tokens generated
} antigravity_candidate_t;

// Generation Result Container Struct
typedef struct {
    antigravity_candidate_t* candidates;
    uint32_t                 candidate_count;
    uint32_t                 best_candidate_index;
    double                   total_latency_ms;
    float                    token_savings_pct;
    bool                     reflection_triggered;
} antigravity_rollout_result_t;

// Verification Result Struct
typedef struct {
    uint32_t selected_index;
    float    confidence_score;
    char*    verifier_reasoning;
} antigravity_verification_result_t;

// High-Level Core Engine API Methods
antigravity_engine_t antigravity_engine_create(const antigravity_config_t* config, const char* model_path);
void antigravity_engine_destroy(antigravity_engine_t engine);

antigravity_rollout_result_t* antigravity_generate_rollouts(
    antigravity_engine_t engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature
);

antigravity_verification_result_t* antigravity_verify_candidates(
    antigravity_engine_t engine,
    const antigravity_rollout_result_t* rollouts
);

void antigravity_free_rollout_result(antigravity_rollout_result_t* result);
void antigravity_free_verification_result(antigravity_verification_result_t* result);
void antigravity_sanitize_buffers(antigravity_engine_t engine);

#ifdef __cplusplus
}
#endif

#endif // ANTIGRAVITY_ENGINE_C_H
