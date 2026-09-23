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
    char*    trace_text;           // Decoded text, or NULL when no tokenizer is bound to this target
    int32_t* token_ids;            // Owned array of `token_count` real token IDs from the transformer
    float    logprob;              // Cumulative log probability reported by the engine
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

/**
 * Token-input rollout entry point. This is the real inference path: it delegates to
 * AntigravityEngineNativeGenerate(), which runs the full multi-layer Metal transformer
 * forward pass over the loaded weights.
 *
 * Candidates carry real `token_ids`; `trace_text` is NULL because this target links no
 * tokenizer. Decode `token_ids` with whatever tokenizer the host app owns.
 *
 * @return Result container, or NULL if weights are not loaded or generation failed.
 */
antigravity_rollout_result_t* antigravity_generate_rollouts_tokens(
    antigravity_engine_t engine,
    const int32_t* prompt_tokens,
    uint32_t prompt_len,
    uint32_t max_tokens,
    float temperature,
    float top_p
);

/**
 * String-prompt convenience wrapper.
 *
 * This demo target links no tokenizer, so it cannot convert `prompt` into token IDs and
 * therefore cannot run real inference. It ALWAYS returns NULL rather than fabricating a
 * trace. Tokenize in the host app and call antigravity_generate_rollouts_tokens().
 *
 * @return Always NULL. Diagnostic is written to stderr.
 */
antigravity_rollout_result_t* antigravity_generate_rollouts(
    antigravity_engine_t engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature
);

/**
 * List-wise verification over candidate rollouts.
 *
 * Scores the real `token_ids` carried by each candidate. Returns NULL if any candidate is
 * missing token IDs; it never substitutes generated or random token data.
 */
antigravity_verification_result_t* antigravity_verify_candidates(
    antigravity_engine_t engine,
    const antigravity_rollout_result_t* rollouts
);

void antigravity_free_rollout_result(antigravity_rollout_result_t* result);
void antigravity_free_verification_result(antigravity_verification_result_t* result);
/**
 * Zero all internal Metal buffers (activations, weights, output, KV cache).
 *
 * This is a plain memory-wipe of shared MTLBuffers. It is not a Secure Enclave operation
 * and makes no hardware-backed guarantee.
 */
void antigravity_sanitize_buffers(antigravity_engine_t engine);

#ifdef __cplusplus
}
#endif

#endif // ANTIGRAVITY_ENGINE_C_H
