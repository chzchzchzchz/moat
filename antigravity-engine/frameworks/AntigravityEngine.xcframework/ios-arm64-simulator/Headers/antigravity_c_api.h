/*
 * Project Antigravity — Public C / Swift SDK API Header
 * Target: Apple Silicon iOS / macOS (A17 Pro / A18 Pro / M1-M4)
 * 
 * Provides zero-copy Metal GPU batched GEMM decode, Paged KV-Cache,
 * candidate rollout generation, and list-wise verification.
 */

#ifndef ANTIGRAVITY_C_API_H
#define ANTIGRAVITY_C_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque engine instance handle
typedef struct AntigravityEngineContext AntigravityEngineContext;

// Engine Configuration Options
typedef struct {
    int32_t n_channels;         // Number of parallel candidate rollout channels N (default: 8)
    int32_t vocab_size;         // Model vocabulary size (default: 32000)
    int32_t hidden_dim;         // Model hidden dimension K (default: 2048)
    int32_t max_seq_len;        // Maximum KV-cache token sequence length (default: 2048)
    bool use_metal_gpu;         // Enable Apple Silicon Metal GPU acceleration (default: true)
} AntigravityConfig;

// Rollout Results Summary
typedef struct {
    int32_t total_tokens_generated;
    int32_t active_channels;
    double execution_wall_time_ms;
    double throughput_tokens_per_sec;
    int32_t best_candidate_channel_idx;
    float best_candidate_score;
} AntigravityRolloutResult;

/**
 * Create a new Antigravity Engine instance configured for Apple Silicon Metal acceleration.
 *
 * @param config Configuration parameters.
 * @return Pointer to context handle, or NULL on error.
 */
AntigravityEngineContext* AntigravityEngineCreate(const AntigravityConfig* config);

/**
 * Destroy engine instance and release all Metal buffers and VRAM allocations.
 *
 * @param ctx Engine handle pointer.
 */
void AntigravityEngineDestroy(AntigravityEngineContext* ctx);

/**
 * Load model weights from a Safetensors model file into Metal VRAM buffers.
 *
 * @param ctx Engine handle pointer.
 * @param model_path Path to model weights file.
 * @return 0 on success, non-zero on failure.
 */
int32_t AntigravityEngineLoadModel(AntigravityEngineContext* ctx, const char* model_path);

/**
 * Execute batched GEMM parallel candidate decode across N channels.
 *
 * @param ctx Engine handle pointer.
 * @param weight_matrix Output projection weight matrix array of size [hidden_dim x vocab_size] in FP16/FP32.
 * @param max_steps Maximum token decoding steps.
 * @param temperature Sampling temperature.
 * @param out_tokens Buffer to receive generated token IDs of size [n_channels x max_steps].
 * @param out_result Result metrics output pointer.
 * @return 0 on success, non-zero on failure.
 */
int32_t AntigravityEngineGenerateRollouts(
    AntigravityEngineContext* ctx,
    const float* weight_matrix,
    int32_t max_steps,
    float temperature,
    int32_t* out_tokens,
    AntigravityRolloutResult* out_result
);

/**
 * Perform List-Wise Verification across N candidate rollout streams.
 *
 * @param ctx Engine handle pointer.
 * @param candidate_tokens Token IDs matrix [n_channels x seq_len].
 * @param seq_len Sequence length per candidate.
 * @param out_scores Output array of verification confidence scores [n_channels].
 * @return Index of the highest scoring candidate channel.
 */
int32_t AntigravityEngineVerifyCandidates(
    AntigravityEngineContext* ctx,
    const int32_t* candidate_tokens,
    int32_t seq_len,
    float* out_scores
);

/**
 * Retrieve total physical VRAM/RAM memory allocated by the engine in bytes.
 */
uint64_t AntigravityEngineGetAllocatedMemoryBytes(const AntigravityEngineContext* ctx);

/**
 * Zero out all internal Metal buffers for Secure Enclave compliance.
 *
 * @param ctx Engine handle pointer.
 */
void AntigravityEngineSanitizeBuffers(AntigravityEngineContext* ctx);

/**
 * Execute full native 22-layer transformer autoregressive decode across N channels.
 * This uses the MetalTransformerEngine (RoPE, GQA, SwiGLU MLP, KV caching) — not
 * the simple GEMM activation loop. Requires prior AntigravityEngineLoadModel() call.
 *
 * @param ctx           Engine handle pointer.
 * @param prompt_tokens Array of prompt token IDs.
 * @param prompt_len    Length of prompt_tokens array.
 * @param max_new_tokens Maximum new tokens to generate per channel.
 * @param temperature   Sampling temperature.
 * @param top_p         Nucleus sampling probability threshold.
 * @param out_tokens    Output buffer [n_channels * max_new_tokens] for generated tokens.
 * @param out_logprobs  Output buffer [n_channels] for cumulative log-probs per channel.
 * @param out_token_counts Output buffer [n_channels] for actual tokens generated per channel.
 * @param out_ttft_ms   Output: time to first token in milliseconds.
 * @param out_total_ms  Output: total wall time in milliseconds.
 * @return 0 on success, -1 if weights not loaded, -2 on other error.
 */
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
);

/**
 * Execute native multimodal autoregressive decode (text tokens + vision patch embeddings).
 *
 * @param ctx               Engine handle pointer.
 * @param text_tokens       Array of text token IDs.
 * @param text_len          Length of text_tokens array.
 * @param image_embeddings  Array of vision patch embeddings [n_patches * hidden_dim] in FP32.
 * @param n_image_patches   Number of image patches.
 * @param max_new_tokens    Maximum new tokens to generate per channel.
 * @param temperature       Sampling temperature.
 * @param top_p             Nucleus sampling threshold.
 * @param out_tokens        Output buffer [n_channels * max_new_tokens] for generated tokens.
 * @param out_logprobs      Output buffer [n_channels] for cumulative log-probs per channel.
 * @param out_token_counts  Output buffer [n_channels] for actual tokens generated per channel.
 * @param out_ttft_ms       Output: time to first token in milliseconds.
 * @param out_total_ms      Output: total wall time in milliseconds.
 * @return 0 on success, -1 if weights not loaded, -2 on invalid input.
 */
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
);

/**
 * Flush all model weights from Metal GPU buffers and KV caches.
 * Used for VRAM swapping between reasoner and verifier models.
 *
 * @param ctx Engine handle pointer.
 */
void AntigravityEngineUnloadWeights(AntigravityEngineContext* ctx);

/**
 * Query whether model weights are currently loaded in Metal GPU buffers.
 *
 * @param ctx Engine handle pointer.
 * @return true if weights are loaded, false otherwise.
 */
bool AntigravityEngineHasWeights(const AntigravityEngineContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // ANTIGRAVITY_C_API_H
