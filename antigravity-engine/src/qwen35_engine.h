#pragma once
//
// Qwen3.5 text decoder on Metal.
//
// Qwen3.5 is not a Llama-shaped model, so MetalTransformerEngine cannot run it. Of the 24 layers
// in the 0.8B, 18 are Gated-DeltaNet linear-attention layers (a causal depthwise conv, then a
// recurrent state update per head) and 6 are full attention with an output gate, per-head
// q/k RMSNorm, partial rotary embeddings (64 of 256 dims) and head_dim 256. Norm weights use
// the (1 + w) convention, embeddings are tied, and the vocabulary is 248,320.
//
// This engine runs that architecture end to end: bf16 weights read as-is, float32 activations
// and recurrent state, float16 KV cache. It implements ITransformerEngine for the text path
// only; speculative, multimodal and tree-search entry points report themselves unsupported.
//
// Selected by AntigravityEngineLoadModel when the config.json next to the weights names a
// qwen3_5 model. The reference it is checked against is transformers' Qwen3_5 implementation,
// run in float32 (tools/compare_forward.py).
//
#include "transformer_engine_interface.h"
#include <memory>
#include <string>

class Qwen35Engine : public ITransformerEngine {
public:
    // True when config.json beside `safetensors_path` declares a qwen3_5 model.
    static bool matches(const std::string& safetensors_path);

    Qwen35Engine(int n_channels, int max_seq_len);
    ~Qwen35Engine() override;

    bool loadWeights(const std::string& safetensors_path) override;
    void allocateUnifiedMemoryMap() override {}

    GenerationResult generate(const int32_t* prompt_tokens, int32_t prompt_len,
                              int32_t max_new_tokens, float temperature, float top_p) override;

    GenerationResult generateSpeculative(ITransformerEngine*, const int32_t*, int32_t, int32_t,
                                         int32_t, float, float) override;
    GenerationResult generateMultimodal(const int32_t*, int32_t, const float*, int32_t, int32_t,
                                        float, float) override;
    MCTSResult generateMCTS(const int32_t*, int32_t, const MCTSConfig&) override;

    bool debugForward(const int32_t* prompt_tokens, int32_t prompt_len,
                      std::vector<float>& hidden, std::vector<float>& logits) override;
    DebugShape debugShape() const override;

    void sanitizeBuffers() override {}
    uint64_t getAllocatedBytes() const override;
    int32_t maxSequenceLength() const override;
    int32_t maxDraftChunkTokens() const override { return 1; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
