#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------------
// Sparse Mixture-of-Experts (MoE) Block
// For Qwen 3.5 4B Hybrid Architecture
// --------------------------------------------------------------------------------

kernel void moe_router(
    const device float* hidden_states [[ buffer(0) ]],
    const device float* router_weights [[ buffer(1) ]],
    device int* expert_indices        [[ buffer(2) ]], // Top-K expert indices
    device float* expert_weights      [[ buffer(3) ]], // Top-K softmax weights
    constant int& batch_size          [[ buffer(4) ]],
    constant int& hidden_dim          [[ buffer(5) ]],
    constant int& num_experts         [[ buffer(6) ]],
    constant int& top_k               [[ buffer(7) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= (uint)batch_size) return;
    
    // Simple naive router max-search for top-k
    int token_offset = gid * hidden_dim;

    // logits is a fixed 64-entry thread array, but num_experts is a runtime value.
    // Without this guard a model with more than 64 experts writes past the end of it.
    if (num_experts > 64) return;

    // Accumulate logits (hidden * router_weights^T)
    thread float logits[64];
    for (int e = 0; e < num_experts; ++e) {
        float val = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            val += hidden_states[token_offset + d] * router_weights[e * hidden_dim + d];
        }
        logits[e] = val;
    }
    
    // Top-K selection & Softmax
    for (int k = 0; k < top_k; ++k) {
        float max_val = -1e9;
        int max_idx = -1;
        for (int e = 0; e < num_experts; ++e) {
            if (logits[e] > max_val) {
                max_val = logits[e];
                max_idx = e;
            }
        }
        expert_indices[gid * top_k + k] = max_idx;
        expert_weights[gid * top_k + k] = max_val; // Pre-softmax
        logits[max_idx] = -1e9; // Mask out
    }
    
    // Softmax normalization over the Top-K.
    // Subtract the maximum first. exp() of a raw router logit overflows to inf for
    // large activations, and the whole Top-K then normalises to NaN. Top-K was
    // selected in descending order, so entry 0 is the maximum. This matches the
    // row-max subtraction already used by softmax_kernel in transformer_ops.metal.
    float max_logit = expert_weights[gid * top_k];
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        expert_weights[gid * top_k + k] = exp(expert_weights[gid * top_k + k] - max_logit);
        sum_exp += expert_weights[gid * top_k + k];
    }
    for (int k = 0; k < top_k; ++k) {
        expert_weights[gid * top_k + k] /= sum_exp;
    }
}
