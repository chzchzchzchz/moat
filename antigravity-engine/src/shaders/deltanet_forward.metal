#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------------
// Gated DeltaNet (Linear Attention / RNN) Forward Pass
// For Qwen 3.5 4B Hybrid Architecture
// Replaces traditional QK^T attention with an O(1) memory recurrent state.
// --------------------------------------------------------------------------------

kernel void deltanet_forward(
    const device float* q_proj [[ buffer(0) ]],
    const device float* k_proj [[ buffer(1) ]],
    const device float* v_proj [[ buffer(2) ]],
    const device float* beta   [[ buffer(3) ]], // Gating factor
    device float* rnn_state    [[ buffer(4) ]], // Fixed size KV state: [batch, n_heads, head_dim_k, head_dim_v]
    device float* out_proj     [[ buffer(5) ]],
    constant int& batch_size   [[ buffer(6) ]],
    constant int& n_heads      [[ buffer(7) ]],
    constant int& head_dim_k   [[ buffer(8) ]],
    constant int& head_dim_v   [[ buffer(9) ]],
    uint3 gid [[thread_position_in_grid]]
) {
    int b = gid.z; // batch
    int h = gid.y; // head
    int d = gid.x; // head_dim_v
    
    if (b >= batch_size || h >= n_heads || d >= head_dim_v) return;
    
    // Pointers for this batch and head
    int head_offset_k = (b * n_heads + h) * head_dim_k;
    int head_offset_v = (b * n_heads + h) * head_dim_v;
    int state_offset = (b * n_heads + h) * head_dim_k * head_dim_v;
    
    float beta_val = beta[b * n_heads + h]; // Gating scalar
    
    // Output accumulator for this dimension
    float out_val = 0.0f;
    
    // 1. Update the RNN State (Linear Attention KV accumulation)
    // S_t = (1 - beta) * S_{t-1} + beta * (K_t^T * V_t)
    for (int k = 0; k < head_dim_k; ++k) {
        float k_val = k_proj[head_offset_k + k];
        float v_val = v_proj[head_offset_v + d];
        
        int s_idx = state_offset + (k * head_dim_v + d);
        float s_prev = rnn_state[s_idx];
        
        // Gated DeltaNet state update
        float s_new = (1.0f - beta_val) * s_prev + (beta_val * k_val * v_val);
        rnn_state[s_idx] = s_new;
        
        // 2. Compute Output: O_t = Q_t * S_t
        float q_val = q_proj[head_offset_k + k];
        out_val += q_val * s_new;
    }
    
    // Write out output projection
    out_proj[head_offset_v + d] = out_val;
}
