#include <metal_stdlib>
using namespace metal;

// 1. rmsnorm_kernel
// Applies RMSNorm with threadgroup reduction for the mean of squares.
kernel void rmsnorm_kernel(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint batch_idx = threadgroup_position_in_grid;
    uint tid = thread_position_in_threadgroup;
    
    device const half* x_b = x + batch_idx * dim;
    device half* out_b = out + batch_idx * dim;
    
    threadgroup float sum_sq_shared[1024]; 
    
    float local_sum = 0.0;
    for (uint i = tid; i < dim; i += threads_per_threadgroup) {
        float val = (float)x_b[i];
        local_sum += val * val;
    }
    sum_sq_shared[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction
    for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_sq_shared[tid] += sum_sq_shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean_sq = sum_sq_shared[0] / (float)dim;
    float rsqrt_val = rsqrt(mean_sq + eps);
    
    for (uint i = tid; i < dim; i += threads_per_threadgroup) {
        out_b[i] = (half)(((float)x_b[i] * rsqrt_val) * (float)weight[i]);
    }
}

// 2. rope_kernel
// Applies LLaMA Rotary Position Embeddings (RoPE) to Q and K.
kernel void rope_kernel(
    device half* q [[buffer(0)]],
    device half* k [[buffer(1)]],
    device const half* freqs_cos [[buffer(2)]],
    device const half* freqs_sin [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& n_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& start_pos [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_seq_idx = gid.x;
    uint head_idx = gid.y;
    uint i = gid.z;
    
    uint half_dim = head_dim / 2;
    if (i >= half_dim) return;
    
    uint seq_pos = batch_seq_idx % seq_len;
    uint absolute_pos = start_pos + seq_pos;
    
    float f_cos = (float)freqs_cos[absolute_pos * half_dim + i];
    float f_sin = (float)freqs_sin[absolute_pos * half_dim + i];
    
    if (head_idx < n_heads) {
        uint base = (batch_seq_idx * n_heads + head_idx) * head_dim;
        float q0 = (float)q[base + i];
        float q1 = (float)q[base + i + half_dim];
        q[base + i]            = (half)(q0 * f_cos - q1 * f_sin);
        q[base + i + half_dim] = (half)(q1 * f_cos + q0 * f_sin);
    }
    
    if (head_idx < n_kv_heads) {
        uint base = (batch_seq_idx * n_kv_heads + head_idx) * head_dim;
        float k0 = (float)k[base + i];
        float k1 = (float)k[base + i + half_dim];
        k[base + i]            = (half)(k0 * f_cos - k1 * f_sin);
        k[base + i + half_dim] = (half)(k1 * f_cos + k0 * f_sin);
    }
}

// 3. gqa_attention_scores_kernel
// Computes unnormalized attention scores with GQA support and max_seq cache stride.
// Supports q_len > 1 for parallel prefill.
kernel void gqa_attention_scores_kernel(
    device const half* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device half* scores [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& max_seq [[buffer(7)]],
    constant uint& q_len [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_head_idx = gid.x;
    uint q_idx = gid.y;
    uint seq_idx = gid.z;
    
    uint batch_idx = batch_head_idx / n_heads;
    uint head_idx = batch_head_idx % n_heads;
    
    if (q_idx >= q_len || seq_idx >= seq_len) return;
    
    uint kv_head_idx = head_idx / (n_heads / n_kv_heads);
    
    // Q is [batch, q_len, n_heads, head_dim] -> we'll layout as [batch * q_len, n_heads, head_dim] in C++
    // Actually, C++ forwardLayer currently lays out Q as: Q [M, H] which is [batch, n_heads * head_dim].
    // If q_len > 1, Q will be [batch * q_len, n_heads * head_dim].
    uint q_row = batch_idx * q_len + q_idx;
    device const half* q_h = q + (q_row * n_heads + head_idx) * head_dim;
    device const half* k_h = k_cache + ((batch_idx * n_kv_heads + kv_head_idx) * max_seq + seq_idx) * head_dim;
    
    float score = 0.0;
    for (uint i = 0; i < head_dim; i++) {
        score += (float)q_h[i] * (float)k_h[i];
    }
    
    score /= sqrt((float)head_dim);
    
    // Apply causal mask during prefill
    // Absolute position of Q is (start_pos + q_idx). Absolute position of K is (seq_idx).
    // If seq_idx > start_pos + q_idx, mask it out.
    // We don't have start_pos here easily, but seq_len = start_pos + q_len.
    // So Q's pos is `seq_len - q_len + q_idx`. K's pos is `seq_idx`.
    uint q_pos = seq_len - q_len + q_idx;
    if (seq_idx > q_pos) {
        score = -1e9;
    }
    
    // scores shape: [batch * n_heads, q_len, seq_len]
    scores[(batch_head_idx * q_len + q_idx) * seq_len + seq_idx] = (half)score;
}

// 4. softmax_kernel
// Computes softmax over attention scores using threadgroup reduction for stability.
kernel void softmax_kernel(
    device half* scores [[buffer(0)]],
    device half* probs [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint batch_head_idx = threadgroup_position_in_grid;
    uint tid = thread_position_in_threadgroup;
    
    device half* scores_bh = scores + batch_head_idx * seq_len;
    device half* probs_bh = probs + batch_head_idx * seq_len;
    
    threadgroup float max_shared[1024];
    threadgroup float sum_shared[1024];
    
    float local_max = -1e9;
    for (uint i = tid; i < seq_len; i += threads_per_threadgroup) {
        local_max = max(local_max, (float)scores_bh[i]);
    }
    max_shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_shared[tid] = max(max_shared[tid], max_shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float max_val = max_shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float local_sum = 0.0;
    for (uint i = tid; i < seq_len; i += threads_per_threadgroup) {
        float val = exp((float)scores_bh[i] - max_val);
        local_sum += val;
    }
    sum_shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float sum_val = sum_shared[0];
    
    for (uint i = tid; i < seq_len; i += threads_per_threadgroup) {
        float val = exp((float)scores_bh[i] - max_val);
        probs_bh[i] = (half)(val / sum_val);
    }
}

// 5. attention_value_kernel
// Computes context values by weighting V cache with attention probabilities and max_seq stride.
kernel void attention_value_kernel(
    device const half* probs [[buffer(0)]],
    device const half* v_cache [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& max_seq [[buffer(7)]],
    constant uint& q_len [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_head_idx = gid.x;
    uint q_idx = gid.y;
    uint dim_idx = gid.z;
    
    uint batch_idx = batch_head_idx / n_heads;
    uint head_idx = batch_head_idx % n_heads;
    
    if (q_idx >= q_len || dim_idx >= head_dim) return;
    
    uint kv_head_idx = head_idx / (n_heads / n_kv_heads);
    
    // probs shape: [batch * n_heads, q_len, seq_len]
    device const half* probs_h = probs + (batch_head_idx * q_len + q_idx) * seq_len;
    
    float val = 0.0;
    for (uint seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        float p = (float)probs_h[seq_idx];
        float v = (float)v_cache[((batch_idx * n_kv_heads + kv_head_idx) * max_seq + seq_idx) * head_dim + dim_idx];
        val += p * v;
    }
    
    // Output shape is Q shape: [batch * q_len, n_heads, head_dim]
    uint q_row = batch_idx * q_len + q_idx;
    out[(q_row * n_heads + head_idx) * head_dim + dim_idx] = (half)val;
}

// 6. silu_elementwise_mul_kernel
// Computes SwiGLU: silu(gate) * up
kernel void silu_elementwise_mul_kernel(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float g = (float)gate[gid];
    float u = (float)up[gid];
    
    float silu_g = g / (1.0 + exp(-g));
    
    out[gid] = (half)(silu_g * u);
}

// 7. residual_add_kernel
// Elementwise addition for residual connections.
kernel void residual_add_kernel(
    device const half* x [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    out[gid] = (half)((float)x[gid] + (float)residual[gid]);
}

// 8. embedding_lookup_kernel
// Embeddings lookup kernel.
kernel void embedding_lookup_kernel(
    device const uint* token_ids [[buffer(0)]],
    device const half* embed_table [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint dim_idx = gid.y;
    
    if (dim_idx >= hidden_dim) return;
    
    uint token_id = token_ids[batch_idx];
    out[batch_idx * hidden_dim + dim_idx] = embed_table[token_id * hidden_dim + dim_idx];
}

// 10. gemv_kernel
// Vector-Matrix multiplication for single-token decode mode (M=1).
// Computes y[N] = x[K] * B[K x N] safely without 8x8 matrix tile out-of-bounds overflow.
kernel void gemv_kernel(
    device const half* x [[buffer(0)]],        // [K]
    device const half* B [[buffer(1)]],        // [K x N]
    device half* y [[buffer(2)]],              // [N]
    constant uint& K_dim [[buffer(3)]],
    constant uint& N_dim [[buffer(4)]],
    uint col [[thread_position_in_grid]]
) {
    if (col >= N_dim) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K_dim; k++) {
        sum += (float)x[k] * (float)B[k * N_dim + col];
    }
    
    y[col] = (half)sum;
}

// 9. kv_cache_append_kernel
// Appends K or V token slice into the KV Cache tensor
kernel void kv_cache_append_kernel(
    device const half* slice [[buffer(0)]],      // [q_len, n_kv_heads, head_dim]
    device half* cache [[buffer(1)]],            // [n_kv_heads, max_seq, head_dim]
    constant uint& n_kv_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& max_seq [[buffer(4)]],
    constant uint& seq_pos [[buffer(5)]],
    constant uint& q_len [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.x;
    uint head_idx = gid.y;
    uint dim_idx = gid.z;
    
    if (q_idx >= q_len || head_idx >= n_kv_heads || dim_idx >= head_dim) return;
    
    uint slice_idx = (q_idx * n_kv_heads + head_idx) * head_dim + dim_idx;
    uint cache_idx = (head_idx * max_seq + seq_pos + q_idx) * head_dim + dim_idx;
    
    cache[cache_idx] = slice[slice_idx];
}
