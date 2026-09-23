#pragma once
//
// GENERATED FILE — do not edit by hand.
// Regenerate with: python3 scripts/embed_shaders.py
//
// Metal shader sources, embedded so the engine can always build a shader
// library even when no .metal or .metallib file ships beside the executable —
// which is the case inside AntigravityEngine.xcframework, where only the static
// library is packaged. See scripts/embed_shaders.py for the full reason.
//
#include <cstddef>

namespace antigravity {
namespace shaders {

struct EmbeddedShader {
    const char* name;      // file stem, e.g. "batched_gemm"
    const char* source;    // the complete .metal source
};

// ---- batched_gemm.metal (10230 bytes) ----
inline const char* const kBatchedGemmSource = R"AGMETAL(#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Project Antigravity — Metal Compute Shader for INT4 Super-Block GEMM
//
// Hardware Target: Apple Silicon GPU (A17 Pro / A18 Pro / M1-M4)
// Uses simdgroup_matrix for hardware matrix tile multiplication.
// On-the-fly INT4 super-block LUT dequantization.
// =============================================================================

#define GROUP_SIZE 32
#define GROUPS_PER_SUPERBLOCK 8
#define ELEMENTS_PER_SUPERBLOCK 256  // 32 * 8

// Super-block memory layout (144 bytes total, 16-byte header + 128-byte payload)
struct SuperBlock {
    half scales[8];            // 8 per-group FP16 scale factors (16 bytes)
    uchar packed_nibbles[128]; // 256 INT4 weights packed into 128 uint8 pairs (128 bytes)
};

// -----------------------------------------------------------------------------
// Dequantization Helper: Unpack INT4 nibble pair and dequantize via LUT
// -----------------------------------------------------------------------------
inline half2 dequantize_nibble_pair(uchar packed_byte, half scale_even, half scale_odd) {
    // Low nibble (even element): bits 0..3
    int raw_even = int(packed_byte & 0x0F) - 8;
    // High nibble (odd element): bits 4..7
    int raw_odd = int((packed_byte >> 4) & 0x0F) - 8;

    half val_even = static_cast<half>(raw_even) * scale_even;
    half val_odd  = static_cast<half>(raw_odd)  * scale_odd;

    return half2(val_even, val_odd);
}

// =============================================================================
// KERNEL 1: Fast INT4 Dequantization Kernel (Super-Block → Dense FP16 Matrix)
// Decouples dequantization so it can feed standard Metal MPS or simdgroup GEMM
// =============================================================================
kernel void dequantize_superblocks_kernel(
    device const SuperBlock* superblocks [[buffer(0)]],
    device half*             out_weights [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // Each thread dequantizes one 256-element super-block
    device const SuperBlock& sb = superblocks[id];
    device half* out_ptr = out_weights + id * ELEMENTS_PER_SUPERBLOCK;

    for (int byte_idx = 0; byte_idx < 128; byte_idx++) {
        uchar packed_byte = sb.packed_nibbles[byte_idx];
        int elem_even = byte_idx * 2;
        int elem_odd  = elem_even + 1;

        int group_even = elem_even / GROUP_SIZE; // 0..7
        int group_odd  = elem_odd / GROUP_SIZE;  // 0..7

        half scale_even = sb.scales[group_even];
        half scale_odd  = sb.scales[group_odd];

        half2 dequantized = dequantize_nibble_pair(packed_byte, scale_even, scale_odd);

        out_ptr[elem_even] = dequantized.x;
        out_ptr[elem_odd]  = dequantized.y;
    }
}

// =============================================================================
// KERNEL 2: Fused Batched GEMM Kernel using Metal SIMD Matrix Tiles
//
// Computes: C [N x M] = A [N x K] * B_dequantized [K x M]
// Uses simdgroup_matrix<half, 8, 8> for hardware acceleration.
// =============================================================================
kernel void batched_gemm_simdgroup(
    device const half*       activations   [[buffer(0)]], // [N x K]
    device const half*       weights       [[buffer(1)]], // [K x M]
    device half*             output        [[buffer(2)]], // [N x M]
    constant uint&           N_batch       [[buffer(3)]],
    constant uint&           K_dim         [[buffer(4)]],
    constant uint&           M_dim         [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_idx [[thread_index_in_simdgroup]]
) {
    uint row_start = group_id.y * 8;
    uint col_start = group_id.x * 8;

    if (row_start >= N_batch || col_start >= M_dim) return;

    // SIMD matrix accumulator (8x8 half precision)
    simdgroup_matrix<half, 8, 8> acc_matrix;
    acc_matrix = simdgroup_matrix<half, 8, 8>(0.0h);

    // Accumulate over K dimension in chunks of 8
    for (uint k = 0; k < K_dim; k += 8) {
        simdgroup_matrix<half, 8, 8> a_tile;
        simdgroup_matrix<half, 8, 8> b_tile;

        // Load Activation Tile (A) [8 x 8] from device memory
        simdgroup_load(a_tile, activations + row_start * K_dim + k, K_dim);

        // Load Weight Tile (B) [8 x 8] from device memory
        simdgroup_load(b_tile, weights + k * M_dim + col_start, M_dim);

        // Hardware Multiply-Accumulate on SIMD Matrix Tile
        simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix);
    }

    // Store result tile back to global memory C [N x M] with bounds checking
    if (row_start + 8 <= N_batch && col_start + 8 <= M_dim) {
        simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim);
    } else {
        threadgroup half edge_tile[64];
        simdgroup_store(acc_matrix, edge_tile, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_idx == 0) {
            for (uint r = 0; r < 8; r++) {
                if (row_start + r < N_batch) {
                    for (uint c = 0; c < 8; c++) {
                        if (col_start + c < M_dim) {
                            output[(row_start + r) * M_dim + (col_start + c)] = edge_tile[r * 8 + c];
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// KERNEL 3: Fused SuperBlock INT4 SIMD-group GEMM Compute Shader Kernel
// Performs on-the-fly INT4 dequantization directly inside threadgroup registers,
// eliminating intermediate FP16 weight buffer allocations.
// =============================================================================
kernel void fused_batched_gemm_int4(
    device const half*       activations   [[buffer(0)]], // [N x K]
    device const SuperBlock* superblocks   [[buffer(1)]], // [ (K*M)/256 SuperBlocks ]
    device half*             output        [[buffer(2)]], // [N x M]
    constant uint&           N_batch       [[buffer(3)]],
    constant uint&           K_dim         [[buffer(4)]],
    constant uint&           M_dim         [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    (void)simd_lane_id;
    uint row_start = group_id.y * 8;
    uint col_start = group_id.x * 8;

    if (row_start >= N_batch || col_start >= M_dim) return;

    simdgroup_matrix<half, 8, 8> acc_matrix = simdgroup_matrix<half, 8, 8>(0.0h);

    for (uint k = 0; k < K_dim; k += 8) {
        simdgroup_matrix<half, 8, 8> a_tile;
        simdgroup_matrix<half, 8, 8> b_tile;

        simdgroup_load(a_tile, activations + row_start * K_dim + k, K_dim);

        threadgroup half b_elements[8][8];
        for (uint r = 0; r < 8; r++) {
            uint global_k = k + r;
            for (uint c = 0; c < 8; c++) {
                uint global_m = col_start + c;
                if (global_k < K_dim && global_m < M_dim) {
                    uint flat_weight_idx = global_k * M_dim + global_m;
                    uint sb_idx = flat_weight_idx / 256;
                    uint in_sb_elem = flat_weight_idx % 256;

                    device const SuperBlock& sb = superblocks[sb_idx];
                    uint byte_idx = in_sb_elem / 2;
                    uchar packed = sb.packed_nibbles[byte_idx];
                    int raw_nibble = (in_sb_elem % 2 == 0) ? (int(packed & 0x0F) - 8) : (int((packed >> 4) & 0x0F) - 8);
                    half scale = sb.scales[in_sb_elem / 32];
                    b_elements[r][c] = static_cast<half>(raw_nibble) * scale;
                } else {
                    b_elements[r][c] = 0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(b_tile, (const threadgroup half*)&b_elements[0][0], 8);

        simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix);
    }

    simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim);
}

// =============================================================================
// KERNEL 4: INT4 Super-Block GEMV (single-row decode)
//
// This is the kernel that makes quantization a throughput win rather than only a
// memory win. Autoregressive decode is memory-bandwidth bound: every token streams
// the entire weight set through the GPU. Reading 4-bit super-blocks instead of FP16
// moves ~1/5.5 the bytes per token, which raises the bandwidth ceiling by the same
// factor. The dequantized value is materialised only in registers, so the weights
// stay 4-bit in VRAM for the whole decode.
//
// The batched simdgroup kernels above tile 8x8; with a single row (M = 1) they
// would leave 7 of 8 rows idle, so decode needs its own GEMV.
//
// Layout matches repack_to_superblocks() in src/dequant.py exactly: 8 FP16 scales
// then 128 packed bytes, 256 INT4 values per block, low nibble = even element,
// values stored biased by +8, one scale per 32 elements.
// =============================================================================
kernel void gemv_int4_kernel(
    device const half*       x           [[buffer(0)]], // [K]
    device const SuperBlock* superblocks [[buffer(1)]], // [(K*N)/256] over B[K x N]
    device half*             y           [[buffer(2)]], // [N]
    constant uint&           K_dim       [[buffer(3)]],
    constant uint&           N_dim       [[buffer(4)]],
    uint col [[thread_position_in_grid]]
) {
    if (col >= N_dim) return;

    float sum = 0.0f;
    for (uint k = 0; k < K_dim; k++) {
        uint flat   = k * N_dim + col;
        uint sb_idx = flat >> 8;          // / ELEMENTS_PER_SUPERBLOCK
        uint in_sb  = flat & 255;         // % ELEMENTS_PER_SUPERBLOCK

        device const SuperBlock& sb = superblocks[sb_idx];
        uchar packed = sb.packed_nibbles[in_sb >> 1];
        int   nib    = (in_sb & 1u) ? (int((packed >> 4) & 0x0F) - 8)
                                    : (int( packed       & 0x0F) - 8);
        half  scale  = sb.scales[in_sb >> 5];   // / GROUP_SIZE

        sum += float(x[k]) * (float(nib) * float(scale));
    }

    y[col] = half(sum);
}
)AGMETAL";

// ---- batched_gemm_fused.metal (2763 bytes) ----
inline const char* const kBatchedGemmFusedSource = R"AGMETAL(#include <metal_stdlib>
using namespace metal;

struct SuperBlock {
    half scales[8];
    uchar packed_nibbles[128];
};

kernel void fused_batched_gemm_int4(
    device const half*       activations   [[buffer(0)]], // [N x K]
    device const SuperBlock* superblocks   [[buffer(1)]], // [ (K*M)/256 SuperBlocks ]
    device half*             output        [[buffer(2)]], // [N x M]
    constant uint&           N_batch       [[buffer(3)]],
    constant uint&           K_dim         [[buffer(4)]],
    constant uint&           M_dim         [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    (void)simd_lane_id;
    uint row_start = group_id.y * 8;
    uint col_start = group_id.x * 8;

    if (row_start >= N_batch || col_start >= M_dim) return;

    simdgroup_matrix<half, 8, 8> acc_matrix = simdgroup_matrix<half, 8, 8>(0.0h);

    for (uint k = 0; k < K_dim; k += 8) {
        simdgroup_matrix<half, 8, 8> a_tile;
        simdgroup_matrix<half, 8, 8> b_tile;

        simdgroup_load(a_tile, activations + row_start * K_dim + k, K_dim);

        // Must be threadgroup, not thread-local: simdgroup_load reads the tile
        // cooperatively across the simdgroup, so a per-thread copy is both the wrong
        // address space (this file has never compiled) and the wrong data.
        threadgroup half b_elements[8][8];
        for (uint r = 0; r < 8; r++) {
            uint global_k = k + r;
            for (uint c = 0; c < 8; c++) {
                uint global_m = col_start + c;
                if (global_k < K_dim && global_m < M_dim) {
                    uint flat_weight_idx = global_k * M_dim + global_m;
                    uint sb_idx = flat_weight_idx / 256;
                    uint in_sb_elem = flat_weight_idx % 256;

                    device const SuperBlock& sb = superblocks[sb_idx];
                    uint byte_idx = in_sb_elem / 2;
                    uchar packed = sb.packed_nibbles[byte_idx];
                    int raw_nibble = (in_sb_elem % 2 == 0) ? (int(packed & 0x0F) - 8) : (int((packed >> 4) & 0x0F) - 8);
                    half scale = sb.scales[in_sb_elem / 32];
                    b_elements[r][c] = static_cast<half>(raw_nibble) * scale;
                } else {
                    b_elements[r][c] = 0.0h;
                }
            }
        }

        // Barrier before the cooperative read: without it the tile is raced.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(b_tile, (const threadgroup half*)&b_elements[0][0], 8);
        simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix);
    }

    simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim);
}
)AGMETAL";

// ---- deltanet_forward.metal (2320 bytes) ----
inline const char* const kDeltanetForwardSource = R"AGMETAL(#include <metal_stdlib>
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
)AGMETAL";

// ---- moe_gemm.metal (2745 bytes) ----
inline const char* const kMoeGemmSource = R"AGMETAL(#include <metal_stdlib>
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
)AGMETAL";

// ---- transformer_ops.metal (12657 bytes) ----
inline const char* const kTransformerOpsSource = R"AGMETAL(#include <metal_stdlib>
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
// Computes softmax over attention scores using SIMD-group reductions for exact numerical stability.
kernel void softmax_kernel(
    device half* scores [[buffer(0)]],
    device half* probs [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint batch_head_idx = threadgroup_position_in_grid;
    uint tid = thread_position_in_threadgroup;
    
    device half* scores_bh = scores + batch_head_idx * seq_len;
    device half* probs_bh = probs + batch_head_idx * seq_len;
    
    threadgroup float max_shared[32];
    threadgroup float sum_shared[32];
    
    float local_max = -1e9f;
    for (uint i = tid; i < seq_len; i += threads_per_threadgroup) {
        local_max = max(local_max, (float)scores_bh[i]);
    }
    
    // 1. Exact SIMD-group hardware reduction
    local_max = simd_max(local_max);
    if (simd_lane_id == 0) {
        max_shared[simd_group_id] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Cross-SIMD-group reduction across SIMD groups
    uint num_simd_groups = (threads_per_threadgroup + 31) / 32;
    if (simd_group_id == 0) {
        float gmax = (simd_lane_id < num_simd_groups) ? max_shared[simd_lane_id] : -1e9f;
        gmax = simd_max(gmax);
        if (simd_lane_id == 0) {
            max_shared[0] = gmax;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = max_shared[0];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < seq_len; i += threads_per_threadgroup) {
        float val = exp((float)scores_bh[i] - max_val);
        local_sum += val;
    }
    
    // 3. Exact SIMD-group hardware reduction for sum
    local_sum = simd_sum(local_sum);
    if (simd_lane_id == 0) {
        sum_shared[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 4. Cross-SIMD-group reduction for sum
    if (simd_group_id == 0) {
        float gsum = (simd_lane_id < num_simd_groups) ? sum_shared[simd_lane_id] : 0.0f;
        gsum = simd_sum(gsum);
        if (simd_lane_id == 0) {
            sum_shared[0] = max(gsum, 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
)AGMETAL";

inline const EmbeddedShader* all(size_t* count) {
    static const EmbeddedShader kShaders[] = {
        { "batched_gemm", kBatchedGemmSource },
        { "batched_gemm_fused", kBatchedGemmFusedSource },
        { "deltanet_forward", kDeltanetForwardSource },
        { "moe_gemm", kMoeGemmSource },
        { "transformer_ops", kTransformerOpsSource },
    };
    if (count) *count = sizeof(kShaders) / sizeof(kShaders[0]);
    return kShaders;
}

// Look up an embedded source by file stem ("batched_gemm"), or nullptr.
inline const char* find(const char* name) {
    if (!name) return nullptr;
    size_t n = 0;
    const EmbeddedShader* list = all(&n);
    for (size_t i = 0; i < n; i++) {
        const char* a = list[i].name;
        const char* b = name;
        while (*a && *a == *b) { a++; b++; }
        if (*a == '\0' && *b == '\0') return list[i].source;
    }
    return nullptr;
}

}  // namespace shaders
}  // namespace antigravity
