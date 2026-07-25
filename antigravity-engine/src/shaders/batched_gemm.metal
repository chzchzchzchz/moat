#include <metal_stdlib>
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
    uint2 group_id [[threadgroup_position_in_grid]]
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

    // Store result tile back to global memory C [N x M]
    simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim);
}
