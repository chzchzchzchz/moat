#include <metal_stdlib>
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

        half b_elements[8][8];
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

        simdgroup_load(b_tile, &b_elements[0][0], 8);
        simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix);
    }

    simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim);
}
