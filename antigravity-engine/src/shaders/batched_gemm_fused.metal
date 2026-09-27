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
    // group_id is uint3 rather than uint2 only so that it matches tg_size: MSL
    // requires every vector kernel input to have the same element count, and
    // mixing uint2 with uint3 is rejected outright.
    uint3 group_id  [[threadgroup_position_in_grid]],
    uint  tid_in_tg [[thread_index_in_threadgroup]],
    uint3 tg_size   [[threads_per_threadgroup]]
) {
    // Threads of this threadgroup share the dequantization of each weight tile, so
    // the split is written against the actual threadgroup size rather than a
    // hardcoded 32. Hardcoding it would mean a change to threadsPerThreadgroup in
    // dispatchGEMM silently left tile cells unwritten, which reads as plausible
    // numbers rather than as a failure.
    uint helpers = tg_size.x * tg_size.y * tg_size.z;
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
        // Threads split the 64 tile elements rather than each thread dequantizing
        // all 64 — see the same loop in batched_gemm.metal.
        threadgroup half b_elements[8][8];
        for (uint e = tid_in_tg; e < 64; e += helpers) {
            uint r = e >> 3;
            uint c = e & 7;
            uint global_k = k + r;
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

        // Barrier before the cooperative read: without it the tile is raced.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(b_tile, (const threadgroup half*)&b_elements[0][0], 8);
        simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix);

        // And after: the next iteration's stores must not overtake this read.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Bounds-checked store, matching batched_gemm_simdgroup. A simdgroup_store always
    // writes a full 8x8 tile, and the guard at the top of this kernel only checks the
    // tile's ORIGIN — so an edge tile writes up to 7 rows and 7 columns past the logical
    // output. In a [rows x M_dim] buffer, running past the last column of row r lands in
    // row r+1, corrupting values that were already computed. That needs M_dim or N_batch
    // to not be a multiple of 8, which every dimension this engine uses today is
    // (2048, 5632, 256, 32000, 151936), but vocabularies like GPT-2's 50257 are not, and
    // the failure would be silently wrong logits rather than a crash.
    if (row_start + 8 <= N_batch && col_start + 8 <= M_dim) {
        simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim);
    } else {
        threadgroup half edge_tile[64];
        simdgroup_store(acc_matrix, edge_tile, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid_in_tg; e < 64; e += helpers) {
            uint r = e >> 3;
            uint c = e & 7;
            if (row_start + r < N_batch && col_start + c < M_dim) {
                output[(row_start + r) * M_dim + (col_start + c)] = edge_tile[e];
            }
        }
    }
}
