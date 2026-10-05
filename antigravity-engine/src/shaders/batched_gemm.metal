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
// NOT DISPATCHED ANYWHERE, and deliberately so.
//
// This expands super-blocks into a full FP16 weight buffer before the GEMM, which is the
// approach INT4 was originally wired for and which gives up the entire point: decode is
// memory-bandwidth bound, so materialising FP16 weights means streaming the same bytes
// per token as never quantizing at all. The memory saving survives only on disk.
//
// gemv_int4_kernel and fused_batched_gemm_int4 below take the other route — they read the
// packed bytes and dequantize into registers inside the inner loop, so the weights stay
// 4-bit in VRAM for the whole decode and the bandwidth ceiling actually rises.
//
// Kept because antigravity_c_api.cpp still creates a pipeline for it (also never
// dispatched) and because the reference semantics are useful. Do not wire it into the
// decode path: doing so would quietly undo the reason INT4 exists.
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
// Uses simdgroup_matrix multiply-accumulate on half tiles into a float accumulator.
// =============================================================================
kernel void batched_gemm_simdgroup(
    device const half*       activations   [[buffer(0)]], // [N x K]
    device const half*       weights       [[buffer(1)]], // [K x M]
    device half*             output        [[buffer(2)]], // [N x M]
    constant uint&           N_batch       [[buffer(3)]],
    constant uint&           K_dim         [[buffer(4)]],
    constant uint&           M_dim         [[buffer(5)]],
    uint3 group_id  [[threadgroup_position_in_grid]],
    uint  tid_in_tg [[thread_index_in_threadgroup]],
    uint3 tg_size   [[threads_per_threadgroup]]
) {
    uint row_start = group_id.y * 8;
    uint col_start = group_id.x * 8;
    uint helpers = tg_size.x * tg_size.y * tg_size.z;

    if (row_start >= N_batch || col_start >= M_dim) return;

    // The running sum is FLOAT. It was half — simdgroup_matrix<half, 8, 8> — so every output
    // was a running FP16 sum over K terms: 2,048 for the attention and MLP input projections,
    // 5,632 for the down projection. FP16 carries about three significant digits; once the sum
    // is large, small terms round away entirely, and past 65,504 it is Inf. The GEMV kernels,
    // which the engine uses for a single row, already accumulated in float, so the same model
    // was computed more accurately with one channel than with eight.
    //
    // Measured, so this is not over-sold: emulating FP16 accumulation on the CPU over TinyLlama
    // left greedy output unchanged over 64 tokens (tools/experiments/fp16_accumulation.py), so
    // this is NOT what broke gsm8k_full_checkpoint.json. It is the better kernel, and it
    // removes one difference between the 1-channel and 8-channel paths.
    //
    // The tiles stay half: half x half products accumulate into the float matrix, so the
    // bandwidth the GEMM reads is unchanged.
    simdgroup_matrix<float, 8, 8> acc_matrix = simdgroup_matrix<float, 8, 8>(0.0f);

    // Accumulate over K dimension in chunks of 8
    for (uint k = 0; k < K_dim; k += 8) {
        simdgroup_matrix<half, 8, 8> a_tile;
        simdgroup_matrix<half, 8, 8> b_tile;

        // Load Activation Tile (A) [8 x 8] from device memory
        simdgroup_load(a_tile, activations + row_start * K_dim + k, K_dim);

        // Load Weight Tile (B) [8 x 8] from device memory
        simdgroup_load(b_tile, weights + k * M_dim + col_start, M_dim);

        simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix);
    }

    // A float matrix cannot be simdgroup_store-d into a half buffer, so every tile is staged
    // through threadgroup memory and converted per element. The copy is spread across the
    // threadgroup and bounds-checked, so a tile hanging past N_batch or M_dim writes only its
    // in-range cells — with 2 channels, which is the sanity check's default, that is every
    // tile. It used to be one thread copying all 64 cells serially.
    threadgroup float out_tile[64];
    simdgroup_store(acc_matrix, out_tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid_in_tg; e < 64; e += helpers) {
        uint r = e >> 3;
        uint c = e & 7;
        if (row_start + r < N_batch && col_start + c < M_dim) {
            output[(row_start + r) * M_dim + (col_start + c)] = half(out_tile[e]);
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

    // Float accumulator, as in batched_gemm_simdgroup — see the note there. The
    // dequantized tile and the activations stay half.
    simdgroup_matrix<float, 8, 8> acc_matrix = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < K_dim; k += 8) {
        simdgroup_matrix<half, 8, 8> a_tile;
        simdgroup_matrix<half, 8, 8> b_tile;

        simdgroup_load(a_tile, activations + row_start * K_dim + k, K_dim);

        // The 8x8 weight tile is dequantized cooperatively, each thread taking
        // every `helpers`-th element. Every thread used to run this whole loop, so
        // at the 32-thread threadgroup dispatchGEMM encodes, all 64 dequantizations
        // were performed 32 times over and 31 of every 32 stores were a redundant
        // write of an identical value.
        threadgroup half b_elements[8][8];
        for (uint e = tid_in_tg; e < 64; e += helpers) {
            uint r = e >> 3;          // e / 8
            uint c = e & 7;           // e % 8
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

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(b_tile, (const threadgroup half*)&b_elements[0][0], 8);

        simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix);

        // Second barrier: without it the next iteration's stores to b_elements can
        // land while another lane is still reading this iteration's tile. That race
        // was present before the split too — identical values do not make it safe,
        // because the values differ between iterations of k.
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
    // A float accumulator cannot be simdgroup_store-d into a half buffer, so every tile is
    // staged and converted per element, with the same bounds check — which also subsumes the
    // fast path, since a full tile is simply one where every cell is in range.
    threadgroup float out_tile[64];
    simdgroup_store(acc_matrix, out_tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid_in_tg; e < 64; e += helpers) {
        uint r = e >> 3;
        uint c = e & 7;
        if (row_start + r < N_batch && col_start + c < M_dim) {
            output[(row_start + r) * M_dim + (col_start + c)] = half(out_tile[e]);
        }
    }
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
