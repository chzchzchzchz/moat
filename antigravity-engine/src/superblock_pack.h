#pragma once
//
// INT4 super-block packing — the CPU side of the quantized weight format.
//
// This header deliberately has no Metal or Objective-C dependency. The engine's
// quantizeToSuperblocks() is a thin wrapper that allocates an MTLBuffer and calls
// packSuperblocks() into it, which means the arithmetic that decides what bytes
// the GPU reads can be tested on a machine with no GPU, and cross-checked against
// the reference implementation in src/dequant.py.
//
// Format (144 bytes per super-block, 256 INT4 values):
//   bytes  0..15  : 8 FP16 group scales, little-endian
//   bytes 16..143 : 128 packed bytes; low nibble = even element, high = odd,
//                   each value biased by +8 so [-8, 7] stores as [0, 15]
//   one scale per 32 consecutive elements
//
// Quantization is symmetric per group: scale = max(|w|) / 7, then
// q = clamp(round_half_to_even(w / fp16(scale)), -8, 7). Rounding against the
// FP16-rounded scale (not the exact one) matters: the kernel multiplies by the
// stored FP16 scale, so quantizing against anything else biases every value.
//
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace antigravity {

constexpr size_t kInt4GroupSize        = 32;
constexpr size_t kInt4GroupsPerBlock   = 8;
constexpr size_t kInt4ElementsPerBlock = kInt4GroupSize * kInt4GroupsPerBlock;  // 256
constexpr size_t kInt4ScaleBytes       = kInt4GroupsPerBlock * 2;               // 16
constexpr size_t kInt4PackedBytes      = kInt4ElementsPerBlock / 2;            // 128
constexpr size_t kInt4BlockBytes       = kInt4ScaleBytes + kInt4PackedBytes;   // 144

// IEEE half <-> float, done in integer arithmetic rather than via _Float16 so the
// result is identical on every host the tests run on.
inline float fp16_bits_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant       = (uint32_t)(h & 0x03FFu);
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       // +/- zero
        } else {
            int shift = -1;                    // subnormal: renormalize
            do { mant <<= 1; shift++; } while ((mant & 0x0400u) == 0);
            mant &= 0x03FFu;
            bits = sign | ((uint32_t)(127 - 15 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);   // Inf / NaN
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t float_to_fp16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    const uint32_t sign   = (x >> 16) & 0x8000u;
    const uint32_t raw_e  = (x >> 23) & 0xFFu;
    uint32_t mant         = x & 0x007FFFFFu;

    if (raw_e == 0xFFu) {                                   // Inf / NaN
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }

    const int32_t exp = (int32_t)raw_e - 127 + 15;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);     // overflow -> Inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;               // underflow -> zero
        mant |= 0x00800000u;                                // restore implicit 1
        const uint32_t shift   = (uint32_t)(14 - exp);      // in [14, 24]
        const uint32_t value   = mant >> shift;
        const uint32_t rem     = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        uint32_t out = value;
        if (rem > halfway || (rem == halfway && (value & 1u))) out++;
        return (uint16_t)(sign | out);
    }

    // Round to nearest, ties to even. A mantissa carry propagates into the
    // exponent field on its own because the two fields are adjacent.
    uint32_t out = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) out++;
    return (uint16_t)(sign | out);
}

// Bytes needed to hold `n_elements` values, or 0 if the count cannot be packed.
inline size_t superblockBytesFor(size_t n_elements) {
    if (n_elements == 0 || n_elements % kInt4ElementsPerBlock != 0) return 0;
    return (n_elements / kInt4ElementsPerBlock) * kInt4BlockBytes;
}

// Pack `n_elements` FP16 values (as raw bits) into super-blocks. `out` must hold
// superblockBytesFor(n_elements) bytes. Returns false if the count is not a
// multiple of 256, in which case nothing is written.
inline bool packSuperblocks(const uint16_t* fp16, size_t n_elements, uint8_t* out) {
    if (!fp16 || !out) return false;
    if (superblockBytesFor(n_elements) == 0) return false;

    const size_t n_sb = n_elements / kInt4ElementsPerBlock;

    for (size_t sb = 0; sb < n_sb; sb++) {
        const uint16_t* block  = fp16 + sb * kInt4ElementsPerBlock;
        uint8_t* sb_out        = out + sb * kInt4BlockBytes;
        uint8_t* packed_out    = sb_out + kInt4ScaleBytes;

        int8_t q[kInt4ElementsPerBlock];

        for (size_t g = 0; g < kInt4GroupsPerBlock; g++) {
            const uint16_t* grp = block + g * kInt4GroupSize;

            float alpha = 0.0f;
            for (size_t i = 0; i < kInt4GroupSize; i++) {
                const float v = std::fabs(fp16_bits_to_float(grp[i]));
                if (v > alpha) alpha = v;
            }

            // An all-zero group would divide by zero; dequant.py substitutes 1.0.
            const uint16_t scale_bits =
                float_to_fp16_bits(alpha == 0.0f ? 1.0f : alpha / 7.0f);
            std::memcpy(sb_out + g * 2, &scale_bits, sizeof(scale_bits));

            const float scale = fp16_bits_to_float(scale_bits);
            for (size_t i = 0; i < kInt4GroupSize; i++) {
                // nearbyint under the default rounding mode is ties-to-even,
                // which is what numpy's np.round does in the reference packer.
                float r = std::nearbyint(fp16_bits_to_float(grp[i]) / scale);
                if (r < -8.0f) r = -8.0f;
                if (r >  7.0f) r =  7.0f;
                q[g * kInt4GroupSize + i] = (int8_t)r;
            }
        }

        for (size_t i = 0; i < kInt4PackedBytes; i++) {
            const uint8_t lo = (uint8_t)((q[2 * i]     + 8) & 0x0F);
            const uint8_t hi = (uint8_t)((q[2 * i + 1] + 8) & 0x0F);
            packed_out[i] = (uint8_t)(lo | (hi << 4));
        }
    }
    return true;
}

// Read one packed value back, using exactly the index arithmetic the Metal
// kernels use. Tests compare this against the packer's input.
inline float dequantSuperblockElement(const uint8_t* blocks, size_t flat_index) {
    const uint8_t* sb   = blocks + (flat_index / kInt4ElementsPerBlock) * kInt4BlockBytes;
    const size_t in_sb  = flat_index % kInt4ElementsPerBlock;

    const uint8_t packed = sb[kInt4ScaleBytes + (in_sb >> 1)];
    const int nib = (in_sb & 1u) ? (int)((packed >> 4) & 0x0F) - 8
                                 : (int)( packed       & 0x0F) - 8;

    uint16_t scale_bits;
    std::memcpy(&scale_bits, sb + (in_sb >> 5) * 2, sizeof(scale_bits));
    return (float)nib * fp16_bits_to_float(scale_bits);
}

}  // namespace antigravity
