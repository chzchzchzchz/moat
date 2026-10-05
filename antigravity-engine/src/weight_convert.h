#pragma once
//
// Converting checkpoint weights into the FP16 the engine's buffers hold.
//
// This lived inside transformer_engine.mm, where nothing could test it. What "verified"
// meant was a Python reimplementation in tools/analyze_existing_artifacts.py agreeing with
// itself over 6,822 bit patterns — the shipping C++ was never run by a test. It is here so
// that it is, and so the F32 path below can be tested at all.
//
#include <cstdint>
#include <cstring>
#include <string>

#include "superblock_pack.h"   // float_to_fp16_bits, fp16_bits_to_float

namespace antigravity {

// BFloat16 (1 sign + 8 exp + 7 mantissa) -> Float16 (1 sign + 5 exp + 10 mantissa).
//
// Moved verbatim from transformer_engine.mm apart from the NaN case, noted below.
inline uint16_t bf16_to_fp16(uint16_t bf16) {
    const uint32_t sign = (uint32_t)((bf16 >> 15) & 1);
    const int32_t  exp  = (int32_t)((bf16 >> 7) & 0xFF) - 127;   // unbias BF16
    const uint32_t mant = (uint32_t)(bf16 & 0x7F);               // 7-bit mantissa

    if (exp == 128) {
        // Inf or NaN. The original returned (sign<<15) | (0x1F<<10) | (mant >> 4), which
        // for any NaN whose mantissa is below 0x10 — 0x7F81, a signalling NaN, among them
        // — shifts the mantissa to zero and yields Inf. A NaN in a checkpoint turning into
        // Inf is a corrupt checkpoint becoming a plausible one, which is the failure this
        // repository keeps finding. Preserve NaN-ness explicitly.
        const uint32_t fp16_mant = mant >> 4;
        if (mant != 0 && fp16_mant == 0) {
            return (uint16_t)((sign << 15) | (0x1F << 10) | 1);   // smallest FP16 NaN
        }
        return (uint16_t)((sign << 15) | (0x1F << 10) | fp16_mant);
    }
    if (exp < -25) {
        // Too small to round up to even the smallest FP16 subnormal (2^-24). BF16 zeros and
        // BF16 subnormals, which are around 1e-40, both land here. Signed zero.
        //
        // The bound was -24, which also discarded every value with exp == -25 — and those
        // are above half the smallest subnormal whenever their mantissa is non-zero, so
        // they should round UP to it rather than to zero. That was 254 patterns (127
        // mantissas x 2 signs), exactly the count still differing from the reference after
        // the subnormal path was taught to round. They go through that path now: with
        // exp == -25 the shift is 11, so the tie case (mantissa zero) rounds to even, which
        // is zero, and everything above the tie rounds to one subnormal ulp.
        return (uint16_t)(sign << 15);
    }

    int32_t  fp16_exp  = exp + 15;          // rebias for FP16
    uint32_t fp16_mant = mant << 3;         // 7-bit mantissa -> 10-bit, lossless

    if (fp16_exp <= 0) {
        // Subnormal in FP16. The original shifted and truncated:
        //
        //     fp16_mant = (0x400 | fp16_mant) >> (1 - fp16_exp);
        //
        // Compared against converting to float and back with round-to-nearest, that
        // disagreed on 1024 of the 65,536 BF16 patterns, every one of them by exactly
        // +1 ulp, over magnitudes from 3e-08 to 7.6e-06 — all below FP16's smallest
        // normal, 6.104e-05. Small, but the worst case is a value of 3.0e-08 truncating
        // to zero where it should round to the smallest subnormal, and rounding is not
        // more expensive than truncating.
        //
        // Round to nearest, ties to even, matching float_to_fp16_bits and IEEE 754.
        // exp >= -25 is guaranteed above, so the shift is between 1 and 11.
        const uint32_t implied = 0x400u | fp16_mant;
        const uint32_t shift = (uint32_t)(1 - fp16_exp);
        const uint32_t lost = implied & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        fp16_mant = implied >> shift;
        if (lost > half || (lost == half && (fp16_mant & 1u) != 0u)) fp16_mant++;
        // Rounding up out of the subnormal range carries into the exponent, giving the
        // smallest normal. Adding rather than OR-ing lets that carry happen; masking
        // fp16_mant to 10 bits here would turn it into zero.
        return (uint16_t)((sign << 15) | fp16_mant);
    }
    if (fp16_exp >= 0x1F) {
        fp16_exp = 0x1F;                    // overflow to Inf
        fp16_mant = 0;
    }

    return (uint16_t)((sign << 15) | ((uint32_t)fp16_exp << 10) | (fp16_mant & 0x3FF));
}

// The dtypes the engine can load into an FP16 buffer.
enum class SourceDtype { Unsupported, FP16, BF16, FP32 };

// safetensors writes dtypes upper-case, but the code this replaces also accepted "bf16"
// and "bfloat16", so those keep working rather than becoming a refusal.
inline SourceDtype sourceDtypeFromName(const std::string& dtype) {
    if (dtype == "BF16" || dtype == "bf16" || dtype == "bfloat16") return SourceDtype::BF16;
    if (dtype == "F16" || dtype == "f16" || dtype == "float16")    return SourceDtype::FP16;
    if (dtype == "F32" || dtype == "f32" || dtype == "float32")    return SourceDtype::FP32;
    return SourceDtype::Unsupported;
}

inline size_t sourceElementSize(SourceDtype dtype) {
    return dtype == SourceDtype::FP32 ? sizeof(float) : sizeof(uint16_t);
}

// Read element `index` from a tensor's raw bytes and give it back as FP16 bits.
//
// The path this replaces branched on BF16 only, and treated everything else — F32
// included — as "already FP16 or compatible, direct copy" of
// min(num_elements * 2, span) bytes. For F32 that copies the first half of the data and
// reads each pair of F32 bytes as one FP16 value: every weight wrong, no error.
inline uint16_t elementAsFp16(const void* base, SourceDtype dtype, size_t index) {
    const char* bytes = (const char*)base;
    switch (dtype) {
        case SourceDtype::FP16: {
            uint16_t h;
            std::memcpy(&h, bytes + index * sizeof(uint16_t), sizeof(uint16_t));
            return h;
        }
        case SourceDtype::BF16: {
            uint16_t b;
            std::memcpy(&b, bytes + index * sizeof(uint16_t), sizeof(uint16_t));
            return bf16_to_fp16(b);
        }
        case SourceDtype::FP32: {
            float f;
            std::memcpy(&f, bytes + index * sizeof(float), sizeof(float));
            return float_to_fp16_bits(f);
        }
        case SourceDtype::Unsupported:
            break;
    }
    return 0;
}

}  // namespace antigravity
