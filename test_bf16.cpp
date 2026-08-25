#include <iostream>
#include <cstdint>
#include <cmath>

static inline uint16_t bf16_to_fp16(uint16_t bf16) {
    uint32_t sign = (bf16 >> 15) & 1;
    int32_t  exp  = ((bf16 >> 7) & 0xFF) - 127;
    uint32_t mant = bf16 & 0x7F;
    if (exp == 128) return (uint16_t)((sign << 15) | (0x1F << 10) | (mant >> 4));
    if (exp < -24) return (uint16_t)(sign << 15);
    int32_t fp16_exp = exp + 15;
    uint32_t fp16_mant = mant << 3;
    if (fp16_exp <= 0) {
        fp16_mant = (0x400 | fp16_mant) >> (1 - fp16_exp);
        fp16_exp = 0;
    } else if (fp16_exp >= 0x1F) {
        fp16_exp = 0x1F;
        fp16_mant = 0;
    }
    return (uint16_t)((sign << 15) | (fp16_exp << 10) | (fp16_mant & 0x3FF));
}

float bf16_to_float(uint16_t bf16) {
    uint32_t u32 = (uint32_t)bf16 << 16;
    float f;
    std::memcpy(&f, &u32, 4);
    return f;
}

float fp16_to_float(uint16_t fp16) {
    _Float16 h;
    std::memcpy(&h, &fp16, 2);
    return (float)h;
}

int main() {
    uint16_t test_vals[] = { 0x3F80, 0x3C00, 0xBF80, 0x3E20, 0x0000 };
    for (uint16_t v : test_vals) {
        float f_orig = bf16_to_float(v);
        uint16_t fp16_bits = bf16_to_fp16(v);
        float f_conv = fp16_to_float(fp16_bits);
        std::cout << BF16:
