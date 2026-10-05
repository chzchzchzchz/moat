// Qwen3.5 text decoder on Metal. See qwen35_engine.h.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <vector>

#include "qwen35_engine.h"
#include "safetensors_header.h"
#include "sampling.h"

namespace {

// ---------------------------------------------------------------------------------------------
// Metal kernels. Weights are bf16 read as raw ushort (a bf16 is the top 16 bits of a float32, so
// widening is a shift and loses nothing); activations, recurrent state and logits are float32.
// ---------------------------------------------------------------------------------------------
const char* kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float bf(ushort v) { return as_type<float>(uint(v) << 16); }
inline float silu_f(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + exp(-x)); }

kernel void k_embed(device const int* tok [[buffer(0)]], device const ushort* emb [[buffer(1)]],
                    device float* out [[buffer(2)]], constant uint& H [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= H) return;
    out[(ulong)gid.y * H + gid.x] = bf(emb[(ulong)tok[gid.y] * H + gid.x]);
}

// out[row] = in[row] / rms(in[row]) * (w + plus1). One threadgroup per row.
kernel void k_rmsnorm(device const float* in [[buffer(0)]], device const ushort* w [[buffer(1)]],
                      device float* out [[buffer(2)]], constant uint& D [[buffer(3)]],
                      constant float& eps [[buffer(4)]], constant uint& plus1 [[buffer(5)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tg_size [[threads_per_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    device const float* x = in + (ulong)row * D;
    float ss = 0.0f;
    for (uint i = tid; i < D; i += tg_size) { float v = x[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    uint nsg = (tg_size + 31) / 32;
    for (uint i = 0; i < nsg; i++) tot += part[i];
    float inv = rsqrt(tot / float(D) + eps);
    for (uint i = tid; i < D; i += tg_size) {
        float wv = bf(w[i]);
        if (plus1 != 0) wv += 1.0f;
        out[(ulong)row * D + i] = x[i] * inv * wv;
    }
}

// C[M,N] (+)= A[M,K] * W[N,K]^T, W in bf16. Each simdgroup owns one output column n and walks
// the weight row once per tile of 16 activation rows, so a decode step with 8 channels reads
// every weight exactly once.
kernel void k_matmul(device const float* A [[buffer(0)]], device const ushort* W [[buffer(1)]],
                     device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                     constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
                     constant uint& accumulate [[buffer(6)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const uint n = tg * 4 + sg;
    if (n >= N) return;
    device const ushort4* wrow = (device const ushort4*)(W + (ulong)n * K);
    for (uint m0 = 0; m0 < M; m0 += 16) {
        float acc[16];
        for (uint i = 0; i < 16; i++) acc[i] = 0.0f;
        for (uint k = lane * 4; k < K; k += 128) {
            ushort4 wb = wrow[k >> 2];
            float4 w4 = float4(bf(wb.x), bf(wb.y), bf(wb.z), bf(wb.w));
            for (uint i = 0; i < 16; i++) {
                if (m0 + i < M) {
                    float4 a4 = *(device const float4*)(A + (ulong)(m0 + i) * K + k);
                    acc[i] += dot(w4, a4);
                }
            }
        }
        for (uint i = 0; i < 16; i++) {
            float s = simd_sum(acc[i]);
            if (lane == 0 && m0 + i < M) {
                device float* c = C + (ulong)(m0 + i) * N + n;
                *c = (accumulate != 0) ? (*c + s) : s;
            }
        }
    }
}


// Faster decode matmul: each simdgroup owns 4 output columns and 8 activation rows, so every
// activation fragment loaded is reused against 4 weight rows, and the 32 partial sums are reduced
// with a transposing shuffle tree (31 shuffles) instead of 32 simd_sums (160).
kernel void k_matmul4(device const float* A [[buffer(0)]], device const ushort* W [[buffer(1)]],
                      device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
                      constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
                      constant uint& accumulate [[buffer(6)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint n0 = (tg * 4 + sg) * 4;
    if (n0 >= N) return;
    device const ushort4* wr0 = (device const ushort4*)(W + (ulong)min(n0 + 0, N - 1) * K);
    device const ushort4* wr1 = (device const ushort4*)(W + (ulong)min(n0 + 1, N - 1) * K);
    device const ushort4* wr2 = (device const ushort4*)(W + (ulong)min(n0 + 2, N - 1) * K);
    device const ushort4* wr3 = (device const ushort4*)(W + (ulong)min(n0 + 3, N - 1) * K);
    for (uint m0 = 0; m0 < M; m0 += 8) {
        float acc[32];
        for (uint j = 0; j < 32; j++) acc[j] = 0.0f;
        for (uint k = lane * 4; k < K; k += 128) {
            float4 a[8];
            for (uint i = 0; i < 8; i++)
                a[i] = (m0 + i < M) ? *(device const float4*)(A + (ulong)(m0 + i) * K + k) : float4(0.0f);
            ushort4 b0 = wr0[k >> 2], b1 = wr1[k >> 2], b2 = wr2[k >> 2], b3 = wr3[k >> 2];
            float4 w0 = float4(bf(b0.x), bf(b0.y), bf(b0.z), bf(b0.w));
            float4 w1 = float4(bf(b1.x), bf(b1.y), bf(b1.z), bf(b1.w));
            float4 w2 = float4(bf(b2.x), bf(b2.y), bf(b2.z), bf(b2.w));
            float4 w3 = float4(bf(b3.x), bf(b3.y), bf(b3.z), bf(b3.w));
            for (uint i = 0; i < 8; i++) {
                acc[0 * 8 + i] += dot(w0, a[i]);
                acc[1 * 8 + i] += dot(w1, a[i]);
                acc[2 * 8 + i] += dot(w2, a[i]);
                acc[3 * 8 + i] += dot(w3, a[i]);
            }
        }
        // Transposing reduction: after the five steps lane l holds the total of acc[l].
        for (uint half_ = 16; half_ >= 1; half_ >>= 1) {
            const bool upper = (lane & half_) != 0;
            for (uint j = 0; j < half_; j++) {
                float keep = upper ? acc[j + half_] : acc[j];
                float send = upper ? acc[j] : acc[j + half_];
                acc[j] = keep + simd_shuffle_xor(send, half_);
            }
        }
        const uint c = lane >> 3, i = lane & 7;
        if (n0 + c < N && m0 + i < M) {
            device float* cp = C + (ulong)(m0 + i) * N + n0 + c;
            *cp = (accumulate != 0) ? (*cp + acc[0]) : acc[0];
        }
    }
}

kernel void k_silu_mul(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                       device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] = silu_f(g[i]) * u[i];
}

// ---- full attention ------------------------------------------------------------------------

inline float rope_angle(uint pos, uint i, uint half_rot, float theta) {
    return float(pos) * pow(theta, -float(i) / float(half_rot));
}

// q_proj output is [row][head][q(256) | gate(256)]. Split it, RMSNorm q per head with (1 + w),
// rotate the first rot_dim dims, and write q and gate as separate [row][head*256] arrays.
kernel void k_attn_prep_q(device const float* qg [[buffer(0)]], device const ushort* qn [[buffer(1)]],
                          device float* q_out [[buffer(2)]], device float* gate_out [[buffer(3)]],
                          constant uint& nH [[buffer(4)]], constant uint& rps [[buffer(5)]],
                          constant uint& pos0 [[buffer(6)]], constant float& theta [[buffer(7)]],
                          constant float& eps [[buffer(8)]], constant uint& rot_dim [[buffer(9)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          uint2 tid2 [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    threadgroup float sh[256];
    const uint d = tid2.x;
    const uint h = tgp.x, row = tgp.y;
    device const float* src = qg + (ulong)row * nH * 512 + h * 512;
    float x = src[d];
    gate_out[(ulong)row * nH * 256 + h * 256 + d] = src[256 + d];
    float ss = simd_sum(x * x);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8; i++) tot += part[i];
    float y = x * rsqrt(tot / 256.0f + eps) * (1.0f + bf(qn[d]));
    sh[d] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot_dim / 2;
    float r = y;
    if (d < rot_dim) {
        uint i = d % half_rot;
        float ang = rope_angle(pos0 + row % rps, i, half_rot, theta);
        float c = cos(ang), s = sin(ang);
        if (d < half_rot) r = y * c - sh[d + half_rot] * s;
        else              r = y * c + sh[d - half_rot] * s;
    }
    q_out[(ulong)row * nH * 256 + h * 256 + d] = r;
}

// k: RMSNorm per kv head, rotate, store to the K cache as half. v: store to the V cache as half.
// Cache layout [slot][kv head][max_seq][256].
kernel void k_attn_prep_kv(device const float* kp [[buffer(0)]], device const float* vp [[buffer(1)]],
                           device const ushort* kn [[buffer(2)]],
                           device half* Kc [[buffer(3)]], device half* Vc [[buffer(4)]],
                           constant uint& nKV [[buffer(5)]], constant uint& rps [[buffer(6)]],
                           constant uint& pos0 [[buffer(7)]], constant float& theta [[buffer(8)]],
                           constant float& eps [[buffer(9)]], constant uint& rot_dim [[buffer(10)]],
                           constant uint& maxSeq [[buffer(11)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           uint2 tid2 [[thread_position_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    threadgroup float sh[256];
    const uint d = tid2.x;
    const uint kvh = tgp.x, row = tgp.y;
    const uint slot = row / rps;
    const uint pos = pos0 + row % rps;
    float x = kp[(ulong)row * nKV * 256 + kvh * 256 + d];
    float ss = simd_sum(x * x);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8; i++) tot += part[i];
    float y = x * rsqrt(tot / 256.0f + eps) * (1.0f + bf(kn[d]));
    sh[d] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_rot = rot_dim / 2;
    float r = y;
    if (d < rot_dim) {
        uint i = d % half_rot;
        float ang = rope_angle(pos, i, half_rot, theta);
        float c = cos(ang), s = sin(ang);
        if (d < half_rot) r = y * c - sh[d + half_rot] * s;
        else              r = y * c + sh[d - half_rot] * s;
    }
    const ulong ci = ((ulong)(slot * nKV + kvh) * maxSeq + pos) * 256 + d;
    Kc[ci] = half(r);
    Vc[ci] = half(vp[(ulong)row * nKV * 256 + kvh * 256 + d]);
}

// One threadgroup (128 threads) per (head, row). Scores for positions 0..pos, softmax, weighted
// sum of V, then the output gate: out = softmax(qK^T/16) V * sigmoid(gate).
kernel void k_attn(device const float* q [[buffer(0)]], device const float* gate [[buffer(1)]],
                   device const half* Kc [[buffer(2)]], device const half* Vc [[buffer(3)]],
                   device float* out [[buffer(4)]],
                   constant uint& nH [[buffer(5)]], constant uint& nKV [[buffer(6)]],
                   constant uint& rps [[buffer(7)]], constant uint& pos0 [[buffer(8)]],
                   constant uint& maxSeq [[buffer(9)]], constant float& scale [[buffer(10)]],
                   uint2 tgp [[threadgroup_position_in_grid]],
                   uint2 tid2 [[thread_position_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    const uint tid = tid2.x;
    threadgroup float sc[4096];
    threadgroup float redm[4];
    threadgroup float reds[4];
    const uint h = tgp.x, row = tgp.y;
    const uint slot = row / rps;
    const uint pos = pos0 + row % rps;
    const uint n = pos + 1;
    const uint kvh = h / (nH / nKV);
    device const half* Kb = Kc + (ulong)(slot * nKV + kvh) * maxSeq * 256;
    device const half* Vb = Vc + (ulong)(slot * nKV + kvh) * maxSeq * 256;

    float qv[8];
    device const float* qp = q + (ulong)row * nH * 256 + h * 256 + lane * 8;
    for (uint i = 0; i < 8; i++) qv[i] = qp[i];

    for (uint j = sg; j < n; j += 4) {
        device const half* kr = Kb + (ulong)j * 256 + lane * 8;
        float acc = 0.0f;
        for (uint i = 0; i < 8; i++) acc += qv[i] * float(kr[i]);
        acc = simd_sum(acc);
        if (lane == 0) sc[j] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    for (uint j = tid; j < n; j += 128) m = max(m, sc[j]);
    m = simd_max(m);
    if (lane == 0) redm[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(redm[0], redm[1]), max(redm[2], redm[3]));

    float s = 0.0f;
    for (uint j = tid; j < n; j += 128) { float e = exp(sc[j] - m); sc[j] = e; s += e; }
    s = simd_sum(s);
    if (lane == 0) reds[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = reds[0] + reds[1] + reds[2] + reds[3];

    const uint d0 = tid * 2;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint j = 0; j < n; j++) {
        float p = sc[j];
        half2 v = *(device const half2*)(Vb + (ulong)j * 256 + d0);
        a0 += p * float(v.x);
        a1 += p * float(v.y);
    }
    const ulong oi = (ulong)row * nH * 256 + h * 256 + d0;
    out[oi]     = (a0 / tot) * sigmoid_f(gate[oi]);
    out[oi + 1] = (a1 / tot) * sigmoid_f(gate[oi + 1]);
}

// ---- Gated DeltaNet (linear attention) -----------------------------------------------------

// Causal depthwise conv, kernel 4, then SiLU, over the qkv channels. State holds the last three
// pre-conv inputs per slot and channel. Rows of one slot are consecutive and are processed in
// order by one thread.
kernel void k_conv1d(device const float* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                     device float* state [[buffer(2)]], device float* out [[buffer(3)]],
                     constant uint& rps [[buffer(4)]], constant uint& D [[buffer(5)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, slot = gid.y;
    if (d >= D) return;
    const float w0 = bf(w[d * 4 + 0]), w1 = bf(w[d * 4 + 1]), w2 = bf(w[d * 4 + 2]), w3 = bf(w[d * 4 + 3]);
    device float* st = state + ((ulong)slot * D + d) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (uint t = 0; t < rps; t++) {
        const ulong idx = (ulong)(slot * rps + t) * D + d;
        const float xin = x[idx];
        out[idx] = silu_f(w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin);
        s0 = s1; s1 = s2; s2 = xin;
    }
    st[0] = s0; st[1] = s1; st[2] = s2;
}

// The gated delta rule, one threadgroup per (head, slot), 256 threads. Thread (half, v) owns
// column v of the 128x128 state for k in [half*64, half*64+64). Per token:
//   q, k <- l2norm; q <- q / sqrt(128);   g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
//   S <- S * exp(g);  d <- (v - S^T k) * beta;  S <- S + k d^T;  out <- S^T q
kernel void k_deltanet(device const float* conv [[buffer(0)]], device const float* bproj [[buffer(1)]],
                       device const float* aproj [[buffer(2)]], device const float* Aneg [[buffer(3)]],
                       device const float* dtb [[buffer(4)]], device float* state [[buffer(5)]],
                       device float* out [[buffer(6)]],
                       constant uint& rps [[buffer(7)]], constant uint& nHeads [[buffer(8)]],
                       constant uint& convDim [[buffer(9)]], constant uint& keyDim [[buffer(10)]],
                       constant uint& valDim [[buffer(11)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint tid = tid2.x;
    threadgroup float qs[128];
    threadgroup float ks[128];
    threadgroup float part1[256];
    threadgroup float part2[256];
    threadgroup float redq[8];
    threadgroup float redk[8];
    const uint h = tgp.x, slot = tgp.y;
    const uint v = tid & 127;
    const uint hf = tid >> 7;
    device float* S = state + (ulong)(slot * nHeads + h) * 128 * 128;
    float s[64];
    for (uint i = 0; i < 64; i++) s[i] = S[(ulong)(hf * 64 + i) * 128 + v];

    for (uint t = 0; t < rps; t++) {
        const uint row = slot * rps + t;
        device const float* cr = conv + (ulong)row * convDim;
        float qv = 0.0f, kv = 0.0f;
        if (tid < 128) { qv = cr[h * 128 + tid]; kv = cr[keyDim + h * 128 + tid]; }
        float qq = simd_sum(qv * qv), kk = simd_sum(kv * kv);
        if (lane == 0) { redq[sg] = qq; redk[sg] = kk; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qsum = 0.0f, ksum = 0.0f;
        for (uint i = 0; i < 8; i++) { qsum += redq[i]; ksum += redk[i]; }
        if (tid < 128) {
            qs[tid] = qv * rsqrt(qsum + 1e-6f) * 0.08838834764831845f;
            ks[tid] = kv * rsqrt(ksum + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float a = aproj[(ulong)row * nHeads + h];
        const float b = bproj[(ulong)row * nHeads + h];
        const float xx = a + dtb[h];
        const float sp = (xx > 20.0f) ? xx : log(1.0f + exp(xx));
        const float decay = exp(Aneg[h] * sp);
        const float beta = sigmoid_f(b);
        const float vt = cr[2 * keyDim + h * 128 + v];

        float kvm = 0.0f;
        for (uint i = 0; i < 64; i++) { s[i] *= decay; kvm += s[i] * ks[hf * 64 + i]; }
        part1[tid] = kvm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float delta = (vt - (part1[v] + part1[128 + v])) * beta;
        float o = 0.0f;
        for (uint i = 0; i < 64; i++) { s[i] += ks[hf * 64 + i] * delta; o += s[i] * qs[hf * 64 + i]; }
        part2[tid] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hf == 0) out[(ulong)row * valDim + h * 128 + v] = part2[v] + part2[128 + v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 64; i++) S[(ulong)(hf * 64 + i) * 128 + v] = s[i];
}

// y = w * rmsnorm(x) * silu(z), per (row, head) of 128. w is float32.
kernel void k_gated_norm(device const float* x [[buffer(0)]], device const float* z [[buffer(1)]],
                         device const float* w [[buffer(2)]], device float* out [[buffer(3)]],
                         constant uint& valDim [[buffer(4)]], constant float& eps [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]],
                         uint2 tid2 [[thread_position_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tid2.x;
    threadgroup float part[4];
    const uint h = tgp.x, row = tgp.y;
    const ulong idx = (ulong)row * valDim + h * 128 + t;
    float xv = x[idx];
    float ss = simd_sum(xv * xv);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = part[0] + part[1] + part[2] + part[3];
    out[idx] = w[t] * (xv * rsqrt(tot / 128.0f + eps)) * silu_f(z[idx]);
}
)MSL";

// ---------------------------------------------------------------------------------------------
// Minimal JSON field readers for config.json. The file is machine-written and its keys are known;
// what matters is reading them from text_config and not from vision_config, which reuses names.
// ---------------------------------------------------------------------------------------------
std::string readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string dirName(const std::string& path) {
    size_t p = path.find_last_of('/');
    return p == std::string::npos ? std::string(".") : path.substr(0, p);
}

// The object that follows `"key":` — the text between its braces.
std::string objectOf(const std::string& text, const std::string& key) {
    size_t k = text.find("\"" + key + "\"");
    if (k == std::string::npos) return std::string();
    size_t open = text.find('{', k);
    if (open == std::string::npos) return std::string();
    int depth = 0;
    for (size_t i = open; i < text.size(); i++) {
        if (text[i] == '{') depth++;
        else if (text[i] == '}' && --depth == 0) return text.substr(open + 1, i - open - 1);
    }
    return std::string();
}

bool numberOf(const std::string& obj, const std::string& key, double& out) {
    size_t k = obj.find("\"" + key + "\"");
    if (k == std::string::npos) return false;
    size_t c = obj.find(':', k);
    if (c == std::string::npos) return false;
    char* end = nullptr;
    out = std::strtod(obj.c_str() + c + 1, &end);
    return end != obj.c_str() + c + 1;
}

struct Dims {
    int H = 0, I = 0, L = 0, nH = 0, nKV = 0, hd = 0, V = 0;
    int convK = 4, linKH = 0, linVH = 0, linKD = 0, linVD = 0;
    int keyDim = 0, valDim = 0, convDim = 0, rotDim = 0;
    float eps = 1e-6f, theta = 1e7f;
    bool tied = true;
    std::vector<bool> isLinear;
    std::vector<int> eos;
};

bool parseDims(const std::string& model_dir, Dims& d, std::string& err) {
    const std::string cfg = readFile(model_dir + "/config.json");
    if (cfg.empty()) { err = "cannot read " + model_dir + "/config.json"; return false; }
    const std::string tc = objectOf(cfg, "text_config");
    if (tc.empty()) { err = "config.json has no text_config"; return false; }
    auto need = [&](const char* key, int& dst) {
        double v;
        if (!numberOf(tc, key, v)) { err = std::string("text_config lacks ") + key; return false; }
        dst = (int)v;
        return true;
    };
    if (!need("hidden_size", d.H) || !need("intermediate_size", d.I) ||
        !need("num_hidden_layers", d.L) || !need("num_attention_heads", d.nH) ||
        !need("num_key_value_heads", d.nKV) || !need("head_dim", d.hd) ||
        !need("vocab_size", d.V) || !need("linear_conv_kernel_dim", d.convK) ||
        !need("linear_num_key_heads", d.linKH) || !need("linear_num_value_heads", d.linVH) ||
        !need("linear_key_head_dim", d.linKD) || !need("linear_value_head_dim", d.linVD)) {
        return false;
    }
    double v;
    if (numberOf(tc, "rms_norm_eps", v)) d.eps = (float)v;
    const std::string rp = objectOf(tc, "rope_parameters");
    double prf = 1.0;
    if (numberOf(rp, "rope_theta", v)) d.theta = (float)v;
    numberOf(rp, "partial_rotary_factor", prf);
    d.rotDim = (int)(d.hd * prf);
    d.keyDim = d.linKH * d.linKD;
    d.valDim = d.linVH * d.linVD;
    d.convDim = 2 * d.keyDim + d.valDim;
    d.tied = cfg.find("\"tie_word_embeddings\": false") == std::string::npos;

    // layer_types: the strings inside the array, in order.
    size_t lt = tc.find("\"layer_types\"");
    if (lt == std::string::npos) { err = "text_config lacks layer_types"; return false; }
    size_t open = tc.find('[', lt), close = tc.find(']', lt);
    std::string arr = tc.substr(open, close - open);
    size_t pos = 0;
    while ((pos = arr.find('"', pos)) != std::string::npos) {
        size_t e = arr.find('"', pos + 1);
        std::string name = arr.substr(pos + 1, e - pos - 1);
        d.isLinear.push_back(name == "linear_attention");
        pos = e + 1;
    }
    if ((int)d.isLinear.size() != d.L) { err = "layer_types length differs from num_hidden_layers"; return false; }

    // Shapes this implementation is written for.
    if (d.hd != 256 || d.linKD != 128 || d.linVD != 128 || d.linKH != d.linVH || d.convK != 4 ||
        d.rotDim <= 0 || d.rotDim > d.hd || d.nH % d.nKV != 0 || d.linKH != 16) {
        err = "model shape is outside what the Qwen3.5 kernels were written for";
        return false;
    }

    if (numberOf(tc, "eos_token_id", v)) d.eos.push_back((int)v);
    // <|im_end|> ends a chat turn; look its id up in the tokenizer rather than assume it.
    const std::string tok = readFile(model_dir + "/tokenizer.json");
    size_t m = tok.find("\"content\": \"<|im_end|>\"");
    if (m == std::string::npos) m = tok.find("\"content\":\"<|im_end|>\"");
    if (m != std::string::npos) {
        size_t idp = tok.rfind("\"id\"", m);
        if (idp != std::string::npos) {
            size_t c = tok.find(':', idp);
            d.eos.push_back((int)std::strtol(tok.c_str() + c + 1, nullptr, 10));
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------------------------

struct Qwen35Engine::Impl {
    int C = 8;
    int maxSeq = 2048;
    Dims dm;
    bool loaded = false;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    std::map<std::string, id<MTLComputePipelineState>> pso;

    // Weights.
    struct Layer {
        bool linear = false;
        id<MTLBuffer> inputNorm, postNorm, gate, up, down;
        // full attention
        id<MTLBuffer> q, k, v, o, qNorm, kNorm;
        // linear attention
        id<MTLBuffer> inQkv, inZ, inB, inA, outProj, conv, aNeg, dtBias, gnorm;
        // state
        id<MTLBuffer> kCache, vCache;      // [C][nKV][maxSeq][256] half
        id<MTLBuffer> convState, recState; // [C][convDim][3] and [C][16][128][128] float
    };
    std::vector<Layer> layers;
    id<MTLBuffer> embed = nil, finalNorm = nil, lmHead = nil;

    // Activation buffers, sized for R rows.
    int R = 0;
    id<MTLBuffer> tokBuf, hid, nrm, qg, qa, gateA, kp, vp, attnOut, qkv, conv, zp, bp, ap, dn, dnn, mg, mu;
    id<MTLBuffer> logits, capture;
    uint64_t allocated = 0;

    // QWEN35_SKIP=matmul,deltanet,attn,conv,norm drops those kernels from the decode graph. The
    // output is garbage; it exists so the time each kernel costs can be read off the step time.
    std::string skip;
    bool skipped(const char* name) const { return skip.find(name) != std::string::npos; }

    bool hasFixedSeed = false;
    uint64_t fixedSeed = 0, generationCalls = 0;
    uint64_t nonFinite = 0, emptyDist = 0;

    id<MTLBuffer> newBuf(size_t bytes) {
        id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        allocated += bytes;
        return b;
    }

    bool initMetal() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) { std::cerr << "[Qwen35] no Metal device\n"; return false; }
        queue = [device newCommandQueue];
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.fastMathEnabled = NO;
        lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                                   options:opts error:&err];
        if (!lib) {
            std::cerr << "[Qwen35] shader compile failed: "
                      << (err ? [[err localizedDescription] UTF8String] : "?") << "\n";
            return false;
        }
        for (NSString* name in @[@"k_embed", @"k_rmsnorm", @"k_matmul", @"k_matmul4", @"k_silu_mul", @"k_attn_prep_q",
                                 @"k_attn_prep_kv", @"k_attn", @"k_conv1d", @"k_deltanet", @"k_gated_norm"]) {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) { std::cerr << "[Qwen35] missing kernel " << [name UTF8String] << "\n"; return false; }
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];
            if (!p) {
                std::cerr << "[Qwen35] pipeline " << [name UTF8String] << " failed: "
                          << (err ? [[err localizedDescription] UTF8String] : "?") << "\n";
                return false;
            }
            pso[[name UTF8String]] = p;
        }
        return true;
    }

    // -- encoding helpers ------------------------------------------------------------------
    void bindU(id<MTLComputeCommandEncoder> e, uint32_t v, int idx) { [e setBytes:&v length:4 atIndex:idx]; }
    void bindF(id<MTLComputeCommandEncoder> e, float v, int idx) { [e setBytes:&v length:4 atIndex:idx]; }

    void rmsnorm(id<MTLComputeCommandEncoder> e, id<MTLBuffer> in, id<MTLBuffer> w, id<MTLBuffer> out,
                 uint32_t rows, uint32_t D, bool plus1) {
        if (skipped("norm")) return;
        [e setComputePipelineState:pso["k_rmsnorm"]];
        [e setBuffer:in offset:0 atIndex:0]; [e setBuffer:w offset:0 atIndex:1]; [e setBuffer:out offset:0 atIndex:2];
        bindU(e, D, 3); bindF(e, dm.eps, 4); bindU(e, plus1 ? 1 : 0, 5);
        [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void matmul(id<MTLComputeCommandEncoder> e, id<MTLBuffer> A, id<MTLBuffer> W, id<MTLBuffer> Cb,
                uint32_t M, uint32_t K, uint32_t N, bool accumulate) {
        if (skipped("matmul")) return;
        const bool v4 = !skipped("oldmatmul");
        [e setComputePipelineState:pso[v4 ? "k_matmul4" : "k_matmul"]];
        [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:Cb offset:0 atIndex:2];
        bindU(e, M, 3); bindU(e, K, 4); bindU(e, N, 5); bindU(e, accumulate ? 1 : 0, 6);
        const uint32_t per_tg = v4 ? 16 : 4;
        [e dispatchThreadgroups:MTLSizeMake((N + per_tg - 1) / per_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    // One decoder layer over `rows` rows. Rows are grouped into slots of `rps` consecutive rows
    // (decode: rps = 1, one slot per channel; prefill: rps = rows, slot 0). `hid` is the residual
    // stream and is updated in place.
    void encodeLayer(id<MTLComputeCommandEncoder> e, int li, uint32_t rows, uint32_t rps, uint32_t pos0) {
        Layer& L = layers[li];
        const uint32_t H = dm.H, I = dm.I;
        const uint32_t nslots = rows / rps;
        rmsnorm(e, hid, L.inputNorm, nrm, rows, H, true);

        if (!L.linear) {
            const uint32_t nH = dm.nH, nKV = dm.nKV;
            matmul(e, nrm, L.q, qg, rows, H, nH * 512, false);
            matmul(e, nrm, L.k, kp, rows, H, nKV * 256, false);
            matmul(e, nrm, L.v, vp, rows, H, nKV * 256, false);

            [e setComputePipelineState:pso["k_attn_prep_q"]];
            [e setBuffer:qg offset:0 atIndex:0]; [e setBuffer:L.qNorm offset:0 atIndex:1];
            [e setBuffer:qa offset:0 atIndex:2]; [e setBuffer:gateA offset:0 atIndex:3];
            bindU(e, nH, 4); bindU(e, rps, 5); bindU(e, pos0, 6); bindF(e, dm.theta, 7); bindF(e, dm.eps, 8);
            bindU(e, dm.rotDim, 9);
            [e dispatchThreadgroups:MTLSizeMake(nH, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [e setComputePipelineState:pso["k_attn_prep_kv"]];
            [e setBuffer:kp offset:0 atIndex:0]; [e setBuffer:vp offset:0 atIndex:1];
            [e setBuffer:L.kNorm offset:0 atIndex:2]; [e setBuffer:L.kCache offset:0 atIndex:3];
            [e setBuffer:L.vCache offset:0 atIndex:4];
            bindU(e, nKV, 5); bindU(e, rps, 6); bindU(e, pos0, 7); bindF(e, dm.theta, 8); bindF(e, dm.eps, 9);
            bindU(e, dm.rotDim, 10); bindU(e, maxSeq, 11);
            [e dispatchThreadgroups:MTLSizeMake(nKV, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            if (!skipped("attn")) {
            [e setComputePipelineState:pso["k_attn"]];
            [e setBuffer:qa offset:0 atIndex:0]; [e setBuffer:gateA offset:0 atIndex:1];
            [e setBuffer:L.kCache offset:0 atIndex:2]; [e setBuffer:L.vCache offset:0 atIndex:3];
            [e setBuffer:attnOut offset:0 atIndex:4];
            bindU(e, nH, 5); bindU(e, nKV, 6); bindU(e, rps, 7); bindU(e, pos0, 8); bindU(e, maxSeq, 9);
            bindF(e, 1.0f / sqrtf((float)dm.hd), 10);
            [e dispatchThreadgroups:MTLSizeMake(nH, rows, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            matmul(e, attnOut, L.o, hid, rows, nH * 256, H, true);
        } else {
            const uint32_t cd = dm.convDim, vd = dm.valDim, nh = dm.linVH;
            matmul(e, nrm, L.inQkv, qkv, rows, H, cd, false);
            matmul(e, nrm, L.inZ, zp, rows, H, vd, false);
            matmul(e, nrm, L.inB, bp, rows, H, nh, false);
            matmul(e, nrm, L.inA, ap, rows, H, nh, false);

            if (!skipped("conv")) {
            [e setComputePipelineState:pso["k_conv1d"]];
            [e setBuffer:qkv offset:0 atIndex:0]; [e setBuffer:L.conv offset:0 atIndex:1];
            [e setBuffer:L.convState offset:0 atIndex:2]; [e setBuffer:conv offset:0 atIndex:3];
            bindU(e, rps, 4); bindU(e, cd, 5);
            [e dispatchThreads:MTLSizeMake(cd, nslots, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }

            if (!skipped("deltanet")) {
            [e setComputePipelineState:pso["k_deltanet"]];
            [e setBuffer:conv offset:0 atIndex:0]; [e setBuffer:bp offset:0 atIndex:1];
            [e setBuffer:ap offset:0 atIndex:2]; [e setBuffer:L.aNeg offset:0 atIndex:3];
            [e setBuffer:L.dtBias offset:0 atIndex:4]; [e setBuffer:L.recState offset:0 atIndex:5];
            [e setBuffer:dn offset:0 atIndex:6];
            bindU(e, rps, 7); bindU(e, nh, 8); bindU(e, cd, 9); bindU(e, dm.keyDim, 10); bindU(e, vd, 11);
            [e dispatchThreadgroups:MTLSizeMake(nh, nslots, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            [e setComputePipelineState:pso["k_gated_norm"]];
            [e setBuffer:dn offset:0 atIndex:0]; [e setBuffer:zp offset:0 atIndex:1];
            [e setBuffer:L.gnorm offset:0 atIndex:2]; [e setBuffer:dnn offset:0 atIndex:3];
            bindU(e, vd, 4); bindF(e, dm.eps, 5);
            [e dispatchThreadgroups:MTLSizeMake(nh, rows, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            matmul(e, dnn, L.outProj, hid, rows, vd, H, true);
        }

        rmsnorm(e, hid, L.postNorm, nrm, rows, H, true);
        matmul(e, nrm, L.gate, mg, rows, H, I, false);
        matmul(e, nrm, L.up, mu, rows, H, I, false);
        [e setComputePipelineState:pso["k_silu_mul"]];
        [e setBuffer:mg offset:0 atIndex:0]; [e setBuffer:mu offset:0 atIndex:1]; [e setBuffer:mg offset:0 atIndex:2];
        bindU(e, rows * I, 3);
        [e dispatchThreads:MTLSizeMake(rows * I, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        matmul(e, mg, L.down, hid, rows, I, H, true);
    }

    void encodeEmbed(id<MTLComputeCommandEncoder> e, uint32_t rows) {
        [e setComputePipelineState:pso["k_embed"]];
        [e setBuffer:tokBuf offset:0 atIndex:0]; [e setBuffer:embed offset:0 atIndex:1];
        [e setBuffer:hid offset:0 atIndex:2]; bindU(e, dm.H, 3);
        [e dispatchThreads:MTLSizeMake(dm.H, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    void encodeHead(id<MTLComputeCommandEncoder> e, uint32_t rows) {
        rmsnorm(e, hid, finalNorm, nrm, rows, dm.H, true);
        matmul(e, nrm, lmHead, logits, rows, dm.H, dm.V, false);
    }

    bool run(id<MTLCommandBuffer> cb) {
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "[Qwen35] command buffer failed: "
                      << (cb.error ? [[cb.error localizedDescription] UTF8String] : "unknown") << "\n";
            return false;
        }
        return true;
    }

    void resetState() {
        for (Layer& L : layers) {
            if (!L.linear) continue;
            std::memset([L.convState contents], 0, [L.convState length]);
            std::memset([L.recState contents], 0, [L.recState length]);
        }
    }

    // Run tokens [0, n) of the prompt through the model on slot 0, in chunks. Fills slot 0 of
    // every cache and state, then copies it to the other slots.
    bool prefill(const int32_t* toks, int n) {
        if (n <= 0) return true;
        for (int c0 = 0; c0 < n; c0 += R) {
            const int T = std::min(R, n - c0);
            std::memcpy([tokBuf contents], toks + c0, (size_t)T * sizeof(int32_t));
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            encodeEmbed(e, T);
            for (int l = 0; l < dm.L; l++) encodeLayer(e, l, T, T, c0);
            [e endEncoding];
            if (!run(cb)) return false;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        for (Layer& L : layers) {
            if (L.linear) {
                const size_t cs = [L.convState length] / C, rs = [L.recState length] / C;
                for (int s = 1; s < C; s++) {
                    [bl copyFromBuffer:L.convState sourceOffset:0 toBuffer:L.convState destinationOffset:s * cs size:cs];
                    [bl copyFromBuffer:L.recState sourceOffset:0 toBuffer:L.recState destinationOffset:s * rs size:rs];
                }
            } else {
                const size_t perHead = (size_t)maxSeq * 256 * sizeof(uint16_t);
                const size_t bytes = (size_t)n * 256 * sizeof(uint16_t);
                for (int s = 1; s < C; s++)
                    for (int h = 0; h < dm.nKV; h++) {
                        const size_t src = (size_t)(0 * dm.nKV + h) * perHead;
                        const size_t dst = (size_t)(s * dm.nKV + h) * perHead;
                        [bl copyFromBuffer:L.kCache sourceOffset:src toBuffer:L.kCache destinationOffset:dst size:bytes];
                        [bl copyFromBuffer:L.vCache sourceOffset:src toBuffer:L.vCache destinationOffset:dst size:bytes];
                    }
            }
        }
        [bl endEncoding];
        return run(cb);
    }

    void seedRngs(std::vector<std::mt19937>& rngs) {
        if (hasFixedSeed) {
            for (size_t c = 0; c < rngs.size(); c++)
                antigravity::seedChannelRng(rngs[c], fixedSeed, generationCalls, (uint32_t)c);
        } else {
            std::random_device rd;
            for (size_t c = 0; c < rngs.size(); c++) rngs[c].seed(rd() + (uint32_t)c * 10007);
        }
        generationCalls++;
    }

    bool isEos(int32_t t) const { return std::find(dm.eos.begin(), dm.eos.end(), t) != dm.eos.end(); }
};

// ---------------------------------------------------------------------------------------------

bool Qwen35Engine::matches(const std::string& safetensors_path) {
    const std::string cfg = readFile(dirName(safetensors_path) + "/config.json");
    return cfg.find("\"model_type\": \"qwen3_5") != std::string::npos ||
           cfg.find("\"model_type\":\"qwen3_5") != std::string::npos;
}

Qwen35Engine::Qwen35Engine(int n_channels, int max_seq_len) : impl_(new Impl) {
    impl_->C = n_channels > 0 ? n_channels : 8;
    if (const char* sk = getenv("QWEN35_SKIP")) impl_->skip = sk;
    impl_->maxSeq = std::min(std::max(max_seq_len, 64), 4096);   // k_attn keeps scores in 4096 floats
    if (const char* seed = getenv("ANTIGRAVITY_SEED")) {
        uint64_t v = 0;
        if (antigravity::parseSeed(seed, v)) {
            impl_->hasFixedSeed = true;
            impl_->fixedSeed = v;
            std::cout << "[Qwen35Engine] ANTIGRAVITY_SEED=" << v << ": generation is reproducible for this process" << std::endl;
        } else {
            std::cerr << "[Qwen35Engine] ignoring ANTIGRAVITY_SEED=\"" << seed << "\": not a decimal integer" << std::endl;
        }
    }
}

Qwen35Engine::~Qwen35Engine() = default;

uint64_t Qwen35Engine::getAllocatedBytes() const { return impl_->allocated; }
int32_t Qwen35Engine::maxSequenceLength() const { return impl_->maxSeq; }

ITransformerEngine::DebugShape Qwen35Engine::debugShape() const {
    DebugShape s;
    s.n_channels = impl_->C;
    s.n_layers = impl_->dm.L;
    s.hidden_dim = impl_->dm.H;
    s.vocab_size = impl_->dm.V;
    return s;
}

bool Qwen35Engine::loadWeights(const std::string& path) {
    Impl& m = *impl_;
    std::string err;
    if (!parseDims(dirName(path), m.dm, err)) { std::cerr << "[Qwen35] " << err << std::endl; return false; }
    if (!m.initMetal()) return false;
    const Dims& d = m.dm;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::cerr << "[Qwen35] cannot open " << path << std::endl; return false; }
    struct stat st;
    fstat(fd, &st);
    const uint64_t fileSize = (uint64_t)st.st_size;
    void* map = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { std::cerr << "[Qwen35] mmap failed" << std::endl; return false; }
    const uint8_t* base = (const uint8_t*)map;
    struct Unmap { void* p; size_t n; ~Unmap() { munmap(p, n); } } unmap{map, (size_t)fileSize};

    uint64_t headerLen = 0;
    std::memcpy(&headerLen, base, 8);
    if (headerLen == 0 || headerLen > fileSize - 8) { std::cerr << "[Qwen35] bad safetensors header" << std::endl; return false; }
    const antigravity::HeaderParseResult parsed = antigravity::parseSafetensorsHeader(
        std::string((const char*)base + 8, (size_t)headerLen), 8 + headerLen, fileSize);
    if (!parsed.ok) { std::cerr << "[Qwen35] " << parsed.error << std::endl; return false; }
    const uint64_t dataStart = 8 + headerLen;

    bool ok = true;
    auto find = [&](const std::string& name, const char* dtype, int64_t elems) -> const antigravity::TensorEntry* {
        auto it = parsed.tensors.find(name);
        if (it == parsed.tensors.end()) { std::cerr << "[Qwen35] missing tensor " << name << std::endl; ok = false; return nullptr; }
        if (it->second.dtype != dtype || it->second.elementCount() != elems) {
            std::cerr << "[Qwen35] tensor " << name << " is " << it->second.dtype << " with "
                      << it->second.elementCount() << " elements; expected " << dtype << " with " << elems << std::endl;
            ok = false;
            return nullptr;
        }
        return &it->second;
    };
    // Raw copy of a bf16 tensor into a GPU buffer.
    auto bf16 = [&](const std::string& name, int64_t elems) -> id<MTLBuffer> {
        const auto* t = find(name, "BF16", elems);
        if (!t) return nil;
        m.allocated += (uint64_t)elems * 2;
        return [m.device newBufferWithBytes:base + dataStart + t->offset_start length:(NSUInteger)elems * 2
                                    options:MTLResourceStorageModeShared];
    };
    // A float32 buffer from an F32 or BF16 tensor.
    auto asFloat = [&](const std::string& name, int64_t elems, bool negExp) -> id<MTLBuffer> {
        auto it = parsed.tensors.find(name);
        if (it == parsed.tensors.end()) { std::cerr << "[Qwen35] missing tensor " << name << std::endl; ok = false; return nil; }
        const auto& t = it->second;
        if (t.elementCount() != elems) { std::cerr << "[Qwen35] tensor " << name << " has the wrong size" << std::endl; ok = false; return nil; }
        std::vector<float> v((size_t)elems);
        const uint8_t* p = base + dataStart + t.offset_start;
        for (int64_t i = 0; i < elems; i++) {
            if (t.dtype == "F32") { std::memcpy(&v[(size_t)i], p + i * 4, 4); }
            else if (t.dtype == "BF16") { uint16_t b; std::memcpy(&b, p + i * 2, 2); uint32_t u = (uint32_t)b << 16; std::memcpy(&v[(size_t)i], &u, 4); }
            else { std::cerr << "[Qwen35] " << name << " has dtype " << t.dtype << std::endl; ok = false; return nil; }
            if (negExp) v[(size_t)i] = -std::exp(v[(size_t)i]);
        }
        m.allocated += (uint64_t)elems * 4;
        return [m.device newBufferWithBytes:v.data() length:(NSUInteger)elems * 4 options:MTLResourceStorageModeShared];
    };

    const std::string P = "model.language_model.";
    m.embed = bf16(P + "embed_tokens.weight", (int64_t)d.V * d.H);
    m.finalNorm = bf16(P + "norm.weight", d.H);
    if (!d.tied && parsed.tensors.count("lm_head.weight")) m.lmHead = bf16("lm_head.weight", (int64_t)d.V * d.H);
    else m.lmHead = m.embed;

    m.layers.resize((size_t)d.L);
    for (int i = 0; i < d.L; i++) {
        Impl::Layer& L = m.layers[(size_t)i];
        L.linear = d.isLinear[(size_t)i];
        const std::string p = P + "layers." + std::to_string(i) + ".";
        L.inputNorm = bf16(p + "input_layernorm.weight", d.H);
        L.postNorm = bf16(p + "post_attention_layernorm.weight", d.H);
        L.gate = bf16(p + "mlp.gate_proj.weight", (int64_t)d.I * d.H);
        L.up = bf16(p + "mlp.up_proj.weight", (int64_t)d.I * d.H);
        L.down = bf16(p + "mlp.down_proj.weight", (int64_t)d.H * d.I);
        if (L.linear) {
            const std::string a = p + "linear_attn.";
            L.inQkv = bf16(a + "in_proj_qkv.weight", (int64_t)d.convDim * d.H);
            L.inZ = bf16(a + "in_proj_z.weight", (int64_t)d.valDim * d.H);
            L.inB = bf16(a + "in_proj_b.weight", (int64_t)d.linVH * d.H);
            L.inA = bf16(a + "in_proj_a.weight", (int64_t)d.linVH * d.H);
            L.outProj = bf16(a + "out_proj.weight", (int64_t)d.H * d.valDim);
            L.conv = bf16(a + "conv1d.weight", (int64_t)d.convDim * d.convK);
            L.aNeg = asFloat(a + "A_log", d.linVH, true);
            L.dtBias = asFloat(a + "dt_bias", d.linVH, false);
            L.gnorm = asFloat(a + "norm.weight", d.linVD, false);
            L.convState = m.newBuf((size_t)m.C * d.convDim * 3 * sizeof(float));
            L.recState = m.newBuf((size_t)m.C * d.linVH * 128 * 128 * sizeof(float));
        } else {
            const std::string a = p + "self_attn.";
            L.q = bf16(a + "q_proj.weight", (int64_t)d.nH * 512 * d.H);
            L.k = bf16(a + "k_proj.weight", (int64_t)d.nKV * 256 * d.H);
            L.v = bf16(a + "v_proj.weight", (int64_t)d.nKV * 256 * d.H);
            L.o = bf16(a + "o_proj.weight", (int64_t)d.H * d.nH * 256);
            L.qNorm = bf16(a + "q_norm.weight", 256);
            L.kNorm = bf16(a + "k_norm.weight", 256);
            const size_t cache = (size_t)m.C * d.nKV * m.maxSeq * 256 * sizeof(uint16_t);
            L.kCache = m.newBuf(cache);
            L.vCache = m.newBuf(cache);
        }
    }
    if (!ok) return false;

    m.R = 128;
    const size_t R = (size_t)std::max(m.R, m.C);
    m.R = (int)R;
    auto fb = [&](size_t cols) { return m.newBuf(R * cols * sizeof(float)); };
    m.tokBuf = m.newBuf(R * sizeof(int32_t));
    m.hid = fb(d.H); m.nrm = fb(d.H);
    m.qg = fb((size_t)d.nH * 512); m.qa = fb((size_t)d.nH * 256); m.gateA = fb((size_t)d.nH * 256);
    m.kp = fb((size_t)d.nKV * 256); m.vp = fb((size_t)d.nKV * 256); m.attnOut = fb((size_t)d.nH * 256);
    m.qkv = fb(d.convDim); m.conv = fb(d.convDim); m.zp = fb(d.valDim);
    m.bp = fb(d.linVH); m.ap = fb(d.linVH); m.dn = fb(d.valDim); m.dnn = fb(d.valDim);
    m.mg = fb(d.I); m.mu = fb(d.I);
    m.logits = m.newBuf((size_t)m.C * d.V * sizeof(float));
    m.capture = m.newBuf((size_t)(d.L + 1) * m.C * d.H * sizeof(float));

    m.loaded = true;
    weightsLoaded_ = true;
    std::cout << "[Qwen35Engine] " << d.L << " layers (" << std::count(d.isLinear.begin(), d.isLinear.end(), true)
              << " linear-attention), hidden " << d.H << ", vocab " << d.V << ", " << m.C << " channels, "
              << (m.allocated >> 20) << " MB on the GPU" << std::endl;
    return true;
}

GenerationResult Qwen35Engine::generate(const int32_t* prompt, int32_t prompt_len, int32_t max_new,
                                        float temperature, float top_p) {
    Impl& m = *impl_;
    const int C = m.C, V = m.dm.V;
    GenerationResult result;
    result.channel_tokens.resize((size_t)C);
    result.channel_logprobs.assign((size_t)C, 0.0f);
    if (!m.loaded) { std::cerr << "[generate] Weights not loaded!" << std::endl; return result; }
    if (prompt_len <= 0 || prompt_len + max_new > m.maxSeq) {
        std::cerr << "[generate] prompt_len " << prompt_len << " + max_new_tokens " << max_new
                  << " does not fit max_seq_len " << m.maxSeq << std::endl;
        return result;
    }
    for (int i = 0; i < prompt_len; i++)
        if (prompt[i] < 0 || prompt[i] >= V) { std::cerr << "[generate] token id out of range" << std::endl; return result; }

    std::vector<std::mt19937> rngs((size_t)C);
    m.seedRngs(rngs);
    std::vector<bool> active((size_t)C, true);
    const auto t0 = std::chrono::high_resolution_clock::now();
    bool ttftDone = false;
    double decodeMs = 0;
    int decodeSteps = 0;

    m.resetState();
    if (!m.prefill(prompt, prompt_len - 1)) return result;

    std::vector<int32_t> next((size_t)C), cur((size_t)C, prompt[prompt_len - 1]);
    std::vector<float> lp((size_t)C);
    for (int step = 0; step < max_new; step++) {
        @autoreleasepool {
            const auto ts = std::chrono::high_resolution_clock::now();
            std::memcpy([m.tokBuf contents], cur.data(), (size_t)C * sizeof(int32_t));
            const bool timing = getenv("QWEN35_TIMING") != nullptr;
            id<MTLCommandBuffer> cb = [m.queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            m.encodeEmbed(e, C);
            for (int l = 0; l < m.dm.L; l++) m.encodeLayer(e, l, C, 1, prompt_len - 1 + step);
            m.encodeHead(e, C);
            [e endEncoding];
            const auto t_enc = std::chrono::high_resolution_clock::now();
            if (!m.run(cb)) return result;
            const auto t_gpu = std::chrono::high_resolution_clock::now();

            std::vector<antigravity::SamplingStats> stats((size_t)C);
            antigravity::sampleChannels(
                (const float*)[m.logits contents], V, C, [&](int c) { return (bool)active[(size_t)c]; },
                temperature, top_p, rngs.data(), next.data(), lp.data(), stats.data(),
                [](int n, auto&& fn) {
                    auto* fnp = &fn;
                    dispatch_apply((size_t)n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                                   ^(size_t i) { (*fnp)((int)i); });
                });
            for (int c = 0; c < C; c++) {
                m.nonFinite += stats[(size_t)c].non_finite_logits;
                if (stats[(size_t)c].empty_distributions && m.emptyDist == 0)
                    std::cerr << "[sampleToken] every one of " << V << " logits is NaN or infinite; the forward pass produced no usable distribution" << std::endl;
                m.emptyDist += stats[(size_t)c].empty_distributions;
                if (!active[(size_t)c]) continue;
                result.channel_logprobs[(size_t)c] += lp[(size_t)c];
                result.channel_tokens[(size_t)c].push_back(next[(size_t)c]);
                result.total_tokens++;
                cur[(size_t)c] = next[(size_t)c];
                if (m.isEos(next[(size_t)c])) active[(size_t)c] = false;
            }
            const auto te = std::chrono::high_resolution_clock::now();
            if (timing && step > 0 && step % 8 == 0) {
                auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                std::cerr << "[Qwen35 timing] encode " << ms(ts, t_enc) << " ms, gpu " << ms(t_enc, t_gpu)
                          << " ms (gpu busy " << (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 << " ms), sample "
                          << ms(t_gpu, te) << " ms" << std::endl;
            }
            if (!ttftDone) { result.ttft_ms = std::chrono::duration<double, std::milli>(te - t0).count(); ttftDone = true; }
            else { decodeMs += std::chrono::duration<double, std::milli>(te - ts).count(); decodeSteps++; }
            if (std::none_of(active.begin(), active.end(), [](bool b) { return b; })) break;
        }
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.tpot_ms = decodeSteps ? decodeMs / decodeSteps : 0;
    result.best_channel = 0;
    result.best_score = result.channel_logprobs[0];
    for (int c = 1; c < C; c++)
        if (result.channel_logprobs[(size_t)c] > result.best_score) { result.best_score = result.channel_logprobs[(size_t)c]; result.best_channel = c; }

    std::cout << "[generate] Done: " << result.total_tokens << " tokens, TTFT=" << result.ttft_ms
              << "ms, TPOT=" << result.tpot_ms << "ms, Total=" << result.total_ms << "ms" << std::endl;
    if (m.emptyDist > 0)
        std::cerr << "[generate] WARNING: " << m.emptyDist << " sampling step(s) had no finite logit at all. Those tokens are not model output." << std::endl;
    if (m.nonFinite > 0)
        std::cerr << "[generate] WARNING: discarded " << m.nonFinite << " non-finite logits during this generation. The forward pass is producing NaN or Inf; the output above is not trustworthy." << std::endl;
    return result;
}

// Same input to every channel: prefill the prompt but its last token, then one decode step that
// copies each layer's output for every channel out of the residual stream.
bool Qwen35Engine::debugForward(const int32_t* prompt, int32_t prompt_len, std::vector<float>& hidden,
                                std::vector<float>& logits) {
    Impl& m = *impl_;
    if (!m.loaded || prompt_len <= 0 || prompt_len > m.maxSeq) return false;
    const int C = m.C, H = m.dm.H, V = m.dm.V, L = m.dm.L;
    m.resetState();
    if (!m.prefill(prompt, prompt_len - 1)) return false;
    std::vector<int32_t> cur((size_t)C, prompt[prompt_len - 1]);
    std::memcpy([m.tokBuf contents], cur.data(), (size_t)C * sizeof(int32_t));
    id<MTLCommandBuffer> cb = [m.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    m.encodeEmbed(e, C);
    [e endEncoding];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    const size_t block = (size_t)C * H * sizeof(float);
    [bl copyFromBuffer:m.hid sourceOffset:0 toBuffer:m.capture destinationOffset:0 size:block];
    [bl endEncoding];
    for (int l = 0; l < L; l++) {
        e = [cb computeCommandEncoder];
        m.encodeLayer(e, l, C, 1, prompt_len - 1);
        [e endEncoding];
        bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:m.hid sourceOffset:0 toBuffer:m.capture destinationOffset:(size_t)(l + 1) * block size:block];
        [bl endEncoding];
    }
    e = [cb computeCommandEncoder];
    m.encodeHead(e, C);
    [e endEncoding];
    if (!m.run(cb)) return false;
    const float* hp = (const float*)[m.capture contents];
    hidden.assign(hp, hp + (size_t)(L + 1) * C * H);
    const float* lp = (const float*)[m.logits contents];
    logits.assign(lp, lp + (size_t)C * V);
    return true;
}

GenerationResult Qwen35Engine::generateSpeculative(ITransformerEngine*, const int32_t*, int32_t, int32_t,
                                                   int32_t, float, float) {
    std::cerr << "[Qwen35Engine] speculative decoding is not implemented for this architecture" << std::endl;
    return GenerationResult();
}
GenerationResult Qwen35Engine::generateMultimodal(const int32_t*, int32_t, const float*, int32_t, int32_t,
                                                  float, float) {
    std::cerr << "[Qwen35Engine] the vision path is not implemented; text only" << std::endl;
    return GenerationResult();
}
MCTSResult Qwen35Engine::generateMCTS(const int32_t*, int32_t, const MCTSConfig&) {
    std::cerr << "[Qwen35Engine] chunk search is not implemented for this architecture" << std::endl;
    return MCTSResult();
}
