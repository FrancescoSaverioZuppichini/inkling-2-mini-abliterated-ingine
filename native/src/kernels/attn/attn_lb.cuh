// The low-bit full-attention readers' shared device code (attn_lb.cu: decode and row packing; attn_lb_prefill.cu:
// prefill): the formats as key and value codes over kv_format.h, the rotation, and the per-block step (K Q^T, online
// softmax, V^T P^T). Included only by those two files; everything lives in their unnamed namespace.
#pragma once

#include "kernels/attn/attn.h"
#include "kernels/attn/attn_common.cuh"
#include "kernels/attn/kv_format.h"
#include "kernels/prefill/common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace ingine::kernels::attn {
namespace {

using prefill::cp_async16;
using prefill::cp_commit;
using prefill::cp_wait;
using prefill::mma_f16;

constexpr int BLK = kvf::BLK, NW = NT / 32;

// x <- H D x (rotate) or D H x (unrotate) for a 128-row held 4 per lane (lane l: dims 4l .. 4l + 3); H is the
// unnormalised Walsh-Hadamard matrix (H H = 128 I). Every lane of the warp must call it.
__device__ __forceinline__ void fwht128(float (&x)[4], int lane) {
    float a = x[0] + x[1], b = x[0] - x[1], c = x[2] + x[3], d = x[2] - x[3];
    x[0] = a + c, x[1] = b + d, x[2] = a - c, x[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool up = lane & m;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, x[e], m);
            x[e] = up ? o - x[e] : x[e] + o;
        }
    }
}
__device__ __forceinline__ void rotate(float (&x)[4], int lane) {
#pragma unroll
    for (int e = 0; e < 4; ++e) x[e] *= kvf::sign_of(4 * lane + e);
    fwht128(x, lane);
}
__device__ __forceinline__ void unrotate(float (&x)[4], int lane) {
    fwht128(x, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) x[e] *= kvf::sign_of(4 * lane + e);
}

// One rotated row rounded to FP16 into a shared row (4 dims per lane); returns the sum of the rounded values.
__device__ __forceinline__ float store_f16_row(const float (&x)[4], __half* row, int lane) {
    float qs = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const __half h = __float2half_rn(x[e]);
        row[4 * lane + e] = h;
        qs += __half2float(h);
    }
    return wsum(qs);
}

__device__ __forceinline__ __half2 h2_of(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t u_of(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ uint32_t hmul(uint32_t a, uint32_t b) { return u_of(__hmul2(h2_of(a), h2_of(b))); }
// |s| capped at cap (NaN -> cap): value scales, so any byte pattern past the newest key decodes finite
__device__ __forceinline__ uint32_t vclamp(uint32_t s, uint32_t cap2) { return u_of(__hmin2(h2_of(s & 0x7fff7fffu), h2_of(cap2))); }

// One lane's 16-byte part of a block: from global memory (no L1 allocation, evict-first L2) or a shared image.
template <bool G>
__device__ __forceinline__ uint4 ld16(const char* p, std::uint64_t pol) {
    if constexpr (G) return ld_kv(p, pol);
    else return *reinterpret_cast<const uint4*>(p);
}
template <int N>
__device__ __forceinline__ void put(uint32_t* w, uint4 v) { w[N] = v.x, w[N + 1] = v.y, w[N + 2] = v.z, w[N + 3] = v.w; }

// (a & b) | c as one lop3 (written in C, ptxas emits two instructions).
__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// 8 int4 codes (nibble i = code n_i) -> 4 FP16 pairs (n0, n4), (n1, n5), (n2, n6), (n3, n7) holding n - 8.
// [Agent] The magic-number decode: OR a nibble into the mantissa of 1024 (0x6400) gives 1024 + n exactly; one
// subtraction (low nibbles) or one FMA by 1/16 (high nibbles, 4 bits up) leaves n - 8. 9 instructions per 8 codes.
// The levels are n - 7.5; the +0.5 is added once per score / output (Qsum, l) instead of per code.
__device__ __forceinline__ void dq4(uint32_t w, uint32_t (&h)[4]) {
    const uint32_t EX = 0x64006400u;
    const __half2 SUB = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));  // 1032
    const __half2 MUL = __halves2half2(__ushort_as_half(0x2c00), __ushort_as_half(0x2c00));  // 1/16
    const __half2 ADD = __halves2half2(__ushort_as_half(0xd480), __ushort_as_half(0xd480));  // -72
    uint32_t lo = lop3_and_or(w, 0x000f000fu, EX), hi = lop3_and_or(w, 0x00f000f0u, EX);
    __half2 r0 = __hsub2(*reinterpret_cast<__half2*>(&lo), SUB);
    __half2 r1 = __hfma2(*reinterpret_cast<__half2*>(&hi), MUL, ADD);
    w >>= 8;
    lo = lop3_and_or(w, 0x000f000fu, EX), hi = lop3_and_or(w, 0x00f000f0u, EX);
    __half2 r2 = __hsub2(*reinterpret_cast<__half2*>(&lo), SUB);
    __half2 r3 = __hfma2(*reinterpret_cast<__half2*>(&hi), MUL, ADD);
    h[0] = *reinterpret_cast<uint32_t*>(&r0), h[1] = *reinterpret_cast<uint32_t*>(&r1);
    h[2] = *reinterpret_cast<uint32_t*>(&r2), h[3] = *reinterpret_cast<uint32_t*>(&r3);
}

// 8 E2M1 codes in the same nibble order -> 4 FP16 pairs, each the code's value times 2^-14.
// [Agent] Sign to bit 15, the 3 magnitude bits to FP16 bits 11..9: exact for normal and subnormal codes, no arithmetic.
__device__ __forceinline__ void e2m1x8(uint32_t w, uint32_t (&h)[4]) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const uint32_t q = w << (12 - 4 * n);  // nibble n of each half at bits 15..12
        h[n] = (q & 0x80008000u) | ((q >> 3) & 0x0e000e00u);
    }
}
// Scale byte b of w -> the FP16 pair (s', s'): the byte is FP16 bits 14..7.
__device__ __forceinline__ uint32_t sbcast(uint32_t w, int b) { return __byte_perm(w, 0u, 0x0404u | (b << 4) | (b << 12)) >> 1; }

// Bytes 0, 1 (i8lo) or 2, 3 (i8hi) of a word of biased int8 codes (c + 128) -> the FP16 pair (c0, c1).
// [Agent] One PRMT puts each code byte under the exponent of 1024 (0x64 | byte = 1024 + c + 128), one HSUB2 of 1152
// leaves c exactly: 2 instructions per 2 values.
__device__ __forceinline__ uint32_t i8lo(uint32_t w) { return u_of(__hsub2(h2_of(__byte_perm(w, 0x64646464u, 0x4140)), h2_of(0x64806480u))); }
__device__ __forceinline__ uint32_t i8hi(uint32_t w) { return u_of(__hsub2(h2_of(__byte_perm(w, 0x64646464u, 0x4342)), h2_of(0x64806480u))); }

// ---- 4-bit packing shared by int4 and fp4 (a warp holds one row, lane l = dims 4 l .. 4 l + 3; codes = its 128
// nibble codes in shared scratch): keys as whole words, values as bytes (kv_format.h int4 order).
__device__ __forceinline__ void pack_k4(const unsigned char* codes, char* img, int j, int lane) {
    if (lane < 16) {
        const int kk = lane >> 2, t4 = lane & 3;
        uint32_t w = 0;
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) w |= uint32_t(codes[kvf::int4::k_dim(kk, t4, nib)]) << (4 * nib);
        reinterpret_cast<uint32_t*>(img)[kvf::int4::k_word(j, kk, t4)] = w;
    }
}
__device__ __forceinline__ void pack_v4(const unsigned char* codes, char* img, int j, int lane) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int pr = lane + 32 * i, d = 16 * (pr >> 3) + (pr & 7);
        img[kvf::int4::v_byte(j, d)] = static_cast<char>(codes[d] | (codes[d + 8] << 4));
    }
}
// Copy key slot js of an image to slot jd of a cache block: whole words / bytes per key, so a concurrent reader of the
// block's other keys never sees a torn value.
__device__ __forceinline__ void copy_k4(const char* img, int js, char* blk, int jd, int lane) {
    if (lane < 16) {
        const int kk = lane >> 2, t4 = lane & 3;
        reinterpret_cast<uint32_t*>(blk)[kvf::int4::k_word(jd, kk, t4)] =
            reinterpret_cast<const uint32_t*>(img)[kvf::int4::k_word(js, kk, t4)];
    }
}
__device__ __forceinline__ void copy_v4(const char* img, int js, char* blk, int jd, int lane) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int pr = lane + 32 * i, d = 16 * (pr >> 3) + (pr & 7);
        blk[kvf::int4::v_byte(jd, d)] = img[kvf::int4::v_byte(js, d)];
    }
}

// One rotated row into int8 codes and its FP16 scale (key or value layout).
template <bool KEY>
__device__ __forceinline__ void enc8(const float (&x)[4], char* img, int j, int lane) {
    float a = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const std::uint16_t s = kvf::int8::scale_of(a);
    const float sf = kvf::h2f(s);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int d = 4 * lane + e;
        img[KEY ? kvf::int8::k_byte(j, d) : kvf::int8::v_byte(j, d)] = static_cast<char>(kvf::int8::code_of(x[e], sf) + 128);
    }
    if (lane == 0) *reinterpret_cast<std::uint16_t*>(img + (KEY ? kvf::int8::k_scale_off(j) : kvf::int8::v_scale_off(j))) = s;
}
// One rotated row into E2M1 codes (scratch) and one scale byte per 16 dims (4 lanes; key or value layout).
template <bool KEY>
__device__ __forceinline__ void encf(const float (&x)[4], char* img, int j, unsigned char* codes, int lane) {
    float a = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
    const std::uint8_t b = kvf::fp4::scale_of(a);
    const float sf = kvf::fp4::scale_val(b);
#pragma unroll
    for (int e = 0; e < 4; ++e) codes[4 * lane + e] = static_cast<unsigned char>(kvf::fp4::code_of(x[e], sf));
    if ((lane & 3) == 0) img[KEY ? kvf::fp4::k_scale_off(j, lane >> 2) : kvf::fp4::v_scale_off(j, lane >> 2)] = static_cast<char>(b);
    __syncwarp();
}

// ---- key codes. A lane's share of a block's keys: load_k, the A operands of k-steps 2 kk and 2 kk + 1 (rows = keys
// gid, gid + 8; k_frags), the per-key score factors of keys gid, gid + 8 (k_scale); encode_k writes one rotated row
// (norm = |k| before the rotation) into slot j of a block image, copy_k moves a slot from an image into the cache.
// KOFF: the per-code level offset, added once per score through the query sum.

// int4 keys: codes of the unit direction, one FP16 least-squares scale.
struct K4 {
    static constexpr int KB = kvf::int4::K_BYTES;
    static constexpr float KOFF = 0.5f;  // decoded n - 8 + 0.5 = the level n - 7.5
    uint32_t k[8], ks;
    template <bool G>
    __device__ __forceinline__ void load_k(const char* kb, int lane, std::uint64_t pol) {
        put<0>(k, ld16<G>(kb + 32 * lane, pol));
        put<4>(k, ld16<G>(kb + 32 * lane + 16, pol));
        ks = *reinterpret_cast<const uint32_t*>(kb + BLK * D / 2 + 4 * (lane >> 2));
    }
    __device__ __forceinline__ void k_frags(int kk, uint32_t (&a0)[4], uint32_t (&a1)[4]) const {
        uint32_t g[4], h[4];
        dq4(k[2 * kk], g);      // key gid
        dq4(k[2 * kk + 1], h);  // key gid + 8
        a0[0] = g[0], a0[1] = h[0], a0[2] = g[1], a0[3] = h[1];
        a1[0] = g[2], a1[1] = h[2], a1[2] = g[3], a1[3] = h[3];
    }
    __device__ __forceinline__ float2 k_scale() const { return __half22float2(h2_of(ks)); }
    // codes of x / |k| (RMS 1, since |H D k| = sqrt(128) |k|) and the least-squares scale
    __device__ static void encode_k(const float (&x)[4], float norm, char* img, int j, unsigned char* codes, int lane) {
        const float inv = norm > 0.f ? 1.f / norm : 0.f;
        float num = 0.f, den = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int n = kvf::int4::code_of(x[e] * inv);
            codes[4 * lane + e] = static_cast<unsigned char>(n);
            const float c = float(n) - 7.5f;
            num = fmaf(x[e], c, num), den = fmaf(c, c, den);
        }
        __syncwarp();
        num = wsum(num), den = wsum(den);
        pack_k4(codes, img, j, lane);
        if (lane == 16) *reinterpret_cast<__half*>(img + kvf::int4::k_scale_off(j)) = __float2half_rn(num / den);
        __syncwarp();
    }
    __device__ static void copy_k(const char* img, int js, char* blk, int jd, int lane) {
        copy_k4(img, js, blk, jd, lane);
        if (lane == 16)
            *reinterpret_cast<__half*>(blk + kvf::int4::k_scale_off(jd)) =
                *reinterpret_cast<const __half*>(img + kvf::int4::k_scale_off(js));
    }
};

// int8 keys: 64 code bytes and the scales of keys gid, gid + 8 per lane; the operand is the code, the scale is a
// per-key score factor (FP32, after the mma).
struct K8 {
    static constexpr int KB = kvf::int8::BYTES;
    static constexpr float KOFF = 0.f;
    uint32_t k[16], ks;
    template <bool G>
    __device__ __forceinline__ void load_k(const char* kb, int lane, std::uint64_t pol) {
        put<0>(k, ld16<G>(kb + 64 * lane, pol));
        put<4>(k, ld16<G>(kb + 64 * lane + 16, pol));
        put<8>(k, ld16<G>(kb + 64 * lane + 32, pol));
        put<12>(k, ld16<G>(kb + 64 * lane + 48, pol));
        ks = *reinterpret_cast<const uint32_t*>(kb + kvf::int8::k_scale_off(lane >> 2));
    }
    __device__ __forceinline__ void k_frags(int kk, uint32_t (&a0)[4], uint32_t (&a1)[4]) const {
        a0[0] = i8lo(k[4 * kk]), a0[1] = i8lo(k[4 * kk + 1]), a0[2] = i8hi(k[4 * kk]), a0[3] = i8hi(k[4 * kk + 1]);
        a1[0] = i8lo(k[4 * kk + 2]), a1[1] = i8lo(k[4 * kk + 3]), a1[2] = i8hi(k[4 * kk + 2]), a1[3] = i8hi(k[4 * kk + 3]);
    }
    __device__ __forceinline__ float2 k_scale() const { return __half22float2(h2_of(ks)); }
    __device__ static void encode_k(const float (&x)[4], float, char* img, int j, unsigned char*, int lane) {
        enc8<true>(x, img, j, lane);
    }
    __device__ static void copy_k(const char* img, int js, char* blk, int jd, int lane) {
        const int d = 16 * (lane >> 2) + 2 * (lane & 3);  // the word holding dims d, d + 1, d + 8, d + 9
        *reinterpret_cast<uint32_t*>(blk + kvf::int8::k_byte(jd, d)) =
            *reinterpret_cast<const uint32_t*>(img + kvf::int8::k_byte(js, d));
        if (lane == 0)
            *reinterpret_cast<std::uint16_t*>(blk + kvf::int8::k_scale_off(jd)) =
                *reinterpret_cast<const std::uint16_t*>(img + kvf::int8::k_scale_off(js));
    }
};

// fp4 keys: E2M1 codes with one scale byte per 16 dims multiplied into the FP16 operand pairs (exact).
struct KF {
    static constexpr int KB = kvf::fp4::BYTES;
    static constexpr float KOFF = 0.f;
    uint32_t k[8], ks[4];
    template <bool G>
    __device__ __forceinline__ void load_k(const char* kb, int lane, std::uint64_t pol) {
        put<0>(k, ld16<G>(kb + 32 * lane, pol));
        put<4>(k, ld16<G>(kb + 32 * lane + 16, pol));
        put<0>(ks, ld16<G>(kb + kvf::fp4::k_scale_off(lane >> 2, 0), pol));
    }
    // ks[0..1]: key gid's scale bytes of groups (= k-steps) 0..7; ks[2..3]: key gid + 8's
    __device__ __forceinline__ void k_frags(int kk, uint32_t (&a0)[4], uint32_t (&a1)[4]) const {
        uint32_t g[4], h[4];
        e2m1x8(k[2 * kk], g);
        e2m1x8(k[2 * kk + 1], h);
        const int w = kk >> 1, b = 2 * (kk & 1);
        const uint32_t sg0 = sbcast(ks[w], b), sg1 = sbcast(ks[w], b + 1), sh0 = sbcast(ks[2 + w], b), sh1 = sbcast(ks[2 + w], b + 1);
        a0[0] = hmul(g[0], sg0), a0[1] = hmul(h[0], sh0), a0[2] = hmul(g[1], sg0), a0[3] = hmul(h[1], sh0);
        a1[0] = hmul(g[2], sg1), a1[1] = hmul(h[2], sh1), a1[2] = hmul(g[3], sg1), a1[3] = hmul(h[3], sh1);
    }
    __device__ __forceinline__ float2 k_scale() const { return make_float2(kvf::fp4::UNIT, kvf::fp4::UNIT); }
    __device__ static void encode_k(const float (&x)[4], float, char* img, int j, unsigned char* codes, int lane) {
        encf<true>(x, img, j, codes, lane);
        pack_k4(codes, img, j, lane);
        __syncwarp();
    }
    __device__ static void copy_k(const char* img, int js, char* blk, int jd, int lane) {
        copy_k4(img, js, blk, jd, lane);
        if (lane < 2)
            reinterpret_cast<uint32_t*>(blk + kvf::fp4::k_scale_off(jd, 0))[lane] =
                reinterpret_cast<const uint32_t*>(img + kvf::fp4::k_scale_off(js, 0))[lane];
    }
};

// ---- value codes: load_v, the A operand of dim tile dt (rows = dims 16 dt + gid, + 8; k = keys; v_frag), encode_v,
// copy_v. VOFF: the per-code level offset, added once per output through l; OUT: the factor from the accumulated
// (rotated) operand units to the model's value after the inverse rotation.

// int4 values: scale-free codes of the unit-RMS rotated row.
struct V4 {
    static constexpr int VB = kvf::int4::V_BYTES;
    static constexpr float VOFF = 0.5f;
    static constexpr float OUT = kvf::int4::STEP * RSQRT_D;  // value level -> unit-RMS value (/ sqrt(128))
    uint32_t v[8];
    template <bool G>
    __device__ __forceinline__ void load_v(const char* vb, int lane, std::uint64_t pol) {
        put<0>(v, ld16<G>(vb + 32 * lane, pol));
        put<4>(v, ld16<G>(vb + 32 * lane + 16, pol));
    }
    __device__ __forceinline__ void v_frag(int dt, uint32_t (&a)[4]) const { dq4(v[dt], a); }
    // codes of x / sqrt(128), no scale
    __device__ static void encode_v(const float (&x)[4], char* img, int j, unsigned char* codes, int lane) {
#pragma unroll
        for (int e = 0; e < 4; ++e) codes[4 * lane + e] = static_cast<unsigned char>(kvf::int4::code_of(x[e] * RSQRT_D));
        __syncwarp();
        pack_v4(codes, img, j, lane);
        __syncwarp();
    }
    __device__ static void copy_v(const char* img, int js, char* blk, int jd, int lane) { copy_v4(img, js, blk, jd, lane); }
};

// int8 values: the per-row scale multiplied into the operand pairs (one HMUL2; the host rounds the same way).
struct V8 {
    static constexpr int VB = kvf::int8::BYTES;
    static constexpr float VOFF = 0.f;
    static constexpr float OUT = 1.f / 128.f;  // D H (H D v) = 128 v
    uint32_t v[16], vs[2];
    template <bool G>
    __device__ __forceinline__ void load_v(const char* vb, int lane, std::uint64_t pol) {
        put<0>(v, ld16<G>(vb + 64 * lane, pol));
        put<4>(v, ld16<G>(vb + 64 * lane + 16, pol));
        put<8>(v, ld16<G>(vb + 64 * lane + 32, pol));
        put<12>(v, ld16<G>(vb + 64 * lane + 48, pol));
        const uint2 s = *reinterpret_cast<const uint2*>(vb + kvf::int8::v_scale_off(2 * (lane & 3)));
        vs[0] = s.x, vs[1] = s.y;  // keys (2 t4, 2 t4 + 1), (2 t4 + 8, 2 t4 + 9)
    }
    __device__ __forceinline__ void v_frag(int dt, uint32_t (&a)[4]) const {
        constexpr uint32_t CAP = kvf::int8::V_SCALE_CAP * 0x10001u;
        const uint32_t s0 = vclamp(vs[0], CAP), s1 = vclamp(vs[1], CAP);
        a[0] = hmul(i8lo(v[2 * dt]), s0), a[1] = hmul(i8lo(v[2 * dt + 1]), s0);
        a[2] = hmul(i8hi(v[2 * dt]), s1), a[3] = hmul(i8hi(v[2 * dt + 1]), s1);
    }
    __device__ static void encode_v(const float (&x)[4], char* img, int j, unsigned char*, int lane) {
        enc8<false>(x, img, j, lane);
    }
    __device__ static void copy_v(const char* img, int js, char* blk, int jd, int lane) {
#pragma unroll
        for (int e = 0; e < 4; ++e) blk[kvf::int8::v_byte(jd, 4 * lane + e)] = img[kvf::int8::v_byte(js, 4 * lane + e)];
        if (lane == 0)
            *reinterpret_cast<std::uint16_t*>(blk + kvf::int8::v_scale_off(jd)) =
                *reinterpret_cast<const std::uint16_t*>(img + kvf::int8::v_scale_off(js));
    }
};

// fp4 values: one scale byte per key and dim tile multiplied into the operand pairs (exact).
struct VF {
    static constexpr int VB = kvf::fp4::BYTES;
    static constexpr float VOFF = 0.f;
    static constexpr float OUT = kvf::fp4::UNIT / 128.f;
    uint32_t v[8], vs[8];
    template <bool G>
    __device__ __forceinline__ void load_v(const char* vb, int lane, std::uint64_t pol) {
        put<0>(v, ld16<G>(vb + 32 * lane, pol));
        put<4>(v, ld16<G>(vb + 32 * lane + 16, pol));
        put<0>(vs, ld16<G>(vb + kvf::fp4::v_scale_off(2 * (lane & 3), 0), pol));
        put<4>(vs, ld16<G>(vb + kvf::fp4::v_scale_off(2 * (lane & 3), 4), pol));
    }
    // vs[dt] = the scale bytes of keys 2 t4, 2 t4 + 1, 2 t4 + 8, 2 t4 + 9 for dim tile dt
    __device__ __forceinline__ void v_frag(int dt, uint32_t (&a)[4]) const {
        constexpr uint32_t CAP = kvf::fp4::V_SCALE_CAP * 0x10001u;
        uint32_t h[4];
        e2m1x8(v[dt], h);
        const uint32_t s0 = vclamp(__byte_perm(vs[dt], 0u, 0x1404u) >> 1, CAP), s1 = vclamp(__byte_perm(vs[dt], 0u, 0x3424u) >> 1, CAP);
        a[0] = hmul(h[0], s0), a[1] = hmul(h[1], s0), a[2] = hmul(h[2], s1), a[3] = hmul(h[3], s1);
    }
    __device__ static void encode_v(const float (&x)[4], char* img, int j, unsigned char* codes, int lane) {
        encf<false>(x, img, j, codes, lane);
        pack_v4(codes, img, j, lane);
        __syncwarp();
    }
    __device__ static void copy_v(const char* img, int js, char* blk, int jd, int lane) {
        copy_v4(img, js, blk, jd, lane);
        if (lane < 8) blk[kvf::fp4::v_scale_off(jd, lane)] = img[kvf::fp4::v_scale_off(js, lane)];
    }
};

// A format = a key code + a value code, both on rotated rows: q'' . k'' = 128 q . k.
template <class K, class V>
struct Fmt : K, V {
    static_assert(K::KB % 16 == 0 && V::VB % 16 == 0, "blocks are read with 16-byte loads");
    static constexpr float SCORE = QSCALE / 128.f;
    template <bool G>
    __device__ __forceinline__ void load(const char* kb, const char* vb, int lane, std::uint64_t pol) {
        K::template load_k<G>(kb, lane, pol);
        V::template load_v<G>(vb, lane, pol);
    }
};
using Int4 = Fmt<K4, V4>;
using Int8 = Fmt<K8, V8>;
using K8V4 = Fmt<K8, V4>;
using Fp4 = Fmt<KF, VF>;

// The format type of a low-bit KvFormat, passed to fn as std::type_identity<F>.
template <class Fn>
void with_format(KvFormat f, Fn&& fn) {
    switch (f) {
        case KvFormat::int4: return fn(std::type_identity<Int4>{});
        case KvFormat::int8: return fn(std::type_identity<Int8>{});
        case KvFormat::k8v4: return fn(std::type_identity<K8V4>{});
        case KvFormat::fp4: return fn(std::type_identity<Fp4>{});
        default: require(false, "not a low-bit KV format");
    }
}

// Online-softmax state of one warp's NTN 8-row tiles: lane (gid, t4) holds rows 8 nn + 2 t4 (+1) of every tile and, for
// them, all 128 dims (dim tile dt: dims 16 dt + gid, + 8) of the unnormalised output. qsum: the format's key level
// offset times the row's query sum (0 without an offset).
template <int NTN>
struct Soft {
    float qsum[NTN][2], m[NTN][2], l[NTN][2], acc[8][NTN][4];
    template <class QS>
    __device__ __forceinline__ void init(QS&& qsum_of, int lane) {
#pragma unroll
        for (int nn = 0; nn < NTN; ++nn) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                qsum[nn][h] = qsum_of(8 * nn + 2 * (lane & 3) + h);
                m[nn][h] = NEG_INIT, l[nn][h] = 0.f;
            }
#pragma unroll
            for (int dt = 0; dt < 8; ++dt)
#pragma unroll
                for (int v = 0; v < 4; ++v) acc[dt][nn][v] = 0.f;
        }
    }
};

// One cache block of 16 keys against this warp's rows. qfrag(nn, ks, h): the rotated query's B fragments. fix(key,
// row, s): the scaled log2 score of (key 0..15, local row) with bias added, or -inf where masked. P goes through the
// warp's own 8-row x 16-key FP16 tiles (pw, row stride pws): no block barrier.
template <class F, int NTN, class QF, class FIX>
__device__ __forceinline__ void attend_block(const F& r, QF&& qfrag, FIX&& fix, Soft<NTN>& st, __half* pw, int pws,
                                             int lane) {
    const int gid = lane >> 2, t4 = lane & 3;
    float s[NTN][4];
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn) s[nn][0] = s[nn][1] = s[nn][2] = s[nn][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        uint32_t a0[4], a1[4];
        r.k_frags(kk, a0, a1);
#pragma unroll
        for (int nn = 0; nn < NTN; ++nn) {
            mma_f16(s[nn], a0, qfrag(nn, 2 * kk, 0), qfrag(nn, 2 * kk, 1));
            mma_f16(s[nn], a1, qfrag(nn, 2 * kk + 1, 0), qfrag(nn, 2 * kk + 1, 1));
        }
    }
    const float2 ks = r.k_scale();
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int key = gid + (v >> 1) * 8, row = 8 * nn + 2 * t4 + (v & 1);
            float sc = s[nn][v];
            if constexpr (F::KOFF != 0.f) sc += st.qsum[nn][v & 1];
            s[nn][v] = fix(key, row, sc * ((v >> 1) ? ks.y : ks.x) * F::SCORE);
        }
    bool grow = false;
    float fac[NTN][2];
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float mx = fmaxf(s[nn][h], s[nn][h + 2]);
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mn = fmaxf(st.m[nn][h], mx);
            fac[nn][h] = exp2f(st.m[nn][h] - mn);
            grow |= mn > st.m[nn][h];
            st.m[nn][h] = mn;
            st.l[nn][h] *= fac[nn][h];
        }
    if (__any_sync(0xffffffffu, grow)) {
#pragma unroll
        for (int dt = 0; dt < 8; ++dt)
#pragma unroll
            for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
                for (int v = 0; v < 4; ++v) st.acc[dt][nn][v] *= fac[nn][v & 1];
    }
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int key = gid + (v >> 1) * 8, row = 8 * nn + 2 * t4 + (v & 1);
            const __half h = __float2half_rn(exp2f(s[nn][v] - st.m[nn][v & 1]));
            st.l[nn][v & 1] += __half2float(h);
            pw[row * pws + key] = h;
        }
    __syncwarp();
    uint32_t pb[NTN][2];
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn) {
        pb[nn][0] = *reinterpret_cast<const uint32_t*>(pw + (8 * nn + gid) * pws + 2 * t4);
        pb[nn][1] = *reinterpret_cast<const uint32_t*>(pw + (8 * nn + gid) * pws + 2 * t4 + 8);
    }
    __syncwarp();
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        uint32_t a[4];
        r.v_frag(dt, a);
#pragma unroll
        for (int nn = 0; nn < NTN; ++nn) mma_f16(st.acc[dt][nn], a, pb[nn][0], pb[nn][1]);
    }
}

// The output of one row (4 dims per lane, rotated, normalised): unrotate, scale, BF16, times the sigmoid gate.
template <class F, class Gate>
__device__ __forceinline__ void write_row(float (&x)[4], __nv_bfloat16* out, Gate&& gate_logit, int lane) {
    unrotate(x, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int dim = 4 * lane + e;
        out[dim] = __float2bfloat16_rn(bf16r(x[e] * F::OUT) * (1.f / (1.f + expf(-gate_logit(dim)))));
    }
}

// The query sum times the key level offset (formats with an offset only).
template <class F>
__device__ __forceinline__ float koff_of(float qsum) {
    if constexpr (F::KOFF != 0.f) return qsum * F::KOFF;
    else return 0.f;
}
// An accumulator plus the value level offset times l (formats with an offset only).
template <class F>
__device__ __forceinline__ float voff_add(float acc, float l) {
    if constexpr (F::VOFF != 0.f) return acc + F::VOFF * l;
    else return acc;
}

}  // namespace
}  // namespace ingine::kernels::attn
