// Full-attention decode (M = 1..8 tokens) and prefill over low-bit KV caches (int4, int8, k8v4, fp4). Contract:
// native/KERNELS.md, area `attn`.
//
// Design
// ------
// * Same arithmetic as attn.cu's full-layer path (attn_common.cuh: q / rel post-processing, dynamic conv, key clip,
//   value norm; relative bias over the last 1024 positions, sink logit, sigmoid gate; the call's own keys attended
//   exactly as the cache will hold them), with the cache in 16-position blocks per KV head written in mma operand
//   order (kv_format.h).
// * A format is a key code and a value code (K4 / K8 / KF, V4 / V8 / VF below). Every format codes rotated rows, so
//   the kernels are one template over the format: q is rotated once per row, the output unrotated once per row.
// * Grid (cs, 8): cs blocks per KV head, 8 warps each. Warp w of block r owns cache blocks r * 8 + w + n * 8 cs: each
//   lane loads its share of the block's codes and scales with 16-byte loads (one warp reads the block contiguously),
//   the next block's loads in flight while this one is consumed. No barrier inside the loop.
// * Per block: S^T (16 keys x 8-row tiles) = K Q^T with mma m16n8k16 f16 (keys decoded to FP16 in registers, the
//   rotated query in FP16), scaled per key, bias / mask, the warp's online softmax (row statistics live in the lanes
//   that hold the rows' accumulators), P to a 16 x 8 tile per warp in shared memory (FP16), then O^T += V^T P^T with
//   the values decoded in registers. Every warp holds all 128 (rotated) dims of its rows.
// * Warps merge in shared memory in a fixed tree order, blocks with attn.cu's last-block merge (counter per head, its
//   workspace layout), then the merging block undoes the rotation per row, scales, rounds to BF16 and gates.
// * Launched like attn.cu (programmatic dependent launch): pos and the weights are read before griddepcontrol.wait,
//   everything the previous kernel writes after it.
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

struct Params {
    const float* proj;
    long long cap;  // cache rows
    const float* q_scale;
    const float* r_scale;
    const float* rel_proj;
    const float* k_conv;  // k_static [4, 1024] by lag
    const float* v_conv;
    float* conv_k;
    float* conv_v;
    char* k_cache;
    char* v_cache;
    const long long* pos;
    int* counter;
    float* part;
    __nv_bfloat16* out;
};

// shared memory (bytes)
template <int M, class F>
struct Smem {
    static constexpr int ROWS = GQA * M, NRP = (ROWS + 7) / 8 * 8, NTN = NRP / 8;
    // staged inputs, dead after the prologue; aliased by the warp merge buffers
    static constexpr int Q = 0;                              // f32 [M][3][D]
    static constexpr int QS = Q + 4 * M * GQA * D;           // f32 [3][D]
    static constexpr int REL = QS + 4 * GQA * D;             // f32 [M][3][16]
    static constexpr int RS = REL + 4 * M * GQA * DREL;      // f32 [3][16]
    static constexpr int ST = RS + 4 * GQA * DREL;           // f32 [3][2][D]
    static constexpr int X = ST + 4 * 3 * 2 * D;             // f32 [M][2][D]
    static constexpr int W = X + 4 * M * 2 * D;              // f32 [2][4][D]
    static constexpr int DYN = W + 4 * 2 * 4 * D;            // f32 [M][2][4]
    static constexpr int NKF = DYN + 4 * M * 2 * 4;          // f32 [2][M][D]
    static constexpr int STAGE_END = NKF + 4 * 2 * M * D;
    static constexpr int RED = 0;                            // f32 [4][NRP][D] (warp merge), then [ROWS][D] (final)
    static constexpr int RED_END = 4 * 4 * NRP * D;
    static constexpr int U_END = (std::max(STAGE_END, RED_END) + 15) / 16 * 16;
    static constexpr int QH = U_END;                         // f16 [NRP][D + 8] rotated query
    static constexpr int QHS = D + 8;
    static constexpr int RELD = QH + 2 * NRP * QHS;          // f32 [NRP][16]
    static constexpr int QSUM = RELD + 4 * NRP * DREL;       // f32 [NRP]
    static constexpr int SINK = QSUM + 4 * NRP;              // f32 [ROWS]
    static constexpr int ML = (SINK + 4 * ROWS + 15) / 16 * 16;  // f32x2 [NW][NRP]
    static constexpr int NST = ML + 8 * NW * NRP;            // f32 [3][2][D] advanced conv state
    static constexpr int KIMG = NST + 4 * 3 * 2 * D;         // the call's keys as one block image
    static constexpr int VIMG = KIMG + F::KB;
    static constexpr int CODES = VIMG + F::VB;               // u8 [NW][D] encode scratch
    static constexpr int PW = CODES + NW * D;                // f16 [NW][NRP][24] P tiles
    static constexpr int PWS = 24;
    static constexpr int END = PW + 2 * NW * NRP * PWS;
    static constexpr size_t BYTES = size_t(END);
};

// [Agent] Two blocks per SM cap a thread at 128 registers. At M <= 2 that holds two blocks' codes of up to 100 bytes per
// lane (int4, k8v4); int8 (140) and fp4 (112) spilled inside the block loop (cuobjdump: 56 / 8 bytes of stack), so they
// run one block per SM, which keeps the same bytes in flight per SM as int4 does with two.
template <int M, class F>
constexpr int min_blocks() { return M > 2 || sizeof(F) > 100 ? 1 : 2; }

template <int M, class F>
__global__ void __launch_bounds__(NT, min_blocks<M, F>()) lb_kernel(const Params p) {
    using L = Smem<M, F>;
    constexpr int ROWS = L::ROWS, NRP = L::NRP, NTN = L::NTN;
    extern __shared__ __align__(16) char smc[];
    auto F32 = [&](int off) { return reinterpret_cast<float*>(smc + off); };

    const int g = blockIdx.y, rank = blockIdx.x, cs = gridDim.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    const bool owner = rank == 0;
    const std::uint64_t pol = l2::policy<KV_L2>();

    // ---- stage the inputs (as attn.cu's full-layer path); weights and pos before the PDL wait
    for (int i = tid; i < GQA * D / 4; i += NT) cp16(F32(L::QS) + 4 * i, p.q_scale + g * GQA * D + 4 * i);
    for (int i = tid; i < GQA * DREL / 4; i += NT) cp16(F32(L::RS) + 4 * i, p.r_scale + g * GQA * DREL + 4 * i);
    if (owner)
        for (int i = tid; i < 2 * D; i += NT) {  // [kv][lag][D]
            const int kv = i / D, q = i % D;
            cp16(F32(L::W) + 4 * i, (kv ? p.v_conv : p.k_conv) + (q / (D / 4)) * CONV_C + g * D + 4 * (q % (D / 4)));
        }
    const int p0 = int(ld_s64(p.pos));
    if (p0 + M > p.cap) return;
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    for (int i = tid; i < M * GQA * D / 4; i += NT)
        cp16(F32(L::Q) + 4 * i, p.proj + size_t(i / (GQA * D / 4)) * R_FULL + g * GQA * D + 4 * (i % (GQA * D / 4)));
    for (int i = tid; i < M * GQA * DREL / 4; i += NT)
        cp16(F32(L::REL) + 4 * i, p.proj + size_t(i / (GQA * DREL / 4)) * R_FULL + OFF_REL + g * GQA * DREL +
                                      4 * (i % (GQA * DREL / 4)));
    for (int i = tid; i < ROWS; i += NT) cp4(F32(L::SINK) + i, p.proj + size_t(i / GQA) * R_FULL + OFF_SINK + g * GQA + i % GQA);
    if (owner) {
        for (int i = tid; i < 3 * 2 * D / 4; i += NT) {
            const int j = i / (2 * D / 4), kv = (i / (D / 4)) % 2, q = i % (D / 4);
            cp16(F32(L::ST) + 4 * i, (kv ? p.conv_v : p.conv_k) + j * CONV_C + g * D + 4 * q);
        }
        for (int i = tid; i < M * 2 * D / 4; i += NT) {
            const int m = i / (2 * D / 4), kv = (i / (D / 4)) % 2, q = i % (D / 4);
            cp16(F32(L::X) + 4 * i, p.proj + size_t(m) * R_FULL + (kv ? OFF_V : OFF_K) + g * D + 4 * q);
        }
        for (int i = tid; i < M * 2; i += NT)
            cp16(F32(L::DYN) + 4 * i, p.proj + size_t(i / 2) * R_FULL + (i % 2 ? OFF_VDYN : OFF_KDYN) + g * 4);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();

    // first cache block of this warp in flight while the prologue runs
    const int nb = (p0 + BLK - 1) / BLK, stride = cs * NW;
    int b = rank * NW + warp;
    F cur, nxt;
    auto blk_k = [&](int bi) { return p.k_cache + (size_t(bi) * HKV + g) * F::KB; };
    auto blk_v = [&](int bi) { return p.v_cache + (size_t(bi) * HKV + g) * F::VB; };
    if (b < nb) cur.template load<true>(blk_k(b), blk_v(b), lane, pol);

    // ---- q (attn_common's post-processing, then rotated: H D q in FP16) and rel, one warp per row
    auto tau_of = [&](int t) { return 1.f + 0.1f * logf(fmaxf(float(p0 + t + 1) * (1.f / 8192.f), 1.f)); };
    __half* const qh = reinterpret_cast<__half*>(smc + L::QH);
    for (int row = warp; row < NRP; row += NW) {
        float x[4] = {0.f, 0.f, 0.f, 0.f};
        if (row < ROWS)
            q_post(x, reinterpret_cast<const float4*>(F32(L::Q) + row * D)[lane],
                   reinterpret_cast<const float4*>(F32(L::QS) + (row % GQA) * D)[lane], tau_of(row / GQA), true);
        rotate(x, lane);
        const float qs = store_f16_row(x, qh + row * L::QHS, lane);
        if (lane == 0) F32(L::QSUM)[row] = qs;
    }
    for (int rr = warp * 2 + (lane >> 4); rr < NRP; rr += 2 * NW) {
        const int c = lane & 15, row = min(rr, ROWS - 1);
        const float y = rel_post(F32(L::REL)[row * DREL + c] * F32(L::RS)[(row % GQA) * DREL + c], tau_of(row / GQA), true);
        F32(L::RELD)[rr * DREL + c] = rr < ROWS ? y : 0.f;
    }

    // ---- owner: the call's keys / values (dynamic conv, clip / RMS norm, BF16), encoded into the block image
    char* const kimg = smc + L::KIMG;
    char* const vimg = smc + L::VIMG;
    if (owner) {
        float* nkf = F32(L::NKF);
        owner_conv<M, true>(F32(L::ST), F32(L::X), F32(L::W), F32(L::DYN), nkf, F32(L::NST), tid);
        __syncthreads();
        unsigned char* codes = reinterpret_cast<unsigned char*>(smc + L::CODES) + warp * D;
        for (int pr = warp; pr < 2 * M; pr += NW) {
            const int kv = pr / M, m = pr % M;
            const float4 xv = reinterpret_cast<const float4*>(nkf + pr * D)[lane];
            const float f = kv_norm(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w), kv == 1);
            float y[4] = {bf16r(xv.x * f), bf16r(xv.y * f), bf16r(xv.z * f), bf16r(xv.w * f)};
            const float norm = sqrtf(wsum(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]));
            rotate(y, lane);
            if (kv == 0) F::encode_k(y, norm, kimg, m, codes, lane);
            else F::encode_v(y, vimg, m, codes, lane);
        }
    }
    __syncthreads();

    // ---- this warp's rows: rotated q fragments, Qsum, softmax state, accumulators
    uint32_t qb[NTN == 1 ? 8 : 1][2];  // M = 1: q B fragments of the 8 k-steps in registers
    if constexpr (NTN == 1) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            qb[ks][0] = *reinterpret_cast<const uint32_t*>(qh + gid * L::QHS + 16 * ks + 2 * t4);
            qb[ks][1] = *reinterpret_cast<const uint32_t*>(qh + gid * L::QHS + 16 * ks + 2 * t4 + 8);
        }
    }
    auto qfrag = [&](int nn, int ks, int h) {
        if constexpr (NTN == 1) return qb[ks][h];
        return *reinterpret_cast<const uint32_t*>(qh + (8 * nn + gid) * L::QHS + 16 * ks + 2 * t4 + 8 * h);
    };
    Soft<NTN> st;
    st.init([&](int row) { return koff_of<F>(F32(L::QSUM)[row]); }, lane);
    __half* const pw = reinterpret_cast<__half*>(smc + L::PW) + warp * NRP * L::PWS;

    // One block of 16 keys (slot j at position kbase + j; own: the call's keys, slot j = token j).
    auto consume = [&](const F& r, const int kbase, const bool own) {
        // the newest key of the block is at distance >= p0 - kbase - 15 from token 0
        const bool near = own || p0 - kbase - (BLK - 1) < EXT;
        attend_block<F, NTN>(r, qfrag, [&](int key, int row, float sc) {
            const int t = row / GQA, d = own ? t - key : p0 + t - (kbase + key);
            const bool ok = row < ROWS && (own ? key <= t : kbase + key < p0);
            if (near && ok && d < EXT) {
                const float* rl = F32(L::RELD) + row * DREL;
                float bsum = 0.f;
#pragma unroll
                for (int k = 0; k < DREL; ++k) bsum = fmaf(rl[k], __ldg(p.rel_proj + k * EXT + d), bsum);
                sc = fmaf(bf16r(bsum), LOG2E, sc);
            }
            return ok ? sc : -INFINITY;
        }, st, pw, L::PWS, lane);
    };

    if (owner && warp == NW - 1) {  // the call's keys (warp 7 has the fewest cache blocks when nb % stride is small)
        F own;
        own.template load<false>(kimg, vimg, lane, pol);
        consume(own, 0, true);
    }
#pragma unroll 1
    for (; b < nb; b += stride) {
        if (b + stride < nb) nxt.template load<true>(blk_k(b + stride), blk_v(b + stride), lane, pol);
        consume(cur, b * BLK, false);
        cur = nxt;
    }

    // ---- warp partials -> block partial: row max over warps, then a fixed-order tree sum (deterministic)
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float l = st.l[nn][h];
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
            st.l[nn][h] = l;
        }
    float2* const ml = reinterpret_cast<float2*>(smc + L::ML);
    if (gid == 0)
#pragma unroll
        for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
            for (int h = 0; h < 2; ++h) ml[warp * NRP + 8 * nn + 2 * t4 + h] = make_float2(st.m[nn][h], st.l[nn][h]);
    __syncthreads();
    float bm[NTN][2], bl[NTN][2];
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 8 * nn + 2 * t4 + h;
            float mx = NEG_INIT;
#pragma unroll
            for (int w = 0; w < NW; ++w) mx = fmaxf(mx, ml[w * NRP + row].x);
            float l = 0.f;
#pragma unroll
            for (int w = 0; w < NW; ++w) l += ml[w * NRP + row].y * exp2f(ml[w * NRP + row].x - mx);
            bm[nn][h] = mx, bl[nn][h] = l;
            const float f = exp2f(st.m[nn][h] - mx);
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {  // the value level offset once per row: + VOFF l
                st.acc[dt][nn][h] = voff_add<F>(st.acc[dt][nn][h], st.l[nn][h]) * f;
                st.acc[dt][nn][h + 2] = voff_add<F>(st.acc[dt][nn][h + 2], st.l[nn][h]) * f;
            }
        }
    float* const red = F32(L::RED);
    auto red_at = [&](int buf, int nn, int v, int dt) {  // [buf][row][dim]
        const int row = 8 * nn + 2 * t4 + (v & 1), dim = 16 * dt + gid + (v >> 1) * 8;
        return red + (buf * NRP + row) * D + dim;
    };
#pragma unroll
    for (int h = NW / 2; h >= 1; h >>= 1) {
        if (warp >= h && warp < 2 * h)
#pragma unroll
            for (int dt = 0; dt < 8; ++dt)
#pragma unroll
                for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
                    for (int v = 0; v < 4; ++v) *red_at(warp - h, nn, v, dt) = st.acc[dt][nn][v];
        __syncthreads();
        if (warp < h)
#pragma unroll
            for (int dt = 0; dt < 8; ++dt)
#pragma unroll
                for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
                    for (int v = 0; v < 4; ++v) st.acc[dt][nn][v] += *red_at(warp, nn, v, dt);
        __syncthreads();
    }
    float* part = p.part + size_t(g * cs + rank) * ROWS * PSTR;
    if (warp == 0) {
#pragma unroll
        for (int dt = 0; dt < 8; ++dt)
#pragma unroll
            for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int row = 8 * nn + 2 * t4 + (v & 1), dim = 16 * dt + gid + (v >> 1) * 8;
                    if (row < ROWS) __stcg(part + row * PSTR + dim, st.acc[dt][nn][v]);
                }
        if (gid == 0)
#pragma unroll
            for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = 8 * nn + 2 * t4 + h;
                    if (row < ROWS) __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(bm[nn][h], bl[nn][h]));
                }
    }

    // ---- owner: new rows into the cache (positions >= p0: nobody in this call reads them), conv states
    if (owner) {
        for (int m = warp; m < M; m += NW) {
            const int pos = p0 + m;
            F::copy_k(kimg, m, blk_k(pos / BLK), pos % BLK, lane);
            F::copy_v(vimg, m, blk_v(pos / BLK), pos % BLK, lane);
        }
        for (int idx = tid; idx < 3 * 2 * D; idx += NT) {
            const int i = idx / (2 * D), kv = (idx / D) % 2, ch = idx % D;
            (kv ? p.conv_v : p.conv_k)[i * CONV_C + g * D + ch] = F32(L::NST)[idx];
        }
    }

    // ---- the head's last block merges its cs partials with the sink, undoes the rotation, gates
    __shared__ int s_last;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(p.counter + 1 + g, 1) == cs - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (tid == 0) p.counter[1 + g] = 0;
    for (int i = tid; i < ROWS * D; i += NT) {
        const int row = i / D, dim = i % D;
        const float* src = p.part + size_t(g * cs) * ROWS * PSTR + row * PSTR;
        const float sink = F32(L::SINK)[row] * LOG2E;
        float mx = sink;
        for (int q = 0; q < cs; ++q) mx = fmaxf(mx, __ldcg(src + q * ROWS * PSTR + D));
        float l = exp2f(sink - mx), a = 0.f;
        for (int q = 0; q < cs; ++q) {
            const float2 ql = __ldcg(reinterpret_cast<const float2*>(src + q * ROWS * PSTR + D));
            const float f = exp2f(ql.x - mx);
            l += ql.y * f;
            a += __ldcg(src + q * ROWS * PSTR + dim) * f;
        }
        red[row * D + dim] = a / l;
    }
    __syncthreads();
    for (int row = warp; row < ROWS; row += NW) {
        float x[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) x[e] = red[row * D + 4 * lane + e];
        const size_t h = size_t(row / GQA) * HQ + g * GQA + row % GQA;
        write_row<F>(x, p.out + h * D, [&](int dim) {
            return __ldcg(p.proj + size_t(row / GQA) * R_FULL + OFF_GATE + (g * GQA + row % GQA) * D + dim);
        }, lane);
    }
}

// ---- pack: T rows per visit, one warp per (token, head)
template <class F>
__global__ void __launch_bounds__(NT) pack_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v, int T, long long pos0,
                                                   char* kc, char* vc) {
    __shared__ __align__(16) char img[NW][F::KB + F::VB];  // per-warp block image of one key (slot 0), K then V
    __shared__ unsigned char codes[NW][D];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int item = blockIdx.x * NW + warp;
    if (item >= T * HKV) return;
    const int t = item / HKV, g = item % HKV;
    const long long pos = pos0 + t;
    for (int kv = 0; kv < 2; ++kv) {
        const __nv_bfloat16* src = (kv ? v : k) + (size_t(t) * HKV + g) * D + 4 * lane;
        float y[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) y[e] = __bfloat162float(src[e]);
        const float norm = sqrtf(wsum(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]));
        rotate(y, lane);
        if (kv == 0) F::encode_k(y, norm, img[warp], 0, codes[warp], lane);
        else F::encode_v(y, img[warp] + F::KB, 0, codes[warp], lane);
    }
    __syncwarp();
    const long long bi = pos / BLK;
    F::copy_k(img[warp], 0, kc + (size_t(bi) * HKV + g) * F::KB, int(pos % BLK), lane);
    F::copy_v(img[warp] + F::KB, 0, vc + (size_t(bi) * HKV + g) * F::VB, int(pos % BLK), lane);
}

// ---- prefill: a chunk's T tokens against every cached key up to each token (the chunk's own rows packed first)
// [Agent] Same block step as decode, organised like the prefill FP8 kernel: a CTA owns one KV head and 128 query rows
// (token, head of the GQA group), each warp 16 of them (two 8-row tiles, accumulators for all 128 dims), and all 8
// warps walk the same key blocks, staged once per CTA with cp.async (4 blocks per stage, double-buffered). Each code
// is decoded once per warp and feeds 2 tiles of rows; at 64 instruction slots per mma on GA102 (half-rate FP32
// accumulation) the decode is not what bounds it.
constexpr int PF_ROWS = NW * 16, PF_STAGE = 4, PF_PWS = 24;

template <class F>
struct PfSmem {
    static constexpr int BLKB = F::KB + F::VB;  // one staged block: K bytes then V bytes
    static constexpr int QHS = D + 8;
    static constexpr int QH = 0;   // f16 [NW][16][QHS] (prologue)
    static constexpr int STG = 0;  // [2][PF_STAGE][BLKB] (main loop)
    static constexpr int OB = 0;   // f32 [NW][16][D] (epilogue)
    static constexpr int U = std::max({2 * NW * 16 * QHS, 2 * PF_STAGE * BLKB, 4 * NW * 16 * D});
    static constexpr int PW = (U + 15) / 16 * 16;          // f16 [NW][16][PF_PWS]
    static constexpr int QSUM = PW + 2 * NW * 16 * PF_PWS;  // f32 [NW][16]
    static constexpr size_t BYTES = size_t(QSUM + 4 * NW * 16);
};

template <class F>
__global__ void __launch_bounds__(NT, 1) lb_prefill_kernel(const PrefillArgs p) {
    using L = PfSmem<F>;
    constexpr int BLKB = L::BLKB;
    extern __shared__ __align__(16) char smc[];
    const int g = blockIdx.y, tile = gridDim.x - 1 - blockIdx.x;  // heaviest row tiles (latest tokens) first
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    const int rows = GQA * p.T, R0 = tile * PF_ROWS, RW = R0 + 16 * warp;
    const char* const kc = static_cast<const char*>(p.k_cache);
    const char* const vc = static_cast<const char*>(p.v_cache);

    // ---- this warp's 16 query rows, rotated, FP16
    __half* const qh = reinterpret_cast<__half*>(smc + L::QH) + warp * 16 * L::QHS;
    float* const qsm = reinterpret_cast<float*>(smc + L::QSUM) + warp * 16;
    for (int r = 0; r < 16; ++r) {
        const int R = RW + r;
        float x[4] = {0.f, 0.f, 0.f, 0.f};
        if (R < rows) {
            const uint2 u = *reinterpret_cast<const uint2*>(p.q + (size_t(R / GQA) * HQ + g * GQA + R % GQA) * D + 4 * lane);
            x[0] = __uint_as_float(u.x << 16), x[1] = __uint_as_float(u.x & 0xffff0000u);
            x[2] = __uint_as_float(u.y << 16), x[3] = __uint_as_float(u.y & 0xffff0000u);
        }
        rotate(x, lane);
        const float qs = store_f16_row(x, qh + r * L::QHS, lane);
        if (lane == 0) qsm[r] = qs;
    }
    __syncwarp();
    uint32_t qb[2][8][2];
#pragma unroll
    for (int nn = 0; nn < 2; ++nn)
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                qb[nn][ks][h] = *reinterpret_cast<const uint32_t*>(qh + (8 * nn + gid) * L::QHS + 16 * ks + 2 * t4 + 8 * h);
    Soft<2> st;
    st.init([&](int row) { return koff_of<F>(qsm[row]); }, lane);
    __syncthreads();  // the query staging is reused by the key stages

    // ---- key blocks [0, nb): every key up to the CTA's last token
    const int t_hi = min(p.T - 1, (R0 + PF_ROWS - 1) / GQA);
    const int nb = (p.pos0 + t_hi + 1 + BLK - 1) / BLK, ng = (nb + PF_STAGE - 1) / PF_STAGE;
    auto issue = [&](int stage, int grp) {
        char* dst = smc + L::STG + size_t(stage) * PF_STAGE * BLKB;
        for (int i = tid; i < PF_STAGE * BLKB / 16; i += NT) {
            const int bb = i / (BLKB / 16), c = i % (BLKB / 16), b = grp * PF_STAGE + bb;
            if (b >= nb) continue;
            const char* src = c < F::KB / 16 ? kc + (size_t(b) * HKV + g) * F::KB + 16 * c
                                             : vc + (size_t(b) * HKV + g) * F::VB + 16 * (c - F::KB / 16);
            cp_async16(dst + bb * BLKB + 16 * c, src);
        }
        cp_commit();
    };
    __half* const pw = reinterpret_cast<__half*>(smc + L::PW) + warp * 16 * PF_PWS;
    auto qfrag = [&](int nn, int ks, int h) { return qb[nn][ks][h]; };
    const uint16_t* const rel = reinterpret_cast<const uint16_t*>(p.rel);
    issue(0, 0);
#pragma unroll 1
    for (int grp = 0; grp < ng; ++grp) {
        if (grp + 1 < ng) {
            issue((grp + 1) & 1, grp + 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const char* stg = smc + L::STG + size_t(grp & 1) * PF_STAGE * BLKB;
#pragma unroll 1
        for (int bb = 0; bb < PF_STAGE; ++bb) {
            const int b = grp * PF_STAGE + bb;
            if (b >= nb) break;
            F r;
            r.template load<false>(stg + bb * BLKB, stg + bb * BLKB + F::KB, lane, 0);
            const int kbase = b * BLK;
            attend_block<F, 2>(r, qfrag, [&](int key, int row, float sc) {
                const int R = RW + row, t = R / GQA, d = p.pos0 + t - (kbase + key);
                const bool ok = R < rows && d >= 0;
                if (ok && d < EXT)
                    sc = fmaf(__uint_as_float(uint32_t(rel[(size_t(t) * HQ + g * GQA + R % GQA) * EXT + d]) << 16), LOG2E, sc);
                return ok ? sc : -INFINITY;
            }, st, pw, PF_PWS, lane);
        }
        __syncthreads();  // the stage is free for the next issue
    }

    // ---- epilogue: sink, value offset, normalise; then per row the inverse rotation, BF16, gate
    float* const ob = reinterpret_cast<float*>(smc + L::OB) + warp * 16 * D;
    const uint16_t* const proj = reinterpret_cast<const uint16_t*>(p.proj);
    auto bf = [](uint16_t u) { return __uint_as_float(uint32_t(u) << 16); };
#pragma unroll
    for (int nn = 0; nn < 2; ++nn)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float l = st.l[nn][h];
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
            const int row = 8 * nn + 2 * t4 + h, R = RW + row;
            const float den = l + (R < rows ? exp2f(bf(proj[size_t(R / GQA) * p.ld + OFF_SINK + g * GQA + R % GQA]) * LOG2E - st.m[nn][h]) : 0.f);
            const float inv = 1.f / den;
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                ob[row * D + 16 * dt + gid] = voff_add<F>(st.acc[dt][nn][h], l) * inv;
                ob[row * D + 16 * dt + gid + 8] = voff_add<F>(st.acc[dt][nn][h + 2], l) * inv;
            }
        }
    __syncwarp();
    for (int r = 0; r < 16 && RW + r < rows; ++r) {
        const int R = RW + r, t = R / GQA, hq = g * GQA + R % GQA;
        float x[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) x[e] = ob[r * D + 4 * lane + e];
        write_row<F>(x, p.out + (size_t(t) * HQ + hq) * D, [&](int dim) { return bf(proj[size_t(t) * p.ld + OFF_GATE + hq * D + dim]); },
                     lane);
    }
}

struct Kernel {
    void (*fn)(Params);
    size_t smem;
    int cs;
};

template <int M, class F>
Kernel kernel_of() {
    static const Kernel k = [] {
        const auto f = lb_kernel<M, F>;
        const size_t smem = Smem<M, F>::BYTES;
        CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
        // [Agent] As many blocks per KV head as fill the GPU in one wave (device facts, never the architecture), at
        // most the MAXCS partials the workspace holds.
        int dev = 0, sms = 0, per_sm = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, f, NT, smem));
        return Kernel{f, smem, std::max(1, std::min(MAXCS, sms * per_sm / HKV))};
    }();
    return k;
}

template <class F>
Kernel select(int M) {
    switch (M) {
        case 1: return kernel_of<1, F>();
        case 2: return kernel_of<2, F>();
        case 3: return kernel_of<3, F>();
        case 4: return kernel_of<4, F>();
        case 5: return kernel_of<5, F>();
        case 6: return kernel_of<6, F>();
        case 7: return kernel_of<7, F>();
        default: return kernel_of<8, F>();
    }
}

}  // namespace

std::size_t cache_bytes(KvFormat f, std::int64_t rows, bool values) {
    if (!low_bit(f)) return std::size_t(rows) * CONV_C * (f == KvFormat::fp8 ? 1 : 2);
    std::size_t per = 0;
    with_format(f, [&](auto t) {
        using F = typename decltype(t)::type;
        per = values ? F::VB : F::KB;
    });
    return std::size_t((rows + BLK - 1) / BLK) * HKV * per;
}

void attn_decode_lb(const DecodeArgs& a, cudaStream_t stream) {
    require(a.log_scaling, "the low-bit reader always applies the log-length temperature");
    Params p{};
    p.proj = a.proj;
    p.cap = a.cache_rows;
    p.q_scale = a.q_scale;
    p.r_scale = a.r_scale;
    p.rel_proj = a.rel_proj;
    p.k_conv = a.k_conv;
    p.v_conv = a.v_conv;
    p.conv_k = a.conv_k_state;
    p.conv_v = a.conv_v_state;
    p.k_cache = static_cast<char*>(a.k_cache);
    p.v_cache = static_cast<char*>(a.v_cache);
    p.pos = reinterpret_cast<const long long*>(a.pos);
    p.counter = static_cast<int*>(a.workspace);
    p.part = reinterpret_cast<float*>(static_cast<char*>(a.workspace) + WS_PART);
    p.out = a.out;
    Kernel k{};
    with_format(a.kv, [&](auto t) { k = select<typename decltype(t)::type>(a.M); });
    cudaLaunchAttribute attrs[2];
    const cudaLaunchConfig_t cfg = launch_config(k.smem, stream, attrs, k.cs, false);
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, k.fn, p));
}

void prefill_attend(const PrefillArgs& a, cudaStream_t stream) {
    require(a.T > 0 && a.q && a.rel && a.proj && a.k_cache && a.v_cache && a.out, "prefill_attend: null argument");
    with_format(a.kv, [&](auto t) {
        using F = typename decltype(t)::type;
        static const bool attr = [] {
            CUDA_CHECK(cudaFuncSetAttribute(lb_prefill_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            int(PfSmem<F>::BYTES)));
            return true;
        }();
        (void)attr;
        lb_prefill_kernel<F><<<dim3((GQA * a.T + PF_ROWS - 1) / PF_ROWS, HKV), NT, PfSmem<F>::BYTES, stream>>>(a);
    });
    CUDA_CHECK(cudaGetLastError());
}

void pack_rows(KvFormat kv, const __nv_bfloat16* k, const __nv_bfloat16* v, int T, std::int64_t pos0, void* k_cache,
               void* v_cache, cudaStream_t stream) {
    if (T <= 0) return;
    with_format(kv, [&](auto t) {
        using F = typename decltype(t)::type;
        pack_kernel<F><<<(T * HKV + NW - 1) / NW, NT, 0, stream>>>(k, v, T, pos0, static_cast<char*>(k_cache),
                                                                    static_cast<char*>(v_cache));
    });
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::attn
