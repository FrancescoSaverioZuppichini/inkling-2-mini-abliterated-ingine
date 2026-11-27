// Full-attention decode over low-bit KV caches (KvFormat other than bf16 / fp8), M = 1..8 tokens, one kernel for
// every format: a format is a policy (bytes per block, register decode into mma operands, row encode), see `Int4`.
//
// Design
// ------
// * Same contract as attn.cu's full-layer path (q / rel post-processing, dynamic key/value convs, key clip, value RMS
//   norm, relative bias over the last 1024 positions, log-length temperature, sink logit, sigmoid gate, the call's
//   own keys attended exactly as the cache will hold them), with the cache in 16-position blocks per KV head written in
//   mma operand order (kv_format.h).
// * Grid (cs, 8): cs blocks per KV head, 8 warps each. Warp w of block r owns cache blocks r * 8 + w + n * 8 cs: each
//   lane loads its 32 B of K codes + 4 B of key scales and its 32 B of V codes (two 16-byte loads each, one warp reads
//   1 KB contiguous), the next block's loads in flight while this one is consumed. No barrier inside the loop.
// * Per block: S^T (16 keys x 8-row tiles) = K Q^T with mma m16n8k16 f16 (keys decoded to FP16 in registers, the
//   rotated query in FP16), scaled per key, bias / mask, the warp's online softmax (row statistics live in the lanes
//   that hold the rows' accumulators), P to a 16 x 8 n-tile per warp in shared memory (FP16), then O^T += V^T P^T with
//   the values decoded in registers. Accumulators: every warp holds all 128 (rotated) dims of its rows.
// * Warps merge in shared memory in a fixed tree order, blocks with attn.cu's last-block merge (counter per head), then
//   the merging block undoes the value rotation per row (FWHT), scales, rounds to BF16 and applies the gate.
// * No PDL wait is needed below sm_90 (no programmatic launch there); the launch sets no PDL attribute.
#include "kernels/attn/attn.h"
#include "kernels/attn/kv_format.h"

#include "kernels/common/l2.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ingine::kernels::attn {

void attn_decode_lb(const DecodeArgs& a, cudaStream_t stream);

namespace {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, GQA = 3, CONV_C = HKV * D, BLK = kvf::BLK;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632;
constexpr int R_FULL = 8664;
constexpr int NT = 256, NW = NT / 32;
constexpr int MAXCS = 16;  // attn.cu's workspace holds 16 partials per KV head
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.08838834764831845f * LOG2E;  // 1/sqrt(128) * log2(e)
constexpr float NEG_INIT = -1e30f;
constexpr float INV_3 = 1.f / 3.f;
constexpr size_t WS_PART = 256;
constexpr int PSTR = D + 4;

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}
__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}
__device__ __forceinline__ float clip_factor(float ss, float t) {
    const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
    return fminf(1.f, __fdividef(t, norm + 1e-6f));
}

// Unnormalised Walsh-Hadamard transform of a 128-row held 4 per lane (lane l: dims 4l .. 4l + 3), in place.
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

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr l2::Evict KV_L2 = l2::Evict::first;

// 16 bytes of a cache block: global (coherent, no L1 allocation, evict-first L2 lines) or the owner's shared image
template <bool G>
__device__ __forceinline__ uint4 ld16(const char* ptr, std::uint64_t pol) {
    if constexpr (!G) return *reinterpret_cast<const uint4*>(ptr);
    uint4 v;
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(ptr), "l"(pol)
                     : "memory");
    else
        asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(ptr)
                     : "memory");
    return v;
}
template <bool G>
__device__ __forceinline__ uint32_t ld4(const char* ptr) {
    if constexpr (!G) return *reinterpret_cast<const uint32_t*>(ptr);
    uint32_t v;
    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

// ---- formats ------------------------------------------------------------------------------------------------------

// 8 int4 codes (nibble i = code n_i) -> 4 FP16 pairs (n0, n4), (n1, n5), (n2, n6), (n3, n7) holding n - 8.
// [Agent] The magic-number decode: OR a nibble into the mantissa of 1024 (0x6400) gives 1024 + n exactly; one
// subtraction (low nibbles) or one FMA by 1/16 (high nibbles, which sit 4 bits up) leaves n - 8. 9 instructions per
// 8 codes. The levels are n - 7.5; the +0.5 is added once per score / output (Qsum, l) instead of per code.
__device__ __forceinline__ void dq4(uint32_t w, uint32_t (&h)[4]) {
    const uint32_t EX = 0x64006400u;
    const __half2 SUB = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));  // 1032
    const __half2 MUL = __halves2half2(__ushort_as_half(0x2c00), __ushort_as_half(0x2c00));  // 1/16
    const __half2 ADD = __halves2half2(__ushort_as_half(0xd480), __ushort_as_half(0xd480));  // -72
    uint32_t lo = (w & 0x000f000fu) | EX, hi = (w & 0x00f000f0u) | EX;
    __half2 r0 = __hsub2(*reinterpret_cast<__half2*>(&lo), SUB);
    __half2 r1 = __hfma2(*reinterpret_cast<__half2*>(&hi), MUL, ADD);
    w >>= 8;
    lo = (w & 0x000f000fu) | EX, hi = (w & 0x00f000f0u) | EX;
    __half2 r2 = __hsub2(*reinterpret_cast<__half2*>(&lo), SUB);
    __half2 r3 = __hfma2(*reinterpret_cast<__half2*>(&hi), MUL, ADD);
    h[0] = *reinterpret_cast<uint32_t*>(&r0), h[1] = *reinterpret_cast<uint32_t*>(&r1);
    h[2] = *reinterpret_cast<uint32_t*>(&r2), h[3] = *reinterpret_cast<uint32_t*>(&r3);
}

// The format policy. Raw: what one lane loads per block. k_frags: A operands of k-steps 2 kk, 2 kk + 1 (rows = keys
// gid, gid + 8). k_scale: per-key score factors of keys gid, gid + 8. v_frag: A operand of dim tile dt (rows = dims
// 16 dt + gid, + 8; k = keys). KOFF / VOFF: the per-code offset added through Qsum / l. OUT: output factor after the
// inverse rotation. encode: one row (4 values per lane) into a block image at slot j.
struct Int4 {
    static constexpr int KB = kvf::int4::K_BYTES, VB = kvf::int4::V_BYTES;
    static constexpr bool ROT = true;
    static constexpr float KOFF = 0.5f, VOFF = 0.5f;
    static constexpr float OUT = kvf::int4::STEP * 0.08838834764831845f;  // step / sqrt(128)
    static constexpr float SCORE = QSCALE / 128.f;                         // q'' . k'' = 128 q . k
    struct Raw {
        uint32_t k[8], s, v[8];
    };
    template <bool G>
    __device__ __forceinline__ static void load(Raw& r, const char* kb, const char* vb, int lane, std::uint64_t pol) {
        const uint4 a = ld16<G>(kb + 32 * lane, pol), b = ld16<G>(kb + 32 * lane + 16, pol);
        r.k[0] = a.x, r.k[1] = a.y, r.k[2] = a.z, r.k[3] = a.w, r.k[4] = b.x, r.k[5] = b.y, r.k[6] = b.z, r.k[7] = b.w;
        r.s = ld4<G>(kb + BLK * D / 2 + 4 * (lane >> 2));
        const uint4 c = ld16<G>(vb + 32 * lane, pol), d = ld16<G>(vb + 32 * lane + 16, pol);
        r.v[0] = c.x, r.v[1] = c.y, r.v[2] = c.z, r.v[3] = c.w, r.v[4] = d.x, r.v[5] = d.y, r.v[6] = d.z, r.v[7] = d.w;
    }
    __device__ __forceinline__ static void k_frags(const Raw& r, int kk, uint32_t (&a0)[4], uint32_t (&a1)[4]) {
        uint32_t g[4], h[4];
        dq4(r.k[2 * kk], g);      // key gid
        dq4(r.k[2 * kk + 1], h);  // key gid + 8
        a0[0] = g[0], a0[1] = h[0], a0[2] = g[1], a0[3] = h[1];
        a1[0] = g[2], a1[1] = h[2], a1[2] = g[3], a1[3] = h[3];
    }
    __device__ __forceinline__ static float2 k_scale(const Raw& r) {
        return __half22float2(*reinterpret_cast<const __half2*>(&r.s));
    }
    __device__ __forceinline__ static void v_frag(const Raw& r, int dt, uint32_t (&a)[4]) { dq4(r.v[dt], a); }

    // One row of 128 values (lane: dims 4 lane .. + 3, already rotated: H D x) into slot j of a block image. Keys
    // carry |k| (unit-RMS units) and get a least-squares FP16 scale; values are unit-RMS after / sqrt(128).
    __device__ static void encode(bool key, const float (&x)[4], float norm, char* kimg, char* vimg, int j,
                                  unsigned char* codes, int lane) {
        // x: H D row. key: x / |k| has RMS 1 (|H D k| = sqrt(128) |k|); value: x / sqrt(128) has RMS 1.
        const float inv = key ? (norm > 0.f ? 1.f / norm : 0.f) : 0.08838834764831845f;
        float num = 0.f, den = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int n = kvf::int4::code_of(x[e] * inv);
            codes[4 * lane + e] = static_cast<unsigned char>(n);
            const float c = float(n) - 7.5f;
            num = fmaf(x[e], c, num), den = fmaf(c, c, den);
        }
        __syncwarp();
        if (key) {
            num = wsum(num), den = wsum(den);
            if (lane < 16) {
                const int kk = lane >> 2, t4 = lane & 3;
                uint32_t w = 0;
#pragma unroll
                for (int nib = 0; nib < 8; ++nib) w |= uint32_t(codes[kvf::int4::k_dim(kk, t4, nib)]) << (4 * nib);
                reinterpret_cast<uint32_t*>(kimg)[kvf::int4::k_word(j, kk, t4)] = w;
            } else if (lane == 16) {
                *reinterpret_cast<__half*>(kimg + kvf::int4::k_scale_off(j)) = __float2half_rn(num / den);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int pr = lane + 32 * i, dt = pr >> 3, gid = pr & 7, d = 16 * dt + gid;
                vimg[kvf::int4::v_byte(j, d)] = static_cast<char>(codes[d] | (codes[d + 8] << 4));
            }
        }
        __syncwarp();
    }
    // Copy key slot js of image (kimg, vimg) to slot jd of a cache block (kc, vc): whole words / bytes per key, so a
    // concurrent reader of the block's other keys never sees a torn value.
    __device__ static void copy_slot(const char* kimg, const char* vimg, int js, char* kc, char* vc, int jd, int lane) {
        if (lane < 16) {
            const int kk = lane >> 2, t4 = lane & 3;
            reinterpret_cast<uint32_t*>(kc)[kvf::int4::k_word(jd, kk, t4)] =
                reinterpret_cast<const uint32_t*>(kimg)[kvf::int4::k_word(js, kk, t4)];
        } else if (lane == 16) {
            *reinterpret_cast<__half*>(kc + kvf::int4::k_scale_off(jd)) =
                *reinterpret_cast<const __half*>(kimg + kvf::int4::k_scale_off(js));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int pr = lane + 32 * i, d = 16 * (pr >> 3) + (pr & 7);
            vc[kvf::int4::v_byte(jd, d)] = vimg[kvf::int4::v_byte(js, d)];
        }
    }
};

// Online-softmax state of one warp's NTN 8-row tiles: lane (gid, t4) holds rows 8 nn + 2 t4 (+1) of every tile and, for
// them, all 128 dims (dim tile dt: dims 16 dt + gid, + 8) of the unnormalised output.
template <int NTN>
struct Soft {
    float qsum[NTN][2], m[NTN][2], l[NTN][2], acc[8][NTN][4];
    template <class QS>
    __device__ __forceinline__ void init(QS&& qsum_of, float koff, int lane) {
#pragma unroll
        for (int nn = 0; nn < NTN; ++nn) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                qsum[nn][h] = qsum_of(8 * nn + 2 * (lane & 3) + h) * koff;
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
__device__ __forceinline__ void attend_block(const typename F::Raw& r, QF&& qfrag, FIX&& fix, Soft<NTN>& st, __half* pw,
                                             int pws, int lane) {
    const int gid = lane >> 2, t4 = lane & 3;
    float s[NTN][4];
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn) s[nn][0] = s[nn][1] = s[nn][2] = s[nn][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        uint32_t a0[4], a1[4];
        F::k_frags(r, kk, a0, a1);
#pragma unroll
        for (int nn = 0; nn < NTN; ++nn) {
            mma_f16(s[nn], a0, qfrag(nn, 2 * kk, 0), qfrag(nn, 2 * kk, 1));
            mma_f16(s[nn], a1, qfrag(nn, 2 * kk + 1, 0), qfrag(nn, 2 * kk + 1, 1));
        }
    }
    const float2 ks = F::k_scale(r);
#pragma unroll
    for (int nn = 0; nn < NTN; ++nn)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int key = gid + (v >> 1) * 8, row = 8 * nn + 2 * t4 + (v & 1);
            s[nn][v] = fix(key, row, (s[nn][v] + st.qsum[nn][v & 1]) * ((v >> 1) ? ks.y : ks.x) * F::SCORE);
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
        F::v_frag(r, dt, a);
#pragma unroll
        for (int nn = 0; nn < NTN; ++nn) mma_f16(st.acc[dt][nn], a, pb[nn][0], pb[nn][1]);
    }
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
    static constexpr int U_END = (STAGE_END > RED_END ? STAGE_END : RED_END + 15) / 16 * 16;
    static constexpr int QH = U_END;                         // f16 [NRP][D + 8] rotated query
    static constexpr int QHS = D + 8;
    static constexpr int RELD = QH + 2 * NRP * QHS;          // f32 [NRP][16]
    static constexpr int QSUM = RELD + 4 * NRP * DREL;       // f32 [NRP]
    static constexpr int SINK = QSUM + 4 * NRP;              // f32 [ROWS]
    static constexpr int ML = (SINK + 4 * ROWS + 15) / 16 * 16;  // f32x2 [NW][NRP]
    static constexpr int NST = ML + 8 * NW * NRP;            // f32 [3][2][D] advanced conv state
    static constexpr int KIMG = NST + 4 * 3 * 2 * D;         // the call's keys as one block image
    static constexpr int VIMG = (KIMG + F::KB + 15) / 16 * 16;
    static constexpr int CODES = VIMG + F::VB;               // u8 [NW][D] encode scratch
    static constexpr int PW = CODES + NW * D;                // f16 [NW][NRP][24] P tiles
    static constexpr int PWS = 24;
    static constexpr int END = PW + 2 * NW * NRP * PWS;
    static constexpr size_t BYTES = size_t(END);
};

template <int M, class F>
__global__ void __launch_bounds__(NT, M > 2 ? 1 : 2) lb_kernel(const Params p) {
    using L = Smem<M, F>;
    constexpr int ROWS = L::ROWS, NRP = L::NRP, NTN = L::NTN;
    extern __shared__ __align__(16) char smc[];
    auto F32 = [&](int off) { return reinterpret_cast<float*>(smc + off); };

    const int g = blockIdx.y, rank = blockIdx.x, cs = gridDim.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    const bool owner = rank == 0;
    const std::uint64_t pol = l2::policy<KV_L2>();
    auto cp16 = [](void* dst, const float* src) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src)
                     : "memory");
    };
    auto cp4 = [](void* dst, const float* src) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src)
                     : "memory");
    };

    // ---- stage the inputs (as attn.cu's full-layer path)
    for (int i = tid; i < GQA * D / 4; i += NT) cp16(F32(L::QS) + 4 * i, p.q_scale + g * GQA * D + 4 * i);
    for (int i = tid; i < GQA * DREL / 4; i += NT) cp16(F32(L::RS) + 4 * i, p.r_scale + g * GQA * DREL + 4 * i);
    if (owner)
        for (int i = tid; i < 2 * D; i += NT) {  // [kv][lag][D]
            const int kv = i / D, q = i % D;
            cp16(F32(L::W) + 4 * i, (kv ? p.v_conv : p.k_conv) + (q / (D / 4)) * CONV_C + g * D + 4 * (q % (D / 4)));
        }
    long long p0l;
    asm volatile("ld.global.s64 %0, [%1];" : "=l"(p0l) : "l"(p.pos) : "memory");
    const int p0 = int(p0l);
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
    typename F::Raw cur, nxt;
    auto blk_k = [&](int bi) { return p.k_cache + (size_t(bi) * HKV + g) * F::KB; };
    auto blk_v = [&](int bi) { return p.v_cache + (size_t(bi) * HKV + g) * F::VB; };
    if (b < nb) F::template load<true>(cur, blk_k(b), blk_v(b), lane, pol);

    // ---- q (scale, clip, BF16, tau, BF16, then rotated: H D q in FP16) and rel, one warp per row
    auto tau_of = [&](int t) { return 1.f + 0.1f * logf(fmaxf(float(p0 + t + 1) * (1.f / 8192.f), 1.f)); };
    __half* const qh = reinterpret_cast<__half*>(smc + L::QH);
    for (int row = warp; row < NRP; row += NW) {
        float x[4] = {0.f, 0.f, 0.f, 0.f};
        if (row < ROWS) {
            const int i = row % GQA;
            const float4 a = reinterpret_cast<const float4*>(F32(L::Q) + row * D)[lane];
            const float4 s = reinterpret_cast<const float4*>(F32(L::QS) + i * D)[lane];
            x[0] = a.x * s.x, x[1] = a.y * s.y, x[2] = a.z * s.z, x[3] = a.w * s.w;
            const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
            const float tau = tau_of(row / GQA);
#pragma unroll
            for (int e = 0; e < 4; ++e) x[e] = bf16r(bf16r(x[e] * f) * tau);
        }
        if constexpr (F::ROT) {
#pragma unroll
            for (int e = 0; e < 4; ++e) x[e] *= kvf::sign_of(4 * lane + e);
            fwht128(x, lane);
        }
        float qs = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const __half h = __float2half_rn(x[e]);
            qh[row * L::QHS + 4 * lane + e] = h;
            qs += __half2float(h);
        }
        qs = wsum(qs);
        if (lane == 0) F32(L::QSUM)[row] = qs;
    }
    for (int rr = warp * 2 + (lane >> 4); rr < NRP; rr += 2 * NW) {
        const int c = lane & 15;
        float y = 0.f;
        if (rr < ROWS) {
            const int i = rr % GQA;
            const float r = F32(L::REL)[rr * DREL + c] * F32(L::RS)[i * DREL + c];
            y = bf16r(bf16r(r * clip_factor(hsum16(r * r), CLIP_REL)) * tau_of(rr / GQA));
        } else {
            hsum16(0.f);
        }
        F32(L::RELD)[rr * DREL + c] = y;
    }

    // ---- owner: the call's keys / values (dynamic conv, clip / RMS norm, BF16), encoded into the block image
    char* const kimg = smc + L::KIMG;
    char* const vimg = smc + L::VIMG;
    if (owner) {
        float* nkf = F32(L::NKF);
        for (int idx = tid; idx < 2 * D; idx += NT) {
            const int kv = idx / D, ch = idx % D;
            float e[3 + M];
#pragma unroll
            for (int j = 0; j < 3; ++j) e[j] = F32(L::ST)[(j * 2 + kv) * D + ch];
#pragma unroll
            for (int m = 0; m < M; ++m) e[3 + m] = F32(L::X)[(m * 2 + kv) * D + ch];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float o = 0.f;
#pragma unroll
                for (int lag = 0; lag < 4; ++lag) {
                    const float w = __fadd_rn(F32(L::W)[(kv * 4 + lag) * D + ch], F32(L::DYN)[(m * 2 + kv) * 4 + lag]);
                    o = __fadd_rn(o, __fmul_rn(w, e[m + 3 - lag]));
                }
                nkf[(kv * M + m) * D + ch] = bf16r(__fmul_rn(__fadd_rn(o, e[m + 3]), INV_3));
            }
#pragma unroll
            for (int i = 0; i < 3; ++i) F32(L::NST)[(i * 2 + kv) * D + ch] = e[M + i];
        }
        __syncthreads();
        unsigned char* codes = reinterpret_cast<unsigned char*>(smc + L::CODES) + warp * D;
        for (int pr = warp; pr < 2 * M; pr += NW) {
            const int kv = pr / M, m = pr % M;
            const float4 xv = reinterpret_cast<const float4*>(nkf + pr * D)[lane];
            const float ss = wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w);
            const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
            float y[4] = {bf16r(xv.x * f), bf16r(xv.y * f), bf16r(xv.z * f), bf16r(xv.w * f)};
            const float norm = sqrtf(wsum(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]));
            if constexpr (F::ROT) {
#pragma unroll
                for (int e = 0; e < 4; ++e) y[e] *= kvf::sign_of(4 * lane + e);
                fwht128(y, lane);
            }
            F::encode(kv == 0, y, norm, kimg, vimg, m, codes, lane);
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
    st.init([&](int row) { return F32(L::QSUM)[row]; }, F::KOFF, lane);
    __half* const pw = reinterpret_cast<__half*>(smc + L::PW) + warp * NRP * L::PWS;

    // One block of 16 keys (slot j at position kbase + j; own: the call's keys, slot j = token j).
    auto consume = [&](const typename F::Raw& r, const int kbase, const bool own) {
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
        typename F::Raw own;
        F::template load<false>(own, kimg, vimg, lane, pol);
        consume(own, 0, true);
    }
#pragma unroll 1
    for (; b < nb; b += stride) {
        if (b + stride < nb) F::template load<true>(nxt, blk_k(b + stride), blk_v(b + stride), lane, pol);
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
            const float off = F::VOFF * st.l[nn][h];  // the +0.5 of every value level, once
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                st.acc[dt][nn][h] = (st.acc[dt][nn][h] + off) * f;
                st.acc[dt][nn][h + 2] = (st.acc[dt][nn][h + 2] + off) * f;
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
            const int pos = p0 + m, bi = pos / BLK;
            F::copy_slot(kimg, vimg, m, p.k_cache + (size_t(bi) * HKV + g) * F::KB,
                         p.v_cache + (size_t(bi) * HKV + g) * F::VB, pos % BLK, lane);
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
            const float2 st = __ldcg(reinterpret_cast<const float2*>(src + q * ROWS * PSTR + D));
            const float f = exp2f(st.x - mx);
            l += st.y * f;
            a += __ldcg(src + q * ROWS * PSTR + dim) * f;
        }
        red[row * D + dim] = a / l;
    }
    __syncthreads();
    for (int row = warp; row < ROWS; row += NW) {
        float x[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) x[e] = red[row * D + 4 * lane + e];
        if constexpr (F::ROT) fwht128(x, lane);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int dim = 4 * lane + e;
            const float o = F::ROT ? x[e] * kvf::sign_of(dim) * F::OUT : x[e] * F::OUT;
            const float gl = __ldcg(p.proj + size_t(row / GQA) * R_FULL + OFF_GATE + (g * GQA + row % GQA) * D + dim);
            const float gate = 1.f / (1.f + expf(-gl));
            p.out[size_t(row / GQA) * HQ * D + (g * GQA + row % GQA) * D + dim] = __float2bfloat16_rn(bf16r(o) * gate);
        }
    }
}

// ---- pack: T rows per visit, one warp per (token, head)
template <class F>
__global__ void __launch_bounds__(NT) pack_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v, int T, long long pos0,
                                                   char* kc, char* vc) {
    __shared__ __align__(16) char img[NW][F::KB + F::VB];  // per-warp block image of one key (slot 0)
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
        if constexpr (F::ROT) {
#pragma unroll
            for (int e = 0; e < 4; ++e) y[e] *= kvf::sign_of(4 * lane + e);
            fwht128(y, lane);
        }
        F::encode(kv == 0, y, norm, img[warp], img[warp] + F::KB, 0, codes[warp], lane);
    }
    const long long bi = pos / BLK;
    F::copy_slot(img[warp], img[warp] + F::KB, 0, kc + (size_t(bi) * HKV + g) * F::KB, vc + (size_t(bi) * HKV + g) * F::VB,
                 int(pos % BLK), lane);
}

// ---- prefill: a chunk's T tokens against every cached key up to each token (the chunk's own rows packed first)
// [Agent] Same block step as decode, organised like the prefill FP8 kernel: a CTA owns one KV head and 128 query rows
// (token, head of the GQA group), each warp 16 of them (two 8-row tiles, accumulators for all 128 dims), and all 8
// warps walk the same key blocks, staged once per CTA with cp.async (4 blocks per stage, double-buffered). Each code
// is decoded once per warp and feeds 2 tiles of rows; at 64 instruction slots per mma on GA102 (half-rate FP32
// accumulation) the decode is not what bounds it.
struct PrefillP {
    const __nv_bfloat16* q;
    const __nv_bfloat16* rel;
    const __nv_bfloat16* proj;
    int ld, T, pos0;
    const char* k_cache;
    const char* v_cache;
    __nv_bfloat16* out;
};

constexpr int PF_ROWS = NW * 16, PF_STAGE = 4, PF_PWS = 24;
constexpr int PF_GATE = HQ * D, PF_SINK = 2 * HQ * D + HQ * DREL;  // proj columns (prefill layout)

template <class F>
struct PfSmem {
    static constexpr int QHS = D + 8;
    static constexpr int QH = 0;                                  // f16 [NW][16][QHS] (prologue)
    static constexpr int STG = 0;                                 // [2][PF_STAGE][KB + VB] (main loop)
    static constexpr int OB = 0;                                  // f32 [NW][16][D] (epilogue)
    static constexpr int U = std::max({2 * NW * 16 * QHS, 2 * PF_STAGE * (F::KB + F::VB), 4 * NW * 16 * D});
    static constexpr int PW = (U + 15) / 16 * 16;                 // f16 [NW][16][PF_PWS]
    static constexpr int QSUM = PW + 2 * NW * 16 * PF_PWS;         // f32 [NW][16]
    static constexpr size_t BYTES = size_t(QSUM + 4 * NW * 16);
};

template <class F>
__global__ void __launch_bounds__(NT, 1) lb_prefill_kernel(const PrefillP p) {
    using L = PfSmem<F>;
    extern __shared__ __align__(16) char smc[];
    const int g = blockIdx.y, tile = gridDim.x - 1 - blockIdx.x;  // heaviest row tiles (latest tokens) first
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    const int rows = GQA * p.T, R0 = tile * PF_ROWS, RW = R0 + 16 * warp;

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
        if constexpr (F::ROT) {
#pragma unroll
            for (int e = 0; e < 4; ++e) x[e] *= kvf::sign_of(4 * lane + e);
            fwht128(x, lane);
        }
        float qs = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const __half h = __float2half_rn(x[e]);
            qh[r * L::QHS + 4 * lane + e] = h;
            qs += __half2float(h);
        }
        qs = wsum(qs);
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
    st.init([&](int row) { return qsm[row]; }, F::KOFF, lane);
    __syncthreads();  // the query staging is reused by the key stages

    // ---- key blocks [0, nb): every key up to the CTA's last token
    const int t_hi = min(p.T - 1, (R0 + PF_ROWS - 1) / GQA);
    const int nb = (p.pos0 + t_hi + 1 + BLK - 1) / BLK, ng = (nb + PF_STAGE - 1) / PF_STAGE;
    constexpr int CH = (F::KB + F::VB) / 16;  // 16-byte chunks per block (K then V)
    auto issue = [&](int stage, int grp) {
        char* dst = smc + L::STG + size_t(stage) * PF_STAGE * (F::KB + F::VB);
        for (int i = tid; i < PF_STAGE * CH; i += NT) {
            const int bb = i / CH, c = i % CH, b = grp * PF_STAGE + bb;
            if (b >= nb) continue;
            const char* src = c < F::KB / 16 ? p.k_cache + (size_t(b) * HKV + g) * F::KB + 16 * c
                                             : p.v_cache + (size_t(b) * HKV + g) * F::VB + 16 * (c - F::KB / 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(unsigned(__cvta_generic_to_shared(
                             dst + bb * (F::KB + F::VB) + 16 * c))),
                         "l"(src)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    __half* const pw = reinterpret_cast<__half*>(smc + L::PW) + warp * 16 * PF_PWS;
    auto qfrag = [&](int nn, int ks, int h) { return qb[nn][ks][h]; };
    const uint16_t* const rel = reinterpret_cast<const uint16_t*>(p.rel);
    issue(0, 0);
#pragma unroll 1
    for (int grp = 0; grp < ng; ++grp) {
        if (grp + 1 < ng) {
            issue((grp + 1) & 1, grp + 1);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        const char* stg = smc + L::STG + size_t(grp & 1) * PF_STAGE * (F::KB + F::VB);
#pragma unroll 1
        for (int bb = 0; bb < PF_STAGE; ++bb) {
            const int b = grp * PF_STAGE + bb;
            if (b >= nb) break;
            typename F::Raw r;
            F::template load<false>(r, stg + bb * (F::KB + F::VB), stg + bb * (F::KB + F::VB) + F::KB, lane, 0);
            const int kbase = b * BLK;
            attend_block<F, 2>(r, qfrag, [&](int key, int row, float sc) {
                const int R = RW + row, t = R / GQA, kp = kbase + key, d = p.pos0 + t - kp;
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
#pragma unroll
    for (int nn = 0; nn < 2; ++nn)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float l = st.l[nn][h];
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
            const int row = 8 * nn + 2 * t4 + h, R = RW + row;
            float den = l;
            if (R < rows) {
                const float sink = __uint_as_float(uint32_t(proj[size_t(R / GQA) * p.ld + PF_SINK + g * GQA + R % GQA]) << 16);
                den += exp2f(sink * LOG2E - st.m[nn][h]);
            }
            const float inv = 1.f / den, off = F::VOFF * l;
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                ob[row * D + 16 * dt + gid] = (st.acc[dt][nn][h] + off) * inv;
                ob[row * D + 16 * dt + gid + 8] = (st.acc[dt][nn][h + 2] + off) * inv;
            }
        }
    __syncwarp();
    for (int r = 0; r < 16; ++r) {
        const int R = RW + r;
        if (R >= rows) break;
        const int t = R / GQA, hq = g * GQA + R % GQA;
        float x[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) x[e] = ob[r * D + 4 * lane + e];
        if constexpr (F::ROT) fwht128(x, lane);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int dim = 4 * lane + e;
            const float o = F::ROT ? x[e] * kvf::sign_of(dim) * F::OUT : x[e] * F::OUT;
            const float gl = __uint_as_float(uint32_t(proj[size_t(t) * p.ld + PF_GATE + hq * D + dim]) << 16);
            p.out[(size_t(t) * HQ + hq) * D + dim] = __float2bfloat16_rn(bf16r(o) * (1.f / (1.f + expf(-gl))));
        }
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
        // most the 16 partials attn.cu's workspace holds.
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

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("attn (low-bit): ") + what);
}

}  // namespace

std::size_t cache_bytes(KvFormat f, std::int64_t rows, bool values) {
    const std::size_t blocks = std::size_t((rows + BLK - 1) / BLK);
    switch (f) {
        case KvFormat::bf16: return std::size_t(rows) * CONV_C * 2;
        case KvFormat::fp8: return std::size_t(rows) * CONV_C;
        case KvFormat::int4: return blocks * HKV * (values ? Int4::VB : Int4::KB);
    }
    throw std::invalid_argument("cache_bytes: unknown format");
}

void attn_decode_lb(const DecodeArgs& a, cudaStream_t stream) {
    require(a.window == 0 && a.log_scaling, "low-bit caches are full layers only");
    require(a.workspace_size >= workspace_bytes(a.M), "workspace too small");
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
    switch (a.kv) {
        case KvFormat::int4: k = select<Int4>(a.M); break;
        default: require(false, "not a low-bit format");
    }
    k.fn<<<dim3(k.cs, HKV), NT, k.smem, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

void prefill_attend(const PrefillArgs& a, cudaStream_t stream) {
    require(a.T > 0 && a.q && a.rel && a.proj && a.k_cache && a.v_cache && a.out, "prefill_attend: null argument");
    PrefillP p{a.q, a.rel, a.proj, a.ld, a.T, a.pos0, static_cast<const char*>(a.k_cache),
               static_cast<const char*>(a.v_cache), a.out};
    const dim3 grid((GQA * a.T + PF_ROWS - 1) / PF_ROWS, HKV);
    switch (a.kv) {
        case KvFormat::int4: {
            static const bool attr = [] {
                CUDA_CHECK(cudaFuncSetAttribute(lb_prefill_kernel<Int4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                int(PfSmem<Int4>::BYTES)));
                return true;
            }();
            (void)attr;
            lb_prefill_kernel<Int4><<<grid, NT, PfSmem<Int4>::BYTES, stream>>>(p);
            break;
        }
        default: require(false, "prefill_attend: not a low-bit format");
    }
    CUDA_CHECK(cudaGetLastError());
}

void pack_rows(KvFormat f, const __nv_bfloat16* k, const __nv_bfloat16* v, int T, std::int64_t pos0, void* k_cache,
               void* v_cache, cudaStream_t stream) {
    if (T <= 0) return;
    const int grid = (T * HKV + NW - 1) / NW;
    switch (f) {
        case KvFormat::int4:
            pack_kernel<Int4><<<grid, NT, 0, stream>>>(k, v, T, pos0, static_cast<char*>(k_cache), static_cast<char*>(v_cache));
            break;
        default: require(false, "pack_rows: not a low-bit format");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::attn
