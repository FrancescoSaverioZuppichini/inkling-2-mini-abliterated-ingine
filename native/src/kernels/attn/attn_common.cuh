// Shared by the full-attention decode kernels: attn.cu (BF16 / FP8 caches) and attn_lb.cu (low-bit caches). The model's
// attention constants, the workspace both write their split partials into, the launch configuration, and the
// per-row arithmetic (q / rel post-processing, dynamic conv, key clip, value norm) whose rounding both must match.
#pragma once

#include "kernels/attn/attn.h"

#include "kernels/common/l2.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ingine::kernels::attn {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, GQA = 3, CONV_C = HKV * D;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632;
constexpr int R_SLIDING = 8600, R_FULL = 8664;
constexpr int CS = 16;  // blocks per KV head = cluster size; block r outputs dims [8r, 8r + 8)
// [Agent] Without clusters (sm_86, sm_89) nothing ties the split to the output layout: the head's last block merges
// all its partials. So the blocks per KV head (Params::cs) are chosen per GPU and width, up to MAXCS; with clusters
// cs is always CS.
constexpr int MAXCS = 16;
constexpr int MAXM = 8;
constexpr int NT = 256;  // threads per block (8 warps) for every M
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float RSQRT_D = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float QSCALE = RSQRT_D * LOG2E;
constexpr float NEG_INIT = -1e30f;
constexpr float INV_SQRT5 = 1.f / 2.2360679774997896f;  // torch: x / scalar == x * (1 / float(scalar))
constexpr float INV_3 = 1.f / 3.f;
// workspace: int32 counters (kpos ordering for M > 1, then one per KV head for the last-block merge; zeroed once at
// allocation) then the split partials fp32 [HKV][MAXCS][3M][PSTR]: o (D) then (max, sum)
constexpr size_t WS_PART = 256;
constexpr int PSTR = D + 4;
constexpr size_t ws_part_bytes(int M) { return size_t(HKV) * MAXCS * GQA * M * PSTR * 4; }
// [Agent] Pre-sm_90 only: the owner's new K/V rows staged bf16 [HKV][2][M][D] after the partials, for the head's last
// block to write into the ring.
constexpr size_t ws_bytes(int M) { return WS_PART + ws_part_bytes(M) + size_t(HKV) * 2 * M * D * 2; }

inline void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("attn: ") + what);
}

// Grid (cs, 8 KV heads) with programmatic dependent launch (the kernel waits with griddepcontrol.wait before touching
// anything an earlier kernel may write; a no-op without PDL), plus a cluster of CS blocks when asked and the GPU has
// clusters. attrs must hold 2 entries.
inline cudaLaunchConfig_t launch_config(size_t smem, cudaStream_t stream, cudaLaunchAttribute* attrs, int cs,
                                        bool cluster) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(cs, HKV);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS, attrs[0].val.clusterDim.y = 1, attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;
    if (!cluster || !device_has_clusters()) cfg.attrs = attrs + 1, cfg.numAttrs = 1;
    return cfg;
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

// clip factor min(1, t / (sqrt(ss) + 1e-6)) without the IEEE sqrt / div slow-path branches (which
// serialise the six reductions of the q / rel prologue); differs from torch by a couple of FP32 ulp
__device__ __forceinline__ float clip_factor(float ss, float t) {
    const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
    return fminf(1.f, __fdividef(t, norm + 1e-6f));
}

// Loads of anything an earlier kernel writes (caches, kpos, pos) are coherent `asm volatile` with a memory
// clobber, so the compiler cannot hoist them above griddepcontrol.wait (programmatic dependent launch).
__device__ __forceinline__ long long ld_s64(const long long* ptr) {
    long long v;
    asm volatile("ld.global.s64 %0, [%1];" : "=l"(v) : "l"(ptr) : "memory");
    return v;
}

// Cached K/V rows are read once per step (the next read of a ring row is a full step, ~1.5 GB of streams, later):
// their L2 lines are filled evict-first (kernels/common/l2.cuh).
constexpr l2::Evict KV_L2 = l2::Evict::first;

__device__ __forceinline__ uint4 ld_kv(const void* ptr, std::uint64_t pol) {
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

__device__ __forceinline__ void cp16(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                 "l"(src)
                 : "memory");
}

__device__ __forceinline__ void cp4(float* dst, const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}

// One q row, 4 dims per lane (a: proj row, s: q_scale): scale, L2 clip, BF16, then (full layers) times tau and BF16.
// Every lane of the warp must call it (the clip reduces over the warp).
__device__ __forceinline__ void q_post(float (&x)[4], const float4 a, const float4 s, const float tau, const bool log_scaling) {
    x[0] = a.x * s.x, x[1] = a.y * s.y, x[2] = a.z * s.z, x[3] = a.w * s.w;
    const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        x[e] = bf16r(x[e] * f);
        if (log_scaling) x[e] = bf16r(x[e] * tau);
    }
}

// One rel value (r = proj rel * r_scale, one per lane of a half-warp): L2 clip over the 16 lanes, BF16, tau, BF16.
__device__ __forceinline__ float rel_post(const float r, const float tau, const bool log_scaling) {
    float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
    if (log_scaling) y = bf16r(y * tau);
    return y;
}

// The owner's key / value convolutions of one KV head's 2 x 128 channels: e = [conv state (3), the call's M inputs],
// conv rounded to BF16 into nkf [2][M][D], the advanced state (last 3 inputs) into nst [3][2][D]. st [3][2][D],
// x [M][2][D]; full layers: w [2][4][D] by lag (0 = current) + dyn [M][2][4], / sqrt(9); sliding: w [2][D][4] in
// conv1d order (tap 3 = current), / sqrt(5).
template <int M, bool FULL>
__device__ __forceinline__ void owner_conv(const float* st, const float* x, const float* w, const float* dyn, float* nkf,
                                           float* nst, int tid) {
    for (int idx = tid; idx < 2 * D; idx += NT) {
        const int kv = idx / D, ch = idx % D;
        float e[3 + M];
#pragma unroll
        for (int j = 0; j < 3; ++j) e[j] = st[(j * 2 + kv) * D + ch];
#pragma unroll
        for (int m = 0; m < M; ++m) e[3 + m] = x[(m * 2 + kv) * D + ch];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float y;
            if constexpr (FULL) {
                float o = 0.f;
#pragma unroll
                for (int lag = 0; lag < 4; ++lag)
                    o = __fadd_rn(o, __fmul_rn(__fadd_rn(w[(kv * 4 + lag) * D + ch], dyn[(m * 2 + kv) * 4 + lag]), e[m + 3 - lag]));
                y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_3);
            } else {
                const float* wc = w + (kv * D + ch) * 4;
                float o = __fmul_rn(wc[0], e[m]);
#pragma unroll
                for (int j = 1; j < 4; ++j) o = __fadd_rn(o, __fmul_rn(wc[j], e[m + j]));
                y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_SQRT5);
            }
            nkf[(kv * M + m) * D + ch] = bf16r(y);
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) nst[(i * 2 + kv) * D + ch] = e[M + i];
    }
}

// Scale of a new key (L2 clip at 22.63) or value (RMS norm without gain) row from its sum of squares.
__device__ __forceinline__ float kv_norm(const float ss, const bool value) {
    return value ? rsqrtf(ss / float(D) + 1e-6f) : clip_factor(ss, CLIP_QK);
}

}  // namespace ingine::kernels::attn
