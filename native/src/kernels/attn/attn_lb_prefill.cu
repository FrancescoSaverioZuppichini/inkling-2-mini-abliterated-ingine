// Full-attention prefill over low-bit KV caches (int4, int8, k8v4, fp4): a chunk's tokens against every cached key up
// to each token, with the decode reader's per-block step (attn_lb.cuh). Contract: native/KERNELS.md, area `attn`.
#include "kernels/attn/attn_lb.cuh"

namespace ingine::kernels::attn {
namespace {

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

}  // namespace

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

}  // namespace ingine::kernels::attn
