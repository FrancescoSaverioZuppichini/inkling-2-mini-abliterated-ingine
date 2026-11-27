// Full-attention prefill over low-bit KV caches (int4, int8, k8v4, fp4): a chunk's tokens against every cached key up
// to each token, with the decode reader's per-block arithmetic (attn_lb.cuh). Contract: native/KERNELS.md, area `attn`.
#include "kernels/attn/attn_lb.cuh"

namespace ingine::kernels::attn {
namespace {

// [Agent] A CTA owns one KV head and 128 query rows (token, head of the GQA group), each warp 16 of them (two 8-row
// tiles, accumulators for all 128 dims); all 8 warps walk the same key blocks, staged with cp.async 4 blocks at a time.
// Decoding per warp would issue ~2.5x the FP8 kernel's instructions for the same MMAs, so:
// * each staged block is decoded once per CTA, not once per warp: warp w turns block w / 2's keys (even w) or values
//   (odd w) into the FP16 MMA fragments every warp needs, one 16-byte fragment per lane (conflict-free shared loads
//   and stores); the next stage is decoded between this stage's score MMAs and its softmax, so that ALU work overlaps
//   the tensor pipe (fragments double-buffered, one barrier per stage);
// * the scores of the 4 staged blocks are computed together: 8 independent MMA chains instead of 2;
// * blocks whose keys all sit EXT or more positions before the warp's first row skip the mask and bias code;
// * P is transposed in registers (movmatrix), not through shared memory.
// Each score is still summed over the same k-steps in the same order, and each block then runs attend_block's softmax
// step in key order (same maxima, exponentials, FP16 roundings, contractions and sums), so the output bits are those
// of the per-block reader.
constexpr int PF_ROWS = NW * 16, PF_STAGE = 4, NFRAG = 16;  // fragments per block: K a0/a1 of 4 k-steps, V of 8 dim tiles

template <class F>
struct PfSmem {
    static constexpr int BLKB = F::KB + F::VB;  // one staged block: K bytes then V bytes
    static constexpr int QHS = D + 8;
    static constexpr int QH = 0;   // f16 [NW][16][QHS] (prologue)
    static constexpr int STG = 0;  // [2][PF_STAGE][BLKB] codes as cached (main loop)
    static constexpr int QSUM = 2 * NW * 16 * QHS;  // f32 [NW][16] (prologue)
    static constexpr int DEC = (2 * PF_STAGE * BLKB + 15) / 16 * 16;  // uint4 [2][PF_STAGE][NFRAG][32] decoded fragments
    static constexpr int DKS = DEC + 2 * PF_STAGE * NFRAG * 32 * 16;  // float2 [2][PF_STAGE][32] key score factors
    static constexpr int OB = 0;   // f32 [NW][16][D] (epilogue)
    static constexpr size_t BYTES = size_t(std::max({QSUM + 4 * NW * 16, DKS + 2 * PF_STAGE * 32 * 8, 4 * NW * 16 * D}));
};

// [Agent] exp2 without exp2f's denormal wrapper: equal to exp2f for every input >= -126; below it exp2f gives a
// denormal and this gives 0, and both round to the same FP16 0 in P (the smallest FP16 is 2^-24).
__device__ __forceinline__ float ex2_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// [Agent] Transpose an 8x8 FP16 tile held as MMA accumulator pairs (lane: row lane / 4, columns 2 (lane % 4) + 0, 1)
// into the B operand layout: the S^T = K Q^T tile becomes P's operand for O^T += V^T P^T without shared memory.
__device__ __forceinline__ uint32_t transpose8x8(uint32_t x) {
    uint32_t y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
__device__ __forceinline__ void mma_u4(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
    const uint32_t r[4] = {a.x, a.y, a.z, a.w};
    mma_f16(c, r, b0, b1);
}

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
    // [Agent] A quarter of the threads copies each staged block, so a thread's block and chunks are fixed: no
    // division per chunk, one bounds check per block.
    constexpr int CPB = NT / PF_STAGE;
    const int ibb = tid / CPB, ic = tid % CPB;
    auto issue = [&](int stage, int grp) {
        const int b = grp * PF_STAGE + ibb;
        if (b < nb) {
            char* const dst = smc + L::STG + (size_t(stage) * PF_STAGE + ibb) * BLKB;
            const char* const ksrc = kc + (size_t(b) * HKV + g) * F::KB;
            const char* const vsrc = vc + (size_t(b) * HKV + g) * F::VB;
#pragma unroll
            for (int c = ic; c < BLKB / 16; c += CPB)
                cp_async16(dst + 16 * c, c < F::KB / 16 ? ksrc + 16 * c : vsrc + 16 * (c - F::KB / 16));
        }
        cp_commit();
    };
    uint4* const dec = reinterpret_cast<uint4*>(smc + L::DEC) + lane;
    float2* const dks = reinterpret_cast<float2*>(smc + L::DKS) + lane;
    const uint16_t* const rel = reinterpret_cast<const uint16_t*>(p.rel);
    auto fix = [&](int kbase, int key, int row, float sc) {
        const int R = RW + row, t = R / GQA, d = p.pos0 + t - (kbase + key);
        const bool ok = R < rows && d >= 0;
        if (ok && d < EXT)
            sc = fmaf(__uint_as_float(uint32_t(rel[(size_t(t) * HQ + g * GQA + R % GQA) * EXT + d]) << 16), LOG2E, sc);
        return ok ? sc : -INFINITY;
    };
    // [Agent] Every distance in a block is >= EXT (no bias, no mask) when the block starts at or before far_lim and all
    // 16 rows of the warp are real: the slow path then returns each score unchanged, so skipping it changes no bit.
    const bool real_rows = RW + 16 <= rows;
    const int far_lim = p.pos0 + RW / GQA - EXT - (BLK - 1);
    // warp w decodes block w / 2's keys (even w) or values (odd w) of stage grp into fragment buffer grp & 1
    auto decode = [&](int grp) {
        const int bb = warp >> 1;
        if (grp * PF_STAGE + bb >= nb) return;
        const char* blk = smc + L::STG + (size_t(grp & 1) * PF_STAGE + bb) * BLKB;
        uint4* const o = dec + ((grp & 1) * PF_STAGE + bb) * NFRAG * 32;
        F r;
        if ((warp & 1) == 0) {
            r.template load_k<false>(blk, lane, 0);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t a0[4], a1[4];
                r.k_frags(kk, a0, a1);
                o[(2 * kk) * 32] = make_uint4(a0[0], a0[1], a0[2], a0[3]);
                o[(2 * kk + 1) * 32] = make_uint4(a1[0], a1[1], a1[2], a1[3]);
            }
            dks[((grp & 1) * PF_STAGE + bb) * 32] = r.k_scale();
        } else {
            r.template load_v<false>(blk + F::KB, lane, 0);
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                uint32_t a[4];
                r.v_frag(dt, a);
                o[(8 + dt) * 32] = make_uint4(a[0], a[1], a[2], a[3]);
            }
        }
    };
    // [Agent] One barrier per stage: after it, stage g's fragments are complete, every warp is done with stage g - 1's
    // (their buffer takes stage g + 1), and stage g's codes are free for stage g + 2.
    issue(0, 0);
    if (ng > 1) issue(1, 1);
    if (ng > 1) cp_wait<1>();
    else cp_wait<0>();
    __syncthreads();
    decode(0);
#pragma unroll 1
    for (int grp = 0; grp < ng; ++grp) {
        cp_wait<0>();
        __syncthreads();
        if (grp + 2 < ng) issue(grp & 1, grp + 2);
        const uint4* const dcur = dec + (grp & 1) * PF_STAGE * NFRAG * 32;
        const float2* const kcur = dks + (grp & 1) * PF_STAGE * 32;
        const int nv = min(PF_STAGE, nb - grp * PF_STAGE);

        // ---- S^T = K Q^T of the 4 blocks: per accumulator the k-steps in attend_block's order
        float s[PF_STAGE][2][4];
#pragma unroll
        for (int bb = 0; bb < PF_STAGE; ++bb)
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) s[bb][nn][0] = s[bb][nn][1] = s[bb][nn][2] = s[bb][nn][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint4 a[PF_STAGE];
#pragma unroll
                for (int bb = 0; bb < PF_STAGE; ++bb) a[bb] = dcur[(bb * NFRAG + 2 * kk + j) * 32];
#pragma unroll
                for (int bb = 0; bb < PF_STAGE; ++bb)
#pragma unroll
                    for (int nn = 0; nn < 2; ++nn) mma_u4(s[bb][nn], a[bb], qb[nn][2 * kk + j][0], qb[nn][2 * kk + j][1]);
            }

        if (grp + 1 < ng) decode(grp + 1);

        // ---- scaled log2 scores, bias and mask, then each block's row maxima (independent across blocks)
        float mx[PF_STAGE][2][2];
#pragma unroll
        for (int bb = 0; bb < PF_STAGE; ++bb) {
            const int kbase = (grp * PF_STAGE + bb) * BLK;
            const float2 ks = kcur[bb * 32];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    float sc = s[bb][nn][v];
                    if constexpr (F::KOFF != 0.f) sc += st.qsum[nn][v & 1];
                    s[bb][nn][v] = sc * ((v >> 1) ? ks.y : ks.x) * F::SCORE;
                }
            if (!(real_rows && kbase <= far_lim)) {
#pragma unroll
                for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                    for (int v = 0; v < 4; ++v)
                        s[bb][nn][v] = fix(kbase, gid + (v >> 1) * 8, 8 * nn + 2 * t4 + (v & 1), s[bb][nn][v]);
            }
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float m = fmaxf(s[bb][nn][h], s[bb][nn][h + 2]);
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                    mx[bb][nn][h] = m;
                }
        }

        // ---- each block in key order: running maximum, rescale, P (FP16, transposed in registers), O^T += V^T P^T
#pragma unroll
        for (int bb = 0; bb < PF_STAGE; ++bb) {
            if (bb >= nv) break;
            bool grow = false;
            float fac[2][2];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float mn = fmaxf(st.m[nn][h], mx[bb][nn][h]);
                    fac[nn][h] = exp2f(st.m[nn][h] - mn);
                    grow |= mn > st.m[nn][h];
                    st.m[nn][h] = mn;
                    st.l[nn][h] *= fac[nn][h];
                }
            if (__any_sync(0xffffffffu, grow)) {
#pragma unroll
                for (int dt = 0; dt < 8; ++dt)
#pragma unroll
                    for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                        for (int v = 0; v < 4; ++v) st.acc[dt][nn][v] *= fac[nn][v & 1];
            }
            uint32_t pb[2][2];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int k8 = 0; k8 < 2; ++k8) {
                    const __half h0 = __float2half_rn(ex2_ftz(s[bb][nn][2 * k8] - st.m[nn][0]));
                    const __half h1 = __float2half_rn(ex2_ftz(s[bb][nn][2 * k8 + 1] - st.m[nn][1]));
                    st.l[nn][0] += __half2float(h0);
                    st.l[nn][1] += __half2float(h1);
                    pb[nn][k8] = transpose8x8(u_of(__halves2half2(h0, h1)));
                }
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                const uint4 a = dcur[(bb * NFRAG + 8 + dt) * 32];
#pragma unroll
                for (int nn = 0; nn < 2; ++nn) mma_u4(st.acc[dt][nn], a, pb[nn][0], pb[nn][1]);
            }
        }
    }
    __syncthreads();  // the epilogue's buffer overlaps the last stage's fragments

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
        [[maybe_unused]] static const bool attr = [] {
            CUDA_CHECK(cudaFuncSetAttribute(lb_prefill_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            int(PfSmem<F>::BYTES)));
            return true;
        }();
        lb_prefill_kernel<F><<<dim3((GQA * a.T + PF_ROWS - 1) / PF_ROWS, HKV), NT, PfSmem<F>::BYTES, stream>>>(a);
    });
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::attn
