// Full-attention decode (M = 1..8 tokens) and row packing over low-bit KV caches (int4, int8, k8v4, fp4); prefill in
// attn_lb_prefill.cu, the shared formats and block step in attn_lb.cuh. Contract: native/KERNELS.md, area `attn`.
//
// Design
// ------
// * Same arithmetic as attn.cu's full-layer path (attn_common.cuh: q / rel post-processing, dynamic conv, key clip,
//   value norm; relative bias over the last 1024 positions, sink logit, sigmoid gate; the call's own keys attended
//   exactly as the cache will hold them), with the cache in 16-position blocks per KV head written in mma operand
//   order (kv_format.h).
// * A format is a key code and a value code (K4 / K8 / KF, V4 / V8 / VF in attn_lb.cuh). Every format codes rotated rows, so
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
#include "kernels/attn/attn_lb.cuh"

namespace ingine::kernels::attn {
namespace {

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
