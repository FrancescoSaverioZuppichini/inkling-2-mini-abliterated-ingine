// Grouped expert GEMMs of the prefill (model._moe_grouped) over NVFP4 / MXFP8 experts decoded on the fly.
//
// Design. A CTA computes up to 128 activation rows x 128 weight rows, K in steps of 64, mma.m16n8k16 BF16 with
// FP32 accumulation. The 8 warps split the weight rows (16 each) and every warp covers all of the tile's
// activation rows, so each weight element is decoded exactly once per CTA, straight from the staged checkpoint
// codes into the MMA B fragments in registers (no BF16 weight tile in shared memory):
//   NVFP4: bf16((e2m1 * e4m3 block scale) * factor)      (model.decode_experts), looked up in a per-tile table
//   MXFP8: bf16(e4m3 * 2^(e8m0 - 127))                    (model.decode_experts)
// Activation rows (gathered by pair for the experts) and the weight codes stream in through 16-byte cp.async:
// activations in 3 stages of 64 elements, the codes in 64-byte row pieces (NVFP4: 128 elements plus their 8 block
// scales in the row padding), several slots ahead, so DRAM reads whole lines and small chunks, which are bound by
// the weight bytes, keep enough loads in flight. One barrier per step. Code rows are padded in shared memory so the
// 8 rows of a fragment load hit distinct banks. m16 row blocks past the tile's row count skip their MMAs
// (zero-filled copies).
// The per-element products and the k order of the FP32 sums are those of the BF16 MMA over decoded rows.
// Epilogues: gate/up (w13 rows interleaved, 2i gate and 2i + 1 up, so each thread holds gate and up of the same
// i): act = bf16(softcap(bf16 g, 4) * sigmoid(bf16 g) * softcap(bf16 u, 25)); down: bf16 per pair placed at
// t * 8 + slot (moe_combine sums the slots in order: deterministic).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace ingine::kernels::prefill {
namespace {

enum class Mode { Up, Down };
constexpr int FMT_NVFP4 = 0, FMT_MXFP8 = 1;

constexpr int TN = 128, BK = 64;
constexpr int AS = 3;  // activation stages (BK elements each)
constexpr int NT = 256;
constexpr int KCH = BK / 8;  // 16-byte chunks per activation tile row

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

template <int FMT, int K, int TM>
struct Cfg {
  static constexpr int CODE_ROW = FMT == FMT_NVFP4 ? K / 2 : K;  // checkpoint bytes per weight row
  static constexpr int SCALE_ROW = FMT == FMT_NVFP4 ? K / 16 : K / 32;
  // weight chunk: 64 code bytes per row (NVFP4 128 elements, FP8 64)
  static constexpr int WK = FMT == FMT_NVFP4 ? 128 : 64;
  static constexpr int R = WK / BK;  // activation steps per weight chunk
  static constexpr int KC = K / WK;  // weight chunks
  // weight slots: the chunk of step s + 1 must have landed (and be published by the barrier) at step s, when its
  // B fragments are decoded under step s's MMAs: R (WS - 1) >= AS. Small-row tiles prefetch further.
  static constexpr int WS = TM <= 32 ? (FMT == FMT_NVFP4 ? 6 : 7) : (FMT == FMT_NVFP4 ? 3 : 4);
  static_assert(R * (WS - 1) >= AS);
  static constexpr int WCODE = FMT == FMT_NVFP4 ? WK / 2 : WK;  // code bytes per row per chunk
  // padded: fragment loads on distinct banks; NVFP4 keeps the chunk's 8 block scales of the row in the padding
  static constexpr int WSTRIDE = WCODE + 16;
  static constexpr int W_BYTES = TN * WSTRIDE;
  static constexpr int A_BYTES = TM * BK * 2;
  static constexpr int SC_BYTES = FMT == FMT_NVFP4 ? 0 : TN * SCALE_ROW;  // MXFP8: block scales of the whole tile
  static constexpr int OFF_W = AS * A_BYTES, OFF_SC = OFF_W + WS * W_BYTES;
  // NVFP4: BF16 value of every (block scale code, E2M1 code) pair for this tile's expert, 256 x 16
  static constexpr int OFF_TBL = (OFF_SC + SC_BYTES + 31) / 32 * 32;
  static constexpr int SMEM = OFF_TBL + (FMT == FMT_NVFP4 ? 256 * 16 * 2 : 0);
  static_assert(K % WK == 0 && SMEM <= 101376);
};

struct GemmArgs {
  const uint8_t* codes = nullptr;   // NVFP4 / MXFP8 expert codes [128, N, ..]
  const uint8_t* scales = nullptr;  // block scales
  const float* factors = nullptr;   // NVFP4 per-expert factor
  const uint16_t* a = nullptr;      // activations (bf16)
  MoeWork w;                        // tiles and pair lists
  uint16_t* out = nullptr;
};

__device__ __forceinline__ void cp_async4(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(smem_dst)), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async8(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_addr(smem_dst)), "l"(gmem_src));
}

__device__ __forceinline__ float softcap(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// One activation step of MMAs for the first MT 16-row blocks: per k16 group, the A fragments by ldmatrix, then the
// MMAs, then `after(kk)` (the next step's B decoding, which runs under the issued MMAs).
template <int MT, int TM, class After>
__device__ __forceinline__ void mma_step(float (&acc)[TM / 16][2][4], unsigned a_addr, const unsigned (&a_off)[BK / 16],
                                         const unsigned (&b)[BK / 16][2][2], After&& after) {
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    unsigned a[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)  // 16 rows further down = 16 * BK * 2 bytes, the swizzle repeats every 8 rows
      ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], a_addr + a_off[kk] + mt * 16 * BK * 2);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      mma_bf16(acc[mt][0], a[mt], b[kk][0][0], b[kk][0][1]);
      mma_bf16(acc[mt][1], a[mt], b[kk][1][0], b[kk][1][1]);
    }
    after(kk);
  }
}

// Up: A = x rows of the pairs' tokens (K = H), W = w13 (N = 2I), out act [pairs, I].
// Down: A = act rows (K = I), W = w2 (N = H), out down [T * 8, H] at pair_dst.
template <int FMT, Mode MODE, int K, int TM>
__global__ void __launch_bounds__(NT, 1) wgemm_kernel(const GemmArgs g) {
  constexpr int KS = K / BK;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;  // expert weight rows
  using C = Cfg<FMT, K, TM>;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* a_smem = smem;
  unsigned char* w_smem = smem + C::OFF_W;
  uint8_t* sc_smem = smem + C::OFF_SC;
  const unsigned char* tbl = smem + C::OFF_TBL;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // ---- tile: n pair rows (from pb) of expert e x weight rows n0..n0+127
  constexpr int NTILES_N = NE / TN;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = (blockIdx.x % NTILES_N) * TN;
  const uint8_t* code_tile = g.codes + (int64_t(e) * NE + n0) * C::CODE_ROW;
  const uint8_t* scale_tile = g.scales + (int64_t(e) * NE + n0) * C::SCALE_ROW;

  // activation rows of this thread's cp.async chunks: rows tid / 8 + 32 i, chunk tid % 8
  const uint16_t* arow[TM / 32];
#pragma unroll
  for (int i = 0; i < TM / 32; ++i) {
    const int r = (tid >> 3) + 32 * i;
    const int rr = r < n ? r : 0;
    if constexpr (MODE == Mode::Up) arow[i] = g.a + size_t(g.w.pair_token[pb + rr]) * K;
    else arow[i] = g.a + size_t(pb + rr) * K;
  }

  // [Agent] Stages and slots are tracked as running byte offsets and the global sources as running pointers: the
  // `% 3` of a 3-stage ring and the per-step 64-bit address rebuild cost ~3 instructions per weight in the K loop.
  const uint16_t* asrc[TM / 32];  // activation source of the next load (step j), this thread's 16 bytes
  unsigned adst[TM / 32];         // its byte offset inside a stage (swizzled)
  int asz[TM / 32];
#pragma unroll
  for (int i = 0; i < TM / 32; ++i) {
    const int r = (tid >> 3) + 32 * i, ch = tid & 7;
    asrc[i] = arow[i] + ch * 8, adst[i] = 2 * swz<KCH>(r, ch), asz[i] = r < n ? 16 : 0;
  }
  const unsigned a_smem_addr = smem_addr(a_smem);
  auto load_a = [&](unsigned stage) {
#pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
      cp_async16(a_smem_addr + stage + adst[i], asrc[i], asz[i]);
      asrc[i] += BK;
    }
  };
  constexpr int CPR = C::WCODE / 16;  // 16-byte pieces per row
  const uint8_t* wsrc[TN * CPR / NT];
  unsigned wdst[TN * CPR / NT];
#pragma unroll
  for (int i = 0; i < TN * CPR / NT; ++i) {
    const int id = tid + i * NT, row = id / CPR, ch = id % CPR;
    wsrc[i] = code_tile + int64_t(row) * C::CODE_ROW + ch * 16, wdst[i] = row * C::WSTRIDE + ch * 16;
  }
  const uint8_t* ssrc = scale_tile + int64_t(tid) * C::SCALE_ROW;
  // the next chunk in K order into `slot`
  auto load_w = [&](unsigned slot) {
    unsigned char* ws = w_smem + slot;
#pragma unroll
    for (int i = 0; i < TN * CPR / NT; ++i) {
      cp_async16(ws + wdst[i], wsrc[i]);
      wsrc[i] += C::WCODE;
    }
    if constexpr (FMT == FMT_NVFP4)
      if (tid < TN) {
        cp_async8(ws + tid * C::WSTRIDE + C::WCODE, ssrc);
        ssrc += C::WK / 16;
      }
  };
  auto ring = [](unsigned off, unsigned size, unsigned count) { return off + size == size * count ? 0u : off + size; };

  // MXFP8: the tile's block scales, with the first activation group
  for (int id = tid; id < C::SC_BYTES / 4; id += NT)
    cp_async4(sc_smem + id * 4, scale_tile + (id * 4 / C::SCALE_ROW) * int64_t(C::SCALE_ROW) + (id * 4) % C::SCALE_ROW);
#pragma unroll
  for (int j = 0; j < C::WS - 1; ++j)
    if (j < C::KC) load_w(j * C::W_BYTES);
  // group j: activation step j (plus whatever weight chunk was issued since the previous commit)
#pragma unroll
  for (int j = 0; j < AS - 1; ++j) {
    load_a(j * C::A_BYTES);
    cp_commit();
  }

  const int c = lane & 3;
  const float factor = FMT == FMT_NVFP4 ? __ldg(g.factors + e) : 1.f;
  // [Agent] NVFP4: a weight's BF16 value bf16((e2m1 * scale) * factor) depends only on its 4-bit code, its block's
  // 8-bit scale code and the expert's factor, so the tile builds all 256 x 16 values once (same expression, same
  // bits) and the decode becomes one shared-memory lookup per weight: no scale conversion, no FP32 multiplies, no
  // rounding in the loop. Published by the first barrier of the K loop.
  if constexpr (FMT == FMT_NVFP4) {
    const float s = e4m3x2_to_float2(uint16_t(tid)).x;  // thread = scale code
    uint4 w[2];
    uint32_t* wv = reinterpret_cast<uint32_t*>(w);
#pragma unroll
    for (int j = 0; j < 8; ++j) wv[j] = pack_bf16x2((kE2M1[2 * j] * s) * factor, (kE2M1[2 * j + 1] * s) * factor);
    uint4* dst = reinterpret_cast<uint4*>(smem + C::OFF_TBL + tid * 32);
    dst[0] = w[0], dst[1] = w[1];
  }
  // NVFP4 decode: bytes c (elements 2c, 2c + 1) of the k16 group's two code words, side by side
  const uint32_t sel_c = uint32_t(c) | uint32_t(c + 4) << 4;
  const int mact = min((n + 15) / 16, TM / 16);  // active m16 row blocks (same for every warp)
  float acc[TM / 16][2][4];
#pragma unroll
  for (int a = 0; a < TM / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  // B fragments of step ks (k16 group kk) decoded from the staged codes of its weight chunk: step ks reads chunk
  // ks / R from weight slot `slot`, part = ks % R (a template constant at every call). NVFP4 scales: the step's 4
  // scale codes per row, raw (they index the table); MXFP8: the 4 k16 groups' power-of-two factors.
  using Scales = std::conditional_t<FMT == FMT_NVFP4, uint32_t[2], float[2][4]>;
  auto scales_of = [&](int ks, unsigned slot, int part, Scales& sc) __attribute__((always_inline)) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      if constexpr (FMT == FMT_NVFP4) {
        sc[nt] = *reinterpret_cast<const uint32_t*>(w_smem + slot + row * C::WSTRIDE + C::WCODE + part * 4);
      } else {
        const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
        sc[nt][0] = sc[nt][1] = exp2f(float(ms[0]) - 127.f);
        sc[nt][2] = sc[nt][3] = exp2f(float(ms[1]) - 127.f);
      }
    }
  };
  auto decode = [&](unsigned slot, int part, int kk, const Scales& sc, unsigned (&b)[BK / 16][2][2]) __attribute__((always_inline)) {
    const unsigned char* ws = w_smem + slot;
    const int kc = part * BK + kk * 16;  // first element of the k16 group inside the chunk
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      if constexpr (FMT == FMT_NVFP4) {
        // elements 2c, 2c+1 (byte c) and 8 + 2c, +1 (byte 4 + c) of the k16 group: v = 4 codes, low nibble first;
        // table byte offset = scale code * 32 | code * 2
        const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + kc / 2);
        const uint32_t v = __byte_perm(wd.x, wd.y, sel_c);
        const uint32_t sb = ((sc[nt] >> (8 * kk)) & 0xffu) << 5;
        const uint32_t t0 = *reinterpret_cast<const uint16_t*>(tbl + (((v << 1) & 0x1eu) | sb));
        const uint32_t t1 = *reinterpret_cast<const uint16_t*>(tbl + (((v >> 3) & 0x1eu) | sb));
        const uint32_t t2 = *reinterpret_cast<const uint16_t*>(tbl + (((v >> 7) & 0x1eu) | sb));
        const uint32_t t3 = *reinterpret_cast<const uint16_t*>(tbl + (((v >> 11) & 0x1eu) | sb));
        b[kk][nt][0] = __byte_perm(t0, t1, 0x5410);
        b[kk][nt][1] = __byte_perm(t2, t3, 0x5410);
      } else {
        const float s = sc[nt][kk];
        const uint4 q = *reinterpret_cast<const uint4*>(ws + row * C::WSTRIDE + kc);
        const uint32_t w0 = (c >> 1) ? q.y : q.x, w1 = (c >> 1) ? q.w : q.z;
        const float2 f0 = e4m3x2_to_float2(uint16_t(w0 >> (16 * (c & 1))));
        const float2 f1 = e4m3x2_to_float2(uint16_t(w1 >> (16 * (c & 1))));
        b[kk][nt][0] = pack_bf16x2(f0.x * s, f0.y * s);
        b[kk][nt][1] = pack_bf16x2(f1.x * s, f1.y * s);
      }
      asm volatile("" : "+r"(b[kk][nt][0]), "+r"(b[kk][nt][1]));
    }
  };

  // Software pipeline: step ks's MMAs run on fragments decoded during step ks - 1; the decoding of step ks + 1
  // is interleaved with them (its chunk has landed: R (WS - 1) >= AS, and the barrier published it).
  // [Agent] The row-block count is a template constant of the whole K loop, not a switch inside it: a runtime
  // `mt < mact` test would be if-converted into predicated MMAs (8x the tensor work for a 16-row tile), and a switch
  // per step costs an indirect branch and splits the step into blocks the scheduler cannot interleave. The loop runs
  // two steps per trip with the fragment sets swapped (no copy of the 16 next-step registers) and peels the last
  // step, so every decode is unconditional and sits in one basic block with the MMAs it hides under.
  unsigned b0[BK / 16][2][2], b1[BK / 16][2][2];
  static_assert(KS % 2 == 0 && (C::R == 1 || C::R == 2) && KS > AS);
  unsigned a_off[BK / 16];  // ldmatrix row address of this lane per k16 group, rows 0..15 of a stage
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) a_off[kk] = 2 * swz<KCH>(lane & 15, kk * 2 + (lane >> 4));
  unsigned a_rd = 0, a_wr = (AS - 1) * C::A_BYTES;  // stage of step ks; stage step ks + AS - 1 loads into
  unsigned w_rd = 0, w_wr = (C::WS - 1) * C::W_BYTES;  // slot of chunk ks / R; slot of chunk ks / R + WS - 1
  // [Agent] The K-loop lambdas are force-inlined: for the MXFP8 kernels sm_86's larger software FP8 decode pushed
  // them past the inliner's size limit, the fragment arrays passed by reference then lived in an 800-byte stack
  // frame with local-memory loads and stores in every step, which made the RTX 3090's MoE GEMMs ~2x slower.
  auto kloop = [&](auto mt_c) __attribute__((always_inline)) {
    constexpr int MT = decltype(mt_c)::value;
    // par = ks % 2. first = step 0: its own fragments are decoded right after the barrier that publishes chunk 0.
    // tail = the last AS - 1 steps, which load no activations.
    auto step = [&](int ks, auto par_c, auto dec_c, auto first_c, auto tail_c, unsigned (&bc)[BK / 16][2][2],
                    unsigned (&bn)[BK / 16][2][2]) __attribute__((always_inline)) {
      constexpr int part = decltype(par_c)::value % C::R, part_n = (decltype(par_c)::value + 1) % C::R;
      constexpr bool DEC = decltype(dec_c)::value;
      cp_wait<AS - 2>();
      __syncthreads();
      // first step of a chunk: the chunk WS - 1 ahead into the slot the previous chunk used (read before the barrier)
      if (part == 0 && ks / C::R + C::WS - 1 < C::KC) {
        load_w(w_wr);
        w_wr = ring(w_wr, C::W_BYTES, C::WS);
      }
      if constexpr (!decltype(tail_c)::value) load_a(a_wr);
      cp_commit();
      a_wr = ring(a_wr, C::A_BYTES, AS);
      if constexpr (decltype(first_c)::value) {
        Scales sc0;
        scales_of(0, 0, 0, sc0);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) decode(0, 0, kk, sc0, bc);
      }
      const unsigned w_nx = part_n == 0 ? ring(w_rd, C::W_BYTES, C::WS) : w_rd;  // slot of step ks + 1
      Scales scn;
      if constexpr (DEC) scales_of(ks + 1, w_nx, part_n, scn);
      mma_step<MT, TM>(acc, a_smem_addr + a_rd, a_off, bc, [&](int kk) {
        if constexpr (DEC) decode(w_nx, part_n, kk, scn, bn);
      });
      a_rd = ring(a_rd, C::A_BYTES, AS);
      w_rd = w_nx;
    };
    using T0 = std::integral_constant<int, 0>;
    using T1 = std::integral_constant<int, 1>;
    using Y = std::true_type;
    using N = std::false_type;
    step(0, T0{}, Y{}, Y{}, N{}, b0, b1);
    step(1, T1{}, Y{}, N{}, N{}, b1, b0);
#pragma unroll 1
    for (int ks = 2; ks < KS - 2; ks += 2) {
      step(ks, T0{}, Y{}, N{}, N{}, b0, b1);
      step(ks + 1, T1{}, Y{}, N{}, N{}, b1, b0);
    }
    step(KS - 2, T0{}, Y{}, N{}, Y{}, b0, b1);
    step(KS - 1, T1{}, N{}, N{}, Y{}, b1, b0);
  };
  switch (mact) {
    case 1: if constexpr (1 <= TM / 16) kloop(std::integral_constant<int, 1>{}); break;
    case 2: if constexpr (2 <= TM / 16) kloop(std::integral_constant<int, 2>{}); break;
    case 3: if constexpr (3 <= TM / 16) kloop(std::integral_constant<int, 3>{}); break;
    case 4: if constexpr (4 <= TM / 16) kloop(std::integral_constant<int, 4>{}); break;
    case 5: if constexpr (5 <= TM / 16) kloop(std::integral_constant<int, 5>{}); break;
    case 6: if constexpr (6 <= TM / 16) kloop(std::integral_constant<int, 6>{}); break;
    case 7: if constexpr (7 <= TM / 16) kloop(std::integral_constant<int, 7>{}); break;
    default: kloop(std::integral_constant<int, TM / 16>{}); break;
  }
  cp_wait<0>();

#pragma unroll
  for (int mt = 0; mt < TM / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * 16 + (lane >> 2) + half * 8;
      if (row >= n) continue;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cl = warp * 16 + nt * 8 + c * 2;  // column inside the tile
        const float c0 = acc[mt][nt][2 * half], c1 = acc[mt][nt][2 * half + 1];
        if constexpr (MODE == Mode::Up) {
          const float gt = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + row) * kInter + (n0 + cl) / 2] = f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + row];
          *reinterpret_cast<uint32_t*>(g.out + size_t(dst) * kHidden + n0 + cl) = pack_bf16x2(c0, c1);
        }
      }
    }
  }
}

template <int FMT, Mode MODE, int K, int TM>
void launch(const GemmArgs& g, int grid, cudaStream_t stream) {
  constexpr int SMEM = Cfg<FMT, K, TM>::SMEM;
  static const cudaError_t attr =
      cudaFuncSetAttribute(wgemm_kernel<FMT, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm: shared memory attribute failed");
  wgemm_kernel<FMT, MODE, K, TM><<<grid, NT, SMEM, stream>>>(g);
}

template <int FMT, Mode MODE, int K>
void launch_rows(const GemmArgs& g, int tm, int grid_per_row_tile, int row_tiles, cudaStream_t stream) {
  if (tm == 32) launch<FMT, MODE, K, 32>(g, grid_per_row_tile * row_tiles, stream);
  else launch<FMT, MODE, K, 128>(g, grid_per_row_tile * row_tiles, stream);
}

GemmArgs expert_args(const void* codes, const void* scales, const float* factors, const __nv_bfloat16* a,
                     const MoeWork& w, __nv_bfloat16* out) {
  GemmArgs g;
  g.codes = static_cast<const uint8_t*>(codes);
  g.scales = static_cast<const uint8_t*>(scales);
  g.factors = factors;
  g.a = reinterpret_cast<const uint16_t*>(a);
  g.w = w;
  g.out = reinterpret_cast<uint16_t*>(out);
  return g;
}

}  // namespace

void moe_gate_up(const MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, __nv_bfloat16* act,
                 cudaStream_t stream) {
  const GemmArgs g = expert_args(e.w13, e.w13_scale, e.w13_factor, x, w, act);
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
  else launch_rows<FMT_MXFP8, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
}

void moe_down(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, __nv_bfloat16* down,
              cudaStream_t stream) {
  const GemmArgs g = expert_args(e.w2, e.w2_scale, e.w2_factor, act, w, down);
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
  else launch_rows<FMT_MXFP8, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
}

}  // namespace ingine::kernels::prefill
