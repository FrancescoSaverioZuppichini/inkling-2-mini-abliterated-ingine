// proj, proj_conv, head, fp8_rows, int8_rows: weight-streaming GEMVs for M = 1..8 tokens.
//
// Design. Decode GEMVs are pure weight streams (M <= 8 activations, K = 2304 or 3072): the goal is to keep
// DRAM busy with the fewest instructions per weight byte. Two kernels share the epilogues:
//
// M = 1, `gemv_kernel` (CUDA cores):
// * One row per lane group: a full warp per BF16 row, a half warp per FP8 or INT8 row, so every lane owns exactly
//   CH = K * bytes / (16 * LANES) 16-byte chunks (BF16 K=2304: 9; FP8 K=2304: 9; K=3072: 12). Chunk i of
//   lane l is bytes [(i * LANES + l) * 16, +16) of the row: each load instruction of a group reads one
//   contiguous 256/512-byte span (fully coalesced).
// * The CH chunk loads of a row are issued at the top of the kernel (read-only streaming loads, no L1
//   allocation; ptxas interleaves them with the FMAs of earlier chunks, leaving ~4 chunks = 2 KB in
//   flight per warp), and with ~32 resident warps per SM (low register count) that is ~64 KB per SM,
//   several times the bandwidth-delay product. (Measured alternatives, no gain: persistent grids, L2 bulk
//   prefetch of the row, 128/512-thread blocks.)
// * The activation is read with 16-byte L1-cached loads at the matching columns; FP32 FMAs, then a
//   warp reduction.
//
// M >= 2, `gemv_mma_kernel` (tensor cores): on CUDA cores every weight costs M FMAs plus the unpacking of
// M activations, which made M = 8 instruction-bound (FP8 proj lost 20-40%; staging the activations in
// shared memory or giving each lane two rows was measured and did not help: both raise register or
// shared-memory pressure and cut occupancy). Instead each warp runs mma.m16n8k16 (BF16 x BF16 -> FP32):
// A = 16 weight rows, B = the activations of up to 8 tokens (N = 8 = max M; missing tokens are zero),
// so one instruction does 16 x 8 x 16 multiply-adds with FP32 accumulation, exact BF16 products as on the
// CUDA-core path. The reduction order over k is free, so the fragments are fed straight from coalesced
// 16-byte loads: lane (g, t) loads columns [c + CPL t, +CPL) of rows g and g + 8 (CPL = 8 BF16 or 16
// FP8 converted exactly to BF16) and the same columns of token g's activation; any k-permutation that A
// and B share gives the same dot products. A block of SPLITK warps owns 16 rows; warp w covers a 1/SPLITK
// slice of K (enough warps in flight for small row counts), and the partial 16 x 8 tiles are summed in
// shared memory in a fixed order. proj blocks that straddle a segment boundary with different inputs run
// one pass per input with the other rows masked to zero.
//
// Epilogues: proj rounds to BF16 (input vector per row from the device-side segment table), head applies
// the softcap, proj_conv applies the post-attention causal conv for its channel (= row) over the call's
// M tokens, rewrites that channel's conv state and optionally records the conv inputs (conv_hist), so
// every op is one launch with no grid-wide dependency. M is a template parameter; FP8 and INT8 rows multiply
// the dot by the row scale.
//
// Programmatic dependent launch (pdl=True): the M = 1 kernel issues all its row's weight loads, the
// M >= 2 kernel its first batch, before griddepcontrol.wait, so they overlap the previous kernel's tail
// (pdl_wait_fenced: a plain wait lets ptxas sink the loads below it; measured M = 1 proj 13.7 -> 11.8 us
// behind a 2 us predecessor); activations are read (coherent, non-hoistable loads: ld_act) and
// outputs/state written only after it, and the next kernel is released right after the wait.
// L2: the weight rows are read once per step, so their lines are filled evict-first (kernels/common/l2.cuh):
// they stop evicting the activations, KV rings and the small kernels' evict-last constants.
// Ported from engine/csrc/dense/gemv.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include "kernels/common/fp8x.cuh"
#include "kernels/common/l2.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

#include <cstdint>
#include <type_traits>

namespace ingine::kernels::dense {
namespace {

constexpr l2::Evict WEIGHT_L2 = l2::Evict::first;

// Weight formats of a Matrix: BF16 rows, FP8 E4M3 rows + row scale, INT8 rows + row scale.
constexpr int F_BF16 = 0, F_FP8 = 1, F_INT8 = 2;

// [Agent] INT8 rows (layout in dense.h) are sign-magnitude codes placed so that one AND turns elements 0 and 1 of a
// 4-byte word into two BF16 words with the sign at bit 15 and |q| in the low mantissa bits, and a byte rotation plus
// the same AND does it for elements 2 and 3. Such a word is the subnormal BF16 |q| * 2^-133; one BF16x2 FMA by 2^127
// (+ -0, which keeps the sign of zero) makes it exactly q * 2^-6: 7 significant bits fit BF16, and BF16 math keeps
// subnormals (the sm_86 E4M3 decode in fp8x.cuh relies on the same). 2.5 instructions per pair against 5 for E4M3
// on sm_86; the 2^6 moves into the row scale, a power of two, so every product and sum keeps its bits.
constexpr float INT8_UNIT = 64.f;
// elements 0, 1 (lo) and 2, 3 (hi) of an INT8 word as BF16x2, value q * 2^-6
__device__ __forceinline__ uint32_t i8_lo_bf16x2(uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(b & 0x807f807fu), "r"(0x7f007f00u), "r"(0x80008000u));
  return r;
}
__device__ __forceinline__ uint32_t i8_hi_bf16x2(uint32_t w) { return i8_lo_bf16x2(__funnelshift_l(w, w, 8)); }

__device__ __forceinline__ void unpack_i8x16(const uint4& v, float* f) {
  const uint32_t ws[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t lo = i8_lo_bf16x2(ws[i]), hi = i8_hi_bf16x2(ws[i]);
    f[4 * i] = __uint_as_float(lo << 16), f[4 * i + 1] = __uint_as_float(lo & 0xffff0000u);
    f[4 * i + 2] = __uint_as_float(hi << 16), f[4 * i + 3] = __uint_as_float(hi & 0xffff0000u);
  }
}

template <int FMT>
__device__ __forceinline__ float row_scale(const float* w_scale, size_t row) {
  if constexpr (FMT == F_BF16) return 1.f;
  else if constexpr (FMT == F_FP8) return __ldg(w_scale + row);
  else return __ldg(w_scale + row) * INT8_UNIT;
}

// ---- epilogues ----------------------------------------------------------------------------------------
// input_of(row): input index of a row (-1: row not computed); x_of(i): that input, bf16 [M, K].
// kScatter epilogues store one (row, token) value at a time (store1); proj_conv needs every token of a
// row together (store: lane-parallel for the CUDA-core kernel; store_row: one thread).

template <int M>
struct ProjEpi {
  static constexpr int kInputs = 3;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* xin;  // [3, M, H]
  const int* segments;       // [S, 3]
  int nseg, R;
  float* out;                // [M, R]
  __device__ int input_of(int row) const {
    for (int s = 0; s < nseg; ++s) {
      const int b = __ldg(segments + 3 * s), e = __ldg(segments + 3 * s + 1);
      if (row >= b && row < e) return __ldg(segments + 3 * s + 2);
    }
    return -1;
  }
  __device__ const __nv_bfloat16* x_of(int i) const { return xin + size_t(i) * M * H; }
  __device__ void store1(int row, int m, float v) const { out[size_t(m) * R + row] = round_bf16(v); }
};

template <int M>
struct HeadEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  int V;
  float* out;  // [M, V]
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const {
    out[size_t(m) * V + row] = tanhf(round_bf16(v) / 30.f) * 30.f;
  }
};

// plain GEMV (drafter): one input x [M, K] (row stride K), out [M, R] = bf16(w . x)
template <int M>
struct GemvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  int R;
  float* out;  // [M, R]
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const { out[size_t(m) * R + row] = round_bf16(v); }
};

template <int M>
struct ConvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = false;  // the conv of token m needs y of tokens m-3 .. m
  const __nv_bfloat16* x;  // [M, 3072]
  const float* conv_w;     // [H, 4], tap 3 = current token
  float* state;            // [3, H], oldest first
  float* out;              // [M, H]
  float* hist;             // [M, H] BF16-rounded conv inputs y, or null
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void load_ext(int c, const float (&acc)[M], float (&ext)[M + 3]) const {
#pragma unroll
    for (int j = 0; j < 3; ++j) ext[j] = state[j * H + c];
#pragma unroll
    for (int m = 0; m < M; ++m) ext[3 + m] = round_bf16(acc[m]);
  }
  __device__ void token(int c, int m, const float (&ext)[M + 3]) const {
    const float4 cw = __ldg(reinterpret_cast<const float4*>(conv_w) + c);
    const float s = cw.x * ext[m] + cw.y * ext[m + 1] + cw.z * ext[m + 2] + cw.w * ext[m + 3];
    out[size_t(m) * H + c] = (s + ext[m + 3]) * (1.f / sqrtf(5.f));
    if (hist) hist[size_t(m) * H + c] = ext[m + 3];
  }
  // lane-parallel (all lanes of the group hold every token's sum)
  __device__ void store(int c, const float (&acc)[M], int gl, unsigned mask) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m)
      if (gl == m) token(c, m, ext);
    __syncwarp(mask);  // the group has read the old state before it is overwritten
#pragma unroll
    for (int j = 0; j < 3; ++j)
      if (gl == 8 + j) state[j * H + c] = ext[M + j];
  }
  // one thread per channel
  __device__ void store_row(int c, const float (&acc)[M]) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m) token(c, m, ext);
#pragma unroll
    for (int j = 0; j < 3; ++j) state[j * H + c] = ext[M + j];
  }
};

// ---- M = 1: CUDA cores --------------------------------------------------------------------------------

template <int FMT, int K, int LANES, int BLOCK, class Epi, bool GATHER = false>
__global__ void __launch_bounds__(BLOCK) gemv_kernel(const uint8_t* __restrict__ w,
                                                     const float* __restrict__ w_scale, int rows, uint32_t zero,
                                                     Epi epi, const int* __restrict__ row_map) {
  constexpr int EPC = FMT != F_BF16 ? 16 : 8;  // elements per 16-byte chunk
  constexpr int CH = K / (EPC * LANES);
  static_assert(K % (EPC * LANES) == 0, "row must split evenly over the lane group");
  constexpr size_t ROW_BYTES = size_t(K) * (FMT != F_BF16 ? 1 : 2);

  const int gl = threadIdx.x % LANES;
  const int row = (blockIdx.x * BLOCK + threadIdx.x) / LANES;
  // A whole group shares `row`, so groups exit together and the shuffles below stay within live groups.
  if (row >= rows) return;
  const int in = epi.input_of(row);
  if (in < 0) return;
  const __nv_bfloat16* x = epi.x_of(in);  // pointer arithmetic only: the segment table is a constant

  const size_t wrow = GATHER ? size_t(__ldg(row_map + row)) : size_t(row);
  const uint8_t* wr = w + wrow * ROW_BYTES;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  uint4 wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * LANES + gl) * 16, wpol);
  pdl_wait_fenced(zero);  // the whole row is in flight; the activation comes from an earlier kernel
  pdl_trigger();

  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * LANES + gl) * EPC;
    float wf[EPC], xf[EPC];
    if constexpr (FMT == F_FP8) unpack_e4m3x16(wv[i], wf);
    else if constexpr (FMT == F_INT8) unpack_i8x16(wv[i], wf);
    else unpack_bf16x8(wv[i], wf);
    unpack_bf16x8(ld_act(x + col), xf);
    if constexpr (EPC == 16) unpack_bf16x8(ld_act(x + col + 8), xf + 8);
#pragma unroll
    for (int j = 0; j < EPC; ++j) acc = fmaf(wf[j], xf[j], acc);
  }

  float v[1] = {group_sum<LANES>(acc) * row_scale<FMT>(w_scale, wrow)};
  if constexpr (Epi::kScatter) {
    if (gl == 0) epi.store1(row, 0, v[0]);
  } else {
    epi.store(row, v, gl, group_mask<LANES>());
  }
}

// ---- M >= 2: tensor cores -----------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#ifndef MMA_U_BF16
#define MMA_U_BF16 6
#endif
#ifndef MMA_U_FP8
#define MMA_U_FP8 3
#endif
// `zero` is 0 at run time (opaque to the compiler): every mma of a batch takes an operand xor-ed with
// (xor of all the batch's loads) & zero, so ptxas cannot interleave the mma with the batch's loads and
// all U steps' loads of a warp are in flight together (otherwise it issues one step's loads at a time,
// ~1 KB per warp, and scattered 64-byte requests from 16 rows run at ~40% of DRAM bandwidth).
// [Agent] An SM schedules warps in four quarters of 16K registers each. On sm_89 ptxas gives the M >= 2 kernel up to
// 108 registers, so a quarter fits only 4 warps of 6-warp blocks: 2 blocks per SM, and proj's 538 blocks run in 2.1
// waves at 80 % of DRAM. Capped at 96, a quarter holds 5 warps, 3 blocks fit, and the M = 8 calls on the RTX 4090 drop
// 3.6 % (proj), 7.6 % (proj_conv), 12.8 % (head); register allocation does not change any result bit.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define GEMV_MMA_BOUNDS(threads) __maxnreg__(96)
#else
#define GEMV_MMA_BOUNDS(threads) __launch_bounds__(threads)
#endif
template <int FMT, int K, int M, int SPLITK, int U, class Epi, bool GATHER = false>
__global__ void GEMV_MMA_BOUNDS(32 * SPLITK) gemv_mma_kernel(const uint8_t* __restrict__ w,
                                                              const float* __restrict__ w_scale, int rows,
                                                              uint32_t zero, Epi epi,
                                                              const int* __restrict__ row_map) {
  constexpr int CPL = FMT != F_BF16 ? 16 : 8;  // columns per lane per 16-byte weight load
  constexpr int SPAN = 4 * CPL;      // columns per step (4 lanes t)
  constexpr int KS = K / SPLITK;     // columns per warp
  static_assert(KS % (SPAN * U) == 0, "K slice must split into unrolled steps");
  constexpr int NJ = KS / SPAN;
  constexpr size_t ROW_BYTES = size_t(K) * (FMT != F_BF16 ? 1 : 2);
  __shared__ float cs[SPLITK][16][8];

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const int ra = r0 + g, rb = r0 + g + 8;
  const int ia = ra < rows ? epi.input_of(ra) : -1, ib = rb < rows ? epi.input_of(rb) : -1;
  const int c0 = warp * KS + t * CPL;
  auto wrow = [&](int r) { return GATHER ? size_t(__ldg(row_map + r)) : size_t(r); };
  const size_t wra = GATHER && ia >= 0 ? wrow(ra) : size_t(ra), wrb = GATHER && ib >= 0 ? wrow(rb) : size_t(rb);

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  bool waited = false;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  auto pass = [&](int cur) {
    const bool la = ia == cur, lb = ib == cur, lx = g < M;
    const uint8_t* wa = w + (la ? wra : 0) * ROW_BYTES;
    const uint8_t* wb = w + (lb ? wrb : 0) * ROW_BYTES;
    const __nv_bfloat16* xg = epi.x_of(cur) + size_t(lx ? g : 0) * K;
#pragma unroll 1
    for (int jb = 0; jb < NJ; jb += U) {
      uint4 A[U], B[U], X[U][CPL / 8];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const size_t off = size_t(c0 + (jb + u) * SPAN) * (FMT != F_BF16 ? 1 : 2);
        A[u] = la ? l2::ld_stream<WEIGHT_L2>(wa + off, wpol) : make_uint4(0, 0, 0, 0);
        B[u] = lb ? l2::ld_stream<WEIGHT_L2>(wb + off, wpol) : make_uint4(0, 0, 0, 0);
      }
      if (!waited) {  // the first batch of weights is in flight; activations come from earlier kernels
        pdl_wait_fenced(zero);
        pdl_trigger();
        waited = true;
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col = c0 + (jb + u) * SPAN;
#pragma unroll
        for (int h = 0; h < CPL / 8; ++h) X[u][h] = lx ? ld_act(xg + col + 8 * h) : make_uint4(0, 0, 0, 0);
      }
      uint32_t dep = 0;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        dep ^= A[u].x ^ A[u].w ^ B[u].x ^ B[u].w;
#pragma unroll
        for (int h = 0; h < CPL / 8; ++h) dep ^= X[u][h].x;
      }
      dep &= zero;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if constexpr (FMT == F_INT8) {
          // as FP8 below, each word's pairs decoded with the INT8 AND / FMA
          const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
          const uint32_t xw[8] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w,
                                  X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s)
            mma_bf16(c, i8_lo_bf16x2(aw[s]), i8_lo_bf16x2(bw[s]), i8_hi_bf16x2(aw[s]), i8_hi_bf16x2(bw[s]),
                     xw[2 * s] ^ dep, xw[2 * s + 1]);
        } else if constexpr (FMT == F_FP8) {
          // 16 columns = 4 k-steps of 4 columns (2 k-pairs) each
          const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
          const uint32_t xw[8] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w,
                                  X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s)
            mma_bf16(c, fp8x::e4m3x2_to_bf16x2(aw[s]), fp8x::e4m3x2_to_bf16x2(bw[s]), fp8x::e4m3x2_to_bf16x2(aw[s] >> 16),
                     fp8x::e4m3x2_to_bf16x2(bw[s] >> 16), xw[2 * s] ^ dep, xw[2 * s + 1]);
        } else {
          mma_bf16(c, A[u].x, B[u].x, A[u].y, B[u].y, X[u][0].x ^ dep, X[u][0].y);
          mma_bf16(c, A[u].z, B[u].z, A[u].w, B[u].w, X[u][0].z ^ dep, X[u][0].w);
        }
      }
    }
  };
  if constexpr (Epi::kInputs == 1) {
    pass(0);
  } else {
    // inputs present among the block's 16 rows (warp-uniform; every warp sees the same rows)
#pragma unroll 1
    for (int i = 0; i < Epi::kInputs; ++i)
      if (__any_sync(0xffffffffu, ia == i || ib == i)) pass(i);
  }

  // C fragment: c0, c1 = rows g, tokens 2t, 2t+1; c2, c3 = row g + 8
  cs[warp][g][2 * t] = c[0];
  cs[warp][g][2 * t + 1] = c[1];
  cs[warp][g + 8][2 * t] = c[2];
  cs[warp][g + 8][2 * t + 1] = c[3];
  __syncthreads();
  auto total = [&](int r, int m) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < SPLITK; ++k) s += cs[k][r][m];
    return s;
  };
  if constexpr (Epi::kScatter) {
    for (int k = threadIdx.x; k < 16 * M; k += 32 * SPLITK) {
      const int r = k / M, m = k % M, row = r0 + r;
      if (row < rows && epi.input_of(row) >= 0)
        epi.store1(row, m, total(r, m) * row_scale<FMT>(w_scale, wrow(row)));
    }
  } else {
    const int r = threadIdx.x, row = r0 + r;
    if (r < 16 && row < rows) {
      const float scale = row_scale<FMT>(w_scale, wrow(row));
      float y[M];
#pragma unroll
      for (int m = 0; m < M; ++m) y[m] = total(r, m) * scale;
      epi.store_row(row, y);
    }
  }
}

// ---- launch helpers -----------------------------------------------------------------------------------

// [Agent] INT8 rows at M = 1 decode on CUDA cores at ~4.3 instructions per weight. That hides behind the DRAM stream
// only where the GPU issues at least one warp instruction per byte DRAM delivers (4 schedulers x SMs x clock /
// bandwidth: ~1.3 on an RTX 4090, ~0.6 on an RTX 3090); below that the tensor-core kernel (~1.4 per weight) wins.
bool int8_m1_tensor_cores() {
  static const bool tc = [] {
    int dev = 0, sms = 0, clk_khz = 0, mem_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    const double issue_per_s = 4.0 * sms * clk_khz * 1e3, bytes_per_s = 2.0 * mem_khz * 1e3 * bus_bits / 8;
    return issue_per_s < bytes_per_s;
  }();
  return tc;
}

// BLOCK: threads per block of the M = 1 kernel (0: M = 1 runs the tensor-core kernel too, for K whose
// whole row per lane would not fit in registers); SPLITK: warps per 16-row block of the M >= 2 kernel;
// UFP8: steps per load batch of its FP8 and INT8 instantiations; I8_TC: INT8 rows at M = 1 may run the tensor-core
// kernel (int8_m1_tensor_cores).
template <int K, int BLOCK, int SPLITK, int UFP8, template <int> class Epi, bool GATHER = false, bool I8_TC = false,
          class Make>
void launch_gemv(const Matrix& w, int M, bool pdl, cudaStream_t stream, Make make) {
  const int fmt = w.scale == nullptr ? F_BF16 : w.int8 ? F_INT8 : F_FP8;
  require(w.w != nullptr && w.rows > 0, "weight matrix is empty");
  require(!w.int8 || w.scale != nullptr, "INT8 rows need their row scales");
  require(GATHER == (w.row_map != nullptr), "row_map: only proj reads mapped rows");
  const int* rm = w.row_map;
  const int rows = w.rows;
  const float* sc = w.scale;
  const uint8_t* wp = static_cast<const uint8_t*>(w.w);
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    auto epi = make(std::integral_constant<int, MM>{});
    using E = decltype(epi);
    if constexpr (MM == 1 && BLOCK > 0) {
      constexpr int B = BLOCK > 0 ? BLOCK : 32;  // (the branch is discarded for BLOCK = 0)
      if (fmt == F_FP8) launch_ex(gemv_kernel<F_FP8, K, 16, B, E, GATHER>, dim3((rows * 16 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, rm);
      else if (fmt == F_INT8) {
        if (I8_TC && int8_m1_tensor_cores()) launch_ex(gemv_mma_kernel<F_INT8, K, MM, SPLITK, UFP8, E, GATHER>, dim3((rows + 15) / 16), 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, rm);
        else launch_ex(gemv_kernel<F_INT8, K, 16, B, E, GATHER>, dim3((rows * 16 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, rm);
      }
      else launch_ex(gemv_kernel<F_BF16, K, 32, B, E, GATHER>, dim3((rows * 32 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, rm);
    } else {
      const dim3 blocks((rows + 15) / 16);
      if (fmt == F_FP8) launch_ex(gemv_mma_kernel<F_FP8, K, MM, SPLITK, UFP8, E, GATHER>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, rm);
      else if (fmt == F_INT8) launch_ex(gemv_mma_kernel<F_INT8, K, MM, SPLITK, UFP8, E, GATHER>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, rm);
      else launch_ex(gemv_mma_kernel<F_BF16, K, MM, SPLITK, MMA_U_BF16, E, GATHER>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi, rm);
    }
  };
  switch (M) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    case 8: go(std::integral_constant<int, 8>{}); break;
    default: require(false, "M must be in 1..8");
  }
}

// Warps per 16-row block of the M >= 2 kernel (K slice per warp must hold a whole number of U-step
// batches). Measured at M = 2 and 8 (shared GPU): proj 6 > 4 > 2 (FP8 +4-8%), head 6 ~ 4 >> 2 >> 1
// (+20-30%: 12.5K blocks need short K slices to keep enough rows in flight), proj_conv 4 ~ 8 ~ 16.
#ifndef PROJ_SPLITK
#define PROJ_SPLITK 6
#endif
#ifndef CONV_SPLITK
#define CONV_SPLITK 8
#endif
// FP8 steps per load batch of proj_conv's M >= 2 kernel: 6 = a warp's whole K slice. Its 144 blocks are all
// resident at once (2 per SM at 120 registers), so every weight byte is requested before the PDL wait, as in
// the M = 1 kernel (measured against 3 steps: 5.9 -> 4.9 us per call at M = 2, 6.6 -> 6.0 at M = 8). proj
// (538 blocks) loses with 6: half its blocks would wait for a slot (14.0 -> 14.5 us at M = 2).
#ifndef CONV_U_FP8
#define CONV_U_FP8 6
#endif
#ifndef HEAD_SPLITK
#define HEAD_SPLITK 6
#endif

__global__ void fp8_rows_kernel(const __nv_bfloat16* __restrict__ w, int rows, int K, __nv_fp8_storage_t* q,
                                float* scale) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  const __nv_bfloat16* wr = w + size_t(row) * K;
  float amax = 0.f;
  for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(__bfloat162float(wr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = amax / 448.f;
  if (lane == 0) scale[row] = s;
  for (int k = lane; k < K; k += 32)
    q[size_t(row) * K + k] =
        s > 0.f ? __nv_cvt_float_to_fp8(__bfloat162float(wr[k]) / s, __NV_SATFINITE, __NV_E4M3) : 0;
}

// INT8 rows: scale = amax / 127, q = rint(w / scale), written in the word layout of dense.h. One warp per row, one
// 4-element word per lane step.
__global__ void int8_rows_kernel(const __nv_bfloat16* __restrict__ w, int rows, int K, uint32_t* q, float* scale) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  const __nv_bfloat16* wr = w + size_t(row) * K;
  float amax = 0.f;
  for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(__bfloat162float(wr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = amax / 127.f;
  if (lane == 0) scale[row] = s;
  for (int k4 = lane; k4 < K / 4; k4 += 32) {
    uint32_t m[4], g[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v = s > 0.f ? rintf(__bfloat162float(wr[4 * k4 + j]) / s) : 0.f;
      m[j] = uint32_t(fminf(fabsf(v), 127.f));
      g[j] = v < 0.f ? 0x80u : 0u;
    }
    q[size_t(row) * (K / 4) + k4] = (m[0] | g[2]) | (m[3] | g[0]) << 8 | (m[1] | g[3]) << 16 | (m[2] | g[1]) << 24;
  }
}

// bf16(q * scale[row]) of INT8 rows ((q 2^-6) (scale 2^6): the same product); one thread per 16 elements.
__global__ void dequant_int8_rows_kernel(const uint4* __restrict__ q, const float* __restrict__ scale, int K,
                                         long long total16, __nv_bfloat16* __restrict__ out) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= total16) return;
  const long long e = i * 16;
  const float s = scale[e / K] * INT8_UNIT;
  float f[16];
  unpack_i8x16(q[i], f);
  __align__(16) __nv_bfloat16 v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = __float2bfloat16_rn(f[j] * s);
  uint4* dst = reinterpret_cast<uint4*>(out + e);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}

// weights.dequant: bf16(float(q) * scale[row]); one thread per 16 elements (K % 16 == 0).
__global__ void dequant_rows_kernel(const __nv_fp8_storage_t* __restrict__ q, const float* __restrict__ scale,
                                    int K, long long total16, __nv_bfloat16* __restrict__ out) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= total16) return;
  const long long e = i * 16;
  const float s = scale[e / K];
  const uint4 u = *reinterpret_cast<const uint4*>(q + e);
  float f[16];
  unpack_e4m3x16(u, f);
  __align__(16) __nv_bfloat16 v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = __float2bfloat16_rn(f[j] * s);
  uint4* dst = reinterpret_cast<uint4*>(out + e);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}

}  // namespace

void proj(const __nv_bfloat16* xin, int M, Matrix w, const int* segments, int nseg, float* out, bool pdl,
          cudaStream_t stream) {
  require(xin && segments && out && nseg > 0, "proj: null argument");
  const int R = w.rows;
  auto make = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ProjEpi<MM>{xin, segments, nseg, R, out};
  };
  // [Agent] INT8 proj at M = 1 may run on tensor cores (int8_m1_tensor_cores: the RTX 3090 yes, the RTX 4090 no).
  // proj_conv (2,304 rows) and head gain nothing from them on either GPU (3090: 10.4 vs 10.7, 520 vs 520 us).
  if (w.row_map) launch_gemv<H, 256, PROJ_SPLITK, MMA_U_FP8, ProjEpi, true, true>(w, M, pdl, stream, make);
  else launch_gemv<H, 256, PROJ_SPLITK, MMA_U_FP8, ProjEpi, false, true>(w, M, pdl, stream, make);
}

void proj_conv(const __nv_bfloat16* x, int M, Matrix w, const float* conv_w, float* conv_state, float* out,
               float* conv_hist, bool pdl, cudaStream_t stream) {
  require(x && conv_w && conv_state && out, "proj_conv: null argument");
  require(w.rows == H, "proj_conv: w must have 2304 rows");
  // M = 1: 2304 rows = 1152-2304 warps; 32-thread blocks spread them evenly over the SMs (measured: equal
  // or better than 64-256)
  launch_gemv<3072, 32, CONV_SPLITK, CONV_U_FP8, ConvEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ConvEpi<MM>{x, conv_w, conv_state, out, conv_hist};
  });
}

void head(const __nv_bfloat16* x, int M, Matrix w, float* out, bool pdl, cudaStream_t stream) {
  require(x && out, "head: null argument");
  const int V = w.rows;
  launch_gemv<H, 256, HEAD_SPLITK, MMA_U_FP8, HeadEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return HeadEpi<MM>{x, V, out};
  });
}

void gemv(const __nv_bfloat16* x, int M, int K, Matrix w, float* out, bool pdl, cudaStream_t stream) {
  require(x && out, "gemv: null argument");
  const int R = w.rows;
  auto make = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return GemvEpi<MM>{x, R, out};
  };
  // 2304 rows = 144 blocks of 16 rows: 16 warps per block (K slices of 192 / 576 columns) keep enough
  // bytes in flight (measured M = 8, BF16: K = 3072 10.4 -> 8.4 us, K = 9216 28.3 -> 26.0 us vs 8 warps).
  // K = 3072, M = 1: as proj_conv; K = 9216: 36 chunks per lane would spill, so M = 1 uses tensor cores.
  if (K == 3072) launch_gemv<3072, 32, 16, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 9216) launch_gemv<9216, 0, 16, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else require(false, "gemv: K must be 3072 or 9216");
}

void fp8_rows(const __nv_bfloat16* w, int rows, int K, void* q, float* scale, cudaStream_t stream) {
  require(rows >= 0 && K > 0, "fp8_rows: bad shape");
  if (rows > 0)
    fp8_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(w, rows, K, static_cast<__nv_fp8_storage_t*>(q), scale);
  CUDA_CHECK(cudaGetLastError());
}

void int8_rows(const __nv_bfloat16* w, int rows, int K, void* q, float* scale, cudaStream_t stream) {
  require(rows >= 0 && K > 0 && K % 16 == 0, "int8_rows: K must be a multiple of 16");
  if (rows > 0) int8_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(w, rows, K, static_cast<uint32_t*>(q), scale);
  CUDA_CHECK(cudaGetLastError());
}

void dequant_rows(const void* q, const float* scale, int rows, int K, __nv_bfloat16* out, cudaStream_t stream,
                  bool int8) {
  require(rows >= 0 && K > 0 && K % 16 == 0, "dequant_rows: K must be a multiple of 16");
  const long long total16 = (long long)rows * K / 16;
  if (total16 > 0 && int8)
    dequant_int8_rows_kernel<<<unsigned((total16 + 255) / 256), 256, 0, stream>>>(static_cast<const uint4*>(q), scale,
                                                                                    K, total16, out);
  else if (total16 > 0)
    dequant_rows_kernel<<<unsigned((total16 + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_fp8_storage_t*>(q), scale, K, total16, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::dense
