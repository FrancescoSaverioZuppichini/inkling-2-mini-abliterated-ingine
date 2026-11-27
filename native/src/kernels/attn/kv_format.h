// Low-bit full-attention KV formats: the byte layout of one cache block (16 positions x 1 KV head) and the scalar
// rules the CUDA writer, the CUDA reader and the host oracle share.
//
// [Agent] The cache is written in the exact order the m16n8k16 tensor-core instruction wants its operand registers,
// so the reader feeds `mma.sync` straight from two 16-byte loads per lane: no shared-memory staging, no ldmatrix, no
// shuffles. Keys are the 16-row operand of S^T = K Q^T (operand swap: the 3 M query rows are the 8-column one), values
// the 16-row operand of O^T = V^T P^T. Inside one mma the order of the k index is free as long as both operands use it,
// so each 32-bit word holds exactly the 8 codes one lane needs.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define KVF_HD __host__ __device__ __forceinline__
#else
#define KVF_HD inline
#endif

namespace ingine::kernels::attn::kvf {

inline constexpr int BLK = 16, D = 128, HKV = 8;

// ---- rotation: x -> H D x, D a fixed +-1 diagonal, H the unnormalised 128-point Walsh-Hadamard matrix (H H = 128 I).
// [Agent] One fixed sign pattern for every visit and head (splitmix64 of a constant), so writer, reader and oracle
// agree without a table in memory.
KVF_HD constexpr std::uint64_t splitmix(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
inline constexpr std::uint64_t SIGN_LO = splitmix(0x5348'4144'4b56'0001ull), SIGN_HI = splitmix(0x5348'4144'4b56'0002ull);
KVF_HD constexpr float sign_of(int d) { return ((d < 64 ? SIGN_LO : SIGN_HI) >> (d & 63)) & 1 ? -1.f : 1.f; }

// ---- int4: rotated keys with one FP16 least-squares scale per key and head, rotated values with no scale.
// A code n in 0..15 stands for the level (n - 7.5) * step on a row scaled to unit RMS (keys: H D k / |k|; values:
// H D v / sqrt(128), whose RMS is 1 because the model RMS-normalises values without gain). step = the MSE-optimal
// 16-level uniform step for a unit Gaussian (Max 1960), which rotated rows follow closely.
namespace int4 {
inline constexpr float STEP = 0.3352f;
inline constexpr int K_BYTES = BLK * D / 2 + BLK * 2;  // codes + FP16 scales: 1056
inline constexpr int V_BYTES = BLK * D / 2;            // 1024

// Key j (0..15) of a block: word w = lane * 8 + kk * 2 + (j >> 3), lane = (j & 7) * 4 + t4, holds dims
// 32 kk + 2 t4 + {0, 8, 16, 24} (nibbles 0..3) and the same + 1 (nibbles 4..7): nibble pairs (n, n + 4) are the
// FP16 pairs of one A register (k-slots 2 t4, 2 t4 + 1 and 2 t4 + 8, 2 t4 + 9 of k-steps 2 kk and 2 kk + 1).
KVF_HD constexpr int k_dim(int kk, int t4, int nib) { return 32 * kk + 2 * t4 + (nib & 3) * 8 + (nib >> 2); }
KVF_HD constexpr int k_word(int j, int kk, int t4) { return ((j & 7) * 4 + t4) * 8 + kk * 2 + (j >> 3); }
KVF_HD constexpr int k_scale_off(int j) { return BLK * D / 2 + 2 * ((j & 7) * 2 + (j >> 3)); }  // bytes
// Dim d of key j: the nibble of the word above.
KVF_HD constexpr int k_nib_of_dim(int d) { const int r = d & 31; return ((r >> 3) & 3) | ((r & 1) << 2); }
KVF_HD constexpr int k_word_of_dim(int j, int d) { return k_word(j, d >> 5, (d & 7) >> 1); }

// Value byte of key j and dims (16 dt + gid, 16 dt + gid + 8) (low, high nibble): word (lane * 8 + dt) with lane =
// gid * 4 + (j & 7) / 2, byte (j & 1) * 2 + (j >> 3). The word decodes to the A registers of dim tile dt: FP16 pairs
// (keys 2 t4, 2 t4 + 1) for dims gid, gid + 8, then (keys 2 t4 + 8, 2 t4 + 9).
KVF_HD constexpr int v_byte(int j, int d) {
  const int dt = d >> 4, gid = d & 7;
  return ((gid * 4 + ((j & 7) >> 1)) * 8 + dt) * 4 + (j & 1) * 2 + (j >> 3);
}
KVF_HD constexpr bool v_high(int d) { return (d & 15) >= 8; }

KVF_HD constexpr int code_of(float x) {  // x in units of the unit-RMS row
  const float t = x * (1.f / STEP);
  int n = int(t >= 0.f ? t : t - 1.f) + 8;  // floor without <cmath>, exact for |t| < 2^23
  if (t < 0.f && float(int(t)) == t) n += 1;  // t a negative integer: floor(t) = t
  return n < 0 ? 0 : n > 15 ? 15 : n;
}
}  // namespace int4

}  // namespace ingine::kernels::attn::kvf
