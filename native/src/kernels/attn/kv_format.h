// Low-bit full-attention KV formats (int4, int8, k8v4, fp4): the byte layout of one cache block (16 positions x 1 KV
// head) and the scalar rules the CUDA writer, the CUDA reader and the host oracle share.
//
// [Agent] The cache is written in the exact order the m16n8k16 tensor-core instruction wants its operand registers,
// so the reader feeds `mma.sync` straight from 16-byte loads per lane: no shared-memory staging, no ldmatrix, no
// shuffles. Keys are the 16-row operand of S^T = K Q^T (operand swap: the 3 M query rows are the 8-column one), values
// the 16-row operand of O^T = V^T P^T. Inside one mma the order of the k index is free as long as both operands use it,
// so each 32-bit word holds exactly the codes one lane needs.
//
// A format is a key code and a value code: int4 = 4-bit keys with a least-squares scale + scale-free 4-bit values,
// int8 = 8-bit keys and values with a scale per row, k8v4 = int8 keys + int4 values, fp4 = NVFP4-style E2M1 codes with
// a scale per 16 dims for both. Every format codes rotated rows (below).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define KVF_HD __host__ __device__ __forceinline__
#else
#define KVF_HD inline
#endif

namespace ingine::kernels::attn::kvf {

inline constexpr int BLK = 16, D = 128;

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

// ---- bit-exact scalar helpers. [Agent] Every rounding step of the int8 and fp4 codes is one IEEE operation
// or integer bit work, never an a * b + c the compiler could contract, so the GPU encoder writes the bytes the host
// oracle writes (f2h equals cvt.rn.f16.f32 on all 2^32 inputs).
KVF_HD std::uint32_t f2u(float x) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(x);
#else
  std::uint32_t u;
  std::memcpy(&u, &x, 4);
  return u;
#endif
}
KVF_HD float u2f(std::uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float x;
  std::memcpy(&x, &u, 4);
  return x;
#endif
}
// float -> FP16 bits, round to nearest even, overflow to inf
KVF_HD std::uint16_t f2h(float f) {
  const std::uint32_t u = f2u(f), s = (u >> 16) & 0x8000u, x = u & 0x7fffffffu;
  if (x >= 0x7f800000u) return std::uint16_t(s | (x > 0x7f800000u ? 0x7fffu : 0x7c00u));
  if (x >= 0x477ff000u) return std::uint16_t(s | 0x7c00u);  // >= 65520 rounds to inf
  if (x < 0x38800000u) {                                     // below 2^-14: subnormal half
    if (x <= 0x33000000u) return std::uint16_t(s);           // <= 2^-25 rounds to 0 (the tie goes to even 0)
    const std::uint32_t e = x >> 23, m = (x & 0x7fffffu) | 0x800000u, sh = 126u - e;
    std::uint32_t q = m >> sh;
    const std::uint32_t r = m & ((1u << sh) - 1u), h = 1u << (sh - 1u);
    q += (r > h || (r == h && (q & 1u))) ? 1u : 0u;
    return std::uint16_t(s | q);
  }
  std::uint32_t q = (x - 0x38000000u) >> 13;
  const std::uint32_t r = x & 0x1fffu;
  q += (r > 0x1000u || (r == 0x1000u && (q & 1u))) ? 1u : 0u;
  return std::uint16_t(s | q);
}
KVF_HD float h2f(std::uint16_t h) {
  const std::uint32_t s = std::uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 0x3ffu;
  if (e == 0) {
    const float v = float(m) * 5.9604644775390625e-08f;  // m * 2^-24, exact
    return s ? -v : v;
  }
  if (e == 31) return u2f(s | 0x7f800000u | (m << 13));
  return u2f(s | ((e + 112u) << 23) | (m << 13));
}
KVF_HD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}
KVF_HD float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
KVF_HD int rint_i(float x) {  // round half to even
#if defined(__CUDA_ARCH__)
  return __float2int_rn(x);
#else
  return int(std::nearbyint(x));
#endif
}
KVF_HD float abs_f(float x) { return u2f(f2u(x) & 0x7fffffffu); }
KVF_HD int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

// ---- int4 (int4 keys and values, k8v4 values): codes n in 0..15 stand for the level (n - 7.5) * STEP on a row scaled
// to unit RMS (keys: H D k / |k|, plus one FP16 least-squares scale per key and head; values: H D v / sqrt(128), whose
// RMS is 1 because the model RMS-normalises values without gain, no scale). STEP = the MSE-optimal 16-level uniform
// step for a unit Gaussian (Max 1960), which rotated rows follow closely.
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

// ---- int8 (int8 keys and values, k8v4 keys): symmetric codes c in -127..127 of the rotated row, stored as c + 128,
// with one FP16 scale per row s = FP16(amax / 127), c = clamp(rint(x / s)). A block is 2,048 code bytes in operand
// order, then the 16 rows' FP16 scales.
namespace int8 {
inline constexpr int BYTES = BLK * D + BLK * 2;  // 2080, keys and values alike
KVF_HD std::uint16_t scale_of(float amax) { return f2h(div_rn(amax, 127.f)); }
KVF_HD int code_of(float x, float s) { return s > 0.f ? clampi(rint_i(div_rn(x, s)), -127, 127) : 0; }
// Keys: lane (gid = j & 7, t4) holds 16 words, word ks * 2 + (j >> 3) = dims 16 ks + 2 t4 + {0, 1, 8, 9}; the scales
// of keys gid, gid + 8 are one word.
KVF_HD constexpr int k_byte(int j, int d) {
  const int ks = d >> 4, sl = d & 15, lane = (j & 7) * 4 + ((sl & 7) >> 1);
  return (lane * 16 + ks * 2 + (j >> 3)) * 4 + (sl & 1) + ((sl >> 3) << 1);
}
KVF_HD constexpr int k_scale_off(int j) { return BLK * D + 4 * (j & 7) + 2 * (j >> 3); }
// Values: lane (gid = dim & 7, t4) holds 16 words, word dt * 2 + (dim >> 3 & 1) = keys 2 t4 + {0, 1, 8, 9}; lane t4
// reads the scales of those 4 keys as 8 bytes.
KVF_HD constexpr int v_byte(int j, int d) {
  const int dt = d >> 4, r = d & 15, lane = (r & 7) * 4 + ((j & 7) >> 1);
  return (lane * 16 + dt * 2 + (r >> 3)) * 4 + (j & 1) + ((j >> 3) << 1);
}
KVF_HD constexpr int v_scale_off(int j) { return BLK * D + 8 * ((j & 7) >> 1) + 2 * ((j & 1) + 2 * (j >> 3)); }
// [Agent] The reader caps value scales at 2.0 (valid rows: |H D v| = 128, so s <= 128 / 127) so that garbage past the
// newest key decodes finite: P = 0 there, and 0 * NaN would poison the output.
inline constexpr std::uint16_t V_SCALE_CAP = 0x4000;
}  // namespace int8

// ---- fp4 (keys and values): E2M1 codes (magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6, sign bit 3) times one scale per row
// and 16 dims, s = amax / 6 kept to 3 mantissa bits as the byte of FP16 bits 14..7 of s * 2^10 (a superset of E4M3,
// so no per-cache FP32 scale). The E2M1 bits placed at FP16 bits 11..9 are the code's value times 2^-14, and their
// FP16 product with the scale byte as FP16 is exact: operand = value * s / 16. Codes in int4's nibble order; a block
// is 1,024 code bytes then 128 scale bytes.
namespace fp4 {
inline constexpr int BYTES = BLK * D / 2 + BLK * D / 16;  // 1152, keys and values alike
inline constexpr float UNIT = 16.f;                         // value = operand * UNIT
KVF_HD std::uint8_t scale_of(float amax) {
  if (!(amax > 0.f)) return 0;
  const std::uint32_t u = f2u(mul_rn(div_rn(amax, 6.f), 1024.f));
  int e = int(u >> 23) - 127;
  std::uint32_t r = (u & 0x7fffffu) >> 20;
  const std::uint32_t rem = u & 0xfffffu;
  r += (rem > 0x80000u || (rem == 0x80000u && (r & 1u))) ? 1u : 0u;
  if (r == 8u) r = 0u, ++e;
  if (e < -6) return std::uint8_t(9u << 3);          // 2^-6
  if (e > 15) return std::uint8_t((30u << 3) | 7u);  // 61440
  return std::uint8_t((std::uint32_t(e + 15) << 3) | r);
}
KVF_HD float scale_val(std::uint8_t b) { return h2f(std::uint16_t(b << 7)) * (1.f / 1024.f); }  // exact
// |x| / s -> magnitude code, round to nearest, ties to the even code, saturating at 6
KVF_HD int mag_of(float t) {
  return int(t > 0.25f) + int(t >= 0.75f) + int(t > 1.25f) + int(t >= 1.75f) + int(t > 2.5f) + int(t >= 3.5f) + int(t > 5.f);
}
KVF_HD int code_of(float x, float s) {
  return int((f2u(x) >> 28) & 8) | (s > 0.f ? mag_of(div_rn(abs_f(x), s)) : 0);
}
KVF_HD float value_of(int nib) {
  const int m = nib & 7;
  const float v = m < 2 ? 0.5f * float(m) : float(2 + (m & 1)) * float(1 << ((m >> 1) - 1)) * 0.5f;
  return (nib & 8) ? -v : v;
}
// Keys: lane gid reads 16 bytes, key gid's 8 group scales then key gid + 8's. Values: lane t4 reads 32 bytes, one word
// per dim tile (= group) holding keys 2 t4, 2 t4 + 1, 2 t4 + 8, 2 t4 + 9.
KVF_HD constexpr int k_scale_off(int j, int g) { return BLK * D / 2 + 16 * (j & 7) + 8 * (j >> 3) + g; }
KVF_HD constexpr int v_scale_off(int j, int g) { return BLK * D / 2 + 32 * ((j & 7) >> 1) + 4 * g + (j & 1) + 2 * (j >> 3); }
inline constexpr std::uint16_t V_SCALE_CAP = 0x7b80;  // 61440, the largest scale the encoder writes
}  // namespace fp4

}  // namespace ingine::kernels::attn::kvf
