// Exact E4M3 decoders for every target architecture.
//
// [Agent] sm_120a converts FP8 pairs in one instruction, sm_89 has only the FP8 -> FP16 one, and sm_86 has neither
// (cuda_fp8.hpp falls back to a branchy per-element routine). The fallback here puts the code's exponent and
// mantissa bits straight into an FP16 word and multiplies by the exponent-bias gap (2^8). That product is exact for
// normal and subnormal codes alike, and FP16 math never flushes subnormals, so every path returns the same bits as
// the hardware convert. (FP4 below sm_100: the per-block lookup table in moe.cu.)
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ingine::kernels::fp8x {

// 2 E4M3 codes (low 16 bits of u, element 0 in the low byte) -> FP16x2.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t u) {
#if __CUDA_ARCH__ >= 890
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
  return r;
#else
  const uint32_t b = ((u & 0x7fu) << 7) | ((u & 0x80u) << 8) | ((u & 0x7f00u) << 15) | ((u & 0x8000u) << 16);
  const __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&b), __half2(__ushort_as_half(0x5c00), __ushort_as_half(0x5c00)));
  return *reinterpret_cast<const uint32_t*>(&h);
#endif
}

// 2 E4M3 codes -> BF16x2 (exact: every E4M3 value fits BF16).
// [Agent] Below sm_89 the code's sign, exponent and mantissa bits go straight into a BF16 word (2^-120 times the
// code's value, subnormal codes included) and one BF16 FMA by 2^120 (plus -0, which keeps the sign of zero) scales
// it back exactly: 5 instructions instead of ~9 through FP16 and FP32.
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t u) {
#if __CUDA_ARCH__ >= 1000
  uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
  return r;
#elif __CUDA_ARCH__ >= 890
  const uint32_t h = e4m3x2_to_f16x2(u);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const uint32_t*>(&b);
#else
  const uint32_t w = __byte_perm(u, 0u, 0x4140);  // [byte 0, 0, byte 1, 0]
  const uint32_t b = ((w << 4) & 0x07f007f0u) | ((w << 8) & 0x80008000u);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(b), "r"(0x7b807b80u), "r"(0x80008000u));
  return r;
#endif
}

}  // namespace ingine::kernels::fp8x
