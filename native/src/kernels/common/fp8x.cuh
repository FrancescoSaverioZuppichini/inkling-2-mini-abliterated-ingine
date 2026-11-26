// Exact E4M3 / E2M1 decoders for every target architecture.
//
// [Agent] sm_120a converts FP4 and FP8 pairs in one instruction, sm_89 has only the FP8 -> FP16 one, and sm_86
// has neither (cuda_fp8.hpp falls back to a branchy per-element routine). The fallback here puts the code's
// exponent and mantissa bits straight into an FP16 word and multiplies by the exponent-bias gap (2^8 for E4M3,
// 2^14 for E2M1). That product is exact for normal and subnormal codes alike, and FP16 math never flushes
// subnormals, so every path returns the same bits as the hardware convert.
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
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t u) {
#if __CUDA_ARCH__ >= 1000
  uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
  return r;
#else
  const uint32_t h = e4m3x2_to_f16x2(u);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const uint32_t*>(&b);
#endif
}

// 2 E2M1 codes (one byte, element 0 in the low nibble) -> FP16x2.
__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t byte) {
#if __CUDA_ARCH__ >= 1000
  uint32_t r;
  asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "r"(byte));
  return r;
#else
  const uint32_t b = ((byte & 0x7u) << 9) | ((byte & 0x8u) << 12) | ((byte & 0x70u) << 21) | ((byte & 0x80u) << 24);
  const __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&b), __half2(__ushort_as_half(0x7400), __ushort_as_half(0x7400)));
  return *reinterpret_cast<const uint32_t*>(&h);
#endif
}

}  // namespace ingine::kernels::fp8x
