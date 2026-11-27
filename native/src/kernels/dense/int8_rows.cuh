// INT8 dense rows (dense::int8_rows): K bytes per row, one fp32 scale per row = amax / 127, q = rint(w / scale).
// [Agent] Each 4-byte word holds 4 sign-magnitude codes as bytes [m0 | s2, m3 | s0, m1 | s3, m2 | s1] (m = |q|,
// s = sign << 7), so that one AND turns elements 0, 1 and a byte rotation plus the same AND turns elements 2, 3 into
// BF16x2 words (the fast decode in gemv.cu). The encoder is int8_rows_kernel (gemv.cu); this scalar decode is the
// reference every other reader uses.
#pragma once

#include <cstdint>

// one 4-byte word as its 4 values q (unscaled)
__device__ __forceinline__ void int8_word(uint32_t w, float* f) {
  auto q = [w](int mb, int sb) {
    const float m = float((w >> (8 * mb)) & 0x7fu);
    return (w >> (8 * sb + 7)) & 1u ? -m : m;
  };
  f[0] = q(0, 1), f[1] = q(2, 3), f[2] = q(3, 0), f[3] = q(1, 2);
}
