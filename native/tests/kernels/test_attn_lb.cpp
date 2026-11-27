// Low-bit full-attention decode and prefill (attn_lb.cu; KvFormat int4, int8, k8v4, fp4) against an FP64 oracle over
// the DEQUANTIZED cache: the oracle reads back the codes and scales the kernels wrote (pack_rows for old rows, the
// decode kernel for the call's own rows) and attends them in double, with the query rotated and rounded to FP16 as
// the kernel does, so the check isolates the reader's arithmetic from the format's loss. Per format, also:
// * pack_rows writes the host encoder's bytes (int8, fp4 and int4 values bit for bit: FP32 rotation in the kernel's
//   butterfly order, every rounding one IEEE operation; int4 keys within one level and 2e-3 of the scale, since their
//   norm and least-squares sums run in another order);
// * the dequantized rows are within the format's error of the true rows (a layout bug the oracle would share);
// * the call's new rows vs a host encode of the oracle's keys / values (rare boundary flips allowed), conv states,
//   and stale rows after p0 + M (random bytes, NaN scales included: output bits must not move).
// Full layers, p0 = 0 / 3000 / 4637 / 9005, M = 1..8; prefill T = 77 and 300. Synthetic layer parameters.
#include "kernels/oracle.h"

#include "kernels/attn/attn.h"
#include "kernels/attn/kv_format.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <utility>

using namespace ingine::test;
namespace attn = ingine::kernels::attn;
namespace kvf = ingine::kernels::attn::kvf;

namespace {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, C = HKV * D, BLK = 16;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632, R = 8664;
constexpr int CAP = 10016;
constexpr double QK_CLIP = 22.627416997969522, REL_CLIP = 8.0;
constexpr float RSQRT_D = 0.08838834764831845f;
cudaStream_t g_stream = nullptr;

// ---- formats: a key code and a value code (kv_format.h)
enum class Code { i4, i8, f4 };
struct Fmt {
  attn::KvFormat kv;
  const char* name;
  Code k, v;
};
const Fmt FORMATS[] = {{attn::KvFormat::int4, "int4", Code::i4, Code::i4},
                       {attn::KvFormat::int8, "int8", Code::i8, Code::i8},
                       {attn::KvFormat::k8v4, "k8v4", Code::i8, Code::i4},
                       {attn::KvFormat::fp4, "fp4", Code::f4, Code::f4}};
// Max RMS relative error of a dequantized random Gaussian row: 8-bit ~0.7 %, 4-bit ~10 %.
double bound_of(Code c) { return c == Code::i8 ? 0.015 : 0.15; }
const Fmt* g_f = nullptr;

int block_bytes(Code c, bool key) {
  switch (c) {
    case Code::i4: return key ? kvf::int4::K_BYTES : kvf::int4::V_BYTES;
    case Code::i8: return kvf::int8::BYTES;
    default: return kvf::fp4::BYTES;
  }
}
int kb() { return block_bytes(g_f->k, true); }
int vb() { return block_bytes(g_f->v, false); }

// Byte and nibble mask of code (j, d) in a block; the bytes of row j's scales.
std::pair<int, uint8_t> code_at(Code c, bool key, int j, int d) {
  if (c == Code::i8) return {key ? kvf::int8::k_byte(j, d) : kvf::int8::v_byte(j, d), 0xff};
  if (key) {
    const int nib = kvf::int4::k_nib_of_dim(d);
    return {4 * kvf::int4::k_word_of_dim(j, d) + nib / 2, uint8_t(nib & 1 ? 0xf0 : 0x0f)};
  }
  return {kvf::int4::v_byte(j, d), uint8_t(kvf::int4::v_high(d) ? 0xf0 : 0x0f)};
}
int code_of_byte(uint8_t b, uint8_t mask) { return mask == 0xf0 ? b >> 4 : mask == 0x0f ? b & 15 : b; }
std::vector<int> scale_bytes(Code c, bool key, int j) {
  switch (c) {
    case Code::i4: return key ? std::vector<int>{kvf::int4::k_scale_off(j), kvf::int4::k_scale_off(j) + 1} : std::vector<int>{};
    case Code::i8: {
      const int o = key ? kvf::int8::k_scale_off(j) : kvf::int8::v_scale_off(j);
      return {o, o + 1};
    }
    default: {
      std::vector<int> o;
      for (int g = 0; g < 8; ++g) o.push_back(key ? kvf::fp4::k_scale_off(j, g) : kvf::fp4::v_scale_off(j, g));
      return o;
    }
  }
}
double half_at(const uint8_t* blk, int off) {
  uint16_t h;
  std::memcpy(&h, blk + off, 2);
  return kvf::h2f(h);
}

// Host decode of slot j, dim d, in the rotated space the kernel attends: keys k'' = H D k, values v'' = H D v. int8
// values are the FP16 product code x scale (HMUL2), fp4 operands are exact.
double key_at(const uint8_t* blk, int j, int d) {
  const auto [byte, mask] = code_at(g_f->k, true, j, d);
  const int n = code_of_byte(blk[byte], mask);
  switch (g_f->k) {
    case Code::i4: return half_at(blk, kvf::int4::k_scale_off(j)) * (n - 7.5);
    case Code::i8: return half_at(blk, kvf::int8::k_scale_off(j)) * (n - 128);
    default: return double(kvf::fp4::value_of(n)) * kvf::fp4::scale_val(blk[kvf::fp4::k_scale_off(j, d >> 4)]);
  }
}
double value_at(const uint8_t* blk, int j, int d) {
  const auto [byte, mask] = code_at(g_f->v, false, j, d);
  const int n = code_of_byte(blk[byte], mask);
  switch (g_f->v) {
    case Code::i4: return (n - 7.5) * kvf::int4::STEP * std::sqrt(128.0);
    case Code::i8: return kvf::h2f(kvf::f2h(float(n - 128) * float(half_at(blk, kvf::int8::v_scale_off(j)))));
    default: return double(kvf::fp4::value_of(n)) * kvf::fp4::scale_val(blk[kvf::fp4::v_scale_off(j, d >> 4)]);
  }
}

// x <- H D x in FP32 with the kernel's butterfly order (bit-identical to its rotation)
void rot_f32(float* x) {
  for (int d = 0; d < D; ++d) x[d] *= kvf::sign_of(d);
  for (int h = 1; h < D; h <<= 1)
    for (int i = 0; i < D; i += 2 * h)
      for (int k = i; k < i + h; ++k) {
        const float a = x[k], b = x[k + h];
        x[k] = a + b, x[k + h] = a - b;
      }
}
// unnormalised Walsh-Hadamard transform in place (H H = 128 I), double
void fwht(std::vector<double>& y) {
  for (int h = 1; h < D; h <<= 1)
    for (int i = 0; i < D; ++i)
      if (!(i & h)) {
        const double a = y[i], b = y[i + h];
        y[i] = a + b, y[i + h] = a - b;
      }
}
// H D x, double
std::vector<double> rot(const double* x) {
  std::vector<double> y(D);
  for (int d = 0; d < D; ++d) y[d] = x[d] * kvf::sign_of(d);
  fwht(y);
  return y;
}
double f16(double x) { return double(__half2float(__float2half_rn(float(x)))); }

// Host encode of one BF16-valued row (model space) into slot j of a K or V block.
void encode(Code c, bool key, const float* row, uint8_t* blk, int j) {
  float x[D];
  std::memcpy(x, row, sizeof x);
  float n2 = 0.f;
  for (int d = 0; d < D; ++d) n2 += x[d] * x[d];
  rot_f32(x);
  auto put = [&](int d, int code) {
    const auto [byte, mask] = code_at(c, key, j, d);
    const int shift = mask == 0xf0 ? 4 : 0;
    blk[byte] = uint8_t((blk[byte] & ~mask) | ((code << shift) & mask));
  };
  if (c == Code::i8) {
    float a = 0.f;
    for (int d = 0; d < D; ++d) a = std::max(a, std::fabs(x[d]));
    const uint16_t s = kvf::int8::scale_of(a);
    for (int d = 0; d < D; ++d) put(d, kvf::int8::code_of(x[d], kvf::h2f(s)) + 128);
    std::memcpy(blk + (key ? kvf::int8::k_scale_off(j) : kvf::int8::v_scale_off(j)), &s, 2);
  } else if (c == Code::f4) {
    for (int g = 0; g < 8; ++g) {
      float a = 0.f;
      for (int d = 16 * g; d < 16 * g + 16; ++d) a = std::max(a, std::fabs(x[d]));
      const uint8_t b = kvf::fp4::scale_of(a);
      for (int d = 16 * g; d < 16 * g + 16; ++d) put(d, kvf::fp4::code_of(x[d], kvf::fp4::scale_val(b)));
      blk[key ? kvf::fp4::k_scale_off(j, g) : kvf::fp4::v_scale_off(j, g)] = b;
    }
  } else if (!key) {
    for (int d = 0; d < D; ++d) put(d, kvf::int4::code_of(x[d] * RSQRT_D));
  } else {  // int4 key: unit-direction codes and the least-squares scale (FP32 sums in host order)
    const float inv = n2 > 0.f ? 1.f / std::sqrt(n2) : 0.f;
    double num = 0, den = 0;
    for (int d = 0; d < D; ++d) {
      const int n = kvf::int4::code_of(x[d] * inv);
      put(d, n);
      num += x[d] * (n - 7.5), den += (n - 7.5) * (n - 7.5);
    }
    const __half s = __float2half_rn(float(num / den));
    std::memcpy(blk + kvf::int4::k_scale_off(j), &s, 2);
  }
}

// The cache as the oracle sees it: [rows][8][128] rotated keys and values.
struct Deq {
  std::vector<double> k, v;
};
Deq dequant(const std::vector<uint8_t>& kc, const std::vector<uint8_t>& vc, int rows) {
  Deq q;
  q.k.resize(size_t(rows) * C), q.v.resize(q.k.size());
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < HKV; ++g) {
      const uint8_t* kblk = &kc[(size_t(r / BLK) * HKV + g) * kb()];
      const uint8_t* vblk = &vc[(size_t(r / BLK) * HKV + g) * vb()];
      for (int d = 0; d < D; ++d) {
        const size_t i = (size_t(r) * HKV + g) * D + d;
        q.k[i] = key_at(kblk, r % BLK, d);
        q.v[i] = value_at(vblk, r % BLK, d);
      }
    }
  return q;
}

// Every packed block against the host encoder (int4 keys: codes within one level, scales within 2e-3; everything else
// bit for bit), and the RMS relative error of the dequantized rows against the true (rotated) rows.
bool check_pack(const std::vector<float>& rk, const std::vector<float>& rv, int rows, const std::vector<uint8_t>& kc,
                const std::vector<uint8_t>& vc, const Deq& q) {
  size_t exact_off = 0, loose_off = 0, loose_big = 0, scales_bad = 0;
  double se[2] = {0, 0}, sr[2] = {0, 0};
  std::vector<uint8_t> blk(std::max(kb(), vb()));
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < HKV; ++g)
      for (int kv = 0; kv < 2; ++kv) {
        const Code c = kv ? g_f->v : g_f->k;
        const int n = kv ? vb() : kb(), j = r % BLK;
        const uint8_t* got = &(kv ? vc : kc)[(size_t(r / BLK) * HKV + g) * n];
        const float* row = &(kv ? rv : rk)[(size_t(r) * HKV + g) * D];
        std::fill(blk.begin(), blk.end(), uint8_t(0));
        encode(c, kv == 0, row, blk.data(), j);
        const bool loose = c == Code::i4 && kv == 0;  // FP32 norm and least-squares sums in another order
        for (int d = 0; d < D; ++d) {
          const auto [byte, mask] = code_at(c, kv == 0, j, d);
          const int a = code_of_byte(got[byte] & mask, mask), b = code_of_byte(blk[byte] & mask, mask);
          (loose ? loose_off : exact_off) += a != b;
          loose_big += loose && std::abs(a - b) > 1;
        }
        if (!loose)
          for (int o : scale_bytes(c, kv == 0, j)) exact_off += got[o] != blk[o];
        else
          scales_bad += std::fabs(half_at(got, kvf::int4::k_scale_off(j)) - half_at(blk.data(), kvf::int4::k_scale_off(j))) >
                        2e-3 * std::fabs(half_at(blk.data(), kvf::int4::k_scale_off(j))) + 1e-6;
        const std::vector<double>& deq = kv ? q.v : q.k;
        std::vector<double> x(row, row + D);
        const std::vector<double> y = rot(x.data());
        for (int d = 0; d < D; ++d) {
          const double e = deq[(size_t(r) * HKV + g) * D + d] - y[d];
          se[kv] += e * e, sr[kv] += y[d] * y[d];
        }
      }
  const bool codes_ok = exact_off == 0 && loose_big == 0 && scales_bad == 0 && loose_off <= size_t(rows) * HKV * D / 500;
  const double ek = std::sqrt(se[0] / sr[0]), ev = std::sqrt(se[1] / sr[1]);
  const bool ok = codes_ok && ek <= bound_of(g_f->k) && ev <= bound_of(g_f->v);
  std::printf("%s %s pack_rows %d rows: %zu codes / scale bytes differ from the exact host encoder; int4 keys: %zu codes "
              "off by one, %zu by more, %zu scales off; row error K %.4f V %.4f (bounds %.3f / %.3f)\n",
              ok ? "PASS" : "FAIL", g_f->name, rows, exact_off, loose_off, loose_big, scales_bad, ek, ev,
              bound_of(g_f->k), bound_of(g_f->v));
  return ok;
}

void clip_rows(std::vector<double>& x, int rows, int n, double t) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += x[size_t(r) * n + i] * x[size_t(r) * n + i];
    const double f = std::min(1.0, t / (std::sqrt(ss) + 1e-6));
    for (int i = 0; i < n; ++i) x[size_t(r) * n + i] *= f;
  }
}

// FP64 attention of one query row (rotated, FP16 as the kernel) over keys 0..p of KV head g of the dequantized cache,
// sink logit in the denominator, bias(distance) added; the output back in model space (D H o / 128).
template <class Bias>
std::vector<double> attend(const Deq& q, const double* qr, int g, int p, double sink, Bias&& bias) {
  std::vector<double> s(p + 1), o(D, 0.0);
  double mx = sink;
  for (int j = 0; j <= p; ++j) {
    double d = 0;
    for (int e = 0; e < D; ++e) d += qr[e] * q.k[(size_t(j) * HKV + g) * D + e];
    s[j] = d / 128.0 / std::sqrt(128.0) + bias(p - j);
    mx = std::max(mx, s[j]);
  }
  double den = std::exp(sink - mx);
  for (int j = 0; j <= p; ++j) {
    const double e = std::exp(s[j] - mx);
    den += e;
    for (int d = 0; d < D; ++d) o[d] += e * q.v[(size_t(j) * HKV + g) * D + d];
  }
  for (auto& x : o) x /= den;
  fwht(o);
  for (int d = 0; d < D; ++d) o[d] *= kvf::sign_of(d) / 128.0;
  return o;
}

// Random rows as the model writes them: BF16 keys clipped at 22.63 (norms 0.5..3.5 x sqrt(128) before the clip),
// unit-RMS values.
void cache_rows(Rng& rng, int rows, std::vector<float>& rk, std::vector<float>& rv) {
  rk.resize(size_t(rows) * C), rv.resize(rk.size());
  for (int r = 0; r < rows * HKV; ++r) {
    std::vector<double> k(D), v(D);
    const double scale = rng.uniform(0.5, 3.5);
    double ss = 0, sv = 0;
    for (int d = 0; d < D; ++d) k[d] = rng.normal() * scale, ss += k[d] * k[d];
    for (int d = 0; d < D; ++d) v[d] = rng.normal(), sv += v[d] * v[d];
    const double f = std::min(1.0, QK_CLIP / (std::sqrt(ss) + 1e-6)), rvn = 1.0 / std::sqrt(sv / D + 1e-6);
    for (int d = 0; d < D; ++d) {
      rk[size_t(r) * D + d] = round_bf16(float(k[d] * f));
      rv[size_t(r) * D + d] = round_bf16(float(v[d] * rvn));
    }
  }
}

// Random bytes in the codes of slot j (and in its scales: random, or NaN-like on the second pass).
void scramble(uint8_t* kblk, uint8_t* vblk, int j, Rng& rng, bool nan) {
  for (int kv = 0; kv < 2; ++kv) {
    const Code c = kv ? g_f->v : g_f->k;
    uint8_t* blk = kv ? vblk : kblk;
    for (int d = 0; d < D; ++d) {
      const auto [byte, mask] = code_at(c, kv == 0, j, d);
      blk[byte] = uint8_t((blk[byte] & ~mask) | (rng.next() & mask));
    }
    const std::vector<int> sb = scale_bytes(c, kv == 0, j);
    for (size_t i = 0; i < sb.size(); ++i)
      blk[sb[i]] = nan ? (c == Code::f4 ? 0xff : i % 2 ? 0x7e : 0x00) : uint8_t(rng.next());
  }
}

bool run(Rng& rng, int M, int p0) {
  // ---- inputs (as test_attn.cpp)
  std::vector<float> proj(size_t(M) * R);
  for (int m = 0; m < M; ++m)
    for (int r = 0; r < R; ++r) {
      double s = 1.0;
      if (r >= OFF_REL && r < OFF_SINK) s = 3.0;
      else if (r >= OFF_SINK && r < OFF_K) s = 2.0;
      else if (r >= OFF_KDYN) s = 0.3;
      proj[size_t(m) * R + r] = round_bf16(float(rng.normal() * s));
    }
  std::vector<float> q_scale = randn(rng, HQ * D, 0.3), r_scale = randn(rng, HQ * DREL, 0.3);
  for (auto& v : q_scale) v = round_bf16(v + 1.f);
  for (auto& v : r_scale) v = round_bf16(v + 1.f);
  const std::vector<float> rel_proj = randn_bf16(rng, DREL * EXT, 0.5);
  const std::vector<float> kconv = randn_bf16(rng, 4 * C, 0.3), vconv = randn_bf16(rng, 4 * C, 0.3);
  const std::vector<float> ck0 = randn_bf16(rng, 3 * C, 1.0), cv0 = randn_bf16(rng, 3 * C, 1.0);
  std::vector<float> rk, rv;
  cache_rows(rng, CAP, rk, rv);
  DeviceArray<__nv_bfloat16> drk(to_bf16(rk)), drv(to_bf16(rv));
  DeviceArray<uint8_t> dk(attn::cache_bytes(g_f->kv, CAP, false)), dv(attn::cache_bytes(g_f->kv, CAP, true));
  attn::pack_rows(g_f->kv, drk.get(), drv.get(), CAP, 0, dk.get(), dv.get(), g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  const bool check_packed = p0 == 0 && M == 1;  // once per format: pack_rows does not depend on the call
  const std::vector<uint8_t> packed_k = check_packed ? dk.download() : std::vector<uint8_t>{};
  const std::vector<uint8_t> packed_v = check_packed ? dv.download() : std::vector<uint8_t>{};

  DeviceArray<float> dproj(proj), dqs(q_scale), drs(r_scale), drp(rel_proj), dkc(kconv), dvc(vconv), dck(ck0), dcv(cv0);
  DeviceArray<int64_t> dpos(std::vector<int64_t>{p0});
  DeviceArray<uint8_t> ws(attn::workspace_bytes(8));
  DeviceArray<__nv_bfloat16> out(size_t(M) * HQ * D);
  attn::DecodeArgs a;
  a.proj = dproj.get(), a.M = M, a.q_scale = dqs.get(), a.r_scale = drs.get(), a.rel_proj = drp.get();
  a.k_conv = dkc.get(), a.v_conv = dvc.get(), a.conv_k_state = dck.get(), a.conv_v_state = dcv.get();
  a.k_cache = dk.get(), a.v_cache = dv.get(), a.kv = g_f->kv, a.cache_rows = CAP;
  a.pos = dpos.get(), a.window = 0, a.log_scaling = true;
  a.workspace = ws.get(), a.workspace_size = ws.count, a.out = out.get();
  attn::attn_decode(a, g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  const std::vector<__nv_bfloat16> out_first = out.download();
  const std::vector<uint8_t> kc = dk.download(), vc = dv.download();
  const Deq q = dequant(kc, vc, p0 + M);
  bool ok = true;
  if (check_packed) ok &= check_pack(rk, rv, CAP, packed_k, packed_v, dequant(packed_k, packed_v, CAP));

  // ---- oracle: q (rotated, FP16 as the kernel) and rel
  std::vector<double> qr(size_t(M) * HQ * D), qplain(qr.size()), rl(size_t(M) * HQ * EXT);
  for (int m = 0; m < M; ++m) {
    const int p = p0 + m;
    const double tau = 1.0 + 0.1 * std::log(std::max((p + 1) / 8192.0, 1.0));
    std::vector<double> qm(HQ * D), rm(HQ * DREL);
    for (int i = 0; i < HQ * D; ++i) qm[i] = float(proj[size_t(m) * R + i] * q_scale[i]);
    for (int i = 0; i < HQ * DREL; ++i) rm[i] = float(proj[size_t(m) * R + OFF_REL + i] * r_scale[i]);
    clip_rows(qm, HQ, D, QK_CLIP);
    clip_rows(rm, HQ, DREL, REL_CLIP);
    for (auto& v : qm) v = rbf(rbf(v) * tau);
    for (auto& v : rm) v = rbf(rbf(v) * tau);
    for (int h = 0; h < HQ; ++h) {
      const std::vector<double> y = rot(&qm[size_t(h) * D]);
      for (int d = 0; d < D; ++d) {
        qr[(size_t(m) * HQ + h) * D + d] = f16(y[d]);
        qplain[(size_t(m) * HQ + h) * D + d] = y[d];
      }
      for (int d = 0; d < EXT; ++d) {
        double s = 0;
        for (int j = 0; j < DREL; ++j) s += rm[h * DREL + j] * rbf(rel_proj[size_t(j) * EXT + d]);
        rl[(size_t(m) * HQ + h) * EXT + d] = rbf(s);
      }
    }
  }
  // ---- oracle: the call's keys / values (dynamic conv, clip / RMS norm, BF16)
  std::vector<double> nk(size_t(M) * C), nv(size_t(M) * C);
  std::vector<float> want_ck(3 * C), want_cv(3 * C);
  for (int kv = 0; kv < 2; ++kv) {
    const std::vector<float>& s0 = kv ? cv0 : ck0;
    const std::vector<float>& w = kv ? vconv : kconv;
    const int off = kv ? OFF_V : OFF_K, offd = kv ? OFF_VDYN : OFF_KDYN;
    std::vector<double>& o = kv ? nv : nk;
    for (int c = 0; c < C; ++c) {
      std::vector<double> ext(M + 3);
      for (int j = 0; j < 3; ++j) ext[j] = s0[size_t(j) * C + c];
      for (int m = 0; m < M; ++m) ext[3 + m] = proj[size_t(m) * R + off + c];
      for (int m = 0; m < M; ++m) {
        double s = 0;
        for (int lag = 0; lag < 4; ++lag)
          s += (double(w[size_t(lag) * C + c]) + proj[size_t(m) * R + offd + (c / D) * 4 + lag]) * ext[m + 3 - lag];
        o[size_t(m) * C + c] = rbf((s + ext[m + 3]) / 3.0);
      }
      for (int j = 0; j < 3; ++j) (kv ? want_cv : want_ck)[size_t(j) * C + c] = float(ext[M + j]);
    }
  }
  clip_rows(nk, M * HKV, D, QK_CLIP);
  for (int r = 0; r < M * HKV; ++r) {
    double ss = 0;
    for (int d = 0; d < D; ++d) ss += nv[size_t(r) * D + d] * nv[size_t(r) * D + d];
    const double rvn = 1.0 / std::sqrt(ss / D + 1e-6);
    for (int d = 0; d < D; ++d) nv[size_t(r) * D + d] *= rvn;
  }
  for (auto& v : nk) v = rbf(v);
  for (auto& v : nv) v = rbf(v);

  // ---- new rows: the kernel's decoded rows vs a host encode -> decode of the oracle's rows (rare flips allowed: the
  // oracle's BF16 rows can differ from the kernel's in the last bit)
  size_t off = 0, rows_bad = 0;
  std::vector<uint8_t> blk(std::max(kb(), vb()));
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < HKV; ++g)
      for (int kv = 0; kv < 2; ++kv) {
        float row[D];
        for (int d = 0; d < D; ++d) row[d] = float((kv ? nv : nk)[(size_t(m) * HKV + g) * D + d]);
        std::fill(blk.begin(), blk.end(), uint8_t(0));
        const int j = (p0 + m) % BLK;
        encode(kv ? g_f->v : g_f->k, kv == 0, row, blk.data(), j);
        const std::vector<double>& got = kv ? q.v : q.k;
        const size_t base = (size_t(p0 + m) * HKV + g) * D;
        double e = 0, n2 = 0;
        for (int d = 0; d < D; ++d) {
          const double want = kv ? value_at(blk.data(), j, d) : key_at(blk.data(), j, d);
          off += std::fabs(got[base + d] - want) > 1e-6 * (1 + std::fabs(want));
          e += (got[base + d] - want) * (got[base + d] - want), n2 += want * want;
        }
        rows_bad += e > 0.01 * n2;
      }
  const size_t nvals = size_t(M) * C * 2;
  const bool rows_ok = rows_bad == 0 && off <= std::max<size_t>(4, nvals / 50);
  std::printf("%s %s p0=%d M=%d new rows: %zu/%zu values differ from the host encoder, %zu rows off by > 10 %% L2\n",
              rows_ok ? "PASS" : "FAIL", g_f->name, p0, M, off, nvals, rows_bad);

  // ---- oracle attention over the dequantized cache (rows < p0 + M, as the kernel wrote them); INFO at the deepest
  // p0: the format's error against attention over the unquantized rows
  std::vector<double> exact(size_t(M) * HQ * D), spec(exact.size()), plain(exact.size());
  const bool info = p0 == 9005;
  Deq raw;
  raw.k.resize(info ? size_t(p0 + M) * C : 0), raw.v.resize(raw.k.size());
  for (int r = 0; info && r < p0 + M; ++r)
    for (int g = 0; g < HKV; ++g) {
      std::vector<double> x(D), y(D);
      for (int d = 0; d < D; ++d) {
        const size_t i = (size_t(r) * HKV + g) * D + d;
        x[d] = r < p0 ? rk[i] : nk[(size_t(r - p0) * HKV + g) * D + d];
        y[d] = r < p0 ? rv[i] : nv[(size_t(r - p0) * HKV + g) * D + d];
      }
      const std::vector<double> xr = rot(x.data()), yr = rot(y.data());
      std::copy(xr.begin(), xr.end(), raw.k.begin() + (size_t(r) * HKV + g) * D);
      std::copy(yr.begin(), yr.end(), raw.v.begin() + (size_t(r) * HKV + g) * D);
    }
  for (int m = 0; m < M; ++m)
    for (int h = 0; h < HQ; ++h) {
      auto bias = [&](int d) { return d < EXT ? rl[(size_t(m) * HQ + h) * EXT + d] : 0.0; };
      const double sink = proj[size_t(m) * R + OFF_SINK + h];
      const std::vector<double> o = attend(q, &qr[(size_t(m) * HQ + h) * D], h / 3, p0 + m, sink, bias);
      const std::vector<double> op = info ? attend(raw, &qplain[(size_t(m) * HQ + h) * D], h / 3, p0 + m, sink, bias) : o;
      for (int d = 0; d < D; ++d) {
        const size_t i = (size_t(m) * HQ + h) * D + d;
        const double gate = 1.0 / (1.0 + std::exp(-double(proj[size_t(m) * R + OFF_GATE + h * D + d])));
        exact[i] = o[d] * gate, spec[i] = rbf(rbf(o[d]) * gate), plain[i] = op[d] * gate;
      }
    }
  const std::string n = std::string(g_f->name) + " p0=" + std::to_string(p0) + " M=" + std::to_string(M);
  const std::vector<float> got = bf16_download(out);
  ok &= rows_ok && Check{(n + " out").c_str(), 1.0 / 128, 2e-3, 0.05}(got, exact, &spec);
  double se = 0, sr = 0;
  for (size_t i = 0; i < got.size(); ++i) se += (got[i] - plain[i]) * (got[i] - plain[i]), sr += plain[i] * plain[i];
  if (info) std::printf("INFO %s: output error vs attention over the unquantized rows, relative RMS %.4f\n", n.c_str(), std::sqrt(se / sr));
  std::vector<double> sk(want_ck.begin(), want_ck.end()), sv(want_cv.begin(), want_cv.end());
  ok &= Check{(n + " conv_k state").c_str(), 0, 0, 0}(dck.download(), sk);
  ok &= Check{(n + " conv_v state").c_str(), 0, 0, 0}(dcv.download(), sv);

  // ---- stale rows: random bytes in every block after the one holding p0 + M - 1, and in the slots >= p0 + M of
  // that block (codes and scales; NaN-like scales on the second pass), conv states reset: output bits unchanged
  size_t diff = 0;
  const int first = (p0 + M + BLK - 1) / BLK;
  for (int rep = 0; rep < 2; ++rep) {
    std::vector<uint8_t> k2 = kc, v2 = vc;
    for (size_t i = size_t(first) * HKV * kb(); i < k2.size(); ++i) k2[i] = uint8_t(rng.next());
    for (size_t i = size_t(first) * HKV * vb(); i < v2.size(); ++i) v2[i] = uint8_t(rng.next());
    for (int j = (p0 + M) % BLK; j > 0 && j < BLK; ++j)
      for (int g = 0; g < HKV; ++g)
        scramble(&k2[(size_t((p0 + M) / BLK) * HKV + g) * kb()], &v2[(size_t((p0 + M) / BLK) * HKV + g) * vb()], j, rng,
                 rep == 1);
    dk.upload(k2);
    dv.upload(v2);
    dck.upload(ck0);
    dcv.upload(cv0);
    attn::attn_decode(a, g_stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
    const std::vector<__nv_bfloat16> again = out.download();
    for (size_t i = 0; i < again.size(); ++i) diff += std::memcmp(&again[i], &out_first[i], 2) != 0;
  }
  std::printf("%s %s stale rows after p0 + M: %zu output values differ\n", diff ? "FAIL" : "PASS", n.c_str(), diff);
  return ok && diff == 0;
}

// Prefill: T tokens at pos0 against the packed cache rows 0 .. pos0 + T - 1, random q / rel logits / gate / sink.
bool run_prefill(Rng& rng, int pos0, int T) {
  const int L = pos0 + T;
  std::vector<float> rk, rv;
  cache_rows(rng, L, rk, rv);
  std::vector<float> qv = randn_bf16(rng, size_t(T) * HQ * D, 1.5), rel = randn_bf16(rng, size_t(T) * HQ * EXT, 1.0),
                     pj = randn_bf16(rng, size_t(T) * R, 1.0);
  DeviceArray<__nv_bfloat16> drk(to_bf16(rk)), drv(to_bf16(rv)), dq(to_bf16(qv)), drel(to_bf16(rel)), dpj(to_bf16(pj));
  DeviceArray<uint8_t> dk(attn::cache_bytes(g_f->kv, L, false)), dv(attn::cache_bytes(g_f->kv, L, true));
  attn::pack_rows(g_f->kv, drk.get(), drv.get(), L, 0, dk.get(), dv.get(), g_stream);
  DeviceArray<__nv_bfloat16> out(size_t(T) * HQ * D);
  attn::PrefillArgs a;
  a.q = dq.get(), a.rel = drel.get(), a.proj = dpj.get(), a.ld = R, a.T = T, a.pos0 = pos0;
  a.k_cache = dk.get(), a.v_cache = dv.get(), a.kv = g_f->kv, a.out = out.get();
  attn::prefill_attend(a, g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  const Deq q = dequant(dk.download(), dv.download(), L);
  constexpr int GATE = HQ * D, SINK = 2 * HQ * D + HQ * DREL;
  std::vector<double> exact(size_t(T) * HQ * D), spec(exact.size());
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < HQ; ++h) {
      std::vector<double> qd(D);
      for (int d = 0; d < D; ++d) qd[d] = qv[(size_t(t) * HQ + h) * D + d];
      std::vector<double> qr = rot(qd.data());
      for (auto& x : qr) x = f16(x);
      const std::vector<double> o = attend(q, qr.data(), h / 3, pos0 + t, pj[size_t(t) * R + SINK + h],
                                           [&](int d) { return d < EXT ? double(rel[(size_t(t) * HQ + h) * EXT + d]) : 0.0; });
      for (int d = 0; d < D; ++d) {
        const size_t i = (size_t(t) * HQ + h) * D + d;
        const double gate = 1.0 / (1.0 + std::exp(-double(pj[size_t(t) * R + GATE + h * D + d])));
        exact[i] = o[d] * gate, spec[i] = rbf(rbf(o[d]) * gate);
      }
    }
  const std::string n = std::string(g_f->name) + " prefill pos0=" + std::to_string(pos0) + " T=" + std::to_string(T);
  return Check{(n + " out").c_str(), 1.0 / 128, 2e-3, 0.05}(bf16_download(out), exact, &spec);
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  bool ok = true;
  for (const Fmt& f : FORMATS) {
    g_f = &f;
    const bool sizes = attn::cache_bytes(f.kv, BLK, false) == size_t(HKV) * kb() &&
                       attn::cache_bytes(f.kv, BLK, true) == size_t(HKV) * vb();
    std::printf("%s %s block bytes K %d V %d (%.3f bits per value)\n", sizes ? "PASS" : "FAIL", f.name, kb(), vb(),
                8.0 * (kb() + vb()) / (2.0 * BLK * D));
    ok &= sizes;
    Rng rng(4243);
    for (int p0 : {0, 3000, 9005})
      for (int M : {1, 2, 3, 8}) ok &= run(rng, M, p0);
    for (int M = 1; M <= 8; ++M) ok &= run(rng, M, 4637);
    ok &= run_prefill(rng, 0, 77);
    ok &= run_prefill(rng, 1500, 300);
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
