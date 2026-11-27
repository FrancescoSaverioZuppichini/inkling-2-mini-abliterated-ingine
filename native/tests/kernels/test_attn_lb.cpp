// Low-bit full-attention decode (attn_lb.cu, KvFormat int4) against an FP64 oracle over the DEQUANTIZED cache: the
// oracle reads back the codes and scales the kernels wrote (pack_rows for old rows, the decode kernel for the call's
// own rows) and attends them in double, with the query rotated and rounded to FP16 as the kernel does. So the check
// isolates the reader's arithmetic from the format's loss. Separately: the call's new rows against a host encode of
// the oracle's keys / values (rare boundary flips allowed), conv states, and stale rows after p0 + M (random bytes,
// NaN scales included: output bits must not move). Full layers, p0 = 0 / 3000 / 4637 / 9005, M = 1..8; prefill
// T = 77 and 300. Synthetic layer parameters.
#include "kernels/oracle.h"

#include "kernels/attn/attn.h"
#include "kernels/attn/kv_format.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>

using namespace ingine::test;
namespace attn = ingine::kernels::attn;
namespace kvf = ingine::kernels::attn::kvf;

namespace {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, C = HKV * D, BLK = 16;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632, R = 8664;
constexpr int CAP = 10016;
constexpr double QK_CLIP = 22.627416997969522, REL_CLIP = 8.0;
cudaStream_t g_stream = nullptr;

void clip_rows(std::vector<double>& x, int rows, int n, double t) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += x[size_t(r) * n + i] * x[size_t(r) * n + i];
    const double f = std::min(1.0, t / (std::sqrt(ss) + 1e-6));
    for (int i = 0; i < n; ++i) x[size_t(r) * n + i] *= f;
  }
}

// unnormalised Walsh-Hadamard transform in place (H H = 128 I)
void fwht(std::vector<double>& y) {
  for (int h = 1; h < D; h <<= 1)
    for (int i = 0; i < D; ++i)
      if (!(i & h)) {
        const double a = y[i], b = y[i + h];
        y[i] = a + b, y[i + h] = a - b;
      }
}
// H D x
std::vector<double> rot(const double* x) {
  std::vector<double> y(D);
  for (int d = 0; d < D; ++d) y[d] = x[d] * kvf::sign_of(d);
  fwht(y);
  return y;
}

double f16(double x) { return double(__half2float(__float2half_rn(float(x)))); }

// The int4 cache as the oracle sees it: per (row, head) the rotated key k'' = s (n - 7.5) and the rotated unit value
// x = (n - 7.5) step; plus the raw codes and scales for the new-row check.
struct Deq {
  std::vector<double> k, v;  // [rows][8][128], rotated space
  std::vector<int> kc, vc;   // codes
  std::vector<float> ks;     // [rows][8]
};

Deq dequant(const std::vector<uint8_t>& kb, const std::vector<uint8_t>& vb, int rows) {
  Deq q;
  q.k.resize(size_t(rows) * C), q.v.resize(q.k.size()), q.kc.resize(q.k.size()), q.vc.resize(q.k.size());
  q.ks.resize(size_t(rows) * HKV);
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < HKV; ++g) {
      const uint8_t* kblk = &kb[(size_t(r / BLK) * HKV + g) * kvf::int4::K_BYTES];
      const uint8_t* vblk = &vb[(size_t(r / BLK) * HKV + g) * kvf::int4::V_BYTES];
      const int j = r % BLK;
      __half hs;
      std::memcpy(&hs, kblk + kvf::int4::k_scale_off(j), 2);
      const double s = __half2float(hs);
      q.ks[size_t(r) * HKV + g] = float(s);
      for (int d = 0; d < D; ++d) {
        uint32_t w;
        std::memcpy(&w, kblk + 4 * kvf::int4::k_word_of_dim(j, d), 4);
        const int nk = (w >> (4 * kvf::int4::k_nib_of_dim(d))) & 15;
        const uint8_t byte = vblk[kvf::int4::v_byte(j, d)];
        const int nv = kvf::int4::v_high(d) ? byte >> 4 : byte & 15;
        const size_t i = (size_t(r) * HKV + g) * D + d;
        q.kc[i] = nk, q.vc[i] = nv;
        q.k[i] = s * (nk - 7.5);
        q.v[i] = (nv - 7.5) * kvf::int4::STEP;
      }
    }
  return q;
}

// host encode of one BF16-valued row: codes (and the key's least-squares scale)
void encode(const double* x, bool key, int* codes, double* scale) {
  const std::vector<double> y = rot(x);
  double n2 = 0;
  for (int d = 0; d < D; ++d) n2 += x[d] * x[d];
  const double inv = key ? (n2 > 0 ? 1.0 / std::sqrt(n2) : 0.0) : 1.0 / std::sqrt(128.0);
  double num = 0, den = 0;
  for (int d = 0; d < D; ++d) {
    codes[d] = kvf::int4::code_of(float(y[d] * inv));
    num += y[d] * (codes[d] - 7.5), den += (codes[d] - 7.5) * (codes[d] - 7.5);
  }
  if (scale) *scale = num / den;
}

// FP64 attention of one query row (rotated, FP16 as the kernel) over keys 0..p of KV head g of the dequantized cache,
// sink logit in the denominator, bias(distance) added; the output back in model space (D H o / sqrt(128)).
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
  for (int d = 0; d < D; ++d) o[d] *= kvf::sign_of(d) / std::sqrt(128.0);
  return o;
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
  // cache rows: BF16 keys clipped at 22.63 (norms 0.5..3.5 x sqrt(128) before the clip), unit-RMS values
  std::vector<float> rk(size_t(CAP) * C), rv(rk.size());
  for (int r = 0; r < CAP; ++r)
    for (int g = 0; g < HKV; ++g) {
      std::vector<double> k(D), v(D);
      const double scale = rng.uniform(0.5, 3.5);
      double ss = 0, sv = 0;
      for (int d = 0; d < D; ++d) k[d] = rng.normal() * scale, ss += k[d] * k[d];
      const double f = std::min(1.0, QK_CLIP / (std::sqrt(ss) + 1e-6));
      for (int d = 0; d < D; ++d) v[d] = rng.normal(), sv += v[d] * v[d];
      const double rvn = 1.0 / std::sqrt(sv / D + 1e-6);
      for (int d = 0; d < D; ++d) {
        rk[(size_t(r) * HKV + g) * D + d] = round_bf16(float(k[d] * f));
        rv[(size_t(r) * HKV + g) * D + d] = round_bf16(float(v[d] * rvn));
      }
    }
  DeviceArray<__nv_bfloat16> drk(to_bf16(rk)), drv(to_bf16(rv));
  DeviceArray<uint8_t> dk(attn::cache_bytes(attn::KvFormat::int4, CAP, false)),
      dv(attn::cache_bytes(attn::KvFormat::int4, CAP, true));
  attn::pack_rows(attn::KvFormat::int4, drk.get(), drv.get(), CAP, 0, dk.get(), dv.get(), g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));

  DeviceArray<float> dproj(proj), dqs(q_scale), drs(r_scale), drp(rel_proj), dkc(kconv), dvc(vconv), dck(ck0), dcv(cv0);
  DeviceArray<int64_t> dpos(std::vector<int64_t>{p0});
  DeviceArray<uint8_t> ws(attn::workspace_bytes(8));
  DeviceArray<__nv_bfloat16> out(size_t(M) * HQ * D);
  attn::DecodeArgs a;
  a.proj = dproj.get(), a.M = M, a.q_scale = dqs.get(), a.r_scale = drs.get(), a.rel_proj = drp.get();
  a.k_conv = dkc.get(), a.v_conv = dvc.get(), a.conv_k_state = dck.get(), a.conv_v_state = dcv.get();
  a.k_cache = dk.get(), a.v_cache = dv.get(), a.kv = attn::KvFormat::int4, a.cache_rows = CAP;
  a.pos = dpos.get(), a.window = 0, a.log_scaling = true;
  a.workspace = ws.get(), a.workspace_size = ws.count, a.out = out.get();
  attn::attn_decode(a, g_stream);
  INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream));
  const std::vector<__nv_bfloat16> out_first = out.download();
  const std::vector<uint8_t> kb = dk.download(), vb = dv.download();
  const Deq q = dequant(kb, vb, CAP);

  // ---- oracle: q (rotated, FP16 as the kernel) and rel
  std::vector<double> qr(size_t(M) * HQ * D), rl(size_t(M) * HQ * EXT);
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
      for (int d = 0; d < D; ++d) qr[(size_t(m) * HQ + h) * D + d] = f16(y[d]);
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

  // ---- new rows: the kernel's codes vs a host encode
  size_t kdiff = 0, vdiff = 0, kbig = 0, sbad = 0;
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < HKV; ++g) {
      int ck[D], cv[D];
      double s = 0;
      encode(&nk[(size_t(m) * HKV + g) * D], true, ck, &s);
      encode(&nv[(size_t(m) * HKV + g) * D], false, cv, nullptr);
      const size_t base = (size_t(p0 + m) * HKV + g) * D;
      for (int d = 0; d < D; ++d) {
        kdiff += q.kc[base + d] != ck[d], vdiff += q.vc[base + d] != cv[d];
        kbig += std::abs(q.kc[base + d] - ck[d]) > 1 || std::abs(q.vc[base + d] - cv[d]) > 1;
      }
      sbad += std::fabs(q.ks[size_t(p0 + m) * HKV + g] - s) > 2e-3 * std::fabs(s) + 1e-6;
    }
  const size_t ncodes = size_t(M) * C;
  const bool rows_ok = kbig == 0 && sbad == 0 && kdiff + vdiff <= std::max<size_t>(2, ncodes / 500);
  std::printf("%s int4 p0=%d M=%d new rows: %zu/%zu key and %zu value codes off by one, %zu off by more, %zu bad scales\n",
              rows_ok ? "PASS" : "FAIL", p0, M, kdiff, ncodes, vdiff, kbig, sbad);

  // ---- oracle attention over the dequantized cache (rows < p0 + M, as the kernel wrote them)
  std::vector<double> exact(size_t(M) * HQ * D), spec(exact.size());
  for (int m = 0; m < M; ++m)
    for (int h = 0; h < HQ; ++h) {
      const std::vector<double> o = attend(q, &qr[(size_t(m) * HQ + h) * D], h / 3, p0 + m, proj[size_t(m) * R + OFF_SINK + h],
                                           [&](int d) { return d < EXT ? rl[(size_t(m) * HQ + h) * EXT + d] : 0.0; });
      for (int d = 0; d < D; ++d) {
        const size_t i = (size_t(m) * HQ + h) * D + d;
        const double gate = 1.0 / (1.0 + std::exp(-double(proj[size_t(m) * R + OFF_GATE + h * D + d])));
        exact[i] = o[d] * gate, spec[i] = rbf(rbf(o[d]) * gate);
      }
    }
  const std::string n = "int4 p0=" + std::to_string(p0) + " M=" + std::to_string(M);
  const std::vector<float> got = bf16_download(out);
  bool ok = rows_ok && Check{(n + " out").c_str(), 1.0 / 128, 2e-3, 0.05}(got, exact, &spec);
  std::vector<double> sk(want_ck.begin(), want_ck.end()), sv(want_cv.begin(), want_cv.end());
  ok &= Check{(n + " conv_k state").c_str(), 0, 0, 0}(dck.download(), sk);
  ok &= Check{(n + " conv_v state").c_str(), 0, 0, 0}(dcv.download(), sv);

  // ---- stale rows: random bytes in every block after the one holding p0 + M - 1, and in the slots >= p0 + M of
  // that block (whole keys: codes and scales), conv states reset; the output must be bit-identical
  size_t diff = 0;
  const int first = (p0 + M + BLK - 1) / BLK;
  for (int rep = 0; rep < 2; ++rep) {
    std::vector<uint8_t> k2 = kb, v2 = vb;
    for (size_t i = size_t(first) * HKV * kvf::int4::K_BYTES; i < k2.size(); ++i) k2[i] = uint8_t(rng.next());
    for (size_t i = size_t(first) * HKV * kvf::int4::V_BYTES; i < v2.size(); ++i) v2[i] = uint8_t(rng.next());
    for (int j = (p0 + M) % BLK; j > 0 && j < BLK; ++j)  // the tail of the last partial block
      for (int g = 0; g < HKV; ++g) {
        uint8_t* kblk = &k2[(size_t((p0 + M) / BLK) * HKV + g) * kvf::int4::K_BYTES];
        uint8_t* vblk = &v2[(size_t((p0 + M) / BLK) * HKV + g) * kvf::int4::V_BYTES];
        for (int d = 0; d < D; ++d) {
          const int w = kvf::int4::k_word_of_dim(j, d), nib = kvf::int4::k_nib_of_dim(d);
          uint32_t x;
          std::memcpy(&x, kblk + 4 * w, 4);
          x = (x & ~(15u << (4 * nib))) | (uint32_t(rng.next() & 15) << (4 * nib));
          std::memcpy(kblk + 4 * w, &x, 4);
          uint8_t& vbyte = vblk[kvf::int4::v_byte(j, d)];
          vbyte = kvf::int4::v_high(d) ? uint8_t((vbyte & 15) | (rng.next() & 15) << 4) : uint8_t((vbyte & 0xf0) | (rng.next() & 15));
        }
        const uint16_t junk = rep ? 0x7e00 : uint16_t(rng.next());  // NaN on the second pass
        std::memcpy(kblk + kvf::int4::k_scale_off(j), &junk, 2);
      }
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
  std::vector<float> rk(size_t(L) * C), rv(rk.size());
  for (int r = 0; r < L * HKV; ++r) {
    double ss = 0, sv = 0;
    std::vector<double> k(D), v(D);
    const double scale = rng.uniform(0.5, 3.5);
    for (int d = 0; d < D; ++d) k[d] = rng.normal() * scale, ss += k[d] * k[d], v[d] = rng.normal(), sv += v[d] * v[d];
    const double f = std::min(1.0, QK_CLIP / (std::sqrt(ss) + 1e-6)), rvn = 1.0 / std::sqrt(sv / D + 1e-6);
    for (int d = 0; d < D; ++d) rk[size_t(r) * D + d] = round_bf16(float(k[d] * f)), rv[size_t(r) * D + d] = round_bf16(float(v[d] * rvn));
  }
  std::vector<float> qv = randn_bf16(rng, size_t(T) * HQ * D, 1.5), rel = randn_bf16(rng, size_t(T) * HQ * EXT, 1.0),
                     pj = randn_bf16(rng, size_t(T) * R, 1.0);
  DeviceArray<__nv_bfloat16> drk(to_bf16(rk)), drv(to_bf16(rv)), dq(to_bf16(qv)), drel(to_bf16(rel)), dpj(to_bf16(pj));
  DeviceArray<uint8_t> dk(attn::cache_bytes(attn::KvFormat::int4, L, false)), dv(attn::cache_bytes(attn::KvFormat::int4, L, true));
  attn::pack_rows(attn::KvFormat::int4, drk.get(), drv.get(), L, 0, dk.get(), dv.get(), g_stream);
  DeviceArray<__nv_bfloat16> out(size_t(T) * HQ * D);
  attn::PrefillArgs a;
  a.q = dq.get(), a.rel = drel.get(), a.proj = dpj.get(), a.ld = R, a.T = T, a.pos0 = pos0;
  a.k_cache = dk.get(), a.v_cache = dv.get(), a.kv = attn::KvFormat::int4, a.out = out.get();
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
  const std::string n = "int4 prefill pos0=" + std::to_string(pos0) + " T=" + std::to_string(T);
  return Check{(n + " out").c_str(), 1.0 / 128, 2e-3, 0.05}(bf16_download(out), exact, &spec);
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  Rng rng(4243);
  bool ok = true;
  for (int p0 : {0, 3000, 9005})
    for (int M : {1, 2, 3, 8}) ok &= run(rng, M, p0);
  for (int M = 1; M <= 8; ++M) ok &= run(rng, M, 4637);
  ok &= run_prefill(rng, 0, 77);
  ok &= run_prefill(rng, 1500, 300);
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
