// Full-attention decode timing: today's FP8 kernel vs the low-bit kernel, one visit (8 KV heads), synthetic caches
// at production depth. Exits 77 (skip) unless INGINE_BENCH=1. Args: depths in tokens (default 229376 507904).
// Per (format, depth, M): 2 warm-ups, then calls interleaved A/B over the formats, median of 15; prints us, the
// cache bytes the call reads and GB/s. The kill metric of S4-08 E1 is int4 GB/s / fp8 GB/s at the same depth.
#include "kernels/attn/attn.h"

#include "common/test_util.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace attn = ingine::kernels::attn;
using namespace ingine::test;

namespace {

constexpr int C = 1024, R = 8664;

__device__ float uni(unsigned long long i, unsigned seed) {
    unsigned long long x = i * 0x9E3779B97F4A7C15ull + seed;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return float((x ^ (x >> 31)) >> 40) * (2.f / 16777216.f) - 1.f;  // [-1, 1)
}
__global__ void fill_bf16(__nv_bfloat16* p, size_t n, float scale, unsigned seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        p[i] = __float2bfloat16(uni(i, seed) * scale);
}
__global__ void fill_fp8(unsigned char* p, size_t n, float scale, unsigned seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        p[i] = __nv_cvt_float_to_fp8(uni(i, seed) * scale, __NV_SATFINITE, __NV_E4M3);
}
__global__ void fill_f32(float* p, size_t n, float scale, unsigned seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        p[i] = uni(i, seed) * scale;
}

struct Arm {
    std::string name;
    attn::KvFormat kv;
    bool fp8;
    void *k = nullptr, *v = nullptr;
    size_t bytes = 0;  // cache bytes one call reads at depth L
};

}  // namespace

int main(int argc, char** argv) {
    const char* on = std::getenv("INGINE_BENCH");
    if (!on || std::string(on) != "1") return kSkip;
    std::vector<long long> depths;
    for (int i = 1; i < argc; ++i) depths.push_back(std::atoll(argv[i]));
    if (depths.empty()) depths = {229376, 507904};
    const long long cap = *std::max_element(depths.begin(), depths.end()) + 16;
    cudaStream_t s;
    INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // layer inputs: small random values (scores stay moderate, as in the model)
    float *proj, *qs, *rs, *rp, *kw, *vw, *ck, *cv;
    INGINE_TEST_CUDA(cudaMalloc(&proj, 8 * R * 4));
    for (float** p : {&qs, &rs, &rp, &kw, &vw, &ck, &cv}) INGINE_TEST_CUDA(cudaMalloc(p, 24 * 1024 * 4));
    fill_f32<<<64, 256, 0, s>>>(proj, 8 * R, 1.f, 1);
    for (float* p : {qs, rs, rp, kw, vw, ck, cv}) fill_f32<<<64, 256, 0, s>>>(p, 24 * 1024, 0.5f, 2);
    void* ws;
    const size_t wsb = attn::workspace_bytes(8);
    INGINE_TEST_CUDA(cudaMalloc(&ws, wsb));
    INGINE_TEST_CUDA(cudaMemset(ws, 0, wsb));
    __nv_bfloat16* out;
    INGINE_TEST_CUDA(cudaMalloc(&out, 8 * 3072 * 2));
    std::int64_t* pos;
    INGINE_TEST_CUDA(cudaMalloc(&pos, 8));

    std::vector<Arm> arms = {{"fp8", attn::KvFormat::fp8, true}, {"int4", attn::KvFormat::int4, false}};
    for (Arm& a : arms) {
        const size_t kb = attn::cache_bytes(a.kv, cap, false), vb = attn::cache_bytes(a.kv, cap, true);
        INGINE_TEST_CUDA(cudaMalloc(&a.k, kb));
        INGINE_TEST_CUDA(cudaMalloc(&a.v, vb));
        if (a.fp8) {
            fill_fp8<<<1024, 256, 0, s>>>(static_cast<unsigned char*>(a.k), kb, 1.5f, 3);
            fill_fp8<<<1024, 256, 0, s>>>(static_cast<unsigned char*>(a.v), vb, 1.7f, 4);
        } else {  // realistic codes: pack random BF16 rows in chunks
            const int chunk = 16384;
            __nv_bfloat16 *rk, *rv;
            INGINE_TEST_CUDA(cudaMalloc(&rk, size_t(chunk) * C * 2));
            INGINE_TEST_CUDA(cudaMalloc(&rv, size_t(chunk) * C * 2));
            for (long long p0 = 0; p0 < cap; p0 += chunk) {
                const int T = int(std::min<long long>(chunk, cap - p0));
                fill_bf16<<<1024, 256, 0, s>>>(rk, size_t(T) * C, 1.5f, unsigned(p0) + 5);
                fill_bf16<<<1024, 256, 0, s>>>(rv, size_t(T) * C, 1.7f, unsigned(p0) + 6);
                attn::pack_rows(a.kv, rk, rv, T, p0, a.k, a.v, s);
            }
            INGINE_TEST_CUDA(cudaStreamSynchronize(s));
            cudaFree(rk), cudaFree(rv);
        }
    }
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    int dev = 0;
    cudaDeviceProp prop{};
    INGINE_TEST_CUDA(cudaGetDevice(&dev));
    INGINE_TEST_CUDA(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::printf("%-5s %8s %2s %10s %10s %8s\n", "fmt", "depth", "M", "us", "MB", "GB/s");

    cudaEvent_t e0, e1;
    INGINE_TEST_CUDA(cudaEventCreate(&e0));
    INGINE_TEST_CUDA(cudaEventCreate(&e1));
    for (long long L : depths)
        for (int M : {1, 2, 4, 8}) {
            const std::int64_t p0 = L - M;
            INGINE_TEST_CUDA(cudaMemcpy(pos, &p0, 8, cudaMemcpyHostToDevice));
            std::vector<std::vector<float>> t(arms.size());
            for (int rep = 0; rep < 17; ++rep)
                for (size_t ai = 0; ai < arms.size(); ++ai) {
                    const Arm& a = arms[ai];
                    if (a.fp8 && L > 300000) continue;  // FP8 at 500K is not a configuration (does not fit)
                    attn::DecodeArgs d;
                    d.proj = proj, d.M = M, d.q_scale = qs, d.r_scale = rs, d.rel_proj = rp, d.k_conv = kw, d.v_conv = vw;
                    d.conv_k_state = ck, d.conv_v_state = cv, d.k_cache = a.k, d.v_cache = a.v, d.cache_fp8 = a.fp8;
                    d.kv = a.kv, d.cache_rows = cap, d.pos = pos, d.window = 0, d.log_scaling = true;
                    d.workspace = ws, d.workspace_size = wsb, d.out = out;
                    INGINE_TEST_CUDA(cudaEventRecord(e0, s));
                    attn::attn_decode(d, s);
                    INGINE_TEST_CUDA(cudaEventRecord(e1, s));
                    INGINE_TEST_CUDA(cudaEventSynchronize(e1));
                    float ms = 0;
                    INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, e0, e1));
                    if (rep >= 2) t[ai].push_back(ms * 1000.f);
                }
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                if (t[ai].empty()) continue;
                std::sort(t[ai].begin(), t[ai].end());
                const float us = t[ai][t[ai].size() / 2];
                const double mb = double(attn::cache_bytes(arms[ai].kv, p0, false) + attn::cache_bytes(arms[ai].kv, p0, true)) / 1e6;
                std::printf("%-5s %8lld %2d %10.1f %10.1f %8.1f  (min %.1f max %.1f)\n", arms[ai].name.c_str(), L, M, us, mb,
                            mb / us * 1e3, t[ai].front(), t[ai].back());
            }
        }
    return 0;
}
