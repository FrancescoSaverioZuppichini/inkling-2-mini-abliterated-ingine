// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// cuBLASLt BF16 GEMM for the prefill projections (from fngine src/kernels/linear/linear_bf16.cu): FP32
// compute, split-K partial sums reduced in FP32 only (torch's allow_bf16_reduced_precision_reduction = False),
// one algorithm cached per shape: the heuristic's first, or on compute capability 8.9 a pinned kernel for large shapes.
#include "kernels/prefill/prefill.h"

#include <cublasLt.h>

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ingine::kernels::prefill {

namespace {

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: status " +
                             std::to_string(static_cast<int>(status)));
}

constexpr std::size_t kWorkspaceBytes = 32u << 20;

}  // namespace

struct Gemm::Impl {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  int pin_min_tiles = 0;  // > 0 on compute capability 8.9: see pinned()
  // (M, N, K, lda, ldb, ldc, fp32 output) -> algorithm
  std::map<std::tuple<int, int, int, int, int, int, bool>, cublasLtMatmulAlgo_t> algos;
};

// [Agent] Ada (compute capability 8.9, measured on the RTX 4090): cuBLASLt's heuristic picks K-sliced sm_80 kernels
// for the large prefill GEMMs, and CUTLASS's 128x128 five-stage kernel (algorithm 21) runs them 24-36 % faster once its
// tiles outnumber half the SMs and K >= 1024. It sums K in plain order, so prefill bits differ from the sliced kernels'.
// Returns false, and the heuristic is used, when this cuBLASLt rejects the config.
static bool pinned(cublasLtHandle_t handle, cublasLtMatmulDesc_t op, cublasLtMatrixLayout_t la,
                   cublasLtMatrixLayout_t lb, cublasLtMatrixLayout_t lc, std::size_t workspace,
                   cublasLtMatmulAlgo_t* algo) {
  if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                             21, algo) != CUBLAS_STATUS_SUCCESS)
    return false;
  const std::uint32_t tile = CUBLASLT_MATMUL_TILE_128x128, stages = CUBLASLT_MATMUL_STAGES_32x5, splitk = 1,
                      reduction = CUBLASLT_REDUCTION_SCHEME_NONE, swizzle = 0, custom = 0;
  cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
  cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages));
  cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk));
  cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction));
  cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom));
  cublasLtMatmulHeuristicResult_t check{};
  return cublasLtMatmulAlgoCheck(handle, op, la, lb, lc, lc, algo, &check) == CUBLAS_STATUS_SUCCESS &&
         check.workspaceSize <= workspace;
}

Gemm::Gemm() : impl_(new Impl) {
  try {
    cublas_check(cublasLtCreate(&impl_->handle), "create");
    if (cudaMalloc(&impl_->workspace, kWorkspaceBytes) != cudaSuccess)
      throw std::runtime_error("prefill Gemm: workspace allocation failed");
    int dev = 0, major = 0, minor = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (major == 8 && minor == 9) impl_->pin_min_tiles = sms / 2;
  } catch (...) {
    if (impl_->handle) cublasLtDestroy(impl_->handle);
    delete impl_;
    throw;
  }
}

Gemm::~Gemm() {
  if (impl_->workspace) cudaFree(impl_->workspace);
  if (impl_->handle) cublasLtDestroy(impl_->handle);
  delete impl_;
}

// Row-major C[M, N] = A[M, K] . B[N, K]^T is column-major C^T[N, M] = B^T(op T on the K x N matrix B) . A
// (the K x M matrix A).
void Gemm::run(const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb, void* C, int ldc, bool c_fp32,
               int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("prefill Gemm: empty shape");
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  auto cleanup = [&] {
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (op) cublasLtMatmulDescDestroy(op);
  };
  try {
    cublas_check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc create");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublas_check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    cublas_check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    cublas_check(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, ldb), "layout a");
    cublas_check(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, lda), "layout b");
    cublas_check(cublasLtMatrixLayoutCreate(&lc, c_fp32 ? CUDA_R_32F : CUDA_R_16BF, N, M, ldc), "layout c");

    const auto key = std::make_tuple(M, N, K, lda, ldb, ldc, c_fp32);
    auto it = impl_->algos.find(key);
    cublasLtMatmulAlgo_t pin;
    if (it == impl_->algos.end() && impl_->pin_min_tiles > 0 && !c_fp32 && K >= 1024 &&
        long((M + 127) / 128) * ((N + 127) / 128) > impl_->pin_min_tiles &&
        pinned(impl_->handle, op, la, lb, lc, kWorkspaceBytes, &pin))
      it = impl_->algos.emplace(key, pin).first;
    if (it == impl_->algos.end()) {
      cublasLtMatmulPreference_t pref = nullptr;
      cublas_check(cublasLtMatmulPreferenceCreate(&pref), "preference create");
      const std::size_t ws = kWorkspaceBytes;
      const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
      cublasLtMatmulHeuristicResult_t result{};
      int found = 0;
      cublasStatus_t st =
          cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                                  sizeof(reduction));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulAlgoGetHeuristic(impl_->handle, op, la, lb, lc, lc, pref, 1, &result, &found);
      cublasLtMatmulPreferenceDestroy(pref);
      cublas_check(st, "heuristic");
      if (found == 0) throw std::runtime_error("prefill Gemm: no cuBLASLt algorithm");
      it = impl_->algos.emplace(key, result.algo).first;
    }
    const float alpha = 1.f, beta = 0.f;
    cublas_check(cublasLtMatmul(impl_->handle, op, &alpha, B, la, A, lb, &beta, C, lc, C, lc, &it->second,
                                impl_->workspace, kWorkspaceBytes, stream),
                 "matmul");
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

}  // namespace ingine::kernels::prefill
