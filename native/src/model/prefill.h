// Native prompt processing: model.Model.prefill for one chunk, without PyTorch.
#pragma once

#include "core/arena.h"
#include "model/model_data.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}

namespace ingine::model {

// Processes prompt tokens [T] (device int32) at positions pos0..pos0+T-1 on top of `state` (exactly
// model.Model.prefill for one chunk): writes caches/conv state, sets state.length = pos0 + T, returns the last
// token's logits (device fp32 [UNPADDED_VOCAB], softcapped like the head) and, if taps != nullptr, every row's
// DSpark taps (bf16 [T, 4, H], model.TAP_VISITS + final normed hidden).
//
// The visit sequence is fast.FastDecoder.decode's (the same block-state wiring and mix / message-gate kernels)
// with T rows; projections are cuBLASLt BF16 GEMMs over the dequantized weight rows (weights.dequant), the
// attention a flash-attention prefill kernel, the MoE grouped GEMMs over the experts decoded on the fly.
// Stream-ordered on `stream`; run() only synchronises for its first use of a block-pointer table.
// Embedding rows computed elsewhere (image tokens from the vision tower) for rows [row, row + count) of a
// chunk: src bf16 [count, HIDDEN] on the device replaces the token embeddings of those rows (which the model
// feeds in as they are: no embedding norm).
struct EmbedRows {
  int row = 0, count = 0;
  const __nv_bfloat16* src = nullptr;
};

class Prefiller {
 public:
  Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream);
  ~Prefiller();
  Prefiller(const Prefiller&) = delete;
  Prefiller& operator=(const Prefiller&) = delete;

  // Tokens with a negative id must be covered by `rows` (their embedding is the image placeholder's otherwise).
  void run(State& state, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
           std::span<const EmbedRows> rows = {});

  int max_chunk() const { return max_chunk_; }
  std::size_t device_bytes() const;  // scratch held by this object
  // [Agent] The front of the scratch holds activations only: inside a run() every byte of it is written before it is
  // read, so between two run() calls it holds no data. A caller may keep its own data there between runs, as long as
  // that data is dead from the first run() it queues until the data is written again.
  DeviceSpan idle() const { return {B0_, idle_bytes_}; }

 private:
  const float* const* block_table(const std::vector<const float*>& blocks);

  const Weights& w_;
  const Ablation& abl_;
  int max_chunk_;
  cudaStream_t stream_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;
  DeviceHeap heap_;  // all scratch, back to back from B0_
  std::size_t idle_bytes_ = 0;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;

  // activations (rows = max_chunk), see prefill.cpp
  float *B0_, *B1_, *B2_[PASSES], *E_[PASSES], *hid_, *part_, *y_, *y2_;
  __nv_bfloat16 *xin_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_;
  __nv_bfloat16 *proj_, *q_, *rel16_, *rel_, *krows_, *vrows_, *attn_out_, *wo_out_;
  int* kpos_;
  int4* bounds_;
  __nv_bfloat16 *gin_, *g16_;
  float *fpre_, *gpre_;
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  __nv_bfloat16 *act_, *down_, *moe_sum_;
  __nv_bfloat16* wscratch_;  // dequantized FP8 / INT8 rows of wproj / wo (quantized dense formats)
  std::int64_t* ids_;
  int kpos_rows_;
  bool split_sliding_;
};

}  // namespace ingine::model
