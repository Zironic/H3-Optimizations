// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// EXPERIMENTAL launcher for the block-sparse pure-INT8 attention kernel.
//
// Pinned to head_dim 128, non-causal, no attention mask. Production uses
// 64Q x 64KV; exact 128Q x 128KV, 128Q x 64KV, and 64Q x 128KV quality arms
// share the same carrier and route ABI. No other template combinations are
// built.

#include "qk_int_sv_i8_sparse_cuda.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

constexpr int SPARSE_HEAD_DIM = 128;

template <int SPARSE_CTA_Q, int SPARSE_CTA_K, typename DTypeOut, bool RETURN_LSE>
void launch_sparse_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o,
                        float *lse, float *q_scale, float *k_scale, float *v_scale,
                        const int32_t *lut, const int32_t *valid_block_num,
                        int lut_stride, int qo_len, int kv_len,
                        int num_qo_heads, int num_kv_groups, int stride_bz_q,
                        int stride_seq_q, int stride_h_q, int stride_bz_k,
                        int stride_seq_k, int stride_h_k, int stride_bz_v,
                        int stride_h_v, int stride_d_v, int stride_bz_o,
                        int stride_seq_o, int stride_h_o, float sm_scale,
                        int stride_bz_q_scale, int stride_h_q_scale,
                        int batch_size, cudaStream_t stream) {
  // Same warp tiling the dense kernel picks for head_dim 128: a 16-row warp
  // tile halves the live FP32 output accumulator set.
  constexpr int WARP_Q = 16;
  constexpr int WARP_K = SPARSE_CTA_K;

  size_t smem_max = std::max(
      static_cast<size_t>(SPARSE_CTA_Q * SPARSE_HEAD_DIM * sizeof(int8_t) +
                          SPARSE_CTA_K * SPARSE_HEAD_DIM * sizeof(int8_t) +
                          SPARSE_CTA_K * SPARSE_HEAD_DIM * sizeof(int8_t)),
      static_cast<size_t>(SPARSE_CTA_Q * SPARSE_HEAD_DIM * sizeof(half)));

  auto kernel = qk_int_sv_i8_sparse_attn_kernel<
      SPARSE_CTA_Q, SPARSE_CTA_K, WARP_Q, WARP_K, SPARSE_HEAD_DIM,
      DataType::kInt8, QuantGranularity::kPerThread,
      QuantGranularity::kPerThread, float, false, DTypeOut,
      ComputeUnit::kCudaCore, MaskMode::kNone, RETURN_LSE, true, false, false, true>;

  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_max));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "sage_attn_sparse failed to request " + std::to_string(smem_max) +
        " bytes of dynamic shared memory: " + cudaGetErrorString(error));
  }

  dim3 grid(div_ceil(qo_len, SPARSE_CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (SPARSE_CTA_Q / WARP_Q) * (SPARSE_CTA_K / WARP_K));

  kernel<<<grid, block, smem_max, stream>>>(
      q, k, v, o, lse, q_scale, k_scale, v_scale, nullptr, nullptr, 0, 0, 0,
      0, 0, qo_len, kv_len, num_kv_groups, stride_bz_q, stride_seq_q,
      stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v,
      stride_h_v, stride_d_v, stride_bz_o, stride_seq_o, stride_h_o,
      stride_bz_q_scale, stride_h_q_scale, sm_scale, lut, valid_block_num,
      static_cast<uint32_t>(lut_stride));

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("sage_attn_sparse kernel launch failed: ") +
        cudaGetErrorString(error));
  }
}

} // anonymous namespace

// The route encoding is a build-time property of the kernel, so report it from
// the binary rather than letting a Python constant drift out of step with it.
// VENDORING CHANGE: these launchers were `extern "C"` upstream, where they are
// reached through nanobind. Here they are internal to this shared library and
// only h3_int8_attention_api.cu calls them, so C linkage buys nothing -- and it
// costs correctness. Under MSVC's /EHsc the trailing `c` means "extern \"C\"
// functions never throw", so the compiler elides the unwind and a throw from
// here terminates the process instead of reaching the catch in the API layer.
// That is the C4297 warning upstream emits, made fatal. C++ linkage fixes it
// regardless of which generator supplies the flags.

const char *sage_attn_sparse_route_encoding() {
  return H3_SPARSE_ROUTE_ENCODING;
}

template <bool RETURN_LSE>
void launch_sparse(
    const void *q, const void *k, const void *v, void *o, void *lse,
    const void *q_scale, const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, int lut_stride, int cta_q, int cta_k,
    int batch_size,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads, int head_dim,
    int stride_bz_q, int stride_seq_q, int stride_h_q, int stride_bz_k,
    int stride_seq_k, int stride_h_k, int stride_bz_v, int stride_h_v,
    int stride_d_v, int stride_bz_o, int stride_seq_o, int stride_h_o,
    int stride_bz_q_scale, int stride_h_q_scale, float sm_scale,
    int output_dtype_code, cudaStream_t stream) {
  if (!((cta_q == 128 || cta_q == 64) &&
        (cta_k == 128 || cta_k == 64))) {
    throw std::runtime_error(
        "sage_attn_sparse: unsupported geometry " + std::to_string(cta_q) +
        "Q x " + std::to_string(cta_k) + "KV");
  }
  if (head_dim != SPARSE_HEAD_DIM) {
    throw std::runtime_error(
        "sage_attn_sparse: the experimental kernel is fixed to head_dim 128, "
        "got " +
        std::to_string(head_dim));
  }
  if (lut == nullptr || valid_block_num == nullptr) {
    throw std::runtime_error("sage_attn_sparse: a route LUT is required");
  }

  int num_kv_groups = num_qo_heads / num_kv_heads;

  // The kernel takes non-const pointers but does not modify its inputs.
  auto q_ = const_cast<int8_t *>(static_cast<const int8_t *>(q));
  auto k_ = const_cast<int8_t *>(static_cast<const int8_t *>(k));
  auto v_ = const_cast<int8_t *>(static_cast<const int8_t *>(v));
  auto qs_ = const_cast<float *>(static_cast<const float *>(q_scale));
  auto ks_ = const_cast<float *>(static_cast<const float *>(k_scale));
  auto vs_ = const_cast<float *>(static_cast<const float *>(v_scale));
  auto lut_ = static_cast<const int32_t *>(lut);
  auto valid_ = static_cast<const int32_t *>(valid_block_num);

#define LAUNCH_SPARSE(CQ, CK, DT)                                                  \
  launch_sparse_impl<CQ, CK, DT, RETURN_LSE>(q_, k_, v_, static_cast<DT *>(o),     \
                         static_cast<float *>(lse), qs_, ks_, vs_, lut_,            \
                         valid_, lut_stride, qo_len, kv_len, num_qo_heads,     \
                         num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, \
                         stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v,   \
                         stride_h_v, stride_d_v, stride_bz_o, stride_seq_o,    \
                         stride_h_o, sm_scale, stride_bz_q_scale,              \
                         stride_h_q_scale, batch_size, stream)

  if (cta_q == 128 && cta_k == 128) {
    if (output_dtype_code == 1) {
      LAUNCH_SPARSE(128, 128, half);
    } else {
      LAUNCH_SPARSE(128, 128, nv_bfloat16);
    }
  } else if (cta_q == 128) {
    if (output_dtype_code == 1) {
      LAUNCH_SPARSE(128, 64, half);
    } else {
      LAUNCH_SPARSE(128, 64, nv_bfloat16);
    }
  } else if (cta_k == 128) {
    if (output_dtype_code == 1) {
      LAUNCH_SPARSE(64, 128, half);
    } else {
      LAUNCH_SPARSE(64, 128, nv_bfloat16);
    }
  } else {
    if (output_dtype_code == 1) {
      LAUNCH_SPARSE(64, 64, half);
    } else {
      LAUNCH_SPARSE(64, 64, nv_bfloat16);
    }
  }

#undef LAUNCH_SPARSE
}

void launch_sage_attn_sparse_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, int lut_stride, int cta_q, int cta_k,
    int batch_size,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads, int head_dim,
    int stride_bz_q, int stride_seq_q, int stride_h_q, int stride_bz_k,
    int stride_seq_k, int stride_h_k, int stride_bz_v, int stride_h_v,
    int stride_d_v, int stride_bz_o, int stride_seq_o, int stride_h_o,
    int stride_bz_q_scale, int stride_h_q_scale, float sm_scale,
    int output_dtype_code, cudaStream_t stream) {
  launch_sparse<false>(
      q, k, v, o, nullptr, q_scale, k_scale, v_scale, lut, valid_block_num,
      lut_stride, cta_q, cta_k, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
      head_dim, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
      stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o, stride_bz_q_scale,
      stride_h_q_scale, sm_scale, output_dtype_code, stream);
}

void launch_sage_attn_sparse_kernel_lse(
    const void *q, const void *k, const void *v, void *o, void *lse,
    const void *q_scale, const void *k_scale, const void *v_scale,
    const void *lut, const void *valid_block_num, int lut_stride, int cta_q,
    int cta_k, int batch_size, int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int head_dim, int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k, int stride_bz_v,
    int stride_h_v, int stride_d_v, int stride_bz_o, int stride_seq_o,
    int stride_h_o, int stride_bz_q_scale, int stride_h_q_scale,
    float sm_scale, int output_dtype_code, cudaStream_t stream) {
  if (lse == nullptr) {
    throw std::runtime_error("sage_attn_sparse_lse: an LSE output is required");
  }
  launch_sparse<true>(
      q, k, v, o, lse, q_scale, k_scale, v_scale, lut, valid_block_num,
      lut_stride, cta_q, cta_k, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
      head_dim, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
      stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o, stride_bz_q_scale,
      stride_h_q_scale, sm_scale, output_dtype_code, stream);
}
