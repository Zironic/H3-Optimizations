// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// EXPERIMENTAL launcher for the block-sparse pure-INT8 attention kernel.
//
// Pinned to head_dim 128, non-causal, no attention mask. Production uses
// 64Q x 64KV routes, traversed by the packed 64Q x 128KV kernel on SM86,
// SM89 and SM12x and by the 64Q x 64KV kernel elsewhere; exact 128Q x 128KV,
// 128Q x 64KV, and 64Q x 128KV quality arms share the same carrier and route
// ABI. 64Q x 32KV routes 32-token subtiles over the same 64-token carrier with
// the packed fine-route kernel. No other template combinations are built.

#include "qk_int_sv_i8_packed_cuda.cuh"
#include "qk_int_sv_i8_packed_fine_cuda.cuh"
#include "qk_int_sv_i8_sparse_cuda.cuh"
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <string>

namespace {

constexpr int SPARSE_HEAD_DIM = 128;

// The packed kernel wins by energy, not cycles: it takes 1.5-4% more cycles
// than 64Q x 64KV but 1.4-5.7% less energy per call, and a power-limited
// card's governor turns that into clock. Use it only where the 64Q x 64KV
// kernel already gets its consumer launch bound (SM86, SM89, SM12x); a card
// that is not power-limited would pay the extra cycles.
bool packed_64_preferred() {
  constexpr int max_devices = 64;
  static std::atomic<int> cached[max_devices];  // zero: unknown; 1 no, 2 yes
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      device >= max_devices) {
    return false;
  }
  int state = cached[device].load(std::memory_order_relaxed);
  if (state == 0) {
    int major = 0;
    int minor = 0;
    const bool known =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) == cudaSuccess;
    const bool consumer =
        known && ((major == 8 && (minor == 6 || minor == 9)) || major == 12);
    state = consumer ? 2 : 1;
    cached[device].store(state, std::memory_order_relaxed);
  }
  return state == 2;
}

template <typename DTypeOut, bool RETURN_LSE, bool DENSE = false>
void launch_packed_64_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o,
                           float *lse, float *q_scale, float *k_scale,
                           float *v_scale, const int32_t *lut,
                           const int32_t *valid_block_num, int lut_stride,
                           int qo_len, int kv_len, int num_qo_heads,
                           int num_kv_groups, int stride_bz_q,
                           int stride_seq_q, int stride_h_q, int stride_bz_k,
                           int stride_seq_k, int stride_h_k, int stride_bz_v,
                           int stride_h_v, int stride_d_v, int stride_bz_o,
                           int stride_seq_o, int stride_h_o, float sm_scale,
                           int stride_bz_q_scale, int stride_h_q_scale,
                           int batch_size, cudaStream_t stream,
                           ChunkedVLaunch chunks = ChunkedVLaunch{}) {
  // Q tile plus one 128-row K tile and one 128-column V tile; the output
  // staging reuses the start of the buffer after the last iteration.
  constexpr size_t smem_max = 64 * SPARSE_HEAD_DIM + 2 * 128 * SPARSE_HEAD_DIM;
#ifdef H3_PACKED64_FINE
  // EXPERIMENT: the fine-route kernel (prefix phase, prefetched route) at 64.
  auto kernel = DENSE
      ? qk_int_sv_i8_packed_attn_kernel<64, DTypeOut, RETURN_LSE, 1, true>
      : qk_int_sv_i8_packed_fine_attn_kernel<64, DTypeOut, RETURN_LSE, 1>;
#else
  auto kernel = qk_int_sv_i8_packed_attn_kernel<64, DTypeOut, RETURN_LSE, 1, DENSE>;
#endif
  if (chunks.chunked()) {
#ifdef H3_PACKED64_FINE
    kernel = DENSE
        ? qk_int_sv_i8_packed_attn_kernel<64, DTypeOut, RETURN_LSE, 1, true, true>
        : qk_int_sv_i8_packed_fine_attn_kernel<64, DTypeOut, RETURN_LSE, 1, true>;
#else
    kernel = qk_int_sv_i8_packed_attn_kernel<64, DTypeOut, RETURN_LSE, 1, DENSE, true>;
#endif
  }
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_max));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "sage_attn_sparse failed to request " + std::to_string(smem_max) +
        " bytes of dynamic shared memory for the packed kernel: " +
        cudaGetErrorString(error));
  }
  dim3 grid(div_ceil(qo_len, 64), num_qo_heads, batch_size);
  dim3 block(32, 4);
  kernel<<<grid, block, smem_max, stream>>>(
      q, k, v, o, lse, q_scale, k_scale, v_scale, qo_len, kv_len,
      num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
      stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o, stride_bz_q_scale,
      stride_h_q_scale, sm_scale, lut, valid_block_num,
      static_cast<uint32_t>(lut_stride), chunks.blocks, chunks.shift);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("sage_attn_sparse packed kernel launch failed: ") +
        cudaGetErrorString(error));
  }
}

template <int SPARSE_CTA_Q, int SPARSE_CTA_K, typename DTypeOut, bool RETURN_LSE,
          bool PER_TILE_KV_LEN = false>
void launch_sparse_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o,
                        float *lse, float *q_scale, float *k_scale, float *v_scale,
                        const int32_t *lut, const int32_t *valid_block_num,
                        const int32_t *kv_tile_len,
                        int lut_stride, int qo_len, int kv_len,
                        int num_qo_heads, int num_kv_groups, int stride_bz_q,
                        int stride_seq_q, int stride_h_q, int stride_bz_k,
                        int stride_seq_k, int stride_h_k, int stride_bz_v,
                        int stride_h_v, int stride_d_v, int stride_bz_o,
                        int stride_seq_o, int stride_h_o, float sm_scale,
                        int stride_bz_q_scale, int stride_h_q_scale,
                        int batch_size, cudaStream_t stream,
                        ChunkedVLaunch chunks = ChunkedVLaunch{}) {
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
      ComputeUnit::kCudaCore, MaskMode::kNone, RETURN_LSE, true, false, false, true,
      PER_TILE_KV_LEN>;
  if constexpr (!PER_TILE_KV_LEN) {
    if (chunks.chunked()) {
      kernel = qk_int_sv_i8_sparse_attn_kernel<
          SPARSE_CTA_Q, SPARSE_CTA_K, WARP_Q, WARP_K, SPARSE_HEAD_DIM,
          DataType::kInt8, QuantGranularity::kPerThread,
          QuantGranularity::kPerThread, float, false, DTypeOut,
          ComputeUnit::kCudaCore, MaskMode::kNone, RETURN_LSE, true, false,
          false, true, false, true>;
    }
  }

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
      static_cast<uint32_t>(lut_stride), kv_tile_len, chunks.blocks,
      chunks.shift);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("sage_attn_sparse kernel launch failed: ") +
        cudaGetErrorString(error));
  }
}

// 32-token routes over the 64-token carrier. Every CUDA target runs it: there
// is no other 32-token kernel, and its cycle count matches packed 64.
template <typename DTypeOut, bool RETURN_LSE>
void launch_packed_32_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o,
                           float *lse, float *q_scale, float *k_scale,
                           float *v_scale, const int32_t *lut,
                           const int32_t *valid_block_num, int lut_stride,
                           int qo_len, int kv_len, int num_qo_heads,
                           int num_kv_groups, int stride_bz_q,
                           int stride_seq_q, int stride_h_q, int stride_bz_k,
                           int stride_seq_k, int stride_h_k, int stride_bz_v,
                           int stride_h_v, int stride_d_v, int stride_bz_o,
                           int stride_seq_o, int stride_h_o, float sm_scale,
                           int stride_bz_q_scale, int stride_h_q_scale,
                           int batch_size, cudaStream_t stream,
                           ChunkedVLaunch chunks = ChunkedVLaunch{},
                           const int32_t *v_changes = nullptr,
                           int v_change_stride = 0) {
  constexpr size_t smem_max = 64 * SPARSE_HEAD_DIM + 2 * 128 * SPARSE_HEAD_DIM;
  if (chunks.chunked() && v_changes == nullptr) {
    throw std::runtime_error(
        "chunk-scaled 64Q x 32KV attention needs the route's V block changes");
  }
  auto kernel =
      qk_int_sv_i8_packed_fine_attn_kernel<32, DTypeOut, RETURN_LSE, 1>;
  if (chunks.chunked()) {
    kernel = qk_int_sv_i8_packed_fine_attn_kernel<32, DTypeOut, RETURN_LSE, 1, true>;
  }
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_max));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "sage_attn_sparse failed to request " + std::to_string(smem_max) +
        " bytes of dynamic shared memory for the packed 32 kernel: " +
        cudaGetErrorString(error));
  }
  dim3 grid(div_ceil(qo_len, 64), num_qo_heads, batch_size);
  dim3 block(32, 4);
  kernel<<<grid, block, smem_max, stream>>>(
      q, k, v, o, lse, q_scale, k_scale, v_scale, qo_len, kv_len,
      num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
      stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o, stride_bz_q_scale,
      stride_h_q_scale, sm_scale, lut, valid_block_num,
      static_cast<uint32_t>(lut_stride), chunks.blocks, chunks.shift,
      v_changes, static_cast<uint32_t>(v_change_stride));
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("sage_attn_sparse packed 32 kernel launch failed: ") +
        cudaGetErrorString(error));
  }
}

} // anonymous namespace

void launch_sage_attn_packed_dense_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, int cta_k, int batch_size,
    int qo_len, int kv_len, int num_qo_heads, int num_kv_heads, int head_dim,
    int stride_bz_q, int stride_seq_q, int stride_h_q, int stride_bz_k,
    int stride_seq_k, int stride_h_k, int stride_bz_v, int stride_h_v,
    int stride_d_v, int stride_bz_o, int stride_seq_o, int stride_h_o,
    float sm_scale, int output_dtype_code, cudaStream_t stream,
    int v_scale_rows) {
  if (cta_k != 64 || head_dim != 128) {
    throw std::runtime_error("packed dense attention requires a 64KV carrier and head_dim 128");
  }
  auto q_ = const_cast<int8_t *>(static_cast<const int8_t *>(q));
  auto k_ = const_cast<int8_t *>(static_cast<const int8_t *>(k));
  auto v_ = const_cast<int8_t *>(static_cast<const int8_t *>(v));
  auto qs_ = const_cast<float *>(static_cast<const float *>(q_scale));
  auto ks_ = const_cast<float *>(static_cast<const float *>(k_scale));
  auto vs_ = const_cast<float *>(static_cast<const float *>(v_scale));
  const int qs_stride_h = div_ceil(qo_len, 128) * 32;
  const ChunkedVLaunch chunks = chunked_v_launch(v_scale_rows, kv_len);
#define LAUNCH_PACKED_DENSE(DT) \
  launch_packed_64_impl<DT, false, true>( \
      q_, k_, v_, static_cast<DT *>(o), nullptr, qs_, ks_, vs_, nullptr, \
      nullptr, 0, qo_len, kv_len, num_qo_heads, num_qo_heads / num_kv_heads, \
      stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, \
      stride_h_k, stride_bz_v, stride_h_v, stride_d_v, stride_bz_o, \
      stride_seq_o, stride_h_o, sm_scale, qs_stride_h * num_qo_heads, \
      qs_stride_h, batch_size, stream, chunks)
  if (output_dtype_code == 1) {
    LAUNCH_PACKED_DENSE(half);
  } else {
    LAUNCH_PACKED_DENSE(nv_bfloat16);
  }
#undef LAUNCH_PACKED_DENSE
}

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
    int output_dtype_code, cudaStream_t stream, int v_scale_rows,
    const void *v_block_changes, int v_change_stride) {
  if (!((cta_q == 128 || cta_q == 64) && (cta_k == 128 || cta_k == 64)) &&
      !(cta_q == 64 && cta_k == 32)) {
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
  const ChunkedVLaunch chunks = chunked_v_launch(v_scale_rows, kv_len);

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
                         valid_, nullptr, lut_stride, qo_len, kv_len, num_qo_heads,     \
                         num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, \
                         stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v,   \
                         stride_h_v, stride_d_v, stride_bz_o, stride_seq_o,    \
                         stride_h_o, sm_scale, stride_bz_q_scale,              \
                         stride_h_q_scale, batch_size, stream, chunks)

  if (cta_q == 64 && cta_k == 32) {
#define LAUNCH_PACKED_32(DT)                                                     launch_packed_32_impl<DT, RETURN_LSE>(                                             q_, k_, v_, static_cast<DT *>(o), static_cast<float *>(lse), qs_, ks_,         vs_, lut_, valid_, lut_stride, qo_len, kv_len, num_qo_heads,                   num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,             stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,                 stride_bz_o, stride_seq_o, stride_h_o, sm_scale, stride_bz_q_scale,            stride_h_q_scale, batch_size, stream, chunks, static_cast<const int32_t *>(v_block_changes), v_change_stride)
    if (output_dtype_code == 1) {
      LAUNCH_PACKED_32(half);
    } else {
      LAUNCH_PACKED_32(nv_bfloat16);
    }
#undef LAUNCH_PACKED_32
  } else if (cta_q == 128 && cta_k == 128) {
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
  } else if (packed_64_preferred()) {
#define LAUNCH_PACKED_64(DT)                                                   \
  launch_packed_64_impl<DT, RETURN_LSE>(                                       \
      q_, k_, v_, static_cast<DT *>(o), static_cast<float *>(lse), qs_, ks_,   \
      vs_, lut_, valid_, lut_stride, qo_len, kv_len, num_qo_heads,             \
      num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,       \
      stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,           \
      stride_bz_o, stride_seq_o, stride_h_o, sm_scale, stride_bz_q_scale,      \
      stride_h_q_scale, batch_size, stream, chunks)
    if (output_dtype_code == 1) {
      LAUNCH_PACKED_64(half);
    } else {
      LAUNCH_PACKED_64(nv_bfloat16);
    }
#undef LAUNCH_PACKED_64
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
    int output_dtype_code, cudaStream_t stream,
    int v_scale_rows, const void *v_block_changes,
    int v_change_stride) {
  launch_sparse<false>(
      q, k, v, o, nullptr, q_scale, k_scale, v_scale, lut, valid_block_num,
      lut_stride, cta_q, cta_k, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
      head_dim, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
      stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o, stride_bz_q_scale,
      stride_h_q_scale, sm_scale, output_dtype_code, stream, v_scale_rows,
      v_block_changes, v_change_stride);
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
    float sm_scale, int output_dtype_code, cudaStream_t stream,
    int v_scale_rows, const void *v_block_changes,
    int v_change_stride) {
  if (lse == nullptr) {
    throw std::runtime_error("sage_attn_sparse_lse: an LSE output is required");
  }
  launch_sparse<true>(
      q, k, v, o, lse, q_scale, k_scale, v_scale, lut, valid_block_num,
      lut_stride, cta_q, cta_k, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
      head_dim, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
      stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o, stride_bz_q_scale,
      stride_h_q_scale, sm_scale, output_dtype_code, stream, v_scale_rows,
      v_block_changes, v_change_stride);
}

// H3 VSA: 64Q x 64KV traversal over padded tiles whose live key count is
// kv_tile_len[tile]. Only the geometry VSA uses is built.
void launch_sage_attn_sparse_tiles_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, const void *kv_tile_len, int lut_stride,
    int batch_size, int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int head_dim, int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k, int stride_bz_v,
    int stride_h_v, int stride_d_v, int stride_bz_o, int stride_seq_o,
    int stride_h_o, int stride_bz_q_scale, int stride_h_q_scale,
    float sm_scale, int output_dtype_code, cudaStream_t stream) {
  if (head_dim != SPARSE_HEAD_DIM) {
    throw std::runtime_error(
        "sage_attn_sparse_tiles: fixed to head_dim 128, got " +
        std::to_string(head_dim));
  }
  if (lut == nullptr || valid_block_num == nullptr || kv_tile_len == nullptr) {
    throw std::runtime_error(
        "sage_attn_sparse_tiles: a route LUT and per-tile KV lengths are required");
  }
  if (kv_len % 64 != 0) {
    throw std::runtime_error(
        "sage_attn_sparse_tiles: the KV sequence must be padded to whole 64-row tiles");
  }
  int num_kv_groups = num_qo_heads / num_kv_heads;
  auto q_ = const_cast<int8_t *>(static_cast<const int8_t *>(q));
  auto k_ = const_cast<int8_t *>(static_cast<const int8_t *>(k));
  auto v_ = const_cast<int8_t *>(static_cast<const int8_t *>(v));
  auto qs_ = const_cast<float *>(static_cast<const float *>(q_scale));
  auto ks_ = const_cast<float *>(static_cast<const float *>(k_scale));
  auto vs_ = const_cast<float *>(static_cast<const float *>(v_scale));
  auto lut_ = static_cast<const int32_t *>(lut);
  auto valid_ = static_cast<const int32_t *>(valid_block_num);
  auto tiles_ = static_cast<const int32_t *>(kv_tile_len);
  if (output_dtype_code == 1) {
    launch_sparse_impl<64, 64, half, false, true>(
        q_, k_, v_, static_cast<half *>(o), nullptr, qs_, ks_, vs_, lut_,
        valid_, tiles_, lut_stride, qo_len, kv_len, num_qo_heads,
        num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
        stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o, sm_scale, stride_bz_q_scale,
        stride_h_q_scale, batch_size, stream);
  } else {
    launch_sparse_impl<64, 64, nv_bfloat16, false, true>(
        q_, k_, v_, static_cast<nv_bfloat16 *>(o), nullptr, qs_, ks_, vs_,
        lut_, valid_, tiles_, lut_stride, qo_len, kv_len, num_qo_heads,
        num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
        stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o, sm_scale, stride_bz_q_scale,
        stride_h_q_scale, batch_size, stream);
  }
}
