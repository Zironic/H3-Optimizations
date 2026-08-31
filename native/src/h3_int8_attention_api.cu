// SPDX-License-Identifier: Apache-2.0
//
// The stable C surface H3-Optimizations loads through ctypes.
//
// The vendored Kitchen launchers are extern "C" but throw std::runtime_error
// on bad configuration or a CUDA fault. MSVC already warns about this
// (C4297: "function assumed not to throw an exception but does"), and under
// /EHc it is entitled to elide the unwind machinery entirely -- so letting an
// exception cross into CPython through ctypes is undefined behaviour, not
// merely untidy. It only appears to work today because nanobind supplies a
// C++ frame that catches.
//
// Every entry point here is noexcept, returns 0 on success or non-zero on
// failure, and leaves a message for h3_int8_last_error(). Python turns that
// into an ordinary exception.
//
// This is also the ABI boundary. Keeping it plain C -- pointers, ints, a
// stream handle -- means no nanobind, no DLPack, and no Python ABI axis, so a
// single binary serves every Python that can call ctypes.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

// Export only this API. The vendored launchers stay internal to the library,
// so the ABI surface is exactly what the loader binds and nothing else.
#if defined(_WIN32)
#define H3_API __declspec(dllexport)
#else
#define H3_API __attribute__((visibility("default")))
#endif

// Vendored Kitchen launchers. See native/src/PROVENANCE for the commit.
void launch_sage_attn_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *mask,
    int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
    int64_t mask_stride_k, int mask_dtype_code, int cta_k, int B, int Lq,
    int Lk, int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, float sm_scale, int output_dtype_code,
    cudaStream_t stream);

void launch_sage_attn_sparse_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, int lut_stride, int cta_q, int cta_k, int B,
    int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, int qs_st_bz, int qs_st_h,
    float sm_scale, int output_dtype_code, cudaStream_t stream);

void launch_sage_attn_sparse_kernel_lse(
    const void *q, const void *k, const void *v, void *o, void *lse,
    const void *q_scale, const void *k_scale, const void *v_scale,
    const void *lut, const void *valid_block_num, int lut_stride, int cta_q,
    int cta_k, int B, int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz,
    int q_st_n, int q_st_h, int k_st_bz, int k_st_n, int k_st_h,
    int v_st_bz, int v_st_h, int v_st_d, int o_st_bz, int o_st_n,
    int o_st_h, int qs_st_bz, int qs_st_h, float sm_scale,
    int output_dtype_code, cudaStream_t stream);

const char *sage_attn_sparse_route_encoding();

void launch_quant_qk_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ,
    int WARPQ, int BLKK, int WARPK, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int input_dtype_code, void *anchor_indices,
    cudaStream_t stream);

void launch_select_k_anchor_from_samples(
    const void *samples, const int *sample_positions, void *anchor_values,
    void *anchor_indices, int B, int H_kv, int full_Lk, int C,
    int64_t stride_b, int64_t stride_h, int64_t stride_n, int input_dtype_code,
    cudaStream_t stream);

void launch_quant_qk_per_thread_int8_into(
    const void *q, const void *k, void *q_int8, void *q_scale, void *k_int8,
    void *k_scale, const void *anchor_values, const void *anchor_indices,
    int B, int H_q, int Lq, int full_Lq, int q_start, int H_kv, int Lk,
    int full_Lk, int k_start, int C, int cta_k, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_n, int64_t k_stride_b,
    int64_t k_stride_h, int64_t k_stride_n, int input_dtype_code,
    cudaStream_t stream);

void launch_quant_q_per_thread_int8_into(
    const void *q, void *q_int8, void *q_scale, int B, int H_q, int Lq,
    int full_Lq, int q_start, int C, int full_Lk, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_n, int input_dtype_code,
    cudaStream_t stream);

void launch_h3_quantize_bf16_rowwise_convrot256(
    const void *input, void *output, void *scales, int64_t rows,
    int64_t columns, cudaStream_t stream);

bool launch_h3_fused_q_cutlass(
    const void *a, const void *b, const void *x_scale,
    const void *weight_scale, const void *norm, const void *freqs, void *debug,
    void *summary, void *q, void *q_scale, int64_t m, int64_t n, int64_t k,
    int full_k_length, float epsilon, cudaStream_t stream);

void launch_quant_v_int8_kernel(const void *v, void *out, void *scale, int B,
                                int H, int N, int D, int padded_N, int64_t sb,
                                int64_t sh, int64_t sn, int input_dtype_code,
                                cudaStream_t stream);

void launch_v_amax_chunk(const void *v, void *amax, int B, int H, int rows,
                         int D, int64_t sb, int64_t sh, int64_t sn,
                         int input_dtype_code, cudaStream_t stream);

void launch_quant_v_chunk_into(const void *v, void *out, const void *scale,
                               int B, int H, int rows, int row_start, int D,
                               int padded_N, int64_t sb, int64_t sh,
                               int64_t sn, int input_dtype_code,
                               cudaStream_t stream);

namespace {

// One slot per thread: a failed call on one stream must not overwrite the
// message another thread is about to read.
thread_local std::string g_last_error;

void set_error(const char *what) {
  g_last_error.assign(what ? what : "unknown native error");
}

} // anonymous namespace

// Wrap a launcher so no exception reaches the caller. Returns 0 on success,
// 1 for a C++ exception, 2 for anything else.
#define H3_GUARD(BODY)                                                         \
  try {                                                                        \
    BODY;                                                                      \
    g_last_error.clear();                                                      \
    return 0;                                                                  \
  } catch (const std::exception &error) {                                      \
    set_error(error.what());                                                   \
    return 1;                                                                  \
  } catch (...) {                                                              \
    set_error("unknown native error");                                         \
    return 2;                                                                  \
  }

extern "C" {

H3_API int h3_int8_abi_version() noexcept { return 4; }

H3_API const char *h3_int8_last_error() noexcept {
  return g_last_error.empty() ? "" : g_last_error.c_str();
}

H3_API const char *h3_int8_route_encoding() noexcept {
  try {
    return sage_attn_sparse_route_encoding();
  } catch (...) {
    return "unknown";
  }
}

H3_API int h3_int8_device_capability(int device, int *major, int *minor) noexcept {
  H3_GUARD({
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
    if (major) *major = properties.major;
    if (minor) *minor = properties.minor;
  })
}

H3_API int h3_int8_dense_attention(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, int cta_k, int B, int Lq, int Lk,
    int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, float sm_scale, int output_dtype_code,
    uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_kernel(
      q, k, v, o, q_scale, k_scale, v_scale, nullptr, 0, 0, 0, 0, -1, cta_k, B,
      Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n, q_st_h, k_st_bz, k_st_n, k_st_h,
      v_st_bz, v_st_h, v_st_d, o_st_bz, o_st_n, o_st_h, sm_scale,
      output_dtype_code, reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_sparse_attention(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, int lut_stride, int cta_q, int cta_k, int B,
    int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, int qs_st_bz, int qs_st_h,
    float sm_scale, int output_dtype_code, uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_sparse_kernel(
      q, k, v, o, q_scale, k_scale, v_scale, lut, valid_block_num, lut_stride,
      cta_q, cta_k, B, Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n, q_st_h,
      k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h, v_st_d, o_st_bz, o_st_n,
      o_st_h, qs_st_bz, qs_st_h, sm_scale, output_dtype_code,
      reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_sparse_attention_lse(
    const void *q, const void *k, const void *v, void *o, void *lse,
    const void *q_scale, const void *k_scale, const void *v_scale,
    const void *lut, const void *valid_block_num, int lut_stride, int cta_q,
    int cta_k, int B, int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz,
    int q_st_n, int q_st_h, int k_st_bz, int k_st_n, int k_st_h,
    int v_st_bz, int v_st_h, int v_st_d, int o_st_bz, int o_st_n,
    int o_st_h, int qs_st_bz, int qs_st_h, float sm_scale,
    int output_dtype_code, uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_sparse_kernel_lse(
      q, k, v, o, lse, q_scale, k_scale, v_scale, lut, valid_block_num,
      lut_stride, cta_q, cta_k, B, Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n,
      q_st_h, k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h, v_st_d, o_st_bz,
      o_st_n, o_st_h, qs_st_bz, qs_st_h, sm_scale, output_dtype_code,
      reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_quantize_qk(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ,
    int WARPQ, int BLKK, int WARPK, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int input_dtype_code, void *anchor_indices,
    uintptr_t stream) noexcept {
  H3_GUARD(launch_quant_qk_per_thread_int8(
      q, q_int8, q_scale, k, k_int8, k_scale, B, H_q, Lq, H_kv, Lk, C, BLKQ,
      WARPQ, BLKK, WARPK, q_stride_b, q_stride_h, q_stride_n, k_stride_b,
      k_stride_h, k_stride_n, input_dtype_code, anchor_indices,
      reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_select_k_anchor(
    const void *samples, const int *sample_positions, void *anchor_values,
    void *anchor_indices, int B, int H_kv, int full_Lk, int C,
    int64_t stride_b, int64_t stride_h, int64_t stride_n, int input_dtype_code,
    uintptr_t stream) noexcept {
  H3_GUARD(launch_select_k_anchor_from_samples(
      samples, sample_positions, anchor_values, anchor_indices, B, H_kv,
      full_Lk, C, stride_b, stride_h, stride_n, input_dtype_code,
      reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_quantize_qk_chunk(
    const void *q, const void *k, void *q_int8, void *q_scale, void *k_int8,
    void *k_scale, const void *anchor_values, const void *anchor_indices,
    int B, int H_q, int Lq, int full_Lq, int q_start, int H_kv, int Lk,
    int full_Lk, int k_start, int C, int cta_k, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_n, int64_t k_stride_b,
    int64_t k_stride_h, int64_t k_stride_n, int input_dtype_code,
    uintptr_t stream) noexcept {
  H3_GUARD(launch_quant_qk_per_thread_int8_into(
      q, k, q_int8, q_scale, k_int8, k_scale, anchor_values, anchor_indices, B,
      H_q, Lq, full_Lq, q_start, H_kv, Lk, full_Lk, k_start, C, cta_k,
      q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h, k_stride_n,
      input_dtype_code, reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_quantize_q_chunk(
    const void *q, void *q_int8, void *q_scale, int B, int H_q, int Lq,
    int full_Lq, int q_start, int C, int full_Lk, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_n, int input_dtype_code,
    uintptr_t stream) noexcept {
  H3_GUARD(launch_quant_q_per_thread_int8_into(
      q, q_int8, q_scale, B, H_q, Lq, full_Lq, q_start, C, full_Lk,
      q_stride_b, q_stride_h, q_stride_n, input_dtype_code,
      reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_quantize_bf16_rowwise_convrot256(
    const void *input, void *output, void *scales, int64_t rows,
    int64_t columns, uintptr_t stream) noexcept {
  H3_GUARD(launch_h3_quantize_bf16_rowwise_convrot256(
      input, output, scales, rows, columns,
      reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_fused_q(
    const void *activation, const void *weight, const void *activation_scale,
    const void *weight_scale, const void *norm, const void *freqs,
    void *summary, void *q, void *q_scale, int64_t rows, int64_t outputs,
    int64_t hidden, int full_k_length, float epsilon,
    uintptr_t stream) noexcept {
  H3_GUARD({
    if (!activation || !weight || !activation_scale || !weight_scale ||
        !norm || !freqs || !summary || !q || !q_scale) {
      throw std::runtime_error("fused H3 Q received a null pointer");
    }
    if (rows <= 0 || outputs <= 0 || hidden <= 0 ||
        rows > std::numeric_limits<int>::max() - 127 ||
        outputs > std::numeric_limits<int>::max() ||
        hidden > std::numeric_limits<int>::max() || full_k_length <= 0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0f) {
      throw std::runtime_error("fused H3 Q received invalid geometry");
    }
    if (!launch_h3_fused_q_cutlass(
            activation, weight, activation_scale, weight_scale, norm, freqs,
            nullptr, summary, q, q_scale, rows, outputs, hidden,
            full_k_length, epsilon,
            reinterpret_cast<cudaStream_t>(stream))) {
      throw std::runtime_error(
          "exact 128x256 CUTLASS H3 Q kernel rejected the request");
    }
  })
}

H3_API int h3_int8_quantize_v(const void *v, void *out, void *scale, int B, int H,
                       int N, int D, int padded_N, int64_t sb, int64_t sh,
                       int64_t sn, int input_dtype_code,
                       uintptr_t stream) noexcept {
  H3_GUARD(launch_quant_v_int8_kernel(v, out, scale, B, H, N, D, padded_N, sb,
                                      sh, sn, input_dtype_code,
                                      reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_v_amax_chunk(const void *v, void *amax, int B, int H,
                                int rows, int D, int64_t sb, int64_t sh,
                                int64_t sn, int input_dtype_code,
                                uintptr_t stream) noexcept {
  H3_GUARD(launch_v_amax_chunk(v, amax, B, H, rows, D, sb, sh, sn,
                               input_dtype_code,
                               reinterpret_cast<cudaStream_t>(stream)))
}

H3_API int h3_int8_quantize_v_chunk_into(
    const void *v, void *out, const void *scale, int B, int H, int rows,
    int row_start, int D, int padded_N, int64_t sb, int64_t sh, int64_t sn,
    int input_dtype_code, uintptr_t stream) noexcept {
  H3_GUARD(launch_quant_v_chunk_into(
      v, out, scale, B, H, rows, row_start, D, padded_N, sb, sh, sn,
      input_dtype_code, reinterpret_cast<cudaStream_t>(stream)))
}

} // extern "C"
