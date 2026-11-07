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
    cudaStream_t stream,
    int v_scale_rows);

void launch_sage_attn_sparse_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, int lut_stride, int cta_q, int cta_k, int B,
    int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, int qs_st_bz, int qs_st_h,
    float sm_scale, int output_dtype_code, cudaStream_t stream,
    int v_scale_rows, const void *v_block_changes, int v_change_stride);

void launch_sage_attn_packed_dense_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, int cta_k, int B, int Lq,
    int Lk, int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, float sm_scale, int output_dtype_code,
    cudaStream_t stream,
    int v_scale_rows);

void launch_sage_attn_sparse_tiles_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, const void *kv_tile_len, int lut_stride,
    int batch_size, int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int head_dim, int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k, int stride_bz_v,
    int stride_h_v, int stride_d_v, int stride_bz_o, int stride_seq_o,
    int stride_h_o, int stride_bz_q_scale, int stride_h_q_scale,
    float sm_scale, int output_dtype_code, cudaStream_t stream);

void launch_sage_attn_sparse_kernel_lse(
    const void *q, const void *k, const void *v, void *o, void *lse,
    const void *q_scale, const void *k_scale, const void *v_scale,
    const void *lut, const void *valid_block_num, int lut_stride, int cta_q,
    int cta_k, int B, int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz,
    int q_st_n, int q_st_h, int k_st_bz, int k_st_n, int k_st_h,
    int v_st_bz, int v_st_h, int v_st_d, int o_st_bz, int o_st_n,
    int o_st_h, int qs_st_bz, int qs_st_h, float sm_scale,
    int output_dtype_code, cudaStream_t stream,
    int v_scale_rows, const void *v_block_changes, int v_change_stride);

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

void launch_h3_quantize_bf16_rms_adaln_rowwise_convrot256(
    const void *input, const void *norm_weight, const void *shift,
    const void *modulation_scale, void *output, void *scales, int64_t rows,
    int64_t columns, float epsilon, int modulation_dtype_code, int mode,
    cudaStream_t stream);

bool launch_h3_fused_q_cutlass(
    const void *a, const void *b, const void *x_scale,
    const void *weight_scale, const void *norm, const void *freqs, void *debug,
    void *summary, void *q, void *q_scale, int64_t m, int64_t n, int64_t k,
    int full_k_length, float epsilon, cudaStream_t stream);

void launch_h3_threshold_route(const void *q, const void *k_video, const void *sigma,
                               const void *kbar, void *lut, void *valid, void *masks, int bh,
                               int tile_count,
                               int64_t q_bh_stride, int64_t q_row_stride, int64_t k_bh_stride,
                               int video_tiles, int pvs, int kv_tiles, int dense_rows, int width,
                               float tau, cudaStream_t stream);

void launch_h3_align_route_v_blocks(const void *indices, const void *counts,
                                    void *out_indices, void *out_counts,
                                    void *out_changes, int rows, int width_in,
                                    int width_out, int change_width,
                                    int tiles_per_block, int group,
                                    int total_tiles, int in_delta,
                                    int out_delta, cudaStream_t stream);

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

void launch_h3_sparse_sage_sa2pp_sm89(
    const void *q, const void *k, const void *v, void *output,
    const void *lut, const void *valid, const void *pv_threshold,
    const void *q_scale, const void *k_scale, const void *v_scale,
    int batch, int q_length, int kv_length, int q_heads, int kv_heads,
    int head_dim, int q_stride_b, int q_stride_n, int q_stride_h,
    int k_stride_b, int k_stride_n, int k_stride_h,
    int v_stride_b, int v_stride_h, int v_stride_d,
    int o_stride_b, int o_stride_n, int o_stride_h,
    float scale, int output_dtype_code, cudaStream_t stream);

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
      output_dtype_code, reinterpret_cast<cudaStream_t>(stream), 0))
}

H3_API int h3_int8_packed_dense_attention(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, int cta_k, int B, int Lq, int Lk,
    int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, float sm_scale,
    int output_dtype_code, uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_packed_dense_kernel(
      q, k, v, o, q_scale, k_scale, v_scale, cta_k, B, Lq, Lk, H_q, H_kv, D,
      q_st_bz, q_st_n, q_st_h, k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h,
      v_st_d, o_st_bz, o_st_n, o_st_h, sm_scale, output_dtype_code,
      reinterpret_cast<cudaStream_t>(stream), 0))
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
      reinterpret_cast<cudaStream_t>(stream), 0, nullptr, 0))
}

// Additive to ABI 4: 64Q x 64KV sparse traversal over padded tiles with a
// per-tile live key count (H3 VSA). Loaders treat it as optional.
H3_API int h3_int8_sparse_attention_tiles(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, const void *kv_tile_len, int lut_stride,
    int B, int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz, int q_st_n,
    int q_st_h, int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h,
    int v_st_d, int o_st_bz, int o_st_n, int o_st_h, int qs_st_bz,
    int qs_st_h, float sm_scale, int output_dtype_code,
    uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_sparse_tiles_kernel(
      q, k, v, o, q_scale, k_scale, v_scale, lut, valid_block_num,
      kv_tile_len, lut_stride, B, Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n,
      q_st_h, k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h, v_st_d, o_st_bz,
      o_st_n, o_st_h, qs_st_bz, qs_st_h, sm_scale, output_dtype_code,
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
      reinterpret_cast<cudaStream_t>(stream), 0, nullptr, 0))
}

// Additive to ABI 4: chunk-scaled INT8 V. Identical to the entry point of the
// same stem, plus v_scale_rows: V scales are [B, H_kv, ceil(Lk / rows), D],
// and, for the sparse ones, the aligned route's V block change list
// (h3_int8_align_route_v_blocks) with its row stride,
// one per block of v_scale_rows keys (a power of two, at least 128), instead
// of one per channel over the whole sequence. Zero keeps the global carrier.
// Loaders treat these as optional.
H3_API int h3_int8_dense_attention_chunked_v(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, int cta_k, int B, int Lq, int Lk,
    int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, float sm_scale, int output_dtype_code,
    int v_scale_rows, uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_kernel(
      q, k, v, o, q_scale, k_scale, v_scale, nullptr, 0, 0, 0, 0, -1, cta_k, B,
      Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n, q_st_h, k_st_bz, k_st_n, k_st_h,
      v_st_bz, v_st_h, v_st_d, o_st_bz, o_st_n, o_st_h, sm_scale,
      output_dtype_code, reinterpret_cast<cudaStream_t>(stream), v_scale_rows))
}

H3_API int h3_int8_packed_dense_attention_chunked_v(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, int cta_k, int B, int Lq, int Lk,
    int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, float sm_scale,
    int output_dtype_code, int v_scale_rows, uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_packed_dense_kernel(
      q, k, v, o, q_scale, k_scale, v_scale, cta_k, B, Lq, Lk, H_q, H_kv, D,
      q_st_bz, q_st_n, q_st_h, k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h,
      v_st_d, o_st_bz, o_st_n, o_st_h, sm_scale, output_dtype_code,
      reinterpret_cast<cudaStream_t>(stream), v_scale_rows))
}

H3_API int h3_int8_sparse_attention_chunked_v(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *lut,
    const void *valid_block_num, int lut_stride, int cta_q, int cta_k, int B,
    int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h, int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h, int qs_st_bz, int qs_st_h,
    float sm_scale, int output_dtype_code, int v_scale_rows, const void *v_block_changes,
    int v_change_stride, uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_sparse_kernel(
      q, k, v, o, q_scale, k_scale, v_scale, lut, valid_block_num, lut_stride,
      cta_q, cta_k, B, Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n, q_st_h,
      k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h, v_st_d, o_st_bz, o_st_n,
      o_st_h, qs_st_bz, qs_st_h, sm_scale, output_dtype_code,
      reinterpret_cast<cudaStream_t>(stream), v_scale_rows,
      v_block_changes, v_change_stride))
}

H3_API int h3_int8_sparse_attention_lse_chunked_v(
    const void *q, const void *k, const void *v, void *o, void *lse,
    const void *q_scale, const void *k_scale, const void *v_scale,
    const void *lut, const void *valid_block_num, int lut_stride, int cta_q,
    int cta_k, int B, int Lq, int Lk, int H_q, int H_kv, int D, int q_st_bz,
    int q_st_n, int q_st_h, int k_st_bz, int k_st_n, int k_st_h,
    int v_st_bz, int v_st_h, int v_st_d, int o_st_bz, int o_st_n,
    int o_st_h, int qs_st_bz, int qs_st_h, float sm_scale,
    int output_dtype_code, int v_scale_rows, const void *v_block_changes,
    int v_change_stride, uintptr_t stream) noexcept {
  H3_GUARD(launch_sage_attn_sparse_kernel_lse(
      q, k, v, o, lse, q_scale, k_scale, v_scale, lut, valid_block_num,
      lut_stride, cta_q, cta_k, B, Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n,
      q_st_h, k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h, v_st_d, o_st_bz,
      o_st_n, o_st_h, qs_st_bz, qs_st_h, sm_scale, output_dtype_code,
      reinterpret_cast<cudaStream_t>(stream), v_scale_rows,
      v_block_changes, v_change_stride))
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

H3_API int h3_sparse_sage_sa2pp_sm89(
    const void *q, const void *k, const void *v, void *output,
    const void *lut, const void *valid, const void *pv_threshold,
    const void *q_scale, const void *k_scale, const void *v_scale,
    int B, int Lq, int Lk, int H_q, int H_kv, int D,
    int q_st_bz, int q_st_n, int q_st_h,
    int k_st_bz, int k_st_n, int k_st_h,
    int v_st_bz, int v_st_h, int v_st_d,
    int o_st_bz, int o_st_n, int o_st_h,
    float sm_scale, int output_dtype_code, uintptr_t stream) noexcept {
  H3_GUARD(launch_h3_sparse_sage_sa2pp_sm89(
      q, k, v, output, lut, valid, pv_threshold, q_scale, k_scale, v_scale,
      B, Lq, Lk, H_q, H_kv, D, q_st_bz, q_st_n, q_st_h,
      k_st_bz, k_st_n, k_st_h, v_st_bz, v_st_h, v_st_d,
      o_st_bz, o_st_n, o_st_h, sm_scale, output_dtype_code,
      reinterpret_cast<cudaStream_t>(stream)))
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

// Additive ABI-4 experiment. mode 0 preserves the BF16 storage boundaries
// around RMSNorm and AdaLN; mode 1 keeps those intermediates in FP32.
H3_API int h3_int8_quantize_bf16_rms_adaln_rowwise_convrot256(
    const void *input, const void *norm_weight, const void *shift,
    const void *modulation_scale, void *output, void *scales, int64_t rows,
    int64_t columns, float epsilon, int modulation_dtype_code, int mode,
    uintptr_t stream) noexcept {
  H3_GUARD(launch_h3_quantize_bf16_rms_adaln_rowwise_convrot256(
      input, norm_weight, shift, modulation_scale, output, scales, rows,
      columns, epsilon, modulation_dtype_code, mode,
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

// Additive to ABI 4: threshold routing for the streamed sparse path (see
// h3_threshold_route.cu). Loaders treat it as optional.
H3_API int h3_int8_threshold_route(
    const void *q, const void *k_video, const void *sigma, const void *kbar, void *lut,
    void *valid, void *masks, int bh, int tile_count, int64_t q_bh_stride, int64_t q_row_stride,
    int64_t k_bh_stride, int video_tiles, int pvs, int kv_tiles, int dense_rows, int width,
    float tau, uintptr_t stream) noexcept {
  H3_GUARD({
    if (!q || !k_video || !sigma || !kbar || !lut || !valid || !masks) {
      throw std::runtime_error("threshold route received a null pointer");
    }
    launch_h3_threshold_route(q, k_video, sigma, kbar, lut, valid, masks, bh, tile_count,
                              q_bh_stride,
                              q_row_stride, k_bh_stride, video_tiles, pvs, kv_tiles, dense_rows,
                              width, tau, reinterpret_cast<cudaStream_t>(stream));
  })
}

// Additive to ABI 4: pad sparse routes so no packed iteration mixes
// chunk-scaled V blocks (see h3_route_align.cu). Optional for loaders.
H3_API int h3_int8_align_route_v_blocks(
    const void *indices, const void *counts, void *out_indices,
    void *out_counts, void *out_changes, int rows, int width_in,
    int width_out, int change_width, int tiles_per_block, int group,
    int total_tiles, int in_delta, int out_delta, uintptr_t stream) noexcept {
  H3_GUARD(launch_h3_align_route_v_blocks(
      indices, counts, out_indices, out_counts, out_changes, rows, width_in,
      width_out, change_width, tiles_per_block, group, total_tiles, in_delta,
      out_delta, reinterpret_cast<cudaStream_t>(stream)))
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
