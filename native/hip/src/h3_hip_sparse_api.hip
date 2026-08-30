// SPDX-FileCopyrightText: Copyright (c) 2026 Zironic
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>

#include <cstdint>
#include <exception>
#include <string>

extern "C" void launch_sol_h3_qk_quant(
    const void* q, void* q_int8, void* q_scale,
    const void* k, void* k_int8, void* k_scale_bias, void* k_mean,
    int num_heads, int q_len, int kv_len, int padded_k,
    int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_h, int64_t k_stride_n,
    int input_dtype_code, hipStream_t stream);

extern "C" void launch_sol_h3_v_quant(
    const void* v, void* output, void* descale, int num_heads,
    int n_ctx, int padded_k, int64_t stride_h, int64_t stride_n,
    int input_dtype_code, hipStream_t stream);

extern "C" void launch_sol_h3_exact(
    const void* q, const void* k, const void* v, void* output,
    const void* q_descale, const void* k_scale_bias, const void* v_descale,
    const int* route, const int* route_counts, int route_width,
    int num_heads, int q_len, int kv_len, int padded_k,
    int64_t output_stride_h,
    int64_t output_stride_n, float sm_scale, int output_dtype_code,
    hipStream_t stream);

#if defined(_WIN32)
#define H3_HIP_API __declspec(dllexport)
#else
#define H3_HIP_API __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string last_error;

template <typename Callback>
int guarded(const char* operation, Callback&& callback) noexcept {
    try {
        callback();
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) {
            last_error = std::string(operation) + ": " + hipGetErrorString(error);
            return 2;
        }
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = std::string(operation) + ": " + error.what();
        return 1;
    } catch (...) {
        last_error = std::string(operation) + ": unknown native exception";
        return 1;
    }
}
}

extern "C" H3_HIP_API int h3_hip_sparse_abi_version() noexcept {
    return 2;
}

extern "C" H3_HIP_API const char* h3_hip_sparse_last_error() noexcept {
    return last_error.c_str();
}

extern "C" H3_HIP_API const char* h3_hip_sparse_route_encoding() noexcept {
    return "absolute";
}

extern "C" H3_HIP_API int h3_hip_sparse_quantize_qk(
    const void* q, void* q_int8, void* q_scale,
    const void* k, void* k_int8, void* k_scale_bias, void* k_mean,
    int num_heads, int q_len, int kv_len, int padded_q, int padded_k,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    int input_dtype_code, uintptr_t stream) noexcept {
    return guarded("Q/K quantization", [&] {
        (void)q_stride_b;
        (void)k_stride_b;
        (void)padded_q;
        launch_sol_h3_qk_quant(
            q, q_int8, q_scale, k, k_int8, k_scale_bias, k_mean,
            num_heads, q_len, kv_len, padded_k,
            q_stride_h, q_stride_n,
            k_stride_h, k_stride_n,
            input_dtype_code, reinterpret_cast<hipStream_t>(stream));
    });
}

extern "C" H3_HIP_API int h3_hip_sparse_quantize_v(
    const void* v, void* v_int8, void* v_scale,
    int num_heads, int kv_len, int tiles,
    int64_t v_stride_h, int64_t v_stride_n,
    int input_dtype_code, uintptr_t stream) noexcept {
    return guarded("V quantization", [&] {
        launch_sol_h3_v_quant(
            v, v_int8, v_scale, num_heads, kv_len, tiles * 64,
            v_stride_h, v_stride_n, input_dtype_code,
            reinterpret_cast<hipStream_t>(stream));
    });
}

extern "C" H3_HIP_API int h3_hip_sparse_attention(
    const void* q, const void* k, const void* v, void* output,
    const void* q_scale, const void* k_scale, const void* v_scale,
    const int* route, const int* route_counts, int route_width,
    int num_heads, int q_len, int kv_len, int padded_q, int padded_k,
    int64_t output_stride_h, int64_t output_stride_n,
    float sm_scale, int output_dtype_code, uintptr_t stream) noexcept {
    return guarded("sparse attention", [&] {
        (void)padded_q;
        launch_sol_h3_exact(
            q, k, v, output, q_scale, k_scale, v_scale,
            route, route_counts, route_width,
            num_heads, q_len, kv_len, padded_k,
            output_stride_h, output_stride_n, sm_scale, output_dtype_code,
            reinterpret_cast<hipStream_t>(stream));
    });
}
