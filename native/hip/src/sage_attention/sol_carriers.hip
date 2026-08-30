// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Carrier adapter for the Sol-Attn exact stage. H3 supplies post-norm, post-RoPE
// Q/K/V, so this deliberately excludes Sol's router and fused QKV producer.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "sol_exact_common.h"

namespace comfy::hip_backend::sol_h3 {
namespace {

constexpr int HD = kHeadDim;

template <typename scalar_t>
__global__ __launch_bounds__(HD) void kmean_kernel(
    const scalar_t* __restrict__ k, float* __restrict__ mean, int H, int T,
    int64_t stride_h, int64_t stride_n) {
    const int h = blockIdx.x, d = threadIdx.x;
    float sum = 0.f;
    const scalar_t* head = k + static_cast<int64_t>(h) * stride_h;
    for (int t = 0; t < T; ++t) sum += static_cast<float>(head[static_cast<int64_t>(t) * stride_n + d]);
    mean[static_cast<size_t>(h) * HD + d] = sum / static_cast<float>(T);
}

template <typename scalar_t>
__global__ __launch_bounds__(kBlock) void quant_q_kernel(
    const scalar_t* __restrict__ q, int8_t* __restrict__ qi, float* __restrict__ qs,
    int H, int T, int64_t stride_h, int64_t stride_n) {
    const int h = blockIdx.y, t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= T) return;
    const scalar_t* row = q + static_cast<int64_t>(h) * stride_h + static_cast<int64_t>(t) * stride_n;
    float a = 0.f;
#pragma unroll
    for (int d = 0; d < HD; ++d) a = fmaxf(a, fabsf(static_cast<float>(row[d])));
    const float scale = fmaxf(a / 127.f, 1e-8f);
    qs[static_cast<size_t>(h) * T + t] = scale;
    int8_t* dst = qi + (static_cast<size_t>(h) * T + t) * HD;
#pragma unroll
    for (int c = 0; c < HD; c += 16) {
        __attribute__((aligned(16))) int8_t out[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) out[j] = q8(static_cast<float>(row[c + j]), 1.f / scale);
        *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(out);
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(kBlock) void quant_k_kernel(
    const scalar_t* __restrict__ k, const float* __restrict__ mean, int8_t* __restrict__ ki,
    float2* __restrict__ ksb, int H, int T, int Tp, int64_t stride_h, int64_t stride_n) {
    const int h = blockIdx.y, t = blockIdx.x * kBlock + threadIdx.x;
    if (t >= Tp) return;
    int8_t* dst = ki + (static_cast<size_t>(h) * Tp + t) * HD;
    if (t >= T) {
#pragma unroll
        for (int c = 0; c < HD; c += 16) *reinterpret_cast<uint4*>(dst + c) = make_uint4(0, 0, 0, 0);
        ksb[static_cast<size_t>(h) * Tp + t] = make_float2(0.f, kNeg);
        return;
    }
    const scalar_t* row = k + static_cast<int64_t>(h) * stride_h + static_cast<int64_t>(t) * stride_n;
    const float* center = mean + static_cast<size_t>(h) * HD;
    float a = 0.f;
#pragma unroll
    for (int d = 0; d < HD; ++d) a = fmaxf(a, fabsf(static_cast<float>(row[d]) - center[d]));
    const float scale = fmaxf(a / 127.f, 1e-8f);
    ksb[static_cast<size_t>(h) * Tp + t] = make_float2(scale, 0.f);
#pragma unroll
    for (int c = 0; c < HD; c += 16) {
        __attribute__((aligned(16))) int8_t out[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) out[j] = q8(static_cast<float>(row[c + j]) - center[c + j], 1.f / scale);
        *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(out);
    }
}

template <typename scalar_t>
__global__ __launch_bounds__(HD) void vscale_kernel(
    const scalar_t* __restrict__ v, float* __restrict__ vsc, int H, int T,
    int64_t stride_h, int64_t stride_n) {
    const int h = blockIdx.x, d = threadIdx.x;
    const scalar_t* head = v + static_cast<int64_t>(h) * stride_h;
    float a = 0.f;
    for (int t = 0; t < T; ++t) a = fmaxf(a, fabsf(static_cast<float>(head[static_cast<int64_t>(t) * stride_n + d])));
    vsc[static_cast<size_t>(h) * HD + d] = fmaxf(a / 127.f, 1e-8f);
}

template <typename scalar_t, int TT>
__global__ __launch_bounds__(256) void vtranspose_kernel(
    const scalar_t* __restrict__ v, const float* __restrict__ vsc, int8_t* __restrict__ vt,
    int T, int Tp, int H, int64_t stride_h, int64_t stride_n) {
    constexpr int LD = HD + 16;
    __shared__ __attribute__((aligned(16))) int8_t tile[TT * LD];
    const int h = blockIdx.y, t0 = blockIdx.x * TT;
    for (int idx = threadIdx.x; idx < TT * (HD / 16); idx += blockDim.x) {
        const int t = idx / (HD / 16), c = (idx % (HD / 16)) * 16;
        __attribute__((aligned(16))) int8_t out[16];
        if (t0 + t < T) {
            const scalar_t* src = v + static_cast<int64_t>(h) * stride_h + static_cast<int64_t>(t0 + t) * stride_n + c;
#pragma unroll
            for (int j = 0; j < 16; ++j) out[j] = q8(static_cast<float>(src[j]), 1.f / vsc[static_cast<size_t>(h) * HD + c + j]);
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) out[j] = 0;
        }
        *reinterpret_cast<uint4*>(tile + t * LD + c) = *reinterpret_cast<const uint4*>(out);
    }
    __syncthreads();
    int8_t* dst = vt + static_cast<int64_t>(h) * HD * Tp + t0;
    for (int idx = threadIdx.x; idx < (HD / 4) * (TT / 4); idx += blockDim.x) {
        const int cg = idx / (TT / 4), tg = (idx % (TT / 4)) * 4;
        if (t0 + tg >= Tp) continue;
        uint32_t r[4], w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) r[j] = *reinterpret_cast<const uint32_t*>(tile + (tg + j) * LD + cg * 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = ((r[0] >> (i * 8)) & 0xffu) | (((r[1] >> (i * 8)) & 0xffu) << 8) |
                   (((r[2] >> (i * 8)) & 0xffu) << 16) | (((r[3] >> (i * 8)) & 0xffu) << 24);
            *reinterpret_cast<uint32_t*>(dst + static_cast<int64_t>(cg * 4 + i) * Tp + tg) = w[i];
        }
    }
}

template <typename scalar_t>
void launch_qk(const void* q, void* qi, void* qs, const void* k, void* ki, void* ksb,
               void* mean, int H, int Tq, int Tk, int Tp, int64_t qh, int64_t qt,
               int64_t kh, int64_t kt, hipStream_t stream) {
    kmean_kernel<scalar_t><<<H, HD, 0, stream>>>(static_cast<const scalar_t*>(k), static_cast<float*>(mean), H, Tk, kh, kt);
    quant_q_kernel<scalar_t><<<dim3((Tq + kBlock - 1) / kBlock, H), kBlock, 0, stream>>>(static_cast<const scalar_t*>(q), static_cast<int8_t*>(qi), static_cast<float*>(qs), H, Tq, qh, qt);
    quant_k_kernel<scalar_t><<<dim3((Tp + kBlock - 1) / kBlock, H), kBlock, 0, stream>>>(static_cast<const scalar_t*>(k), static_cast<const float*>(mean), static_cast<int8_t*>(ki), static_cast<float2*>(ksb), H, Tk, Tp, kh, kt);
}

template <typename scalar_t>
void launch_v(const void* v, void* vt, void* vsc, int H, int T, int Tp,
              int64_t vh, int64_t vt_stride, hipStream_t stream) {
    vscale_kernel<scalar_t><<<H, HD, 0, stream>>>(static_cast<const scalar_t*>(v), static_cast<float*>(vsc), H, T, vh, vt_stride);
    vtranspose_kernel<scalar_t, 256><<<dim3((Tp + 255) / 256, H), 256, 0, stream>>>(static_cast<const scalar_t*>(v), static_cast<const float*>(vsc), static_cast<int8_t*>(vt), T, Tp, H, vh, vt_stride);
}

}  // namespace
}  // namespace comfy::hip_backend::sol_h3

using namespace comfy::hip_backend::sol_h3;

extern "C" void launch_sol_h3_qk_quant(
    const void* q, void* qi, void* qs, const void* k, void* ki, void* ksb, void* mean,
    int H, int Tq, int Tk, int Tp, int64_t qh, int64_t qt, int64_t kh, int64_t kt,
    int dtype, hipStream_t stream) {
    if (dtype == 1) launch_qk<__half>(q, qi, qs, k, ki, ksb, mean, H, Tq, Tk, Tp, qh, qt, kh, kt, stream);
    else if (dtype == 2) launch_qk<__bf16>(q, qi, qs, k, ki, ksb, mean, H, Tq, Tk, Tp, qh, qt, kh, kt, stream);
    else throw std::runtime_error("Sol exact carriers require float16 or bfloat16 input");
}

extern "C" void launch_sol_h3_v_quant(
    const void* v, void* vt, void* vsc, int H, int T, int Tp, int64_t vh, int64_t vn,
    int dtype, hipStream_t stream) {
    if (dtype == 1) launch_v<__half>(v, vt, vsc, H, T, Tp, vh, vn, stream);
    else if (dtype == 2) launch_v<__bf16>(v, vt, vsc, H, T, Tp, vh, vn, stream);
    else throw std::runtime_error("Sol exact carriers require float16 or bfloat16 input");
}
