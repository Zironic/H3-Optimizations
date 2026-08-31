// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn's exact INT8 stage adapted to H3's absolute 64Q x 64KV route.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "sol_exact_common.h"

namespace comfy::hip_backend::sol_h3 {
namespace {

constexpr int HD = kHeadDim;
constexpr int BQ = kBlock;
constexpr int BK = kBlock;
constexpr int NWAVE = BQ / 16;
constexpr int kThreads = NWAVE * 32;
constexpr int DT = HD / 16;
constexpr int KT = BK / 16;
constexpr int NT = HD / 16;

#if defined(COMFY_MMA_GFX11)
constexpr int kLdsPad = 16;
#else
constexpr int kLdsPad = 8;
#endif
constexpr int kStrideK = HD + kLdsPad;
constexpr int kStrideV = BK + kLdsPad;

template <typename OutT>
__global__ __launch_bounds__(kThreads) void exact_kernel(
    const int8_t* __restrict__ qi, const float* __restrict__ qs,
    const int8_t* __restrict__ ki, const float2* __restrict__ ksb,
    const int8_t* __restrict__ vt, const float* __restrict__ vsc,
    const int32_t* __restrict__ route, const int32_t* __restrict__ counts,
    OutT* __restrict__ out, int Tq, int Tp, int H, int NQ, int NTB,
    int route_width, int64_t out_h, int64_t out_n, float scale_log2) {
    __shared__ __attribute__((aligned(16))) int8_t sK[BK * kStrideK];
    __shared__ __attribute__((aligned(16))) int8_t sVt[HD * kStrideV];
    __shared__ float2 sKsb[BK];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int r = frag_row(lane);
    const int qb = blockIdx.x, h = blockIdx.y;
    const int qrow = qb * BQ + wave * 16 + r;
    const int qread = imin(qrow, Tq - 1);

    typename MmaInt8::Frag qfrag[DT];
    const int8_t* qptr = qi + (static_cast<size_t>(h) * Tq + qread) * HD;
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) qfrag[dt] = MmaInt8::load_aligned(qptr, 0, dt * 16, 0, lane);
    const float qscale = qs[static_cast<size_t>(h) * Tq + qread] * scale_log2;

    const size_t route_row = static_cast<size_t>(h) * NQ + qb;
    const int32_t* indices = route + route_row * route_width;
    const int nblocks = counts[route_row];
    const size_t kbase = static_cast<size_t>(h) * Tp;
    const size_t vbase = static_cast<size_t>(h) * HD * Tp;

    float oacc[NT][8];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) oacc[nt][e] = 0.f;
    }
    float m = kNeg, l = 0.f;

    for (int selected = 0; selected < nblocks; ++selected) {
        const int block = indices[selected];
        if (block < 0 || block >= NTB) continue;
        const int64_t k0 = static_cast<int64_t>(block) * BK;
        __syncthreads();
        for (int idx = tid; idx < BK * (HD / 8); idx += kThreads) {
            const int p = idx / (HD / 8), c = (idx % (HD / 8)) * 8;
            *reinterpret_cast<uint2*>(sK + p * kStrideK + c) =
                *reinterpret_cast<const uint2*>(ki + (kbase + k0 + p) * HD + c);
        }
        for (int idx = tid; idx < HD * (BK / 8); idx += kThreads) {
            const int c = idx / (BK / 8), part = (idx % (BK / 8)) * 8;
            *reinterpret_cast<uint2*>(sVt + c * kStrideV + part) =
                *reinterpret_cast<const uint2*>(vt + vbase + static_cast<int64_t>(c) * Tp + k0 + part);
        }
        for (int idx = tid; idx < BK; idx += kThreads) sKsb[idx] = ksb[kbase + k0 + idx];
        __syncthreads();

        typename MmaInt8::Acc score[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            score[kt] = MmaInt8::zero();
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                score[kt] = MmaInt8::mma(
                    MmaInt8::load_aligned(sK, kt * 16 + r, dt * 16, kStrideK, lane),
                    qfrag[dt], score[kt]);
            }
        }

        float probability[KT][8];
        float mnew = m;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float2 sk = sKsb[kt * 16 + acc_row(lane, e)];
                const float value = fmaf(static_cast<float>(score[kt][e]), qscale * sk.x, sk.y);
                probability[kt][e] = value;
                mnew = fmaxf(mnew, value);
            }
        }
        mnew = fmaxf(mnew, swap_half_wave(mnew));
        const float alpha = exp2f(m - mnew);
        m = mnew;

        const float moff = mnew - kProbU8Offset;
        typename MmaInt8::Frag pfrag[KT];
        uint32_t li = 0;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            uint32_t packed[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                packed[e] = prob_to_u8(exp2f(probability[kt][e] - moff));
                li += packed[e];
            }
            pfrag[kt] = pack_prob_frag(packed, lane);
        }
        li += swap_half_wave_b32(li);
        l = fmaf(l, alpha, static_cast<float>(li));

#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            typename MmaInt8::Acc partial = MmaInt8::zero();
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                partial = MmaInt8::mma_ub(
                    MmaInt8::load_aligned(sVt, nt * 16 + r, kt * 16, kStrideV, lane),
                    pfrag[kt], partial);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) oacc[nt][e] = fmaf(oacc[nt][e], alpha, static_cast<float>(partial[e]));
        }
    }

    if (qrow >= Tq) return;
    OutT* row = out + static_cast<int64_t>(h) * out_h + static_cast<int64_t>(qrow) * out_n;
    const float inv = 1.f / fmaxf(l, 1e-30f);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        float values[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int d = nt * 16 + acc_row(lane, e);
            values[e] = oacc[nt][e] * inv * vsc[static_cast<size_t>(h) * HD + d];
        }
        store_o_tile(row, nt * 16, values, lane);
    }
}

template <typename OutT>
void launch_typed(const void* qi, const void* qs, const void* ki, const void* ksb,
                  const void* vt, const void* vsc, const void* route, const void* counts,
                  void* out, int H, int Tq, int Tk, int Tp, int route_width,
                  int64_t out_h, int64_t out_n, float scale, hipStream_t stream) {
    const int nq = (Tq + kBlock - 1) / kBlock;
    const int ntb = (Tk + kBlock - 1) / kBlock;
    exact_kernel<OutT><<<dim3(nq, H), kThreads, 0, stream>>>(
        static_cast<const int8_t*>(qi), static_cast<const float*>(qs),
        static_cast<const int8_t*>(ki), static_cast<const float2*>(ksb),
        static_cast<const int8_t*>(vt), static_cast<const float*>(vsc),
        static_cast<const int32_t*>(route), static_cast<const int32_t*>(counts),
        static_cast<OutT*>(out), Tq, Tp, H, nq, ntb, route_width, out_h, out_n,
        scale * 1.4426950408889634f);
}

}  // namespace
}  // namespace comfy::hip_backend::sol_h3

using namespace comfy::hip_backend::sol_h3;

extern "C" void launch_sol_h3_exact(
    const void* qi, const void* ki, const void* vt, void* out, const void* qs,
    const void* ksb, const void* vsc, const void* route, const void* counts,
    int route_width, int H, int Tq, int Tk, int Tp, int64_t out_h, int64_t out_n,
    float scale, int output_dtype, hipStream_t stream) {
    if (H <= 0 || Tq <= 0 || Tk <= 0 || Tp < Tk || Tp % kBlock != 0 || route_width <= 0)
        throw std::runtime_error("Sol exact attention received invalid shape or route geometry");
    if (output_dtype == 1) launch_typed<__half>(qi, qs, ki, ksb, vt, vsc, route, counts, out, H, Tq, Tk, Tp, route_width, out_h, out_n, scale, stream);
    else if (output_dtype == 2) launch_typed<__bf16>(qi, qs, ki, ksb, vt, vsc, route, counts, out, H, Tq, Tk, Tp, route_width, out_h, out_n, scale, stream);
    else throw std::runtime_error("Sol exact attention requires float16 or bfloat16 output");
}
