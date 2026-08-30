// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_runtime.h>

#include "../mma.h"
#include "sage_common.h"

namespace comfy::hip_backend::sol_h3 {

using namespace comfy::hip_backend::sage;

constexpr int kHeadDim = 128;
constexpr int kBlock = 64;
constexpr float kNeg = -3.0e38f;

__forceinline__ __device__ int imin(int a, int b) { return a < b ? a : b; }
__forceinline__ __device__ int imax(int a, int b) { return a > b ? a : b; }

__forceinline__ __device__ int8_t q8(float x, float inv) {
    const int r = static_cast<int>(rintf(x * inv));
    return static_cast<int8_t>(imax(-127, imin(127, r)));
}

__forceinline__ __device__ uint32_t prob_to_u8(float p) {
    return static_cast<uint32_t>(fminf(255.0f, fmaxf(0.0f, rintf(p))));
}

__forceinline__ __device__ MmaInt8::Frag pack_prob_frag(const uint32_t p[8], int lane) {
    const uint32_t lo = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
    const uint32_t hi = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
#if !defined(COMFY_MMA_GFX11)
    (void)lane;
    MmaInt8::Frag f;
    f[0] = static_cast<int>(lo);
    f[1] = static_cast<int>(hi);
    return f;
#else
    const uint32_t partner_lo = swap_half_wave_b32(lo);
    const uint32_t partner_hi = swap_half_wave_b32(hi);
    const bool even_half = lane < 16;
    const uint32_t even0 = even_half ? lo : partner_lo;
    const uint32_t odd0 = even_half ? partner_lo : lo;
    const uint32_t even1 = even_half ? hi : partner_hi;
    const uint32_t odd1 = even_half ? partner_hi : hi;
    MmaInt8::Frag f;
    f[0] = static_cast<int>(__builtin_amdgcn_perm(odd0, even0, 0x05010400u));
    f[1] = static_cast<int>(__builtin_amdgcn_perm(odd0, even0, 0x07030602u));
    f[2] = static_cast<int>(__builtin_amdgcn_perm(odd1, even1, 0x05010400u));
    f[3] = static_cast<int>(__builtin_amdgcn_perm(odd1, even1, 0x07030602u));
    return f;
#endif
}

template <typename OutT>
__forceinline__ __device__ void store_o_tile(OutT* __restrict__ row, int d_base,
                                             const float* vals, int lane) {
#if defined(COMFY_MMA_GFX12)
    __attribute__((aligned(16))) OutT packed[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) packed[e] = static_cast<OutT>(vals[e]);
    *reinterpret_cast<uint4*>(row + d_base + 8 * (lane / 16)) =
        *reinterpret_cast<const uint4*>(packed);
#else
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        row[d_base + acc_row(lane, e)] = static_cast<OutT>(vals[e]);
    }
#endif
}

}  // namespace comfy::hip_backend::sol_h3
