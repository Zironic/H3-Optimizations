/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * H3's fused Q producer, adapted from the ComfyKitchen CUTLASS INT8 GEMM
 * prototype based on ComfyKitchen commit c6b7ba49ab219c7c1236089b0c8908ea671993ee.
 * This production copy intentionally contains only the measured exact
 * 128-row x 256-column configuration.
 */

#include <cuda_runtime.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "float_utils.cuh"

namespace {

using namespace cute;

__forceinline__ __device__ void h3_convrot4(float *values) {
  const float x0 = values[0];
  const float x1 = values[1];
  const float x2 = values[2];
  const float x3 = values[3];
  const float a0 = x0 + x1;
  const float a1 = x0 - x1;
  const float a2 = x2 + x3;
  const float a3 = x2 - x3;
  values[0] = (a0 + a2) * 0.5f;
  values[1] = (a1 + a3) * 0.5f;
  values[2] = (a0 - a2) * 0.5f;
  values[3] = (a1 - a3) * 0.5f;
}

__forceinline__ __device__ void h3_convrot_sign128(float *values, int lane) {
  constexpr uint32_t signs_0 = 0x1035997bu;
  constexpr uint32_t signs_1 = 0x8087f5eeu;
  constexpr uint32_t signs_2 = 0xee2e4e1au;
  constexpr uint32_t signs_3 = 0x71132418u;
  const uint32_t signs = lane < 8    ? signs_0
                         : lane < 16 ? signs_1
                         : lane < 24 ? signs_2
                                     : signs_3;
  const int shift = (lane & 7) * 4;
#pragma unroll
  for (int channel = 0; channel < 4; ++channel) {
    const uint32_t flip = ((signs >> (shift + channel)) & 1u) ^ 1u;
    values[channel] =
        __uint_as_float(__float_as_uint(values[channel]) ^ (flip << 31));
  }
}

__forceinline__ __device__ void h3_convrot128(float *values) {
  const int lane = threadIdx.x & 31;
  h3_convrot_sign128(values, lane);
  h3_convrot4(values);
#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int channel = 0; channel < 4; ++channel) {
      const float other =
          __shfl_xor_sync(0xffffffffu, values[channel], bit);
      values[channel] =
          (lane & bit) ? other - values[channel] : values[channel] + other;
    }
  }
#pragma unroll
  for (int channel = 0; channel < 4; ++channel)
    values[channel] *= 0.1767766952966369f;
}

template <class ThreadMap, class Element, int TBM, int TBN>
struct VisitorH3QStore {
  struct Arguments {
    int8_t *q = nullptr;
    float *q_scale = nullptr;
    Element *debug = nullptr;
    Element *summary = nullptr;
    const Element *norm = nullptr;
    const Element *freqs = nullptr;
    int m = 0;
    int heads = 0;
    int full_k_length = 0;
    float epsilon = 0.f;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &, Arguments const &args, void *) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) {
    return 0;
  }

  struct SharedStorage {
    alignas(16) Element tile[TBM * TBN];
  };

  static int constexpr vec_bits =
      ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<Element>::value;
  using VecType = cute::uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE VisitorH3QStore() = default;

  CUTLASS_HOST_DEVICE
  VisitorH3QStore(Params const &params,
                  SharedStorage const &shared_storage)
      : params_ptr(&params),
        tile(const_cast<Element *>(shared_storage.tile)) {}

  Params const *params_ptr;
  Element *tile;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks
      : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(RTensor &&tC_rAux, CTensor &&tC_cAux,
              ProblemShape problem_shape, Params const *params_ptr,
              Element *tile, int tile_m, int tile_n)
        : tC_rAux(cute::forward<RTensor>(tC_rAux)),
          tC_cAux(cute::forward<CTensor>(tC_cAux)),
          problem_shape(problem_shape), params_ptr(params_ptr), tile(tile),
          tile_m(tile_m), tile_n(tile_n) {}

    RTensor tC_rAux;
    CTensor tC_cAux;
    ProblemShape problem_shape;
    Params const *params_ptr;
    Element *tile;
    int tile_m;
    int tile_n;

    CUTLASS_DEVICE void begin_step(int) { clear(tC_rAux); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(int, int, int, int frg_idx,
          cutlass::Array<ElementAccumulator, FragmentSize> const &,
          cutlass::Array<ElementInput, FragmentSize> const &frg_input) {
      using ConvertInput = cutlass::NumericArrayConverter<
          Element, ElementInput, FragmentSize,
          cutlass::FloatRoundStyle::round_to_nearest>;
      ConvertInput convert_input{};
      auto tC_rAux_frg =
          recast<cutlass::Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(frg_idx) = convert_input(frg_input);
      return frg_input;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
#pragma unroll
      for (int i = 0; i < size(src_v); ++i) {
        const bool guard = elem_less(coord_v(i), problem_shape);
        if (guard) {
          const int row = int(get<0>(coord_v(i))) - tile_m;
          const int column = int(get<1>(coord_v(i))) - tile_n;
          *reinterpret_cast<VecType *>(&tile[row * TBN + column]) = src_v(i);
        }
      }
    }

    CUTLASS_DEVICE void end_epilogue() {
      __syncthreads();

      const int lane = threadIdx.x & 31;
      const int head_in_tile = threadIdx.x >> 7;
      const int warp_in_head = (threadIdx.x >> 5) & 3;
      const int head = tile_n / 128 + head_in_tile;
      if (head >= params_ptr->heads)
        return;

      const int channel = lane * 4;
      const int head_column = head_in_tile * 128 + channel;
      const int q_scale_tiles = (params_ptr->m + 127) / 128;
      const int q_scale_head = head * q_scale_tiles * 32;
      const int q_scale_tile = (tile_m / 128) * 32;
      float summary_sum[2][4] = {};

#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
#pragma unroll
        for (int group = 0; group < 2; ++group) {
          const int row_group = warp_in_head * 2 + group;
          float values[16];

#pragma unroll
          for (int row_iter = 0; row_iter < 4; ++row_iter) {
            const int local_row = sub * 32 + row_group + row_iter * 8;
            const int global_row = tile_m + local_row;
            float *row_values = &values[row_iter * 4];
            float sum = 0.f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const float value =
                  global_row < params_ptr->m
                      ? static_cast<float>(
                            tile[local_row * TBN + head_column + item])
                      : 0.f;
              row_values[item] = value;
              sum += value * value;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 16);
            sum += __shfl_xor_sync(0xffffffffu, sum, 8);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            const float rrms =
                rsqrtf(sum / 128.f + params_ptr->epsilon);

            if (global_row < params_ptr->m) {
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                const float normalized =
                    row_values[item] * rrms *
                    static_cast<float>(params_ptr->norm[channel + item]);
                row_values[item] = static_cast<float>(Element(normalized));
              }

              const int pair_lane = lane < 12   ? lane + 12
                                    : lane < 24 ? lane - 12
                                                : lane;
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                const float other = __shfl_sync(
                    0xffffffffu, row_values[item], pair_lane);
                if (lane < 24) {
                  const float low = lane < 12 ? row_values[item] : other;
                  const float high = lane < 12 ? other : row_values[item];
                  const int pair = (lane % 12) * 4 + item;
                  const Element *rotation =
                      params_ptr->freqs + global_row * 48 * 4 + pair * 4;
                  const int rotation_row = lane < 12 ? 0 : 2;
                  const float first_rotation =
                      static_cast<float>(rotation[rotation_row]);
                  const float second_rotation =
                      static_cast<float>(rotation[rotation_row + 1]);
                  const float first =
                      static_cast<float>(Element(low * first_rotation));
                  const float second =
                      static_cast<float>(Element(high * second_rotation));
                  row_values[item] =
                      static_cast<float>(Element(first + second));
                }
              }
              if (params_ptr->debug != nullptr) {
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                  params_ptr->debug[(int64_t)global_row *
                                        params_ptr->heads * 128 +
                                    (int64_t)head * 128 + channel + item] =
                      Element(row_values[item]);
                }
              }
              if (params_ptr->summary != nullptr) {
#pragma unroll
                for (int item = 0; item < 4; ++item)
                  summary_sum[sub >> 1][item] += row_values[item];
              }
            } else {
#pragma unroll
              for (int item = 0; item < 4; ++item)
                row_values[item] = 0.f;
            }

            if (params_ptr->full_k_length > 256)
              h3_convrot128(row_values);
            else
              h3_convrot4(row_values);
          }

          float maximum = 0.f;
#pragma unroll
          for (int item = 0; item < 16; ++item)
            maximum = fmaxf(maximum, fabsf(values[item]));
          maximum = comfy::warp_reduce_fmax(maximum);
          const float scale = maximum / 127.f + 1e-7f;
          const float inverse_scale = 1.f / scale;
          if (lane == 0) {
            params_ptr->q_scale[q_scale_head + q_scale_tile + sub * 8 +
                                row_group] = scale;
          }

#pragma unroll
          for (int row_iter = 0; row_iter < 4; ++row_iter) {
            const int local_row = sub * 32 + row_group + row_iter * 8;
            const int global_row = tile_m + local_row;
            if (global_row < params_ptr->m) {
              float *row_values = &values[row_iter * 4];
              int8_t *output =
                  params_ptr->q +
                  ((int64_t)head * params_ptr->m + global_row) * 128 + channel;
              comfy::store4_i8(
                  output, comfy::quant_int8_rcp(row_values[0], inverse_scale),
                  comfy::quant_int8_rcp(row_values[1], inverse_scale),
                  comfy::quant_int8_rcp(row_values[2], inverse_scale),
                  comfy::quant_int8_rcp(row_values[3], inverse_scale));
            }
          }
        }
      }

      if (params_ptr->summary != nullptr) {
        __syncthreads();
        float *summary_partial = reinterpret_cast<float *>(tile);
#pragma unroll
        for (int local_tile = 0; local_tile < 2; ++local_tile) {
#pragma unroll
          for (int item = 0; item < 4; ++item) {
            summary_partial[(((head_in_tile * 2 + local_tile) * 4 +
                              warp_in_head) *
                                 128) +
                            channel + item] = summary_sum[local_tile][item];
          }
        }
        __syncthreads();
        const int summary_tiles = (params_ptr->m + 63) / 64;
        for (int index = threadIdx.x; index < (TBN / 128) * 2 * 128;
             index += blockDim.x) {
          const int local_head = index / 256;
          const int local_tile = (index / 128) & 1;
          const int summary_channel = index & 127;
          const int summary_head = tile_n / 128 + local_head;
          const int summary_tile = tile_m / 64 + local_tile;
          const int row_start = local_tile * 64;
          const int remaining_rows = params_ptr->m - tile_m - row_start;
          const int valid_rows = remaining_rows < 64 ? remaining_rows : 64;
          if (summary_head < params_ptr->heads &&
              summary_tile < summary_tiles && valid_rows > 0) {
            float sum = 0.f;
#pragma unroll
            for (int warp = 0; warp < 4; ++warp) {
              sum += summary_partial[(((local_head * 2 + local_tile) * 4 +
                                       warp) *
                                          128) +
                                     summary_channel];
            }
            params_ptr->summary[((int64_t)summary_head * summary_tiles +
                                 summary_tile) *
                                    128 +
                                summary_channel] = Element(sum / valid_rows);
          }
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(cutlass::gemm::GemmCoord threadblock_tile_offset,
                int thread_idx, ProblemShape problem_shape) {
    const int64_t m = get<0>(problem_shape);
    const int64_t n = get<1>(problem_shape);
    auto dummy = make_tensor(make_gmem_ptr(static_cast<Element *>(nullptr)),
                             problem_shape,
                             cute::Stride<int64_t, _1, int64_t>{n, _1{},
                                                                 m * n});
    auto partitioned = group_modes<3, 6>(
        ThreadMap::partition(dummy, thread_idx, threadblock_tile_offset));
    auto tC_rAux =
        make_tensor_like(take<0, 3>(recast<VecType>(partitioned)));
    auto coordinates = make_identity_tensor(dummy.shape());
    auto tC_cAux = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(
            coordinates, thread_idx, threadblock_tile_offset)),
        Shape<Int<VecLength>>{}, (_0{}));
    return Callbacks<decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr,
        tile, threadblock_tile_offset.m() * TBM,
        threadblock_tile_offset.n() * TBN);
  }
};

struct H3FusedQGemm {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementAcc = int32_t;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TB = cutlass::gemm::GemmShape<128, 256, 64>;
  using Warp = cutlass::gemm::GemmShape<64, 64, 64>;
  using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int Align = 16;
  static constexpr int AlignC = 8;
  static constexpr int EVTStages = 1;
  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      TB, Warp, ElementC, AlignC, EVTStages>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      ThreadMap, ElementCompute, cute::Stride<_0, _1, int32_t>>;
  using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 =
      cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
  using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT1 =
      cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
  using StoreQ = VisitorH3QStore<ThreadMap, ElementC, 128, 256>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreQ, EVT1>;
  using GemmKernel =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
          ElementA, LayoutA, cutlass::ComplexTransform::kNone, Align, ElementB,
          LayoutB, cutlass::ComplexTransform::kNone, Align, ElementC, LayoutC,
          AlignC, ElementAcc, ElementCompute,
          cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst,
          EVTD,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
          cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static bool run(const int8_t *a, const int8_t *b, const float *x_scale,
                  const float *weight_scale, const ElementC *norm,
                  const ElementC *freqs, ElementC *debug, ElementC *summary,
                  int8_t *q, float *q_scale, int m, int n, int k,
                  int full_k_length, float epsilon, cudaStream_t stream) {
    if (n % 256 != 0 || n / 128 <= 0)
      return false;
    cutlass::gemm::GemmCoord problem(m, n, k);
    typename EVTD::Arguments callbacks{
        {{{},
          {const_cast<float *>(x_scale), 0.f, {_1{}, _0{}, m}},
          {}},
         {const_cast<float *>(weight_scale), 0.f, {_0{}, _1{}, n}},
         {}},
        {q, q_scale, debug, summary, norm, freqs, m, n / 128,
         full_k_length, epsilon}};
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, callbacks,
        const_cast<int8_t *>(a), const_cast<int8_t *>(b), nullptr, nullptr,
        (int64_t)m * k, (int64_t)n * k, 0, 0, k, k, 0, 0);
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess)
      return false;
    const size_t workspace_size = Gemm::get_workspace_size(args);
    if (workspace_size != 0)
      return false;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
      return false;
    return gemm(stream) == cutlass::Status::kSuccess;
  }
};

} // namespace

bool launch_h3_fused_q_cutlass(
    const void *a, const void *b, const void *x_scale,
    const void *weight_scale, const void *norm, const void *freqs, void *debug,
    void *summary, void *q, void *q_scale, int64_t m, int64_t n, int64_t k,
    int full_k_length, float epsilon, cudaStream_t stream) {
  if (m == 0 || n == 0 || k == 0)
    return true;
  return H3FusedQGemm::run(
      static_cast<const int8_t *>(a), static_cast<const int8_t *>(b),
      static_cast<const float *>(x_scale),
      static_cast<const float *>(weight_scale),
      static_cast<const cutlass::bfloat16_t *>(norm),
      static_cast<const cutlass::bfloat16_t *>(freqs),
      static_cast<cutlass::bfloat16_t *>(debug),
      static_cast<cutlass::bfloat16_t *>(summary), static_cast<int8_t *>(q),
      static_cast<float *>(q_scale), static_cast<int>(m), static_cast<int>(n),
      static_cast<int>(k), full_k_length, epsilon, stream);
}
