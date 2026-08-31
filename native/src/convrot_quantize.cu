/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from ComfyKitchen's BF16 ConvRot-256 activation quantizer. The
 * exact source revision is recorded in PROVENANCE.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreadsPerWarp = 32;
constexpr int kConvRotGroup = 256;

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

template <int NumWarps>
__device__ __forceinline__ float block_reduce_max(
    float value, float *warp_smem, float *block_smem) {
  const int lane = threadIdx.x & (kThreadsPerWarp - 1);
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_max(value);
  if (lane == 0) {
    warp_smem[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < NumWarps ? warp_smem[lane] : 0.0f;
    total = warp_reduce_max(total);
    if (lane == 0) {
      *block_smem = total;
    }
  }
  __syncthreads();
  return *block_smem;
}

template <int Stride>
__device__ __forceinline__ void convrot_stage64(
    const float *__restrict__ source, float *__restrict__ destination,
    int lane) {
  const int base = (lane % Stride) + (lane / Stride) * (4 * Stride);
  const float x0 = source[base];
  const float x1 = source[base + Stride];
  const float x2 = source[base + 2 * Stride];
  const float x3 = source[base + 3 * Stride];
  destination[base] = 0.5f * (x0 + x1 + x2 - x3);
  destination[base + Stride] = 0.5f * (x0 + x1 - x2 + x3);
  destination[base + 2 * Stride] = 0.5f * (x0 - x1 + x2 + x3);
  destination[base + 3 * Stride] = 0.5f * (-x0 + x1 + x2 + x3);
}

template <int Stride>
__device__ __forceinline__ float convrot_stage64_store_absmax(
    const float *__restrict__ source, float *__restrict__ output, int lane) {
  const int base = (lane % Stride) + (lane / Stride) * (4 * Stride);
  const float x0 = source[base];
  const float x1 = source[base + Stride];
  const float x2 = source[base + 2 * Stride];
  const float x3 = source[base + 3 * Stride];
  const float y0 = 0.5f * (x0 + x1 + x2 - x3);
  const float y1 = 0.5f * (x0 + x1 - x2 + x3);
  const float y2 = 0.5f * (x0 - x1 + x2 + x3);
  const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
  output[base] = y0;
  output[base + Stride] = y1;
  output[base + 2 * Stride] = y2;
  output[base + 3 * Stride] = y3;
  return fmaxf(fmaxf(fabsf(y0), fabsf(y1)),
               fmaxf(fabsf(y2), fabsf(y3)));
}

__device__ __forceinline__ float quant_div_bf16(float value, float scale) {
  const float value_bf16 = __bfloat162float(__float2bfloat16_rn(value));
  const float scale_bf16 = __bfloat162float(__float2bfloat16_rn(scale));
  return __bfloat162float(
      __float2bfloat16_rn(value_bf16 / scale_bf16));
}

template <int BlockThreads>
__global__ void quantize_bf16_rowwise_convrot256_kernel(
    const nv_bfloat16 *__restrict__ input, int8_t *__restrict__ output,
    float *__restrict__ scales, int columns) {
  constexpr int kGroupThreads = 64;
  constexpr int kGroupsInFlight = BlockThreads / kGroupThreads;
  constexpr int kWarps = BlockThreads / kThreadsPerWarp;

  extern __shared__ float smem[];
  float *row_buffer = smem;
  float *temporary = smem + columns;

  __shared__ float warp_smem[kWarps];
  __shared__ float block_smem;

  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int thread = threadIdx.x;
  const int group_slot = thread / kGroupThreads;
  const int lane = thread % kGroupThreads;
  const int64_t row_offset = row * columns;
  const int group_count = columns / kConvRotGroup;

  float *buffer0 = temporary + group_slot * (2 * kConvRotGroup);
  float *buffer1 = buffer0 + kConvRotGroup;
  float abs_max = 0.0f;

  const int iterations =
      (group_count + kGroupsInFlight - 1) / kGroupsInFlight;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int group = iteration * kGroupsInFlight + group_slot;
    const bool active = group < group_count;
    const int base = lane * 4;
    const int group_column = group * kConvRotGroup;
    const int column = group_column + base;

    const float x0 =
        active ? __bfloat162float(input[row_offset + column]) : 0.0f;
    const float x1 =
        active ? __bfloat162float(input[row_offset + column + 1]) : 0.0f;
    const float x2 =
        active ? __bfloat162float(input[row_offset + column + 2]) : 0.0f;
    const float x3 =
        active ? __bfloat162float(input[row_offset + column + 3]) : 0.0f;
    buffer1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buffer1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buffer1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buffer1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_stage64<4>(buffer1, buffer0, lane);
    __syncthreads();
    convrot_stage64<16>(buffer0, buffer1, lane);
    __syncthreads();

    if (active) {
      abs_max = fmaxf(
          abs_max, convrot_stage64_store_absmax<64>(
                       buffer1, row_buffer + group_column, lane));
    }
    __syncthreads();
  }

  abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
  const float scale =
      fmaxf(fminf(abs_max, 3.38953139e38f) * (1.0f / 127.0f), 1.0e-30f);
  if (thread == 0) {
    scales[row] = scale;
  }

  for (int column = thread; column < columns; column += BlockThreads) {
    const int64_t index = row_offset + column;
    float quantized = nearbyintf(quant_div_bf16(row_buffer[column], scale));
    quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
    output[index] = static_cast<int8_t>(quantized);
  }
}

template <int BlockThreads>
void launch_quantizer(const void *input, void *output, void *scales,
                      int64_t rows, int columns, cudaStream_t stream) {
  auto kernel = quantize_bf16_rowwise_convrot256_kernel<BlockThreads>;
  constexpr int kGroupsInFlight = BlockThreads / 64;
  const size_t smem_bytes =
      (static_cast<size_t>(columns) +
       kGroupsInFlight * 2 * kConvRotGroup) *
      sizeof(float);
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("ConvRot-256 quantizer shared memory request failed: ") +
        cudaGetErrorString(status));
  }
  kernel<<<static_cast<unsigned int>(rows), BlockThreads, smem_bytes, stream>>>(
      static_cast<const nv_bfloat16 *>(input),
      static_cast<int8_t *>(output), static_cast<float *>(scales), columns);
}

} // namespace

void launch_h3_quantize_bf16_rowwise_convrot256(
    const void *input, void *output, void *scales, int64_t rows,
    int64_t columns, cudaStream_t stream) {
  if (rows == 0 || columns == 0) {
    return;
  }
  if (!input || !output || !scales) {
    throw std::runtime_error("ConvRot-256 quantizer received a null pointer");
  }
  if (rows < 0 || rows > std::numeric_limits<int>::max()) {
    throw std::runtime_error("ConvRot-256 quantizer row count is out of range");
  }
  if (columns < 0 || columns > std::numeric_limits<int>::max() ||
      columns % kConvRotGroup != 0) {
    throw std::runtime_error(
        "ConvRot-256 quantizer requires K divisible by 256 and within INT_MAX");
  }

  if (rows == 1) {
    launch_quantizer<512>(input, output, scales, rows,
                          static_cast<int>(columns), stream);
  } else if (columns == kConvRotGroup) {
    launch_quantizer<64>(input, output, scales, rows,
                         static_cast<int>(columns), stream);
  } else if (columns == 2560) {
    launch_quantizer<640>(input, output, scales, rows,
                          static_cast<int>(columns), stream);
  } else if (columns == 6144) {
    launch_quantizer<768>(input, output, scales, rows,
                          static_cast<int>(columns), stream);
  } else {
    launch_quantizer<1024>(input, output, scales, rows,
                           static_cast<int>(columns), stream);
  }

  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA BF16 ConvRot-256 quantization failed: ") +
        cudaGetErrorString(status));
  }
}
