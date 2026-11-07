// SPDX-License-Identifier: Apache-2.0
//
// Chunk-scaled INT8 V.
//
// The global V carrier holds one dequant scale per (batch, KV head, channel),
// so every key waits for the whole-sequence maximum before it can be
// quantized: either a full BF16 copy of V is retained or V is projected a
// second time. The chunk-scaled carrier holds one scale per (batch, KV head,
// block, channel), where a block is the 2^shift consecutive keys of the QKV
// projection chunk that produced them. Each chunk is quantized the moment it
// is projected. Scales are laid out [B, H_kv, blocks, D]; with one block that
// is exactly the global layout, which is why a single-block carrier runs the
// ordinary kernels unchanged.
//
// The P x V accumulators stay in units of the current block's scale. Moving to
// keys of another block multiplies them per channel by s_current / s_next --
// the same pattern the online softmax already applies per tile for its
// running maximum -- and the fused V-scale epilogue multiplies by the final
// block's scale instead of the global one. A dense walk or an ascending sparse
// route crosses each boundary once, so the rescale runs at most blocks - 1
// times per CTA rather than once per tile.
//
// Packed kernels sum several routed subtiles into one INT32 product, so an
// iteration must not mix blocks. The router guarantees it: in every route row,
// each block except the last contributes a whole number of iterations
// (int8_attention.align_route_to_v_blocks pads with neighbouring tiles). The
// block check then runs once per iteration, at its top: placed beside P x V,
// the same check cost the register-bound fine-route kernel its copy/MMA
// overlap (+67% there, against +4% at the top).

#pragma once

#include <stdexcept>
#include <string>

#include "attn_utils.cuh"

// Host side: validate a carrier's block size and derive the kernel arguments.
// scale_rows == 0 is the global carrier. A carrier that fits in one block has
// the global layout and runs the ordinary kernels.
struct ChunkedVLaunch {
  uint32_t blocks = 1;
  uint32_t shift = 0;
  bool chunked() const { return blocks > 1; }
};

inline ChunkedVLaunch chunked_v_launch(int scale_rows, int kv_len) {
  ChunkedVLaunch launch;
  if (scale_rows == 0) {
    return launch;
  }
  // 128 keeps every dense and sparse KV tile inside one block; a power of two
  // makes the block of a key a shift.
  if (scale_rows < 128 || (scale_rows & (scale_rows - 1)) != 0) {
    throw std::runtime_error(
        "chunk-scaled V needs a power-of-two block of at least 128 rows, got " +
        std::to_string(scale_rows));
  }
  while ((1 << launch.shift) < scale_rows) {
    launch.shift++;
  }
  launch.blocks = static_cast<uint32_t>((kv_len + scale_rows - 1) / scale_rows);
  if (launch.blocks == 0) {
    launch.blocks = 1;
  }
  if (launch.blocks > 0xffffu) {
    throw std::runtime_error("chunk-scaled V supports at most 65535 blocks");
  }
  return launch;
}

// s_from / s_to for this thread's four channels of one 16-channel V tile, in
// the (k / 4) * 2 + (k % 2) order the fused V-scale epilogue uses.
template <uint32_t head_dim>
__device__ __forceinline__ void chunked_v_ratio(const float *head,
                                                uint32_t from, uint32_t to,
                                                uint32_t fv, float (&r)[4]) {
  const uint32_t channel = fv * 16 + (get_lane_id() % 4) * 2;
  const float *source = head + static_cast<int64_t>(from) * head_dim + channel;
  const float *target = head + static_cast<int64_t>(to) * head_dim + channel;
  const float2 a0 = *reinterpret_cast<const float2 *>(source);
  const float2 a1 = *reinterpret_cast<const float2 *>(source + 8);
  const float2 b0 = *reinterpret_cast<const float2 *>(target);
  const float2 b1 = *reinterpret_cast<const float2 *>(target + 8);
  // Fast division need not return exactly one for equal operands.
  r[0] = a0.x == b0.x ? 1.0f : a0.x / b0.x;
  r[1] = a0.y == b0.y ? 1.0f : a0.y / b0.y;
  r[2] = a1.x == b1.x ? 1.0f : a1.x / b1.x;
  r[3] = a1.y == b1.y ? 1.0f : a1.y / b1.y;
}

template <uint32_t num_tiles_q, uint32_t num_tiles_v, uint32_t head_dim>
__device__ __forceinline__ void chunked_v_rescale(float RO[][num_tiles_v][8],
                                               const float *head,
                                               uint32_t from, uint32_t to) {
#pragma unroll
  for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
    float r[4];
    chunked_v_ratio<head_dim>(head, from, to, fv, r);
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        RO[fq][fv][k] *= r[(k / 4) * 2 + (k % 2)];
      }
    }
  }
}

// Express the accumulators in `block` before a P x V product whose keys all
// lie in it.
template <uint32_t num_tiles_q, uint32_t num_tiles_v, uint32_t head_dim>
__device__ __forceinline__ void chunked_v_enter(float RO[][num_tiles_v][8],
                                                const float *head,
                                                uint32_t &current,
                                                uint32_t block) {
  if (block != current) {
    chunked_v_rescale<num_tiles_q, num_tiles_v, head_dim>(RO, head, current,
                                                          block);
    current = block;
  }
}

// The fused V-scale epilogue's base pointer for this thread's channels.
template <uint32_t head_dim>
__device__ __forceinline__ const float *chunked_v_epilogue(const float *head,
                                                           uint32_t current) {
  return head + static_cast<int64_t>(current) * head_dim +
         (get_lane_id() % 4) * 2;
}
