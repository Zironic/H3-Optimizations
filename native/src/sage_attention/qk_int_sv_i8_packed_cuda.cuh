// SPDX-License-Identifier: Apache-2.0
//
// Packed block-sparse traversal. A 64Q x 128KV CTA consumes N_SUB = 128 /
// SUB_K routed SUB_K-wide KV tiles per iteration, reading the ordinary
// SUB_K-granular carrier and route unchanged. Routing keeps the fine SUB_K
// granularity while the online-softmax/barrier cadence runs at 128.
//
// Production runs SUB_K = 64 in place of the 64Q x 64KV kernel on the
// power-limited consumer architectures (see sage_attn_sparse_launcher.cu). It
// takes slightly more cycles than 64Q x 64KV but less energy per call, which
// the power governor turns into a higher clock. P is quantized against the
// 128-column running maximum, so attention error vs FP32 moves from about
// 1.62% to 1.66%.
//
// Only the production configuration is built: head_dim 128, INT8 QK with
// per-thread scales, UINT8 P x INT8 V, fused V scale, no mask, optional LSE,
// delta-encoded routes.
// Each warp copies 32 contiguous K rows and each lane one 16-byte V column
// chunk, so for SUB_K in {32, 64} every thread's K and V addresses come from
// exactly one subtile. Only the per-thread K dequant scale differs across the
// thread's columns, one per subtile.
// Dense SUB_K=128 retains the ordinary 128-token K quantization grid.

#pragma once

#include "qk_int_sv_i8_sparse_cuda.cuh"

template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          uint32_t TILES_PER_SUB>
__device__ __forceinline__ void update_mdo_i32_u8_packed(
    int32_t RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2],
    float d[][2], const float (&scale)[num_tiles_k / TILES_PER_SUB],
    const float exp_offset_value, uint32_t RS_u8[][num_tiles_k / 2][4]) {
  constexpr uint32_t N_SUB = num_tiles_k / TILES_PER_SUB;
  static_assert(TILES_PER_SUB % 2 == 0,
                "probability packing pairs 16-column tiles within one subtile");
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      const float m_prev = m[fq][k];
      // Scales are positive, so each subtile's maximum is its integer maximum
      // times its scale; the row maximum is the largest of those.
      float m_temp = -CUDART_INF_F;
#pragma unroll
      for (uint32_t s = 0; s < N_SUB; s++) {
        int32_t m_sub = INT_MIN;
#pragma unroll
        for (uint32_t t = 0; t < TILES_PER_SUB; t++) {
          const uint32_t fk = s * TILES_PER_SUB + t;
          const int32_t m_local =
              max(max(RS[fq][fk][k * 2], RS[fq][fk][k * 2 + 1]),
                  max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
          m_sub = max(m_sub, m_local);
        }
        m_temp = max(m_temp, fmaf(__int2float_rz(m_sub), scale[s],
                                  -exp_offset_value));
      }
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
      const float tile_m = m_temp;
      m[fq][k] = max(m_prev, tile_m);

      const float smaller_scale = math::ptx_exp2(-fabsf(m_prev - tile_m));
      const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
      const float tile_scale = tile_m < m_prev ? smaller_scale : 1.0f;
      d[fq][k] *= o_scale;
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        RO[fq][fv][k * 2] *= o_scale;
        RO[fq][fv][k * 2 + 1] *= o_scale;
        RO[fq][fv][k * 2 + 4] *= o_scale;
        RO[fq][fv][k * 2 + 5] *= o_scale;
      }

      const float negative_m = -tile_m;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
        const float sub_scale = scale[(fk * 2) / TILES_PER_SUB];
        const PackedU8RowSum probabilities_0 = pack_scaled_exp2_u8x4(
            RS[fq][fk * 2][k * 2], RS[fq][fk * 2][k * 2 + 1],
            RS[fq][fk * 2][k * 2 + 4], RS[fq][fk * 2][k * 2 + 5], sub_scale,
            negative_m);
        const PackedU8RowSum probabilities_1 = pack_scaled_exp2_u8x4(
            RS[fq][fk * 2 + 1][k * 2], RS[fq][fk * 2 + 1][k * 2 + 1],
            RS[fq][fk * 2 + 1][k * 2 + 4], RS[fq][fk * 2 + 1][k * 2 + 5],
            sub_scale, negative_m);
        RS_u8[fq][fk][k] = probabilities_0.probabilities;
        RS_u8[fq][fk][k + 2] = probabilities_1.probabilities;
        d[fq][k] += probabilities_0.denominator * tile_scale;
        d[fq][k] += probabilities_1.denominator * tile_scale;
      }
      RS[fq][0][k] = __float_as_int(tile_scale);
    }
  }
}

template <uint32_t SUB_K, typename DTypeOut, bool return_lse,
          uint32_t MIN_BLOCKS, bool DENSE = false, bool block_v_scale = false>
__global__ __launch_bounds__(128, MIN_BLOCKS) void qk_int_sv_i8_packed_attn_kernel(
    int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V,
    DTypeOut *__restrict__ O, float *__restrict__ Lse,
    float *__restrict__ Q_scale, float *__restrict__ K_scale,
    float *__restrict__ V_scale, const uint32_t qo_len, const uint32_t kv_len,
    const uint32_t num_kv_groups, const uint32_t stride_bz_q,
    const uint32_t stride_seq_q, const uint32_t stride_h_q,
    const uint32_t stride_bz_k, const uint32_t stride_seq_k,
    const uint32_t stride_h_k, const uint32_t stride_bz_v,
    const uint32_t stride_h_v, const uint32_t stride_d_v,
    const uint32_t stride_bz_o, const uint32_t stride_seq_o,
    const uint32_t stride_h_o, const uint32_t stride_bz_q_scale,
    const uint32_t stride_h_q_scale, float sm_scale,
    const int32_t *__restrict__ BlockLut,
    const int32_t *__restrict__ ValidBlockNum, const uint32_t lut_stride,
    const uint32_t v_scale_rows = 0) {
  constexpr uint32_t CTA_Q = 64;
  constexpr uint32_t CTA_K = 128;
  constexpr uint32_t WARP_Q = 16;
  constexpr uint32_t head_dim = 128;
  static_assert(SUB_K == 32 || SUB_K == 64 || (SUB_K == 128 && DENSE),
                "128-token carriers require dense traversal");
  constexpr uint32_t N_SUB = CTA_K / SUB_K;
  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = 1;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = CTA_K / MMA_QK_N;
  constexpr uint32_t TILES_PER_SUB = SUB_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = head_dim / MMA_QK_K;
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;
  static_assert(num_tiles_q == 1);
  constexpr uint32_t QK_SMEM_STRIDE = head_dim;
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = CTA_K;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;
  const uint32_t kv_head = head_id / num_kv_groups;
  const uint32_t scale_rows = v_scale_rows ? v_scale_rows : SUB_K;
  const uint32_t tiles_per_scale = scale_rows / SUB_K;
  const bool scale_power_of_two = (tiles_per_scale & (tiles_per_scale - 1)) == 0;
  const uint32_t scale_shift = __ffs(tiles_per_scale) - 1;
  auto v_scale_for_block = [&](uint32_t block) -> const float * {
    if constexpr (block_v_scale) {
      const uint32_t index = scale_power_of_two ? block >> scale_shift : block / tiles_per_scale;
      const uint32_t head = batch_id * (num_qo_heads / num_kv_groups) + kv_head;
      return V_scale + (static_cast<int64_t>(head) * div_ceil(kv_len, scale_rows) + index) * head_dim;
    }
    return nullptr;
  };

  sm_scale *= math::log2e;

  int32_t RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];

  const uint32_t q_scale_idx =
      batch_id * stride_bz_q_scale + head_id * stride_h_q_scale +
      bx * ((CTA_Q / 32) * 8) +
      (get_warp_idx_q<num_warps_q, num_warps_k>() / 2) * 8 + lane_id / 4;
  // The carrier is SUB_K-granular: four per-thread K scales per subtile.
  const uint32_t num_sub_blocks = div_ceil(kv_len, SUB_K);
  const float *const k_scale_lane =
      K_scale + (batch_id * (num_qo_heads / num_kv_groups) + kv_head) *
                    (num_sub_blocks * 4) +
      lane_id % 4;

#pragma unroll
  for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
      RO[0][fv][k] = 0.0f;
    }
  }
#pragma unroll
  for (uint32_t k = 0; k < 2; k++) {
    m[0][k] = -50000.0f;
    d[0][k] = 1.0f;
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;
  constexpr SwizzleMode swizzle_mode_QK = SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(
      smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V = SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(
      smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t line_lanes_QK = 8;
  constexpr uint32_t lines_per_warp_QK = 4;
  constexpr uint32_t line_lanes_V = 8;
  constexpr uint32_t lines_per_warp_V = 4;
  constexpr uint32_t line_lanes_O = 8;
  constexpr uint32_t lines_per_warp_O = 4;
  constexpr uint32_t QK_smem_iters_row =
      QK_SMEM_STRIDE / (line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row =
      V_SMEM_STRIDE / (line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col = head_dim / (num_warps * lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row =
      O_SMEM_STRIDE / (line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * lines_per_warp_O);

  // A warp copies CTA_K / num_warps contiguous K rows, all inside one subtile.
  constexpr uint32_t K_rows_per_warp = CTA_K / num_warps;
  static_assert(SUB_K % K_rows_per_warp == 0);
  const uint32_t k_sub = (K_rows_per_warp * warp_id) / SUB_K;
  const uint32_t k_row_in_sub =
      (K_rows_per_warp * warp_id) % SUB_K + lane_id / line_lanes_QK;
  // A lane copies one 16-byte V column chunk of each channel row.
  const uint32_t v_column = (lane_id % line_lanes_V) * PACK_SIZE_V;
  const uint32_t v_sub = v_column / SUB_K;

  int8_t *Q_lane_ptr =
      Q + batch_id * stride_bz_q + head_id * stride_h_q +
      (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / line_lanes_QK) *
          stride_seq_q +
      (lane_id % line_lanes_QK) * PACK_SIZE_QK;
  int8_t *const K_lane_origin =
      K + batch_id * stride_bz_k + kv_head * stride_h_k +
      static_cast<int64_t>(k_row_in_sub) * stride_seq_k +
      (lane_id % line_lanes_QK) * PACK_SIZE_QK;
  int8_t *const V_lane_origin =
      V + batch_id * stride_bz_v + kv_head * stride_h_v +
      (head_dim / num_warps * warp_id + lane_id / line_lanes_V) * stride_d_v +
      v_column % SUB_K;

  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(
      warp_id * lines_per_warp_QK * Q_smem_iters_col + lane_id / line_lanes_QK,
      lane_id % line_lanes_QK);
  const uint32_t K_smem_offset_load = smem_K.get_permuted_offset(
      warp_id * lines_per_warp_QK * K_smem_iters_col + lane_id / line_lanes_QK,
      lane_id % line_lanes_QK);
  const uint32_t V_smem_offset_load = smem_V.get_permuted_offset(
      warp_id * lines_per_warp_V * V_smem_iters_col + lane_id / line_lanes_V,
      lane_id % line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16,
      lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(
      lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t Q_load_idx_lane_base =
      bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / line_lanes_QK;

  const uint32_t route_idx =
      (batch_id * num_qo_heads + head_id) * gridDim.x + bx;
  const int32_t *__restrict__ lut_row =
      DENSE ? nullptr : BlockLut + static_cast<int64_t>(route_idx) * lut_stride;
  const uint32_t num_entries = DENSE ? num_sub_blocks
      : static_cast<uint32_t>(ValidBlockNum[route_idx]);

  if (num_entries == 0) {
    const uint32_t tid = warp_id * WARP_SIZE + lane_id;
    constexpr uint32_t cta_threads = num_warps * WARP_SIZE;
    for (uint32_t idx = tid; idx < CTA_Q * head_dim; idx += cta_threads) {
      const uint32_t q_idx = bx * CTA_Q + idx / head_dim;
      if (q_idx < qo_len) {
        O[batch_id * stride_bz_o + head_id * stride_h_o +
          static_cast<int64_t>(q_idx) * stride_seq_o + (idx % head_dim)] =
            DTypeOut(0.0f);
      }
    }
    if constexpr (return_lse) {
      for (uint32_t q_idx = tid; q_idx < CTA_Q; q_idx += cta_threads) {
        const uint32_t row = bx * CTA_Q + q_idx;
        if (row < qo_len) {
          Lse[batch_id * (qo_len * num_qo_heads) + head_id * qo_len + row] =
              -CUDART_INF_F;
        }
      }
    }
    return;
  }

  const uint32_t num_iterations = div_ceil(num_entries, N_SUB);
  const uint32_t last_valid_subs = num_entries - (num_iterations - 1) * N_SUB;

  // Delta route: entry 0 is absolute, later entries are positive steps. Past
  // the final entry a subtile repeats the previous block so every address
  // stays in bounds; the final iteration masks those subtiles out.
  uint32_t cursor = 0;
  uint32_t running_block = 0;
  auto decode = [&](uint32_t (&blocks)[N_SUB]) {
#pragma unroll
    for (uint32_t s = 0; s < N_SUB; s++) {
      if constexpr (DENSE) {
        running_block = min(cursor + s, num_entries - 1);
      } else {
        if (cursor + s < num_entries) {
          running_block += static_cast<uint32_t>(lut_row[cursor + s]);
        }
      }
      blocks[s] = running_block;
    }
    cursor += N_SUB;
  };
  auto select_block = [&](const uint32_t (&blocks)[N_SUB],
                          uint32_t sub) -> uint32_t {
    uint32_t block = blocks[0];
#pragma unroll
    for (uint32_t s = 1; s < N_SUB; s++) {
      block = sub == s ? blocks[s] : block;
    }
    return block;
  };
  auto load_K = [&](const uint32_t (&blocks)[N_SUB]) {
    const uint32_t block = select_block(blocks, k_sub);
    int8_t *ptr = K_lane_origin + static_cast<int64_t>(block) * SUB_K * stride_seq_k;
    uint32_t offset = K_smem_offset_load;
    load_global_to_share<line_lanes_QK, lines_per_warp_QK, QK_smem_iters_row,
                         K_smem_iters_col, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        &ptr, offset, stride_seq_k, smem_K, block * SUB_K + k_row_in_sub,
        kv_len);
  };
  auto load_V = [&](const uint32_t (&blocks)[N_SUB]) {
    const uint32_t block = select_block(blocks, v_sub);
    int8_t *ptr = V_lane_origin + static_cast<int64_t>(block) * SUB_K;
    uint32_t offset = V_smem_offset_load;
    load_int8_V_global_to_share<line_lanes_V, lines_per_warp_V,
                                V_smem_iters_row, V_smem_iters_col,
                                swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V,
                                CTA_K>(&ptr, offset, stride_d_v, smem_V);
  };

  uint32_t cur[N_SUB];
  uint32_t nxt[N_SUB];
  decode(cur);

  load_global_to_share<line_lanes_QK, lines_per_warp_QK, QK_smem_iters_row,
                       Q_smem_iters_col, swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
      &Q_lane_ptr, Q_smem_offset_load, stride_seq_q, smem_Q,
      Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  load_K(cur);
  cp_async::commit_group();

  const float qk_scale = sm_scale * Q_scale[q_scale_idx];
  float scale[N_SUB];
  auto set_scales = [&](const uint32_t (&blocks)[N_SUB]) {
#pragma unroll
    for (uint32_t s = 0; s < N_SUB; s++) {
      scale[s] = qk_scale * k_scale_lane[blocks[s] * 4];
    }
  };
  set_scales(cur);

  load_V(cur);
  cp_async::commit_group();

  for (uint32_t iter = 0; iter + 1 < num_iterations; iter++) {
    cp_async::wait_group<1>();
    __syncthreads();
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                   num_tiles_qk_inner, swizzle_mode_QK,
                   QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    __syncthreads();

    decode(nxt);
    load_K(nxt);
    cp_async::commit_group();

    uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
    update_mdo_i32_u8_packed<num_tiles_q, num_tiles_k, num_tiles_v,
                             TILES_PER_SUB>(RS, RO, m, d, scale, S_U8_OFFSET,
                                            RS_u8);
    set_scales(nxt);

    cp_async::wait_group<1>();
    __syncthreads();
    compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                    num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, block_v_scale>(
        smem_V, RS, RS_u8, RO, v_scale_for_block(cur[0]));
    __syncthreads();

    load_V(nxt);
    cp_async::commit_group();
#pragma unroll
    for (uint32_t s = 0; s < N_SUB; s++) {
      cur[s] = nxt[s];
    }
  }

  // Final iteration: float path with the ragged-sequence and missing-subtile
  // masks. Scores are pre-scaled because the dequant scale varies by subtile.
  {
    cp_async::wait_group<1>();
    __syncthreads();
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                   num_tiles_qk_inner, swizzle_mode_QK,
                   QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

    float RS_soft[num_tiles_q][num_tiles_k][8];
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      const uint32_t s = fk / TILES_PER_SUB;
      const bool sub_valid = s < last_valid_subs;
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t column =
            fk * 16 + 8 * (k / 4) + (k % 2) + 2 * (lane_id % 4);
        const uint32_t kv_idx = cur[s] * SUB_K + column % SUB_K;
        RS_soft[0][fk][k] = (sub_valid && kv_idx < kv_len)
                                ? __int2float_rz(RS[0][fk][k]) * scale[s]
                                : -50000.0f;
      }
    }

    float pv_scale[num_tiles_q][2];
    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, true>(
        RS_soft, RO, m, d, pv_scale, 1.0f, S_U8_OFFSET);
    uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
    RS_to_u8<num_tiles_q, num_tiles_k>(RS_soft, RS_u8);
    accumulate_d<num_tiles_q, num_tiles_k>(RS_soft, d, pv_scale);
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      RS[0][0][k] = __float_as_int(pv_scale[0][k]);
    }

    cp_async::wait_group<0>();
    __syncthreads();
    compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                    num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, block_v_scale>(
        smem_V, RS, RS_u8, RO, v_scale_for_block(cur[0]));
    __syncthreads();
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  if constexpr (!block_v_scale) {
    float v_scale[4];
    const float *V_scale_base_ptr =
        V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim +
        kv_head * head_dim + (lane_id % 4) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      ((float2 *)v_scale)[0] = *((const float2 *)(V_scale_base_ptr + fv * 16));
      ((float2 *)v_scale)[1] =
          *((const float2 *)(V_scale_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        RO[0][fv][k] *= v_scale[(k / 4) * 2 + (k % 2)];
      }
    }
  }

  const uint32_t smem_O_row_base =
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
    uint32_t offset_O = smem_O.get_permuted_offset(
        smem_O_row_base, fv * (MMA_SV_N / PACK_SIZE_O));
    uint32_t RO_f16[4];
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      if constexpr (std::is_same<DTypeOut, half>::value) {
        ((half2 *)RO_f16)[k] = __float22half2_rn(((float2 *)RO[0][fv])[k]);
      } else {
        ((nv_bfloat162 *)RO_f16)[k] =
            __float22bfloat162_rn(((float2 *)RO[0][fv])[k]);
      }
    }
    ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
    ((uint32_t *)(smem_O.base + offset_O +
                  8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];
    offset_O = smem_O.get_permuted_offset(
        smem_O_row_base, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
    ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
    ((uint32_t *)(smem_O.base + offset_O +
                  8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
  }
  __syncwarp();

  DTypeOut *O_lane_ptr =
      O + batch_id * stride_bz_o + head_id * stride_h_o +
      (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() +
       lane_id / line_lanes_O) *
          stride_seq_o +
      lane_id % line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
          lane_id / line_lanes_O,
      lane_id % line_lanes_O);
  uint32_t O_load_idx_lane_base =
      bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / line_lanes_O;
#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++) {
      if (O_load_idx_lane_base < qo_len) {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<line_lanes_O>(offset_O);
    }
    offset_O = smem_O.advance_offset_by_row<lines_per_warp_O>(
        offset_O - (O_smem_iters_row * line_lanes_O));
    O_lane_ptr += ((lines_per_warp_O * stride_seq_o) -
                   (O_smem_iters_row * line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += lines_per_warp_O;
  }

  if constexpr (return_lse) {
    const uint32_t warp_q = WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    if ((lane_id % 4) < 2) {
      const uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 2) + warp_q;
      if (lse_idx < qo_len) {
        Lse[batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx] =
            math::ptx_log2(d[0][lane_id % 2]) + m[0][lane_id % 2];
      }
    }
  }
}
