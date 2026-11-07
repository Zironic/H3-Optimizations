// SPDX-License-Identifier: Apache-2.0
//
// Packed fine-route traversal. A 64Q x 128KV CTA consumes 128 / SUB_K routed
// SUB_K-token subtiles per iteration from the ordinary 64-token carrier; the
// route counts SUB_K-token subtiles. Production builds SUB_K = 32, selected
// by a 32-token route (cta_k 32) over the 64-token carrier.
//
// No shared-memory route or scale tables and no extra barrier:
//   - every thread decodes the iteration's route deltas in registers; the
//     deltas are loaded one iteration ahead, so the decode that precedes the
//     next K copy never waits on global memory;
//   - a warp copies 32 contiguous K rows (whole subtiles) and a lane one
//     16-byte V column chunk (inside one subtile);
//   - K dequant scales stay per 64-row carrier block and lane group; the
//     softmax applies one scale per subtile.
// Each CTA first measures its route's leading run 0,1,2,... (a dense row is
// all of it; a sparse row's context prefix is found with one warp scan).
// Iterations inside that run take their blocks arithmetically and load two K
// scales, so contiguous work runs at packed-64 cost. At SUB_K = 32 every
// routed piece is whole 32-byte V sectors.
// The ragged-sequence mask is a per-iteration bit mask of subtiles that are
// the final partial carrier subtile.

#pragma once

#include "qk_int_sv_i8_sparse_cuda.cuh"
#include "chunked_v_scale.cuh"

namespace h3_packed_fine {

__device__ __forceinline__ PackedU8RowSum
pack_scaled_exp2_u8x4_2(const int32_t a, const int32_t b, const int32_t c,
                        const int32_t d, const float scale_ab,
                        const float scale_cd, const float negative_m) {
  const float pa = math::ptx_exp2(fmaf(__int2float_rz(a), scale_ab, negative_m));
  const float pb = math::ptx_exp2(fmaf(__int2float_rz(b), scale_ab, negative_m));
  const float pc = math::ptx_exp2(fmaf(__int2float_rz(c), scale_cd, negative_m));
  const float pd = math::ptx_exp2(fmaf(__int2float_rz(d), scale_cd, negative_m));
  const uint32_t probabilities = pack_u8x4(pa, pb, pc, pd);
  const uint32_t denominator = __dp4a(probabilities, 0x01010101u, 0u);
  return {probabilities, __uint2float_rn(denominator)};
}

// psc[j] is the dequant scale of the thread's column pair in 8-column group j
// (columns 8j + 2*(lane%4) + {0,1}). Pairs sharing a subtile share a scale, so
// the row maximum is taken in integers over each group of PPS pairs first.
template <uint32_t num_tiles_k, uint32_t num_tiles_v, uint32_t PPS>
__device__ __forceinline__ void update_mdo_packed2(
    int32_t RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2],
    float d[][2], const float (&psc)[2 * num_tiles_k],
    const float exp_offset_value, uint32_t RS_u8[][num_tiles_k / 2][4]) {
  constexpr uint32_t pairs = 2 * num_tiles_k;
#pragma unroll
  for (uint32_t k = 0; k < 2; k++) {
    const float m_prev = m[0][k];
    float m_temp = -CUDART_INF_F;
#pragma unroll
    for (uint32_t g = 0; g < pairs / PPS; g++) {
      int32_t mx = INT_MIN;
#pragma unroll
      for (uint32_t p = 0; p < PPS; p++) {
        const uint32_t j = g * PPS + p;
        const uint32_t fk = j / 2;
        const uint32_t e = k * 2 + 4 * (j % 2);
        mx = max(mx, max(RS[0][fk][e], RS[0][fk][e + 1]));
      }
      m_temp = max(m_temp, fmaf(__int2float_rz(mx), psc[g * PPS], -exp_offset_value));
    }
    m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
    m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
    const float tile_m = m_temp;
    m[0][k] = max(m_prev, tile_m);

    const float smaller_scale = math::ptx_exp2(-fabsf(m_prev - tile_m));
    const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
    const float tile_scale = tile_m < m_prev ? smaller_scale : 1.0f;
    d[0][k] *= o_scale;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      RO[0][fv][k * 2] *= o_scale;
      RO[0][fv][k * 2 + 1] *= o_scale;
      RO[0][fv][k * 2 + 4] *= o_scale;
      RO[0][fv][k * 2 + 5] *= o_scale;
    }

    const float negative_m = -tile_m;
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      const PackedU8RowSum p0 = pack_scaled_exp2_u8x4_2(
          RS[0][fk * 2][k * 2], RS[0][fk * 2][k * 2 + 1],
          RS[0][fk * 2][k * 2 + 4], RS[0][fk * 2][k * 2 + 5],
          psc[4 * fk], psc[4 * fk + 1], negative_m);
      const PackedU8RowSum p1 = pack_scaled_exp2_u8x4_2(
          RS[0][fk * 2 + 1][k * 2], RS[0][fk * 2 + 1][k * 2 + 1],
          RS[0][fk * 2 + 1][k * 2 + 4], RS[0][fk * 2 + 1][k * 2 + 5],
          psc[4 * fk + 2], psc[4 * fk + 3], negative_m);
      RS_u8[0][fk][k] = p0.probabilities;
      RS_u8[0][fk][k + 2] = p1.probabilities;
      d[0][k] += p0.denominator * tile_scale;
      d[0][k] += p1.denominator * tile_scale;
    }
    RS[0][0][k] = __float_as_int(tile_scale);
  }
}

} // namespace h3_packed_fine

template <uint32_t SUB_K, typename DTypeOut, bool return_lse, uint32_t MIN_BLOCKS,
          bool chunked_v = false>
__global__ __launch_bounds__(128, MIN_BLOCKS) void qk_int_sv_i8_packed_fine_attn_kernel(
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
    const uint32_t v_scale_blocks = 1, const uint32_t v_scale_shift = 0,
    const int32_t *__restrict__ VBlockChanges = nullptr,
    const uint32_t v_change_stride = 0) {
  static_assert(SUB_K == 16 || SUB_K == 32 || SUB_K == 64,
                "SUB_K must be 16, 32 or 64");
  static_assert(!chunked_v || SUB_K >= 32,
                "chunk-scaled V needs whole subtiles per 32-key P x V group");
  constexpr uint32_t CARRIER_K = 64;
  constexpr uint32_t CTA_Q = 64;
  constexpr uint32_t CTA_K = 128;
  constexpr uint32_t WARP_Q = 16;
  constexpr uint32_t head_dim = 128;
  constexpr uint32_t N_SUB = CTA_K / SUB_K;
  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = 1;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = CTA_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = head_dim / MMA_QK_K;
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;
  static_assert(num_tiles_q == 1);
  constexpr uint32_t TILES_PER_SUB = SUB_K / MMA_QK_N;
  static_assert(num_tiles_k == N_SUB * TILES_PER_SUB);
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

  sm_scale *= math::log2e;

  int32_t RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];

  const uint32_t q_scale_idx =
      batch_id * stride_bz_q_scale + head_id * stride_h_q_scale +
      bx * ((CTA_Q / 32) * 8) +
      (get_warp_idx_q<num_warps_q, num_warps_k>() / 2) * 8 + lane_id / 4;
  const uint32_t num_carrier_blocks = div_ceil(kv_len, CARRIER_K);
  const float *const k_scale_lane =
      K_scale + (batch_id * (num_qo_heads / num_kv_groups) + kv_head) *
                    (num_carrier_blocks * 4) +
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
  static_assert(QK_smem_iters_row == 1);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * lines_per_warp_QK);
  constexpr uint32_t K_rows_per_warp = K_smem_iters_col * lines_per_warp_QK;
  static_assert(K_rows_per_warp % SUB_K == 0 || SUB_K % K_rows_per_warp == 0,
                "a warp's K rows are whole subtiles or part of one");
  constexpr uint32_t K_iters_per_sub = SUB_K / lines_per_warp_QK;
  constexpr uint32_t V_smem_iters_row =
      V_SMEM_STRIDE / (line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col = head_dim / (num_warps * lines_per_warp_V);
  static_assert(line_lanes_V * PACK_SIZE_V == CTA_K && SUB_K % PACK_SIZE_V == 0,
                "a lane's V chunk lies inside one subtile");
  constexpr uint32_t O_smem_iters_row =
      O_SMEM_STRIDE / (line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * lines_per_warp_O);

  const uint32_t k_sub0 = (K_rows_per_warp * warp_id) / SUB_K;
  // Nonzero only at SUB_K = 64, where a warp's 32 rows are half a subtile.
  const uint32_t k_row_base = (K_rows_per_warp * warp_id) % SUB_K;
  const uint32_t k_row = lane_id / line_lanes_QK;
  const uint32_t v_column = (lane_id % line_lanes_V) * PACK_SIZE_V;
  const uint32_t v_sub = v_column / SUB_K;

  int8_t *Q_lane_ptr =
      Q + batch_id * stride_bz_q + head_id * stride_h_q +
      (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / line_lanes_QK) *
          stride_seq_q +
      (lane_id % line_lanes_QK) * PACK_SIZE_QK;
  const int8_t *const K_lane_origin =
      K + batch_id * stride_bz_k + kv_head * stride_h_k +
      static_cast<int64_t>(k_row) * stride_seq_k +
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
      BlockLut + static_cast<int64_t>(route_idx) * lut_stride;
  const uint32_t num_entries = static_cast<uint32_t>(ValidBlockNum[route_idx]);

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
  // The final carrier subtile may be partial: tokens [16 * tail_block, kv_len).
  const uint32_t tail_block = (kv_len - 1) / SUB_K;
  const uint32_t tail_valid = kv_len - tail_block * SUB_K;

  // Length of the route's leading run 0,1,2,... in subtiles. A route holding
  // every subtile is that run; otherwise each warp scans it, four entries per
  // lane per step, and stops at the first entry that breaks it.
  uint32_t coarse_iters = 0;
  {
    uint32_t prefix = num_entries;
    if (num_entries != div_ceil(kv_len, SUB_K)) {
      for (uint32_t base = 0; base < num_entries; base += 4 * WARP_SIZE) {
        uint32_t first_bad = 4;
#pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
          const uint32_t idx = base + lane_id * 4 + j;
          if (idx < num_entries && first_bad == 4 &&
              lut_row[idx] != (idx == 0 ? 0 : 1)) {
            first_bad = j;
          }
        }
        const uint32_t bad = __ballot_sync(0xffffffff, first_bad < 4);
        if (bad != 0) {
          const uint32_t lane = __ffs(bad) - 1;
          prefix = base + lane * 4 + __shfl_sync(0xffffffff, first_bad, lane);
          break;
        }
      }
    }
    coarse_iters = prefix / N_SUB;
  }

  // Delta route: entry 0 is absolute, later entries are positive steps. Past
  // the final entry a subtile repeats the previous block so every address
  // stays in bounds; the final iteration masks those subtiles out.
  uint32_t cursor = 0;
  uint32_t running_block = 0;
  uint32_t blocks[N_SUB];
  // Chunk-scaled V (chunked_v): the block the P x V accumulators are
  // expressed in, and this CTA's [blocks, head_dim] scales on demand.
  uint32_t v_block = 0;
  auto v_scale_head = [&]() -> const float * {
    return V_scale +
           static_cast<int64_t>(batch_id * (num_qo_heads / num_kv_groups) +
                                kv_head) *
               v_scale_blocks * head_dim;
  };
  // Routes must be block-aligned: within each route row, every V scale
  // block except the last contributes a multiple of N_SUB entries, so an
  // iteration never mixes blocks, and the router lists where blocks change.
  uint32_t decoded_iters = 0;
  bool blocks_coarse = false;
  int32_t pending[N_SUB];
  auto fetch = [&](uint32_t at) {
#pragma unroll
    for (uint32_t s = 0; s < N_SUB; s++) {
      pending[s] = at + s < num_entries ? lut_row[at + s] : 0;
    }
  };
  if (coarse_iters == 0) {
    fetch(0);
  }
  auto decode = [&]() {
    blocks_coarse = decoded_iters < coarse_iters;
    if (blocks_coarse) {
#pragma unroll
      for (uint32_t s = 0; s < N_SUB; s++) {
        blocks[s] = cursor + s;
      }
      running_block = cursor + N_SUB - 1;
    } else {
#pragma unroll
      for (uint32_t s = 0; s < N_SUB; s++) {
        running_block += static_cast<uint32_t>(pending[s]);
        blocks[s] = running_block;
      }
    }
    cursor += N_SUB;
    decoded_iters++;
    // Deltas for the next decode, unless it is still inside the prefix.
    if (decoded_iters >= coarse_iters && cursor < num_entries) {
      fetch(cursor);
    }
  };
  auto tail_bits = [&]() -> uint32_t {
    uint32_t bits = 0;
#pragma unroll
    for (uint32_t s = 0; s < N_SUB; s++) {
      bits |= static_cast<uint32_t>(blocks[s] == tail_block) << s;
    }
    return bits;
  };
  auto select_block = [&](uint32_t sub) -> uint32_t {
    uint32_t block = blocks[0];
#pragma unroll
    for (uint32_t s = 1; s < N_SUB; s++) {
      block = sub == s ? blocks[s] : block;
    }
    return block;
  };
  auto load_K = [&]() {
    const uint32_t block0 = select_block(k_sub0);
    uint32_t block1 = block0;
    if constexpr (K_rows_per_warp > SUB_K) {
      block1 = select_block(k_sub0 + 1);
    }
    uint32_t offset = K_smem_offset_load;
#pragma unroll
    for (uint32_t i = 0; i < K_smem_iters_col; i++) {
      const uint32_t block = i < K_iters_per_sub ? block0 : block1;
      const uint32_t row = block * SUB_K + k_row_base + k_row +
                           (i % K_iters_per_sub) * lines_per_warp_QK;
      smem_K.load_128b_async<cp_async::SharedMemFillMode::kNoFill>(
          offset, K_lane_origin + static_cast<int64_t>(row - k_row) * stride_seq_k,
          row < kv_len);
      offset = smem_K.advance_offset_by_row<lines_per_warp_QK>(offset);
    }
  };
  auto load_V = [&]() {
    int8_t *ptr = V_lane_origin + static_cast<int64_t>(select_block(v_sub)) * SUB_K;
    uint32_t offset = V_smem_offset_load;
    load_int8_V_global_to_share<line_lanes_V, lines_per_warp_V,
                                V_smem_iters_row, V_smem_iters_col,
                                swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V,
                                CTA_K>(&ptr, offset, stride_d_v, smem_V);
  };
  // Raw per-thread K scales; the Q scale is applied where they are used so the
  // loads have the P x V and the next QK to land.
  float k_scale[N_SUB];
  auto load_scales = [&]() {
    if (blocks_coarse) {
      // Coarse iterations are two whole, aligned carrier blocks.
      const float first = k_scale_lane[(blocks[0] * SUB_K / CARRIER_K) * 4];
      const float second =
          k_scale_lane[(blocks[N_SUB / 2] * SUB_K / CARRIER_K) * 4];
#pragma unroll
      for (uint32_t s = 0; s < N_SUB; s++) {
        k_scale[s] = s < N_SUB / 2 ? first : second;
      }
    } else {
#pragma unroll
      for (uint32_t s = 0; s < N_SUB; s++) {
        k_scale[s] = k_scale_lane[(blocks[s] * SUB_K / CARRIER_K) * 4];
      }
    }
  };

  decode();
  uint32_t tail_cur = tail_bits();

  load_global_to_share<line_lanes_QK, lines_per_warp_QK, QK_smem_iters_row,
                       Q_smem_iters_col, swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
      &Q_lane_ptr, Q_smem_offset_load, stride_seq_q, smem_Q,
      Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  load_K();
  cp_async::commit_group();

  const float qk_scale = sm_scale * Q_scale[q_scale_idx];
  load_scales();

  load_V();
  cp_async::commit_group();

  // One integer-softmax iteration over the routed subtiles in shared memory.
  auto iteration = [&]() {
      cp_async::wait_group<1>();
      __syncthreads();
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DataType::kInt8>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
      __syncthreads();

      decode();
      const uint32_t tail_next = tail_bits();
      load_K();
      cp_async::commit_group();

      {
        float psc[2 * num_tiles_k];
#pragma unroll
        for (uint32_t j = 0; j < 2 * num_tiles_k; j++) {
          psc[j] = qk_scale * k_scale[j / (SUB_K / 8)];
        }
        uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
        h3_packed_fine::update_mdo_packed2<num_tiles_k, num_tiles_v, SUB_K / 8>(
            RS, RO, m, d, psc, S_U8_OFFSET, RS_u8);
        load_scales();

        cp_async::wait_group<1>();
        __syncthreads();
        compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                        num_tiles_v, swizzle_mode_V,
                        V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS, RS_u8, RO);
      }
      __syncthreads();

      load_V();
      cp_async::commit_group();
      tail_cur = tail_next;
  };

  if constexpr (chunked_v) {
    // Aligned routes (int8_attention.align_route_to_v_blocks) list, per route
    // row, the iterations where the V scale block changes, ending with a
    // sentinel. The block changes only there, so the hot loop is the global
    // kernel's: the rescale runs between inner loops, never beside P x V.
    const int32_t *changes =
        VBlockChanges + static_cast<int64_t>(route_idx) * v_change_stride;
    v_block = (blocks[0] * SUB_K) >> v_scale_shift;  // accumulators still zero
    uint32_t iter = 0;
    uint32_t next = static_cast<uint32_t>(changes[0]);
    while (true) {
      const uint32_t stop = min(next, num_iterations - 1);
      for (; iter < stop; iter++) {
        iteration();
      }
      if (iter != next) {
        break;  // the final iteration is next and stays in this block
      }
      chunked_v_enter<num_tiles_q, num_tiles_v, head_dim>(
          RO, v_scale_head(), v_block, (blocks[0] * SUB_K) >> v_scale_shift);
      next = static_cast<uint32_t>(*++changes);
    }
  } else {
    for (uint32_t iter = 0; iter + 1 < num_iterations; iter++) {
      iteration();
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
      const uint32_t sub = fk / TILES_PER_SUB;
      const bool sub_valid = sub < last_valid_subs;
      const bool in_tail = (tail_cur >> sub) & 1u;
      const float scale = qk_scale * k_scale[sub];
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t column_in_sub = (fk % TILES_PER_SUB) * MMA_QK_N +
                                       8 * (k / 4) + (k % 2) + 2 * (lane_id % 4);
        const bool valid = sub_valid && (!in_tail || column_in_sub < tail_valid);
        RS_soft[0][fk][k] =
            valid ? __int2float_rz(RS[0][fk][k]) * scale : -50000.0f;
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
                    num_tiles_v, swizzle_mode_V,
                    V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS, RS_u8, RO);
    __syncthreads();
  }

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  {
    float v_scale[4];
    const float *V_scale_base_ptr =
        chunked_v
            ? chunked_v_epilogue<head_dim>(v_scale_head(), v_block)
            : V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim +
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
