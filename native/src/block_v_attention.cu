#include "sage_attention/qk_int_sv_i8_cuda.cuh"
#include "sage_attention/qk_int_sv_i8_packed_cuda.cuh"
#include <cstdio>

namespace {
thread_local char last_error[512];

template<int CK, typename Out>
cudaError_t launch(
    const void *q, const void *k, const void *v, void *o,
    const void *qs, const void *ks, const void *vs,
    int B, int Lq, int Lk, int Hq, int Hk,
    int qb, int qn, int qh, int kb, int kn, int kh,
    int vb, int vh, int vd, int ob, int on, int oh,
    float scale, int scale_rows, cudaStream_t stream) {
  constexpr int shared = 128 * 128 + 2 * CK * 128;
  auto kernel = qk_int_sv_i8_attn_kernel<
      128, CK, 16, CK, 128, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread,
      float, false, Out, ComputeUnit::kCudaCore, MaskMode::kNone,
      false, true, false, false, true, true>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<dim3(div_ceil(Lq, 128), Hq, B), dim3(32, 8), shared, stream>>>(
      (int8_t *)q, (int8_t *)k, (int8_t *)v, (Out *)o, nullptr,
      (float *)qs, (float *)ks, (float *)vs, nullptr,
      nullptr, 0, 0, 0, 0, -1, Lq, Lk, Hq / Hk,
      qb, qn, qh, kb, kn, kh, vb, vh, vd, ob, on, oh, scale, scale_rows);
  return cudaGetLastError();
}

template<int CK, typename Out>
cudaError_t launch_packed(
    const void *q, const void *k, const void *v, void *o,
    const void *qs, const void *ks, const void *vs,
    int B, int Lq, int Lk, int Hq, int Hk,
    int qb, int qn, int qh, int kb, int kn, int kh,
    int vb, int vh, int vd, int ob, int on, int oh,
    float scale, int scale_rows, int qsb, int qsh, cudaStream_t stream) {
  constexpr int shared = 64 * 128 + 2 * 128 * 128;
  auto kernel = qk_int_sv_i8_packed_attn_kernel<CK, Out, false, 1, true, true>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<dim3(div_ceil(Lq, 64), Hq, B), dim3(32, 4), shared, stream>>>(
      (int8_t *)q, (int8_t *)k, (int8_t *)v, (Out *)o, nullptr,
      (float *)qs, (float *)ks, (float *)vs, Lq, Lk, Hq / Hk,
      qb, qn, qh, kb, kn, kh, vb, vh, vd, ob, on, oh,
      qsb, qsh, scale,
      nullptr, nullptr, 0, scale_rows);
  return cudaGetLastError();
}
}

#ifdef _WIN32
#define H3_EXPORT extern "C" __declspec(dllexport)
#else
#define H3_EXPORT extern "C" __attribute__((visibility("default")))
#endif

H3_EXPORT const char *h3_block_v_last_error() { return last_error; }

H3_EXPORT int h3_block_v_attention_v3(
    const void *q, const void *k, const void *v, void *o,
    const void *qs, const void *ks, const void *vs,
    int cta_k, int B, int Lq, int Lk, int Hq, int Hk, int D,
    int qb, int qn, int qh, int kb, int kn, int kh,
    int vb, int vh, int vd, int ob, int on, int oh,
    int qsb, int qsh, float scale, int dtype, int scale_rows, int packed, size_t stream) {
  if (!q || !k || !v || !o || !qs || !ks || !vs || D != 128 ||
      B < 1 || Lq < 1 || Lk < 1 || Hk < 1 || Hq < 1 || Hq % Hk ||
      (cta_k != 64 && cta_k != 128) || scale_rows < 1 || scale_rows % cta_k ||
      (dtype != 1 && dtype != 2) || (packed != 0 && packed != 1) ||
      (packed && scale_rows % 128)) {
    std::snprintf(last_error, sizeof(last_error), "invalid block-V attention geometry or dtype");
    return 1;
  }
  cudaError_t status;
#define RUN(CK, OUT) launch<CK, OUT>(q, k, v, o, qs, ks, vs, B, Lq, Lk, Hq, Hk, qb, qn, qh, kb, kn, kh, vb, vh, vd, ob, on, oh, scale, scale_rows, (cudaStream_t)stream)
#define RUN_PACKED(CK, OUT) launch_packed<CK, OUT>(q, k, v, o, qs, ks, vs, B, Lq, Lk, Hq, Hk, qb, qn, qh, kb, kn, kh, vb, vh, vd, ob, on, oh, scale, scale_rows, qsb, qsh, (cudaStream_t)stream)
  if (packed) {
    if (dtype == 2) {
      status = cta_k == 128 ? RUN_PACKED(128, nv_bfloat16) : RUN_PACKED(64, nv_bfloat16);
    } else {
      status = cta_k == 128 ? RUN_PACKED(128, half) : RUN_PACKED(64, half);
    }
  } else if (dtype == 2) {
    status = cta_k == 128 ? RUN(128, nv_bfloat16) : RUN(64, nv_bfloat16);
  } else {
    status = cta_k == 128 ? RUN(128, half) : RUN(64, half);
  }
#undef RUN
#undef RUN_PACKED
  if (status != cudaSuccess) {
    std::snprintf(last_error, sizeof(last_error), "%s", cudaGetErrorString(status));
    return 1;
  }
  return 0;
}
