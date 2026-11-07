// SPDX-License-Identifier: Apache-2.0
//
// Align sparse routes to chunk-scaled V blocks.
//
// The packed attention kernels consume `group` routed tiles per iteration
// (four 32-key subtiles at 64Q x 32KV, two 64-key subtiles at 64Q x 64KV) and
// sum them in one INT32 product, so with chunk-scaled V no iteration may mix V
// scale blocks. This pass makes every block's run in a route row, except the
// row's last run, a multiple of `group` tiles by adding unselected tiles of
// the same block: first directly after the run, then directly before it, then
// in gaps inside it. Adding keys only moves sparse attention towards dense.
//
// One thread per route row streams the row once to find each run and once to
// write it, so the output stays ascending without sorting. Routes may be read
// and written absolute or delta-encoded (entry 0 absolute, then steps).

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

struct RowWriter {
  int32_t *out;
  int width;
  bool delta;
  int written = 0;
  int previous = 0;

  __device__ void emit(int tile) {
    if (written < width) {
      out[written] = (delta && written > 0) ? tile - previous : tile;
    }
    previous = tile;
    written++;
  }
};

__global__ void align_route_v_blocks_kernel(
    const int32_t *__restrict__ indices, const int32_t *__restrict__ counts,
    int32_t *__restrict__ out_indices, int32_t *__restrict__ out_counts,
    int32_t *__restrict__ out_changes, int rows, int width_in, int width_out,
    int change_width, int tiles_per_block, int group, int total_tiles,
    int in_delta, int out_delta) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t *src = indices + static_cast<int64_t>(row) * width_in;
  RowWriter writer{out_indices + static_cast<int64_t>(row) * width_out,
                   width_out, out_delta != 0};
  int32_t *changes = out_changes + static_cast<int64_t>(row) * change_width;
  int change = 0;
  const int count = counts[row];
  int i = 0;
  int previous = 0;  // absolute tile before entry i (delta decoding)
  while (i < count) {
    const int first = in_delta ? (i == 0 ? src[0] : previous + src[i]) : src[i];
    const int block = first / tiles_per_block;
    int last = first;
    int j = i + 1;
    while (j < count) {
      const int tile = in_delta ? last + src[j] : src[j];
      if (tile / tiles_per_block != block) {
        break;
      }
      last = tile;
      j++;
    }
    const int length = j - i;
    // The row's last run may end in a partial iteration: the kernels mask it.
    int need = j == count ? 0 : (group - length % group) % group;
    const int block_start = block * tiles_per_block;
    const int block_end = min(block_start + tiles_per_block, total_tiles);
    const int above = min(need, block_end - 1 - last);
    need -= above;
    const int below = min(need, first - block_start);
    need -= below;  // what remains comes from gaps inside the run

    // Every earlier run ends on a whole iteration, so this run starts one:
    // record its iteration for the kernel's block change list.
    if (i > 0 && change < change_width - 1) {
      changes[change++] = writer.written / group;
    }
    for (int tile = first - below; tile < first; tile++) {
      writer.emit(tile);
    }
    writer.emit(first);
    int tile = first;
    for (int k = i + 1; k < j; k++) {
      const int next = in_delta ? tile + src[k] : src[k];
      for (int gap = tile + 1; gap < next && need > 0; gap++, need--) {
        writer.emit(gap);
      }
      writer.emit(next);
      tile = next;
    }
    for (int extra = last + 1; extra <= last + above; extra++) {
      writer.emit(extra);
    }
    previous = last;
    i = j;
  }
  out_counts[row] = writer.written;
  for (; change < change_width; change++) {
    changes[change] = 0x7fffffff;  // sentinel, and deterministic padding
  }
}

}  // namespace

void launch_h3_align_route_v_blocks(const void *indices, const void *counts,
                                    void *out_indices, void *out_counts,
                                    void *out_changes, int rows, int width_in,
                                    int width_out, int change_width,
                                    int tiles_per_block, int group,
                                    int total_tiles, int in_delta,
                                    int out_delta, cudaStream_t stream) {
  if (group < 1 || tiles_per_block < group || tiles_per_block % group != 0) {
    throw std::runtime_error(
        "route alignment needs V blocks of whole packed iterations");
  }
  if (rows <= 0) {
    return;
  }
  constexpr int threads = 128;
  align_route_v_blocks_kernel<<<(rows + threads - 1) / threads, threads, 0,
                                stream>>>(
      static_cast<const int32_t *>(indices),
      static_cast<const int32_t *>(counts),
      static_cast<int32_t *>(out_indices), static_cast<int32_t *>(out_counts),
      static_cast<int32_t *>(out_changes), rows, width_in, width_out,
      change_width, tiles_per_block, group, total_tiles, in_delta, out_delta);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("route alignment launch failed: ") +
                             cudaGetErrorString(error));
  }
}
