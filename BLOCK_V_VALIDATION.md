# Experimental single-pass V validation

Local build: Windows x64, RTX 4080, CUDA 13.2, torch 2.14.0+cu132.
`native/bin/h3_block_v_v3.dll` SHA256:
`6095F74075F6644F7F854592B55CC0F2984B4F59BE479350A0BC43A79EF1FADC`.

The extension reuses the existing INT8 attention kernel with V channel scales
applied to each KV tile's contribution before accumulation. Existing binaries
and default attention modes are unchanged.

Checks passed:

- CPU block packing, independent channel scales, zero V, tail padding,
  invalid chunk coverage, overwrite-safe direct residuals and failure cleanup.
- A producer contract checks one QKV projection per chunk, no V reprojection
  and no full-sequence BF16 V quantization call.
- CUDA attention against SDPA: FP16/BF16, 64/128 KV tiles, lengths
  1/65/129/513, batch 2, grouped-query attention and differing V block scales.
  Tolerance: absolute 0.007, relative 0.035.
- A small real ConvRot INT8 DiT block compares ordinary buffered output with
  direct residual output across modulation/chunk boundaries, with fused norm
  quantization enabled and disabled. Tolerance: absolute 0.001, relative 0.01.
- Existing producer, streamed output, V staging, plan, composition, dense
  backend, status and public node tests pass.

These are correctness checks on synthetic inputs. No production checkpoint
generation, performance benchmark, full-workflow VRAM measurement or video
quality assessment was performed.

The saved hidden-size BF16 output is replaced by packed INT8 Q and V scale
metadata. The actual peak also depends on projection workspace, chunk size,
weight residency and conditioning; this accounting is not a measured VRAM
guarantee. Per-block V quantization differs numerically from global V scaling.

## V2 follow-up

Q slices are consumed with their actual strides instead of copied to a
contiguous tensor. V channel scales are stored once per projection chunk,
rather than repeated for every attention tile. `v_scale_rows` carries that
granularity through the native producer and attention carrier.

CUDA checks establish bitwise equality of Q views versus copies and compact
versus repeated scales. They cover FP16/BF16, grouped-query attention, ragged
tails and 128/256/384/4096-row scale blocks. The 384-row case exercises the
non-power-of-two indexing path. The ordinary small-block checks also pass.

For 167210 tokens, 56 heads, 128-dimensional heads, 4096-row projection chunks
and 128-token KV tiles, scale residency falls from 35.74 MiB to 1.12 MiB.
An additional Q-copy temporary of up to 28 MiB is removed. These reductions
need not coincide with the full-workflow peak. Avoided logical Q-copy reads
and writes total about 112 GiB per 50-block forward; caches determine how much
of that would have reached DRAM.

SM89 resource inspection reports 168 registers/thread at 64 KV and 255 at
128 KV, with no local spills. An experimental fused-tail rewrite did not
reduce those counts and was reverted. No timing claim follows from this
inspection. The pytest-based producer suite could not run because pytest is
absent; a CPU test directly checks native finalization preserves scale
granularity, and the CUDA block test exercises the real producer.

## Attention mass-bound research probe

`benchmarks/probe_attention_bounds.py` is a CPU-only synthetic experiment,
separate from inference. Its center/radius bound successfully limits local
attention output error in all tested cases. At a dropped-mass budget of
0.001, tightly clustered keys use 23.83% of blocks on average, moderate
clusters use 85.25%, and unstructured/wide-cluster keys use 100%.

The bound does not cover quantization error, subsequent-layer amplification
or video quality. No real H3 activations were assessed and this research path
is not enabled in production. It does not establish a speedup.

## V3 packed consumer

Long rows on devices accepted by the existing packed-attention capability
check now use the shared 64Q x 128KV packed kernel. Block-scaled V is supported
for both 64- and 128-token K carriers; the producer keeps its original K
quantization grid. Short rows, unsupported devices and incompatible V-scale
block sizes retain the generic consumer. The packed consumer also reads Q
scale views by stride, avoiding another slab copy.

CUDA checks compare packed and generic consumers with SDPA across FP16/BF16,
batch 2, grouped queries, both K layouts, lengths 1025/1088/1151/1280 and
128/384-row V-scale blocks. Tolerances remain absolute 0.007, relative 0.035;
packed and generic kernels are not claimed bit-identical. Zero-copy Q and Q
scale views do match their copied equivalents bitwise within each consumer.
Small quantized-block checks cover both short and packed long-row execution.

A CUDA driver occupancy query on the RTX 4080 reports:

| 128-KV consumer | Registers/thread | Threads/CTA | Shared memory/CTA | Resident CTAs/SM | Resident warps/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic | 255 | 256 | 48 KiB | 1 | 8 |
| Packed | 254 | 128 | 40 KiB | 2 | 8 |

The extra independent CTA may improve scheduling across block barriers; the
resident warp count is unchanged. This is a resource query, not a measured
speedup or a reduction in total model VRAM. No timing benchmark was run.

## Compensated head accumulation prototype

`benchmarks/probe_compensated_head_sum.py` tests a prospective grouped-head
output projection. It stores BF16 residuals plus an INT8, per-row-scaled
rounding remainder. Seven grouped additions can then retain much of the
benefit of FP32 accumulation without a full-sequence FP32 residual allocation.

Three synthetic seeds, four BF16 gate strengths and a 7168-wide projection
were checked. At gate 0.1, relative L2 errors against the FP32 projection
reference were about 0.296% for native INT8, 0.438% for ordinary grouped BF16
addition and 0.171% with the compressed remainder. Aligned 1024-column
ConvRot weight slices were checked bitwise against full dequantization.
The output width was 256; these are projection-only tests, not real H3
activations, an entire block or video-quality validation.

A proposed full head-streamed block would preserve a quantized original
Norm/AdaLN input before modifying residuals and retain only one head group's
Q/K/V. At the example geometry its main-state accounting is about 1.19 GiB
smaller than full packed Q/K/V. This excludes weights and workspace and is
not an implemented or measured full-block peak. BF16 output projection and
extra grouped GEMMs change execution and numerical behavior. The prototype
is not enabled in inference.
