# H3 Optimizations

Production optimization nodes for MiniMax H3 in ComfyUI. The pack can reduce
peak VRAM use and, with optional sparse attention, make longer generations
faster.

## Quick start

1. Add **H3 Memory Optimization** between your H3 model and sampler.
2. Leave its settings at their defaults.
3. If you want more speed and accept a quality tradeoff, also add **H3 Sparse
   Attention**.

Most users do not need the Advanced Sparse Attention or AIMDO nodes.

## Quality and output changes

**H3 Memory Optimization** changes how compatible work is divided and stored.
It is designed to preserve the model's computation while reducing temporary
memory use.

**H3 Sparse Attention** deliberately removes some video-to-video attention
connections. Lower budgets are faster, but can affect prompt adherence, motion,
detail, composition, or other parts of the result. No sparse percentage is
guaranteed to be lossless for every prompt.

Example comparisons are available in the
[H3 attention breakpoint dataset](https://huggingface.co/datasets/Zironic/h3-attention-breakpoint-10s).

## Install

The package requires **ComfyUI 0.33.0 or newer** and is intended for installation
through ComfyUI-Manager.

For a manual installation, run this from ComfyUI's `custom_nodes` directory:

```text
git clone https://github.com/Zironic/H3-Optimizations
```

Restart ComfyUI after installing. The nodes appear under
`H3-Optimizations > Model Patches`.

## Choose a node

| Node | Use it when | Recommendation |
| --- | --- | --- |
| **H3 Memory Optimization** | You want lower peak VRAM use | Start here and keep the defaults |
| **H3 Sparse Attention** | You want more speed and accept a quality tradeoff | Start with the default 15% budget and denser-early ramp |
| **H3 VSA Attention (FastH3)** | You use a FastH3/VSA-trained checkpoint | Use the keep percentage the checkpoint was trained with |
| **H3 Sparse Attention (Advanced)** | You need manual schedules, backends, or token order | Advanced users only |
| **H3 AIMDO Residency Limiter** | You need to force model-weight residency for diagnostics | Benchmarking and debugging only |

The Memory, Sparse, and AIMDO nodes are order-independent with each other.
Unsupported model families pass through unchanged. Automatic modes retain the
existing implementation when a specialized H3 path is incompatible with the
checkpoint, attention backend, or runtime.

Compatible external attention and `ModelPatcher` changes are preserved where
possible. If another patch conflicts with an H3 sub-optimization, that part of
the H3 plan is disabled instead of silently overwriting the other patch.

## H3 Memory Optimization

H3 creates several large temporary tensors while generating video. This node
processes compatible work in smaller pieces and releases temporary data earlier,
which can substantially reduce peak VRAM use.

For normal use, keep these settings:

- **MLP memory optimization: Auto**
- **QKV streaming: Auto**
- **Precision mode: Auto**
- **Attention memory mode: Standard**

Choose **Lower VRAM (slower)** only when you need to reduce the peak further.
The **Activation chunk rows** control trades temporary memory for throughput;
the default is the normal recommendation.

Embedding assembly tensors are released before block 0. ComfyUI 0.39.2 and
newer do this themselves; on older recognized versions the node does it.
Unrecognized implementations retain ComfyUI's normal embedding lifetime instead
of failing the workflow.

This node is different from Sparse Attention: it changes how the work is
executed and stored rather than intentionally removing video attention
connections.

## Experimental single-pass V and direct residual

In **H3 Memory Optimization**, choose **Attention memory mode: Single-pass V +
direct residual (experimental)**. This explicitly selects dense Kitchen INT8;
disconnect other H3 block/attention patches. Sparse Attention and trained VSA
checkpoints are unsupported in this mode.

Q/K/V are projected together once per chunk. V is quantized immediately with
per-block channel scales, and packed INT8 Q is retained so attention output can
be projected and gated directly into the residual without overwriting future
queries. It avoids V reprojection and the full hidden-size output buffer.
Per-block V quantization changes numerical results; rendered-video quality and
full-workflow speed/peak VRAM have not yet been measured. Existing modes and
defaults are unchanged.

The local Windows build is `native/bin/h3_block_v_v3.dll` (SM89 SASS plus SM80 PTX).
Rebuild from a PowerShell terminal with CUDA and Visual Studio C++ Build Tools:
`./native/build_block_v.ps1`. A missing library fails explicitly in this mode.

Correctness checks: `tests/test_block_v.py` and `tests/test_block_v_cuda.py`.

## H3 Sparse Attention

Sparse Attention reduces the amount of video-to-video attention H3 calculates.
The lower the **Video attention budget**, the less work is done.

The default budget is **15%**, with **Denser Early ramp** enabled:

- Lower values are faster but more likely to change the result.
- Higher values generally stay closer to dense attention.
- `1.0` keeps all video-attention connections.
- Text, reference conditioning, audio, non-video queries, and mixed boundary
  tiles remain dense.

H3 is especially sensitive to reduced attention while the scene is being
established. The default ramp therefore starts sampling at no less than 50%
video attention and gradually moves toward the selected budget. Budgets already
at or above 50% are unchanged.

A `1.0` budget still enters the Sparse node's backend-selection path. Bypass the
node when you want the ordinary dense baseline.

VSA-trained checkpoints such as FastH3 must use **H3 VSA Attention (FastH3)**
instead. The normal Sparse nodes detect their learned gate layers and leave the
model unchanged rather than applying an incompatible sparse pattern.

## H3 VSA Attention (FastH3)

Use this node only with a VSA-trained MiniMax H3 checkpoint. It reproduces the
checkpoint's trained sparse pattern and learned coarse attention branch. It
rejects an ordinary H3 checkpoint instead of silently applying VSA semantics to
incompatible weights.

Set **Video tiles kept** to the checkpoint's trained value:

- FastH3 8-Step V2: **20%**
- FastH3 Preview v1: **10%**

The dense-step and dense-layer controls are diagnostic overrides, not the
trained default. The normal node uses the measured **1x8x8** target-video order
and restores raster order at model output.

The **Auto** backend runs a one-time numerical check on each GPU before using
the shipped INT8 tile kernel. A missing or failing kernel falls back to BF16
Triton. ROCm uses BF16 Triton because the AMD native library does not expose the
VSA tile entry point.

## Measured performance

These results are MiniMax H3 text-to-video at 1376x768 (1.0 MP, 16:9) on an RTX
4070 12 GB. Each timing is the median steady-state sampler step from a five-step
run; the 20-step columns are projections. VRAM is the ComfyUI process's dedicated
GPU memory, not whole-card `nvidia-smi` usage.

| configuration | 5s step | 5s 20-step | 5s VRAM | 10s step | 10s 20-step | 10s VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Comfy Kitchen INT8 dense | 24.9 s | ~8m18s | 5769 MiB | 71.7 s | ~23m55s | 9947 MiB |
| SageAttention dense | 24.2 s | ~8m04s | 6201 MiB | 74.4 s | ~24m47s | 10747 MiB |
| SageAttention dense + H3 Memory Optimization | 24.6 s | ~8m12s | 2963 MiB | 70.6 s | ~23m31s | 4435 MiB |
| Native INT8 attention (KV 100%) + H3 Memory Optimization | 25.0 s | ~8m20s | 2933 MiB | 70.2 s | ~23m24s | 4501 MiB |
| **Native INT8 attention (fixed KV 30%) + H3 Memory Optimization** | **15.5 s** | **~5m09s** | **2933 MiB** | **37.3 s** | **~12m25s** | **4501 MiB** |
| Native INT8 attention (fixed KV 30%) + H3 Memory Optimization (Lower VRAM) | 16.5 s | ~5m30s | 2549 MiB | 39.2 s | ~13m04s | 3733 MiB |

The three native KV rows were rerun on September 30, 2026 with the local
**64Q x 32KV** production selection and threshold routing at KV30; KV100
bypasses selection. The dense/Sage rows and packed64 stage progression below
are earlier measurements, not fresh paired controls.

Lower VRAM saved a further **384 MiB at 5 seconds and 768 MiB at 10 seconds**
against that standard fixed-30% path, with measured speed costs of 6.8% and
5.2%, respectively.

The fixed-30% row is a constant-budget comparison point, not the current default.
The current 15% middle budget plus denser-early ramp redistributes roughly the
same full-schedule attention work toward quality-sensitive early steps. It is
not shown here because a short run containing only those early steps would
overstate its full-schedule cost.

Results vary with GPU, checkpoint format, attention backend, sequence length,
residency, and system configuration. See
[Benchmark methodology](#benchmark-methodology) before comparing or reproducing
these numbers.

## Compatibility at a glance

- ComfyUI 0.33.0 or newer with MiniMax H3 support
- Python 3.10 or newer
- Windows x64 or Linux x86-64 for the shipped native Kitchen binaries
- NVIDIA SM75 or newer for the shipped native Kitchen path
- Other ComfyUI-supported H3 backends retain the dense fallback and compatible
  architecture-neutral memory optimizations

SM75/Turing is a reduced-feature target. Memory Optimization, AIMDO, bounded
FP16 execution, dense Comfy Kitchen INT8, and native sparse Kitchen are eligible;
BF16 Triton, FROST, FP8, and Sparse Sage are not.

AMD sparse support has additional hardware-specific limits. See
[AMD Sparse Kitchen](#amd-sparse-kitchen).

### LoRAs

Use ComfyUI's standard **Load LoRA** node. Its weight patches apply normally
with every node in this pack.

Bypass-mode LoRAs, such as ComfyUI's **Load LoRA (Bypass)** debugging nodes or
custom nodes that load LoRAs in bypass mode, are not supported with H3 Memory Optimization or H3 Sparse
Attention. A bypass LoRA works by replacing a layer's `forward`, but these nodes
compute the attention projections and MLP layers with their own chunked and
fused kernels, which never call that `forward`. The LoRA is silently ignored on
those layers, with no error. Supporting it would need a way for an adapter to
apply its delta to part of a layer's output, which ComfyUI's bypass hook does
not provide.

---

## Technical reference

The sections below cover manual controls, benchmark methodology, backend
routing, implementation contracts, and developer validation. They are not
required for normal use.

### H3 Sparse Attention (Advanced)

The Advanced node exposes separate budgets for the beginning, middle, and end
of sampling, an early schedule, an explicit backend, and target-video token
order.

- **Video attention budget** controls the middle steps.
- **Early schedule** selects **Hold** or **Ramp**. Hold keeps Early KV fixed;
  Ramp moves linearly from Early KV toward the middle budget.
- **Early steps / Early KV** control the beginning of sampling.
- **Late steps / Late KV** control the final steps.
- If early and late windows overlap, the denser requested budget wins.
- **Video token order** defaults to **1x8x8**. Measured **1x16x4** and **4x4x4**
  layouts are comparison options; **Raster** is stock H3 order.
- **Kitchen INT8 64Q x 32KV (Quality)** is the default geometry. Compatible
  native builds use threshold routing; other builds retain top-k routing.
  Auto selection falls back to 64KV if the 32KV device check fails.
- **Kitchen INT8 64Q x 128KV (Faster)** uses coarser KV tiles. It was faster in
  the measured SM89 full-block comparison, but can change quality and is not
  guaranteed to be faster on every GPU.

The Advanced defaults use an eight-step **Ramp** starting at **68.33%** and
moving toward the 15% middle budget, with no late override on the default
20-step schedule.

Explicit backend choices are hard requirements. If the selected backend is not
available, the node errors instead of silently switching. Use the normal Sparse
Attention node for automatic fallback behavior.

### H3 AIMDO Residency Limiter

AIMDO is ComfyUI's DynamicVRAM model-weight streaming system. This node manually
limits how much H3 model weight its VBAR keeps persistently resident on the GPU.
It is intended for benchmarking and diagnostics, not as a normal optimization
recommendation.

The default **`0 blocks`** is the maximum-offload setting. It keeps no H3 model
blocks persistently resident and can be substantially slower because weights
must be streamed more often.

- `0 blocks` keeps no persistent H3 model blocks in the AIMDO VBAR.
- `1 block`, `2 blocks`, and `4 blocks` allow progressively more residency and
  consume progressively more VRAM.
- `stock` removes the limiter and uses ComfyUI's normal AIMDO policy.

Numeric limits require DynamicVRAM with asynchronous weight offloading. They are
block-equivalent byte budgets, not a guarantee that coherent transformer blocks
remain resident. They do not cap activations, temporary buffers, force-loaded
weights, or total GPU memory.

The limiter verifies its watermark and fails closed if persistent pages remain
above the requested cap.

### Benchmark methodology

The table above was measured end to end through a real sampler with
`res_multistep`/`simple`. Before the matrix, the harness performs one discarded
priming request so checkpoint loading is outside the measured arms. Each speed
request then executes five sampler steps: the extra first boundary establishes
timing, the first of the four resulting durations is discarded as warmup, and
the final three durations produce the reported median.

The RTX 4070 ran at its stock **200 W** board-power limit. The harness verifies
that limit but does not pin clocks. Observed board-power peaks ranged from 192 W
to 197 W.

A real cached Qwen conditioning entry and an empty native latent keep the text
encoder and VAE out of the timings and memory figures. Speed runs use stock
AIMDO residency. Separate VRAM runs unload the model, set AIMDO to `0 blocks`,
and execute one first-sigma step so retained weights do not distort activation
memory differences.

All arms force compatible ConvRot-256 INT8 QKV linears to Comfy Kitchen CUTLASS
config 0. This is a benchmark control, not a recommended workflow setting. The
Memory/Sage row also pins **Precision = Preserve native**; later native rows use
the normal automatic policy. Both use QKV streaming.

The historical packed64 stage progression at 5 seconds was:

| conceptual stage | speed | VRAM change |
| --- | ---: | ---: |
| Comfy Kitchen dense to dense SageAttention | 1.03x | +432 MiB |
| Stream ConvRot INT8 QKV into dense Sage; chunk MLP and FinalLayer | 0.99x | -3238 MiB |
| Use native 64Q x 64KV at 100% KV with automatic precision | 0.93x | -24 MiB |
| Reduce video KV density to 30% | 1.56x | +0 MiB |
| Stage V in two passes | 0.99x | -336 MiB |

A matched native-allocator replay across all 50 H3 layers measured a 1730.7 MiB
Standard peak and a 1381.2 MiB Lower VRAM peak, a 349.5 MiB reduction. The
process-memory runs independently measured reductions of 336 MiB at 5 seconds
and 720 MiB at 10 seconds.

CUDA sysmem fallback was disabled, so exceeding VRAM failed instead of silently
paging to system memory.

### Memory Optimization internals

Memory Optimization preserves the dense attention selected by ComfyUI and adds
compatible memory/execution providers around it. This includes external
SageAttention, external Comfy Kitchen, and ComfyUI's normal attention backend.

Compatible implementations avoid materializing the full fused Q/K/V projection
at once. Depending on checkpoint format and consumer, the node streams bounded Q
chunks while retaining the K/V form required by the backend. MLP and FinalLayer
work are also processed in bounded token chunks.

The precision policies are:

- `Auto` preserves floating checkpoint precision and keeps supported native
  quantized formats native.
- `BF16` materializes supported weights for BF16 execution.
- `Preserve native` does not introduce a new weight conversion.
- `Force quant` keeps supported quantized checkpoints native and converts
  floating H3 linears once to canonical ConvRot-256 INT8.

Force quant modifies only the isolated in-memory model branch; it does not
rewrite the checkpoint. Reloading for conversion can temporarily require close
to two BF16 copies in system memory. Unsupported custom loaders fail clearly.

Legacy `Preserve precision` and `Allow FP8 conversion` workflow values remain
accepted as aliases for `Preserve native` and `Auto`.

#### MLP and residual paths

Auto uses ordinary 4096-row MLP chunking by default. A chunk size at or above
the packed input length uses the ordinary unsliced MLP for that invocation.

On compatible ConvRot INT8 checkpoints, Comfy Kitchen 0.2.34+ and NVIDIA SM80+
can fold the gated residual add into fc2. The output is bit-identical and avoids
one hidden-width temporary. Other hardware, older Kitchen versions, and
per-token modulation segments retain the ordinary add.

Floating MLPs that keep their own precision compute SwiGLU in place, saving one
activation-width temporary per chunk with identical output.

#### QKV streaming

`Auto` preserves the dense attention selection and adds a compatible bounded
carrier when supported. Known Comfy consumers retain global K/V, stream bounded
Q, and write output-projection chunks into disposable block input.

An explicitly selected external Comfy Kitchen dense backend retains a full INT8
Q/K/V carrier while streaming bounded output-projection slabs. Unknown explicit
overrides retain their full-Q single-call contract unless they opt into the
streamed-H3 interface.

`Forced` authorizes the private full-density Kitchen path when compatible. `Off`
disables the Memory node's request; a separate Sparse request remains
authoritative.

#### Attention memory mode

`Standard` prioritizes speed. `Lower VRAM (slower)` can use additional passes to
reduce peak memory for compatible V-carrier paths. Unsupported carriers stay on
the standard path.

### Sparse routing and backends

On NVIDIA, automatic Sparse Attention prefers the shipped native Kitchen
backend when its per-GPU self-test passes, then tries compatible Sparse Sage,
BF16 Triton, FlexAttention, and finally the resolved dense H3 path.

On RDNA2 gfx103x, Auto skips Kitchen INT8, Sparse Sage, and BF16 Triton. It tries
ROCm FlexAttention, then a probed sparse-over-existing-dense adapter, and finally
ordinary dense attention.

#### Native Kitchen

The automatic production sparse geometry is 64Q x 32KV, with 64KV retained as
a device-validated fallback and an explicit choice. Compatible ConvRot-256 TensorWise
INT8 QKV can feed the sparse carrier without materializing full-sequence BF16
Q/K/V.

Packed32 consumes up to four routed 32-token tiles per 64Q x 128KV iteration
over the same 64-token carrier. Compatible native builds select tiles with
threshold routing using BF16 summaries. The requested budget targets average
video density rather than an exact per-row tile count. Non-video context stays
dense, 100% KV uses a full route, and builds without the threshold router retain
top-k selection. Explicit 64KV and 128KV backends remain available.

For compatible ConvRot-256 INT8 checkpoints on SM80+, the path can use an exact
fused Q producer. Q projection dimensions are 128x256, while packed32 routing
uses 32-token KV summaries over the same 64-token carrier. A one-time parity
check falls back to the established producer when needed.

The repository ships Windows x64 and Linux x86-64 binaries. On first use, the
loader checks the ABI and runs a cached per-GPU numerical self-test. Nothing is
downloaded, compiled, or installed at startup.

CUDA targets are SM75, SM80, SM86, SM89, and SM120 on Windows, with SM90a also
on Linux. Each target includes SASS, with an SM89 PTX fallback for forward
compatibility.

#### FROST BF16

FROST is an explicit SM89-only 64Q x 64KV backend using a packaged cubin based
on NVIDIA's Apache-2.0 FROST SM80 template. It is not part of automatic backend
selection.

#### Sparse Sage

Sparse Sage requires a separately installed compatible `spas_sage_attn` build.
Its kernel ABI, geometry, scale layout, carrier, accumulator, routing summaries,
and callables must match. Automatic mode moves on when incompatible; an explicit
selection errors.

#### BF16 Triton

BF16 Triton uses 64Q x 64KV routing and can stream supported BF16, ConvRot-256,
W4A8, and FP8 projections into its BF16 carrier. RDNA2 gfx103x is excluded
because it lacks the required BF16 matrix multiply.

#### FlexAttention

On NVIDIA, FlexAttention can use FP8 carriers. Hopper and Blackwell request FA4
when its CuTe package is installed; other supported NVIDIA runtimes use Triton
Flex.

On ROCm, FlexAttention retains BF16/FP16 Q/K/V and uses PyTorch FlexAttention's
Triton lowering. If first-use validation fails, automatic mode retires that
signature and returns to dense attention.

#### AMD Sparse Kitchen

AMD Sparse Kitchen adapts the exact INT8 stage from Comfy Kitchen's HIP
Sol-Attn work to H3's 64Q x 64KV route, with H3 choosing every KV block. Native
targets are gfx11 (RDNA 3/3.5) and gfx12 (RDNA 4).

Prebuilt Windows x64 and Linux x86-64 libraries were compiled with ROCm 7.2.1.
The loader checks `native/hip/bin`, `native/hip/lib`, `native/hip/build`, and
`native/hip/build/Release`. A custom absolute path can be placed in the ignored
`native/hip/library_path.txt` file.

Automatic selection on gfx11/gfx12 tries AMD Sparse Kitchen before other sparse
fallbacks. Cached per-device tests compare full and sparse routes against
PyTorch attention. Missing libraries, unsupported hardware, or failed tests
leave the normal fallback chain in control. Explicit Kitchen selection remains
a hard requirement and reports failures.

RDNA2 uses the separate fail-open adapter described above; it is not evidence
for the native HIP kernel.

The libraries have passed compile and ABI-export checks on Ubuntu 24.04 and
Windows Server 2022 but have not yet been run on AMD hardware; the per-device
tests above guard first use.

Credit to Deluxa for the RDNA3+ Sol exact HIP kernel adapted here.

### ModelPatcher composition

H3 stores an immutable optimization plan on the cloned `ModelPatcher` and
reconciles consumer-dependent choices at ComfyUI's prepare-sampling boundary.
Downstream nodes may clone, apply LoRA/weight patches, select compile options, or
change attention without requiring H3 to be the last node before the sampler.
Bypass-mode LoRAs are the exception; see [LoRAs](#loras).

Each descendant receives a fresh request-local runtime session and execution
callables. Package-owned wrappers and clone callbacks are replaced rather than
appended, so repeated cloning does not accumulate hooks.

Foreign block, attention, and FinalLayer patches are preserved per conflicting
key. The conflicting H3 sub-optimization is disabled and reported in status.

#### Dense attention overrides Sparse may replace

Sparse replaces an upstream override only when it is a recognized dense H3
kernel:

- this pack's installed dense backend
- Comfy Kitchen INT8 selected by ComfyUI or an external node
- KJNodes' SageAttention patch

Other overrides are preserved and sparse attention is disabled because an
unknown override may not be dense. With no sparse request, Memory Optimization
preserves the override and can use the streamed-H3 contract below.

#### Spectrum forecast output

Cube-ordered video rows are restored at FinalLayer, including Spectrum forecast
steps that bypass the normal H3 forward. Audio rows are unchanged. A foreign
FinalLayer patch keeps raster order instead. Use a current Spectrum version for
current ComfyUI and PDD checkpoints.

### External streamed-H3 attention consumers

An `optimized_attention_override` can opt into bounded-Q execution without being
imported by this pack:

```python
def consume(
    q_chunk,
    global_k,
    global_v,
    q_start,
    q_total,
    layer_index,
    transformer_options,
):
    ...

override.supports_streamed_h3_qkv = True
override.consume = consume
```

The tensors use HND layout. The consumer must support rectangular attention,
return an HND tensor matching `q_chunk`, and interpret `q_start` as the chunk's
offset within `q_total`.

Memory Optimization owns source-aware projection, carrier lifetime, chunk
scheduling, and bounded output projection. The external consumer owns attention
math and state keyed by layer and global rows. Advertising the marker without a
callable `consume` is an error. Unmarked overrides retain full-Q behavior.

### Detailed requirements and architecture support

- ComfyUI 0.33.0+ with MiniMax H3 and `comfy_api.latest`
- Python 3.10+
- Linux glibc 2.34+ and `GLIBCXX_3.4.21`+ for the shipped Linux native binary
- Any ComfyUI-supported MiniMax H3 backend for the final dense fallback
- NVIDIA SM75+ for native Kitchen
- AMD gfx11/gfx12 plus a compatible ROCm PyTorch runtime for AMD Sparse Kitchen
- NVIDIA SM80+ with Triton for BF16 Triton; RDNA2 is unsupported
- A ROCm PyTorch build with a working dense H3 consumer for the RDNA2 adapter
- An FP8-capable NVIDIA GPU with PyTorch FlexAttention for NVIDIA Flex
- ROCm PyTorch with FlexAttention/Triton for AMD Flex
- CUDA SM80, SM86, SM87, SM89, SM90, or SM120 for compatible Sparse Sage builds

Production node IDs are `H3MemoryOptimization`, `H3SparseAttention`,
`H3VSAAttention`, `H3SparseAttentionAdvanced`, and
`H3AIMDOResidencyLimiter`.

### Validation

CPU tests cover node schemas, AIMDO arithmetic, plan composition, backend
classification, streamed/chunked projection, native shipping, sparse routing,
non-H3 pass-through, schedule behavior, token-order permutations, and source
isolation. RDNA2 contracts also verify that known-incompatible backends are
skipped and adapter failures return to the existing dense consumer.

Run the CPU suite from the ComfyUI root:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
.\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v
```

GPU validation is separate because it requires matching hardware and compiled
backends. Live SM89 gates compare full and streamed carriers, exercise every
shipped Kitchen geometry, and check backend outputs against dense or masked SDPA
references with route-specific numerical limits.

## Acknowledgements

- Thanks to [Pizzawookiee](https://github.com/Pizzawookiee) for the low-VRAM
  experiments in [PR #13](https://github.com/Zironic/H3-Optimizations/pull/13)
  and [PR #26](https://github.com/Zironic/H3-Optimizations/pull/26). They helped
  inform the streamed QKV and FinalLayer chunking work that later shipped.
- Sparse attention draws ideas from
  [MoBA](https://github.com/MoonshotAI/MoBA) and
  [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/).
- This project relies heavily on
  [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) for quantization,
  ConvRot execution, and the native attention kernel foundations.

## Support

[Support the project on Ko-fi](https://ko-fi.com/zironic).
