# H3 Optimizations

Production optimization nodes for MiniMax H3 in ComfyUI.

The pack focuses on two things: reducing the amount of VRAM H3 needs and making
longer H3 generations faster with optional sparse attention.

> **Experimental AMD branch:** On ROCm gfx11/gfx12 GPUs, the normal H3 Sparse
> Attention node tries the shipped AMD Sparse Kitchen INT8 backend first. On
> RDNA 2 gfx103x it skips sparse backends that require unavailable matrix/kernel
> support, then tries ROCm FlexAttention when usable and finally a probed
> sparse-over-existing-dense adapter. Adapter failures fail open to the same full
> dense ComfyUI attention consumer. This path is intended for hardware testing;
> the shipped gfx11/gfx12 native libraries have not yet passed a live AMD run.
> See [Experimental AMD Sparse Kitchen](#experimental-amd-sparse-kitchen).

# How does it affect quality?

You can see my test videos here 
https://huggingface.co/datasets/Zironic/h3-attention-breakpoint-10s

## Install

The package requires **ComfyUI 0.33.0 or newer** and is intended for installation
through ComfyUI-Manager.

Manual installation is also available from the ComfyUI `custom_nodes` directory:

```text
git clone https://github.com/Zironic/H3-Optimizations
```

Restart ComfyUI after installing. The nodes appear under
`H3-Optimizations > Model Patches`.

## Which nodes should I use?

> **For most users: add H3 Memory Optimization and leave it at default. Add H3
> Sparse Attention only if you want more speed and accept a quality tradeoff.
> You normally do not need the AIMDO or Advanced nodes.**

The pack exposes three normal optimization nodes plus one manual AIMDO residency
control:

| Node | What it is for | Normal recommendation |
| --- | --- | --- |
| **H3 Memory Optimization** | Reduces peak VRAM use during H3 generation | Add it and leave the defaults alone |
| **H3 Sparse Attention** | Makes H3 faster by calculating less video attention | Use it when speed matters and you accept a quality/speed tradeoff |
| **H3 Sparse Attention (Advanced)** | Gives manual control over sparse-attention scheduling and backend selection | Use only when you know why you need the extra controls |
| **H3 AIMDO Residency Limiter** | Manually controls how much of the H3 model DynamicVRAM keeps persistently resident in VRAM | Mainly for benchmarking, debugging AIMDO, or deliberately forcing low residency |

The **H3-Optimizations production nodes are order-independent with each other**.
You do not need to worry about whether Memory Optimization comes before or after
Sparse Attention.

Compatible external attention and `ModelPatcher` changes are preserved where
supported. A conflicting third-party patch can disable the corresponding H3
sub-optimization rather than being overwritten, so order-independence should not
be read as a guarantee about arbitrary external nodes.

Unsupported model families pass through unchanged. Automatic modes keep the
existing implementation when a specialized H3 path is not compatible with the
current checkpoint, attention backend, or runtime.

## H3 Memory Optimization

**Use this to reduce H3's VRAM usage.**

H3 normally creates several very large temporary tensors while generating a
video. This node processes compatible parts of that work in smaller pieces and
releases temporary data earlier, which can substantially reduce peak VRAM use.

For normal use, add the node and leave its settings at their defaults. It will
choose compatible optimizations for the current checkpoint, attention backend,
and GPU.

On recognized compatible ComfyUI versions, embedding assembly tensors are
released before block 0. Unrecognized implementations retain ComfyUI's stock
embedding lifetime instead of failing the workflow.

This is different from Sparse Attention: Memory Optimization is primarily about
how the same H3 work is executed and stored, rather than deliberately removing
video attention connections for speed.

### Normal settings

- **MLP memory optimization: Auto** — leave enabled.
- **QKV streaming: Auto** — leave enabled. It uses a lower-memory path when the
  active attention backend can consume it safely.
- **Precision mode: Auto** — lets the node choose a compatible execution path.
- **Attention memory mode: Standard** — prioritizes speed. `Lower VRAM (slower)`
  is available when you need to squeeze peak memory further.
- **Activation chunk rows** is an advanced control. Larger chunks can be faster
  but use more temporary memory. The UI range and 256-row step are editing
  recommendations; saved workflows may use any positive integer. A value at or
  above the current packed input length uses the ordinary unsliced MLP for that
  invocation instead of entering an effectively unchunked two-slice mode.

The advanced precision and streaming controls are mainly useful for debugging,
benchmarking, or deliberately forcing a particular execution policy.

## H3 AIMDO Residency Limiter

**Use this when you specifically want to control how much H3 model weight
DynamicVRAM keeps persistently resident on the GPU.**

This node exists mainly for benchmarking, debugging AIMDO behavior, and forcing
minimal persistent H3 model residency. It is not intended to imply that one
particular block count is universally optimal.

The default is **`0 blocks`**. This is the **maximum-offload setting**: it keeps
no H3 model blocks persistently resident. It can cause substantially more weight
streaming and may therefore be substantially slower than allowing some residency.
The default exists to make the node's manual/benchmarking purpose explicit, not
because `0 blocks` is generally preferable for normal generation.

- `0 blocks` keeps no H3 model blocks persistently resident in the AIMDO VBAR.
  The current weights are still staged through DynamicVRAM's temporary streaming
  buffers as they are needed.
- `1 block`, `2 blocks`, and `4 blocks` allow progressively more persistent model
  residency. That can reduce weight streaming but consumes more VRAM.
- `stock` disables the limiter and leaves ComfyUI's normal AIMDO residency policy
  unchanged.

The numeric settings require DynamicVRAM with asynchronous weight offloading.
The limiter only controls persistent H3 model residency; it does **not** place a
hard cap on activations, temporary buffers, force-loaded weights, or total GPU
memory use.

## H3 Sparse Attention

**Use this to make H3 faster at the cost of potentially changing the result.**

Sparse Attention reduces the amount of video-to-video attention H3 calculates.
The lower the **Video attention budget**, the less video attention work is done.

The default video attention budget is **15%**.

- Lower values are faster, but are more likely to affect prompt adherence,
  motion, detail, composition, or other parts of the result.
- Higher values retain more video attention and generally stay closer to dense
  attention.
- `1.0` keeps full video-attention connectivity, so no video connections are
  removed by sparsification.

`1.0` does **not** necessarily mean “use the same dense backend I would have used
without this node.” The Sparse Attention node can still resolve through its own
backend path. Bypass the node when you want the ordinary dense baseline.

The displayed `0.01` to `1.0` budget range is the recommended editing range,
not a server-side execution limit. Finite values below it retain at least one
whole video KV tile, while values above it saturate at the full video route.

Text, reference conditioning, audio, non-video queries, and mixed boundary tiles
remain dense.

> **Sparse attention changes model computation. It is not free acceleration.**
> There is no attention percentage that is guaranteed to be lossless for every
> prompt. H3 is especially sensitive to reducing attention in the early sampling
> steps.

The **Denser Early ramp** setting is enabled by default. It starts sampling at
no less than 50% video attention, then gradually reduces attention toward the
selected budget. For ordinary sparse budgets it targets 12 additional percentage
points per sampler step on average. Budgets already at or above 50% are
unchanged; on unusually short samplers, the 50% first-step minimum takes
precedence over the target average. H3 is especially sensitive to reduced
attention while the scene is being established, and the ramp avoids an abrupt
drop from a short dense prefix to the low budget.

## H3 Sparse Attention (Advanced)

**Use this when you want to tune Sparse Attention rather than simply turn it
on.**

The Advanced node exposes separate attention budgets for the beginning, middle,
and end of sampling, an early-schedule shape, and an explicit sparse-backend
selector.

- **Video attention budget** controls the middle steps.
- **Early schedule** selects **Hold** or **Ramp**. Hold keeps Early KV fixed for
  the configured Early steps. Ramp starts at Early KV and moves linearly toward
  Video attention budget over those steps. Set Early steps to `0` to disable it.
- **Early steps / Early KV** control the duration and held or starting budget.
- **Late steps / Late KV** do the same for the final steps.
- Step counts above the displayed `1000` editing limit remain valid; only
  negative step counts are rejected.
- If the early and late windows overlap, the denser requested budget wins.
- **Sparse backend** lets you explicitly select Kitchen INT8, FROST BF16,
  Sparse Sage, BF16 Triton, or FP8 FlexAttention.

The defaults preserve the existing **Hold** behavior: four early steps at 50%,
a 15% middle budget, and no late override, matching a 20-step schedule. Choose
**Ramp** for a gradual transition and increase Early steps for a longer ramp.
Late controls remain available for experiments, but denser late steps have not
shown enough benefit to justify their compute cost as a default.

Explicit backend choices are hard requirements: if you select a backend that is
not available on the current system, the node errors instead of silently
switching to another backend. Use the normal H3 Sparse Attention node when you
want automatic backend selection.

## Performance

MiniMax H3 text-to-video at 1376x768 (1.0 MP, 16:9) on an RTX 4070 12 GB,
`res_multistep`/`simple`, measured end to end through a real sampler. Each cell
executes five sampler steps and reports the median of the last three; step 1
carries model initialization and the step after it is a discarded warmup. The
20-step columns are projections from that median, not separate wall-clock runs.

The card was capped to **160 W** of its 200 W stock board power for these runs,
which is quieter and holds a steadier clock than the stock limit. A stock-power
card is faster than every row below, but not uniformly: the cap costs each
configuration in proportion to how power-saturated it already was, and dense
INT8 attention at 10 seconds is the most saturated cell in the matrix. The
Comfy Kitchen baseline is therefore the row that transfers least well to a
stock-power card, and the speedups below are, if anything, flattered by it.

Benchmark-only synthetic Qwen states and an empty native latent stand in for
the text encoder and VAE, so neither is loaded and neither contributes to the
timings or the memory figures.

Every benchmark arm also shares two controls that are **not normal user
settings**:

- `H3BenchmarkForceQKVConfig0` forces compatible ConvRot-256 INT8 QKV linears to
  Comfy Kitchen CUTLASS config 0 so the Kitchen dispatcher cannot choose a
  different large-sequence configuration between arms.
- `H3 AIMDO Residency Limiter` is explicitly fixed to `0 blocks` so model-weight
  residency cannot hide or amplify activation-memory differences between arms.

Both benchmark controls come from the benchmark setup; the first is provided by
the sibling H3-Extended pack. They should not be interpreted as recommended
workflow settings.

The VRAM columns are the **ComfyUI process's own dedicated GPU memory**, read
from the Windows GPU Process Memory counters that Task Manager reports. This
replaces the whole-GPU `nvidia-smi` figure used in earlier revisions of this
table, which carried the desktop compositor and every other application on the
card. That baseline is not a constant: measured here it sat near 2.9 GB during
the 5-second arms and collapsed to roughly 1.0 GB during the 10-second ones, as
Windows evicted desktop surfaces under pressure. Whole-GPU peaks from the two
durations were therefore never comparable with each other. Arm-to-arm
differences within one duration were much less affected, since the baseline
largely cancels.

| configuration | 5s step | 5s 20-step | 5s VRAM | 10s step | 10s 20-step | 10s VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Comfy Kitchen INT8 dense | 28.2 s | ~9m24s | 6193 MiB | 89.0 s | ~29m40s | 10849 MiB |
| SageAttention dense | 27.7 s | ~9m13s | 6737 MiB | out of memory | - | - |
| SageAttention + streamed QKV, chunked MLP/FinalLayer | 28.0 s | ~9m19s | 2801 MiB | 77.9 s | ~25m58s | 4787 MiB |
| H3 Memory Optimization + Sparse Attention (KV 100%) | 29.5 s | ~9m50s | 2963 MiB | 91.5 s | ~30m30s | 4595 MiB |
| **H3 Memory Optimization + Sparse Attention (KV 30%, measured)** | **18.7 s** | **~6m14s** | **2963 MiB** | **46.1 s** | **~15m21s** | **4563 MiB** |

Against dense Comfy Kitchen attention, the measured 30% configuration achieved
**1.51x faster at 5 seconds and 1.93x at 10 seconds**, using 3.2 GB less VRAM at
5 seconds and 6.1 GB less at 10 seconds.

These measurements predate the 15% middle-step and 50% early-step defaults; they
should not be read as performance evidence for the new schedule.

The rows isolate successive **conceptual optimization stages**, but they are not
a ladder of untouched UI defaults. The Sage memory arm pins
`Precision = Preserve native`, while the following 100% KV arm uses the normal
automatic precision policy. Both leave `QKV streaming = Auto`, and both stream:
the Sage memory arm projects ConvRot INT8 QKV straight into the dense Sage
carrier (`streamed_dense_sage_qkv`), which the benchmark verifies per arm with a
route assertion. Streaming is therefore not what separates those two rows; the
attention path and the precision policy are. The stage comparison is:

| conceptual stage | 5s speed | 5s VRAM |
| --- | ---: | ---: |
| Comfy Kitchen dense to dense SageAttention | 1.02x | +544 MiB |
| stream ConvRot INT8 QKV into the dense Sage carrier, chunk MLP and FinalLayer | 0.99x | -3936 MiB |
| swap that carrier for the native 64Q x 64KV path at 100% KV, precision on the automatic policy | 0.95x | +162 MiB |
| reduce video KV density to 30 percent | 1.58x | +0 MiB |

The native 64Q x 64KV path costs 162 MiB at 5 seconds rather than saving memory,
and pays it back at 10 seconds, where it sits below the dense Sage carrier
(4595 MiB against 4787 MiB, and 4563 MiB once KV drops to 30%). Dense Sage with
streamed QKV is the more memory-efficient choice until the sequence grows long
enough to invert that.

The measurements were taken with the NVIDIA driver's CUDA sysmem fallback
disabled, so exceeding VRAM fails instead of silently paging to system memory.
With fallback enabled, near-limit runs can page to system RAM and become much
slower.

---

# Technical reference

Everything below describes implementation details, backend routing, compatibility
contracts, and validation. You do not need to understand this section to use the
production nodes.

## Memory Optimization internals

H3 Memory Optimization preserves the dense attention selected by ComfyUI and
applies compatible memory/execution providers around it. This includes external
SageAttention, external Comfy Kitchen, and ComfyUI's normal attention backend.

Compatible QKV implementations can avoid materializing the full fused Q/K/V
projection at once. Depending on checkpoint format and attention consumer, the
node can stream bounded Q chunks while retaining the K/V representation required
by the backend. MLP and FinalLayer work are also processed in bounded token
chunks.

On recognized compatible ComfyUI versions, visual, audio, and source-row
embedding tensors are released immediately after the packed hidden state is
assembled, before block 0. Unrecognized implementations retain ComfyUI's stock
embedding lifetime.

The advanced precision selector exposes four policies:

- `Auto` chooses the best compatible path and may use FP8 conversion as a
  fallback.
- `BF16` materializes supported weights for BF16 execution.
- `Preserve native` does not introduce a new weight conversion.
- `Force quant` keeps supported native quantized checkpoints native and converts
  floating H3 linears to execution-scoped ConvRot-256 INT8 where supported.

Saved legacy `Preserve precision` and `Allow FP8 conversion` workflow values are
accepted as compatibility aliases for `Preserve native` and `Auto`.

### QKV streaming

`QKV streaming = Auto` preserves the current dense attention selection and adds
a compatible bounded Q/K/V carrier when that consumer supports it. Known Comfy
attention consumers can retain global K/V, stream bounded Q, and write each
output-projection chunk into the disposable block input.

An explicitly selected external Comfy Kitchen dense backend is a special case:
it retains a full INT8 Q/K/V carrier while streaming bounded output-projection
slabs.

Unknown explicit attention overrides preserve their ordinary full-Q single-call
contract unless they explicitly opt into the streamed-H3 consumer interface.

`Forced` authorizes the private full-density Kitchen path when compatible.
`Off` disables the Memory node's QKV streaming request. A separate Sparse
Attention request remains authoritative in the shared optimization plan.

### Attention memory mode

`Standard` prioritizes speed. `Lower VRAM (slower)` can use additional passes to
reduce peak memory for compatible V-carrier paths. Unsupported carrier formats
remain on the standard path.

## AIMDO implementation details

AIMDO is ComfyUI's DynamicVRAM model-weight streaming system. The residency
limiter applies a watermark to the H3 model's VBAR after dynamic model loading.
The numeric options are block-equivalent byte budgets rather than a promise that
particular coherent transformer blocks remain resident.

At `0 blocks`, no persistent VBAR pages are retained. The current block can still
execute because DynamicVRAM stages required weights through temporary async
buffers. This is the maximum-offload policy and can increase weight-streaming
cost substantially; it is not asserted to be an optimal general-use residency
level.

The limiter verifies the applied watermark and fails closed if persistent pages
remain resident above the requested cap. It is inactive when DynamicVRAM is not
enabled and requires asynchronous weight offloading for numeric limits.

## Sparse routing and backends

The standard H3 Sparse Attention node uses automatic backend selection. On
NVIDIA it prefers the shipped native Kitchen sparse backend when its per-GPU
self-test passes, then tries compatible Sparse Sage, BF16 Triton, FlexAttention,
and finally the resolved dense H3 path. On RDNA2 gfx103x, Auto skips Kitchen
INT8, Sparse Sage, and BF16 Triton because those paths are known incompatible;
it tries ROCm FlexAttention when available, then the probed sparse-over-existing-
dense adapter, and finally ordinary dense attention.

A `1.0` video budget leaves the video route fully connected but still enters this
backend-selection path. Bypass H3 Sparse Attention when comparing against the
ordinary dense attention route selected without the node.

The Advanced node instead uses the backend explicitly selected in its dropdown.
It does not traverse the automatic fallback chain.

### Native Kitchen

The shipped native Kitchen sparse path uses 64Q x 64KV routing for its production
sparse geometry. Compatible ConvRot-256 TensorWise INT8 QKV can feed the sparse
carrier without materializing full-sequence BF16 Q/K/V.

The 0.2.26 hotfix retained the full INT8 Q carrier. Version 0.2.27 restores
bounded streamed Q with a native Q-only producer that receives the global K
length and therefore uses the same quantization transform as the retained K/V
carrier.

The repository ships Windows x64 and Linux x86-64 native binaries. At first
resolution the local binary is loaded, its ABI is checked, and a cached per-GPU
numerical self-test is run. Nothing is downloaded, compiled, or installed during
startup.

The shipped CUDA targets are SM75, SM80, SM89, and SM120 on Windows, with SM90a
also included on Linux. Each target ships real SASS and one SM89 PTX fallback is
retained for forward compatibility.

### Experimental AMD Sparse Kitchen

This branch adapts the exact INT8 stage from Comfy Kitchen's HIP Sol-Attn work
to H3's existing 64Q x 64KV sparse route. H3 still chooses every KV block; the
Sol router, approximate tail, and fused QKV producer are not used. Supported
native targets are gfx11 (RDNA 3/3.5) and gfx12 (RDNA 4).

RDNA 2 gfx103x does not enter the native Kitchen path and does not try BF16
Triton: the latter requires BF16 matrix multiply that RDNA2 does not provide.
Sparse Sage is likewise skipped because it is an NVIDIA CUDA extension. RDNA2
Auto therefore tries ROCm FlexAttention when the installed stack can lower it;
otherwise it probes the existing ComfyUI dense attention consumer for 64Q x
64KV and then 128Q x 128KV packed sparse execution. If the adapter later meets
an unprobed route, shape, batch, or consumer restriction, it fails open to full
dense attention for the affected invocation or streamed Q chunk.

Credit to Deluxa for the RDNA3+ Sol exact HIP kernel adapted by this branch.

The branch ships prebuilt Linux x86-64 and Windows x64 libraries in
`native/hip/bin`, compiled with ROCm 7.2.1 for the supported gfx11/gfx12 targets. Testers do
not need CMake, a compiler, or the ROCm development SDK; they need only their
normal ROCm ComfyUI runtime and a supported GPU.

The loader searches `native/hip/bin`, `native/hip/lib`, `native/hip/build`, and
`native/hip/build/Release`. If the build emits the library elsewhere, place its
absolute path in the ignored `native/hip/library_path.txt` file.

Automatic sparse selection on gfx11/gfx12 AMD hardware tries AMD Sparse Kitchen
before the ordinary sparse fallback chain. The first resolution runs cached,
per-device full-route and genuinely sparse numerical comparisons against
PyTorch attention. These checks use a non-64-divisible sequence, production-
shaped strided HND inputs, NHD output, varying per-head/query-block routes, and
delta route conversion. A missing library, unsupported architecture, or failed
self-test retires AMD Sparse Kitchen and leaves the existing automatic fallback
behavior in control. Explicitly selecting `Kitchen INT8` remains a hard
requirement and reports the failure instead.

RDNA2 uses a different fail-open adapter path and is not evidence for the native
HIP kernel. The adapter's architecture-neutral routing, packing, probe, and
fallback contracts can be exercised on CUDA, but only live gfx103x hardware can
validate the actual ROCm dense consumer and runtime behavior.

This is an experimental tester branch. The shipped gfx11/gfx12 libraries have
been compiled and ABI-export checked in CI on Ubuntu 24.04 and Windows Server
2022, but have not been executed on AMD hardware. A live gfx11 or gfx12 run is
still required to establish numerical correctness and performance.

### FROST BF16

FROST BF16 is an explicit SM89-only 64Q x 64KV backend. It uses the packaged
cubin built from NVIDIA's open Apache-2.0 FROST SM80 template. It is not part of
the normal automatic backend selection.

### Sparse Sage

Sparse Sage requires a separately installed compatible `spas_sage_attn` build.
The active kernel ABI, Q/K tile geometry, scale layout, V carrier, accumulator,
routing summaries, and callables must match before the specialized producer is
selected.

A mismatched or unavailable Sparse Sage configuration moves to the next backend
only in automatic mode. An explicit Sparse Sage selection errors instead.

### BF16 Triton

The package BF16 Triton backend is available on supported Triton runtimes and
uses the same 64Q x 64KV routing geometry as the native Kitchen default. It can
stream projection chunks from supported BF16, ConvRot-256, W4A8, and FP8
checkpoints into its BF16 attention carrier without retaining a full fused
projection temporary. RDNA2 gfx103x is explicitly excluded because it lacks the
BF16 matrix multiply required by this backend.

### FlexAttention

On supported NVIDIA runtimes, FlexAttention can use FP8 carriers. Hopper and
Blackwell request the FA4 backend when its CuTe package is installed; other
supported NVIDIA runtimes use the Triton Flex kernel.

On ROCm, FlexAttention keeps Q/K/V in native BF16/FP16 and uses PyTorch
FlexAttention's Triton lowering with the same H3 sparse block mask. The ROCm
path is validated on first execution; if the installed PyTorch/ROCm/Triton stack
cannot lower the kernel, automatic mode retires that sparse signature and
returns to dense attention.

## ModelPatcher composition

H3 stores an immutable optimization plan on the cloned `ModelPatcher` and
reconciles consumer-dependent choices again at ComfyUI's prepare-sampling
boundary. A downstream node may therefore clone the model, add keyed wrappers,
apply LoRA/weight patches, select compile options, or change attention without
requiring H3 to be the last node before the sampler.

Every descendant clone receives a fresh request-local H3 runtime session and
fresh H3 execution callables. Package-owned keyed wrappers and clone callbacks
are replaced rather than appended, so repeated cloning does not accumulate
hooks.

Explicit external attention overrides are preserved. Foreign block, attention,
and FinalLayer patches are preserved per conflicting key; the conflicting H3
sub-optimization is disabled and reported in status instead of overwriting the
other patch.

## External streamed-H3 attention consumers

An `optimized_attention_override` can opt into the Memory node's bounded-Q
execution without being imported by this pack. The callable stored in
`transformer_options["optimized_attention_override"]` must expose:

```python
override.supports_streamed_h3_qkv = True
override.consume = consume

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
```

`q_chunk`, `global_k`, and `global_v` use HND layout. The consumer must support
rectangular attention, return an HND tensor with the same shape and device as
`q_chunk`, and treat `q_start` as the chunk's row offset within `q_total`.

H3 Memory Optimization owns source-aware Q/K/V projection, carrier lifetime,
chunk scheduling, and bounded output projection. The external consumer owns its
attention math and any state keyed by `layer_index` and global query rows.
Advertising the marker without a callable `consume` is an error. Unmarked
overrides retain full-Q single-call behavior.

## Requirements and architecture support

- ComfyUI 0.33.0 or newer with MiniMax H3 support and the `comfy_api.latest`
  extension API
- Python 3.10 or newer
- Windows x64 or Linux x86-64 for the shipped native Kitchen binaries
- Linux glibc 2.34+ and libstdc++ with `GLIBCXX_3.4.21`+ for the shipped Linux
  native Kitchen binary
- Any backend supported by ComfyUI's MiniMax H3 implementation for the final
  dense fallback
- NVIDIA SM75 or newer for the shipped native Kitchen default
- AMD gfx11 (RDNA 3/3.5) or gfx12 (RDNA 4) and a compatible ROCm PyTorch runtime for
  the shipped experimental AMD Sparse Kitchen library
- NVIDIA SM80 or newer with Triton for the BF16 Triton sparse fallback; RDNA2
  gfx103x is explicitly unsupported by that backend
- A ROCm-capable PyTorch build with a working existing H3 dense attention
  consumer for the RDNA2 sparse-over-dense adapter
- An FP8-capable NVIDIA GPU with PyTorch FlexAttention for the NVIDIA Flex path
- A ROCm-capable PyTorch build with FlexAttention/Triton for the AMD sparse Flex
  path
- NVIDIA CUDA SM80, SM86, SM87, SM89, SM90, or SM120 for compatible Sparse Sage
  builds

SM75/Turing is a reduced-feature supported target. H3 Memory Optimization,
AIMDO, bounded FP16 QKV/MLP/FinalLayer execution, dense Comfy Kitchen INT8, and
the shipped native sparse Kitchen path are eligible. BF16 Triton, FROST, FP8,
and Sparse Sage remain unavailable on SM75. GPUs below SM75 retain
architecture-neutral memory management and chunking only.

Production node IDs are `H3MemoryOptimization`, `H3AIMDOResidencyLimiter`,
`H3SparseAttention`, and `H3SparseAttentionAdvanced`. H3-Extended is not
required for the production nodes.

## Validation

CPU tests cover node schemas, AIMDO limiter arithmetic and load callbacks, plan
composition, backend classification, streamed and chunked projection contracts,
FROST ABI behavior, native shipping contracts, explicit sparse-backend
selection, chunk boundaries and RoPE slices, non-H3 no-op behavior, sparse route
geometry, runtime step/layout publication, early/middle/late density schedules,
and source isolation. RDNA2 CPU contracts additionally verify that known-dead
Auto backends are skipped and that adapter preparation/execution failures fail
open to the existing dense consumer.

GPU kernel validation is intentionally separate because it requires matching
hardware and compiled backend packages. A CUDA contract test also exercises the
architecture-neutral RDNA2 adapter with an unknown dense consumer that passes the
probe and later rejects an unprobed packed shape; this validates fail-open control
flow on real GPU tensors but does not validate ROCm execution.

Live SM89 gates compare whole-carrier and streamed Kitchen Q/Q-scale exactly,
exercise every shipped Kitchen geometry, and check the declared SM89
attention-backend matrix against dense or explicitly masked SDPA references.
The matrix requires finite output plus route-specific relative-L2 and maximum
absolute-error limits. Its streamed dense Sage row runs lazy normalized input
through the real Sage kernel and chunked output projection, covering the full
source/output lifetime boundary rather than only a prepared carrier.

Run the CPU suite from the ComfyUI root:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
.\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v
```

The AIMDO residency benchmark runs every limiter level in a fresh process and
reports persistent VBAR pages, temporary cast buffers, AIMDO usage, and
whole-device VRAM separately.

The attention benchmark is `benchmarks/bench_attention_arms.py`; it drives a
running ComfyUI server over the prompt API. It applies the same two shared
benchmark controls described above to every arm: `H3BenchmarkForceQKVConfig0`
and AIMDO `0 blocks`.

## Acknowledgements

- Thanks to [Pizzawookiee](https://github.com/Pizzawookiee) for the low-VRAM
  experiments in [PR #13](https://github.com/Zironic/H3-Optimizations/pull/13)
  and [PR #26](https://github.com/Zironic/H3-Optimizations/pull/26). Although
  neither PR was merged as-is, those experiments helped inform the streamed
  QKV and FinalLayer chunking work that later shipped.
- The sparse-attention work draws ideas from
  [MoBA](https://github.com/MoonshotAI/MoBA) and
  [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/).
- This project relies heavily on
  [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) for quantization,
  ConvRot execution, and the kernel foundations used by the native attention
  backend.

  ## Kofi
  https://ko-fi.com/zironic
