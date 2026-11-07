# Vendored INT8 attention

The pack ships its own compiled INT8 attention instead of depending on a
comfy-kitchen release.

## Why this exists

ComfyUI pins `comfy-kitchen==0.2.31` in its requirements. Anything that lives
only in a later Kitchen, or in a fork, reaches nobody — which is exactly what
happened to the chunked QKV producer: fully integrated on this side, calling a
Kitchen API that was never released, so every install silently fell back to
standard projection at roughly half the speed.

Four pieces are needed and none is in 0.2.31:

- **the external INT8 attention producer**, which lets a chunked QKV
  projection quantize each chunk as it is produced rather than materializing
  the whole BF16 sequence;
- **block-sparse KV traversal**, which walks a routed subset of the KV tiles.
- **composable sparse softmax state**, which lets the same native traversal
  return its base-2 row normalizer for a separate rejected-tile residual.
- **the exact H3 fused-Q producer**, which combines the ConvRot INT8 Q GEMM,
  RMSNorm, RoPE, Kitchen Q packing, and 64-row routing summaries without a
  full BF16 Q intermediate.

## Provenance

`src/sage_attention/` and the two headers beside it are vendored from Comfy
Kitchen (Apache-2.0, see `LICENSE` and `NOTICE.upstream`), which derives them
in turn from SageAttention. The exact commit is recorded in `src/PROVENANCE`.

The launcher change is marked with a `VENDORING CHANGE` comment: the launchers
lost their `extern "C"` linkage. Upstream they are reached through nanobind;
here they are internal to
this library and only `h3_int8_attention_api.cu` calls them. C linkage bought
nothing and cost correctness — under MSVC's `/EHsc` the trailing `c` means
*extern "C" functions never throw*, so the compiler elides the unwind and a
throw terminates the process instead of reaching the catch in the API layer.
Upstream warns about this as C4297 on every build; here it was fatal until the
linkage changed.

The exact fused-Q kernel uses the C++ header tree from NVIDIA CUTLASS 4.6.0.
Those headers and their BSD-3-Clause license are vendored under
`third_party/cutlass`; they are build inputs only and introduce no runtime or
Python-package dependency. The source revision is recorded in
`third_party/cutlass/PROVENANCE`.

## The API boundary

`src/h3_int8_attention_api.cu` is the stable surface. Every entry point is
`noexcept`, returns `0` on success or non-zero on failure, and leaves a message
for `h3_int8_last_error()`. `h3_optimizations/native/loader.py` turns that into
an ordinary Python exception.

It is plain C — pointers, ints, a stream handle — loaded through `ctypes`. No
nanobind, no DLPack, and therefore no Python ABI dimension: one binary per OS
and architecture set serves every interpreter, rather than separate cp310,
cp311 and abi3 builds.

ABI 4 adds Q-only carrier packing with the global K length kept as transform
dispatch metadata, without allocating or rewriting a dummy K carrier. ABI 3
added explicit sparse Q geometry and Q-scale strides. It exposes exact
128Q x 128KV, 128Q x 64KV, 64Q x 128KV, and 64Q x 64KV traversal without
changing Kitchen's Q128 quantization carrier. ABI 2 added
`h3_int8_sparse_attention_lse`; that output
remains one FP32 base-2 log-sum-exp value per query row. The packaged Windows
DLL is build-versioned so a running Comfy process can keep an older mapped DLL
until the next normal restart.

ABI 4 also accepts the additive
`h3_int8_quantize_bf16_rowwise_convrot256` and `h3_int8_fused_q` symbols. They
are optional when loading an older ABI-4 binary, so unsupported installs keep
the established Q projection and packing path. The fixed exact 128x256 fused-Q
kernel is selected only on SM80 or newer; this does not change the SM75 support
of the existing attention kernels.

ABI 4 also accepts the additive `h3_int8_threshold_route` symbol. It selects
video KV tiles whose score exceeds the row's mean plus tau standard deviations,
computed from the K-summary mean and covariance, so no score matrix, top-k, or
sort is needed. It requires SM80 or newer and is optional when loading an older
binary; the Python side then keeps top-k routing.

64Q x 64KV routes run on the packed 64Q x 128KV kernel on SM86, SM89, and SM12x,
where the card is usually power-limited and the packed kernel's lower energy
per call becomes clock. Other targets keep the 64Q x 64KV kernel. 64Q x 32KV
routes walk 32-token subtiles of the same 64-token carrier with the packed
fine-route kernel on every target.

ABI 4 also accepts the additive
`h3_int8_quantize_bf16_rms_adaln_rowwise_convrot256` producer. The H3 ConvRot
path uses its explicit-BF16-staging mode to fold RMSNorm and scalar AdaLN into
the rowwise activation quantizer and feed the carrier directly to fc1. The
explicit two-slice path reuses one carrier for both fc1 slices. Per-token
modulation selectors, non-BF16 execution, compile
tracing, and older binaries keep the eager path. The FP32-through mode remains
available for numerical experiments. Compare both modes against the eager
producer with:

```text
python benchmarks/bench_fused_norm_adaln_quantize.py `
  --library native/build-vsa/h3_int8_attention.dll `
  --checkpoint path/to/minimax_h3_ref2va_pruned_int8_convrot.safetensors `
  --block-index 24 --rows 4096 --hidden 5376
```

## Building

Lower VRAM's chunk-scaled V uses four optional ABI-4 entry points:
`h3_int8_dense_attention_chunked_v`, `h3_int8_packed_dense_attention_chunked_v`,
`h3_int8_sparse_attention_chunked_v`, and `h3_int8_sparse_attention_lse_chunked_v`.
They consume per-channel scales laid out as `[batch, KV heads, chunks, 128]`.
Chunks must contain a power-of-two number of rows, at least 128. The new
multi-chunk consumers support unmasked H3 attention with head dimension 128.
The original entry points and global-scale carriers remain unchanged.

The packed sparse kernels sum several routed subtiles per iteration, so an
iteration must not span two V chunks. `h3_int8_align_route_v_blocks` pads each
route row so every chunk except the last contributes whole iterations, adding
unselected tiles of that chunk (after its run, before it, then in gaps), and
writes the iterations where the chunk changes. The sparse `_chunked_v` entry
points take that list; the fine-route kernel rescales between inner loops
there instead of checking every iteration. Builds without these symbols, or
GPUs whose `selftest.chunk_scaled_v_check` fails (chunked kernels must
reproduce the global ones bit for bit when all chunk scales are equal), use
the two-pass producer for Lower VRAM instead.

GPU correctness tests are in `tests/test_chunk_scaled_v_cuda.py`. Real block
comparisons are in H3-Extended's `benchmarks/bench_chunk_scaled_v.py`; pass the
candidate pack explicitly and verify its recorded native path and hash.

```
cmake -S native -B native/build -G Ninja
cmake --build native/build --config Release
```

On Windows, run those commands from an x64 Visual Studio developer shell. The
Visual Studio CUDA MSBuild targets hardcode `/EHsc` after project flags, so the
Visual Studio generator is rejected; Ninja preserves the required `/EHs`
exception boundary described above.

The loader first uses the platform binary committed under `native/bin/`, then
checks local `native/lib/`, `native/build/`, and `native/build/Release/` paths.

To point it somewhere else, write that path into `native/library_path.txt`;
the first non-blank, non-`#` line wins and is searched ahead of everything
above. This replaces the old `H3_INT8_ATTENTION_LIBRARY` environment variable.
The Registry's automated package scanner reports any runtime `os.environ` read
as environment manipulation, and a flagged release leaves ComfyUI-Manager
pinned to the previous approved version, so the override became a file the
pack owns. It is ignored by both Git and `.comfyignore`, so it never exists in
a published install.

Architectures default to
`75-real;80-real;86-real;89-real;120f-real;89-virtual` on Windows and add
`90a-real` on Linux. Every shipped target has real SASS, while one `compute_89`
PTX payload provides the forward-compatible fallback without duplicating the
family-limited `compute_120f` PTX. The 64Q x 64KV specialization requests three
resident blocks only on the SM86, SM89, and SM120 consumer architectures;
datacenter targets retain their unconstrained register allocation. Override
with `-DH3_CUDA_ARCHS=89` to build just one while iterating.

SM75 composes the Ampere-shaped INT8 MMA fragments from Turing's m8n8k16
instructions and uses synchronous shared-memory copies where newer GPUs use
`cp.async`. The packaged path is still accepted only after its per-GPU dense
and sparse numerical self-test passes.

Linux release binaries are built in the CUDA 13 Ubuntu 22.04 image with GCC 11
and pinned CMake 3.28.3. The packaged library requires no newer than
`GLIBCXX_3.4.21` and keeps its existing `GLIBC_2.34` floor. The shipping test
rejects a binary built against a newer libstdc++ ABI. CUDA fatbins use size
compression, unused ELF sections are discarded, and the final shared library
is stripped while retaining its exported C ABI.

CUTLASS is vendored as compile-time headers. There is no external CUTLASS,
cuBLAS, flash-attention, or Python runtime dependency; the shipped shared
library needs only the CUDA runtime and driver already required by ComfyUI.
