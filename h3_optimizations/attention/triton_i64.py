"""Int64-offset per-thread INT8 Q/K quantization for long H3 sequences.

Derived from SageAttention 2.2's `sageattention/triton/quant_per_thread.py`:

    Copyright (c) 2024 by SageAttention team.
    Licensed under the Apache License, Version 2.0.
    http://www.apache.org/licenses/LICENSE-2.0

Modifications from the original, per Apache 2.0 §4(b):

  1. Every pointer offset is computed in `tl.int64`. The original accumulates
     `offs_n * stride_in` in int32, which wraps for strided inputs at long
     sequence lengths.
  2. The int4 kernels are omitted - they are not on the H3 path.
  3. The vestigial `sm_scale` computation is dropped; the original computes it
     and never passes it to either kernel.

Nothing else changes: identical loads, identical scale arithmetic, identical
rounding, identical stores. Below the overflow this is **bit-identical** to stock
SageAttention by construction, not by measurement - which matters, because every
existing benchmark on this machine was taken with the stock kernel.

Why it is needed
----------------
MiniMax H3's attention splits one fused QKV projection, so q and k are
non-contiguous views with a sequence stride of `heads * head_dim * 3` = 21,504.
In the original kernels `offs_n * stride_in` is int32, so the first row index
whose offset wraps is

    (2**31 - 1) // 21504 + 1 = 99,865

so a sequence must reach length 99,866 to contain that row. Reaching it produces
an invalid pointer and an illegal memory access. That is fatal rather
than recoverable: the `try`/`except` around `sageattn` in ComfyUI catches it, but
the CUDA context is already poisoned, so the process dies with
`Fatal Python error: Aborted` and takes the prompt worker with it.

On a 12 GB card OOM arrives around S=120-150k, so the window where this bites is
real but narrow: roughly C=209 upward (S=102,640). See PLAN.md §5.

This module exists so the fix lives in this repository rather than as an edit to
`site-packages`, which a `pip install -U sageattention` silently reverts.
"""

import logging

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    TRITON_AVAILABLE = False
    triton = None
    tl = None


_original = None


if TRITON_AVAILABLE:

    @triton.jit
    def quant_query_per_thread_int8_kernel(Input, Output, Scale, L,
                                           stride_iz, stride_ih, stride_in,
                                           stride_oz, stride_oh, stride_on,
                                           stride_sz, stride_sh,
                                           C: tl.constexpr, BLK: tl.constexpr):
        off_blk = tl.program_id(0) // 8
        off_tld = tl.program_id(0) % 8
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
        offs_k = tl.arange(0, C)

        # the modification: every term promoted before it is multiplied
        off_b64 = off_b.to(tl.int64)
        off_h64 = off_h.to(tl.int64)
        off_blk64 = off_blk.to(tl.int64)
        off_tld64 = off_tld.to(tl.int64)
        offs_n64 = offs_n.to(tl.int64)
        offs_k64 = offs_k.to(tl.int64)

        input_ptrs = (Input
                      + off_b64 * stride_iz.to(tl.int64)
                      + off_h64 * stride_ih.to(tl.int64)
                      + offs_n64[:, None] * stride_in.to(tl.int64)
                      + offs_k64[None, :])
        output_ptrs = (Output
                       + off_b64 * stride_oz.to(tl.int64)
                       + off_h64 * stride_oh.to(tl.int64)
                       + offs_n64[:, None] * stride_on.to(tl.int64)
                       + offs_k64[None, :])
        scale_ptrs = (Scale
                      + off_b64 * stride_sz.to(tl.int64)
                      + off_h64 * stride_sh.to(tl.int64)
                      + off_blk64 * 8
                      + off_tld64)

        x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
        x = x.to(tl.float32)
        scale = tl.max(tl.abs(x)) / 127. + 0.0000001
        x_int8 = x / scale
        x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
        x_int8 = x_int8.to(tl.int8)
        tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
        tl.store(scale_ptrs, scale)

    @triton.jit
    def quant_key_per_thread_int8_kernel(Input, Output, Scale, L,
                                         stride_iz, stride_ih, stride_in,
                                         stride_oz, stride_oh, stride_on,
                                         stride_sz, stride_sh,
                                         C: tl.constexpr, BLK: tl.constexpr):
        off_blk = tl.program_id(0) // 4
        off_tld = tl.program_id(0) % 4
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
        offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
        offs_k = tl.arange(0, C)

        off_b64 = off_b.to(tl.int64)
        off_h64 = off_h.to(tl.int64)
        off_blk64 = off_blk.to(tl.int64)
        off_tld64 = off_tld.to(tl.int64)
        offs_n0_64 = offs_n0.to(tl.int64)
        offs_n1_64 = offs_n1.to(tl.int64)
        offs_k64 = offs_k.to(tl.int64)

        input_ptrs0 = (Input
                       + off_b64 * stride_iz.to(tl.int64)
                       + off_h64 * stride_ih.to(tl.int64)
                       + offs_n0_64[:, None] * stride_in.to(tl.int64)
                       + offs_k64[None, :])
        input_ptrs1 = (Input
                       + off_b64 * stride_iz.to(tl.int64)
                       + off_h64 * stride_ih.to(tl.int64)
                       + offs_n1_64[:, None] * stride_in.to(tl.int64)
                       + offs_k64[None, :])
        output_ptrs0 = (Output
                        + off_b64 * stride_oz.to(tl.int64)
                        + off_h64 * stride_oh.to(tl.int64)
                        + offs_n0_64[:, None] * stride_on.to(tl.int64)
                        + offs_k64[None, :])
        output_ptrs1 = (Output
                        + off_b64 * stride_oz.to(tl.int64)
                        + off_h64 * stride_oh.to(tl.int64)
                        + offs_n1_64[:, None] * stride_on.to(tl.int64)
                        + offs_k64[None, :])
        scale_ptrs = (Scale
                      + off_b64 * stride_sz.to(tl.int64)
                      + off_h64 * stride_sh.to(tl.int64)
                      + off_blk64 * 4
                      + off_tld64)

        x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
        x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
        x0 = x0.to(tl.float32)
        x1 = x1.to(tl.float32)
        scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
        x0_int8 = x0 / scale
        x1_int8 = x1 / scale
        x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
        x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
        x0_int8 = x0_int8.to(tl.int8)
        x1_int8 = x1_int8.to(tl.int8)
        tl.store(output_ptrs0, x0_int8, mask=offs_n0[:, None] < L)
        tl.store(output_ptrs1, x1_int8, mask=offs_n1[:, None] < L)
        tl.store(scale_ptrs, scale)


def per_thread_int8_i64(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64,
                        sm_scale=None, tensor_layout="HND"):
    """Drop-in replacement for `sageattention.core.per_thread_int8_triton`.

    Signature matches the original exactly, including the unused `sm_scale`.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not available; cannot run the int64 Q/K quantizer")

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError("Unknown tensor layout: %s" % tensor_layout)

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8),
                          device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4),
                          device=q.device, dtype=torch.float32)

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    quant_query_per_thread_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ)

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    quant_key_per_thread_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK)

    return q_int8, q_scale, k_int8, k_scale


# --------------------------------------------------------------------------
# installation into SageAttention
# --------------------------------------------------------------------------

def first_wrapping_row(stride):
    """First row *index* whose base offset wraps signed int32 at `stride`.

    Row `r` sits at `r * stride`, so it wraps when `r * stride > 2**31 - 1`. Note
    that `(2**31 - 1) // stride` is the last *safe* index, one below this; a
    tensor of sequence length L covers rows 0..L-1 and therefore only contains a
    wrapping row when `L > first_wrapping_row(stride)`.

    At H3's fused stride of 21,504 that is row 99,865 - matching the original
    field report - and a sequence must reach length 99,866 to include it.
    """
    return (2 ** 31 - 1) // stride + 1


def wraps(shape, stride):
    """Whether a tensor's maximum linear element offset exceeds signed int32."""
    return sum((n - 1) * s for n, s in zip(shape, stride)) > 2 ** 31 - 1


def install():
    """Point SageAttention's per-thread INT8 path at the int64 kernels.

    `sageattention.core` does `from .triton.quant_per_thread import
    per_thread_int8 as per_thread_int8_triton`, so the binding to replace is the
    one in `core`, not the one in the `triton` submodule.

    This is deliberately global rather than H3-scoped. The replacement is
    bit-identical below the overflow and strictly more correct above it, so
    every model benefits and none regresses - and scoping it would mean leaving
    a known-fatal bug live on other architectures for no gain.

    Idempotent. Returns True if the swap is in place.
    """
    global _original
    if not TRITON_AVAILABLE:
        logging.warning("[H3 attention] triton unavailable; int64 Q/K quantizer not installed")
        return False
    try:
        import sageattention.core as sage_core
    except ImportError:
        return False

    current = getattr(sage_core, "per_thread_int8_triton", None)
    if current is None:
        logging.warning("[H3 attention] sageattention.core has no per_thread_int8_triton; "
                        "the package layout changed and the int64 fix was NOT applied")
        return False
    if getattr(current, "_h3_optimizations_int64", False):
        return True

    _original = current
    per_thread_int8_i64._h3_optimizations_int64 = True
    sage_core.per_thread_int8_triton = per_thread_int8_i64
    logging.debug("[H3 attention] int64 Q/K quantizer installed "
                 "(stock kernels wrap from row %d at H3's 21504 stride)",
                 first_wrapping_row(56 * 128 * 3))
    return True


def uninstall():
    global _original
    if _original is None:
        return
    try:
        import sageattention.core as sage_core
    except ImportError:
        return
    if getattr(sage_core.per_thread_int8_triton, "_h3_optimizations_int64", False):
        sage_core.per_thread_int8_triton = _original
        _original = None


def is_installed():
    try:
        import sageattention.core as sage_core
    except ImportError:
        return False
    return getattr(
        getattr(sage_core, "per_thread_int8_triton", None),
        "_h3_optimizations_int64",
        False,
    )
