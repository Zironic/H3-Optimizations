"""H3-owned exact 128x256 fused Kitchen Q producer."""

from __future__ import annotations

import math

import torch

from . import loader


_SYMBOL = "h3_int8_fused_q"
_HEAD_DIM = 128
_TILE_M = 128
_TILE_N = 256
_ROPE_PAIRS = 48
_MAX_INT = 2**31 - 1


def fused_h3_q_is_available(device=None):
    """Whether this CUDA device can use the fixed CUTLASS producer."""
    if not torch.cuda.is_available():
        return False
    try:
        if tuple(torch.cuda.get_device_capability(device)) < (8, 0):
            return False
        library = loader.load()
    except (loader.NativeUnavailableError, RuntimeError):
        return False
    if getattr(library, _SYMBOL, None) is None:
        return False
    from . import selftest

    return selftest.fused_q_check(device)


def _tensor(name, value, *, dtype, device, dimensions=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor" % name)
    if value.dtype != dtype:
        raise TypeError("%s must have dtype %s, got %s" % (name, dtype, value.dtype))
    if not value.is_cuda:
        raise ValueError("%s must be a CUDA tensor" % name)
    if value.device != device:
        raise ValueError("%s must be on %s" % (name, device))
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError("%s must have %d dimensions" % (name, dimensions))
    if not value.is_contiguous():
        raise ValueError("%s must be contiguous" % name)


def fused_h3_q_from_int8(
    activation,
    weight,
    activation_scale,
    weight_scale,
    norm,
    freqs,
    *,
    full_k_length,
    epsilon,
):
    """Produce Kitchen Q bytes, scales, and 64-row routing summaries."""
    if not isinstance(activation, torch.Tensor):
        raise TypeError("activation must be a torch.Tensor")
    if activation.ndim != 2:
        raise ValueError("activation must be two-dimensional")
    device = activation.device
    _tensor(
        "activation",
        activation,
        dtype=torch.int8,
        device=device,
        dimensions=2,
    )
    _tensor("weight", weight, dtype=torch.int8, device=device, dimensions=2)
    _tensor(
        "activation_scale",
        activation_scale,
        dtype=torch.float32,
        device=device,
        dimensions=2,
    )
    _tensor(
        "weight_scale",
        weight_scale,
        dtype=torch.float32,
        device=device,
        dimensions=1,
    )
    _tensor("norm", norm, dtype=torch.bfloat16, device=device)
    _tensor("freqs", freqs, dtype=torch.bfloat16, device=device)

    rows, hidden = (int(value) for value in activation.shape)
    outputs, weight_hidden = (int(value) for value in weight.shape)
    if not 0 < rows <= _MAX_INT or not 0 < hidden <= _MAX_INT:
        raise ValueError("activation dimensions exceed the fused-Q kernel limits")
    if not 0 < outputs <= _MAX_INT or weight_hidden != hidden:
        raise ValueError("weight must be [N, K] with the activation K dimension")
    if outputs % _TILE_N:
        raise ValueError("Q output width must be divisible by %d" % _TILE_N)
    if tuple(activation_scale.shape) != (rows, 1):
        raise ValueError("activation_scale must have shape [M, 1]")
    if int(weight_scale.numel()) != outputs:
        raise ValueError("weight_scale must contain one value per output row")
    if int(norm.numel()) != _HEAD_DIM:
        raise ValueError("norm must contain %d BF16 values" % _HEAD_DIM)
    if int(freqs.numel()) != rows * _ROPE_PAIRS * 2 * 2:
        raise ValueError("freqs must contain M*48*2*2 BF16 values")
    full_k_length = int(full_k_length)
    if not 0 < full_k_length <= _MAX_INT:
        raise ValueError("full_k_length must be a positive 32-bit integer")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if tuple(torch.cuda.get_device_capability(device)) < (8, 0):
        raise loader.NativeUnavailableError(
            "the exact H3 fused-Q producer requires CUDA capability 8.0 or newer"
        )

    library = loader.load()
    function = getattr(library, _SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError(
            "the loaded ABI-4 native library has no exact H3 fused-Q producer; "
            "rebuild native/"
        )

    heads = outputs // _HEAD_DIM
    q_scale_count = ((rows + _TILE_M - 1) // _TILE_M) * 32
    summary_tiles = (rows + 63) // 64
    q = torch.empty((1, heads, rows, _HEAD_DIM), dtype=torch.int8, device=device)
    q_scale = torch.empty(
        (1, heads, q_scale_count), dtype=torch.float32, device=device
    )
    summary = torch.empty(
        (1, heads, summary_tiles, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    loader.check(
        function(
            activation.data_ptr(),
            weight.data_ptr(),
            activation_scale.data_ptr(),
            weight_scale.data_ptr(),
            norm.data_ptr(),
            freqs.data_ptr(),
            summary.data_ptr(),
            q.data_ptr(),
            q_scale.data_ptr(),
            rows,
            outputs,
            hidden,
            full_k_length,
            epsilon,
            torch.cuda.current_stream(device).cuda_stream,
        ),
        "fused_h3_q_exact_128x256",
    )
    return q, q_scale, summary


__all__ = ["fused_h3_q_from_int8", "fused_h3_q_is_available"]
