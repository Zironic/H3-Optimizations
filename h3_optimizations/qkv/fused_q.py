"""Exact H3 fused-Q binding for the native Kitchen carrier."""

from __future__ import annotations

import torch

import comfy.model_management
from comfy.quant_ops import QuantizedTensor

from .. import diagnostics
from ..native import (
    fused_h3_q_from_int8,
    fused_h3_q_is_available,
    int8_rowwise_convrot256_is_available,
    quantize_int8_rowwise_convrot256,
)
from .formats import describe_linear
from .int8 import HeldConvRotINT8Linear
from .streamed import PROJECTION_NATIVE


HEAD_DIM = 128
CONVROT_GROUP = 256


class FusedH3QUnavailableError(RuntimeError):
    pass


def fused_h3_q_supported(module, x, rope_freqs, projection_mode):
    """Whether the exact measured producer matches this execution contract."""
    if projection_mode != PROJECTION_NATIVE:
        return False
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or rope_freqs is None
        or not isinstance(rope_freqs, torch.Tensor)
        or rope_freqs.dtype != torch.bfloat16
        or rope_freqs.device != x.device
        or int(x.shape[1]) <= 0
        or int(x.shape[1]) % CONVROT_GROUP
        or int(getattr(module, "head_dim", 0)) != HEAD_DIM
        or (int(getattr(module, "heads", 0)) * HEAD_DIM) % 256
        or not describe_linear(module.qkv_proj).convrot_int8_256
    ):
        return False
    return bool(
        int8_rowwise_convrot256_is_available()
        and fused_h3_q_is_available(x.device)
    )


class HeldExactH3FusedQ:
    """Hold the native ConvRot Q weight while producing bounded Q slabs."""

    def __init__(self, module, sample, rope_freqs, projection_mode):
        self.module = module
        self.sample = sample
        self.rope_freqs = rope_freqs
        self.projection_mode = projection_mode
        self.binding = None
        self.weight_qdata = None
        self.weight_scale = None
        self.norm = None

    def __enter__(self):
        if not fused_h3_q_supported(
            self.module,
            self.sample,
            self.rope_freqs,
            self.projection_mode,
        ):
            raise FusedH3QUnavailableError(
                "the exact H3 fused-Q producer does not support this projection"
            )

        self.binding = HeldConvRotINT8Linear(self.module.qkv_proj, self.sample)
        self.binding.__enter__()
        try:
            weight = self.binding.weight
            if not isinstance(weight, QuantizedTensor):
                raise FusedH3QUnavailableError(
                    "the exact H3 fused-Q producer requires ConvRot INT8 QKV weights"
                )
            inner = int(self.module.heads) * int(self.module.head_dim)
            if int(weight._qdata.shape[0]) < inner:
                raise FusedH3QUnavailableError("the H3 QKV weight has no complete Q slice")
            self.weight_qdata = weight._qdata[:inner].contiguous()
            scale = weight._params.scale.to(torch.float32).reshape(-1)
            if int(scale.numel()) != int(weight._qdata.shape[0]):
                raise FusedH3QUnavailableError(
                    "the H3 QKV weight must have one scale per output row"
                )
            self.weight_scale = scale[:inner].contiguous()
            self.norm = comfy.model_management.cast_to(
                self.module.q_norm.weight,
                dtype=torch.bfloat16,
                device=self.weight_qdata.device,
            ).contiguous()
            if int(self.norm.numel()) != HEAD_DIM:
                raise FusedH3QUnavailableError(
                    "the exact H3 fused-Q producer requires a 128-value Q norm"
                )
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        self.weight_qdata = None
        self.weight_scale = None
        self.norm = None
        self.sample = None
        self.rope_freqs = None
        if self.binding is not None:
            self.binding.__exit__(None, None, None)
            self.binding = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def project(self, x, rope_freqs, start, stop, full_k_length):
        if self.binding is None:
            raise RuntimeError("the exact H3 fused-Q binding is not active")
        rows = x[start:stop].contiguous()
        freqs = rope_freqs[0, start:stop, 0].contiguous()
        with diagnostics.stage('q_activation_quant'):
            activation, activation_scale = quantize_int8_rowwise_convrot256(rows)
        with diagnostics.stage('fused_q_projection'):
            return fused_h3_q_from_int8(
                activation,
                self.weight_qdata,
                activation_scale,
                self.weight_scale,
                self.norm,
                freqs,
                full_k_length=full_k_length,
                epsilon=float(self.module.q_norm.eps),
            )


__all__ = [
    "FusedH3QUnavailableError",
    "HeldExactH3FusedQ",
    "fused_h3_q_supported",
]
