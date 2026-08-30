"""Dense SM89 Sage backend adapter for projected fused-QKV carriers."""

from __future__ import annotations

import logging

import torch

from .attention import stats
from .attention.sage_mem_eff import (
    EfficientSageError,
    PreparedSM89,
    guard_v_stride,
)
from .attention.sage_v_fp8 import prepare_sage_v_fp8

from .dense_fused_qkv import (
    DENSE_QK_FORMAT,
    DenseFusedQKVError,
    validate_prepared_dense_fused_qkv,
)


class ProjectedSM89SageBackend:
    """Extend the existing dense backend with a no-requantization entrypoint."""

    name = "sage_mem_eff"
    requires_registered_sage = True
    requires_runtime_context = False
    approximate = False
    projected_qkv_format = DENSE_QK_FORMAT

    def __init__(self, delegate):
        if getattr(delegate, "name", None) != self.name:
            raise TypeError("ProjectedSM89SageBackend requires sage_mem_eff")
        self.delegate = delegate
        self.api = delegate.api
        self.allow_cpu_for_tests = bool(delegate.allow_cpu_for_tests)
        self.runtime_listeners = tuple(
            getattr(delegate, "runtime_listeners", ())
        )
        self._projected_logged = False

    @property
    def installation_signature(self):
        api = self.api
        return (
            self.name,
            "projected_dense_qkv",
            str(getattr(api, "version", "unknown")),
            str(getattr(api, "kernel_name", "unknown")),
            self.projected_qkv_format,
        )

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        return self.delegate.prepare(
            q,
            k,
            v,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )

    def prepare_projected(self, projected, *, layer_index, transformer_options):
        try:
            validate_prepared_dense_fused_qkv(projected)
        except DenseFusedQKVError as exc:
            raise EfficientSageError(str(exc)) from exc

        if int(projected.layer_index) != int(layer_index):
            raise EfficientSageError(
                "dense fused QKV layer %d does not match attention layer %d"
                % (projected.layer_index, layer_index)
            )
        if not self.allow_cpu_for_tests:
            if not projected.q_int8.is_cuda:
                raise EfficientSageError("projected dense Sage requires CUDA")
            capability = tuple(
                torch.cuda.get_device_capability(projected.q_int8.device)
            )
            if capability != (8, 9):
                raise EfficientSageError(
                    "projected dense Sage is SM89-only; device capability is %d.%d"
                    % capability
                )
        v_fp8, v_scale = prepare_sage_v_fp8(
            guard_v_stride(projected.v),
            self.api.per_channel_fp8,
            scale_max=self.api.v_scale_max,
        )

        stats.observe_sequence(projected.sequence)
        if not self._projected_logged:
            logging.debug(
                "[H3 attention] sage_mem_eff projected QKV active: "
                "SageAttention %s, per-thread INT8 Q/K, fused projection, "
                "accumulation=%s, kernel=%s via %s",
                self.api.version,
                self.api.accumulation,
                self.api.kernel_name,
                self.api.kernel_source,
            )
            self._projected_logged = True

        return PreparedSM89(
            q_int8=projected.q_int8,
            q_scale=projected.q_scale,
            k_int8=projected.k_int8,
            k_scale=projected.k_scale,
            v_fp8=v_fp8,
            v_scale=v_scale,
            output_dtype=projected.output_dtype,
            layer_index=int(layer_index),
            sequence=int(projected.sequence),
            heads=int(projected.heads),
            head_dim=int(projected.head_dim),
            softmax_scale=int(projected.head_dim) ** -0.5,
            kernel=self.api.kernel,
            kernel_name=self.api.kernel_name,
        )

    def execute(self, prepared):
        return self.delegate.execute(prepared)

    def as_status(self):
        return {
            "backend": self.name,
            "projected_qkv_format": self.projected_qkv_format,
            "sageattention": str(getattr(self.api, "version", "unknown")),
            "kernel": str(getattr(self.api, "kernel_name", "unknown")),
        }
