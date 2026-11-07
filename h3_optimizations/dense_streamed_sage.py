"""Stream H3 Q and attention output through native Sage carrier formats."""

from dataclasses import dataclass
import math

import torch

import comfy.model_management

from . import diagnostics
from .attention import stats
from .attention.sage_mem_eff import EfficientSageError
from .attention.sage_v_staging import (
    BACKEND_TORCH as SAGE_STAGING_TORCH,
    BACKEND_TRITON as SAGE_STAGING_TRITON,
    TwoPassSageVCarrier,
)
from .normalized_rows import attention_output_buffer
from .plan import (
    V_MEMORY_RETAIN,
    V_MEMORY_TWO_PASS,
    V_MEMORY_REQUESTS,
    v_memory_without_chunk_scales,
)
from .qkv.streamed import (
    create_held_qkv,
    project_kv_hnd,
    project_q_hnd,
    project_v_hnd,
)


@dataclass
class StreamedDenseSageQKV:
    module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    k_int8: torch.Tensor | None
    k_scale: torch.Tensor | None
    v: torch.Tensor | None
    output_dtype: torch.dtype
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    query_chunk_rows: int
    projection_mode: str
    held_factory: object
    # Set only when the projector staged V in two passes; the backend then
    # skips its own quantization instead of re-preparing a carrier.
    staged_v_carrier: torch.Tensor | None = None
    staged_v_scale: torch.Tensor | None = None

    def release(self):
        self.module = None
        self.x = None
        self.rope_freqs = None
        self.k_int8 = None
        self.k_scale = None
        self.v = None
        self.staged_v_carrier = None
        self.staged_v_scale = None
        self.held_factory = None


@dataclass
class PreparedStreamedDenseSage:
    projected: StreamedDenseSageQKV
    v_carrier: torch.Tensor | None
    v_scale: torch.Tensor | None

    def release(self):
        self.projected.release()
        self.v_carrier = None
        self.v_scale = None


class StreamedDenseSageQKVProjector:
    name = "streamed_dense_sage_qkv"
    streamed_q = True
    consumer_native_carrier = True

    def __init__(
        self,
        backend,
        *,
        chunk_rows,
        projection_mode,
        v_mode=V_MEMORY_RETAIN,
        held_factory=create_held_qkv,
        allow_cpu_for_tests=False,
    ):
        self.backend = backend
        self.chunk_rows = int(chunk_rows)
        self.projection_mode = projection_mode
        if v_mode not in V_MEMORY_REQUESTS:
            raise ValueError('unknown dense Sage V mode %r' % v_mode)
        self.requested_v_mode = v_mode
        # Sage carriers have no chunk scales; Lower VRAM stages V in two passes.
        v_mode = v_memory_without_chunk_scales(v_mode)
        # Resolve the request against what this architecture can actually do,
        # so v_mode always names what will happen. Status reads this field, and
        # a request the backend declined must not read back as granted.
        parameters = getattr(backend, "v_staging_parameters", None)
        self._staging_parameters = None if parameters is None else parameters()
        self.v_mode = (
            V_MEMORY_TWO_PASS
            if v_mode == V_MEMORY_TWO_PASS and self._staging_parameters is not None
            else V_MEMORY_RETAIN
        )
        self.held_factory = held_factory
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        q_tile = int(backend.projected_q_tile)
        k_tile = int(backend.projected_k_tile)
        alignment = math.lcm(q_tile, k_tile)
        if self.chunk_rows <= 0 or self.chunk_rows % alignment:
            raise ValueError(
                "dense Sage stream rows must be a positive multiple of %d"
                % alignment
            )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.chunk_rows,
            self.projection_mode,
            self.v_mode,
            getattr(self.backend, "name", type(self.backend).__name__),
            int(self.backend.projected_q_tile),
            int(self.backend.projected_k_tile),
        )

    def bind(self, module):
        return None

    def _staging(self, sequence, heads, x):
        """Build a two-pass V carrier, or None to keep retaining BF16 V.

        The delegate decides whether a second pass can pay: an architecture
        whose V carrier is FP16 is the same size as the BF16 source and
        declines by returning None, so the request degrades to Standard
        instead of costing a reprojection for nothing.
        """
        if self.v_mode != V_MEMORY_TWO_PASS:
            return None
        scale_max, pad_to = self._staging_parameters
        return TwoPassSageVCarrier(
            1,
            heads,
            sequence,
            128,
            scale_max=scale_max,
            device=x.device,
            dtype=x.dtype,
            pad_to=pad_to,
            backend=(
                SAGE_STAGING_TORCH
                if self.allow_cpu_for_tests
                else SAGE_STAGING_TRITON
            ),
        )

    def project(self, module, x, rope_freqs, *, layer_index, transformer_options):
        del transformer_options
        if x.ndim != 2 or x.dtype not in (torch.float16, torch.bfloat16):
            raise EfficientSageError(
                "streamed dense Sage requires rank-2 BF16/FP16 activations"
            )
        if not self.allow_cpu_for_tests and not x.is_cuda:
            raise EfficientSageError("streamed dense Sage requires CUDA")
        if comfy.model_management.in_training:
            raise EfficientSageError("streamed dense Sage is inference-only")
        if int(module.head_dim) != 128:
            raise EfficientSageError("streamed dense Sage requires head_dim 128")

        sequence = int(x.shape[0])
        heads = int(module.heads)
        k_int8 = torch.empty(
            (1, heads, sequence, 128),
            dtype=torch.int8,
            device=x.device,
        )
        staging = self._staging(sequence, heads, x)
        staged_v_carrier = None
        staged_v_scale = None
        v = (
            None
            if staging is not None
            else torch.empty(
                (1, heads, sequence, 128),
                dtype=x.dtype,
                device=x.device,
            )
        )
        k_scales = []
        held = self.held_factory(module, x[:1], self.projection_mode)
        held.__enter__()
        try:
            for start in range(0, sequence, self.chunk_rows):
                stop = min(start + self.chunk_rows, sequence)
                k, chunk_v = project_kv_hnd(
                    held,
                    x,
                    rope_freqs,
                    start,
                    stop,
                )
                chunk_k, chunk_scale = self.backend.quantize_projected_k(k)
                k_int8[..., start:stop, :].copy_(chunk_k)
                k_scales.append(chunk_scale)
                if staging is None:
                    v[..., start:stop, :].copy_(chunk_v)
                else:
                    with diagnostics.stage("v_amax_update"):
                        staging.update(chunk_v)
                del k, chunk_v, chunk_k, chunk_scale
            if staging is not None:
                # Second pass: V only. K is already packed, so this reprojects
                # strictly the rows the carrier still needs.
                staging.finalize_scale()
                for start in range(0, sequence, self.chunk_rows):
                    stop = min(start + self.chunk_rows, sequence)
                    with diagnostics.stage("v_reprojection"):
                        chunk_v = project_v_hnd(held, x, rope_freqs, start, stop)
                    with diagnostics.stage("v_carrier_pack"):
                        staging.quantize(chunk_v, start)
                    del chunk_v
                staged_v_carrier, staged_v_scale = staging.finish()
        finally:
            held.__exit__(None, None, None)

        return StreamedDenseSageQKV(
            module=module,
            x=x,
            rope_freqs=rope_freqs,
            k_int8=k_int8,
            k_scale=torch.cat(k_scales, dim=-1).contiguous(),
            v=v,
            staged_v_carrier=staged_v_carrier,
            staged_v_scale=staged_v_scale,
            output_dtype=x.dtype,
            sequence=sequence,
            heads=heads,
            head_dim=128,
            layer_index=int(layer_index),
            query_chunk_rows=self.chunk_rows,
            projection_mode=self.projection_mode,
            held_factory=self.held_factory,
        )


class StreamedDenseSageBackend:
    def __init__(self, delegate):
        self.delegate = delegate
        self.name = delegate.name
        self.requires_registered_sage = getattr(
            delegate,
            "requires_registered_sage",
            True,
        )
        self.requires_runtime_context = getattr(
            delegate,
            "requires_runtime_context",
            False,
        )
        self.approximate = False
        self.runtime_listeners = tuple(getattr(delegate, "runtime_listeners", ()))

    @property
    def installation_signature(self):
        return (
            self.name,
            "streamed_dense_sage",
            getattr(self.delegate, "projected_q_tile", None),
            getattr(self.delegate, "projected_k_tile", None),
        )

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        return self.delegate.prepare(
            q,
            k,
            v,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )

    def execute(self, prepared):
        return self.delegate.execute(prepared)

    def requires_fallback_inputs(self, prepared):
        check = getattr(self.delegate, "requires_fallback_inputs", None)
        return False if check is None else bool(check(prepared))

    def prepare_projected(self, projected, *, layer_index, transformer_options):
        del transformer_options
        if not isinstance(projected, StreamedDenseSageQKV):
            raise EfficientSageError("dense Sage received an invalid streamed carrier")
        if int(projected.layer_index) != int(layer_index):
            raise EfficientSageError(
                "streamed dense Sage layer %d does not match attention layer %d"
                % (projected.layer_index, layer_index)
            )
        stats.observe_sequence(projected.sequence)
        if projected.staged_v_carrier is not None:
            v_carrier = projected.staged_v_carrier
            v_scale = projected.staged_v_scale
            projected.staged_v_carrier = None
            projected.staged_v_scale = None
        else:
            v_carrier, v_scale = self.delegate.prepare_streamed_v(projected.v)
        projected.v = None
        return PreparedStreamedDenseSage(projected, v_carrier, v_scale)

    def execute_projected(self, module, prepared):
        if not isinstance(prepared, PreparedStreamedDenseSage):
            return None
        projected = prepared.projected
        source_module = getattr(module, "_module", module)
        if source_module is not projected.module:
            raise EfficientSageError(
                "streamed dense Sage module changed between prepare and execute"
            )

        result = attention_output_buffer(projected.x)
        hidden = int(result.shape[1])
        try:
            for start in range(0, projected.sequence, projected.query_chunk_rows):
                stop = min(start + projected.query_chunk_rows, projected.sequence)
                held = projected.held_factory(
                    module,
                    projected.x[start:start + 1],
                    projected.projection_mode,
                )
                held.__enter__()
                try:
                    q = project_q_hnd(
                        held,
                        projected.x,
                        projected.rope_freqs,
                        start,
                        stop,
                    )
                finally:
                    held.__exit__(None, None, None)
                q_int8, q_scale = self.delegate.quantize_projected_q(q)
                del q
                raw = self.delegate.execute_rectangular(
                    q_int8,
                    q_scale,
                    projected.k_int8,
                    projected.k_scale,
                    prepared.v_carrier,
                    prepared.v_scale,
                    output_dtype=projected.output_dtype,
                    softmax_scale=projected.head_dim ** -0.5,
                    layer_index=projected.layer_index,
                )
                del q_int8, q_scale
                if stop == projected.sequence:
                    projected.k_int8 = None
                    projected.k_scale = None
                    prepared.v_carrier = None
                    prepared.v_scale = None
                rows = stop - start
                flat = raw.transpose(1, 2).reshape(
                    rows,
                    projected.heads * projected.head_dim,
                )
                del raw
                with diagnostics.stage("attention_out"):
                    projected_rows = module.out_proj(flat)
                    if tuple(projected_rows.shape) != (rows, hidden):
                        raise EfficientSageError(
                            "streamed dense Sage out_proj returned %s"
                            % (tuple(projected_rows.shape),)
                        )
                    result[start:stop].copy_(projected_rows)
                del flat, projected_rows
            return result
        finally:
            prepared.release()

    def as_status(self):
        status = getattr(self.delegate, "as_status", None)
        data = {} if status is None else dict(status())
        data.update(
            {
                "backend": self.name,
                "streamed_q": True,
                "streamed_attention_output": True,
            }
        )
        return data
