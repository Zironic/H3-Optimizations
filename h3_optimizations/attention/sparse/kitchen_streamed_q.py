"""Streamed-query composition for Sparse Kitchen.

Sparse Sage, Triton, and FROST already use the same lifetime contract:
retain global K/V plus K routing summaries, then project one Q slab, build only
that slab's route, execute it, and discard Q immediately.  Sparse Kitchen used
to be the exception: it retained a full INT8 Q carrier and a full Q-summary
array before consuming query slabs.

This module closes that gap without changing Kitchen's carrier math.  It hooks
the existing Kitchen projector only for the production sparse composition
(`routing_summaries + stream_output`); dense Kitchen and the legacy chunked
producer keep their existing implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import math

import torch

import comfy.model_management

from ... import diagnostics
from ...normalized_rows import attention_output_buffer
from ...kitchen_qkv import (
    CHUNK_ROWS,
    PRODUCER_ABI_VERSION,
    V_MODE_TWO_PASS,
    V_MODE_CHUNK_SCALED,
    ChunkedKitchenQKVProjector,
    _native_bf16_format,
    _project_anchor_samples,
    _quantize_q_chunk as _pack_q_chunk,
    _qk_chunk_kwargs,
    _supports_streamed_producer,
    producer_api_available,
    resolve_kitchen,
)
from ...native import threshold_route as native_threshold_route
from ...native.int8_attention import carrier_kv_tile
from ...qkv.formats import describe_linear
from ...qkv.fused_q import HeldExactH3FusedQ, fused_h3_q_supported
from ...qkv.streamed import (
    PROJECTION_FORCE_BF16,
    PROJECTION_FORCE_FP8,
    PROJECTION_FORCE_INT8,
    PROJECTION_NATIVE,
    create_held_qkv,
    project_kv_hnd,
    project_q_hnd,
    project_v_hnd,
)
from .config import resolve_video_budget
from .kitchen_sparse import (
    OUTPUT_NHD,
    SparseKitchenBackend as _BaseSparseKitchenBackend,
    SparseKitchenError,
    route_metadata,
    snapshot_for,
)
from .router import SparseRouterError


@dataclass
class StreamedSparseKitchenQKV:
    module: object
    producer_module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    carrier: object
    k_summary: torch.Tensor | None
    projection_mode: str
    output_buffer: torch.Tensor | None
    fused_q: bool = False

    def release(self):
        self.module = None
        self.producer_module = None
        self.x = None
        self.rope_freqs = None
        self.carrier = None
        self.k_summary = None
        self.output_buffer = None
        self.fused_q = False


# Route selection mode, read per request from transformer_options (so a
# benchmark or experiment can set it on a patched model). 'topk' keeps exactly
# the budget's tiles per row; 'threshold' keeps tiles scoring above the row's
# mean + tau * std, tau from the budget (native/threshold_route.py), and falls
# back to top-k where the native router is unavailable.
ROUTE_MODE_KEY = "h3_sparse_route_mode"
ROUTE_MODE_TOPK = "topk"
ROUTE_MODE_THRESHOLD = "threshold"
DEFAULT_ROUTE_MODE = ROUTE_MODE_THRESHOLD
DENSE_QUERY_KEY = "h3_dense_query_dispatch"


@dataclass
class _StreamedRoutePlan:
    geometry: object
    retained: int
    k_summary: torch.Tensor | None
    batch: int
    heads: int
    mode: str = ROUTE_MODE_TOPK
    k_mean: torch.Tensor | None = None
    k_cov: torch.Tensor | None = None
    tau: float = 0.0

    def release(self):
        self.k_summary = None
        self.k_mean = None
        self.k_cov = None


@dataclass
class PreparedStreamedSparseKitchen:
    projected: StreamedSparseKitchenQKV
    route_plan: _StreamedRoutePlan | None
    metadata: dict
    dense_query_rows: int = 0

    def release(self):
        if self.projected is not None:
            self.projected.release()
        if self.route_plan is not None:
            self.route_plan.release()
        self.route_plan = None


def _tile_mean(x, tile):
    sequence = int(x.shape[-2])
    tile = int(tile)
    full = sequence // tile
    remainder = sequence % tile
    pieces = []
    if full:
        pieces.append(
            x[..., : full * tile, :]
            .reshape(*x.shape[:-2], full, tile, x.shape[-1])
            .mean(dim=-2)
        )
    if remainder:
        pieces.append(x[..., full * tile :, :].mean(dim=-2, keepdim=True))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def _projection_mode(projector):
    if projector.force_weights_bf16:
        return PROJECTION_FORCE_BF16
    if projector.fp8_projection:
        return PROJECTION_FORCE_FP8
    if projector.convrot_int8_projection:
        return PROJECTION_FORCE_INT8
    return PROJECTION_NATIVE


def _format_supported(projector, fmt):
    native_bf16 = _native_bf16_format(fmt)
    if projector.force_weights_bf16:
        return bool(
            getattr(fmt, "plain_float", False)
            or getattr(fmt, "convrot_int8_256", False)
            or getattr(fmt, "w4a8", False)
            or getattr(fmt, "fp8", False)
        )
    if projector.convrot_int8_projection:
        return bool(
            getattr(fmt, "convrot_int8_256", False)
            or getattr(fmt, "plain_float", False)
        )
    if projector.fp8_projection:
        return bool(
            getattr(fmt, "fp8", False)
            or getattr(fmt, "plain_float", False)
        )
    return bool(
        getattr(fmt, "convrot_int8_256", False)
        or getattr(fmt, "w4a8", False)
        or native_bf16
    )


def _run_streamed_sparse_kitchen_qkv(
    projector,
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
):
    del layer_index
    kitchen = resolve_kitchen(x.device)
    if kitchen is None or not _supports_streamed_producer(kitchen, x.device):
        return None

    fmt = describe_linear(module.qkv_proj)
    if (
        torch.compiler.is_compiling()
        or comfy.model_management.in_training
        or x.ndim != 2
        or not x.is_cuda
        or not _format_supported(projector, fmt)
        or not producer_api_available(kitchen, x.device)
    ):
        return None

    q_tile = int(projector.q_tile or 64)
    # kv_tile is the route tile; a 32-token route walks the 64-token carrier.
    kv_tile = int(projector.kv_tile or 64)
    carrier_tile = carrier_kv_tile(kv_tile)
    chunk_rows = int(projector.chunk_rows)
    alignment = math.lcm(q_tile, carrier_tile)
    if chunk_rows <= 0 or chunk_rows % alignment:
        return None

    sequence = int(x.shape[0])
    projection_mode = _projection_mode(projector)
    fused_q = bool(
        q_tile == 64
        and fused_h3_q_supported(module, x[:1], rope_freqs, projection_mode)
    )
    shape = (1, int(module.heads), sequence, int(module.head_dim))
    q_shape = (1, int(module.heads), 1, int(module.head_dim))
    try:
        spec = kitchen.int8_attention_producer_spec(
            q_shape,
            shape,
            dtype=x.dtype,
            device=x.device,
            cta_k=carrier_tile,
        )
    except kitchen.Int8AttentionProducerUnavailableError:
        return None
    if (
        getattr(spec, "abi_version", None) != PRODUCER_ABI_VERSION
        or chunk_rows % int(spec.sequence_alignment)
    ):
        return None

    held = create_held_qkv(module, x[:1], projection_mode)
    held.__enter__()
    try:
        with diagnostics.stage("anchor_projection"):
            samples = _project_anchor_samples(
                module,
                x,
                rope_freqs,
                spec.k_anchor_positions,
                projector=held,
            )
        anchor = kitchen.select_int8_attention_k_anchor(spec, samples)
        del samples
        producer = kitchen.create_int8_attention_producer(spec, anchor)
        del anchor

        staging = None
        chunked_v = None
        if projector.v_mode == V_MODE_TWO_PASS:
            from ...native.v_staging import TwoPassVCarrier

            staging = TwoPassVCarrier(spec)
        elif projector.v_mode == V_MODE_CHUNK_SCALED:
            from ...native.v_staging import ChunkScaledVCarrier

            chunked_v = ChunkScaledVCarrier(spec, chunk_rows)
        retained_v = None
        kv_tiles = (sequence + kv_tile - 1) // kv_tile
        k_summary = x.new_empty(
            (1, int(module.heads), kv_tiles, int(module.head_dim))
        )
        chunk_kwargs = _qk_chunk_kwargs(kitchen, projector.strided_qk_input)

        for start in range(0, sequence, chunk_rows):
            end = min(start + chunk_rows, sequence)
            k, v = project_kv_hnd(held, x, rope_freqs, start, end)
            if staging is None and chunked_v is None and retained_v is None:
                retained_v = v.new_empty(shape)
            kitchen.quantize_int8_attention_k_chunk(
                producer,
                k,
                k_start=start,
                **chunk_kwargs,
            )
            k_mean = _tile_mean(k, kv_tile)
            k_start = start // kv_tile
            k_summary[
                ..., k_start : k_start + int(k_mean.shape[-2]), :
            ].copy_(k_mean)
            if chunked_v is not None:
                with diagnostics.stage("v_carrier_pack"):
                    chunked_v.quantize(v, start)
            elif staging is None:
                retained_v[..., start:end, :].copy_(v)
            else:
                with diagnostics.stage("v_amax_update"):
                    staging.update(v)
            del k_mean, k, v

        if chunked_v is not None:
            producer.v, producer.v_scale = chunked_v.finish()
            producer.v_scale_rows = chunked_v.block_rows
        elif staging is None:
            kitchen.quantize_int8_attention_v(producer, retained_v)
            del retained_v
        else:
            staging.finalize_scale()
            for start in range(0, sequence, chunk_rows):
                end = min(start + chunk_rows, sequence)
                with diagnostics.stage("v_reprojection"):
                    v = project_v_hnd(held, x, rope_freqs, start, end)
                with diagnostics.stage("v_carrier_pack"):
                    staging.quantize(v, start)
                del v
            producer.v, producer.v_scale = staging.finish()
        carrier = kitchen.finalize_int8_attention_producer(producer)
    finally:
        held.__exit__(None, None, None)

    return StreamedSparseKitchenQKV(
        module=module,
        producer_module=kitchen,
        x=x,
        rope_freqs=rope_freqs,
        carrier=carrier,
        k_summary=k_summary,
        projection_mode=projection_mode,
        output_buffer=x,
        fused_q=fused_q,
    )


def _prepare_route_plan(router, k_summary, layout, video_budget, mode=ROUTE_MODE_TOPK):
    geometry = router.geometry(layout)
    if tuple(k_summary.shape[-2:]) != (
        geometry.kv_tiles,
        k_summary.shape[-1],
    ):
        raise SparseRouterError("K router summary shape does not match layout")
    retained = router._retained(video_budget, geometry)
    metadata = router._metadata(geometry, video_budget, retained)
    plan = _StreamedRoutePlan(
        geometry=geometry,
        retained=retained,
        k_summary=k_summary,
        batch=int(k_summary.shape[0]),
        heads=int(k_summary.shape[1]),
    )
    if (
        mode == ROUTE_MODE_THRESHOLD
        and 0 < retained < geometry.pure_video_kv_tiles
        and k_summary.dtype == torch.bfloat16
        and native_threshold_route.threshold_route_is_available(k_summary.device)
    ):
        plan.mode = ROUTE_MODE_THRESHOLD
        plan.k_mean, plan.k_cov = native_threshold_route.video_statistics(
            k_summary[..., geometry.pure_video_kv_start :, :]
        )
        plan.tau = native_threshold_route.budget_tau(
            retained / geometry.pure_video_kv_tiles
        )
    return plan, metadata


def _build_threshold_route_chunk(plan, q_summary, *, tile_start):
    geometry = plan.geometry
    return native_threshold_route.threshold_route(
        q_summary,
        plan.k_summary,
        plan.k_mean,
        plan.k_cov,
        pure_video_kv_start=geometry.pure_video_kv_start,
        dense_rows=geometry.pure_video_q_start - tile_start,
        tau=plan.tau,
    )


def _build_route_chunk(router, plan, q_summary, *, tile_start):
    geometry = plan.geometry
    tile_start = int(tile_start)
    tile_count = int(q_summary.shape[-2])
    tile_end = tile_start + tile_count
    if (
        plan.k_summary is None
        or q_summary.ndim != 4
        or tuple(q_summary.shape[:2]) != (plan.batch, plan.heads)
        or q_summary.shape[-1] != plan.k_summary.shape[-1]
        or q_summary.device != plan.k_summary.device
        or tile_count <= 0
        or not 0 <= tile_start < tile_end <= geometry.q_tiles
    ):
        raise SparseRouterError("Q router summary chunk does not match route plan")
    if plan.mode == ROUTE_MODE_THRESHOLD:
        return _build_threshold_route_chunk(plan, q_summary, tile_start=tile_start)

    dense = torch.arange(
        geometry.kv_tiles, device=q_summary.device, dtype=torch.int32
    )
    dense_delta = torch.cat((dense[:1], dense[1:] - dense[:-1]))
    lut = dense_delta.view(1, 1, 1, -1).expand(
        plan.batch, plan.heads, tile_count, -1
    ).clone()
    valid = torch.full(
        (plan.batch, plan.heads, tile_count),
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q_summary.device,
    )

    sparse_start = max(tile_start, geometry.pure_video_q_start)
    if plan.retained < geometry.pure_video_kv_tiles and sparse_start < tile_end:
        local_start = sparse_start - tile_start
        indices = router._select_indices(
            q_summary[..., local_start:, :],
            plan.k_summary[..., geometry.pure_video_kv_start :, :],
            plan.retained,
        )
        sparse_rows = router._pack_rows(
            indices,
            geometry,
            dense,
            dense_delta,
        )
        lut[..., local_start:, : sparse_rows.shape[-1]].copy_(sparse_rows)
        valid[..., local_start:] = (
            geometry.pure_video_kv_start + plan.retained
        )
    return lut.contiguous(), valid.contiguous()


def _quantize_q_chunk(kitchen, global_carrier, q):
    return _pack_q_chunk(kitchen, global_carrier, q)


def _query_spans(sequence, chunk_rows, dense_rows):
    for start in range(0, sequence, chunk_rows):
        stop = min(start + chunk_rows, sequence)
        if start < dense_rows < stop:
            yield start, dense_rows
            start = dense_rows
        yield start, stop


class StreamedSparseKitchenBackend(_BaseSparseKitchenBackend):
    """Sparse Kitchen with the same lazy-Q route lifecycle as other backends."""

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        if not isinstance(projected, StreamedSparseKitchenQKV):
            return super().prepare_projected(
                projected,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )

        sequence = int(projected.x.shape[0])
        snapshot = snapshot_for(transformer_options, sequence)
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            with diagnostics.stage("sparse_route"):
                route_plan, mask_metadata = _prepare_route_plan(
                    self.router,
                    projected.k_summary,
                    snapshot.layout,
                    video_budget,
                    mode=(transformer_options or {}).get(
                        ROUTE_MODE_KEY, DEFAULT_ROUTE_MODE
                    ),
                )
        except SparseRouterError as exc:
            projected.release()
            raise SparseKitchenError("sparse routing failed: %s" % exc) from exc

        projected.k_summary = None
        dense_query_rows = 0
        dense_available = getattr(projected.producer_module, "packed_dense_attention_is_available", None)
        if (
            (transformer_options or {}).get(DENSE_QUERY_KEY, False)
            and self.executor.q_tile == 64
            and projected.carrier.cta_k == 64
            and projected.carrier.k.shape[-1] == 128
            and dense_available is not None
            and dense_available(projected.carrier.k.device)
        ):
            geometry = route_plan.geometry
            dense_query_rows = (
                sequence if route_plan.retained == geometry.pure_video_kv_tiles
                else min(sequence, geometry.pure_video_q_start * 64)
            )
        metadata = route_metadata(
            mask_metadata,
            layer_index,
            int(projected.carrier.k.shape[1]),
        )
        metadata.update(
            {
                "qkv_lifetime": "streamed_q_global_kitchen_kv",
                "route_mode": route_plan.mode,
                "dense_query_rows": dense_query_rows,
                "router_lifetime": "k_summary_q_slab_selection_lazy_kitchen_lut",
                "attention_output": "chunked_out_proj_inplace",
                "q_producer": (
                    "h3_native_exact_128x256_fused"
                    if projected.fused_q
                    else "project_norm_rope_then_pack"
                ),
            }
        )
        return PreparedStreamedSparseKitchen(
            projected=projected,
            route_plan=route_plan,
            metadata=metadata,
            dense_query_rows=dense_query_rows,
        )

    def execute_projected(self, module, prepared):
        if not isinstance(prepared, PreparedStreamedSparseKitchen):
            return super().execute_projected(module, prepared)

        projected = prepared.projected
        if projected.module is not module and getattr(module, "_module", module) is not projected.module:
            prepared.release()
            raise SparseKitchenError(
                "streamed Sparse Kitchen module changed between prepare and execute"
            )
        if prepared.route_plan is None:
            raise SparseKitchenError("streamed Sparse Kitchen route was released")

        kitchen = self.executor.kitchen
        producer_module = projected.producer_module
        if producer_module is None:
            raise SparseKitchenError("streamed Sparse Kitchen producer was released")
        residual_source = projected.output_buffer
        project_residual = getattr(module, "_h3_out_proj_residual", None)
        in_place_residual = bool(
            callable(project_residual)
            and getattr(residual_source, "attention_residual_pending", False)
        )
        output = (
            None
            if in_place_residual
            else attention_output_buffer(residual_source)
        )
        if output is None and not in_place_residual:
            raise SparseKitchenError(
                "streamed Sparse Kitchen requires an output-capturing projector"
            )

        sequence = int(projected.x.shape[0])
        q_tile = int(self.executor.q_tile)
        query_rows = int(self.query_chunk_rows)
        if query_rows <= 0 or query_rows % q_tile:
            raise SparseKitchenError(
                "streamed Sparse Kitchen query rows must align to Q tiles"
            )

        fused = None
        try:
            if projected.fused_q:
                fused = HeldExactH3FusedQ(
                    projected.module,
                    projected.x[:1],
                    projected.rope_freqs,
                    projected.projection_mode,
                )
                fused.__enter__()
            full_k_length = (
                int(projected.carrier.k.shape[-2]) if fused is not None else None
            )
            for start, stop in _query_spans(sequence, query_rows, prepared.dense_query_rows):
                dense_query = stop <= prepared.dense_query_rows
                q_x = projected.x
                q_rope = projected.rope_freqs
                q_start = start
                q_stop = stop
                if fused is None:
                    held = create_held_qkv(
                        projected.module,
                        q_x[q_start : q_start + 1],
                        projected.projection_mode,
                    )
                    held.__enter__()
                    try:
                        q = project_q_hnd(
                            held,
                            q_x,
                            q_rope,
                            q_start,
                            q_stop,
                        )
                    finally:
                        held.__exit__(None, None, None)
                    if not dense_query:
                        with diagnostics.stage("sparse_route"):
                            q_summary = _tile_mean(q, q_tile)
                    chunk_carrier = _quantize_q_chunk(
                        producer_module,
                        projected.carrier,
                        q,
                    )
                    del q
                else:
                    packed_q, q_scale, q_summary = fused.project(
                        q_x,
                        q_rope,
                        q_start,
                        q_stop,
                        full_k_length,
                    )
                    chunk_carrier = replace(
                        projected.carrier,
                        q=packed_q,
                        q_scale=q_scale,
                    )
                    del packed_q, q_scale
                if dense_query:
                    if fused is not None:
                        del q_summary
                    with diagnostics.stage('dense_attention_kernel'):
                        raw = producer_module.packed_int8_attention_from_prequantized(
                            chunk_carrier, output_layout=OUTPUT_NHD,
                        )
                else:
                    tile_start = start // q_tile
                    with diagnostics.stage("sparse_route"):
                        lut, counts = _build_route_chunk(
                            self.router,
                            prepared.route_plan,
                            q_summary,
                            tile_start=tile_start,
                        )
                        del q_summary
                    route = kitchen.BlockSparseRoute(
                        indices=lut,
                        counts=counts,
                        q_tile=q_tile,
                        kv_tile=int(self.executor.kv_tile),
                        encoding="delta",
                    )
                    del lut, counts

                    with diagnostics.stage("sparse_attention_kernel"):
                        raw = kitchen.block_sparse_int8_attention_from_prequantized(
                            chunk_carrier, route, output_layout=OUTPUT_NHD,
                        )
                    del route

                del chunk_carrier

                if stop == sequence:
                    projected.carrier = None
                    if prepared.route_plan is not None:
                        prepared.route_plan.release()

                flat = raw.transpose(1, 2).reshape(
                    raw.shape[0],
                    raw.shape[2],
                    module.heads * module.head_dim,
                )
                del raw
                with diagnostics.stage("attention_out"):
                    projection_input = flat.squeeze(0)
                    if in_place_residual:
                        residual_source.write_attention_residual(
                            projection_input,
                            start,
                            stop,
                            module.out_proj,
                            project_residual,
                        )
                    else:
                        projected_rows = module.out_proj(projection_input)
                        output[start:stop].copy_(projected_rows)
                        del projected_rows
                del flat
            return (
                residual_source.finish_attention_residual()
                if in_place_residual
                else output
            )
        finally:
            if fused is not None:
                fused.__exit__(None, None, None)
            prepared.release()


_ORIGINAL_PROJECTOR_INIT = ChunkedKitchenQKVProjector.__init__
_ORIGINAL_PROJECTOR_TRY_PROJECT = ChunkedKitchenQKVProjector.try_project


def _streamed_sparse_projector_init(self, *args, **kwargs):
    _ORIGINAL_PROJECTOR_INIT(self, *args, **kwargs)
    if self.routing_summaries and self.stream_output:
        self.streamed_q = True


def _streamed_sparse_try_project(
    self,
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
):
    sparse_streamed = bool(
        self.routing_summaries and self.stream_output and self.streamed_q
    )
    if sparse_streamed:
        projected = _run_streamed_sparse_kitchen_qkv(
            self,
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        if projected is not None:
            return projected

        fallback = copy.copy(self)
        fallback.streamed_q = False
        return _ORIGINAL_PROJECTOR_TRY_PROJECT(
            fallback,
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )

    return _ORIGINAL_PROJECTOR_TRY_PROJECT(
        self,
        module,
        x,
        rope_freqs,
        layer_index=layer_index,
        transformer_options=transformer_options,
    )


def install():
    """Install the Sparse-Kitchen streamed-Q composition exactly once."""
    import h3_optimizations.attention.sparse.kitchen_sparse as kitchen_sparse

    if getattr(ChunkedKitchenQKVProjector, "_h3_sparse_kitchen_streamed_q", False):
        return
    ChunkedKitchenQKVProjector.__init__ = _streamed_sparse_projector_init
    ChunkedKitchenQKVProjector.try_project = _streamed_sparse_try_project
    ChunkedKitchenQKVProjector._h3_sparse_kitchen_streamed_q = True
    kitchen_sparse.SparseKitchenBackend = StreamedSparseKitchenBackend


__all__ = [
    "PreparedStreamedSparseKitchen",
    "StreamedSparseKitchenBackend",
    "StreamedSparseKitchenQKV",
    "install",
]
