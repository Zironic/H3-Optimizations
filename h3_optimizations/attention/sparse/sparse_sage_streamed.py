"""Low-VRAM streamed-query execution for the supported Sparse Sage backend.

This intentionally implements only the parts of the old experimental streamed
Sparge path that remain useful and have a clean lifetime contract:

* keep global K and the final V carrier, but never materialize full-sequence Q;
* retain the global K summary and select each Q slab's route immediately before
  its attention launch;
* run Sparge with a bounded Q/output slab using its qo_len != kv_len support;
* project the bounded attention output immediately and reuse the attention
  input tensor for the final hidden-size result.

The source QKV weight is acquired and released independently for every Q chunk.
It is never held across a Sparge launch or ``out_proj`` acquisition. Native
ConvRot keeps its direct INT8 Q-only path; the other supported checkpoint and
forced-precision modes produce only one bounded BF16 Q slab. This avoids the
reusable Comfy cast-buffer aliasing hazard that made the original PR #13 path
unsafe on offloaded/low-VRAM models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ... import diagnostics
from ...normalized_rows import attention_output_buffer
from ...qkv.formats import describe_linear
from ...plan import V_MEMORY_RETAIN, V_MEMORY_TWO_PASS
from ...qkv.streamed import (
    PROJECTION_NATIVE,
    create_held_qkv,
    project_kv_hnd,
    project_q_hnd,
    project_v_hnd,
)
from ..sage_v_staging import TwoPassSageVCarrier
from . import fused_qkv as _fused_qkv_mod
from .chunked_qkv import pack_sparse_qk_chunk_into
from .config import resolve_video_budget
from .fused_qkv import FusedQKVError, HEAD_DIM, sparse_fused_qkv_contract_mismatch
from .router import SparseRouterError
from .sparse_sage import SparseSageError


DEFAULT_PROJECT_CHUNK_ROWS = 4096
DEFAULT_QUERY_CHUNK_ROWS = 4096
OUT_PROJ_CHUNK_ROWS = 2048


@dataclass
class StreamedSparseSageQKV:
    module: object
    x: torch.Tensor
    rope_freqs: torch.Tensor | None
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v: torch.Tensor | None
    k_summary: torch.Tensor | None
    output_dtype: torch.dtype
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    project_chunk_rows: int
    query_chunk_rows: int
    projection_mode: str
    # Set only when the projector staged V in two passes; prepare then adopts
    # this carrier instead of quantizing a full-sequence BF16 V.
    staged_v_carrier: torch.Tensor | None = None
    staged_v_scale: torch.Tensor | None = None


@dataclass
class StreamedRoutePlan:
    geometry: object
    retained: int
    k_summary: torch.Tensor | None
    batch: int
    heads: int

    def release(self):
        self.k_summary = None


@dataclass
class PreparedStreamedSparseSage:
    projected: StreamedSparseSageQKV
    route_plan: StreamedRoutePlan | None
    v_carrier: torch.Tensor | None
    v_scale: torch.Tensor | None
    metadata: dict

    def release(self):
        projected = self.projected
        projected.k_summary = None
        projected.k_int8 = None
        projected.k_scale = None
        projected.v = None
        self.v_carrier = None
        self.v_scale = None
        if self.route_plan is not None:
            self.route_plan.release()
        self.route_plan = None


def _validate_chunk_rows(value, q_tile, kv_tile, *, name):
    value = int(value)
    alignment = math.lcm(int(q_tile), int(kv_tile))
    if value <= 0 or value % alignment:
        raise SparseSageError(
            "%s must be a positive multiple of %d" % (name, alignment)
        )
    return value


def _validate_streamed_projected(projected, spec):
    if not isinstance(projected, StreamedSparseSageQKV):
        raise SparseSageError("expected StreamedSparseSageQKV")
    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise SparseSageError(
            "streamed Sparse Sage carrier contract mismatch: %s" % mismatch
        )
    sequence = int(projected.sequence)
    heads = int(projected.heads)
    head_dim = int(projected.head_dim)
    if sequence <= 0 or heads <= 0 or head_dim != HEAD_DIM:
        raise SparseSageError("streamed Sparse Sage metadata is invalid")
    if (
        not projected.x.is_cuda
        or projected.x.dtype != torch.bfloat16
        or projected.x.ndim != 2
        or int(projected.x.shape[0]) != sequence
    ):
        raise SparseSageError(
            "streamed Sparse Sage requires rank-2 CUDA BF16 attention input"
        )
    k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    expected = (
        ("k_int8", projected.k_int8, (1, heads, sequence, head_dim), torch.int8),
        ("k_scale", projected.k_scale, (1, heads, k_blocks), torch.float32),
        (
            "k_summary",
            projected.k_summary,
            (1, heads, k_blocks, head_dim),
            projected.output_dtype,
        ),
    )
    for name, tensor, shape, dtype in expected:
        if tensor is None or tuple(tensor.shape) != tuple(shape):
            raise SparseSageError(
                "%s shape does not match streamed Sparse Sage contract" % name
            )
        if tensor.dtype != dtype or tensor.device != projected.x.device:
            raise SparseSageError(
                "%s dtype/device does not match streamed Sparse Sage contract"
                % name
            )
        if not tensor.is_contiguous():
            raise SparseSageError("%s must be contiguous" % name)
    if projected.staged_v_carrier is None:
        if projected.v is None or tuple(projected.v.shape) != (
            1,
            heads,
            sequence,
            head_dim,
        ):
            raise SparseSageError("streamed Sparse Sage V projection is invalid")
        if projected.v.dtype != projected.output_dtype:
            raise SparseSageError("streamed Sparse Sage V dtype is invalid")
    elif projected.v is not None:
        # Two-pass staging owns the carrier outright. A full-sequence BF16 V
        # alongside it means the projector kept both, which is the exact peak
        # this mode exists to remove.
        raise SparseSageError(
            "streamed Sparse Sage staged V must not retain a BF16 V tensor"
        )
    _validate_chunk_rows(
        projected.project_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="project_chunk_rows",
    )
    _validate_chunk_rows(
        projected.query_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="query_chunk_rows",
    )
    return projected


def _v_staging(spec, v_mode, sequence, heads, head_dim, x, project_chunk):
    """Build a two-pass V carrier, or None to keep retaining BF16 V.

    The spec declines when its V carrier is not FP8, since a second pass would
    then cost a reprojection for no saving. The injected project_chunk seam
    used by tests has no V-only entry point, so it also stays on one pass.
    """
    if v_mode != V_MEMORY_TWO_PASS or project_chunk is not None:
        return None
    parameters = getattr(spec, "v_staging_parameters", None)
    parameters = None if parameters is None else parameters()
    if parameters is None:
        return None
    scale_max, pad_to = parameters
    return TwoPassSageVCarrier(
        1,
        heads,
        sequence,
        head_dim,
        scale_max=scale_max,
        device=x.device,
        dtype=x.dtype,
        pad_to=pad_to,
    )


def _assemble_streamed_sparse_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    spec,
    project_chunk_rows,
    query_chunk_rows,
    projection_mode=PROJECTION_NATIVE,
    v_mode=V_MEMORY_RETAIN,
    packer=pack_sparse_qk_chunk_into,
    project_chunk=None,
    held_factory=create_held_qkv,
):
    """Produce global K/V and routing summaries without a global Q carrier."""
    mismatch = sparse_fused_qkv_contract_mismatch(spec)
    if mismatch is not None:
        raise SparseSageError(
            "streamed Sparse Sage QKV contract mismatch: %s" % mismatch
        )
    project_chunk_rows = _validate_chunk_rows(
        project_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="project_chunk_rows",
    )
    query_chunk_rows = _validate_chunk_rows(
        query_chunk_rows,
        spec.q_tile,
        spec.kv_tile,
        name="query_chunk_rows",
    )
    sequence = int(x.shape[0])
    heads = int(module.heads)
    head_dim = int(module.head_dim)
    if sequence <= 0 or head_dim != HEAD_DIM:
        raise SparseSageError("streamed Sparse Sage QKV requires head_dim 128")

    k_blocks = (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile)
    k_int8 = torch.empty(
        (1, heads, sequence, head_dim), dtype=torch.int8, device=x.device
    )
    k_scale = torch.empty(
        (1, heads, k_blocks), dtype=torch.float32, device=x.device
    )
    staging = _v_staging(spec, v_mode, sequence, heads, head_dim, x, project_chunk)
    staged_v_carrier = None
    staged_v_scale = None
    v = (
        None
        if staging is not None
        else torch.empty(
            (1, heads, sequence, head_dim), dtype=x.dtype, device=x.device
        )
    )
    k_summary = torch.empty(
        (1, heads, k_blocks, head_dim), dtype=x.dtype, device=x.device
    )

    held = None
    if project_chunk is None:
        held = held_factory(module, x[:1], projection_mode)
        held.__enter__()
    try:
        for start in range(0, sequence, project_chunk_rows):
            end = min(start + project_chunk_rows, sequence)
            if held is None:
                projected_chunk = project_chunk(
                    module,
                    x,
                    rope_freqs,
                    start,
                    end,
                )
                if len(projected_chunk) == 3:
                    q, k, chunk_v = projected_chunk
                    del q
                else:
                    k, chunk_v = projected_chunk
            else:
                k, chunk_v = project_kv_hnd(
                    held,
                    x,
                    rope_freqs,
                    start,
                    end,
                )
            try:
                packer(
                    k,
                    k_int8,
                    k_scale,
                    k_summary,
                    row_start=start,
                    block_size=spec.kv_tile,
                )
                if staging is None:
                    v[..., start:end, :].copy_(chunk_v)
                else:
                    with diagnostics.stage("v_amax_update"):
                        staging.update(chunk_v)
            finally:
                del k, chunk_v
        if staging is not None:
            # Second pass: V only. K and its summaries are already packed, so
            # this reprojects strictly what the carrier still needs.
            staging.finalize_scale()
            for start in range(0, sequence, project_chunk_rows):
                end = min(start + project_chunk_rows, sequence)
                with diagnostics.stage("v_reprojection"):
                    chunk_v = project_v_hnd(held, x, rope_freqs, start, end)
                try:
                    with diagnostics.stage("v_carrier_pack"):
                        staging.quantize(chunk_v, start)
                finally:
                    del chunk_v
            staged_v_carrier, staged_v_scale = staging.finish()
    finally:
        if held is not None:
            held.__exit__(None, None, None)

    return StreamedSparseSageQKV(
        module=module,
        x=x,
        rope_freqs=rope_freqs,
        k_int8=k_int8,
        k_scale=k_scale,
        v=v,
        staged_v_carrier=staged_v_carrier,
        staged_v_scale=staged_v_scale,
        k_summary=k_summary,
        output_dtype=x.dtype,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        layer_index=int(layer_index),
        project_chunk_rows=project_chunk_rows,
        query_chunk_rows=query_chunk_rows,
        projection_mode=projection_mode,
    )


def run_streamed_sparse_qkv(
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    spec,
    project_chunk_rows=DEFAULT_PROJECT_CHUNK_ROWS,
    query_chunk_rows=DEFAULT_QUERY_CHUNK_ROWS,
    projection_mode=PROJECTION_NATIVE,
    v_mode=V_MEMORY_RETAIN,
):
    import comfy.model_management

    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise SparseSageError(
            "streamed Sparse Sage QKV requires rank-2 CUDA BF16 input"
        )
    if comfy.model_management.in_training:
        raise SparseSageError("streamed Sparse Sage QKV is inference-only")
    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or rope_freqs.device != x.device
    ):
        raise SparseSageError("streamed Sparse Sage QKV received invalid RoPE")
    return _assemble_streamed_sparse_qkv(
        module,
        x,
        rope_freqs,
        layer_index=layer_index,
        spec=spec,
        project_chunk_rows=project_chunk_rows,
        query_chunk_rows=query_chunk_rows,
        projection_mode=projection_mode,
        v_mode=v_mode,
    )


class StreamedSparseSageQKVProjector:
    name = "chunked_sparse_sage_qkv"
    qk_format = "streamed_q_sparge_block_int8"
    streamed_q = True

    def __init__(
        self,
        spec,
        *,
        project_chunk_rows=DEFAULT_PROJECT_CHUNK_ROWS,
        query_chunk_rows=DEFAULT_QUERY_CHUNK_ROWS,
        projection_mode=PROJECTION_NATIVE,
        v_mode=V_MEMORY_RETAIN,
    ):
        self.spec = spec
        self.projection_mode = projection_mode
        if v_mode not in (V_MEMORY_RETAIN, V_MEMORY_TWO_PASS):
            raise ValueError('unknown sparse Sage V mode %r' % v_mode)
        self.requested_v_mode = v_mode
        # Resolve against what the spec can do, so v_mode always names what
        # will happen; status reads this field.
        parameters = getattr(spec, "v_staging_parameters", None)
        self.v_mode = (
            V_MEMORY_TWO_PASS
            if v_mode == V_MEMORY_TWO_PASS
            and parameters is not None
            and parameters() is not None
            else V_MEMORY_RETAIN
        )
        self.chunk_rows = _validate_chunk_rows(
            project_chunk_rows,
            spec.q_tile,
            spec.kv_tile,
            name="project_chunk_rows",
        )
        self.query_chunk_rows = _validate_chunk_rows(
            query_chunk_rows,
            spec.q_tile,
            spec.kv_tile,
            name="query_chunk_rows",
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.qk_format,
            self.chunk_rows,
            self.query_chunk_rows,
            self.projection_mode,
            self.v_mode,
            self.spec.signature,
        )

    def project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        del transformer_options
        return run_streamed_sparse_qkv(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            spec=self.spec,
            project_chunk_rows=self.chunk_rows,
            query_chunk_rows=self.query_chunk_rows,
            projection_mode=self.projection_mode,
            v_mode=self.v_mode,
        )


def _prepare_streamed_route_plan(
    router,
    k_summary,
    layout,
    video_budget,
):
    """Prepare K-owned routing state without projecting Q ahead of execution."""
    if k_summary.ndim != 4:
        raise SparseRouterError("K router summary must be a rank-4 HND tensor")
    if not math.isfinite(float(video_budget)):
        raise SparseRouterError("video_budget must be finite")

    geometry = router.geometry(layout)
    if tuple(k_summary.shape[-2:]) != (
        geometry.kv_tiles,
        k_summary.shape[-1],
    ):
        raise SparseRouterError("K router summary shape does not match layout")

    retained = router._retained(video_budget, geometry)
    metadata = router._metadata(geometry, video_budget, retained)
    return (
        StreamedRoutePlan(
            geometry=geometry,
            retained=retained,
            k_summary=k_summary,
            batch=int(k_summary.shape[0]),
            heads=int(k_summary.shape[1]),
        ),
        metadata,
    )


def _build_streamed_lut_chunk(
    router,
    route_plan,
    q_summary,
    *,
    tile_start,
):
    geometry = route_plan.geometry
    if route_plan.k_summary is None:
        raise SparseRouterError("streamed route K summary was released")
    if q_summary.ndim != 4:
        raise SparseRouterError("Q router summary must be a rank-4 HND tensor")
    tile_start = int(tile_start)
    tile_count = int(q_summary.shape[-2])
    tile_end = tile_start + tile_count
    if (
        tuple(q_summary.shape[:2]) != (route_plan.batch, route_plan.heads)
        or q_summary.shape[-1] != route_plan.k_summary.shape[-1]
        or q_summary.device != route_plan.k_summary.device
        or tile_count <= 0
        or not 0 <= tile_start < tile_end <= geometry.q_tiles
    ):
        raise SparseRouterError("Q router summary chunk does not match route plan")
    dense = torch.arange(
        geometry.kv_tiles,
        device=q_summary.device,
        dtype=torch.int32,
    )
    dense_delta = torch.cat((dense[:1], dense[1:] - dense[:-1]))
    lut = dense_delta.view(1, 1, 1, -1).expand(
        route_plan.batch,
        route_plan.heads,
        tile_count,
        -1,
    ).clone()
    valid = torch.full(
        (route_plan.batch, route_plan.heads, tile_count),
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q_summary.device,
    )

    sparse_start = max(tile_start, geometry.pure_video_q_start)
    if (
        route_plan.retained < geometry.pure_video_kv_tiles
        and sparse_start < tile_end
    ):
        local_start = sparse_start - tile_start
        indices = router._select_indices(
            q_summary[..., local_start:, :],
            route_plan.k_summary[..., geometry.pure_video_kv_start:, :],
            route_plan.retained,
        )
        sparse_rows = router._pack_rows(
            indices,
            geometry,
            dense,
            dense_delta,
        )
        lut[..., local_start:, :sparse_rows.shape[-1]].copy_(sparse_rows)
        valid[..., local_start:] = (
            geometry.pure_video_kv_start + route_plan.retained
        )
    return lut.contiguous(), valid.contiguous()


def _validate_staged_v(carrier, scale, projected):
    """Check a projector-staged carrier the way _validate_v checks a prepared one.

    The staged path bypasses the executor's own preparer, so the shape contract
    it would have enforced has to be enforced here instead. Padding is the one
    that can drift silently: the kernel ABI wants 128, and a spec that reported
    a different pad_to would otherwise produce a quietly wrong carrier.
    """
    expected_padded = (int(projected.sequence) + 127) // 128 * 128
    expected = (1, int(projected.heads), int(projected.head_dim), expected_padded)
    if (
        tuple(carrier.shape) != expected
        or carrier.dtype != torch.float8_e4m3fn
        or not carrier.is_contiguous()
    ):
        raise SparseSageError("staged V produced an invalid FP8 carrier")
    if (
        scale is None
        or tuple(scale.shape) != (1, int(projected.heads), int(projected.head_dim))
        or scale.dtype != torch.float32
        or scale.device != carrier.device
        or not scale.is_contiguous()
    ):
        raise SparseSageError("staged V produced an invalid FP8 scale")


def prepare_streamed_sparse_sage(
    backend,
    projected,
    *,
    layer_index,
    transformer_options,
):
    projected = _validate_streamed_projected(projected, backend.executor.spec)
    if int(projected.layer_index) != int(layer_index):
        raise SparseSageError(
            "streamed QKV layer %d does not match attention layer %d"
            % (projected.layer_index, layer_index)
        )
    snapshot = backend._snapshot(transformer_options, projected.sequence)
    video_budget = resolve_video_budget(
        backend.config,
        snapshot.step_index,
        snapshot.total_steps,
        layer_index,
    )
    try:
        with diagnostics.stage("sparse_route"):
            route_plan, mask_metadata = _prepare_streamed_route_plan(
                backend.router,
                projected.k_summary,
                snapshot.layout,
                video_budget,
            )
    except SparseRouterError as exc:
        raise SparseSageError("sparse routing failed: %s" % exc) from exc

    # The route plan owns K summaries until the final Q slab has selected its
    # route. The projected carrier no longer owns a second reference.
    projected.k_summary = None
    with diagnostics.stage("sparse_carrier_prepare"):
        if projected.staged_v_carrier is not None:
            v_carrier = projected.staged_v_carrier
            v_scale = projected.staged_v_scale
            projected.staged_v_carrier = None
            projected.staged_v_scale = None
            _validate_staged_v(v_carrier, v_scale, projected)
        else:
            v_carrier, v_scale = backend.executor._prepare_v(projected.v)
            backend.executor._validate_v(
                projected.v,
                v_carrier,
                v_scale,
                projected.sequence,
            )
    projected.v = None

    metadata = backend._metadata(
        mask_metadata,
        layer_index,
        projected.heads,
    )
    metadata.update(
        {
            "qkv_lifetime": "streamed_q_global_k",
            "attention_output": "chunked_out_proj_inplace",
            "router_lifetime": "k_summary_q_slab_selection_lazy_sparge_lut",
            "project_chunk_rows": projected.project_chunk_rows,
            "query_chunk_rows": projected.query_chunk_rows,
            "out_proj_chunk_rows": OUT_PROJ_CHUNK_ROWS,
        }
    )
    return PreparedStreamedSparseSage(
        projected=projected,
        route_plan=route_plan,
        v_carrier=v_carrier,
        v_scale=v_scale,
        metadata=metadata,
    )


def _run_fused_q_only_into(
    module,
    x,
    rope_freqs,
    q_int8,
    q_scale,
    q_summary_scratch,
):
    """Self-contained ConvRot Q projection; the weight handle never escapes."""
    import comfy.model_management
    import comfy.ops

    if not _fused_qkv_mod.TRITON_AVAILABLE:
        raise FusedQKVError("Q-only fused H3 projection requires Triton")
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise FusedQKVError(
            "Q-only fused H3 projection requires rank-2 CUDA BF16 input"
        )
    if comfy.model_management.in_training:
        raise FusedQKVError("Q-only fused H3 projection is inference-only")
    if int(module.head_dim) != HEAD_DIM:
        raise FusedQKVError("Q-only fused H3 projection requires head_dim 128")

    sequence, hidden = x.shape
    heads = int(module.heads)
    q_blocks = (sequence + _fused_qkv_mod.Q_TILE - 1) // _fused_qkv_mod.Q_TILE
    if (
        tuple(q_int8.shape) != (1, heads, sequence, HEAD_DIM)
        or q_int8.dtype != torch.int8
        or not q_int8.is_contiguous()
        or q_int8.device != x.device
    ):
        raise FusedQKVError("Q-only fused H3 Q destination is invalid")
    if (
        tuple(q_scale.shape) != (1, heads, q_blocks)
        or q_scale.dtype != torch.float32
        or not q_scale.is_contiguous()
        or q_scale.device != x.device
    ):
        raise FusedQKVError("Q-only fused H3 Q-scale destination is invalid")
    if (
        tuple(q_summary_scratch.shape) != (1, heads, q_blocks, HEAD_DIM)
        or q_summary_scratch.dtype != x.dtype
        or not q_summary_scratch.is_contiguous()
        or q_summary_scratch.device != x.device
    ):
        raise FusedQKVError("Q-only fused H3 summary scratch is invalid")
    if hidden % 256:
        raise FusedQKVError(
            "Q-only fused H3 projection requires ConvRot-256 hidden dimension"
        )
    if float(module.q_norm.eps) != float(module.k_norm.eps):
        raise FusedQKVError("Q-only fused H3 projection requires matching Q/K eps")

    if rope_freqs is None:
        rope = x.new_empty((1, 1, 1, 16, 2, 2))
        rope_strides = (0, 0, 0, 0)
        has_rope = False
    else:
        if (
            rope_freqs.ndim != 6
            or tuple(rope_freqs.shape[:3]) != (1, sequence, 1)
            or int(rope_freqs.shape[-3]) * 2 != _fused_qkv_mod.ROT_DIM
            or tuple(rope_freqs.shape[-2:]) != (2, 2)
            or rope_freqs.device != x.device
        ):
            raise FusedQKVError(
                "Q-only fused H3 projection requires H3 split-half RoPE"
            )
        rope = rope_freqs
        rope_strides = (
            rope.stride(1),
            rope.stride(3),
            rope.stride(4),
            rope.stride(5),
        )
        has_rope = True

    qdata, weight_scale, handle, held_weight, bias = (
        _fused_qkv_mod._plain_qkv_weight(module, x)
    )
    try:
        inner = heads * HEAD_DIM
        expected_weight = (inner * 3, hidden)
        if (
            tuple(qdata.shape) != expected_weight
            or qdata.dtype != torch.int8
            or qdata.device != x.device
        ):
            raise FusedQKVError(
                "Q-only fused H3 weight shape is %s; expected %s"
                % (tuple(qdata.shape), expected_weight)
            )
        weight_scale = weight_scale.reshape(-1).contiguous()
        if (
            weight_scale.numel() != inner * 3
            or weight_scale.dtype != torch.float32
            or weight_scale.device != x.device
        ):
            raise FusedQKVError("Q-only fused H3 weight scale shape is invalid")
        q_norm = comfy.model_management.cast_to(
            module.q_norm.weight,
            device=x.device,
            dtype=x.dtype,
        ).contiguous()
        if q_norm.numel() != HEAD_DIM or q_norm.dtype != x.dtype:
            raise FusedQKVError("Q-only fused H3 RMSNorm weight is invalid")

        x_int8, x_scale = _fused_qkv_mod._quantize_projection_input(x)
        grid = (
            _fused_qkv_mod.triton.cdiv(sequence, _fused_qkv_mod.Q_TILE),
            heads,
        )
        _fused_qkv_mod._fused_qk_kernel[grid](
            x_int8,
            qdata,
            x_scale,
            weight_scale,
            q_norm,
            q_norm,
            rope,
            q_int8,
            q_scale,
            q_int8,
            q_scale,
            q_summary_scratch,
            q_summary_scratch,
            sequence=sequence,
            hidden=hidden,
            heads=heads,
            weight_stride_output=qdata.stride(0),
            weight_stride_inner=qdata.stride(1),
            rope_stride_seq=rope_strides[0],
            rope_stride_dim=rope_strides[1],
            rope_stride_rot=rope_strides[2],
            rope_stride_pair=rope_strides[3],
            epsilon=float(module.q_norm.eps),
            has_rope=has_rope,
            KIND=0,
            BLOCK_M=_fused_qkv_mod.Q_TILE,
            BLOCK_N=HEAD_DIM,
            BLOCK_K=128,
            num_warps=8,
            num_stages=3,
        )
    finally:
        # Critical safety property: the QKV weight is uncast before Sparge or
        # out_proj can acquire another reusable Comfy cast buffer.
        comfy.ops.uncast_bias_weight(
            module.qkv_proj,
            held_weight,
            bias,
            handle,
        )


def _project_streamed_q_into(
    projected,
    row_start,
    row_end,
    q_int8,
    q_scale,
    q_summary_scratch,
    *,
    block_size,
    held_factory=create_held_qkv,
    packer=pack_sparse_qk_chunk_into,
):
    """Project and pack one Q slab without retaining its weight binding."""
    actual = describe_linear(projected.module.qkv_proj)
    if projected.projection_mode == PROJECTION_NATIVE and actual.convrot_int8_256:
        chunk_rope = (
            None
            if projected.rope_freqs is None
            else projected.rope_freqs[:, row_start:row_end]
        )
        _run_fused_q_only_into(
            projected.module,
            projected.x[row_start:row_end],
            chunk_rope,
            q_int8,
            q_scale,
            q_summary_scratch,
        )
        return

    held = held_factory(
        projected.module,
        projected.x[row_start:row_start + 1],
        projected.projection_mode,
    )
    held.__enter__()
    try:
        q = project_q_hnd(
            held,
            projected.x,
            projected.rope_freqs,
            row_start,
            row_end,
        )
        try:
            packer(
                q,
                q_int8,
                q_scale,
                q_summary_scratch,
                row_start=0,
                block_size=block_size,
            )
        finally:
            del q
    finally:
        held.__exit__(None, None, None)


def execute_streamed_sparse_sage(module, backend, prepared):
    if not isinstance(prepared, PreparedStreamedSparseSage):
        return None
    projected = prepared.projected
    spec = backend.executor.spec
    if module is not projected.module:
        raise SparseSageError(
            "streamed Sparse Sage module changed between prepare and execute"
        )
    if prepared.route_plan is None:
        raise SparseSageError("streamed Sparse Sage route was already released")

    sequence = int(projected.sequence)
    heads = int(projected.heads)
    hidden = int(projected.x.shape[1])
    q_tile = int(spec.q_tile)
    max_rows = min(int(projected.query_chunk_rows), sequence)
    max_q_tiles = (max_rows + q_tile - 1) // q_tile
    result = attention_output_buffer(projected.x)

    q_int8_buffer = torch.empty(
        (1, heads, max_rows, HEAD_DIM),
        dtype=torch.int8,
        device=result.device,
    )
    q_scale_buffer = torch.empty(
        (1, heads, max_q_tiles),
        dtype=torch.float32,
        device=result.device,
    )
    q_summary_buffer = torch.empty(
        (1, heads, max_q_tiles, HEAD_DIM),
        dtype=projected.output_dtype,
        device=result.device,
    )
    output_buffer = torch.empty(
        (1, heads, max_rows, HEAD_DIM),
        dtype=projected.output_dtype,
        device=result.device,
    )
    pv_threshold = torch.full(
        (heads,), 50.0, dtype=torch.float32, device=result.device
    )

    try:
        for tile_start in range(
            0,
            prepared.route_plan.geometry.q_tiles,
            max_q_tiles,
        ):
            tile_end = min(
                tile_start + max_q_tiles,
                prepared.route_plan.geometry.q_tiles,
            )
            row_start = int(tile_start) * q_tile
            row_end = min(int(tile_end) * q_tile, sequence)
            rows = row_end - row_start
            local_q_tiles = int(tile_end) - int(tile_start)
            if rows <= 0:
                raise SparseSageError("streamed Sparse Sage produced empty Q chunk")

            full_chunk = rows == max_rows and local_q_tiles == max_q_tiles
            if full_chunk:
                q_int8 = q_int8_buffer
                q_scale = q_scale_buffer
                q_summary_scratch = q_summary_buffer
                output = output_buffer
            else:
                # Head stride depends on qo_len, so the tail must be exact-size
                # rather than a non-contiguous view into the reusable buffers.
                q_int8 = torch.empty(
                    (1, heads, rows, HEAD_DIM),
                    dtype=torch.int8,
                    device=result.device,
                )
                q_scale = torch.empty(
                    (1, heads, local_q_tiles),
                    dtype=torch.float32,
                    device=result.device,
                )
                q_summary_scratch = torch.empty(
                    (1, heads, local_q_tiles, HEAD_DIM),
                    dtype=projected.output_dtype,
                    device=result.device,
                )
                output = torch.empty(
                    (1, heads, rows, HEAD_DIM),
                    dtype=projected.output_dtype,
                    device=result.device,
                )

            _project_streamed_q_into(
                projected,
                row_start,
                row_end,
                q_int8,
                q_scale,
                q_summary_scratch,
                block_size=q_tile,
            )

            with diagnostics.stage("sparse_route"):
                lut_chunk, valid_chunk = _build_streamed_lut_chunk(
                    backend.router,
                    prepared.route_plan,
                    q_summary_scratch,
                    tile_start=tile_start,
                )

            with diagnostics.stage("sparse_attention_kernel"):
                spec.dispatch(
                    q_int8,
                    projected.k_int8,
                    prepared.v_carrier,
                    output,
                    lut_chunk,
                    valid_chunk,
                    pv_threshold,
                    q_scale,
                    projected.k_scale,
                    prepared.v_scale,
                    projected.output_dtype,
                )
            del lut_chunk, valid_chunk

            # On the final Q slab the global carriers are dead immediately
            # after dispatch.  Releasing their Python references here lets the
            # same-stream allocator reuse them for out_proj safely.
            if int(tile_end) == int(prepared.route_plan.geometry.q_tiles):
                projected.k_int8 = None
                projected.k_scale = None
                prepared.v_carrier = None
                prepared.v_scale = None

            with diagnostics.stage("attention_out"):
                for local_start in range(0, rows, OUT_PROJ_CHUNK_ROWS):
                    local_end = min(local_start + OUT_PROJ_CHUNK_ROWS, rows)
                    proj_rows = local_end - local_start
                    attention_rows = (
                        output[..., local_start:local_end, :]
                        .transpose(1, 2)
                        .reshape(proj_rows, heads * HEAD_DIM)
                    )
                    projected_rows = module.out_proj(attention_rows)
                    if tuple(projected_rows.shape) != (proj_rows, hidden):
                        raise SparseSageError(
                            "streamed Sparse Sage out_proj shape %s is invalid"
                            % (tuple(projected_rows.shape),)
                        )
                    result[
                        row_start + local_start:row_start + local_end
                    ].copy_(projected_rows)
                    del attention_rows, projected_rows

            if not full_chunk:
                del q_int8, q_scale, q_summary_scratch, output
    finally:
        prepared.release()
        del (
            q_int8_buffer,
            q_scale_buffer,
            q_summary_buffer,
            output_buffer,
            pv_threshold,
        )

    return result
