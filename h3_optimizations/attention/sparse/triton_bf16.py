'''Portable BF16 Triton fallback for fixed-density H3 sparse attention.'''

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

from ...qkv.bf16 import PreparedBF16QKV
from ...runtime.context import get_runtime_snapshot
from .config import HybridSparseConfig, resolve_video_budget
from .router import SparseRouterError, SparseTileRouter

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_AVAILABLE = False


Q_TILE = 64
KV_TILE = 64
HEAD_DIM = 128


class TritonBF16Error(RuntimeError):
    pass


@dataclass(frozen=True)
class TritonBF16Spec:
    q_tile: int = Q_TILE
    kv_tile: int = KV_TILE
    head_dim: int = HEAD_DIM
    implementation: str = 'triton_bf16_qk_bf16pv_fp32'

    @property
    def signature(self):
        return (
            self.implementation,
            int(self.q_tile),
            int(self.kv_tile),
            int(self.head_dim),
        )


@dataclass
class PreparedTritonBF16:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    sparse_lut: torch.Tensor
    dense_q_tiles: int
    sparse_q_tiles: int
    sparse_selected: int
    layer_index: int
    metadata: dict


def preflight_triton_bf16(
    *, cuda_available, capability_getter, triton_available=None
):
    if not cuda_available():
        raise TritonBF16Error('BF16 Triton sparse attention requires CUDA')
    available = TRITON_AVAILABLE if triton_available is None else bool(triton_available)
    if not available:
        raise TritonBF16Error('BF16 Triton sparse attention requires Triton')
    capability = capability_getter()
    if capability is None:
        raise TritonBF16Error('BF16 Triton GPU capability is unavailable')
    capability = tuple(int(value) for value in capability)
    if len(capability) != 2 or capability[0] < 8:
        raise TritonBF16Error(
            'BF16 Triton sparse attention requires NVIDIA compute capability '
            '>= 8.0; got %s' % (capability,)
        )
    return TritonBF16Spec()


def _validate_qkv(q, k, v):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise TritonBF16Error('BF16 Triton requires equal HND rank-4 Q/K/V')
    if q.shape[0] != 1 or q.shape[-1] != HEAD_DIM:
        raise TritonBF16Error('BF16 Triton requires batch 1 and head dimension 128')
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TritonBF16Error('BF16 Triton requires BF16 Q/K/V')
    if q.device != k.device or q.device != v.device or not q.is_cuda:
        raise TritonBF16Error('BF16 Triton requires CUDA Q/K/V on one device')
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise TritonBF16Error('BF16 Triton requires contiguous head dimensions')


def _compact_route(lut, valid, metadata):
    if lut.ndim != 4 or valid.ndim != 3:
        raise TritonBF16Error('BF16 Triton route ranks are invalid')
    if lut.dtype != torch.int32 or valid.dtype != torch.int32:
        raise TritonBF16Error('BF16 Triton route must be INT32')
    if not lut.is_contiguous() or not valid.is_contiguous():
        raise TritonBF16Error('BF16 Triton route must be contiguous')
    batch, heads, q_tiles, kv_tiles = (int(value) for value in lut.shape)
    if tuple(valid.shape) != (batch, heads, q_tiles):
        raise TritonBF16Error('BF16 Triton route shapes differ')
    dense_q_tiles = int(metadata['dense_q_tiles'])
    sparse_q_tiles = int(metadata['sparse_q_tiles'])
    if dense_q_tiles + sparse_q_tiles != q_tiles:
        raise TritonBF16Error('BF16 Triton Q route geometry differs')
    if sparse_q_tiles:
        selected = (
            kv_tiles
            - int(metadata['pure_video_kv_tiles'])
            + int(metadata['retained_video_kv_tiles'])
        )
        if not 0 < selected < kv_tiles:
            raise TritonBF16Error('BF16 Triton sparse route is invalid')
        delta = lut[
            ...,
            dense_q_tiles:dense_q_tiles + sparse_q_tiles,
            :selected,
        ]
        absolute = torch.cumsum(delta, dim=-1, dtype=torch.int32).contiguous()
    else:
        selected = 0
        absolute = lut.new_empty((batch, heads, 0, 0))
    return absolute, dense_q_tiles, sparse_q_tiles, selected


if TRITON_AVAILABLE:
    @triton.jit
    def _bf16_sparse_kernel(
        Q,
        K,
        V,
        LUT,
        O,
        sequence,
        Q_BLOCK_START,
        Q_BLOCK_COUNT,
        Q_STORAGE_ROW_START,
        Q_STORAGE_ROWS,
        stride_qh: tl.constexpr,
        stride_qn: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_on: tl.constexpr,
        N_SELECTED: tl.constexpr,
        USE_ROUTE: tl.constexpr,
        softmax_scale: tl.constexpr,
        Q_TILE_: tl.constexpr,
        KV_TILE_: tl.constexpr,
        D: tl.constexpr,
    ):
        local_q_block = tl.program_id(0)
        q_block = Q_BLOCK_START + local_q_block
        bh = tl.program_id(1)

        global_q_rows = q_block * Q_TILE_ + tl.arange(0, Q_TILE_)
        q_rows = global_q_rows - Q_STORAGE_ROW_START
        kv_rows = tl.arange(0, KV_TILE_)
        dims = tl.arange(0, D)
        q_mask = (
            (global_q_rows < sequence)
            & (q_rows >= 0)
            & (q_rows < Q_STORAGE_ROWS)
        )
        q = tl.load(
            Q + bh * stride_qh + q_rows[:, None] * stride_qn + dims[None, :],
            mask=q_mask[:, None],
            other=0.0,
        )
        row_max = tl.full((Q_TILE_,), -float('inf'), dtype=tl.float32)
        row_sum = tl.zeros((Q_TILE_,), dtype=tl.float32)
        output = tl.zeros((Q_TILE_, D), dtype=tl.float32)

        for route_position in tl.range(0, N_SELECTED):
            if USE_ROUTE:
                route_offset = (
                    (bh * Q_BLOCK_COUNT + local_q_block) * N_SELECTED
                    + route_position
                )
                key_block = tl.load(LUT + route_offset)
            else:
                key_block = route_position
            k_rows = key_block * KV_TILE_ + kv_rows
            k_mask = k_rows < sequence
            k = tl.load(
                K + bh * stride_kh + k_rows[None, :] * stride_kn + dims[:, None],
                mask=k_mask[None, :],
                other=0.0,
            )
            logits = tl.dot(q, k) * (softmax_scale * 1.4426950408889634)
            logits = tl.where(k_mask[None, :], logits, -float('inf'))
            v = tl.load(
                V + bh * stride_vh + k_rows[:, None] * stride_vn + dims[None, :],
                mask=k_mask[:, None],
                other=0.0,
            )

            tile_max = tl.max(logits, axis=1)
            new_row_max = tl.maximum(row_max, tile_max)
            probability = tl.math.exp2(logits - new_row_max[:, None])
            tile_sum = tl.sum(probability, axis=1)
            old_scale = tl.math.exp2(row_max - new_row_max)

            output = output * old_scale[:, None]
            output += tl.dot(probability.to(v.dtype), v)
            row_sum = row_sum * old_scale + tile_sum
            row_max = new_row_max

        tl.store(
            O + bh * stride_oh + q_rows[:, None] * stride_on + dims[None, :],
            (output / row_sum[:, None]).to(O.type.element_ty),
            mask=q_mask[:, None],
        )


def _launch(prepared):
    if not TRITON_AVAILABLE:
        raise TritonBF16Error('BF16 Triton sparse attention requires Triton')
    if not isinstance(prepared, PreparedTritonBF16):
        raise TritonBF16Error('invalid BF16 Triton payload')
    _validate_qkv(prepared.q, prepared.k, prepared.v)
    sequence = int(prepared.q.shape[2])
    heads = int(prepared.q.shape[1])
    q_tiles = (sequence + Q_TILE - 1) // Q_TILE
    kv_tiles = (sequence + KV_TILE - 1) // KV_TILE
    if prepared.dense_q_tiles + prepared.sparse_q_tiles != q_tiles:
        raise TritonBF16Error('prepared BF16 Triton Q geometry differs')
    expected = (1, heads, prepared.sparse_q_tiles, prepared.sparse_selected)
    if tuple(prepared.sparse_lut.shape) != expected:
        raise TritonBF16Error('prepared BF16 Triton sparse LUT shape differs')
    if (
        prepared.sparse_lut.dtype != torch.int32
        or not prepared.sparse_lut.is_contiguous()
        or prepared.sparse_lut.device != prepared.q.device
    ):
        raise TritonBF16Error('prepared BF16 Triton sparse LUT is invalid')

    output = torch.empty(
        prepared.q.shape,
        dtype=prepared.q.dtype,
        device=prepared.q.device,
    )

    def launch_group(q_start, q_count, selected, use_route):
        if not q_count:
            return
        _bf16_sparse_kernel[(q_count, heads)](
            prepared.q,
            prepared.k,
            prepared.v,
            prepared.sparse_lut,
            output,
            sequence,
            q_start,
            q_count,
            0,
            sequence,
            stride_qh=prepared.q.stride(1),
            stride_qn=prepared.q.stride(2),
            stride_kh=prepared.k.stride(1),
            stride_kn=prepared.k.stride(2),
            stride_vh=prepared.v.stride(1),
            stride_vn=prepared.v.stride(2),
            stride_oh=output.stride(1),
            stride_on=output.stride(2),
            N_SELECTED=selected,
            USE_ROUTE=use_route,
            softmax_scale=HEAD_DIM**-0.5,
            Q_TILE_=Q_TILE,
            KV_TILE_=KV_TILE,
            D=HEAD_DIM,
            num_warps=4,
            num_stages=1,
        )

    launch_group(0, prepared.dense_q_tiles, kv_tiles, False)
    launch_group(
        prepared.dense_q_tiles,
        prepared.sparse_q_tiles,
        prepared.sparse_selected,
        True,
    )
    return output


def _launch_streamed_chunk(
    q,
    k,
    v,
    sparse_lut,
    *,
    dense_q_tiles,
    sparse_q_tiles,
    sparse_selected,
    sequence,
    q_row_start,
    sparse_lut_q_start=0,
):
    if not TRITON_AVAILABLE:
        raise TritonBF16Error('BF16 Triton sparse attention requires Triton')
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise TritonBF16Error('streamed BF16 Triton Q/K/V ranks differ')
    if (
        q.shape[0] != 1
        or k.shape[0] != 1
        or q.shape[1] != k.shape[1]
        or q.shape[-1] != HEAD_DIM
        or k.shape[-1] != HEAD_DIM
        or int(k.shape[-2]) != int(sequence)
    ):
        raise TritonBF16Error('streamed BF16 Triton Q/K/V shapes differ')
    if (
        q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or q.device != k.device
        or q.device != v.device
        or not q.is_cuda
    ):
        raise TritonBF16Error('streamed BF16 Triton requires CUDA BF16 Q/K/V')
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise TritonBF16Error('streamed BF16 Triton head dimensions must be contiguous')

    q_row_start = int(q_row_start)
    rows = int(q.shape[-2])
    if q_row_start < 0 or q_row_start % Q_TILE or not 0 < rows <= int(sequence):
        raise TritonBF16Error('streamed BF16 Triton Q slab is invalid')
    q_tile_start = q_row_start // Q_TILE
    q_tile_count = (rows + Q_TILE - 1) // Q_TILE
    total_q_tiles = int(dense_q_tiles) + int(sparse_q_tiles)
    if q_tile_start + q_tile_count > total_q_tiles:
        raise TritonBF16Error('streamed BF16 Triton Q slab exceeds its route')

    heads = int(q.shape[1])
    kv_tiles = (int(sequence) + KV_TILE - 1) // KV_TILE
    sparse_lut_q_start = int(sparse_lut_q_start)
    expected = (1, heads, int(sparse_lut.shape[-2]), int(sparse_selected))
    if tuple(sparse_lut.shape) != expected:
        raise TritonBF16Error('streamed BF16 Triton sparse route shape differs')

    # Each Triton program owns one (Q tile, head), loads that complete Q tile
    # before entering the KV loop, and no other program reads those Q rows for
    # that head. The store at the end can therefore safely overwrite the Q slab
    # in place. This makes one bounded allocation serve as both Q and O.
    output = q

    def launch_group(q_start, q_count, selected, use_route, lut):
        if not q_count:
            return
        _bf16_sparse_kernel[(q_count, heads)](
            q,
            k,
            v,
            lut,
            output,
            int(sequence),
            q_start,
            q_count,
            q_row_start,
            rows,
            stride_qh=q.stride(1),
            stride_qn=q.stride(2),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            stride_oh=output.stride(1),
            stride_on=output.stride(2),
            N_SELECTED=selected,
            USE_ROUTE=use_route,
            softmax_scale=HEAD_DIM**-0.5,
            Q_TILE_=Q_TILE,
            KV_TILE_=KV_TILE,
            D=HEAD_DIM,
            num_warps=4,
            num_stages=1,
        )

    dense_count = max(
        0,
        min(q_tile_start + q_tile_count, int(dense_q_tiles)) - q_tile_start,
    )
    launch_group(q_tile_start, dense_count, kv_tiles, False, sparse_lut)
    sparse_count = q_tile_count - dense_count
    if sparse_count:
        sparse_global_start = q_tile_start + dense_count
        route_start = (
            sparse_global_start
            - int(dense_q_tiles)
            - sparse_lut_q_start
        )
        if route_start < 0 or route_start + sparse_count > sparse_lut.shape[-2]:
            raise TritonBF16Error('streamed BF16 Triton sparse route slab differs')
        route = sparse_lut[
            ...,
            route_start:route_start + sparse_count,
            :,
        ].contiguous()
        launch_group(
            sparse_global_start,
            sparse_count,
            int(sparse_selected),
            True,
            route,
        )
    return output


class TritonBF16Backend:
    name = 'triton_sparse_bf16'
    requires_runtime_context = True
    approximate = True

    def __init__(self, config=None, *, router=None, projector=None, spec=None):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        self.spec = spec or TritonBF16Spec()
        if not isinstance(self.spec, TritonBF16Spec):
            raise TypeError('spec must be TritonBF16Spec')
        self.projector = projector
        self._streamed_q_announced = False
        self.router = router or SparseTileRouter(
            self.config, q_tile=Q_TILE, kv_tile=KV_TILE
        )
        if (self.router.q_tile, self.router.kv_tile) != (Q_TILE, KV_TILE):
            raise TritonBF16Error('router geometry must be 64Q x 64KV')

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.spec.signature,
            None if self.projector is None else self.projector.installation_signature,
        )

    @staticmethod
    def _snapshot(transformer_options, sequence):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise TritonBF16Error('BF16 Triton requires a valid H3 runtime layout')
        if int(snapshot.layout.seq_len) != int(sequence):
            raise TritonBF16Error('BF16 Triton runtime sequence differs from QKV')
        return snapshot

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        _validate_qkv(q, k, v)
        snapshot = self._snapshot(transformer_options, q.shape[-2])
        budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            lut, valid, route_metadata = self.router.build_lut(
                q,
                k,
                snapshot.layout,
                budget,
            )
        except SparseRouterError as exc:
            raise TritonBF16Error('BF16 Triton routing failed: %s' % exc) from exc
        metadata = route_metadata.as_dict()
        sparse_lut, dense, sparse, selected = _compact_route(lut, valid, metadata)
        metadata.update(
            {
                'layer': int(layer_index),
                'sparse_backend': self.name,
                'route_format': 'dense_implicit_plus_sparse_absolute_int32',
                'program_shape': 'one_64q_tile_x_one_head_x_full_d128',
            }
        )
        return PreparedTritonBF16(
            q=q,
            k=k,
            v=v,
            sparse_lut=sparse_lut,
            dense_q_tiles=dense,
            sparse_q_tiles=sparse,
            sparse_selected=selected,
            layer_index=int(layer_index),
            metadata=metadata,
        )

    def prepare_projected(
        self, projected, *, layer_index, transformer_options
    ):
        from .triton_bf16_streamed import (
            StreamedTritonBF16QKV,
            prepare_streamed_triton_bf16,
        )

        if isinstance(projected, StreamedTritonBF16QKV):
            try:
                return prepare_streamed_triton_bf16(
                    self,
                    projected,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
            except Exception as exc:
                projected.release()
                raise TritonBF16Error(
                    'streamed BF16 Triton preparation failed'
                ) from exc
        if not isinstance(projected, PreparedBF16QKV):
            raise TritonBF16Error('BF16 Triton requires chunked BF16 Q/K/V')
        prepared = self.prepare(
            projected.q,
            projected.k,
            projected.v,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        prepared.metadata['qkv_projection'] = (
            'bounded_bf16_chunks_into_final_hnd_carrier'
        )
        return prepared

    def execute_projected(self, module, prepared):
        from .triton_bf16_streamed import (
            PreparedStreamedTritonBF16,
            execute_streamed_triton_bf16,
        )

        if isinstance(prepared, PreparedStreamedTritonBF16):
            try:
                result = execute_streamed_triton_bf16(
                    module,
                    self,
                    prepared,
                )
                if not self._streamed_q_announced:
                    self._streamed_q_announced = True
                    logging.debug(
                        '[H3 Optimizations] streamed BF16 Triton ran: '
                        'global K/V, aliased Q/output slab, chunk_rows=%d',
                        int(prepared.metadata['query_chunk_rows']),
                    )
                return result
            except Exception as exc:
                prepared.release()
                raise TritonBF16Error(
                    'streamed BF16 Triton execution failed'
                ) from exc
        return None

    def execute(self, prepared):
        from .triton_bf16_streamed import PreparedStreamedTritonBF16

        if isinstance(prepared, PreparedStreamedTritonBF16):
            raise TritonBF16Error(
                'streamed BF16 Triton must execute through execute_projected'
            )
        try:
            return _launch(prepared)
        except Exception as exc:
            layer = getattr(prepared, 'layer_index', -1)
            raise TritonBF16Error(
                'BF16 Triton kernel failed at layer %d' % layer
            ) from exc

    def as_status(self):
        return {
            'mode': self.name,
            'video_budget': float(self.config.video_budget),
            'denser_early_late_steps': bool(self.config.denser_early_late_steps),
            'density_mode': self.config.density_mode,
            'sparse_q_tile': Q_TILE,
            'sparse_kv_tile': KV_TILE,
            'qkv_carrier': 'bf16_hnd',
            'probability_value_path': 'bf16_x_bf16_fp32',
            'route_format': 'dense_implicit_plus_sparse_absolute_int32',
            'chunked_qkv': self.projector is not None,
            'streamed_q': bool(getattr(self.projector, 'streamed_q', False)),
            'qkv_lifetime': (
                'global_bf16_kv_bounded_q_inplace_o'
                if getattr(self.projector, 'streamed_q', False)
                else 'global_bf16_qkv'
            ),
            'approximate': True,
        }


__all__ = [
    'PreparedTritonBF16',
    'TritonBF16Backend',
    'TritonBF16Error',
    'TritonBF16Spec',
    '_compact_route',
    '_launch',
    '_launch_streamed_chunk',
    'preflight_triton_bf16',
]
