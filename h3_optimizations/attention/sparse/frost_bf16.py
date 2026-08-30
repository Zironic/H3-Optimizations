'''NVIDIA FROST-derived BF16 block-sparse attention for SM89 MiniMax H3.'''

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch

from ... import diagnostics
from ...runtime.context import get_runtime_snapshot
from .config import HybridSparseConfig, resolve_video_budget
from .frost_loader import (
    FROST_ABI,
    FrostDriverError,
    artifact_path,
    driver_available,
    launch,
    symbol_path,
    unavailable_reason,
)
from .frost_route import build_full_absolute_route
from .router import SparseRouterError, SparseTileRouter
from .triton_route import TritonRouteError


Q_TILE = 64
KV_TILE = 64
HEADS = 56
HEAD_DIM = 128


class FrostBF16Error(RuntimeError):
    pass


@dataclass(frozen=True)
class FrostBF16Spec:
    q_tile: int = Q_TILE
    kv_tile: int = KV_TILE
    heads: int = HEADS
    head_dim: int = HEAD_DIM
    abi: int = FROST_ABI

    @property
    def signature(self):
        return (
            'frost_bf16_sm89',
            int(self.abi),
            int(self.q_tile),
            int(self.kv_tile),
            int(self.heads),
            int(self.head_dim),
        )


@dataclass
class PreparedFrostBF16:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    route: torch.Tensor
    counts: torch.Tensor
    output_storage: torch.Tensor
    output: torch.Tensor
    metadata: dict


def preflight_frost_bf16(
    *,
    cuda_available,
    capability_getter,
    driver_probe=driver_available,
):
    if not bool(cuda_available()):
        raise FrostBF16Error('FROST BF16 requires NVIDIA CUDA')
    capability = tuple(capability_getter())
    if capability != (8, 9):
        raise FrostBF16Error(
            'FROST BF16 is compiled for SM89; found SM%d%d' % capability
        )
    if not artifact_path().is_file() or not symbol_path().is_file():
        raise FrostBF16Error('the packaged FROST BF16 SM89 artifact is missing')
    if not driver_probe():
        raise FrostBF16Error(unavailable_reason() or 'the CUDA Driver API is unavailable')
    return FrostBF16Spec()


def _check_stride(name, tensor):
    if int(tensor.data_ptr()) % 16:
        raise FrostBF16Error('%s does not satisfy the FROST 16-byte alignment ABI' % name)
    if tensor.stride(-1) != 1:
        raise FrostBF16Error('%s head dimension must be contiguous' % name)
    if any(int(value) < 0 or int(value) > 0x7FFFFFFF for value in tensor.stride()[:3]):
        raise FrostBF16Error('%s stride exceeds the FROST int32 ABI' % name)


def _check_sequence_major(name, tensor):
    _check_stride(name, tensor)
    sequence = int(tensor.shape[2])
    heads = int(tensor.shape[1])
    head_dim = int(tensor.shape[3])
    expected = (
        sequence * heads * head_dim,
        head_dim,
        heads * head_dim,
        1,
    )
    if tuple(int(value) for value in tensor.stride()) != expected:
        raise FrostBF16Error(
            '%s must be an HND view over contiguous sequence-major storage'
            % name
        )


class FrostBF16Executor:
    def __init__(
        self,
        spec=None,
        *,
        launcher=launch,
        stream_getter=None,
        allow_cpu_for_tests=False,
    ):
        self.spec = spec or FrostBF16Spec()
        self.launcher = launcher
        self.stream_getter = stream_getter or (
            lambda device: torch.cuda.current_stream(device).cuda_stream
        )
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)

    def prepare(self, q, k, v, route, counts, *, layer_index, metadata):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise FrostBF16Error('FROST BF16 expects rank-4 HND Q/K/V')
        if k.shape != v.shape:
            raise FrostBF16Error('FROST BF16 requires equal K/V shapes')
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise FrostBF16Error('FROST BF16 requires BF16 Q/K/V')
        if not self.allow_cpu_for_tests and not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise FrostBF16Error('FROST BF16 requires CUDA Q/K/V')
        expected = (1, self.spec.heads, self.spec.head_dim)
        for name, tensor in (('Q', q), ('K', k), ('V', v)):
            if (int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[3])) != expected:
                raise FrostBF16Error(
                    'FROST BF16 requires [1,%d,S,%d] %s, got %s' % (
                        self.spec.heads,
                        self.spec.head_dim,
                        name,
                        tuple(int(value) for value in tensor.shape),
                    )
                )
        if q.device != k.device or q.device != v.device:
            raise FrostBF16Error('FROST BF16 Q/K/V devices differ')
        for name, tensor in (('Q', q), ('K', k), ('V', v)):
            _check_sequence_major(name, tensor)
        sequence_q = int(q.shape[-2])
        sequence_kv = int(k.shape[-2])
        q_tiles = (sequence_q + self.spec.q_tile - 1) // self.spec.q_tile
        kv_tiles = (sequence_kv + self.spec.kv_tile - 1) // self.spec.kv_tile
        if tuple(route.shape) != (1, self.spec.heads, q_tiles, kv_tiles):
            raise FrostBF16Error('FROST route shape does not match Q/K/V')
        if tuple(counts.shape) != (1, self.spec.heads, q_tiles):
            raise FrostBF16Error('FROST route counts shape does not match Q/K/V')
        if route.dtype != torch.int32 or counts.dtype != torch.int32:
            raise FrostBF16Error('FROST route and counts must be int32')
        if not route.is_contiguous() or not counts.is_contiguous():
            raise FrostBF16Error('FROST route and counts must be contiguous')
        _check_stride('route', route)
        _check_stride('counts', counts)
        if route.device != q.device or counts.device != q.device:
            raise FrostBF16Error('FROST route and Q/K/V devices differ')
        storage = torch.empty(
            (1, sequence_q, self.spec.heads, self.spec.head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
        return PreparedFrostBF16(
            q=q,
            k=k,
            v=v,
            route=route,
            counts=counts,
            output_storage=storage,
            output=storage.permute(0, 2, 1, 3),
            metadata=dict(metadata, layer=int(layer_index)),
        )

    def execute(self, prepared):
        try:
            self.launcher(
                prepared.q,
                prepared.k,
                prepared.v,
                prepared.output_storage,
                prepared.route,
                prepared.counts,
                scale_log2=(1.0 / math.sqrt(self.spec.head_dim)) * math.log2(math.e),
                stream=self.stream_getter(prepared.q.device),
            )
        except FrostDriverError as error:
            raise FrostBF16Error('FROST BF16 launch failed: %s' % error) from error
        return prepared.output


class FrostBF16Backend:
    name = 'frost_bf16_sm89'
    requires_runtime_context = True
    approximate = True
    output_layout = 'nhd'

    def __init__(
        self,
        config=None,
        *,
        spec=None,
        router=None,
        executor=None,
        projector=None,
        allow_cpu_for_tests=False,
    ):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        self.spec = spec or FrostBF16Spec()
        self.router = router or SparseTileRouter(
            self.config,
            q_tile=self.spec.q_tile,
            kv_tile=self.spec.kv_tile,
        )
        self.executor = executor or FrostBF16Executor(
            self.spec,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        self.projector = projector
        self._streamed_q_announced = False
        if (self.router.q_tile, self.router.kv_tile) != (
            self.spec.q_tile,
            self.spec.kv_tile,
        ):
            raise FrostBF16Error('FROST router geometry does not match the cubin ABI')

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.spec.signature,
            (type(self.router).__module__, type(self.router).__qualname__),
            getattr(self.projector, 'installation_signature', None),
        )

    @staticmethod
    def _snapshot(transformer_options, sequence):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise FrostBF16Error('FROST BF16 requires a valid H3 runtime layout')
        if int(snapshot.layout.seq_len) != int(sequence):
            raise FrostBF16Error('FROST BF16 runtime layout sequence mismatch')
        return snapshot

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        snapshot = self._snapshot(transformer_options, q.shape[-2])
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            with diagnostics.stage('sparse_route'):
                route, counts, mask_metadata = build_full_absolute_route(
                    self.router,
                    q,
                    k,
                    snapshot.layout,
                    video_budget,
                )
        except (SparseRouterError, TritonRouteError) as error:
            raise FrostBF16Error('FROST sparse routing failed: %s' % error) from error
        metadata = mask_metadata.as_dict()
        metadata.update(
            {
                'sparse_sage_heads': int(q.shape[1]),
                'total_q_video_tiles': (
                    int(mask_metadata.pure_video_q_tiles) * int(q.shape[1])
                ),
            }
        )
        return self.executor.prepare(
            q,
            k,
            v,
            route,
            counts,
            layer_index=layer_index,
            metadata=metadata,
        )

    def execute(self, prepared):
        from .frost_bf16_streamed import PreparedStreamedFrostBF16

        if isinstance(prepared, PreparedStreamedFrostBF16):
            raise FrostBF16Error(
                'streamed FROST BF16 must execute through execute_projected'
            )
        return self.executor.execute(prepared)

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        from .frost_bf16_streamed import (
            StreamedFrostBF16QKV,
            prepare_streamed_frost_bf16,
        )

        if not isinstance(projected, StreamedFrostBF16QKV):
            raise FrostBF16Error('FROST BF16 requires streamed BF16 QKV')
        try:
            return prepare_streamed_frost_bf16(
                self,
                projected,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
        except Exception as error:
            projected.release()
            raise FrostBF16Error(
                'streamed FROST BF16 preparation failed'
            ) from error

    def execute_projected(self, module, prepared):
        from .frost_bf16_streamed import (
            PreparedStreamedFrostBF16,
            execute_streamed_frost_bf16,
        )

        if not isinstance(prepared, PreparedStreamedFrostBF16):
            return None
        try:
            result = execute_streamed_frost_bf16(module, self, prepared)
            if not self._streamed_q_announced:
                self._streamed_q_announced = True
                logging.debug(
                    '[H3 Optimizations] streamed FROST BF16 ran: '
                    'global sequence-major K/V, bounded Q/output, chunk_rows=%d',
                    int(prepared.metadata['query_chunk_rows']),
                )
            return result
        except Exception as error:
            prepared.release()
            raise FrostBF16Error(
                'streamed FROST BF16 execution failed'
            ) from error

    def as_status(self):
        return {
            'mode': self.config.mode,
            'video_budget': float(self.config.video_budget),
            'density_mode': self.config.density_mode,
            'sparse_architecture': 'frost_bf16_sm89',
            'sparse_q_tile': self.spec.q_tile,
            'sparse_kv_tile': self.spec.kv_tile,
            'sparse_v_format': 'bf16',
            'route_encoding': 'absolute_full_int32_direct',
            'output_layout': self.output_layout,
            'approximate': True,
            'fused_qkv': self.projector is not None,
            'qkv_projector': getattr(self.projector, 'name', None),
            'streamed_q': bool(getattr(self.projector, 'streamed_q', False)),
            'qkv_lifetime': (
                'global_sequence_major_bf16_kv_bounded_q'
                if getattr(self.projector, 'streamed_q', False)
                else 'global_sequence_major_bf16_qkv'
            ),
            'frost_abi': self.spec.abi,
        }
