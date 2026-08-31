'''Sparse H3 routing executed through ComfyUI's existing dense attention.

This backend knows nothing about the physical attention kernel. It owns only
routing and bounded K/V packing. Each packed sparse problem is handed back to
the same Comfy attention entry point that already runs dense H3 on the device.
The selected consumer is probed at 64Q x 64KV first and then 128Q x 128KV.
'''

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

import torch

import comfy.ldm.minimax.model as h3_model

from ... import diagnostics
from ...normalized_rows import attention_output_buffer
from ...qkv.bf16 import BF16QKVBindingError
from ...qkv.fp8 import FP8BindingError
from ...qkv.int8 import ConvRotINT8BindingError
from ...qkv.projectors import TritonSparseQKVProjector
from ...qkv.streamed import StreamedQKVBindingError
from ...qkv.w4a8 import W4A8BindingError
from ...runtime.context import get_runtime_snapshot
from .config import HybridSparseConfig, resolve_video_budget
from .router import SparseRouterError, SparseTileRouter
from .triton_route import (
    TritonRouteError,
    build_compact_absolute_route_chunk,
    prepare_compact_absolute_route_chunks,
)


LOG_PREFIX = '[H3 Optimizations]'
HEAD_DIM = 128
PACK_TARGET_BYTES = 256 * 1024 * 1024
OUT_PROJ_CHUNK_ROWS = 2048
# Match the established dense-Sage relative-L2 gate while retaining a tighter
# absolute-error bound than the direct Sage carrier tests.
_PROBE_REL_L2 = 0.05
_PROBE_MAX_ABS = 0.10
_probe_lock = threading.Lock()
_probe_results = {}
_packed_runtime_fallback_warned = False


class ExistingDenseSparseError(RuntimeError):
    pass


class ExistingDenseConsumerError(ExistingDenseSparseError):
    pass


@dataclass(frozen=True)
class ExistingDenseSparseSpec:
    q_tile: int
    kv_tile: int
    max_batch_entries: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def signature(self):
        return (
            'existing_dense_sparse',
            int(self.q_tile),
            int(self.kv_tile),
            int(self.max_batch_entries),
            str(self.dtype),
        )


@dataclass
class PreparedExistingDenseSparse:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    lut: torch.Tensor
    valid: torch.Tensor
    metadata: dict
    transformer_options: dict


@dataclass
class PreparedStreamedExistingDenseSparse:
    projected: object
    route_plan: object
    dense_q_tiles: int
    sparse_q_tiles: int
    metadata: dict
    transformer_options: dict

    def release(self):
        if self.projected is not None:
            self.projected.release()
        if self.route_plan is not None:
            self.route_plan.release()
        self.projected = None
        self.route_plan = None


class ExistingDenseSparseQKVProjector(TritonSparseQKVProjector):
    '''Reuse the existing streamed-BF16 carrier without requiring Triton.'''

    # This name intentionally matches the already documented retained-K/V
    # lifetime in the status formatter.  The projected object itself is the
    # generic streamed BF16 carrier that Triton also consumes; projection does
    # not import or launch Triton.
    name = 'streamed_dense_bf16_qkv'

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
            if self.required:
                raise ExistingDenseSparseError(
                    'required streamed QKV needs rank-2 CUDA/ROCm BF16 input'
                )
            return None
        try:
            return super().try_project(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
        except (
            BF16QKVBindingError,
            ConvRotINT8BindingError,
            FP8BindingError,
            StreamedQKVBindingError,
            W4A8BindingError,
        ):
            if self.required:
                raise
            return None


def _probe_options(transformer_options):
    options = transformer_options or {}
    override = options.get('optimized_attention_override')
    return {} if override is None else {'optimized_attention_override': override}


def _call_existing_dense(q, k, v, transformer_options, *, heads):
    return h3_model.optimized_attention(
        q,
        k,
        v,
        int(heads),
        mask=None,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=transformer_options or {},
    )


def _relative_l2(actual, expected):
    error = (actual.float() - expected.float()).norm()
    return (error / expected.float().norm().clamp_min(1e-12)).item()


def _reference_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    return torch.matmul(scores.softmax(dim=-1), v.float()).to(q.dtype)


def _probe_case(
    device,
    transformer_options,
    *,
    q_rows,
    k_rows,
    batch,
    dtype=torch.bfloat16,
):
    generator = torch.Generator(device=device).manual_seed(
        20260831 + int(q_rows) * 17 + int(k_rows) * 31 + int(batch)
    )
    q = torch.randn(
        (int(batch), 1, int(q_rows), HEAD_DIM),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        (int(batch), 1, int(k_rows), HEAD_DIM),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        (int(batch), 1, int(k_rows), HEAD_DIM),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    actual = _call_existing_dense(
        q,
        k,
        v,
        transformer_options,
        heads=1,
    )
    if tuple(actual.shape) != tuple(q.shape):
        raise ExistingDenseSparseError(
            'existing attention returned %s for probe input %s'
            % (tuple(actual.shape), tuple(q.shape))
        )
    expected = _reference_attention(q, k, v)
    finite = bool(torch.isfinite(actual).all())
    rel_l2 = _relative_l2(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    if not (
        finite
        and rel_l2 < _PROBE_REL_L2
        and max_abs < _PROBE_MAX_ABS
    ):
        raise ExistingDenseSparseError(
            'existing attention probe numerics failed: finite=%s rel_l2=%.6f '
            'max_abs=%.6f' % (finite, rel_l2, max_abs)
        )


def _probe_geometry(
    device,
    transformer_options,
    q_tile,
    kv_tile,
    *,
    dtype=torch.bfloat16,
):
    # First establish the logical shape independently of batching.  The ragged
    # rectangular case mirrors a selected route that includes the final partial
    # KV tile.
    for k_rows in (kv_tile, kv_tile * 2, kv_tile * 2 - 7):
        _probe_case(
            device,
            transformer_options,
            q_rows=q_tile,
            k_rows=k_rows,
            batch=1,
            dtype=dtype,
        )

    # Batching is an optimization only.  If an aggressive dense consumer has a
    # batch restriction, retain correctness and use smaller packed groups.
    for batch in (16, 8, 4, 2):
        try:
            _probe_case(
                device,
                transformer_options,
                q_rows=q_tile,
                k_rows=kv_tile * 2 - 7,
                batch=batch,
                dtype=dtype,
            )
            return batch
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            continue
    return 1


def _probe_key(device, transformer_options, *, dtype=torch.bfloat16):
    device = torch.device(device)
    index = (
        torch.cuda.current_device()
        if device.index is None
        else int(device.index)
    )
    properties = torch.cuda.get_device_properties(index)
    architecture = getattr(properties, 'gcnArchName', None)
    if architecture is None:
        architecture = 'sm%d%d' % (properties.major, properties.minor)
    architecture = str(architecture).split(':')[0]
    override = (transformer_options or {}).get('optimized_attention_override')
    consumer = h3_model.optimized_attention if override is None else override
    return (
        index,
        architecture,
        torch.cuda.get_device_name(index),
        id(consumer),
        dtype,
    )


def probe_existing_dense_sparse(
    transformer_options=None,
    device=None,
    *,
    force=False,
    dtype=torch.bfloat16,
):
    '''Return a proven logical geometry for the active Comfy dense consumer.'''
    if dtype not in (torch.bfloat16, torch.float32):
        raise ExistingDenseSparseError(
            'existing-dense sparse attention supports BF16 or FP32 execution, got %s'
            % dtype
        )
    device = torch.device('cuda' if device is None else device)
    options = _probe_options(transformer_options)
    key = _probe_key(device, options, dtype=dtype)
    with _probe_lock:
        if not force and key in _probe_results:
            result = _probe_results[key]
            if isinstance(result, Exception):
                raise ExistingDenseSparseError(str(result))
            return result

        failures = []
        for q_tile, kv_tile in ((64, 64), (128, 128)):
            try:
                max_batch = _probe_geometry(
                    device,
                    options,
                    q_tile,
                    kv_tile,
                    dtype=dtype,
                )
                spec = ExistingDenseSparseSpec(
                    q_tile=q_tile,
                    kv_tile=kv_tile,
                    max_batch_entries=max_batch,
                    dtype=dtype,
                )
                _probe_results[key] = spec
                logging.info(
                    '%s existing-dense sparse probe selected %dQ x %dKV %s '
                    '(packed batch <= %d)',
                    LOG_PREFIX,
                    q_tile,
                    kv_tile,
                    dtype,
                    max_batch,
                )
                return spec
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as error:
                failures.append(
                    '%dQ x %dKV: %s: %s'
                    % (q_tile, kv_tile, type(error).__name__, error)
                )

        error = ExistingDenseSparseError(
            'the existing Comfy attention rejected both sparse adapter '
            'geometries (%s)' % '; '.join(failures)
        )
        _probe_results[key] = error
        raise error


def _validate_qkv(q, k, v, *, dtype=None):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires equal HND rank-4 Q/K/V'
        )
    if q.shape[0] != 1 or q.shape[-1] != HEAD_DIM:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires batch 1 and head_dim 128'
        )
    if q.dtype not in (torch.bfloat16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires matching BF16 or FP32 Q/K/V'
        )
    if dtype is not None and q.dtype != dtype:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention was probed for %s but received %s Q/K/V'
            % (dtype, q.dtype)
        )
    if q.device != k.device or q.device != v.device or not q.is_cuda:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires one CUDA/ROCm device'
        )
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires contiguous head dimensions'
        )


def _packed_batch_limit(spec, k_rows, head_dim, element_size):
    entry_bytes = max(1, int(k_rows) * int(head_dim) * int(element_size) * 2)
    memory_limit = max(1, PACK_TARGET_BYTES // entry_bytes)
    return max(1, min(int(spec.max_batch_entries), int(memory_limit)))


def _execute_packed_sparse(
    q,
    k,
    v,
    selected_tiles,
    transformer_options,
    *,
    spec,
):
    '''Execute independent (Q tile, head) routes as dense batch entries.'''
    if q.ndim != 4 or q.shape[0] != 1:
        raise ExistingDenseSparseError('packed sparse Q must be HND batch 1')
    if selected_tiles.ndim != 4 or selected_tiles.shape[0] != 1:
        raise ExistingDenseSparseError('packed sparse route must be rank-4 batch 1')

    heads = int(q.shape[1])
    sequence = int(k.shape[-2])
    rows = int(q.shape[-2])
    q_tiles = (rows + int(spec.q_tile) - 1) // int(spec.q_tile)
    if tuple(selected_tiles.shape[:3]) != (1, heads, q_tiles):
        raise ExistingDenseSparseError(
            'packed sparse route shape %s does not match Q slab %s'
            % (tuple(selected_tiles.shape), tuple(q.shape))
        )
    selected_count = int(selected_tiles.shape[-1])
    if selected_count <= 0:
        raise ExistingDenseSparseError('packed sparse route selected no KV tiles')

    kv_tile = int(spec.kv_tile)
    q_tile = int(spec.q_tile)
    kv_tiles = (sequence + kv_tile - 1) // kv_tile
    tail_pad = kv_tiles * kv_tile - sequence

    selected = selected_tiles[0].reshape(heads * q_tiles, selected_count).to(torch.int64)
    if bool(((selected < 0) | (selected >= kv_tiles)).any()):
        raise ExistingDenseSparseError('packed sparse route contains invalid KV tiles')

    head_ids = (
        torch.arange(heads, device=q.device, dtype=torch.int64)
        .view(heads, 1)
        .expand(heads, q_tiles)
        .reshape(-1)
    )
    tile_ids = (
        torch.arange(q_tiles, device=q.device, dtype=torch.int64)
        .view(1, q_tiles)
        .expand(heads, q_tiles)
        .reshape(-1)
    )
    output = torch.empty_like(q)
    tail_selected = (
        selected[:, -1].eq(kv_tiles - 1)
        if tail_pad
        else torch.zeros(selected.shape[0], dtype=torch.bool, device=q.device)
    )

    offsets = torch.arange(kv_tile, device=q.device, dtype=torch.int64)
    q_offsets = torch.arange(q_tile, device=q.device, dtype=torch.int64)

    for uses_tail in ((False, True) if tail_pad else (False,)):
        entry_ids = torch.nonzero(
            tail_selected if uses_tail else ~tail_selected,
            as_tuple=False,
        ).flatten()
        entry_count = int(entry_ids.numel())
        if not entry_count:
            continue
        actual_k_rows = selected_count * kv_tile - (tail_pad if uses_tail else 0)
        batch_limit = _packed_batch_limit(
            spec,
            actual_k_rows,
            q.shape[-1],
            k.element_size(),
        )

        for first in range(0, entry_count, batch_limit):
            ids = entry_ids[first:first + batch_limit]
            entry_heads = head_ids.index_select(0, ids)
            entry_tiles = tile_ids.index_select(0, ids)
            entry_selected = selected.index_select(0, ids)

            kv_rows = (
                entry_selected[..., None] * kv_tile + offsets
            ).reshape(ids.numel(), -1)
            if uses_tail:
                # Absolute tile ids are sorted, so the ragged final tile is the
                # final packed block whenever it is present.
                kv_rows = kv_rows[:, :actual_k_rows]

            q_rows = entry_tiles[:, None] * q_tile + q_offsets
            q_valid = q_rows < rows
            safe_q_rows = q_rows.clamp_max(max(0, rows - 1))
            q_batch = q[0, entry_heads[:, None], safe_q_rows, :]
            if not bool(q_valid.all()):
                q_batch = q_batch.masked_fill(~q_valid[..., None], 0)

            k_batch = k[0, entry_heads[:, None], kv_rows, :]
            v_batch = v[0, entry_heads[:, None], kv_rows, :]
            try:
                with diagnostics.stage('sparse_attention_kernel'):
                    dense_out = _call_existing_dense(
                        q_batch.unsqueeze(1),
                        k_batch.unsqueeze(1),
                        v_batch.unsqueeze(1),
                        transformer_options,
                        heads=1,
                    )
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as error:
                raise ExistingDenseConsumerError(
                    'existing attention rejected packed sparse input: %s: %s'
                    % (type(error).__name__, error)
                ) from error
            expected = (ids.numel(), 1, q_tile, int(q.shape[-1]))
            if tuple(dense_out.shape) != expected:
                raise ExistingDenseConsumerError(
                    'existing attention returned %s for packed sparse shape %s'
                    % (tuple(dense_out.shape), expected)
                )

            output_heads = entry_heads[:, None].expand_as(q_rows)[q_valid]
            output_rows = q_rows[q_valid]
            output[0, output_heads, output_rows, :] = dense_out[:, 0][q_valid]
            del q_batch, k_batch, v_batch, dense_out

    return output


def _execute_packed_or_dense(
    q,
    k,
    v,
    selected_tiles,
    transformer_options,
    *,
    spec,
):
    '''Fail an unsupported packed consumer call open to dense for this Q slab.'''
    global _packed_runtime_fallback_warned
    try:
        return _execute_packed_sparse(
            q,
            k,
            v,
            selected_tiles,
            transformer_options,
            spec=spec,
        )
    except ExistingDenseConsumerError as error:
        if not _packed_runtime_fallback_warned:
            _packed_runtime_fallback_warned = True
            logging.warning(
                '%s packed sparse call failed; using full existing dense '
                'attention for the affected Q slab: %s: %s',
                LOG_PREFIX,
                type(error).__name__,
                error,
            )
        return _call_existing_dense(
            q,
            k,
            v,
            transformer_options,
            heads=int(q.shape[1]),
        )


def _project_hnd_rows(module, destination, hnd, start):
    rows = int(hnd.shape[-2])
    heads = int(hnd.shape[1])
    head_dim = int(hnd.shape[-1])
    flat = hnd.transpose(1, 2).reshape(rows, heads * head_dim)
    for local_start in range(0, rows, OUT_PROJ_CHUNK_ROWS):
        local_end = min(local_start + OUT_PROJ_CHUNK_ROWS, rows)
        projected = module.out_proj(flat[local_start:local_end])
        destination[
            int(start) + local_start:int(start) + local_end
        ].copy_(projected)
        del projected
    del flat


class ExistingDenseSparseBackend:
    name = 'existing_dense_sparse'
    requires_runtime_context = True
    approximate = True

    def __init__(self, config=None, *, spec):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        if not isinstance(spec, ExistingDenseSparseSpec):
            raise TypeError('spec must be ExistingDenseSparseSpec')
        self.spec = spec
        self.router = SparseTileRouter(
            self.config,
            q_tile=spec.q_tile,
            kv_tile=spec.kv_tile,
        )

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.spec.signature,
        )

    @staticmethod
    def _snapshot(transformer_options, sequence):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise ExistingDenseSparseError(
                'existing-dense sparse attention requires a valid H3 runtime layout'
            )
        if int(snapshot.layout.seq_len) != int(sequence):
            raise ExistingDenseSparseError(
                'existing-dense sparse runtime sequence differs from QKV'
            )
        return snapshot

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        _validate_qkv(q, k, v, dtype=self.spec.dtype)
        snapshot = self._snapshot(transformer_options, q.shape[-2])
        budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            lut, valid, metadata = self.router.build_lut(
                q,
                k,
                snapshot.layout,
                budget,
            )
        except SparseRouterError as error:
            raise ExistingDenseSparseError(
                'existing-dense sparse routing failed: %s' % error
            ) from error
        details = metadata.as_dict()
        details.update(
            {
                'layer': int(layer_index),
                'sparse_backend': self.name,
                'route_format': 'delta_int32_then_packed_dense',
                'logical_geometry': '%dQx%dKV'
                % (self.spec.q_tile, self.spec.kv_tile),
            }
        )
        return PreparedExistingDenseSparse(
            q=q,
            k=k,
            v=v,
            lut=lut,
            valid=valid,
            metadata=details,
            transformer_options=transformer_options,
        )

    def execute(self, prepared):
        if not isinstance(prepared, PreparedExistingDenseSparse):
            raise ExistingDenseSparseError('invalid existing-dense sparse payload')
        q, k, v = prepared.q, prepared.k, prepared.v
        sequence = int(q.shape[-2])
        heads = int(q.shape[1])
        dense_q_tiles = int(prepared.metadata['dense_q_tiles'])
        sparse_q_tiles = int(prepared.metadata['sparse_q_tiles'])
        if not sparse_q_tiles:
            return _call_existing_dense(
                q,
                k,
                v,
                prepared.transformer_options,
                heads=heads,
            )

        output = torch.empty_like(q)
        dense_rows = min(sequence, dense_q_tiles * int(self.spec.q_tile))
        if dense_rows:
            with diagnostics.stage('sparse_attention_kernel'):
                output[..., :dense_rows, :].copy_(
                    _call_existing_dense(
                        q[..., :dense_rows, :],
                        k,
                        v,
                        prepared.transformer_options,
                        heads=heads,
                    )
                )

        context_tiles = int(prepared.metadata['kv_tiles']) - int(
            prepared.metadata['pure_video_kv_tiles']
        )
        selected_count = context_tiles + int(
            prepared.metadata['retained_video_kv_tiles']
        )
        sparse_lut = prepared.lut[..., dense_q_tiles:, :selected_count]
        absolute = torch.cumsum(sparse_lut, dim=-1, dtype=torch.int32).contiguous()
        sparse_output = _execute_packed_or_dense(
            q[..., dense_rows:, :],
            k,
            v,
            absolute,
            prepared.transformer_options,
            spec=self.spec,
        )
        output[..., dense_rows:, :].copy_(sparse_output)
        return output

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        from .triton_bf16_streamed import StreamedTritonBF16QKV

        if not isinstance(projected, StreamedTritonBF16QKV):
            raise ExistingDenseSparseError(
                'existing-dense sparse attention expected streamed BF16 QKV'
            )
        if self.spec.dtype != torch.bfloat16:
            projected.release()
            raise ExistingDenseSparseError(
                'streamed BF16 QKV does not match the probed %s execution dtype'
                % self.spec.dtype
            )
        if int(projected.chunk_rows) % int(self.spec.q_tile):
            projected.release()
            raise ExistingDenseSparseError(
                'streamed Q chunk size must be divisible by the selected Q tile'
            )
        snapshot = self._snapshot(transformer_options, projected.sequence)
        budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
            layer_index,
        )
        try:
            k_summary = self.router._mean_pool(projected.k, self.spec.kv_tile)
            route_plan = prepare_compact_absolute_route_chunks(
                self.router,
                k_summary,
                snapshot.layout,
                budget,
            )
        except (SparseRouterError, TritonRouteError, Exception) as error:
            projected.release()
            raise ExistingDenseSparseError(
                'streamed existing-dense sparse route preparation failed: %s'
                % error
            ) from error
        projected.k_summary = None
        metadata = route_plan.metadata.as_dict()
        metadata.update(
            {
                'layer': int(layer_index),
                'sparse_backend': self.name,
                'logical_geometry': '%dQx%dKV'
                % (self.spec.q_tile, self.spec.kv_tile),
                'qkv_lifetime': 'streamed_q_global_bf16_kv',
                'packed_dense_batch_max': int(self.spec.max_batch_entries),
            }
        )
        return PreparedStreamedExistingDenseSparse(
            projected=projected,
            route_plan=route_plan,
            dense_q_tiles=int(route_plan.metadata.dense_q_tiles),
            sparse_q_tiles=int(route_plan.metadata.sparse_q_tiles),
            metadata=metadata,
            transformer_options=transformer_options,
        )

    def execute_projected(self, module, prepared):
        if not isinstance(prepared, PreparedStreamedExistingDenseSparse):
            return None
        projected = prepared.projected
        if getattr(module, '_module', module) is not projected.module:
            prepared.release()
            raise ExistingDenseSparseError(
                'streamed existing-dense sparse attention module changed'
            )

        result = attention_output_buffer(projected.x)
        sequence = int(projected.sequence)
        heads = int(projected.heads)
        q_tile = int(self.spec.q_tile)
        try:
            for start in range(0, sequence, int(projected.chunk_rows)):
                end = min(start + int(projected.chunk_rows), sequence)
                q = projected.project_q(start, end)
                q_summary = self.router._mean_pool(q, q_tile)
                try:
                    selected = build_compact_absolute_route_chunk(
                        self.router,
                        q_summary,
                        prepared.route_plan,
                        q_tile_start=start // q_tile,
                    )
                except TritonRouteError as error:
                    raise ExistingDenseSparseError(
                        'streamed existing-dense sparse Q routing failed: %s'
                        % error
                    ) from error
                del q_summary

                dense_end = min(
                    end,
                    prepared.dense_q_tiles * q_tile,
                )
                dense_rows = max(0, dense_end - start)
                if dense_rows:
                    with diagnostics.stage('sparse_attention_kernel'):
                        dense_out = _call_existing_dense(
                            q[..., :dense_rows, :],
                            projected.k,
                            projected.v,
                            prepared.transformer_options,
                            heads=heads,
                        )
                    _project_hnd_rows(module, result, dense_out, start)
                    del dense_out

                sparse_rows = (end - start) - dense_rows
                if sparse_rows:
                    sparse_q = q[..., dense_rows:, :]
                    sparse_out = _execute_packed_or_dense(
                        sparse_q,
                        projected.k,
                        projected.v,
                        selected,
                        prepared.transformer_options,
                        spec=self.spec,
                    )
                    _project_hnd_rows(
                        module,
                        result,
                        sparse_out,
                        start + dense_rows,
                    )
                    del sparse_q, sparse_out

                if end == sequence:
                    projected.release_weight()
                    if prepared.route_plan is not None:
                        prepared.route_plan.release()
                        prepared.route_plan = None
                    projected.k = None
                    projected.v = None
                del q, selected
            return result
        finally:
            prepared.release()

    def as_status(self):
        return {
            'mode': self.name,
            'video_budget': float(self.config.video_budget),
            'density_mode': self.config.density_mode,
            'sparse_q_tile': int(self.spec.q_tile),
            'sparse_kv_tile': int(self.spec.kv_tile),
            'logical_geometry': '%dQ x %dKV'
            % (self.spec.q_tile, self.spec.kv_tile),
            'dense_consumer': 'existing ComfyUI attention',
            'packed_batch_max': int(self.spec.max_batch_entries),
            'probe_dtype': str(self.spec.dtype),
            'approximate': True,
        }


__all__ = [
    'ExistingDenseSparseBackend',
    'ExistingDenseConsumerError',
    'ExistingDenseSparseError',
    'ExistingDenseSparseQKVProjector',
    'ExistingDenseSparseSpec',
    'PreparedExistingDenseSparse',
    'PreparedStreamedExistingDenseSparse',
    'probe_existing_dense_sparse',
]
