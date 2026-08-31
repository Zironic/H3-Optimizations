'''Experimental AMD policy layered after the ordinary H3 optimization policy.'''

from __future__ import annotations

from dataclasses import replace
import logging

import torch

from . import apply as _base
from . import apply_policy as _policy  # noqa: F401 - installs ordinary policy first
from .attention.sparse import existing_dense_sparse as _dense_sparse
from .attention.sparse.existing_dense_sparse import (
    ExistingDenseSparseBackend,
    ExistingDenseSparseError,
    ExistingDenseSparseQKVProjector,
    probe_existing_dense_sparse,
)
from .environment import BACKEND_ROCM
from .plan import (
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_OFF,
    FUSED_QKV_REQUIRED,
    QKV_STREAMING_FORCED,
    QKV_STREAMING_OFF,
    SPARSE_BACKEND_AUTO,
)
from .qkv import policy as _qkv_policy
from .qkv.providers import (
    QKV_BF16_CHUNKED,
    QKV_FORCE_BF16_CHUNKED,
    QKV_FORCE_CONVROT_INT8_CHUNKED,
    QKV_STANDARD,
    QKVProviderResolution,
)


LOG_PREFIX = '[H3 Optimizations]'
RDNA2_ARCHES = frozenset(
    ('gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036')
)
SELECTED_EXISTING_DENSE_SPARSE = 'rocm_existing_dense_sparse'

_POLICY_RESOLVE_ATTENTION = _base._resolve_attention
_ORIGINAL_TRITON_PREFLIGHT = _base.preflight_triton_sparse
_ORIGINAL_PACKED_EXECUTE = _dense_sparse._execute_packed_sparse
_packed_runtime_fallback_warned = False


def _active_rocm_architecture(environment=None):
    if environment is not None and getattr(environment, 'backend', None) != BACKEND_ROCM:
        return None
    if not getattr(torch.version, 'hip', None) and environment is None:
        return None
    try:
        index = (
            torch.cuda.current_device()
            if environment is None or getattr(environment, 'device_index', None) is None
            else int(environment.device_index)
        )
        name = torch.cuda.get_device_properties(index).gcnArchName
        return str(name).split(':')[0]
    except Exception:
        return None


def _is_rdna2(environment):
    return (
        getattr(environment, 'backend', None) == BACKEND_ROCM
        and _active_rocm_architecture(environment) in RDNA2_ARCHES
    )


def preflight_triton_sparse(*args, **kwargs):
    '''Do not compile the already-known-broken BF16 Triton path on gfx103x.'''
    architecture = _active_rocm_architecture()
    if architecture in RDNA2_ARCHES:
        raise _base.TritonSparseError(
            'RDNA2 BF16 Triton is disabled after gfx103x rejected BF16 dot '
            'lowering; Auto will try the existing-dense sparse adapter'
        )
    return _ORIGINAL_TRITON_PREFLIGHT(*args, **kwargs)


# The ordinary resolver looks this global up when it reaches Triton.  Replacing
# it here skips the compile-failing RDNA2 probe while leaving every other device
# on the original implementation.
_base.preflight_triton_sparse = preflight_triton_sparse


def _safe_execute_packed_sparse(
    q,
    k,
    v,
    selected_tiles,
    transformer_options,
    *,
    spec,
):
    '''Fail an unanticipated packed-shape rejection open to full dense attention.'''
    global _packed_runtime_fallback_warned
    try:
        return _ORIGINAL_PACKED_EXECUTE(
            q,
            k,
            v,
            selected_tiles,
            transformer_options,
            spec=spec,
        )
    except Exception as error:
        if not _packed_runtime_fallback_warned:
            _packed_runtime_fallback_warned = True
            logging.warning(
                '%s RDNA2 packed sparse call failed at runtime; using full '
                'existing dense attention for the affected Q slab: %s: %s',
                LOG_PREFIX,
                type(error).__name__,
                error,
            )
        return _dense_sparse._call_existing_dense(
            q,
            k,
            v,
            transformer_options,
            heads=int(q.shape[1]),
        )


_safe_execute_packed_sparse._h3_rdna2_dense_fallback = True
_dense_sparse._execute_packed_sparse = _safe_execute_packed_sparse


def _common_streamable(inventory):
    return bool(
        getattr(inventory, 'qkv_plain_float', False)
        or getattr(inventory, 'qkv_convrot_int8_256', False)
        or getattr(inventory, 'qkv_w4a8', False)
        or getattr(inventory, 'qkv_fp8', False)
    )


def _resolve_qkv(plan, inventory):
    request = _base._qkv_request(plan)
    memory = getattr(plan, 'memory', None)
    if request == FUSED_QKV_OFF or (
        memory is not None and memory.qkv_streaming == QKV_STREAMING_OFF
    ):
        return (
            QKVProviderResolution(
                QKV_STANDARD,
                False,
                'QKV streaming is disabled for the RDNA2 sparse adapter',
            ),
            None,
        )

    native_format = _qkv_policy._native_stream_format(inventory)
    required = request == FUSED_QKV_REQUIRED or (
        memory is not None and memory.qkv_streaming == QKV_STREAMING_FORCED
    )

    if request == FUSED_QKV_FORCE_BF16:
        can_stream = _common_streamable(inventory)
        projection_mode = _base.PROJECTION_FORCE_BF16
        provider = QKV_FORCE_BF16_CHUNKED
        projection_label = 'forced BF16'
    elif request == FUSED_QKV_FORCE_QUANT and getattr(
        inventory, 'qkv_plain_float', False
    ):
        can_stream = True
        projection_mode = _base.PROJECTION_FORCE_INT8
        provider = QKV_FORCE_CONVROT_INT8_CHUNKED
        projection_label = 'runtime ConvRot-256 INT8'
    else:
        can_stream = native_format is not None
        projection_mode = _base.PROJECTION_NATIVE
        provider = QKV_BF16_CHUNKED
        projection_label = (
            'checkpoint-native %s' % native_format
            if native_format is not None
            else 'checkpoint-native'
        )

    if not can_stream:
        if required:
            raise ExistingDenseSparseError(
                'required QKV streaming has no held binding for the checkpoint '
                'QKV format on the RDNA2 sparse adapter'
            )
        return (
            QKVProviderResolution(
                QKV_STANDARD,
                False,
                'RDNA2 sparse adapter leaves unsupported QKV storage on the '
                'standard H3 projection path',
            ),
            None,
        )

    chunk_rows = (
        4096
        if memory is None
        else _base._effective_qkv_chunk_rows(memory.chunk_rows)
    )
    projector = ExistingDenseSparseQKVProjector(
        required=required,
        chunk_rows=chunk_rows,
        projection_mode=projection_mode,
    )
    return (
        QKVProviderResolution(
            provider,
            True,
            '%s QKV retains global BF16 K/V and streams bounded BF16 Q into '
            'the RDNA2 existing-dense sparse adapter' % projection_label,
        ),
        projector,
    )


def _fallback_device(environment):
    index = getattr(environment, 'device_index', None)
    return torch.device('cuda' if index is None else 'cuda:%d' % int(index))


def resolve_attention(plan, model, inventory, environment):
    '''Substitute sparse-over-dense only for RDNA2 Auto that otherwise went dense.'''
    attention, qkv = _POLICY_RESOLVE_ATTENTION(
        plan,
        model,
        inventory,
        environment,
    )
    sparse = getattr(plan, 'sparse', None)
    if (
        sparse is None
        or sparse.backend != SPARSE_BACKEND_AUTO
        or not _is_rdna2(environment)
        or attention.selected != _base.ATTENTION_EXISTING
    ):
        return attention, qkv

    transformer_options = (
        getattr(model, 'model_options', {}).get('transformer_options', {}) or {}
    )
    try:
        spec = probe_existing_dense_sparse(
            transformer_options,
            device=_fallback_device(environment),
        )
        resolved_qkv, projector = _resolve_qkv(plan, inventory)
        config = _base.HybridSparseConfig(
            mode=_base.MODE_SAGE128,
            **_base._sparse_config_kwargs(plan),
        )
        backend = ExistingDenseSparseBackend(config, spec=spec)
        reason = (
            '%s; RDNA2 fallback uses the existing ComfyUI dense attention '
            'consumer over packed %dQ x %dKV routes'
            % (attention.reason, spec.q_tile, spec.kv_tile)
        )
        resolved = _base.ResolvedAttention(
            requested=attention.requested,
            selected=SELECTED_EXISTING_DENSE_SPARSE,
            backend=backend,
            reason=reason,
            # Reuse the generic sparse installer category; this is an internal
            # execution kind, not a public BF16 Triton selection.
            backend_kind=_base.ATTENTION_TRITON_SPARSE,
            projector=projector,
            dense_resolution=attention.dense_resolution,
        )
        return resolved, resolved_qkv
    except Exception as error:
        logging.warning(
            '%s RDNA2 existing-dense sparse fallback unavailable: %s: %s',
            LOG_PREFIX,
            type(error).__name__,
            error,
        )
        return (
            replace(
                attention,
                reason=(
                    '%s; RDNA2 existing-dense sparse unavailable: %s: %s'
                    % (attention.reason, type(error).__name__, error)
                ),
            ),
            qkv,
        )


resolve_attention._h3_amd_rdna2_policy = True
_base._resolve_attention = resolve_attention


__all__ = [
    'RDNA2_ARCHES',
    'SELECTED_EXISTING_DENSE_SPARSE',
    'preflight_triton_sparse',
    'resolve_attention',
]
