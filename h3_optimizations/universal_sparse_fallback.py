'''Universal final sparse fallback over ComfyUI's existing dense attention.

This layer runs after the ordinary sparse resolver (and the experimental AMD
policy). It changes nothing while Kitchen, Sparse Sage, BF16 Triton, FROST, or
FlexAttention resolves successfully. If Auto would otherwise fall all the way
back to ordinary existing ComfyUI attention, it gets one last architecture-
neutral sparse attempt: H3 routing packs the selected K/V rows and hands each
small problem back to the same dense attention consumer that already works on
the device.

The adapter is deliberately runtime-probed using the actual Q/K/V dtype. Probe,
routing, packing, or consumer failures fail open to the original dense problem,
so this final fallback cannot turn an otherwise-working dense H3 path into a
hard generation failure.
'''

from __future__ import annotations

import logging
import threading

import torch

from . import apply as _base
from .attention.sparse import existing_dense_sparse as _dense_sparse
from .attention.sparse.existing_dense_sparse import (
    ExistingDenseSparseBackend,
    ExistingDenseSparseError,
    ExistingDenseSparseSpec,
)
from .plan import (
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_REQUIRED,
    QKV_STREAMING_FORCED,
    SPARSE_BACKEND_AUTO,
)
from .qkv.providers import QKV_STANDARD, QKVProviderResolution


LOG_PREFIX = '[H3 Optimizations]'
SELECTED_EXISTING_DENSE_SPARSE = 'existing_dense_sparse'
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_PREVIOUS_RESOLVE_ATTENTION = _base._resolve_attention
_probe_lock = threading.Lock()
_probe_results = {}
_runtime_fallback_lock = threading.Lock()
_runtime_fallback_warned = False


def _validate_qkv(q, k, v):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires equal HND rank-4 Q/K/V'
        )
    if q.shape[0] != 1 or q.shape[-1] != _dense_sparse.HEAD_DIM:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires batch 1 and head_dim 128'
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires FP16, BF16, or FP32 Q/K/V'
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires matching Q/K/V dtypes'
        )
    if q.device != k.device or q.device != v.device or not q.is_cuda:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires one CUDA/ROCm device'
        )
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires contiguous head dimensions'
        )


def _warn_runtime_fallback(scope, error):
    global _runtime_fallback_warned
    with _runtime_fallback_lock:
        if _runtime_fallback_warned:
            return
        _runtime_fallback_warned = True
    logging.warning(
        '%s existing-dense sparse fallback failed during %s; using full '
        'existing dense attention for the affected scope: %s: %s',
        LOG_PREFIX,
        scope,
        type(error).__name__,
        error,
    )


# existing_dense_sparse predates the universal policy and originally carried
# RDNA2-specific BF16 validation/log wording. Its routing/packing implementation
# is architecture-neutral; install the universal runtime contract here after the
# AMD policy has loaded.
_dense_sparse._validate_qkv = _validate_qkv
_dense_sparse._warn_runtime_fallback = _warn_runtime_fallback


def _probe_case(device, transformer_options, *, q_rows, k_rows, batch, dtype):
    generator = torch.Generator(device=device).manual_seed(
        20260831
        + int(q_rows) * 17
        + int(k_rows) * 31
        + int(batch)
        + _SUPPORTED_DTYPES.index(dtype) * 101
    )
    shape_q = (int(batch), 1, int(q_rows), _dense_sparse.HEAD_DIM)
    shape_kv = (int(batch), 1, int(k_rows), _dense_sparse.HEAD_DIM)
    q = torch.randn(shape_q, dtype=dtype, device=device, generator=generator)
    k = torch.randn(shape_kv, dtype=dtype, device=device, generator=generator)
    v = torch.randn(shape_kv, dtype=dtype, device=device, generator=generator)
    actual = _dense_sparse._call_existing_dense(
        q,
        k,
        v,
        transformer_options,
        heads=1,
    )
    if tuple(actual.shape) != tuple(q.shape):
        raise ExistingDenseSparseError(
            'existing attention returned %s for %s probe input %s'
            % (tuple(actual.shape), dtype, tuple(q.shape))
        )
    expected = _dense_sparse._reference_attention(q, k, v)
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(actual).all())
    rel_l2 = _dense_sparse._relative_l2(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    if not (
        finite
        and rel_l2 < _dense_sparse._PROBE_REL_L2
        and max_abs < _dense_sparse._PROBE_MAX_ABS
    ):
        raise ExistingDenseSparseError(
            'existing attention %s probe numerics failed: finite=%s rel_l2=%.6f '
            'max_abs=%.6f' % (dtype, finite, rel_l2, max_abs)
        )


def _probe_geometry(device, transformer_options, q_tile, kv_tile, dtype):
    for k_rows in (kv_tile, kv_tile * 2, kv_tile * 2 - 7):
        _probe_case(
            device,
            transformer_options,
            q_rows=q_tile,
            k_rows=k_rows,
            batch=1,
            dtype=dtype,
        )

    # Batching only affects efficiency. If the dense consumer rejects larger
    # packed batches, progressively reduce them without changing the route.
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
        except Exception:
            continue
    return 1


def probe_existing_dense_sparse(
    transformer_options=None,
    device=None,
    *,
    dtype,
    force=False,
):
    '''Probe the active dense consumer at the dtype real H3 QKV produced.'''
    if dtype not in _SUPPORTED_DTYPES:
        raise ExistingDenseSparseError(
            'existing-dense sparse cannot probe unsupported QKV dtype %s' % dtype
        )
    device = torch.device('cuda' if device is None else device)
    options = _dense_sparse._probe_options(transformer_options)
    key = _dense_sparse._probe_key(device, options) + (str(dtype),)
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
                    dtype,
                )
                spec = ExistingDenseSparseSpec(q_tile, kv_tile, max_batch)
                _probe_results[key] = spec
                logging.info(
                    '%s existing-dense sparse probe selected %dQ x %dKV for %s '
                    '(packed batch <= %d)',
                    LOG_PREFIX,
                    q_tile,
                    kv_tile,
                    str(dtype).replace('torch.', ''),
                    max_batch,
                )
                return spec
            except Exception as error:
                failures.append(
                    '%dQ x %dKV: %s: %s'
                    % (q_tile, kv_tile, type(error).__name__, error)
                )

        error = ExistingDenseSparseError(
            'the existing Comfy attention rejected both %s sparse adapter '
            'geometries (%s)' % (dtype, '; '.join(failures))
        )
        _probe_results[key] = error
        raise error


class UniversalExistingDenseSparseBackend(ExistingDenseSparseBackend):
    '''Existing-dense sparse with runtime dtype and geometry negotiation.'''

    name = SELECTED_EXISTING_DENSE_SPARSE

    def __init__(self, config=None):
        # The installer needs a stable backend before real QKV exists. The first
        # invocation replaces this placeholder with a geometry proven for the
        # actual dense consumer and dtype.
        super().__init__(config, spec=ExistingDenseSparseSpec(64, 64, 1))
        self._runtime_spec_lock = threading.Lock()
        self._runtime_dtype = None

    def _activate_spec(self, dtype, spec):
        with self._runtime_spec_lock:
            if self._runtime_dtype == dtype and self.spec.signature == spec.signature:
                return
            self.spec = spec
            self.router = _dense_sparse.SparseTileRouter(
                self.config,
                q_tile=spec.q_tile,
                kv_tile=spec.kv_tile,
            )
            self._runtime_dtype = dtype

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        options = transformer_options or {}
        try:
            _validate_qkv(q, k, v)
            spec = probe_existing_dense_sparse(
                options,
                device=q.device,
                dtype=q.dtype,
            )
            self._activate_spec(q.dtype, spec)
        except Exception as error:
            return _dense_sparse.PreparedExistingDenseSparse(
                q=q,
                k=k,
                v=v,
                lut=None,
                valid=None,
                metadata={
                    'layer': int(layer_index),
                    'sparse_backend': self.name,
                    'fallback': 'existing_dense',
                },
                transformer_options=options,
                fallback_reason=_dense_sparse._fallback_reason(
                    'runtime dtype/geometry probe',
                    error,
                ),
            )
        return super().prepare(
            q,
            k,
            v,
            layer_index=layer_index,
            transformer_options=options,
        )

    def as_status(self):
        status = super().as_status()
        status['runtime_qkv_dtype'] = (
            None if self._runtime_dtype is None else str(self._runtime_dtype)
        )
        status['probe_mode'] = 'runtime_actual_qkv_dtype'
        status['logical_geometry'] = 'runtime-probed 64Qx64KV or 128Qx128KV'
        return status


def _stock_qkv_allowed(plan):
    request = _base._qkv_request(plan)
    memory = getattr(plan, 'memory', None)
    if request in (
        FUSED_QKV_REQUIRED,
        FUSED_QKV_FORCE_BF16,
        FUSED_QKV_FORCE_QUANT,
    ):
        return False
    return not (
        memory is not None
        and memory.qkv_streaming == QKV_STREAMING_FORCED
    )


def _resolve_final_fallback(plan, attention, qkv):
    # Unknown explicit attention overrides are classified as existing_full_q;
    # keep their single-call contract rather than probing/packing them.
    if attention.backend_kind != _base.ATTENTION_EXISTING:
        return attention, qkv
    if not _stock_qkv_allowed(plan):
        return attention, qkv

    config = _base.HybridSparseConfig(
        mode=_base.MODE_SAGE128,
        **_base._sparse_config_kwargs(plan),
    )
    backend = UniversalExistingDenseSparseBackend(config)
    resolved_qkv = QKVProviderResolution(
        QKV_STANDARD,
        False,
        'final sparse fallback preserves stock Comfy QKV projection so the '
        'adapter receives the execution dtype and representation already proven '
        'to work on this device',
    )
    reason = (
        '%s; final Auto fallback will runtime-probe the existing ComfyUI dense '
        'attention consumer for packed sparse execution before using it fully dense'
        % attention.reason
    )
    return (
        _base.ResolvedAttention(
            requested=_base.ATTENTION_SPARSE,
            selected=SELECTED_EXISTING_DENSE_SPARSE,
            backend=backend,
            reason=reason,
            # Reuse the generic sparse installation category. The backend itself
            # launches no Triton; this only opts into the sparse runtime wrapper.
            backend_kind=_base.ATTENTION_TRITON_SPARSE,
            projector=None,
            dense_resolution=attention.dense_resolution,
        ),
        resolved_qkv,
    )


def resolve_attention(plan, model, inventory, environment):
    attention, qkv = _PREVIOUS_RESOLVE_ATTENTION(
        plan,
        model,
        inventory,
        environment,
    )
    sparse = getattr(plan, 'sparse', None)
    if sparse is None or sparse.backend != SPARSE_BACKEND_AUTO:
        return attention, qkv

    # A previous layer may already have produced an actual sparse backend. In
    # particular, RDNA2's tested fallback currently resolves before this layer.
    if attention.backend_kind in _base.SPARSE_EXECUTION_BACKENDS:
        return attention, qkv

    try:
        return _resolve_final_fallback(plan, attention, qkv)
    except Exception as error:
        logging.warning(
            '%s universal existing-dense sparse fallback could not be installed; '
            'keeping the resolved dense path: %s: %s',
            LOG_PREFIX,
            type(error).__name__,
            error,
        )
        return attention, qkv


resolve_attention._h3_universal_sparse_fallback = True
_base._resolve_attention = resolve_attention


__all__ = [
    'SELECTED_EXISTING_DENSE_SPARSE',
    'UniversalExistingDenseSparseBackend',
    'probe_existing_dense_sparse',
    'resolve_attention',
]
