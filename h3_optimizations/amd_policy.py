'''Experimental AMD policy layered after the ordinary H3 optimization policy.'''

from __future__ import annotations

from dataclasses import replace
import logging
import threading

import torch

from . import apply as _base
from . import apply_policy as _policy  # noqa: F401 - installs ordinary policy first
from .attention.sparse import existing_dense_sparse as _dense_sparse
from .attention.sparse.existing_dense_sparse import (
    ExistingDenseSparseBackend,
    ExistingDenseSparseError,
    ExistingDenseSparseSpec,
)
from .environment import BACKEND_ROCM
from .plan import (
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_REQUIRED,
    QKV_STREAMING_FORCED,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
)
from .qkv.providers import QKV_STANDARD, QKVProviderResolution


LOG_PREFIX = '[H3 Optimizations]'
RDNA2_ARCHES = frozenset(
    ('gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036')
)
SELECTED_EXISTING_DENSE_SPARSE = 'rocm_existing_dense_sparse'

_POLICY_RESOLVE_ATTENTION = _base._resolve_attention
_ORIGINAL_RESOLVE_TRITON_SPARSE = _base._resolve_triton_sparse
_rdna2_probe_lock = threading.Lock()
_rdna2_probe_results = {}


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


def _resolve_triton_sparse(plan, environment, inventory, fallback_reason):
    '''Reject the known-impossible RDNA2 BF16 Triton path before preflight.'''
    architecture = _active_rocm_architecture(environment)
    if architecture in RDNA2_ARCHES:
        raise _base.TritonSparseError(
            'BF16 Triton sparse attention is unavailable on RDNA2 %s because '
            'gfx103x does not provide the BF16 matrix multiply required by this '
            'backend' % architecture
        )
    return _ORIGINAL_RESOLVE_TRITON_SPARSE(
        plan,
        environment,
        inventory,
        fallback_reason,
    )


# apply.py resolves this global at execution time, including calls made through
# apply_policy's saved base resolver. Explicit BF16 Triton therefore gets the
# same selected-device-aware RDNA2 rejection as automatic resolution.
_base._resolve_triton_sparse = _resolve_triton_sparse


def _validate_rdna2_qkv(q, k, v):
    '''The existing-dense adapter is dtype-neutral for H3's BF16/FP32 modes.'''
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires equal HND rank-4 Q/K/V'
        )
    if q.shape[0] != 1 or q.shape[-1] != _dense_sparse.HEAD_DIM:
        raise ExistingDenseSparseError(
            'existing-dense sparse attention requires batch 1 and head_dim 128'
        )
    if q.dtype not in (torch.bfloat16, torch.float32):
        raise ExistingDenseSparseError(
            'RDNA2 existing-dense sparse attention requires BF16 or FP32 Q/K/V'
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


# The generic adapter was originally BF16-only because it was introduced behind
# a BF16 streamed projector. RDNA2 now deliberately uses stock Comfy QKV, whose
# real execution dtype can be FP32 on older ROCm stacks. The packing/router math
# itself is dtype-neutral for the two H3 inference dtypes.
_dense_sparse._validate_qkv = _validate_rdna2_qkv


def _probe_case_dtype(device, transformer_options, *, q_rows, k_rows, batch, dtype):
    generator = torch.Generator(device=device).manual_seed(
        20260831
        + int(q_rows) * 17
        + int(k_rows) * 31
        + int(batch)
        + (1 if dtype == torch.float32 else 0)
    )
    q = torch.randn(
        (int(batch), 1, int(q_rows), _dense_sparse.HEAD_DIM),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        (int(batch), 1, int(k_rows), _dense_sparse.HEAD_DIM),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        (int(batch), 1, int(k_rows), _dense_sparse.HEAD_DIM),
        dtype=dtype,
        device=device,
        generator=generator,
    )
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


def _probe_geometry_dtype(device, transformer_options, q_tile, kv_tile, dtype):
    for k_rows in (kv_tile, kv_tile * 2, kv_tile * 2 - 7):
        _probe_case_dtype(
            device,
            transformer_options,
            q_rows=q_tile,
            k_rows=k_rows,
            batch=1,
            dtype=dtype,
        )
    for batch in (16, 8, 4, 2):
        try:
            _probe_case_dtype(
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


def probe_rdna2_existing_dense_sparse(
    transformer_options=None,
    device=None,
    *,
    dtype,
    force=False,
):
    '''Probe the same dtype that stock H3 QKV actually produced at runtime.'''
    if dtype not in (torch.bfloat16, torch.float32):
        raise ExistingDenseSparseError(
            'RDNA2 sparse-over-dense cannot probe unsupported QKV dtype %s' % dtype
        )
    device = torch.device('cuda' if device is None else device)
    options = _dense_sparse._probe_options(transformer_options)
    key = _dense_sparse._probe_key(device, options) + (str(dtype),)
    with _rdna2_probe_lock:
        if not force and key in _rdna2_probe_results:
            result = _rdna2_probe_results[key]
            if isinstance(result, Exception):
                raise ExistingDenseSparseError(str(result))
            return result

        failures = []
        for q_tile, kv_tile in ((64, 64), (128, 128)):
            try:
                max_batch = _probe_geometry_dtype(
                    device,
                    options,
                    q_tile,
                    kv_tile,
                    dtype,
                )
                spec = ExistingDenseSparseSpec(q_tile, kv_tile, max_batch)
                _rdna2_probe_results[key] = spec
                logging.info(
                    '%s RDNA2 existing-dense sparse probe selected %dQ x %dKV '
                    'for %s (packed batch <= %d)',
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
        _rdna2_probe_results[key] = error
        raise error


class RDNA2ExistingDenseSparseBackend(ExistingDenseSparseBackend):
    '''Existing-dense sparse with runtime dtype/geometry negotiation.'''

    def __init__(self, config=None):
        # Placeholder geometry only satisfies the generic installation contract.
        # The first real QKV invocation replaces it after probing that dtype.
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
            _validate_rdna2_qkv(q, k, v)
            spec = probe_rdna2_existing_dense_sparse(
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
        return status


def _resolve_qkv(plan):
    '''RDNA2 Auto keeps QKV on the exact stock Comfy execution path.'''
    request = _base._qkv_request(plan)
    memory = getattr(plan, 'memory', None)
    required = request == FUSED_QKV_REQUIRED or (
        memory is not None and memory.qkv_streaming == QKV_STREAMING_FORCED
    )
    if required or request in (FUSED_QKV_FORCE_BF16, FUSED_QKV_FORCE_QUANT):
        raise ExistingDenseSparseError(
            'RDNA2 sparse-over-dense does not override explicitly forced QKV '
            'execution; use Auto/Off QKV streaming or let attention fall back dense'
        )
    return (
        QKVProviderResolution(
            QKV_STANDARD,
            False,
            'RDNA2 sparse-over-dense preserves stock Comfy QKV projection so '
            'quantized checkpoints may dequantize/materialize in the execution '
            'dtype supported by the installed ROCm/PyTorch stack',
        ),
        None,
    )


def _preserves_external_override(plan, model):
    sparse = getattr(plan, 'sparse', None)
    if sparse is None:
        return False
    options = getattr(model, 'model_options', {}).get('transformer_options', {}) or {}
    override = options.get('optimized_attention_override')
    return bool(
        override is not None
        and not _base.is_installed_dense_attention(options)
        and not _base.is_comfy_kitchen_dense_attention(options)
    )


def _resolve_existing_dense_adapter(
    plan,
    model,
    inventory,
    environment,
    attention,
    qkv,
    fallback_reason,
):
    del model, inventory, environment
    if attention.selected != _base.ATTENTION_EXISTING:
        return (
            replace(
                attention,
                reason='%s; %s' % (fallback_reason, attention.reason),
            ),
            qkv,
        )

    try:
        resolved_qkv, projector = _resolve_qkv(plan)
        config = _base.HybridSparseConfig(
            mode=_base.MODE_SAGE128,
            **_base._sparse_config_kwargs(plan),
        )
        backend = RDNA2ExistingDenseSparseBackend(config)
        reason = (
            '%s; RDNA2 fallback preserves stock Comfy QKV and probes the '
            'existing dense attention consumer at runtime using the actual QKV '
            'dtype, selecting 64Q x 64KV or 128Q x 128KV; adapter failures fail '
            'open to the same full dense consumer' % fallback_reason
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
                    '%s; RDNA2 existing-dense sparse unavailable: %s: %s; %s'
                    % (
                        fallback_reason,
                        type(error).__name__,
                        error,
                        attention.reason,
                    )
                ),
            ),
            qkv,
        )


def resolve_attention(plan, model, inventory, environment):
    '''Use only potentially viable sparse paths for RDNA2 selection.'''
    sparse = getattr(plan, 'sparse', None)
    rdna2 = _is_rdna2(environment)
    if sparse is not None and rdna2:
        architecture = _active_rocm_architecture(environment) or 'gfx103x'
        if sparse.backend == SPARSE_BACKEND_KITCHEN:
            raise _base.SparseKitchenError(
                'Kitchen INT8 sparse attention is unavailable on RDNA2 %s; '
                'the experimental AMD native library targets gfx11/gfx12'
                % architecture
            )
        if sparse.backend == SPARSE_BACKEND_SAGE:
            raise _base.SparseSageError(
                'Sparse Sage is unavailable on RDNA2 %s because its packaged '
                'kernels are NVIDIA CUDA extensions' % architecture
            )
        if sparse.backend == SPARSE_BACKEND_TRITON:
            raise _base.TritonSparseError(
                'BF16 Triton sparse attention is unavailable on RDNA2 %s because '
                'gfx103x does not provide the BF16 matrix multiply required by '
                'this backend' % architecture
            )

    if (
        sparse is None
        or sparse.backend != SPARSE_BACKEND_AUTO
        or not rdna2
        or _preserves_external_override(plan, model)
    ):
        return _POLICY_RESOLVE_ATTENTION(
            plan,
            model,
            inventory,
            environment,
        )

    # gfx103x has no compatible native Kitchen kernel, Sparse Sage is an NVIDIA
    # extension, and this BF16 Triton path requires BF16 matrix multiply that
    # RDNA2 does not provide. Do not spend startup time probing known-dead paths.
    dense_attention, dense_qkv = _base._resolve_dense(
        plan,
        model,
        inventory,
        environment,
    )
    skipped = (
        'RDNA2 gfx103x skips Kitchen INT8, Sparse Sage, and BF16 Triton because '
        'those sparse backends require hardware/runtime support unavailable on '
        'RDNA2'
    )

    try:
        return _base._resolve_fp8_flex(
            plan,
            environment,
            inventory,
            skipped,
            dense_attention,
        )
    except _base.FP8FlexError as flex_error:
        fallback_reason = '%s; FP8 FlexAttention unavailable: %s' % (
            skipped,
            flex_error,
        )

    return _resolve_existing_dense_adapter(
        plan,
        model,
        inventory,
        environment,
        dense_attention,
        dense_qkv,
        fallback_reason,
    )


resolve_attention._h3_amd_rdna2_policy = True
_base._resolve_attention = resolve_attention


__all__ = [
    'RDNA2_ARCHES',
    'RDNA2ExistingDenseSparseBackend',
    'SELECTED_EXISTING_DENSE_SPARSE',
    'probe_rdna2_existing_dense_sparse',
    'resolve_attention',
]
