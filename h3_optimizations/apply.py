'''Resolve and apply the complete H3 optimization plan.'''

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

import comfy.model_management
import comfy.quant_ops

from .attention.sparse import (
    FP8FlexBackend,
    FP8FlexError,
    HybridSparseBackend,
    HybridSparseConfig,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
    SparseSageError,
    TritonSparseBackend,
    TritonSparseError,
    preflight_fp8_flex,
    preflight_sparse_sage,
    preflight_triton_sparse,
)
from .attention.sparse.fused_qkv import (
    TRITON_AVAILABLE as SPARSE_TRITON_AVAILABLE,
)
from .attention.sparse.kitchen_sparse import OUTPUT_NHD, SparseKitchenError
from .dense_resolver import (
    install_dense_attention,
    preserve_dense_attention,
    resolve_dense_attention,
)
from .environment import RuntimeEnvironment
from .kitchen_qkv import (
    PRODUCER_ABI as KITCHEN_PRODUCER_ABI,
    ChunkedKitchenAttentionBackend,
    ChunkedKitchenQKVProjector,
    producer_api_available,
)
from .memory.config import ActivationMemoryConfig
from .memory.patch import install as install_memory_patch
from .mlp_sharing.execution import install_sharing
from .model import get_h3_blocks, is_minimax_h3
from .patch import configure_backend
from .plan import (
    ATTENTION_EXISTING,
    DENSITY_FIXED,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    H3OptimizationPlan,
    PLAN_KEY,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
    STATUS_KEY,
)
from .qkv.formats import inspect_h3_linears
from .qkv.providers import (
    MLP_OFF,
    MLP_PRESERVE_UPSTREAM,
    QKV_DENSE_FP8_CHUNKED,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_SPARSE_CONVROT_INT8,
    QKV_SPARSE_FP8_CHUNKED,
    QKV_STANDARD,
    QKV_TRITON_SPARSE_CHUNKED,
    resolve_mlp_provider,
    resolve_qkv_provider,
)
from .runtime.context import (
    H3RuntimeSession,
    RUNTIME_SESSION_KEY,
    install_runtime_wrapper,
)

LOG_PREFIX = '[H3 Optimizations]'
ATTENTION_SPARSE = 'sparse_sage'
ATTENTION_TRITON_SPARSE = 'triton_sparse_int8'
ATTENTION_FP8_FLEX = 'flex_attention_fp8'
ATTENTION_KITCHEN_SPARSE = 'sparse_kitchen_int8'


@dataclass(frozen=True)
class ResolvedAttention:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str
    projector: object | None = None
    dense_resolution: object | None = None


# Backends that are materially slower than the one that was asked for. A
# fallback here is not a detail: it roughly halves attention throughput, and it
# used to be visible only to someone who went looking at the status output.
_SLOW_SPARSE_FALLBACKS = {
    ATTENTION_TRITON_SPARSE: 'INT8 Triton sparse is roughly half the speed of the native sparse kernel',
    ATTENTION_FP8_FLEX: 'FP8 FlexAttention is far slower than the native sparse kernel',
}


def _warn_about_slow_paths(attention, qkv):
    """Say loudly when a fast path was wanted and something slower was used.

    Only fires when the resolution actually degraded. Choosing a backend
    explicitly is not a fallback and stays quiet.
    """
    requested_sparse = attention.requested in (
        ATTENTION_SPARSE,
        ATTENTION_KITCHEN_SPARSE,
    )
    if requested_sparse and attention.selected != attention.requested:
        cost = _SLOW_SPARSE_FALLBACKS.get(
            attention.selected,
            'this path is substantially slower than the native sparse kernel',
        )
        logging.warning(
            '%s SPARSE ATTENTION FELL BACK to %s. %s. Reason: %s',
            LOG_PREFIX,
            attention.selected,
            cost,
            attention.reason or 'unknown',
        )

    # The chunked Comfy Kitchen QKV producer is the fast QKV path. When its
    # Kitchen-side API is missing the pack silently projects the slow way,
    # which is what made a missing dependency look like a working install.
    if qkv.provider_id == QKV_STANDARD and 'Kitchen' in (qkv.reason or ''):
        logging.warning(
            '%s FUSED QKV IS NOT RUNNING - falling back to standard projection, '
            'which is roughly half the speed. Reason: %s',
            LOG_PREFIX,
            qkv.reason,
        )


def _qkv_request(plan):
    if plan.memory is not None:
        return plan.memory.fused_qkv
    if plan.sparse is not None:
        return FUSED_QKV_AUTO
    return FUSED_QKV_OFF


def _fp8_execution_available(environment):
    if not bool(getattr(environment, 'cuda_available', False)):
        return False
    capability = getattr(environment, 'capability', None)
    if capability is None or tuple(capability) < (8, 9):
        return False
    if not bool(getattr(comfy.quant_ops, '_CK_AVAILABLE', False)):
        return False
    index = getattr(environment, 'device_index', None)
    device = torch.device('cuda', index) if index is not None else torch.device('cuda')
    try:
        return bool(comfy.model_management.supports_fp8_compute(device))
    except (AttributeError, RuntimeError, TypeError):
        return False


def _sparse_config_kwargs(plan):
    sparse = plan.sparse
    return {
        'video_budget': float(sparse.video_budget),
        'density_mode': DENSITY_FIXED,
        'denser_early_late_steps': bool(sparse.denser_early_late_steps),
        'early_steps': sparse.early_steps,
        'early_kv': sparse.early_kv,
        'late_steps': sparse.late_steps,
        'late_kv': sparse.late_kv,
        'layer_video_budgets': sparse.layer_video_budgets,
        'strict': True,
    }


def _resolve_dense(plan, model, inventory, environment=None):
    memory = plan.memory
    dense = (
        preserve_dense_attention('no memory optimization requested')
        if memory is None
        else (
            preserve_dense_attention('existing dense attention was requested')
            if memory.attention == ATTENTION_EXISTING
            else resolve_dense_attention(model)
        )
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=dense.backend_kind,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None)
        ),
        memory_optimize=memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    backend = None
    projector = None
    if qkv.provider_id in (
        QKV_DENSE_KITCHEN_CHUNKED,
        QKV_DENSE_FP8_CHUNKED,
    ):
        backend = ChunkedKitchenAttentionBackend()
        # The chunk quantizer is handed the same strided Q/K views here as on
        # the sparse path, and takes them through the same guarded predicate.
        # The sequence-major output layout is deliberately not enabled on this
        # path: it was measured on the sparse route only.
        projector = ChunkedKitchenQKVProjector(
            fp8_projection=qkv.provider_id == QKV_DENSE_FP8_CHUNKED,
            strided_qk_input=True,
        )
    return (
        ResolvedAttention(
            requested=dense.requested,
            selected=dense.selected,
            backend=backend,
            reason=dense.reason,
            backend_kind=dense.backend_kind,
            projector=projector,
            dense_resolution=dense,
        ),
        qkv,
    )


def _resolve_sparse(plan, environment, inventory):
    kernel_spec = preflight_sparse_sage(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=ATTENTION_SPARSE,
        triton_available=bool(SPARSE_TRITON_AVAILABLE),
        sparse_spec=kernel_spec,
        memory_optimize=plan.memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    use_projected = qkv.provider_id in (
        QKV_SPARSE_CONVROT_INT8,
        QKV_SPARSE_FP8_CHUNKED,
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128_FUSED_QKV if use_projected else MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    projector = None
    if qkv.provider_id == QKV_SPARSE_CONVROT_INT8:
        from .qkv.projectors import SparseFusedQKVProjector

        projector = SparseFusedQKVProjector(kernel_spec, chunk_rows=4096)
    elif qkv.provider_id == QKV_SPARSE_FP8_CHUNKED:
        from .attention.sparse.fp8_qkv import FP8SparseQKVProjector

        projector = FP8SparseQKVProjector(kernel_spec, chunk_rows=4096)
    backend = HybridSparseBackend(
        config,
        kernel_spec=kernel_spec,
        projector=projector,
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_SPARSE,
            selected=ATTENTION_SPARSE,
            backend=backend,
            reason='explicit fixed-density Sparse Sage attention',
            backend_kind=ATTENTION_SPARSE,
            projector=projector,
        ),
        qkv,
    )


def _resolve_fp8_flex(
    plan,
    environment,
    inventory,
    fallback_reason,
    dense_attention,
):
    spec = preflight_fp8_flex(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
        device=getattr(environment, 'device_index', None),
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=ATTENTION_FP8_FLEX,
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    backend = FP8FlexBackend(config, spec=spec)
    if fallback_reason is None:
        reason = 'explicit FP8 FlexAttention selection'
        dense_resolution = None
    else:
        reason = '%s; using FP8 FlexAttention' % fallback_reason
        dense_resolution = dense_attention.dense_resolution
    return (
        ResolvedAttention(
            requested=ATTENTION_SPARSE,
            selected=ATTENTION_FP8_FLEX,
            backend=backend,
            reason=reason,
            backend_kind=ATTENTION_FP8_FLEX,
            dense_resolution=dense_resolution,
        ),
        qkv,
    )


def _resolve_triton_sparse(plan, environment, inventory, fallback_reason):
    spec = preflight_triton_sparse(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=ATTENTION_TRITON_SPARSE,
        triton_available=True,
    )
    config = HybridSparseConfig(
        mode=MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    projector = None
    if qkv.provider_id == QKV_TRITON_SPARSE_CHUNKED:
        from .qkv.projectors import TritonSparseQKVProjector

        projector = TritonSparseQKVProjector(chunk_rows=4096)
    backend = TritonSparseBackend(
        config,
        spec=spec,
        projector=projector,
    )
    reason = (
        'explicit INT8 Triton sparse attention selection'
        if fallback_reason is None
        else '%s; using INT8 Triton sparse attention' % fallback_reason
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_SPARSE,
            selected=ATTENTION_TRITON_SPARSE,
            backend=backend,
            reason=reason,
            backend_kind=ATTENTION_TRITON_SPARSE,
            projector=projector,
        ),
        qkv,
    )


def _resolve_kitchen_sparse(plan, environment, inventory):
    """Explicit Kitchen block-sparse INT8, with no Sparge anywhere in it."""
    from .attention.sparse.kitchen_sparse import (
        SparseKitchenBackend,
        preflight_sparse_kitchen,
    )

    kitchen = preflight_sparse_kitchen(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_qkv_request(plan),
        backend_kind=ATTENTION_KITCHEN_SPARSE,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None),
        ),
    )
    use_projected = qkv.provider_id == QKV_DENSE_KITCHEN_CHUNKED
    config = HybridSparseConfig(
        mode=MODE_SAGE128_FUSED_QKV if use_projected else MODE_SAGE128,
        **_sparse_config_kwargs(plan),
    )
    projector = (
        ChunkedKitchenQKVProjector(
            routing_summaries=True, strided_qk_input=True
        )
        if use_projected
        else None
    )
    # Sequence-major output storage and releasing the carrier before the
    # output projection are only worth anything together: measured separately
    # the layout alone moved nothing and the release alone cost 10 ms, while
    # together they take 923 MiB off peak allocated and 852 MiB off peak
    # reserved at no cost in time. Block output stays bit-identical.
    backend = SparseKitchenBackend(
        config,
        kitchen=kitchen,
        projector=projector,
        output_layout=OUTPUT_NHD,
        release_carrier_before_out_proj=True,
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_KITCHEN_SPARSE,
            selected=ATTENTION_KITCHEN_SPARSE,
            backend=backend,
            reason='explicit Kitchen block-sparse INT8 attention',
            backend_kind=ATTENTION_KITCHEN_SPARSE,
            projector=projector,
        ),
        qkv,
    )


def _resolve_attention(plan, model, inventory, environment):
    if plan.sparse is not None:
        backend_request = plan.sparse.backend
        if backend_request == SPARSE_BACKEND_SAGE:
            return _resolve_sparse(plan, environment, inventory)
        if backend_request == SPARSE_BACKEND_TRITON:
            return _resolve_triton_sparse(plan, environment, inventory, None)
        if backend_request == SPARSE_BACKEND_KITCHEN:
            return _resolve_kitchen_sparse(plan, environment, inventory)
        if backend_request == SPARSE_BACKEND_FLEX:
            return _resolve_fp8_flex(
                plan,
                environment,
                inventory,
                None,
                None,
            )
        if backend_request != SPARSE_BACKEND_AUTO:
            raise ValueError('unknown sparse backend request %r' % backend_request)

    dense_attention, dense_qkv = _resolve_dense(
        plan,
        model,
        inventory,
        environment,
    )
    if plan.sparse is None:
        return dense_attention, dense_qkv

    try:
        return _resolve_kitchen_sparse(plan, environment, inventory)
    except SparseKitchenError as kitchen_exc:
        try:
            return _resolve_sparse(plan, environment, inventory)
        except SparseSageError as sparse_exc:
            fallback_reason = (
                'Kitchen INT8 unavailable: %s; Sparse Sage unavailable: %s'
                % (kitchen_exc, sparse_exc)
            )
            try:
                return _resolve_triton_sparse(
                    plan,
                    environment,
                    inventory,
                    fallback_reason,
                )
            except TritonSparseError as triton_exc:
                fallback_reason = (
                    '%s; INT8 Triton unavailable: %s'
                    % (fallback_reason, triton_exc)
                )
                try:
                    return _resolve_fp8_flex(
                        plan,
                        environment,
                        inventory,
                        fallback_reason,
                        dense_attention,
                    )
                except FP8FlexError as flex_exc:
                    return (
                        ResolvedAttention(
                            requested=ATTENTION_SPARSE,
                            selected=dense_attention.selected,
                            backend=dense_attention.backend,
                            reason=(
                                '%s; FP8 FlexAttention unavailable: %s; %s'
                                % (
                                    fallback_reason,
                                    flex_exc,
                                    dense_attention.reason,
                                )
                            ),
                            backend_kind=dense_attention.backend_kind,
                            projector=dense_attention.projector,
                            dense_resolution=dense_attention.dense_resolution,
                        ),
                        dense_qkv,
                    )


def _install_mlp(model_patcher, plan, inventory, environment):
    memory = plan.memory
    if memory is None:
        return resolve_mlp_provider(inventory, request='off'), 0
    resolution = resolve_mlp_provider(
        inventory,
        request=memory.mlp_memory,
        fp8_available=_fp8_execution_available(environment),
    )
    if resolution.provider_id in (MLP_OFF, MLP_PRESERVE_UPSTREAM):
        return resolution, 0
    config = ActivationMemoryConfig(
        mode=resolution.activation_mode,
        chunk_rows=int(memory.chunk_rows),
        strict=bool(memory.mlp_strict),
        prefer_held_weights=bool(memory.prefer_held_weights),
    )
    return resolution, int(install_memory_patch(model_patcher, config))


def _ensure_sparse_runtime(model_patcher):
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    session = options.get(RUNTIME_SESSION_KEY)
    if session is not None:
        if not isinstance(session, H3RuntimeSession):
            raise TypeError(
                '%s is not an H3 Optimizations runtime session'
                % RUNTIME_SESSION_KEY
            )
        session.strict_layout = True
        return session, False
    session = H3RuntimeSession(strict_layout=True)
    install_runtime_wrapper(model_patcher, session)
    return session, True


def _inventory_status(inventory):
    return {
        'qkv': list(inventory.labels('qkv')),
        'fc1': list(inventory.labels('fc1')),
        'fc2': list(inventory.labels('fc2')),
    }


def _status(
    plan,
    environment,
    attention,
    qkv,
    mlp,
    *,
    attention_blocks,
    mlp_blocks,
    mlp_sharing_installed,
    runtime_installed,
    inventory,
):
    return {
        'plan_version': int(plan.version),
        'plan_signature': plan.signature,
        'attention': {
            'requested': attention.requested,
            'selected': attention.selected,
            'reason': attention.reason,
            'patched_blocks': int(attention_blocks),
        },
        'sparse': (
            None
            if plan.sparse is None
            else {
                'backend': plan.sparse.backend,
                'video_budget': float(plan.sparse.video_budget),
                'denser_early_late_steps': bool(
                    plan.sparse.denser_early_late_steps
                ),
                'early_steps': plan.sparse.early_steps,
                'early_kv': plan.sparse.early_kv,
                'late_steps': plan.sparse.late_steps,
                'late_kv': plan.sparse.late_kv,
                'layer_video_budgets': (
                    None
                    if plan.sparse.layer_video_budgets is None
                    else list(plan.sparse.layer_video_budgets)
                ),
            }
        ),
        'fused_qkv': {
            'requested': _qkv_request(plan),
            'provider': qkv.provider_id,
            'fused': bool(qkv.fused),
            'reason': qkv.reason,
            'projector': getattr(attention.projector, 'name', None),
            'chunk_rows': getattr(attention.projector, 'chunk_rows', None),
            'producer_abi': (
                KITCHEN_PRODUCER_ABI
                if qkv.provider_id in (
                    QKV_DENSE_KITCHEN_CHUNKED,
                    QKV_DENSE_FP8_CHUNKED,
                )
                else None
            ),
        },
        'mlp': {
            'requested': 'off' if plan.memory is None else plan.memory.mlp_memory,
            'provider': mlp.provider_id,
            'activation_mode': mlp.activation_mode,
            'reason': mlp.reason,
            'chunk_rows': (
                None if plan.memory is None else int(plan.memory.chunk_rows)
            ),
            'patched_blocks': int(mlp_blocks),
        },
        'mlp_sharing': (
            None
            if plan.mlp_sharing is None
            else {
                'installed': bool(mlp_sharing_installed),
                'selector': plan.mlp_sharing.selector,
                'requested_removal_fraction': float(
                    plan.mlp_sharing.removal_fraction
                ),
                'start_after_step': int(plan.mlp_sharing.start_after_step),
                'layers': list(plan.mlp_sharing.layers),
                'selector_seed': int(plan.mlp_sharing.selector_seed),
                'geometry': list(plan.mlp_sharing.geometry),
                'reconstruction': 'representative',
            }
        ),
        'weight_formats': _inventory_status(inventory),
        'runtime_installed': bool(runtime_installed),
        'device': {
            'name': environment.device_name,
            'backend': environment.backend,
            'architecture': environment.architecture,
            'capability': (
                None
                if environment.capability is None
                else [int(value) for value in environment.capability]
            ),
        },
    }


def apply_plan(model, plan: H3OptimizationPlan):
    '''Apply compatible H3 features; other model families are exact no-ops.'''
    if not isinstance(plan, H3OptimizationPlan):
        raise TypeError('plan must be H3OptimizationPlan')
    if not is_minimax_h3(model):
        return model

    blocks = get_h3_blocks(model)
    inventory = inspect_h3_linears(blocks)
    environment = RuntimeEnvironment.detect()
    attention, qkv = _resolve_attention(plan, model, inventory, environment)

    patched = model.clone()
    attention_blocks = 0
    sparse_execution_selected = attention.backend_kind in (
        ATTENTION_SPARSE,
        ATTENTION_TRITON_SPARSE,
        ATTENTION_FP8_FLEX,
        ATTENTION_KITCHEN_SPARSE,
    )
    flex_dense_fallback = (
        attention.backend_kind == ATTENTION_FP8_FLEX
        and plan.sparse is not None
        and plan.sparse.backend == SPARSE_BACKEND_AUTO
    )
    if sparse_execution_selected:
        if attention.backend_kind == ATTENTION_FP8_FLEX:
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
                backend_fallback_to_dense=flex_dense_fallback,
            )
        else:
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
            )
        if flex_dense_fallback and attention.dense_resolution is not None:
            install_dense_attention(patched, attention.dense_resolution)
    elif plan.memory is not None:
        if qkv.provider_id in (
            QKV_DENSE_KITCHEN_CHUNKED,
            QKV_DENSE_FP8_CHUNKED,
        ):
            _backend, attention_blocks = configure_backend(
                patched,
                attention.backend,
                projector=attention.projector,
                projector_fallback_to_original=True,
            )
        install_dense_attention(patched, attention.dense_resolution)

    mlp, mlp_blocks = _install_mlp(
        patched,
        plan,
        inventory,
        environment,
    )
    mlp_sharing_installed = False
    if plan.mlp_sharing is not None and plan.memory is not None:
        if mlp.provider_id in (MLP_OFF, MLP_PRESERVE_UPSTREAM):
            raise RuntimeError(
                'H3 MLP Sharing requires an executable H3 Memory Optimization '
                'MLP provider; resolved %s' % mlp.provider_id
            )
        install_sharing(patched, plan.mlp_sharing)
        mlp_sharing_installed = True
    runtime_installed = False
    if sparse_execution_selected:
        _session, _created = _ensure_sparse_runtime(patched)
        runtime_installed = True
    if mlp_sharing_installed:
        runtime_installed = True

    patched.model_options[PLAN_KEY] = plan
    options = patched.model_options['transformer_options'] = (
        patched.model_options.get('transformer_options', {}).copy()
    )
    options[STATUS_KEY] = _status(
        plan,
        environment,
        attention,
        qkv,
        mlp,
        attention_blocks=attention_blocks,
        mlp_blocks=mlp_blocks,
        mlp_sharing_installed=mlp_sharing_installed,
        runtime_installed=runtime_installed,
        inventory=inventory,
    )
    _warn_about_slow_paths(attention, qkv)
    logging.info(
        '%s armed: attention=%s qkv=%s mlp=%s device=%s',
        LOG_PREFIX,
        attention.selected,
        qkv.provider_id,
        mlp.provider_id,
        environment.device_name,
    )
    return patched
