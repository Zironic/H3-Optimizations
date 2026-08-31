'''Experimental AMD policy layered after the ordinary H3 optimization policy.'''

from __future__ import annotations

from dataclasses import replace

import torch

from . import apply as _base
from . import apply_policy as _policy  # noqa: F401 - installs ordinary policy first
from .environment import BACKEND_ROCM
from .plan import (
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
)


RDNA2_ARCHES = frozenset(
    ('gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036')
)

_POLICY_RESOLVE_ATTENTION = _base._resolve_attention
_ORIGINAL_RESOLVE_TRITON_SPARSE = _base._resolve_triton_sparse


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


def resolve_attention(plan, model, inventory, environment):
    '''Skip known-dead RDNA2 sparse paths and leave final fallback to core policy.'''
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
        # Return the proven dense path. The architecture-neutral universal layer
        # loaded after this policy gets the final opportunity to promote it to
        # sparse-over-existing-dense. Keeping the layers separate makes RDNA2 and
        # CUDA use exactly the same adapter implementation and fail-open contract.
        return (
            replace(
                dense_attention,
                reason=(
                    '%s; FP8 FlexAttention unavailable: %s; %s'
                    % (skipped, flex_error, dense_attention.reason)
                ),
            ),
            dense_qkv,
        )


resolve_attention._h3_amd_rdna2_policy = True
_base._resolve_attention = resolve_attention


__all__ = [
    'RDNA2_ARCHES',
    'resolve_attention',
]
