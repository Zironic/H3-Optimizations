'''Stable public surface for the BF16 Triton sparse backend.

The production backend keeps projected Q/K/V in BF16 and uses ordinary BF16
dots with FP32 online-softmax state at 64Q x 64KV. NVIDIA CUDA is the validated
production target. ROCm gfx103x uses a cached numerical gate before this
backend becomes eligible; newer RDNA and known CDNA targets remain experimental
Triton targets.
'''

import torch

from .triton_bf16 import (
    PreparedTritonBF16,
    TRITON_AVAILABLE,
    TritonBF16Backend,
    TritonBF16Error,
    TritonBF16Spec,
    preflight_triton_bf16,
)


TritonSparseError = TritonBF16Error
TritonSparseSpec = TritonBF16Spec

_ROCM_MATRIX_ARCHES = frozenset(
    ('gfx908', 'gfx90a', 'gfx940', 'gfx941', 'gfx942', 'gfx950')
)
_ROCM_RDNA2_ARCHES = frozenset(
    ('gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036')
)


def _rocm_architecture():
    try:
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_properties(index).gcnArchName
        return str(name).split(':')[0]
    except Exception:
        return None


def _rocm_bf16_dot_supported(architecture):
    architecture = str(architecture or '')
    return (
        architecture in _ROCM_RDNA2_ARCHES
        or architecture.startswith(('gfx11', 'gfx12'))
        or architecture in _ROCM_MATRIX_ARCHES
    )


def TritonSparseBackend(config=None, **kwargs):
    '''Construct the portable BF16 Triton backend.'''
    return TritonBF16Backend(config, **kwargs)


TritonSparseBackend.name = TritonBF16Backend.name


def preflight_triton_sparse(
    *,
    rocm_available=None,
    rocm_arch_getter=None,
    rocm_selftest=None,
    **kwargs,
):
    '''Validate the portable BF16 Triton fallback.

    ROCm exposes GPU tensors through PyTorch's ``cuda`` device type but does not
    have an NVIDIA compute capability. The kernel itself is plain Triton. RDNA2
    may lower BF16 dots through a non-matrix fallback, so gfx103x must pass a
    small per-device numerical check before automatic selection. Unknown future
    architectures fail closed to the existing Flex/dense fallback chain.
    '''
    if rocm_available is None:
        rocm_available = lambda: bool(getattr(torch.version, 'hip', None))
    if rocm_available():
        available = kwargs.pop('triton_available', None)
        available = TRITON_AVAILABLE if available is None else bool(available)
        if not available:
            raise TritonBF16Error('BF16 Triton sparse attention requires Triton')
        architecture = (
            _rocm_architecture()
            if rocm_arch_getter is None
            else rocm_arch_getter()
        )
        if not _rocm_bf16_dot_supported(architecture):
            raise TritonBF16Error(
                'experimental ROCm BF16 Triton requires gfx103x, gfx11, gfx12, '
                'or known CDNA; got %s'
                % (architecture or 'unknown')
            )
        if architecture in _ROCM_RDNA2_ARCHES:
            if rocm_selftest is None:
                from . import triton_bf16_selftest

                rocm_selftest = triton_bf16_selftest.check
            if not rocm_selftest():
                raise TritonBF16Error(
                    'experimental RDNA2 BF16 Triton numerical self-test failed'
                )
        return TritonBF16Spec()
    return preflight_triton_bf16(**kwargs)


__all__ = [
    'PreparedTritonBF16',
    'TritonSparseBackend',
    'TritonSparseError',
    'TritonSparseSpec',
    'preflight_triton_sparse',
]
