"""H3-owned BF16 ConvRot-256 activation quantization."""

from __future__ import annotations

import torch

from . import loader

_SYMBOL = 'h3_int8_quantize_bf16_rowwise_convrot256'
_GROUP_SIZE = 256


def int8_rowwise_convrot256_is_available():
    """Whether the loaded ABI-4 library includes the additive quantizer."""
    try:
        library = loader.load()
    except loader.NativeUnavailableError:
        return False
    return getattr(library, _SYMBOL, None) is not None


def quantize_int8_rowwise_convrot256(input):
    """Rotate contiguous BF16 rows in 256-wide groups and quantize to INT8."""
    if not isinstance(input, torch.Tensor):
        raise TypeError('input must be a torch.Tensor')
    if input.ndim != 2:
        raise ValueError('input must be two-dimensional, got %s' % (tuple(input.shape),))
    if input.dtype != torch.bfloat16:
        raise TypeError('input must have dtype torch.bfloat16, got %r' % input.dtype)
    if not input.is_cuda:
        raise ValueError('input must be a CUDA tensor')
    if not input.is_contiguous():
        raise ValueError('input must be contiguous')
    rows, columns = input.shape
    if columns <= 0 or columns % _GROUP_SIZE:
        raise ValueError('input width must be a positive multiple of 256, got %d' % columns)

    library = loader.load()
    function = getattr(library, _SYMBOL, None)
    if function is None:
        raise loader.NativeUnavailableError(
            'the loaded ABI-4 native library has no ConvRot-256 quantizer; rebuild native/'
        )

    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=input.device)
    loader.check(
        function(
            input.data_ptr(), output.data_ptr(), scales.data_ptr(),
            rows, columns, torch.cuda.current_stream(input.device).cuda_stream,
        ),
        'quantize_bf16_rowwise_convrot256',
    )
    return output, scales
