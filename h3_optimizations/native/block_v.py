"""Single-pass, per-block V quantization for dense H3 attention."""

import ctypes
from functools import lru_cache
from pathlib import Path
import platform

import torch

from . import v_staging
from .int8_attention import _DTYPE_TO_CODE, _pad_to, _ptr, _stream, packed_dense_attention_is_available


@lru_cache(maxsize=1)
def _library():
    name = 'h3_block_v_v3.dll' if platform.system() == 'Windows' else 'libh3_block_v_v3.so'
    path = Path(__file__).resolve().parents[2] / 'native' / 'bin' / name
    if not path.is_file():
        raise RuntimeError('Single-pass V requires the compiled H3 block-V attention library')
    library = ctypes.CDLL(str(path))
    library.h3_block_v_last_error.restype = ctypes.c_char_p
    library.h3_block_v_attention_v3.restype = ctypes.c_int
    library.h3_block_v_attention_v3.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int] * 21
        + [ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    )
    return library


def available():
    try:
        _library()
        return v_staging.native_v_staging_available()
    except (OSError, RuntimeError):
        return False


class SinglePassVCarrier:
    def __init__(self, spec, *, block_rows=None, backend=v_staging.BACKEND_NATIVE):
        self.spec = spec
        self.batch, self.heads, self.sequence, self.dim = spec.k_input_shape
        self.padded = _pad_to(self.sequence, spec.cta_k)
        self.block_rows = spec.cta_k if block_rows is None else int(block_rows)
        if self.block_rows <= 0 or self.block_rows % spec.cta_k:
            raise ValueError('V scale blocks must align to the attention KV tile')
        self.blocks = (self.sequence + self.block_rows - 1) // self.block_rows
        if backend not in (v_staging.BACKEND_NATIVE, v_staging.BACKEND_TORCH):
            raise ValueError('unknown block-V staging backend')
        self.library = v_staging._native_library() if backend == v_staging.BACKEND_NATIVE else None
        if backend == v_staging.BACKEND_NATIVE and self.library is None:
            raise RuntimeError('Single-pass V requires native V staging kernels')
        self.v = torch.empty(self.batch * self.heads * self.dim, self.padded,
                             dtype=torch.int8, device=spec.device)
        self.scales = torch.empty(self.batch, self.heads, self.blocks, self.dim,
                                  dtype=torch.float32, device=spec.device)
        v_staging._zero_padding(self.v, self.sequence)
        self.next_row = 0

    def quantize(self, value, start):
        rows = value.shape[2]
        end = start + rows
        if (start != self.next_row or start % self.block_rows or
                end != min(start + self.block_rows, self.sequence)):
            raise ValueError('block-V chunks must cover aligned rows in order')
        if (tuple(value.shape) != (self.batch, self.heads, rows, self.dim) or
                value.dtype != self.spec.input_dtype or value.device != self.v.device or value.stride(-1) != 1):
            raise ValueError('block-V chunk does not match its producer spec')
        amax = torch.zeros(self.batch, self.heads, self.dim, dtype=torch.float32, device=value.device)
        if self.library is None:
            v_staging._torch_update(amax, value)
        else:
            v_staging._check(self.library, self.library.h3_int8_v_amax_chunk(
                _ptr(value), _ptr(amax), self.batch, self.heads, rows, self.dim,
                *value.stride()[:3], _DTYPE_TO_CODE[value.dtype], _stream()), 'block-V amax')
        scale = (amax * (1.0 / 127.0)).clamp_min_(1e-12)
        self.scales[:, :, start // self.block_rows].copy_(scale)
        if self.library is None:
            v_staging._torch_quantize(value, self.v, scale, start)
        else:
            v_staging._check(self.library, self.library.h3_int8_quantize_v_chunk_into(
                _ptr(value), _ptr(self.v), _ptr(scale), self.batch, self.heads, rows,
                start, self.dim, self.padded, *value.stride()[:3],
                _DTYPE_TO_CODE[value.dtype], _stream()), 'block-V quantize')
        self.next_row = end

    def finish(self):
        if self.next_row != self.sequence:
            raise ValueError('block-V chunks do not cover the sequence')
        return self.v, self.scales


def attention(quantized, output, dtype, geometry, strides):
    batch, _, length, _, heads, dim = geometry
    scale_rows = quantized.v_scale_rows or quantized.cta_k
    if scale_rows <= 0 or scale_rows % quantized.cta_k:
        raise ValueError('V scale blocks must align to the attention KV tile')
    expected = (batch, heads, (length + scale_rows - 1) // scale_rows, dim)
    if (tuple(quantized.v_scale.shape) != expected or dim != 128 or
            quantized.v_scale.dtype != torch.float32 or not quantized.v_scale.is_contiguous() or
            quantized.v_scale.device != quantized.q.device):
        raise ValueError('invalid block-V scales or head dimension')
    library = _library()
    packed = (scale_rows % 128 == 0 and length > 1024
              and packed_dense_attention_is_available(quantized.q.device))
    q_scale = quantized.q_scale if packed else quantized.q_scale.contiguous()
    status = library.h3_block_v_attention_v3(
        _ptr(quantized.q), _ptr(quantized.k), _ptr(quantized.v), _ptr(output),
        _ptr(q_scale), _ptr(quantized.k_scale), _ptr(quantized.v_scale),
        quantized.cta_k, *geometry, *strides,
        q_scale.stride(0), q_scale.stride(1),
        quantized.attention_scale, _DTYPE_TO_CODE[dtype], scale_rows, int(packed), _stream())
    if status:
        raise RuntimeError(library.h3_block_v_last_error().decode('utf-8', errors='replace'))
