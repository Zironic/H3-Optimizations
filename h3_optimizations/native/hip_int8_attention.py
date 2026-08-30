"""Experimental gfx12 64Q x 64KV Sparse Kitchen attention."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .. import diagnostics
from . import hip_loader as loader


__version__ = 'experimental-gfx12-bm64'
IS_HIP_SPARSE_KITCHEN = True
KERNEL_ROUTE_ENCODING = 'absolute'
OUTPUT_HND = 'hnd'
OUTPUT_NHD = 'nhd'
OUTPUT_LAYOUTS = (OUTPUT_HND, OUTPUT_NHD)
SPARSE_GEOMETRIES = ((64, 64),)
Q_TILE = 64
KV_TILE = 64
HEAD_DIM = 128
_SUPPORTED_ARCHITECTURES = ('gfx1200', 'gfx1201')
_DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _pad_to(value, multiple):
    return ((int(value) + multiple - 1) // multiple) * multiple


def _ptr(tensor):
    return tensor.data_ptr()


def _stream(device=None):
    return torch.cuda.current_stream(device).cuda_stream


def device_architecture(device=None):
    if not getattr(torch.version, 'hip', None) or not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(device)
    value = str(getattr(properties, 'gcnArchName', ''))
    return value.split(':', 1)[0] or None


@dataclass(frozen=True)
class PrequantizedInt8Attention:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    original_head_dim: int
    input_dtype: torch.dtype
    attention_scale: float
    cta_k: int
    q_length: int
    kv_length: int
    anchor_indices: torch.Tensor


@dataclass(frozen=True)
class BlockSparseRoute:
    indices: torch.Tensor
    counts: torch.Tensor
    q_tile: int
    kv_tile: int
    encoding: str = 'absolute'

    def __post_init__(self):
        if self.encoding not in ('absolute', 'delta'):
            raise ValueError("encoding must be 'absolute' or 'delta'")

    def to_absolute(self):
        if self.encoding == 'absolute':
            return self
        live = torch.arange(
            self.indices.shape[-1], device=self.indices.device
        ) < self.counts.to(torch.int64).unsqueeze(-1)
        positions = self.indices.to(torch.int64).cumsum(dim=-1)
        indices = torch.where(live, positions, torch.zeros_like(positions))
        return replace(
            self,
            indices=indices.to(torch.int32).contiguous(),
            encoding='absolute',
        )

    def for_kernel(self):
        return self.to_absolute()


def _validate_inputs(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('q, k, and v must have shape [batch, heads, sequence, head_dim]')
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError('experimental AMD Sparse Kitchen currently supports batch 1')
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError('experimental AMD Sparse Kitchen requires equal Q/K/V head counts')
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        raise ValueError('experimental AMD Sparse Kitchen requires head_dim 128')
    if k.shape[-2] != v.shape[-2]:
        raise ValueError('k and v sequence lengths must match')
    if q.dtype not in _DTYPE_TO_CODE or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError('q, k, and v must share float16 or bfloat16 dtype')
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError('q, k, and v must be on the same ROCm device')
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError('q, k, and v head dimensions must be contiguous')


def prequantize_int8_attention(q, k, v, *, cta_k=KV_TILE):
    _validate_inputs(q, k, v)
    if int(cta_k) != KV_TILE:
        raise ValueError('experimental AMD Sparse Kitchen requires cta_k=64')
    library = loader.load()
    _, heads, q_length, _ = q.shape
    kv_length = k.shape[-2]
    padded_q = _pad_to(q_length, 128)
    padded_k = _pad_to(kv_length, KV_TILE)
    tiles = padded_k // KV_TILE
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty((1, heads, padded_k, HEAD_DIM), dtype=torch.int8, device=q.device)
    v_int8 = torch.empty(
        (1, heads, tiles, HEAD_DIM, KV_TILE), dtype=torch.int8, device=q.device
    )
    q_scale = torch.empty((1, heads, padded_q), dtype=torch.float32, device=q.device)
    k_scale = torch.empty((1, heads, padded_k // 16), dtype=torch.float32, device=q.device)
    v_scale = torch.empty((1, heads, tiles, HEAD_DIM), dtype=torch.float32, device=q.device)
    anchor_indices = torch.empty((1, heads), dtype=torch.int32, device=q.device)
    stream = _stream(q.device)
    dtype_code = _DTYPE_TO_CODE[q.dtype]
    with diagnostics.stage('qk_carrier_pack'):
        loader.check(
            library.h3_hip_sparse_quantize_qk(
                _ptr(q), _ptr(q_int8), _ptr(q_scale),
                _ptr(k), _ptr(k_int8), _ptr(k_scale), _ptr(anchor_indices),
                heads, q_length, kv_length, padded_q, padded_k,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                dtype_code, stream,
            ),
            'HIP Sparse Kitchen Q/K quantization',
        )
    with diagnostics.stage('v_carrier_pack'):
        loader.check(
            library.h3_hip_sparse_quantize_v(
                _ptr(v), _ptr(v_int8), _ptr(v_scale),
                heads, kv_length, tiles, v.stride(1), v.stride(2),
                dtype_code, stream,
            ),
            'HIP Sparse Kitchen V quantization',
        )
    return PrequantizedInt8Attention(
        q=q_int8,
        k=k_int8,
        v=v_int8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        original_head_dim=HEAD_DIM,
        input_dtype=q.dtype,
        attention_scale=HEAD_DIM ** -0.5,
        cta_k=KV_TILE,
        q_length=q_length,
        kv_length=kv_length,
        anchor_indices=anchor_indices,
    )


def _validate_route(quantized, route):
    if not isinstance(route, BlockSparseRoute):
        raise TypeError('route must be BlockSparseRoute')
    if (int(route.q_tile), int(route.kv_tile)) != (Q_TILE, KV_TILE):
        raise ValueError('experimental AMD Sparse Kitchen requires 64Q x 64KV routes')
    route = route.for_kernel()
    q_blocks = _pad_to(quantized.q_length, Q_TILE) // Q_TILE
    expected = (1, quantized.q.shape[1], q_blocks)
    if route.indices.ndim != 4 or tuple(route.indices.shape[:-1]) != expected:
        raise ValueError('route indices must have shape [1, heads, q_blocks, selected]')
    if tuple(route.counts.shape) != expected:
        raise ValueError('route counts must have shape [1, heads, q_blocks]')
    if route.indices.shape[-1] <= 0:
        raise ValueError('route must reserve at least one selected KV tile')
    if route.indices.dtype != torch.int32 or route.counts.dtype != torch.int32:
        raise TypeError('route indices and counts must be int32')
    if route.indices.device != quantized.q.device or route.counts.device != quantized.q.device:
        raise ValueError('route and carrier must be on the same ROCm device')
    return replace(
        route,
        indices=route.indices.contiguous(),
        counts=route.counts.contiguous(),
    )


def _output(quantized, output_layout):
    if output_layout == OUTPUT_HND:
        output = torch.empty(
            (1, quantized.q.shape[1], quantized.q_length, HEAD_DIM),
            dtype=quantized.input_dtype,
            device=quantized.q.device,
        )
    elif output_layout == OUTPUT_NHD:
        output = torch.empty(
            (1, quantized.q_length, quantized.q.shape[1], HEAD_DIM),
            dtype=quantized.input_dtype,
            device=quantized.q.device,
        ).transpose(1, 2)
    else:
        raise ValueError('unknown output layout %r' % output_layout)
    return output


def block_sparse_int8_attention_from_prequantized(
    quantized, route, *, output_layout=OUTPUT_HND, validate_geometry=True
):
    del validate_geometry
    library = loader.load()
    route = _validate_route(quantized, route)
    output = _output(quantized, output_layout)
    with diagnostics.stage('sparse_attention_kernel'):
        loader.check(
            library.h3_hip_sparse_attention(
                _ptr(quantized.q), _ptr(quantized.k), _ptr(quantized.v), _ptr(output),
                _ptr(quantized.q_scale), _ptr(quantized.k_scale), _ptr(quantized.v_scale),
                _ptr(route.indices), _ptr(route.counts), route.indices.shape[-1],
                quantized.q.shape[1], quantized.q_length, quantized.kv_length,
                quantized.q_scale.shape[-1], quantized.k.shape[-2],
                output.stride(1), output.stride(2), quantized.attention_scale,
                _DTYPE_TO_CODE[output.dtype], _stream(output.device),
            ),
            'HIP Sparse Kitchen attention',
        )
    return output


def int8_attention_from_prequantized(quantized, *, output_layout=OUTPUT_HND):
    kv_tiles = _pad_to(quantized.kv_length, KV_TILE) // KV_TILE
    q_tiles = _pad_to(quantized.q_length, Q_TILE) // Q_TILE
    indices = torch.arange(kv_tiles, dtype=torch.int32, device=quantized.q.device)
    route = BlockSparseRoute(
        indices=indices.view(1, 1, 1, kv_tiles)
        .expand(1, quantized.q.shape[1], q_tiles, kv_tiles)
        .contiguous(),
        counts=torch.full(
            (1, quantized.q.shape[1], q_tiles),
            kv_tiles,
            dtype=torch.int32,
            device=quantized.q.device,
        ),
        q_tile=Q_TILE,
        kv_tile=KV_TILE,
    )
    return block_sparse_int8_attention_from_prequantized(
        quantized, route, output_layout=output_layout
    )


def route_encoding():
    return loader.route_encoding()


def int8_attention_is_available(device=None):
    if device_architecture(device) not in _SUPPORTED_ARCHITECTURES:
        return False
    if not loader.is_available():
        return False
    from . import hip_selftest

    return hip_selftest.check(device)
