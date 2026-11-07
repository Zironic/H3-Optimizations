"""Two-pass V carrier staging without a full-sequence BF16 V tensor."""

from __future__ import annotations

import ctypes

import torch

from . import loader
from .int8_attention import _DTYPE_TO_CODE, _pad_to, _ptr, _stream


BACKEND_NATIVE = 'native'
BACKEND_TORCH = 'torch_reference'
_NATIVE_SYMBOLS = (
    'h3_int8_v_amax_chunk',
    'h3_int8_quantize_v_chunk_into',
)


class VStagingError(RuntimeError):
    pass


def _bind(library):
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    integer64 = ctypes.c_int64
    stream = ctypes.c_size_t
    library.h3_int8_v_amax_chunk.restype = integer
    library.h3_int8_v_amax_chunk.argtypes = (
        [pointer, pointer] + [integer] * 4 + [integer64] * 3 + [integer, stream]
    )
    library.h3_int8_quantize_v_chunk_into.restype = integer
    library.h3_int8_quantize_v_chunk_into.argtypes = (
        [pointer, pointer, pointer]
        + [integer] * 6
        + [integer64] * 3
        + [integer, stream]
    )
    return library


def _native_library():
    try:
        library = loader.load()
    except loader.NativeUnavailableError:
        return None
    if not all(hasattr(library, symbol) for symbol in _NATIVE_SYMBOLS):
        return None
    return _bind(library)


def native_v_staging_available():
    return _native_library() is not None


def _check(library, status, what):
    if status:
        loader.check(status, what)


def _inverse_permutation_16(device):
    source = torch.arange(16, dtype=torch.int64, device=device)
    return (
        (source & 1)
        | (((source >> 3) & 1) << 1)
        | (((source >> 1) & 1) << 2)
        | (((source >> 2) & 1) << 3)
    )


def _zero_padding(v_int8, sequence):
    padded = int(v_int8.shape[-1])
    if padded == int(sequence):
        return
    source = torch.arange(
        int(sequence), padded, dtype=torch.int64, device=v_int8.device
    )
    destination = (source & ~15) | _inverse_permutation_16(v_int8.device)[
        source & 15
    ]
    v_int8.index_fill_(1, destination, 0)


def _torch_update(amax, v_chunk):
    chunk = v_chunk.to(torch.float32).abs().amax(dim=-2)
    amax.copy_(torch.maximum(amax, chunk))


def _torch_quantize(v_chunk, v_int8, scale, row_start):
    batch, heads, rows, head_dim = v_chunk.shape
    source = torch.arange(
        int(row_start),
        int(row_start) + int(rows),
        dtype=torch.int64,
        device=v_chunk.device,
    )
    destination = (source & ~15) | _inverse_permutation_16(v_chunk.device)[
        source & 15
    ]
    inverse = (
        1.0 / scale.reshape(batch, heads, head_dim).to(torch.float32)
    ).unsqueeze(-2)
    quantized = torch.clamp(
        torch.round(v_chunk.to(torch.float32) * inverse), -128.0, 127.0
    ).to(torch.int8)
    v_int8.view(batch, heads, head_dim, -1).index_copy_(
        3,
        destination,
        quantized.permute(0, 1, 3, 2).contiguous(),
    )


class TwoPassVCarrier:
    def __init__(self, spec, *, backend=BACKEND_NATIVE):
        self.spec = spec
        self.batch = int(spec.k_input_shape[0])
        self.heads = int(spec.k_input_shape[1])
        self.sequence = int(spec.k_input_shape[2])
        self.head_dim = int(spec.kernel_head_dim)
        self.padded = _pad_to(self.sequence, int(spec.cta_k))
        if backend not in (BACKEND_NATIVE, BACKEND_TORCH):
            raise VStagingError('unknown V staging backend %r' % backend)
        self.backend = backend
        self.library = _native_library()
        if backend == BACKEND_NATIVE and self.library is None:
            raise VStagingError(
                'two-pass V requires native V staging kernels in the H3 library'
            )
        self.amax = torch.zeros(
            self.batch,
            self.heads,
            self.head_dim,
            dtype=torch.float32,
            device=spec.device,
        )
        self.scale = None
        self.v_int8 = None
        self._covered = []

    def _check_chunk(self, v_chunk):
        if v_chunk.ndim != 4:
            raise VStagingError('V chunks must be [batch, heads, rows, dim]')
        if tuple(v_chunk.shape[:2]) != (self.batch, self.heads):
            raise VStagingError('V chunk batch/head shape does not match spec')
        if int(v_chunk.shape[-1]) != self.head_dim:
            raise VStagingError('V chunk head dimension does not match spec')
        if v_chunk.dtype != self.spec.input_dtype:
            raise VStagingError('V chunk dtype does not match spec')
        if v_chunk.stride(-1) != 1:
            raise VStagingError('V chunk head dimension must be contiguous')

    def update(self, v_chunk):
        if self.scale is not None:
            raise VStagingError('V scale is already finalized')
        self._check_chunk(v_chunk)
        if self.backend == BACKEND_TORCH:
            _torch_update(self.amax, v_chunk)
            return
        batch, heads, rows, head_dim = v_chunk.shape
        _check(
            self.library,
            self.library.h3_int8_v_amax_chunk(
                _ptr(v_chunk),
                _ptr(self.amax),
                batch,
                heads,
                rows,
                head_dim,
                v_chunk.stride(0),
                v_chunk.stride(1),
                v_chunk.stride(2),
                _DTYPE_TO_CODE[v_chunk.dtype],
                _stream(),
            ),
            'v_amax_chunk',
        )

    def finalize_scale(self):
        if self.scale is None:
            self.scale = torch.clamp(
                self.amax * (1.0 / 127.0), min=1e-12
            ).reshape(-1).contiguous()
            self.v_int8 = torch.empty(
                self.batch * self.heads * self.head_dim,
                self.padded,
                dtype=torch.int8,
                device=self.spec.device,
            )
            _zero_padding(self.v_int8, self.sequence)
            self.amax = None
        return self.scale

    def quantize(self, v_chunk, row_start):
        if self.scale is None:
            raise VStagingError('V scale must be finalized before quantization')
        self._check_chunk(v_chunk)
        row_start = int(row_start)
        rows = int(v_chunk.shape[2])
        if row_start < 0 or row_start + rows > self.sequence:
            raise VStagingError('V chunk falls outside the carrier sequence')
        if self.backend == BACKEND_TORCH:
            _torch_quantize(v_chunk, self.v_int8, self.scale, row_start)
        else:
            batch, heads, _, head_dim = v_chunk.shape
            _check(
                self.library,
                self.library.h3_int8_quantize_v_chunk_into(
                    _ptr(v_chunk),
                    _ptr(self.v_int8),
                    _ptr(self.scale),
                    batch,
                    heads,
                    rows,
                    row_start,
                    head_dim,
                    self.padded,
                    v_chunk.stride(0),
                    v_chunk.stride(1),
                    v_chunk.stride(2),
                    _DTYPE_TO_CODE[v_chunk.dtype],
                    _stream(),
                ),
                'quantize_v_chunk_into',
            )
        self._covered.append((row_start, row_start + rows))

    def finish(self):
        covered = 0
        for start, end in sorted(self._covered):
            if start != covered:
                raise VStagingError('V chunks contain a gap or overlap')
            covered = end
        if covered != self.sequence:
            raise VStagingError('V chunks do not cover the carrier sequence')
        return self.v_int8, self.scale


def chunk_scaled_block_rows_ok(block_rows):
    """Block sizes the chunk-scaled kernels accept: a power of two >= 128."""
    block_rows = int(block_rows)
    return block_rows >= 128 and block_rows & (block_rows - 1) == 0


class ChunkScaledVCarrier(TwoPassVCarrier):
    """Single-pass V carrier with one scale per block of ``block_rows`` keys.

    Each projection chunk is quantized against its own per-channel maximum
    the moment it is produced, so neither a full-sequence BF16 V nor a second
    V projection is needed. The scales are [batch, heads, blocks, head_dim];
    the carrier declares ``v_scale_rows = block_rows`` and only the
    chunk-scaled attention entry points may read it. Chunks must arrive in
    order and each must be exactly one block (the last may be short).
    """

    def __init__(self, spec, block_rows, *, backend=BACKEND_NATIVE):
        super().__init__(spec, backend=backend)
        if not chunk_scaled_block_rows_ok(block_rows):
            raise VStagingError(
                'chunk-scaled V needs a power-of-two block of at least 128 '
                'rows, got %r' % (block_rows,)
            )
        self.block_rows = int(block_rows)
        self.blocks = -(-self.sequence // self.block_rows)
        self.block_scales = torch.empty(
            self.batch,
            self.heads,
            self.blocks,
            self.head_dim,
            dtype=torch.float32,
            device=spec.device,
        )
        # The native quantizer reads one contiguous [B*H*D] scale per chunk.
        self.scale = torch.empty(
            self.batch * self.heads * self.head_dim,
            dtype=torch.float32,
            device=spec.device,
        )
        self.v_int8 = torch.empty(
            self.batch * self.heads * self.head_dim,
            self.padded,
            dtype=torch.int8,
            device=spec.device,
        )
        _zero_padding(self.v_int8, self.sequence)

    def finalize_scale(self):
        raise VStagingError('chunk-scaled V has no whole-sequence scale')

    def quantize(self, v_chunk, row_start):
        self._check_chunk(v_chunk)
        row_start = int(row_start)
        rows = int(v_chunk.shape[2])
        expected = self._covered[-1][1] if self._covered else 0
        if (
            row_start != expected
            or row_start % self.block_rows
            or rows != min(self.block_rows, self.sequence - row_start)
        ):
            raise VStagingError(
                'chunk-scaled V chunks must be whole %d-row blocks, in order'
                % self.block_rows
            )
        self.amax.zero_()
        if self.backend == BACKEND_TORCH:
            _torch_update(self.amax, v_chunk)
        else:
            batch, heads, rows, head_dim = v_chunk.shape
            _check(
                self.library,
                self.library.h3_int8_v_amax_chunk(
                    _ptr(v_chunk), _ptr(self.amax), batch, heads, rows,
                    head_dim, v_chunk.stride(0), v_chunk.stride(1),
                    v_chunk.stride(2), _DTYPE_TO_CODE[v_chunk.dtype],
                    _stream(),
                ),
                'v_amax_chunk',
            )
        torch.mul(self.amax.view(-1), 1.0 / 127.0, out=self.scale)
        self.scale.clamp_(min=1e-12)
        self.block_scales[:, :, row_start // self.block_rows].copy_(
            self.scale.view(self.batch, self.heads, self.head_dim)
        )
        if self.backend == BACKEND_TORCH:
            _torch_quantize(v_chunk, self.v_int8, self.scale, row_start)
        else:
            batch, heads, _, head_dim = v_chunk.shape
            _check(
                self.library,
                self.library.h3_int8_quantize_v_chunk_into(
                    _ptr(v_chunk), _ptr(self.v_int8), _ptr(self.scale),
                    batch, heads, rows, row_start, head_dim, self.padded,
                    v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2),
                    _DTYPE_TO_CODE[v_chunk.dtype], _stream(),
                ),
                'quantize_v_chunk_into',
            )
        self._covered.append((row_start, row_start + rows))

    def finish(self):
        if not self._covered or self._covered[-1][1] != self.sequence:
            raise VStagingError('V chunks do not cover the carrier sequence')
        self.amax = None
        self.scale = None
        return self.v_int8, self.block_scales.view(-1)
