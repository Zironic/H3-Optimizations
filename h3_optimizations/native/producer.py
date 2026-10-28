"""Build INT8 attention carriers a chunk at a time.

A chunked QKV projection can quantize each chunk as it is produced instead of
holding the whole BF16 sequence, which is where the QKV speedup comes from.
The single-shot path has to see all of Q, K and V at once; this one only ever
holds a chunk plus the carriers being filled.

Chunking costs one thing. The single-shot quantizer discovers the K anchor by
scanning, so it is chosen up
front from nine sampled rows and passed in, and the spec's
``sequence_alignment`` keeps chunk boundaries on tile boundaries so the
per-thread scales come out identical to the whole-tensor path. Identical is
the word: ``tests/test_native_producer.py`` compares the two byte for byte,
because a carrier that is merely close is a carrier nothing downstream can
trust.

The API mirrors the one h3_optimizations/kitchen_qkv.py already calls, so that
integration works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .. import diagnostics
from . import loader
from .int8_attention import (
    CTA_K,
    LARGE_CTA_K,
    PrequantizedInt8Attention,
    Q_TILE,
    _DTYPE_TO_CODE,
    _kernel_head_dim,
    _pad_to,
    _ptr,
    _stream,
    int8_attention_is_available,
    select_cta_k,
)

INT8_ATTENTION_PRODUCER_ABI_VERSION = 1
# The chunk quantizer takes explicit Q/K strides, so a caller holding an
# already-correct strided view does not have to materialize a contiguous copy
# first. Declared so kitchen_qkv can tell this module apart from an installed
# comfy-kitchen that has no such option.
SUPPORTS_STRIDED_QK_CHUNK = True
_K_ANCHOR_SAMPLES = 9
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class Int8AttentionProducerUnavailableError(RuntimeError):
    """The producer cannot run here, for a reason worth reporting."""


@dataclass(frozen=True)
class Int8AttentionProducerSpec:
    abi_version: int
    backend: str
    device: torch.device
    input_dtype: torch.dtype
    q_input_shape: tuple
    k_input_shape: tuple
    original_head_dim: int
    kernel_head_dim: int
    attention_scale: float
    cta_k: int
    q_tile: int
    k_tile: int
    sequence_alignment: int
    k_anchor_positions: tuple


@dataclass(frozen=True)
class Int8AttentionKAnchor:
    values: torch.Tensor
    indices: torch.Tensor


@dataclass
class Int8AttentionProducer:
    spec: Int8AttentionProducerSpec
    anchor: Int8AttentionKAnchor
    q: torch.Tensor
    k: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None
    _q_ranges: list = field(default_factory=list, repr=False)
    _k_ranges: list = field(default_factory=list, repr=False)
    _finalized: bool = field(default=False, repr=False)
    v_scale_rows: int = 0


def int8_attention_producer_is_available(device=None):
    """The producer is usable only when the consuming native kernel is proven."""
    return int8_attention_is_available(device)


def int8_attention_k_anchor_positions(kv_length):
    """The absolute K rows the stabilization detector samples."""
    if not isinstance(kv_length, int) or isinstance(kv_length, bool) or kv_length <= 0:
        raise ValueError('kv_length must be a positive integer, got %r' % kv_length)
    return tuple(
        sample * (kv_length - 1) // (_K_ANCHOR_SAMPLES - 1)
        for sample in range(_K_ANCHOR_SAMPLES)
    )


def _shape4(name, shape):
    normalized = tuple(int(value) for value in shape)
    if len(normalized) != 4:
        raise ValueError('%s must have four dimensions, got %s' % (name, normalized))
    if any(value <= 0 for value in normalized):
        raise ValueError('%s dimensions must be positive, got %s' % (name, normalized))
    return normalized


def int8_attention_producer_spec(
    q_shape, k_shape, *, dtype, device, scale=None, cta_k=None
):
    """Fix the carrier geometry before any chunk is produced."""
    if not int8_attention_producer_is_available(device):
        raise Int8AttentionProducerUnavailableError(
            'the native INT8 attention library is unavailable: %s'
            % (loader.unavailable_reason() or 'no CUDA device')
        )
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError('dtype must be float32, float16 or bfloat16, got %r' % dtype)

    q_input_shape = _shape4('q_shape', q_shape)
    k_input_shape = _shape4('k_shape', k_shape)
    if q_input_shape[0] != k_input_shape[0] or q_input_shape[3] != k_input_shape[3]:
        raise ValueError('q and k must share batch and head_dim')

    original_head_dim = q_input_shape[3]
    kernel_head_dim = _kernel_head_dim(original_head_dim)
    kv_length = k_input_shape[2]
    if cta_k is None:
        cta_k = select_cta_k(kernel_head_dim, kv_length)
    else:
        cta_k = int(cta_k)
        if cta_k not in (CTA_K, LARGE_CTA_K):
            raise ValueError(
                'cta_k must be %d or %d' % (CTA_K, LARGE_CTA_K)
            )
        if kernel_head_dim == 64 and cta_k != CTA_K:
            raise ValueError('head_dim 64 requires cta_k %d' % CTA_K)
    return Int8AttentionProducerSpec(
        abi_version=INT8_ATTENTION_PRODUCER_ABI_VERSION,
        backend='h3_native',
        device=torch.device(device),
        input_dtype=dtype,
        q_input_shape=q_input_shape,
        k_input_shape=k_input_shape,
        original_head_dim=original_head_dim,
        kernel_head_dim=kernel_head_dim,
        attention_scale=(
            original_head_dim ** -0.5 if scale is None else float(scale)
        ),
        cta_k=cta_k,
        q_tile=Q_TILE,
        k_tile=cta_k,
        # Chunks must land on both tile boundaries, or a chunk's per-thread
        # scales cover a different span than the whole-tensor path would.
        sequence_alignment=Q_TILE if Q_TILE % cta_k == 0 else Q_TILE * cta_k,
        k_anchor_positions=int8_attention_k_anchor_positions(kv_length),
    )


def _pad_last_dim(tensor, width):
    if tensor.shape[-1] == width:
        return tensor.contiguous()
    return torch.nn.functional.pad(
        tensor, (0, width - tensor.shape[-1])
    ).contiguous()


def _kernel_accepts_strided(tensor, width):
    """Whether the chunk quantizer can read this tensor where it already is.

    Mirrors the launcher's own predicate rather than approximating it: the
    head dimension must be exactly the kernel width and contiguous, and the
    base pointer and B/H/N strides must preserve 4-element vector alignment.
    An extent below two is exempt because its index never reaches an address.
    Getting this wrong does not corrupt anything -- the launcher raises -- but
    checking here keeps the fast path from depending on that.
    """
    if tensor.shape[-1] != width or tensor.stride(-1) != 1:
        return False
    element_size = tensor.element_size()
    if tensor.data_ptr() % (4 * element_size):
        return False
    for dimension in range(3):
        extent = tensor.shape[dimension]
        stride = tensor.stride(dimension)
        if extent >= 2 and (stride <= 0 or stride % 4):
            return False
    return True


def _prepare_chunk_input(tensor, width, allow_strided):
    if allow_strided and _kernel_accepts_strided(tensor, width):
        return tensor
    return _pad_last_dim(tensor, width)


def select_int8_attention_k_anchor(spec, k_samples):
    """Choose the K row to centre on, from the sampled rows."""
    expected = (
        spec.k_input_shape[0], spec.k_input_shape[1],
        _K_ANCHOR_SAMPLES, spec.original_head_dim,
    )
    if tuple(k_samples.shape) != expected:
        raise ValueError(
            'k_samples must have shape %s, got %s' % (expected, tuple(k_samples.shape))
        )
    if k_samples.dtype != spec.input_dtype:
        raise TypeError('k_samples must have dtype %r' % spec.input_dtype)

    with diagnostics.stage('anchor_selection'):
        return _select_k_anchor(spec, k_samples)


def _select_k_anchor(spec, k_samples):
    library = loader.load()
    samples = _pad_last_dim(k_samples, spec.kernel_head_dim)
    positions = torch.tensor(
        spec.k_anchor_positions, dtype=torch.int32, device=spec.device
    )
    values = torch.empty(
        spec.k_input_shape[0], spec.k_input_shape[1], spec.kernel_head_dim,
        dtype=spec.input_dtype, device=spec.device,
    )
    indices = torch.empty(
        spec.k_input_shape[0], spec.k_input_shape[1],
        dtype=torch.int32, device=spec.device,
    )
    import ctypes

    loader.check(
        library.h3_int8_select_k_anchor(
            _ptr(samples),
            ctypes.cast(_ptr(positions), ctypes.POINTER(ctypes.c_int)),
            _ptr(values), _ptr(indices),
            spec.k_input_shape[0], spec.k_input_shape[1],
            spec.k_input_shape[2], spec.kernel_head_dim,
            samples.stride(0), samples.stride(1), samples.stride(2),
            _DTYPE_TO_CODE[spec.input_dtype], _stream(),
        ),
        'select_k_anchor',
    )
    return Int8AttentionKAnchor(values=values, indices=indices)


def create_int8_attention_producer(spec, anchor):
    """Allocate the carriers the chunks will fill."""
    if spec.abi_version != INT8_ATTENTION_PRODUCER_ABI_VERSION:
        raise Int8AttentionProducerUnavailableError(
            'producer spec ABI %d, expected %d'
            % (spec.abi_version, INT8_ATTENTION_PRODUCER_ABI_VERSION)
        )
    batch, q_heads, q_length, _ = spec.q_input_shape
    _, kv_heads, kv_length, _ = spec.k_input_shape
    device, head_dim = spec.device, spec.kernel_head_dim
    q_scales_per_tile = 64 if head_dim == 256 else 32
    return Int8AttentionProducer(
        spec=spec,
        anchor=anchor,
        q=torch.empty(batch, q_heads, q_length, head_dim, dtype=torch.int8, device=device),
        k=torch.empty(batch, kv_heads, kv_length, head_dim, dtype=torch.int8, device=device),
        q_scale=torch.empty(
            batch, q_heads,
            ((q_length + Q_TILE - 1) // Q_TILE) * q_scales_per_tile,
            dtype=torch.float32, device=device,
        ),
        k_scale=torch.empty(
            batch, kv_heads,
            ((kv_length + spec.cta_k - 1) // spec.cta_k) * 4,
            dtype=torch.float32, device=device,
        ),
    )


def _check_chunk(name, start, length, full, alignment):
    if start < 0 or start + length > full:
        raise ValueError(
            '%s chunk [%d, %d) falls outside the sequence of %d'
            % (name, start, start + length, full)
        )
    if start % alignment:
        raise ValueError(
            '%s chunk starts at %d, which is not a multiple of the %d-row '
            'alignment; the per-thread scales would cover a different span '
            'than the whole-tensor path' % (name, start, alignment)
        )
    if (start + length) != full and length % alignment:
        raise ValueError(
            '%s chunk of %d rows is not a multiple of the %d-row alignment'
            % (name, length, alignment)
        )


def quantize_int8_attention_qk_chunk(
    producer, q, k, *, q_start, k_start, allow_strided_input=False
):
    """Quantize one aligned slice of Q and K into the carriers."""
    if producer._finalized:
        raise RuntimeError('the producer has already been finalized')
    spec = producer.spec
    alignment = spec.sequence_alignment
    _check_chunk('q', q_start, q.shape[2], spec.q_input_shape[2], alignment)
    _check_chunk('k', k_start, k.shape[2], spec.k_input_shape[2], alignment)
    if q.dtype != spec.input_dtype or k.dtype != spec.input_dtype:
        raise TypeError('chunks must have dtype %r' % spec.input_dtype)

    library = loader.load()
    with diagnostics.stage('qk_pack_input_contiguous'):
        q = _prepare_chunk_input(q, spec.kernel_head_dim, allow_strided_input)
        k = _prepare_chunk_input(k, spec.kernel_head_dim, allow_strided_input)
    with diagnostics.stage('qk_carrier_pack'):
        _quantize_qk_chunk(library, producer, spec, q, k, q_start, k_start)
    producer._q_ranges.append((q_start, q_start + q.shape[2]))
    producer._k_ranges.append((k_start, k_start + k.shape[2]))


def _dummy_chunk(spec, heads):
    return torch.zeros(
        spec.q_input_shape[0],
        int(heads),
        1,
        spec.original_head_dim,
        dtype=spec.input_dtype,
        device=spec.device,
    )


def quantize_int8_attention_q_chunk(
    producer, q, *, q_start, allow_strided_input=False
):
    """Pack Q while doing only one row of companion K work."""
    if producer._finalized:
        raise RuntimeError('the producer has already been finalized')
    spec = producer.spec
    _check_chunk(
        'q', q_start, q.shape[2], spec.q_input_shape[2], spec.sequence_alignment
    )
    if spec.k_input_shape[2] != 1:
        raise ValueError('Q-only packing requires a one-row K companion')
    if q.dtype != spec.input_dtype:
        raise TypeError('chunks must have dtype %r' % spec.input_dtype)

    k = _dummy_chunk(spec, spec.k_input_shape[1])
    library = loader.load()
    with diagnostics.stage('qk_pack_input_contiguous'):
        q = _prepare_chunk_input(q, spec.kernel_head_dim, allow_strided_input)
        k = _prepare_chunk_input(k, spec.kernel_head_dim, False)
    with diagnostics.stage('qk_carrier_pack'):
        _quantize_qk_chunk(library, producer, spec, q, k, q_start, 0)
    producer._q_ranges.append((q_start, q_start + q.shape[2]))
    if not producer._k_ranges:
        producer._k_ranges.append((0, 1))


def quantize_int8_attention_q(q, *, full_k_length, allow_strided_input=False):
    """Pack a standalone Q slab with the transform selected by global K."""
    full_k_length = int(full_k_length)
    if full_k_length <= 0:
        raise ValueError('full_k_length must be positive')
    shape = _shape4('q_shape', q.shape)
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError('q must have dtype float32, float16 or bfloat16')
    if not q.is_cuda:
        raise ValueError('q must be a CUDA tensor')
    kernel_head_dim = _kernel_head_dim(shape[3])
    q = _prepare_chunk_input(q, kernel_head_dim, allow_strided_input)
    batch, heads, sequence, _ = q.shape
    q_int8 = torch.empty(
        batch,
        heads,
        sequence,
        kernel_head_dim,
        dtype=torch.int8,
        device=q.device,
    )
    q_scales_per_tile = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch,
        heads,
        ((sequence + Q_TILE - 1) // Q_TILE) * q_scales_per_tile,
        dtype=torch.float32,
        device=q.device,
    )
    library = loader.load()
    with diagnostics.stage('qk_carrier_pack'):
        loader.check(
            library.h3_int8_quantize_q_chunk(
                _ptr(q),
                _ptr(q_int8),
                _ptr(q_scale),
                batch,
                heads,
                sequence,
                sequence,
                0,
                kernel_head_dim,
                full_k_length,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                _DTYPE_TO_CODE[q.dtype],
                _stream(),
            ),
            'quantize_q_chunk',
        )
    return q_int8, q_scale


def quantize_int8_attention_k_chunk(
    producer, k, *, k_start, allow_strided_input=False
):
    """Pack K while retaining only a one-row companion Q carrier."""
    if producer._finalized:
        raise RuntimeError('the producer has already been finalized')
    spec = producer.spec
    _check_chunk(
        'k', k_start, k.shape[2], spec.k_input_shape[2], spec.sequence_alignment
    )
    if spec.q_input_shape[2] != 1:
        raise ValueError('K-only packing requires a one-row Q companion')
    if k.dtype != spec.input_dtype:
        raise TypeError('chunks must have dtype %r' % spec.input_dtype)

    q = _dummy_chunk(spec, spec.q_input_shape[1])
    library = loader.load()
    with diagnostics.stage('qk_pack_input_contiguous'):
        q = _prepare_chunk_input(q, spec.kernel_head_dim, False)
        k = _prepare_chunk_input(k, spec.kernel_head_dim, allow_strided_input)
    with diagnostics.stage('qk_carrier_pack'):
        _quantize_qk_chunk(library, producer, spec, q, k, 0, k_start)
    if not producer._q_ranges:
        producer._q_ranges.append((0, 1))
    producer._k_ranges.append((k_start, k_start + k.shape[2]))


def _quantize_qk_chunk(library, producer, spec, q, k, q_start, k_start):
    loader.check(
        library.h3_int8_quantize_qk_chunk(
            _ptr(q), _ptr(k), _ptr(producer.q), _ptr(producer.q_scale),
            _ptr(producer.k), _ptr(producer.k_scale),
            _ptr(producer.anchor.values), _ptr(producer.anchor.indices),
            spec.q_input_shape[0], spec.q_input_shape[1], q.shape[2],
            spec.q_input_shape[2], q_start,
            spec.k_input_shape[1], k.shape[2], spec.k_input_shape[2], k_start,
            spec.kernel_head_dim, spec.cta_k,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            _DTYPE_TO_CODE[spec.input_dtype], _stream(),
        ),
        'quantize_qk_chunk',
    )


def quantize_int8_attention_v(producer, v):
    """Quantize V. V needs no norm or rotation, so it arrives whole."""
    if producer._finalized:
        raise RuntimeError('the producer has already been finalized')
    spec = producer.spec
    if v.shape[2] != spec.k_input_shape[2]:
        raise ValueError(
            'v must cover the full key sequence of %d, got %d'
            % (spec.k_input_shape[2], v.shape[2])
        )

    with diagnostics.stage('v_carrier_pack'):
        _quantize_v(producer, spec, v)


def _quantize_v(producer, spec, v):
    library = loader.load()
    v = _pad_last_dim(v, spec.kernel_head_dim)
    batch, kv_heads = spec.k_input_shape[0], spec.k_input_shape[1]
    kv_length, head_dim = spec.k_input_shape[2], spec.kernel_head_dim
    padded = _pad_to(kv_length, spec.cta_k)
    v_int8 = torch.empty(
        batch * kv_heads * head_dim, padded, dtype=torch.int8, device=spec.device
    )
    v_scale = torch.empty(
        batch * kv_heads * head_dim, dtype=torch.float32, device=spec.device
    )
    loader.check(
        library.h3_int8_quantize_v(
            _ptr(v), _ptr(v_int8), _ptr(v_scale),
            batch, kv_heads, kv_length, head_dim, padded,
            v.stride(0), v.stride(1), v.stride(2),
            _DTYPE_TO_CODE[spec.input_dtype], _stream(),
        ),
        'quantize_v',
    )
    producer.v = v_int8
    producer.v_scale = v_scale


def _covers(ranges, length):
    """Whether the chunks tile the sequence exactly, with no gap or overlap."""
    covered = 0
    for start, end in sorted(ranges):
        if start != covered:
            return False
        covered = end
    return covered == length


def finalize_int8_attention_producer(producer):
    """Check the chunks covered everything, and hand back the carrier."""
    if producer._finalized:
        raise RuntimeError('the producer has already been finalized')
    spec = producer.spec
    if not _covers(producer._q_ranges, spec.q_input_shape[2]):
        raise RuntimeError('Q chunks do not cover the full query sequence')
    if not _covers(producer._k_ranges, spec.k_input_shape[2]):
        raise RuntimeError('K chunks do not cover the full key sequence')
    if producer.v is None or producer.v_scale is None:
        raise RuntimeError('V was never quantized')

    producer._finalized = True
    return PrequantizedInt8Attention(
        q=producer.q, k=producer.k, v=producer.v,
        q_scale=producer.q_scale, k_scale=producer.k_scale,
        v_scale=producer.v_scale,
        original_head_dim=spec.original_head_dim,
        input_dtype=spec.input_dtype,
        attention_scale=spec.attention_scale,
        cta_k=spec.cta_k,
        anchor_indices=producer.anchor.indices,
        v_scale_rows=producer.v_scale_rows,
    )
