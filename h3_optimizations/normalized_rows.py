'''Lazily normalized attention input rows.

The QKV producers only ever read their input as ``x[start:end]`` row slices,
plus ``shape``/``dtype``/``device``/``new_empty``. That lets the pre-attention
``norm1`` + modulation run per requested slice instead of materializing a full
``[sequence, hidden]`` tensor that stays resident for the whole block.

The lazy row source is private to the package-owned attention/QKV transaction.
The DiT block still calls attention with an ordinary Tensor; the optimized
attention forward extracts this row source from its private transformer option
and hands it only to a projector that explicitly owns the projected-QKV path.
Foreign or standard attention forwards therefore never have to emulate a new
Tensor protocol.

The trade is recomputation: a route that walks the sequence more than once
(streamed Kitchen projects K/V and Q in separate passes, and two-pass V adds a
third) normalizes those rows once per pass. RMSNorm at H3 shapes is bandwidth
bound, so this is close to even on time and removes one full-sequence BF16
tensor from the block's peak.

Consumers that need a real tensor can call ``materialize()``. Streamed
attention consumers normally call ``attention_output_buffer()`` when they are
ready to write the projected result. A compatible INT8 output projection can
instead consume each slab through ``write_attention_residual()`` and update the
wrapped block residual directly.
'''

from __future__ import annotations

import torch

import comfy.ops

from .native import (
    RMS_ADALN_BF16_STAGED,
    quantize_rms_adaln_int8_rowwise_convrot256,
)


NORM1_SOURCE_KEY = 'h3_optimizations_norm1_source'


class NormalizedRowsUnsupported(AttributeError):
    pass


class NormalizedRows:
    '''A read-only ``[sequence, hidden]`` row source with fused norm + modulation.

    Only the surface the QKV producers actually use is implemented. Anything
    else is deliberately absent so an unsupported projector fails loudly
    instead of silently reading unnormalized rows.
    '''

    __slots__ = (
        '_x',
        '_norm',
        '_segments',
        '_shift',
        '_scale',
        '_apply',
        '_output',
        '_attention_gate',
        '_attention_next',
        '_fused_convrot_quantize',
        'shape',
        'dtype',
        'device',
        'ndim',
        'is_cuda',
        'requires_grad',
    )

    def __init__(self, x, norm, segments, shift, scale, apply_modulation):
        self._x = x
        self._norm = norm
        self._segments = tuple(segments)
        self._shift = shift
        self._scale = scale
        self._apply = apply_modulation
        self._output = None
        self._attention_gate = None
        self._attention_next = 0
        self._fused_convrot_quantize = False
        self.shape = x.shape
        self.dtype = x.dtype
        self.device = x.device
        self.ndim = x.ndim
        self.is_cuda = x.is_cuda
        self.requires_grad = False

    def new_empty(self, *args, **kwargs):
        return self._x.new_empty(*args, **kwargs)

    def size(self, dim=None):
        return self._x.shape if dim is None else self._x.shape[dim]

    def dim(self):
        return self._x.dim()

    def numel(self):
        return self._x.numel()

    def element_size(self):
        return self._x.element_size()

    def stride(self, dim=None):
        return self._x.stride() if dim is None else self._x.stride(dim)

    def is_contiguous(self, *args, **kwargs):
        return self._x.is_contiguous(*args, **kwargs)

    def index_select(self, dim, index):
        '''Gathered rows, normalized and modulated. Anchor/sample selection.'''
        if dim != 0:
            raise NormalizedRowsUnsupported(
                'NormalizedRows supports row (dim 0) selection only'
            )
        rows = self._norm(self._x.index_select(0, index))
        self._apply(rows, self._shift, self._scale, self._row_selectors(index))
        return rows

    def __len__(self):
        return int(self._x.shape[0])

    def __getitem__(self, item):
        start, stop = self._bounds(item)
        rows = self._norm(self._x[start:stop])
        self._modulate(rows, start, stop)
        return rows

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types, args, kwargs
        raise NormalizedRowsUnsupported(
            'NormalizedRows does not support torch.%s' % func.__name__
        )

    def __getattr__(self, name):
        raise NormalizedRowsUnsupported(
            'NormalizedRows does not expose tensor attribute %r' % name
        )

    def _bounds(self, item):
        sequence = int(self._x.shape[0])
        if isinstance(item, slice):
            if item.step not in (None, 1):
                raise NormalizedRowsUnsupported(
                    'NormalizedRows supports contiguous row slices only'
                )
            start, stop, _ = item.indices(sequence)
            if stop < start:
                stop = start
            return start, stop
        raise NormalizedRowsUnsupported(
            'NormalizedRows supports row slices only, got %r' % type(item).__name__
        )

    def _modulate(self, rows, start, stop):
        for seg_start, seg_stop, selector in self._segments:
            lo = seg_start if seg_start > start else start
            hi = seg_stop if seg_stop < stop else stop
            if lo >= hi:
                continue
            if torch.is_tensor(selector):
                sub = selector[lo - seg_start:hi - seg_start]
            else:
                sub = selector
            self._apply(rows[lo - start:hi - start], self._shift, self._scale, sub)

    def _row_selectors(self, index):
        out = torch.zeros(
            int(index.shape[0]), dtype=torch.long, device=index.device
        )
        for seg_start, seg_stop, selector in self._segments:
            mask = (index >= seg_start) & (index < seg_stop)
            if torch.is_tensor(selector):
                local = index[mask] - seg_start
                if selector.device != index.device:
                    selector = selector.to(device=index.device)
                out[mask] = selector[local]
            else:
                out[mask] = int(selector)
        return out

    def materialize(self):
        '''Full ``[sequence, hidden]`` tensor, for consumers that need one.'''
        return self[0:int(self._x.shape[0])]

    def enable_fused_convrot_quantization(self):
        self._fused_convrot_quantize = True

    def quantize_convrot256(self, start, stop):
        '''Produce the staged Norm1 + AdaLN ConvRot carrier, or decline.'''
        if not self._fused_convrot_quantize:
            return None
        start = int(start)
        stop = int(stop)
        spans = []
        for seg_start, seg_stop, selector in self._segments:
            lo = max(start, seg_start)
            hi = min(stop, seg_stop)
            if lo >= hi:
                continue
            if torch.is_tensor(selector):
                return None
            spans.append((lo, hi, selector))
        if not spans:
            raise RuntimeError('fused Norm1 quantizer received an empty row range')
        norm_weight = comfy.ops.cast_to_input(self._norm.weight, self._x).contiguous()
        pieces = [
            quantize_rms_adaln_int8_rowwise_convrot256(
                self._x[lo:hi],
                norm_weight,
                self._shift[selector].contiguous(),
                self._scale[selector].contiguous(),
                self._norm.eps,
                mode=RMS_ADALN_BF16_STAGED,
            )
            for lo, hi, selector in spans
        ]
        if len(pieces) == 1:
            return pieces[0]
        return (
            torch.cat([piece[0] for piece in pieces], dim=0),
            torch.cat([piece[1] for piece in pieces], dim=0),
        )

    def enable_attention_residual(self, gate):
        '''Offer the wrapped residual to a compatible streamed out projection.'''
        self._attention_gate = gate
        self._attention_next = 0

    @property
    def attention_residual_pending(self):
        return (
            self._attention_gate is not None
            and self._attention_next < int(self._x.shape[0])
        )

    @property
    def attention_residual_complete(self):
        return (
            self._attention_gate is not None
            and self._attention_next == int(self._x.shape[0])
        )

    def write_attention_residual(
        self,
        rows,
        start,
        stop,
        project,
        project_residual,
    ):
        '''Project one ordered attention slab straight into the block residual.'''
        start = int(start)
        stop = int(stop)
        if self._attention_gate is None:
            raise RuntimeError('attention residual output was not enabled')
        if start != self._attention_next or stop <= start:
            raise RuntimeError(
                'attention residual slabs must cover rows in order; got [%d, %d) after %d'
                % (start, stop, self._attention_next)
            )
        if tuple(rows.shape[:1]) != (stop - start,):
            raise RuntimeError('attention residual slab row count does not match its range')

        pieces = []
        for seg_start, seg_stop, selector in self._segments:
            lo = max(seg_start, start)
            hi = min(seg_stop, stop)
            if lo >= hi:
                continue
            pieces.append((lo, hi, seg_start, selector))

        if len(pieces) == 1 and not torch.is_tensor(pieces[0][3]):
            lo, hi, _seg_start, selector = pieces[0]
            gate = self._attention_gate[selector].to(dtype=self._x.dtype)
            if project_residual(rows, self._x[lo:hi], gate):
                self._attention_next = stop
                return

        projected = project(rows)
        for lo, hi, seg_start, selector in pieces:
            residual = self._x[lo:hi]
            if torch.is_tensor(selector):
                sub = selector[lo - seg_start:hi - seg_start]
                if sub.device != self._attention_gate.device:
                    sub = sub.to(device=self._attention_gate.device)
                gate = self._attention_gate[sub].to(dtype=residual.dtype)
            else:
                gate = self._attention_gate[selector].to(dtype=residual.dtype)
            residual.addcmul_(projected[lo - start:hi - start], gate)

        self._attention_next = stop

    def finish_attention_residual(self):
        if not self.attention_residual_complete:
            raise RuntimeError('attention residual output did not cover the full sequence')
        return self._x

    def output_buffer(self):
        '''Allocate the full attention output only after input projection.'''
        if self._output is None:
            self._output = self._x.new_empty(self.shape)
        return self._output


def attention_output_buffer(x):
    '''Return the disposable tensor that streamed attention may overwrite.'''
    if isinstance(x, NormalizedRows):
        return x.output_buffer()
    return x
