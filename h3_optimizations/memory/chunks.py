"""Segment-aware token slab planning for MiniMax H3."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenChunk:
    start: int
    stop: int
    mod_row: object

    @property
    def rows(self):
        return self.stop - self.start


def _normalize_selector(selector, span, index, mod_rows):
    if torch.is_tensor(selector):
        if selector.ndim != 1:
            raise ValueError(
                "segment %d modulation selector must be rank 1, got rank %d"
                % (index, selector.ndim)
            )
        if selector.numel() != span:
            raise ValueError(
                "segment %d modulation selector has %d rows, expected %d"
                % (index, selector.numel(), span)
            )
        if selector.dtype != torch.long:
            raise TypeError(
                "segment %d modulation selector must use torch.long integer dtype, got %s"
                % (index, selector.dtype)
            )
        # Do not reduce CUDA selectors here to validate values. This function is
        # called inside every DiT block and a min/max .item() would synchronize
        # the device repeatedly. PyTorch indexing will still reject an invalid
        # row if an upstream producer violates the selector contract.
        return selector

    row = int(selector)
    if row < 0 or (mod_rows is not None and row >= int(mod_rows)):
        raise ValueError(
            "segment %d modulation row %d is outside [0, %s)"
            % (index, row, "?" if mod_rows is None else int(mod_rows))
        )
    return row


def validate_mod_segments(segments, seq_len, mod_rows=None):
    """Return normalized ``(start, stop, selector)`` tuples.

    H3's modulation segments must cover the packed sequence contiguously. The
    selector is either one scalar modulation-row index for the whole segment or
    a rank-1 LongTensor containing one modulation-row index per token.
    """
    seq_len = int(seq_len)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    normalized = []
    expected = 0
    for index, segment in enumerate(segments):
        if len(segment) != 3:
            raise ValueError(
                "segment %d must contain (start, stop, modulation selector)"
                % index
            )
        start, stop = (int(v) for v in segment[:2])
        if start != expected:
            relation = "gap" if start > expected else "overlap"
            raise ValueError(
                "segment %d starts at %d, expected %d (%s)"
                % (index, start, expected, relation)
            )
        if stop <= start:
            raise ValueError(
                "segment %d has non-positive span [%d, %d)" % (index, start, stop)
            )
        if stop > seq_len:
            raise ValueError(
                "segment %d stops at %d past sequence length %d"
                % (index, stop, seq_len)
            )
        selector = _normalize_selector(
            segment[2],
            stop - start,
            index,
            mod_rows,
        )
        normalized.append((start, stop, selector))
        expected = stop

    if expected != seq_len:
        raise ValueError(
            "segments cover [0, %d), expected [0, %d)" % (expected, seq_len)
        )
    if seq_len and not normalized:
        raise ValueError("non-empty sequence requires at least one modulation segment")
    return tuple(normalized)


def iter_mod_chunks(segments, seq_len, max_rows, alignment=1, mod_rows=None):
    """Yield the largest aligned chunks that stay inside modulation segments."""
    max_rows = int(max_rows)
    alignment = int(alignment)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    alignment = min(alignment, max_rows)

    normalized = validate_mod_segments(segments, seq_len, mod_rows=mod_rows)
    for segment_start, segment_stop, selector in normalized:
        start = segment_start
        while start < segment_stop:
            remaining = segment_stop - start
            size = min(remaining, max_rows)
            if size < remaining and alignment > 1:
                size = (size // alignment) * alignment
                if size <= 0:
                    raise ValueError(
                        "alignment %d leaves no rows inside max_rows %d"
                        % (alignment, max_rows)
                    )
            stop = start + size
            chunk_selector = selector
            if torch.is_tensor(selector):
                offset = start - segment_start
                chunk_selector = selector[offset:offset + size]
            yield TokenChunk(start, stop, chunk_selector)
            start = stop


def iter_modulation_chunks(segments, max_rows):
    """Bound only per-token selector gathers; scalar segments stay unsplit.

    Unlike ``iter_mod_chunks`` this never splits a segment that carries one
    scalar modulation row, so the whole-segment fast path is preserved.
    """
    max_rows = int(max_rows)
    for start, stop, selector in segments:
        if not torch.is_tensor(selector):
            yield start, stop, selector
            continue
        offset = 0
        while start + offset < stop:
            size = min(max_rows, stop - start - offset)
            yield (
                start + offset,
                start + offset + size,
                selector[offset:offset + size],
            )
            offset += size
