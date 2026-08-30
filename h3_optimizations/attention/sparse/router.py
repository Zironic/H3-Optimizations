'''Fixed-density Sparse Sage tile routing for MiniMax H3.'''

from dataclasses import dataclass
import math

import torch

from .config import DENSITY_FIXED
Q_TILE = 128
KV_TILE = 64


class SparseRouterError(RuntimeError):
    pass


@torch.library.custom_op(
    'h3_optimizations::sort_selected_indices',
    mutates_args=(),
    device_types='cuda',
)
def sort_selected_indices_op(indices: torch.Tensor) -> torch.Tensor:
    return indices.sort(dim=-1).values


@sort_selected_indices_op.register_fake
def _sort_selected_indices_fake(indices):
    return torch.empty_like(indices)


def sort_selected_indices(indices):
    if torch.compiler.is_compiling() or indices.is_cuda:
        return sort_selected_indices_op(indices)
    return indices.sort(dim=-1).values


@dataclass(frozen=True)
class SparseTileGeometry:
    signature: tuple
    sequence: int
    q_tiles: int
    kv_tiles: int
    pure_video_q_start: int
    pure_video_kv_start: int

    @property
    def pure_video_q_tiles(self):
        return self.q_tiles - self.pure_video_q_start

    @property
    def pure_video_kv_tiles(self):
        return self.kv_tiles - self.pure_video_kv_start


@dataclass(frozen=True)
class SparseMaskMetadata:
    requested_video_budget: float
    actual_video_tile_density: float
    full_mask_density: float
    dense_q_tiles: int
    sparse_q_tiles: int
    q_tiles: int
    kv_tiles: int
    pure_video_q_tiles: int
    pure_video_kv_tiles: int
    retained_video_kv_tiles: int
    density_mode: str = DENSITY_FIXED
    def as_dict(self):
        return dict(vars(self))


class SparseTileRouter:
    '''Build a per-head route using the resolved Sparse Sage geometry.'''

    def __init__(
        self,
        config=None,
        *,
        spec=None,
        q_tile=None,
        kv_tile=None,
        score_chunk_tiles=None,
    ):
        self.config = config
        self.spec = spec
        # None keeps the whole [B, H, q_tiles, kv_tiles] score slab live at
        # once. A chunk count scores that many query tiles at a time and drops
        # each slab before the next, which is selection-equivalent because a
        # query tile's top-K depends on no other query tile -- but see
        # `_select_indices`: equivalent is not automatically bit-identical,
        # because a matmul may reduce in a different order at a different M.
        if score_chunk_tiles is not None:
            score_chunk_tiles = int(score_chunk_tiles)
            if score_chunk_tiles <= 0:
                raise ValueError('score_chunk_tiles must be positive')
        self.score_chunk_tiles = score_chunk_tiles
        self.q_tile = int(
            q_tile if q_tile is not None else getattr(spec, 'q_tile', Q_TILE)
        )
        self.kv_tile = int(
            kv_tile if kv_tile is not None else getattr(spec, 'kv_tile', KV_TILE)
        )
        if self.q_tile <= 0 or self.kv_tile <= 0:
            raise ValueError('Sparse Sage tile sizes must be positive')
        self._geometry_cache = {}

    @staticmethod
    def _layout_signature(layout):
        return (
            int(layout.seq_len),
            tuple(int(x) for x in layout.video_range),
            tuple(
                (int(start), int(stop), str(kind))
                for start, stop, kind in layout.segments
            ),
            tuple(int(x) for x in layout.video_shape),
            int(layout.audio_t),
        )

    def geometry(self, layout):
        signature = self._layout_signature(layout)
        cached = self._geometry_cache.get(signature)
        if cached is not None:
            return cached
        sequence = int(layout.seq_len)
        video_start, video_stop = (int(x) for x in layout.video_range)
        if sequence <= 0:
            raise SparseRouterError('packed sequence is empty')
        if not (0 <= video_start < video_stop == sequence):
            raise SparseRouterError(
                'Sparse Sage requires target video to be the final packed '
                'segment; got video=%s sequence=%d'
                % (tuple(layout.video_range), sequence)
            )
        geometry = SparseTileGeometry(
            signature=signature,
            sequence=sequence,
            q_tiles=(sequence + self.q_tile - 1) // self.q_tile,
            kv_tiles=(sequence + self.kv_tile - 1) // self.kv_tile,
            pure_video_q_start=(
                video_start + self.q_tile - 1
            ) // self.q_tile,
            pure_video_kv_start=(
                video_start + self.kv_tile - 1
            ) // self.kv_tile,
        )
        if not geometry.pure_video_q_tiles:
            raise SparseRouterError(
                'packed layout has no pure-video query tiles'
            )
        if not geometry.pure_video_kv_tiles:
            raise SparseRouterError(
                'packed layout has no pure-video KV tiles'
            )
        self._geometry_cache[signature] = geometry
        return geometry

    @staticmethod
    def _mean_pool(x, block):
        sequence = x.shape[-2]
        full = sequence // block
        remainder = sequence % block
        pieces = []
        if full:
            pieces.append(
                x[..., :full * block, :]
                .reshape(*x.shape[:-2], full, block, x.shape[-1])
                .mean(dim=-2)
            )
        if remainder:
            pieces.append(
                x[..., full * block:, :].mean(dim=-2, keepdim=True)
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)

    @staticmethod
    def _retained(video_budget, geometry):
        video_budget = float(video_budget)
        if video_budget <= 0.0:
            return 1
        if video_budget >= 1.0:
            return geometry.pure_video_kv_tiles
        return min(
            geometry.pure_video_kv_tiles,
            max(
                1,
                int(
                    math.ceil(
                        video_budget * geometry.pure_video_kv_tiles
                    )
                ),
            ),
        )

    def _metadata(self, geometry, video_budget, retained):
        pure_q = geometry.pure_video_q_tiles
        pure_kv = geometry.pure_video_kv_tiles
        sparse_q = pure_q if retained < pure_kv else 0
        non_video_kv = geometry.kv_tiles - pure_kv
        true_blocks = (
            (geometry.q_tiles - pure_q) * geometry.kv_tiles
            + pure_q * (non_video_kv + retained)
        )
        return SparseMaskMetadata(
            requested_video_budget=float(video_budget),
            actual_video_tile_density=float(retained) / pure_kv,
            full_mask_density=float(true_blocks)
            / (geometry.q_tiles * geometry.kv_tiles),
            dense_q_tiles=geometry.q_tiles - sparse_q,
            sparse_q_tiles=sparse_q,
            q_tiles=geometry.q_tiles,
            kv_tiles=geometry.kv_tiles,
            pure_video_q_tiles=pure_q,
            pure_video_kv_tiles=pure_kv,
            retained_video_kv_tiles=retained,
        )

    @staticmethod
    def _dense_lut(source, geometry, metadata):
        batch, heads = source.shape[:2]
        dense = torch.arange(
            geometry.kv_tiles,
            device=source.device,
            dtype=torch.int32,
        )
        delta = torch.cat((dense[:1], dense[1:] - dense[:-1]))
        lut = delta.view(1, 1, 1, -1).expand(
            batch,
            heads,
            geometry.q_tiles,
            -1,
        ).clone()
        valid = torch.full(
            (batch, heads, geometry.q_tiles),
            geometry.kv_tiles,
            dtype=torch.int32,
            device=source.device,
        )
        return lut.contiguous(), valid.contiguous(), metadata

    def build_lut(self, q, k, layout, video_budget):
        if q.ndim != 4 or k.ndim != 4:
            raise SparseRouterError('tile router expects HND rank-4 Q/K')
        if q.shape != k.shape:
            raise SparseRouterError(
                'tile router requires equal Q/K shapes; got %s %s'
                % (tuple(q.shape), tuple(k.shape))
            )
        if not math.isfinite(float(video_budget)):
            raise SparseRouterError('video_budget must be finite')
        geometry = self.geometry(layout)
        if q.shape[-2] != geometry.sequence:
            raise SparseRouterError(
                'layout sequence %d does not match Q/K sequence %d'
                % (geometry.sequence, q.shape[-2])
            )
        retained = self._retained(video_budget, geometry)
        metadata = self._metadata(geometry, video_budget, retained)
        if retained == geometry.pure_video_kv_tiles:
            return self._dense_lut(q, geometry, metadata)
        return self._build_lut_from_summaries(
            self._mean_pool(q, self.q_tile),
            self._mean_pool(k, self.kv_tile),
            geometry,
            video_budget,
        )

    def build_lut_from_summaries(
        self,
        q_summary,
        k_summary,
        layout,
        video_budget,
    ):
        if q_summary.ndim != 4 or k_summary.ndim != 4:
            raise SparseRouterError(
                'tile router summaries must be rank-4 HND tensors'
            )
        if q_summary.shape[:2] != k_summary.shape[:2]:
            raise SparseRouterError(
                'Q/K router summary batch and head shapes differ'
            )
        if q_summary.shape[-1] != k_summary.shape[-1]:
            raise SparseRouterError('Q/K router summary dimensions differ')
        if q_summary.device != k_summary.device:
            raise SparseRouterError('Q/K router summary devices differ')
        if not math.isfinite(float(video_budget)):
            raise SparseRouterError('video_budget must be finite')
        geometry = self.geometry(layout)
        expected_q = (geometry.q_tiles, q_summary.shape[-1])
        expected_k = (geometry.kv_tiles, k_summary.shape[-1])
        if tuple(q_summary.shape[-2:]) != expected_q:
            raise SparseRouterError(
                'Q router summary shape %s does not match %s'
                % (tuple(q_summary.shape[-2:]), expected_q)
            )
        if tuple(k_summary.shape[-2:]) != expected_k:
            raise SparseRouterError(
                'K router summary shape %s does not match %s'
                % (tuple(k_summary.shape[-2:]), expected_k)
            )
        retained = self._retained(video_budget, geometry)
        return self._build_lut_from_summaries(
            q_summary,
            k_summary,
            geometry,
            video_budget,
        )

    def _select_indices(self, q_video, k_video, retained):
        '''Top-K video KV tiles per query tile, optionally chunk by chunk.

        The unchunked branch is kept verbatim rather than expressed as a
        one-chunk special case, so enabling chunking cannot quietly change the
        shipped route: whatever cuBLAS does for the full M is what production
        still gets.
        '''
        keys = k_video.transpose(-1, -2)
        if self.score_chunk_tiles is None:
            return torch.topk(torch.matmul(q_video, keys), retained, dim=-1).indices

        query_tiles = q_video.shape[-2]
        pieces = []
        for start in range(0, query_tiles, self.score_chunk_tiles):
            stop = min(start + self.score_chunk_tiles, query_tiles)
            scores = torch.matmul(q_video[..., start:stop, :], keys)
            pieces.append(torch.topk(scores, retained, dim=-1).indices)
            del scores
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)

    @staticmethod
    def _pack_rows(indices, geometry, dense, dense_delta):
        batch, heads = indices.shape[:2]
        absolute = indices.to(torch.int32) + geometry.pure_video_kv_start
        selected = sort_selected_indices(absolute)
        context_count = geometry.pure_video_kv_start
        context = dense_delta[:context_count].view(1, 1, 1, -1).expand(
            batch,
            heads,
            indices.shape[-2],
            -1,
        )
        previous = dense[context_count - 1] if context_count else 0
        selected_delta = torch.cat(
            (
                selected[..., :1] - previous,
                selected[..., 1:] - selected[..., :-1],
            ),
            dim=-1,
        )
        if not context_count:
            return selected_delta
        return torch.cat((context, selected_delta), dim=-1)

    def _build_lut_from_summaries(
        self,
        q_means,
        k_means,
        geometry,
        video_budget,
    ):
        batch, heads = q_means.shape[:2]
        retained = self._retained(video_budget, geometry)
        metadata = self._metadata(geometry, video_budget, retained)
        if retained == geometry.pure_video_kv_tiles:
            return self._dense_lut(q_means, geometry, metadata)
        indices = self._select_indices(
            q_means[..., geometry.pure_video_q_start:, :],
            k_means[..., geometry.pure_video_kv_start:, :],
            retained,
        )
        return self._build_lut_from_indices(
            q_means,
            geometry,
            metadata,
            indices,
        )

    def _build_lut_from_indices(
        self,
        source,
        geometry,
        metadata,
        indices,
    ):
        batch, heads = source.shape[:2]
        dense = torch.arange(
            geometry.kv_tiles,
            device=source.device,
            dtype=torch.int32,
        )
        dense_delta = torch.cat((dense[:1], dense[1:] - dense[:-1]))
        lut = dense_delta.view(1, 1, 1, -1).expand(
            batch,
            heads,
            geometry.q_tiles,
            -1,
        ).clone()
        valid = torch.full(
            (batch, heads, geometry.q_tiles),
            geometry.kv_tiles,
            dtype=torch.int32,
            device=source.device,
        )
        sparse_rows = self._pack_rows(
            indices,
            geometry,
            dense,
            dense_delta,
        )
        lut[
            ...,
            geometry.pure_video_q_start:,
            :sparse_rows.shape[-1],
        ].copy_(sparse_rows)
        valid[..., geometry.pure_video_q_start:] = (
            geometry.pure_video_kv_start + metadata.retained_video_kv_tiles
        )
        return lut.contiguous(), valid.contiguous(), metadata
