'''Direct full-width absolute routes for the BF16 FROST backend.'''

from dataclasses import dataclass
import math

import torch

from .router import SparseRouterError, sort_selected_indices
from .triton_route import (
    TritonRouteError,
    build_compact_absolute_route,
)


@dataclass(frozen=True)
class FrostRoutePlan:
    geometry: object
    retained: int
    metadata: object
    batch: int
    heads: int


def build_full_absolute_route(
    router,
    q,
    k,
    layout,
    video_budget,
):
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise TritonRouteError('FROST route expects equal rank-4 HND Q/K')
    try:
        q_summary = router._mean_pool(q, router.q_tile)
        k_summary = router._mean_pool(k, router.kv_tile)
    except Exception as error:
        raise TritonRouteError('FROST route mean pooling failed: %s' % error) from error
    return build_full_absolute_route_from_summaries(
        router,
        q_summary,
        k_summary,
        layout,
        video_budget,
    )


def build_full_absolute_route_from_summaries(
    router,
    q_summary,
    k_summary,
    layout,
    video_budget,
):
    compact, metadata = build_compact_absolute_route(
        router,
        q_summary,
        k_summary,
        layout,
        video_budget,
    )
    geometry = router.geometry(layout)
    batch, heads = q_summary.shape[:2]
    dense = torch.arange(
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q_summary.device,
    )
    route = dense.view(1, 1, 1, -1).expand(
        batch,
        heads,
        geometry.q_tiles,
        -1,
    ).clone()
    counts = torch.full(
        (batch, heads, geometry.q_tiles),
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q_summary.device,
    )
    if compact.numel():
        start = geometry.pure_video_q_start
        route[..., start:, :compact.shape[-1]].copy_(compact)
        counts[..., start:] = compact.shape[-1]
    return route.contiguous(), counts.contiguous(), metadata


def prepare_full_absolute_route_chunks(
    router,
    k_summary,
    layout,
    video_budget,
):
    if k_summary.ndim != 4:
        raise TritonRouteError('FROST K route summary must be rank-4 HND')
    if not math.isfinite(float(video_budget)):
        raise TritonRouteError('video_budget must be finite')
    try:
        geometry = router.geometry(layout)
    except SparseRouterError as error:
        raise TritonRouteError(str(error)) from error
    if tuple(k_summary.shape[-2:]) != (
        geometry.kv_tiles,
        k_summary.shape[-1],
    ):
        raise TritonRouteError('FROST K route summary does not match layout')
    retained = router._retained(video_budget, geometry)
    return FrostRoutePlan(
        geometry=geometry,
        retained=retained,
        metadata=router._metadata(geometry, video_budget, retained),
        batch=int(k_summary.shape[0]),
        heads=int(k_summary.shape[1]),
    )


def build_full_absolute_route_chunk(
    router,
    q_summary,
    k_summary,
    route_plan,
    *,
    q_tile_start,
):
    if q_summary.ndim != 4 or k_summary.ndim != 4:
        raise TritonRouteError('FROST route summaries must be rank-4 HND')
    if q_summary.shape[:2] != k_summary.shape[:2]:
        raise TritonRouteError('FROST route Q/K batch and head shapes differ')
    if q_summary.shape[-1] != k_summary.shape[-1]:
        raise TritonRouteError('FROST route Q/K summary dimensions differ')
    if q_summary.device != k_summary.device:
        raise TritonRouteError('FROST route Q/K summary devices differ')

    geometry = route_plan.geometry
    q_tile_start = int(q_tile_start)
    q_tiles = int(q_summary.shape[-2])
    q_tile_end = q_tile_start + q_tiles
    if (
        int(q_summary.shape[0]) != route_plan.batch
        or int(q_summary.shape[1]) != route_plan.heads
        or q_tiles <= 0
        or not 0 <= q_tile_start < q_tile_end <= geometry.q_tiles
        or int(k_summary.shape[-2]) != geometry.kv_tiles
    ):
        raise TritonRouteError('FROST route summary chunk does not match plan')

    dense = torch.arange(
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q_summary.device,
    )
    route = dense.view(1, 1, 1, -1).expand(
        route_plan.batch,
        route_plan.heads,
        q_tiles,
        -1,
    ).clone()
    counts = torch.full(
        (route_plan.batch, route_plan.heads, q_tiles),
        geometry.kv_tiles,
        dtype=torch.int32,
        device=q_summary.device,
    )

    sparse_start = max(q_tile_start, geometry.pure_video_q_start)
    if (
        route_plan.retained < geometry.pure_video_kv_tiles
        and sparse_start < q_tile_end
    ):
        local_start = sparse_start - q_tile_start
        indices = router._select_indices(
            q_summary[..., local_start:, :],
            k_summary[
                ..., geometry.pure_video_kv_start:, :
            ],
            route_plan.retained,
        )
        selected = sort_selected_indices(
            indices.to(torch.int32) + geometry.pure_video_kv_start
        )
        context_count = int(geometry.pure_video_kv_start)
        if context_count:
            context = dense[:context_count].view(1, 1, 1, -1).expand(
                route_plan.batch,
                route_plan.heads,
                q_tile_end - sparse_start,
                -1,
            )
            selected = torch.cat((context, selected), dim=-1)
        route[..., local_start:, :selected.shape[-1]].copy_(selected)
        counts[..., local_start:] = selected.shape[-1]
    return route.contiguous(), counts.contiguous()
