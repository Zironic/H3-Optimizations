"""MiniMax H3 cube-major target-video token ordering."""

from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
import logging

import torch
import torch.nn.functional as F

from comfy.ldm import common_dit
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, patchify_video, unpatchify_video
from .memory import final_layer
from .model import unpatched_model_object
from .plan import (
    VIDEO_TOKEN_ORDER_1X16X4,
    VIDEO_TOKEN_ORDER_1X8X8,
    VIDEO_TOKEN_ORDER_4X4X4,
    VIDEO_TOKEN_ORDER_RASTER,
)


CUBE_SHAPES = ((1, 8, 8), (1, 16, 4), (4, 4, 4))
CUBE_SHAPE = (1, 8, 8)
ROUTER_TILE = 64
FORWARD_KEY = "diffusion_model._forward"
LOG_PREFIX = "[H3 cube order]"
TOKEN_ORDER_SHAPES = {
    VIDEO_TOKEN_ORDER_1X8X8: (1, 8, 8),
    VIDEO_TOKEN_ORDER_1X16X4: (1, 16, 4),
    VIDEO_TOKEN_ORDER_4X4X4: (4, 4, 4),
    VIDEO_TOKEN_ORDER_RASTER: None,
}

_ACTIVE_TOPOLOGY = ContextVar("h3_cube_order_active", default=None)
_LATEST_TOPOLOGY = ContextVar("h3_cube_order_latest", default=None)


class H3CubeOrderPatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class CubeOrderTopology:
    grid_shape: tuple
    forward: tuple
    inverse: tuple

    def order_video_selector(self, video_seg):
        """Cube-order a raster per-token FinalLayer selector.

        Native ``_forward`` already hands FinalLayer ordered selectors; only a
        caller that bypassed it, such as a forecast, still holds raster ones.
        """
        start, stop, row = video_seg
        if not torch.is_tensor(row) or row.ndim == 0:
            return video_seg
        if int(row.shape[0]) != int(stop) - int(start):
            raise H3CubeOrderPatchError(
                "per-token FinalLayer selector does not match the target-video rows"
            )
        index = torch.tensor(self.forward, dtype=torch.long, device=row.device)
        return (start, stop, row.index_select(0, index))

    def restore_video_output(self, output):
        """Return FinalLayer's projected video rows to raster order."""
        if not isinstance(output, (list, tuple)) or len(output) != 2:
            raise H3CubeOrderPatchError(
                "MiniMax H3 FinalLayer returned an unexpected output contract"
            )
        video = output[0]
        if not torch.is_tensor(video) or video.ndim < 1:
            raise H3CubeOrderPatchError(
                "MiniMax H3 FinalLayer returned an invalid video projection"
            )
        if int(video.shape[0]) != len(self.inverse):
            raise H3CubeOrderPatchError(
                "FinalLayer video projection does not match the cube-order topology"
            )
        index = torch.tensor(self.inverse, dtype=torch.long, device=video.device)
        restored = video.index_select(0, index)
        if isinstance(output, tuple):
            return (restored, output[1])
        return [restored, output[1]]


class CubeOrderState:
    """One execution-context topology shared by _forward and FinalLayer."""

    def enter(self, topology):
        return _ACTIVE_TOPOLOGY.set((self, topology))

    def leave(self, token, topology, succeeded):
        _ACTIVE_TOPOLOGY.reset(token)
        if succeeded:
            _LATEST_TOPOLOGY.set((self, topology))
        else:
            latest = _LATEST_TOPOLOGY.get()
            if latest is not None and latest[0] is self:
                _LATEST_TOPOLOGY.set(None)

    def resolve(self, video_rows):
        entry = _ACTIVE_TOPOLOGY.get()
        if entry is not None and entry[0] is not self:
            raise H3CubeOrderPatchError(
                "FinalLayer call belongs to another active cube-order state"
            )
        active = entry is not None
        if entry is None:
            entry = _LATEST_TOPOLOGY.get()
        if entry is None or entry[0] is not self:
            raise H3CubeOrderPatchError(
                "FinalLayer bypass has no completed cube-order topology"
            )
        topology = entry[1]
        if len(topology.forward) != int(video_rows):
            raise H3CubeOrderPatchError(
                "FinalLayer target-video rows do not match the cube-order topology"
            )
        return topology, active


def _cube_groups(grid_shape, cube_shape=CUBE_SHAPE):
    t, h, w = (int(value) for value in grid_shape)
    ct, ch, cw = (int(value) for value in cube_shape)
    if min(t, h, w, ct, ch, cw) <= 0:
        raise ValueError("grid and cube dimensions must be positive")

    groups = []
    for t0 in range(0, t, ct):
        for h0 in range(0, h, ch):
            for w0 in range(0, w, cw):
                group = []
                for ti in range(t0, min(t0 + ct, t)):
                    for hi in range(h0, min(h0 + ch, h)):
                        for wi in range(w0, min(w0 + cw, w)):
                            group.append((ti * h + hi) * w + wi)
                groups.append(tuple(group))
    return tuple(groups)


def _with_inverse(forward):
    forward = tuple(forward)
    inverse = [0] * len(forward)
    for cube_row, raster_row in enumerate(forward):
        inverse[raster_row] = cube_row
    return forward, tuple(inverse)


@lru_cache(maxsize=2)
def cube_major_indices(grid_shape, cube_shape=CUBE_SHAPE):
    """Return cube-major-to-raster and raster-to-cube row mappings."""
    forward = [row for group in _cube_groups(grid_shape, cube_shape) for row in group]
    return _with_inverse(forward)


@lru_cache(maxsize=2)
def tile_aligned_cube_major_indices(
    grid_shape,
    sequence_offset,
    cube_shape=CUBE_SHAPE,
    router_tile=ROUTER_TILE,
):
    """Put complete cubes on global router-tile boundaries without fake rows."""
    router_tile = int(router_tile)
    if router_tile <= 0:
        raise ValueError("router tile size must be positive")
    groups = _cube_groups(grid_shape, cube_shape)
    full = [group for group in groups if len(group) == router_tile]
    edge_rows = [row for group in groups if len(group) != router_tile for row in group]

    prefix_rows = (-int(sequence_offset)) % router_tile
    if len(edge_rows) < prefix_rows:
        if not full:
            raise ValueError("video grid cannot satisfy the router alignment prefix")
        edge_rows.extend(full.pop(0))

    prefix = edge_rows[:prefix_rows]
    suffix = edge_rows[prefix_rows:]
    forward = prefix + [row for group in full for row in group] + suffix
    return _with_inverse(forward)


def patch_grid_shape(video, patch_size):
    return tuple(
        int(video.shape[axis + 2] // patch_size[axis])
        for axis in range(3)
    )


def reorder_video_patches(video, indices, patch_size):
    """Reorder complete patch cells while preserving values inside each patch."""
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("MiniMax H3 cube ordering expects [1, C, T, H, W]")
    grid_t, grid_h, grid_w = patch_grid_shape(video, patch_size)
    if len(indices) != grid_t * grid_h * grid_w:
        raise ValueError("cube-order row count does not match the video patch grid")
    rows = patchify_video(video, patch_size)
    if torch.is_tensor(indices):
        index = indices.to(device=rows.device, dtype=torch.long)
    else:
        index = torch.tensor(indices, dtype=torch.long, device=rows.device)
    rows = rows.index_select(0, index)
    return unpatchify_video(
        rows,
        grid_t,
        grid_h,
        grid_w,
        int(video.shape[1]),
        patch_size,
    )


def pad_mask(mask, video_shape):
    if mask.ndim != 5:
        raise ValueError("MiniMax H3 denoise mask must be [B, C, T, H, W]")
    target = tuple(int(value) for value in video_shape[-3:])
    current = tuple(int(value) for value in mask.shape[-3:])
    if any(source > dest for source, dest in zip(current, target)):
        raise ValueError("MiniMax H3 denoise mask exceeds the padded video shape")
    padding = (
        0, target[2] - current[2],
        0, target[1] - current[1],
        0, target[0] - current[0],
    )
    return F.pad(mask, padding, mode="replicate")


def reorder_target_positions(layout, indices, grid_shape, cube_shape=CUBE_SHAPE):
    """Copy a packed layout and reorder only its final target-video positions."""
    video_segments = [segment for segment in layout.segments if segment[2] == "video"]
    if not video_segments:
        raise ValueError("MiniMax H3 packed layout has no target-video segment")
    start, stop, _kind = video_segments[-1]
    if stop - start != len(indices):
        raise ValueError("target-video layout rows do not match the cube-order patch grid")

    ordered = copy(layout)
    position_ids = layout.position_ids.clone()
    index = torch.tensor(indices, dtype=torch.long, device=position_ids.device)
    position_ids[start:stop] = position_ids[start:stop].index_select(0, index)
    ordered.position_ids = position_ids
    ordered.h3_cube_order = {
        "cube_shape": tuple(int(value) for value in cube_shape),
        "grid_shape": tuple(int(value) for value in grid_shape),
        "video_range": (int(start), int(stop)),
        "edge_tokens_only": True,
        "router_tile": ROUTER_TILE,
        "alignment_prefix_rows": (-int(start)) % ROUTER_TILE,
    }
    return ordered


def _matching_layout(layout, signature):
    return layout is not None and tuple(layout.signature) == tuple(signature)


def _layout(payload, context, padded_video, audio):
    signature = (
        int(context.shape[1]),
        int(padded_video.shape[2]),
        int(padded_video.shape[3]),
        int(padded_video.shape[4]),
        int(audio.shape[-1]),
    )
    layout = payload.get("layout")
    if not _matching_layout(layout, signature):
        layout = PackedLayout(
            *signature,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )
    return layout


def make_forward(model, original_forward, cube_shape=CUBE_SHAPE, state=None):
    patch_size = tuple(int(value) for value in model.patch_size)
    cube_shape = tuple(int(value) for value in cube_shape)

    def cube_order_forward(
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        denoise_mask=None,
        audio_denoise_mask=None,
        **kwargs,
    ):
        video, audio = x[0], x[1]
        original_shape = tuple(int(value) for value in video.shape[-3:])
        padded_video = common_dit.pad_to_patch_size(video, patch_size)
        grid_shape = patch_grid_shape(padded_video, patch_size)

        payload = dict(minimax_payload or {})
        layout = _layout(payload, context, padded_video, audio)
        video_start = next(
            start for start, _stop, kind in layout.segments if kind == "video"
        )
        forward, inverse = tile_aligned_cube_major_indices(
            grid_shape, int(video_start), cube_shape
        )
        topology = CubeOrderTopology(grid_shape, forward, inverse)
        payload["layout"] = reorder_target_positions(
            layout, forward, grid_shape, cube_shape
        )

        forward_index = torch.tensor(forward, dtype=torch.long, device=video.device)
        ordered_x = list(x)
        ordered_x[0] = reorder_video_patches(padded_video, forward_index, patch_size)

        ordered_mask = denoise_mask
        if denoise_mask is not None:
            ordered_mask = reorder_video_patches(
                pad_mask(denoise_mask, padded_video.shape),
                forward_index,
                patch_size,
            )

        token = state.enter(topology) if state is not None else None
        succeeded = False
        try:
            output = original_forward(
                ordered_x,
                timestep,
                context,
                transformer_options,
                minimax_payload=payload,
                denoise_mask=ordered_mask,
                audio_denoise_mask=audio_denoise_mask,
                **kwargs,
            )
            if not isinstance(output, (list, tuple)) or len(output) < 2:
                raise H3CubeOrderPatchError(
                    "MiniMax H3 _forward returned an unexpected output contract"
                )

            restored = output[0]
            if state is None:
                inverse_index = torch.tensor(
                    inverse, dtype=torch.long, device=video.device
                )
                restored = reorder_video_patches(restored, inverse_index, patch_size)
            restored = restored[
                :, :, :original_shape[0], :original_shape[1], :original_shape[2]
            ]
            result = list(output)
            result[0] = restored
            succeeded = True
            return tuple(result) if isinstance(output, tuple) else result
        finally:
            if state is not None:
                state.leave(token, topology, succeeded)

    cube_order_forward._h3_cube_order = True
    cube_order_forward._h3_cube_order_shape = cube_shape
    cube_order_forward._h3_cube_order_original = original_forward
    cube_order_forward._h3_cube_order_state = state
    return cube_order_forward


def _same_callable(left, right):
    if left is right:
        return True
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
        and getattr(left, "__func__", None) is not None
    )


def clear(model_patcher):
    patches = getattr(model_patcher, "object_patches", {})
    current = patches.get(FORWARD_KEY)
    if current is None or not getattr(current, "_h3_cube_order", False):
        return False
    original = getattr(current, "_h3_cube_order_original", None)
    if original is None:
        raise H3CubeOrderPatchError(
            "installed H3 cube-order patch has no recoverable original"
        )
    state = getattr(current, "_h3_cube_order_state", None)
    if state is not None:
        final_layer.clear_cube_state(model_patcher, state)
    model = model_patcher.get_model_object("diffusion_model")
    stock = unpatched_model_object(model_patcher, FORWARD_KEY, model._forward)
    if _same_callable(original, stock):
        patches.pop(FORWARD_KEY)
    else:
        patches[FORWARD_KEY] = original
    return True


def install(model_patcher, cube_shape=CUBE_SHAPE):
    cube_shape = tuple(int(value) for value in cube_shape)
    if (
        len(cube_shape) != 3
        or min(cube_shape) <= 0
        or cube_shape[0] * cube_shape[1] * cube_shape[2] != ROUTER_TILE
    ):
        raise H3CubeOrderPatchError(
            "cube-order geometry must have positive dimensions and contain 64 tokens"
        )
    try:
        model = model_patcher.get_model_object("diffusion_model")
    except Exception as exc:
        raise H3CubeOrderPatchError(
            "This model has no diffusion_model; cube ordering only applies to MiniMax H3"
        ) from exc
    if not isinstance(model, MiniMaxH3Model):
        raise H3CubeOrderPatchError(
            "cube ordering can only patch MiniMaxH3Model; got %s"
            % type(model).__name__
        )
    if tuple(model.patch_size) != (1, 2, 2):
        raise H3CubeOrderPatchError(
            "expected MiniMax H3 video patch size (1, 2, 2), got %s"
            % (tuple(model.patch_size),)
        )

    original = model_patcher.get_model_object(FORWARD_KEY)
    if getattr(original, "_h3_cube_order", False):
        if getattr(original, "_h3_cube_order_shape", None) == cube_shape:
            return False
        raise H3CubeOrderPatchError("another H3 cube-order configuration is installed")

    existing_final = getattr(model_patcher, "object_patches", {}).get(
        final_layer.FINAL_LAYER_KEY
    )
    chunk_rows = (
        getattr(existing_final, final_layer.SIGNATURE_MARKER, None)
        if getattr(existing_final, final_layer.OWNER_MARKER, False)
        else None
    )
    state = CubeOrderState()
    final_layer.install(
        model_patcher,
        chunk_rows,
        cube_state=state,
    )
    installed_final = getattr(model_patcher, "object_patches", {}).get(
        final_layer.FINAL_LAYER_KEY
    )
    if getattr(installed_final, final_layer.CUBE_STATE_MARKER, None) is not state:
        logging.warning(
            "%s using raster target-video order because a foreign FinalLayer "
            "patch is active",
            LOG_PREFIX,
        )
        return False

    model_patcher.add_object_patch(
        FORWARD_KEY,
        make_forward(model, original, cube_shape, state),
    )
    logging.debug(
        "%s armed: cube=%s edge_tokens_only=true",
        LOG_PREFIX,
        cube_shape,
    )
    return True


__all__ = [
    "CUBE_SHAPE",
    "CUBE_SHAPES",
    "CubeOrderState",
    "CubeOrderTopology",
    "FORWARD_KEY",
    "H3CubeOrderPatchError",
    "ROUTER_TILE",
    "TOKEN_ORDER_SHAPES",
    "clear",
    "cube_major_indices",
    "install",
    "make_forward",
    "reorder_target_positions",
    "reorder_video_patches",
    "tile_aligned_cube_major_indices",
]
