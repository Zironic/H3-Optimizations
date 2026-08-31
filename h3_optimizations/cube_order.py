"""MiniMax H3 cube-major target-video token ordering."""

from copy import copy
from functools import lru_cache
import inspect
import logging

import torch
import torch.nn.functional as F

from comfy.ldm import common_dit
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, patchify_video, unpatchify_video
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
_PACKED_LAYOUT_PARAMETERS = inspect.signature(PackedLayout).parameters


class H3CubeOrderPatchError(RuntimeError):
    pass


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
        kwargs = {
            "keyframes": payload.get("keyframes"),
            "refs": payload.get("refs"),
        }
        # ComfyUI v0.33 requires frame_count to resolve a last-frame FL2VA
        # anchor. v0.34 removed that constructor parameter when it generalized
        # PackedLayout to arbitrary keyframe positions, so only forward metadata
        # that the installed Comfy constructor actually accepts.
        if "frame_count" in _PACKED_LAYOUT_PARAMETERS:
            kwargs["frame_count"] = payload.get("frame_count")
        layout = PackedLayout(*signature, **kwargs)
    return layout


def make_forward(model, original_forward, cube_shape=CUBE_SHAPE):
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
        payload["layout"] = reorder_target_positions(
            layout, forward, grid_shape, cube_shape
        )

        forward_index = torch.tensor(forward, dtype=torch.long, device=video.device)
        inverse_index = torch.tensor(inverse, dtype=torch.long, device=video.device)
        ordered_x = list(x)
        ordered_x[0] = reorder_video_patches(padded_video, forward_index, patch_size)

        ordered_mask = denoise_mask
        if denoise_mask is not None:
            ordered_mask = reorder_video_patches(
                pad_mask(denoise_mask, padded_video.shape),
                forward_index,
                patch_size,
            )

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

        restored = reorder_video_patches(output[0], inverse_index, patch_size)
        restored = restored[
            :, :, :original_shape[0], :original_shape[1], :original_shape[2]
        ]
        result = list(output)
        result[0] = restored
        return tuple(result) if isinstance(output, tuple) else result

    cube_order_forward._h3_cube_order = True
    cube_order_forward._h3_cube_order_shape = cube_shape
    cube_order_forward._h3_cube_order_original = original_forward
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
    model = model_patcher.get_model_object("diffusion_model")
    if _same_callable(original, model._forward):
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

    model_patcher.add_object_patch(
        FORWARD_KEY,
        make_forward(model, original, cube_shape),
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
