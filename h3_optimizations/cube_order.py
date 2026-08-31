"""MiniMax H3 cube-major target-video token ordering."""

from contextvars import ContextVar
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
CUBE_STATE_KEY = "h3_optimizations_cube_order_state"
SPECTRUM_RUNTIME_KEY = "spectrum_h3_runtime"
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


class CubeOrderState:
    """Per-patcher row-order state shared by _forward and FinalLayer.

    Native H3 calls use absolute packed-sequence offsets while forecast consumers
    such as Spectrum call FinalLayer on compact [audio | video] target tensors.
    Keep the exact native segment for ordinary calls and the latest matching
    video-row topology for compact bypass calls.
    """

    def __init__(self, cube_shape=CUBE_SHAPE):
        self.cube_shape = tuple(int(value) for value in cube_shape)
        self._entries = {}
        self._latest = None
        self._index_cache = {}
        self._active_entry = ContextVar(
            "h3_optimizations_cube_order_active_entry",
            default=None,
        )
        self._warned_state_residual = False

    def clear(self):
        self._entries.clear()
        self._latest = None
        self._index_cache.clear()
        self._warned_state_residual = False

    def record_cube(self, start, stop, forward, inverse, grid_shape):
        entry = {
            "mode": "cube",
            "start": int(start),
            "stop": int(stop),
            "rows": int(stop) - int(start),
            "forward": tuple(int(value) for value in forward),
            "inverse": tuple(int(value) for value in inverse),
            "grid_shape": tuple(int(value) for value in grid_shape),
            "cube_shape": self.cube_shape,
        }
        self._entries[(entry["start"], entry["stop"])] = entry
        self._latest = entry
        return entry

    def record_raster(self, start, stop, grid_shape):
        entry = {
            "mode": "raster",
            "start": int(start),
            "stop": int(stop),
            "rows": int(stop) - int(start),
            "forward": None,
            "inverse": None,
            "grid_shape": tuple(int(value) for value in grid_shape),
            "cube_shape": None,
        }
        self._entries[(entry["start"], entry["stop"])] = entry
        self._latest = entry
        return entry

    def begin_call(self, entry):
        return self._active_entry.set(entry)

    def end_call(self, token):
        self._active_entry.reset(token)

    def active_entry(self):
        return self._active_entry.get()

    def resolve(self, video_seg):
        first, last, _row = video_seg
        key = (int(first), int(last))
        entry = self._entries.get(key)
        if entry is not None:
            return entry
        rows = key[1] - key[0]
        latest = self._latest
        if latest is not None and int(latest["rows"]) == rows:
            return latest
        matches = [
            candidate
            for candidate in self._entries.values()
            if int(candidate["rows"]) == rows
        ]
        return matches[0] if len(matches) == 1 else None

    def _index(self, entry, kind, device):
        values = entry[kind]
        if values is None:
            return None
        key = (id(entry), kind, str(device))
        index = self._index_cache.get(key)
        if index is None or index.device != device:
            index = torch.tensor(values, dtype=torch.long, device=device)
            self._index_cache[key] = index
        return index

    def reorder_native_selector_for_bypass(self, selector, video_seg):
        entry = self.resolve(video_seg)
        if entry is None or entry["mode"] != "cube" or not torch.is_tensor(selector):
            return selector
        if self.active_entry() is entry:
            # Native _forward already reordered the denoise mask, therefore its
            # per-row timestep selector is already in cube order.
            return selector
        if selector.ndim == 0:
            return selector
        if int(selector.shape[0]) != int(entry["rows"]):
            raise H3CubeOrderPatchError(
                "FinalLayer video selector row count does not match cube-order topology"
            )
        return selector.index_select(
            0,
            self._index(entry, "forward", selector.device),
        )

    def restore_projected_video(self, rows, video_seg):
        entry = self.resolve(video_seg)
        if entry is None or entry["mode"] != "cube":
            return rows
        if not torch.is_tensor(rows) or rows.ndim < 2:
            raise H3CubeOrderPatchError(
                "FinalLayer video projection has an unexpected tensor contract"
            )
        if int(rows.shape[0]) != int(entry["rows"]):
            raise H3CubeOrderPatchError(
                "FinalLayer video projection row count does not match cube-order topology"
            )
        return rows.index_select(
            0,
            self._index(entry, "inverse", rows.device),
        )

    def warn_state_residual_fallback(self):
        if self._warned_state_residual:
            return
        self._warned_state_residual = True
        logging.warning(
            "%s using raster target-video order because Spectrum "
            "state_conditioned_residual mixes forecast residuals with a native "
            "raster input embedding before FinalLayer",
            LOG_PREFIX,
        )


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
        if "frame_count" in _PACKED_LAYOUT_PARAMETERS:
            kwargs["frame_count"] = payload.get("frame_count")
        layout = PackedLayout(*signature, **kwargs)
    return layout


def _spectrum_state_conditioned_residual(transformer_options):
    runtime = (transformer_options or {}).get(SPECTRUM_RUNTIME_KEY)
    if runtime is None:
        return False
    return bool(
        getattr(runtime, "state_conditioned_residual", False)
        or getattr(runtime, "active_state_conditioned_residual", False)
        or getattr(runtime, "active_state_residual_mode", False)
    )


def make_forward(model, original_forward, cube_shape=CUBE_SHAPE, state=None):
    patch_size = tuple(int(value) for value in model.patch_size)
    cube_shape = tuple(int(value) for value in cube_shape)
    standalone_restore = state is None
    state = CubeOrderState(cube_shape) if state is None else state

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
        video_start, video_stop, _kind = next(
            segment for segment in layout.segments if segment[2] == "video"
        )

        if _spectrum_state_conditioned_residual(transformer_options):
            payload["layout"] = layout
            entry = state.record_raster(video_start, video_stop, grid_shape)
            state.warn_state_residual_fallback()
            token = state.begin_call(entry)
            try:
                return original_forward(
                    x,
                    timestep,
                    context,
                    transformer_options,
                    minimax_payload=payload,
                    denoise_mask=denoise_mask,
                    audio_denoise_mask=audio_denoise_mask,
                    **kwargs,
                )
            finally:
                state.end_call(token)

        forward, inverse = tile_aligned_cube_major_indices(
            grid_shape, int(video_start), cube_shape
        )
        payload["layout"] = reorder_target_positions(
            layout, forward, grid_shape, cube_shape
        )
        entry = state.record_cube(
            video_start,
            video_stop,
            forward,
            inverse,
            grid_shape,
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

        token = state.begin_call(entry)
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
        finally:
            state.end_call(token)
        if not isinstance(output, (list, tuple)) or len(output) < 2:
            raise H3CubeOrderPatchError(
                "MiniMax H3 _forward returned an unexpected output contract"
            )

        result = list(output)
        if standalone_restore:
            inverse_index = torch.tensor(
                inverse,
                dtype=torch.long,
                device=result[0].device,
            )
            result[0] = reorder_video_patches(
                result[0],
                inverse_index,
                patch_size,
            )
        # Installed operation restores rows inside FinalLayer; both paths still
        # need the crop because cube ordering fed padded video into native _forward.
        result[0] = result[0][
            :, :, :original_shape[0], :original_shape[1], :original_shape[2]
        ]
        return tuple(result) if isinstance(output, tuple) else result

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


def _transformer_options(model_patcher):
    options = getattr(model_patcher, "model_options", {})
    options["transformer_options"] = options.get("transformer_options", {}).copy()
    return options["transformer_options"]


def get_state(model_patcher):
    state = getattr(model_patcher, "model_options", {}).get(
        "transformer_options",
        {},
    ).get(CUBE_STATE_KEY)
    return state if isinstance(state, CubeOrderState) else None


def _clear_order_only_final_layer(model_patcher):
    from .memory import final_layer as final_layer_patch

    current = getattr(model_patcher, "object_patches", {}).get(
        final_layer_patch.FINAL_LAYER_KEY
    )
    if (
        current is not None
        and getattr(current, final_layer_patch.OWNER_MARKER, False)
        and getattr(current, final_layer_patch.SIGNATURE_MARKER, None) is None
    ):
        final_layer_patch.clear(model_patcher)


def clear(model_patcher):
    state = get_state(model_patcher)
    if state is not None:
        state.clear()
    _transformer_options(model_patcher).pop(CUBE_STATE_KEY, None)
    _clear_order_only_final_layer(model_patcher)
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


def _install_final_layer_restore(model_patcher, state):
    from .memory import final_layer as final_layer_patch

    model = model_patcher.get_model_object("diffusion_model")
    if getattr(model, "final_layer", None) is None:
        # Keeps the low-level cube-order helper usable with deliberately minimal
        # test/dummy models. Real MiniMaxH3Model always owns FinalLayer.
        return
    current = getattr(model_patcher, "object_patches", {}).get(
        final_layer_patch.FINAL_LAYER_KEY
    )
    chunk_rows = None
    if current is not None and getattr(
        current,
        final_layer_patch.OWNER_MARKER,
        False,
    ):
        chunk_rows = getattr(current, final_layer_patch.SIGNATURE_MARKER, None)
    final_layer_patch.install(
        model_patcher,
        chunk_rows,
        cube_state=state,
        force_rebuild=True,
    )


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

    state = CubeOrderState(cube_shape)
    _transformer_options(model_patcher)[CUBE_STATE_KEY] = state
    model_patcher.add_object_patch(
        FORWARD_KEY,
        make_forward(model, original, cube_shape, state),
    )
    _install_final_layer_restore(model_patcher, state)
    logging.debug(
        "%s armed: cube=%s edge_tokens_only=true restore=FinalLayer",
        LOG_PREFIX,
        cube_shape,
    )
    return True


__all__ = [
    "CUBE_SHAPE",
    "CUBE_SHAPES",
    "CUBE_STATE_KEY",
    "CubeOrderState",
    "FORWARD_KEY",
    "H3CubeOrderPatchError",
    "ROUTER_TILE",
    "SPECTRUM_RUNTIME_KEY",
    "TOKEN_ORDER_SHAPES",
    "clear",
    "cube_major_indices",
    "get_state",
    "install",
    "make_forward",
    "reorder_target_positions",
    "reorder_video_patches",
    "tile_aligned_cube_major_indices",
]
