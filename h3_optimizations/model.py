"""Exact MiniMax H3 model identity and block discovery."""

from __future__ import annotations


def get_minimax_h3_model(model_patcher, model_type=None):
    """Return the diffusion model only when it is an actual MiniMax H3 model."""

    try:
        diffusion_model = model_patcher.get_model_object("diffusion_model")
    except Exception:
        return None

    if model_type is None:
        try:
            from comfy.ldm.minimax.model import MiniMaxH3Model
        except ImportError:
            return None
        model_type = MiniMaxH3Model

    return diffusion_model if isinstance(diffusion_model, model_type) else None


def unpatched_model_object(model_patcher, name, live):
    """Return the value ``name`` holds with no object patches applied.

    Comfy leaves a loaded patcher's object patches applied to the shared
    model, so while an earlier clone stays loaded ``live`` may be that clone's
    wrapper. Capturing it as an original nests the old wrapper inside the new
    one. Clones share ``object_patches_backup``, which holds the stock value
    of every key the loaded patcher replaced.
    """

    backup = getattr(model_patcher, "object_patches_backup", None)
    if isinstance(backup, dict) and name in backup:
        return backup[name]
    return live


def is_minimax_h3(model_patcher, model_type=None):
    return get_minimax_h3_model(model_patcher, model_type=model_type) is not None


def is_minimax_h3_image(model_patcher):
    """Whether the diffusion model is MiniMax H3 Image (ComfyUI 0.45+), which has no audio stream."""
    try:
        from comfy.ldm.minimax.model import MiniMaxH3ImageModel
    except ImportError:
        return False
    return get_minimax_h3_model(model_patcher, model_type=MiniMaxH3ImageModel) is not None


def get_h3_blocks(model_patcher, model_type=None):
    model = get_minimax_h3_model(model_patcher, model_type=model_type)
    if model is None:
        return ()
    try:
        blocks = tuple(model_patcher.get_model_object("diffusion_model.blocks"))
    except Exception:
        blocks = tuple(getattr(model, "blocks", ()))
    return blocks
