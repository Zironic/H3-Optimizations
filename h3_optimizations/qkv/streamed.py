"""Source-aware held QKV bindings for streamed attention consumers."""

from __future__ import annotations

from .bf16 import HeldBF16QKV
from .formats import describe_linear
from .fp8 import HeldFP8QKV
from .int8 import HeldConvRotINT8QKV
from .w4a8 import HeldW4A8QKV


PROJECTION_NATIVE = "native"
PROJECTION_FORCE_BF16 = "force_bf16"
PROJECTION_FORCE_FP8 = "force_fp8"
PROJECTION_FORCE_INT8 = "force_int8"
PROJECTION_MODES = frozenset(
    (
        PROJECTION_NATIVE,
        PROJECTION_FORCE_BF16,
        PROJECTION_FORCE_FP8,
        PROJECTION_FORCE_INT8,
    )
)


class StreamedQKVBindingError(RuntimeError):
    pass


def _has_runtime_weight_patch(linear):
    """Whether Comfy may materialize an effective weight for this linear.

    LoRAs, adapters and similar ModelPatcher features are represented through
    weight/bias functions.  Their effective runtime value is authoritative over
    the checkpoint storage layout.  This intentionally does not identify the
    extension that installed the patch.
    """
    return bool(getattr(linear, "weight_function", ())) or bool(
        getattr(linear, "bias_function", ())
    )


def _native_binding(module, sample):
    linear = module.qkv_proj
    actual = describe_linear(linear)
    if _has_runtime_weight_patch(linear):
        # Preserve semantics: native/auto mode follows the effective value
        # Comfy computes after all runtime weight patches.  Bounded/streamed
        # projection keeps activation VRAM low even when that value is BF16.
        return HeldBF16QKV(module, sample, allow_quantized_source=True)
    if actual.convrot_int8_256:
        return HeldConvRotINT8QKV(module, sample)
    if actual.w4a8:
        return HeldW4A8QKV(module, sample)
    if actual.fp8:
        return HeldFP8QKV(module, sample)
    dtype = str(getattr(actual, "logical_dtype", "")).lower()
    if actual.plain_float and ("bfloat16" in dtype or "bf16" in dtype):
        return HeldBF16QKV(module, sample)
    raise StreamedQKVBindingError(
        "native streamed QKV does not support %s" % actual.label
    )


def create_held_qkv(module, sample, projection_mode=PROJECTION_NATIVE):
    """Create, but do not enter, the requested execution-scoped QKV binding.

    Explicit force modes remain authoritative even when runtime weight patches
    exist: force BF16 materializes the effective value in BF16, while force FP8
    or INT8 requantizes the effective value.  Native mode preserves the effective
    precision chosen by Comfy.
    """
    if projection_mode not in PROJECTION_MODES:
        raise ValueError("unknown streamed QKV projection mode %r" % projection_mode)
    if projection_mode == PROJECTION_FORCE_BF16:
        return HeldBF16QKV(module, sample, allow_quantized_source=True)
    if projection_mode == PROJECTION_FORCE_FP8:
        return HeldFP8QKV(module, sample, allow_float_conversion=True)
    if projection_mode == PROJECTION_FORCE_INT8:
        return HeldConvRotINT8QKV(module, sample, allow_float_conversion=True)
    return _native_binding(module, sample)


def project_q_hnd(held, x, rope_freqs, start, end):
    """Project bounded Q, using a true Q-only path when the source exposes one."""
    project_q = getattr(held, "project_q_hnd", None)
    if callable(project_q):
        return project_q(x, rope_freqs, start, end)
    q, k, v = held.project_hnd(x, rope_freqs, start, end)
    del k, v
    return q


def project_kv_hnd(held, x, rope_freqs, start, end):
    """Project K/V without Q when the source binding exposes a split path."""
    project_kv = getattr(held, "project_kv_hnd", None)
    if callable(project_kv):
        return project_kv(x, rope_freqs, start, end)
    q, k, v = held.project_hnd(x, rope_freqs, start, end)
    del q
    return k, v


__all__ = [
    "PROJECTION_FORCE_BF16",
    "PROJECTION_FORCE_FP8",
    "PROJECTION_FORCE_INT8",
    "PROJECTION_MODES",
    "PROJECTION_NATIVE",
    "StreamedQKVBindingError",
    "create_held_qkv",
    "project_kv_hnd",
    "project_q_hnd",
]
