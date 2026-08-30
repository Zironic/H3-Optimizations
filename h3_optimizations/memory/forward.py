'''H3 DiT block forward with bounded MLP activations.'''

import logging

import torch

import comfy.model_management

from .chunks import (
    iter_mod_chunks,
    iter_modulation_chunks,
    validate_mod_segments,
)
from .linear import (
    ConvRotTwoSliceMLP,
    HeldMLP,
    UnsafeHeldWeights,
    bind_convrot_mlp,
    module_fc1,
    module_swiglu_fc2,
    swiglu_eager,
)
from ..normalized_rows import NORM1_SOURCE_KEY, NormalizedRows
from ..qkv.fp8 import FP8BindingError, HeldFP8MLP
from ..qkv.int8 import (
    ConvRotINT8BindingError,
    HeldConvRotINT8MLP,
)

LOG_PREFIX = '[H3 Optimizations]'
_MLP_FALLBACK_LOGGED = set()


def _mod_row(values, selector, dtype):
    if torch.is_tensor(selector) and selector.device != values.device:
        selector = selector.to(device=values.device)
    return values[selector].to(dtype)


def _scale_shift(h, shift, scale, selector):
    scale_rows = _mod_row(scale, selector, h.dtype)
    h.mul_(1.0 + scale_rows)
    del scale_rows
    shift_rows = _mod_row(shift, selector, h.dtype)
    h.add_(shift_rows)
    del shift_rows
    return h


def _gate_add(x, other, gate, selector):
    gate_rows = _mod_row(gate, selector, x.dtype)
    x.addcmul_(other, gate_rows)
    del gate_rows
    return x


def _attention_supports_lazy_norm(attention):
    forward = getattr(attention, 'forward', None)
    if forward is None:
        return False
    function = getattr(forward, '__func__', forward)
    # Identity, not truthiness: mocks and __getattr__ proxies can synthesize a
    # truthy value for arbitrary attributes. A false positive here would pass
    # the raw unnormalized residual to a foreign attention forward.
    return getattr(
        function,
        '_h3_optimizations_lazy_norm_source',
        False,
    ) is True


def _open_generic_held(block, sample, config):
    if not config.prefer_held_weights:
        return None, None
    held = HeldMLP(block.mlp, sample, force_bf16=config.bf16_swiglu)
    try:
        held.__enter__()
        return held, None
    except UnsafeHeldWeights as exc:
        if config.bf16_swiglu:
            raise
        return None, str(exc)
    except (RuntimeError, TypeError, ValueError) as exc:
        if config.strict:
            raise
        return None, '%s: %s' % (type(exc).__name__, exc)


def _open_fp8(block, sample, config):
    allow_float_conversion = not hasattr(block.mlp.fc1.weight, '_layout_cls')
    held = HeldFP8MLP(
        block.mlp,
        sample,
        allow_float_conversion=allow_float_conversion,
    )
    try:
        held.__enter__()
        return held, None
    except (FP8BindingError, RuntimeError, TypeError, ValueError) as exc:
        if config.strict:
            raise
        held.release()
        return None, '%s: %s' % (type(exc).__name__, exc)


def _open_runtime_convrot_int8(block, sample, config):
    held = HeldConvRotINT8MLP(
        block.mlp,
        sample,
        allow_float_conversion=True,
    )
    try:
        held.__enter__()
        return held, None
    except (
        ConvRotINT8BindingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if config.strict:
            raise
        held.release()
        return None, '%s: %s' % (type(exc).__name__, exc)


def _open_mlp(block, sample, config):
    if config.runtime_convrot_int8:
        held, error = _open_runtime_convrot_int8(block, sample, config)
        return held, 'int8' if held is not None else 'module', error
    if config.fp8:
        held, error = _open_fp8(block, sample, config)
        return held, 'fp8' if held is not None else 'module', error

    if not config.convrot_2slice:
        held, error = _open_generic_held(block, sample, config)
        return held, 'held' if held is not None else 'module', error

    held = ConvRotTwoSliceMLP(block.mlp, sample)
    try:
        held.__enter__()
        return held, 'convrot', None
    except (RuntimeError, TypeError, ValueError) as exc:
        if config.strict:
            raise
        held, generic_error = _open_generic_held(block, sample, config)
        detail = '%s: %s' % (type(exc).__name__, exc)
        if generic_error is not None:
            detail += '; held fallback unavailable: %s' % generic_error
        return held, 'held' if held is not None else 'module', detail


def _log_mlp_fallback(layer_index, reason):
    reason = str(reason)
    if reason in _MLP_FALLBACK_LOGGED:
        return
    logging.info(
        '%s preferred MLP optimization is unavailable for this model path '
        '(first seen in block %d); using a compatible fallback instead. '
        'Output is unaffected, but VRAM use or speed may be slightly worse. '
        'Detail: %s',
        LOG_PREFIX,
        layer_index,
        reason,
    )
    _MLP_FALLBACK_LOGGED.add(reason)


def _run_mlp(block, h, held, mlp_path, config):
    expanded = None
    if mlp_path == 'convrot':
        out, path = held.fc1_fc2(h)
    elif mlp_path in ('fp8', 'int8'):
        out, path = held.fc1_fc2(h, swiglu_eager)
    elif mlp_path == 'held':
        expanded = held.fc1(h)
        out, path = held.fc2_swiglu(
            expanded,
            native=config.native_swiglu,
        )
    else:
        expanded = module_fc1(block.mlp, h)
        out, path = module_swiglu_fc2(
            block.mlp,
            expanded,
            native=config.native_swiglu,
        )
    return out, expanded, path


def _run_fc1_swiglu(block, h, held, mlp_path):
    """Diagnostic-only SwiGLU activation for a small row subset."""
    if mlp_path == 'convrot':
        return held.fc1_swiglu(h)
    if mlp_path in ('fp8', 'int8'):
        return swiglu_eager(held.fc1_binding.linear(h))
    if mlp_path == 'held':
        return swiglu_eager(held.fc1(h))
    return swiglu_eager(module_fc1(block.mlp, h))


def _run_fc2_activation(block, activated, held, mlp_path):
    """Diagnostic-only fc2 applied to an already-activated SwiGLU vector."""
    if mlp_path == 'convrot':
        return held.fc2_activation(activated)
    if mlp_path in ('fp8', 'int8'):
        return held.fc2_binding.linear(activated)
    if mlp_path == 'held':
        return held.fc2_weight.linear(activated)
    return block.mlp.fc2(activated)


def make_forward(block, layer_index, config, original_forward=None):
    '''Build an unbound replacement for one MiniMax H3 DiT block.'''

    original_forward = original_forward or block.forward
    if config.convrot_2slice and isinstance(block.mlp, torch.nn.Module):
        bind_convrot_mlp(block.mlp)

    def forward(
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
    ):
        if comfy.model_management.in_training:
            raise RuntimeError(
                'H3 Memory Optimization is inference-only; training requires '
                'the original block forward'
            )

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = block.adaln_proj(t_emb)
        segments = validate_mod_segments(
            mod_segments,
            x.shape[0],
            mod_rows=shift_msa.shape[0],
        )

        # Keep lazy Norm1 below the attention abstraction boundary. The block
        # always calls attention with a real Tensor; only a package-owned
        # projected-QKV forward receives the private row source. Unknown or
        # standard attention forwards get the ordinary materialized Norm1 path.
        h = NormalizedRows(
            x,
            block.norm1,
            segments,
            shift_msa,
            scale_msa,
            _scale_shift,
        )
        if _attention_supports_lazy_norm(block.attn):
            attention_options = dict(transformer_options or {})
            attention_options[NORM1_SOURCE_KEY] = h
            attn_out = block.attn(
                x,
                rope_freqs=rope_freqs,
                transformer_options=attention_options,
            )
        else:
            h = h.materialize()
            attn_out = block.attn(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
        for start, stop, selector in iter_modulation_chunks(
            segments,
            config.chunk_rows,
        ):
            _gate_add(
                x[start:stop],
                attn_out[start:stop],
                gate_msa,
                selector,
            )
        del h, attn_out

        if int(config.chunk_rows) >= int(x.shape[0]):
            h = block.norm2(x)
            for start, stop, selector in segments:
                _scale_shift(
                    h[start:stop],
                    shift_mlp,
                    scale_mlp,
                    selector,
                )
            out = block.mlp(h)
            for start, stop, selector in segments:
                _gate_add(
                    x[start:stop],
                    out[start:stop],
                    gate_mlp,
                    selector,
                )
            del h, out
            return x

        chunks = tuple(
            iter_mod_chunks(
                segments,
                x.shape[0],
                config.chunk_rows,
                alignment=config.alignment,
                mod_rows=shift_mlp.shape[0],
            )
        )

        held, mlp_path, held_error = _open_mlp(block, x[:1], config)
        if held_error is not None:
            _log_mlp_fallback(layer_index, held_error)

        try:
            for chunk in chunks:
                h = block.norm2(x[chunk.start:chunk.stop])
                _scale_shift(
                    h,
                    shift_mlp,
                    scale_mlp,
                    chunk.mod_row,
                )

                out, expanded, _path = _run_mlp(
                    block,
                    h,
                    held,
                    mlp_path,
                    config,
                )
                _gate_add(
                    x[chunk.start:chunk.stop],
                    out,
                    gate_mlp,
                    chunk.mod_row,
                )
                del h, expanded, out
        finally:
            if held is not None:
                held.__exit__(None, None, None)
        return x

    forward._h3_optimizations_memory = True
    forward._h3_optimizations_memory_signature = config.signature
    forward._h3_optimizations_memory_layer = int(layer_index)
    forward._h3_optimizations_memory_original = original_forward
    return forward
