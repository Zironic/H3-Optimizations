'''H3 DiT block forward with bounded MLP activations.'''

import logging

import torch

import comfy.model_management
import comfy.ops

from .. import diagnostics
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
    HeldConvRotINT8Linear,
    HeldConvRotINT8MLP,
)
from ..native import (
    RMS_ADALN_BF16_STAGED,
    quantize_rms_adaln_int8_rowwise_convrot256,
    rms_adaln_int8_rowwise_convrot256_is_available,
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
    logging.debug(
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
        with diagnostics.stage('mlp_fc1_fc2'):
            out, path = held.fc1_fc2(h)
    elif mlp_path == 'fp8':
        with diagnostics.stage('mlp_fc1_fc2'):
            out, path = held.fc1_fc2(h, swiglu_eager)
    elif mlp_path == 'int8':
        with diagnostics.stage('mlp_fc1'):
            expanded = held.fc1(h)
        with diagnostics.stage('mlp_swiglu_fc2'):
            out, path = held.fc2_swiglu(expanded)
    elif mlp_path == 'held':
        with diagnostics.stage('mlp_fc1'):
            expanded = held.fc1(h)
        with diagnostics.stage('mlp_swiglu_fc2'):
            out, path = held.fc2_swiglu(
                expanded,
                native=config.native_swiglu,
            )
    else:
        with diagnostics.stage('mlp_fc1'):
            expanded = module_fc1(block.mlp, h)
        with diagnostics.stage('mlp_swiglu_fc2'):
            out, path = module_swiglu_fc2(
                block.mlp,
                expanded,
                native=config.native_swiglu,
            )
    return out, expanded, path


def _gated_path(mlp_path, config):
    '''Sessions whose fc2 can fold the gated residual (see fused_residual).'''
    return mlp_path == 'int8' or (mlp_path == 'held' and config.native_swiglu)


def _short_sequence_uses_held(config):
    '''Whether a sequence no longer than one chunk should still use the held path.

    Runtime ConvRot INT8 must: the upstream single-call MLP would run the
    unconverted floating weights (measured on a 4455-token BF16 H3 block: held
    INT8 with one slab per modulation segment 18.5 ms / 81 MiB, upstream BF16
    40.8 ms / 458 MiB). Checkpoint-native paths keep the upstream call: splitting
    per modulation segment re-reads the full MLP weights once per segment, which
    cost more than the fused residual saved (16.3 vs 14.3 ms on INT8 ConvRot).
    '''
    return config.runtime_convrot_int8


def _run_held_native_gated(held, h, x_rows, gate, selector, expanded=None):
    '''Held MLP plus gated residual for one scalar-modulation slab.

    Prefers Kitchen's fused INT8 residual kernel, which writes
    ``x_rows + gate * fc2(swiglu(fc1(h)))`` straight into ``x_rows``; otherwise
    runs the ordinary fc2 and ``addcmul_``. Both give bit-identical results.
    '''
    if expanded is None:
        with diagnostics.stage('mlp_fc1'):
            expanded = held.fc1(h)
    gate_row = _mod_row(gate, selector, x_rows.dtype)
    with diagnostics.stage('mlp_swiglu_fc2_gated'):
        if held.fc2_swiglu_residual_(expanded, x_rows, gate_row):
            return 'held_int8_fused_residual'
    del gate_row
    with diagnostics.stage('mlp_swiglu_fc2'):
        out, path = held.fc2_swiglu(expanded, native=True)
    del expanded
    with diagnostics.stage('final_mlp_gate'):
        _gate_add(x_rows, out, gate, selector)
    return path


def _quantize_norm_modulation(x, norm, shift, scale, selector):
    if torch.is_tensor(selector):
        return None
    norm_weight = comfy.ops.cast_to_input(norm.weight, x).contiguous()
    return quantize_rms_adaln_int8_rowwise_convrot256(
        x,
        norm_weight,
        shift[selector].contiguous(),
        scale[selector].contiguous(),
        norm.eps,
        mode=RMS_ADALN_BF16_STAGED,
    )


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

    def _forward(
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
        attention=None,
    ):
        if comfy.model_management.in_training:
            raise RuntimeError(
                'H3 Memory Optimization is inference-only; training requires '
                'the original block forward'
            )
        # ComfyUI block replace patches (Fun Control, Block Sparse Attention)
        # may substitute the attention callable for this block.
        attention = block.attn if attention is None else attention

        with diagnostics.stage('adaln_proj'):
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
        use_fused_norm_quantize = (
            config.fused_norm_quantize
            and not torch.compiler.is_compiling()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and int(x.shape[1]) % 256 == 0
            and HeldConvRotINT8Linear.prequantized_is_available()
            and rms_adaln_int8_rowwise_convrot256_is_available()
        )
        if use_fused_norm_quantize:
            h.enable_fused_convrot_quantization()
        if _attention_supports_lazy_norm(attention):
            h.enable_attention_residual(gate_msa)
            attention_options = dict(transformer_options or {})
            attention_options[NORM1_SOURCE_KEY] = h
            attn_out = attention(
                x,
                rope_freqs=rope_freqs,
                transformer_options=attention_options,
            )
            attention_residual_complete = h.attention_residual_complete
        else:
            h = h.materialize()
            attn_out = attention(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
            attention_residual_complete = False
        if not attention_residual_complete:
            with diagnostics.stage('attention_residual_gate'):
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

        if (
            int(config.chunk_rows) >= int(x.shape[0])
            and not _short_sequence_uses_held(config)
        ):
            with diagnostics.stage('norm2_modulation'):
                h = block.norm2(x)
                for start, stop, selector in segments:
                    _scale_shift(
                        h[start:stop],
                        shift_mlp,
                        scale_mlp,
                        selector,
                    )
            with diagnostics.stage('mlp_fc1_fc2'):
                out = block.mlp(h)
            with diagnostics.stage('final_mlp_gate'):
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
            fused_mlp_norm = use_fused_norm_quantize and (
                mlp_path == 'convrot'
                or (mlp_path == 'held' and held.supports_prequantized_fc1())
            )
            for chunk in chunks:
                if fused_mlp_norm:
                    with diagnostics.stage('norm2_modulation_quantize'):
                        carrier = _quantize_norm_modulation(
                            x[chunk.start:chunk.stop],
                            block.norm2,
                            shift_mlp,
                            scale_mlp,
                            chunk.mod_row,
                        )
                    if carrier is not None:
                        if mlp_path == 'convrot':
                            with diagnostics.stage('mlp_fc1_fc2'):
                                out, _path = held.fc1_fc2_prequantized(
                                    *carrier,
                                    x.dtype,
                                )
                            with diagnostics.stage('final_mlp_gate'):
                                _gate_add(
                                    x[chunk.start:chunk.stop],
                                    out,
                                    gate_mlp,
                                    chunk.mod_row,
                                )
                            del carrier, out
                            continue
                        with diagnostics.stage('mlp_fc1'):
                            expanded = held.fc1_prequantized(*carrier, x.dtype)
                        del carrier
                        if expanded is not None:
                            _run_held_native_gated(
                                held,
                                None,
                                x[chunk.start:chunk.stop],
                                gate_mlp,
                                chunk.mod_row,
                                expanded=expanded,
                            )
                            del expanded
                            continue
                with diagnostics.stage('norm2_modulation'):
                    h = block.norm2(x[chunk.start:chunk.stop])
                    _scale_shift(
                        h,
                        shift_mlp,
                        scale_mlp,
                        chunk.mod_row,
                    )

                if (
                    _gated_path(mlp_path, config)
                    and not torch.is_tensor(chunk.mod_row)
                ):
                    # The fused kernel takes one gate per output channel, so
                    # only scalar modulation segments qualify.
                    _run_held_native_gated(
                        held,
                        h,
                        x[chunk.start:chunk.stop],
                        gate_mlp,
                        chunk.mod_row,
                    )
                    del h
                    continue

                out, expanded, _path = _run_mlp(
                    block,
                    h,
                    held,
                    mlp_path,
                    config,
                )
                with diagnostics.stage('final_mlp_gate'):
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

    def forward(
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
        attention=None,
    ):
        with diagnostics.stage('block_total'):
            return _forward(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
                attention=attention,
            )

    forward._h3_optimizations_memory = True
    forward._h3_optimizations_memory_signature = config.signature
    forward._h3_optimizations_memory_layer = int(layer_index)
    forward._h3_optimizations_memory_original = original_forward
    return forward
