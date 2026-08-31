'''Bounded and representation-safe MiniMax H3 FinalLayer execution.'''

import inspect
import logging

import torch

import comfy.ops
from comfy.ldm.minimax.model import time_shift_sigma

from ..cube_order import CubeOrderState
from ..model import get_minimax_h3_model


FINAL_LAYER_KEY = 'diffusion_model.final_layer.forward'
OWNER_MARKER = '_h3_optimizations_final_layer'
SIGNATURE_MARKER = '_h3_optimizations_final_layer_signature'
CONFIG_MARKER = '_h3_optimizations_final_layer_config'
ORIGINAL_MARKER = '_h3_optimizations_final_layer_original'


class H3FinalLayerPatchError(RuntimeError):
    pass


def _selector(value, start, stop):
    return value if value.ndim == 1 else value[start:stop]


def _pdd_head(head, value, n, start, stop, flow_shift):
    grid = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    dt = (
        1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)
    ).diff()[start:stop]
    blend = (dt / dt.sum()).to(value)
    with comfy.ops.CastBiasWeightContext(
        head, value, offloadable=True
    ) as (weight, bias):
        rows = weight.reshape(n, -1, weight.shape[1])
        bias_rows = bias.reshape(n, -1)
        first = max(start, 1)
        return torch.nn.functional.linear(
            value,
            rows[0]
            + torch.einsum(
                'n,noi->oi', blend[first - start:], rows[first:stop]
            ),
            bias_rows[0]
            + torch.einsum(
                'n,no->o', blend[first - start:], bias_rows[first:stop]
            ),
        )


def chunked_final_layer(
    layer,
    x,
    t_emb,
    video_seg,
    audio_seg,
    chunk_rows,
    sigma=None,
    sample_sigmas=None,
    shifts=None,
):
    shift, scale = layer.adaln_proj(t_emb)

    n = layer.video_out.weight.shape[0] // layer.video_out.out_features
    pdd = None
    if n > 1:
        if sample_sigmas is None:
            raise ValueError(
                "MiniMax H3 PDD heads need the sampler's sigma schedule"
            )
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        start, stop = (
            round(float(1.0 - time_shift_sigma(s, shifts[0], 1.0)) * n)
            for s in (sigma, sigma_next)
        )
        start = min(start, n - 1)
        stop = max(stop, start + 1)
        pdd = (n, start, stop)

    def project(segment, output, flow_shift=None):
        first, last, row = segment
        selected_shift = shift[row]
        selected_scale = scale[row]
        pieces = []
        for start in range(first, last, int(chunk_rows)):
            stop = min(start + int(chunk_rows), last)
            local_start = start - first
            local_stop = stop - first
            value = (
                layer.norm(x[start:stop])
                * (1.0 + _selector(selected_scale, local_start, local_stop))
                + _selector(selected_shift, local_start, local_stop)
            ).to(torch.float32)
            pieces.append(
                output(value)
                if pdd is None
                else _pdd_head(output, value, *pdd, flow_shift)
            )
        if not pieces:
            value = (
                layer.norm(x[first:last])
                * (1.0 + _selector(selected_scale, 0, 0))
                + _selector(selected_shift, 0, 0)
            ).to(torch.float32)
            return (
                output(value)
                if pdd is None
                else _pdd_head(output, value, *pdd, flow_shift)
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    video_shift = shifts[0] if pdd is not None else None
    audio_shift = shifts[1] if pdd is not None else None
    return (
        project(video_seg, layer.video_out, video_shift),
        project(audio_seg, layer.audio_out, audio_shift),
    )


def _chunk_count(rows, chunk_rows):
    if rows <= 0:
        return 0
    return (int(rows) + int(chunk_rows) - 1) // int(chunk_rows)


def _supports_current_contract(forward):
    try:
        parameters = inspect.signature(forward).parameters.values()
    except (TypeError, ValueError):
        return True
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True
    names = {parameter.name for parameter in parameters}
    return bool({'sigma', 'sample_sigmas', 'shifts'} & names)


def _call_original(
    original_forward,
    x,
    t_emb,
    video_seg,
    audio_seg,
    sigma,
    sample_sigmas,
    shifts,
):
    if _supports_current_contract(original_forward):
        return original_forward(
            x,
            t_emb,
            video_seg,
            audio_seg,
            sigma,
            sample_sigmas,
            shifts,
        )
    return original_forward(x, t_emb, video_seg, audio_seg)


def _ordered_video_segment(video_seg, cube_state):
    if cube_state is None:
        return video_seg
    first, last, row = video_seg
    ordered_row = cube_state.reorder_native_selector_for_bypass(row, video_seg)
    return (first, last, ordered_row)


def _restore_output_order(result, video_seg, cube_state):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise H3FinalLayerPatchError(
            'MiniMax H3 FinalLayer returned an unexpected output contract'
        )
    if cube_state is None:
        return result
    video = cube_state.restore_projected_video(result[0], video_seg)
    return (video, result[1]) if isinstance(result, tuple) else [video, result[1]]


def make_forward(
    layer,
    chunk_rows=None,
    *,
    original_forward=None,
    cube_state=None,
):
    chunk_rows = None if chunk_rows is None else int(chunk_rows)
    original_forward = layer.forward if original_forward is None else original_forward
    if cube_state is not None and not isinstance(cube_state, CubeOrderState):
        raise TypeError('cube_state must be CubeOrderState or None')
    announced = []

    def forward(
        x,
        t_emb,
        video_seg,
        audio_seg,
        sigma=None,
        sample_sigmas=None,
        shifts=None,
    ):
        effective_video_seg = _ordered_video_segment(video_seg, cube_state)
        if chunk_rows is None:
            result = _call_original(
                original_forward,
                x,
                t_emb,
                effective_video_seg,
                audio_seg,
                sigma,
                sample_sigmas,
                shifts,
            )
        else:
            if not announced:
                announced.append(True)
                video_rows = int(video_seg[1]) - int(video_seg[0])
                audio_rows = int(audio_seg[1]) - int(audio_seg[0])
                logging.debug(
                    '[H3 Optimizations] chunked FinalLayer ran: %d rows, '
                    'video %d in %d chunk(s), audio %d in %d chunk(s), '
                    'chunk_rows=%d',
                    int(x.shape[0]),
                    video_rows,
                    _chunk_count(video_rows, chunk_rows),
                    audio_rows,
                    _chunk_count(audio_rows, chunk_rows),
                    chunk_rows,
                )
            result = chunked_final_layer(
                layer,
                x,
                t_emb,
                effective_video_seg,
                audio_seg,
                chunk_rows,
                sigma,
                sample_sigmas,
                shifts,
            )
        return _restore_output_order(result, video_seg, cube_state)

    setattr(forward, OWNER_MARKER, True)
    setattr(forward, SIGNATURE_MARKER, chunk_rows)
    setattr(
        forward,
        CONFIG_MARKER,
        (
            chunk_rows,
            None if cube_state is None else tuple(cube_state.cube_shape),
        ),
    )
    setattr(forward, ORIGINAL_MARKER, original_forward)
    return forward


def _same_callable(left, right):
    if left is right:
        return True
    return (
        getattr(left, '__self__', None) is getattr(right, '__self__', None)
        and getattr(left, '__func__', None) is getattr(right, '__func__', None)
        and getattr(left, '__func__', None) is not None
    )


def _options(model_patcher):
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    return options


def clear(model_patcher):
    patches = getattr(model_patcher, 'object_patches', {})
    current = patches.get(FINAL_LAYER_KEY)
    if current is None or not getattr(current, OWNER_MARKER, False):
        return False
    original = getattr(current, ORIGINAL_MARKER, None)
    if original is None:
        raise H3FinalLayerPatchError(
            'installed H3 FinalLayer patch has no recoverable original'
        )
    model = get_minimax_h3_model(model_patcher)
    layer = None if model is None else getattr(model, 'final_layer', None)
    native = None if layer is None else layer.forward
    if native is not None and _same_callable(original, native):
        patches.pop(FINAL_LAYER_KEY)
    else:
        patches[FINAL_LAYER_KEY] = original
    return True


def install(
    model_patcher,
    chunk_rows=None,
    *,
    cube_state=None,
    force_rebuild=False,
):
    '''Patch FinalLayer for chunking, cube restore, or both.'''

    if chunk_rows is not None:
        chunk_rows = int(chunk_rows)
        if chunk_rows <= 0:
            raise ValueError('chunk_rows must be positive')
    if cube_state is not None and not isinstance(cube_state, CubeOrderState):
        raise TypeError('cube_state must be CubeOrderState or None')

    model = get_minimax_h3_model(model_patcher)
    if model is None:
        raise H3FinalLayerPatchError(
            'H3 FinalLayer optimization can only patch MiniMaxH3Model'
        )
    layer = getattr(model, 'final_layer', None)
    if layer is None:
        raise H3FinalLayerPatchError('MiniMax H3 has no final layer')

    patches = getattr(model_patcher, 'object_patches', {})
    current = patches.get(FINAL_LAYER_KEY)
    current_owned = bool(current is not None and getattr(current, OWNER_MARKER, False))
    if current_owned:
        original = getattr(current, ORIGINAL_MARKER, None)
        if original is None:
            raise H3FinalLayerPatchError(
                'installed H3 FinalLayer patch has no recoverable original'
            )
        foreign_base = not _same_callable(original, layer.forward)
    elif current is not None:
        original = current
        foreign_base = True
    else:
        original = layer.forward
        foreign_base = False

    effective_chunk_rows = chunk_rows
    if foreign_base and chunk_rows is not None:
        # Preserve the foreign computation exactly. Cube restoration is a pure
        # row permutation around its result and can still compose safely.
        effective_chunk_rows = None

    options = _options(model_patcher)
    options['h3_optimizations_preserved_final_layer_patch'] = bool(foreign_base)

    if effective_chunk_rows is None and cube_state is None:
        if current_owned:
            return clear(model_patcher)
        if foreign_base and chunk_rows is not None:
            logging.debug(
                '[H3 Optimizations] preserved foreign %s; FinalLayer chunking '
                'is disabled',
                FINAL_LAYER_KEY,
            )
        return False

    desired = (
        effective_chunk_rows,
        None if cube_state is None else tuple(cube_state.cube_shape),
    )
    if (
        current_owned
        and getattr(current, CONFIG_MARKER, None) == desired
        and not force_rebuild
    ):
        return False

    model_patcher.add_object_patch(
        FINAL_LAYER_KEY,
        make_forward(
            layer,
            effective_chunk_rows,
            original_forward=original,
            cube_state=cube_state,
        ),
    )
    logging.debug(
        '[H3 Optimizations] patched FinalLayer: chunk_rows=%s cube_restore=%s foreign_base=%s',
        effective_chunk_rows,
        None if cube_state is None else tuple(cube_state.cube_shape),
        foreign_base,
    )
    return True
