'''Bounded MiniMax H3 FinalLayer execution.'''

import inspect
import logging

import torch

import comfy.ops
from comfy.ldm.minimax.model import time_shift_sigma

from ..model import get_minimax_h3_model, unpatched_model_object


FINAL_LAYER_KEY = 'diffusion_model.final_layer.forward'
OWNER_MARKER = '_h3_optimizations_final_layer'
SIGNATURE_MARKER = '_h3_optimizations_final_layer_signature'
ORIGINAL_MARKER = '_h3_optimizations_final_layer_original'
CUBE_STATE_MARKER = '_h3_optimizations_cube_order_state'
_CURRENT_FORWARD_PARAMETERS = frozenset({'sigma', 'sample_sigmas', 'shifts'})


class H3FinalLayerPatchError(RuntimeError):
    pass


def _pdd_weights(head, value, n, start, stop, flow_shift):
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
        return (
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
    precomputed_scale = scale.shape[0] <= int(chunk_rows)
    if precomputed_scale:
        scale = 1.0 + scale

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
        per_token = torch.is_tensor(row) and row.ndim > 0
        if not per_token:
            selected_shift = shift[row]
            selected_scale = scale[row]
            if not precomputed_scale:
                selected_scale = 1.0 + selected_scale
        result = None
        weight = bias = None
        starts = range(first, last, int(chunk_rows))
        if last <= first:
            starts = (first,)
        for start in starts:
            stop = min(start + int(chunk_rows), last)
            if per_token:
                local_row = row[start - first:stop - first]
                selected_scale = scale[local_row]
                if not precomputed_scale:
                    selected_scale = 1.0 + selected_scale
            value = layer.norm(x[start:stop]) * selected_scale
            if per_token:
                del selected_scale
                selected_shift = shift[local_row]
            value = (value + selected_shift).to(torch.float32)
            if per_token:
                del selected_shift
            if pdd is None:
                projected = output(value)
            else:
                if weight is None:
                    weight, bias = _pdd_weights(output, value, *pdd, flow_shift)
                projected = torch.nn.functional.linear(value, weight, bias)
            if result is None:
                if stop == last:
                    return projected
                result = projected.new_empty((last - first, *projected.shape[1:]))
            result[start - first:stop - first].copy_(projected)
            del value, projected
        return result

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


def _accepts_current_contract(forward):
    parameters = inspect.signature(forward).parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True
    return _CURRENT_FORWARD_PARAMETERS.issubset(
        parameter.name for parameter in parameters
    )


def _call_original(
    original_forward,
    current_contract,
    x,
    t_emb,
    video_seg,
    audio_seg,
    sigma,
    sample_sigmas,
    shifts,
):
    if current_contract:
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


def _ordered_selector(video_seg, topology):
    start, stop, row = video_seg
    rows = int(stop) - int(start)
    if not torch.is_tensor(row) or row.ndim == 0:
        return video_seg
    if int(row.shape[0]) != rows:
        raise H3FinalLayerPatchError(
            'per-token FinalLayer selector does not match the target-video rows'
        )
    index = torch.tensor(topology.forward, dtype=torch.long, device=row.device)
    return (start, stop, row.index_select(0, index))


def _restore_video_output(output, topology):
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        raise H3FinalLayerPatchError(
            'MiniMax H3 FinalLayer returned an unexpected output contract'
        )
    video = output[0]
    if not torch.is_tensor(video) or video.ndim < 1:
        raise H3FinalLayerPatchError(
            'MiniMax H3 FinalLayer returned an invalid video projection'
        )
    if int(video.shape[0]) != len(topology.inverse):
        raise H3FinalLayerPatchError(
            'FinalLayer video projection does not match the cube-order topology'
        )
    index = torch.tensor(topology.inverse, dtype=torch.long, device=video.device)
    restored = video.index_select(0, index)
    return (restored, output[1]) if isinstance(output, tuple) else [restored, output[1]]


def make_forward(
    layer,
    chunk_rows=None,
    *,
    original_forward=None,
    cube_state=None,
):
    signature = None if chunk_rows is None else int(chunk_rows)
    if signature is not None and signature <= 0:
        raise ValueError('chunk_rows must be positive')
    original_forward = original_forward or layer.forward
    current_contract = _accepts_current_contract(original_forward)
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
        topology = None
        active = False
        local_video_seg = video_seg
        if cube_state is not None:
            topology, active = cube_state.resolve(int(video_seg[1]) - int(video_seg[0]))
            if not active:
                local_video_seg = _ordered_selector(video_seg, topology)

        if signature is not None:
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
                    _chunk_count(video_rows, signature),
                    audio_rows,
                    _chunk_count(audio_rows, signature),
                    signature,
                )
            output = chunked_final_layer(
                layer,
                x,
                t_emb,
                local_video_seg,
                audio_seg,
                signature,
                sigma,
                sample_sigmas,
                shifts,
            )
        else:
            output = _call_original(
                original_forward,
                current_contract,
                x,
                t_emb,
                local_video_seg,
                audio_seg,
                sigma,
                sample_sigmas,
                shifts,
            )

        return output if topology is None else _restore_video_output(output, topology)

    setattr(forward, OWNER_MARKER, True)
    setattr(forward, SIGNATURE_MARKER, signature)
    setattr(forward, ORIGINAL_MARKER, original_forward)
    setattr(forward, CUBE_STATE_MARKER, cube_state)
    return forward


def _set_preserved_flag(model_patcher, value):
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options['h3_optimizations_preserved_final_layer_patch'] = bool(value)


def _stock_forward(model_patcher, layer):
    return unpatched_model_object(model_patcher, FINAL_LAYER_KEY, layer.forward)


def _restore_original(model_patcher, layer, original_forward):
    patches = model_patcher.object_patches
    stock = _stock_forward(model_patcher, layer)
    if (
        getattr(original_forward, '__self__', None) is layer
        and getattr(original_forward, '__func__', None)
        is getattr(stock, '__func__', None)
    ) or original_forward is stock:
        patches.pop(FINAL_LAYER_KEY, None)
    else:
        patches[FINAL_LAYER_KEY] = original_forward


def install(
    model_patcher,
    chunk_rows=None,
    *,
    cube_state=None,
    force_rebuild=False,
):
    '''Compose H3-owned FinalLayer behavior without wrapping foreign patches.'''

    if chunk_rows is not None:
        chunk_rows = int(chunk_rows)
        if chunk_rows <= 0:
            raise ValueError('chunk_rows must be positive')
    if chunk_rows is None and cube_state is None:
        raise ValueError('FinalLayer installation has no requested behavior')

    model = get_minimax_h3_model(model_patcher)
    if model is None:
        raise H3FinalLayerPatchError(
            'H3 Memory Optimization can only patch MiniMaxH3Model'
        )
    layer = getattr(model, 'final_layer', None)
    if layer is None:
        raise H3FinalLayerPatchError('MiniMax H3 has no final layer')

    existing = getattr(model_patcher, 'object_patches', {}).get(FINAL_LAYER_KEY)
    original_forward = _stock_forward(model_patcher, layer)
    if existing is not None:
        if not getattr(existing, OWNER_MARKER, False):
            _set_preserved_flag(model_patcher, True)
            logging.debug(
                '[H3 Optimizations] preserved foreign %s; H3 FinalLayer '
                'optimizations are disabled',
                FINAL_LAYER_KEY,
            )
            return False
        original_forward = getattr(existing, ORIGINAL_MARKER, None)
        if original_forward is None:
            raise H3FinalLayerPatchError(
                'installed H3 FinalLayer patch has no recoverable original'
            )
        if (
            getattr(existing, SIGNATURE_MARKER, None) == chunk_rows
            and getattr(existing, CUBE_STATE_MARKER, None) is cube_state
            and not force_rebuild
        ):
            return False

    model_patcher.add_object_patch(
        FINAL_LAYER_KEY,
        make_forward(
            layer,
            chunk_rows,
            original_forward=original_forward,
            cube_state=cube_state,
        ),
    )
    _set_preserved_flag(model_patcher, False)
    if chunk_rows is not None:
        logging.debug(
            '[H3 Optimizations] patched FinalLayer: chunk_rows=%d',
            chunk_rows,
        )
    return True


def clear_cube_state(model_patcher, cube_state):
    '''Remove one cube-order owner while retaining FinalLayer chunking.'''

    existing = getattr(model_patcher, 'object_patches', {}).get(FINAL_LAYER_KEY)
    if (
        existing is None
        or not getattr(existing, OWNER_MARKER, False)
        or getattr(existing, CUBE_STATE_MARKER, None) is not cube_state
    ):
        return False
    original_forward = getattr(existing, ORIGINAL_MARKER, None)
    if original_forward is None:
        raise H3FinalLayerPatchError(
            'installed H3 FinalLayer patch has no recoverable original'
        )
    model = get_minimax_h3_model(model_patcher)
    layer = getattr(model, 'final_layer', None) if model is not None else None
    if layer is None:
        raise H3FinalLayerPatchError('MiniMax H3 has no final layer')

    chunk_rows = getattr(existing, SIGNATURE_MARKER, None)
    if chunk_rows is None:
        _restore_original(model_patcher, layer, original_forward)
    else:
        model_patcher.add_object_patch(
            FINAL_LAYER_KEY,
            make_forward(
                layer,
                chunk_rows,
                original_forward=original_forward,
            ),
        )
    return True
