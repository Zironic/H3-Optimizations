'''Bounded MiniMax H3 FinalLayer execution.'''

import logging

import torch

import comfy.ops
from comfy.ldm.minimax.model import time_shift_sigma

from ..model import get_minimax_h3_model


FINAL_LAYER_KEY = 'diffusion_model.final_layer.forward'
OWNER_MARKER = '_h3_optimizations_final_layer'
SIGNATURE_MARKER = '_h3_optimizations_final_layer_signature'


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


def make_forward(layer, chunk_rows):
    signature = int(chunk_rows)
    # One line the first time the patched forward actually executes. The
    # install-time message only proves the patch was attached; it stays silent
    # when routing sends the forward somewhere else, which is exactly the case
    # that has to be visible. Reporting the chunk counts also separates real
    # chunking from a segment that fits in one chunk and is therefore bounded
    # in name only.
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
        return chunked_final_layer(
            layer,
            x,
            t_emb,
            video_seg,
            audio_seg,
            signature,
            sigma,
            sample_sigmas,
            shifts,
        )

    setattr(forward, OWNER_MARKER, True)
    setattr(forward, SIGNATURE_MARKER, signature)
    return forward


def install(model_patcher, chunk_rows, *, force_rebuild=False):
    '''Patch FinalLayer once; identical installation is idempotent.'''

    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    model = get_minimax_h3_model(model_patcher)
    if model is None:
        raise H3FinalLayerPatchError(
            'H3 Memory Optimization can only patch MiniMaxH3Model'
        )
    layer = getattr(model, 'final_layer', None)
    if layer is None:
        raise H3FinalLayerPatchError('MiniMax H3 has no final layer')

    existing = getattr(model_patcher, 'object_patches', {}).get(FINAL_LAYER_KEY)
    if existing is not None:
        if not getattr(existing, OWNER_MARKER, False):
            options = model_patcher.model_options['transformer_options'] = (
                model_patcher.model_options.get('transformer_options', {}).copy()
            )
            options['h3_optimizations_preserved_final_layer_patch'] = True
            logging.debug(
                '[H3 Optimizations] preserved foreign %s; FinalLayer chunking '
                'is disabled',
                FINAL_LAYER_KEY,
            )
            return False
        installed = getattr(existing, SIGNATURE_MARKER, None)
        if installed == chunk_rows and not force_rebuild:
            return False

    model_patcher.add_object_patch(
        FINAL_LAYER_KEY,
        make_forward(layer, chunk_rows),
    )
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options['h3_optimizations_preserved_final_layer_patch'] = False
    logging.debug(
        '[H3 Optimizations] patched FinalLayer: chunk_rows=%d',
        chunk_rows,
    )
    return True
