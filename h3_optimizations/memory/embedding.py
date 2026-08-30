'''Early release of dead MiniMax H3 embedding-assembly tensors.'''

import hashlib
import inspect
import logging

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
from comfy.ldm.minimax import model as minimax

from ..model import get_minimax_h3_model


FORWARD_KEY = 'diffusion_model._forward'
OWNER_MARKER = '_h3_optimizations_embedding_memory'
SIGNATURE_MARKER = '_h3_optimizations_embedding_memory_signature'
ORIGINAL_MARKER = '_h3_optimizations_embedding_memory_original'
FALLBACK_REASON_KEY = 'h3_optimizations_embedding_memory_fallback'
UPSTREAM_FORWARD_SHA256 = '14bdfccd6860f252005b8d43ab446aa9a938a13dc819061724b8f914218f5fd1'


class H3EmbeddingMemoryPatchError(RuntimeError):
    pass


def _source_digest(forward):
    return hashlib.sha256(inspect.getsource(forward).encode()).hexdigest()


def _validate_upstream_forward(forward):
    try:
        digest = _source_digest(forward)
    except (OSError, TypeError) as exc:
        raise H3EmbeddingMemoryPatchError(
            'cannot inspect MiniMax H3 _forward for embedding-memory compatibility'
        ) from exc
    if digest != UPSTREAM_FORWARD_SHA256:
        raise H3EmbeddingMemoryPatchError(
            'MiniMax H3 _forward changed; refusing the experimental embedding-memory patch'
        )


def make_forward(model, original_forward):
    announced = []

    def forward(x, timestep, context, transformer_options={}, minimax_payload=None,
                denoise_mask=None, audio_denoise_mask=None, **kwargs):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError('MiniMax H3 supports batch size 1')
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        layout = payload.get('layout')
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = minimax.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                          keyframes=payload.get('keyframes'),
                                          refs=payload.get('refs'))

        shift_v = float(transformer_options.get('minimax_h3_sigma_shift_video', model.sigma_shift_video))
        shift_a = float(transformer_options.get('minimax_h3_sigma_shift_audio', model.sigma_shift_audio))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - minimax.time_shift_sigma(sigma_v, shift_v, shift_a))

        vis_aug = float(payload.get('visual_cond_noise_aug', minimax.VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get('audio_cond_noise_aug', minimax.AUDIO_COND_TIMESTEP))
        seg_t = {'text': t_v, 'video': t_v, 'audio': t_a,
                 'cond': max(t_v, vis_aug), 'ref_img': max(t_v, vis_aug),
                 'cond_audio': max(t_a, aud_aug), 'ref_audio': max(t_a, aud_aug)}

        t_pin_v = max(t_v, minimax.VISUAL_COND_TIMESTEP)
        t_pin_a = max(t_a, minimax.AUDIO_COND_TIMESTEP)
        video_rows_t = None
        audio_rows_t = None
        if denoise_mask is not None:
            m = minimax.mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
            if m is not None:
                rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)
                if rows_t.unique().numel() == 1:
                    seg_t['video'] = float(rows_t[0])
                else:
                    video_rows_t = rows_t
        if audio_denoise_mask is not None:
            m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
            if not bool((m >= 1.0 - 1e-3).all()):
                sigma_a = 1.0 - t_a
                rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)
                if rows_t.unique().numel() == 1:
                    seg_t['audio'] = float(rows_t[0])
                else:
                    audio_rows_t = rows_t

        unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                          | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                          | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {'text': 1, 'video': 0, 'audio': 2, 'cond': 0, 'ref_img': 0, 'cond_audio': 2, 'ref_audio': 2}

        def rows_to_mod_index(rows_t, tag):
            levels = rows_t.unique()
            base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()],
                                dtype=torch.long, device=rows_t.device)
            return base[torch.searchsorted(levels, rows_t)]

        text_tags = payload.get('text_token_tags')
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == 'text' and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            elif kind == 'video' and video_rows_t is not None:
                mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))
            elif kind == 'audio' and audio_rows_t is not None:
                mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = minimax.patchify_video(video_x.to(torch.float32), model.patch_size)
        audio_rows = minimax.pack_audio(audio_x.to(torch.float32))
        cond_video_rows = model._cond_video_rows(payload, device)
        cond_audio_rows = model._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = model.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = model.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != model.hidden_size:
            text_states = model.token_refiner(model.condition_proj(text_states),
                                              transformer_options=transformer_options)

        h = torch.empty(layout.seq_len, model.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == 'text':
                h[a:b] = text_states
            elif kind in ('cond', 'ref_img', 'video'):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        del video_embed, audio_embed
        del all_video_rows, all_audio_rows
        del video_rows, audio_rows
        del cond_video_rows, cond_audio_rows
        del img_update, audio_update
        del text_states
        if not announced:
            announced.append(True)
            logging.debug(
                '[H3 Optimizations] released dead embedding tensors before block 0: rows=%d',
                int(h.shape[0]),
            )

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if model.use_adaln_curves:
            table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = model.time_embedder(t_vals).to(dtype)

        rope_freqs = minimax.rope_rotation_table(model.rope_freqs(layout.position_ids, device), dtype)

        patches_replace = transformer_options.get('patches_replace', {})
        blocks_replace = patches_replace.get('dit', {})
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(model.blocks), device, transformer_options)
        for i, block in enumerate(model.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            if ('double_block', i) in blocks_replace:
                def block_wrap(args):
                    return {'img': block(args['img'], args['t_emb'], args['mod_segments'], args['rope_freqs'],
                                         transformer_options=args['transformer_options'])}
                h = blocks_replace[('double_block', i)](
                    {'img': h, 't_emb': t_emb, 'mod_segments': mod_segments, 'rope_freqs': rope_freqs,
                     'transformer_options': transformer_options},
                    {'original_block': block_wrap})['img']
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        va, vb, _ = next(s for s in layout.segments if s[2] == 'video')
        aa, ab, _ = next(s for s in layout.segments if s[2] == 'audio')
        if video_rows_t is not None:
            video_seg = (va, vb, rows_to_mod_index(video_rows_t, 0) // 3)
        else:
            video_seg = (va, vb, t_row[seg_t['video']])
        if audio_rows_t is not None:
            audio_seg = (aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3)
        else:
            audio_seg = (aa, ab, t_row[seg_t['audio']])
        v, a = model.final_layer(h, t_emb, video_seg, audio_seg)

        video_out = minimax.unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, model.latents_dim, model.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = minimax.unpack_audio(a)

        return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]

    setattr(forward, OWNER_MARKER, True)
    setattr(forward, SIGNATURE_MARKER, 'release')
    setattr(forward, ORIGINAL_MARKER, original_forward)
    return forward


def install(model_patcher, *, force_rebuild=False, strict=False):
    model = get_minimax_h3_model(model_patcher)
    if model is None:
        raise H3EmbeddingMemoryPatchError(
            'H3 Memory Optimization can only patch MiniMaxH3Model'
        )
    existing = getattr(model_patcher, 'object_patches', {}).get(FORWARD_KEY)
    if existing is not None and not getattr(existing, OWNER_MARKER, False):
        options = model_patcher.model_options['transformer_options'] = (
            model_patcher.model_options.get('transformer_options', {}).copy()
        )
        options['h3_optimizations_preserved_embedding_patch'] = True
        return False
    if existing is not None:
        if not force_rebuild:
            return False
        original = getattr(existing, ORIGINAL_MARKER, None)
        if original is None:
            raise H3EmbeddingMemoryPatchError(
                'installed embedding-memory patch has no recoverable original'
            )
    else:
        original = model._forward
        if getattr(original, OWNER_MARKER, False):
            original = getattr(original, ORIGINAL_MARKER, None)
            if original is None:
                raise H3EmbeddingMemoryPatchError(
                    'installed embedding-memory patch has no recoverable original'
                )
    try:
        _validate_upstream_forward(original)
    except H3EmbeddingMemoryPatchError as exc:
        if strict:
            raise
        if existing is not None and getattr(existing, OWNER_MARKER, False):
            model_patcher.object_patches.pop(FORWARD_KEY)
        options = model_patcher.model_options['transformer_options'] = (
            model_patcher.model_options.get('transformer_options', {}).copy()
        )
        reason = str(exc)
        if options.get(FALLBACK_REASON_KEY) != reason:
            logging.warning(
                '[H3 Optimizations] embedding-memory release is unavailable; '
                'using ComfyUI stock lifetime: %s',
                reason,
            )
        options[FALLBACK_REASON_KEY] = reason
        options['h3_optimizations_preserved_embedding_patch'] = False
        return False
    model_patcher.add_object_patch(FORWARD_KEY, make_forward(model, original))
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options.pop(FALLBACK_REASON_KEY, None)
    options['h3_optimizations_preserved_embedding_patch'] = False
    return True


def is_installed(model_patcher):
    current = getattr(model_patcher, 'object_patches', {}).get(FORWARD_KEY)
    return bool(current is not None and getattr(current, OWNER_MARKER, False))


def clear(model_patcher):
    patches = getattr(model_patcher, 'object_patches', {})
    current = patches.get(FORWARD_KEY)
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options.pop(FALLBACK_REASON_KEY, None)
    if current is not None and getattr(current, OWNER_MARKER, False):
        patches.pop(FORWARD_KEY)
        return True
    return False
