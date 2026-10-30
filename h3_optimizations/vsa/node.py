'''H3 VSA Attention: run a VSA-trained H3 checkpoint (FastH3) as trained.

VSA checkpoints learn their sparsity pattern, so this node reproduces
FastVideo's tiling, selection and coarse branch instead of offering the
normal sparse node's schedules, token orders and backends. It installs block
replace patches that hand ComfyUI's ``attention=`` slot a VSA callable, which
composes with H3 Memory Optimization's block forward.
'''

from __future__ import annotations

import logging
import re

from comfy_api.latest import io

import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model

from ..node_constants import NODE_CATEGORY
import torch

from . import kernel
from .attention import BACKEND_BF16, BACKEND_INT8, _int8_fine, vsa_attention
from .tiling import build_geometry, compute_topk, sparsity_from_keep_percent

LOG_PREFIX = '[H3 VSA]'
PATCH_KEY = 'h3_vsa_attention'
GEOMETRY_CACHE = 4
DEFAULT_KEEP_PERCENT = 20.0
BACKEND_AUTO_OPTION = 'Auto'
BACKEND_INT8_OPTION = 'INT8 (Kitchen)'
BACKEND_BF16_OPTION = 'BF16 (Triton)'
BACKEND_OPTIONS = (BACKEND_AUTO_OPTION, BACKEND_INT8_OPTION, BACKEND_BF16_OPTION)
MEMORY_STANDARD = 'Standard'
MEMORY_LOWER_VRAM = 'Lower VRAM (slower)'
MEMORY_OPTIONS = (MEMORY_STANDARD, MEMORY_LOWER_VRAM)
# INT8 attention error against BF16 sits around 1-3% rel_l2; a broken
# carrier, route or tile mask lands far above this.
INT8_PARITY_LIMIT = 0.08
_INT8_PARITY = {}


def int8_parity(device):
    """One-time per-device check of the INT8 tiles path against BF16 Triton.

    Returns (ok, detail). Ragged edge tiles make it exercise the per-tile KV
    mask, not only full tiles.
    """
    key = str(device)
    if key in _INT8_PARITY:
        return _INT8_PARITY[key]
    try:
        from ..native import int8_attention as native
        if not native.sparse_tiles_attention_is_available():
            result = (False, 'the loaded INT8 library has no per-tile KV traversal')
        else:
            grid = (5, 6, 7)
            segments = [(0, 29, 'text'), (29, 129, 'audio'), (129, 339, 'video')]
            geometry = build_geometry(segments, grid, device)
            heads = 4
            generator = torch.Generator(device='cpu').manual_seed(7)
            buffers = []
            for _ in range(3):
                buffer = torch.zeros(
                    (heads, geometry.padded_len, 128), dtype=torch.bfloat16, device=device,
                )
                values = torch.randn(heads, geometry.seq_len, 128, generator=generator)
                buffer.index_copy_(1, geometry.inv, values.to(device=device, dtype=torch.bfloat16))
                buffers.append(buffer)
            block_len = geometry.block_len.to(device)
            prefix, video = geometry.num_prefix_tiles, geometry.num_video_tiles
            keep = compute_topk(0.8, video)
            scores = torch.randn(heads, video, video, generator=generator).to(device)
            route = torch.cat((
                torch.arange(prefix, device=device).expand(heads, video, prefix),
                scores.topk(keep, -1).indices.sort(-1).values + prefix,
            ), -1).to(torch.int32).contiguous()
            reference = kernel.sparse_attention(*buffers, block_len, route, prefix)
            candidate = _int8_fine(list(buffers), block_len, route, geometry)
            live = geometry.inv
            a = candidate[:, live].float()
            b = reference[:, live].float()
            error = float((a - b).norm() / b.norm())
            ok = bool(torch.isfinite(a).all()) and error < INT8_PARITY_LIMIT
            result = (ok, 'rel_l2 %.4f against BF16 Triton' % error)
    except Exception as error:  # a missing or broken native build falls back
        result = (False, '%s: %s' % (type(error).__name__, error))
    _INT8_PARITY[key] = result
    return result


def is_vsa_checkpoint(model):
    '''True for an H3 checkpoint that carries VSA's to_gate_compress layers.'''
    try:
        diffusion_model = model.get_model_object('diffusion_model')
    except Exception:
        return False
    if not isinstance(diffusion_model, MiniMaxH3Model) or not len(diffusion_model.blocks):
        return False
    return getattr(diffusion_model.blocks[0].attn, 'to_gate_compress', None) is not None


def parse_block_list(text):
    '''"0, 1, 47-49" -> {0, 1, 47, 48, 49}.'''
    blocks = set()
    for part in re.findall(r'\d+\s*-\s*\d+|\d+', text or ''):
        if '-' in part:
            first, last = (int(value) for value in part.split('-'))
            blocks.update(range(min(first, last), max(first, last) + 1))
        else:
            blocks.add(int(part))
    return blocks


def _pause_compiler():
    return comfy.model_prefetch.pause_malloc_graph()


class VSAState:
    '''Options and per-run caches for one patched model.'''

    def __init__(self, sparsity, dense_first_steps, dense_layers, verbose,
                 backend=BACKEND_AUTO_OPTION, memory_mode=MEMORY_STANDARD):
        self.backend_request = backend
        self.lower_vram = memory_mode == MEMORY_LOWER_VRAM
        self.sparsity = sparsity
        self.dense_first_steps = dense_first_steps
        self.dense_layers = dense_layers
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.geometries = {}
        self._logged = set()

    def backend(self, device):
        if self.backend_request == BACKEND_BF16_OPTION:
            return BACKEND_BF16
        ok, detail = int8_parity(device)
        if ok:
            self.log_once(('backend', str(device)), 'INT8 fine stage (%s)' % detail)
            return BACKEND_INT8
        if self.backend_request == BACKEND_INT8_OPTION:
            raise RuntimeError('H3 VSA INT8 backend is unavailable: %s' % detail)
        if ('int8_fallback', str(device)) not in self._logged:
            self._logged.add(('int8_fallback', str(device)))
            logging.warning('%s INT8 unavailable, using BF16 Triton: %s', LOG_PREFIX, detail)
        return BACKEND_BF16

    def log_once(self, key, message):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            logging.info('%s %s', LOG_PREFIX, message)

    def geometry(self, layout, device):
        key = (tuple(layout.signature), tuple(tuple(s) for s in layout.segments), str(device))
        geometry = self.geometries.get(key)
        if geometry is None:
            _text_len, latent_t, lat_h, lat_w, _audio_t = layout.signature
            grid = (int(latent_t), int(lat_h) // 2, int(lat_w) // 2)
            # the cached tile maps outlive one block; keep them out of the
            # comfy compiler's per-block allocation graph
            with _pause_compiler():
                geometry = build_geometry(layout.segments, grid, device)
            while len(self.geometries) >= GEOMETRY_CACHE:
                del self.geometries[next(iter(self.geometries))]
            self.geometries[key] = geometry
            self.log_once(
                ('geometry', key),
                '%d rows -> %d prefix + %d video tiles (%d padded rows)'
                % (geometry.seq_len, geometry.num_prefix_tiles,
                   geometry.num_video_tiles, geometry.padded_len),
            )
        return geometry

    def step_index(self, transformer_options):
        sigmas = transformer_options.get('sigmas')
        sample_sigmas = transformer_options.get('sample_sigmas')
        if sigmas is None or sample_sigmas is None:
            return None
        sigma = float(sigmas.reshape(-1)[0])
        return int((sample_sigmas.float().cpu() - sigma).abs().argmin())

    def dense_for(self, transformer_options, block_index):
        if block_index in self.dense_layers:
            return True
        if self.dense_first_steps:
            step = self.step_index(transformer_options)
            if step is None:
                self.log_once('no_step', 'no sampler sigmas; dense_first_steps ignored')
            elif step < self.dense_first_steps:
                return True
        return False


def _make_block_patch(block, block_index, state):
    def attention(h, rope_freqs=None, transformer_options={}):
        layout = transformer_options.get('minimax_h3_layout')
        if layout is None or layout.seq_len != h.shape[0]:
            raise RuntimeError(
                'H3 VSA Attention needs the packed H3 layout for every block call'
            )
        geometry = state.geometry(layout, h.device)
        return vsa_attention(
            block.attn, h, rope_freqs, geometry, state.sparsity,
            dense=state.dense_for(transformer_options, block_index),
            backend=state.backend(h.device),
            lower_vram=state.lower_vram,
        )

    def block_patch(args, extra):
        return extra['original_block']({**args, 'attention': attention})

    return block_patch


def apply_vsa(model, keep_percent, dense_first_steps=0, dense_layers='', verbose=False,
              backend=BACKEND_AUTO_OPTION, memory_mode=MEMORY_STANDARD):
    if not is_vsa_checkpoint(model):
        raise ValueError(
            'H3 VSA Attention needs a VSA-trained MiniMax H3 checkpoint (one with '
            'to_gate_compress layers, such as FastH3). Use H3 Sparse Attention for '
            'the base model.'
        )
    state = VSAState(
        sparsity=sparsity_from_keep_percent(keep_percent),
        dense_first_steps=int(dense_first_steps),
        dense_layers=parse_block_list(dense_layers),
        verbose=bool(verbose),
        backend=backend,
        memory_mode=memory_mode,
    )
    patched = model.clone()
    diffusion_model = model.get_model_object('diffusion_model')
    for index, block in enumerate(diffusion_model.blocks):
        patched.set_model_patch_replace(
            _make_block_patch(block, index, state), 'dit', 'double_block', index,
        )
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
        PATCH_KEY,
        lambda model_patcher: state.reset(),
    )
    logging.info(
        '%s keep %.4g%% of video tiles (sparsity %.6g), dense first steps %d, '
        'dense layers %s, backend %s, memory %s',
        LOG_PREFIX, float(keep_percent), state.sparsity, state.dense_first_steps,
        sorted(state.dense_layers) or 'none', backend, memory_mode,
    )
    return patched


class H3VSAAttention(io.ComfyNode):
    '''Trained VSA sparse attention for FastH3-style checkpoints.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3VSAAttention',
            display_name='H3 VSA Attention (FastH3)',
            category=NODE_CATEGORY,
            description=(
                'Runs a VSA-trained MiniMax H3 checkpoint (FastH3) with the sparse '
                'attention it was trained on: 4x4x4 video tiles, trained top-k tile '
                'selection and the gated coarse branch. Text and audio stay dense. '
                'Errors on checkpoints without VSA gate layers; use H3 Sparse '
                'Attention for the base model.'
            ),
            search_aliases=['VSA', 'FastH3', 'video sparse attention', 'H3 VSA'],
            inputs=[
                io.Model.Input('model'),
                io.Float.Input(
                    'keep_percent',
                    display_name='Video tiles kept (%)',
                    default=DEFAULT_KEEP_PERCENT,
                    min=1.0,
                    max=100.0,
                    step=0.5,
                    tooltip=(
                        'Percent of video tiles each video query tile attends exactly. '
                        'Use the value the checkpoint was trained at: 20 for FastH3 '
                        '8-Step V2 (80% sparse), 10 for FastH3 Preview v1 (90% '
                        'sparse). Other values leave the trained operating point; '
                        'more is slower and not guaranteed to look better.'
                    ),
                ),
                io.Int.Input(
                    'dense_first_steps',
                    default=0,
                    min=0,
                    max=1000,
                    advanced=True,
                    tooltip=(
                        'Sampler steps that keep every video tile. FastH3 was trained '
                        'sparse on every step, so 0 matches training.'
                    ),
                ),
                io.String.Input(
                    'dense_layers',
                    default='',
                    advanced=True,
                    tooltip=(
                        "Blocks that keep every video tile, e.g. '0, 1, 47-49'. "
                        'Empty matches training.'
                    ),
                ),
                io.Combo.Input(
                    'backend',
                    options=list(BACKEND_OPTIONS),
                    default=BACKEND_AUTO_OPTION,
                    advanced=True,
                    tooltip=(
                        'Kernel for the exact sparse pass. Auto uses the vendored '
                        'Kitchen INT8 kernel when it is present and passes a one-time '
                        'check against BF16 on this GPU, else BF16 Triton. Tile '
                        'selection and the coarse branch always run in fp32.'
                    ),
                ),
                io.Combo.Input(
                    'memory_mode',
                    options=list(MEMORY_OPTIONS),
                    default=MEMORY_STANDARD,
                    advanced=True,
                    tooltip=(
                        'INT8 only. Lower VRAM stages V in two passes instead of '
                        'keeping it in BF16, and writes attention output into the '
                        'block input it replaces. Slower; use it when the long '
                        'sequences do not fit.'
                    ),
                ),
                io.Boolean.Input(
                    'verbose',
                    default=False,
                    advanced=True,
                    tooltip='Log the tile geometry once per shape.',
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, keep_percent=DEFAULT_KEEP_PERCENT, dense_first_steps=0,
                dense_layers='', backend=BACKEND_AUTO_OPTION,
                memory_mode=MEMORY_STANDARD, verbose=False):
        return io.NodeOutput(apply_vsa(
            model, keep_percent, dense_first_steps, dense_layers, verbose, backend,
            memory_mode,
        ))
