'''Tiny real ComfyUI 0.45+ H3 models: patched forwards match stock.'''

from pathlib import Path
import sys
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.ops  # noqa: E402
from comfy.ldm.minimax import model as minimax  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402

from h3_optimizations import cube_order  # noqa: E402
from h3_optimizations.memory import final_layer  # noqa: E402
from h3_optimizations.runtime.layout import resolve_layout  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]

HAS_IMAGE_MODEL = hasattr(minimax, 'MiniMaxH3ImageModel')
DIMS = dict(
    hidden_size=128, num_layers=2, token_refiner_num_layers=1,
    num_attention_heads=1, attention_head_dim=128, ffn_hidden_size=128,
    latents_dim=8, text_dim=32, timestep_input_dim=32,
    time_embed_hidden_size=64, time_embed_dim=64,
    dtype=torch.float32, device='cpu',
    operations=comfy.ops.disable_weight_init,
)


def _randomized(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
        model.rope.inv_freq.copy_(
            1.0 / (10000 ** (torch.arange(16, dtype=torch.float32) / 16))
        )
    # Kitchen's in-place RoPE rejects tensors that require grad.
    return model.requires_grad_(False)


class _Root(torch.nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model


def _run(model, setup, *args, **kwargs):
    patcher = ModelPatcher(_Root(model), torch.device('cpu'), torch.device('cpu'))
    if setup is not None:
        setup(patcher)
    patcher.patch_model(load_weights=False)
    try:
        with torch.inference_mode():
            return patcher.model.diffusion_model(*args, **kwargs)
    finally:
        patcher.unpatch_model()


SETUPS = {
    'chunked FinalLayer': lambda patcher: final_layer.install(patcher, 16),
    'cube order': lambda patcher: cube_order.install(patcher, (1, 8, 8)),
    'both': lambda patcher: (
        final_layer.install(patcher, 16),
        cube_order.install(patcher, (1, 8, 8)),
    ),
}


@unittest.skipUnless(HAS_IMAGE_MODEL, 'MiniMax H3 Image needs ComfyUI 0.45+')
class H3ImageTests(unittest.TestCase):
    def test_patched_image_forward_matches_stock(self):
        # 21x19 patches: ragged against the 8x8 cubes and the router tile.
        x = torch.randn(1, 8, 1, 42, 38)
        context = torch.randn(1, 6, 32)
        sample_sigmas = torch.tensor([1.0, 0.7, 0.3, 0.0])
        for heads, grid in (
            (1, [1.0, 0.0]),
            (4, [1.0, 0.8, 0.5, 0.2, 0.0]),
        ):
            model = _randomized(
                minimax.MiniMaxH3ImageModel(pdd_sigmas=grid, pdd_steps=heads, **DIMS)
            )
            for sigma in (0.7, 0.3):
                timestep = torch.tensor([sigma * 1000.0])
                stock = _run(
                    model, None, x, timestep, context,
                    transformer_options={'sample_sigmas': sample_sigmas},
                )
                for name, setup in SETUPS.items():
                    with self.subTest(heads=heads, sigma=sigma, patch=name):
                        patched = _run(
                            model, setup, x, timestep, context,
                            transformer_options={'sample_sigmas': sample_sigmas},
                        )
                        self.assertTrue(torch.is_tensor(patched))
                        self.assertEqual(patched.shape, stock.shape)
                        torch.testing.assert_close(patched, stock, atol=1e-5, rtol=0)

    def test_layout_resolves_one_latent_without_audio(self):
        x = torch.randn(1, 8, 1, 42, 38)
        context = torch.randn(1, 6, 32)
        layout = resolve_layout(x, context, {})
        self.assertEqual(layout.audio_t, 0)
        self.assertEqual(layout.audio_range, (layout.video_range[0],) * 2)
        self.assertEqual(layout.video_range[1], layout.seq_len)
        self.assertEqual(layout.video_shape, (1, 21, 19))

    def test_baked_pdd_video_forward_matches_stock(self):
        # ComfyUI 0.45 hands FinalLayer per-stream grids instead of shifts.
        model = _randomized(
            minimax.MiniMaxH3Model(audio_latents_dim=4, pdd_steps=4, **DIMS)
        )
        x = [torch.randn(1, 8, 2, 34, 30), torch.randn(1, 4, 2, 3)]
        context = torch.randn(1, 5, 32)
        sample_sigmas = torch.tensor([1.0, 0.9, 0.6, 0.3, 0.0])
        for sigma in (0.9, 0.6, 0.3):
            with self.subTest(sigma=sigma):
                timestep = torch.tensor([sigma * 1000.0])
                options = lambda: {'sample_sigmas': sample_sigmas}
                stock = _run(model, None, x, timestep, context,
                             transformer_options=options())
                patched = _run(model, SETUPS['both'], x, timestep, context,
                               transformer_options=options())
                for want, got in zip(stock, patched):
                    self.assertEqual(got.shape, want.shape)
                    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)


if __name__ == '__main__':
    unittest.main()
