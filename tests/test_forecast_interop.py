"""CPU contracts for forecast consumers that bypass MiniMax H3 _forward."""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402
from h3_optimizations.cube_order import (  # noqa: E402
    FORECAST_CONSUMERS_KEY,
    FORECAST_INTEROP_API,
    FORECAST_REPRESENTATION_KEY,
    FORWARD_KEY,
    SPECTRUM_WRAPPER_KEY,
    forecast_representation_contract,
    install,
    restore_forecast_target_hidden_to_raster,
    tile_aligned_cube_major_indices,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class _FakePatcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name == "diffusion_model":
            return self.model
        if name == FORWARD_KEY:
            return self.model._forward
        raise KeyError(name)

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


def _model():
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    model._forward = lambda *args, **kwargs: None
    return model


def _activate_spectrum(patcher):
    patcher.model_options["transformer_options"]["wrappers"] = {
        "diffusion_model": {
            SPECTRUM_WRAPPER_KEY: [lambda executor, *args, **kwargs: None],
        }
    }


class ForecastInteropTests(unittest.TestCase):
    def test_spectrum_without_contract_falls_back_to_raster(self):
        patcher = _FakePatcher(_model())
        _activate_spectrum(patcher)

        self.assertFalse(install(patcher, (1, 8, 8)))
        self.assertNotIn(FORWARD_KEY, patcher.object_patches)
        self.assertNotIn(
            FORECAST_REPRESENTATION_KEY,
            patcher.model_options["transformer_options"],
        )

    def test_spectrum_contract_allows_requested_cube_order(self):
        patcher = _FakePatcher(_model())
        _activate_spectrum(patcher)
        patcher.model_options["transformer_options"][FORECAST_CONSUMERS_KEY] = {
            SPECTRUM_WRAPPER_KEY: {
                "api": FORECAST_INTEROP_API,
                "accepts_representation_adapter": True,
            }
        }

        self.assertTrue(install(patcher, (1, 8, 8)))
        self.assertIn(FORWARD_KEY, patcher.object_patches)
        contract = patcher.model_options["transformer_options"][
            FORECAST_REPRESENTATION_KEY
        ]
        self.assertEqual(contract["api"], FORECAST_INTEROP_API)
        self.assertEqual(contract["video_token_order"], (1, 8, 8))
        self.assertTrue(callable(contract["to_native_target_hidden"]))

    def test_adapter_restores_only_target_video_rows_to_native_raster(self):
        audio_rows = 5
        video_rows = 128
        hidden = 3
        layout = type(
            "Layout",
            (),
            {"segments": [(0, audio_rows, "audio"), (audio_rows, audio_rows + video_rows, "video")]},
        )()
        raster_video = torch.arange(
            video_rows * hidden,
            dtype=torch.float32,
        ).reshape(video_rows, hidden)
        forward, _inverse = tile_aligned_cube_major_indices(
            (1, 8, 16),
            audio_rows,
            (1, 8, 8),
        )
        self.assertNotEqual(tuple(forward), tuple(range(video_rows)))
        audio = torch.full((audio_rows, hidden), -7.0)
        cube_target = torch.cat((audio, raster_video[list(forward)]), dim=0).unsqueeze(0)

        restored = restore_forecast_target_hidden_to_raster(
            cube_target,
            layout=layout,
            video_shape=(1, 16, 32),
            patch_size=(1, 2, 2),
            cube_shape=(1, 8, 8),
        )

        self.assertTrue(torch.equal(restored[0, :audio_rows], audio))
        self.assertTrue(torch.equal(restored[0, audio_rows:], raster_video))
        self.assertFalse(torch.equal(restored, cube_target))

    def test_contract_is_geometry_agnostic_across_supported_cube_shapes(self):
        for shape in ((1, 8, 8), (1, 16, 4), (4, 4, 4)):
            with self.subTest(shape=shape):
                contract = forecast_representation_contract(shape)
                self.assertEqual(contract["api"], FORECAST_INTEROP_API)
                self.assertEqual(contract["video_token_order"], shape)
                self.assertTrue(callable(contract["to_native_target_hidden"]))


if __name__ == "__main__":
    unittest.main()
