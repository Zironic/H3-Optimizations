"""CPU contracts for _forward-bypassing forecast consumers and cube ordering."""

import os
from pathlib import Path
from types import SimpleNamespace
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

from h3_optimizations.cube_order import (  # noqa: E402
    CubeOrderState,
    SPECTRUM_RUNTIME_KEY,
    make_forward as make_cube_forward,
    tile_aligned_cube_major_indices,
)
from h3_optimizations.memory import final_layer  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class _PassThroughFinalLayer:
    def __init__(self):
        self.seen_video_selector = None

    def forward(
        self,
        x,
        _t_emb,
        video_seg,
        audio_seg,
        _sigma=None,
        _sample_sigmas=None,
        _shifts=None,
    ):
        va, vb, selector = video_seg
        aa, ab, _audio_selector = audio_seg
        self.seen_video_selector = selector
        return x[va:vb].clone(), x[aa:ab].clone()


def _nontrivial_mapping(sequence_offset=37):
    # Two complete 64-token cubes make a nontrivial tile-aligned permutation
    # when the packed video segment begins off a router-tile boundary.
    grid = (1, 8, 16)
    forward, inverse = tile_aligned_cube_major_indices(
        grid,
        sequence_offset,
        (1, 8, 8),
    )
    if tuple(forward) == tuple(range(128)):
        raise AssertionError("test geometry unexpectedly produced raster order")
    return grid, forward, inverse


class ForecastInteropTests(unittest.TestCase):
    def test_compact_forecast_final_layer_restores_raster_by_video_row_count(self):
        native_video_start = 37
        grid, forward, inverse = _nontrivial_mapping(native_video_start)
        state = CubeOrderState((1, 8, 8))
        state.record_cube(
            native_video_start,
            native_video_start + 128,
            forward,
            inverse,
            grid,
        )

        audio_rows = 6
        hidden = 3
        audio = torch.full((audio_rows, hidden), -7.0)
        raster_video = torch.arange(128 * hidden, dtype=torch.float32).reshape(
            128,
            hidden,
        )
        cube_video = raster_video[list(forward)]
        compact = torch.cat((audio, cube_video), dim=0)

        layer = _PassThroughFinalLayer()
        wrapped = final_layer.make_forward(layer, cube_state=state)
        video, restored_audio = wrapped(
            compact,
            None,
            (audio_rows, audio_rows + 128, 0),
            (0, audio_rows, 0),
        )

        self.assertTrue(torch.equal(video, raster_video))
        self.assertTrue(torch.equal(restored_audio, audio))

    def test_bypass_per_token_selector_is_forward_permuted_before_final_layer(self):
        native_video_start = 37
        grid, forward, inverse = _nontrivial_mapping(native_video_start)
        state = CubeOrderState((1, 8, 8))
        state.record_cube(
            native_video_start,
            native_video_start + 128,
            forward,
            inverse,
            grid,
        )
        layer = _PassThroughFinalLayer()
        wrapped = final_layer.make_forward(layer, cube_state=state)
        selector = torch.arange(128, dtype=torch.long)
        compact = torch.zeros(134, 2)

        wrapped(
            compact,
            None,
            (6, 134, selector),
            (0, 6, 0),
        )

        self.assertTrue(
            torch.equal(
                layer.seen_video_selector,
                selector[list(forward)],
            )
        )

    def test_native_cube_call_does_not_double_permute_existing_selector(self):
        native_video_start = 37
        grid, forward, inverse = _nontrivial_mapping(native_video_start)
        state = CubeOrderState((1, 8, 8))
        entry = state.record_cube(
            native_video_start,
            native_video_start + 128,
            forward,
            inverse,
            grid,
        )
        layer = _PassThroughFinalLayer()
        wrapped = final_layer.make_forward(layer, cube_state=state)
        already_cube = torch.arange(128, dtype=torch.long)[list(forward)]
        packed = torch.zeros(native_video_start + 128, 2)

        token = state.begin_call(entry)
        try:
            wrapped(
                packed,
                None,
                (native_video_start, native_video_start + 128, already_cube),
                (0, 6, 0),
            )
        finally:
            state.end_call(token)

        self.assertTrue(torch.equal(layer.seen_video_selector, already_cube))

    def test_spectrum_state_conditioned_residual_keeps_native_raster_input(self):
        patch_size = (1, 2, 2)
        model = SimpleNamespace(patch_size=patch_size)
        state = CubeOrderState((1, 8, 8))
        captured = {}

        def original(
            x,
            _timestep,
            _context,
            _transformer_options,
            minimax_payload=None,
            **_kwargs,
        ):
            captured["video"] = x[0]
            captured["layout"] = minimax_payload["layout"]
            return [x[0].clone(), x[1]]

        wrapped = make_cube_forward(
            model,
            original,
            (1, 8, 8),
            state,
        )
        video = torch.arange(1 * 1 * 1 * 16 * 16, dtype=torch.float32).reshape(
            1,
            1,
            1,
            16,
            16,
        )
        audio = torch.zeros(1, 2, 2, 3)
        runtime = SimpleNamespace(state_conditioned_residual=True)

        output = wrapped(
            [video, audio],
            torch.tensor([500.0]),
            torch.zeros(1, 3, 8),
            {SPECTRUM_RUNTIME_KEY: runtime},
            minimax_payload={},
        )

        self.assertIs(captured["video"], video)
        self.assertTrue(torch.equal(output[0], video))
        self.assertFalse(hasattr(captured["layout"], "h3_cube_order"))
        video_segment = next(
            segment
            for segment in captured["layout"].segments
            if segment[2] == "video"
        )
        entry = state.resolve((video_segment[0], video_segment[1], 0))
        self.assertIsNotNone(entry)
        self.assertEqual(entry["mode"], "raster")

    def test_ordinary_spectrum_runtime_does_not_disable_cube_order(self):
        patch_size = (1, 2, 2)
        model = SimpleNamespace(patch_size=patch_size)
        state = CubeOrderState((1, 8, 8))
        captured = {}

        def original(
            x,
            _timestep,
            _context,
            _transformer_options,
            minimax_payload=None,
            **_kwargs,
        ):
            captured["video"] = x[0]
            captured["layout"] = minimax_payload["layout"]
            return [x[0].clone(), x[1]]

        wrapped = make_cube_forward(
            model,
            original,
            (1, 8, 8),
            state,
        )
        video = torch.arange(1 * 1 * 1 * 16 * 32, dtype=torch.float32).reshape(
            1,
            1,
            1,
            16,
            32,
        )
        audio = torch.zeros(1, 2, 2, 3)
        runtime = SimpleNamespace(state_conditioned_residual=False)

        wrapped(
            [video, audio],
            torch.tensor([500.0]),
            torch.zeros(1, 3, 8),
            {SPECTRUM_RUNTIME_KEY: runtime},
            minimax_payload={},
        )

        self.assertTrue(hasattr(captured["layout"], "h3_cube_order"))
        self.assertFalse(
            torch.equal(
                captured["video"],
                video,
            )
        )


if __name__ == "__main__":
    unittest.main()
