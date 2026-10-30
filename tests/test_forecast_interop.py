"""CPU contracts for FinalLayer calls that bypass MiniMax H3 _forward."""

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

from comfy.ldm.minimax.model import (  # noqa: E402
    MiniMaxH3Model,
    pack_audio,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)
from comfy.model_patcher import ModelPatcher  # noqa: E402
from comfy.patcher_extension import WrappersMP  # noqa: E402
from h3_optimizations.cube_order import (  # noqa: E402
    CUBE_SHAPES,
    CubeOrderState,
    CubeOrderTopology,
    H3CubeOrderPatchError,
    install,
)
from h3_optimizations.memory import final_layer  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class _IdentityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_selector = None

    def forward(self, x, _t_emb, video_seg, audio_seg, _sigma=None,
                _sample_sigmas=None, _shifts=None):
        self.video_selector = video_seg[2]
        return x[video_seg[0]:video_seg[1]], x[audio_seg[0]:audio_seg[1]]


def _topology(forward):
    inverse = [0] * len(forward)
    for cube_row, raster_row in enumerate(forward):
        inverse[raster_row] = cube_row
    return CubeOrderTopology((1, 2, len(forward) // 2), tuple(forward), tuple(inverse))


def _publish(state, topology):
    token = state.enter(topology)
    state.leave(token, topology, True)


class ForecastFinalLayerInteropTests(unittest.TestCase):
    def test_registered_spectrum_bypass_matches_native_output_for_every_geometry(self):
        grid = (5, 17, 19)
        video_rows = grid[0] * grid[1] * grid[2]
        raster = torch.arange(video_rows * 4, dtype=torch.float32).reshape(video_rows, 4)
        video = unpatchify_video(raster, *grid, 1, (1, 2, 2))
        audio = torch.arange(32, dtype=torch.float32).reshape(1, 4, 2, 4)
        context = torch.zeros(1, 3, 4)

        for geometry in CUBE_SHAPES:
            with self.subTest(geometry=geometry):
                model = MiniMaxH3Model.__new__(MiniMaxH3Model)
                torch.nn.Module.__init__(model)
                model.patch_size = (1, 2, 2)
                model.final_layer = _IdentityLayer()
                captured = {}

                def actual(x, timestep, context, options, minimax_payload=None, **kwargs):
                    audio_hidden = pack_audio(x[1])
                    video_hidden = patchify_video(x[0], model.patch_size)
                    compact = torch.cat((audio_hidden, video_hidden))
                    captured['hidden'] = compact
                    hidden = torch.cat((context[0], compact))
                    audio_start = context.shape[1]
                    video_start = audio_start + audio_hidden.shape[0]
                    projected_video, projected_audio = model.final_layer(
                        hidden, None,
                        (video_start, hidden.shape[0], 0),
                        (audio_start, video_start, 1),
                    )
                    return [
                        unpatchify_video(projected_video, *grid, 1, model.patch_size),
                        unpack_audio(projected_audio),
                    ]

                model._forward = actual
                root = torch.nn.Module()
                root.diffusion_model = model
                patcher = ModelPatcher(root, torch.device('cpu'), torch.device('cpu'))
                # Spectrum registers here, not in transformer_options['wrappers'].
                patcher.add_wrapper_with_key(
                    WrappersMP.DIFFUSION_MODEL, 'spectrum_minimax_h3',
                    lambda executor, *args, **kwargs: executor(*args, **kwargs),
                )
                self.assertTrue(install(patcher, geometry))
                patcher.patch_model(load_weights=False)
                try:
                    native_video, native_audio = model._forward(
                        [video, audio], torch.tensor([500.0]), context, {},
                    )
                    compact = captured['hidden']
                    audio_rows = compact.shape[0] - video_rows
                    self.assertFalse(torch.equal(compact[audio_rows:], raster))
                    # A forecast reuses the captured representation and calls
                    # FinalLayer directly with compact, rebased target segments.
                    forecast_video, forecast_audio = model.final_layer(
                        compact, None,
                        (audio_rows, compact.shape[0], 0),
                        (0, audio_rows, 1),
                    )
                    forecast_video = unpatchify_video(
                        forecast_video, *grid, 1, model.patch_size,
                    )
                    forecast_audio = unpack_audio(forecast_audio)
                    torch.testing.assert_close(native_video, video, rtol=0, atol=0)
                    torch.testing.assert_close(native_audio, audio, rtol=0, atol=0)
                    torch.testing.assert_close(forecast_video, native_video, rtol=0, atol=0)
                    torch.testing.assert_close(forecast_audio, native_audio, rtol=0, atol=0)
                finally:
                    patcher.unpatch_model(unpatch_weights=False)

    def test_compact_bypass_restores_only_target_video_rows(self):
        state = CubeOrderState()
        topology = _topology((2, 0, 3, 1))
        _publish(state, topology)
        layer = _IdentityLayer()
        forward = final_layer.make_forward(layer, cube_state=state)

        audio = torch.tensor([[100.0], [101.0]])
        raster_video = torch.arange(4, dtype=torch.float32).unsqueeze(1)
        cube_video = raster_video.index_select(0, torch.tensor(topology.forward))
        compact = torch.cat((audio, cube_video), dim=0)

        video, projected_audio = forward(
            compact,
            None,
            (2, 6, 0),
            (0, 2, 1),
        )

        torch.testing.assert_close(video, raster_video)
        torch.testing.assert_close(projected_audio, audio)

    def test_bypass_reorders_a_per_token_video_selector_to_cube_order(self):
        state = CubeOrderState()
        topology = _topology((2, 0, 3, 1))
        _publish(state, topology)
        layer = _IdentityLayer()
        forward = final_layer.make_forward(layer, cube_state=state)
        selector = torch.arange(4)

        forward(torch.arange(6).unsqueeze(1), None, (2, 6, selector), (0, 2, 0))

        torch.testing.assert_close(
            layer.video_selector,
            selector.index_select(0, torch.tensor(topology.forward)),
        )

    def test_native_call_keeps_its_already_ordered_selector(self):
        state = CubeOrderState()
        topology = _topology((2, 0, 3, 1))
        layer = _IdentityLayer()
        forward = final_layer.make_forward(layer, cube_state=state)
        selector = torch.tensor([20, 10, 30, 11])

        token = state.enter(topology)
        try:
            forward(torch.arange(6).unsqueeze(1), None, (2, 6, selector), (0, 2, 0))
        finally:
            state.leave(token, topology, True)

        self.assertIs(layer.video_selector, selector)

    def test_bypass_without_a_completed_native_topology_fails_closed(self):
        state = CubeOrderState()
        forward = final_layer.make_forward(_IdentityLayer(), cube_state=state)

        with self.assertRaisesRegex(
            H3CubeOrderPatchError,
            "no completed cube-order topology",
        ):
            forward(torch.arange(6).unsqueeze(1), None, (2, 6, 0), (0, 2, 0))

    def test_mismatched_bypass_rows_fail_instead_of_guessing_by_size(self):
        state = CubeOrderState()
        _publish(state, _topology((2, 0, 3, 1)))
        forward = final_layer.make_forward(_IdentityLayer(), cube_state=state)

        with self.assertRaisesRegex(
            H3CubeOrderPatchError,
            "do not match the cube-order topology",
        ):
            forward(torch.arange(7).unsqueeze(1), None, (2, 7, 0), (0, 2, 0))

    def test_state_retains_one_plain_topology_and_no_tensor_cache(self):
        state = CubeOrderState()
        for forward in ((2, 0, 3, 1), (1, 3, 0, 2), (0, 2, 1, 3)):
            _publish(state, _topology(forward))

        latest, active = state.resolve(4)
        self.assertFalse(active)
        self.assertEqual(latest.forward, (0, 2, 1, 3))
        self.assertEqual(state.__dict__, {})
        self.assertFalse(any(torch.is_tensor(value) for value in state.__dict__.values()))

    def test_failed_native_call_invalidates_the_previous_topology(self):
        state = CubeOrderState()
        first = _topology((2, 0, 3, 1))
        failed = _topology((1, 3, 0, 2))
        _publish(state, first)

        token = state.enter(failed)
        state.leave(token, failed, False)

        with self.assertRaisesRegex(
            H3CubeOrderPatchError,
            "no completed cube-order topology",
        ):
            state.resolve(4)

    def test_new_wrapper_state_cannot_reuse_a_previous_wrappers_topology(self):
        previous = CubeOrderState()
        current = CubeOrderState()
        _publish(previous, _topology((2, 0, 3, 1)))

        with self.assertRaisesRegex(
            H3CubeOrderPatchError,
            "no completed cube-order topology",
        ):
            current.resolve(4)


if __name__ == "__main__":
    unittest.main()
