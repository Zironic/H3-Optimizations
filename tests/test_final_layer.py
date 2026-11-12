'''Production contracts for bounded H3 FinalLayer execution.'''

from types import SimpleNamespace
from pathlib import Path
import inspect
import sys
import unittest
from unittest import mock

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

from h3_optimizations.memory import final_layer
from h3_optimizations.cube_order import CubeOrderState, CubeOrderTopology
import h3_optimizations.apply as apply_module
from h3_optimizations.plan import H3OptimizationPlan, MemoryRequest
from h3_optimizations.qkv.providers import MLPProviderResolution
from comfy.model_patcher import ModelPatcher

sys.argv = [sys.argv[0], *TEST_ARGS]


def _core_schedule(heads, shifts=(12.0, 3.0)):
    '''The last FinalLayer argument the installed core passes for these shifts.'''
    if hasattr(minimax, 'pdd_grid'):
        # ComfyUI 0.45+ passes one sigma grid per stream.
        return tuple(minimax.pdd_grid(heads, shift) for shift in shifts)
    return shifts


def _shifted_grid(heads, shift):
    '''ComfyUI 0.45's pdd_grid, so grid schedules are testable on any core.'''
    return [
        shift * (k / heads) / (1.0 + (shift - 1.0) * (k / heads))
        for k in range(heads, -1, -1)
    ]


class _Projection(torch.nn.Module):
    comfy_cast_weights = False
    weight_function = ()
    bias_function = ()

    def __init__(self, weight, bias, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = torch.nn.Parameter(bias, requires_grad=False)
        self.out_features = out_features

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight, self.bias)


class _Layer:
    norm = staticmethod(lambda value: value * 0.5)

    def __init__(self):
        self.video_out = _Projection(
            torch.arange(12, dtype=torch.float32).reshape(4, 3).T,
            torch.zeros(3, dtype=torch.float32),
            3,
        )
        self.audio_out = _Projection(
            torch.arange(8, dtype=torch.float32).reshape(4, 2).T,
            torch.zeros(2, dtype=torch.float32),
            2,
        )

    @staticmethod
    def adaln_proj(_t_emb):
        shift = torch.tensor(
            [[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=torch.float32
        )
        scale = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
            dtype=torch.float32,
        )
        return shift, scale

    def forward(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.adaln_proj(t_emb)

        def project(segment, output):
            first, last, row = segment
            value = (
                self.norm(x[first:last]) * (1.0 + scale[row]) + shift[row]
            ).float()
            return output(value)

        return project(video_seg, self.video_out), project(
            audio_seg, self.audio_out
        )


class _PDDLayer(_Layer):
    def __init__(self):
        super().__init__()
        video_rows = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        audio_rows = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self.video_out = _Projection(
            torch.cat(
                (video_rows, video_rows + 2, video_rows + 6, video_rows + 12)
            ),
            torch.cat(
                (
                    torch.tensor([0.0, 1.0, 2.0]),
                    torch.tensor([2.0, 4.0, 6.0]),
                    torch.tensor([6.0, 8.0, 10.0]),
                    torch.tensor([12.0, 14.0, 16.0]),
                )
            ),
            3,
        )
        self.audio_out = _Projection(
            torch.cat(
                (audio_rows, audio_rows + 1, audio_rows + 3, audio_rows + 7)
            ),
            torch.cat(
                (
                    torch.tensor([0.0, 1.0]),
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([3.0, 5.0]),
                    torch.tensor([7.0, 9.0]),
                )
            ),
            2,
        )


class _BakedPDDLayer(_PDDLayer):
    '''ComfyUI 0.42+ builds PDD heads at full width and records video_dim.'''

    video_dim = 3

    def __init__(self):
        super().__init__()
        self.video_out.out_features = self.video_out.weight.shape[0]
        self.audio_out.out_features = self.audio_out.weight.shape[0]


class _CurrentIdentityLayer:
    def __init__(self):
        self.received = None

    def forward(
        self,
        x,
        _t_emb,
        video_seg,
        audio_seg,
        sigma,
        sample_sigmas,
        shifts,
    ):
        self.received = (sigma, sample_sigmas, shifts)
        return x[video_seg[0]:video_seg[1]], x[audio_seg[0]:audio_seg[1]]


class _Patcher:
    def __init__(self):
        self.object_patches = {}
        self.model_options = {'transformer_options': {}}

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


class FinalLayerTests(unittest.TestCase):
    def _assert_matches_stock(self, segments):
        layer = _Layer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        expected = layer.forward(x, None, *segments)
        actual = final_layer.chunked_final_layer(
            layer, x, None, *segments, chunk_rows=3
        )
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-4, rtol=0))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-4, rtol=0))

    def test_ragged_scalar_selectors_match_stock(self):
        self._assert_matches_stock(((0, 7, 0), (7, 11, 1)))

    def test_per_token_selectors_match_stock(self):
        self._assert_matches_stock((
            (0, 7, torch.tensor([0, 1, 0, 1, 1, 0, 1])),
            (7, 11, torch.tensor([1, 0, 0, 1])),
        ))

    def test_per_token_modulation_is_gathered_per_chunk(self):
        layer = _Layer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        segments = (
            (0, 7, torch.tensor([0, 1, 0, 1, 1, 0, 1])),
            (7, 11, torch.tensor([1, 0, 0, 1])),
        )
        expected = layer.forward(x, None, *segments)
        gathered = []

        class Table:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, index):
                gathered.append(int(index.numel()))
                return self.values[index]

        tables = tuple(Table(value) for value in layer.adaln_proj(None))
        with mock.patch.object(layer, 'adaln_proj', return_value=tables):
            actual = final_layer.chunked_final_layer(
                layer, x, None, *segments, chunk_rows=3
            )
        # Two tables, chunks of 3+3+1 and 3+1 rows: never a whole stream.
        self.assertEqual(gathered, [3, 3, 3, 3, 1, 1, 3, 3, 1, 1])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=0)

    def test_matches_core_final_layer_for_every_selector_kind(self):
        # The fakes above mirror core by hand; this pins the real module so a
        # core change to selector semantics fails here, not in a masked run.
        import comfy.ops
        from comfy.ldm.minimax.model import FinalLayer

        torch.manual_seed(7)
        layer = FinalLayer(
            32, 24, 6, 4, 1e-6, dtype=torch.float32, device='cpu',
            operations=comfy.ops.disable_weight_init,
        )
        for parameter in layer.parameters():
            parameter.detach().copy_(torch.randn_like(parameter) * 0.1)
        x = torch.randn(20, 32)
        t_emb = torch.randn(3, 24)
        cases = {
            'scalar': ((0, 13, 1), (13, 20, 2)),
            'per_token': ((0, 13, torch.randint(0, 3, (13,))), (13, 20, 2)),
            'both_per_token': (
                (0, 13, torch.randint(0, 3, (13,))),
                (13, 20, torch.randint(0, 3, (7,))),
            ),
            'empty_audio': ((0, 20, torch.randint(0, 3, (20,))), (20, 20, 0)),
        }
        for name, segments in cases.items():
            with self.subTest(name):
                expected = layer.forward(
                    x, t_emb, *segments, None, None, _core_schedule(1)
                )
                actual = final_layer.chunked_final_layer(
                    layer, x, t_emb, *segments, chunk_rows=5,
                )
                for a, b in zip(actual, expected):
                    self.assertEqual(a.shape, b.shape)
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_chunked_streams_do_not_concatenate(self):
        layer = _Layer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        segments = ((0, 7, 0), (7, 11, 1))
        expected = layer.forward(x, None, *segments)
        with mock.patch.object(torch, 'cat', side_effect=AssertionError('cat')):
            actual = final_layer.chunked_final_layer(
                layer, x, None, *segments, chunk_rows=3
            )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=0)

    def test_pdd_head_is_blended_once_per_stream(self):
        layer = _PDDLayer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        args = (
            layer, x, None, (0, 7, 0), (7, 11, 1), 3,
            torch.tensor(0.75), torch.tensor([1.0, 0.75, 0.25, 0.0]),
            (1.0, 2.0),
        )
        single = final_layer.chunked_final_layer(*args[:5], 11, *args[6:])
        with mock.patch.object(
            final_layer, '_pdd_weights', wraps=final_layer._pdd_weights,
        ) as blend:
            chunked = final_layer.chunked_final_layer(*args)
        self.assertEqual(
            [call.args[0] for call in blend.call_args_list],
            [layer.video_out, layer.audio_out],
        )
        for a, b in zip(chunked, single):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=0)

    def test_empty_stream_matches_stock(self):
        self._assert_matches_stock(((0, 11, 0), (11, 11, 1)))

    def test_forward_supports_current_seven_argument_contract(self):
        layer = _Layer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        segments = ((0, 7, 0), (7, 11, 1))
        expected = layer.forward(x, None, *segments)
        actual = final_layer.make_forward(layer, 3)(
            x,
            None,
            *segments,
            torch.tensor(0.75),
            torch.tensor([1.0, 0.75, 0.25, 0.0]),
            (12.0, 3.0),
        )
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-4, rtol=0))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-4, rtol=0))

    def test_pdd_head_bank_matches_schedule_weighted_blend(self):
        self._assert_pdd_head_bank_blend(_PDDLayer(), (1.0, 2.0))

    def test_baked_pdd_head_bank_matches_schedule_weighted_blend(self):
        self._assert_pdd_head_bank_blend(_BakedPDDLayer(), (1.0, 2.0))

    def test_grid_schedule_matches_the_equivalent_flow_shifts(self):
        # ComfyUI 0.45 passes the shifted grids instead of the shifts. Core
        # weighs them in float32 rather than float64, so compare relatively.
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        sample_sigmas = torch.tensor([1.0, 0.75, 0.25, 0.0])
        for layer in (_PDDLayer(), _BakedPDDLayer()):
            for sigma in sample_sigmas[:-1]:
                args = (x, None, (0, 7, 0), (7, 11, 1), sigma, sample_sigmas)
                forward = final_layer.make_forward(layer, 3)
                shifted = forward(*args, (1.0, 2.0))
                gridded = forward(
                    *args, (_shifted_grid(4, 1.0), _shifted_grid(4, 2.0))
                )
                for want, got in zip(shifted, gridded):
                    torch.testing.assert_close(got, want, atol=0, rtol=1e-6)

    def test_one_stream_layer_returns_one_projection(self):
        # MiniMax H3 Image has no audio head and passes audio_seg=None.
        layer = _BakedPDDLayer()
        layer.audio_out = None
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        grid = [1.0, 0.9, 0.6, 0.25, 0.0]
        sample_sigmas = torch.tensor([1.0, 0.6, 0.0])
        for chunk_rows in (3, None):
            with self.subTest(chunk_rows=chunk_rows):
                output = final_layer.make_forward(layer, chunk_rows or 11)(
                    x, None, (0, 7, 0), None,
                    torch.tensor(0.6), sample_sigmas, (grid,),
                )
                self.assertIsInstance(output, tuple)
                self.assertEqual(len(output), 1)
                # sigma 0.6 -> 0.0 spans heads 2 and 3 of this grid.
                shift, scale = layer.adaln_proj(None)
                value = (layer.norm(x[0:7]) * (1.0 + scale[0]) + shift[0]).float()
                weights = layer.video_out.weight.reshape(4, -1, 4)
                biases = layer.video_out.bias.reshape(4, -1)
                blend = (0.35 / 0.6, 0.25 / 0.6)
                expected = torch.nn.functional.linear(
                    value,
                    weights[0] + blend[0] * weights[2] + blend[1] * weights[3],
                    biases[0] + blend[0] * biases[2] + blend[1] * biases[3],
                )
                torch.testing.assert_close(output[0], expected, atol=1e-4, rtol=0)

    def _assert_pdd_head_bank_blend(self, layer, schedule):
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        video_seg = (0, 7, 0)
        audio_seg = (7, 11, 1)
        actual = final_layer.make_forward(layer, 3)(
            x,
            None,
            video_seg,
            audio_seg,
            torch.tensor(0.75),
            torch.tensor([1.0, 0.75, 0.25, 0.0]),
            schedule,
        )

        shift, scale = layer.adaln_proj(None)

        def expected(segment, output, blend):
            first, last, row = segment
            value = (
                layer.norm(x[first:last]) * (1.0 + scale[row]) + shift[row]
            ).float()
            weights = output.weight.reshape(4, -1, 4)
            biases = output.bias.reshape(4, -1)
            return torch.nn.functional.linear(
                value,
                weights[0] + blend[0] * weights[1] + blend[1] * weights[2],
                biases[0] + blend[0] * biases[1] + blend[1] * biases[2],
            )

        self.assertTrue(
            torch.allclose(
                expected(video_seg, layer.video_out, (0.5, 0.5)),
                actual[0],
                atol=1e-4,
                rtol=0,
            )
        )
        self.assertTrue(
            torch.allclose(
                expected(audio_seg, layer.audio_out, (5.0 / 12.0, 7.0 / 12.0)),
                actual[1],
                atol=1e-4,
                rtol=0,
            )
        )

    def test_pdd_head_bank_requires_sample_sigmas(self):
        layer = _PDDLayer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        with self.assertRaisesRegex(ValueError, "sampler's sigma schedule"):
            final_layer.make_forward(layer, 3)(
                x,
                None,
                (0, 7, 0),
                (7, 11, 1),
                torch.tensor(0.75),
                None,
                (1.0, 1.0),
            )

    def test_pdd_head_bank_matches_installed_comfy_final_layer(self):
        heads = 4
        baked = 'pdd_steps' in inspect.signature(minimax.FinalLayer.__init__).parameters
        torch.manual_seed(0)
        layer = minimax.FinalLayer(
            16, 8, 12, 4, 1e-5,
            operations=comfy.ops.disable_weight_init,
            **({'pdd_steps': heads} if baked else {}),
        )
        for module in layer.modules():
            for name, value in list(module.named_parameters(recurse=False)):
                setattr(module, name, torch.nn.Parameter(
                    torch.randn_like(value) * 0.1, requires_grad=False
                ))
        if not baked:
            # Before ComfyUI 0.42 only a PDD LoRA grows the head rows.
            for head in (layer.video_out, layer.audio_out):
                head.weight = torch.nn.Parameter(
                    torch.randn(head.weight.shape[0] * heads, head.weight.shape[1]) * 0.1,
                    requires_grad=False,
                )
                head.bias = torch.nn.Parameter(
                    torch.randn(head.bias.shape[0] * heads) * 0.1, requires_grad=False
                )
        x = torch.randn(23, 16)
        t_emb = torch.randn(2, 8)
        segments = ((0, 13, 0), (13, 20, torch.tensor([0, 1, 1, 0, 1, 0, 0])))
        sample_sigmas = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])
        forward = final_layer.make_forward(layer, 3)
        schedule = _core_schedule(heads)
        with torch.inference_mode():
            for sigma in sample_sigmas[:-1]:
                expected = layer(x, t_emb, *segments, sigma, sample_sigmas, schedule)
                actual = forward(x, t_emb, *segments, sigma, sample_sigmas, schedule)
                for want, got in zip(expected, actual):
                    self.assertEqual(want.shape, got.shape)
                    self.assertTrue(torch.allclose(want, got, atol=1e-5, rtol=0))

    @unittest.skipUnless(
        hasattr(minimax, 'MiniMaxH3ImageModel'),
        'MiniMax H3 Image needs ComfyUI 0.45+',
    )
    def test_one_stream_pdd_head_bank_matches_installed_comfy_final_layer(self):
        # The image checkpoint has no audio head and its own sigma grid.
        grid = [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.0]
        torch.manual_seed(1)
        layer = minimax.FinalLayer(
            16, 8, 12, None, 1e-5,
            operations=comfy.ops.disable_weight_init, pdd_steps=len(grid) - 1,
        )
        for module in layer.modules():
            for name, value in list(module.named_parameters(recurse=False)):
                setattr(module, name, torch.nn.Parameter(
                    torch.randn_like(value) * 0.1, requires_grad=False
                ))
        x = torch.randn(17, 16)
        t_emb = torch.randn(2, 8)
        segment = (3, 17, 1)
        sample_sigmas = torch.tensor([1.0, 0.95, 0.8, 0.5, 0.2, 0.0])
        forward = final_layer.make_forward(layer, 4)
        with torch.inference_mode():
            for sigma in sample_sigmas[:-1]:
                expected = layer(x, t_emb, segment, None, sigma, sample_sigmas, (grid,))
                actual = forward(x, t_emb, segment, None, sigma, sample_sigmas, (grid,))
                self.assertEqual(len(actual), len(expected))
                self.assertEqual(actual[0].shape, expected[0].shape)
                torch.testing.assert_close(actual[0], expected[0], atol=1e-5, rtol=0)

    def test_install_is_owned_and_idempotent(self):
        patcher = _Patcher()
        model = SimpleNamespace(final_layer=_Layer())
        with mock.patch.object(
            final_layer, 'get_minimax_h3_model', return_value=model
        ):
            self.assertTrue(final_layer.install(patcher, 4096))
            self.assertFalse(final_layer.install(patcher, 4096))
            self.assertTrue(final_layer.install(patcher, 2048))
            self.assertEqual(
                getattr(
                    patcher.object_patches[final_layer.FINAL_LAYER_KEY],
                    final_layer.SIGNATURE_MARKER,
                ),
                2048,
            )

    def test_foreign_final_layer_patch_is_preserved(self):
        patcher = _Patcher()
        foreign = lambda *_args: 'foreign'
        patcher.object_patches[final_layer.FINAL_LAYER_KEY] = foreign
        model = SimpleNamespace(final_layer=_Layer())
        with mock.patch.object(
            final_layer, 'get_minimax_h3_model', return_value=model
        ):
            self.assertFalse(final_layer.install(patcher, 4096))
        self.assertIs(
            patcher.object_patches[final_layer.FINAL_LAYER_KEY],
            foreign,
        )

    def test_cube_restore_composes_with_chunking(self):
        layer = _Layer()
        state = CubeOrderState()
        topology = CubeOrderTopology(
            (1, 2, 2),
            (2, 0, 3, 1),
            (1, 3, 0, 2),
        )
        token = state.enter(topology)
        state.leave(token, topology, True)
        raster_video = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        audio = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
        cube_video = raster_video.index_select(0, torch.tensor(topology.forward))

        actual = final_layer.make_forward(layer, 2, cube_state=state)(
            torch.cat((audio, cube_video)),
            None,
            (2, 6, 0),
            (0, 2, 1),
        )
        expected = layer.forward(
            torch.cat((audio, raster_video)),
            None,
            (2, 6, 0),
            (0, 2, 1),
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_clearing_cube_state_retains_owned_chunking(self):
        patcher = _Patcher()
        model = SimpleNamespace(final_layer=_Layer())
        state = CubeOrderState()
        with mock.patch.object(
            final_layer, 'get_minimax_h3_model', return_value=model
        ):
            self.assertTrue(final_layer.install(patcher, 3))
            self.assertTrue(final_layer.install(patcher, 3, cube_state=state))
            self.assertTrue(final_layer.clear_cube_state(patcher, state))

        current = patcher.object_patches[final_layer.FINAL_LAYER_KEY]
        self.assertEqual(getattr(current, final_layer.SIGNATURE_MARKER), 3)
        self.assertIsNone(getattr(current, final_layer.CUBE_STATE_MARKER))

    def test_clearing_order_only_patch_restores_native_forward(self):
        patcher = _Patcher()
        model = SimpleNamespace(final_layer=_Layer())
        state = CubeOrderState()
        with mock.patch.object(
            final_layer, 'get_minimax_h3_model', return_value=model
        ):
            self.assertTrue(final_layer.install(patcher, cube_state=state))
            self.assertTrue(final_layer.clear_cube_state(patcher, state))

        self.assertNotIn(final_layer.FINAL_LAYER_KEY, patcher.object_patches)

    def test_order_only_wrapper_preserves_the_current_forward_contract(self):
        layer = _CurrentIdentityLayer()
        state = CubeOrderState()
        topology = CubeOrderTopology(
            (1, 2, 2),
            (2, 0, 3, 1),
            (1, 3, 0, 2),
        )
        token = state.enter(topology)
        state.leave(token, topology, True)
        sigma = torch.tensor(0.5)
        sample_sigmas = torch.tensor([1.0, 0.5, 0.0])
        shifts = (3.0, 4.0)

        final_layer.make_forward(layer, cube_state=state)(
            torch.arange(6).unsqueeze(1),
            None,
            (2, 6, 0),
            (0, 2, 0),
            sigma,
            sample_sigmas,
            shifts,
        )

        self.assertIs(layer.received[0], sigma)
        self.assertIs(layer.received[1], sample_sigmas)
        self.assertIs(layer.received[2], shifts)

    def test_real_model_patcher_dispatches_current_forward_contract(self):
        root = torch.nn.Module()
        root.diffusion_model = torch.nn.Module()
        layer = torch.nn.Module()
        implementation = _Layer()
        layer.norm = implementation.norm
        layer.video_out = implementation.video_out
        layer.audio_out = implementation.audio_out
        layer.adaln_proj = implementation.adaln_proj
        layer.forward = implementation.forward
        root.diffusion_model.final_layer = layer
        patcher = ModelPatcher(root, torch.device('cpu'), torch.device('cpu'))

        with mock.patch.object(
            final_layer,
            'get_minimax_h3_model',
            return_value=root.diffusion_model,
        ):
            final_layer.install(patcher, 3)
        patcher.patch_model(load_weights=False)
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        with mock.patch.object(final_layer.logging, 'debug') as debug:
            root.diffusion_model.final_layer(
                x,
                None,
                (0, 7, 0),
                (7, 11, 1),
                torch.tensor(0.75),
                torch.tensor([1.0, 0.75, 0.25, 0.0]),
                (12.0, 3.0),
            )
        patcher.unpatch_model(unpatch_weights=False)

        self.assertEqual(len(debug.call_args_list), 1)
        self.assertIn('chunked FinalLayer ran', debug.call_args.args[0])

    def test_memory_plan_installs_final_layer_even_when_mlp_is_off(self):
        patcher = _Patcher()
        plan = H3OptimizationPlan(
            memory=MemoryRequest(mlp_memory='off', chunk_rows=4096)
        )
        disabled = MLPProviderResolution('off', 'off', 'disabled')
        with mock.patch.object(
            apply_module, 'install_final_layer'
        ) as install, mock.patch.object(
            apply_module, 'resolve_mlp_provider', return_value=disabled
        ):
            resolution, patched_blocks = apply_module._install_mlp(
                patcher, plan, object(), object()
            )

        install.assert_called_once_with(patcher, 4096)
        self.assertIs(resolution, disabled)
        self.assertEqual(patched_blocks, 0)


if __name__ == '__main__':
    unittest.main()


class FinalLayerExecutionLogTests(unittest.TestCase):
    '''The patched forward must announce that it actually ran.

    Regression cover for a real diagnostic gap: the install-time message proves
    only that the patch was attached. Three benchmark runs looked correctly
    configured while routing sent the forward elsewhere, and nothing in a normal
    workflow said so.
    '''

    @staticmethod
    def _run(chunk_rows, segments, rows=11, calls=1):
        layer = _Layer()
        forward = final_layer.make_forward(layer, chunk_rows)
        x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        with mock.patch.object(final_layer.logging, 'debug') as debug:
            for _ in range(calls):
                forward(x, None, *segments)
        return [call.args for call in debug.call_args_list]

    def test_first_execution_logs_once(self):
        logged = self._run(3, ((0, 7, 0), (7, 11, 1)))
        self.assertEqual(len(logged), 1)
        self.assertIn('chunked FinalLayer ran', logged[0][0])

    def test_repeat_executions_stay_quiet(self):
        # A 20-step sampler must not emit 20 identical lines.
        logged = self._run(3, ((0, 7, 0), (7, 11, 1)), calls=20)
        self.assertEqual(len(logged), 1)

    def test_log_reports_rows_and_chunk_counts(self):
        logged = self._run(3, ((0, 7, 0), (7, 11, 1)))
        args = logged[0]
        self.assertEqual(args[1], 11)     # total rows
        self.assertEqual(args[2], 7)      # video rows
        self.assertEqual(args[3], 3)      # video chunks: ceil(7/3)
        self.assertEqual(args[4], 4)      # audio rows
        self.assertEqual(args[5], 2)      # audio chunks: ceil(4/3)
        self.assertEqual(args[6], 3)      # chunk_rows

    def test_single_chunk_is_visible_as_such(self):
        # Bounded in name only: the segment fits in one chunk, so the log must
        # not imply the activation memory was actually split.
        logged = self._run(4096, ((0, 7, 0), (7, 11, 1)))
        args = logged[0]
        self.assertEqual(args[3], 1)
        self.assertEqual(args[5], 1)

    def test_empty_audio_segment_reports_zero_chunks(self):
        logged = self._run(3, ((0, 11, 0), (11, 11, 1)))
        args = logged[0]
        self.assertEqual(args[4], 0)
        self.assertEqual(args[5], 0)

    def test_logging_does_not_change_the_result(self):
        layer = _Layer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        segments = ((0, 7, 0), (7, 11, 1))
        expected = layer.forward(x, None, *segments)
        actual = final_layer.make_forward(layer, 3)(x, None, *segments)
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-4, rtol=0))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-4, rtol=0))

    def test_chunk_count_helper(self):
        self.assertEqual(final_layer._chunk_count(0, 64), 0)
        self.assertEqual(final_layer._chunk_count(1, 64), 1)
        self.assertEqual(final_layer._chunk_count(64, 64), 1)
        self.assertEqual(final_layer._chunk_count(65, 64), 2)
