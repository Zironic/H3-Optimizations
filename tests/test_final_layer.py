'''Production contracts for bounded H3 FinalLayer execution.'''

from types import SimpleNamespace
from pathlib import Path
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

from h3_optimizations.memory import final_layer
import h3_optimizations.apply as apply_module
from h3_optimizations.plan import (
    EMBEDDING_MEMORY_RELEASE,
    H3OptimizationPlan,
    MemoryRequest,
)
from h3_optimizations.qkv.providers import MLPProviderResolution
from comfy.model_patcher import ModelPatcher

sys.argv = [sys.argv[0], *TEST_ARGS]


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

    def test_empty_stream_matches_stock(self):
        self._assert_matches_stock(((0, 11, 0), (11, 11, 1)))

    def test_forward_supports_legacy_four_argument_contract(self):
        layer = _Layer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        segments = ((0, 7, 0), (7, 11, 1))
        expected = layer.forward(x, None, *segments)
        actual = final_layer.make_forward(layer, 3)(x, None, *segments)
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-4, rtol=0))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-4, rtol=0))

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
        layer = _PDDLayer()
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
            (1.0, 2.0),
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
            apply_module, 'install_embedding_memory'
        ) as install_embedding, mock.patch.object(
            apply_module, 'resolve_mlp_provider', return_value=disabled
        ):
            resolution, patched_blocks = apply_module._install_mlp(
                patcher, plan, object(), object()
            )

        install.assert_called_once_with(patcher, 4096)
        install_embedding.assert_called_once_with(patcher)
        self.assertIs(resolution, disabled)
        self.assertEqual(patched_blocks, 0)

    def test_explicit_embedding_release_is_strict(self):
        patcher = _Patcher()
        plan = H3OptimizationPlan(
            memory=MemoryRequest(
                mlp_memory='off',
                embedding_memory=EMBEDDING_MEMORY_RELEASE,
            )
        )
        disabled = MLPProviderResolution('off', 'off', 'disabled')
        with mock.patch.object(
            apply_module, 'install_final_layer'
        ), mock.patch.object(
            apply_module, 'install_embedding_memory'
        ) as install_embedding, mock.patch.object(
            apply_module, 'resolve_mlp_provider', return_value=disabled
        ):
            apply_module._install_mlp(patcher, plan, object(), object())

        install_embedding.assert_called_once_with(patcher, strict=True)


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
