'''CPU contracts for bounded MLP execution and ConvRot two-slice math.'''

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import DiTBlock  # noqa: E402

from h3_optimizations.memory import chunks  # noqa: E402
from h3_optimizations.memory.config import (  # noqa: E402
    MODE_BF16,
    MODE_CONVROT_2SLICE,
    MODE_NATIVE,
    ActivationMemoryConfig,
)
from h3_optimizations.memory.forward import make_forward  # noqa: E402
import h3_optimizations.memory.forward as forward_module  # noqa: E402
from h3_optimizations.normalized_rows import NormalizedRowsUnsupported  # noqa: E402
import h3_optimizations.memory.linear as linear_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class MemoryTests(unittest.TestCase):
    def test_config_contains_no_epilogue_mode(self):
        config = ActivationMemoryConfig()
        self.assertEqual(config.mode, MODE_NATIVE)
        self.assertEqual(config.chunk_rows, 4096)
        self.assertTrue(
            ActivationMemoryConfig(
                mode=MODE_CONVROT_2SLICE
            ).convrot_2slice
        )
        with self.assertRaises(ValueError):
            ActivationMemoryConfig(
                mode='mlp_chunked_convrot_epilogue'
            )
        compatibility = ActivationMemoryConfig(
            mode=MODE_BF16,
            prefer_held_weights=False,
        )
        self.assertTrue(compatibility.bf16_swiglu)
        self.assertFalse(compatibility.prefer_held_weights)

    def test_chunk_planner_preserves_modulation_boundaries(self):
        result = list(
            chunks.iter_mod_chunks(
                [(0, 5, 0), (5, 19, 1)],
                19,
                max_rows=8,
                alignment=4,
            )
        )
        self.assertEqual(
            [
                (chunk.start, chunk.stop, chunk.mod_row)
                for chunk in result
            ],
            [(0, 5, 0), (5, 13, 1), (13, 19, 1)],
        )
        with self.assertRaisesRegex(ValueError, 'gap'):
            chunks.validate_mod_segments(
                [(0, 4, 0), (5, 8, 1)],
                8,
            )
        with self.assertRaisesRegex(ValueError, 'overlap'):
            chunks.validate_mod_segments(
                [(0, 5, 0), (4, 8, 1)],
                8,
            )

    def test_chunk_planner_slices_per_token_modulation_selector(self):
        selector = torch.tensor(
            [0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9, 0, 3],
            dtype=torch.long,
        )
        result = list(
            chunks.iter_mod_chunks(
                [(0, 5, 1), (5, 19, selector)],
                19,
                max_rows=8,
                alignment=4,
                mod_rows=12,
            )
        )
        self.assertEqual(
            [(chunk.start, chunk.stop) for chunk in result],
            [(0, 5), (5, 13), (13, 19)],
        )
        self.assertEqual(result[0].mod_row, 1)
        self.assertTrue(torch.equal(result[1].mod_row, selector[:8]))
        self.assertTrue(torch.equal(result[2].mod_row, selector[8:]))

        with self.assertRaisesRegex(ValueError, 'has 3 rows, expected 4'):
            chunks.validate_mod_segments(
                [(0, 4, torch.tensor([0, 1, 2], dtype=torch.long))],
                4,
            )
        with self.assertRaisesRegex(TypeError, 'integer dtype'):
            chunks.validate_mod_segments(
                [(0, 4, torch.zeros(4, dtype=torch.float32))],
                4,
            )

    def test_acquired_weight_release_is_exactly_once(self):
        acquired = linear_module.AcquiredLinear(
            'module',
            'weight',
            'bias',
            'handle',
        )
        with patch.object(
            comfy.ops,
            'uncast_bias_weight',
        ) as release:
            acquired.release()
            acquired.release()
        release.assert_called_once_with(
            'module',
            'weight',
            'bias',
            'handle',
        )

    def test_forced_bf16_held_mlp_disables_requantization(self):
        fc1 = object()
        fc2 = object()
        mlp = type('MLP', (), {'fc1': fc1, 'fc2': fc2})()
        sample = torch.empty((1, 8), dtype=torch.bfloat16)
        weights = iter(
            (
                (torch.empty((16, 8), dtype=torch.bfloat16), None, 'fc1'),
                (torch.empty((8, 8), dtype=torch.bfloat16), None, 'fc2'),
            )
        )
        with patch.object(
            comfy.ops,
            'cast_bias_weight',
            side_effect=lambda *_args, **_kwargs: next(weights),
        ) as acquire, patch.object(comfy.ops, 'uncast_bias_weight'):
            with linear_module.HeldMLP(mlp, sample, force_bf16=True):
                pass
        self.assertEqual(acquire.call_count, 2)
        self.assertTrue(
            all(call.kwargs['want_requant'] is False for call in acquire.call_args_list)
        )

    def test_convrot_two_slice_matches_unsliced_fake_math(self):
        class FakeQuantized:
            def __init__(self, qdata, scale):
                self.qdata = qdata
                self.scale = scale
                self._layout_cls = 'TensorWiseINT8Layout'
                self._params = type(
                    'Params',
                    (),
                    {
                        'transposed': False,
                        'convrot': True,
                        'convrot_groupsize': 256,
                    },
                )()

        class FakeLayout:
            @staticmethod
            def get_plain_tensors(weight):
                return weight.qdata, weight.scale

        class FakeLinear:
            def __init__(self, weight):
                self.weight = weight

        hidden = 256
        ffn = 512
        torch.manual_seed(27)
        fc1_q = torch.randint(
            -1,
            2,
            (ffn * 2, hidden),
            dtype=torch.int8,
        )
        fc2_q = torch.randint(
            -1,
            2,
            (hidden, ffn),
            dtype=torch.int8,
        )
        mlp = type(
            'MLP',
            (),
            {
                'fc1': FakeLinear(
                    FakeQuantized(fc1_q, torch.ones(ffn * 2))
                ),
                'fc2': FakeLinear(
                    FakeQuantized(fc2_q, torch.ones(hidden))
                ),
            },
        )()

        def fake_convrot(x, qdata, _scale, input_act=None):
            if input_act == 'swiglu':
                gate, up = x.chunk(2, dim=-1)
                x = torch.nn.functional.silu(gate) * up
            return x @ qdata.to(x.dtype).t()

        def fake_cast(module, _sample, **_kwargs):
            return module.weight, None, None

        x = torch.randn(3, hidden, dtype=torch.bfloat16) * 0.01
        with patch.object(
            linear_module,
            'QuantizedTensor',
            FakeQuantized,
        ), patch.object(
            linear_module,
            'TensorWiseINT8Layout',
            FakeLayout,
        ), patch.object(
            comfy.ops,
            'cast_bias_weight',
            side_effect=fake_cast,
        ), patch.object(
            comfy.ops,
            'uncast_bias_weight',
        ):
            with linear_module.ConvRotTwoSliceMLP(
                mlp,
                x[:1],
                fake_convrot,
            ) as session:
                actual, path = session.fc1_fc2(x)

        gate = x @ fc1_q[:ffn].to(torch.bfloat16).t()
        up = x @ fc1_q[ffn:].to(torch.bfloat16).t()
        expected = (
            torch.nn.functional.silu(gate) * up
        ) @ fc2_q.to(torch.bfloat16).t()
        self.assertEqual(actual.shape, (3, hidden))
        self.assertTrue(
            torch.allclose(actual, expected, atol=0.25, rtol=0.0)
        )
        self.assertEqual(path, 'held_convrot_2slice')
        self.assertIsNone(session.tiles)

    @staticmethod
    def _make_block():
        block = DiTBlock(
            hidden=32,
            heads=2,
            head_dim=16,
            ffn=48,
            t_dim=24,
            eps=1e-6,
            qk_eps=1e-6,
            dtype=torch.float32,
            device='cpu',
            operations=comfy.ops.disable_weight_init,
        )
        for parameter in block.parameters():
            parameter.detach().copy_(
                torch.randn_like(parameter) * 0.03
            )
            parameter.requires_grad_(False)
        return block

    def test_generic_chunked_forward_matches_core(self):
        torch.manual_seed(2)
        block = self._make_block()

        torch.manual_seed(3)
        x = torch.randn(19, 32) * 0.1
        t_emb = torch.randn(1, 24) * 0.1
        segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
        expected = block.forward(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        actual = make_forward(
            block,
            0,
            ActivationMemoryConfig(
                mode=MODE_NATIVE,
                chunk_rows=256,
                alignment=256,
            ),
        )(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-5, atol=2e-6)
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_masked_per_token_forward_matches_core_across_chunks(self):
        torch.manual_seed(31)
        block = self._make_block()

        torch.manual_seed(32)
        x = torch.randn(300, 32) * 0.1
        # Four timestep embeddings produce twelve modulation rows (3 modalities
        # per timestep), matching ComfyUI's per-token noise-mask representation.
        t_emb = torch.randn(4, 24) * 0.1
        selector = torch.tensor(
            ([0, 3, 6, 9] * 65),
            dtype=torch.long,
        )
        segments = [
            (0, 20, 1),
            (20, 280, selector),
            (280, 300, 2),
        ]
        expected = block.forward(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        actual = make_forward(
            block,
            0,
            ActivationMemoryConfig(
                mode=MODE_NATIVE,
                chunk_rows=256,
                alignment=256,
            ),
        )(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-5, atol=2e-6)
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_attention_attribute_error_is_not_treated_as_row_fallback(self):
        torch.manual_seed(33)
        block = self._make_block()
        block.attn.forward = Mock(
            side_effect=AttributeError('attention implementation bug')
        )
        forward = make_forward(
            block,
            0,
            ActivationMemoryConfig(
                mode=MODE_NATIVE,
                chunk_rows=256,
                alignment=256,
            ),
        )
        with self.assertRaisesRegex(AttributeError, 'implementation bug'):
            forward(
                torch.randn(8, 32),
                torch.randn(1, 24),
                [(0, 8, 0)],
                rope_freqs=None,
                transformer_options={},
            )

    def test_attention_row_fallback_is_logged_once_across_blocks_and_retried(self):
        torch.manual_seed(34)
        attention_inputs = []
        reason = 'test consumer requires a tensor'
        forward_module._ATTENTION_FALLBACK_LOGGED.discard(reason)

        def make_attention_block():
            block = self._make_block()

            def attention(value, **_kwargs):
                if not torch.is_tensor(value):
                    raise NormalizedRowsUnsupported(reason)
                attention_inputs.append(value)
                return torch.zeros_like(value)

            block.attn.forward = Mock(side_effect=attention)
            return block

        forwards = [
            make_forward(
                make_attention_block(),
                layer,
                ActivationMemoryConfig(
                    mode=MODE_NATIVE,
                    chunk_rows=256,
                    alignment=256,
                ),
            )
            for layer in (7, 8)
        ]
        with self.assertLogs(level='INFO') as captured:
            for forward in forwards:
                forward(
                    torch.randn(8, 32),
                    torch.randn(1, 24),
                    [(0, 8, 0)],
                    rope_freqs=None,
                    transformer_options={},
                )

        messages = [
            message for message in captured.output
            if 'Norm1 fusion is unavailable' in message
        ]
        self.assertEqual(len(messages), 1)
        self.assertIn('block 7', messages[0])
        self.assertIn('slightly more VRAM', messages[0])
        self.assertIn(reason, messages[0])
        self.assertEqual(len(attention_inputs), 2)
        self.assertTrue(all(torch.is_tensor(value) for value in attention_inputs))


if __name__ == '__main__':
    unittest.main()
