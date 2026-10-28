"""Packed-Q staging must finish before a block overwrites its residual."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1])]
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from h3_optimizations import kitchen_qkv
from h3_optimizations.attention_forward import make_forward
from h3_optimizations.kitchen_qkv import ChunkedKitchenAttentionBackend, PreparedStreamedKitchenQKV
from h3_optimizations.native.int8_attention import PrequantizedInt8Attention
sys.argv = [sys.argv[0], *TEST_ARGS]


class StagedKitchenTests(unittest.TestCase):
    def test_staging_preserves_output_forward_hooks_and_releases_on_error(self):
        for finish in ('complete', 'close', 'error', 'projection_error'):
            with self.subTest(finish=finish):
                torch.manual_seed(5)
                x = (torch.arange(513 * 256).reshape(513, 256) % 7).to(torch.bfloat16)
                weight = QuantizedTensor.from_float(
                    torch.randn(256, 256, dtype=torch.bfloat16) * 0.01,
                    'TensorWiseINT8Layout', scale='recalculate', is_weight=True,
                    per_channel=True, convrot=True, convrot_groupsize=256,
                )
                class OutputProjection(comfy.ops.manual_cast.Linear):
                    def forward(self, value):
                        if finish == 'projection_error':
                            raise RuntimeError('attention failed')
                        return super().forward(value) + 1
                out = OutputProjection(256, 256, bias=False, dtype=torch.bfloat16)
                out.weight = torch.nn.Parameter(weight, requires_grad=False)
                output_calls = []
                out.register_forward_hook(lambda _module, _args, _result: output_calls.append(True))
                module = SimpleNamespace(heads=2, head_dim=128, out_proj=out)
                carrier = PrequantizedInt8Attention(
                    q=torch.empty(1, 2, 1, 128, dtype=torch.int8),
                    k=torch.empty(1, 2, 513, 128, dtype=torch.int8),
                    v=torch.empty(256, 513, dtype=torch.int8),
                    q_scale=torch.empty(1, 2, 1), k_scale=torch.ones(1), v_scale=torch.ones(1),
                    original_head_dim=128, input_dtype=x.dtype, attention_scale=1.0, cta_k=64,
                )
                prepared = PreparedStreamedKitchenQKV(module, x, None, carrier, 'native', x)
                projected_ranges = []
                def project(_held, source, _rope, start, stop):
                    projected_ranges.append((start, stop))
                    return source[start:stop].reshape(stop - start, 2, 128).transpose(0, 1).unsqueeze(0)
                class Kitchen:
                    @staticmethod
                    def quantize_int8_attention_q(q, **_kwargs):
                        return q.to(torch.int8), torch.ones(1, 2, ((q.shape[-2] + 127) // 128) * 32)
                    @staticmethod
                    def int8_attention_from_prequantized(chunk, **_kwargs):
                        if finish == 'error':
                            raise RuntimeError('attention failed')
                        return chunk.q.to(torch.bfloat16)
                source_weight = mock.MagicMock()
                expected = torch.nn.functional.linear(x, weight) + 1
                with (
                    mock.patch.object(kitchen_qkv, 'resolve_kitchen', return_value=Kitchen()),
                    mock.patch.object(kitchen_qkv, 'create_held_qkv', return_value=source_weight) as acquire,
                    mock.patch.object(kitchen_qkv, 'project_q_hnd', side_effect=project),
                    mock.patch.object(comfy.ops, 'cast_bias_weight', return_value=(weight, None, object())) as cast,
                    mock.patch.object(comfy.ops, 'uncast_bias_weight') as release,
                    mock.patch.object(kitchen_qkv, 'attention_output_buffer', side_effect=AssertionError('full output allocated')),
                ):
                    rows = ChunkedKitchenAttentionBackend(stream_output=True, query_chunk_rows=128).iter_projected_rows(module, prepared)
                    if finish in ('error', 'projection_error'):
                        with self.assertRaisesRegex(RuntimeError, 'attention failed'):
                            next(rows)
                    else:
                        first, last, value = next(rows)
                        self.assertEqual(len(projected_ranges), 5)
                        self.assertIsNone(prepared.x)
                        x.fill_(99)
                        torch.testing.assert_close(value, expected[first:last], rtol=0, atol=0)
                        if finish == 'close':
                            rows.close()
                        else:
                            for first, last, value in rows:
                                torch.testing.assert_close(value, expected[first:last], rtol=0, atol=0)
                    acquire.assert_called_once()
                    source_weight.__exit__.assert_called_once()
                    calls = {'complete': 5, 'close': 1, 'error': 0, 'projection_error': 0}[finish]
                    self.assertEqual(cast.call_count, calls)
                    self.assertEqual(release.call_count, calls)
                    self.assertEqual(len(output_calls), calls)
                    self.assertIsNone(prepared.carrier)
                    self.assertIsNone(prepared.output_buffer)

    def test_short_sequences_decline_but_output_hooks_do_not_block_staging(self):
        weight = QuantizedTensor.from_float(
            torch.randn(256, 256, dtype=torch.bfloat16), 'TensorWiseINT8Layout',
            scale='recalculate', is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256,
        )
        out = comfy.ops.manual_cast.Linear(256, 256, bias=False, dtype=torch.bfloat16)
        out.weight = torch.nn.Parameter(weight, requires_grad=False)
        module = SimpleNamespace(heads=2, head_dim=128, out_proj=out)
        projector = SimpleNamespace(streamed_q=True, chunk_rows=128, v_mode='retain', try_project=mock.Mock(return_value=None))
        forward = make_forward(module, 0, backend=ChunkedKitchenAttentionBackend(stream_output=True), projector=projector)
        stream = forward._h3_optimizations_projected_rows
        self.assertIsNone(stream(torch.empty(128, 256, dtype=torch.bfloat16), None, {}))
        projector.try_project.assert_not_called()
        handle = out.register_forward_hook(lambda _module, _args, result: result)
        self.assertIsNone(stream(torch.empty(513, 256, dtype=torch.bfloat16), None, {}))
        handle.remove()
        projector.try_project.assert_called_once()


if __name__ == '__main__':
    unittest.main()
