"""Block-V packing and overwrite-safe residual output contracts."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1])]
saved_args = sys.argv
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from h3_optimizations.native.block_v import SinglePassVCarrier
from h3_optimizations.native.int8_attention import PrequantizedInt8Attention
from h3_optimizations.native.v_staging import BACKEND_TORCH, _inverse_permutation_16
from h3_optimizations.kitchen_qkv import ChunkedKitchenAttentionBackend, PreparedChunkedKitchenQKV
from h3_optimizations.normalized_rows import NormalizedRows
from h3_optimizations.plan import H3OptimizationPlan, MemoryRequest, SparseRequest
from h3_optimizations.memory_migration_node import _memory_request_for_modes, KITCHEN_V_MEMORY_MODE_BLOCK
from h3_optimizations import kitchen_qkv
from h3_optimizations.native.producer import Int8AttentionProducer, finalize_int8_attention_producer
sys.argv = saved_args


def spec_for(value, cta=64):
    return SimpleNamespace(k_input_shape=tuple(value.shape), input_dtype=value.dtype,
                           device=value.device, cta_k=cta)


class BlockVTests(unittest.TestCase):
    def test_native_producer_retains_scale_granularity(self):
        spec = SimpleNamespace(q_input_shape=(1, 2, 129, 128), k_input_shape=(1, 2, 129, 128),
                               original_head_dim=128, input_dtype=torch.bfloat16,
                               attention_scale=.1, cta_k=64)
        packed = torch.empty(1, 2, 129, 128, dtype=torch.int8)
        producer = Int8AttentionProducer(
            spec, SimpleNamespace(indices=torch.zeros(1, dtype=torch.long)),
            packed, packed, torch.ones(1), torch.ones(1),
            v=torch.empty(256, 192, dtype=torch.int8), v_scale=torch.ones(1, 2, 2, 128),
            v_scale_rows=128,
        )
        producer._q_ranges.append((0, 129))
        producer._k_ranges.append((0, 129))
        carrier = finalize_int8_attention_producer(producer)
        self.assertEqual(carrier.v_scale_rows, 128)
        self.assertIs(carrier.v_scale, producer.v_scale)

    def test_producer_projects_each_chunk_once_without_v_reprojection(self):
        value = torch.randn(1, 2, 257, 128, dtype=torch.bfloat16)
        spec = spec_for(value)
        spec.kernel_head_dim = 128
        spec.q_tile, spec.k_tile, spec.k_anchor_positions = 128, 64, [0]
        producer = SimpleNamespace(v=None, v_scale=None)
        kitchen = mock.Mock()
        kitchen.create_int8_attention_producer.return_value = producer
        kitchen.finalize_int8_attention_producer.side_effect = lambda produced: produced
        module = SimpleNamespace(heads=2, head_dim=128, to_gate_compress=None, qkv_proj=None)

        def project(module, rows, rope, start, stop, projector):
            slab = value[:, :, start:stop]
            return slab, slab, slab

        with (
            mock.patch.object(kitchen_qkv, 'resolve_kitchen', return_value=kitchen),
            mock.patch.object(kitchen_qkv, 'describe_linear', return_value=SimpleNamespace(plain_float=False)),
            mock.patch.object(kitchen_qkv, '_project_anchor_samples', return_value=torch.zeros(1)),
            mock.patch.object(kitchen_qkv, 'project_chunk_hnd', side_effect=project) as projection,
            mock.patch.object(kitchen_qkv, 'project_v_hnd', side_effect=AssertionError('V reprojected')),
            mock.patch('h3_optimizations.native.block_v.available', return_value=True),
            mock.patch('h3_optimizations.native.block_v.SinglePassVCarrier',
                       side_effect=lambda spec, block_rows: SinglePassVCarrier(spec, block_rows=block_rows, backend=BACKEND_TORCH)),
        ):
            result = kitchen_qkv.run_chunked_kitchen_qkv(
                module, torch.empty(257, 256), None, layer_index=0, transformer_options={},
                spec=spec, chunk_rows=128, v_mode='blockwise',
            )
        self.assertEqual(projection.call_count, 3)
        self.assertEqual(kitchen.quantize_int8_attention_qk_chunk.call_count, 3)
        kitchen.quantize_int8_attention_v.assert_not_called()
        self.assertEqual(result.carrier.v_scale.shape, (1, 2, 3, 128))

    def test_per_block_scales_and_ragged_padding(self):
        torch.manual_seed(42)
        value = torch.randn(2, 3, 257, 128, dtype=torch.bfloat16)
        value[:, :, :128] *= .01
        value[:, :, 128:256] *= 10
        value[:, :, 256:] = 0
        for cta in (64, 128):
            with self.subTest(cta=cta):
                staging = SinglePassVCarrier(spec_for(value, cta), block_rows=128, backend=BACKEND_TORCH)
                for start in range(0, 257, 128):
                    staging.quantize(value[:, :, start:start + 128], start)
                packed, scales = staging.finish()
                positions = torch.arange(staging.padded)
                permutation = (positions & ~15) | _inverse_permutation_16(value.device)[positions & 15]
                unpacked = packed.reshape(2, 3, 128, -1).index_select(-1, permutation).transpose(-1, -2)
                for start in range(0, 257, 128):
                    stop = min(start + 128, 257)
                    scale = value[:, :, start:stop].float().abs().amax(dim=2).mul(1 / 127).clamp_min(1e-12)
                    torch.testing.assert_close(scales[:, :, start // 128], scale)
                    expected = (value[:, :, start:stop].float() * (1.0 / scale).unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)
                    torch.testing.assert_close(unpacked[:, :, start:stop], expected, atol=0, rtol=0)
                self.assertFalse(unpacked[:, :, 257:].count_nonzero())

    def test_invalid_chunk_coverage(self):
        value = torch.zeros(1, 2, 129, 128, dtype=torch.bfloat16)
        staging = SinglePassVCarrier(spec_for(value), backend=BACKEND_TORCH)
        with self.assertRaisesRegex(ValueError, 'cover'):
            staging.finish()
        with self.assertRaisesRegex(ValueError, 'aligned'):
            staging.quantize(value[:, :, :63], 0)
        staging.quantize(value[:, :, :64], 0)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            staging.quantize(value[:, :, :64], 0)

    def test_direct_residual_stages_all_q_before_overwrite(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                original = (torch.arange(257 * 256).reshape(257, 256) % 7).float()
                x = original.clone()
                source = NormalizedRows(x, lambda rows: rows.clone(), ((0, 257, 0),), None, None,
                                        lambda rows, *args: rows)
                gates = torch.tensor([[.25] * 256])
                source.enable_attention_residual(gates)
                carrier = PrequantizedInt8Attention(
                    original.reshape(1, 257, 2, 128).transpose(1, 2).to(torch.int8).contiguous(),
                    torch.empty(1, 2, 257, 128, dtype=torch.int8),
                    torch.empty(256, 320, dtype=torch.int8),
                    torch.ones(1, 2, 96), torch.ones(1), torch.ones(1, 2, 5, 128),
                    128, torch.float32, 1., 64,
                )
                projection = mock.Mock(side_effect=lambda rows: rows + 1)
                module = SimpleNamespace(heads=2, head_dim=128, out_proj=projection)
                prepared = PreparedChunkedKitchenQKV(carrier, output_buffer=source)
                def attention(chunk, **kwargs):
                    if fail:
                        raise RuntimeError('kernel failure')
                    return chunk.q.float()

                kitchen = mock.Mock(int8_attention_from_prequantized=attention)
                backend = ChunkedKitchenAttentionBackend(stream_output=True, query_chunk_rows=128)
                with (
                    mock.patch('h3_optimizations.kitchen_qkv.resolve_kitchen', return_value=kitchen),
                    mock.patch('h3_optimizations.kitchen_qkv.create_held_qkv', side_effect=AssertionError('Q was already packed')),
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'kernel failure'):
                            backend.execute_projected(module, prepared)
                    else:
                        actual = backend.execute_projected(module, prepared)
                        self.assertIs(actual, x)
                        torch.testing.assert_close(actual, original + (original + 1) * .25)
                        self.assertTrue(source.attention_residual_complete)
                self.assertIsNone(source._output)
                self.assertIsNone(prepared.carrier)

    def test_public_mode_forces_dense_streaming(self):
        request = _memory_request_for_modes(
            fused_qkv='auto', mlp_memory='auto', chunk_rows=4096, precision_mode='Auto',
            qkv_streaming_mode='Auto', kitchen_v_memory_mode=KITCHEN_V_MEMORY_MODE_BLOCK,
        )
        self.assertEqual(request.attention_v_memory, 'blockwise')
        self.assertEqual(request.qkv_streaming, 'forced')
        with self.assertRaisesRegex(ValueError, 'dense attention'):
            H3OptimizationPlan(memory=request, sparse=SparseRequest())


if __name__ == '__main__':
    unittest.main()
