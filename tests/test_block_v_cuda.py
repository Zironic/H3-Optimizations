"""Small CUDA numerical checks; no performance benchmarks or model downloads."""

import unittest
from dataclasses import replace
from unittest import mock
from itertools import product

import torch
import torch.nn.functional as F

from test_block_v import spec_for
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from comfy.ldm.minimax.model import DiTBlock, rope_rotation_table
from h3_optimizations.attention_forward import make_forward as make_attention
from h3_optimizations.kitchen_qkv import ChunkedKitchenAttentionBackend, ChunkedKitchenQKVProjector, DENSE_KITCHEN_BACKEND, H3_ATTENTION_BACKEND_KEY
from h3_optimizations.memory.config import ActivationMemoryConfig
from h3_optimizations.memory.forward import make_forward as make_block
from h3_optimizations.native.block_v import SinglePassVCarrier
from h3_optimizations.native import block_v
from h3_optimizations.native.int8_attention import (
    prequantize_int8_attention, int8_attention_from_prequantized,
)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class BlockVCudaTests(unittest.TestCase):
    def test_packed_block_v_against_generic_and_sdpa(self):
        torch.manual_seed(29)
        for dtype, length, block_rows, cta in product(
            (torch.bfloat16, torch.float16), (1025, 1088, 1151, 1280), (128, 384), (64, 128),
        ):
            with self.subTest(dtype=dtype, length=length, block_rows=block_rows, cta=cta):
                q = torch.randn(2, 4, 129, 128, device='cuda', dtype=dtype) * .3
                k = torch.randn(2, 2, length, 128, device='cuda', dtype=dtype) * .3
                v = torch.randn_like(k)
                v[:, :, :128] *= .01
                staging = SinglePassVCarrier(spec_for(v, cta), block_rows=block_rows)
                for start in range(0, length, block_rows):
                    staging.quantize(v[:, :, start:start + block_rows], start)
                packed, scales = staging.finish()
                carrier = replace(prequantize_int8_attention(q, k, v, cta_k=cta),
                                  v=packed, v_scale=scales, v_scale_rows=block_rows)
                with mock.patch.object(block_v, 'packed_dense_attention_is_available', return_value=True):
                    actual = int8_attention_from_prequantized(carrier, output_layout='nhd')
                with mock.patch.object(block_v, 'packed_dense_attention_is_available', return_value=False):
                    generic = int8_attention_from_prequantized(carrier, output_layout='nhd')
                reference = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
                torch.testing.assert_close(actual, reference, atol=.007, rtol=.035)
                torch.testing.assert_close(actual, generic, atol=.007, rtol=.035)

    def test_compact_scales_match_repeated_bitwise(self):
        torch.manual_seed(23)
        for dtype in (torch.bfloat16, torch.float16):
            for cta in (64, 128):
                for block_rows in (128, 256, 384, 4096):
                    with self.subTest(dtype=dtype, cta=cta, block_rows=block_rows):
                        q = torch.randn(2, 4, 19, 128, device='cuda', dtype=dtype)
                        k = torch.randn(2, 2, 513, 128, device='cuda', dtype=dtype)
                        v = torch.randn_like(k)
                        v[:, :, :128] *= .01
                        staging = SinglePassVCarrier(spec_for(v, cta), block_rows=block_rows)
                        for start in range(0, 513, block_rows):
                            staging.quantize(v[:, :, start:start + block_rows], start)
                        packed, scales = staging.finish()
                        compact = replace(prequantize_int8_attention(q, k, v, cta_k=cta),
                                          v=packed, v_scale=scales, v_scale_rows=block_rows)
                        expanded = scales.repeat_interleave(block_rows // cta, dim=2)
                        expanded = expanded[:, :, :(513 + cta - 1) // cta].contiguous()
                        repeated = replace(compact, v_scale=expanded, v_scale_rows=cta)
                        actual = int8_attention_from_prequantized(compact, output_layout='nhd')
                        expected = int8_attention_from_prequantized(repeated, output_layout='nhd')
                        self.assertTrue(torch.equal(actual, expected))

    def test_query_views_match_copies_bitwise(self):
        torch.manual_seed(17)
        for dtype, length in product((torch.bfloat16, torch.float16), (257, 1025)):
            q = torch.randn(2, 3, 385, 128, device='cuda', dtype=dtype)
            k = torch.randn(2, 3, length, 128, device='cuda', dtype=dtype)
            v = torch.randn_like(k)
            staging = SinglePassVCarrier(spec_for(v, 128))
            for start in range(0, length, 128):
                staging.quantize(v[:, :, start:start + 128], start)
            packed, scales = staging.finish()
            full = replace(prequantize_int8_attention(q, k, v, cta_k=128), v=packed, v_scale=scales)
            for start, stop in ((0, 128), (128, 384), (384, 385)):
                view = replace(full, q=full.q[:, :, start:stop],
                               q_scale=full.q_scale[:, :, start // 128 * 32:((stop + 127) // 128) * 32])
                copy = replace(view, q=view.q.contiguous(), q_scale=view.q_scale.contiguous())
                self.assertFalse(view.q.is_contiguous())
                self.assertFalse(view.q_scale.is_contiguous())
                actual = int8_attention_from_prequantized(view, output_layout='nhd')
                expected = int8_attention_from_prequantized(copy, output_layout='nhd')
                self.assertTrue(torch.equal(actual, expected))

    def test_complete_quantized_block_direct_residual(self):
        for sequence in (257, 1025):
            with self.subTest(sequence=sequence):
                self._complete_quantized_block(sequence)

    def _complete_quantized_block(self, sequence):
        torch.manual_seed(7)
        block = DiTBlock(256, 2, 128, 256, 32, 1e-6, 1e-6,
                         adaln_dtype=torch.float32, dtype=torch.bfloat16,
                         device='cuda', operations=comfy.ops.manual_cast)
        for p in block.parameters():
            p.requires_grad_(False)
            p.normal_(std=.03)
        block.norm1.weight.fill_(1)
        block.norm2.weight.fill_(1)
        block.attn.q_norm.weight.fill_(1)
        block.attn.k_norm.weight.fill_(1)
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
        for owner, name in ((block.attn, 'qkv_proj'), (block.attn, 'out_proj'),
                            (block.mlp, 'fc1'), (block.mlp, 'fc2')):
            old = getattr(owner, name)
            layer = ops.Linear(old.in_features, old.out_features, bias=False, device='cuda')
            layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
                old.weight, 'TensorWiseINT8Layout', scale='recalculate',
                is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256), requires_grad=False)
            setattr(owner, name, layer)
        block.attn.forward = make_attention(
            block.attn, 0,
            backend=ChunkedKitchenAttentionBackend(stream_output=True, query_chunk_rows=128),
            projector=ChunkedKitchenQKVProjector(chunk_rows=128, stream_output=True,
                                                streamed_q=True, strided_qk_input=True, v_mode='blockwise'),
        )
        x = torch.randn(sequence, 256, dtype=torch.bfloat16, device='cuda') * .1
        t = torch.randn(1, 32, dtype=torch.float32, device='cuda') * .1
        rope = rope_rotation_table(torch.randn(sequence, 96, device='cuda'), x.dtype)
        segments = [(0, 64, 1), (64, 129, 2), (129, sequence, 0)]
        options = {H3_ATTENTION_BACKEND_KEY: DENSE_KITCHEN_BACKEND}
        expected = block(x.clone(), t, segments, rope, transformer_options=options)
        for fused in (False, True):
            forward = make_block(block, 0, ActivationMemoryConfig(chunk_rows=128, fused_norm_quantize=fused))
            actual = forward(x.clone(), t, segments, rope, transformer_options=options)
            torch.testing.assert_close(actual, expected, atol=.001, rtol=.01)

    def test_block_scales_against_sdpa(self):
        torch.manual_seed(42)
        for dtype in (torch.bfloat16, torch.float16):
            for cta in (64, 128):
                for length in (1, 65, 129, 513):
                    with self.subTest(dtype=dtype, cta=cta, length=length):
                        q = torch.randn(2, 2, 19, 128, device='cuda', dtype=dtype) * .3
                        k = torch.randn(2, 1, length, 128, device='cuda', dtype=dtype) * .3
                        v = torch.randn_like(k)
                        v[:, :, :64] *= .01
                        staging = SinglePassVCarrier(spec_for(v, cta), block_rows=128)
                        for start in range(0, length, 128):
                            staging.quantize(v[:, :, start:start + 128], start)
                        packed, scales = staging.finish()
                        carrier = prequantize_int8_attention(q, k, v, cta_k=cta)
                        carrier = replace(carrier, v=packed, v_scale=scales, v_scale_rows=128)
                        actual = int8_attention_from_prequantized(carrier, output_layout='nhd')
                        expected = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
                        torch.testing.assert_close(actual, expected, atol=.007, rtol=.035)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    unittest.main()
