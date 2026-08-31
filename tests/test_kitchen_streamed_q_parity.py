"""Live numerical gates for the streamed Kitchen Q carrier boundary.

Run this file with CUDA visible. These tests deliberately compare the carrier
representations, not only the Python control path that produces them.
"""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))

from h3_optimizations import native  # noqa: E402
from h3_optimizations.native import selftest  # noqa: E402


HEADS = 2
HEAD_DIM = 128


def _samples(sequence, seed, *, head_dim=HEAD_DIM, dtype=torch.bfloat16):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    return tuple(
        torch.randn(
            1,
            HEADS,
            sequence,
            head_dim,
            device='cuda',
            dtype=dtype,
            generator=generator,
        )
        for _ in range(3)
    )


def _full_route(q_length, kv_length, q_tile, kv_tile):
    q_tiles = (q_length + q_tile - 1) // q_tile
    kv_tiles = (kv_length + kv_tile - 1) // kv_tile
    indices = torch.arange(kv_tiles, dtype=torch.int32, device='cuda')
    return native.BlockSparseRoute(
        indices=indices.view(1, 1, 1, -1)
        .expand(1, HEADS, q_tiles, -1)
        .contiguous(),
        counts=torch.full(
            (1, HEADS, q_tiles),
            kv_tiles,
            dtype=torch.int32,
            device='cuda',
        ),
        q_tile=q_tile,
        kv_tile=kv_tile,
        encoding='absolute',
    )


class KitchenStreamedQParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is unavailable')
        if not native.is_available():
            raise unittest.SkipTest(
                'vendored INT8 attention is unavailable: %s'
                % native.unavailable_reason()
            )

    def test_native_production_selftest_passes(self):
        passed, detail = selftest.run('cuda', verbose=True)
        self.assertTrue(passed, detail)

    def test_q_and_scale_match_whole_carrier_across_transform_boundary(self):
        cases = {
            256: ((0, 128), (128, 256)),
            257: ((0, 128), (128, 257)),
        }
        for dtype in (torch.bfloat16, torch.float16):
            for head_dim in (64, 128, 256):
                scales_per_tile = 64 if head_dim == 256 else 32
                for sequence, chunks in cases.items():
                    with self.subTest(
                        dtype=dtype,
                        head_dim=head_dim,
                        sequence=sequence,
                    ):
                        q, k, v = _samples(
                            sequence,
                            20260851 + sequence + head_dim,
                            head_dim=head_dim,
                            dtype=dtype,
                        )
                        whole = native.prequantize_int8_attention(
                            q,
                            k,
                            v,
                            cta_k=64,
                        )
                        for start, stop in chunks:
                            packed_q, packed_scale = native.quantize_int8_attention_q(
                                q[..., start:stop, :],
                                full_k_length=sequence,
                                allow_strided_input=True,
                            )
                            scale_start = (start // 128) * scales_per_tile
                            scale_stop = (
                                (stop + 127) // 128
                            ) * scales_per_tile
                            self.assertTrue(
                                torch.equal(
                                    packed_q,
                                    whole.q[..., start:stop, :],
                                ),
                                'Q carrier differs at sequence=%d rows=[%d,%d)'
                                % (sequence, start, stop),
                            )
                            self.assertTrue(
                                torch.equal(
                                    packed_scale,
                                    whole.q_scale[..., scale_start:scale_stop],
                                ),
                                'Q scale differs at sequence=%d rows=[%d,%d)'
                                % (sequence, start, stop),
                            )

        q, k, v = _samples(640, 20260851)
        whole = native.prequantize_int8_attention(q, k, v, cta_k=64)
        for start, stop in ((0, 128), (128, 384), (384, 640)):
            packed_q, packed_scale = native.quantize_int8_attention_q(
                q[..., start:stop, :],
                full_k_length=640,
                allow_strided_input=True,
            )
            self.assertTrue(torch.equal(packed_q, whole.q[..., start:stop, :]))
            self.assertTrue(
                torch.equal(
                    packed_scale,
                    whole.q_scale[..., (start // 128) * 32:(stop // 128) * 32],
                )
            )

    def test_streamed_q_full_route_matches_whole_carrier_output(self):
        sequence = 640
        q, k, v = _samples(sequence, 20260852)
        for q_tile, kv_tile in ((128, 64), (64, 128), (64, 64), (128, 128)):
            with self.subTest(q_tile=q_tile, kv_tile=kv_tile):
                whole = native.prequantize_int8_attention(
                    q,
                    k,
                    v,
                    cta_k=kv_tile,
                )
                dense = native.int8_attention_from_prequantized(whole)
                expected = native.block_sparse_int8_attention_from_prequantized(
                    whole,
                    _full_route(sequence, sequence, q_tile, kv_tile),
                    validate_geometry=False,
                )
                self.assertTrue(
                    torch.equal(expected, dense),
                    'whole-carrier 100%% route differs for %dQx%dKV'
                    % (q_tile, kv_tile),
                )
                chunks = []
                for start in range(0, sequence, 128):
                    stop = min(start + 128, sequence)
                    packed_q, packed_scale = native.quantize_int8_attention_q(
                        q[..., start:stop, :],
                        full_k_length=sequence,
                        allow_strided_input=True,
                    )
                    carrier = replace(
                        whole,
                        q=packed_q,
                        q_scale=packed_scale,
                    )
                    chunks.append(
                        native.block_sparse_int8_attention_from_prequantized(
                            carrier,
                            _full_route(
                                stop - start,
                                sequence,
                                q_tile,
                                kv_tile,
                            ),
                            validate_geometry=False,
                        )
                    )
                actual = torch.cat(chunks, dim=-2)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(actual, expected),
                    'streamed Q output differs for %dQx%dKV' % (q_tile, kv_tile),
                )


if __name__ == '__main__':
    unittest.main()
