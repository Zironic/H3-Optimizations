'''CUDA correctness of chunk-scaled (single-pass) INT8 V. Requires a GPU.

Run explicitly; it allocates VRAM:
    python -m unittest discover -s tests -p test_chunk_scaled_v_cuda.py
'''

from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401 - parse now, not under the runner's argv

sys.argv = [sys.argv[0], *TEST_ARGS]

HEADS = 4
DIM = 128
SEQUENCE = 1100  # ragged: neither a multiple of the blocks nor of 128
SPARSE_GEOMETRIES = ((64, 32), (64, 64), (128, 128), (128, 64), (64, 128))


def _relative_l2(actual, reference):
    return float((actual.float() - reference.float()).norm() / reference.float().norm())


@unittest.skipUnless(torch.cuda.is_available(), 'chunk-scaled V kernels need CUDA')
class ChunkScaledVCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from h3_optimizations.native import int8_attention as native
        from h3_optimizations.native.v_staging import ChunkScaledVCarrier

        if not native.chunk_scaled_v_is_available():
            raise unittest.SkipTest('the loaded native library has no chunk-scaled V')
        cls.native = native
        cls.Carrier = ChunkScaledVCarrier
        cls.device = torch.device('cuda')

    def _qkv(self, seed, *, equal_block_scales=False):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        def sample():
            return torch.randn(1, SEQUENCE, HEADS, DIM, device=self.device,
                               generator=generator).transpose(1, 2)

        q, k, v = sample(), sample(), sample()
        rows = torch.arange(SEQUENCE, device=self.device)
        if equal_block_scales:
            # The first row of every 128-row block holds the same per-channel
            # maximum, so every block scale equals the global scale.
            peak = 6.0 * torch.sign(torch.randn(1, HEADS, 1, DIM, device=self.device,
                                                generator=generator))
            v = v.clamp(-5.5, 5.5)
            v[:, :, rows % 128 == 0, :] = peak
        else:
            # Block magnitudes that differ, as H3's sequence segments do.
            v = v * (1.0 + (rows // 128) % 5).view(1, 1, -1, 1)
        return (q.to(torch.bfloat16).contiguous(), k.to(torch.bfloat16).contiguous(),
                v.to(torch.bfloat16).contiguous())

    def _carriers(self, q, k, v, cta_k, block_rows):
        global_carrier = self.native.prequantize_int8_attention(q, k, v, cta_k=cta_k)
        spec = SimpleNamespace(
            k_input_shape=tuple(v.shape), kernel_head_dim=DIM, cta_k=cta_k,
            device=self.device, input_dtype=v.dtype,
        )
        staging = self.Carrier(spec, block_rows)
        for start in range(0, SEQUENCE, block_rows):
            staging.quantize(v[..., start:start + block_rows, :], start)
        v_int8, v_scale = staging.finish()
        chunked = replace(global_carrier, v=v_int8, v_scale=v_scale,
                          v_scale_rows=block_rows)
        return global_carrier, chunked

    def _reference(self, q, k, v, scale, allowed=None):
        logits = torch.einsum('bhqd,bhkd->bhqk', q.float(), k.float()) * scale
        if allowed is not None:
            logits = logits.masked_fill(~allowed, float('-inf'))
        return torch.softmax(logits, dim=-1) @ v.float()

    def _route(self, q_tile, kv_tile, seed):
        generator = torch.Generator().manual_seed(seed)
        q_tiles = math.ceil(SEQUENCE / q_tile)
        kv_tiles = math.ceil(SEQUENCE / kv_tile)
        indices = torch.zeros(1, HEADS, q_tiles, kv_tiles, dtype=torch.int32)
        counts = torch.zeros(1, HEADS, q_tiles, dtype=torch.int32)
        for h in range(HEADS):
            for t in range(q_tiles):
                keep = torch.rand(kv_tiles, generator=generator) < 0.45
                keep[0] = True
                tiles = torch.nonzero(keep).flatten()
                indices[0, h, t, :len(tiles)] = tiles.to(torch.int32)
                counts[0, h, t] = len(tiles)
        route = self.native.BlockSparseRoute(
            indices=indices.to(self.device), counts=counts.to(self.device),
            q_tile=q_tile, kv_tile=kv_tile, encoding='absolute',
        )
        # Chunk-scaled carriers pad routes so packed iterations never mix V
        # blocks; give the global carrier and the reference the same keys.
        route = self.native.align_route_to_v_blocks(
            route, 128, self.native.packed_route_group(q_tile, kv_tile, self.device),
            SEQUENCE)
        return route, self._allowed(route)

    def _allowed(self, route):
        route = route.to_absolute()
        q_tile, kv_tile = int(route.q_tile), int(route.kv_tile)
        indices, counts = route.indices.cpu(), route.counts.cpu()
        allowed = torch.zeros(1, HEADS, SEQUENCE, SEQUENCE, dtype=torch.bool)
        for h in range(HEADS):
            for t in range(indices.shape[2]):
                for tile in indices[0, h, t, :int(counts[0, h, t])].tolist():
                    allowed[0, h, t * q_tile:(t + 1) * q_tile,
                            tile * kv_tile:(tile + 1) * kv_tile] = True
        return allowed.to(self.device)

    def _as_global(self, chunked):
        '''The same INT8 V bytes declared as a global carrier (block 0 scales).'''
        first = chunked.v_scale.view(1, HEADS, -1, DIM)[:, :, 0].reshape(-1).contiguous()
        return replace(chunked, v_scale=first, v_scale_rows=0)

    def test_equal_block_scales_reproduce_the_global_kernels(self):
        q, k, v = self._qkv(11, equal_block_scales=True)
        for cta_k in (64, 128):
            with self.subTest(dense_cta_k=cta_k):
                global_carrier, chunked = self._carriers(q, k, v, cta_k, 128)
                scales = chunked.v_scale.view(1, HEADS, -1, DIM)
                self.assertTrue(torch.equal(scales, scales[:, :, :1].expand_as(scales)))
                torch.testing.assert_close(
                    scales[:, :, 0].reshape(-1), global_carrier.v_scale, rtol=1e-6, atol=0)
                expected = self.native.int8_attention_from_prequantized(self._as_global(chunked))
                actual = self.native.int8_attention_from_prequantized(chunked)
                self.assertTrue(torch.equal(actual, expected))
        for q_tile, kv_tile in SPARSE_GEOMETRIES:
            with self.subTest(sparse='%dx%d' % (q_tile, kv_tile)):
                carrier_tile = self.native.carrier_kv_tile(kv_tile)
                _, chunked = self._carriers(q, k, v, carrier_tile, 128)
                route, _ = self._route(q_tile, kv_tile, 3)
                expected = self.native.block_sparse_int8_attention_from_prequantized(
                    self._as_global(chunked), route, validate_geometry=False)
                actual = self.native.block_sparse_int8_attention_from_prequantized(
                    chunked, route, validate_geometry=False)
                # Aligned routes never mix blocks inside a packed iteration,
                # so every kernel reproduces the global kernel bit for bit.
                self.assertTrue(torch.equal(actual, expected))

    def test_dense_is_more_accurate_than_the_global_carrier(self):
        q, k, v = self._qkv(21)
        for cta_k in (64, 128):
            for block_rows in (128, 256, 4096):
                with self.subTest(cta_k=cta_k, block_rows=block_rows):
                    global_carrier, chunked = self._carriers(q, k, v, cta_k, block_rows)
                    reference = self._reference(q, k, v, global_carrier.attention_scale)
                    global_error = _relative_l2(
                        self.native.int8_attention_from_prequantized(global_carrier), reference)
                    chunked_error = _relative_l2(
                        self.native.int8_attention_from_prequantized(chunked), reference)
                    self.assertLess(chunked_error, 0.05)
                    if block_rows < SEQUENCE:
                        self.assertLess(chunked_error, global_error)
                    else:
                        self.assertEqual(chunked_error, global_error)

    def test_packed_dense_is_more_accurate_than_the_global_carrier(self):
        if not self.native.packed_dense_attention_is_available(self.device):
            self.skipTest('packed dense attention is not built for this GPU')
        q, k, v = self._qkv(31)
        global_carrier, chunked = self._carriers(q, k, v, 64, 128)
        reference = self._reference(q, k, v, global_carrier.attention_scale)
        global_error = _relative_l2(
            self.native.packed_int8_attention_from_prequantized(global_carrier), reference)
        chunked_error = _relative_l2(
            self.native.packed_int8_attention_from_prequantized(chunked), reference)
        self.assertLess(chunked_error, global_error)
        self.assertLess(chunked_error, 0.05)

    def test_native_route_alignment_matches_the_reference(self):
        for seed in range(3):
            for kv_tile, group in ((32, 4), (64, 2)):
                for encoding in ('absolute', 'delta'):
                    with self.subTest(seed=seed, kv_tile=kv_tile, encoding=encoding):
                        generator = torch.Generator().manual_seed(seed)
                        tiles = math.ceil(SEQUENCE / kv_tile)
                        keep = torch.rand(1, HEADS, 9, tiles, generator=generator) < 0.35
                        keep[..., 0] = True
                        position = torch.arange(tiles).expand_as(keep)
                        ordered = torch.where(keep, position, tiles).sort(dim=-1).values
                        route = self.native.BlockSparseRoute(
                            indices=torch.where(ordered == tiles, 0, ordered).to(torch.int32).contiguous(),
                            counts=keep.sum(dim=-1).to(torch.int32).contiguous(),
                            q_tile=64, kv_tile=kv_tile, encoding='absolute')
                        if encoding == 'delta':
                            route = route.to_delta()
                        expected = self.native._align_route_reference(route, 128 // kv_tile, group, SEQUENCE)
                        moved = replace(route, indices=route.indices.to(self.device),
                                        counts=route.counts.to(self.device))
                        actual = self.native.align_route_to_v_blocks(moved, 128, group, SEQUENCE)
                        self.assertTrue(torch.equal(actual.counts.cpu(), expected.counts))
                        self.assertTrue(torch.equal(actual.v_block_changes.cpu(),
                                                    expected.v_block_changes))
                        a, e = actual.to_absolute(), expected.to_absolute()
                        live = torch.arange(a.indices.shape[-1]) < a.counts.cpu().unsqueeze(-1)
                        width = e.indices.shape[-1]
                        self.assertTrue(torch.equal(
                            torch.where(live[..., :width], a.indices.cpu()[..., :width], 0),
                            torch.where(live[..., :width], e.indices, 0)))

    def test_aligned_routes_hold_the_packed_contract(self):
        for q_tile, kv_tile in SPARSE_GEOMETRIES:
            group = self.native.packed_route_group(q_tile, kv_tile, self.device)
            route, _ = self._route(q_tile, kv_tile, 5)
            route = route.to_absolute()
            per_block = 128 // kv_tile
            for h in range(HEADS):
                for t in range(route.indices.shape[2]):
                    count = int(route.counts[0, h, t])
                    blocks = (route.indices[0, h, t, :count] // per_block).tolist()
                    for start in range(0, count, group):
                        self.assertEqual(len(set(blocks[start:start + group])), 1)

    def test_sparse_routes_across_block_boundaries(self):
        q, k, v = self._qkv(41)
        for q_tile, kv_tile in SPARSE_GEOMETRIES:
            with self.subTest(geometry='%dx%d' % (q_tile, kv_tile)):
                carrier_tile = self.native.carrier_kv_tile(kv_tile)
                global_carrier, chunked = self._carriers(q, k, v, carrier_tile, 128)
                route, allowed = self._route(q_tile, kv_tile, 7)
                reference = self._reference(
                    q, k, v, global_carrier.attention_scale, allowed)
                global_out = self.native.block_sparse_int8_attention_from_prequantized(
                    global_carrier, route, validate_geometry=False)
                chunked_out = self.native.block_sparse_int8_attention_from_prequantized(
                    chunked, route, validate_geometry=False)
                global_error = _relative_l2(global_out, reference)
                chunked_error = _relative_l2(chunked_out, reference)
                self.assertTrue(torch.isfinite(chunked_out).all())
                self.assertLess(chunked_error, 0.05)
                self.assertLess(chunked_error, global_error)

    def test_sparse_lse_does_not_depend_on_v_scales(self):
        q, k, v = self._qkv(51)
        global_carrier, chunked = self._carriers(q, k, v, 64, 128)
        route, allowed = self._route(64, 64, 9)
        expected, expected_lse = (
            self.native.block_sparse_int8_attention_with_lse_from_prequantized(
                global_carrier, route, validate_geometry=False))
        actual, actual_lse = (
            self.native.block_sparse_int8_attention_with_lse_from_prequantized(
                chunked, route, validate_geometry=False))
        self.assertTrue(torch.equal(actual_lse, expected_lse))
        reference = self._reference(q, k, v, global_carrier.attention_scale, allowed)
        self.assertLess(_relative_l2(actual, reference), _relative_l2(expected, reference))


if __name__ == '__main__':
    unittest.main()
