'''CPU contracts for chunk-scaled (single-pass) INT8 V.'''

import ctypes
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

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

import h3_optimizations.kitchen_qkv as kitchen_qkv  # noqa: E402
from h3_optimizations import memory_migration_node, status  # noqa: E402
from h3_optimizations.native import int8_attention, loader  # noqa: E402
from h3_optimizations.native.v_staging import (  # noqa: E402
    BACKEND_TORCH,
    ChunkScaledVCarrier,
    VStagingError,
    _inverse_permutation_16,
    chunk_scaled_block_rows_ok,
)
from h3_optimizations.plan import (  # noqa: E402
    MemoryRequest,
    V_MEMORY_CHUNK_SCALED,
    V_MEMORY_RETAIN,
    V_MEMORY_TWO_PASS,
    v_memory_without_chunk_scales,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def _spec(sequence, heads=2, head_dim=8, cta_k=64):
    return SimpleNamespace(
        k_input_shape=(1, heads, sequence, head_dim),
        kernel_head_dim=head_dim,
        cta_k=cta_k,
        device=torch.device('cpu'),
        input_dtype=torch.bfloat16,
    )


def _unpermute(v_int8, batch, heads, head_dim, sequence):
    '''Carrier [B*H*D, padded] back to [B, H, S, D] in sequence order.'''
    rows = torch.arange(sequence)
    columns = (rows & ~15) | _inverse_permutation_16(v_int8.device)[rows & 15]
    return v_int8.view(batch, heads, head_dim, -1)[..., columns].permute(0, 1, 3, 2)


class ChunkScaledCarrierTests(unittest.TestCase):
    def _source(self, sequence=300, heads=2, head_dim=8):
        torch.manual_seed(0)
        source = torch.randn(1, heads, sequence, head_dim)
        # Give each 128-row block a different magnitude, so a global scale and
        # block scales differ.
        source *= (1.0 + torch.arange(sequence) // 128).view(1, 1, -1, 1)
        return source.to(torch.bfloat16)

    def test_each_block_is_quantized_against_its_own_maximum(self):
        source = self._source()
        carrier = ChunkScaledVCarrier(_spec(300), 128, backend=BACKEND_TORCH)
        for start in range(0, 300, 128):
            carrier.quantize(source[..., start:start + 128, :], start)
        v_int8, scales = carrier.finish()

        self.assertEqual(carrier.blocks, 3)
        self.assertEqual(tuple(scales.shape), (1 * 2 * 3 * 8,))
        scales = scales.view(1, 2, 3, 8)
        values = source.float()
        for block, start in enumerate(range(0, 300, 128)):
            chunk = values[..., start:start + 128, :]
            expected = torch.clamp(chunk.abs().amax(dim=-2) / 127.0, min=1e-12)
            torch.testing.assert_close(scales[:, :, block], expected, rtol=1e-6, atol=0)
            quantized = _unpermute(v_int8, 1, 2, 8, 300)[..., start:start + 128, :]
            reference = torch.clamp(
                torch.round(chunk * (1.0 / expected).unsqueeze(-2)), -128, 127
            )
            self.assertTrue(torch.equal(quantized.float(), reference))

    def test_padding_columns_stay_zero(self):
        source = self._source(sequence=200)
        carrier = ChunkScaledVCarrier(_spec(200), 128, backend=BACKEND_TORCH)
        carrier.quantize(source[..., :128, :], 0)
        carrier.quantize(source[..., 128:, :], 128)
        v_int8, _ = carrier.finish()
        self.assertEqual(int(v_int8.shape[-1]), 256)
        rows = torch.arange(200, 256)
        columns = (rows & ~15) | _inverse_permutation_16(v_int8.device)[rows & 15]
        self.assertTrue(torch.equal(v_int8[:, columns], torch.zeros_like(v_int8[:, columns])))

    def test_dequantized_error_is_below_the_global_carrier(self):
        source = self._source()
        carrier = ChunkScaledVCarrier(_spec(300), 128, backend=BACKEND_TORCH)
        for start in range(0, 300, 128):
            carrier.quantize(source[..., start:start + 128, :], start)
        v_int8, scales = carrier.finish()
        q = _unpermute(v_int8, 1, 2, 8, 300).float()
        block_scale = scales.view(1, 2, 3, 8).repeat_interleave(128, dim=2)[:, :, :300]
        chunked_error = (q * block_scale - source.float()).norm()

        global_scale = source.float().abs().amax(dim=-2, keepdim=True) / 127.0
        global_q = torch.round(source.float() / global_scale) * global_scale
        global_error = (global_q - source.float()).norm()
        self.assertLess(float(chunked_error), float(global_error))

    def test_chunks_must_be_whole_ordered_blocks(self):
        source = self._source()
        carrier = ChunkScaledVCarrier(_spec(300), 128, backend=BACKEND_TORCH)
        with self.assertRaisesRegex(VStagingError, 'whole 128-row blocks'):
            carrier.quantize(source[..., 128:256, :], 128)
        with self.assertRaisesRegex(VStagingError, 'whole 128-row blocks'):
            carrier.quantize(source[..., :100, :], 0)
        carrier.quantize(source[..., :128, :], 0)
        with self.assertRaisesRegex(VStagingError, 'do not cover'):
            carrier.finish()

    def test_has_no_whole_sequence_scale(self):
        carrier = ChunkScaledVCarrier(_spec(300), 128, backend=BACKEND_TORCH)
        with self.assertRaisesRegex(VStagingError, 'no whole-sequence scale'):
            carrier.finalize_scale()

    def test_block_rows_must_be_a_power_of_two_of_at_least_128(self):
        for rows in (64, 192, 4095):
            self.assertFalse(chunk_scaled_block_rows_ok(rows))
            with self.assertRaisesRegex(VStagingError, 'power-of-two'):
                ChunkScaledVCarrier(_spec(300), rows, backend=BACKEND_TORCH)
        for rows in (128, 256, 4096, 65536):
            self.assertTrue(chunk_scaled_block_rows_ok(rows))


class _Held:
    def __init__(self):
        self.kv_calls = []
        self.v_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def project_rows(self, _x, _rope, rows):
        values = torch.zeros((1, 2, len(rows), 128), dtype=torch.bfloat16)
        return values, values, values

    def project_kv_hnd(self, _x, _rope, start, end):
        self.kv_calls.append((start, end))
        values = torch.zeros(1, 2, end - start, 128, dtype=torch.bfloat16)
        return values, values + 1

    def project_v_hnd(self, _x, _rope, start, end):
        self.v_calls.append((start, end))
        raise AssertionError('single-pass V must not project V a second time')


class _Kitchen:
    Int8AttentionProducerUnavailableError = RuntimeError

    def __init__(self):
        self.retained = []

    @staticmethod
    def select_int8_attention_k_anchor(_spec, _samples):
        return object()

    @staticmethod
    def create_int8_attention_producer(_spec, _anchor):
        return SimpleNamespace(v=None, v_scale=None, v_scale_rows=0)

    @staticmethod
    def quantize_int8_attention_k_chunk(*_args, **_kwargs):
        return None

    def quantize_int8_attention_v(self, _producer, v):
        self.retained.append(v)

    @staticmethod
    def finalize_int8_attention_producer(producer):
        return producer


class _Carrier:
    instance = None

    def __init__(self, _spec, block_rows):
        _Carrier.instance = self
        self.block_rows = block_rows
        self.quantized = []
        self.output = (object(), object())

    def quantize(self, value, start):
        self.quantized.append((start, int(value.shape[2])))

    def finish(self):
        return self.output


class ProducerTests(unittest.TestCase):
    def test_streamed_producer_quantizes_each_chunk_once_and_retains_nothing(self):
        held = _Held()
        kitchen = _Kitchen()
        with (
            mock.patch.object(kitchen_qkv, 'resolve_kitchen', return_value=kitchen),
            mock.patch.object(kitchen_qkv, 'create_held_qkv', return_value=held),
            mock.patch('h3_optimizations.native.v_staging.ChunkScaledVCarrier', _Carrier),
        ):
            projected = kitchen_qkv.run_streamed_kitchen_qkv(
                SimpleNamespace(heads=2, head_dim=128),
                torch.zeros(5, 256, dtype=torch.bfloat16),
                None,
                layer_index=0,
                transformer_options={},
                spec=SimpleNamespace(k_anchor_positions=(0,) * 9),
                chunk_rows=2,
                projection_mode='native',
                v_mode=kitchen_qkv.V_MODE_CHUNK_SCALED,
            )
        self.assertEqual(held.kv_calls, [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(held.v_calls, [])
        self.assertEqual(kitchen.retained, [])
        self.assertEqual(_Carrier.instance.block_rows, 2)
        self.assertEqual(_Carrier.instance.quantized, [(0, 2), (2, 2), (4, 1)])
        carrier = projected.carrier
        self.assertIs(carrier.v, _Carrier.instance.output[0])
        self.assertIs(carrier.v_scale, _Carrier.instance.output[1])
        self.assertEqual(carrier.v_scale_rows, 2)


class ProjectorResolutionTests(unittest.TestCase):
    def _projector(self, available, chunk_rows=4096):
        with mock.patch.object(
            int8_attention, 'chunk_scaled_v_is_available', return_value=available
        ):
            return kitchen_qkv.ChunkedKitchenQKVProjector(
                chunk_rows=chunk_rows,
                v_mode=kitchen_qkv.V_MODE_CHUNK_SCALED,
            )

    def test_runs_chunk_scaled_when_the_binary_supports_it(self):
        projector = self._projector(True)
        self.assertEqual(projector.v_mode, kitchen_qkv.V_MODE_CHUNK_SCALED)
        self.assertIn(kitchen_qkv.V_MODE_CHUNK_SCALED, projector.installation_signature)

    def test_falls_back_to_two_pass_without_kernels(self):
        projector = self._projector(False)
        self.assertEqual(projector.v_mode, kitchen_qkv.V_MODE_TWO_PASS)

    def test_falls_back_to_two_pass_for_an_unsupported_chunk_size(self):
        projector = self._projector(True, chunk_rows=3000)
        self.assertEqual(projector.v_mode, kitchen_qkv.V_MODE_TWO_PASS)


class ConsumerDispatchTests(unittest.TestCase):
    def test_global_carrier_uses_the_base_entry_point(self):
        library = SimpleNamespace(h3_int8_dense_attention='base')
        entry, extra = int8_attention._attention_entry(
            library, 'h3_int8_dense_attention', SimpleNamespace(v_scale_rows=0)
        )
        self.assertEqual((entry, extra), ('base', ()))

    def test_chunk_scaled_carrier_uses_the_chunked_entry_point(self):
        library = SimpleNamespace(
            h3_int8_dense_attention='base',
            h3_int8_dense_attention_chunked_v='chunked',
        )
        entry, extra = int8_attention._attention_entry(
            library, 'h3_int8_dense_attention', SimpleNamespace(v_scale_rows=4096)
        )
        self.assertEqual((entry, extra), ('chunked', (4096,)))

    def test_chunk_scaled_carrier_never_falls_back_to_global_scales(self):
        library = SimpleNamespace(h3_int8_dense_attention='base')
        with self.assertRaisesRegex(RuntimeError, 'cannot read chunk-scaled V'):
            int8_attention._attention_entry(
                library, 'h3_int8_dense_attention', SimpleNamespace(v_scale_rows=4096)
            )

    def test_vsa_tiles_refuse_chunk_scaled_carriers(self):
        library = SimpleNamespace(h3_int8_sparse_attention_tiles=object())
        quantized = SimpleNamespace(cta_k=64, v_scale_rows=4096)
        route = SimpleNamespace(q_tile=64, kv_tile=64)
        with mock.patch.object(loader, 'load', return_value=library):
            with self.assertRaisesRegex(ValueError, 'no chunk-scaled V'):
                int8_attention.block_sparse_int8_attention_tiles_from_prequantized(
                    quantized, route, torch.ones(1, dtype=torch.int32)
                )

    def test_loader_binds_chunked_symbols_with_the_block_size_before_the_stream(self):
        class Function:
            def __init__(self):
                self.argtypes = None
                self.restype = None

        class Library:
            pass

        library = Library()
        names = [
            'h3_int8_abi_version', 'h3_int8_last_error', 'h3_int8_route_encoding',
            'h3_int8_device_capability', 'h3_int8_dense_attention',
            'h3_int8_sparse_attention', 'h3_int8_sparse_attention_lse',
            'h3_int8_quantize_qk', 'h3_int8_select_k_anchor',
            'h3_int8_quantize_qk_chunk', 'h3_int8_quantize_q_chunk',
            'h3_int8_quantize_v',
            'h3_int8_dense_attention_chunked_v',
            'h3_int8_sparse_attention_chunked_v',
        ]
        for name in names:
            setattr(library, name, Function())
        try:
            loader._bind(library)
        except AttributeError:
            # Later optional groups are probed with getattr; a bare fake may
            # lack some, which is fine as long as the attention bindings ran.
            pass
        base = library.h3_int8_dense_attention.argtypes
        self.assertEqual(library.h3_int8_dense_attention_chunked_v.argtypes,
                         list(base[:-1]) + [ctypes.c_int, ctypes.c_size_t])
        # Sparse entries also take the aligned route's V block change list.
        base = library.h3_int8_sparse_attention.argtypes
        self.assertEqual(library.h3_int8_sparse_attention_chunked_v.argtypes,
                         list(base[:-1]) + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_size_t])
        self.assertFalse(hasattr(library, 'h3_int8_packed_dense_attention_chunked_v'))


class RouteAlignmentTests(unittest.TestCase):
    '''The packed kernels never split an iteration across V scale blocks.'''

    KV_TILE = 32
    BLOCK_ROWS = 512  # 16 tiles per block keeps the test small
    KV_LENGTH = 1500  # ragged: the final block is partial

    def _route(self, seed, density=0.3, encoding='absolute'):
        generator = torch.Generator().manual_seed(seed)
        tiles = -(-self.KV_LENGTH // self.KV_TILE)
        keep = torch.rand(2, 3, 5, tiles, generator=generator) < density
        keep[0, 0, 0] = False  # an empty row
        keep[0, 0, 1] = True   # a full row
        counts = keep.sum(dim=-1).to(torch.int32)
        position = torch.arange(tiles).expand_as(keep)
        ordered = torch.where(keep, position, tiles).sort(dim=-1).values
        indices = torch.where(ordered == tiles, 0, ordered).to(torch.int32)
        route = int8_attention.BlockSparseRoute(
            indices=indices.contiguous(), counts=counts.contiguous(),
            q_tile=64, kv_tile=self.KV_TILE, encoding='absolute')
        return route.to_delta() if encoding == 'delta' else route

    @staticmethod
    def _rows(route):
        absolute = route.to_absolute()
        indices = absolute.indices.reshape(-1, absolute.indices.shape[-1])
        counts = absolute.counts.reshape(-1)
        return [indices[r, :int(counts[r])].tolist() for r in range(counts.numel())]

    def _check(self, before, after, group):
        per_block = self.BLOCK_ROWS // self.KV_TILE
        for old, new in zip(self._rows(before), self._rows(after)):
            self.assertEqual(new, sorted(set(new)))
            self.assertTrue(set(old) <= set(new))
            blocks = [tile // per_block for tile in new]
            if not blocks:
                continue
            last = blocks[-1]
            for block in set(blocks) - {last}:
                self.assertEqual(blocks.count(block) % group, 0)
            # Iterations of `group` consecutive entries never mix blocks.
            for start in range(0, len(blocks), group):
                self.assertEqual(len(set(blocks[start:start + group])), 1)
            # Only what alignment needs is added.
            added = len(new) - len(old)
            self.assertLess(added, group * len(set(blocks)))

    def test_every_iteration_stays_inside_one_block(self):
        for seed in range(6):
            for group in (2, 4):
                with self.subTest(seed=seed, group=group):
                    route = self._route(seed)
                    aligned = int8_attention.align_route_to_v_blocks(
                        route, self.BLOCK_ROWS, group, self.KV_LENGTH)
                    self._check(route, aligned, group)

    def test_delta_routes_align_the_same(self):
        route = self._route(3, encoding='delta')
        aligned = int8_attention.align_route_to_v_blocks(
            route, self.BLOCK_ROWS, 4, self.KV_LENGTH)
        expected = int8_attention.align_route_to_v_blocks(
            self._route(3), self.BLOCK_ROWS, 4, self.KV_LENGTH)
        self.assertEqual(self._rows(aligned), self._rows(expected))

    def _single_row(self, tiles):
        indices = torch.zeros(1, 1, 1, 47, dtype=torch.int32)
        indices[0, 0, 0, :len(tiles)] = torch.tensor(tiles)
        route = int8_attention.BlockSparseRoute(
            indices=indices, counts=torch.tensor([[[len(tiles)]]], dtype=torch.int32),
            q_tile=64, kv_tile=self.KV_TILE, encoding='absolute')
        aligned = int8_attention.align_route_to_v_blocks(
            route, self.BLOCK_ROWS, 4, self.KV_LENGTH)
        return self._rows(aligned)[0]

    def test_change_list_marks_each_new_block_iteration(self):
        route = int8_attention.BlockSparseRoute(
            indices=torch.tensor([[[[0, 1, 2, 20, 40, 41, 0]]]], dtype=torch.int32),
            counts=torch.tensor([[[6]]], dtype=torch.int32),
            q_tile=64, kv_tile=self.KV_TILE, encoding='absolute')
        aligned = int8_attention.align_route_to_v_blocks(
            route, self.BLOCK_ROWS, 4, self.KV_LENGTH)
        # Blocks 0 (padded to 4), 1 (padded to 4), 2 (last, exempt):
        # iterations 0, 1, 2 start blocks 0, 1, 2.
        self.assertEqual(self._rows(aligned)[0], [0, 1, 2, 3, 20, 21, 22, 23, 40, 41])
        self.assertEqual(aligned.v_block_changes.reshape(-1)[:3].tolist(), [1, 2, 0x7FFFFFFF])

    def test_padding_follows_the_run_then_precedes_it_then_fills_gaps(self):
        # Blocks hold 16 tiles: block 1 is tiles 16-31, block 2 is 32-47.
        self.assertEqual(self._single_row([20, 40]), [20, 21, 22, 23, 40])
        self.assertEqual(self._single_row([30, 40]), [28, 29, 30, 31, 40])
        self.assertEqual(self._single_row([16, 31, 40]), [16, 17, 18, 31, 40])
        self.assertEqual(self._single_row([16, 17, 18, 19]), [16, 17, 18, 19])  # aligned
        self.assertEqual(self._single_row([40]), [40])  # last block exempt

    def test_single_tile_iterations_are_left_alone(self):
        route = self._route(1)
        self.assertIs(
            int8_attention.align_route_to_v_blocks(route, self.BLOCK_ROWS, 1, self.KV_LENGTH),
            route)


class PlanNodeStatusTests(unittest.TestCase):
    def test_plan_accepts_chunk_scaled_v(self):
        request = MemoryRequest(attention_v_memory=V_MEMORY_CHUNK_SCALED)
        self.assertEqual(request.attention_v_memory, V_MEMORY_CHUNK_SCALED)

    def test_carriers_without_chunk_scales_run_two_pass(self):
        self.assertEqual(v_memory_without_chunk_scales(V_MEMORY_CHUNK_SCALED), V_MEMORY_TWO_PASS)
        self.assertEqual(v_memory_without_chunk_scales(V_MEMORY_TWO_PASS), V_MEMORY_TWO_PASS)
        self.assertEqual(v_memory_without_chunk_scales(V_MEMORY_RETAIN), V_MEMORY_RETAIN)

    def test_lower_vram_option_requests_chunk_scaled_v(self):
        request = memory_migration_node._memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=4096,
            precision_mode=memory_migration_node.PRECISION_MODE_AUTO,
            qkv_streaming_mode=memory_migration_node.QKV_STREAMING_MODE_AUTO,
            kitchen_v_memory_mode=memory_migration_node.KITCHEN_V_MEMORY_MODE_LOWER_VRAM,
        )
        self.assertEqual(request.attention_v_memory, V_MEMORY_CHUNK_SCALED)

    def test_status_names_chunk_scaled_v_and_reports_fallbacks(self):
        def qkv(requested, actual):
            return {'v_memory_requested': requested, 'v_memory': actual}

        # Two-pass V is the expected route wherever chunk scales are unavailable;
        # it honours Lower VRAM, so only a path that retains V is reported.
        self.assertIsNone(status._v_memory_notice(qkv('chunk_scaled', 'chunk_scaled')))
        self.assertIsNone(status._v_memory_notice(qkv('chunk_scaled', 'two_pass')))
        self.assertIn('running Standard', status._v_memory_notice(qkv('chunk_scaled', 'retain')))
        self.assertIsNone(status._v_memory_notice(qkv('two_pass', 'two_pass')))
        self.assertIn('running Standard', status._v_memory_notice(qkv('two_pass', None)))
        with mock.patch.object(status, '_format_qkv_execution', return_value='route'):
            text = status.format_qkv_execution({'fused_qkv': qkv('chunk_scaled', 'chunk_scaled')})
        self.assertEqual(text, 'route; chunk-scaled V')


if __name__ == '__main__':
    unittest.main()
