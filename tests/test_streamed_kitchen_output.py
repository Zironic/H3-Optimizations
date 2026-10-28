'''Production contracts for query-sliced Kitchen output projection.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.kitchen_sparse import (
    PreparedSparseKitchen,
    SparseKitchenBackend,
)
from h3_optimizations.attention.sparse import kitchen_streamed_q
from h3_optimizations.attention.sparse.kitchen_streamed_q import (
    PreparedStreamedSparseKitchen,
    StreamedSparseKitchenBackend,
    StreamedSparseKitchenQKV,
)
from h3_optimizations.kitchen_qkv import (
    ChunkedKitchenAttentionBackend,
    ChunkedKitchenQKVProjector,
    PreparedChunkedKitchenQKV,
    PreparedStreamedKitchenQKV,
)
from h3_optimizations.normalized_rows import NormalizedRows
from h3_optimizations.native.int8_attention import (
    BlockSparseRoute,
    PrequantizedInt8Attention,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class _Kitchen:
    __version__ = 'test'

    def __init__(self):
        self.calls = []

    def block_sparse_int8_attention_from_prequantized(
        self, carrier, route, output_layout
    ):
        self.calls.append(
            (carrier.q.shape[-2], route.indices.shape[-2], output_layout)
        )
        return carrier.q.to(torch.float32)


class _DenseKitchen:
    def __init__(self):
        self.calls = []

    def int8_attention_from_prequantized(self, carrier, output_layout):
        self.calls.append((carrier.q.shape[-2], output_layout))
        return carrier.q.to(torch.float32)


class _Executor:
    output_layout = 'nhd'

    def __init__(self, kitchen, q_tile=64, kv_tile=64):
        self.kitchen = kitchen
        self.q_tile = q_tile
        self.kv_tile = kv_tile


class _Router:
    score_chunk_tiles = None

    def __init__(self, q_tile=64, kv_tile=64):
        self.q_tile = q_tile
        self.kv_tile = kv_tile


class _Module:
    heads = 2
    head_dim = 128
    out_proj = staticmethod(lambda value: value)


class StreamedKitchenOutputTests(unittest.TestCase):
    def _backend(self, kitchen, query_chunk_rows=128):
        projector = ChunkedKitchenQKVProjector(stream_output=True)
        return SparseKitchenBackend(
            kitchen=kitchen,
            executor=_Executor(kitchen),
            router=_Router(),
            projector=projector,
            stream_output=True,
            query_chunk_rows=query_chunk_rows,
        )

    def test_64q_routes_and_128q_scales_are_sliced_independently(self):
        q = torch.arange(2 * 257 * 128).to(torch.int8).reshape(1, 2, 257, 128)
        carrier = PrequantizedInt8Attention(
            q=q,
            k=torch.empty_like(q),
            v=torch.empty(2 * 128, 257, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 96),
            k_scale=torch.empty(1),
            v_scale=torch.empty(1),
            original_head_dim=128,
            input_dtype=torch.float32,
            attention_scale=1.0,
            cta_k=64,
        )
        route = BlockSparseRoute(
            indices=torch.zeros(1, 2, 5, 1, dtype=torch.int32),
            counts=torch.ones(1, 2, 5, dtype=torch.int32),
            q_tile=64,
            kv_tile=64,
            encoding='delta',
        )
        output = torch.empty(257, 256)
        prepared = PreparedSparseKitchen(
            carrier, route, 128, 0, {}, output_buffer=output
        )
        kitchen = _Kitchen()

        actual = self._backend(kitchen).execute_projected(_Module(), prepared)

        expected = q.transpose(1, 2).reshape(1, 257, 256).squeeze(0).float()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            kitchen.calls,
            [(128, 2, 'nhd'), (128, 2, 'nhd'), (1, 1, 'nhd')],
        )
        self.assertIsNone(prepared.quantized)
        self.assertIsNone(prepared.route)
        self.assertIsNone(prepared.output_buffer)

    def test_streaming_requires_a_capturing_projector(self):
        kitchen = _Kitchen()
        with self.assertRaisesRegex(RuntimeError, 'output-capturing projector'):
            SparseKitchenBackend(
                kitchen=kitchen,
                executor=_Executor(kitchen),
                router=_Router(),
                projector=ChunkedKitchenQKVProjector(),
                stream_output=True,
            )

    def test_sparse_kitchen_keeps_lazy_residual_separate_from_output(self):
        sequence = 128
        q = torch.arange(2 * sequence * 128).to(torch.int8).reshape(
            1, 2, sequence, 128
        )
        carrier = PrequantizedInt8Attention(
            q=q,
            k=torch.empty_like(q),
            v=torch.empty(2 * 128, sequence, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 32),
            k_scale=torch.empty(1),
            v_scale=torch.empty(1),
            original_head_dim=128,
            input_dtype=torch.float32,
            attention_scale=1.0,
            cta_k=64,
        )
        route = BlockSparseRoute(
            indices=torch.zeros(1, 2, 2, 1, dtype=torch.int32),
            counts=torch.ones(1, 2, 2, dtype=torch.int32),
            q_tile=64,
            kv_tile=64,
            encoding='delta',
        )
        residual = torch.full((sequence, 256), 11.0)
        source = NormalizedRows(
            residual,
            lambda rows: rows.clone(),
            ((0, sequence, 0),),
            None,
            None,
            lambda rows, _shift, _scale, _selector: rows,
        )
        source.output_buffer().fill_(-7)
        prepared = PreparedSparseKitchen(
            carrier,
            route,
            128,
            0,
            {},
            output_buffer=source,
        )

        actual = self._backend(_Kitchen(), query_chunk_rows=128).execute_projected(
            _Module(),
            prepared,
        )

        expected = q.transpose(1, 2).reshape(sequence, 256).float()
        self.assertIs(actual, source.output_buffer())
        self.assertTrue(torch.equal(residual, source.materialize()))
        torch.testing.assert_close(actual, expected)

    def test_low_level_defaults_remain_opt_in(self):
        self.assertFalse(ChunkedKitchenQKVProjector().stream_output)
        kitchen = _Kitchen()
        backend = SparseKitchenBackend(
            kitchen=kitchen,
            executor=_Executor(kitchen),
            router=_Router(),
        )
        self.assertFalse(backend.stream_output)

    def test_dense_kitchen_streams_query_and_output_slabs(self):
        q = torch.arange(2 * 257 * 128).to(torch.int8).reshape(1, 2, 257, 128)
        carrier = PrequantizedInt8Attention(
            q=q,
            k=torch.empty_like(q),
            v=torch.empty(2 * 128, 257, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 96),
            k_scale=torch.empty(1),
            v_scale=torch.empty(1),
            original_head_dim=128,
            input_dtype=torch.float32,
            attention_scale=1.0,
            cta_k=64,
        )
        output = torch.empty(257, 256)
        prepared = PreparedChunkedKitchenQKV(
            carrier,
            output_buffer=output,
        )
        kitchen = _DenseKitchen()
        backend = ChunkedKitchenAttentionBackend(
            stream_output=True,
            query_chunk_rows=128,
        )

        with mock.patch(
            'h3_optimizations.kitchen_qkv.resolve_kitchen',
            return_value=kitchen,
        ):
            actual = backend.execute_projected(_Module(), prepared)

        expected = q.transpose(1, 2).reshape(1, 257, 256).squeeze(0).float()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(kitchen.calls, [(128, 'nhd'), (128, 'nhd'), (1, 'nhd')])
        self.assertIsNone(prepared.carrier)
        self.assertIsNone(prepared.output_buffer)

    def test_dense_kitchen_streamed_q_keeps_lazy_input_separate_from_output(self):
        sequence = 256
        residual = (
            torch.arange(sequence * 256).reshape(sequence, 256) % 32
        ).to(torch.float32)
        source = NormalizedRows(
            residual,
            lambda rows: rows.clone(),
            ((0, sequence, 0),),
            None,
            None,
            lambda rows, _shift, _scale, _selector: rows,
        )
        source.output_buffer().fill_(-7)
        carrier = PrequantizedInt8Attention(
            q=torch.empty(1, 2, 1, 128, dtype=torch.int8),
            k=torch.empty(1, 2, sequence, 128, dtype=torch.int8),
            v=torch.empty(2 * 128, sequence, dtype=torch.int8),
            q_scale=torch.empty(1, 2, 1),
            k_scale=torch.empty(1),
            v_scale=torch.empty(1),
            original_head_dim=128,
            input_dtype=torch.float32,
            attention_scale=1.0,
            cta_k=64,
        )
        prepared = PreparedStreamedKitchenQKV(
            module=_Module(),
            x=source,
            rope_freqs=None,
            carrier=carrier,
            projection_mode='native',
            output_buffer=source,
        )

        class Kitchen(_DenseKitchen):
            @staticmethod
            def quantize_int8_attention_q(q, **_kwargs):
                return q.to(torch.int8), torch.ones(1, 2, 1)

        class Held:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def project_q(_held, rows, _rope, start, stop):
            count = stop - start
            return (
                rows[start:stop]
                .reshape(count, 2, 128)
                .transpose(0, 1)
                .unsqueeze(0)
                .contiguous()
            )

        kitchen = Kitchen()
        backend = ChunkedKitchenAttentionBackend(
            stream_output=True,
            query_chunk_rows=128,
        )
        held = mock.MagicMock(wraps=Held())
        with mock.patch(
            'h3_optimizations.kitchen_qkv.resolve_kitchen',
            return_value=kitchen,
        ), mock.patch(
            'h3_optimizations.kitchen_qkv.create_held_qkv',
            return_value=held,
        ) as acquire, mock.patch(
            'h3_optimizations.kitchen_qkv.project_q_hnd',
            side_effect=project_q,
        ):
            actual = backend.execute_projected(_Module(), prepared)

        self.assertIs(actual, source.output_buffer())
        acquire.assert_called_once()
        held.__enter__.assert_called_once()
        held.__exit__.assert_called_once()
        self.assertTrue(torch.equal(residual, source.materialize()))
        torch.testing.assert_close(actual, residual)

    def test_dense_streamed_q_releases_weights_and_carrier_on_error(self):
        for failure in ('acquire', 'projection', 'quantize', 'attention', 'output'):
            with self.subTest(failure=failure):
                module = mock.Mock(heads=2, head_dim=128)
                module.out_proj.side_effect = lambda rows: rows
                source = torch.zeros(257, 256)
                prepared = PreparedStreamedKitchenQKV(
                    module, source, None, object(), 'native', source,
                )
                held = mock.MagicMock()
                error = RuntimeError('failed ' + failure)
                kernel = mock.Mock(side_effect=lambda chunk, **kwargs: chunk.q)

                def project(binding, rows, rope, start, stop):
                    self.assertIs(binding, held)
                    held.__enter__.assert_called_once()
                    held.__exit__.assert_not_called()
                    return rows[start:stop].reshape(1, stop - start, 2, 128).transpose(1, 2)

                projection = mock.Mock(side_effect=project)
                quantize = mock.Mock(side_effect=lambda kitchen, carrier, q: mock.Mock(q=q, q_scale=torch.ones(1)))
                failing = {
                    'acquire': held.__enter__,
                    'projection': projection,
                    'quantize': quantize,
                    'attention': kernel,
                    'output': module.out_proj,
                }[failure]
                failing.side_effect = error
                kitchen = mock.Mock(int8_attention_from_prequantized=kernel)
                backend = ChunkedKitchenAttentionBackend(stream_output=True, query_chunk_rows=128)
                with (
                    mock.patch('h3_optimizations.kitchen_qkv.resolve_kitchen', return_value=kitchen),
                    mock.patch('h3_optimizations.kitchen_qkv.create_held_qkv', return_value=held) as acquire,
                    mock.patch('h3_optimizations.kitchen_qkv.project_q_hnd', projection),
                    mock.patch('h3_optimizations.kitchen_qkv._quantize_q_chunk', quantize),
                    self.assertRaisesRegex(RuntimeError, 'failed ' + failure),
                ):
                    backend.execute_projected(module, prepared)
                acquire.assert_called_once()
                if failure == 'acquire':
                    held.__exit__.assert_not_called()
                else:
                    held.__exit__.assert_called_once()
                self.assertIsNone(prepared.carrier)
                self.assertIsNone(prepared.x)
                self.assertIsNone(prepared.output_buffer)

    def test_sparse_kitchen_streamed_q_keeps_lazy_input_separate_from_output(self):
        class Held:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class Kitchen:
            BlockSparseRoute = staticmethod(lambda **_kwargs: object())

            @staticmethod
            def block_sparse_int8_attention_from_prequantized(
                carrier, _route, *, output_layout
            ):
                return carrier

        sequence = 128
        residual = torch.arange(sequence, dtype=torch.float32).reshape(sequence, 1)
        source = NormalizedRows(
            residual,
            lambda rows: rows.clone(),
            ((0, sequence, 0),),
            None,
            None,
            lambda rows, _shift, _scale, _selector: rows,
        )
        module = type(
            'Module',
            (),
            {
                'heads': 1,
                'head_dim': 1,
                'out_proj': staticmethod(lambda rows: rows),
            },
        )()
        projected = StreamedSparseKitchenQKV(
            module=module,
            producer_module=object(),
            x=source,
            rope_freqs=None,
            carrier=object(),
            k_summary=None,
            projection_mode='native',
            output_buffer=source,
        )
        prepared = PreparedStreamedSparseKitchen(
            projected=projected,
            route_plan=type('RoutePlan', (), {'release': lambda self: None})(),
            metadata={},
        )
        backend = StreamedSparseKitchenBackend.__new__(StreamedSparseKitchenBackend)
        backend.executor = type(
            'Executor',
            (),
            {'kitchen': Kitchen(), 'q_tile': 64, 'kv_tile': 64},
        )()
        backend.query_chunk_rows = 64
        backend.router = object()

        def project_q(_held, rows, _rope, start, stop):
            return rows[start:stop].transpose(0, 1).reshape(1, 1, stop - start, 1)

        source.output_buffer().fill_(-7)
        with mock.patch.object(
            kitchen_streamed_q,
            'create_held_qkv',
            return_value=Held(),
        ), mock.patch.object(
            kitchen_streamed_q,
            'project_q_hnd',
            side_effect=project_q,
        ), mock.patch.object(
            kitchen_streamed_q,
            '_build_route_chunk',
            return_value=(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1)),
        ), mock.patch.object(
            kitchen_streamed_q,
            '_quantize_q_chunk',
            side_effect=lambda _producer, _carrier, q: q,
        ):
            actual = backend.execute_projected(module, prepared)

        self.assertIs(actual, source.output_buffer())
        self.assertTrue(torch.equal(residual, source.materialize()))
        torch.testing.assert_close(actual, residual)

    def test_sparse_kitchen_streamed_q_writes_projected_slabs_into_residual(self):
        class Held:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class Kitchen:
            BlockSparseRoute = staticmethod(lambda **_kwargs: object())

            @staticmethod
            def block_sparse_int8_attention_from_prequantized(
                carrier, _route, *, output_layout
            ):
                return carrier

        sequence = 128
        original = torch.arange(sequence, dtype=torch.float32).reshape(sequence, 1)
        residual = original.clone()
        source = NormalizedRows(
            residual,
            lambda rows: rows.clone(),
            ((0, 32, 0), (32, sequence, 1)),
            None,
            None,
            lambda rows, _shift, _scale, _selector: rows,
        )
        gate = torch.tensor([[0.5], [1.0]])
        source.enable_attention_residual(gate)

        def project_residual(rows, target, scale):
            target.addcmul_(rows, scale)
            return True

        module = type(
            'Module',
            (),
            {
                'heads': 1,
                'head_dim': 1,
                'out_proj': staticmethod(lambda rows: rows),
                '_h3_out_proj_residual': staticmethod(project_residual),
            },
        )()
        projected = StreamedSparseKitchenQKV(
            module=module,
            producer_module=object(),
            x=source,
            rope_freqs=None,
            carrier=object(),
            k_summary=None,
            projection_mode='native',
            output_buffer=source,
        )
        prepared = PreparedStreamedSparseKitchen(
            projected=projected,
            route_plan=type('RoutePlan', (), {'release': lambda self: None})(),
            metadata={},
        )
        backend = StreamedSparseKitchenBackend.__new__(StreamedSparseKitchenBackend)
        backend.executor = type(
            'Executor',
            (object,),
            {'kitchen': Kitchen(), 'q_tile': 64, 'kv_tile': 64},
        )()
        backend.query_chunk_rows = 64
        backend.router = object()

        def project_q(_held, rows, _rope, start, stop):
            return rows[start:stop].transpose(0, 1).reshape(1, 1, stop - start, 1)

        with mock.patch.object(
            kitchen_streamed_q,
            'create_held_qkv',
            return_value=Held(),
        ), mock.patch.object(
            kitchen_streamed_q,
            'project_q_hnd',
            side_effect=project_q,
        ), mock.patch.object(
            kitchen_streamed_q,
            '_build_route_chunk',
            return_value=(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1)),
        ), mock.patch.object(
            kitchen_streamed_q,
            '_quantize_q_chunk',
            side_effect=lambda _producer, _carrier, q: q,
        ):
            actual = backend.execute_projected(module, prepared)

        expected = original.clone()
        expected[:32].addcmul_(original[:32], gate[0])
        expected[32:].addcmul_(original[32:], gate[1])
        self.assertIs(actual, residual)
        self.assertIsNone(source._output)
        self.assertTrue(source.attention_residual_complete)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
