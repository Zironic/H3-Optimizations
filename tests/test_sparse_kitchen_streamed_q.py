"""CPU contracts for Sparse Kitchen's streamed-query routing composition."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from dataclasses import dataclass, replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.kitchen_streamed_q import (
    PreparedStreamedSparseKitchen,
    StreamedSparseKitchenBackend,
    StreamedSparseKitchenQKV,
    _build_route_chunk,
    _prepare_route_plan,
)
from h3_optimizations.attention.sparse import kitchen_streamed_q
from h3_optimizations.attention.sparse.router import SparseTileRouter
from h3_optimizations.attention.sparse import kitchen_sparse
from h3_optimizations.kitchen_qkv import ChunkedKitchenQKVProjector
from h3_optimizations.kitchen_qkv import V_MODE_RETAIN, V_MODE_TWO_PASS
from h3_optimizations.normalized_rows import NormalizedRows

sys.argv = [sys.argv[0], *TEST_ARGS]


class _Layout:
    def __init__(self, sequence=512, video_start=128):
        self.seq_len = sequence
        self.video_range = (video_start, sequence)
        self.segments = (
            (0, 64, "text"),
            (64, video_start, "audio"),
            (video_start, sequence, "video"),
        )
        self.video_shape = (1, 1, sequence - video_start)
        self.audio_t = video_start - 64


class StreamedSparseKitchenRoutingTests(unittest.TestCase):
    def test_exact_fused_q_eligibility_receives_a_tensor_from_lazy_norm(self):
        class StopAfterEligibility(Exception):
            pass

        def modulate(rows, _shift, _scale, _selector):
            rows.add_(0.25)

        source = NormalizedRows(
            torch.zeros(64, 256, dtype=torch.bfloat16),
            lambda rows: rows.clone(),
            ((0, 64, 0),),
            None,
            None,
            modulate,
        )
        source.is_cuda = True
        projector = SimpleNamespace(
            force_weights_bf16=False,
            fp8_projection=False,
            convrot_int8_projection=False,
            q_tile=64,
            kv_tile=64,
            chunk_rows=64,
        )
        module = SimpleNamespace(qkv_proj=object())
        rope = torch.zeros(1, 64, 1, 48, 2, 2, dtype=torch.bfloat16)

        with (
            mock.patch.object(kitchen_streamed_q, "resolve_kitchen", return_value=object()),
            mock.patch.object(
                kitchen_streamed_q, "_supports_streamed_producer", return_value=True
            ),
            mock.patch.object(
                kitchen_streamed_q,
                "describe_linear",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(kitchen_streamed_q, "_format_supported", return_value=True),
            mock.patch.object(
                kitchen_streamed_q, "producer_api_available", return_value=True
            ),
            mock.patch.object(
                kitchen_streamed_q,
                "fused_h3_q_supported",
                side_effect=StopAfterEligibility,
            ) as supported,
        ):
            with self.assertRaises(StopAfterEligibility):
                kitchen_streamed_q._run_streamed_sparse_kitchen_qkv(
                    projector,
                    module,
                    source,
                    rope,
                    layer_index=0,
                    transformer_options={},
                )

        sample = supported.call_args.args[1]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(tuple(sample.shape), (1, 256))
        torch.testing.assert_close(
            sample,
            torch.full((1, 256), 0.25, dtype=torch.bfloat16),
        )

    def test_query_chunks_use_the_carrier_producer_not_the_sparse_executor(self):
        class Held:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class StopAfterQuantize(Exception):
            pass

        producer_module = object()
        sparse_executor_module = object()
        module = SimpleNamespace(heads=1, head_dim=1)
        projected = StreamedSparseKitchenQKV(
            module=module,
            producer_module=producer_module,
            x=torch.zeros(64, 1),
            rope_freqs=None,
            carrier=object(),
            k_summary=None,
            projection_mode="native",
            output_buffer=torch.empty(64, 1),
        )
        prepared = PreparedStreamedSparseKitchen(
            projected=projected,
            route_plan=SimpleNamespace(release=lambda: None),
            metadata={},
        )
        backend = StreamedSparseKitchenBackend.__new__(StreamedSparseKitchenBackend)
        backend.executor = SimpleNamespace(
            kitchen=sparse_executor_module,
            q_tile=64,
            kv_tile=64,
        )
        backend.query_chunk_rows = 64
        backend.router = object()

        with (
            mock.patch.object(kitchen_streamed_q, "create_held_qkv", return_value=Held()),
            mock.patch.object(
                kitchen_streamed_q,
                "project_q_hnd",
                return_value=torch.zeros(1, 1, 64, 1),
            ),
            mock.patch.object(
                kitchen_streamed_q,
                "_build_route_chunk",
                return_value=(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1)),
            ),
            mock.patch.object(
                kitchen_streamed_q,
                "_quantize_q_chunk",
                side_effect=StopAfterQuantize,
            ) as quantize,
        ):
            with self.assertRaises(StopAfterQuantize):
                backend.execute_projected(module, prepared)

        self.assertIs(quantize.call_args.args[0], producer_module)
        self.assertIsNot(quantize.call_args.args[0], sparse_executor_module)

    def test_production_sparse_projector_marks_streamed_q(self):
        projector = ChunkedKitchenQKVProjector(
            routing_summaries=True,
            q_tile=64,
            kv_tile=64,
            stream_output=True,
        )
        self.assertTrue(projector.streamed_q)

    def test_exact_fused_q_replaces_only_the_query_carrier_and_summary(self):
        @dataclass
        class Carrier:
            q: object
            k: torch.Tensor
            q_scale: object

        class Fused:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def project(self, x, rope, start, stop, full_k_length):
                self.calls.append((x, rope, start, stop, full_k_length))
                return (
                    torch.zeros(1, 1, stop - start, 1, dtype=torch.int8),
                    torch.ones(1, 1, 32),
                    torch.zeros(1, 1, 1, 1),
                )

        class Kitchen:
            BlockSparseRoute = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))

            @staticmethod
            def block_sparse_int8_attention_from_prequantized(
                carrier, _route, *, output_layout
            ):
                self.assertEqual(output_layout, kitchen_streamed_q.OUTPUT_NHD)
                self.assertEqual(carrier.q.dtype, torch.int8)
                return carrier.q.to(torch.float32)

        module = SimpleNamespace(heads=1, head_dim=1, out_proj=lambda rows: rows)
        source = torch.zeros(64, 1)
        rope = torch.zeros(1, 64, 1, 1)
        projected = StreamedSparseKitchenQKV(
            module=module,
            producer_module=object(),
            x=source,
            rope_freqs=rope,
            carrier=Carrier(
                q=object(),
                k=torch.zeros(1, 1, 64, 1),
                q_scale=object(),
            ),
            k_summary=None,
            projection_mode="native",
            output_buffer=source,
            fused_q=True,
        )
        prepared = PreparedStreamedSparseKitchen(
            projected=projected,
            route_plan=SimpleNamespace(release=lambda: None),
            metadata={},
        )
        backend = StreamedSparseKitchenBackend.__new__(StreamedSparseKitchenBackend)
        backend.executor = SimpleNamespace(kitchen=Kitchen(), q_tile=64, kv_tile=64)
        backend.query_chunk_rows = 64
        backend.router = object()
        fused = Fused()

        with (
            mock.patch.object(
                kitchen_streamed_q, "HeldExactH3FusedQ", return_value=fused
            ),
            mock.patch.object(
                kitchen_streamed_q,
                "_build_route_chunk",
                return_value=(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1)),
            ),
            mock.patch.object(
                kitchen_streamed_q,
                "_quantize_q_chunk",
                side_effect=AssertionError("generic Q pack must not run"),
            ),
            mock.patch.object(
                kitchen_streamed_q,
                "create_held_qkv",
                side_effect=AssertionError("generic Q projection must not run"),
            ),
        ):
            actual = backend.execute_projected(module, prepared)

        self.assertTrue(torch.equal(actual, source))
        self.assertEqual(len(fused.calls), 1)
        self.assertEqual(fused.calls[0][2:], (0, 64, 64))

    def test_dense_projector_default_is_unchanged(self):
        projector = ChunkedKitchenQKVProjector()
        self.assertFalse(projector.streamed_q)
        self.assertEqual(projector.v_mode, V_MODE_RETAIN)

    def test_two_pass_v_is_part_of_projector_identity(self):
        projector = ChunkedKitchenQKVProjector(v_mode=V_MODE_TWO_PASS)
        self.assertEqual(projector.v_mode, V_MODE_TWO_PASS)
        self.assertIn(V_MODE_TWO_PASS, projector.installation_signature)

    def test_sparse_backend_symbol_is_upgraded(self):
        self.assertIs(kitchen_sparse.SparseKitchenBackend, StreamedSparseKitchenBackend)

    def test_projector_fallback_does_not_require_shared_flag_mutation(self):
        import inspect
        from h3_optimizations.attention.sparse import kitchen_streamed_q

        source = inspect.getsource(kitchen_streamed_q._streamed_sparse_try_project)
        self.assertIn("fallback = copy.copy(self)", source)
        self.assertNotIn("self.streamed_q = False", source)

    def test_lazy_route_chunks_reconstruct_full_route(self):
        router = SparseTileRouter(q_tile=64, kv_tile=64)
        layout = _Layout()
        geometry = router.geometry(layout)
        heads = 2
        dim = 8

        k_summary = torch.arange(
            heads * geometry.kv_tiles * dim,
            dtype=torch.float32,
        ).reshape(1, heads, geometry.kv_tiles, dim)
        q_summary = torch.arange(
            heads * geometry.q_tiles * dim,
            dtype=torch.float32,
        ).reshape(1, heads, geometry.q_tiles, dim).flip(-1)

        budget = 0.25
        full_lut, full_counts, full_meta = router.build_lut_from_summaries(
            q_summary,
            k_summary,
            layout,
            budget,
        )
        plan, metadata = _prepare_route_plan(
            router,
            k_summary,
            layout,
            budget,
        )
        self.assertEqual(
            metadata.retained_video_kv_tiles,
            full_meta.retained_video_kv_tiles,
        )

        lut_parts = []
        count_parts = []
        chunk_tiles = 3
        for tile_start in range(0, geometry.q_tiles, chunk_tiles):
            tile_stop = min(tile_start + chunk_tiles, geometry.q_tiles)
            lut, counts = _build_route_chunk(
                router,
                plan,
                q_summary[..., tile_start:tile_stop, :],
                tile_start=tile_start,
            )
            lut_parts.append(lut)
            count_parts.append(counts)

        lazy_lut = torch.cat(lut_parts, dim=-2)
        lazy_counts = torch.cat(count_parts, dim=-1)
        self.assertTrue(torch.equal(lazy_lut, full_lut))
        self.assertTrue(torch.equal(lazy_counts, full_counts))


if __name__ == "__main__":
    unittest.main()
