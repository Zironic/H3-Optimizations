'''CPU contracts for sparse routing through an existing dense attention consumer.'''

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

import h3_optimizations.apply_policy as apply_policy  # noqa: E402
apply_module = apply_policy._base
from h3_optimizations.attention.sparse import existing_dense_sparse  # noqa: E402
from h3_optimizations.normalized_rows import NormalizedRows  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_TRITON,
    SparseRequest,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_FORCE_BF16_CHUNKED,
    QKV_STANDARD,
    QKVProviderResolution,
)
from h3_optimizations.qkv.projectors import (  # noqa: E402
    TritonSparseQKVProjector,
)
from h3_optimizations.qkv.w4a8 import W4A8BindingError  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def _reference(q, k, v):
    return existing_dense_sparse._reference_attention(q, k, v)


class ExistingDenseSparsePackingTests(unittest.TestCase):
    def test_per_head_routes_and_ragged_tail_match_independent_dense_attention(self):
        torch.manual_seed(7)
        q = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 2, 5, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 2, 5, 8), dtype=torch.bfloat16)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 4)
        selected = torch.tensor(
            [[
                [[0, 2], [0, 1]],
                [[1, 2], [0, 2]],
            ]],
            dtype=torch.int32,
        )

        with mock.patch.object(
            existing_dense_sparse,
            '_call_existing_dense',
            side_effect=lambda q, k, v, _options, *, heads: _reference(q, k, v),
        ):
            actual = existing_dense_sparse._execute_packed_sparse(
                q, k, v, selected, {}, spec=spec
            )

        expected = torch.empty_like(q)
        for head in range(2):
            for q_tile in range(2):
                rows = [
                    row
                    for tile in selected[0, head, q_tile].tolist()
                    for row in range(tile * 2, min(tile * 2 + 2, 5))
                ]
                q_rows = slice(q_tile * 2, q_tile * 2 + 2)
                expected[:, head:head + 1, q_rows, :] = _reference(
                    q[:, head:head + 1, q_rows, :],
                    k[:, head:head + 1, rows, :],
                    v[:, head:head + 1, rows, :],
                )
        self.assertTrue(torch.equal(actual, expected))

    def test_validate_qkv_accepts_certified_fp32(self):
        device = object()
        q = SimpleNamespace(
            shape=(1, 2, 4, 128),
            ndim=4,
            dtype=torch.float32,
            device=device,
            is_cuda=True,
            stride=lambda dim: 1,
        )
        existing_dense_sparse._validate_qkv(
            q,
            q,
            q,
            dtype=torch.float32,
        )

    def test_validate_qkv_rejects_a_dtype_not_certified_by_the_probe(self):
        device = object()
        q = SimpleNamespace(
            shape=(1, 2, 4, 128),
            ndim=4,
            dtype=torch.float32,
            device=device,
            is_cuda=True,
            stride=lambda dim: 1,
        )
        with self.assertRaisesRegex(
            existing_dense_sparse.ExistingDenseSparseError,
            'probed for torch.bfloat16',
        ):
            existing_dense_sparse._validate_qkv(
                q,
                q,
                q,
                dtype=torch.bfloat16,
            )

    def test_packed_consumer_failure_is_classified(self):
        q = torch.randn((1, 1, 2, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 1, 2, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 1, 2, 8), dtype=torch.bfloat16)
        selected = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 1)

        with (
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
                side_effect=RuntimeError('packed shape rejected'),
            ),
            self.assertRaisesRegex(
                existing_dense_sparse.ExistingDenseConsumerError,
                'packed shape rejected',
            ),
        ):
            existing_dense_sparse._execute_packed_sparse(
                q, k, v, selected, {}, spec=spec
            )

    def test_packed_failure_uses_full_existing_dense_for_the_q_slab(self):
        q = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 2, 6, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 2, 6, 8), dtype=torch.bfloat16)
        selected = torch.zeros((1, 2, 2, 1), dtype=torch.int32)
        expected = torch.randn_like(q)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 4)

        with (
            mock.patch.object(
                existing_dense_sparse,
                '_execute_packed_sparse',
                side_effect=existing_dense_sparse.ExistingDenseConsumerError(
                    'packed shape rejected'
                ),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
                return_value=expected,
            ) as dense,
        ):
            actual = existing_dense_sparse._execute_packed_or_dense(
                q, k, v, selected, {'test': True}, spec=spec
            )

        self.assertIs(actual, expected)
        args, kwargs = dense.call_args
        self.assertEqual(args[:3], (q, k, v))
        self.assertEqual(kwargs['heads'], 2)

    def test_non_consumer_packing_failure_is_not_hidden(self):
        q = torch.randn((1, 1, 2, 8), dtype=torch.bfloat16)
        selected = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 1)
        with (
            mock.patch.object(
                existing_dense_sparse,
                '_execute_packed_sparse',
                side_effect=existing_dense_sparse.ExistingDenseSparseError(
                    'invalid route'
                ),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
            ) as dense,
            self.assertRaisesRegex(
                existing_dense_sparse.ExistingDenseSparseError,
                'invalid route',
            ),
        ):
            existing_dense_sparse._execute_packed_or_dense(
                q, q, q, selected, {}, spec=spec
            )
        dense.assert_not_called()

    def test_cuda_oom_is_not_hidden(self):
        q = torch.randn((1, 1, 2, 8), dtype=torch.bfloat16)
        selected = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 1)
        with (
            mock.patch.object(
                existing_dense_sparse,
                '_execute_packed_sparse',
                side_effect=torch.cuda.OutOfMemoryError('synthetic OOM'),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
            ) as dense,
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            existing_dense_sparse._execute_packed_or_dense(
                q, q, q, selected, {}, spec=spec
            )
        dense.assert_not_called()

    def test_partial_final_q_tile_discards_padded_rows(self):
        torch.manual_seed(11)
        q = torch.randn((1, 1, 3, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 1, 4, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 1, 4, 8), dtype=torch.bfloat16)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 2)
        selected = torch.tensor([[[[0], [1]]]], dtype=torch.int32)

        with mock.patch.object(
            existing_dense_sparse,
            '_call_existing_dense',
            side_effect=lambda q, k, v, _options, *, heads: _reference(q, k, v),
        ):
            actual = existing_dense_sparse._execute_packed_sparse(
                q, k, v, selected, {}, spec=spec
            )

        expected = torch.empty_like(q)
        expected[..., :2, :] = _reference(
            q[..., :2, :], k[..., :2, :], v[..., :2, :]
        )
        expected[..., 2:3, :] = _reference(
            q[..., 2:3, :], k[..., 2:4, :], v[..., 2:4, :]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_streamed_execute_keeps_lazy_input_separate_from_output(self):
        sequence = 2
        residual = torch.arange(sequence * 128, dtype=torch.float32).reshape(
            sequence, 128
        )
        source = NormalizedRows(
            residual,
            lambda rows: rows.clone(),
            ((0, sequence, 0),),
            None,
            None,
            lambda rows, _shift, _scale, _selector: rows,
        )
        q = torch.arange(sequence * 128, dtype=torch.float32).reshape(
            1, 1, sequence, 128
        )
        module = SimpleNamespace(out_proj=lambda rows: rows)

        class Projected:
            def __init__(self):
                self.module = module
                self.x = source
                self.sequence = sequence
                self.heads = 1
                self.chunk_rows = sequence
                self.k = q.clone()
                self.v = q.clone()
                self.released = False

            def project_q(self, _start, _end):
                return q.clone()

            def release_weight(self):
                pass

            def release(self):
                self.released = True

        projected = Projected()
        prepared = existing_dense_sparse.PreparedStreamedExistingDenseSparse(
            projected=projected,
            route_plan=SimpleNamespace(release=lambda: None),
            dense_q_tiles=1,
            sparse_q_tiles=0,
            metadata={},
            transformer_options={},
        )
        backend = existing_dense_sparse.ExistingDenseSparseBackend(
            spec=existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 1)
        )

        with (
            mock.patch.object(
                existing_dense_sparse,
                'build_compact_absolute_route_chunk',
                return_value=torch.zeros((1, 1, 1, 1), dtype=torch.int32),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
                return_value=q,
            ),
        ):
            actual = backend.execute_projected(module, prepared)

        self.assertIs(actual, source.output_buffer())
        self.assertTrue(torch.equal(actual, q.reshape(sequence, 128)))
        self.assertTrue(torch.equal(source.materialize(), residual))
        self.assertTrue(projected.released)


class ExistingDenseSparseProjectorTests(unittest.TestCase):
    @staticmethod
    def _input(dtype):
        return SimpleNamespace(ndim=2, is_cuda=True, dtype=dtype)

    def test_optional_projector_declines_fp32_for_stock_qkv(self):
        projector = existing_dense_sparse.ExistingDenseSparseQKVProjector(
            required=False,
            projection_mode='native',
        )
        self.assertIsNone(
            projector.try_project(
                object(),
                self._input(torch.float32),
                None,
                layer_index=0,
                transformer_options={},
            )
        )

    def test_optional_projector_declines_runtime_binding_failure(self):
        projector = existing_dense_sparse.ExistingDenseSparseQKVProjector(
            required=False,
            projection_mode='native',
        )
        with mock.patch.object(
            TritonSparseQKVProjector,
            'try_project',
            side_effect=W4A8BindingError(
                'W4A8 acquisition materialized an unquantized weight'
            ),
        ):
            self.assertIsNone(
                projector.try_project(
                    object(),
                    self._input(torch.bfloat16),
                    None,
                    layer_index=0,
                    transformer_options={},
                )
            )

    def test_required_projector_does_not_hide_runtime_binding_failure(self):
        projector = existing_dense_sparse.ExistingDenseSparseQKVProjector(
            required=True,
            projection_mode='native',
        )
        with (
            mock.patch.object(
                TritonSparseQKVProjector,
                'try_project',
                side_effect=W4A8BindingError(
                    'W4A8 acquisition materialized an unquantized weight'
                ),
            ),
            self.assertRaises(W4A8BindingError),
        ):
            projector.try_project(
                object(),
                self._input(torch.bfloat16),
                None,
                layer_index=0,
                transformer_options={},
            )

    def test_optional_projector_does_not_hide_unrelated_failures(self):
        projector = existing_dense_sparse.ExistingDenseSparseQKVProjector(
            required=False,
            projection_mode='native',
        )
        with (
            mock.patch.object(
                TritonSparseQKVProjector,
                'try_project',
                side_effect=RuntimeError('invalid projected route'),
            ),
            self.assertRaisesRegex(RuntimeError, 'invalid projected route'),
        ):
            projector.try_project(
                object(),
                self._input(torch.bfloat16),
                None,
                layer_index=0,
                transformer_options={},
            )


class ExistingDenseSparseProbeTests(unittest.TestCase):
    def setUp(self):
        existing_dense_sparse._probe_results.clear()

    def test_probe_prefers_64_then_falls_back_to_128(self):
        calls = []

        def geometry(_device, _options, q_tile, kv_tile, *, dtype):
            calls.append((q_tile, kv_tile))
            self.assertEqual(dtype, torch.bfloat16)
            if q_tile == 64:
                raise RuntimeError('64 rejected')
            return 4

        with (
            mock.patch.object(
                existing_dense_sparse,
                '_probe_key',
                return_value=('test',),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_probe_geometry',
                side_effect=geometry,
            ),
        ):
            spec = existing_dense_sparse.probe_existing_dense_sparse(device='cpu')

        self.assertEqual(calls, [(64, 64), (128, 128)])
        self.assertEqual((spec.q_tile, spec.kv_tile), (128, 128))
        self.assertEqual(spec.max_batch_entries, 4)
        self.assertEqual(spec.dtype, torch.bfloat16)

    def test_probe_accepts_established_dense_sage_relative_error(self):
        def approximate(q, k, v, _options, *, heads):
            return _reference(q, k, v) * 1.04

        with (
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
                side_effect=approximate,
            ),
        ):
            existing_dense_sparse._probe_case(
                torch.device('cpu'),
                {},
                q_rows=4,
                k_rows=5,
                batch=1,
            )

    def test_probe_uses_requested_fp32_execution_dtype(self):
        observed = []

        def exact(q, k, v, _options, *, heads):
            observed.append((q.dtype, k.dtype, v.dtype))
            return _reference(q, k, v)

        with (
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
                side_effect=exact,
            ),
        ):
            existing_dense_sparse._probe_case(
                torch.device('cpu'),
                {},
                q_rows=4,
                k_rows=5,
                batch=1,
                dtype=torch.float32,
            )

        self.assertEqual(
            observed,
            [(torch.float32, torch.float32, torch.float32)],
        )

    def test_probe_geometry_does_not_reduce_batch_after_cuda_oom(self):
        with (
            mock.patch.object(
                existing_dense_sparse,
                '_probe_case',
                side_effect=torch.cuda.OutOfMemoryError('synthetic OOM'),
            ) as probe_case,
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            existing_dense_sparse._probe_geometry(
                torch.device('cpu'), {}, 64, 64
            )
        probe_case.assert_called_once()

    def test_probe_does_not_try_another_geometry_after_cuda_oom(self):
        with (
            mock.patch.object(
                existing_dense_sparse,
                '_probe_key',
                return_value=('test',),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_probe_geometry',
                side_effect=torch.cuda.OutOfMemoryError('synthetic OOM'),
            ) as probe_geometry,
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            existing_dense_sparse.probe_existing_dense_sparse(device='cpu')
        probe_geometry.assert_called_once()

    def test_probe_key_uses_cuda_compute_capability_without_gcn_arch(self):
        override = object()
        properties = SimpleNamespace(major=8, minor=9)
        with (
            mock.patch.object(
                torch.cuda,
                'get_device_properties',
                return_value=properties,
            ),
            mock.patch.object(
                torch.cuda,
                'get_device_name',
                return_value='Test CUDA',
            ),
        ):
            key = existing_dense_sparse._probe_key(
                torch.device('cuda:0'),
                {'optimized_attention_override': override},
                dtype=torch.float32,
            )

        self.assertEqual(
            key,
            (0, 'sm89', 'Test CUDA', id(override), torch.float32),
        )


class ExistingDenseSparseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.override = object()
        self.execution_dtype = torch.bfloat16
        self.model = SimpleNamespace(
            model_options={
                'transformer_options': {
                    'optimized_attention_override': self.override,
                }
            },
            get_model_object=lambda name: self.execution_dtype,
            model=SimpleNamespace(
                get_dtype_inference=lambda: self.execution_dtype,
            ),
        )
        self.inventory = SimpleNamespace()
        self.environment = SimpleNamespace(
            cuda_available=True,
            device_index=0,
            capability=(8, 9),
        )

    def test_auto_unknown_override_selects_sparse_over_existing_dense(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_AUTO)
        )
        spec = existing_dense_sparse.ExistingDenseSparseSpec(64, 64, 8)
        qkv = QKVProviderResolution('standard_h3_qkv', False, 'synthetic')

        with (
            mock.patch.object(
                apply_module,
                'is_installed_dense_attention',
                return_value=False,
            ),
            mock.patch.object(
                apply_module,
                'is_comfy_kitchen_dense_attention',
                return_value=False,
            ),
            mock.patch.object(
                apply_module,
                'resolve_current_dense_attention',
                return_value=SimpleNamespace(
                    selected=apply_module.ATTENTION_EXISTING,
                    backend_kind=apply_module.ATTENTION_EXISTING_FULL_Q,
                ),
            ),
            mock.patch.object(
                apply_module,
                'probe_existing_dense_sparse',
                return_value=spec,
            ),
            mock.patch.object(
                apply_module,
                'resolve_qkv_provider',
                return_value=qkv,
            ),
        ):
            attention, actual_qkv = apply_module._resolve_attention(
                plan,
                self.model,
                self.inventory,
                self.environment,
            )

        self.assertIs(actual_qkv, qkv)
        self.assertEqual(attention.selected, apply_module.ATTENTION_EXISTING_SPARSE)
        self.assertEqual(attention.backend.name, 'existing_dense_sparse')
        self.assertIn('64Q x 64KV', attention.reason)

    def test_fp32_auto_uses_stock_qkv_and_fp32_probe(self):
        self.execution_dtype = torch.float32
        plan = H3OptimizationPlan(sparse=SparseRequest())
        spec = existing_dense_sparse.ExistingDenseSparseSpec(
            64,
            64,
            8,
            dtype=torch.float32,
        )
        dense_qkv = QKVProviderResolution(
            QKV_FORCE_BF16_CHUNKED,
            False,
            'dense fallback',
        )
        dense = apply_module.ResolvedAttention(
            requested=apply_module.ATTENTION_SPARSE,
            selected=apply_module.ATTENTION_EXISTING,
            backend=None,
            reason='preserved unknown override',
            backend_kind=apply_module.ATTENTION_EXISTING_FULL_Q,
        )
        with (
            mock.patch.object(
                apply_module,
                'probe_existing_dense_sparse',
                return_value=spec,
            ) as probe,
            mock.patch.object(
                apply_module,
                'resolve_qkv_provider',
            ) as resolve_qkv,
        ):
            attention, qkv = apply_module._resolve_existing_dense_sparse(
                plan,
                self.model,
                self.inventory,
                self.environment,
                dense,
                dense_qkv,
            )

        probe.assert_called_once_with(
            self.model.model_options['transformer_options'],
            device=torch.device('cuda:0'),
            dtype=torch.float32,
        )
        resolve_qkv.assert_not_called()
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertIsNone(attention.projector)
        self.assertEqual(attention.backend.spec.dtype, torch.float32)

    def test_explicit_backend_does_not_use_existing_dense_fallback(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_TRITON)
        )
        with (
            mock.patch.object(
                apply_module,
                'is_installed_dense_attention',
                return_value=False,
            ),
            mock.patch.object(
                apply_module,
                'is_comfy_kitchen_dense_attention',
                return_value=False,
            ),
            mock.patch.object(
                apply_module,
                'resolve_current_dense_attention',
                return_value=SimpleNamespace(
                    selected=apply_module.ATTENTION_EXISTING,
                    backend_kind=apply_module.ATTENTION_EXISTING_FULL_Q,
                ),
            ),
            mock.patch.object(
                apply_module,
                'probe_existing_dense_sparse',
                side_effect=AssertionError('must not probe'),
            ),
        ):
            attention, _qkv = apply_module._resolve_attention(
                plan,
                self.model,
                self.inventory,
                self.environment,
            )
        self.assertEqual(attention.selected, apply_module.ATTENTION_EXISTING)
        self.assertIn('sparse attention is disabled', attention.reason)

    def test_streamable_qkv_uses_existing_dense_sparse_projector(self):
        plan = H3OptimizationPlan(sparse=SparseRequest())
        spec = existing_dense_sparse.ExistingDenseSparseSpec(64, 64, 8)
        qkv = QKVProviderResolution(
            QKV_FORCE_BF16_CHUNKED,
            True,
            'synthetic forced BF16',
        )
        dense = apply_module.ResolvedAttention(
            requested=apply_module.ATTENTION_SPARSE,
            selected=apply_module.ATTENTION_EXISTING,
            backend=None,
            reason='preserved unknown override',
            backend_kind=apply_module.ATTENTION_EXISTING_FULL_Q,
        )
        with (
            mock.patch.object(
                apply_module,
                'probe_existing_dense_sparse',
                return_value=spec,
            ),
            mock.patch.object(
                apply_module,
                'resolve_qkv_provider',
                return_value=qkv,
            ),
        ):
            attention, actual_qkv = apply_module._resolve_existing_dense_sparse(
                plan,
                self.model,
                self.inventory,
                self.environment,
                dense,
                object(),
            )

        self.assertIs(actual_qkv, qkv)
        self.assertIsInstance(
            attention.projector,
            existing_dense_sparse.ExistingDenseSparseQKVProjector,
        )
        self.assertTrue(attention.projector.required)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
