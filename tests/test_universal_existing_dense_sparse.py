'''Architecture-neutral final sparse fallback policy contracts.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
PACK = Path(__file__).resolve().parents[1]
ROOT = next(parent for parent in PACK.parents if (parent / 'comfy').is_dir())
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations import apply as base_apply  # noqa: E402
from h3_optimizations import apply_policy  # noqa: F401,E402
from h3_optimizations import amd_policy  # noqa: F401,E402
from h3_optimizations import universal_sparse_fallback as universal  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_FORCE_BF16,
    H3OptimizationPlan,
    MemoryRequest,
    SparseRequest,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class UniversalExistingDenseSparsePolicyTests(unittest.TestCase):
    def setUp(self):
        universal._probe_results.clear()
        universal._runtime_fallback_warned = False
        self.environment = SimpleNamespace(
            backend='cuda',
            device_index=0,
            cuda_available=True,
        )
        self.model = SimpleNamespace(model_options={'transformer_options': {}})
        self.dense_resolution = object()
        self.dense = base_apply.ResolvedAttention(
            requested=base_apply.ATTENTION_SPARSE,
            selected=base_apply.ATTENTION_EXISTING,
            backend=None,
            reason='all specialized sparse backends unavailable; existing dense',
            backend_kind=base_apply.ATTENTION_EXISTING,
            dense_resolution=self.dense_resolution,
        )
        self.qkv = object()

    def test_auto_promotes_existing_dense_to_final_sparse_fallback(self):
        with mock.patch.object(
            universal,
            '_PREVIOUS_RESOLVE_ATTENTION',
            return_value=(self.dense, self.qkv),
        ):
            attention, qkv = universal.resolve_attention(
                H3OptimizationPlan(sparse=SparseRequest()),
                self.model,
                SimpleNamespace(),
                self.environment,
            )

        self.assertEqual(
            attention.selected,
            universal.SELECTED_EXISTING_DENSE_SPARSE,
        )
        self.assertIsInstance(
            attention.backend,
            universal.UniversalExistingDenseSparseBackend,
        )
        self.assertEqual(attention.backend_kind, base_apply.ATTENTION_TRITON_SPARSE)
        self.assertIsNone(attention.projector)
        self.assertIs(attention.dense_resolution, self.dense_resolution)
        self.assertEqual(qkv.provider_id, 'standard_h3_qkv')
        self.assertFalse(qkv.fused)
        self.assertIn('stock Comfy QKV', qkv.reason)
        self.assertIn('runtime-probe', attention.reason)

    def test_existing_sparse_result_is_left_unchanged(self):
        sparse = base_apply.ResolvedAttention(
            requested=base_apply.ATTENTION_SPARSE,
            selected=base_apply.ATTENTION_TRITON_SPARSE,
            backend=object(),
            reason='BF16 Triton works',
            backend_kind=base_apply.ATTENTION_TRITON_SPARSE,
        )
        expected = (sparse, self.qkv)
        with mock.patch.object(
            universal,
            '_PREVIOUS_RESOLVE_ATTENTION',
            return_value=expected,
        ):
            actual = universal.resolve_attention(
                H3OptimizationPlan(sparse=SparseRequest()),
                self.model,
                SimpleNamespace(),
                self.environment,
            )
        self.assertEqual(actual, expected)

    def test_unknown_full_q_override_is_not_packed(self):
        dense = base_apply.ResolvedAttention(
            requested=base_apply.ATTENTION_SPARSE,
            selected=base_apply.ATTENTION_EXISTING,
            backend=None,
            reason='preserved external override',
            backend_kind=base_apply.ATTENTION_EXISTING_FULL_Q,
            dense_resolution=self.dense_resolution,
        )
        expected = (dense, self.qkv)
        with mock.patch.object(
            universal,
            '_PREVIOUS_RESOLVE_ATTENTION',
            return_value=expected,
        ):
            actual = universal.resolve_attention(
                H3OptimizationPlan(sparse=SparseRequest()),
                self.model,
                SimpleNamespace(),
                self.environment,
            )
        self.assertEqual(actual, expected)

    def test_explicit_forced_qkv_policy_keeps_dense_fallback(self):
        plan = H3OptimizationPlan(
            memory=MemoryRequest(fused_qkv=FUSED_QKV_FORCE_BF16),
            sparse=SparseRequest(),
        )
        expected = (self.dense, self.qkv)
        with mock.patch.object(
            universal,
            '_PREVIOUS_RESOLVE_ATTENTION',
            return_value=expected,
        ):
            actual = universal.resolve_attention(
                plan,
                self.model,
                SimpleNamespace(),
                self.environment,
            )
        self.assertEqual(actual, expected)

    def test_runtime_qkv_validator_accepts_h3_float_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                q = torch.empty((1, 1, 8, 128), dtype=dtype)
                with mock.patch.object(
                    torch.Tensor,
                    'is_cuda',
                    new_callable=mock.PropertyMock,
                    return_value=True,
                ):
                    universal._validate_qkv(q, q, q)

    def test_probe_cache_is_separate_per_runtime_dtype(self):
        observed = []

        def geometry(device, options, q_tile, kv_tile, dtype):
            del device, options, q_tile, kv_tile
            observed.append(dtype)
            return 4

        with (
            mock.patch.object(
                universal._dense_sparse,
                '_probe_key',
                return_value=(0, 'test', 'gpu', 123),
            ),
            mock.patch.object(universal, '_probe_geometry', side_effect=geometry),
        ):
            bf16 = universal.probe_existing_dense_sparse(
                device='cpu',
                dtype=torch.bfloat16,
            )
            fp32 = universal.probe_existing_dense_sparse(
                device='cpu',
                dtype=torch.float32,
            )
            bf16_cached = universal.probe_existing_dense_sparse(
                device='cpu',
                dtype=torch.bfloat16,
            )

        self.assertEqual((bf16.q_tile, bf16.kv_tile), (64, 64))
        self.assertEqual((fp32.q_tile, fp32.kv_tile), (64, 64))
        self.assertIs(bf16_cached, bf16)
        self.assertEqual(observed, [torch.bfloat16, torch.float32])

    def test_runtime_probe_failure_defers_to_dense_execution(self):
        q = torch.randn((1, 1, 8, 128), dtype=torch.bfloat16)
        backend = universal.UniversalExistingDenseSparseBackend()
        with mock.patch.object(
            universal,
            'probe_existing_dense_sparse',
            side_effect=RuntimeError('consumer probe failed'),
        ):
            prepared = backend.prepare(
                q,
                q,
                q,
                layer_index=2,
                transformer_options={},
            )
        self.assertIsNotNone(prepared.fallback_reason)
        self.assertIn('runtime dtype/geometry probe', prepared.fallback_reason)
        self.assertEqual(prepared.metadata['fallback'], 'existing_dense')


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
