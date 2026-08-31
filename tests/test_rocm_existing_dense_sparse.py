'''Policy contracts for the RDNA2 sparse-over-existing-dense fallback.'''

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
from h3_optimizations import amd_policy  # noqa: E402
from h3_optimizations.attention.sparse import existing_dense_sparse  # noqa: E402
from h3_optimizations.environment import BACKEND_ROCM  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    MemoryRequest,
    SparseRequest,
    QKV_STREAMING_FORCED,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_TRITON,
)
from h3_optimizations.qkv.providers import QKV_STANDARD  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RDNA2PolicyTests(unittest.TestCase):
    def setUp(self):
        self.environment = SimpleNamespace(backend=BACKEND_ROCM, device_index=0)
        self.model = SimpleNamespace(model_options={'transformer_options': {}})
        self.dense = base_apply.ResolvedAttention(
            requested=base_apply.ATTENTION_SPARSE,
            selected=base_apply.ATTENTION_EXISTING,
            backend=None,
            reason='existing dense attention',
            backend_kind=base_apply.ATTENTION_EXISTING,
        )

    def test_auto_skips_known_dead_backends_and_defers_probe_to_runtime(self):
        with (
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                base_apply, '_resolve_dense', return_value=(self.dense, object())
            ),
            mock.patch.object(
                base_apply,
                '_resolve_fp8_flex',
                side_effect=base_apply.FP8FlexError('flex unavailable'),
            ),
            mock.patch.object(
                amd_policy,
                'probe_rdna2_existing_dense_sparse',
                side_effect=AssertionError('runtime probe must not run at resolution'),
            ),
            mock.patch.object(
                amd_policy,
                '_POLICY_RESOLVE_ATTENTION',
                side_effect=AssertionError('ordinary sparse chain must not run'),
            ),
        ):
            attention, resolved_qkv = amd_policy.resolve_attention(
                H3OptimizationPlan(sparse=SparseRequest()),
                self.model,
                SimpleNamespace(),
                self.environment,
            )

        self.assertEqual(resolved_qkv.provider_id, QKV_STANDARD)
        self.assertFalse(resolved_qkv.fused)
        self.assertIsNone(attention.projector)
        self.assertIsInstance(
            attention.backend,
            amd_policy.RDNA2ExistingDenseSparseBackend,
        )
        self.assertEqual(
            attention.selected,
            amd_policy.SELECTED_EXISTING_DENSE_SPARSE,
        )
        self.assertIn('actual QKV dtype', attention.reason)

    def test_rdna2_adapter_preserves_stock_qkv(self):
        qkv, projector = amd_policy._resolve_qkv(
            H3OptimizationPlan(sparse=SparseRequest())
        )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertFalse(qkv.fused)
        self.assertIsNone(projector)
        self.assertIn('stock Comfy QKV projection', qkv.reason)

    def test_forced_qkv_streaming_disables_adapter(self):
        plan = H3OptimizationPlan(
            memory=MemoryRequest(qkv_streaming=QKV_STREAMING_FORCED),
            sparse=SparseRequest(),
        )
        with self.assertRaisesRegex(
            existing_dense_sparse.ExistingDenseSparseError,
            'does not override explicitly forced QKV execution',
        ):
            amd_policy._resolve_qkv(plan)

    def test_auto_keeps_working_flex(self):
        expected = (object(), object())
        with (
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                base_apply, '_resolve_dense', return_value=(self.dense, object())
            ),
            mock.patch.object(base_apply, '_resolve_fp8_flex', return_value=expected),
            mock.patch.object(
                amd_policy,
                'probe_rdna2_existing_dense_sparse',
                side_effect=AssertionError('adapter must not probe'),
            ),
        ):
            actual = amd_policy.resolve_attention(
                H3OptimizationPlan(sparse=SparseRequest()),
                self.model,
                SimpleNamespace(),
                self.environment,
            )
        self.assertEqual(actual, expected)

    def test_explicit_flex_still_delegates(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_FLEX)
        )
        with (
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                amd_policy,
                '_POLICY_RESOLVE_ATTENTION',
                return_value=(self.dense, 'qkv'),
            ) as resolver,
        ):
            actual = amd_policy.resolve_attention(
                plan, self.model, SimpleNamespace(), self.environment
            )
        resolver.assert_called_once()
        self.assertEqual(actual, (self.dense, 'qkv'))

    def test_explicit_triton_fails_without_probe_chain(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_TRITON)
        )
        with (
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                amd_policy, '_active_rocm_architecture', return_value='gfx1030'
            ),
            mock.patch.object(
                amd_policy,
                '_POLICY_RESOLVE_ATTENTION',
                side_effect=AssertionError('must not probe alternatives'),
            ) as resolver,
            self.assertRaisesRegex(
                base_apply.TritonSparseError,
                'unavailable on RDNA2 gfx1030',
            ),
        ):
            amd_policy.resolve_attention(
                plan, self.model, SimpleNamespace(), self.environment
            )
        resolver.assert_not_called()

    def test_triton_resolver_uses_selected_device_architecture(self):
        with (
            mock.patch.object(
                amd_policy, '_active_rocm_architecture', return_value='gfx1030'
            ) as architecture,
            mock.patch.object(
                amd_policy,
                '_ORIGINAL_RESOLVE_TRITON_SPARSE',
                side_effect=AssertionError('must not resolve Triton'),
            ),
            self.assertRaises(base_apply.TritonSparseError),
        ):
            amd_policy._resolve_triton_sparse(
                object(), self.environment, object(), None
            )
        architecture.assert_called_once_with(self.environment)


class RDNA2RuntimeProbeTests(unittest.TestCase):
    def setUp(self):
        amd_policy._rdna2_probe_results.clear()

    def test_probe_cache_is_dtype_specific(self):
        calls = []

        def geometry(_device, _options, q_tile, kv_tile, dtype):
            calls.append((q_tile, kv_tile, dtype))
            return 4

        with (
            mock.patch.object(
                existing_dense_sparse,
                '_probe_key',
                return_value=('device', 'consumer'),
            ),
            mock.patch.object(
                amd_policy,
                '_probe_geometry_dtype',
                side_effect=geometry,
            ),
        ):
            bf16 = amd_policy.probe_rdna2_existing_dense_sparse(
                device='cpu', dtype=torch.bfloat16
            )
            fp32 = amd_policy.probe_rdna2_existing_dense_sparse(
                device='cpu', dtype=torch.float32
            )
            bf16_cached = amd_policy.probe_rdna2_existing_dense_sparse(
                device='cpu', dtype=torch.bfloat16
            )

        self.assertEqual((bf16.q_tile, bf16.kv_tile), (64, 64))
        self.assertEqual((fp32.q_tile, fp32.kv_tile), (64, 64))
        self.assertIs(bf16_cached, bf16)
        self.assertEqual(
            calls,
            [
                (64, 64, torch.bfloat16),
                (64, 64, torch.float32),
            ],
        )

    def test_validator_accepts_h3_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):
            q = SimpleNamespace(
                shape=(1, 2, 8, 128),
                ndim=4,
                dtype=dtype,
                device='cuda:0',
                is_cuda=True,
                stride=lambda dim: 1,
            )
            amd_policy._validate_rdna2_qkv(q, q, q)

    def test_runtime_probe_failure_prepares_dense_fallback(self):
        backend = amd_policy.RDNA2ExistingDenseSparseBackend()
        q = torch.empty((1, 2, 8, 128), dtype=torch.float32)
        with (
            mock.patch.object(
                amd_policy,
                '_validate_rdna2_qkv',
                return_value=None,
            ),
            mock.patch.object(
                amd_policy,
                'probe_rdna2_existing_dense_sparse',
                side_effect=RuntimeError('runtime rejected dtype'),
            ),
        ):
            prepared = backend.prepare(
                q,
                q,
                q,
                layer_index=3,
                transformer_options={},
            )
        self.assertIsNotNone(prepared.fallback_reason)
        self.assertIn('runtime rejected dtype', prepared.fallback_reason)
        self.assertIs(prepared.q, q)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
