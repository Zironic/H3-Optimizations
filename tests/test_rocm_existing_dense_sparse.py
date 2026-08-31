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
    SparseRequest,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_TRITON,
)

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

    def test_auto_skips_known_dead_backends_then_uses_dense_adapter(self):
        spec = existing_dense_sparse.ExistingDenseSparseSpec(64, 64, 8)
        qkv = object()
        projector = object()
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
                amd_policy, 'probe_existing_dense_sparse', return_value=spec
            ),
            mock.patch.object(
                amd_policy, '_resolve_qkv', return_value=(qkv, projector)
            ),
            mock.patch.object(
                amd_policy, '_fallback_device', return_value=torch.device('cpu')
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

        self.assertIs(resolved_qkv, qkv)
        self.assertIs(attention.projector, projector)
        self.assertEqual(
            attention.selected,
            amd_policy.SELECTED_EXISTING_DENSE_SPARSE,
        )
        self.assertIn('skips Kitchen INT8, Sparse Sage, and BF16 Triton', attention.reason)

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
                'probe_existing_dense_sparse',
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


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
