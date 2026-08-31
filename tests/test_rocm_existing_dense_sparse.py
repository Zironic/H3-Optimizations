'''Policy contracts for RDNA2 using the universal sparse-over-dense fallback.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

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
from h3_optimizations import universal_sparse_fallback as universal  # noqa: E402
from h3_optimizations.environment import BACKEND_ROCM  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    SparseRequest,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_STANDARD,
    QKVProviderResolution,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class RDNA2PolicyTests(unittest.TestCase):
    def setUp(self):
        self.environment = SimpleNamespace(
            backend=BACKEND_ROCM,
            device_index=0,
            cuda_available=False,
        )
        self.model = SimpleNamespace(model_options={'transformer_options': {}})
        self.dense_resolution = SimpleNamespace(
            requested='auto',
            selected='existing',
            backend=None,
            reason='existing',
            backend_kind='existing',
        )
        self.dense = base_apply.ResolvedAttention(
            requested=base_apply.ATTENTION_SPARSE,
            selected=base_apply.ATTENTION_EXISTING,
            backend=None,
            reason='existing dense attention',
            backend_kind=base_apply.ATTENTION_EXISTING,
            dense_resolution=self.dense_resolution,
        )
        self.qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'stock projection',
        )

    def test_auto_skips_known_dead_backends_before_final_fallback(self):
        with (
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                base_apply,
                '_resolve_dense',
                return_value=(self.dense, self.qkv),
            ),
            mock.patch.object(
                base_apply,
                '_resolve_fp8_flex',
                side_effect=base_apply.FP8FlexError('flex unavailable'),
            ),
            mock.patch.object(
                amd_policy,
                '_POLICY_RESOLVE_ATTENTION',
                side_effect=AssertionError('ordinary sparse chain must not run'),
            ),
        ):
            attention, qkv = amd_policy.resolve_attention(
                H3OptimizationPlan(sparse=SparseRequest()),
                self.model,
                SimpleNamespace(),
                self.environment,
            )

        self.assertIs(qkv, self.qkv)
        self.assertEqual(attention.selected, base_apply.ATTENTION_EXISTING)
        self.assertIsNone(attention.backend)
        self.assertIn('skips Kitchen INT8, Sparse Sage, and BF16 Triton', attention.reason)
        self.assertIn('FP8 FlexAttention unavailable', attention.reason)

    def test_universal_layer_promotes_rdna2_dense_result(self):
        resolved = (self.dense, self.qkv)
        with mock.patch.object(
            universal,
            '_PREVIOUS_RESOLVE_ATTENTION',
            return_value=resolved,
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
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertFalse(qkv.fused)
        self.assertIn('stock Comfy QKV', qkv.reason)

    def test_auto_keeps_working_flex(self):
        expected = (object(), object())
        with (
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                base_apply,
                '_resolve_dense',
                return_value=(self.dense, self.qkv),
            ),
            mock.patch.object(base_apply, '_resolve_fp8_flex', return_value=expected),
        ):
            actual = amd_policy.resolve_attention(
                H3OptimizationPlan(sparse=SparseRequest()),
                self.model,
                SimpleNamespace(),
                self.environment,
            )
        self.assertIs(actual, expected)

    def test_explicit_flex_still_delegates(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_FLEX)
        )
        expected = (self.dense, self.qkv)
        with (
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                amd_policy,
                '_POLICY_RESOLVE_ATTENTION',
                return_value=expected,
            ) as resolver,
        ):
            actual = amd_policy.resolve_attention(
                plan,
                self.model,
                SimpleNamespace(),
                self.environment,
            )
        resolver.assert_called_once()
        self.assertIs(actual, expected)

    def test_explicit_known_dead_backends_fail_without_fallback_probes(self):
        cases = (
            (SPARSE_BACKEND_KITCHEN, base_apply.SparseKitchenError, 'Kitchen INT8'),
            (SPARSE_BACKEND_SAGE, base_apply.SparseSageError, 'Sparse Sage'),
            (SPARSE_BACKEND_TRITON, base_apply.TritonSparseError, 'BF16 Triton'),
        )
        for backend, error_type, message in cases:
            with self.subTest(backend=backend):
                plan = H3OptimizationPlan(sparse=SparseRequest(backend=backend))
                with (
                    mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
                    mock.patch.object(
                        amd_policy,
                        '_active_rocm_architecture',
                        return_value='gfx1030',
                    ),
                    mock.patch.object(
                        amd_policy,
                        '_POLICY_RESOLVE_ATTENTION',
                        side_effect=AssertionError('must not probe alternatives'),
                    ) as resolver,
                    self.assertRaisesRegex(error_type, message),
                ):
                    amd_policy.resolve_attention(
                        plan,
                        self.model,
                        SimpleNamespace(),
                        self.environment,
                    )
                resolver.assert_not_called()

    def test_triton_resolver_uses_selected_device_architecture(self):
        with (
            mock.patch.object(
                amd_policy,
                '_active_rocm_architecture',
                return_value='gfx1030',
            ) as architecture,
            mock.patch.object(
                amd_policy,
                '_ORIGINAL_RESOLVE_TRITON_SPARSE',
                side_effect=AssertionError('must not resolve Triton'),
            ),
            self.assertRaises(base_apply.TritonSparseError),
        ):
            amd_policy._resolve_triton_sparse(
                object(),
                self.environment,
                object(),
                None,
            )
        architecture.assert_called_once_with(self.environment)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
