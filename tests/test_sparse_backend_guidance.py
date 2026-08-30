'''CPU contracts for actionable explicit sparse-backend failures.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.apply_policy as policy  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_SAGE,
    SparseRequest,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class SparseBackendGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.inventory = object()

    @staticmethod
    def _plan(backend):
        return H3OptimizationPlan(sparse=SparseRequest(backend=backend))

    def test_sm75_sparse_sage_failure_recommends_detected_kitchen(self):
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(7, 5),
            device_index=0,
            backend='nvidia_cuda',
            architecture='sm75',
        )
        original = policy._base.SparseSageError(
            'Sparse Sage does not support device capability 7.5'
        )

        with mock.patch.object(
            policy,
            '_BASE_RESOLVE_ATTENTION',
            side_effect=original,
        ), mock.patch.object(
            policy._base,
            'preflight_sparse_kitchen',
            return_value=object(),
        ), mock.patch.object(
            policy._base,
            'preflight_frost_bf16',
            side_effect=policy.FrostBF16Error('SM89 only'),
        ), mock.patch.object(
            policy._base,
            'preflight_triton_sparse',
            side_effect=policy._base.TritonSparseError('SM80 or newer'),
        ), mock.patch.object(
            policy._base,
            'preflight_fp8_flex',
            side_effect=policy._base.FP8FlexError('FP8 unavailable'),
        ):
            with self.assertRaises(policy._base.SparseSageError) as raised:
                policy.resolve_attention(
                    self._plan(SPARSE_BACKEND_SAGE),
                    self.model,
                    self.inventory,
                    environment,
                )

        text = str(raised.exception)
        self.assertIn('Sparse Sage is unavailable on sm75', text)
        self.assertIn('device capability 7.5', text)
        self.assertIn(
            'Available sparse backends detected on this system: Kitchen INT8',
            text,
        )
        self.assertNotIn('BF16 Triton,', text)
        self.assertIn('H3 Sparse Attention', text)
        self.assertIn('select a compatible backend automatically', text)
        self.assertIn('fall back to dense attention', text)

    def test_rocm_sparse_sage_failure_recommends_detected_triton_and_flex(self):
        environment = SimpleNamespace(
            cuda_available=False,
            capability=None,
            device_index=0,
            backend='rocm',
            architecture='rocm',
        )
        original = policy._base.SparseSageError(
            'Hybrid Sparse Attention requires CUDA'
        )

        with mock.patch.object(
            policy,
            '_BASE_RESOLVE_ATTENTION',
            side_effect=original,
        ), mock.patch.object(
            policy._base,
            'preflight_sparse_kitchen',
            side_effect=policy._base.SparseKitchenError('requires CUDA'),
        ), mock.patch.object(
            policy._base,
            'preflight_frost_bf16',
            side_effect=policy.FrostBF16Error('requires NVIDIA CUDA'),
        ), mock.patch.object(
            policy._base,
            'preflight_triton_sparse',
            return_value=object(),
        ), mock.patch.object(
            policy._base,
            'preflight_fp8_flex',
            return_value=object(),
        ):
            with self.assertRaises(policy._base.SparseSageError) as raised:
                policy.resolve_attention(
                    self._plan(SPARSE_BACKEND_SAGE),
                    self.model,
                    self.inventory,
                    environment,
                )

        text = str(raised.exception)
        self.assertIn('Sparse Sage is unavailable on rocm', text)
        self.assertIn('Hybrid Sparse Attention requires CUDA', text)
        self.assertIn(
            'Available sparse backends detected on this system: '
            'BF16 Triton, FP8 FlexAttention',
            text,
        )
        self.assertIn('select a compatible backend automatically', text)

    def test_no_alternative_keeps_original_reason_and_suggests_auto(self):
        environment = SimpleNamespace(
            cuda_available=False,
            capability=None,
            device_index=None,
            backend='cpu',
            architecture='cpu',
        )
        original = policy._base.SparseSageError('requires a GPU')

        with mock.patch.object(
            policy,
            '_BASE_RESOLVE_ATTENTION',
            side_effect=original,
        ), mock.patch.object(
            policy,
            '_probe_sparse_backend',
            side_effect=RuntimeError('unavailable'),
        ):
            with self.assertRaises(policy._base.SparseSageError) as raised:
                policy.resolve_attention(
                    self._plan(SPARSE_BACKEND_SAGE),
                    self.model,
                    self.inventory,
                    environment,
                )

        text = str(raised.exception)
        self.assertIn('requires a GPU', text)
        self.assertIn(
            'No compatible alternative sparse backend was detected',
            text,
        )
        self.assertIn('H3 Sparse Attention', text)

    def test_architecture_errors_name_the_backend_system_and_recovery(self):
        environment = SimpleNamespace(
            backend='nvidia_cuda',
            architecture='sm75',
        )
        cases = (
            (
                policy.SPARSE_BACKEND_FROST,
                policy.FrostBF16Error(
                    'FROST BF16 is compiled for SM89; found SM75'
                ),
                'FROST BF16 (SM89) is unavailable on sm75',
            ),
            (
                policy.SPARSE_BACKEND_TRITON,
                policy._base.TritonSparseError(
                    'BF16 Triton requires NVIDIA compute capability 8.0 or newer'
                ),
                'BF16 Triton is unavailable on sm75',
            ),
            (
                policy.SPARSE_BACKEND_FLEX,
                policy._base.FP8FlexError(
                    'FP8 compute is unsupported on device capability 7.5'
                ),
                'FP8 FlexAttention is unavailable on sm75',
            ),
        )
        with mock.patch.object(
            policy,
            '_available_sparse_alternatives',
            return_value=['Kitchen INT8'],
        ):
            for backend, error, heading in cases:
                with self.subTest(backend=backend):
                    text = policy._explicit_backend_error(
                        backend,
                        error,
                        environment,
                    )
                    self.assertIn(heading, text)
                    self.assertIn(str(error), text)
                    self.assertIn('Kitchen INT8', text)
                    self.assertIn('H3 Sparse Attention (Advanced)', text)
                    self.assertIn('fall back to dense attention', text)

    def test_auto_errors_are_not_rewritten_or_probed(self):
        environment = SimpleNamespace()
        original = policy._base.SparseSageError('synthetic auto failure')

        with mock.patch.object(
            policy,
            '_BASE_RESOLVE_ATTENTION',
            side_effect=original,
        ), mock.patch.object(
            policy,
            '_available_sparse_alternatives',
        ) as alternatives:
            with self.assertRaisesRegex(
                policy._base.SparseSageError,
                '^synthetic auto failure$',
            ):
                policy.resolve_attention(
                    self._plan(SPARSE_BACKEND_AUTO),
                    self.model,
                    self.inventory,
                    environment,
                )

        alternatives.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
