'''CPU contracts for experimental ROCm BF16 Triton selection.'''

import os
from pathlib import Path
import sys
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

from h3_optimizations.attention.sparse import triton_sparse  # noqa: E402
from h3_optimizations.attention.sparse import triton_bf16_selftest  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RocmTritonSelectionTests(unittest.TestCase):
    def test_rdna2_selftest_route_is_sparse_varied_and_includes_tail(self):
        rows = triton_bf16_selftest._sparse_rows(2, 3, 3)

        self.assertEqual(rows[0][0], [0, 2])
        self.assertEqual(rows[0][1], [1, 2])
        self.assertEqual(rows[1][0], [1, 2])
        self.assertTrue(all(len(row) == 2 for head in rows for row in head))

    def test_rocm_rdna3_preflight_accepts_triton_without_nvidia_capability(self):
        cuda_probe = mock.Mock(side_effect=AssertionError('must not probe CUDA'))
        capability_probe = mock.Mock(
            side_effect=AssertionError('must not probe NVIDIA capability')
        )

        spec = triton_sparse.preflight_triton_sparse(
            cuda_available=cuda_probe,
            capability_getter=capability_probe,
            triton_available=True,
            rocm_available=lambda: True,
            rocm_arch_getter=lambda: 'gfx1100',
        )

        self.assertEqual(
            spec.signature,
            ('triton_bf16_qk_bf16pv_fp32', 64, 64, 128),
        )
        cuda_probe.assert_not_called()
        capability_probe.assert_not_called()

    def test_rocm_cdna_preflight_accepts_triton(self):
        spec = triton_sparse.preflight_triton_sparse(
            cuda_available=lambda: False,
            capability_getter=lambda: None,
            triton_available=True,
            rocm_available=lambda: True,
            rocm_arch_getter=lambda: 'gfx942',
        )
        self.assertEqual(spec.q_tile, 64)
        self.assertEqual(spec.kv_tile, 64)

    def test_rocm_rdna2_accepts_after_numerical_selftest(self):
        selftest = mock.Mock(return_value=True)
        spec = triton_sparse.preflight_triton_sparse(
            cuda_available=lambda: False,
            capability_getter=lambda: None,
            triton_available=True,
            rocm_available=lambda: True,
            rocm_arch_getter=lambda: 'gfx1030',
            rocm_selftest=selftest,
        )

        self.assertEqual(spec.q_tile, 64)
        self.assertEqual(spec.kv_tile, 64)
        selftest.assert_called_once_with()

    def test_rocm_rdna2_selftest_failure_falls_through(self):
        with self.assertRaisesRegex(
            triton_sparse.TritonSparseError,
            'numerical self-test failed',
        ):
            triton_sparse.preflight_triton_sparse(
                cuda_available=lambda: False,
                capability_getter=lambda: None,
                triton_available=True,
                rocm_available=lambda: True,
                rocm_arch_getter=lambda: 'gfx1030',
                rocm_selftest=lambda: False,
            )

    def test_rocm_rdna3_does_not_run_rdna2_selftest(self):
        selftest = mock.Mock(side_effect=AssertionError('must not self-test'))
        triton_sparse.preflight_triton_sparse(
            cuda_available=lambda: False,
            capability_getter=lambda: None,
            triton_available=True,
            rocm_available=lambda: True,
            rocm_arch_getter=lambda: 'gfx1100',
            rocm_selftest=selftest,
        )
        selftest.assert_not_called()

    def test_rocm_unknown_architecture_fails_closed(self):
        with self.assertRaisesRegex(
            triton_sparse.TritonSparseError,
            'got gfx9999',
        ):
            triton_sparse.preflight_triton_sparse(
                cuda_available=lambda: False,
                capability_getter=lambda: None,
                triton_available=True,
                rocm_available=lambda: True,
                rocm_arch_getter=lambda: 'gfx9999',
            )

    def test_rocm_preflight_still_requires_triton(self):
        with self.assertRaisesRegex(
            triton_sparse.TritonSparseError,
            'requires Triton',
        ):
            triton_sparse.preflight_triton_sparse(
                cuda_available=lambda: False,
                capability_getter=lambda: None,
                triton_available=False,
                rocm_available=lambda: True,
                rocm_arch_getter=lambda: 'gfx1100',
            )

    def test_non_rocm_keeps_nvidia_preflight_contract(self):
        with mock.patch.object(
            triton_sparse,
            'preflight_triton_bf16',
            return_value='nvidia-spec',
        ) as preflight:
            result = triton_sparse.preflight_triton_sparse(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 9),
                triton_available=True,
                rocm_available=lambda: False,
                rocm_arch_getter=lambda: 'must-not-be-used',
            )

        self.assertEqual(result, 'nvidia-spec')
        preflight.assert_called_once()
        self.assertNotIn('rocm_available', preflight.call_args.kwargs)
        self.assertNotIn('rocm_arch_getter', preflight.call_args.kwargs)
        self.assertNotIn('rocm_selftest', preflight.call_args.kwargs)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
