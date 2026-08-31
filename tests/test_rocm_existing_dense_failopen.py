'''Fail-open contract for the RDNA2 sparse-over-dense adapter.'''

import os
from pathlib import Path
import sys
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

from h3_optimizations import amd_policy  # noqa: E402
from h3_optimizations.attention.sparse import existing_dense_sparse  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RDNA2PackedFailOpenTests(unittest.TestCase):
    def test_packed_failure_uses_full_existing_dense_for_the_q_slab(self):
        q = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 2, 6, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 2, 6, 8), dtype=torch.bfloat16)
        selected = torch.zeros((1, 2, 2, 1), dtype=torch.int32)
        expected = torch.randn_like(q)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 4)

        with (
            mock.patch.object(
                amd_policy,
                '_ORIGINAL_PACKED_EXECUTE',
                side_effect=RuntimeError('packed shape rejected'),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
                return_value=expected,
            ) as dense,
        ):
            actual = amd_policy._safe_execute_packed_sparse(
                q,
                k,
                v,
                selected,
                {'test': True},
                spec=spec,
            )

        self.assertIs(actual, expected)
        dense.assert_called_once()
        args, kwargs = dense.call_args
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], v)
        self.assertEqual(kwargs['heads'], 2)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
