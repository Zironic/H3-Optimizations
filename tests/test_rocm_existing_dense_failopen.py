'''Fail-open contracts for the RDNA2 sparse-over-existing-dense adapter.'''

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

from h3_optimizations.attention.sparse import existing_dense_sparse  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RDNA2ExistingDenseFailOpenTests(unittest.TestCase):
    def setUp(self):
        existing_dense_sparse._runtime_fallback_warned = False

    def test_packed_failure_recomputes_entire_attention_invocation_dense(self):
        q = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 2, 6, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 2, 6, 8), dtype=torch.bfloat16)
        expected = torch.randn_like(q)
        spec = existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 4)
        backend = existing_dense_sparse.ExistingDenseSparseBackend(spec=spec)
        prepared = existing_dense_sparse.PreparedExistingDenseSparse(
            q=q,
            k=k,
            v=v,
            lut=torch.tensor(
                [[[[0], [0]], [[0], [0]]]],
                dtype=torch.int32,
            ),
            valid=torch.ones((1, 2, 2), dtype=torch.int32),
            metadata={
                'dense_q_tiles': 0,
                'sparse_q_tiles': 2,
                'kv_tiles': 3,
                'pure_video_kv_tiles': 3,
                'retained_video_kv_tiles': 1,
            },
            transformer_options={'test': True},
        )

        with (
            mock.patch.object(
                existing_dense_sparse,
                '_execute_packed_sparse',
                side_effect=RuntimeError('packed shape rejected'),
            ),
            mock.patch.object(
                existing_dense_sparse,
                '_call_existing_dense',
                return_value=expected,
            ) as dense,
        ):
            actual = backend.execute(prepared)

        self.assertIs(actual, expected)
        dense.assert_called_once()
        args, kwargs = dense.call_args
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], v)
        self.assertEqual(kwargs['heads'], 2)

    def test_prepare_failure_is_deferred_to_full_dense_execution(self):
        q = torch.randn((1, 1, 4, 128), dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        expected = torch.randn_like(q)
        backend = existing_dense_sparse.ExistingDenseSparseBackend(
            spec=existing_dense_sparse.ExistingDenseSparseSpec(2, 2, 2)
        )

        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=3,
            transformer_options={'test': True},
        )
        self.assertIsNotNone(prepared.fallback_reason)
        self.assertIsNone(prepared.lut)

        with mock.patch.object(
            existing_dense_sparse,
            '_call_existing_dense',
            return_value=expected,
        ) as dense:
            actual = backend.execute(prepared)

        self.assertIs(actual, expected)
        dense.assert_called_once()
        args, kwargs = dense.call_args
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], v)
        self.assertEqual(kwargs['heads'], 1)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
