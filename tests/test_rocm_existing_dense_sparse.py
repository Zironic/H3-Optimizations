'''CPU contracts for the RDNA2 sparse-over-existing-dense fallback.'''

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
from h3_optimizations.attention.sparse import existing_dense_sparse as dense_sparse  # noqa: E402
from h3_optimizations.environment import BACKEND_ROCM  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    SparseRequest,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_TRITON,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def _reference(q, k, v):
    return dense_sparse._reference_attention(q, k, v)


class ExistingDenseSparsePackingTests(unittest.TestCase):
    def test_per_head_routes_and_ragged_tail_match_independent_dense_attention(self):
        torch.manual_seed(7)
        q = torch.randn((1, 2, 4, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 2, 5, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 2, 5, 8), dtype=torch.bfloat16)
        spec = dense_sparse.ExistingDenseSparseSpec(
            q_tile=2,
            kv_tile=2,
            max_batch_entries=4,
        )
        # kv tiles are [0:2], [2:4], [4:5].  Different heads and Q tiles
        # deliberately choose different pairs; two entries include the ragged
        # final tile and two do not.
        selected = torch.tensor(
            [[
                [[0, 2], [0, 1]],
                [[1, 2], [0, 2]],
            ]],
            dtype=torch.int32,
        )

        def dense(q_batch, k_batch, v_batch, _options, *, heads):
            self.assertEqual(heads, 1)
            return _reference(q_batch, k_batch, v_batch)

        with mock.patch.object(
            dense_sparse,
            '_call_existing_dense',
            side_effect=dense,
        ):
            actual = dense_sparse._execute_packed_sparse(
                q,
                k,
                v,
                selected,
                {},
                spec=spec,
            )

        expected = torch.empty_like(q)
        for head in range(2):
            for q_tile in range(2):
                tiles = selected[0, head, q_tile].tolist()
                rows = [
                    row
                    for tile in tiles
                    for row in range(tile * 2, min(tile * 2 + 2, 5))
                ]
                q_rows = slice(q_tile * 2, q_tile * 2 + 2)
                expected[:, head:head + 1, q_rows, :] = _reference(
                    q[:, head:head + 1, q_rows, :],
                    k[:, head:head + 1, rows, :],
                    v[:, head:head + 1, rows, :],
                )

        self.assertTrue(torch.equal(actual, expected))

    def test_partial_final_q_tile_discards_padded_probe_rows(self):
        torch.manual_seed(11)
        q = torch.randn((1, 1, 3, 8), dtype=torch.bfloat16)
        k = torch.randn((1, 1, 4, 8), dtype=torch.bfloat16)
        v = torch.randn((1, 1, 4, 8), dtype=torch.bfloat16)
        spec = dense_sparse.ExistingDenseSparseSpec(2, 2, 2)
        selected = torch.tensor([[[[0], [1]]]], dtype=torch.int32)

        with mock.patch.object(
            dense_sparse,
            '_call_existing_dense',
            side_effect=lambda q, k, v, _o, *, heads: _reference(q, k, v),
        ):
            actual = dense_sparse._execute_packed_sparse(
                q, k, v, selected, {}, spec=spec
            )

        expected = torch.empty_like(q)
        expected[..., :2, :] = _reference(q[..., :2, :], k[..., :2, :], v[..., :2, :])
        expected[..., 2:3, :] = _reference(q[..., 2:3, :], k[..., 2:4, :], v[..., 2:4, :])
        self.assertTrue(torch.equal(actual, expected))


class ExistingDenseSparseProbeTests(unittest.TestCase):
    def setUp(self):
        dense_sparse._probe_results.clear()

    def test_probe_prefers_64_then_falls_back_to_128(self):
        calls = []

        def geometry(_device, _options, q_tile, kv_tile):
            calls.append((q_tile, kv_tile))
            if q_tile == 64:
                raise RuntimeError('64 rejected')
            return 4

        with (
            mock.patch.object(dense_sparse, '_probe_key', return_value=('test',)),
            mock.patch.object(dense_sparse, '_probe_geometry', side_effect=geometry),
        ):
            spec = dense_sparse.probe_existing_dense_sparse(device='cpu')

        self.assertEqual(calls, [(64, 64), (128, 128)])
        self.assertEqual((spec.q_tile, spec.kv_tile), (128, 128))
        self.assertEqual(spec.max_batch_entries, 4)

    def test_probe_keeps_64_when_accepted(self):
        with (
            mock.patch.object(dense_sparse, '_probe_key', return_value=('test',)),
            mock.patch.object(dense_sparse, '_probe_geometry', return_value=8),
        ):
            spec = dense_sparse.probe_existing_dense_sparse(device='cpu')
        self.assertEqual((spec.q_tile, spec.kv_tile), (64, 64))
        self.assertEqual(spec.max_batch_entries, 8)


class AMDPolicyTests(unittest.TestCase):
    def test_rdna2_auto_replaces_final_dense_fallback(self):
        plan = H3OptimizationPlan(sparse=SparseRequest())
        incoming = base_apply.ResolvedAttention(
            requested=base_apply.ATTENTION_SPARSE,
            selected=base_apply.ATTENTION_EXISTING,
            backend=None,
            reason='specialized sparse backends unavailable',
            backend_kind=base_apply.ATTENTION_EXISTING,
        )
        old_qkv = object()
        new_qkv = object()
        projector = object()
        spec = dense_sparse.ExistingDenseSparseSpec(64, 64, 8)
        model = SimpleNamespace(model_options={'transformer_options': {}})
        environment = SimpleNamespace(backend=BACKEND_ROCM, device_index=0)

        with (
            mock.patch.object(
                amd_policy,
                '_POLICY_RESOLVE_ATTENTION',
                return_value=(incoming, old_qkv),
            ),
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                amd_policy,
                'probe_existing_dense_sparse',
                return_value=spec,
            ),
            mock.patch.object(
                amd_policy,
                '_resolve_qkv',
                return_value=(new_qkv, projector),
            ),
            mock.patch.object(
                amd_policy,
                '_fallback_device',
                return_value=torch.device('cpu'),
            ),
        ):
            attention, qkv = amd_policy.resolve_attention(
                plan,
                model,
                SimpleNamespace(),
                environment,
            )

        self.assertIs(qkv, new_qkv)
        self.assertEqual(
            attention.selected,
            amd_policy.SELECTED_EXISTING_DENSE_SPARSE,
        )
        self.assertEqual(attention.backend.name, 'rocm_existing_dense_sparse')
        self.assertEqual(attention.backend_kind, base_apply.ATTENTION_TRITON_SPARSE)
        self.assertIs(attention.projector, projector)
        self.assertIn('64Q x 64KV', attention.reason)

    def test_explicit_sparse_backend_is_not_replaced(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_TRITON)
        )
        incoming = base_apply.ResolvedAttention(
            requested=base_apply.ATTENTION_TRITON_SPARSE,
            selected=base_apply.ATTENTION_EXISTING,
            backend=None,
            reason='explicit failure placeholder',
            backend_kind=base_apply.ATTENTION_EXISTING,
        )
        environment = SimpleNamespace(backend=BACKEND_ROCM, device_index=0)
        with (
            mock.patch.object(
                amd_policy,
                '_POLICY_RESOLVE_ATTENTION',
                return_value=(incoming, 'qkv'),
            ),
            mock.patch.object(amd_policy, '_is_rdna2', return_value=True),
            mock.patch.object(
                amd_policy,
                'probe_existing_dense_sparse',
                side_effect=AssertionError('must not probe'),
            ),
        ):
            attention, qkv = amd_policy.resolve_attention(
                plan,
                SimpleNamespace(model_options={}),
                SimpleNamespace(),
                environment,
            )
        self.assertIs(attention, incoming)
        self.assertEqual(qkv, 'qkv')

    def test_rdna2_triton_preflight_fails_before_compilation(self):
        original = mock.Mock(side_effect=AssertionError('must not compile'))
        with (
            mock.patch.object(
                amd_policy,
                '_active_rocm_architecture',
                return_value='gfx1030',
            ),
            mock.patch.object(
                amd_policy,
                '_ORIGINAL_TRITON_PREFLIGHT',
                original,
            ),
            self.assertRaisesRegex(
                base_apply.TritonSparseError,
                'RDNA2 BF16 Triton is disabled',
            ),
        ):
            amd_policy.preflight_triton_sparse()
        original.assert_not_called()

    def test_non_rdna2_triton_preflight_delegates(self):
        with (
            mock.patch.object(
                amd_policy,
                '_active_rocm_architecture',
                return_value='gfx1100',
            ),
            mock.patch.object(
                amd_policy,
                '_ORIGINAL_TRITON_PREFLIGHT',
                return_value='spec',
            ) as original,
        ):
            result = amd_policy.preflight_triton_sparse(test=True)
        self.assertEqual(result, 'spec')
        original.assert_called_once_with(test=True)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
