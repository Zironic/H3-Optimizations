'''Pure tests for the order-independent optimization plan.'''

import math
from pathlib import Path
import sys
import unittest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
    EMBEDDING_MEMORY_AUTO,
    EMBEDDING_MEMORY_RELEASE,
    EARLY_SCHEDULE_HOLD,
    EARLY_SCHEDULE_RAMP,
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    FUSED_QKV_REQUIRED,
    H3OptimizationPlan,
    KITCHEN_V_MEMORY_RETAIN,
    KITCHEN_V_MEMORY_TWO_PASS,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    MLP_MEMORY_PRESERVE,
    MemoryRequest,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_TRITON,
    SparseRequest,
)


class PlanTests(unittest.TestCase):
    def test_memory_request_defaults_to_four_thousand_rows(self):
        request = MemoryRequest()
        self.assertEqual(request.chunk_rows, 4096)
        self.assertTrue(request.prefer_held_weights)
        self.assertFalse(request.mlp_strict)
        self.assertEqual(request.embedding_memory, EMBEDDING_MEMORY_AUTO)
        self.assertEqual(request.attention_v_memory, KITCHEN_V_MEMORY_RETAIN)

    def test_embedding_memory_is_part_of_memory_identity(self):
        request = MemoryRequest(embedding_memory=EMBEDDING_MEMORY_RELEASE)
        self.assertIn(EMBEDDING_MEMORY_RELEASE, request.signature)
        with self.assertRaisesRegex(ValueError, 'embedding memory'):
            MemoryRequest(embedding_memory='unknown')

    def test_attention_v_memory_is_explicit_and_part_of_identity(self):
        request = MemoryRequest(attention_v_memory=KITCHEN_V_MEMORY_TWO_PASS)
        self.assertIn(KITCHEN_V_MEMORY_TWO_PASS, request.signature)
        with self.assertRaisesRegex(ValueError, 'attention V memory'):
            MemoryRequest(attention_v_memory='unknown')

    def test_preserve_precision_is_a_valid_memory_request(self):
        request = MemoryRequest(mlp_memory=MLP_MEMORY_PRESERVE)
        self.assertEqual(request.mlp_memory, MLP_MEMORY_PRESERVE)
        self.assertIn(MLP_MEMORY_PRESERVE, request.signature)

    def test_existing_attention_plus_qkv_off_normalizes_to_preserved_bf16(self):
        request = MemoryRequest(
            attention=ATTENTION_EXISTING,
            fused_qkv=FUSED_QKV_OFF,
        )
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertIn(FUSED_QKV_PRESERVE_BF16, request.signature)
        self.assertEqual(MemoryRequest(fused_qkv=FUSED_QKV_OFF).fused_qkv, FUSED_QKV_OFF)

    def test_legacy_adapter_options_are_part_of_memory_identity(self):
        request = MemoryRequest(
            attention=ATTENTION_EXISTING,
            fused_qkv=FUSED_QKV_REQUIRED,
            mlp_memory=MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
            chunk_rows=2048,
            prefer_held_weights=False,
            mlp_strict=True,
        )
        self.assertEqual(request.signature[0], ATTENTION_EXISTING)
        self.assertEqual(request.signature[1], FUSED_QKV_REQUIRED)
        self.assertFalse(request.prefer_held_weights)
        self.assertTrue(request.mlp_strict)

    def test_sparse_request_defaults_to_fifteen_percent_video_budget(self):
        request = SparseRequest()
        self.assertEqual(request.video_budget, 0.15)
        self.assertEqual(request.backend, SPARSE_BACKEND_AUTO)
        self.assertEqual(request.early_schedule, EARLY_SCHEDULE_HOLD)
        self.assertFalse(request.advanced_schedule)

    def test_legacy_sparse_request_positional_shape_is_preserved(self):
        request = SparseRequest(0.3, False, 2, 0.5, 2, 0.5)
        self.assertEqual(request.backend, SPARSE_BACKEND_AUTO)
        self.assertEqual(
            (
                request.early_steps,
                request.early_kv,
                request.late_steps,
                request.late_kv,
            ),
            (2, 0.5, 2, 0.5),
        )

    def test_legacy_kitchen_label_is_normalized(self):
        request = SparseRequest(backend='Kitchen INT8 (experimental)')
        self.assertEqual(request.backend, SPARSE_BACKEND_KITCHEN)
        self.assertIn(SPARSE_BACKEND_KITCHEN, request.signature)

    def test_explicit_sparse_schedule_is_part_of_request_identity(self):
        request = SparseRequest(
            video_budget=0.3,
            backend=SPARSE_BACKEND_TRITON,
            early_steps=2,
            early_kv=0.5,
            late_steps=2,
            late_kv=0.5,
            early_schedule=EARLY_SCHEDULE_RAMP,
        )
        self.assertTrue(request.advanced_schedule)
        self.assertIn(SPARSE_BACKEND_TRITON, request.signature)
        self.assertIn(EARLY_SCHEDULE_RAMP, request.signature)
        self.assertEqual(request.signature[-5:-1], (2, 0.5, 2, 0.5))

    def test_node_order_does_not_change_the_plan(self):
        memory = MemoryRequest()
        sparse = SparseRequest(video_budget=0.5)
        first = (
            H3OptimizationPlan()
            .with_memory(memory)
            .with_sparse(sparse)
        )
        second = (
            H3OptimizationPlan()
            .with_sparse(sparse)
            .with_memory(memory)
        )
        self.assertEqual(first, second)
        self.assertEqual(first.signature, second.signature)

    def test_identical_requests_are_idempotent(self):
        memory = MemoryRequest()
        sparse = SparseRequest()
        plan = (
            H3OptimizationPlan()
            .with_memory(memory)
            .with_sparse(sparse)
        )
        self.assertEqual(plan.with_memory(memory), plan)
        self.assertEqual(plan.with_sparse(sparse), plan)

    def test_conflicting_duplicate_requests_fail(self):
        plan = H3OptimizationPlan().with_memory(MemoryRequest())
        with self.assertRaisesRegex(ValueError, 'different H3 Memory'):
            plan.with_memory(MemoryRequest(fused_qkv='off'))
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        with self.assertRaisesRegex(ValueError, 'different H3 Sparse'):
            plan.with_sparse(SparseRequest(video_budget=0.4))
    def test_validation_boundaries(self):
        for chunk_rows in (1, 255, 257, 65_536, 65_792):
            self.assertEqual(
                MemoryRequest(chunk_rows=chunk_rows).chunk_rows,
                chunk_rows,
            )
        for chunk_rows in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                MemoryRequest(chunk_rows=chunk_rows)
        for budget in (-1.0, 0.0, 1.01, 2.0):
            self.assertEqual(
                SparseRequest(video_budget=budget).video_budget,
                budget,
            )
        for budget in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                SparseRequest(video_budget=budget)
        with self.assertRaisesRegex(ValueError, 'unknown sparse backend'):
            SparseRequest(backend='dense')
        with self.assertRaisesRegex(ValueError, 'unknown early schedule'):
            SparseRequest(early_schedule='Curve')
        SparseRequest(
            early_steps=0,
            early_kv=0.0,
            late_steps=1001,
            late_kv=1.01,
        )
        with self.assertRaises(ValueError):
            SparseRequest(early_steps=2)
        with self.assertRaises(ValueError):
            SparseRequest(
                early_steps=-1,
                early_kv=0.5,
                late_steps=2,
                late_kv=0.5,
            )
        SparseRequest(
            early_steps=2,
            early_kv=-0.5,
            late_steps=2000,
            late_kv=1.5,
        )
        with self.assertRaises(ValueError):
            SparseRequest(
                denser_early_late_steps=True,
                early_steps=2,
                early_kv=0.5,
                late_steps=2,
                late_kv=0.5,
            )


if __name__ == '__main__':
    unittest.main()
