'''CPU-only tests for the production ComfyUI node registry.'''

import asyncio
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.aimdo_limiter import H3AIMDOResidencyLimiter  # noqa: E402
from h3_optimizations.memory_migration_node import (  # noqa: E402
    H3MemoryOptimization,
    KITCHEN_V_MEMORY_MODE_RETAIN,
    KITCHEN_V_MEMORY_MODE_TWO_PASS,
    PRECISION_MODE_ALLOW_FP8,
    PRECISION_MODE_AUTO,
    PRECISION_MODE_BF16,
    PRECISION_MODE_FORCE_QUANT,
    PRECISION_MODE_OPTIONS,
    PRECISION_MODE_PRESERVE,
    PRECISION_MODE_PRESERVE_NATIVE,
    QKV_STREAMING_MODE_AUTO,
    QKV_STREAMING_MODE_FORCED,
    QKV_STREAMING_MODE_OFF,
    _memory_request_for_modes,
    _normalize_precision_mode,
    _qkv_streaming_request,
)
from h3_optimizations.nodes import (  # noqa: E402
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)
from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_PRESERVE_BF16,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_BF16,
    MLP_MEMORY_FORCE_QUANT,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
    QKV_STREAMING_AUTO,
    QKV_STREAMING_FORCED,
    QKV_STREAMING_OFF,
    KITCHEN_V_MEMORY_RETAIN,
    KITCHEN_V_MEMORY_TWO_PASS,
)
from h3_optimizations.public_nodes import H3OptimizationsExtension  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class PublicNodeTests(unittest.TestCase):
    def test_public_registry_contains_only_production_nodes(self):
        nodes = asyncio.run(H3OptimizationsExtension().get_node_list())
        self.assertEqual(
            nodes,
            [
                H3MemoryOptimization,
                H3AIMDOResidencyLimiter,
                H3SparseAttention,
                H3SparseAttentionAdvanced,
            ],
        )
        self.assertFalse(any('MLPSharing' in node.__name__ for node in nodes))

    def test_legacy_precision_modes_map_to_new_policies(self):
        self.assertEqual(
            _normalize_precision_mode(PRECISION_MODE_PRESERVE),
            PRECISION_MODE_PRESERVE_NATIVE,
        )
        self.assertEqual(
            _normalize_precision_mode(PRECISION_MODE_ALLOW_FP8),
            PRECISION_MODE_AUTO,
        )

    def test_qkv_streaming_mode_maps_to_plan_values(self):
        self.assertEqual(_qkv_streaming_request(QKV_STREAMING_MODE_OFF), QKV_STREAMING_OFF)
        self.assertEqual(_qkv_streaming_request(QKV_STREAMING_MODE_AUTO), QKV_STREAMING_AUTO)
        self.assertEqual(_qkv_streaming_request(QKV_STREAMING_MODE_FORCED), QKV_STREAMING_FORCED)

    def test_streaming_auto_preserves_current_attention(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_AUTO)
        self.assertEqual(request.attention_v_memory, KITCHEN_V_MEMORY_RETAIN)

    def test_two_pass_v_mode_is_explicit(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
            kitchen_v_memory_mode=KITCHEN_V_MEMORY_MODE_TWO_PASS,
        )
        self.assertEqual(request.attention_v_memory, KITCHEN_V_MEMORY_TWO_PASS)

    def test_streaming_forced_claims_attention(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_FORCED,
        )
        self.assertEqual(request.attention, ATTENTION_AUTO)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_FORCED)

    def test_streaming_off_preserves_attention_and_native_qkv_policy(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
            qkv_streaming_mode=QKV_STREAMING_MODE_OFF,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_PRESERVE_BF16)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_OFF)

    def test_precision_modes_map_to_distinct_execution_policies(self):
        expected = {
            PRECISION_MODE_AUTO: (FUSED_QKV_AUTO, MLP_MEMORY_AUTO, False),
            PRECISION_MODE_BF16: (FUSED_QKV_FORCE_BF16, MLP_MEMORY_BF16, True),
            PRECISION_MODE_PRESERVE_NATIVE: (
                FUSED_QKV_PRESERVE_BF16,
                MLP_MEMORY_PRESERVE,
                False,
            ),
            PRECISION_MODE_FORCE_QUANT: (
                FUSED_QKV_FORCE_QUANT,
                MLP_MEMORY_FORCE_QUANT,
                True,
            ),
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                request = _memory_request_for_modes(
                    fused_qkv='auto',
                    mlp_memory='auto',
                    chunk_rows=2048,
                    precision_mode=mode,
                    qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
                )
                self.assertEqual(
                    (request.fused_qkv, request.mlp_memory, request.mlp_strict),
                    values,
                )

    def test_streaming_off_remains_authoritative_over_precision_policy(self):
        request = _memory_request_for_modes(
            fused_qkv='auto',
            mlp_memory='auto',
            chunk_rows=2048,
            precision_mode=PRECISION_MODE_AUTO,
            qkv_streaming_mode=QKV_STREAMING_MODE_OFF,
        )
        self.assertEqual(request.attention, ATTENTION_EXISTING)
        self.assertEqual(request.fused_qkv, FUSED_QKV_AUTO)
        self.assertEqual(request.qkv_streaming, QKV_STREAMING_OFF)

    def test_memory_schema_appends_streaming_after_precision_mode(self):
        schema = H3MemoryOptimization.define_schema()
        inputs = schema.inputs
        ids = [item.id for item in inputs]
        legacy_index = ids.index('preserve_precision')
        precision_index = ids.index('precision_mode')
        streaming_index = ids.index('qkv_streaming_mode')
        embedding_index = ids.index('embedding_memory_mode')
        v_memory_index = ids.index('kitchen_v_memory_mode')
        self.assertEqual(precision_index, legacy_index + 1)
        self.assertEqual(streaming_index, precision_index + 1)
        self.assertEqual(embedding_index, streaming_index + 1)
        self.assertEqual(v_memory_index, embedding_index + 1)

        legacy = inputs[legacy_index]
        precision = inputs[precision_index]
        streaming = inputs[streaming_index]
        self.assertTrue(legacy.extra_dict.get('hidden'))
        self.assertEqual(precision.options, list(PRECISION_MODE_OPTIONS))
        self.assertEqual(precision.default, PRECISION_MODE_AUTO)
        self.assertEqual(streaming.default, QKV_STREAMING_MODE_AUTO)
        self.assertEqual(inputs[embedding_index].default, 'Auto')
        self.assertTrue(inputs[embedding_index].extra_dict.get('hidden'))
        attention_memory = inputs[v_memory_index]
        self.assertEqual(attention_memory.display_name, 'Attention memory mode')
        self.assertEqual(
            attention_memory.options,
            ['Standard', 'Lower VRAM (slower)'],
        )
        self.assertEqual(attention_memory.default, KITCHEN_V_MEMORY_MODE_RETAIN)
        visible_text = ' '.join(
            [attention_memory.display_name, attention_memory.tooltip]
            + attention_memory.options
        ).lower()
        self.assertNotIn('kitchen', visible_text)
        self.assertNotIn('two-pass', visible_text)
        self.assertNotIn('bf16 v', visible_text)

    def test_memory_node_accepts_legacy_precision_values(self):
        self.assertTrue(H3MemoryOptimization.validate_inputs(PRECISION_MODE_PRESERVE))
        self.assertTrue(H3MemoryOptimization.validate_inputs(PRECISION_MODE_ALLOW_FP8))

    def test_memory_node_accepts_any_positive_chunk_size(self):
        for chunk_rows in (1, 257, 65_537):
            self.assertTrue(
                H3MemoryOptimization.validate_inputs(
                    PRECISION_MODE_AUTO,
                    chunk_rows,
                )
            )
        for chunk_rows in (0, -1):
            self.assertIsInstance(
                H3MemoryOptimization.validate_inputs(
                    PRECISION_MODE_AUTO,
                    chunk_rows,
                ),
                str,
            )


if __name__ == '__main__':
    unittest.main()
