'''Regression tests for ComfyUI static validation of linked node inputs.'''

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

from h3_optimizations.memory_migration_node import (  # noqa: E402
    H3MemoryOptimization,
    PRECISION_MODE_AUTO,
)
from h3_optimizations.nodes import (  # noqa: E402
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)
from h3_optimizations.plan import EARLY_SCHEDULE_HOLD  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class LinkedInputValidationTests(unittest.TestCase):
    def test_simple_sparse_accepts_linked_video_budget(self):
        self.assertTrue(H3SparseAttention.validate_inputs(None))

    def test_advanced_sparse_accepts_each_linked_validated_input(self):
        defaults = {
            'backend': 'Kitchen INT8',
            'early_schedule': EARLY_SCHEDULE_HOLD,
            'video_budget': 0.15,
            'early_steps': 4,
            'early_kv': 0.5,
            'late_steps': 0,
            'late_kv': 0.5,
        }
        for input_name in defaults:
            with self.subTest(input_name=input_name):
                values = dict(defaults)
                values[input_name] = None
                self.assertTrue(
                    H3SparseAttentionAdvanced.validate_inputs(**values)
                )

    def test_memory_accepts_each_linked_validated_input(self):
        self.assertTrue(
            H3MemoryOptimization.validate_inputs(None, chunk_rows=4096)
        )
        self.assertTrue(
            H3MemoryOptimization.validate_inputs(
                PRECISION_MODE_AUTO,
                chunk_rows=None,
            )
        )
        self.assertTrue(H3MemoryOptimization.validate_inputs(None, chunk_rows=None))

    def test_concrete_invalid_values_still_fail_validation(self):
        self.assertIsInstance(H3SparseAttention.validate_inputs(float('nan')), str)
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs(
                'Kitchen INT8',
                early_steps=-1,
            ),
            str,
        )
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs('not a backend'),
            str,
        )
        self.assertIsInstance(
            H3MemoryOptimization.validate_inputs(
                PRECISION_MODE_AUTO,
                chunk_rows=0,
            ),
            str,
        )
        self.assertIsInstance(
            H3MemoryOptimization.validate_inputs(
                'not a precision mode',
                chunk_rows=4096,
            ),
            str,
        )


if __name__ == '__main__':
    unittest.main()
