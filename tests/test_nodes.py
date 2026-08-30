'''CPU-only schema, disabled-node, and non-H3 no-op tests.'''

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.apply import apply_plan  # noqa: E402
from h3_optimizations.environment import RuntimeEnvironment  # noqa: E402
from h3_optimizations.nodes import (  # noqa: E402
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)
from h3_optimizations.plan import (  # noqa: E402
    EARLY_SCHEDULE_HOLD,
    EARLY_SCHEDULE_RAMP,
    H3OptimizationPlan,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def input_by_id(schema, input_id):
    return next(item for item in schema.inputs if item.id == input_id)


class NodeTests(unittest.TestCase):
    def test_node_schemas_are_small_and_stable(self):
        sparse = H3SparseAttention.define_schema()
        advanced = H3SparseAttentionAdvanced.define_schema()
        self.assertEqual(sparse.node_id, 'H3SparseAttention')
        self.assertEqual(sparse.display_name, 'H3 Sparse Attention')
        self.assertEqual(
            sparse.category,
            'H3-Optimizations/Model Patches',
        )
        self.assertEqual(
            [item.id for item in sparse.inputs],
            [
                'model',
                'video_budget',
                'denser_early_late_steps',
            ],
        )
        self.assertEqual(input_by_id(sparse, 'video_budget').default, 0.15)
        self.assertTrue(
            input_by_id(sparse, 'denser_early_late_steps').default
        )
        self.assertEqual(advanced.node_id, 'H3SparseAttentionAdvanced')
        self.assertEqual(
            advanced.display_name,
            'H3 Sparse Attention (Advanced)',
        )
        self.assertEqual(
            advanced.category,
            'H3-Optimizations/Model Patches',
        )
        self.assertEqual(
            [item.id for item in advanced.inputs],
            [
                'model',
                'video_budget',
                'early_steps',
                'early_kv',
                'late_steps',
                'late_kv',
                'backend',
                'early_schedule',
            ],
        )
        self.assertEqual(
            [item.id for item in advanced.inputs[:6]],
            [
                'model',
                'video_budget',
                'early_steps',
                'early_kv',
                'late_steps',
                'late_kv',
            ],
        )
        backend = input_by_id(advanced, 'backend')
        self.assertEqual(backend.default, 'Kitchen INT8')
        self.assertEqual(
            backend.options,
            [
                'Kitchen INT8',
                'FROST BF16 (SM89)',
                'Sparse Sage',
                'BF16 Triton',
                'FP8 FlexAttention',
            ],
        )
        self.assertIn('Kitchen INT8 64x64 is the default', advanced.description)
        self.assertIn('Bypass this node', backend.tooltip)
        self.assertIn(
            'BF16 Triton and FP8 FlexAttention use the same 64Q x 64KV',
            backend.tooltip,
        )
        self.assertIn('FROST BF16 uses 64Q x 64KV', backend.tooltip)
        self.assertNotIn('experimental', ' '.join(backend.options).lower())
        early_schedule = input_by_id(advanced, 'early_schedule')
        self.assertEqual(early_schedule.default, EARLY_SCHEDULE_HOLD)
        self.assertEqual(
            early_schedule.options,
            [EARLY_SCHEDULE_HOLD, EARLY_SCHEDULE_RAMP],
        )
        self.assertTrue(H3SparseAttentionAdvanced.validate_inputs('auto'))
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs(
                'Native INT8 128x128 + Sol residual 64x64'
            ),
            str,
        )
        self.assertTrue(
            H3SparseAttentionAdvanced.validate_inputs(
                'Kitchen INT8 (experimental)'
            )
        )
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs('not a backend'),
            str,
        )
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs(
                'Kitchen INT8',
                'Curve',
            ),
            str,
        )
        self.assertTrue(H3SparseAttention.validate_inputs(0.0))
        self.assertTrue(H3SparseAttention.validate_inputs(1.2))
        self.assertIsInstance(H3SparseAttention.validate_inputs(float('nan')), str)
        self.assertTrue(
            H3SparseAttentionAdvanced.validate_inputs(
                'Kitchen INT8',
                video_budget=-1.0,
                early_steps=1001,
                early_kv=1.2,
                late_steps=2000,
                late_kv=0.0,
            )
        )
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs(
                'Kitchen INT8',
                early_steps=-1,
            ),
            str,
        )
        self.assertIsInstance(
            H3SparseAttentionAdvanced.validate_inputs(
                'Kitchen INT8',
                late_kv=float('inf'),
            ),
            str,
        )
        self.assertEqual(input_by_id(advanced, 'video_budget').default, 0.15)
        self.assertEqual(input_by_id(advanced, 'early_steps').default, 4)
        self.assertEqual(input_by_id(advanced, 'early_kv').default, 0.5)
        self.assertEqual(input_by_id(advanced, 'late_steps').default, 0)
        self.assertEqual(input_by_id(advanced, 'late_kv').default, 0.5)
        self.assertNotIn('Experimental', sparse.display_name)
        self.assertNotIn('Experimental', advanced.display_name)

    def test_non_h3_models_do_not_probe_the_runtime(self):
        class OtherModel:
            model_options = {}

            @staticmethod
            def get_model_object(_name):
                raise KeyError('not a diffusion model')

        model = OtherModel()
        with patch.object(
            RuntimeEnvironment,
            'detect',
            side_effect=AssertionError('CUDA detection must not run'),
        ):
            self.assertIs(
                apply_plan(model, H3OptimizationPlan()),
                model,
            )


if __name__ == '__main__':
    unittest.main()
