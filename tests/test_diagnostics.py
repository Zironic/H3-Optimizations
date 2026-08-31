'''The stage hook must be inert by default and cover the real producer path.'''

from pathlib import Path
import ast
import sys
import unittest

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))

from h3_optimizations import diagnostics  # noqa: E402

SOURCE = PACK / 'h3_optimizations'


class RecordingStages:
    def __init__(self):
        self.entered = []
        self.closed = []

    def stage(self, name):
        return _Region(self, name)


class _Region:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def __enter__(self):
        self.owner.entered.append(self.name)
        return None

    def __exit__(self, exc_type, exc, traceback):
        self.owner.closed.append(self.name)
        return False


class DiagnosticsHookTests(unittest.TestCase):
    def test_default_is_the_shared_no_op(self):
        self.assertIs(diagnostics.active(), diagnostics.NULL_STAGES)
        with diagnostics.stage('anything'):
            pass

    def test_the_no_op_allocates_nothing_per_call(self):
        first = diagnostics.stage('a')
        second = diagnostics.stage('b')
        self.assertIs(first, second)
        self.assertIs(first, diagnostics.NULL_STAGES)

    def test_installed_recorder_sees_regions_and_is_restored(self):
        recorder = RecordingStages()
        with diagnostics.installed(recorder):
            self.assertIs(diagnostics.active(), recorder)
            with diagnostics.stage('outer'):
                with diagnostics.stage('inner'):
                    pass
        self.assertEqual(recorder.entered, ['outer', 'inner'])
        self.assertEqual(recorder.closed, ['inner', 'outer'])
        self.assertIs(diagnostics.active(), diagnostics.NULL_STAGES)

    def test_installed_restores_after_an_exception(self):
        with self.assertRaises(ValueError):
            with diagnostics.installed(RecordingStages()):
                raise ValueError('boom')
        self.assertIs(diagnostics.active(), diagnostics.NULL_STAGES)

    def test_nesting_restores_the_outer_recorder(self):
        outer, inner = RecordingStages(), RecordingStages()
        with diagnostics.installed(outer):
            with diagnostics.installed(inner):
                self.assertIs(diagnostics.active(), inner)
            self.assertIs(diagnostics.active(), outer)


class StageCoverageTests(unittest.TestCase):
    '''Every stage the attribution needs has to exist in production source.'''

    REQUIRED = {
        'attention_total': 'attention_forward.py',
        'attention_out': 'attention_forward.py',
        'qkv_linear': 'attention_forward.py',
        'qk_norm_rope': 'attention_forward.py',
        'q_activation_quant': 'qkv/fused_q.py',
        'fused_q_projection': 'qkv/fused_q.py',
        'qkv_producer_total': 'kitchen_qkv.py',
        'anchor_projection': 'kitchen_qkv.py',
        'routing_summary_generation': 'kitchen_qkv.py',
        'v_retention_copy': 'kitchen_qkv.py',
        'v_amax_update': 'kitchen_qkv.py',
        'v_reprojection': 'kitchen_qkv.py',
        'carrier_finalize': 'kitchen_qkv.py',
        'producer_create': 'kitchen_qkv.py',
        'anchor_selection': 'native/producer.py',
        'qk_pack_input_contiguous': 'native/producer.py',
        'qk_carrier_pack': 'native/producer.py',
        'v_carrier_pack': 'native/producer.py',
        'sparse_attention_kernel': 'native/int8_attention.py',
        'full_carrier_pack': 'attention/sparse/kitchen_sparse.py',
        'sparse_route': 'attention/sparse/kitchen_sparse.py',
        'sparse_carrier_prepare': 'attention/sparse/kitchen_sparse.py',
    }

    def test_every_required_stage_is_present_where_it_belongs(self):
        for name, relative in self.REQUIRED.items():
            text = (SOURCE / relative).read_text(encoding='utf-8')
            self.assertIn(
                "diagnostics.stage('%s')" % name,
                text,
                '%s is missing stage %s' % (relative, name),
            )

    def test_held_chunk_projectors_expose_projection_stages(self):
        for relative in ('qkv/w4a8.py', 'qkv/fp8.py'):
            text = (SOURCE / relative).read_text(encoding='utf-8')
            for name in ('qkv_linear', 'qk_norm_rope'):
                self.assertIn(
                    "diagnostics.stage('%s')" % name,
                    text,
                    '%s is missing stage %s' % (relative, name),
                )

    def test_declared_names_cover_the_stages_in_source(self):
        found = set()
        for path in SOURCE.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'stage'
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    found.add(node.args[0].value)
        self.assertTrue(found)
        self.assertEqual(
            found - set(diagnostics.STAGE_NAMES),
            set(),
            'undeclared stage names in source',
        )
        self.assertEqual(found, set(self.REQUIRED))

    def test_the_hook_module_never_touches_cuda(self):
        text = (SOURCE / 'diagnostics.py').read_text(encoding='utf-8')
        for fragment in ('import torch', 'cuda', 'Event'):
            self.assertNotIn(fragment, text.replace('CUDA event', ''))


if __name__ == '__main__':
    unittest.main()
