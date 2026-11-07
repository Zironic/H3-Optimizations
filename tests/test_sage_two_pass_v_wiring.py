"""CPU contracts for routing the Lower VRAM request into the Sage carriers."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
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

from h3_optimizations.attention.sage_v_fp8 import TRITON_AVAILABLE  # noqa: E402
from h3_optimizations.attention.sparse.sparse_sage import (  # noqa: E402
    SparseSageKernelSpec,
)
from h3_optimizations.attention.sparse.sparse_sage_streamed import (  # noqa: E402
    StreamedSparseSageQKVProjector,
)
from h3_optimizations.dense_streamed_sage import (  # noqa: E402
    StreamedDenseSageQKVProjector,
)
from h3_optimizations.plan import (  # noqa: E402
    V_MEMORY_CHUNK_SCALED,
    V_MEMORY_RETAIN,
    V_MEMORY_TWO_PASS,
)
from h3_optimizations.status import format_qkv_execution  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def _spec(v_format, **overrides):
    fields = dict(
        version='test',
        architecture='sm89',
        capability=(8, 9),
        q_tile=128,
        kv_tile=64,
        v_format=v_format,
        kernel=lambda *_args: None,
        accumulator='f32',
        v_quant_bound=2.25,
    )
    fields.update(overrides)
    return SparseSageKernelSpec(**fields)


def _dense_backend(parameters):
    backend = SimpleNamespace(
        name='fake-dense-sage',
        projected_q_tile=128,
        projected_k_tile=64,
    )
    backend.v_staging_parameters = lambda: parameters
    return backend


class SparseSageVModeTests(unittest.TestCase):
    def test_fp8_spec_accepts_the_two_pass_request(self):
        projector = StreamedSparseSageQKVProjector(
            _spec('fp8'), v_mode=V_MEMORY_TWO_PASS
        )

        self.assertEqual(projector.requested_v_mode, V_MEMORY_TWO_PASS)
        self.assertEqual(projector.v_mode, V_MEMORY_TWO_PASS)

    def test_lower_vram_request_runs_two_pass(self):
        # Lower VRAM requests chunk-scaled V, which Sage carriers cannot
        # hold; the same memory saving comes from staging V in two passes.
        projector = StreamedSparseSageQKVProjector(
            _spec('fp8'), v_mode=V_MEMORY_CHUNK_SCALED
        )

        self.assertEqual(projector.requested_v_mode, V_MEMORY_CHUNK_SCALED)
        self.assertEqual(projector.v_mode, V_MEMORY_TWO_PASS)

    def test_fp16_v_spec_declines_and_reports_retain(self):
        # The Ampere kernels carry V in FP16, which is the same size as the
        # BF16 source: a second pass would cost a reprojection for no saving.
        projector = StreamedSparseSageQKVProjector(
            _spec('fp16'), v_mode=V_MEMORY_TWO_PASS
        )

        self.assertEqual(projector.requested_v_mode, V_MEMORY_TWO_PASS)
        self.assertEqual(projector.v_mode, V_MEMORY_RETAIN)

    def test_v_mode_is_part_of_the_projector_identity(self):
        retain = StreamedSparseSageQKVProjector(
            _spec('fp8'), v_mode=V_MEMORY_RETAIN
        )
        two_pass = StreamedSparseSageQKVProjector(
            _spec('fp8'), v_mode=V_MEMORY_TWO_PASS
        )

        self.assertNotEqual(
            retain.installation_signature,
            two_pass.installation_signature,
        )

    def test_unknown_v_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown sparse Sage V mode'):
            StreamedSparseSageQKVProjector(_spec('fp8'), v_mode='sometimes')

    def test_fp8_spec_reports_staging_parameters_with_kernel_padding(self):
        # 128 is the padding the kernel ABI expects; _validate_staged_v checks
        # the same number, so a drift here would surface as an invalid carrier.
        parameters = _spec('fp8').v_staging_parameters()

        if not TRITON_AVAILABLE:
            self.assertIsNone(parameters)
            self.skipTest('two-pass Sage V staging requires Triton')
        self.assertEqual(parameters, (2.25, 128))

    def test_fp16_spec_reports_no_staging_parameters(self):
        self.assertIsNone(_spec('fp16').v_staging_parameters())


class DenseSageVModeTests(unittest.TestCase):
    def test_backend_parameters_accept_the_two_pass_request(self):
        projector = StreamedDenseSageQKVProjector(
            _dense_backend((2.25, 64)),
            chunk_rows=128,
            projection_mode='native',
            v_mode=V_MEMORY_TWO_PASS,
        )

        self.assertEqual(projector.v_mode, V_MEMORY_TWO_PASS)

    def test_lower_vram_request_runs_two_pass(self):
        projector = StreamedDenseSageQKVProjector(
            _dense_backend((2.25, 64)),
            chunk_rows=128,
            projection_mode='native',
            v_mode=V_MEMORY_CHUNK_SCALED,
        )

        self.assertEqual(projector.requested_v_mode, V_MEMORY_CHUNK_SCALED)
        self.assertEqual(projector.v_mode, V_MEMORY_TWO_PASS)

    def test_backend_without_staging_support_reports_retain(self):
        projector = StreamedDenseSageQKVProjector(
            _dense_backend(None),
            chunk_rows=128,
            projection_mode='native',
            v_mode=V_MEMORY_TWO_PASS,
        )

        self.assertEqual(projector.requested_v_mode, V_MEMORY_TWO_PASS)
        self.assertEqual(projector.v_mode, V_MEMORY_RETAIN)

    def test_v_mode_is_part_of_the_projector_identity(self):
        backend = _dense_backend((2.25, 64))
        retain = StreamedDenseSageQKVProjector(
            backend, chunk_rows=128, projection_mode='native',
            v_mode=V_MEMORY_RETAIN,
        )
        two_pass = StreamedDenseSageQKVProjector(
            backend, chunk_rows=128, projection_mode='native',
            v_mode=V_MEMORY_TWO_PASS,
        )

        self.assertNotEqual(
            retain.installation_signature,
            two_pass.installation_signature,
        )

    def test_default_is_retain(self):
        projector = StreamedDenseSageQKVProjector(
            _dense_backend((2.25, 64)),
            chunk_rows=128,
            projection_mode='native',
        )

        self.assertEqual(projector.v_mode, V_MEMORY_RETAIN)


class SageStatusTests(unittest.TestCase):
    def _status(self, provider, projector, v_memory):
        return {
            'attention': {'selected': 'sparse_sage'},
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16'] * 50},
            'mlp': {'provider': 'off'},
            'fused_qkv': {
                'provider': provider,
                'projector': projector,
                'chunk_rows': 4096,
                'streamed_q': True,
                'v_memory': v_memory,
            },
        }

    def test_sparse_sage_reports_two_pass_v(self):
        text = format_qkv_execution(
            self._status(
                'chunked_fp8_sparse_sage', 'chunked_sparse_sage_qkv', 'two_pass'
            )
        )

        self.assertIn('two-pass V', text)
        self.assertIn('staged Sparse Sage V', text)
        self.assertNotIn('retained Sparse Sage K/V', text)

    def test_dense_sage_reports_two_pass_v(self):
        text = format_qkv_execution(
            self._status(
                'force_convrot_int8_qkv', 'streamed_dense_sage_qkv', 'two_pass'
            )
        )

        self.assertIn('two-pass V', text)
        self.assertIn('staged native Sage V', text)
        self.assertNotIn('retained native Sage K/V', text)

    def test_retained_sage_does_not_claim_two_pass(self):
        text = format_qkv_execution(
            self._status(
                'chunked_fp8_sparse_sage', 'chunked_sparse_sage_qkv', 'retain'
            )
        )

        self.assertNotIn('two-pass', text)

    def test_kitchen_still_reports_two_pass_exactly_once(self):
        text = format_qkv_execution(
            self._status(
                'streamed_bf16_kitchen_qkv', 'chunked_kitchen_qkv', 'two_pass'
            )
        )

        self.assertEqual(text.count('two-pass V'), 1)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
