'''CPU contracts for user-facing logging noise.'''

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

from h3_optimizations import aimdo_limiter  # noqa: E402
from h3_optimizations.memory import forward as forward_module  # noqa: E402
from h3_optimizations.native import selftest  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class _FakeVBAR:
    def __init__(self):
        self.base_addr = 0x10000000
        self.pages = 2
        self.watermark = 2
        self.residency = [1, 1]

    def get_nr_pages(self):
        return self.pages

    def set_watermark(self, size_bytes):
        requested = (
            size_bytes + aimdo_limiter.PAGE_SIZE - 1
        ) // aimdo_limiter.PAGE_SIZE
        self.watermark = min(requested, self.pages)
        for page in range(self.watermark, self.pages):
            self.residency[page] = 0

    def get_watermark(self):
        return self.watermark

    def get_residency(self):
        return list(self.residency)


class _FakeBlock:
    def __init__(self, vbar):
        self._v = (
            vbar,
            vbar.base_addr,
            aimdo_limiter.PAGE_SIZE,
        )

    def modules(self):
        yield self


class _FakePatcher:
    @staticmethod
    def is_dynamic():
        return True


class LoggingNoiseTests(unittest.TestCase):
    def test_mlp_fallback_is_debug_once_per_reason(self):
        reason = 'test incompatible optimized MLP format'
        forward_module._MLP_FALLBACK_LOGGED.discard(reason)

        with patch.object(forward_module.logging, 'debug') as debug:
            forward_module._log_mlp_fallback(3, reason)
            forward_module._log_mlp_fallback(17, reason)
            forward_module._log_mlp_fallback(3, reason)

        self.assertEqual(debug.call_count, 1)
        args = debug.call_args.args
        self.assertIn('preferred MLP optimization is unavailable', args[0])
        self.assertEqual(args[2], 3)
        self.assertEqual(args[3], reason)

    def test_partial_native_geometry_failure_is_debug_only(self):
        detail = {
            'dense_int8_passed': True,
            'production_sparse_passed': True,
            'full_route_passed': {
                '64x64': False,
                '128x64': True,
                '128x128': False,
            },
        }
        with (
            patch.object(selftest, '_result', None),
            patch.object(selftest, '_detail_result', None),
            patch.object(selftest.torch.cuda, 'is_available', return_value=True),
            patch.object(selftest.loader, 'is_available', return_value=True),
            patch.object(selftest, '_cache_key', return_value='test-gpu'),
            patch.object(selftest, '_read_cache', return_value={}),
            patch.object(selftest, '_write_cache'),
            patch.object(selftest, 'run', return_value=(True, dict(detail))),
            patch.object(selftest.logging, 'warning') as warning,
            patch.object(selftest.logging, 'info') as info,
            patch.object(selftest.logging, 'debug') as debug,
        ):
            passed, actual = selftest._load_result(force=True)

        self.assertTrue(passed)
        self.assertTrue(actual['passed'])
        warning.assert_not_called()
        info.assert_not_called()
        self.assertEqual(debug.call_count, 2)
        messages = [call.args[0] for call in debug.call_args_list]
        self.assertTrue(any(
            'validated fallback geometry remains available' in message
            for message in messages
        ))
        self.assertTrue(any(
            'native sparse self-test detail' in message
            for message in messages
        ))

    def test_full_native_selftest_failure_stays_warning(self):
        detail = {
            'dense_int8_passed': False,
            'production_sparse_passed': False,
            'full_route_passed': {'64x64': False, '128x64': False},
        }
        with (
            patch.object(selftest, '_result', None),
            patch.object(selftest, '_detail_result', None),
            patch.object(selftest.torch.cuda, 'is_available', return_value=True),
            patch.object(selftest.loader, 'is_available', return_value=True),
            patch.object(selftest, '_cache_key', return_value='test-gpu'),
            patch.object(selftest, '_read_cache', return_value={}),
            patch.object(selftest, '_write_cache'),
            patch.object(selftest, 'run', return_value=(False, dict(detail))),
            patch.object(selftest.logging, 'warning') as warning,
            patch.object(selftest.logging, 'info') as info,
        ):
            passed, actual = selftest._load_result(force=True)

        self.assertFalse(passed)
        self.assertFalse(actual['passed'])
        warning.assert_called_once()
        self.assertIn('NATIVE PRODUCTION SELF-TEST FAILED', warning.call_args.args[0])
        info.assert_not_called()

    def test_successful_aimdo_residency_application_is_debug_only(self):
        vbar = _FakeVBAR()
        patcher = _FakePatcher()
        with (
            patch.object(aimdo_limiter.comfy.model_management, 'NUM_STREAMS', 1),
            patch.object(
                aimdo_limiter,
                'get_h3_blocks',
                return_value=(_FakeBlock(vbar),),
            ),
            patch.object(aimdo_limiter.logging, 'debug') as debug,
            patch.object(aimdo_limiter.logging, 'info') as info,
        ):
            aimdo_limiter._apply_residency_cap(patcher, 1)

        debug.assert_called_once()
        self.assertIn('AIMDO residency limited', debug.call_args.args[0])
        info.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
