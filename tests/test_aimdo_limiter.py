'''CPU-only AIMDO residency limiter contracts.'''

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

from comfy.patcher_extension import CallbacksMP  # noqa: E402
from h3_optimizations import aimdo_limiter  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeVBAR:
    def __init__(self, pages=16, base_addr=0x10000000, residency=None):
        self.base_addr = base_addr
        self.pages = pages
        self.watermark = pages
        self.residency = list(residency if residency is not None else [1] * pages)
        self.set_calls = []

    def get_nr_pages(self):
        return self.pages

    def set_watermark(self, size_bytes):
        self.set_calls.append(size_bytes)
        requested = (size_bytes + aimdo_limiter.PAGE_SIZE - 1) // aimdo_limiter.PAGE_SIZE
        self.watermark = min(requested, self.pages)
        for page in range(self.watermark, self.pages):
            if not self.residency[page] & 2:
                self.residency[page] = 0

    def get_watermark(self):
        return self.watermark

    def get_residency(self):
        return list(self.residency)


class FakeModule:
    def __init__(self, allocation=None, children=()):
        if allocation is not None:
            self._v = allocation
        self.children = tuple(children)

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()


class FakePatcher:
    def __init__(self, dynamic=True, vbar=None):
        self.dynamic = dynamic
        self.vbar = vbar
        self.callbacks = {}

    def is_dynamic(self):
        return self.dynamic

    def _vbar_get(self):
        return self.vbar

    def clone(self):
        clone = FakePatcher(dynamic=self.dynamic, vbar=self.vbar)
        clone.callbacks = {
            call_type: {
                key: list(callbacks)
                for key, callbacks in keyed_callbacks.items()
            }
            for call_type, keyed_callbacks in self.callbacks.items()
        }
        return clone

    def remove_callbacks_with_key(self, call_type, key):
        self.callbacks.get(call_type, {}).pop(key, None)

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {}).setdefault(key, []).append(callback)


def allocation(vbar, page, pages=1, offset=0):
    return (
        vbar,
        vbar.base_addr + page * aimdo_limiter.PAGE_SIZE + offset,
        pages * aimdo_limiter.PAGE_SIZE,
    )


def blocks_for(vbar):
    return (
        FakeModule(children=(FakeModule(allocation(vbar, 0)),)),
        FakeModule(children=(FakeModule(allocation(vbar, 1, pages=2)),)),
        FakeModule(children=(FakeModule(allocation(vbar, 3, pages=3)),)),
    )


class AIMDOLimiterTests(unittest.TestCase):
    def test_schema_exposes_explicit_residency_arms(self):
        schema = aimdo_limiter.H3AIMDOResidencyLimiter.define_schema()
        self.assertEqual(schema.node_id, 'H3AIMDOResidencyLimiter')
        self.assertEqual(schema.display_name, 'H3 AIMDO Residency Limiter')
        self.assertEqual(schema.category, 'H3-Optimizations/Model Patches')
        self.assertEqual([item.id for item in schema.inputs], ['model', 'residency'])
        residency = schema.inputs[1]
        self.assertEqual(
            residency.options,
            ['0 blocks', '1 block', '2 blocks', '4 blocks', 'stock'],
        )
        self.assertEqual(residency.default, '0 blocks')
        self.assertEqual(residency.options[0], residency.default)
        self.assertIn('model-agnostic', residency.tooltip)
        self.assertIn('keeps no DynamicVRAM VBAR pages persistently resident', residency.tooltip)
        self.assertIn('require DynamicVRAM', residency.tooltip)

    def test_block_budget_uses_page_footprints(self):
        vbar = FakeVBAR()
        found_vbar, counts = aimdo_limiter._block_page_counts(blocks_for(vbar))
        self.assertIs(found_vbar, vbar)
        self.assertEqual(counts, (1, 2, 3))
        expected = {0: 0, 1: 3, 2: 5, 4: 6}
        for blocks, pages in expected.items():
            with self.subTest(blocks=blocks):
                self.assertEqual(
                    aimdo_limiter._residency_cap_pages(counts, blocks),
                    pages,
                )

    def test_page_count_includes_straddling_allocations(self):
        vbar = FakeVBAR()
        item = FakeModule(
            (
                vbar,
                vbar.base_addr + aimdo_limiter.PAGE_SIZE - 512,
                1024,
            )
        )
        _, counts = aimdo_limiter._block_page_counts((item,))
        self.assertEqual(counts, (2,))

    def test_callback_replaces_existing_policy(self):
        patcher = FakePatcher()
        aimdo_limiter.install_aimdo_limiter(patcher, 1)
        first = patcher.callbacks[CallbacksMP.ON_LOAD][aimdo_limiter.CALLBACK_KEY][0]
        aimdo_limiter.install_aimdo_limiter(patcher, 2)
        callbacks = patcher.callbacks[CallbacksMP.ON_LOAD][aimdo_limiter.CALLBACK_KEY]
        self.assertEqual(len(callbacks), 1)
        self.assertIsNot(callbacks[0], first)

    def test_node_arms_limiter_on_a_dynamic_h3_clone(self):
        patcher = FakePatcher()
        with patch.object(aimdo_limiter, 'is_minimax_h3', return_value=True):
            output = aimdo_limiter.H3AIMDOResidencyLimiter.execute(
                patcher,
                residency='2 blocks',
            )
        patched = output.args[0]
        self.assertIsNot(patched, patcher)
        callbacks = patched.callbacks[CallbacksMP.ON_LOAD][aimdo_limiter.CALLBACK_KEY]
        self.assertEqual(len(callbacks), 1)

    def test_node_arms_zero_residency_on_a_non_h3_dynamic_clone(self):
        patcher = FakePatcher(vbar=FakeVBAR())
        with patch.object(aimdo_limiter, 'is_minimax_h3', return_value=False):
            output = aimdo_limiter.H3AIMDOResidencyLimiter.execute(
                patcher,
                residency='0 blocks',
            )
        patched = output.args[0]
        self.assertIsNot(patched, patcher)
        callbacks = patched.callbacks[CallbacksMP.ON_LOAD][aimdo_limiter.CALLBACK_KEY]
        self.assertEqual(len(callbacks), 1)

    def test_positive_block_budget_still_passes_through_non_h3_models(self):
        patcher = FakePatcher()
        with patch.object(aimdo_limiter, 'is_minimax_h3', return_value=False):
            output = aimdo_limiter.H3AIMDOResidencyLimiter.execute(
                patcher,
                residency='2 blocks',
            )
        self.assertIs(output.args[0], patcher)
        self.assertEqual(patcher.callbacks, {})

    def test_stock_removes_a_previous_limiter_from_the_clone(self):
        patcher = FakePatcher()
        aimdo_limiter.install_aimdo_limiter(patcher, 2)
        with patch.object(aimdo_limiter, 'is_minimax_h3', return_value=True):
            output = aimdo_limiter.H3AIMDOResidencyLimiter.execute(
                patcher,
                residency='stock',
            )
        patched = output.args[0]
        self.assertIsNot(patched, patcher)
        self.assertNotIn(
            aimdo_limiter.CALLBACK_KEY,
            patched.callbacks[CallbacksMP.ON_LOAD],
        )
        self.assertIn(
            aimdo_limiter.CALLBACK_KEY,
            patcher.callbacks[CallbacksMP.ON_LOAD],
        )

    def test_two_block_callback_applies_and_verifies_page_cap(self):
        vbar = FakeVBAR(pages=8)
        patcher = FakePatcher()
        with (
            patch.object(aimdo_limiter.comfy.model_management, 'NUM_STREAMS', 2),
            patch.object(aimdo_limiter, 'get_h3_blocks', return_value=blocks_for(vbar)),
        ):
            aimdo_limiter._apply_residency_cap(patcher, 2)
        self.assertEqual(vbar.set_calls, [5 * aimdo_limiter.PAGE_SIZE])
        self.assertEqual(vbar.get_watermark(), 5)
        self.assertFalse(any(value & 1 for value in vbar.residency[5:]))

    def test_zero_block_callback_clears_generic_vbar_without_h3_discovery(self):
        vbar = FakeVBAR(pages=8)
        patcher = FakePatcher(vbar=vbar)
        with (
            patch.object(aimdo_limiter.comfy.model_management, 'NUM_STREAMS', 2),
            patch.object(
                aimdo_limiter,
                'get_h3_blocks',
                side_effect=AssertionError('generic zero residency must not inspect H3 blocks'),
            ),
        ):
            aimdo_limiter._apply_residency_cap(patcher, 0)
        self.assertEqual(vbar.set_calls, [0])
        self.assertEqual(vbar.get_watermark(), 0)
        self.assertFalse(any(value & 1 for value in vbar.residency))

    def test_async_offload_is_required_before_mutating_watermark(self):
        vbar = FakeVBAR(pages=8)
        patcher = FakePatcher()
        with (
            patch.object(aimdo_limiter.comfy.model_management, 'NUM_STREAMS', 0),
            patch.object(aimdo_limiter, 'get_h3_blocks', return_value=blocks_for(vbar)),
            self.assertRaisesRegex(
                aimdo_limiter.AIMDOResidencyLimiterError,
                'requires async weight offloading',
            ),
        ):
            aimdo_limiter._apply_residency_cap(patcher, 2)
        self.assertEqual(vbar.set_calls, [])

    def test_non_dynamic_callback_is_an_exact_no_op(self):
        patcher = FakePatcher(dynamic=False)
        with patch.object(
            aimdo_limiter,
            'get_h3_blocks',
            side_effect=AssertionError('H3 blocks must not be inspected'),
        ):
            aimdo_limiter._apply_residency_cap(patcher, 2)

    def test_resident_pinned_page_above_cap_fails_closed(self):
        residency = [1] * 8
        residency[7] = 3
        vbar = FakeVBAR(pages=8, residency=residency)
        patcher = FakePatcher()
        with (
            patch.object(aimdo_limiter.comfy.model_management, 'NUM_STREAMS', 2),
            patch.object(aimdo_limiter, 'get_h3_blocks', return_value=blocks_for(vbar)),
            self.assertRaisesRegex(
                aimdo_limiter.AIMDOResidencyLimiterError,
                'resident above the limiter watermark',
            ),
        ):
            aimdo_limiter._apply_residency_cap(patcher, 2)


if __name__ == '__main__':
    unittest.main()
