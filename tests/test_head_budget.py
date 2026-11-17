'''CPU contracts for calibrated per-head budgets on the threshold sparse route.

Nothing here touches CUDA or a model: the bundled table, the allocation, the
covariance scaling that turns it into a per-head tau, the fallbacks, and the
request/config/backend wiring. Quality evidence lives in H3-Extended
(h3_head_budget) and in the release notes, not here.
'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations import apply as h3_apply  # noqa: E402
from h3_optimizations.attention.sparse import head_budget as hb  # noqa: E402
from h3_optimizations.attention.sparse import kitchen_streamed_q as ksq  # noqa: E402
from h3_optimizations.attention.sparse.config import HybridSparseConfig  # noqa: E402
from h3_optimizations.native import threshold_route  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    DEFAULT_HEAD_BUDGETS,
    HEAD_BUDGETS_CALIBRATED,
    HEAD_BUDGETS_OFF,
    H3OptimizationPlan,
    SparseRequest,
    VIDEO_TOKEN_ORDER_RASTER,
)

sys.argv = [sys.argv[0], *TEST_ARGS]

HEADS = 56
TOTAL = 1300  # pure-video KV tiles at ~42k video tokens with 32-token tiles


def _plan(retained, *, heads=HEADS, mode=ksq.ROUTE_MODE_THRESHOLD, dim=8, seed=3):
    torch.manual_seed(seed)
    root = torch.randn(1, heads, dim, dim)
    cov = (root @ root.transpose(-1, -2) / dim).to(torch.bfloat16)
    geometry = SimpleNamespace(pure_video_kv_tiles=TOTAL)
    return ksq._StreamedRoutePlan(
        geometry=geometry, retained=retained, k_summary=None, batch=1, heads=heads, mode=mode,
        k_mean=None, k_cov=cov, tau=threshold_route.budget_tau(retained / TOTAL),
    )


class BundledProfileTests(unittest.TestCase):
    def test_bundled_table_is_the_calibrated_geometry(self):
        profile = hb.bundled_profile()
        self.assertIsNotNone(profile, 'the bundled table must load')
        self.assertEqual((profile.q_tile, profile.kv_tile, profile.heads, profile.layer_count), (64, 32, 56, 50))
        self.assertEqual(profile.video_token_order, '1x8x8')
        self.assertEqual(sorted(profile.layers), list(range(50)))
        self.assertEqual(profile.fractions[0], 0.01)

    def test_allocation_is_compute_matched_and_stays_below_half(self):
        profile = hb.bundled_profile()
        for layer in (0, 24, 45, 49):
            for retained in (130, 196, 390):
                with self.subTest(layer=layer, retained=retained):
                    fractions = profile.fractions_for(layer, TOTAL, retained)
                    self.assertIsNotNone(fractions)
                    self.assertEqual(len(fractions), HEADS)
                    self.assertGreater(len(set(fractions)), 1, 'heads get different fractions')
                    self.assertLess(max(fractions), 0.5)
                    densities = [hb.density_points(profile.fractions, row) for row in profile.density[layer]]
                    shared = sum(hb.interpolate(points, retained / TOTAL) for points in densities)
                    spent = sum(hb.interpolate(points, value) for points, value in zip(densities, fractions))
                    self.assertAlmostEqual(spent, shared, places=6, msg='realized tiles match the shared tau')

    def test_golden_allocation_pins_the_table(self):
        # Values from the H3-Extended allocator the 2026-11-17 validation ran.
        profile = hb.bundled_profile()
        self.assertEqual(profile.fractions_for(45, TOTAL, 196)[:3], (0.075, 0.2, 0.05))
        self.assertEqual(profile.fractions_for(49, TOTAL, 196)[:3], (0.39999999999999997, 0.1, 0.15))

    def test_matches_the_h3_extended_allocator_when_installed(self):
        extended = ROOT / 'custom_nodes' / 'ComfyUI-H3-Extended'
        source = Path(r'D:\AI\ComfyUI\Output\h3_quality_eval\h3_head_budget\profiles\heads_t32_v1.json')
        if not (extended / 'h3_head_budget' / 'profile.py').is_file() or not source.is_file():
            self.skipTest('H3-Extended or its calibration output is not installed here')
        sys.path.insert(0, str(extended))
        try:
            from h3_head_budget.profile import HeadAllocator, HeadBudgetProfile
        finally:
            sys.path.remove(str(extended))
        reference = HeadAllocator(HeadBudgetProfile.load(source), 0.0, 1.0)
        profile = hb.bundled_profile()
        for layer in (0, 1, 24, 45, 49):
            for total, retained in ((TOTAL, 130), (TOTAL, 196), (TOTAL, 546), (TOTAL, 650), (2600, 260)):
                self.assertEqual(profile.fractions_for(layer, total, retained),
                                 reference.fractions(layer, total, retained, HEADS))


class ApplyHeadTauTests(unittest.TestCase):
    def test_each_head_thresholds_at_its_own_tau(self):
        plan = _plan(196)
        original = plan.k_cov.clone()
        fractions = hb.apply_head_tau(plan, 45, q_tile=64, kv_tile=32)
        self.assertEqual(fractions, plan.head_fractions)
        q = torch.randn(32, 8, dtype=torch.float64)
        for head, fraction in enumerate(fractions):
            scale = hb.threshold_tau(fraction) / plan.tau
            # k_cov stays BF16: exactly one rounding of Sigma_h * (tau_h / tau)^2
            expected = (original[0, head].float() * (scale * scale)).to(torch.bfloat16)
            self.assertTrue(torch.equal(plan.k_cov[0, head], expected), 'head %d covariance' % head)
            # and in exact arithmetic tau * sqrt(q' (s^2 Sigma) q) == tau_h * sqrt(q' Sigma q)
            sigma = original[0, head].double()
            lhs = plan.tau * torch.einsum('rd,de,re->r', q, sigma * scale * scale, q).sqrt()
            rhs = hb.threshold_tau(fraction) * torch.einsum('rd,de,re->r', q, sigma, q).sqrt()
            torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)
        self.assertEqual(plan.k_cov.dtype, torch.bfloat16)

    def test_fallbacks_keep_the_shared_tau_bit_identically(self):
        cases = {
            'top-k route': (_plan(196, mode=ksq.ROUTE_MODE_TOPK), 45, 64, 32),
            'dense step': (_plan(TOTAL), 45, 64, 32),
            'budget at 0.5': (_plan(650), 45, 64, 32),
            'other KV tile': (_plan(196), 45, 64, 64),
            'other Q tile': (_plan(196), 45, 128, 32),
            'other head count': (_plan(196, heads=24), 45, 64, 32),
            'uncalibrated layer': (_plan(196), 50, 64, 32),
            'no layer index': (_plan(196), None, 64, 32),
        }
        for name, (plan, layer, q_tile, kv_tile) in cases.items():
            with self.subTest(name):
                before = plan.k_cov.clone()
                self.assertIsNone(hb.apply_head_tau(plan, layer, q_tile=q_tile, kv_tile=kv_tile))
                self.assertTrue(torch.equal(plan.k_cov, before))
                self.assertIsNone(plan.head_fractions)

    def test_double_application_is_refused(self):
        plan = _plan(196)
        hb.apply_head_tau(plan, 3, q_tile=64, kv_tile=32)
        with self.assertRaises(hb.HeadBudgetError):
            hb.apply_head_tau(plan, 3, q_tile=64, kv_tile=32)

    def test_unreadable_table_falls_back_once(self):
        with mock.patch.dict(hb._bundled, {'profile': None, 'error': None}), \
             mock.patch.object(hb, 'BUNDLED_PROFILE', PACK / 'missing.json.gz'):
            self.assertIsNone(hb.bundled_profile())
            plan = _plan(196)
            self.assertIsNone(hb.apply_head_tau(plan, 3, q_tile=64, kv_tile=32))


class RequestWiringTests(unittest.TestCase):
    def test_request_defaults_to_calibrated_on_the_calibrated_order(self):
        self.assertEqual(DEFAULT_HEAD_BUDGETS, HEAD_BUDGETS_CALIBRATED)
        self.assertTrue(SparseRequest().head_budgets_active)
        self.assertFalse(SparseRequest(head_budgets=HEAD_BUDGETS_OFF).head_budgets_active)
        self.assertFalse(SparseRequest(video_token_order=VIDEO_TOKEN_ORDER_RASTER).head_budgets_active)
        with self.assertRaises(ValueError):
            SparseRequest(head_budgets='Uniform')
        self.assertNotEqual(SparseRequest().signature, SparseRequest(head_budgets=HEAD_BUDGETS_OFF).signature)

    def test_config_carries_the_active_flag(self):
        on = h3_apply._sparse_config_kwargs(H3OptimizationPlan(sparse=SparseRequest()))
        off = h3_apply._sparse_config_kwargs(H3OptimizationPlan(sparse=SparseRequest(head_budgets=HEAD_BUDGETS_OFF)))
        self.assertTrue(on['head_budgets'])
        self.assertFalse(off['head_budgets'])
        self.assertFalse(HybridSparseConfig().head_budgets, 'internal configs stay shared-tau unless asked')
        self.assertNotEqual(HybridSparseConfig(**on).signature, HybridSparseConfig(**off).signature)


class BackendWiringTests(unittest.TestCase):
    def _prepare(self, config, options, layout=None):
        plan = _plan(196)
        original = plan.k_cov.clone()
        backend = ksq.StreamedSparseKitchenBackend.__new__(ksq.StreamedSparseKitchenBackend)
        backend.config = config
        backend.router = object()
        backend.executor = SimpleNamespace(q_tile=64, kv_tile=32)
        carrier = SimpleNamespace(k=torch.zeros(1, HEADS, 8, 8), cta_k=32)
        projected = ksq.StreamedSparseKitchenQKV(
            SimpleNamespace(), SimpleNamespace(), torch.zeros(16, 4), None, carrier, torch.zeros(1), 'native', None)
        snapshot = SimpleNamespace(step_index=0, total_steps=1, layout=layout)
        with mock.patch.object(ksq, 'snapshot_for', return_value=snapshot), \
             mock.patch.object(ksq, '_prepare_route_plan', return_value=(plan, {})), \
             mock.patch.object(ksq, 'route_metadata', return_value={}):
            prepared = backend.prepare_projected(projected, layer_index=45, transformer_options=options)
        return prepared, original

    def test_backend_applies_per_head_tau_only_when_configured(self):
        prepared, original = self._prepare(HybridSparseConfig(video_budget=0.15, head_budgets=True), {})
        self.assertIsNotNone(prepared.route_plan.head_fractions)
        self.assertFalse(torch.equal(prepared.route_plan.k_cov, original))
        self.assertTrue(prepared.metadata['head_budgets'])

        prepared, original = self._prepare(HybridSparseConfig(video_budget=0.15, head_budgets=False), {})
        self.assertIsNone(prepared.route_plan.head_fractions)
        self.assertTrue(torch.equal(prepared.route_plan.k_cov, original))
        self.assertFalse(prepared.metadata['head_budgets'])

        prepared, original = self._prepare(HybridSparseConfig(video_budget=0.15, head_budgets=True),
                                           {hb.HEAD_BUDGETS_OFF_KEY: True})
        self.assertIsNone(prepared.route_plan.head_fractions, 'the per-forward off switch keeps the shared tau')
        self.assertTrue(torch.equal(prepared.route_plan.k_cov, original))

    def test_h3_image_keeps_the_shared_tau(self):
        # MiniMax H3 Image packs no audio; the table is calibrated on the video models only.
        config = HybridSparseConfig(video_budget=0.15, head_budgets=True)
        prepared, original = self._prepare(config, {}, layout=SimpleNamespace(audio_t=0))
        self.assertIsNone(prepared.route_plan.head_fractions)
        self.assertTrue(torch.equal(prepared.route_plan.k_cov, original))
        prepared, _ = self._prepare(config, {}, layout=SimpleNamespace(audio_t=31))
        self.assertIsNotNone(prepared.route_plan.head_fractions, 'video models with audio keep per-head budgets')

    def test_h3_image_detection(self):
        from h3_optimizations import model as h3_model
        try:
            from comfy.ldm.minimax.model import MiniMaxH3ImageModel, MiniMaxH3Model
        except ImportError:
            self.skipTest('this ComfyUI predates MiniMax H3 Image')

        def patcher(cls):
            instance = cls.__new__(cls)
            return SimpleNamespace(get_model_object=lambda name: instance)
        self.assertTrue(h3_model.is_minimax_h3_image(patcher(MiniMaxH3ImageModel)))
        self.assertFalse(h3_model.is_minimax_h3_image(patcher(MiniMaxH3Model)))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
