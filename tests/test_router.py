'''CPU tests for fixed-density Sparse Sage routing.'''

import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
    resolve_video_budget,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def layout(sequence=384, video_start=128):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=[
            (0, video_start - 32, 'text'),
            (video_start - 32, video_start, 'audio'),
            (video_start, sequence, 'video'),
        ],
        video_shape=(1, 1, sequence - video_start),
        audio_t=16,
    )


def routed_inputs():
    q = torch.zeros((1, 2, 384, 2), dtype=torch.float32)
    k = torch.zeros_like(q)
    q[0, 0, 128:, 0] = 1
    q[0, 1, 128:, 1] = 1
    for index, value in enumerate(((4, 0), (3, 0), (2, 0), (1, 0)), start=2):
        k[0, 0, index * 64:(index + 1) * 64] = torch.tensor(value)
    for index, value in enumerate(((0, 1), (0, 2), (0, 3), (0, 4)), start=2):
        k[0, 1, index * 64:(index + 1) * 64] = torch.tensor(value)
    return q, k


def decode(lut, valid):
    mask = torch.zeros(lut.shape, dtype=torch.bool)
    for index in range(valid.shape[-1]):
        count = int(valid[..., index].max().item())
        if count:
            delta = lut[..., index, :count]
            mask[..., index, :] = torch.nn.functional.one_hot(
                torch.cumsum(delta, dim=-1).long(),
                num_classes=lut.shape[-1],
            ).any(dim=-2)
    return mask


class RouterTests(unittest.TestCase):
    def test_short_video_tile_boundaries_use_dense_routes(self):
        class NoSelectionRouter(SparseTileRouter):
            def _select_indices(self, *_args, **_kwargs):
                raise AssertionError('dense boundary must not run Top-K')

        video_start = 5263
        cases = (
            (64, 64, 5312, 0, 0),
            (64, 64, 5313, 1, 1),
            (64, 128, 5312, 0, 0),
            (64, 128, 5313, 1, 0),
            (64, 128, 5376, 1, 0),
            (64, 128, 5377, 2, 1),
            (128, 64, 5313, 0, 1),
        )
        for q_tile, kv_tile, sequence, pure_q, pure_kv in cases:
            with self.subTest(
                q_tile=q_tile,
                kv_tile=kv_tile,
                sequence=sequence,
            ):
                router = NoSelectionRouter(q_tile=q_tile, kv_tile=kv_tile)
                current_layout = layout(sequence, video_start)
                geometry = router.geometry(current_layout)
                self.assertEqual(geometry.pure_video_q_tiles, pure_q)
                self.assertEqual(geometry.pure_video_kv_tiles, pure_kv)

                q = torch.randn((1, 2, sequence, 4))
                routes = (
                    router.build_lut(q, q, current_layout, 0.15),
                    router.build_lut_from_summaries(
                        router._mean_pool(q, q_tile),
                        router._mean_pool(q, kv_tile),
                        current_layout,
                        0.15,
                    ),
                )
                for lut, valid, metadata in routes:
                    self.assertTrue(decode(lut, valid).all())
                    self.assertEqual(
                        metadata.retained_video_kv_tiles,
                        pure_kv,
                    )
                    self.assertEqual(metadata.actual_video_tile_density, 1.0)
                    self.assertEqual(metadata.full_mask_density, 1.0)
                    self.assertEqual(metadata.sparse_q_tiles, 0)
                    self.assertEqual(metadata.dense_q_tiles, geometry.q_tiles)

    def test_optional_early_ramp_is_bounded(self):
        config = HybridSparseConfig(
            video_budget=0.15,
            denser_early_late_steps=True,
        )
        budgets = [resolve_video_budget(config, step, 10) for step in range(10)]
        self.assertEqual(budgets[0], 0.5)
        self.assertTrue(
            all(left >= right for left, right in zip(budgets, budgets[1:]))
        )
        self.assertAlmostEqual(sum(value - 0.15 for value in budgets), 1.2)
        self.assertEqual(budgets[6:], [0.15] * 4)
        self.assertEqual(resolve_video_budget(config, -1, 10), 0.15)
        already_denser = HybridSparseConfig(
            video_budget=0.85,
            denser_early_late_steps=True,
        )
        self.assertEqual(resolve_video_budget(already_denser, 0, 10), 0.85)
        eleven_steps = [
            resolve_video_budget(config, step, 11) for step in range(11)
        ]
        self.assertEqual(eleven_steps[0], 0.5)
        self.assertAlmostEqual(
            sum(value - 0.15 for value in eleven_steps),
            1.32,
        )

    def test_per_head_top_k_and_dense_context(self):
        q, k = routed_inputs()
        lut, valid, metadata = SparseTileRouter().build_lut(
            q,
            k,
            layout(),
            0.5,
        )
        mask = decode(lut, valid)
        self.assertEqual(mask.shape, (1, 2, 3, 6))
        self.assertTrue(mask[..., :2].all())
        self.assertTrue(mask[:, :, 0].all())
        self.assertEqual(
            set(torch.where(mask[0, 0, 1, 2:])[0].tolist()),
            {0, 1},
        )
        self.assertEqual(
            set(torch.where(mask[0, 1, 1, 2:])[0].tolist()),
            {2, 3},
        )
        self.assertEqual(metadata.retained_video_kv_tiles, 2)
        self.assertEqual(metadata.actual_video_tile_density, 0.5)

    def test_mixed_boundaries_and_partial_tiles_stay_safe(self):
        q = torch.randn((1, 1, 350, 4))
        lut, valid, metadata = SparseTileRouter().build_lut(
            q,
            q,
            layout(sequence=350, video_start=96),
            0.5,
        )
        mask = decode(lut, valid)
        self.assertTrue(mask[:, :, 0].all())
        self.assertTrue(mask[..., 1].all())
        self.assertEqual((metadata.q_tiles, metadata.kv_tiles), (3, 6))
        self.assertEqual(metadata.pure_video_q_tiles, 2)
        self.assertEqual(metadata.pure_video_kv_tiles, 4)

    def test_full_budget_skips_similarity_scoring(self):
        class NoPoolingRouter(SparseTileRouter):
            @staticmethod
            def _mean_pool(_x, _block):
                raise AssertionError('full budget must not pool Q or K')

        q = torch.randn((1, 2, 384, 8))
        lut, valid, metadata = NoPoolingRouter().build_lut(
            q,
            q,
            layout(),
            1.0,
        )
        self.assertTrue(decode(lut, valid).all())
        self.assertEqual(metadata.full_mask_density, 1.0)
        self.assertEqual(metadata.sparse_q_tiles, 0)

    def test_finite_budgets_outside_the_ui_range_saturate(self):
        q, k = routed_inputs()
        router = SparseTileRouter()
        _lut, _valid, sparse = router.build_lut(q, k, layout(), -1.0)
        lut, valid, dense = router.build_lut(q, k, layout(), 1.2)

        self.assertEqual(sparse.requested_video_budget, -1.0)
        self.assertEqual(sparse.retained_video_kv_tiles, 1)
        self.assertEqual(dense.requested_video_budget, 1.2)
        self.assertEqual(dense.retained_video_kv_tiles, 4)
        self.assertTrue(decode(lut, valid).all())
        self.assertEqual(router._retained(-1e308, router.geometry(layout())), 1)
        self.assertEqual(router._retained(1e308, router.geometry(layout())), 4)

        HybridSparseConfig(
            video_budget=-1.0,
            early_steps=1001,
            early_kv=1.2,
            late_steps=2000,
            late_kv=0.0,
        )
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaisesRegex(ValueError, 'finite'):
                HybridSparseConfig(video_budget=value)

if __name__ == '__main__':
    unittest.main()
