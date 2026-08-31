"""CPU contracts for H3-owned cube-major token ordering."""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

from comfy.ldm.minimax.model import MiniMaxH3Model, patchify_video  # noqa: E402
from h3_optimizations.cube_order import (  # noqa: E402
    CUBE_SHAPE,
    CUBE_SHAPES,
    FORWARD_KEY,
    H3CubeOrderPatchError,
    TOKEN_ORDER_SHAPES,
    clear,
    cube_major_indices,
    install,
    make_forward,
    tile_aligned_cube_major_indices,
)
from h3_optimizations.plan import (  # noqa: E402
    VIDEO_TOKEN_ORDER_1X16X4,
    VIDEO_TOKEN_ORDER_1X8X8,
    VIDEO_TOKEN_ORDER_4X4X4,
    VIDEO_TOKEN_ORDER_RASTER,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def _raster_video(t, h, w):
    return torch.arange(t * h * w, dtype=torch.float32).reshape(1, 1, t, h, w)


class _FakePatcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}
        self.model_options = {}

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name == "diffusion_model":
            return self.model
        if name == FORWARD_KEY:
            return self.model._forward
        raise KeyError(name)

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


def _model():
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    model._forward = lambda *args, **kwargs: None
    return model


class CubeOrderTests(unittest.TestCase):
    def test_all_supported_geometries_are_exact_tile_aligned_permutations(self):
        grid = (5, 17, 19)
        sequence_offset = 11
        prefix = (-sequence_offset) % 64
        for shape in CUBE_SHAPES:
            with self.subTest(shape=shape):
                forward, inverse = cube_major_indices(grid, shape)
                self.assertEqual(
                    [forward[index] for index in inverse],
                    list(range(5 * 17 * 19)),
                )
                aligned, aligned_inverse = tile_aligned_cube_major_indices(
                    grid,
                    sequence_offset,
                    shape,
                )
                self.assertEqual(
                    [aligned[index] for index in aligned_inverse],
                    list(range(5 * 17 * 19)),
                )
                rows = aligned[prefix : prefix + 64]
                coords = [
                    (
                        row // (grid[1] * grid[2]),
                        (row // grid[2]) % grid[1],
                        row % grid[2],
                    )
                    for row in rows
                ]
                self.assertEqual(len({t // shape[0] for t, _h, _w in coords}), 1)
                self.assertEqual(len({h // shape[1] for _t, h, _w in coords}), 1)
                self.assertEqual(len({w // shape[2] for _t, _h, w in coords}), 1)

    def test_whole_model_order_reorders_input_and_restores_raster_output(self):
        patch_size = (1, 2, 2)
        video = _raster_video(5, 10, 14)
        audio = torch.zeros(1, 2, 2, 4)
        context = torch.zeros(1, 3, 8)
        captured = {}
        model = type("Model", (), {"patch_size": patch_size})()

        def original(x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
            captured["video"] = x[0]
            captured["layout"] = minimax_payload["layout"]
            return [x[0].clone(), x[1].clone()]

        output = make_forward(model, original)(
            [video, audio],
            torch.tensor([500.0]),
            context,
            {},
            minimax_payload={},
        )
        self.assertTrue(torch.equal(output[0], video))
        layout = captured["layout"]
        start = next(start for start, _stop, kind in layout.segments if kind == "video")
        order, _inverse = tile_aligned_cube_major_indices((5, 5, 7), start)
        self.assertTrue(
            torch.equal(
                patchify_video(captured["video"], patch_size),
                patchify_video(video, patch_size)[list(order)],
            )
        )
        self.assertEqual(layout.h3_cube_order["cube_shape"], CUBE_SHAPE)

    def test_token_order_mapping_has_three_cube_arms_and_stock_raster(self):
        self.assertEqual(
            TOKEN_ORDER_SHAPES,
            {
                VIDEO_TOKEN_ORDER_1X8X8: (1, 8, 8),
                VIDEO_TOKEN_ORDER_1X16X4: (1, 16, 4),
                VIDEO_TOKEN_ORDER_4X4X4: (4, 4, 4),
                VIDEO_TOKEN_ORDER_RASTER: None,
            },
        )

    def test_clear_exposes_inner_embedding_wrapper_for_plan_rebuild(self):
        model = _model()
        patcher = _FakePatcher(model)
        first_embedding = lambda *args, **kwargs: None
        first_embedding._h3_optimizations_embedding_memory = True
        patcher.object_patches[FORWARD_KEY] = first_embedding

        self.assertTrue(install(patcher, (1, 8, 8)))
        self.assertIs(
            patcher.object_patches[FORWARD_KEY]._h3_cube_order_original,
            first_embedding,
        )
        self.assertTrue(clear(patcher))
        self.assertIs(patcher.object_patches[FORWARD_KEY], first_embedding)

        second_embedding = lambda *args, **kwargs: None
        second_embedding._h3_optimizations_embedding_memory = True
        patcher.object_patches[FORWARD_KEY] = second_embedding
        self.assertTrue(install(patcher, (4, 4, 4)))
        self.assertIs(
            patcher.object_patches[FORWARD_KEY]._h3_cube_order_original,
            second_embedding,
        )

    def test_clear_removes_wrapper_when_it_wraps_stock_forward(self):
        model = _model()
        patcher = _FakePatcher(model)
        self.assertTrue(install(patcher))
        self.assertFalse(install(patcher))
        with self.assertRaises(H3CubeOrderPatchError):
            install(patcher, (4, 4, 4))
        self.assertTrue(clear(patcher))
        self.assertNotIn(FORWARD_KEY, patcher.object_patches)
        self.assertFalse(clear(patcher))


if __name__ == "__main__":
    unittest.main()
