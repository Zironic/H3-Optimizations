"""CPU contracts for H3-owned cube-major token ordering."""

import inspect
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

from comfy.ldm import common_dit  # noqa: E402
from comfy.ldm.minimax.model import (  # noqa: E402
    MiniMaxH3Model,
    PackedLayout,
    patchify_video,
)
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
    pad_mask,
    reorder_video_patches,
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


def _packed_layout_supports(name):
    return name in inspect.signature(PackedLayout).parameters


def _compatible_keyframe(frame_count):
    # v0.33 only accepts first/last anchors; v0.34 generalized PackedLayout to
    # arbitrary keyframe positions. Exercise the broader contract where it
    # exists without feeding an impossible layout to the older release.
    return 2 if not _packed_layout_supports("frame_count") else frame_count - 1


def _packed_layout(*args, keyframes=None, refs=None, frame_count=None):
    kwargs = {"keyframes": keyframes, "refs": refs}
    if _packed_layout_supports("frame_count"):
        kwargs["frame_count"] = frame_count
    return PackedLayout(*args, **kwargs)


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

    def test_all_geometries_preserve_padding_masks_and_mixed_layout(self):
        patch_size = (1, 2, 2)
        video = _raster_video(5, 33, 17)
        audio = torch.arange(1 * 32 * 2 * 3, dtype=torch.float32).reshape(
            1, 32, 2, 3
        )
        context = torch.zeros(1, 3, 8)
        denoise_mask = torch.linspace(
            0.0,
            1.0,
            5 * 33 * 17,
            dtype=torch.float32,
        ).reshape(1, 1, 5, 33, 17)
        audio_denoise_mask = torch.tensor(
            [[[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]]],
            dtype=torch.float32,
        )
        padded_video = common_dit.pad_to_patch_size(video, patch_size)
        frame_count = 5
        keyframes = [{
            "resolved_frame_index": _compatible_keyframe(frame_count),
            "latent": torch.zeros(1, 24, 1, 34, 18),
            "audio_latent": torch.zeros(1, 32, 2, 2),
        }]
        refs = [
            {"kind": "image", "latent_h": 34, "latent_w": 18},
            {"kind": "audio", "ref_audio_t": 2},
            {
                "kind": "video_audio",
                "latent_t": 2,
                "latent_h": 34,
                "latent_w": 18,
                "ref_audio_t": 2,
            },
        ]
        base_layout = _packed_layout(
            context.shape[1],
            padded_video.shape[2],
            padded_video.shape[3],
            padded_video.shape[4],
            audio.shape[-1],
            keyframes=keyframes,
            refs=refs,
            frame_count=frame_count,
        )
        original_position_ids = base_layout.position_ids.clone()
        payload = {
            "layout": base_layout,
            "keyframes": keyframes,
            "refs": refs,
            "frame_count": frame_count,
            "sentinel": object(),
        }
        transformer_options = {"sentinel": object()}
        model = type("Model", (), {"patch_size": patch_size})()
        sentinel = object()

        for cube_shape in CUBE_SHAPES:
            with self.subTest(cube_shape=cube_shape):
                captured = {}

                def original(
                    x,
                    timestep,
                    actual_context,
                    actual_options,
                    minimax_payload=None,
                    denoise_mask=None,
                    audio_denoise_mask=None,
                    **kwargs,
                ):
                    captured["video"] = x[0]
                    captured["audio"] = x[1]
                    captured["context"] = actual_context
                    captured["options"] = actual_options
                    captured["layout"] = minimax_payload["layout"]
                    captured["denoise_mask"] = denoise_mask
                    captured["audio_denoise_mask"] = audio_denoise_mask
                    return (x[0].clone(), x[1], sentinel)

                output = make_forward(model, original, cube_shape)(
                    (video, audio),
                    torch.tensor([500.0]),
                    context,
                    transformer_options,
                    minimax_payload=payload,
                    denoise_mask=denoise_mask,
                    audio_denoise_mask=audio_denoise_mask,
                )

                self.assertIsInstance(output, tuple)
                self.assertTrue(torch.equal(output[0], video))
                self.assertIs(output[1], audio)
                self.assertIs(output[2], sentinel)
                self.assertIs(captured["audio"], audio)
                self.assertIs(captured["context"], context)
                self.assertIs(captured["options"], transformer_options)
                self.assertIs(
                    captured["audio_denoise_mask"],
                    audio_denoise_mask,
                )
                self.assertEqual(captured["video"].shape, padded_video.shape)
                self.assertEqual(captured["video"].dtype, video.dtype)

                ordered_layout = captured["layout"]
                self.assertIsNot(ordered_layout, base_layout)
                video_start, video_stop, _kind = next(
                    segment
                    for segment in ordered_layout.segments
                    if segment[2] == "video"
                )
                order, _inverse = tile_aligned_cube_major_indices(
                    (5, 17, 9),
                    video_start,
                    cube_shape,
                )
                expected_rows = patchify_video(padded_video, patch_size)[
                    list(order)
                ]
                self.assertTrue(torch.equal(
                    patchify_video(captured["video"], patch_size),
                    expected_rows,
                ))
                expected_mask = reorder_video_patches(
                    pad_mask(denoise_mask, padded_video.shape),
                    order,
                    patch_size,
                )
                self.assertTrue(torch.equal(
                    captured["denoise_mask"],
                    expected_mask,
                ))
                self.assertTrue(torch.equal(
                    ordered_layout.position_ids[:video_start],
                    original_position_ids[:video_start],
                ))
                self.assertTrue(torch.equal(
                    ordered_layout.position_ids[video_start:video_stop],
                    original_position_ids[video_start:video_stop][list(order)],
                ))
                self.assertEqual(
                    ordered_layout.h3_cube_order["cube_shape"],
                    cube_shape,
                )
                self.assertEqual(
                    ordered_layout.h3_cube_order["grid_shape"],
                    (5, 17, 9),
                )
                self.assertIs(payload["layout"], base_layout)
                self.assertTrue(torch.equal(
                    base_layout.position_ids,
                    original_position_ids,
                ))

    def test_layout_fallback_preserves_supported_keyframe_metadata(self):
        patch_size = (1, 2, 2)
        video = _raster_video(5, 34, 18)
        audio = torch.zeros(1, 32, 2, 3)
        context = torch.zeros(1, 3, 8)
        frame_count = 5
        keyframe_index = _compatible_keyframe(frame_count)
        keyframes = [{
            "resolved_frame_index": keyframe_index,
            "latent": torch.zeros(1, 24, 1, 34, 18),
            "audio_latent": torch.zeros(1, 32, 2, 2),
        }]
        captured = {}
        model = type("Model", (), {"patch_size": patch_size})()

        def original(x, timestep, actual_context, actual_options, minimax_payload=None, **kwargs):
            captured["layout"] = minimax_payload["layout"]
            return [x[0].clone(), x[1]]

        output = make_forward(model, original)(
            [video, audio],
            torch.tensor([500.0]),
            context,
            {},
            minimax_payload={
                "keyframes": keyframes,
                "frame_count": frame_count,
            },
        )

        self.assertTrue(torch.equal(output[0], video))
        layout = captured["layout"]
        cond_start, _cond_stop, _kind = next(
            segment for segment in layout.segments if segment[2] == "cond"
        )
        expected = _packed_layout(
            context.shape[1],
            video.shape[2],
            video.shape[3],
            video.shape[4],
            audio.shape[-1],
            keyframes=keyframes,
            frame_count=frame_count,
        )
        self.assertTrue(torch.equal(
            layout.position_ids[cond_start],
            expected.position_ids[cond_start],
        ))

    def test_short_unalignable_grid_fails_before_downstream_forward(self):
        model = type("Model", (), {"patch_size": (1, 2, 2)})()
        called = False

        def original(*args, **kwargs):
            nonlocal called
            called = True
            return args[0]

        wrapped = make_forward(model, original)
        with self.assertRaisesRegex(
            ValueError,
            "video grid cannot satisfy the router alignment prefix",
        ):
            wrapped(
                [_raster_video(1, 3, 3), torch.zeros(1, 32, 2, 1)],
                torch.tensor([500.0]),
                torch.zeros(1, 3, 8),
                {},
                minimax_payload={},
            )
        self.assertFalse(called)

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
