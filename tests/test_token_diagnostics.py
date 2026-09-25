'''Shape-only token diagnostic contracts for MiniMax H3 attention.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations import attention_forward  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class _Layout:
    def __init__(self):
        self.seq_len = 62
        self.signature = (7, 2, 8, 10, 3)
        self.segments = [
            (0, 7, 'text'),
            (7, 12, 'ref_img'),
            (12, 16, 'ref_audio'),
            (16, 22, 'audio'),
            (22, 62, 'video'),
        ]


class _Backend:
    name = attention_forward.DENSE_KITCHEN_PREQUANTIZED
    query_chunk_rows = 4096


class _Projector:
    name = 'chunked_kitchen_qkv'
    streamed_q = True
    chunk_rows = 4096


class _ShapeOnlyInput:
    shape = (62, 4)


def _render_log(call):
    return call.args[0]


class TokenDiagnosticTests(unittest.TestCase):
    def test_console_log_uses_direct_tqdm_write(self):
        with mock.patch.object(attention_forward.tqdm, 'write') as write:
            attention_forward._tqdm_info('diagnostic message\ndetail line')

        write.assert_called_once_with(
            '[INFO] diagnostic message\n[INFO] detail line'
        )

    def test_diagnostic_uses_shapes_and_metadata_without_cuda_sync(self):
        with (
            mock.patch.object(attention_forward.tqdm, 'write') as info,
            mock.patch.object(
                torch.cuda,
                'synchronize',
                side_effect=AssertionError('CUDA synchronization is not allowed'),
            ) as synchronize,
        ):
            attention_forward.log_token_diagnostic_once(
                _ShapeOnlyInput(),
                {'minimax_h3_layout': _Layout()},
                backend=_Backend(),
                projector=_Projector(),
            )

        info.assert_called_once()
        synchronize.assert_not_called()

    def test_layout_ranges_and_ref2va_decomposition(self):
        layout = _Layout()
        x = torch.empty(62, 4)

        with mock.patch.object(attention_forward.tqdm, 'write') as info:
            attention_forward.log_token_diagnostic_once(
                x,
                {'minimax_h3_layout': layout},
                backend=_Backend(),
                projector=_Projector(),
            )

        info.assert_called_once()
        message = _render_log(info.call_args)
        self.assertIn('packed=62 layout_total=62', message)
        self.assertIn(
            'text_tokens=7 conditioning_tokens=0 reference_tokens=9 '
            'audio_tokens=6 video_tokens=40',
            message,
        )
        self.assertIn('video_shape=(2,4,5)', message)
        self.assertIn('text_range=(0,7)', message)
        self.assertIn('conditioning_range=none', message)
        self.assertIn('audio_range=(16,22)', message)
        self.assertIn('video_range=(22,62)', message)
        self.assertIn('reference_segments=2', message)
        self.assertIn('ref_img[7:12]=5,ref_audio[12:16]=4', message)
        self.assertIn('route=dense backend=Comfy Kitchen INT8', message)
        self.assertIn('qkv_streaming=on q_chunk=4096', message)
        self.assertIn('component_total_without_conditioning=62', message)
        self.assertIn('component_total_with_conditioning=62 component_check=ok', message)

    def test_local_packed_layout_decomposes_image_references(self):
        layout = attention_forward.h3_model.PackedLayout(
            7,
            2,
            8,
            10,
            3,
            refs=[
                {'kind': 'image', 'latent_h': 8, 'latent_w': 10},
                {'kind': 'image', 'latent_h': 8, 'latent_w': 10},
            ],
        )

        with mock.patch.object(attention_forward.tqdm, 'write') as info:
            attention_forward.log_token_diagnostic_once(
                torch.empty(layout.seq_len, 4),
                {'minimax_h3_layout': layout},
            )

        message = _render_log(info.call_args)
        self.assertIn('reference_tokens=40', message)
        self.assertIn('reference_segments=2', message)
        self.assertIn('component_check=ok', message)

    def test_runtime_conditioning_is_included_in_component_check(self):
        layout = _Layout()
        layout.seq_len = 116780
        layout.signature = (7486, 102, 48, 84, 575)
        layout.segments = [
            (0, 7486, 'text'),
            (7486, 8494, 'cond'),
            (8494, 9358, 'ref_img'),
            (9358, 10222, 'ref_img'),
            (10222, 11086, 'ref_img'),
            (11086, 11950, 'ref_img'),
            (11950, 12814, 'ref_img'),
            (12814, 13964, 'audio'),
            (13964, 116780, 'video'),
        ]

        with mock.patch.object(attention_forward.tqdm, 'write') as info:
            attention_forward.log_token_diagnostic_once(
                torch.empty(116780, 1),
                {'minimax_h3_layout': layout},
                backend=_Backend(),
                projector=_Projector(),
            )

        info.assert_called_once()
        message = _render_log(info.call_args)
        self.assertIn(
            '[H3 Tokens] packed=116780 layout_total=116780 '
            'text_tokens=7486 conditioning_tokens=1008 reference_tokens=4320 '
            'audio_tokens=1150 video_tokens=102816 video_shape=(102,24,42) '
            'q_total=116780 route=dense backend=Comfy Kitchen INT8 '
            'qkv_streaming=on q_chunk=4096',
            message,
        )
        self.assertIn('component_total_without_conditioning=115772', message)
        self.assertIn(
            'component_total_with_conditioning=116780 component_check=ok',
            message,
        )
        self.assertIn(
            '[H3 Tokens Detail] text_range=(0,7486) '
            'conditioning_range=(7486,8494) reference_segments=5',
            message,
        )
        self.assertIn('audio_range=(12814,13964)', message)
        self.assertIn('video_range=(13964,116780)', message)

    def test_missing_layout_is_a_noop(self):
        with mock.patch.object(attention_forward.tqdm, 'write') as info:
            attention_forward.log_token_diagnostic_once(
                torch.empty(3, 2),
                {},
                backend=_Backend(),
                projector=_Projector(),
            )
        info.assert_not_called()

    def test_logs_once_per_layout_not_once_per_shape(self):
        first = _Layout()
        second = _Layout()
        first_state = vars(first).copy()
        x = torch.empty(62, 4)

        with mock.patch.object(attention_forward.tqdm, 'write') as info:
            for _ in range(3):
                attention_forward.log_token_diagnostic_once(
                    x,
                    {'minimax_h3_layout': first},
                    backend=_Backend(),
                    projector=_Projector(),
                )
            attention_forward.log_token_diagnostic_once(
                x,
                {'minimax_h3_layout': second},
                backend=_Backend(),
                projector=_Projector(),
            )

        self.assertEqual(info.call_count, 2)
        self.assertEqual(vars(first), first_state)

    def test_diagnostic_does_not_change_attention_output_or_call_count(self):
        module = mock.Mock()
        module.heads = 1
        module.head_dim = 2
        module.out_proj.side_effect = lambda value: value
        q = torch.arange(124, dtype=torch.float32).reshape(62, 1, 2)
        k = q + 100
        v = q + 200
        attention = mock.Mock(side_effect=lambda _q, _k, value, *_args, **_kwargs: value)
        forward = attention_forward.make_forward(module, 0, attention=attention)

        with (
            mock.patch.object(attention_forward, 'project_qkv', return_value=(q, k, v)),
            mock.patch.object(attention_forward.tqdm, 'write'),
        ):
            without_layout = forward(torch.empty(62, 2), transformer_options={})
            with_layout = forward(
                torch.empty(62, 2),
                transformer_options={'minimax_h3_layout': _Layout()},
            )

        self.assertEqual(attention.call_count, 2)
        self.assertTrue(torch.equal(without_layout, with_layout))

    def test_component_mismatch_is_logged_without_raising(self):
        layout = _Layout()
        layout.segments = [
            (0, 7, 'text'),
            (7, 10, 'cond'),
            (10, 15, 'ref_img'),
            (15, 19, 'ref_audio'),
            (19, 25, 'audio'),
            (25, 65, 'video'),
        ]
        layout.seq_len = 65

        with mock.patch.object(attention_forward.tqdm, 'write') as info:
            attention_forward.log_token_diagnostic_once(
                torch.empty(66, 4),
                {'minimax_h3_layout': layout},
            )

        message = _render_log(info.call_args)
        self.assertIn('packed=66 layout_total=65', message)
        self.assertIn('conditioning_tokens=3', message)
        self.assertIn('conditioning_range=(7,10)', message)
        self.assertIn('component_total_without_conditioning=62', message)
        self.assertIn(
            'component_total_with_conditioning=65 component_check=mismatch',
            message,
        )


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
