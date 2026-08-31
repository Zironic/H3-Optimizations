'''CPU tests for standalone sampler-step and packed-layout publication.'''

from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.runtime.context import (  # noqa: E402
    CLONE_CALLBACK_KEY,
    H3RuntimeSession,
    OUTER_WRAPPER_KEY,
    RUNTIME_KEY,
    RUNTIME_SESSION_KEY,
    WRAPPER_KEY,
    get_runtime_snapshot,
    install_runtime_wrapper,
    make_diffusion_wrapper,
    make_outer_wrapper,
)
from comfy.model_patcher import ModelPatcher  # noqa: E402
from comfy.patcher_extension import CallbacksMP, WrappersMP  # noqa: E402
from h3_optimizations.runtime.layout import resolve_layout  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class RuntimeTests(unittest.TestCase):
    @staticmethod
    def _patcher():
        model = torch.nn.Module()
        model.device = torch.device('cpu')
        return ModelPatcher(
            model,
            torch.device('cpu'),
            torch.device('cpu'),
        )

    def test_sampler_callback_owns_step_progress(self):
        layout = SimpleNamespace(seq_len=384)
        options = {'sample_sigmas': torch.empty((11,))}
        session = H3RuntimeSession(strict_layout=True)
        token = session.begin_request(10)
        try:
            with patch(
                'h3_optimizations.runtime.context.resolve_layout',
                return_value=layout,
            ):
                first = session.observe(
                    [torch.zeros((1, 1))],
                    torch.zeros((1, 1)),
                    options,
                    {},
                )
                repeated = session.observe(
                    [torch.zeros((1, 1))],
                    torch.zeros((1, 1)),
                    options,
                    {},
                )
                session.complete_step(0, 10)
                second = session.observe(
                    [torch.zeros((1, 1))],
                    torch.zeros((1, 1)),
                    options,
                    {},
                )
        finally:
            session.end_request(token)
        self.assertEqual(
            [
                first.step_index,
                repeated.step_index,
                second.step_index,
            ],
            [0, 0, 1],
        )
        self.assertEqual(
            [
                first.total_steps,
                repeated.total_steps,
                second.total_steps,
            ],
            [10, 10, 10],
        )

    def test_resolve_layout_rebuilds_current_packed_layout_for_odd_video(self):
        video = torch.zeros(1, 24, 3, 9, 11)
        audio = torch.zeros(1, 32, 2, 4)
        context = torch.zeros(1, 5, 8)
        payload = {"frame_count": 81}

        layout = resolve_layout([video, audio], context, payload)

        self.assertEqual(layout.text_range, (0, 5))
        self.assertEqual(layout.audio_range, (5, 13))
        self.assertEqual(layout.video_range, (13, 103))
        self.assertEqual(layout.video_shape, (3, 5, 6))
        self.assertEqual(layout.audio_t, 4)
        self.assertEqual(layout.seq_len, 103)
        self.assertEqual(payload, {"frame_count": 81})

    def test_outer_wrapper_preserves_and_extends_the_callback(self):
        session = H3RuntimeSession()
        observed = []

        def original_callback(step, _x0, _x, total_steps):
            observed.append((step, total_steps))

        def executor(
            _noise,
            _latent,
            _sampler,
            _sigmas,
            _mask,
            callback,
        ):
            self.assertEqual(session._step_index, 0)
            callback(0, None, None, 10)
            self.assertEqual(session._step_index, 1)
            return 'ok'

        result = make_outer_wrapper(session)(
            executor,
            None,
            None,
            None,
            torch.empty((11,)),
            None,
            original_callback,
        )
        self.assertEqual(result, 'ok')
        self.assertEqual(observed, [(0, 10)])
        self.assertEqual(session._step_index, -1)

    def test_session_publishes_only_package_owned_state(self):
        layout = SimpleNamespace(seq_len=384)
        options = {}
        session = H3RuntimeSession(strict_layout=True)
        with patch(
            'h3_optimizations.runtime.context.resolve_layout',
            return_value=layout,
        ):
            snapshot = session.observe(
                [torch.zeros((1, 1))],
                torch.zeros((1, 1)),
                options,
                {},
            )
        self.assertIs(snapshot.layout, layout)
        self.assertEqual(snapshot.step_index, -1)
        self.assertIs(options[RUNTIME_KEY], snapshot)
        self.assertIs(get_runtime_snapshot(options), snapshot)
        self.assertEqual(set(options), {'h3_optimizations_runtime'})

    def test_diffusion_wrapper_publishes_before_execution(self):
        layout = SimpleNamespace(seq_len=384)
        options = {}
        session = H3RuntimeSession(strict_layout=True)

        def executor(*_args, **_kwargs):
            self.assertIs(get_runtime_snapshot(options).layout, layout)
            return 'ok'

        wrapper = make_diffusion_wrapper(session)
        with patch(
            'h3_optimizations.runtime.context.resolve_layout',
            return_value=layout,
        ):
            result = wrapper(
                executor,
                [torch.zeros((1, 1))],
                torch.zeros((1,)),
                torch.zeros((1, 1)),
                options,
                minimax_payload={},
            )
        self.assertEqual(result, 'ok')

    def test_clone_reconstructs_runtime_session_and_keyed_wrappers(self):
        parent = self._patcher()
        parent_session = install_runtime_wrapper(
            parent,
            H3RuntimeSession(strict_layout=True),
        )

        child_a = parent.clone()
        child_b = parent.clone()
        child_a_session = child_a.model_options['transformer_options'][
            RUNTIME_SESSION_KEY
        ]
        child_b_session = child_b.model_options['transformer_options'][
            RUNTIME_SESSION_KEY
        ]

        self.assertIsNot(parent_session, child_a_session)
        self.assertIsNot(parent_session, child_b_session)
        self.assertIsNot(child_a_session, child_b_session)
        self.assertTrue(child_a_session.strict_layout)

        for child in (child_a, child_b):
            wrappers = child.model_options['transformer_options']['wrappers']
            self.assertEqual(
                len(wrappers[WrappersMP.OUTER_SAMPLE][OUTER_WRAPPER_KEY]),
                1,
            )
            self.assertEqual(
                len(wrappers[WrappersMP.DIFFUSION_MODEL][WRAPPER_KEY]),
                1,
            )
            self.assertEqual(
                len(child.callbacks[CallbacksMP.ON_CLONE][CLONE_CALLBACK_KEY]),
                1,
            )

    def test_sibling_clone_requests_can_overlap(self):
        parent = self._patcher()
        install_runtime_wrapper(parent, H3RuntimeSession(strict_layout=True))
        child_a = parent.clone()
        child_b = parent.clone()
        session_a = child_a.model_options['transformer_options'][
            RUNTIME_SESSION_KEY
        ]
        session_b = child_b.model_options['transformer_options'][
            RUNTIME_SESSION_KEY
        ]

        token_a = session_a.begin_request(10)
        try:
            token_b = session_b.begin_request(10)
            session_b.end_request(token_b)
        finally:
            session_a.end_request(token_a)

    def test_repeated_cloning_does_not_accumulate_runtime_hooks(self):
        patcher = self._patcher()
        install_runtime_wrapper(patcher, H3RuntimeSession(strict_layout=True))

        for _ in range(4):
            patcher = patcher.clone()
            wrappers = patcher.model_options['transformer_options']['wrappers']
            self.assertEqual(
                len(wrappers[WrappersMP.OUTER_SAMPLE][OUTER_WRAPPER_KEY]),
                1,
            )
            self.assertEqual(
                len(wrappers[WrappersMP.DIFFUSION_MODEL][WRAPPER_KEY]),
                1,
            )
            self.assertEqual(
                len(patcher.callbacks[CallbacksMP.ON_CLONE][CLONE_CALLBACK_KEY]),
                1,
            )

    def test_compile_change_repairs_the_runtime_wrapper_type(self):
        parent = self._patcher()
        install_runtime_wrapper(parent, H3RuntimeSession(strict_layout=True))
        child = parent.clone()
        session = child.model_options['transformer_options'][
            RUNTIME_SESSION_KEY
        ]
        child.model_options['torch_compile_kwargs'] = {'backend': 'inductor'}

        install_runtime_wrapper(child, session)

        wrappers = child.model_options['transformer_options']['wrappers']
        self.assertNotIn(
            WRAPPER_KEY,
            wrappers.get(WrappersMP.DIFFUSION_MODEL, {}),
        )
        self.assertEqual(
            len(wrappers[WrappersMP.APPLY_MODEL][WRAPPER_KEY]),
            1,
        )


if __name__ == '__main__':
    unittest.main()
