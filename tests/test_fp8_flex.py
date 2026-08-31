'''CPU-only contracts for the FP8 FlexAttention sparse fallback.'''

from copy import deepcopy
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
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.apply as apply_module  # noqa: E402
import h3_optimizations.attention_forward as attention_forward  # noqa: E402
from h3_optimizations.attention import (  # noqa: E402
    AttentionBackendUnavailable,
)
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
)
from h3_optimizations.attention.sparse.fp8_flex import (  # noqa: E402
    FLEX_BACKEND_FLASH,
    FLEX_BACKEND_TRITON,
    FP8FlexBackend,
    FP8FlexError,
    FP8FlexSpec,
    block_mask_from_delta_lut,
    load_fp8_flex_spec,
    preflight_fp8_flex,
    select_flex_kernel_backend,
)
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    MemoryRequest,
    SparseRequest,
    STATUS_KEY,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLPProviderResolution,
    QKVProviderResolution,
    QKV_STANDARD,
)
from h3_optimizations.status import format_sparse_status  # noqa: E402
from torch.nn.attention.flex_attention import BlockMask  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeModel:
    def __init__(self, options=None):
        self.model_options = deepcopy(options or {})
        self.object_patches = {}
        self.callbacks = {}
        self.wrappers = {}

    def clone(self):
        cloned = FakeModel(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        return cloned

    def remove_callbacks_with_key(self, call_type, key):
        self.callbacks.get(call_type, {}).pop(key, None)

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {})[key] = [callback]

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {})[key] = [wrapper]


class FakeMetadata:
    def as_dict(self):
        return {'requested_video_budget': 0.5}


class FakeRouter:
    q_tile = 64
    kv_tile = 64

    def build_lut(self, q, _k, _layout, _video_budget, *, sink=None):
        del sink
        q_tiles = (q.shape[-2] + self.q_tile - 1) // self.q_tile
        kv_tiles = (q.shape[-2] + self.kv_tile - 1) // self.kv_tile
        dense_delta = torch.ones(kv_tiles, dtype=torch.int32)
        dense_delta[0] = 0
        lut = dense_delta.view(1, 1, 1, -1).expand(
            q.shape[0],
            q.shape[1],
            q_tiles,
            -1,
        ).clone()
        valid = torch.full(
            (q.shape[0], q.shape[1], q_tiles),
            kv_tiles,
            dtype=torch.int32,
        )
        return lut, valid, FakeMetadata()


class FP8FlexTests(unittest.TestCase):
    def setUp(self):
        cube_install = mock.patch.object(
            apply_module,
            'install_cube_order',
            return_value=True,
        )
        cube_install.start()
        self.addCleanup(cube_install.stop)

    @staticmethod
    def _spec(
        attention=lambda *_args, **_kwargs: None,
        kernel_backend=FLEX_BACKEND_TRITON,
    ):
        return FP8FlexSpec(
            version='test-flex',
            attention=attention,
            block_mask_type=BlockMask,
            kernel_backend=kernel_backend,
        )

    def test_preflight_requires_cuda_fp8_and_dynamo(self):
        with self.assertRaisesRegex(FP8FlexError, 'requires NVIDIA CUDA'):
            preflight_fp8_flex(
                cuda_available=lambda: False,
                capability_getter=lambda: None,
            )

        with self.assertRaisesRegex(FP8FlexError, 'unsupported'):
            preflight_fp8_flex(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 0),
                fp8_supported=lambda: False,
            )

        with self.assertRaisesRegex(FP8FlexError, 'Dynamo'):
            preflight_fp8_flex(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 9),
                fp8_supported=lambda: True,
                dynamo_supported=lambda: False,
            )

        spec = self._spec(kernel_backend=FLEX_BACKEND_FLASH)
        selected = []
        self.assertIs(
            preflight_fp8_flex(
                cuda_available=lambda: True,
                capability_getter=lambda: (12, 0),
                fp8_supported=lambda: True,
                dynamo_supported=lambda: True,
                flash_available=lambda: True,
                loader=lambda backend: selected.append(backend) or spec,
            ),
            spec,
        )
        self.assertEqual(selected, [FLEX_BACKEND_FLASH])

    def test_kernel_backend_uses_flash_only_with_complete_eligible_runtime(self):
        self.assertEqual(
            select_flex_kernel_backend(
                (12, 0),
                flash_available=lambda: True,
            ),
            FLEX_BACKEND_FLASH,
        )
        self.assertEqual(
            select_flex_kernel_backend(
                (8, 9),
                flash_available=lambda: True,
            ),
            FLEX_BACKEND_TRITON,
        )
        self.assertEqual(
            select_flex_kernel_backend(
                (12, 0),
                flash_available=lambda: False,
            ),
            FLEX_BACKEND_TRITON,
        )

    def test_loader_compiles_flex_attention_for_sparse_execution(self):
        compiled_attention = object()
        with mock.patch.object(
            torch,
            'compile',
            return_value=compiled_attention,
        ) as compile_attention:
            spec = load_fp8_flex_spec(FLEX_BACKEND_FLASH)

        self.assertIs(spec.attention, compiled_attention)
        self.assertEqual(spec.kernel_backend, FLEX_BACKEND_FLASH)
        compile_attention.assert_called_once_with(
            mock.ANY,
            fullgraph=True,
        )

    def test_delta_lut_becomes_compact_flex_block_indices(self):
        lut = torch.tensor(
            [[[[0, 2, 1], [0, 1, 1], [0, 1, 1]]]],
            dtype=torch.int32,
        )
        valid = torch.tensor([[[2, 3, 3]]], dtype=torch.int32)

        block_mask = block_mask_from_delta_lut(
            self._spec(),
            lut,
            valid,
            192,
        )

        self.assertEqual(block_mask.BLOCK_SIZE, (64, 64))
        self.assertEqual(block_mask.seq_lengths, (192, 192))
        self.assertTrue(torch.equal(block_mask.kv_num_blocks, valid))
        self.assertEqual(
            block_mask.kv_indices.tolist(),
            [[[[0, 2, 3], [0, 1, 2], [0, 1, 2]]]],
        )
        self.assertIsNone(block_mask.q_indices)

    def test_backend_quantizes_all_qkv_and_restores_output_dtype(self):
        calls = []

        def attention(q, k, v, **kwargs):
            calls.append((q, k, v, kwargs))
            return torch.ones_like(q)

        backend = FP8FlexBackend(
            HybridSparseConfig(video_budget=0.5),
            spec=self._spec(attention),
            router=FakeRouter(),
            chunk_rows=128,
            allow_cpu_for_tests=True,
        )
        q = torch.empty((1, 2, 192, 128), dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        q[:, 0].fill_(1)
        q[:, 1].fill_(2)
        k[:, 0].fill_(2)
        k[:, 1].fill_(4)
        v[:, 0].fill_(3)
        v[:, 1].fill_(6)
        snapshot = SimpleNamespace(
            valid_layout=True,
            error=None,
            layout=SimpleNamespace(seq_len=192),
            step_index=4,
            total_steps=20,
        )

        with mock.patch.object(
            backend,
            '_snapshot',
            return_value=snapshot,
        ):
            prepared = backend.prepare(
                q,
                k,
                v,
                layer_index=7,
                transformer_options={},
            )

        self.assertEqual(prepared.q_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(prepared.k_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(prepared.v_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(prepared.q_fp8.stride(-1), 1)
        self.assertEqual(prepared.k_fp8.stride(-1), 1)
        self.assertEqual(prepared.v_fp8.stride(-2), 1)
        torch.testing.assert_close(
            prepared.qk_scale,
            torch.tensor(
                [[2 / (448 ** 2), 8 / (448 ** 2)]],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            prepared.v_scale,
            torch.tensor([[3 / 448, 6 / 448]], dtype=torch.float32),
        )

        output = backend.execute(prepared)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertEqual(len(calls), 1)
        call_q, call_k, call_v, kwargs = calls[0]
        self.assertIs(call_q, prepared.q_fp8)
        self.assertIs(call_k, prepared.k_fp8)
        self.assertIs(call_v, prepared.v_fp8)
        self.assertIs(kwargs['block_mask'], prepared.block_mask)
        self.assertEqual(
            kwargs['kernel_options']['BACKEND'],
            FLEX_BACKEND_TRITON,
        )
        self.assertEqual(kwargs['kernel_options']['BLOCK_M'], 64)
        self.assertEqual(kwargs['kernel_options']['BLOCK_N'], 64)
        self.assertTrue(
            kwargs['kernel_options']['ROWS_GUARANTEED_SAFE']
        )
        self.assertIs(
            kwargs['score_mod'].__closure__[0].cell_contents,
            prepared.qk_scale,
        )
        restored = kwargs['score_mod'](
            torch.tensor(2.0, dtype=torch.float8_e4m3fn),
            torch.tensor(0),
            torch.tensor(1),
            torch.tensor(0),
            torch.tensor(0),
        )
        self.assertEqual(restored.dtype, torch.float32)
        torch.testing.assert_close(restored, prepared.qk_scale[0, 1] * 2)
        torch.testing.assert_close(
            output[:, 0].float(),
            torch.full_like(output[:, 0].float(), 3 / 448),
            atol=2e-5,
            rtol=2e-3,
        )

    def test_first_execution_failure_retires_only_the_unvalidated_signature(self):
        calls = 0

        def attention(q, _k, _v, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError('synthetic lowering failure')
            return torch.ones_like(q)

        backend = FP8FlexBackend(
            spec=self._spec(attention),
            router=FakeRouter(),
            chunk_rows=128,
            allow_cpu_for_tests=True,
        )
        q = torch.ones((1, 1, 128, 128), dtype=torch.bfloat16)
        snapshot = SimpleNamespace(
            valid_layout=True,
            error=None,
            layout=SimpleNamespace(seq_len=128),
            step_index=0,
            total_steps=1,
        )
        with mock.patch.object(backend, '_snapshot', return_value=snapshot):
            prepared = backend.prepare(
                q,
                q,
                q,
                layer_index=0,
                transformer_options={},
            )

        with self.assertRaisesRegex(
            AttentionBackendUnavailable,
            'synthetic lowering failure',
        ):
            backend.execute(prepared)
        with self.assertRaises(AttentionBackendUnavailable):
            backend.prepare(
                q,
                q,
                q,
                layer_index=1,
                transformer_options={},
            )
        self.assertEqual(calls, 1)

    def test_failure_after_successful_validation_remains_hard(self):
        calls = 0

        def attention(q, _k, _v, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('validated runtime failure')
            return torch.ones_like(q)

        backend = FP8FlexBackend(
            spec=self._spec(attention),
            router=FakeRouter(),
            chunk_rows=128,
            allow_cpu_for_tests=True,
        )
        q = torch.ones((1, 1, 128, 128), dtype=torch.bfloat16)
        snapshot = SimpleNamespace(
            valid_layout=True,
            error=None,
            layout=SimpleNamespace(seq_len=128),
            step_index=0,
            total_steps=1,
        )
        with mock.patch.object(backend, '_snapshot', return_value=snapshot):
            first = backend.prepare(
                q,
                q,
                q,
                layer_index=0,
                transformer_options={},
            )
            self.assertTrue(backend.requires_fallback_inputs(first))
            backend.execute(first)
            second = backend.prepare(
                q,
                q,
                q,
                layer_index=1,
                transformer_options={},
            )

        self.assertFalse(backend.requires_fallback_inputs(second))
        with self.assertRaisesRegex(RuntimeError, 'validated runtime failure'):
            backend.execute(second)

    def test_attention_forward_uses_dense_qkv_when_backend_is_unavailable(self):
        class Backend:
            name = 'synthetic'

            @staticmethod
            def prepare(*_args, **_kwargs):
                raise AttentionBackendUnavailable('not compiled')

        q = torch.randn((3, 2, 4), dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        dense_calls = []

        def dense(_module, call_q, call_k, call_v, options, attention=None):
            dense_calls.append((call_q, call_k, call_v, options, attention))
            return torch.ones((1, 2, 3, 4), dtype=torch.bfloat16)

        module = SimpleNamespace(
            heads=2,
            head_dim=4,
            out_proj=lambda value: value,
        )
        forward = attention_forward.make_forward(
            module,
            0,
            backend=Backend(),
            backend_fallback_to_dense=True,
        )
        options = {'marker': True}
        with mock.patch.object(
            attention_forward,
            'project_qkv',
            return_value=(q, k, v),
        ), mock.patch.object(
            attention_forward,
            '_legacy_attention',
            side_effect=dense,
        ):
            output = forward(object(), transformer_options=options)

        self.assertEqual(tuple(output.shape), (3, 8))
        self.assertEqual(len(dense_calls), 1)
        self.assertTrue(dense_calls[0][2].is_contiguous())
        self.assertIs(dense_calls[0][3], options)

    def test_selection_uses_flex_after_sparse_sage_and_triton_fail(self):
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        dense_qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'standard projection',
        )
        dense = apply_module.ResolvedAttention(
            requested='existing',
            selected='existing',
            backend=None,
            reason='normal Comfy attention',
            backend_kind='existing',
            dense_resolution=SimpleNamespace(backend=None),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(12, 0),
            device_index=0,
        )
        inventory = SimpleNamespace(qkv=())

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('native missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('ABI missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            side_effect=apply_module.TritonSparseError('Triton missing'),
        ), mock.patch.object(
            apply_module,
            'preflight_fp8_flex',
            return_value=self._spec(),
        ):
            attention, qkv = apply_module._resolve_attention(
                plan,
                object(),
                inventory,
                environment,
            )

        self.assertEqual(attention.selected, 'flex_attention_fp8')
        self.assertEqual(attention.requested, 'sparse_sage')
        self.assertIs(attention.dense_resolution, dense.dense_resolution)
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertIn('ABI missing', attention.reason)
        self.assertIn('Triton missing', attention.reason)

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('native missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('ABI missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            side_effect=apply_module.TritonSparseError('Triton missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
            side_effect=apply_module.FP8FlexError('FP8 missing'),
        ):
            attention, qkv = apply_module._resolve_attention(
                plan,
                object(),
                inventory,
                environment,
            )

        self.assertEqual(attention.selected, 'existing')
        self.assertIs(qkv, dense_qkv)
        self.assertIn('Triton missing', attention.reason)
        self.assertIn('FP8 missing', attention.reason)

    def test_sparse_sage_success_does_not_probe_other_fallbacks(self):
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        resolved = (
            apply_module.ResolvedAttention(
                requested='sparse_sage',
                selected='sparse_sage',
                backend=object(),
                reason='Sparse Sage selected',
                backend_kind='sparse_sage',
            ),
            QKVProviderResolution(
                QKV_STANDARD,
                False,
                'standard projection',
            ),
        )

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=resolved,
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('native missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            return_value=resolved,
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton_sparse, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            actual = apply_module._resolve_attention(
                plan,
                object(),
                object(),
                object(),
            )

        self.assertIs(actual, resolved)
        triton_sparse.assert_not_called()
        flex.assert_not_called()

    def test_apply_installs_flex_as_the_sparse_execution_backend(self):
        plan = H3OptimizationPlan(
            memory=MemoryRequest(),
            sparse=SparseRequest(),
        )
        qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'standard projection',
        )
        mlp = MLPProviderResolution('off', 'off', 'disabled')
        backend = SimpleNamespace(name='flex_attention_fp8')
        dense_resolution = SimpleNamespace(backend=object())
        attention = apply_module.ResolvedAttention(
            requested='sparse_sage',
            selected='flex_attention_fp8',
            backend=backend,
            reason='Sparse Sage unavailable; using FP8 FlexAttention',
            backend_kind='flex_attention_fp8',
            dense_resolution=dense_resolution,
        )
        environment = SimpleNamespace(
            cuda_available=True,
            device_index=0,
            capability=(12, 0),
            device_name='fake NVIDIA',
            backend='nvidia_cuda',
            architecture='sm120',
        )
        inventory = SimpleNamespace(labels=lambda _name: ())

        with mock.patch.object(
            apply_module,
            'is_minimax_h3',
            return_value=True,
        ), mock.patch.object(
            apply_module,
            'get_h3_blocks',
            return_value=(object(),),
        ), mock.patch.object(
            apply_module,
            'inspect_h3_linears',
            return_value=inventory,
        ), mock.patch.object(
            apply_module.RuntimeEnvironment,
            'detect',
            return_value=environment,
        ), mock.patch.object(
            apply_module,
            '_resolve_attention',
            return_value=(attention, qkv),
        ), mock.patch.object(
            apply_module,
            'configure_backend',
            return_value=(backend, 50),
        ) as configure, mock.patch.object(
            apply_module,
            '_install_mlp',
            return_value=(mlp, 0),
        ), mock.patch.object(
            apply_module,
            '_ensure_sparse_runtime',
            return_value=(object(), True),
        ) as runtime:
            with mock.patch.object(
                apply_module,
                'install_dense_attention',
                return_value=True,
            ) as install_dense:
                patched = apply_module.apply_plan(FakeModel(), plan)

        configure.assert_called_once_with(
            mock.ANY,
            backend,
            projector=None,
            backend_fallback_to_dense=True,
        )
        install_dense.assert_called_once_with(patched, dense_resolution)
        runtime.assert_called_once()
        status = patched.model_options['transformer_options'][STATUS_KEY]
        self.assertEqual(status['attention']['selected'], 'flex_attention_fp8')
        self.assertTrue(status['runtime_installed'])
        self.assertIn('Attention: FP8 FlexAttention', format_sparse_status(patched))


if __name__ == '__main__':
    unittest.main()
