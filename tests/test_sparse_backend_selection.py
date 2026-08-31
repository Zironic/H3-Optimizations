'''CPU contracts for explicit sparse backend selection.'''

from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.apply_policy as apply_policy  # noqa: E402
apply_module = apply_policy._base
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    SparseKitchenError,
)
from h3_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_OFF,
    H3OptimizationPlan,
    MemoryRequest,
    PLAN_KEY,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_FROST,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_KITCHEN_64X128,
    SPARSE_BACKEND_TRITON,
    STATUS_KEY,
    SparseRequest,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLPProviderResolution,
    QKV_FORCE_BF16_STREAMED_KITCHEN,
    QKVProviderResolution,
    QKV_STREAMED_BF16_KITCHEN,
)
from h3_optimizations.qkv.policy import (  # noqa: E402
    resolve_qkv_provider as resolve_policy_qkv_provider,
)
from h3_optimizations.status import format_sparse_status  # noqa: E402

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


def qkv_resolution():
    return QKVProviderResolution(
        'standard_h3_qkv',
        False,
        'synthetic',
    )


def resolved(kind, dense_resolution=None):
    return apply_module.ResolvedAttention(
        requested=apply_module.ATTENTION_SPARSE,
        selected=kind,
        backend=SimpleNamespace(name=kind),
        reason='synthetic',
        backend_kind=kind,
        projector=None,
        dense_resolution=dense_resolution,
    )


class SparseBackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.inventory = object()
        self.environment = object()
        self.qkv = qkv_resolution()
        cube_install = mock.patch.object(
            apply_module,
            'install_cube_order',
            return_value=True,
        )
        cube_install.start()
        self.addCleanup(cube_install.stop)

    def test_forced_sparse_sage_does_not_fall_through(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_SAGE)
        )
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('synthetic unavailable'),
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            with self.assertRaisesRegex(
                apply_module.SparseSageError,
                'synthetic unavailable',
            ):
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                )
        dense.assert_not_called()
        sage.assert_called_once_with(plan, self.environment, self.inventory)
        triton.assert_not_called()
        flex.assert_not_called()

    def test_forced_triton_bypasses_sage_flex_and_dense(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_TRITON)
        )
        target = (resolved(apply_module.ATTENTION_TRITON_SPARSE), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            return_value=target,
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        dense.assert_not_called()
        sage.assert_not_called()
        triton.assert_called_once_with(
            plan,
            self.environment,
            self.inventory,
            None,
        )
        flex.assert_not_called()

    def test_forced_kitchen_bypasses_every_other_backend(self):
        """Explicit Kitchen remains a hard requirement without fallbacks."""
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_KITCHEN)
        )
        target = (resolved(apply_module.ATTENTION_KITCHEN_SPARSE), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex, mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            return_value=target,
        ) as kitchen:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        dense.assert_not_called()
        sage.assert_not_called()
        triton.assert_not_called()
        flex.assert_not_called()
        kitchen.assert_called_once_with(
            plan,
            self.environment,
            self.inventory,
        )

    def test_forced_frost_is_a_hard_bf16_backend_selection(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_FROST)
        )
        target = (resolved(apply_module.ATTENTION_FROST_BF16), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex, mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
        ) as kitchen, mock.patch.object(
            apply_module,
            '_resolve_frost_bf16',
            return_value=target,
        ) as frost:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        dense.assert_not_called()
        sage.assert_not_called()
        triton.assert_not_called()
        flex.assert_not_called()
        kitchen.assert_not_called()
        frost.assert_called_once_with(plan, self.environment, self.inventory)

    def test_kitchen_resolver_selects_streamed_sparse_producer(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(
                backend=SPARSE_BACKEND_KITCHEN,
            )
        )
        inventory = SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=True,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=False,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
        )
        kitchen = SimpleNamespace(__version__='test')

        with mock.patch.object(
            apply_module,
            'preflight_sparse_kitchen',
            return_value=kitchen,
        ) as preflight, mock.patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, qkv = apply_module._resolve_kitchen_sparse(
                plan,
                environment,
                inventory,
            )

        self.assertEqual(qkv.provider_id, 'streamed_bf16_kitchen_qkv')
        self.assertTrue(qkv.fused)
        self.assertTrue(attention.projector.routing_summaries)
        self.assertTrue(attention.projector.stream_output)
        self.assertTrue(attention.backend.stream_output)
        self.assertIs(attention.backend.projector, attention.projector)
        self.assertEqual(
            (attention.projector.q_tile, attention.projector.kv_tile),
            (64, 64),
        )
        self.assertEqual(
            (
                attention.backend.executor.q_tile,
                attention.backend.executor.kv_tile,
            ),
            (64, 64),
        )
        self.assertEqual(
            (
                preflight.call_args.kwargs['q_tile'],
                preflight.call_args.kwargs['kv_tile'],
            ),
            (64, 64),
        )

    def test_rectangular_kitchen_request_reaches_64x128_resolver(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_KITCHEN_64X128)
        )
        target = (resolved('sparse_kitchen_int8'), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            return_value=target,
        ) as kitchen:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        kitchen.assert_called_once_with(
            plan,
            self.environment,
            self.inventory,
            q_tile=64,
            kv_tile=128,
        )

    def test_preserve_precision_defaults_to_streamed_sparse_kitchen(self):
        plan = H3OptimizationPlan(
            memory=MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_OFF,
            ),
            sparse=SparseRequest(backend=SPARSE_BACKEND_KITCHEN),
        )
        inventory = SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=True,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=False,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
        )

        with mock.patch.object(
            apply_module,
            'preflight_sparse_kitchen',
            return_value=SimpleNamespace(__version__='test'),
        ), mock.patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, qkv = apply_module._resolve_kitchen_sparse(
                plan,
                environment,
                inventory,
            )

        self.assertEqual(qkv.provider_id, QKV_STREAMED_BF16_KITCHEN)
        self.assertTrue(qkv.fused)
        self.assertEqual(
            (attention.projector.q_tile, attention.projector.kv_tile),
            (64, 64),
        )
        self.assertTrue(attention.projector.strided_qk_input)
        self.assertTrue(attention.projector.stream_output)
        self.assertTrue(attention.backend.stream_output)

    def test_forced_bf16_sparse_kitchen_streams_projected_chunks(self):
        plan = H3OptimizationPlan(
            memory=MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_FORCE_BF16,
            ),
            sparse=SparseRequest(backend=SPARSE_BACKEND_KITCHEN),
        )
        inventory = SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=False,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=True,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('Parameter:torch.bfloat16',),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
        )

        with mock.patch.object(
            apply_module,
            'preflight_sparse_kitchen',
            return_value=SimpleNamespace(__version__='test'),
        ), mock.patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ), mock.patch.object(
            apply_module,
            'resolve_qkv_provider',
            resolve_policy_qkv_provider,
        ):
            attention, qkv = apply_module._resolve_kitchen_sparse(
                plan,
                environment,
                inventory,
            )

        self.assertEqual(
            qkv.provider_id,
            QKV_FORCE_BF16_STREAMED_KITCHEN,
        )
        self.assertTrue(attention.projector.force_weights_bf16)
        self.assertTrue(attention.projector.routing_summaries)
        self.assertTrue(attention.projector.stream_output)
        self.assertTrue(attention.backend.stream_output)

    def test_auto_prefers_kitchen(self):
        plan = H3OptimizationPlan(sparse=SparseRequest())
        target = (resolved(apply_module.ATTENTION_KITCHEN_SPARSE), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(resolved('dense'), self.qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex, mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            return_value=target,
        ) as kitchen:
            actual = apply_module._resolve_attention(
                plan,
                self.model,
                self.inventory,
                self.environment,
            )
        self.assertIs(actual, target)
        kitchen.assert_called_once_with(plan, self.environment, self.inventory)
        sage.assert_not_called()
        triton.assert_not_called()
        flex.assert_not_called()

    def test_auto_uses_sparse_sage_after_kitchen_failure(self):
        plan = H3OptimizationPlan(sparse=SparseRequest())
        target = (resolved(apply_module.ATTENTION_SPARSE), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(resolved('dense'), self.qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=SparseKitchenError('native self-test failed'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            return_value=target,
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton:
            actual = apply_module._resolve_attention(
                plan,
                self.model,
                self.inventory,
                self.environment,
            )
        self.assertIs(actual, target)
        sage.assert_called_once_with(plan, self.environment, self.inventory)
        triton.assert_not_called()

    def test_forced_flex_bypasses_sage_triton_and_dense(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_FLEX)
        )
        target = (resolved(apply_module.ATTENTION_FP8_FLEX), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
            return_value=target,
        ) as flex:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        dense.assert_not_called()
        sage.assert_not_called()
        triton.assert_not_called()
        flex.assert_called_once_with(
            plan,
            self.environment,
            self.inventory,
            None,
            None,
        )

    def test_only_auto_enables_flex_dense_runtime_fallback(self):
        inventory = SimpleNamespace(labels=lambda _name: ())
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
            device_name='fake',
            backend='nvidia_cuda',
            architecture='sm89',
        )
        mlp = MLPProviderResolution('off', 'off', 'synthetic')
        dense_resolution = object()
        attention = resolved(
            apply_module.ATTENTION_FP8_FLEX,
            dense_resolution=dense_resolution,
        )

        for request, expected_fallback in (
            (SPARSE_BACKEND_AUTO, True),
            (SPARSE_BACKEND_FLEX, False),
        ):
            with self.subTest(request=request):
                plan = H3OptimizationPlan(
                    sparse=SparseRequest(backend=request)
                )
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
                    return_value=(attention, self.qkv),
                ), mock.patch.object(
                    apply_module,
                    'configure_backend',
                    return_value=(object(), 50),
                ) as configure, mock.patch.object(
                    apply_module,
                    'install_dense_attention',
                    return_value=True,
                ) as install_dense, mock.patch.object(
                    apply_module,
                    '_install_mlp',
                    return_value=(mlp, 0),
                ), mock.patch.object(
                    apply_module,
                    '_ensure_sparse_runtime',
                    return_value=(object(), True),
                ):
                    apply_module.apply_plan(FakeModel(), plan)

                self.assertEqual(
                    configure.call_args.kwargs['backend_fallback_to_dense'],
                    expected_fallback,
                )
                if expected_fallback:
                    install_dense.assert_called_once_with(
                        mock.ANY,
                        dense_resolution,
                    )
                else:
                    install_dense.assert_not_called()

    def test_kitchen_sparse_is_installed_as_sparse_execution(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_KITCHEN)
        )
        inventory = SimpleNamespace(labels=lambda _name: ())
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
            device_name='fake',
            backend='nvidia_cuda',
            architecture='sm89',
        )
        mlp = MLPProviderResolution('off', 'off', 'synthetic')
        projector = object()
        attention = apply_module.ResolvedAttention(
            requested=apply_module.ATTENTION_KITCHEN_SPARSE,
            selected=apply_module.ATTENTION_KITCHEN_SPARSE,
            backend=SimpleNamespace(name=apply_module.ATTENTION_KITCHEN_SPARSE),
            reason='synthetic',
            backend_kind=apply_module.ATTENTION_KITCHEN_SPARSE,
            projector=projector,
        )

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
            return_value=(attention, self.qkv),
        ), mock.patch.object(
            apply_module,
            'configure_backend',
            return_value=(object(), 50),
        ) as configure, mock.patch.object(
            apply_module,
            '_install_mlp',
            return_value=(mlp, 0),
        ), mock.patch.object(
            apply_module,
            '_ensure_sparse_runtime',
            return_value=(object(), True),
        ) as ensure_runtime:
            apply_module.apply_plan(FakeModel(), plan)

        configure.assert_called_once_with(
            mock.ANY,
            attention.backend,
            projector=projector,
        )
        ensure_runtime.assert_called_once()

    def test_explicit_backend_status_is_not_called_a_fallback(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(
                backend=SPARSE_BACKEND_TRITON,
                early_steps=2,
                early_kv=0.5,
                late_steps=2,
                late_kv=0.5,
            )
        )
        model = FakeModel(
            {
                PLAN_KEY: plan,
                'transformer_options': {
                    STATUS_KEY: {
                        'attention': {
                            'selected': apply_module.ATTENTION_TRITON_SPARSE,
                            'reason': 'explicit selection',
                        },
                        'sparse': {
                            'backend': SPARSE_BACKEND_TRITON,
                            'video_budget': 0.3,
                            'early_steps': 2,
                            'early_kv': 0.5,
                            'late_steps': 2,
                            'late_kv': 0.5,
                        },
                        'fused_qkv': {
                            'provider': 'standard_h3_qkv',
                            'reason': 'synthetic',
                        },
                        'mlp': {'provider': 'off'},
                    }
                },
            }
        )
        text = format_sparse_status(model)
        self.assertIn('Requested sparse backend: BF16 Triton', text)
        self.assertNotIn('Sparse fallback:', text)

    def test_frost_status_uses_the_public_backend_name(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_FROST)
        )
        model = FakeModel(
            {
                PLAN_KEY: plan,
                'transformer_options': {
                    STATUS_KEY: {
                        'attention': {
                            'selected': apply_module.ATTENTION_FROST_BF16,
                            'reason': 'explicit selection',
                        },
                        'sparse': {
                            'backend': SPARSE_BACKEND_FROST,
                            'video_budget': 0.3,
                        },
                        'fused_qkv': {'provider': 'standard_h3_qkv'},
                        'mlp': {'provider': 'off'},
                    }
                },
            }
        )

        text = format_sparse_status(model)
        self.assertIn('Attention: FROST BF16 (SM89)', text)
        self.assertIn('Requested sparse backend: FROST BF16 (SM89)', text)
        self.assertNotIn('Sparse fallback:', text)


if __name__ == '__main__':
    unittest.main()
