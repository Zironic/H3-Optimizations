'''Apply-plan composition through Memory then Sparse and the reverse.'''

from copy import deepcopy
import logging
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

import comfy.ldm.modules.attention as comfy_attention  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402
import comfy.patcher_extension  # noqa: E402
from comfy_extras.nodes_model_advanced import ModelAttentionBackend  # noqa: E402
import h3_optimizations.apply as apply_module  # noqa: E402
from h3_optimizations.cube_order import TOKEN_ORDER_SHAPES  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    H3OptimizationPlan,
    MemoryRequest,
    QKV_STREAMING_OFF,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_PUBLIC_REQUESTS,
    SparseRequest,
    STATUS_KEY,
    PLAN_KEY,
    VIDEO_TOKEN_ORDER_REQUESTS,
    VIDEO_TOKEN_ORDER_RASTER,
    read_plan,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLPProviderResolution,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKVProviderResolution,
    QKV_STANDARD,
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
        cloned.callbacks = deepcopy(self.callbacks)
        cloned.wrappers = deepcopy(self.wrappers)
        return cloned

    def remove_callbacks_with_key(self, call_type, key):
        self.callbacks.get(call_type, {}).pop(key, None)

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {})[key] = [callback]

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {})[key] = [wrapper]

    def set_model_optimized_attention(self, attention):
        ModelPatcher.set_model_optimized_attention(self, attention)


def resolved_attention(plan):
    selected = 'sparse_sage' if plan.sparse else 'comfy_kitchen_int8'
    return apply_module.ResolvedAttention(
        requested=selected,
        selected=selected,
        backend=SimpleNamespace(name=selected),
        reason='synthetic',
        backend_kind=selected,
        projector=None,
    )


def apply_in_order(base, first_request, second_request):
    plan = H3OptimizationPlan()
    if isinstance(first_request, MemoryRequest):
        plan = plan.with_memory(first_request)
    else:
        plan = plan.with_sparse(first_request)
    first = apply_module.apply_plan(base, plan)

    plan = read_plan(first)
    if isinstance(second_request, MemoryRequest):
        plan = plan.with_memory(second_request)
    else:
        plan = plan.with_sparse(second_request)
    return apply_module.apply_plan(first, plan)


def run_prepare_wrappers(model):
    wrapper_type = comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING
    wrappers = model.wrappers[wrapper_type][
        apply_module.PREPARE_WRAPPER_KEY
    ]
    return comfy.patcher_extension.WrapperExecutor.new_executor(
        lambda patcher, *_args, **_kwargs: patcher,
        wrappers,
    ).execute(model)


class ApplyCompositionTests(unittest.TestCase):
    def setUp(self):
        cube_install = mock.patch.object(
            apply_module,
            'install_cube_order',
            return_value=True,
        )
        self.install_cube_order = cube_install.start()
        self.addCleanup(cube_install.stop)

    def test_live_option_sync_is_a_no_op_for_the_patcher_options_object(self):
        wrapper = lambda executor, *args, **kwargs: executor(*args, **kwargs)
        model = FakeModel({
            'transformer_options': {
                'h3_optimizations_status': {'ready': True},
                'wrappers': {
                    'outer_sample': {
                        apply_module.OUTER_WRAPPER_KEY: [wrapper],
                    },
                },
            },
        })

        apply_module._sync_h3_live_options(model, model.model_options)

        self.assertIs(
            model.model_options['transformer_options']['wrappers'][
                'outer_sample'
            ][apply_module.OUTER_WRAPPER_KEY][0],
            wrapper,
        )

    def test_prepare_finalization_observes_downstream_wrapper_mutations(self):
        external = lambda value: value * 3
        plan = H3OptimizationPlan(memory=MemoryRequest())

        def reconcile(patcher, actual_plan, **_kwargs):
            self.assertIs(actual_plan, plan)
            self.assertIs(
                patcher.model_options['transformer_options'][
                    'optimized_attention_override'
                ],
                external,
            )
            return patcher

        def downstream_wrapper(executor, patcher, *args, **kwargs):
            kwargs['model_options']['transformer_options'][
                'optimized_attention_override'
            ] = external
            return executor(patcher, *args, **kwargs)

        def executor(patcher, *_args, model_options=None, **_kwargs):
            override = model_options['transformer_options'][
                'optimized_attention_override'
            ]
            self.assertIs(
                patcher.model_options['transformer_options'][
                    'optimized_attention_override'
                ],
                external,
            )
            return override(2)

        for wrappers in (
            (apply_module._prepare_sampling_wrapper, downstream_wrapper),
            (downstream_wrapper, apply_module._prepare_sampling_wrapper),
        ):
            with self.subTest(order=wrappers):
                model = FakeModel()
                model.model_options[PLAN_KEY] = plan
                live_options = {'transformer_options': {}}
                with mock.patch.object(
                    apply_module, 'is_minimax_h3', return_value=True
                ), mock.patch.object(
                    apply_module, '_reconcile_plan', side_effect=reconcile
                ):
                    result = comfy.patcher_extension.WrapperExecutor.new_executor(
                        executor,
                        list(wrappers),
                    ).execute(
                        model,
                        None,
                        None,
                        model_options=live_options,
                    )
                self.assertEqual(result, 6)

    def test_sparse_policy_preserves_an_explicit_external_attention_override(self):
        external = lambda value: value
        model = FakeModel({
            'transformer_options': {
                'optimized_attention_override': external,
            },
        })
        dense = SimpleNamespace(
            requested='existing',
            selected='existing',
            backend=None,
            reason='preserved external',
            backend_kind=apply_module.ATTENTION_EXISTING_FULL_Q,
        )
        plan = H3OptimizationPlan(sparse=SparseRequest())
        with mock.patch.object(
            apply_module,
            'resolve_current_dense_attention',
            return_value=dense,
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
        ) as kitchen:
            attention, qkv = apply_module._resolve_attention(
                plan,
                model,
                object(),
                object(),
            )

        kitchen.assert_not_called()
        self.assertEqual(attention.selected, 'existing')
        self.assertIsNone(attention.backend)
        self.assertEqual(qkv.provider_id, 'standard_h3_qkv')
        self.assertIn('explicit external attention override', attention.reason)

    def test_actual_kitchen_backend_node_composes_with_sparse(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_KITCHEN),
        )
        sparse_resolution = (object(), object())
        inventory = object()
        environment = object()

        # CPU startup hides the Kitchen choice, so expose the real registered
        # callable without executing its CUDA attention implementation.
        with mock.patch.object(
            comfy_attention,
            'COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE',
            True,
        ), mock.patch.dict(
            comfy_attention.REGISTERED_ATTENTION_FUNCTIONS,
            {
                'comfy_kitchen_int8': (
                    comfy_attention.attention_comfy_kitchen_int8
                ),
            },
        ):
            choices = ModelAttentionBackend.INPUT_TYPES()[
                'required'
            ]['attention'][0]
            self.assertIn('comfy kitchen attention', choices)
            model, = ModelAttentionBackend().patch(
                FakeModel({'transformer_options': {}}),
                'comfy kitchen attention',
            )
            options = model.model_options['transformer_options']
            self.assertTrue(
                apply_module.is_comfy_kitchen_dense_attention(options)
            )
            with mock.patch.object(
                apply_module,
                '_resolve_kitchen_sparse',
                return_value=sparse_resolution,
            ) as kitchen:
                actual = apply_module._resolve_attention(
                    plan,
                    model,
                    inventory,
                    environment,
                )

        kitchen.assert_called_once_with(plan, environment, inventory)
        self.assertEqual(actual, sparse_resolution)

    def test_sparse_only_dense_fallback_disables_memory_qkv(self):
        plan = H3OptimizationPlan(sparse=SparseRequest())
        inventory = SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=True,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=False,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
        )
        auto_qkv = apply_module.resolve_qkv_provider(
            inventory,
            request=FUSED_QKV_AUTO,
            backend_kind='comfy_kitchen_int8',
            kitchen_producer_available=True,
        )
        self.assertEqual(auto_qkv.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        dense = SimpleNamespace(
            requested='existing',
            selected='comfy_kitchen_int8',
            backend=None,
            reason='preserved external Kitchen',
            backend_kind='comfy_kitchen_int8',
        )

        with mock.patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ), mock.patch.object(
            apply_module,
            'preserve_dense_attention',
            return_value=dense,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                FakeModel(),
                inventory,
                SimpleNamespace(cuda_available=False),
            )

        self.assertIsNone(attention.backend)
        self.assertIsNone(attention.projector)
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertIn('disabled', qkv.reason)

    def test_streaming_off_disables_qkv_carriers_with_sparse_present(self):
        plan = H3OptimizationPlan(
            memory=MemoryRequest(qkv_streaming=QKV_STREAMING_OFF),
            sparse=SparseRequest(),
        )
        self.assertEqual(apply_module._qkv_request(plan), FUSED_QKV_OFF)

    def test_both_node_orders_resolve_identically(self):
        memory = MemoryRequest()
        sparse = SparseRequest(
            video_budget=0.5,
            denser_early_late_steps=True,
        )
        inventory = SimpleNamespace(
            labels=lambda _name: (
                'TensorWiseINT8Layout+convrot256',
            ) * 50,
        )
        qkv = QKVProviderResolution(
            'standard_h3_qkv',
            False,
            'Comfy Kitchen external producer API is unavailable',
        )
        mlp = MLPProviderResolution(
            'generic_chunked_quantized',
            'mlp_chunked_native',
            'synthetic',
        )

        def resolve(plan, *_args):
            return resolved_attention(plan), qkv

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
            return_value=SimpleNamespace(
                cuda_available=True,
                capability=(12, 0),
                device_name='fake SM120',
                backend='nvidia_cuda',
                architecture='sm120',
            ),
        ), mock.patch.object(
            apply_module,
            '_resolve_dense',
            side_effect=resolve,
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=resolve,
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=resolve,
        ), mock.patch.object(
            apply_module,
            'configure_backend',
            return_value=(object(), 50),
        ), mock.patch.object(
            apply_module,
            'install_dense_attention',
            return_value=True,
        ), mock.patch.object(
            apply_module,
            '_install_mlp',
            return_value=(mlp, 50),
        ), mock.patch.object(
            apply_module,
            '_ensure_sparse_runtime',
            return_value=(object(), True),
        ):
            with self.assertLogs(level='DEBUG') as provisional_logs:
                memory_only = apply_module.apply_plan(
                    FakeModel(),
                    H3OptimizationPlan(memory=memory),
                )
                sparse_only = apply_module.apply_plan(
                    FakeModel(),
                    H3OptimizationPlan(sparse=sparse),
                )
                left = apply_in_order(FakeModel(), memory, sparse)
                right = apply_in_order(FakeModel(), sparse, memory)
                apply_module._reconcile_plan(
                    left,
                    read_plan(left),
                    phase='clone',
                    force_rebuild=True,
                )

            self.assertFalse(any(
                record.levelno >= logging.INFO
                for record in provisional_logs.records
            ))
            with self.assertLogs(level='DEBUG') as logs:
                run_prepare_wrappers(memory_only)
                run_prepare_wrappers(sparse_only)
                run_prepare_wrappers(left)
                run_prepare_wrappers(right)

        applied_logs = [
            line for line in logs.output if ' applied plan: ' in line
        ]
        self.assertEqual(len(applied_logs), 4)
        self.assertTrue(all('phase=prepare' in line for line in applied_logs))
        self.assertTrue(all(line.startswith('DEBUG:') for line in applied_logs))
        final_logs = [
            line for line in logs.output if ' final plan: ' in line
        ]
        self.assertEqual(len(final_logs), 4)
        self.assertTrue(all(line.startswith('INFO:') for line in final_logs))
        self.assertEqual(sum(
            'final plan: Memory Optimization; attention: Comfy Kitchen INT8.'
            in line for line in final_logs
        ), 1)
        self.assertEqual(sum(
            'final plan: Sparse Attention; attention: Sparse Sage.'
            in line for line in final_logs
        ), 1)
        self.assertEqual(sum(
            'final plan: Memory Optimization + Sparse Attention; '
            'attention: Sparse Sage.' in line for line in final_logs
        ), 2)
        self.assertFalse(any('qkv_provider=' in line for line in final_logs))
        warning_logs = [
            record.getMessage()
            for record in logs.records
            if record.levelno == logging.WARNING
        ]
        self.assertEqual(len(warning_logs), 4)
        self.assertTrue(all(
            'FUSED QKV IS NOT RUNNING' in line for line in warning_logs
        ))
        self.assertTrue(all(
            'qkv_weights=TensorWiseINT8Layout+convrot256 qkv_layers=50' in line
            for line in applied_logs
        ))
        self.assertTrue(all('qkv_provider=' in line for line in applied_logs))
        self.assertTrue(any(
            'features=memory+sparse' in line for line in applied_logs
        ))
        self.assertFalse(any(
            'replaces_attention=' in line for line in applied_logs
        ))
        self.assertFalse(any(' armed: ' in line for line in logs.output))
        self.assertEqual(
            read_plan(left).signature,
            read_plan(right).signature,
        )
        left_status = left.model_options[
            'transformer_options'
        ][STATUS_KEY]
        right_status = right.model_options[
            'transformer_options'
        ][STATUS_KEY]
        self.assertEqual(
            left_status['plan_signature'],
            right_status['plan_signature'],
        )
        self.assertTrue(
            left_status['sparse']['denser_early_late_steps']
        )

    def test_public_backends_share_token_order_installation_contract(self):
        inventory = SimpleNamespace(
            labels=lambda _name: (),
            out_proj_plain_float=False,
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_name='fake SM89',
            backend='nvidia_cuda',
            architecture='sm89',
        )
        attention = apply_module.ResolvedAttention(
            requested='synthetic',
            selected=apply_module.ATTENTION_SPARSE,
            backend=SimpleNamespace(
                name=apply_module.ATTENTION_SPARSE,
                as_status=lambda: {},
            ),
            reason='synthetic sparse backend',
            backend_kind=apply_module.ATTENTION_SPARSE,
        )
        qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'standard projection',
        )
        mlp = MLPProviderResolution('off', 'off', 'disabled')

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
            return_value=(object(), 1),
        ), mock.patch.object(
            apply_module,
            '_install_mlp',
            return_value=(mlp, 0),
        ), mock.patch.object(
            apply_module,
            '_ensure_sparse_runtime',
            return_value=(object(), True),
        ):
            for backend in SPARSE_BACKEND_PUBLIC_REQUESTS:
                for token_order in VIDEO_TOKEN_ORDER_REQUESTS:
                    with self.subTest(
                        backend=backend,
                        token_order=token_order,
                    ):
                        self.install_cube_order.reset_mock()
                        patched = apply_module.apply_plan(
                            FakeModel(),
                            H3OptimizationPlan(sparse=SparseRequest(
                                backend=backend,
                                video_token_order=token_order,
                            )),
                        )
                        cube_shape = TOKEN_ORDER_SHAPES[token_order]
                        if cube_shape is None:
                            self.install_cube_order.assert_not_called()
                        else:
                            self.install_cube_order.assert_called_once_with(
                                patched,
                                cube_shape,
                            )

    def test_missing_sparse_backend_preserves_dense_h3(self):
        inventory = SimpleNamespace(labels=lambda _name: ())
        qkv = QKVProviderResolution(
            'standard_h3_qkv',
            False,
            'standard projection',
        )
        mlp = MLPProviderResolution('off', 'off', 'disabled')
        dense = apply_module.ResolvedAttention(
            requested='existing',
            selected='existing',
            backend=None,
            reason='normal Comfy attention',
            backend_kind='existing',
            dense_resolution=SimpleNamespace(backend=None),
        )
        environment = SimpleNamespace(
            cuda_available=False,
            capability=None,
            device_name='cpu',
            backend='cpu',
            architecture='cpu',
        )
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        raster_plan = H3OptimizationPlan().with_sparse(SparseRequest(
            video_token_order=VIDEO_TOKEN_ORDER_RASTER,
        ))

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
            '_resolve_dense',
            return_value=(dense, qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError(
                'Kitchen sparse attention is unavailable'
            ),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError(
                'requires NVIDIA CUDA'
            ),
        ), mock.patch.object(
            apply_module,
            '_install_mlp',
            return_value=(mlp, 0),
        ), mock.patch.object(
            apply_module,
            'configure_backend',
        ) as configure, mock.patch.object(
            apply_module,
            '_ensure_sparse_runtime',
        ) as sparse_runtime:
            patched = apply_module.apply_plan(FakeModel(), plan)
            raster_patched = apply_module.apply_plan(
                FakeModel(),
                raster_plan,
            )

        status = patched.model_options['transformer_options'][STATUS_KEY]
        self.assertEqual(status['attention']['requested'], 'sparse_sage')
        self.assertEqual(status['attention']['selected'], 'existing')
        self.assertIn('requires NVIDIA CUDA', status['attention']['reason'])
        self.assertEqual(status['device']['backend'], 'cpu')
        self.assertFalse(status['runtime_installed'])
        self.assertIn('Attention: existing', format_sparse_status(patched))
        self.assertIn('Sparse fallback:', format_sparse_status(patched))
        self.assertIn('Video token order: 1x8x8', format_sparse_status(patched))
        self.install_cube_order.assert_called_once_with(patched, (1, 8, 8))
        self.assertIn(
            'Video token order: Raster (stock H3 order)',
            format_sparse_status(raster_patched),
        )
        configure.assert_not_called()
        sparse_runtime.assert_not_called()

    def test_sparse_failure_uses_resolved_memory_dense_path(self):
        inventory = SimpleNamespace(labels=lambda _name: ())
        qkv = QKVProviderResolution(
            'standard_h3_qkv',
            False,
            'standard projection',
        )
        mlp = MLPProviderResolution('off', 'off', 'disabled')
        dense_resolution = SimpleNamespace(backend=object())
        dense = apply_module.ResolvedAttention(
            requested='auto',
            selected='comfy_kitchen_int8',
            backend=None,
            reason='selected dense provider',
            backend_kind='comfy_kitchen_int8',
            dense_resolution=dense_resolution,
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(10, 0),
            device_name='future NVIDIA',
            backend='nvidia_cuda',
            architecture='sm100',
        )
        plan = H3OptimizationPlan(
            memory=MemoryRequest(),
            sparse=SparseRequest(),
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
            '_resolve_dense',
            return_value=(dense, qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError(
                'Kitchen sparse attention is unavailable'
            ),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError(
                'does not support device capability 10.0'
            ),
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            side_effect=apply_module.TritonSparseError(
                'BF16 Triton is unavailable'
            ),
        ), mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
            side_effect=apply_module.FP8FlexError(
                'FP8 FlexAttention is unavailable'
            ),
        ), mock.patch.object(
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
        ) as sparse_runtime:
            patched = apply_module.apply_plan(FakeModel(), plan)

        status = patched.model_options['transformer_options'][STATUS_KEY]
        self.assertEqual(
            status['attention']['selected'],
            'comfy_kitchen_int8',
        )
        install_dense.assert_called_once_with(patched, dense_resolution)
        sparse_runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
