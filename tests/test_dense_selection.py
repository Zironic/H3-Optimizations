'''CPU-only contracts for public dense-attention backend selection.'''

import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
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

from comfy.model_patcher import ModelPatcher  # noqa: E402
import h3_optimizations.apply_policy as apply_policy  # noqa: E402
apply_module = apply_policy._base
from h3_optimizations.dense_resolver import (  # noqa: E402
    ATTENTION_COMFY_KITCHEN_INT8,
    ATTENTION_EXISTING,
    ATTENTION_SAGE,
    ATTENTION_SAGE_PREFIX,
    ATTENTION_SAGE_SM89,
    DenseResolution,
    install_dense_attention,
    resolve_current_dense_attention,
    resolve_dense_attention,
)
from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_PRESERVE_BF16,
    H3OptimizationPlan,
    MemoryRequest,
    QKV_STREAMING_OFF,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_BF16_CHUNKED,
    QKV_DENSE_KITCHEN_CHUNKED,
    QKV_STANDARD,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakePatcher:
    def __init__(self):
        self.model_options = {'transformer_options': {}}

    def set_model_optimized_attention(self, attention):
        ModelPatcher.set_model_optimized_attention(self, attention)


class ChunkRowsTests(unittest.TestCase):
    def test_qkv_chunk_rows_preserve_kernel_alignment(self):
        self.assertEqual(apply_module._effective_qkv_chunk_rows(1), 256)
        self.assertEqual(apply_module._effective_qkv_chunk_rows(257), 256)
        self.assertEqual(apply_module._effective_qkv_chunk_rows(65_537), 65_536)


def kitchen_attention(*_args, **_kwargs):
    return None


def kitchen_containers(*_args, **_kwargs):
    return None


kitchen_attention.container_function = kitchen_containers


def pytorch_attention(*_args, **_kwargs):
    return None


def sage_attention(*_args, **_kwargs):
    return None


class DenseSelectionTests(unittest.TestCase):
    @staticmethod
    def _convrot_inventory():
        return SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=True,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=False,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
        )

    @staticmethod
    def _bf16_inventory():
        item = SimpleNamespace(
            plain_float=True,
            logical_dtype='torch.bfloat16',
            label='Tensor:torch.bfloat16',
        )
        return SimpleNamespace(
            qkv=(item,),
            qkv_convrot_int8_256=False,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=True,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: (item.label,),
        )

    @staticmethod
    def _source_inventory(source):
        convrot, w4a8, fp8, plain = {
            'bf16': (False, False, False, True),
            'convrot': (True, False, False, False),
            'w4a8': (False, True, False, False),
            'fp8': (False, False, True, False),
        }[source]
        item = SimpleNamespace(
            plain_float=plain,
            convrot_int8_256=convrot,
            w4a8=w4a8,
            fp8=fp8,
            logical_dtype='torch.bfloat16' if plain else source,
            label=source,
        )
        return SimpleNamespace(
            qkv=(item,),
            qkv_convrot_int8_256=convrot,
            qkv_w4a8=w4a8,
            qkv_fp8=fp8,
            qkv_plain_float=plain,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: (source,),
        )

    def test_public_lookup_and_setter_retain_container_function(self):
        model = FakePatcher()
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ) as lookup:
            resolution = resolve_dense_attention(model)

        lookup.assert_called_once_with('comfy_kitchen_int8', None)
        self.assertEqual(resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertTrue(install_dense_attention(model, resolution))
        override = model.model_options['transformer_options']['optimized_attention_override']
        self.assertIs(override.container_function, kitchen_containers)

    def test_existing_explicit_override_is_preserved(self):
        model = FakePatcher()
        model.set_model_optimized_attention(pytorch_attention)
        original = model.model_options['transformer_options']['optimized_attention_override']

        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            resolution = resolve_dense_attention(model)

        self.assertEqual(resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertFalse(install_dense_attention(model, resolution))
        self.assertIs(
            model.model_options['transformer_options']['optimized_attention_override'],
            original,
        )

    def test_external_sage_selector_resolves_known_native_consumer(self):
        model = FakePatcher()
        model.set_model_optimized_attention(sage_attention)
        backend = SimpleNamespace(
            name='sage_mem_eff',
            api=SimpleNamespace(version='2.2.test', kernel_name='fake'),
            allow_cpu_for_tests=True,
            runtime_listeners=(),
            projected_q_tile=128,
            projected_k_tile=64,
        )
        environment = SimpleNamespace(capability=(8, 9))

        def lookup(name, _fallback):
            return sage_attention if name == ATTENTION_SAGE else kitchen_attention

        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            side_effect=lookup,
        ), patch(
            'h3_optimizations.attention.sage_mem_eff.SM89SageMemoryEfficientBackend',
            return_value=backend,
        ):
            resolution = resolve_current_dense_attention(model, environment)

        self.assertEqual(resolution.selected, ATTENTION_SAGE)
        self.assertEqual(resolution.backend_kind, ATTENTION_SAGE_SM89)
        self.assertIs(resolution.backend, backend)

    def test_external_sage_selector_uses_capability_specific_backend_kind(self):
        model = FakePatcher()
        model.set_model_optimized_attention(sage_attention)
        backend = SimpleNamespace(name='prepared_sage')

        def lookup(name, _fallback):
            return sage_attention if name == ATTENTION_SAGE else None

        capabilities = (
            (7, 5),
            (8, 0),
            (8, 6),
            (8, 7),
            (8, 9),
            (9, 0),
            (10, 0),
            (12, 0),
            (12, 1),
        )
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            side_effect=lookup,
        ), patch(
            'h3_optimizations.dense_resolver._prepared_sage_backend',
            return_value=backend,
        ):
            for capability in capabilities:
                with self.subTest(capability=capability):
                    resolution = resolve_current_dense_attention(
                        model,
                        SimpleNamespace(capability=capability),
                    )
                    self.assertEqual(
                        resolution.backend_kind,
                        '%s%d%d'
                        % (ATTENTION_SAGE_PREFIX, capability[0], capability[1]),
                    )
                    self.assertIs(resolution.backend, backend)

    def test_unavailable_kitchen_leaves_normal_comfy_selection(self):
        model = FakePatcher()
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=None,
        ):
            resolution = resolve_dense_attention(model)

        self.assertEqual(resolution.selected, ATTENTION_EXISTING)
        self.assertFalse(install_dense_attention(model, resolution))
        self.assertNotIn(
            'optimized_attention_override',
            model.model_options['transformer_options'],
        )

    def test_official_override_composes_before_and_after_h3_selection(self):
        before = FakePatcher()
        before.set_model_optimized_attention(pytorch_attention)
        before_override = before.model_options['transformer_options']['optimized_attention_override']
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            before_resolution = resolve_dense_attention(before)
        self.assertEqual(before_resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertIs(
            before.model_options['transformer_options']['optimized_attention_override'],
            before_override,
        )

        after = FakePatcher()
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            automatic = resolve_dense_attention(after)
        install_dense_attention(after, automatic)
        after.set_model_optimized_attention(pytorch_attention)
        official_override = after.model_options['transformer_options']['optimized_attention_override']
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ):
            after_resolution = resolve_dense_attention(after)
        self.assertEqual(after_resolution.selected, ATTENTION_COMFY_KITCHEN_INT8)
        self.assertIs(
            after.model_options['transformer_options']['optimized_attention_override'],
            official_override,
        )

    def test_h3_dense_auto_installs_chunked_producer_only_for_kitchen(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(MemoryRequest())
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ), patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                self._convrot_inventory(),
            )
        self.assertEqual(qkv.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        self.assertEqual(attention.projector.name, 'chunked_kitchen_qkv')
        self.assertEqual(attention.backend.name, 'comfy_kitchen_int8_prequantized')

        model.set_model_optimized_attention(pytorch_attention)
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ), patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                self._convrot_inventory(),
            )
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertIsNone(attention.projector)
        self.assertIn('full-Q', attention.reason)

    def test_preserve_precision_convrot_can_stream_through_dense_kitchen(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(fused_qkv=FUSED_QKV_PRESERVE_BF16)
        )
        with patch(
            'h3_optimizations.dense_resolver.get_attention_function',
            return_value=kitchen_attention,
        ), patch.object(
            apply_module,
            'producer_api_available',
            return_value=True,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                self._convrot_inventory(),
            )
        self.assertEqual(qkv.provider_id, QKV_DENSE_KITCHEN_CHUNKED)
        self.assertEqual(attention.projector.name, 'chunked_kitchen_qkv')

    def test_existing_dense_bf16_streams_q_against_full_kv(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_PRESERVE_BF16,
                chunk_rows=2048,
            )
        )

        attention, qkv = apply_module._resolve_dense(
            plan,
            model,
            self._bf16_inventory(),
        )

        self.assertEqual(qkv.provider_id, QKV_BF16_CHUNKED)
        self.assertEqual(attention.selected, ATTENTION_EXISTING)
        self.assertEqual(attention.projector.name, 'streamed_dense_bf16_qkv')
        self.assertEqual(attention.projector.chunk_rows, 2048)

    def test_existing_dense_non_bf16_streams_q_against_full_kv(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_PRESERVE_BF16,
                chunk_rows=2048,
            )
        )
        inventory = self._convrot_inventory()
        inventory.qkv = (SimpleNamespace(
            plain_float=False,
            logical_dtype='TensorWiseINT8Layout+convrot256',
        ),)

        with patch.object(
            apply_module,
            'resolve_qkv_provider',
            return_value=SimpleNamespace(
                provider_id=QKV_BF16_CHUNKED,
                fused=False,
                reason='test non-BF16 bounded provider',
            ),
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                model,
                inventory,
            )

        self.assertEqual(qkv.provider_id, QKV_BF16_CHUNKED)
        self.assertEqual(attention.projector.name, 'streamed_dense_bf16_qkv')
        self.assertEqual(attention.projector.projection_mode, 'native')

    def test_external_streamed_consumer_resolves_the_declared_matrix(self):
        def override(*_args, **_kwargs):
            return None

        override.supports_streamed_h3_qkv = True
        override.consume = lambda **_kwargs: None
        model = FakePatcher()
        model.model_options['transformer_options'][
            'optimized_attention_override'
        ] = override
        requests = (
            FUSED_QKV_AUTO,
            FUSED_QKV_FORCE_BF16,
            FUSED_QKV_PRESERVE_BF16,
            FUSED_QKV_FORCE_QUANT,
        )
        for source in ('bf16', 'convrot', 'w4a8', 'fp8'):
            for request in requests:
                with self.subTest(source=source, request=request):
                    plan = H3OptimizationPlan().with_memory(
                        MemoryRequest(
                            attention=ATTENTION_EXISTING,
                            fused_qkv=request,
                        )
                    )
                    attention, _qkv = apply_module._resolve_dense(
                        plan,
                        model,
                        self._source_inventory(source),
                    )
                    expected_mode = 'native'
                    if request == FUSED_QKV_FORCE_BF16:
                        expected_mode = 'force_bf16'
                    elif request == FUSED_QKV_FORCE_QUANT and source == 'bf16':
                        expected_mode = 'force_int8'
                    self.assertEqual(
                        attention.projector.projection_mode,
                        expected_mode,
                    )
                    self.assertTrue(attention.projector.streamed_q)

    def test_known_dense_sage_resolution_uses_direct_carriers_for_matrix(self):
        backend = SimpleNamespace(
            name='sage_mem_eff',
            api=SimpleNamespace(version='2.2.test', kernel_name='fake'),
            allow_cpu_for_tests=True,
            runtime_listeners=(),
            projected_q_tile=128,
            projected_k_tile=64,
        )
        dense = DenseResolution(
            ATTENTION_SAGE,
            ATTENTION_SAGE,
            backend,
            'known Sage test consumer',
            ATTENTION_SAGE_SM89,
        )
        requests = (
            FUSED_QKV_AUTO,
            FUSED_QKV_FORCE_BF16,
            FUSED_QKV_PRESERVE_BF16,
            FUSED_QKV_FORCE_QUANT,
        )
        environment = SimpleNamespace(
            capability=(8, 9),
            cuda_available=False,
            device_index=None,
        )
        with patch.object(
            apply_module,
            'resolve_current_dense_attention',
            return_value=dense,
        ), patch(
            'h3_optimizations.attention.triton_i64.TRITON_AVAILABLE',
            True,
        ), patch.object(
            apply_module,
            '_fp8_execution_available',
            return_value=True,
        ):
            for source in ('bf16', 'convrot', 'w4a8', 'fp8'):
                for request in requests:
                    with self.subTest(source=source, request=request):
                        plan = H3OptimizationPlan().with_memory(
                            MemoryRequest(
                                attention=ATTENTION_EXISTING,
                                fused_qkv=request,
                            )
                        )
                        attention, _qkv = apply_module._resolve_dense(
                            plan,
                            FakePatcher(),
                            self._source_inventory(source),
                            environment,
                        )
                        expected_mode = 'native'
                        if request == FUSED_QKV_FORCE_BF16:
                            expected_mode = 'force_bf16'
                        elif request == FUSED_QKV_FORCE_QUANT and source == 'bf16':
                            expected_mode = 'force_int8'
                        self.assertEqual(
                            attention.projector.projection_mode,
                            expected_mode,
                        )
                        self.assertTrue(
                            attention.projector.consumer_native_carrier
                        )
                        self.assertTrue(attention.projector.streamed_q)
                        self.assertIs(attention.backend.delegate, backend)

    def test_non_sm89_dense_sage_installs_architecture_carrier(self):
        backend = SimpleNamespace(
            name='sage_mem_eff_sm90',
            projected_qkv_format='sage_per_thread_64_128',
            projected_q_tile=64,
            projected_k_tile=128,
            requires_h3_triton=False,
            quantize_projected_qk=lambda q, k: (q, q, k, k),
        )
        dense = DenseResolution(
            ATTENTION_SAGE,
            ATTENTION_SAGE,
            backend,
            'known SM90 Sage consumer',
            'dense_sage_sm90',
        )
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_PRESERVE_BF16,
            )
        )
        environment = SimpleNamespace(
            capability=(9, 0),
            cuda_available=False,
            device_index=None,
        )
        with patch.object(
            apply_module,
            'resolve_current_dense_attention',
            return_value=dense,
        ):
            attention, qkv = apply_module._resolve_dense(
                plan,
                FakePatcher(),
                self._source_inventory('convrot'),
                environment,
            )

        self.assertTrue(qkv.fused)
        self.assertIs(attention.backend.delegate, backend)
        self.assertIs(attention.projector.backend, backend)
        self.assertEqual(attention.projector.name, 'streamed_dense_sage_qkv')
        self.assertTrue(attention.projector.streamed_q)

    def test_streaming_off_preserves_upstream_qkv_and_attention(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(
            MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_PRESERVE_BF16,
                qkv_streaming=QKV_STREAMING_OFF,
            )
        )
        environment = SimpleNamespace(
            capability=(8, 9),
            device_index=None,
            cuda_available=False,
        )
        attention, qkv = apply_module._resolve_dense(
            plan,
            model,
            self._convrot_inventory(),
            environment,
        )

        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertFalse(qkv.fused)
        self.assertEqual(attention.selected, ATTENTION_EXISTING)
        self.assertIsNone(attention.backend)
        self.assertIsNone(attention.projector)

    def test_legacy_existing_request_preserves_incoming_attention(self):
        model = FakePatcher()
        plan = H3OptimizationPlan().with_memory(MemoryRequest(attention='existing'))
        attention, qkv = apply_module._resolve_dense(
            plan,
            model,
            self._convrot_inventory(),
        )
        self.assertEqual(attention.selected, ATTENTION_EXISTING)
        self.assertEqual(qkv.provider_id, QKV_BF16_CHUNKED)
        self.assertEqual(attention.projector.name, 'streamed_dense_bf16_qkv')


if __name__ == '__main__':
    unittest.main()
