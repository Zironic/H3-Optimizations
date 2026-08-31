'''Readable QKV status contracts.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import STATUS_KEY  # noqa: E402
from h3_optimizations.status import (  # noqa: E402
    format_qkv_execution,
    format_sparse_status,
)


class QKVStatusReadabilityTests(unittest.TestCase):
    def setUp(self):
        self.status = {
            'attention': {'selected': 'sparse_kitchen_int8'},
            'sparse': {'video_budget': 0.05},
            'fused_qkv': {
                'provider': 'streamed_bf16_kitchen_qkv',
                'projector': 'chunked_kitchen_qkv',
                'chunk_rows': 4096,
                'output_streamed': True,
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16'] * 50},
            'mlp': {'provider': 'off'},
        }

    def test_streamed_bf16_kitchen_describes_the_execution(self):
        self.assertEqual(
            format_qkv_execution(self.status),
            (
                'BF16 weights -> 4096-row BF16 chunks -> Kitchen INT8 carrier; '
                'output streamed'
            ),
        )

    def test_sparse_preview_hides_the_internal_route_name(self):
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: self.status},
            }
        )

        text = format_sparse_status(model)

        self.assertIn(
            'QKV: BF16 weights -> 4096-row BF16 chunks -> Kitchen INT8 carrier',
            text,
        )
        self.assertNotIn('streamed_bf16_kitchen_qkv', text)

    def _status_with_v_memory(self, requested, effective):
        status = dict(self.status)
        status['fused_qkv'] = dict(self.status['fused_qkv'])
        status['fused_qkv']['v_memory_requested'] = requested
        status['fused_qkv']['v_memory'] = effective
        return SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: status},
            }
        )

    def test_ignored_lower_vram_request_is_reported_not_silent(self):
        model = self._status_with_v_memory('two_pass', None)

        text = format_sparse_status(model)

        self.assertIn('Lower VRAM requested but not available', text)
        self.assertIn('running Standard', text)

    def test_honoured_lower_vram_request_reports_no_warning(self):
        model = self._status_with_v_memory('two_pass', 'two_pass')

        text = format_sparse_status(model)

        self.assertNotIn('not available', text)

    def test_standard_request_reports_no_warning(self):
        model = self._status_with_v_memory('retain', None)

        text = format_sparse_status(model)

        self.assertNotIn('not available', text)

    def test_auto_kitchen_success_is_not_labeled_as_fallback(self):
        status = dict(self.status)
        status['attention'] = {
            'selected': 'sparse_kitchen_int8',
            'reason': 'native Kitchen INT8 64Q x 64KV sparse attention',
        }
        status['sparse'] = {'backend': 'auto', 'video_budget': 0.3}
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: status},
            }
        )

        text = format_sparse_status(model)

        self.assertNotIn('Sparse fallback:', text)

    def test_kitchen_attention_uses_a_readable_name(self):
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: self.status},
            }
        )

        text = format_sparse_status(model)

        self.assertIn('Attention: Comfy Kitchen INT8 Sparse', text)
        self.assertNotIn('sparse_kitchen_int8', text)

    def test_existing_dense_sparse_attention_uses_a_readable_name(self):
        status = dict(self.status)
        status['attention'] = {'selected': 'existing_dense_sparse'}
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: status},
            }
        )

        text = format_sparse_status(model)

        self.assertIn('Attention: Existing Dense Sparse', text)
        self.assertNotIn('existing_dense_sparse', text)

    def test_auto_sparse_sage_is_labeled_as_fallback(self):
        status = dict(self.status)
        status['attention'] = {
            'selected': 'sparse_sage',
            'reason': 'Kitchen INT8 unavailable: synthetic',
        }
        status['sparse'] = {'backend': 'auto', 'video_budget': 0.3}
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: status},
            }
        )

        text = format_sparse_status(model)

        self.assertIn('Sparse fallback: Kitchen INT8 unavailable: synthetic', text)

    def test_advanced_ramp_describes_its_peak_floor_and_duration(self):
        status = dict(self.status)
        status['sparse'] = {
            'video_budget': 0.15,
            'early_steps': 8,
            'early_kv': 0.5,
            'late_steps': 0,
            'late_kv': 0.5,
            'early_schedule': 'Ramp',
        }
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: status},
            }
        )

        text = format_sparse_status(model)

        self.assertIn('Early ramp: 50.0% -> 15.0% KV over 8 steps', text)

    def test_sparse_preview_reports_runtime_int8_output_projection(self):
        self.status['fused_qkv']['out_proj_runtime_convrot_int8'] = True
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: self.status},
            }
        )

        self.assertIn(
            'Attention output: runtime ConvRot-256 INT8',
            format_sparse_status(model),
        )

    def test_composition_status_reports_preserved_external_and_object_patches(self):
        self.status['composition'] = {
            'external_attention_preserved': True,
            'preserved_object_patches': {
                'attention': ['block.0.attn.forward'],
                'blocks': ['block.1.forward'],
                'final_layer': True,
            },
        }
        model = SimpleNamespace(
            model_options={
                'transformer_options': {STATUS_KEY: self.status},
            }
        )

        text = format_sparse_status(model)

        self.assertIn('preserved explicit external attention', text)
        self.assertIn('1 attention, 1 block, FinalLayer', text)
        self.assertIn('conflicting H3 sub-optimizations are disabled', text)

    def test_standard_path_keeps_the_fallback_reason(self):
        status = {
            'fused_qkv': {
                'provider': 'standard_h3_qkv',
                'reason': 'Kitchen producer unavailable',
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16']},
        }

        self.assertEqual(
            format_qkv_execution(status),
            'BF16 weights -> standard QKV path (Kitchen producer unavailable)',
        )

    def test_force_quant_triton_reports_the_actual_retained_carrier(self):
        status = {
            'fused_qkv': {
                'provider': 'force_convrot_int8_triton_qkv',
                'chunk_rows': 4096,
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16']},
        }
        self.assertEqual(
            format_qkv_execution(status),
            (
                'BF16 weights -> runtime ConvRot-256 INT8 projection -> '
                'retained BF16 K/V + 4096-row BF16 Q slabs -> Triton'
            ),
        )

    def test_native_bf16_triton_reports_the_actual_retained_carrier(self):
        status = {
            'fused_qkv': {
                'provider': 'force_bf16_qkv',
                'projector': 'chunked_triton_sparse_qkv',
                'chunk_rows': 4096,
                'streamed_q': True,
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16']},
        }
        self.assertEqual(
            format_qkv_execution(status),
            (
                'BF16 weights -> forced BF16 projection; retained BF16 K/V + '
                '4096-row BF16 Q slabs -> Triton'
            ),
        )

    def test_dense_sage_reports_bounded_q_and_output(self):
        status = {
            'fused_qkv': {
                'provider': 'force_bf16_qkv',
                'projector': 'streamed_dense_sage_qkv',
                'chunk_rows': 4096,
                'streamed_q': True,
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16']},
        }
        self.assertEqual(
            format_qkv_execution(status),
            (
                'BF16 weights -> forced BF16 projection; retained native '
                'Sage K/V + 4096-row Q/output slabs'
            ),
        )

    def test_projector_name_does_not_claim_streaming_without_capability(self):
        status = {
            'fused_qkv': {
                'provider': 'force_bf16_qkv',
                'projector': 'chunked_triton_sparse_qkv',
                'chunk_rows': 4096,
                'streamed_q': False,
            },
            'weight_formats': {'qkv': ['Parameter:torch.bfloat16']},
        }
        self.assertIn('full BF16 Q/K/V', format_qkv_execution(status))

    def test_every_public_route_has_a_readable_description(self):
        expected = {
            'chunked_bf16_qkv': 'full BF16 Q/K/V',
            'force_bf16_qkv': 'forced BF16 projection',
            'force_bf16_streamed_kitchen_qkv': 'forced BF16 projection',
            'force_fp8_qkv': 'forced FP8 projection',
            'force_convrot_int8_qkv': 'runtime ConvRot-256 INT8 projection',
            'force_convrot_int8_kitchen_qkv': 'runtime ConvRot-256 INT8 projection',
            'force_convrot_int8_triton_qkv': 'runtime ConvRot-256 INT8 projection',
            'convrot_int8_dense_sage': 'dense Sage carrier',
            'chunked_kitchen_qkv': 'Kitchen INT8 carrier',
            'streamed_bf16_kitchen_qkv': 'Kitchen INT8 carrier',
            'chunked_fp8_kitchen_qkv': 'FP8 projection',
            'convrot_int8_sparse_sage': 'Sparse Sage carrier',
            'chunked_fp8_sparse_sage': 'Sparse Sage carrier',
            'chunked_triton_bf16_sparse': 'Triton BF16 carrier',
        }
        for provider, phrase in expected.items():
            with self.subTest(provider=provider):
                status = {
                    'fused_qkv': {
                        'provider': provider,
                        'chunk_rows': 4096,
                    },
                    'weight_formats': {
                        'qkv': ['Parameter:torch.bfloat16'],
                    },
                }

                text = format_qkv_execution(status)

                self.assertIn(phrase, text)
                self.assertNotIn(provider, text)


if __name__ == '__main__':
    unittest.main()
