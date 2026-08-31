"""CPU contracts for selecting and holding the exact H3 fused-Q path."""

import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.qkv import fused_q  # noqa: E402
from h3_optimizations.qkv.streamed import PROJECTION_NATIVE  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FusedQBindingTests(unittest.TestCase):
    def test_selection_is_native_bf16_convrot_sm80_only(self):
        module = SimpleNamespace(
            heads=2,
            head_dim=128,
            qkv_proj=object(),
        )
        x = torch.zeros(64, 256, dtype=torch.bfloat16)
        rope = torch.zeros(1, 64, 1, 48, 2, 2, dtype=torch.bfloat16)
        with (
            mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                fused_q, "describe_linear", return_value=SimpleNamespace(convrot_int8_256=True)
            ),
            mock.patch.object(
                fused_q, "int8_rowwise_convrot256_is_available", return_value=True
            ),
            mock.patch.object(fused_q, "fused_h3_q_is_available", return_value=True),
        ):
            self.assertTrue(
                fused_q.fused_h3_q_supported(module, x, rope, PROJECTION_NATIVE)
            )
            self.assertFalse(
                fused_q.fused_h3_q_supported(module, x.float(), rope, PROJECTION_NATIVE)
            )
            self.assertFalse(
                fused_q.fused_h3_q_supported(module, x, rope, "force_bf16")
            )
            self.assertFalse(
                fused_q.fused_h3_q_supported(
                    module,
                    torch.zeros(64, 384, dtype=torch.bfloat16),
                    rope,
                    PROJECTION_NATIVE,
                )
            )
            module.heads = 1
            self.assertFalse(
                fused_q.fused_h3_q_supported(module, x, rope, PROJECTION_NATIVE)
            )

    def test_project_uses_h3_owned_quantizer_and_fixed_native_producer(self):
        class FakeQuantizedTensor:
            def __init__(self):
                self._qdata = torch.arange(768 * 256, dtype=torch.int8).reshape(768, 256)
                self._params = SimpleNamespace(
                    scale=torch.ones(768, dtype=torch.float32)
                )

        class Binding:
            def __init__(self):
                self.weight = FakeQuantizedTensor()
                self.exited = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.exited = True
                return False

        q_norm = SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16), eps=1e-6)
        module = SimpleNamespace(
            heads=2,
            head_dim=128,
            qkv_proj=object(),
            q_norm=q_norm,
        )
        x = torch.zeros(64, 256, dtype=torch.bfloat16)
        rope = torch.zeros(1, 64, 1, 48, 2, 2, dtype=torch.bfloat16)
        binding = Binding()
        activation = torch.zeros(64, 256, dtype=torch.int8)
        activation_scale = torch.ones(64, 1)
        expected = (object(), object(), object())

        with (
            mock.patch.object(fused_q, "QuantizedTensor", FakeQuantizedTensor),
            mock.patch.object(fused_q, "fused_h3_q_supported", return_value=True),
            mock.patch.object(
                fused_q, "HeldConvRotINT8Linear", return_value=binding
            ),
            mock.patch.object(
                fused_q.comfy.model_management,
                "cast_to",
                return_value=q_norm.weight,
            ),
            mock.patch.object(
                fused_q,
                "quantize_int8_rowwise_convrot256",
                return_value=(activation, activation_scale),
            ) as quantize,
            mock.patch.object(
                fused_q, "fused_h3_q_from_int8", return_value=expected
            ) as produce,
        ):
            held = fused_q.HeldExactH3FusedQ(
                module, x[:1], rope, PROJECTION_NATIVE
            )
            with held:
                actual = held.project(x, rope, 0, 64, 4096)

        self.assertIs(actual, expected)
        quantize.assert_called_once()
        self.assertEqual(tuple(produce.call_args.args[1].shape), (256, 256))
        self.assertEqual(int(produce.call_args.args[3].numel()), 256)
        self.assertEqual(produce.call_args.kwargs["full_k_length"], 4096)
        self.assertEqual(produce.call_args.kwargs["epsilon"], 1e-6)
        self.assertTrue(binding.exited)

    def test_production_source_does_not_import_the_external_kitchen_cuda_binding(self):
        source = (PACK / "h3_optimizations" / "qkv" / "fused_q.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("comfy_kitchen.backends", source)
        self.assertNotIn("cutlass_h3_fused_q", source)


if __name__ == "__main__":
    unittest.main()
