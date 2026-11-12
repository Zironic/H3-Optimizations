"""CPU contracts for runtime ConvRot-256 INT8 H3 linears."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
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

import comfy.ops  # noqa: E402
from comfy.quant_ops import TensorWiseINT8Layout  # noqa: E402
from h3_optimizations.memory import fused_residual  # noqa: E402
from h3_optimizations.qkv.int8 import (  # noqa: E402
    ConvRotINT8BindingError,
    HeldConvRotINT8Linear,
    LazyConvRotINT8Linear,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def floating_linear(weight, bias=None):
    return SimpleNamespace(
        weight=weight,
        bias=bias,
        weight_function=[],
        bias_function=[],
        _full_precision_mm=False,
    )


class RuntimeConvRotINT8Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weight = torch.randn((256, 256), dtype=torch.bfloat16)
        self.sample = torch.randn((2, 256), dtype=torch.bfloat16)
        self.module = floating_linear(self.weight)

    def test_float_weight_becomes_reference_convrot_layout(self):
        handle = object()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, handle),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight") as release:
            with HeldConvRotINT8Linear(
                self.module,
                self.sample[:1],
                allow_float_conversion=True,
            ) as binding:
                self.assertTrue(binding.converted_from_float)
                self.assertEqual(binding.weight._layout_cls, "TensorWiseINT8Layout")
                self.assertTrue(binding.weight._params.convrot)
                self.assertEqual(binding.weight._params.convrot_groupsize, 256)
                qdata, scale = TensorWiseINT8Layout.get_plain_tensors(
                    binding.weight
                )
                self.assertEqual(qdata.dtype, torch.int8)
                self.assertEqual(scale.dtype, torch.float32)
                self.assertEqual(tuple(scale.shape), (256, 1))
                output = binding.linear(self.sample)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(tuple(output.shape), (2, 256))

            release.assert_called_once_with(
                self.module,
                self.weight,
                None,
                handle,
            )

    def test_fp16_activation_uses_convrot_layout(self):
        sample = self.sample.to(torch.float16)
        weight = self.weight.to(torch.float16)
        module = floating_linear(weight)
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(weight, None, object()),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with HeldConvRotINT8Linear(
                module,
                sample[:1],
                allow_float_conversion=True,
            ) as binding:
                output = binding.linear(sample)

        self.assertIsNone(binding.weight)
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(tuple(output.shape), (2, 256))

    def test_fp32_activation_runs_native_convrot_weight(self):
        # SM75 executes H3 with FP32 manual cast (issue #89).
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, object()),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with HeldConvRotINT8Linear(
                self.module,
                self.sample[:1],
                allow_float_conversion=True,
            ) as binding:
                native = binding.weight
        module = floating_linear(native)
        sample = self.sample.float()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(native, None, object()),
        ) as acquire, mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with HeldConvRotINT8Linear(module, sample[:1]) as binding:
                self.assertFalse(binding.converted_from_float)
                output = binding.linear(sample)
                sliced = binding.linear_range(sample, 0, 96)

        self.assertEqual(acquire.call_args.kwargs["compute_dtype"], torch.float32)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (2, 256))
        torch.testing.assert_close(sliced, output[:, :96], rtol=0, atol=0)

    def test_fp32_activation_does_not_convert_fp32_weight(self):
        sample = self.sample.float()
        weight = self.weight.float()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(weight, None, object()),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with self.assertRaises(ConvRotINT8BindingError):
                HeldConvRotINT8Linear(
                    floating_linear(weight),
                    sample[:1],
                    allow_float_conversion=True,
                ).__enter__()

    def test_unsupported_activation_dtype_is_named(self):
        sample = self.sample.double()
        with self.assertRaisesRegex(RuntimeError, "got torch.float64"):
            HeldConvRotINT8Linear(self.module, sample[:1]).__enter__()

    def test_lazy_binding_quantizes_once_for_multiple_output_slabs(self):
        handle = object()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, handle),
        ) as acquire, mock.patch.object(
            comfy.ops,
            "uncast_bias_weight",
        ) as release:
            session = LazyConvRotINT8Linear(self.module)
            first = session.linear(self.sample[:1])
            second = session.linear(self.sample[1:])
            self.assertEqual(tuple(first.shape), (1, 256))
            self.assertEqual(tuple(second.shape), (1, 256))
            self.assertEqual(acquire.call_count, 1)
            session.release()
            self.assertEqual(release.call_count, 1)

    def test_lazy_binding_offers_plain_linear_to_residual_kernel(self):
        residual = torch.zeros_like(self.sample)
        gate = torch.ones(256, dtype=torch.bfloat16)
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, object()),
        ) as acquire, mock.patch.object(
            comfy.ops,
            "uncast_bias_weight",
        ), mock.patch.object(
            fused_residual,
            "linear_residual_",
            return_value=True,
        ) as fused:
            session = LazyConvRotINT8Linear(self.module)
            self.assertTrue(session.residual_(self.sample, residual, gate))
            self.assertTrue(session.residual_(self.sample, residual, gate))
            self.assertEqual(acquire.call_count, 1)
            session.release()

        self.assertEqual(fused.call_count, 2)
        self.assertIs(fused.call_args.kwargs["convrot"], True)
        self.assertEqual(fused.call_args.kwargs["convrot_groupsize"], 256)

    def test_output_slices_match_one_full_convrot_linear(self):
        handle = object()
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, handle),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with HeldConvRotINT8Linear(
                self.module,
                self.sample[:1],
                allow_float_conversion=True,
            ) as binding:
                full = binding.linear(self.sample)
                sliced = torch.cat(
                    (
                        binding.linear_range(self.sample, 0, 96),
                        binding.linear_range(self.sample, 96, 256),
                    ),
                    dim=-1,
                )

        torch.testing.assert_close(sliced, full, rtol=0, atol=0)

    def test_benchmark_override_receives_full_and_sliced_weights(self):
        seen = []

        def override(x, weight, bias):
            seen.append((int(weight.shape[0]), bias))
            return x.new_zeros((x.shape[0], weight.shape[0]))

        self.module._h3_benchmark_convrot_linear = override
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, None, object()),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with HeldConvRotINT8Linear(
                self.module,
                self.sample[:1],
                allow_float_conversion=True,
            ) as binding:
                full = binding.linear(self.sample)
                sliced = binding.linear_range(self.sample, 32, 96)

        self.assertEqual(tuple(full.shape), (2, 256))
        self.assertEqual(tuple(sliced.shape), (2, 64))
        self.assertEqual(seen, [(256, None), (64, None)])

    def test_bias_is_not_silently_quantized(self):
        bias = torch.zeros((256,), dtype=torch.bfloat16)
        module = floating_linear(self.weight, bias)
        with mock.patch.object(
            comfy.ops,
            "cast_bias_weight",
            return_value=(self.weight, bias, object()),
        ), mock.patch.object(comfy.ops, "uncast_bias_weight"):
            with self.assertRaisesRegex(RuntimeError, "bias-free"):
                HeldConvRotINT8Linear(
                    module,
                    self.sample[:1],
                    allow_float_conversion=True,
                ).__enter__()


if __name__ == "__main__":
    unittest.main()
