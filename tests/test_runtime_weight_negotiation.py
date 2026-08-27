"""Generic runtime-weight negotiation contracts for streamed H3 QKV."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

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

from comfy.quant_ops import QuantizedTensor  # noqa: E402
from h3_optimizations.qkv.bf16 import HeldBF16QKV  # noqa: E402
from h3_optimizations.qkv.int8 import HeldConvRotINT8QKV  # noqa: E402
from h3_optimizations.qkv.streamed import (  # noqa: E402
    PROJECTION_FORCE_INT8,
    PROJECTION_NATIVE,
    create_held_qkv,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def patched_convrot_attention():
    weight = torch.randn((768, 256), dtype=torch.bfloat16)
    quantized = QuantizedTensor.from_float(
        weight,
        "TensorWiseINT8Layout",
        scale="recalculate",
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=256,
    )
    qkv_proj = SimpleNamespace(
        weight=quantized,
        bias=None,
        weight_function=[object()],
        bias_function=[],
    )
    return SimpleNamespace(qkv_proj=qkv_proj)


class RuntimeWeightNegotiationTests(unittest.TestCase):
    def test_native_streaming_preserves_effective_runtime_precision(self):
        attention = patched_convrot_attention()
        sample = torch.zeros((1, 256), dtype=torch.bfloat16)
        binding = create_held_qkv(attention, sample, PROJECTION_NATIVE)
        self.assertIsInstance(binding, HeldBF16QKV)
        self.assertTrue(binding.allow_quantized_source)

    def test_force_quant_remains_authoritative(self):
        attention = patched_convrot_attention()
        sample = torch.zeros((1, 256), dtype=torch.bfloat16)
        binding = create_held_qkv(attention, sample, PROJECTION_FORCE_INT8)
        self.assertIsInstance(binding, HeldConvRotINT8QKV)
        self.assertTrue(binding.binding.allow_float_conversion)


if __name__ == "__main__":
    unittest.main()
