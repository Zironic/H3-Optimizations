"""CPU-visible contracts for the exact native H3 fused-Q producer."""

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.native import fused_q, loader, selftest  # noqa: E402


class _Function:
    def __init__(self):
        self.args = None

    def __call__(self, *args):
        self.args = args
        return 0


class _Library:
    def __init__(self, include_fused):
        self.include_fused = include_fused

    def __getattr__(self, name):
        if name == fused_q._SYMBOL and not self.include_fused:
            raise AttributeError(name)
        function = _Function()
        setattr(self, name, function)
        return function


class NativeFusedQTests(unittest.TestCase):
    def test_device_selftest_skips_pre_ampere_without_loading_the_kernel(self):
        with (
            mock.patch.object(
                selftest.torch.cuda, "get_device_capability", return_value=(7, 5)
            ),
            mock.patch.object(selftest.loader, "load") as load,
        ):
            detail = selftest._run_fused_q("cuda")
        self.assertFalse(detail["available"])
        self.assertEqual(detail["reason"], "requires_sm80")
        load.assert_not_called()

    def test_availability_requires_the_device_parity_selftest(self):
        library = SimpleNamespace(**{fused_q._SYMBOL: object()})
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(8, 9)
            ),
            mock.patch.object(fused_q.loader, "load", return_value=library),
            mock.patch.object(selftest, "fused_q_check", return_value=False) as check,
        ):
            self.assertFalse(fused_q.fused_h3_q_is_available("cuda"))
            check.return_value = True
            self.assertTrue(fused_q.fused_h3_q_is_available("cuda"))
        self.assertEqual(check.call_args_list, [mock.call("cuda"), mock.call("cuda")])

    def test_loader_treats_fused_q_as_an_additive_abi4_symbol(self):
        old_library = _Library(False)
        self.assertIs(loader._bind(old_library), old_library)

        new_library = _Library(True)
        loader._bind(new_library)
        function = getattr(new_library, fused_q._SYMBOL)
        self.assertIs(function.restype, ctypes.c_int)
        self.assertEqual(
            function.argtypes,
            [ctypes.c_void_p] * 9
            + [ctypes.c_int64] * 3
            + [ctypes.c_int, ctypes.c_float, ctypes.c_size_t],
        )

    def test_fake_call_returns_kitchen_q_scale_and_router_summary(self):
        call = _Function()
        library = SimpleNamespace(**{fused_q._SYMBOL: call})
        activation = torch.zeros(65, 256, dtype=torch.int8)
        weight = torch.zeros(512, 256, dtype=torch.int8)
        activation_scale = torch.ones(65, 1, dtype=torch.float32)
        weight_scale = torch.ones(512, dtype=torch.float32)
        norm = torch.ones(128, dtype=torch.bfloat16)
        freqs = torch.zeros(65, 48, 2, 2, dtype=torch.bfloat16)
        with (
            mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(8, 9)
            ),
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=1234),
            ),
            mock.patch.object(fused_q.loader, "load", return_value=library),
            mock.patch.object(fused_q.loader, "check") as check,
        ):
            q, q_scale, summary = fused_q.fused_h3_q_from_int8(
                activation,
                weight,
                activation_scale,
                weight_scale,
                norm,
                freqs,
                full_k_length=4096,
                epsilon=1e-6,
            )

        self.assertEqual(tuple(q.shape), (1, 4, 65, 128))
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(tuple(q_scale.shape), (1, 4, 32))
        self.assertEqual(q_scale.dtype, torch.float32)
        self.assertEqual(tuple(summary.shape), (1, 4, 2, 128))
        self.assertEqual(summary.dtype, torch.bfloat16)
        self.assertEqual(call.args[9:14], (65, 512, 256, 4096, 1e-6))
        self.assertEqual(call.args[-1], 1234)
        check.assert_called_once_with(0, "fused_h3_q_exact_128x256")

    def test_rejects_one_head_and_pre_ampere_devices(self):
        tensors = {
            "activation": torch.zeros(64, 256, dtype=torch.int8),
            "weight": torch.zeros(128, 256, dtype=torch.int8),
            "activation_scale": torch.ones(64, 1),
            "weight_scale": torch.ones(128),
            "norm": torch.ones(128, dtype=torch.bfloat16),
            "freqs": torch.zeros(64, 48, 2, 2, dtype=torch.bfloat16),
        }
        with mock.patch.object(
            torch.Tensor,
            "is_cuda",
            new_callable=mock.PropertyMock,
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "divisible by 256"):
                fused_q.fused_h3_q_from_int8(
                    **tensors, full_k_length=64, epsilon=1e-6
                )

        tensors["weight"] = torch.zeros(256, 256, dtype=torch.int8)
        tensors["weight_scale"] = torch.ones(256)
        with (
            mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(7, 5)
            ),
        ):
            with self.assertRaisesRegex(loader.NativeUnavailableError, "8.0"):
                fused_q.fused_h3_q_from_int8(
                    **tensors, full_k_length=64, epsilon=1e-6
                )

    def test_source_owns_only_the_measured_exact_configuration(self):
        source = (PACK / "native" / "src" / "h3_fused_q_cutlass.cu").read_text(
            encoding="utf-8"
        )
        cmake = (PACK / "native" / "CMakeLists.txt").read_text(encoding="utf-8")
        api = (
            PACK / "native" / "src" / "h3_int8_attention_api.cu"
        ).read_text(encoding="utf-8")
        self.assertIn("GemmShape<128, 256, 64>", source)
        self.assertIn("cutlass::arch::Sm80", source)
        self.assertNotIn("math_mode", source)
        self.assertIn("src/h3_fused_q_cutlass.cu", cmake)
        self.assertIn("third_party/cutlass/include", cmake)
        self.assertIn(fused_q._SYMBOL, api)
        self.assertIn("h3_int8_abi_version() noexcept { return 4; }", api)


if __name__ == "__main__":
    unittest.main()
