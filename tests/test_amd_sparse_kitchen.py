"""CPU/static contracts for the experimental gfx12 Sparse Kitchen branch."""

import hashlib
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
for path in (str(PACK), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_optimizations.apply_policy as apply_policy  # noqa: E402
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    SparseKitchenError,
    preflight_sparse_kitchen,
)
from h3_optimizations.native import hip_int8_attention  # noqa: E402
from h3_optimizations.native import hip_selftest  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    SparseRequest,
)

apply_module = apply_policy._base
sys.argv = [sys.argv[0], *TEST_ARGS]


class ExperimentalAMDSparseKitchenTests(unittest.TestCase):
    @staticmethod
    def _hip_module(architecture='gfx1201', available=True):
        return SimpleNamespace(
            IS_HIP_SPARSE_KITCHEN=True,
            SPARSE_GEOMETRIES=((64, 64),),
            __version__='experimental-test',
            device_architecture=lambda: architecture,
            int8_attention_is_available=lambda: available,
        )

    def test_preflight_accepts_gfx12_without_nvidia_capability(self):
        cuda_probe = mock.Mock(side_effect=AssertionError('must not probe CUDA'))
        capability_probe = mock.Mock(
            side_effect=AssertionError('must not probe NVIDIA capability')
        )
        kitchen = self._hip_module()

        selected = preflight_sparse_kitchen(
            cuda_available=cuda_probe,
            capability_getter=capability_probe,
            backend='rocm',
            kitchen=kitchen,
            q_tile=64,
            kv_tile=64,
        )

        self.assertIs(selected, kitchen)
        cuda_probe.assert_not_called()
        capability_probe.assert_not_called()

    def test_preflight_rejects_gfx11_and_failed_selftest(self):
        with self.assertRaisesRegex(SparseKitchenError, 'gfx1200 or gfx1201'):
            preflight_sparse_kitchen(
                cuda_available=lambda: False,
                capability_getter=lambda: None,
                backend='rocm',
                kitchen=self._hip_module('gfx1100'),
                q_tile=64,
                kv_tile=64,
            )
        with self.assertRaisesRegex(SparseKitchenError, 'self-test'):
            preflight_sparse_kitchen(
                cuda_available=lambda: False,
                capability_getter=lambda: None,
                backend='rocm',
                kitchen=self._hip_module(available=False),
                q_tile=64,
                kv_tile=64,
            )

    def test_rocm_auto_prefers_sparse_kitchen_without_cuda_producer(self):
        plan = H3OptimizationPlan(sparse=SparseRequest())
        model = SimpleNamespace(model_options={})
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
            cuda_available=False,
            capability=None,
            device_index=0,
            backend='rocm',
        )
        kitchen = self._hip_module()

        with mock.patch.object(
            apply_module,
            'preflight_sparse_kitchen',
            return_value=kitchen,
        ) as preflight, mock.patch.object(
            apply_module,
            'producer_api_available',
            side_effect=AssertionError('CUDA producer must not be probed on ROCm'),
        ), mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(object(), object()),
        ):
            attention, qkv = apply_module._resolve_attention(
                plan, model, inventory, environment
            )

        self.assertEqual(attention.selected, apply_module.ATTENTION_KITCHEN_SPARSE)
        self.assertIn('experimental AMD', attention.reason)
        self.assertEqual(attention.projector.name, 'chunked_bf16_qkv')
        self.assertIs(attention.backend.projector, attention.projector)
        self.assertFalse(attention.backend.stream_output)
        self.assertEqual(qkv.provider_id, 'chunked_bf16_qkv')
        self.assertEqual(preflight.call_args.kwargs['backend'], 'rocm')

    def test_delta_routes_are_converted_to_absolute_for_the_hip_kernel(self):
        route = hip_int8_attention.BlockSparseRoute(
            indices=torch.tensor([[[[2, 3, 1, 0]]]], dtype=torch.int32),
            counts=torch.tensor([[[3]]], dtype=torch.int32),
            q_tile=64,
            kv_tile=64,
            encoding='delta',
        )
        converted = route.for_kernel()
        self.assertEqual(converted.encoding, 'absolute')
        self.assertEqual(converted.indices.tolist(), [[[[2, 5, 6, 0]]]])

    def test_hardware_selftest_route_is_sparse_varied_and_includes_tail(self):
        rows = hip_selftest._sparse_absolute_rows(4, 11, 11)

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row) == 5 for head in rows for row in head))
        self.assertTrue(all(row == sorted(set(row)) for head in rows for row in head))
        self.assertTrue(all(row[0] == 0 and row[-1] == 10 for head in rows for row in head))
        self.assertNotEqual(rows[0][0], rows[0][1])
        self.assertNotEqual(rows[0][0], rows[1][0])
        self.assertEqual(hip_selftest._delta_rows([[[0, 2, 5, 7, 10]]]), [[[0, 2, 3, 2, 3]]])

    def test_architecture_header_allows_the_hip_clang_host_pass(self):
        header = (
            PACK / 'native' / 'hip' / 'src' / 'architecture_config.h'
        ).read_text(encoding='utf-8')
        self.assertNotIn('#error', header)
        self.assertIn('#if defined(__gfx1200__) || defined(__gfx1201__)', header)

    def test_prebuilt_libraries_match_manifests_and_target_both_gfx12_arches(self):
        binary_root = PACK / 'native' / 'hip' / 'bin'
        linux_binary = binary_root / 'libh3_hip_sparse_kitchen.so'
        windows_binary = binary_root / 'h3_hip_sparse_kitchen.dll'
        linux_bytes = linux_binary.read_bytes()
        windows_bytes = windows_binary.read_bytes()

        self.assertEqual(linux_bytes[:4], b'\x7fELF')
        self.assertEqual(windows_bytes[:2], b'MZ')
        for contents in (linux_bytes, windows_bytes):
            self.assertIn(b'gfx1200', contents)
            self.assertIn(b'gfx1201', contents)

        linux_info = (binary_root / 'BUILD_INFO-linux.txt').read_text().splitlines()
        windows_info = (binary_root / 'BUILD_INFO-windows.txt').read_text().splitlines()
        linux_values = dict(line.split('=', 1) for line in linux_info[:4])
        windows_values = dict(line.split('=', 1) for line in windows_info)
        self.assertRegex(linux_values['source_sha'], r'^[0-9a-f]{40}$')
        self.assertEqual(linux_values['source_sha'], windows_values['source_sha'])
        self.assertEqual(linux_values['rocm_version'], '7.2.1')
        self.assertEqual(windows_values['rocm_version'], '7.2.1')
        self.assertEqual(linux_values['architectures'], 'gfx1200;gfx1201')
        self.assertEqual(windows_values['architectures'], 'gfx1200;gfx1201')
        self.assertEqual(hashlib.sha256(linux_bytes).hexdigest(), linux_info[4].split()[0])
        self.assertEqual(hashlib.sha256(windows_bytes).hexdigest(), windows_values['sha256'])

    def test_vendored_kernel_is_bm64_with_indexed_kv_traversal(self):
        source = (
            PACK / 'native' / 'hip' / 'src' / 'sage_attention' /
            'int8_attn_sparse.hip'
        ).read_text(encoding='utf-8')
        long_kernel = source[
            source.index('constexpr int kLongHeadDim'):
            source.index('template <int HD, int CTA_K')
        ]
        self.assertIn('constexpr int kLongBlockM = 64;', long_kernel)
        self.assertIn('constexpr int kLongWarps = 4;', long_kernel)
        self.assertIn('for (int selected = 0; selected < selected_tiles; ++selected)', long_kernel)
        self.assertIn('const int tile = route_row[selected];', long_kernel)
        self.assertNotIn('for (int tile = 0; tile < tiles; ++tile)', long_kernel)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
