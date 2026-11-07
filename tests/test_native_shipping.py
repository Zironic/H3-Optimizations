'''CPU contracts for packaged native backend shipping.'''

import os
from pathlib import Path
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
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

from h3_optimizations import __version__  # noqa: E402
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    SparseKitchenError,
    preflight_sparse_kitchen,
)
from h3_optimizations.native import artifacts, int8_attention, loader, selftest  # noqa: E402


BIN_DIR = PACK / 'native' / 'bin'
MAX_LINUX_BINARY_BYTES = 50 * 1024 * 1024


class NativeShippingTests(unittest.TestCase):
    def test_native_binaries_are_packaged(self):
        windows_binary = BIN_DIR / 'h3_int8_attention_v10.dll'
        linux_binary = BIN_DIR / 'libh3_int8_attention.so'

        self.assertEqual(loader._LIBRARY_NAMES['Windows'], windows_binary.name)
        self.assertGreater(windows_binary.stat().st_size, 1_000_000)
        self.assertEqual(windows_binary.read_bytes()[:2], b'MZ')
        self.assertGreater(linux_binary.stat().st_size, 1_000_000)
        self.assertEqual(linux_binary.read_bytes()[:4], b'\x7fELF')
        self.assertEqual(
            (BIN_DIR / 'BUILD_ID').read_text(encoding='utf-8').strip(),
            artifacts.NATIVE_BUILD,
        )

    def test_linux_binary_stays_below_registry_warning_threshold(self):
        linux_binary = BIN_DIR / 'libh3_int8_attention.so'
        self.assertLess(linux_binary.stat().st_size, MAX_LINUX_BINARY_BYTES)

    def test_three_block_launch_bound_targets_consumer_architectures(self):
        cmake = (PACK / 'native' / 'CMakeLists.txt').read_text(encoding='utf-8')
        source = (
            PACK / 'native' / 'src' / 'sage_attention'
            / 'qk_int_sv_i8_sparse_cuda.cuh'
        ).read_text(encoding='utf-8')

        self.assertIn('80-real;86-real;89-real', cmake)
        self.assertIn(
            '#if defined(__CUDA_ARCH__) && \\\n'
            '    (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || \\\n'
            '     __CUDA_ARCH__ == 1200)',
            source,
        )

    def test_shipped_binaries_export_the_current_kernel_surface(self):
        # BUILD_ID is a claim, not evidence: a stale binary alongside a bumped
        # BUILD_ID passes every size and magic-number check above. Assert the
        # entry points the Python bindings dlsym at runtime are really present.
        required = (
            b'h3_int8_fused_q',
            b'h3_int8_quantize_bf16_rowwise_convrot256',
            b'h3_int8_quantize_bf16_rms_adaln_rowwise_convrot256',
            b'h3_int8_quantize_q_chunk',
            b'h3_int8_quantize_v',
            b'h3_int8_v_amax_chunk',
            b'h3_int8_quantize_v_chunk_into',
            b'h3_sparse_sage_sa2pp_sm89',
            b'h3_int8_sparse_attention_tiles',
            b'h3_int8_align_route_v_blocks',
            *(name.encode() for name in loader.CHUNK_SCALED_V_SYMBOLS),
        )
        for name in ('h3_int8_attention_v10.dll', 'libh3_int8_attention.so'):
            contents = (BIN_DIR / name).read_bytes()
            for symbol in required:
                self.assertIn(symbol, contents, '%s is missing %s' % (name, symbol.decode()))

    def test_v_staging_symbol_list_matches_the_shipped_binaries(self):
        from h3_optimizations.native import v_staging

        contents = (BIN_DIR / 'libh3_int8_attention.so').read_bytes()
        for symbol in v_staging._NATIVE_SYMBOLS:
            self.assertIn(symbol.encode(), contents, symbol)

    def test_obsolete_windows_binary_is_not_shipped(self):
        self.assertFalse((BIN_DIR / 'h3_int8_attention_v4.dll').exists())

    def test_linux_binary_keeps_old_libstdcxx_compatibility(self):
        contents = (BIN_DIR / 'libh3_int8_attention.so').read_bytes()
        versions = {
            tuple(int(part) for part in match.split(b'.'))
            for match in re.findall(rb'GLIBCXX_(\d+\.\d+\.\d+)', contents)
        }

        self.assertTrue(versions)
        self.assertLessEqual(max(versions), (3, 4, 21))

    def test_package_versions_match(self):
        metadata = tomllib.loads((PACK / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertEqual(metadata['project']['version'], __version__)

    def test_frost_artifact_has_reproducible_source_and_license(self):
        frost = PACK / 'native' / 'frost'
        for name in (
            'h3_frost_bf16_sm89.cubin',
            'h3_frost_bf16_sm89.symbol',
            'frost_h3.patch',
            'compile_sm89.py',
            'Dockerfile',
            'PROVENANCE',
            'LICENSE.txt',
        ):
            self.assertTrue((frost / name).is_file(), name)
        provenance = (frost / 'PROVENANCE').read_text(encoding='utf-8')
        self.assertIn('ae8705effeea3804585b6aca554beaca1a76a3da', provenance)
        self.assertIn('64690d05f52335bd252c6ecd9ad5d470ad5cff1df0d48f59c35396d0f775188c', provenance)

    def test_native_availability_requires_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(8, 9)),
            mock.patch('h3_optimizations.native.selftest.check', return_value=True) as selftest_check,
        ):
            self.assertTrue(int8_attention.int8_attention_is_available('cuda'))
            selftest_check.assert_called_once_with('cuda')

    def test_native_availability_allows_sm75_only_after_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(7, 5)),
            mock.patch('h3_optimizations.native.selftest.check', return_value=True) as selftest_check,
        ):
            self.assertTrue(int8_attention.int8_attention_is_available('cuda'))
            selftest_check.assert_called_once_with('cuda')

    def test_native_availability_rejects_unsupported_capability_before_selftest(self):
        with (
            mock.patch.object(int8_attention.torch.cuda, 'is_available', return_value=True),
            mock.patch.object(int8_attention.loader, 'is_available', return_value=True),
            mock.patch.object(int8_attention.torch.cuda, 'get_device_capability', return_value=(7, 0)),
            mock.patch('h3_optimizations.native.selftest.check') as selftest_check,
        ):
            self.assertFalse(int8_attention.int8_attention_is_available('cuda'))
            selftest_check.assert_not_called()

    def test_sparse_preflight_allows_sm75_after_native_selftest(self):
        kitchen = mock.Mock(
            SPARSE_GEOMETRIES=((64, 64),),
        )
        kitchen.int8_attention_is_available.return_value = True

        self.assertIs(
            preflight_sparse_kitchen(
                cuda_available=lambda: True,
                capability_getter=lambda: (7, 5),
                kitchen=kitchen,
                q_tile=64,
                kv_tile=64,
            ),
            kitchen,
        )

    def test_sparse_preflight_rejects_capabilities_below_sm75(self):
        with self.assertRaisesRegex(SparseKitchenError, 'compute capability 7.5'):
            preflight_sparse_kitchen(
                cuda_available=lambda: True,
                capability_getter=lambda: (7, 0),
                kitchen=mock.Mock(),
            )

    def test_sparse_preflight_keeps_sm75_fail_closed(self):
        kitchen = mock.Mock()
        kitchen.int8_attention_is_available.return_value = False

        with self.assertRaisesRegex(SparseKitchenError, 'extension is not available'):
            preflight_sparse_kitchen(
                cuda_available=lambda: True,
                capability_getter=lambda: (7, 5),
                kitchen=kitchen,
            )

    def test_selftest_covers_every_shipped_sparse_geometry(self):
        self.assertEqual(
            selftest._SPARSE_PARITY_GEOMETRIES,
            int8_attention.SPARSE_GEOMETRIES,
        )

    def test_selftest_cache_key_includes_revision(self):
        with (
            mock.patch.object(
                selftest.torch.cuda, 'get_device_capability', return_value=(12, 0)
            ),
            mock.patch.object(
                selftest.torch.cuda,
                'get_device_name',
                return_value='NVIDIA GeForce RTX 5070 Ti',
            ),
            mock.patch(
                'h3_optimizations.native.bootstrap.installed_build_id',
                return_value='native-v8',
            ),
            mock.patch.object(loader, '_override_path', return_value=None),
        ):
            key = selftest._cache_key('cuda')

        self.assertEqual(
            key,
            'sm120|native-v8|%s|NVIDIA GeForce RTX 5070 Ti'
            % selftest._SELFTEST_REVISION,
        )

    def test_library_override_file_tolerates_a_bom(self):
        # PowerShell 5.1 and Windows editors write UTF-8 with a BOM; it must not
        # turn a leading comment into a (missing) path and silently fall back.
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            pointer = Path(folder) / 'library_path.txt'
            pointer.write_bytes('\ufeff# test build\r\nC:\\builds\\h3_dev.dll\r\n'.encode('utf-8'))
            with mock.patch.object(loader, '_OVERRIDE_POINTER', pointer):
                self.assertEqual(loader._override_path(), Path('C:\\builds\\h3_dev.dll'))

    def test_selftest_cache_key_separates_a_library_override(self):
        override = mock.Mock()
        override.name = 'h3_int8_attention_dev.dll'
        override.stat.return_value.st_mtime = 1234.5
        with (
            mock.patch.object(
                selftest.torch.cuda, 'get_device_capability', return_value=(8, 9)
            ),
            mock.patch.object(
                selftest.torch.cuda, 'get_device_name', return_value='RTX 4070'
            ),
            mock.patch(
                'h3_optimizations.native.bootstrap.installed_build_id',
                return_value='native-v8',
            ),
            mock.patch.object(loader, '_override_path', return_value=override),
        ):
            key = selftest._cache_key('cuda')

        self.assertEqual(
            key,
            'sm89|override:h3_int8_attention_dev.dll:1234|%s|RTX 4070'
            % selftest._SELFTEST_REVISION,
        )


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
