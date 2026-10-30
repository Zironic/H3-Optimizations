'''The entry point refuses ComfyUI releases older than the declared floor.'''

from importlib import util as importlib_util
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], *TEST_ARGS]


def _load_entry_point(comfyui_version):
    fake = types.ModuleType('comfyui_version')
    fake.__version__ = comfyui_version
    spec = importlib_util.spec_from_file_location(
        'h3_optimizations_entry_point_under_test', PACK / '__init__.py'
    )
    module = importlib_util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {'comfyui_version': fake}):
        spec.loader.exec_module(module)
    return module


class ComfyFloorTests(unittest.TestCase):
    def test_older_comfyui_loads_no_nodes_and_names_the_last_release(self):
        with self.assertLogs(level='ERROR') as logs:
            module = _load_entry_point('0.39.2')

        self.assertEqual(module.NODE_CLASS_MAPPINGS, {})
        self.assertFalse(hasattr(module, 'comfy_entrypoint'))
        self.assertIn('found 0.39.2', logs.output[0])
        self.assertIn(module.LAST_RELEASE_FOR_OLDER_COMFYUI, logs.output[0])

    def test_floor_release_loads_the_extension(self):
        module = _load_entry_point('0.40.0')

        self.assertTrue(callable(getattr(module, 'comfy_entrypoint', None)))
        self.assertFalse(hasattr(module, 'NODE_CLASS_MAPPINGS'))

    def test_floor_matches_registry_metadata(self):
        metadata = tomllib.loads((PACK / 'pyproject.toml').read_text(encoding='utf-8'))
        declared = metadata['tool']['comfy']['requires-comfyui']
        module = _load_entry_point('0.40.0')

        self.assertEqual(
            declared, '>=%s' % '.'.join(map(str, module.MINIMUM_COMFYUI))
        )
        self.assertIsNotNone(re.fullmatch(r'\d+\.\d+\.\d+', module.LAST_RELEASE_FOR_OLDER_COMFYUI))


if __name__ == '__main__':
    unittest.main()
