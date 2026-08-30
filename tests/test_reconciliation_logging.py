'''Human-readable labels for the final reconciled H3 plan summary.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations import apply as apply_module  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def _attention(selected):
    return SimpleNamespace(selected=selected)


def test_final_summary_uses_human_attention_names():
    assert apply_module._attention_summary_name(
        _attention(apply_module.ATTENTION_SPARSE)
    ) == 'Sparse Sage'
    assert apply_module._attention_summary_name(
        _attention('comfy_kitchen_int8')
    ) == 'Comfy Kitchen INT8'


def test_unknown_attention_name_remains_readable():
    assert apply_module._attention_summary_name(
        _attention('future_attention_backend')
    ) == 'future attention backend'
