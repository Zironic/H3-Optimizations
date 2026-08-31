"""Contracts for the Kitchen block-sparse INT8 attention backend.

These need a GPU, and the rest of this suite is a CPU contract suite: most of
its files call ``os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')`` at
import, and one of them sorts before this file. Running the whole directory
therefore hides the GPU and everything here skips -- passing, silently, while
testing nothing.

Because it is ``setdefault``, an explicit value wins:

    CUDA_VISIBLE_DEVICES=0 pytest tests/

or run this file on its own.

The gate that matters is the same one the kernel itself is built on: at a 100%
video budget the routed traversal must reproduce Kitchen's dense INT8 output
bit-for-bit. Here it runs through the whole backend -- runtime snapshot, the
production router, carrier quantization, route encoding -- so anything that
mis-wires those layers shows up as a mismatch rather than as a plausible
number nobody checks.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
    MODE_SAGE128_FUSED_QKV,
)
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    HEAD_DIM,
    KV_TILE,
    PRODUCTION_KV_TILE,
    PRODUCTION_Q_TILE,
    Q_TILE,
    SparseKitchenBackend,
    SparseKitchenError,
)
from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402

# Gate on what the backend actually uses, not on which module supplies it:
# the vendored library first, an installed comfy-kitchen carrying the sparse
# kernel otherwise. Gating on comfy_kitchen alone silently skipped everything
# once the backend started preferring the vendored build.
from h3_optimizations.native import int8_attention as _vendored  # noqa: E402

if _vendored.int8_attention_is_available():
    ck = _vendored
    _SPARSE_READY = True
else:
    try:
        import comfy_kitchen as ck

        _SPARSE_READY = bool(
            hasattr(ck, "block_sparse_int8_attention_from_prequantized")
            and torch.cuda.is_available()
            and ck.int8_attention_is_available()
        )
    except ImportError:  # pragma: no cover - environment dependent
        ck = None
        _SPARSE_READY = False

requires_kitchen_sparse = pytest.mark.skipif(
    not _SPARSE_READY,
    reason="needs the vendored INT8 library or a comfy-kitchen with the sparse kernel",
)

TEXT_LEN = 226
HEADS = 8


class StubLayout:
    """The parts of H3's packed layout the router reads."""

    def __init__(self, seq_len, video_start):
        self.seq_len = seq_len
        self.video_range = (video_start, seq_len)
        self.segments = [
            (0, TEXT_LEN, "text"),
            (TEXT_LEN, video_start, "audio"),
            (video_start, seq_len, "video"),
        ]
        self.video_shape = (max(1, (seq_len - video_start) // 1008), 24, 42)
        self.audio_t = (video_start - TEXT_LEN) // 2


def _options(seq_len, video_start, *, step=0, steps=4):
    return {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=1,
            step_index=step,
            total_steps=steps,
            layout=StubLayout(seq_len, video_start),
            compute_dtype=torch.bfloat16,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    }


def _qkv(seq_len, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            1, seq_len, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ).transpose(1, 2)
        for _ in range(3)
    )


# --------------------------------------------------------------------------
# Contracts that need no GPU.
# --------------------------------------------------------------------------


def test_fused_qkv_mode_requires_the_chunked_kitchen_producer():
    config = HybridSparseConfig(mode=MODE_SAGE128_FUSED_QKV)
    with pytest.raises(SparseKitchenError, match="chunked Kitchen producer"):
        SparseKitchenBackend(config, kitchen=object())


def test_rejects_a_foreign_config():
    with pytest.raises(TypeError, match="HybridSparseConfig"):
        SparseKitchenBackend(config={"video_budget": 0.3}, kitchen=object())


def test_production_geometry_is_64x64():
    assert (PRODUCTION_Q_TILE, PRODUCTION_KV_TILE) == (64, 64)


def test_router_geometry_must_match_the_kernel():
    from h3_optimizations.attention.sparse.router import SparseTileRouter

    mismatched = SparseTileRouter(q_tile=Q_TILE, kv_tile=64)
    with pytest.raises(SparseKitchenError, match="router geometry"):
        SparseKitchenBackend(kitchen=object(), router=mismatched)


@pytest.mark.parametrize('q_tile,kv_tile', [(128, 64), (64, 128), (64, 64)])
def test_exact_quality_geometry_reaches_router_executor_and_status(q_tile, kv_tile):
    backend = SparseKitchenBackend(
        kitchen=object(),
        q_tile=q_tile,
        kv_tile=kv_tile,
    )
    assert (backend.router.q_tile, backend.router.kv_tile) == (q_tile, kv_tile)
    assert (backend.executor.q_tile, backend.executor.kv_tile) == (q_tile, kv_tile)
    status = backend.as_status()
    assert (status['sparse_q_tile'], status['sparse_kv_tile']) == (
        q_tile,
        kv_tile,
    )


def test_the_backend_imports_nothing_from_sparge():
    """The point of this backend is that it does not reach for Sparge.

    Checks the import graph rather than the text, because the module talks
    about Sparge in prose and a substring match would only ever test that.
    """
    import ast
    import h3_optimizations.attention.sparse.kitchen_sparse as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {name for name in imported if "spas_sage" in name or "sparse_sage" in name}
    assert not forbidden, f"Kitchen backend imports Sparge: {forbidden}"


# --------------------------------------------------------------------------
# The correctness gate, end to end through the backend.
# --------------------------------------------------------------------------


@requires_kitchen_sparse
@pytest.mark.parametrize("seq_len,video_start", [(4096, 640), (3000, 512)])
def test_full_budget_matches_dense_kitchen_bitwise(seq_len, video_start):
    """A 100% video budget routes every tile, so it must equal dense INT8."""
    q, k, v = _qkv(seq_len)
    backend = SparseKitchenBackend(HybridSparseConfig(video_budget=1.0))
    prepared = backend.prepare(
        q, k, v, layer_index=0, transformer_options=_options(seq_len, video_start)
    )
    routed = backend.execute(prepared)

    dense = ck.int8_attention_from_prequantized(
        ck.prequantize_int8_attention(q, k, v, cta_k=KV_TILE)
    )
    assert torch.equal(routed, dense), (
        "the routed backend diverged from dense Kitchen INT8 at a full budget"
    )


@requires_kitchen_sparse
def test_sparse_budget_produces_finite_output_of_the_right_shape():
    seq_len, video_start = 4096, 640
    q, k, v = _qkv(seq_len)
    backend = SparseKitchenBackend(HybridSparseConfig(video_budget=0.3))
    prepared = backend.prepare(
        q, k, v, layer_index=3, transformer_options=_options(seq_len, video_start)
    )
    out = backend.execute(prepared)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()
    assert prepared.metadata["layer"] == 3
    assert prepared.metadata["full_mask_density"] < 1.0


@requires_kitchen_sparse
def test_a_sparser_budget_attends_to_fewer_tiles():
    seq_len, video_start = 4096, 640
    q, k, v = _qkv(seq_len)
    options = _options(seq_len, video_start)

    densities = []
    for budget in (0.5, 0.2):
        backend = SparseKitchenBackend(HybridSparseConfig(video_budget=budget))
        prepared = backend.prepare(
            q, k, v, layer_index=0, transformer_options=options
        )
        densities.append(prepared.metadata["full_mask_density"])
    assert densities[1] < densities[0]


@requires_kitchen_sparse
def test_status_reports_the_kitchen_geometry():
    backend = SparseKitchenBackend(HybridSparseConfig(video_budget=0.3))
    status = backend.as_status()
    assert status["sparse_architecture"] == "comfy_kitchen_int8"
    assert status["sparse_kv_tile"] == KV_TILE
    assert status["sparse_v_format"] == "int8"
    assert status["fused_qkv"] is False
    assert "sparge_attention" not in status


@requires_kitchen_sparse
def test_requires_a_runtime_snapshot():
    q, k, v = _qkv(1024)
    backend = SparseKitchenBackend(HybridSparseConfig(video_budget=0.3))
    with pytest.raises(SparseKitchenError, match="runtime snapshot"):
        backend.prepare(q, k, v, layer_index=0, transformer_options={})


@requires_kitchen_sparse
def test_rejects_a_layout_of_the_wrong_length():
    q, k, v = _qkv(2048)
    backend = SparseKitchenBackend(HybridSparseConfig(video_budget=0.3))
    with pytest.raises(SparseKitchenError, match="does not match attention sequence"):
        backend.prepare(
            q, k, v, layer_index=0, transformer_options=_options(4096, 640)
        )
