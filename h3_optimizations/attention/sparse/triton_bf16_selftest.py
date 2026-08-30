'''Per-device numerical gate for experimental RDNA2 BF16 Triton.'''

from __future__ import annotations

import logging
import threading

import torch

from . import triton_bf16


LOG_PREFIX = '[H3 Optimizations]'
_lock = threading.Lock()
_results = {}
_REL_L2_TOLERANCE = 0.02
_MAX_ABS_TOLERANCE = 0.05


def _relative_l2(actual, expected):
    error = (actual.float() - expected.float()).norm()
    return (error / expected.float().norm().clamp_min(1e-12)).item()


def _metrics(actual, expected):
    return {
        'finite': bool(torch.isfinite(actual).all()),
        'rel_l2': _relative_l2(actual, expected),
        'max_abs': (actual.float() - expected.float()).abs().max().item(),
    }


def _metrics_pass(metrics):
    return bool(
        metrics['finite']
        and metrics['rel_l2'] < _REL_L2_TOLERANCE
        and metrics['max_abs'] < _MAX_ABS_TOLERANCE
    )


def _sparse_rows(heads, q_tiles, kv_tiles):
    rows = []
    for head in range(heads):
        head_rows = []
        for q_tile in range(q_tiles):
            first = (head + q_tile) % (kv_tiles - 1)
            head_rows.append(sorted((first, kv_tiles - 1)))
        rows.append(head_rows)
    return rows


def _sparse_reference(q, k, v, rows):
    expected = torch.empty_like(q)
    for head, head_rows in enumerate(rows):
        for q_block, tiles in enumerate(head_rows):
            q_start = q_block * triton_bf16.Q_TILE
            q_stop = min(q_start + triton_bf16.Q_TILE, q.shape[-2])
            positions = [
                position
                for tile in tiles
                for position in range(
                    tile * triton_bf16.KV_TILE,
                    min((tile + 1) * triton_bf16.KV_TILE, k.shape[-2]),
                )
            ]
            expected[:, head:head + 1, q_start:q_stop] = (
                torch.nn.functional.scaled_dot_product_attention(
                    q[:, head:head + 1, q_start:q_stop].float(),
                    k[:, head:head + 1, positions].float(),
                    v[:, head:head + 1, positions].float(),
                ).to(q.dtype)
            )
    return expected


def _prepared(q, k, v, sparse_lut, *, dense_q_tiles, sparse_q_tiles):
    return triton_bf16.PreparedTritonBF16(
        q=q,
        k=k,
        v=v,
        sparse_lut=sparse_lut,
        dense_q_tiles=dense_q_tiles,
        sparse_q_tiles=sparse_q_tiles,
        sparse_selected=int(sparse_lut.shape[-1]),
        layer_index=-1,
        metadata={'selftest': True},
    )


def _key(device):
    index = torch.cuda.current_device() if device is None else torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    architecture = str(properties.gcnArchName).split(':')[0]
    version = getattr(triton_bf16.triton, '__version__', 'unknown')
    return index, architecture, torch.cuda.get_device_name(index), version


def run(device=None):
    device = torch.device('cuda' if device is None else device)
    detail = {}
    try:
        generator = torch.Generator(device=device).manual_seed(20260830)
        q_storage, k_storage, v_storage = (
            torch.randn(
                (1, 129, 2, triton_bf16.HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            for _ in range(3)
        )
        q, k, v = (
            tensor.transpose(1, 2)
            for tensor in (q_storage, k_storage, v_storage)
        )
        q_tiles = (q.shape[-2] + triton_bf16.Q_TILE - 1) // triton_bf16.Q_TILE
        kv_tiles = (k.shape[-2] + triton_bf16.KV_TILE - 1) // triton_bf16.KV_TILE

        full_route = torch.empty(
            (1, q.shape[1], 0, 0), dtype=torch.int32, device=device
        )
        full_actual = triton_bf16._launch(
            _prepared(
                q,
                k,
                v,
                full_route,
                dense_q_tiles=q_tiles,
                sparse_q_tiles=0,
            )
        )
        full_expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(q.dtype)

        rows = _sparse_rows(q.shape[1], q_tiles, kv_tiles)
        sparse_route = torch.tensor(
            [rows], dtype=torch.int32, device=device
        ).contiguous()
        sparse_actual = triton_bf16._launch(
            _prepared(
                q,
                k,
                v,
                sparse_route,
                dense_q_tiles=0,
                sparse_q_tiles=q_tiles,
            )
        )
        sparse_expected = _sparse_reference(q, k, v, rows)
        torch.cuda.synchronize(device)

        full_metrics = _metrics(full_actual, full_expected)
        sparse_metrics = _metrics(sparse_actual, sparse_expected)
        passed = _metrics_pass(full_metrics) and _metrics_pass(sparse_metrics)
        detail.update(
            {
                'input_layout': 'HND views over NHD storage',
                'sequence_length': q.shape[-2],
                'selected_kv_tiles': sparse_route.shape[-1],
                'available_kv_tiles': kv_tiles,
                'full_route_finite': full_metrics['finite'],
                'full_route_rel_l2': round(full_metrics['rel_l2'], 6),
                'full_route_max_abs': round(full_metrics['max_abs'], 6),
                'sparse_route_finite': sparse_metrics['finite'],
                'sparse_route_rel_l2': round(sparse_metrics['rel_l2'], 6),
                'sparse_route_max_abs': round(sparse_metrics['max_abs'], 6),
                'passed': passed,
            }
        )
        return passed, detail
    except Exception as error:
        detail['error'] = '%s: %s' % (type(error).__name__, error)
        detail['passed'] = False
        return False, detail


def check(device=None, *, force=False):
    key = _key(device)
    with _lock:
        if not force and key in _results:
            return bool(_results[key]['passed'])
        passed, detail = run(device)
        _results[key] = detail
        if not passed:
            logging.warning(
                '%s EXPERIMENTAL RDNA2 BF16 TRITON SELF-TEST FAILED on %s: %s',
                LOG_PREFIX,
                key,
                detail,
            )
        return passed
