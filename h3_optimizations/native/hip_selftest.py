"""Per-device numerical gate for experimental gfx12 Sparse Kitchen."""

from __future__ import annotations

import logging
import threading

import torch


LOG_PREFIX = '[H3 Optimizations]'
_lock = threading.Lock()
_results = {}
_REL_L2_TOLERANCE = 0.15
_MAX_ABS_TOLERANCE = 1.0


def _relative_l2(actual, expected):
    error = (actual.float() - expected.float()).norm()
    return (error / expected.float().norm().clamp_min(1e-12)).item()


def _sparse_absolute_rows(heads, q_blocks, kv_tiles, selected_tiles=5):
    selected_tiles = min(int(selected_tiles), int(kv_tiles))
    rows = []
    for head in range(heads):
        head_rows = []
        for q_block in range(q_blocks):
            chosen = {0, kv_tiles - 1}
            candidate = head * 3 + q_block * 5
            while len(chosen) < selected_tiles:
                chosen.add(candidate % kv_tiles)
                candidate += 2
            head_rows.append(sorted(chosen)[:selected_tiles])
        rows.append(head_rows)
    return rows


def _delta_rows(absolute_rows):
    return [
        [
            [tile if index == 0 else tile - row[index - 1] for index, tile in enumerate(row)]
            for row in head_rows
        ]
        for head_rows in absolute_rows
    ]


def _sparse_reference(q, k, v, absolute_rows, q_tile, kv_tile):
    expected = torch.empty_like(q)
    for head, head_rows in enumerate(absolute_rows):
        for q_block, tiles in enumerate(head_rows):
            q_start = q_block * q_tile
            q_stop = min(q_start + q_tile, q.shape[-2])
            positions = [
                position
                for tile in tiles
                for position in range(
                    tile * kv_tile,
                    min((tile + 1) * kv_tile, k.shape[-2]),
                )
            ]
            selected_k = k[:, head:head + 1, positions]
            selected_v = v[:, head:head + 1, positions]
            expected[:, head:head + 1, q_start:q_stop] = (
                torch.nn.functional.scaled_dot_product_attention(
                    q[:, head:head + 1, q_start:q_stop].float(),
                    selected_k.float(),
                    selected_v.float(),
                ).to(q.dtype)
            )
    return expected


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


def _key(native, device):
    index = torch.cuda.current_device() if device is None else torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    return (
        index,
        native.device_architecture(index),
        torch.cuda.get_device_name(index),
    )


def run(device=None):
    from . import hip_int8_attention as native

    device = torch.device('cuda' if device is None else device)
    detail = {}
    try:
        generator = torch.Generator(device=device).manual_seed(20260830)
        q_storage, k_storage, v_storage = (
            torch.randn(
                (1, 641, 4, 128),
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
        carrier = native.prequantize_int8_attention(q, k, v)
        full_actual = native.int8_attention_from_prequantized(
            carrier, output_layout=native.OUTPUT_NHD
        )
        full_expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(torch.bfloat16)

        q_blocks = (q.shape[-2] + native.Q_TILE - 1) // native.Q_TILE
        kv_tiles = (k.shape[-2] + native.KV_TILE - 1) // native.KV_TILE
        absolute_rows = _sparse_absolute_rows(q.shape[1], q_blocks, kv_tiles)
        sparse_route = native.BlockSparseRoute(
            indices=torch.tensor(
                [_delta_rows(absolute_rows)], dtype=torch.int32, device=device
            ),
            counts=torch.full(
                (1, q.shape[1], q_blocks),
                len(absolute_rows[0][0]),
                dtype=torch.int32,
                device=device,
            ),
            q_tile=native.Q_TILE,
            kv_tile=native.KV_TILE,
            encoding='delta',
        )
        sparse_actual = native.block_sparse_int8_attention_from_prequantized(
            carrier, sparse_route, output_layout=native.OUTPUT_NHD
        )
        sparse_expected = _sparse_reference(
            q, k, v, absolute_rows, native.Q_TILE, native.KV_TILE
        )

        full_metrics = _metrics(full_actual, full_expected)
        sparse_metrics = _metrics(sparse_actual, sparse_expected)
        torch.cuda.synchronize(device)
        passed = bool(
            full_actual.transpose(1, 2).is_contiguous()
            and sparse_actual.transpose(1, 2).is_contiguous()
            and _metrics_pass(full_metrics)
            and _metrics_pass(sparse_metrics)
        )
        detail.update(
            {
                'input_layout': 'HND views over NHD storage',
                'sequence_length': q.shape[-2],
                'output_layout': native.OUTPUT_NHD,
                'full_output_nhd_contiguous': full_actual.transpose(1, 2).is_contiguous(),
                'sparse_output_nhd_contiguous': sparse_actual.transpose(1, 2).is_contiguous(),
                'selected_kv_tiles': len(absolute_rows[0][0]),
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
    from . import hip_int8_attention as native

    key = _key(native, device)
    with _lock:
        if not force and key in _results:
            return bool(_results[key]['passed'])
        passed, detail = run(device)
        _results[key] = detail
        if not passed:
            logging.warning(
                '%s EXPERIMENTAL AMD SPARSE KITCHEN SELF-TEST FAILED on %s: %s',
                LOG_PREFIX,
                key,
                detail,
            )
        return passed
