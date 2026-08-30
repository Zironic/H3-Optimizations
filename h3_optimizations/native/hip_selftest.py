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
        q, k, v = (
            torch.randn(
                (1, 4, 640, 128),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            for _ in range(3)
        )
        carrier = native.prequantize_int8_attention(q, k, v)
        actual = native.int8_attention_from_prequantized(carrier)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(torch.bfloat16)
        relative_l2 = _relative_l2(actual, expected)
        max_abs = (actual.float() - expected.float()).abs().max().item()
        finite = bool(torch.isfinite(actual).all())
        torch.cuda.synchronize(device)
        passed = bool(
            finite
            and relative_l2 < _REL_L2_TOLERANCE
            and max_abs < _MAX_ABS_TOLERANCE
        )
        detail.update(
            {
                'finite': finite,
                'full_route_rel_l2': round(relative_l2, 6),
                'full_route_max_abs': round(max_abs, 6),
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
