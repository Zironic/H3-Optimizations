"""Prove the native kernels work on *this* GPU before trusting them.

The library is compiled for several architectures and validated directly on
one. A prebuilt binary makes "it was built for this card" an assumption rather
than something anyone checked, so check it here once and cache the verdict.

The dense CTA_K=64 carrier path is the common producer gate. Sparse geometries
are validated independently: production Kitchen is usable when either 64x64 or
its carrier-compatible 128x64 fallback is finite and numerically close to the
matched dense Kitchen output. Exact equality remains diagnostic because
harmless BF16 reduction-order differences may vary across GPU architectures.
The legacy 128x128/LSE path is validated only when an LSE-dependent
experimental backend explicitly asks for it, so it cannot disable normal
Kitchen.

Blackwell deliberately uses different probability math for short CTA_K=64
dense rows. Sparse kernels use the fused path. Geometry comparison therefore runs
with K > 512 so dense and sparse exercise the same probability implementation;
otherwise a healthy SM120 64KV kernel fails a meaningless comparison.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading

import torch

from . import loader

LOG_PREFIX = '[H3 Optimizations]'

_CACHE = pathlib.Path(__file__).resolve().parent.parent.parent / 'native' / 'selftest.json'
_lock = threading.Lock()
_result = None
_detail_result = None
_lse_lock = threading.Lock()
_lse_results = {}

# Bump whenever the meaning of a cached pass/fail changes without changing the
# native binary build ID. Otherwise an old failure can survive a Python-only fix.
_SELFTEST_REVISION = 'v7'

# K > 512 is intentional. Blackwell's dense CTA_K=64 launcher chooses a
# lower-pressure probability path at K <= 512 while sparse always uses fused
# probability math. At 640 rows the outputs compare like with like on SM120.
_BATCH, _HEADS, _Q_LEN, _KV_LEN, _HEAD_DIM = 1, 4, 640, 640, 128

# Keep this local to the native startup layer: importing the high-level sparse
# backend here would pull Comfy runtime modules into pre-startup initialization.
_SPARSE_PARITY_GEOMETRIES = ((128, 128), (128, 64), (64, 64))
_PRODUCTION_GEOMETRIES = ((64, 64), (128, 64))

# Healthy INT8 error is ~0.016 relative L2; the mildest corruption injected in
# testing produced 0.42. This sits between, well clear of both.
_INT8_TOLERANCE = 0.15
# Focused corruption tests accept a local BF16 ULP while rejecting distributed
# drift, a localized 0.02 error, and non-finite output.
_SPARSE_REL_L2_TOLERANCE = 0.002
_SPARSE_MAX_ABS_TOLERANCE = 0.01


def _geometry_key(q_tile, kv_tile):
    return '%dx%d' % (int(q_tile), int(kv_tile))


def _relative_l2(actual, expected):
    error = (actual.float() - expected.float()).norm()
    return (error / expected.float().norm().clamp_min(1e-12)).item()


def _sparse_output_health(actual, expected):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return False, {
            'finite': False,
            'rel_l2': None,
            'max_abs': None,
            'passed': False,
        }
    relative_l2 = _relative_l2(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    passed = bool(
        relative_l2 < _SPARSE_REL_L2_TOLERANCE
        and max_abs < _SPARSE_MAX_ABS_TOLERANCE
    )
    return passed, {
        'finite': True,
        'rel_l2': round(relative_l2, 6),
        'max_abs': round(max_abs, 6),
        'passed': passed,
    }


def _carrier_lse_reference(carrier):
    batch, heads, q_length, _head_dim = carrier.q.shape
    kv_length = carrier.k.shape[-2]
    q_rows = torch.arange(q_length, device=carrier.q.device)
    q_scale_index = (q_rows // 32) * 8 + q_rows % 8
    q_scale = carrier.q_scale.reshape(batch, heads, -1).index_select(
        -1, q_scale_index
    )
    k_rows = torch.arange(kv_length, device=carrier.k.device)
    k_scale_index = (k_rows // carrier.cta_k) * 4 + (k_rows % 8) // 2
    k_scale = carrier.k_scale.reshape(batch, heads, -1).index_select(
        -1, k_scale_index
    )
    q = carrier.q.float() * q_scale.unsqueeze(-1)
    k = carrier.k.float() * k_scale.unsqueeze(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * carrier.attention_scale
    return torch.logsumexp(scores, dim=-1) * 1.4426950408889634


def _cache_key(device):
    major, minor = torch.cuda.get_device_capability(device)
    from .bootstrap import installed_build_id

    return 'sm%d%d|%s|%s|%s' % (
        major,
        minor,
        installed_build_id() or 'local',
        _SELFTEST_REVISION,
        torch.cuda.get_device_name(device),
    )


def _read_cache():
    try:
        return json.loads(_CACHE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _write_cache(cache):
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    except OSError:
        pass


def _samples(device, seed=20260823):
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(
            _BATCH, _Q_LEN, _HEADS, _HEAD_DIM,
            device=device, dtype=torch.bfloat16, generator=generator,
        ).transpose(1, 2)
        for _ in range(3)
    )


def _full_route(native, device, q_tile, kv_tile):
    kv_tiles = (_KV_LEN + kv_tile - 1) // kv_tile
    q_tiles = (_Q_LEN + q_tile - 1) // q_tile
    indices = torch.arange(kv_tiles, dtype=torch.int32, device=device)
    return native.BlockSparseRoute(
        indices=indices.view(1, 1, 1, -1)
        .expand(_BATCH, _HEADS, q_tiles, -1)
        .contiguous(),
        counts=torch.full(
            (_BATCH, _HEADS, q_tiles), kv_tiles,
            dtype=torch.int32, device=device,
        ),
        q_tile=q_tile,
        kv_tile=kv_tile,
        encoding='absolute',
    )


def run(device=None, *, verbose=False):
    """Return (production_passed, detail). Never raises on a kernel fault."""
    from . import int8_attention as native

    device = torch.device('cuda' if device is None else device)
    detail = {}
    try:
        q, k, v = _samples(device)

        # At 640 rows this uses CTA_K=64, exactly the carrier required by the
        # production 64x64 and 128x64 sparse geometries.
        dense_carrier = native.prequantize_int8_attention(q, k, v)
        dense = native.int8_attention_from_prequantized(dense_carrier)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float()
        ).to(torch.bfloat16)
        int8_error = _relative_l2(dense, reference)
        detail['int8_vs_sdpa_rel_l2'] = round(int8_error, 6)
        dense_ok = int8_error < _INT8_TOLERANCE and bool(torch.isfinite(dense).all())
        detail['dense_int8_passed'] = bool(dense_ok)

        carriers = {}
        dense_by_kv = {}
        parity = {}
        geometry_health = {}
        geometry_passed = {}
        for q_tile, kv_tile in _SPARSE_PARITY_GEOMETRIES:
            key = _geometry_key(q_tile, kv_tile)
            try:
                if kv_tile not in carriers:
                    carrier = native.prequantize_int8_attention(
                        q, k, v, cta_k=kv_tile
                    )
                    carriers[kv_tile] = carrier
                    dense_by_kv[kv_tile] = native.int8_attention_from_prequantized(
                        carrier
                    )
                carrier = carriers[kv_tile]
                route = _full_route(native, device, q_tile, kv_tile)
                # The self-test is the authority that populates the geometry
                # verdict, so bypass the runtime geometry gate here. Calling
                # the normal gated path would recurse back into this test.
                routed = native.block_sparse_int8_attention_from_prequantized(
                    carrier, route, validate_geometry=False
                )
                torch.cuda.synchronize(device)
                parity[key] = torch.equal(routed, dense_by_kv[kv_tile])
                geometry_passed[key], geometry_health[key] = (
                    _sparse_output_health(routed, dense_by_kv[kv_tile])
                )
            except Exception as error:  # geometry-local rejection when possible
                parity[key] = False
                geometry_passed[key] = False
                detail.setdefault('geometry_errors', {})[key] = (
                    '%s: %s' % (type(error).__name__, error)
                )
        detail['full_route_bit_identical'] = parity
        detail['full_route_numerical_health'] = geometry_health
        detail['full_route_passed'] = geometry_passed

        production_sparse_ok = any(
            bool(geometry_passed.get(_geometry_key(*geometry), False))
            for geometry in _PRODUCTION_GEOMETRIES
        )
        detail['production_sparse_passed'] = bool(production_sparse_ok)
        torch.cuda.synchronize(device)
    except Exception as error:  # noqa: BLE001 - reporting is the job
        detail['error'] = '%s: %s' % (type(error).__name__, error)
        detail.setdefault('dense_int8_passed', False)
        detail.setdefault('production_sparse_passed', False)
        return False, detail

    passed = bool(dense_ok and production_sparse_ok)
    if verbose:
        print('  dense INT8 vs FP32 SDPA : rel_l2 %.6f (tolerance %s) %s'
              % (int8_error, _INT8_TOLERANCE, 'ok' if dense_ok else 'FAIL'))
        print('  100%% route health       : %s'
              % ', '.join('%s=%s' % item for item in geometry_health.items()))
        print('  bit-identical diagnostic: %s'
              % ', '.join('%s=%s' % item for item in parity.items()))
        print('  production 64KV sparse  : %s'
              % ('ok' if production_sparse_ok else 'FAIL'))
    return passed, detail


def run_lse(device=None):
    """Validate the legacy 128x128 sparse-LSE path independently."""
    from . import int8_attention as native

    device = torch.device('cuda' if device is None else device)
    detail = {}
    try:
        q, k, v = _samples(device, seed=20260824)
        carrier = native.prequantize_int8_attention(q, k, v, cta_k=128)
        dense = native.int8_attention_from_prequantized(carrier)
        route = _full_route(native, device, 128, 128)
        routed = native.block_sparse_int8_attention_from_prequantized(
            carrier, route, validate_geometry=False
        )
        routed_lse_output, routed_lse = (
            native.block_sparse_int8_attention_with_lse_from_prequantized(
                carrier,
                route,
                validate_geometry=False,
            )
        )
        lse_reference = _carrier_lse_reference(carrier)
        lse_error = (routed_lse - lse_reference).abs().max().item()
        parity = torch.equal(routed, dense)
        output_parity = torch.equal(routed_lse_output, routed)
        finite = bool(torch.isfinite(routed_lse).all())
        torch.cuda.synchronize(device)
    except Exception as error:  # noqa: BLE001 - reporting is the job
        detail['error'] = '%s: %s' % (type(error).__name__, error)
        detail['passed'] = False
        return False, detail

    passed = bool(parity and output_parity and finite and lse_error < 0.02)
    detail.update(
        {
            '128x128_full_route_bit_identical': bool(parity),
            'sparse_lse_output_bit_identical': bool(output_parity),
            'sparse_lse_max_abs': round(lse_error, 6),
            'passed': passed,
        }
    )
    return passed, detail


def _load_result(device=None, *, force=False):
    global _result, _detail_result
    with _lock:
        if _result is not None and _detail_result is not None and not force:
            return _result, _detail_result
        if not torch.cuda.is_available() or not loader.is_available():
            _result, _detail_result = False, {}
            return _result, _detail_result

        key = _cache_key(device)
        cache = _read_cache()
        if not force and key in cache:
            detail = dict(cache[key])
            _result = bool(detail.get('passed'))
            _detail_result = detail
            return _result, _detail_result

        passed, detail = run(device)
        detail['passed'] = passed
        cache[key] = detail
        _write_cache(cache)
        _result, _detail_result = passed, detail
        if not passed:
            logging.warning(
                '%s NATIVE PRODUCTION SELF-TEST FAILED on %s - refusing the '
                'native Kitchen production path. Detail: %s',
                LOG_PREFIX, key, detail,
            )
        else:
            failed = [
                geometry for geometry, ok
                in detail.get('full_route_passed', {}).items()
                if not ok
            ]
            if failed:
                logging.debug(
                    '%s native sparse self-test disabled geometry %s on %s; '
                    'a validated fallback geometry remains available.',
                    LOG_PREFIX,
                    ', '.join(failed),
                    key,
                )
                logging.debug(
                    '%s native sparse self-test detail for %s: %s',
                    LOG_PREFIX,
                    key,
                    detail,
                )
        return _result, _detail_result


def check(device=None, *, force=False):
    """Cached gate for the production Kitchen carrier + 64KV sparse path."""
    return _load_result(device, force=force)[0]


def sparse_geometry_check(q_tile, kv_tile, device=None, *, force=False):
    """Whether one shipped sparse geometry passed its numerical health test."""
    geometry = (int(q_tile), int(kv_tile))
    if geometry not in _SPARSE_PARITY_GEOMETRIES:
        return False
    _passed, detail = _load_result(device, force=force)
    if not bool(detail.get('dense_int8_passed', False)):
        return False
    geometry_passed = detail.get('full_route_passed', {})
    return bool(geometry_passed.get(_geometry_key(*geometry), False))


def sparse_lse_check(device=None, *, force=False):
    """Whether the separately tested native 128x128 sparse-LSE path is healthy."""
    if not torch.cuda.is_available() or not loader.is_available():
        return False
    key = _cache_key(device)
    with _lse_lock:
        if not force and key in _lse_results:
            return bool(_lse_results[key]['passed'])
        passed, detail = run_lse(device)
        _lse_results[key] = detail
        if not passed:
            logging.warning(
                '%s NATIVE SPARSE-LSE SELF-TEST FAILED on %s. Detail: %s',
                LOG_PREFIX, key, detail,
            )
        return passed
