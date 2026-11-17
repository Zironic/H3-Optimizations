'''Calibrated per-head video budgets for the threshold sparse route.

The threshold router keeps a video KV tile when ``q.k > q.kbar + tau *
sqrt(q^T Sigma q)``, with one ``tau`` for every head from the step's budget.
Heads are not equally sensitive to the tiles they lose, so a calibrated table
gives each head of a layer its own ``tau_h`` while the layer's heads together
spend the same realized tiles the shared tau would (compute-matched per layer
and step). ``tau_h`` is applied by scaling that head's K covariance in the
route plan by ``(tau_h / tau) ** 2``; the native router and kernels are
unchanged.

The bundled table (``data/head_budgets_t32_v1.json.gz``) was calibrated in
H3-Extended on 64Q x 32KV threshold routing with the 1x8x8 video token order,
over 14 T2VA, I2V, FL2VA and Ref2VA runs. It applies only to that geometry
and to H3's 56-head, 50-layer transformer; anything else keeps the shared tau.

The allocation below is pure Python and mirrors ``h3_head_budget.profile`` in
H3-Extended exactly, so a route here gets the fractions that were validated
there.
'''

from __future__ import annotations

import gzip
import json
import logging
import math
from pathlib import Path
from statistics import NormalDist
import threading

import torch


LOG = logging.getLogger(__name__)

PROFILE_FORMAT = 'h3_head_budget_profile'
PROFILE_VERSION = 1
ROUTE_THRESHOLD = 'threshold'
SIGN_MARGIN = 0.02
BUNDLED_PROFILE = Path(__file__).resolve().parents[2] / 'data' / 'head_budgets_t32_v1.json.gz'
# A forward whose transformer options set this runs the shared tau (diagnostic arms).
HEAD_BUDGETS_OFF_KEY = 'h3_sparse_head_budgets_off'


class HeadBudgetError(ValueError):
    pass


def threshold_tau(fraction):
    '''``threshold_route.budget_tau``: tau keeping ~``fraction`` of the tiles.'''
    fraction = min(max(float(fraction), 1e-6), 1.0 - 1e-6)
    return NormalDist().inv_cdf(1.0 - fraction)


# --------------------------------------------------------------------------
# Piecewise-linear allocation (identical to H3-Extended h3_head_budget.profile)


def interpolate(points, x):
    '''Piecewise-linear value of ascending ``points`` at ``x``; clamped at the ends.'''
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def _monotone(values):
    '''Running maximum: realized density can only grow with the fraction.'''
    result = []
    best = 0.0
    for value in values:
        best = max(best, float(value))
        result.append(best)
    return result


def lower_hull(points):
    '''Lower convex hull of points sorted by x (Andrew's monotone chain).'''
    hull = []
    for point in points:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1) <= 0.0:
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def _clip(points, low, high):
    inside = [(x, y) for x, y in points if low < x < high]
    return [(low, interpolate(points, low))] + inside + [(high, interpolate(points, high))]


def allocate_continuous(curves, total, lows, highs):
    '''Real-valued ``x_h`` minimizing ``sum_h e_h(x_h)`` with ``sum x_h == total``.

    Greedy exchange over the lower convex hull of each clipped curve; exact for
    piecewise-linear convex curves. ``None`` when the rails cannot meet ``total``.
    '''
    total = float(total)
    if sum(lows) > total + 1e-9 or sum(highs) < total - 1e-9:
        return None
    segments = []
    for head, (points, low, high) in enumerate(zip(curves, lows, highs)):
        if high <= low:
            continue
        hull = lower_hull(_clip(points, float(low), float(high)))
        for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
            length = x1 - x0
            if length > 0.0:
                segments.append((-(y0 - y1) / length, head, length))
    segments.sort()
    values = [float(low) for low in lows]
    remaining = total - sum(values)
    for _negative_gain, head, length in segments:
        if remaining <= 0.0:
            break
        take = min(length, remaining)
        values[head] += take
        remaining -= take
    return values


def inverse(points, y):
    '''``x`` where ascending-in-both ``points`` reach ``y``; clamped at the ends.'''
    return interpolate([(value, x) for x, value in points], y)


def density_points(fractions, density):
    '''Ascending ``[(fraction, realized density), ...]`` including both ends.'''
    return [(0.0, 0.0)] + list(zip(fractions, _monotone(density))) + [(1.0, 1.0)]


def threshold_error_points(fractions, errors, density, pure_kv_tiles):
    '''``[(realized tiles, error), ...]`` for one threshold head, ending at (T, 0).'''
    total = float(int(pure_kv_tiles))
    points = []
    for value, error in zip(_monotone(density), errors):
        x = value * total
        if points and x <= points[-1][0]:
            if error < points[-1][1]:
                points[-1] = (points[-1][0], float(error))
            continue
        points.append((x, float(error)))
    if not points or points[-1][0] < total:
        points.append((total, 0.0))
    return points


def sign_rails(fraction, low, high, margin=SIGN_MARGIN):
    '''Per-head fraction rails that keep ``tau_h`` on ``tau``'s side of zero.

    ``None`` when ``fraction`` is too close to 0.5 to scale tau per head.
    '''
    fraction = float(fraction)
    if abs(fraction - 0.5) < margin:
        return None
    if fraction < 0.5:
        low, high = low, min(high, 0.5 - margin)
    else:
        low, high = max(low, 0.5 + margin), high
    if high <= low or not low <= fraction <= high:
        return None
    return low, high


def allocate_threshold(fractions, errors, densities, pure_kv_tiles, fraction, low, high):
    '''Per-head fractions matching the shared-tau route's predicted realized tiles, or ``None``.'''
    rails = sign_rails(fraction, low, high)
    if rails is None:
        return None
    low, high = rails
    total_tiles = float(int(pure_kv_tiles))
    curves = [threshold_error_points(fractions, row, density, total_tiles)
              for row, density in zip(errors, densities)]
    dpoints = [density_points(fractions, density) for density in densities]
    budget = sum(interpolate(points, fraction) for points in dpoints) * total_tiles
    lows = [interpolate(points, low) * total_tiles for points in dpoints]
    highs = [interpolate(points, high) * total_tiles for points in dpoints]
    targets = allocate_continuous(curves, budget, lows, highs)
    if targets is None:
        return None
    head_fractions = []
    for points, target in zip(dpoints, targets):
        value = inverse(points, target / total_tiles)
        head_fractions.append(min(high, max(low, value)))
    return tuple(head_fractions)


# --------------------------------------------------------------------------
# Profile


class HeadBudgetProfile:
    '''A threshold head-budget table: per-(layer, head) error and realized-density curves.'''

    def __init__(self, data):
        if data.get('format') != PROFILE_FORMAT or int(data.get('version', 0)) != PROFILE_VERSION:
            raise HeadBudgetError('not a version %d H3 head budget profile' % PROFILE_VERSION)
        if data.get('route_mode') != ROUTE_THRESHOLD:
            raise HeadBudgetError('only threshold head budget profiles are supported')
        self.name = str(data.get('name', 'unnamed'))
        self.q_tile = int(data['q_tile'])
        self.kv_tile = int(data['kv_tile'])
        self.heads = int(data['heads'])
        self.layer_count = int(data['layer_count'])
        self.video_token_order = data.get('video_token_order')
        self.fractions = tuple(float(value) for value in data['fractions'])
        if list(self.fractions) != sorted(self.fractions) or len(self.fractions) < 2:
            raise HeadBudgetError('profile fractions must be ascending and at least two')
        self.layers = {}
        self.density = {}
        for key, rows in data['layers'].items():
            layer = int(key)
            dense_rows = data['density'][key]
            for table in (rows, dense_rows):
                if len(table) != self.heads or any(len(row) != len(self.fractions) for row in table):
                    raise HeadBudgetError('profile layer %d has the wrong shape' % layer)
            self.layers[layer] = tuple(tuple(max(0.0, float(v)) for v in row) for row in rows)
            self.density[layer] = tuple(tuple(min(1.0, max(0.0, float(v))) for v in row) for row in dense_rows)
        self.min_fraction = self.fractions[0]  # never extrapolate below the smallest calibrated point
        self.max_fraction = 1.0
        self._cache = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = path.read_bytes()
        if path.suffix == '.gz':
            raw = gzip.decompress(raw)
        return cls(json.loads(raw.decode('utf-8')))

    def matches(self, *, q_tile, kv_tile, heads, video_token_order=None):
        return (
            int(q_tile) == self.q_tile
            and int(kv_tile) == self.kv_tile
            and int(heads) == self.heads
            and (video_token_order is None or self.video_token_order is None
                 or video_token_order == self.video_token_order)
        )

    def fractions_for(self, layer, pure_kv_tiles, retained):
        '''Per-head fractions for one layer and step, or ``None`` to keep the shared tau.'''
        layer, total, retained = int(layer), int(pure_kv_tiles), int(retained)
        key = (layer, total, retained)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        result = None
        if layer in self.layers and 0 < retained < total:
            result = allocate_threshold(
                self.fractions, self.layers[layer], self.density[layer],
                total, retained / total, self.min_fraction, self.max_fraction,
            )
        with self._lock:
            self._cache[key] = result
        return result


_bundled = {'profile': None, 'error': None}
_bundled_lock = threading.Lock()


def bundled_profile():
    '''The shipped calibration, loaded once; ``None`` (logged once) if it cannot be read.'''
    with _bundled_lock:
        if _bundled['profile'] is None and _bundled['error'] is None:
            try:
                _bundled['profile'] = HeadBudgetProfile.load(BUNDLED_PROFILE)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                _bundled['error'] = str(exc)
                LOG.warning('[H3] per-head budgets unavailable, using one shared tau: %s', exc)
        return _bundled['profile']


def apply_head_tau(plan, layer_index, *, q_tile, kv_tile, video_token_order=None, profile=None):
    '''Give each head of a threshold route plan its calibrated tau, in place.

    Returns the per-head fractions applied, or ``None`` when the plan keeps
    one shared tau (not a threshold route, a dense step, an uncalibrated
    geometry or layer, or a budget too close to 0.5 to scale tau per head).
    '''
    if plan is None or getattr(plan, 'mode', None) != ROUTE_THRESHOLD or plan.k_cov is None:
        return None
    if getattr(plan, 'head_fractions', None) is not None:
        raise HeadBudgetError('per-head budgets were already applied to this route plan')
    if layer_index is None:
        return None
    profile = bundled_profile() if profile is None else profile
    if profile is None or not profile.matches(q_tile=q_tile, kv_tile=kv_tile, heads=plan.heads,
                                              video_token_order=video_token_order):
        return None
    geometry = plan.geometry
    total = int(geometry.pure_video_kv_tiles)
    fractions = profile.fractions_for(layer_index, total, plan.retained)
    if fractions is None:
        return None
    tau = float(plan.tau)
    scales = [threshold_tau(value) / tau for value in fractions]
    if min(scales) <= 0.0:
        # sign_rails keep every head on tau's side of zero; refuse rather than invert a threshold
        raise HeadBudgetError('per-head tau crossed zero')
    factor = torch.tensor([value * value for value in scales], dtype=torch.float32, device=plan.k_cov.device)
    plan.k_cov = (plan.k_cov.float() * factor.view(1, -1, 1, 1)).to(plan.k_cov.dtype).contiguous()
    plan.head_fractions = fractions
    return fractions


__all__ = [
    'BUNDLED_PROFILE',
    'HEAD_BUDGETS_OFF_KEY',
    'HeadBudgetError',
    'HeadBudgetProfile',
    'allocate_threshold',
    'apply_head_tau',
    'bundled_profile',
    'threshold_tau',
]
