'''Benchmark the vendored native INT8 attention geometries on one GPU.'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import statistics
import sys
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))


GEOMETRIES = ((128, 128), (128, 64), (64, 128), (64, 64))


@dataclass(frozen=True)
class RouteGeometry:
    sequence: int
    q_tile: int
    kv_tile: int
    q_tiles: int
    kv_tiles: int
    pure_video_q_start: int
    pure_video_kv_start: int

    @property
    def pure_video_q_tiles(self):
        return self.q_tiles - self.pure_video_q_start

    @property
    def pure_video_kv_tiles(self):
        return self.kv_tiles - self.pure_video_kv_start


def route_geometry(sequence, video_start, q_tile, kv_tile):
    return RouteGeometry(
        sequence=int(sequence),
        q_tile=int(q_tile),
        kv_tile=int(kv_tile),
        q_tiles=(int(sequence) + int(q_tile) - 1) // int(q_tile),
        kv_tiles=(int(sequence) + int(kv_tile) - 1) // int(kv_tile),
        pure_video_q_start=(int(video_start) + int(q_tile) - 1) // int(q_tile),
        pure_video_kv_start=(int(video_start) + int(kv_tile) - 1) // int(kv_tile),
    )


def retained_video_tiles(density, geometry):
    return min(
        geometry.pure_video_kv_tiles,
        max(1, math.ceil(float(density) * geometry.pure_video_kv_tiles)),
    )


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples):
    return {
        'median_ms': statistics.median(samples),
        'p10_ms': percentile(samples, 0.1),
        'p90_ms': percentile(samples, 0.9),
        'samples_ms': samples,
    }


def density_label(density):
    percent = float(density) * 100.0
    return '%g%%' % percent


def arm_specs(densities):
    specs = [('dense_128x128', 'dense', 1.0, 128, 128)]
    ordered = [1.0]
    ordered.extend(value for value in densities if not math.isclose(value, 1.0))
    for density in ordered:
        label = density_label(density).replace('%', '')
        for q_tile, kv_tile in GEOMETRIES:
            specs.append(
                (
                    'sparse_%dx%d_kv%s' % (q_tile, kv_tile, label),
                    'sparse',
                    float(density),
                    q_tile,
                    kv_tile,
                )
            )
    return specs


def _coprime_stride(size, seed):
    if size <= 1:
        return 1
    candidate = int(seed) % size or 1
    while math.gcd(candidate, size) != 1:
        candidate += 1
        if candidate == size:
            candidate = 1
    return candidate


def build_route(torch, native, device, heads, geometry, density, seed):
    dense = torch.arange(geometry.kv_tiles, device=device, dtype=torch.int32)
    indices = dense.view(1, 1, 1, -1).expand(
        1, heads, geometry.q_tiles, -1
    ).clone()
    counts = torch.full(
        (1, heads, geometry.q_tiles),
        geometry.kv_tiles,
        device=device,
        dtype=torch.int32,
    )

    retained = retained_video_tiles(density, geometry)
    if retained < geometry.pure_video_kv_tiles:
        sparse_q_tiles = geometry.pure_video_q_tiles
        rows = heads * sparse_q_tiles
        positions = torch.arange(retained, device=device, dtype=torch.int64)
        row_ids = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(-1)
        stride = _coprime_stride(
            geometry.pure_video_kv_tiles,
            int(seed) + geometry.q_tile * 17 + geometry.kv_tile * 31,
        )
        offsets = (
            row_ids * 1103515245 + int(seed) * 12345
        ) % geometry.pure_video_kv_tiles
        selected = (
            positions.unsqueeze(0) * stride + offsets
        ) % geometry.pure_video_kv_tiles
        selected = selected.sort(dim=-1).values + geometry.pure_video_kv_start
        context = dense[:geometry.pure_video_kv_start].to(torch.int64)
        context = context.view(1, -1).expand(rows, -1)
        sparse_rows = torch.cat((context, selected), dim=-1).to(torch.int32)
        sparse_rows = sparse_rows.view(heads, sparse_q_tiles, -1)
        live = sparse_rows.shape[-1]
        indices[0, :, geometry.pure_video_q_start:, :live].copy_(sparse_rows)
        counts[..., geometry.pure_video_q_start:] = live

    route = native.BlockSparseRoute(
        indices=indices.contiguous(),
        counts=counts.contiguous(),
        q_tile=geometry.q_tile,
        kv_tile=geometry.kv_tile,
        encoding='absolute',
    )
    return route.for_kernel(), retained


def benchmark_call(torch, call, warmup, iterations):
    for _ in range(warmup):
        output = call()
        del output
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = call()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        del output
    return summarize(samples)


def parse_capability(value):
    parts = str(value).split('.')
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError('capability must look like 9.0')
    return tuple(int(part) for part in parts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Benchmark native 128x128, 128x64, 64x128, and 64x64 INT8 attention.'
        )
    )
    parser.add_argument('--expected-capability', type=parse_capability)
    parser.add_argument('--sequence', type=int, default=54_006)
    parser.add_argument('--heads', type=int, default=56)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--densities', type=float, nargs='+', default=[1.0, 0.3])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or not 0 < args.video_start < args.sequence:
        parser.error('sequence/video-start arguments are invalid')
    if args.heads <= 0:
        parser.error('--heads must be positive')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('warmup/iteration arguments are invalid')
    if any(not 0.01 <= density <= 1.0 for density in args.densities):
        parser.error('--densities values must be in [0.01, 1]')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def main(argv=None):
    args = parse_args(argv)

    import torch

    from h3_optimizations import native
    from h3_optimizations.native import selftest

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    capability = tuple(torch.cuda.get_device_capability(device))
    if args.expected_capability is not None and capability != args.expected_capability:
        raise SystemExit(
            'expected compute capability %d.%d but worker has %d.%d'
            % (*args.expected_capability, *capability)
        )

    passed, selftest_detail = selftest.run(device)
    result = {
        'status': 'selftest_passed' if passed else 'selftest_failed',
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(capability),
        },
        'torch_version': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'native_abi': native.ABI_VERSION,
        'native_route_encoding': native.route_encoding(),
        'native_selftest': selftest_detail,
        'config': {
            'sequence': args.sequence,
            'heads': args.heads,
            'head_dim': 128,
            'video_start': args.video_start,
            'densities': args.densities,
            'warmup': args.warmup,
            'iterations': args.iterations,
            'seed': args.seed,
        },
        'arms': [],
        'recommendations': {},
    }
    if not passed:
        print(json.dumps(result, sort_keys=True))
        return 2

    generator = torch.Generator(device=device).manual_seed(args.seed)
    q, k, v = (
        torch.randn(
            1,
            args.heads,
            args.sequence,
            128,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    carriers = {
        128: native.prequantize_int8_attention(q, k, v, cta_k=128),
        64: native.prequantize_int8_attention(q, k, v, cta_k=64),
    }
    del q, k, v
    torch.cuda.synchronize(device)

    by_density = {}
    for label, kind, density, q_tile, kv_tile in arm_specs(args.densities):
        carrier = carriers[kv_tile]
        arm = {
            'label': label,
            'kind': kind,
            'density': density,
            'q_tile': q_tile,
            'kv_tile': kv_tile,
        }
        if kind == 'dense':
            call = lambda carrier=carrier: native.int8_attention_from_prequantized(
                carrier
            )
        else:
            geometry = route_geometry(
                args.sequence, args.video_start, q_tile, kv_tile
            )
            route, retained = build_route(
                torch, native, device, args.heads, geometry, density, args.seed
            )
            arm['route'] = {
                'q_tiles': geometry.q_tiles,
                'kv_tiles': geometry.kv_tiles,
                'dense_q_tiles': geometry.pure_video_q_start,
                'sparse_q_tiles': (
                    geometry.pure_video_q_tiles
                    if retained < geometry.pure_video_kv_tiles
                    else 0
                ),
                'non_video_kv_tiles': geometry.pure_video_kv_start,
                'retained_video_kv_tiles': retained,
                'video_kv_tiles': geometry.pure_video_kv_tiles,
                'encoding': route.encoding,
            }
            call = lambda carrier=carrier, route=route: (
                native.block_sparse_int8_attention_from_prequantized(
                    carrier, route
                )
            )
        arm['timing'] = benchmark_call(
            torch, call, args.warmup, args.iterations
        )
        result['arms'].append(arm)
        del call
        if kind == 'sparse':
            by_density.setdefault(density_label(density), []).append(arm)
            del route

    for label, arms in by_density.items():
        winner = min(arms, key=lambda arm: arm['timing']['median_ms'])
        result['recommendations'][label] = '%dx%d' % (
            winner['q_tile'], winner['kv_tile']
        )
    preferred_density = density_label(
        min(float(density) for density in args.densities)
    )
    result['recommended_geometry'] = result['recommendations'][preferred_density]
    result['status'] = 'completed'
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
