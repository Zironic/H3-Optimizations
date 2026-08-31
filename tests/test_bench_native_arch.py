'''CPU contracts for the native architecture benchmark.'''

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest


BENCHMARK = Path(__file__).resolve().parents[1] / 'benchmarks' / 'bench_native_arch.py'
SPEC = importlib.util.spec_from_file_location('bench_native_arch', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class NativeArchitectureBenchmarkTests(unittest.TestCase):
    def test_defaults_match_the_h3_reference_shape(self):
        args = bench.parse_args(['--i-understand-this-uses-gpu'])
        self.assertEqual(args.sequence, 54_006)
        self.assertEqual(args.heads, 56)
        self.assertEqual(args.video_start, 256)
        self.assertEqual(args.densities, [1.0, 0.3])
        self.assertEqual((args.warmup, args.iterations), (2, 5))

    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])

    def test_geometry_keeps_context_dense_and_matches_router_rounding(self):
        geometry = bench.route_geometry(54_006, 256, 64, 64)
        self.assertEqual(geometry.pure_video_q_start, 4)
        self.assertEqual(geometry.pure_video_kv_start, 4)
        self.assertEqual(geometry.q_tiles, 844)
        self.assertEqual(geometry.kv_tiles, 844)
        self.assertEqual(
            bench.retained_video_tiles(0.3, geometry),
            252,
        )

    def test_arm_matrix_has_dense_reference_and_four_geometries_per_density(self):
        arms = bench.arm_specs([1.0, 0.3])
        self.assertEqual(len(arms), 9)
        self.assertEqual(arms[0], ('dense_128x128', 'dense', 1.0, 128, 128))
        self.assertEqual(
            {(arm[2], arm[3], arm[4]) for arm in arms[1:]},
            {
                (1.0, 128, 128), (1.0, 128, 64),
                (1.0, 64, 128), (1.0, 64, 64),
                (0.3, 128, 128), (0.3, 128, 64),
                (0.3, 64, 128), (0.3, 64, 64),
            },
        )

    def test_percentiles_are_interpolated(self):
        summary = bench.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary['median_ms'], 3.0)
        self.assertAlmostEqual(summary['p10_ms'], 1.4)
        self.assertAlmostEqual(summary['p90_ms'], 4.6)


if __name__ == '__main__':
    unittest.main()
