"""Synthetic CUDA A/B for bounded FinalLayer and BF16 MLP execution."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import types

PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACK), str(PACK.parents[1])]


def baseline_module(relative, ref):
    source = subprocess.check_output(
        ['git', 'show', '%s:%s' % (ref, relative)], cwd=PACK,
    ).decode('utf-8')
    name = 'h3_optimizations.memory._baseline_' + Path(relative).stem
    module = types.ModuleType(name)
    module.__package__ = 'h3_optimizations.memory'
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


def measure(torch, function, iterations):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        result = function()
        del result
    stop.record()
    stop.synchronize()
    return {
        'ms': start.elapsed_time(stop) / iterations,
        'incremental_peak_mib': (torch.cuda.max_memory_allocated() - before) / 2**20,
    }


def compare(torch, old, new, iterations):
    expected, actual = old(), new()
    expected = (expected,) if torch.is_tensor(expected) else expected
    actual = (actual,) if torch.is_tensor(actual) else actual
    equal = all(torch.equal(a, b) for a, b in zip(expected, actual))
    del expected, actual
    rounds = {'baseline': [], 'optimized': []}
    for order in (('baseline', 'optimized'), ('optimized', 'baseline'), ('baseline', 'optimized')):
        for key in order:
            rounds[key].append(measure(torch, old if key == 'baseline' else new, iterations))
    return {
        'bitwise_equal': equal,
        **{
            key: {field: statistics.median(r[field] for r in samples)
                  for field in ('ms', 'incremental_peak_mib')}
            for key, samples in rounds.items()
        },
        'rounds': rounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ref', default='c6bc3cfc58c18c3a6220aeea09116e1eebd91669')
    parser.add_argument('--rows', type=int, default=16384)
    parser.add_argument('--hidden', type=int, default=5376)
    parser.add_argument('--chunk', type=int, default=4096)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    import torch
    import torch.nn.functional as F
    import comfy.ops
    from h3_optimizations.memory import final_layer, forward
    from h3_optimizations.memory.config import ActivationMemoryConfig, MODE_BF16

    old_final = baseline_module('h3_optimizations/memory/final_layer.py', args.baseline_ref)
    torch.manual_seed(20261028)
    hidden = args.hidden
    x = torch.randn(args.rows, hidden, device='cuda', dtype=torch.bfloat16)
    norm_weight = torch.ones(hidden, device='cuda', dtype=torch.bfloat16)
    shift = torch.randn(4, hidden, device='cuda') * .1
    scale = torch.randn_like(shift) * .1

    class Head:
        weight_function = ()
        bias_function = ()

        def __init__(self, width, bank):
            self.out_features = width
            self.weight = torch.randn(width * bank, hidden, device='cuda') * .01
            self.bias = torch.randn(width * bank, device='cuda') * .01

        def __call__(self, value):
            return F.linear(value, self.weight, self.bias)

    split = args.rows - 257
    selector = torch.randint(4, (args.rows,), device='cuda')
    schedule = torch.tensor([1., .75, .25, 0.])
    results = {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'baseline_revision': subprocess.check_output(
            ['git', 'rev-parse', args.baseline_ref], cwd=PACK,
        ).decode().strip(),
        'rows': args.rows, 'hidden': hidden, 'chunk': args.chunk,
        'synthetic': True, 'cases': {},
    }
    for bank in (1, 8):
        layer = types.SimpleNamespace(
            norm=lambda rows: F.rms_norm(rows, (hidden,), norm_weight, 1e-6),
            adaln_proj=lambda _t: (shift, scale),
            video_out=Head(128, bank), audio_out=Head(64, bank),
        )
        for per_token in (False, True):
            video = (0, split, selector[:split] if per_token else 0)
            audio = (split, args.rows, selector[split:] if per_token else 1)
            call_args = (layer, x, None, video, audio, args.chunk,
                         torch.tensor(.75), schedule, (1., 2.))
            name = ('pdd' if bank > 1 else 'plain') + ('_tensor' if per_token else '_scalar')
            results['cases'][name] = compare(
                torch, lambda: old_final.chunked_final_layer(*call_args),
                lambda: final_layer.chunked_final_layer(*call_args), args.iterations,
            )
            print(name, json.dumps(results['cases'][name]), flush=True)
    del layer, x

    if args.mlp:
        old_forward = baseline_module('h3_optimizations/memory/forward.py', args.baseline_ref)
        ffn = 14336
        mlp = types.SimpleNamespace(
            fc1=comfy.ops.disable_weight_init.Linear(hidden, 2 * ffn, bias=False, device='cuda', dtype=torch.bfloat16),
            fc2=comfy.ops.disable_weight_init.Linear(ffn, hidden, bias=False, device='cuda', dtype=torch.bfloat16),
        )
        mlp.fc1.weight.requires_grad_(False)
        mlp.fc2.weight.requires_grad_(False)
        mlp.fc1.weight.normal_(0., .01)
        mlp.fc2.weight.normal_(0., .01)
        norm = torch.nn.RMSNorm(hidden, eps=1e-6, device='cuda', dtype=torch.bfloat16)
        norm.weight.requires_grad_(False)
        zero = torch.zeros(1, hidden, device='cuda')
        block = types.SimpleNamespace(
            norm1=norm, norm2=norm, mlp=mlp,
            adaln_proj=lambda _t: (zero, zero, zero, zero, zero, zero + .1),
            attn=lambda rows, **_kwargs: torch.zeros_like(rows),
            forward=lambda *_args: None,
        )
        config = ActivationMemoryConfig(mode=MODE_BF16, chunk_rows=args.chunk)
        previous = old_forward.make_forward(block, 0, config)
        current = forward.make_forward(block, 0, config)
        data = torch.randn(2 * args.chunk, hidden, device='cuda', dtype=torch.bfloat16) * .1
        segments = [(0, data.shape[0], 0)]
        results['cases']['bf16_mlp'] = compare(
            torch, lambda: previous(data.clone(), None, segments, None),
            lambda: current(data.clone(), None, segments, None), args.iterations,
        )
        print('bf16_mlp', json.dumps(results['cases']['bf16_mlp']), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
