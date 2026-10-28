"""Compare staged packed Q and streamed residuals on one real H3 block."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PACK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(PACK)]

from bench_h3_block import build_block, tensor_error
from bench_chunked_kitchen_qkv import benchmark_case, make_rope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sequence', type=int, default=16384)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--output', required=True)
    parser.add_argument('--offload-attention', action='store_true')
    parser.add_argument('--compile-eager', action='store_true')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]

    import torch
    from h3_optimizations.attention_forward import make_forward as make_attention
    from h3_optimizations.kitchen_qkv import (
        ChunkedKitchenAttentionBackend, ChunkedKitchenQKVProjector,
        DENSE_KITCHEN_BACKEND, H3_ATTENTION_BACKEND_KEY,
    )
    from h3_optimizations.memory.config import ActivationMemoryConfig, MODE_CONVROT_2SLICE
    from h3_optimizations.memory.forward import make_forward

    device = torch.device('cuda')
    torch.manual_seed(1234)
    block, prefix, hidden, t_dim = build_block(torch, args.checkpoint, 0, device)
    if args.offload_attention:
        block.attn.qkv_proj.to('cpu')
        block.attn.out_proj.to('cpu')
    x = torch.randn(args.sequence, hidden, device=device, dtype=torch.bfloat16) * 0.1
    t_emb = torch.randn(1, t_dim, device=device, dtype=torch.float32) * 0.1
    rope = make_rope(torch, args.sequence, device)
    segments = [(0, 512, 1), (512, 768, 2), (768, args.sequence, 0)]
    options = {H3_ATTENTION_BACKEND_KEY: DENSE_KITCHEN_BACKEND}
    forward = make_attention(
        block.attn, 0,
        backend=ChunkedKitchenAttentionBackend(stream_output=True, query_chunk_rows=4096),
        projector=ChunkedKitchenQKVProjector(
            chunk_rows=4096, stream_output=True, streamed_q=True, strided_qk_input=True,
        ),
    )
    stream = forward._h3_optimizations_projected_rows
    block.attn.forward = forward
    run_block = make_forward(block, 0, ActivationMemoryConfig(mode=MODE_CONVROT_2SLICE))
    if args.compile_eager:
        run_block = torch.compile(run_block, backend='eager')

    def run():
        return run_block(x.clone(), t_emb, segments, rope, transformer_options=options)

    del forward._h3_optimizations_projected_rows
    reference = run()
    forward._h3_optimizations_projected_rows = stream
    if args.compile_eager:
        torch.compiler.reset()
    actual = run()
    error = tensor_error(torch, reference, actual)
    del reference, actual
    results = {}
    for name in ('baseline', 'staged', 'baseline_repeat', 'staged_repeat'):
        if name.startswith('baseline'):
            del forward._h3_optimizations_projected_rows
        else:
            forward._h3_optimizations_projected_rows = stream
        if args.compile_eager:
            torch.compiler.reset()
        results[name] = benchmark_case(torch, run, 1, args.iterations, device)
        print(name, results[name], flush=True)
    result = {'sequence': args.sequence, 'offload_attention': args.offload_attention, 'gpu': torch.cuda.get_device_name(), 'checkpoint_prefix': prefix,
              'parity': error, 'results': results}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
