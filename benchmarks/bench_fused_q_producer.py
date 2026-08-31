'''Compare the current streamed H3 Q producer with the experimental fused Config-0 epilogue.'''

from __future__ import annotations

import argparse
from functools import partial
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from bench_chunked_kitchen_qkv import (
    HEAD_DIM,
    build_attention,
    chunked_kitchen_carrier,
    make_rope,
    output_error_metrics,
    project_qkv,
    resolve_checkpoint,
    weight_contract,
)


CONFIG = 0
ROT_DIM = 96
FUSED_VARIANTS = (
    ('exact_256', 0, 256),
    ('rope_fp32_256', 1, 256),
    ('norm_rope_fp32_256', 2, 256),
    ('exact_128', 0, 128),
    ('rope_fp32_128', 1, 128),
    ('norm_rope_fp32_128', 2, 128),
)


class GpuSampler:
    def __init__(self, interval_ms=100):
        self.interval_ms = int(interval_ms)
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        self.process = subprocess.Popen(
            [
                'nvidia-smi',
                '--query-gpu=memory.used,power.draw,clocks.sm',
                '--format=csv,noheader,nounits',
                '-lms',
                str(self.interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.samples.append(tuple(float(value.strip()) for value in line.split(',')))
            except ValueError:
                continue

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def report(self):
        names = ('memory_used_mib', 'power_w', 'clocks_sm_mhz')
        return {
            'sample_count': len(self.samples),
            **{
                name: {
                    'min': min(values),
                    'median': statistics.median(values),
                    'max': max(values),
                }
                for index, name in enumerate(names)
                if (values := [sample[index] for sample in self.samples if len(sample) == 3])
            },
        }


def summarize(samples):
    return {
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'max_ms': max(samples),
        'samples_ms': samples,
    }


def alternating_time(torch, cases, warmup, iterations, repeats, device):
    for _ in range(warmup):
        for _name, function in cases:
            for _repeat in range(repeats):
                function()
    torch.cuda.synchronize(device)

    samples = {name: [] for name, _function in cases}
    wall_samples = {name: [] for name, _function in cases}
    for iteration in range(iterations):
        order = cases if iteration % 2 == 0 else tuple(reversed(cases))
        for name, function in order:
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _repeat in range(repeats):
                function()
            stop.record()
            stop.synchronize()
            samples[name].append(float(start.elapsed_time(stop)) / repeats)
            wall_samples[name].append((time.perf_counter() - wall_start) * 1000.0 / repeats)
    return {
        name: {
            'cuda': summarize(samples[name]),
            'wall': summarize(wall_samples[name]),
        }
        for name, _function in cases
    }


def carrier_scale_indices(torch, rows, device):
    position = torch.arange(rows, device=device, dtype=torch.int64)
    within = position % 128
    return (position // 128) * 32 + (within // 32) * 8 + within % 8


def parity_report(torch, baseline_q, baseline_scale, fused_q, fused_scale):
    indices = carrier_scale_indices(torch, int(baseline_q.shape[1]), baseline_q.device)
    baseline_dequant = baseline_q.float() * baseline_scale[:, indices, None]
    fused_dequant = fused_q.float() * fused_scale[:, indices, None]
    difference = fused_dequant - baseline_dequant
    reference_norm = torch.linalg.vector_norm(baseline_dequant)
    return {
        'q_exact': bool(torch.equal(baseline_q, fused_q)),
        'q_mismatch_fraction': float((baseline_q != fused_q).float().mean().item()),
        'scale_exact': bool(torch.equal(baseline_scale, fused_scale)),
        'scale_max_abs': float((fused_scale - baseline_scale).abs().max().item()),
        'dequant_max_abs': float(difference.abs().max().item()),
        'dequant_mean_abs': float(difference.abs().mean().item()),
        'dequant_rel_l2': float(
            (torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-12)).item()
        ),
    }


def tensor_difference(torch, actual, reference):
    difference = actual.float() - reference.float()
    return {
        'exact': bool(torch.equal(actual, reference)),
        'max_abs': float(difference.abs().max().item()),
        'rel_l2': float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            ).item()
        ),
    }


def dequantize_carrier(torch, q, scale):
    indices = carrier_scale_indices(torch, int(q.shape[1]), q.device)
    return q.float() * scale[:, indices, None]


def oracle_error(torch, actual, reference):
    difference = actual.float() - reference.float()
    return {
        'max_abs': float(difference.abs().max().item()),
        'mean_abs': float(difference.abs().mean().item()),
        'rel_l2': float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            ).item()
        ),
    }


def bf16_oracle_carrier(torch, q_hnd, full_k_length):
    q = q_hnd.float()
    if full_k_length <= 256:
        h4 = torch.tensor(
            ((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1)),
            dtype=torch.float32,
            device=q.device,
        ) * 0.5
        return torch.matmul(q.reshape(*q.shape[:-1], 32, 4), h4).reshape(q.shape)

    h2 = torch.tensor(((1, 1), (1, -1)), dtype=torch.float32, device=q.device)
    hadamard = h2
    for _ in range(6):
        hadamard = torch.kron(hadamard, h2)
    hadamard *= 128 ** -0.5
    words = (0x1035997B, 0x8087F5EE, 0xEE2E4E1A, 0x71132418)
    signs = [
        1.0 if (words[index // 32] >> (index % 32)) & 1 else -1.0
        for index in range(128)
    ]
    sign = torch.tensor(signs, dtype=torch.float32, device=q.device)
    return torch.matmul(q * sign, hadamard)


def bf16_attention_oracle(torch, module, x, rope):
    q, k, v = project_qkv(torch, module, x, rope)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=HEAD_DIM ** -0.5,
    )
    del q, k, v
    return output


def fused_cases(experiment, method_name):
    method = getattr(experiment, method_name)
    return tuple(
        (name, partial(method, math_mode=math_mode, tile_n=tile_n))
        for name, math_mode, tile_n in FUSED_VARIANTS
    )


def route_report(route):
    counts = route.counts
    kv_tiles = int(route.indices.shape[-1])
    selected = int(counts.sum().item())
    possible = int(counts.numel()) * kv_tiles
    return {
        'encoding': str(route.encoding),
        'q_tile': int(route.q_tile),
        'kv_tile': int(route.kv_tile),
        'q_tiles': int(counts.shape[-1]),
        'kv_tiles': kv_tiles,
        'selected_blocks': selected,
        'possible_blocks': possible,
        'executed_density': selected / possible,
        'count_min': int(counts.min().item()),
        'count_max': int(counts.max().item()),
    }


def route_overlap_report(torch, baseline, candidate):
    baseline = baseline.to_absolute()
    candidate = candidate.to_absolute()
    active_columns = max(
        int(baseline.counts.max().item()),
        int(candidate.counts.max().item()),
    )
    baseline_indices = baseline.indices[..., :active_columns]
    candidate_indices = candidate.indices[..., :active_columns]
    positions = torch.arange(active_columns, device=baseline.indices.device)
    baseline_live = positions < baseline.counts.unsqueeze(-1)
    candidate_live = positions < candidate.counts.unsqueeze(-1)
    matched = (
        (baseline_indices.unsqueeze(-1) == candidate_indices.unsqueeze(-2))
        & baseline_live.unsqueeze(-1)
        & candidate_live.unsqueeze(-2)
    ).any(dim=-1)
    overlap = int(matched.sum().item())
    baseline_selected = int(baseline.counts.sum().item())
    candidate_selected = int(candidate.counts.sum().item())
    union = baseline_selected + candidate_selected - overlap
    return {
        'selected_block_overlap_fraction': overlap / max(baseline_selected, 1),
        'selected_block_jaccard': overlap / max(union, 1),
        'changed_selected_block_fraction_vs_baseline': (
            baseline_selected - overlap
        ) / max(baseline_selected, 1),
    }


class FusedQExperiment:
    def __init__(
        self,
        torch,
        module,
        x,
        rope,
        full_k_length,
        arm='both',
        defer_q_buffers=False,
    ):
        import torch.nn.functional as torch_functional
        from comfy_kitchen.backends import cuda
        from h3_optimizations.native import loader
        from h3_optimizations.native.int8_attention import _DTYPE_TO_CODE, _ptr, _stream

        self.torch = torch
        self.torch_functional = torch_functional
        self.cuda = cuda
        self.extension = cuda._C
        self.native_loader = loader
        self.dtype_codes = _DTYPE_TO_CODE
        self.ptr = _ptr
        self.native_stream = _stream
        self.module = module
        self.x = x
        self.rope = rope
        self.rows = int(x.shape[0])
        self.arm = str(arm)
        if self.arm not in ('both', 'baseline', 'fused'):
            raise ValueError('arm must be both, baseline, or fused')
        self.full_k_length = int(full_k_length)
        self.hidden = int(x.shape[1])
        self.heads = int(module.heads)
        self.inner = self.heads * HEAD_DIM
        self.stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

        self.config0 = getattr(self.extension, 'cutlass_int8_dequant_config', None)
        self.fused = getattr(self.extension, 'cutlass_h3_fused_q', None)
        if self.config0 is None or self.fused is None:
            raise RuntimeError('the source Kitchen extension lacks the experiment bindings')

        weight = module.qkv_proj.weight
        self.groupsize = int(weight._params.convrot_groupsize)
        self.weight_qdata = weight._qdata[:self.inner].contiguous()
        weight_scale = weight._params.scale.to(torch.float32).reshape(-1)
        if int(weight_scale.numel()) != int(weight._qdata.shape[0]):
            raise RuntimeError('the experiment requires one ConvRot scale per output row')
        self.weight_scale = weight_scale[:self.inner].contiguous()
        self.norm = module.q_norm.weight.contiguous()
        self.freqs = rope[0, :, 0].contiguous()
        self.scale_count = ((self.rows + 127) // 128) * 32
        self.summary_tiles = (self.rows + 63) // 64
        self.act_qdata = None
        self.act_scale = None
        self.projected_q = None
        self.baseline_q = None
        self.baseline_scale = None
        self.fused_q = None
        self.fused_scale = None
        self.baseline_summary = None
        self.fused_summary = None
        self.debug_q = None
        self.no_debug = None
        self.normalized_q = None
        if not defer_q_buffers:
            self.allocate_q_buffers()
        self.global_carrier = None
        self.router = None
        self.route_plan = None
        self.common_route = None
        self.q_start = None
        self.sparse_metadata = None

    def allocate_q_buffers(self):
        self.act_qdata = self.torch.empty(
            (self.rows, self.hidden), dtype=self.torch.int8, device=self.x.device
        )
        self.act_scale = self.torch.empty(
            (self.rows, 1), dtype=self.torch.float32, device=self.x.device
        )
        baseline = self.arm in ('both', 'baseline')
        fused = self.arm in ('both', 'fused')
        q_shape = (self.heads, self.rows, HEAD_DIM)
        scale_shape = (self.heads, self.scale_count)
        summary_shape = (self.heads, self.summary_tiles, HEAD_DIM)
        self.projected_q = (
            self.torch.empty(
                (self.rows, self.inner),
                dtype=self.torch.bfloat16,
                device=self.x.device,
            )
            if baseline else None
        )
        self.baseline_q = (
            self.torch.empty(q_shape, dtype=self.torch.int8, device=self.x.device)
            if baseline else None
        )
        self.baseline_scale = (
            self.torch.empty(
                scale_shape, dtype=self.torch.float32, device=self.x.device
            )
            if baseline else None
        )
        self.fused_q = (
            self.torch.empty(q_shape, dtype=self.torch.int8, device=self.x.device)
            if fused else None
        )
        self.fused_scale = (
            self.torch.empty(
                scale_shape, dtype=self.torch.float32, device=self.x.device
            )
            if fused else None
        )
        self.baseline_summary = (
            self.torch.empty(
                summary_shape, dtype=self.torch.bfloat16, device=self.x.device
            )
            if baseline else None
        )
        self.fused_summary = (
            self.torch.empty(
                summary_shape, dtype=self.torch.bfloat16, device=self.x.device
            )
            if fused else None
        )
        self.debug_q = (
            self.torch.empty(
                (self.rows, self.inner),
                dtype=self.torch.bfloat16,
                device=self.x.device,
            )
            if self.arm == 'both' else None
        )
        self.no_debug = self.torch.empty(
            0, dtype=self.torch.bfloat16, device=self.x.device
        )

    def quantize_activation(self):
        self.extension.quantize_int8_rowwise_convrot64(
            self.cuda._wrap_for_dlpack(self.x),
            self.cuda._wrap_for_dlpack(self.act_qdata),
            self.cuda._wrap_for_dlpack(self.act_scale),
            self.groupsize,
            False,
            0,
            0,
            self.stream_ptr,
        )

    def config0_gemm(self):
        used = self.config0(
            self.cuda._wrap_for_dlpack(self.act_qdata),
            self.cuda._wrap_for_dlpack(self.weight_qdata),
            self.cuda._wrap_for_dlpack(self.act_scale),
            self.cuda._wrap_for_dlpack(self.weight_scale),
            self.cuda._wrap_for_dlpack(self.projected_q),
            self.cuda.DTYPE_TO_CODE[self.torch.bfloat16],
            CONFIG,
            self.stream_ptr,
        )
        if not used:
            raise RuntimeError('Kitchen declined Config 0')

    def pack_baseline(self, q_hnd):
        library = self.native_loader.load()
        self.native_loader.check(
            library.h3_int8_quantize_q_chunk(
                self.ptr(q_hnd),
                self.ptr(self.baseline_q),
                self.ptr(self.baseline_scale),
                1,
                self.heads,
                self.rows,
                self.rows,
                0,
                HEAD_DIM,
                self.full_k_length,
                q_hnd.stride(0),
                q_hnd.stride(1),
                q_hnd.stride(2),
                self.dtype_codes[q_hnd.dtype],
                self.native_stream(),
            ),
            'quantize_q_chunk',
        )

    def baseline_from_quantized(self):
        self.config0_gemm()
        projected = self.projected_q.view(1, self.rows, self.heads, HEAD_DIM)
        self.normalized_q = self.torch_functional.rms_norm(
            projected,
            (HEAD_DIM,),
            weight=self.norm,
            eps=self.module.q_norm.eps,
        )
        self.cuda.apply_rope_split_half1_(
            self.normalized_q[..., :ROT_DIM],
            self.rope,
        )
        q_hnd = self.normalized_q.transpose(1, 2)
        self.baseline_summary = q_hnd.reshape(
            1, self.heads, self.summary_tiles, 64, HEAD_DIM
        ).mean(dim=-2).squeeze(0)
        self.pack_baseline(q_hnd)
        return self.baseline_q

    def fused_from_quantized(self, math_mode=0, tile_n=256, debug=False):
        debug_output = self.debug_q if debug else self.no_debug
        used = self.fused(
            self.cuda._wrap_for_dlpack(self.act_qdata),
            self.cuda._wrap_for_dlpack(self.weight_qdata),
            self.cuda._wrap_for_dlpack(self.act_scale),
            self.cuda._wrap_for_dlpack(self.weight_scale),
            self.cuda._wrap_for_dlpack(self.norm),
            self.cuda._wrap_for_dlpack(self.freqs),
            self.cuda._wrap_for_dlpack(debug_output),
            self.cuda._wrap_for_dlpack(self.fused_summary),
            self.cuda._wrap_for_dlpack(self.fused_q),
            self.cuda._wrap_for_dlpack(self.fused_scale),
            self.full_k_length,
            self.module.q_norm.eps,
            math_mode,
            tile_n,
            self.stream_ptr,
        )
        if not used:
            raise RuntimeError(
                'Kitchen declined the fused H3 Q kernel '
                f'(math_mode={math_mode}, tile_n={tile_n})'
            )
        return self.fused_q

    def baseline_full(self):
        self.quantize_activation()
        return self.baseline_from_quantized()

    def fused_full(self, math_mode=0, tile_n=256):
        self.quantize_activation()
        return self.fused_from_quantized(math_mode=math_mode, tile_n=tile_n)

    def prepare_sparse_attention(
        self,
        global_x,
        global_rope,
        block_index,
        q_start,
        video_start,
        video_budget,
        v_mode='retain',
        build_common_route=True,
    ):
        from h3_optimizations.attention.sparse.kitchen_streamed_q import (
            _build_route_chunk,
            _prepare_route_plan,
            _run_streamed_sparse_kitchen_qkv,
        )
        from h3_optimizations.attention.sparse.router import SparseTileRouter
        from h3_optimizations.kitchen_qkv import (
            ChunkedKitchenQKVProjector,
        )

        projector = ChunkedKitchenQKVProjector(
            chunk_rows=4096,
            routing_summaries=True,
            q_tile=64,
            kv_tile=64,
            strided_qk_input=True,
            stream_output=True,
            v_mode=v_mode,
        )
        projected = _run_streamed_sparse_kitchen_qkv(
            projector,
            self.module,
            global_x,
            global_rope,
            layer_index=block_index,
            transformer_options={},
        )
        if projected is None:
            raise RuntimeError('the current streamed sparse Kitchen producer was not reached')
        self.global_carrier = projected.carrier
        k_summary = projected.k_summary
        projected.carrier = None
        projected.k_summary = None
        projected.release()

        sequence = int(global_x.shape[0])
        layout = SimpleNamespace(
            seq_len=sequence,
            video_range=(int(video_start), sequence),
            segments=(
                (0, int(video_start), 'context'),
                (int(video_start), sequence, 'video'),
            ),
            video_shape=(1, 1, sequence - int(video_start)),
            audio_t=0,
        )
        self.router = SparseTileRouter(q_tile=64, kv_tile=64)
        self.route_plan, metadata = _prepare_route_plan(
            self.router,
            k_summary,
            layout,
            video_budget,
        )
        self._build_route_chunk = _build_route_chunk
        self.q_start = int(q_start)
        self.sparse_metadata = metadata.as_dict()
        if build_common_route:
            self.common_route = self.build_sparse_route(self.baseline_summary)

    def build_sparse_route(self, summary):
        from h3_optimizations.kitchen_qkv import resolve_kitchen

        lut, counts = self._build_route_chunk(
            self.router,
            self.route_plan,
            summary.unsqueeze(0),
            tile_start=self.q_start // 64,
        )
        kitchen = resolve_kitchen(summary.device)
        return kitchen.BlockSparseRoute(
            indices=lut,
            counts=counts,
            q_tile=64,
            kv_tile=64,
            encoding='delta',
        )

    def consume_sparse_q(self, q, q_scale, route):
        from dataclasses import replace
        from h3_optimizations.kitchen_qkv import resolve_kitchen

        carrier = replace(
            self.global_carrier,
            q=q.unsqueeze(0),
            q_scale=q_scale.unsqueeze(0),
        )
        kitchen = resolve_kitchen(q.device)
        return kitchen.block_sparse_int8_attention_from_prequantized(
            carrier,
            route,
            output_layout='nhd',
        )

    def release_q_producer_buffers(self, summary_attr):
        setattr(self, summary_attr, None)
        self.act_qdata = None
        self.act_scale = None
        self.weight_qdata = None
        self.weight_scale = None
        self.projected_q = None
        self.normalized_q = None
        self.debug_q = None
        self.no_debug = None

    def baseline_q_sparse_attention(self, release_producer=False):
        self.baseline_from_quantized()
        route = self.build_sparse_route(self.baseline_summary)
        if release_producer:
            self.release_q_producer_buffers('baseline_summary')
        return self.consume_sparse_q(self.baseline_q, self.baseline_scale, route)

    def fused_q_sparse_attention(
        self,
        math_mode=0,
        tile_n=256,
        release_producer=False,
    ):
        self.fused_from_quantized(math_mode=math_mode, tile_n=tile_n)
        route = self.build_sparse_route(self.fused_summary)
        if release_producer:
            self.release_q_producer_buffers('fused_summary')
        return self.consume_sparse_q(self.fused_q, self.fused_scale, route)

    def baseline_full_sparse_attention(self, release_producer=False):
        self.quantize_activation()
        return self.baseline_q_sparse_attention(release_producer=release_producer)

    def fused_full_sparse_attention(
        self,
        math_mode=0,
        tile_n=256,
        release_producer=False,
    ):
        self.quantize_activation()
        return self.fused_q_sparse_attention(
            math_mode=math_mode,
            tile_n=tile_n,
            release_producer=release_producer,
        )

    def baseline_q_sparse_common_route(self):
        self.baseline_from_quantized()
        return self.consume_sparse_q(
            self.baseline_q,
            self.baseline_scale,
            self.common_route,
        )

    def fused_q_sparse_common_route(self, math_mode=0, tile_n=256):
        self.fused_from_quantized(math_mode=math_mode, tile_n=tile_n)
        return self.consume_sparse_q(
            self.fused_q,
            self.fused_scale,
            self.common_route,
        )

    def prepare_attention_carrier(self, block_index):
        self.global_carrier = chunked_kitchen_carrier(
            self.torch,
            self.module,
            self.x,
            self.rope,
            self.rows,
            block_index,
        )

    def consume_q(self, q, q_scale):
        from dataclasses import replace
        from h3_optimizations.kitchen_qkv import resolve_kitchen

        carrier = replace(
            self.global_carrier,
            q=q.unsqueeze(0),
            q_scale=q_scale.unsqueeze(0),
        )
        kitchen = resolve_kitchen(q.device)
        if kitchen is None:
            raise RuntimeError('the H3 carrier consumer is unavailable')
        return kitchen.int8_attention_from_prequantized(carrier)

    def baseline_q_attention(self):
        self.baseline_from_quantized()
        return self.consume_q(self.baseline_q, self.baseline_scale)

    def fused_q_attention(self, math_mode=0, tile_n=256):
        self.fused_from_quantized(math_mode=math_mode, tile_n=tile_n)
        return self.consume_q(self.fused_q, self.fused_scale)

    def baseline_full_attention(self):
        self.quantize_activation()
        return self.baseline_q_attention()

    def fused_full_attention(self, math_mode=0, tile_n=256):
        self.quantize_activation()
        return self.fused_q_attention(math_mode=math_mode, tile_n=tile_n)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--kitchen-source', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=4096)
    parser.add_argument('--global-sequence', type=int, default=54006)
    parser.add_argument('--q-start', type=int, default=4096)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--video-budget', type=float, default=0.15)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=9)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--with-attention', action='store_true')
    parser.add_argument('--with-sparse-attention', action='store_true')
    parser.add_argument('--v-mode', choices=('retain', 'two_pass'), default='retain')
    parser.add_argument('--memory-arm', choices=('baseline', 'fused'))
    parser.add_argument('--memory-snapshot')
    parser.add_argument('--disable-cuda-malloc', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.sequence % 128:
        parser.error('sequence must be a positive multiple of 128')
    if args.global_sequence <= 0:
        parser.error('global-sequence must be positive')
    if args.q_start < 0 or args.q_start % 64:
        parser.error('q-start must be a non-negative multiple of 64')
    if args.q_start + args.sequence > args.global_sequence:
        parser.error('the Q slab must fit within global-sequence')
    if not 0 <= args.video_start < args.global_sequence:
        parser.error('video-start must lie within global-sequence')
    if not 0.0 < args.video_budget <= 1.0:
        parser.error('video-budget must be in (0, 1]')
    if args.with_attention and args.with_sparse_attention:
        parser.error('choose dense or sparse attention, not both')
    if args.v_mode != 'retain' and not args.with_sparse_attention:
        parser.error('v-mode only applies to sparse attention')
    if bool(args.memory_arm) != bool(args.memory_snapshot):
        parser.error('memory-arm and memory-snapshot must be supplied together')
    if args.memory_arm and not args.with_sparse_attention:
        parser.error('memory capture requires with-sparse-attention')
    if args.memory_arm and not args.disable_cuda_malloc:
        parser.error('memory capture requires disable-cuda-malloc')
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error('warmup/iterations are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error('pass --i-understand-this-uses-gpu after the managed preflight')
    kitchen_source = Path(args.kitchen_source).resolve()
    if not (kitchen_source / 'comfy_kitchen').is_dir():
        parser.error('--kitchen-source does not contain comfy_kitchen')
    args.kitchen_source = kitchen_source
    if args.memory_snapshot:
        args.memory_snapshot = Path(args.memory_snapshot).resolve()
        if not args.memory_snapshot.parent.is_dir():
            parser.error('memory-snapshot parent directory does not exist')
    return args


def main(argv=None):
    args = parse_args(argv)
    comfy_root = Path(__file__).resolve().parents[3]
    pack_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.kitchen_source))
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(pack_root))

    if args.disable_cuda_malloc:
        from comfy.cli_args import args as comfy_args

        comfy_args.cuda_malloc = False
        comfy_args.disable_cuda_malloc = True
        import cuda_malloc  # noqa: F401

    import torch

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    if args.memory_snapshot:
        allocator = torch.cuda.memory.get_allocator_backend()
        if allocator != 'native':
            raise SystemExit('memory capture requires the native allocator, got %s' % allocator)
        torch.cuda.memory._record_memory_history(max_entries=400000)
    device = torch.device('cuda')
    checkpoint = resolve_checkpoint(args.checkpoint)
    module, hidden, prefix = build_attention(
        torch, checkpoint, args.block, args.epsilon, device
    )
    contract = weight_contract(module)
    if not (
        contract['quantized']
        and contract['layout'] == 'TensorWiseINT8Layout'
        and contract['convrot']
        and contract['convrot_groupsize'] == 256
    ):
        raise SystemExit('checkpoint QKV is not ConvRot-256 TensorWise INT8')

    generator = torch.Generator(device=device).manual_seed(1234)
    full_sequence = (
        args.global_sequence if args.with_sparse_attention else args.sequence
    )
    global_x = torch.randn(
        (full_sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    global_rope = make_rope(torch, full_sequence, device)
    if args.with_sparse_attention:
        x = global_x[args.q_start : args.q_start + args.sequence].contiguous()
        rope = global_rope[
            :, args.q_start : args.q_start + args.sequence
        ].contiguous()
    else:
        x = global_x
        rope = global_rope
    experiment = FusedQExperiment(
        torch,
        module,
        x,
        rope,
        full_k_length=full_sequence,
        arm=args.memory_arm or 'both',
        defer_q_buffers=bool(args.memory_arm),
    )
    if args.memory_arm:
        experiment.prepare_sparse_attention(
            global_x,
            global_rope,
            args.block,
            args.q_start,
            args.video_start,
            args.video_budget,
            v_mode=args.v_mode,
            build_common_route=False,
        )
        experiment.allocate_q_buffers()
        if args.memory_arm == 'baseline':
            output = experiment.baseline_full_sparse_attention(release_producer=True)
        else:
            output = experiment.fused_full_sparse_attention(release_producer=True)
        torch.cuda.synchronize(device)
        torch.cuda.memory._dump_snapshot(str(args.memory_snapshot))
        print(json.dumps({
            'allocator': torch.cuda.memory.get_allocator_backend(),
            'arm': args.memory_arm,
            'v_mode': args.v_mode,
            'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'output_shape': list(output.shape),
            'snapshot': str(args.memory_snapshot),
        }, indent=2))
        return 0
    experiment.quantize_activation()
    experiment.baseline_from_quantized()
    torch.cuda.synchronize(device)
    baseline_post_hnd = experiment.normalized_q.transpose(1, 2)
    baseline_post = experiment.normalized_q.reshape(experiment.rows, experiment.inner)
    oracle_carrier = bf16_oracle_carrier(
        torch,
        baseline_post_hnd,
        experiment.full_k_length,
    ).squeeze(0)
    kitchen_oracle = oracle_error(
        torch,
        dequantize_carrier(
            torch,
            experiment.baseline_q,
            experiment.baseline_scale,
        ),
        oracle_carrier,
    )
    projected = experiment.projected_q.view(
        1, experiment.rows, experiment.heads, HEAD_DIM
    )
    rrms = torch.rsqrt(
        projected.float().square().mean(dim=-1, keepdim=True)
        + experiment.module.q_norm.eps
    )
    manual_post = (
        projected.float()
        * rrms
        * experiment.norm.float().view(1, 1, 1, HEAD_DIM)
    ).to(torch.bfloat16)
    experiment.cuda.apply_rope_split_half1_(
        manual_post[..., :ROT_DIM],
        experiment.rope,
    )
    manual_post = manual_post.reshape(experiment.rows, experiment.inner)
    parity = {
        'bf16_oracle': {
            'definition': (
                'Config-0 BF16 Q projection followed by production BF16 '
                'RMSNorm/RoPE and carrier ConvRot, without INT8 quantization.'
            ),
            'kitchen': kitchen_oracle,
            'manual_float_formula_vs_production': tensor_difference(
                torch,
                manual_post,
                baseline_post,
            ),
        },
        'variants': {},
    }
    sparse_route = None
    baseline_attention = None
    oracle_attention = None
    baseline_sparse = None
    baseline_common = None
    baseline_route = None
    if args.with_attention:
        experiment.prepare_attention_carrier(args.block)
        oracle_attention = bf16_attention_oracle(
            torch,
            module,
            x,
            rope,
        )
        baseline_attention = experiment.baseline_q_attention()
        parity['bf16_attention_oracle'] = {
            'definition': (
                'BF16 Q/K/V projection, RMSNorm, RoPE, and PyTorch scaled-dot-product attention.'
            ),
            'kitchen': output_error_metrics(
                torch,
                baseline_attention,
                oracle_attention,
            ),
        }
    elif args.with_sparse_attention:
        experiment.prepare_sparse_attention(
            global_x,
            global_rope,
            args.block,
            args.q_start,
            args.video_start,
            args.video_budget,
            v_mode=args.v_mode,
        )
        baseline_route = experiment.build_sparse_route(experiment.baseline_summary)
        sparse_route = {
            'router_metadata': experiment.sparse_metadata,
            'baseline': route_report(baseline_route),
            'variants': {},
            'consumer': (
                'h3_optimizations.native.'
                'block_sparse_int8_attention_from_prequantized'
            ),
        }
        baseline_sparse = experiment.consume_sparse_q(
            experiment.baseline_q,
            experiment.baseline_scale,
            baseline_route,
        )
        baseline_common = experiment.consume_sparse_q(
            experiment.baseline_q,
            experiment.baseline_scale,
            experiment.common_route,
        )

    for name, math_mode, tile_n in FUSED_VARIANTS:
        experiment.fused_from_quantized(
            math_mode=math_mode,
            tile_n=tile_n,
            debug=True,
        )
        report = parity_report(
            torch,
            experiment.baseline_q,
            experiment.baseline_scale,
            experiment.fused_q,
            experiment.fused_scale,
        )
        report['post_norm_rope'] = tensor_difference(
            torch,
            experiment.debug_q,
            baseline_post,
        )
        report['router_summary'] = tensor_difference(
            torch,
            experiment.fused_summary,
            experiment.baseline_summary,
        )
        report['bf16_oracle'] = oracle_error(
            torch,
            dequantize_carrier(
                torch,
                experiment.fused_q,
                experiment.fused_scale,
            ),
            oracle_carrier,
        )
        candidate_rel_l2 = report['bf16_oracle']['rel_l2']
        kitchen_rel_l2 = kitchen_oracle['rel_l2']
        report['bf16_oracle_rel_l2_delta_vs_kitchen'] = (
            candidate_rel_l2 - kitchen_rel_l2
        )
        report['bf16_oracle_rel_l2_ratio_vs_kitchen'] = (
            candidate_rel_l2 / max(kitchen_rel_l2, 1e-12)
        )
        report['bf16_oracle_not_worse_with_0p1pct_margin'] = bool(
            candidate_rel_l2 <= kitchen_rel_l2 * 1.001 + 1e-12
        )
        if baseline_attention is not None:
            candidate_attention = experiment.consume_q(
                experiment.fused_q,
                experiment.fused_scale,
            )
            torch.cuda.synchronize(device)
            report['attention_vs_kitchen'] = output_error_metrics(
                torch,
                candidate_attention,
                baseline_attention,
            )
            report['attention_vs_bf16_oracle'] = output_error_metrics(
                torch,
                candidate_attention,
                oracle_attention,
            )
            report['attention_bf16_oracle_not_worse_with_0p1pct_margin'] = bool(
                report['attention_vs_bf16_oracle']['relative_rmse']
                <= parity['bf16_attention_oracle']['kitchen']['relative_rmse'] * 1.001
                + 1e-12
            )
            del candidate_attention
        elif baseline_sparse is not None:
            candidate_route = experiment.build_sparse_route(experiment.fused_summary)
            active_columns = max(
                int(baseline_route.counts.max().item()),
                int(candidate_route.counts.max().item()),
            )
            baseline_active = baseline_route.indices[..., :active_columns]
            candidate_active = candidate_route.indices[..., :active_columns]
            sparse_route['variants'][name] = {
                'route': route_report(candidate_route),
                **route_overlap_report(torch, baseline_route, candidate_route),
                'indices_exact': bool(
                    torch.equal(baseline_route.indices, candidate_route.indices)
                ),
                'counts_exact': bool(
                    torch.equal(baseline_route.counts, candidate_route.counts)
                ),
                'index_mismatch_fraction': float(
                    (
                        baseline_route.indices != candidate_route.indices
                    ).float().mean().item()
                ),
                'active_delta_index_mismatch_fraction': float(
                    (baseline_active != candidate_active).float().mean().item()
                ),
            }
            candidate_sparse = experiment.consume_sparse_q(
                experiment.fused_q,
                experiment.fused_scale,
                candidate_route,
            )
            candidate_common = experiment.consume_sparse_q(
                experiment.fused_q,
                experiment.fused_scale,
                experiment.common_route,
            )
            torch.cuda.synchronize(device)
            report['sparse_attention_current_route_vs_kitchen'] = (
                output_error_metrics(
                    torch,
                    candidate_sparse,
                    baseline_sparse,
                )
            )
            report['sparse_attention_common_route_vs_kitchen'] = (
                output_error_metrics(
                    torch,
                    candidate_common,
                    baseline_common,
                )
            )
            del candidate_sparse, candidate_common, candidate_route
        parity['variants'][name] = report

    del oracle_carrier, baseline_post_hnd, manual_post, projected, rrms
    if baseline_attention is not None:
        del baseline_attention, oracle_attention
    if baseline_sparse is not None:
        del baseline_sparse, baseline_common, baseline_route, global_x, global_rope

    sampler = GpuSampler()
    sampler.start()
    try:
        timings = {
            'bare_config0_q_gemm': alternating_time(
                torch,
                (('config0', experiment.config0_gemm),),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            ),
            'prequantized_q_producer': alternating_time(
                torch,
                (('baseline', experiment.baseline_from_quantized),)
                + fused_cases(experiment, 'fused_from_quantized'),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            ),
            'including_activation_quant': alternating_time(
                torch,
                (('baseline', experiment.baseline_full),)
                + fused_cases(experiment, 'fused_full'),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            ),
        }
        if args.with_attention:
            timings['prequantized_q_plus_attention'] = alternating_time(
                torch,
                (('baseline', experiment.baseline_q_attention),)
                + fused_cases(experiment, 'fused_q_attention'),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            )
            timings['including_activation_quant_plus_attention'] = alternating_time(
                torch,
                (('baseline', experiment.baseline_full_attention),)
                + fused_cases(experiment, 'fused_full_attention'),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            )
        elif args.with_sparse_attention:
            timings['prequantized_q_plus_sparse_current_route'] = alternating_time(
                torch,
                (('baseline', experiment.baseline_q_sparse_attention),)
                + fused_cases(experiment, 'fused_q_sparse_attention'),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            )
            timings['including_activation_quant_plus_sparse_current_route'] = alternating_time(
                torch,
                (('baseline', experiment.baseline_full_sparse_attention),)
                + fused_cases(experiment, 'fused_full_sparse_attention'),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            )
            timings['prequantized_q_plus_sparse_common_route'] = alternating_time(
                torch,
                (('baseline', experiment.baseline_q_sparse_common_route),)
                + fused_cases(experiment, 'fused_q_sparse_common_route'),
                args.warmup,
                args.iterations,
                args.repeats,
                device,
            )
    finally:
        sampler.stop()
    for boundary in timings.values():
        if 'baseline' not in boundary:
            continue
        baseline_ms = boundary['baseline']['cuda']['median_ms']
        boundary['speedup_vs_baseline_percent'] = {
            name: (baseline_ms / result['cuda']['median_ms'] - 1.0) * 100.0
            for name, result in boundary.items()
            if name != 'baseline' and isinstance(result, dict) and 'cuda' in result
        }

    bare_gemm_ms = timings['bare_config0_q_gemm']['config0']['cuda']['median_ms']
    producer_baseline_ms = timings['prequantized_q_producer']['baseline']['cuda']['median_ms']
    producer_headroom = {
        'bare_config0_bf16_store_ms': bare_gemm_ms,
        'current_producer_ms': producer_baseline_ms,
        'current_over_bare_config0_percent': (
            producer_baseline_ms / bare_gemm_ms - 1.0
        ) * 100.0,
        'variants': {
            name: {
                'producer_ms': timings['prequantized_q_producer'][name]['cuda']['median_ms'],
                'over_bare_config0_percent': (
                    timings['prequantized_q_producer'][name]['cuda']['median_ms']
                    / bare_gemm_ms
                    - 1.0
                ) * 100.0,
            }
            for name, _math_mode, _tile_n in FUSED_VARIANTS
        },
        'note': (
            'Config 0 still writes BF16 Q, so this is a practical GEMM reference, '
            'not a hard mathematical floor for a fused epilogue.'
        ),
    }

    from comfy_kitchen.backends import cuda

    result = {
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
        },
        'versions': {
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
        },
        'kernel': {
            'config': CONFIG,
            'variants': [
                {'name': name, 'math_mode': math_mode, 'tile': [128, tile_n, 64]}
                for name, math_mode, tile_n in FUSED_VARIANTS
            ],
            'sparse_consumer_tile': [64, 64],
            'extension': str(Path(cuda._C.__file__).resolve()),
        },
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': args.block,
        'sequence': args.sequence,
        'global_sequence': full_sequence,
        'q_start': args.q_start if args.with_sparse_attention else 0,
        'video_start': args.video_start if args.with_sparse_attention else None,
        'video_budget': args.video_budget if args.with_sparse_attention else None,
        'repeats_per_sample': args.repeats,
        'hidden': hidden,
        'heads': module.heads,
        'weight_contract': contract,
        'parity': parity,
        'sparse_route': sparse_route,
        'producer_headroom': producer_headroom,
        'telemetry': sampler.report(),
        'timings': timings,
    }
    if args.summary_only:
        result['timings'] = {
            boundary_name: {
                **{
                    case_name: case_result['cuda']['median_ms']
                    for case_name, case_result in boundary.items()
                    if isinstance(case_result, dict) and 'cuda' in case_result
                },
                **(
                    {'speedup_vs_baseline_percent': boundary['speedup_vs_baseline_percent']}
                    if 'speedup_vs_baseline_percent' in boundary
                    else {}
                ),
            }
            for boundary_name, boundary in timings.items()
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
