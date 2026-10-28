"""Small projection correctness probe; no timing or full-model quality claims."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('--output', type=Path)
args = cli.parse_args()
sys.argv = [sys.argv[0], '--cpu']

import torch
import torch.nn.functional as F
from comfy.quant_ops import QuantizedTensor


def probe():
    torch.set_grad_enabled(False)
    records = []
    for seed in (3, 7, 19):
        torch.manual_seed(seed)
        value = torch.randn(65, 7168, device='cuda', dtype=torch.bfloat16) * .3
        weight = torch.randn(256, 7168, device='cuda', dtype=torch.bfloat16) * .01
        quantized = QuantizedTensor.from_float(
            weight, 'TensorWiseINT8Layout', scale='recalculate', is_weight=True,
            per_channel=True, convrot=True, convrot_groupsize=256,
        )
        restored = quantized.dequantize()
        partial_weights = []
        for start in range(0, 7168, 1024):
            tile = QuantizedTensor(
                quantized._qdata[:, start:start + 1024].contiguous(), quantized._layout_cls,
                replace(quantized._params, orig_shape=(256, 1024)),
            ).dequantize()
            if not torch.equal(tile, restored[:, start:start + 1024]):
                raise AssertionError('ConvRot weight slice changed the rotation basis')
            partial_weights.append(tile)
        native = F.linear(value, quantized)
        fp32 = F.linear(value.float(), restored.float())
        residual = torch.randn(65, 256, device='cuda', dtype=torch.bfloat16) * .1
        for strength in (1., .1, .01, .001):
            gate = torch.tensor(strength, device='cuda', dtype=torch.bfloat16)
            reference = residual.float().addcmul(fp32, gate.float())
            ordinary = residual.clone().addcmul_(native, gate)
            naive, compensated = residual.clone(), residual.clone()
            error_bytes = torch.zeros_like(residual, dtype=torch.int8)
            error_scale = torch.ones(65, 1, device='cuda', dtype=torch.float32)
            for index, start in enumerate(range(0, 7168, 1024)):
                partial = F.linear(value[:, start:start + 1024], partial_weights[index])
                naive.addcmul_(partial, gate)
                total = compensated.float() + error_bytes.float() * error_scale
                total.addcmul_(partial.float(), gate.float())
                rounded = total.to(torch.bfloat16)
                remainder = total - rounded.float()
                error_scale = (remainder.abs().amax(dim=-1, keepdim=True) / 127).clamp_min_(1e-30)
                error_bytes = (remainder / error_scale).round().clamp(-127, 127).to(torch.int8)
                compensated.copy_(rounded)
            def relative(output):
                return float((output.float() - reference).norm() / reference.norm())
            records.append(dict(seed=seed, gate=strength,
                                native_relative_l2=relative(ordinary),
                                naive_relative_l2=relative(naive),
                                compensated_relative_l2=relative(compensated)))
    return dict(synthetic_projection_only=True, no_timing=True, groups=7,
                group_width=1024, weight_slice_exact=True, records=records)


if __name__ == '__main__':
    text = json.dumps(probe(), indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding='utf-8')
    print(text)
