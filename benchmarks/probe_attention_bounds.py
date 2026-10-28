"""CPU-only numerical probe for block attention mass bounds, not a speed test."""

import argparse
import json
import math
from pathlib import Path

import torch


def evaluate(q, k, v, epsilon):
    blocks, rows, dim = k.shape
    center = k.mean(1)
    radius = (k - center[:, None]).norm(dim=-1).amax(1)
    upper = (q @ center.T + q.norm(dim=-1, keepdim=True) * radius) / math.sqrt(dim)
    magnitude = float(v.abs().amax())
    fractions, errors, certificates = [], [], []
    for query, limits in zip(q, upper):
        order = limits.argsort(descending=True).tolist()
        scores, values = [], []
        for used, index in enumerate(order, 1):
            scores.append(k[index] @ query / math.sqrt(dim))
            values.append(v[index])
            selected = torch.cat(scores)
            log_z = torch.logsumexp(selected, 0)
            mass = 0.
            if used < blocks:
                log_u = torch.logsumexp(limits[order[used:]] + math.log(rows), 0)
                mass = float(torch.sigmoid(log_u - log_z))
            if mass <= epsilon:
                break
        estimate = selected.softmax(0) @ torch.cat(values)
        exact = (k.reshape(-1, dim) @ query / math.sqrt(dim)).softmax(0) @ v.reshape(-1, v.shape[-1])
        error = float((estimate - exact).abs().amax())
        certificate = 2 * magnitude * mass
        if error > certificate + 2e-5:
            raise AssertionError((error, certificate))
        fractions.append(used / blocks)
        errors.append(error)
        certificates.append(certificate)
    return dict(epsilon=epsilon, mean_blocks_computed=sum(fractions) / len(fractions),
                max_error=max(errors), max_error_bound=max(certificates))


def probe():
    torch.manual_seed(19)
    torch.set_num_threads(2)
    dim, blocks, rows, queries = 128, 64, 64, 16
    results = []
    for spread in (None, .05, .25, 1.):
        centers = torch.randn(blocks, dim)
        centers = centers / centers.norm(dim=-1, keepdim=True) * math.sqrt(dim)
        k = torch.randn(blocks, rows, dim) if spread is None else centers[:, None] + spread * torch.randn(blocks, rows, dim)
        k = k / k.norm(dim=-1, keepdim=True) * math.sqrt(dim)
        v = torch.randn(blocks, rows, 16)
        q = torch.randn(queries, dim) if spread is None else centers[:queries] + .05 * torch.randn(queries, dim)
        q = q / q.norm(dim=-1, keepdim=True) * math.sqrt(dim)
        results.append(dict(
            distribution='unstructured' if spread is None else 'clustered_' + str(spread),
            results=[evaluate(q, k, v, epsilon) for epsilon in (.01, .001)],
        ))
    return dict(synthetic_only=True, keys=blocks * rows, head_dim=dim, queries=queries, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(probe(), indent=2)
    if args.output is not None:
        args.output.write_text(report, encoding='utf-8')
    print(report)
