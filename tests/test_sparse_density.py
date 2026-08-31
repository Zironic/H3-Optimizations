'''CPU-only contracts for the H3 Sparse Attention density policy.'''

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _PACK)
sys.path.insert(0, _ROOT)

sys.argv = [sys.argv[0], '--cpu']
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.backend import (  # noqa: E402
    HybridSparseBackend,
)
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
    resolve_video_budget,
)
from h3_optimizations.nodes import (  # noqa: E402
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)
from h3_optimizations.runtime.context import (  # noqa: E402
    H3RuntimeSession,
    RUNTIME_KEY,
    RuntimeSnapshot,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print('  ok: %s' % message)


class TensorStub:
    shape = (1, 2, 384, 128)


class MaskMetadata:
    pure_video_q_tiles = 4

    def __init__(self, budget):
        self.budget = float(budget)

    def as_dict(self):
        return {'requested_video_budget': self.budget}


class RecordingRouter:
    def __init__(self):
        self.budgets = []

    def build_lut(self, _q, _k, _layout, budget, *, sink=None):
        del sink
        self.budgets.append(float(budget))
        return object(), object(), MaskMetadata(budget)

    def build_lut_from_summaries(
        self,
        _q_summary,
        _k_summary,
        _layout,
        budget,
        *,
        sink=None,
    ):
        del sink
        self.budgets.append(float(budget))
        return object(), object(), MaskMetadata(budget)


class RecordingExecutor:
    def prepare(
        self,
        _q,
        _k,
        _v,
        _lut,
        _valid_block_num,
        *,
        layer_index,
        metadata,
    ):
        return SimpleNamespace(layer_index=layer_index, metadata=metadata)

    def prepare_projected(
        self,
        _projected,
        _lut,
        _valid_block_num,
        *,
        metadata,
    ):
        return SimpleNamespace(metadata=metadata)


def make_backend(config):
    backend = object.__new__(HybridSparseBackend)
    backend.config = config
    backend.router = RecordingRouter()
    backend.executor = RecordingExecutor()
    backend.projector = None
    return backend


def options(step_index, total_steps=20):
    layout = SimpleNamespace(seq_len=384)
    snapshot = RuntimeSnapshot(
        request_id=0,
        step_index=step_index,
        total_steps=total_steps,
        layout=layout,
        compute_dtype=None,
        device=None,
    )
    return {RUNTIME_KEY: snapshot}


def input_by_id(schema, input_id):
    return next(item for item in schema.inputs if item.id == input_id)


def test_node_schema_and_request():
    print('H3 Sparse Attention node policy')
    schema = H3SparseAttention.define_schema()
    check(
        [item.id for item in schema.inputs]
        == [
            'model',
            'video_budget',
            'denser_early_late_steps',
        ],
        'standard schema exposes only its production controls',
    )
    denser = input_by_id(schema, 'denser_early_late_steps')
    check(
        denser.display_name == 'Denser Early ramp'
        and denser.default is True
        and 'targets 12 additional percentage points' in denser.tooltip
        and 'no less than 50%' in denser.tooltip,
        'simple early density ramp is described and defaults on',
    )

    model = SimpleNamespace(model_options={})
    patched = SimpleNamespace(model_options={})
    with mock.patch(
        'h3_optimizations.nodes.apply_plan',
        return_value=patched,
    ) as apply:
        result = H3SparseAttention.execute(
            model,
            video_budget=0.5,
        )
    request = apply.call_args.args[1].sparse
    check(
        result.args[0] is patched
        and request.video_budget == 0.5
        and request.backend == 'auto'
        and request.video_token_order == '1x8x8'
        and request.denser_early_late_steps is True
        and request.advanced_schedule is False,
        'standard node carries the simple denser-step policy',
    )

def test_advanced_node_schema_and_request():
    print('H3 Sparse Attention Advanced node policy')
    schema = H3SparseAttentionAdvanced.define_schema()
    check(
        [item.id for item in schema.inputs]
        == [
            'model',
            'video_budget',
            'early_steps',
            'early_kv',
            'late_steps',
            'late_kv',
            'backend',
            'early_schedule',
            'video_token_order',
        ],
        'advanced schema exposes only production sparse controls',
    )
    backend = input_by_id(schema, 'backend')
    check(
        backend.default == 'Kitchen INT8'
        and backend.options
        == [
            'Kitchen INT8',
            'Kitchen INT8 64x128 (experimental)',
            'FROST BF16 (SM89)',
            'Sparse Sage',
            'BF16 Triton',
            'FP8 FlexAttention',
        ],
        'advanced backend selector exposes the supported sparse backends',
    )
    check(
        input_by_id(schema, 'video_budget').default == 0.15
        and input_by_id(schema, 'early_steps').default == 4
        and input_by_id(schema, 'early_kv').default == 0.5
        and input_by_id(schema, 'late_steps').default == 0
        and input_by_id(schema, 'late_kv').default == 0.5
        and input_by_id(schema, 'early_schedule').default == 'Hold'
        and input_by_id(schema, 'early_schedule').options == ['Hold', 'Ramp']
        and input_by_id(schema, 'video_token_order').default == '1x8x8',
        'advanced early and late defaults match the public contract',
    )

    model = SimpleNamespace(model_options={})
    patched = SimpleNamespace(model_options={})
    with mock.patch(
        'h3_optimizations.nodes.apply_plan',
        return_value=patched,
    ) as apply:
        result = H3SparseAttentionAdvanced.execute(
            model,
            video_budget=0.3,
            early_steps=3,
            early_kv=0.6,
            late_steps=4,
            late_kv=0.7,
            backend='BF16 Triton',
            early_schedule='Ramp',
        )
    request = apply.call_args.args[1].sparse
    check(
        result.args[0] is patched
        and request.backend == 'BF16 Triton'
        and request.video_budget == 0.3
        and request.early_steps == 3
        and request.early_kv == 0.6
        and request.late_steps == 4
        and request.late_kv == 0.7
        and request.early_schedule == 'Ramp'
        and request.video_token_order == '1x8x8'
        and request.denser_early_late_steps is False,
        'advanced node carries the explicit schedule and production TopK routing',
    )


def test_step_budgets():
    print('H3 Sparse Attention smart early ramp')
    config = HybridSparseConfig(
        video_budget=0.1,
        denser_early_late_steps=True,
    )
    backend = make_backend(config)
    q = k = v = TensorStub()
    budgets = [resolve_video_budget(config, step, 20) for step in range(20)]
    check(abs(budgets[0] - 0.5) < 1e-9, 'the first step uses 50% video KV')
    check(
        all(left >= right for left, right in zip(budgets, budgets[1:])),
        'the early schedule declines monotonically',
    )
    check(
        all(abs(value - 0.1) < 1e-9 for value in budgets[11:]),
        'the 20-step ramp returns to the configured floor after step 11',
    )
    check(
        abs(sum(value - 0.1 for value in budgets) - 2.4) < 1e-9,
        'the ramp spends 12 average percentage points of extra KV',
    )
    for step_index in (-1, 0, 5, 10, 11, 19):
        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=0,
            transformer_options=options(step_index),
        )
        check(
            abs(
                prepared.sparse.metadata['requested_video_budget']
                - (0.1 if step_index < 0 else budgets[step_index])
            )
            < 1e-9,
            'step %d uses the resolved ramp budget' % step_index,
        )

    projected = SimpleNamespace(
        sequence=384,
        q_summary=object(),
        k_summary=object(),
        heads=2,
    )
    prepared = backend.prepare_projected(
        projected,
        layer_index=0,
        transformer_options=options(5),
    )
    check(
        abs(
            prepared.sparse.metadata['requested_video_budget'] - budgets[5]
        ) < 1e-9,
        'fused projected-QKV routing uses the same early ramp',
    )
    check(
        resolve_video_budget(
            HybridSparseConfig(
                video_budget=0.85,
                denser_early_late_steps=True,
            ),
            0,
            20,
        )
        == 0.85,
        'simple ramp does not alter an already denser base budget',
    )
    short_budgets = [resolve_video_budget(config, step, 8) for step in range(8)]
    check(
        abs(short_budgets[0] - 0.5) < 1e-9
        and abs(sum(value - 0.1 for value in short_budgets) - 0.96) < 1e-9
        and all(left >= right for left, right in zip(short_budgets, short_budgets[1:])),
        'the normalized ramp preserves its area and shape at eight steps',
    )
    check(
        resolve_video_budget(HybridSparseConfig(video_budget=0.15), 0, 20)
        == 0.15,
        'disabled simple ramp preserves the configured budget',
    )
    all_ramps_bounded = True
    all_areas_exact = True
    for total_steps in range(1, 33):
        for base_budget in (0.01, 0.1, 0.15, 0.3, 0.49, 0.5, 0.85, 1.0):
            varied = HybridSparseConfig(
                video_budget=base_budget,
                denser_early_late_steps=True,
            )
            resolved = [
                resolve_video_budget(varied, step, total_steps)
                for step in range(total_steps)
            ]
            all_ramps_bounded = all_ramps_bounded and (
                all(
                    base_budget <= value <= 1.0
                    for value in resolved
                )
                and all(
                    left >= right
                    for left, right in zip(resolved, resolved[1:])
                )
            )
            peak_extra = max(0.0, 0.5 - base_budget)
            expected_extra = 0.0
            if peak_extra:
                expected_extra = max(
                    min(0.12 * total_steps, peak_extra * total_steps),
                    peak_extra,
                )
            all_areas_exact = all_areas_exact and (
                abs(
                    sum(value - base_budget for value in resolved)
                    - expected_extra
                ) < 1e-9
            )
    check(all_ramps_bounded, 'ramps remain bounded and monotonic across step counts')
    check(all_areas_exact, 'ramps preserve normalized area across step counts')


def test_advanced_step_budgets():
    print('H3 Sparse Attention explicit step budgets')
    config = HybridSparseConfig(
        video_budget=0.3,
        early_steps=2,
        early_kv=0.5,
        late_steps=3,
        late_kv=0.7,
    )
    expected = {
        -1: 0.3,
        0: 0.5,
        1: 0.5,
        2: 0.3,
        16: 0.3,
        17: 0.7,
        18: 0.7,
        19: 0.7,
    }
    for step_index, budget in expected.items():
        check(
            resolve_video_budget(config, step_index, 20) == budget,
            'explicit step %d resolves to %.0f%% video budget'
            % (step_index, budget * 100.0),
        )

    check(
        resolve_video_budget(
            HybridSparseConfig(
                video_budget=0.7,
                early_steps=1,
                early_kv=0.2,
                late_steps=0,
                late_kv=0.4,
            ),
            0,
            20,
        )
        == 0.2,
        'explicit early KV may be lower than the middle-step budget',
    )
    check(
        resolve_video_budget(
            HybridSparseConfig(
                video_budget=0.3,
                early_steps=2,
                early_kv=0.4,
                late_steps=2,
                late_kv=0.6,
            ),
            1,
            3,
        )
        == 0.6,
        'overlapping early and late windows use the denser requested budget',
    )

    ramp = HybridSparseConfig(
        video_budget=0.3,
        early_steps=4,
        early_kv=0.7,
        late_steps=0,
        late_kv=0.7,
        early_schedule='Ramp',
    )
    ramp_budgets = [
        resolve_video_budget(ramp, step, 20) for step in range(6)
    ]
    check(
        all(
            abs(actual - expected) < 1e-9
            for actual, expected in zip(
                ramp_budgets,
                [0.7, 0.6, 0.5, 0.4, 0.3, 0.3],
            )
        ),
        'advanced ramp moves linearly from Early KV to the base budget',
    )
    overlap = HybridSparseConfig(
        video_budget=0.3,
        early_steps=3,
        early_kv=0.9,
        late_steps=2,
        late_kv=0.8,
        early_schedule='Ramp',
    )
    check(
        resolve_video_budget(overlap, 1, 3) == 0.8,
        'late KV wins when it is denser than an overlapping early ramp',
    )


def test_explicit_per_step_budgets():
    print('H3 Sparse Attention benchmark per-step budgets')
    budgets = (0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2) + (0.1,) * 10
    config = HybridSparseConfig(
        video_budget=0.1,
        step_video_budgets=budgets,
    )
    check(
        [resolve_video_budget(config, step, 20) for step in range(20)]
        == list(budgets),
        'per-step schedule resolves every sampler step exactly',
    )
    try:
        resolve_video_budget(config, 0, 19)
    except ValueError as exc:
        check('19 steps' in str(exc), 'sampler length mismatch fails clearly')
    else:
        raise AssertionError('sampler length mismatch should fail')
    try:
        HybridSparseConfig(
            video_budget=0.1,
            denser_early_late_steps=True,
            step_video_budgets=budgets,
        )
    except ValueError as exc:
        check('cannot be combined' in str(exc), 'per-step schedule is exclusive')
    else:
        raise AssertionError('combined schedules should fail')


def test_runtime_step_resolution():
    print('H3 runtime sampler-step publication')
    session = H3RuntimeSession()
    layout = SimpleNamespace(seq_len=384)
    context = SimpleNamespace(dtype=None)
    schedule = torch.tensor([1.0, 0.7, 0.4, 0.1, 0.0])
    token = session.begin_request(4)
    with mock.patch(
        'h3_optimizations.runtime.context.resolve_layout',
        return_value=layout,
    ):
        try:
            first = session.observe(
                TensorStub(),
                context,
                {'sample_sigmas': schedule},
            )
            session.complete_step(1, 4)
            middle = session.observe(
                TensorStub(),
                context,
                {'sample_sigmas': schedule},
            )
        finally:
            session.end_request(token)
        unknown = session.observe(
            TensorStub(),
            context,
            {'sample_sigmas': schedule},
        )
    check(
        first.step_index == 0 and first.total_steps == 4,
        'request boundary publishes the first sampler step',
    )
    check(
        middle.step_index == 2 and middle.total_steps == 4,
        'sampler callback advances step publication',
    )
    check(
        unknown.step_index == -1 and unknown.total_steps == 4,
        'missing current-step metadata preserves the base-budget fallback',
    )


def main():
    test_node_schema_and_request()
    test_advanced_node_schema_and_request()
    test_step_budgets()
    test_advanced_step_budgets()
    test_explicit_per_step_budgets()
    test_runtime_step_resolution()
    print('\nall H3 sparse density tests passed')


if __name__ == '__main__':
    main()
