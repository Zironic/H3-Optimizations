'''Composable production nodes for MiniMax H3 optimization.'''

import logging
import math

from comfy_api.latest import io, ui

from .apply import apply_plan
from .node_constants import NODE_CATEGORY
from .plan import (
    DEFAULT_EDGE_KV,
    DEFAULT_HEAD_BUDGETS,
    DEFAULT_LATE_STEPS,
    DEFAULT_VIDEO_BUDGET,
    DEFAULT_VIDEO_TOKEN_ORDER,
    EARLY_SCHEDULE_HOLD,
    EARLY_SCHEDULE_OPTIONS,
    EARLY_SCHEDULE_RAMP,
    HEAD_BUDGETS_REQUESTS,
    SPARSE_BACKEND_COMPAT_REQUESTS,
    SPARSE_BACKEND_KITCHEN_64X32,
    SPARSE_BACKEND_PUBLIC_REQUESTS,
    SparseRequest,
    VIDEO_TOKEN_ORDER_REQUESTS,
    read_plan,
)
from .status import (
    format_memory_status,
    format_sparse_status,
)
from .vsa.node import is_vsa_checkpoint

VSA_PASSTHROUGH_MESSAGE = (
    'VSA-trained H3 checkpoint detected (to_gate_compress layers, e.g. FastH3). '
    'H3 Sparse Attention passed the model through unchanged: this checkpoint was '
    'trained on its own sparse pattern. Use H3 VSA Attention (FastH3) instead.'
)


def _vsa_passthrough(model):
    """Leave VSA-trained checkpoints to the VSA node rather than re-routing them."""
    if not is_vsa_checkpoint(model):
        return None
    logging.warning('[H3 Optimizations] %s', VSA_PASSTHROUGH_MESSAGE)
    return io.NodeOutput(model, ui=ui.PreviewText(VSA_PASSTHROUGH_MESSAGE))

# On a 20-step sampler, this eight-step ramp spends the same 2.4 cumulative
# extra video-budget steps as the simple node's default normalized early ramp.
_ADVANCED_DEFAULT_EARLY_STEPS = 8
_ADVANCED_DEFAULT_EARLY_KV = 41.0 / 60.0
_ADVANCED_DEFAULT_EARLY_SCHEDULE = EARLY_SCHEDULE_RAMP


def _video_budget_input():
    return io.Float.Input(
        'video_budget',
        display_name='Video attention budget',
        default=DEFAULT_VIDEO_BUDGET,
        min=0.01,
        max=1.0,
        step=0.01,
        tooltip=(
            'Controls the speed/quality tradeoff for target-video attention. '
            'Lower values are faster but retain fewer video attention connections '
            'and can reduce prompt adherence, change motion/detail, or otherwise '
            'change the result. There is no universally safe value: some prompts '
            'tolerate very low budgets while others require substantially more. '
            'Native threshold routing targets this density with variable tile '
            'counts per row; other routers round up to whole KV tiles. Non-video '
            'context and mixed boundary tiles stay dense. 1.0 retains the full video route. The '
            'displayed range is the recommended editing range; finite workflow '
            'values outside it saturate to at least one or at most all video tiles.'
        ),
    )


def _validate_budget(name, value):
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return '%s must be finite' % name
    return None


def _validate_steps(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        return '%s must be a non-negative integer' % name
    return None


class H3SparseAttention(io.ComfyNode):
    '''Fixed-density sparse attention for MiniMax H3.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3SparseAttention',
            display_name='H3 Sparse Attention',
            category=NODE_CATEGORY,
            description=(
                'Budgeted sparse attention for MiniMax H3. Lower video '
                'attention budgets are faster but can reduce prompt adherence, '
                'change motion/detail, or otherwise change the generated result; '
                'no percentage is lossless for every prompt. Text, reference '
                'conditioning, audio, non-video queries, and mixed boundary tiles '
                'remain dense. Target-video tokens use the measured 1x8x8 '
                'router-aligned order. Backend auto prefers native Kitchen INT8, then '
                'Sparse Sage, BF16 Triton, FP8 FlexAttention, and finally the '
                'resolved dense attention path.'
            ),
            search_aliases=[
                'H3 sparse',
                'H3 sparse attention',
                'MiniMax sparse',
                'Sparse Sage',
                'Sparge',
                'H3 acceleration',
            ],
            inputs=[
                io.Model.Input('model'),
                _video_budget_input(),
                io.Boolean.Input(
                    'denser_early_late_steps',
                    display_name='Denser Early ramp',
                    default=True,
                    tooltip=(
                        'Starts at no less than 50% video attention, then gradually '
                        'reduces toward the selected budget. The ramp targets 12 '
                        'additional percentage points per sampler step on '
                        'average. Budgets already at or above 50% are unchanged.'
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        video_budget=DEFAULT_VIDEO_BUDGET,
        denser_early_late_steps=True,
    ):
        passthrough = _vsa_passthrough(model)
        if passthrough is not None:
            return passthrough
        plan = read_plan(model).with_sparse(
            SparseRequest(
                video_budget=float(video_budget),
                denser_early_late_steps=bool(denser_early_late_steps),
                video_token_order=DEFAULT_VIDEO_TOKEN_ORDER,
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_sparse_status(patched)),
        )

    @classmethod
    def validate_inputs(cls, video_budget=DEFAULT_VIDEO_BUDGET):
        return _validate_budget('video_budget', video_budget) or True


class H3SparseAttentionAdvanced(io.ComfyNode):
    '''Sparse attention with explicit backend and sampling-step schedules.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3SparseAttentionAdvanced',
            display_name='H3 Sparse Attention (Advanced)',
            category=NODE_CATEGORY,
            description=(
                'Advanced fixed-density sparse attention for MiniMax H3. '
                'Video attention budget controls middle sampling steps. Early KV '
                'can be held or ramped toward that budget; Late KV overrides the '
                'final configured step count. The default eight-step Ramp starts '
                'at 68.33% and matches the simple node default cumulative video '
                'attention budget on a 20-step sampler. Video token order defaults '
                'to the measured 1x8x8 geometry and can be restored to stock raster '
                'order. Lower budgets are faster but can change the generated '
                'result, and the quality cost depends on the prompt and where '
                'removed in the denoising schedule. Kitchen INT8 64Q x 32KV '
                '(Quality) is the default; the other geometries and backends '
                'are available as explicit alternatives.'
            ),
            search_aliases=[
                'H3 sparse advanced',
                'H3 sparse schedule',
                'Sparse Sage advanced',
                'H3 early late KV',
                'H3 sparse backend',
            ],
            inputs=[
                io.Model.Input('model'),
                _video_budget_input(),
                io.Int.Input(
                    'early_steps',
                    display_name='Early steps',
                    default=_ADVANCED_DEFAULT_EARLY_STEPS,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip=(
                        'Hold uses Early KV for this many opening steps. Ramp '
                        'moves from Early KV toward Video attention budget over '
                        'this many steps. Set 0 to disable the early schedule. '
                        'Values above the displayed editing range remain valid.'
                    ),
                ),
                io.Float.Input(
                    'early_kv',
                    display_name='Early KV',
                    default=_ADVANCED_DEFAULT_EARLY_KV,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        'Hold uses this budget throughout the early window. Ramp '
                        'uses it as the starting budget. Increasing it can preserve '
                        'prompt/timeline adherence at the cost of speed.'
                    ),
                ),
                io.Int.Input(
                    'late_steps',
                    display_name='Late steps',
                    default=DEFAULT_LATE_STEPS,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip=(
                        'Number of final sampling steps that use Late KV. The default '
                        'is 0 because denser late steps have not shown enough benefit '
                        'to justify their compute cost. Values above the displayed '
                        'editing range remain valid.'
                    ),
                ),
                io.Float.Input(
                    'late_kv',
                    display_name='Late KV',
                    default=DEFAULT_EDGE_KV,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        'Video attention budget used during the late-step window. '
                        'Higher values retain more exact video attention at the '
                        'cost of speed.'
                    ),
                ),
                io.Combo.Input(
                    'backend',
                    display_name='Sparse backend',
                    options=list(SPARSE_BACKEND_PUBLIC_REQUESTS),
                    default=SPARSE_BACKEND_KITCHEN_64X32,
                    tooltip=(
                        'Kitchen INT8 64Q x 32KV (Quality) is the default, using '
                        'threshold routing when the native router is available. '
                        'Kitchen INT8 64Q x 128KV (Faster) was faster in the '
                        'measured full-block SM89 comparison, but its coarser '
                        'KV selections can change output quality and the speed '
                        'gain is not guaranteed on every GPU. '
                        'Kitchen INT8 64Q x 32KV (Quality) routes finer '
                        '32-token KV tiles over the same carrier: lower error '
                        'at a given budget, measured faster than 64Q x 64KV '
                        'at equal quality on SM89. '
                        'FROST BF16, BF16 Triton, and FP8 FlexAttention use '
                        '64Q x 64KV routing. Sparse Sage uses 128Q x 64KV on '
                        'SM80/86/87/89/120 and 64Q x 128KV on SM90. '
                        'Each alternative is selected explicitly. '
                        'Explicit backend choices fail if that backend is '
                        'unavailable and do not switch to another backend. '
                        'Bypass this node to force dense attention.'
                    ),
                ),
                io.Combo.Input(
                    'early_schedule',
                    display_name='Early schedule',
                    options=list(EARLY_SCHEDULE_OPTIONS),
                    default=_ADVANCED_DEFAULT_EARLY_SCHEDULE,
                    tooltip=(
                        'Ramp starts at Early KV and moves linearly toward Video '
                        'attention budget over Early steps. Hold keeps Early KV '
                        'constant instead. Set Early steps to 0 to disable either '
                        'schedule.'
                    ),
                ),
                io.Combo.Input(
                    'video_token_order',
                    display_name='Video token order',
                    options=list(VIDEO_TOKEN_ORDER_REQUESTS),
                    default=DEFAULT_VIDEO_TOKEN_ORDER,
                    tooltip=(
                        '1x8x8 is the measured default and groups each 64-token '
                        'router tile as temporal x height x width. The other '
                        '64-token geometries are experimental comparison arms. '
                        'Raster restores unchanged H3 target-video ordering.'
                    ),
                ),
                io.Combo.Input(
                    'head_budgets',
                    display_name='Per-head budgets',
                    options=list(HEAD_BUDGETS_REQUESTS),
                    default=DEFAULT_HEAD_BUDGETS,
                    tooltip=(
                        'Calibrated gives each attention head its own share of a '
                        "layer's video budget from a bundled calibration, at the "
                        'same total compute: heads that lose more when tiles are '
                        'dropped keep more. It applies to Kitchen INT8 64Q x 32KV '
                        'threshold routing with the 1x8x8 token order; other '
                        'backends, geometries and orders keep one shared budget. '
                        'Off restores one shared budget for every head.'
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        video_budget=DEFAULT_VIDEO_BUDGET,
        early_steps=_ADVANCED_DEFAULT_EARLY_STEPS,
        early_kv=_ADVANCED_DEFAULT_EARLY_KV,
        late_steps=DEFAULT_LATE_STEPS,
        late_kv=DEFAULT_EDGE_KV,
        backend=SPARSE_BACKEND_KITCHEN_64X32,
        early_schedule=_ADVANCED_DEFAULT_EARLY_SCHEDULE,
        video_token_order=DEFAULT_VIDEO_TOKEN_ORDER,
        head_budgets=DEFAULT_HEAD_BUDGETS,
    ):
        passthrough = _vsa_passthrough(model)
        if passthrough is not None:
            return passthrough
        plan = read_plan(model).with_sparse(
            SparseRequest(
                video_budget=float(video_budget),
                early_steps=int(early_steps),
                early_kv=float(early_kv),
                late_steps=int(late_steps),
                late_kv=float(late_kv),
                backend=backend,
                early_schedule=early_schedule,
                video_token_order=video_token_order,
                head_budgets=head_budgets,
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_sparse_status(patched)),
        )

    @classmethod
    def validate_inputs(
        cls,
        backend,
        early_schedule=_ADVANCED_DEFAULT_EARLY_SCHEDULE,
        video_budget=DEFAULT_VIDEO_BUDGET,
        early_steps=_ADVANCED_DEFAULT_EARLY_STEPS,
        early_kv=_ADVANCED_DEFAULT_EARLY_KV,
        late_steps=DEFAULT_LATE_STEPS,
        late_kv=DEFAULT_EDGE_KV,
        video_token_order=DEFAULT_VIDEO_TOKEN_ORDER,
        head_budgets=DEFAULT_HEAD_BUDGETS,
    ):
        if head_budgets is not None and head_budgets not in HEAD_BUDGETS_REQUESTS:
            return 'unknown per-head budget option %r' % head_budgets
        if backend is not None and backend not in SPARSE_BACKEND_COMPAT_REQUESTS:
            return 'unknown sparse backend %r' % backend
        if early_schedule is not None and early_schedule not in EARLY_SCHEDULE_OPTIONS:
            return 'unknown early schedule %r' % early_schedule
        if video_token_order is not None and video_token_order not in VIDEO_TOKEN_ORDER_REQUESTS:
            return 'unknown video token order %r' % video_token_order
        for name, value in (
            ('video_budget', video_budget),
            ('early_kv', early_kv),
            ('late_kv', late_kv),
        ):
            error = _validate_budget(name, value)
            if error is not None:
                return error
        for name, value in (
            ('early_steps', early_steps),
            ('late_steps', late_steps),
        ):
            error = _validate_steps(name, value)
            if error is not None:
                return error
        return True
