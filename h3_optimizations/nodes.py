'''Composable production nodes for MiniMax H3 optimization.'''

import math

from comfy_api.latest import io, ui

from .apply import apply_plan
from .node_constants import NODE_CATEGORY
from .plan import (
    DEFAULT_EDGE_KV,
    DEFAULT_EDGE_STEPS,
    DEFAULT_LATE_STEPS,
    DEFAULT_VIDEO_BUDGET,
    DEFAULT_VIDEO_TOKEN_ORDER,
    EARLY_SCHEDULE_HOLD,
    EARLY_SCHEDULE_OPTIONS,
    SPARSE_BACKEND_COMPAT_REQUESTS,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_PUBLIC_REQUESTS,
    SparseRequest,
    VIDEO_TOKEN_ORDER_REQUESTS,
    read_plan,
)
from .status import (
    format_memory_status,
    format_sparse_status,
)

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
            'The request rounds up to whole KV tiles; non-video context and mixed '
            'boundary tiles stay dense. 1.0 retains the full video route. The '
            'displayed range is the recommended editing range; finite workflow '
            'values outside it saturate to at least one or at most all video tiles.'
        ),
    )


def _validate_budget(name, value):
    if not math.isfinite(float(value)):
        return '%s must be finite' % name
    return None


def _validate_steps(name, value):
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
                'Fixed-density sparse attention for MiniMax H3. Lower video '
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
                'final configured step count. Video token order defaults to the '
                'measured 1x8x8 geometry and can be restored to stock raster order. '
                'Lower budgets are faster but can change the generated result, and '
                'the quality cost depends on the prompt and where attention is '
                'removed in the denoising schedule. Kitchen INT8 64x64 is the '
                'default; FROST BF16, Sparse Sage, BF16 Triton, and FP8 '
                'FlexAttention are available as explicit alternatives.'
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
                    default=DEFAULT_EDGE_STEPS,
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
                    default=DEFAULT_EDGE_KV,
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
                    default=SPARSE_BACKEND_KITCHEN,
                    tooltip=(
                        'Kitchen INT8 uses the shipped native 64Q x 64KV path. '
                        'Kitchen INT8 64x128 is an experimental image-quality '
                        'arm with the same 64-row query routing but coarser '
                        '128-row KV selections. '
                        'FROST BF16 uses 64Q x 64KV routing and is available '
                        'only on SM89. '
                        'BF16 Triton and FP8 FlexAttention use the same 64Q x '
                        '64KV routing geometry. Sparse Sage uses its installed '
                        'kernel geometry. Each alternative is selected explicitly. '
                        'Explicit backend choices fail if that backend is '
                        'unavailable and do not switch to another backend. '
                        'Bypass this node to force dense attention.'
                    ),
                ),
                io.Combo.Input(
                    'early_schedule',
                    display_name='Early schedule',
                    options=list(EARLY_SCHEDULE_OPTIONS),
                    default=EARLY_SCHEDULE_HOLD,
                    tooltip=(
                        'Hold keeps Early KV constant for Early steps, matching '
                        'existing Advanced workflows. Ramp starts at Early KV and '
                        'moves linearly toward Video attention budget over Early '
                        'steps. Set Early steps to 0 to disable either schedule.'
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
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        video_budget=DEFAULT_VIDEO_BUDGET,
        early_steps=DEFAULT_EDGE_STEPS,
        early_kv=DEFAULT_EDGE_KV,
        late_steps=DEFAULT_LATE_STEPS,
        late_kv=DEFAULT_EDGE_KV,
        backend=SPARSE_BACKEND_KITCHEN,
        early_schedule=EARLY_SCHEDULE_HOLD,
        video_token_order=DEFAULT_VIDEO_TOKEN_ORDER,
    ):
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
        early_schedule=EARLY_SCHEDULE_HOLD,
        video_budget=DEFAULT_VIDEO_BUDGET,
        early_steps=DEFAULT_EDGE_STEPS,
        early_kv=DEFAULT_EDGE_KV,
        late_steps=DEFAULT_LATE_STEPS,
        late_kv=DEFAULT_EDGE_KV,
        video_token_order=DEFAULT_VIDEO_TOKEN_ORDER,
    ):
        if backend not in SPARSE_BACKEND_COMPAT_REQUESTS:
            return 'unknown sparse backend %r' % backend
        if early_schedule not in EARLY_SCHEDULE_OPTIONS:
            return 'unknown early schedule %r' % early_schedule
        if video_token_order not in VIDEO_TOKEN_ORDER_REQUESTS:
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
