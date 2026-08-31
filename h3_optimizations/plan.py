'''Immutable, order-independent configuration for H3 optimizations.'''

from __future__ import annotations

from dataclasses import dataclass, replace
import math

PLAN_KEY = 'h3_optimizations_plan'
STATUS_KEY = 'h3_optimizations_status'
PLAN_VERSION = 3

ATTENTION_AUTO = 'auto'
ATTENTION_EXISTING = 'existing'
ATTENTION_REQUESTS = (ATTENTION_AUTO, ATTENTION_EXISTING)

FUSED_QKV_AUTO = 'auto'
FUSED_QKV_OFF = 'off'
FUSED_QKV_REQUIRED = 'required'
# Internal requests used by the public precision policy.
FUSED_QKV_FORCE_BF16 = 'force_bf16'
FUSED_QKV_FORCE_QUANT = 'force_quant'
FUSED_QKV_PRESERVE_BF16 = 'preserve_bf16'
FUSED_QKV_REQUESTS = (
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    FUSED_QKV_REQUIRED,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_PRESERVE_BF16,
)

QKV_STREAMING_OFF = 'off'
QKV_STREAMING_AUTO = 'auto'
QKV_STREAMING_FORCED = 'forced'
QKV_STREAMING_REQUESTS = (
    QKV_STREAMING_OFF,
    QKV_STREAMING_AUTO,
    QKV_STREAMING_FORCED,
)

MLP_MEMORY_AUTO = 'auto'
MLP_MEMORY_OFF = 'off'
MLP_MEMORY_BF16 = 'bf16'
MLP_MEMORY_FORCE_QUANT = 'force_quant'
MLP_MEMORY_PRESERVE = 'preserve_precision'
MLP_MEMORY_LEGACY_BF16 = 'legacy_bf16'
MLP_MEMORY_LEGACY_NATIVE = 'legacy_native'
MLP_MEMORY_LEGACY_CONVROT_REQUIRED = 'legacy_convrot_2slice_required'
MLP_MEMORY_REQUESTS = (
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    MLP_MEMORY_BF16,
    MLP_MEMORY_FORCE_QUANT,
    MLP_MEMORY_PRESERVE,
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_NATIVE,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
)

SPARSE_BACKEND_AUTO = 'auto'
SPARSE_BACKEND_SAGE = 'Sparse Sage'
SPARSE_BACKEND_TRITON = 'BF16 Triton'
SPARSE_BACKEND_TRITON_LEGACY = 'INT8 Triton'
SPARSE_BACKEND_FLEX = 'FP8 FlexAttention'
SPARSE_BACKEND_FROST = 'FROST BF16 (SM89)'
SPARSE_BACKEND_KITCHEN = 'Kitchen INT8'
SPARSE_BACKEND_KITCHEN_64X128 = 'Kitchen INT8 64x128 (experimental)'
SPARSE_BACKEND_KITCHEN_LEGACY = 'Kitchen INT8 (experimental)'
SPARSE_BACKEND_REQUESTS = (
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_FROST,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_KITCHEN_64X128,
)

EMBEDDING_MEMORY_AUTO = 'auto'
EMBEDDING_MEMORY_STOCK = 'stock'
EMBEDDING_MEMORY_RELEASE = 'release'
EMBEDDING_MEMORY_REQUESTS = (
    EMBEDDING_MEMORY_AUTO,
    EMBEDDING_MEMORY_STOCK,
    EMBEDDING_MEMORY_RELEASE,
)

V_MEMORY_RETAIN = 'retain'
V_MEMORY_TWO_PASS = 'two_pass'
V_MEMORY_REQUESTS = (
    V_MEMORY_RETAIN,
    V_MEMORY_TWO_PASS,
)
# Two-pass V started out Kitchen-only. It now also covers the Sage FP8
# carriers, so the neutral names above are canonical; these aliases keep the
# original spelling working for callers that still use it.
KITCHEN_V_MEMORY_RETAIN = V_MEMORY_RETAIN
KITCHEN_V_MEMORY_TWO_PASS = V_MEMORY_TWO_PASS
KITCHEN_V_MEMORY_REQUESTS = V_MEMORY_REQUESTS
SPARSE_BACKEND_COMPAT_REQUESTS = (
    *SPARSE_BACKEND_REQUESTS,
    SPARSE_BACKEND_KITCHEN_LEGACY,
    SPARSE_BACKEND_TRITON_LEGACY,
)
SPARSE_BACKEND_PUBLIC_REQUESTS = (
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_KITCHEN_64X128,
    SPARSE_BACKEND_FROST,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
    SPARSE_BACKEND_FLEX,
)

MIN_CHUNK_ROWS = 256
MAX_CHUNK_ROWS = 65_536
CHUNK_ALIGNMENT = 256
DENSITY_FIXED = 'fixed'
DEFAULT_VIDEO_BUDGET = 0.15
DEFAULT_EDGE_STEPS = 4
DEFAULT_LATE_STEPS = 0
DEFAULT_EDGE_KV = 0.5
VIDEO_TOKEN_ORDER_1X8X8 = '1x8x8'
VIDEO_TOKEN_ORDER_1X16X4 = '1x16x4'
VIDEO_TOKEN_ORDER_4X4X4 = '4x4x4'
VIDEO_TOKEN_ORDER_RASTER = 'Raster (stock H3 order)'
DEFAULT_VIDEO_TOKEN_ORDER = VIDEO_TOKEN_ORDER_1X8X8
VIDEO_TOKEN_ORDER_REQUESTS = (
    VIDEO_TOKEN_ORDER_1X8X8,
    VIDEO_TOKEN_ORDER_1X16X4,
    VIDEO_TOKEN_ORDER_4X4X4,
    VIDEO_TOKEN_ORDER_RASTER,
)
EARLY_SCHEDULE_HOLD = 'Hold'
EARLY_SCHEDULE_RAMP = 'Ramp'
EARLY_SCHEDULE_OPTIONS = (
    EARLY_SCHEDULE_HOLD,
    EARLY_SCHEDULE_RAMP,
)


def _validate_sparse_budget(name, value):
    budget = float(value)
    if not math.isfinite(budget):
        raise ValueError('%s must be finite' % name)


def _validate_edge_schedule(early_steps, early_kv, late_steps, late_kv):
    values = (early_steps, early_kv, late_steps, late_kv)
    if not any(value is not None for value in values):
        return
    if not all(value is not None for value in values):
        raise ValueError(
            'early_steps, early_kv, late_steps, and late_kv must be set together'
        )
    for name, value in (
        ('early_steps', early_steps),
        ('late_steps', late_steps),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError('%s must be a non-negative integer' % name)
    _validate_sparse_budget('early_kv', early_kv)
    _validate_sparse_budget('late_kv', late_kv)


@dataclass(frozen=True)
class MemoryRequest:
    '''Execution and activation-memory options owned by the memory node.'''

    attention: str = ATTENTION_AUTO
    fused_qkv: str = FUSED_QKV_AUTO
    mlp_memory: str = MLP_MEMORY_AUTO
    chunk_rows: int = 4096
    qkv_streaming: str = QKV_STREAMING_AUTO
    prefer_held_weights: bool = True
    mlp_strict: bool = False
    embedding_memory: str = EMBEDDING_MEMORY_AUTO
    attention_v_memory: str = V_MEMORY_RETAIN

    def __post_init__(self):
        if self.attention not in ATTENTION_REQUESTS:
            raise ValueError('unknown dense attention request %r' % self.attention)
        if self.qkv_streaming not in QKV_STREAMING_REQUESTS:
            raise ValueError('unknown QKV streaming request %r' % self.qkv_streaming)

        # The public Preserve precision policy is represented today by keeping
        # the existing attention backend and disabling QKV requantization. Turn
        # that specific combination into an internal BF16-preserving request so
        # a real BF16 checkpoint can still use bounded QKV projection. Explicit
        # QKV streaming Off keeps the ordinary upstream QKV path instead.
        if (
            self.qkv_streaming != QKV_STREAMING_OFF
            and self.attention == ATTENTION_EXISTING
            and self.fused_qkv == FUSED_QKV_OFF
        ):
            object.__setattr__(
                self,
                'fused_qkv',
                FUSED_QKV_PRESERVE_BF16,
            )

        if self.fused_qkv not in FUSED_QKV_REQUESTS:
            raise ValueError('unknown fused QKV request %r' % self.fused_qkv)
        if self.mlp_memory not in MLP_MEMORY_REQUESTS:
            raise ValueError('unknown MLP memory request %r' % self.mlp_memory)
        if self.embedding_memory not in EMBEDDING_MEMORY_REQUESTS:
            raise ValueError(
                'unknown embedding memory request %r' % self.embedding_memory
            )
        if self.attention_v_memory not in V_MEMORY_REQUESTS:
            raise ValueError(
                'unknown attention V memory request %r' % self.attention_v_memory
            )
        chunk_rows = int(self.chunk_rows)
        if (
            isinstance(self.chunk_rows, bool)
            or chunk_rows != self.chunk_rows
            or chunk_rows <= 0
        ):
            raise ValueError('chunk_rows must be a positive integer')

    @property
    def signature(self):
        return (
            self.attention,
            self.fused_qkv,
            self.mlp_memory,
            int(self.chunk_rows),
            self.qkv_streaming,
            bool(self.prefer_held_weights),
            bool(self.mlp_strict),
            self.embedding_memory,
            self.attention_v_memory,
        )


@dataclass(frozen=True)
class SparseRequest:
    '''Fixed-density sparse attention request.'''

    video_budget: float = DEFAULT_VIDEO_BUDGET
    denser_early_late_steps: bool = False
    early_steps: int | None = None
    early_kv: float | None = None
    late_steps: int | None = None
    late_kv: float | None = None
    backend: str = SPARSE_BACKEND_AUTO
    early_schedule: str = EARLY_SCHEDULE_HOLD
    step_video_budgets: tuple[float, ...] | None = None
    video_token_order: str = DEFAULT_VIDEO_TOKEN_ORDER

    def __post_init__(self):
        _validate_sparse_budget('video_budget', self.video_budget)
        if self.backend == SPARSE_BACKEND_KITCHEN_LEGACY:
            object.__setattr__(self, 'backend', SPARSE_BACKEND_KITCHEN)
        if self.backend == SPARSE_BACKEND_TRITON_LEGACY:
            object.__setattr__(self, 'backend', SPARSE_BACKEND_TRITON)
        if self.backend not in SPARSE_BACKEND_REQUESTS:
            raise ValueError('unknown sparse backend request %r' % self.backend)
        if self.early_schedule not in EARLY_SCHEDULE_OPTIONS:
            raise ValueError('unknown early schedule %r' % self.early_schedule)
        if self.video_token_order not in VIDEO_TOKEN_ORDER_REQUESTS:
            raise ValueError(
                'unknown video token order %r' % self.video_token_order
            )
        _validate_edge_schedule(
            self.early_steps,
            self.early_kv,
            self.late_steps,
            self.late_kv,
        )
        if self.step_video_budgets is not None:
            budgets = tuple(float(value) for value in self.step_video_budgets)
            if not budgets:
                raise ValueError('step_video_budgets must not be empty')
            for step_index, budget in enumerate(budgets):
                _validate_sparse_budget(
                    'step_video_budgets[%d]' % step_index,
                    budget,
                )
            object.__setattr__(self, 'step_video_budgets', budgets)
        if self.advanced_schedule and self.denser_early_late_steps:
            raise ValueError(
                'explicit early/late budgets cannot be combined with the '
                'simple denser early schedule'
            )
        if self.step_video_budgets is not None and (
            self.denser_early_late_steps
            or self.advanced_schedule
        ):
            raise ValueError(
                'per-step budgets cannot be combined with other sparse schedules'
            )

    @property
    def advanced_schedule(self):
        return self.early_steps is not None

    @property
    def signature(self):
        return (
            float(self.video_budget),
            DENSITY_FIXED,
            bool(self.denser_early_late_steps),
            self.backend,
            self.early_schedule,
            None if self.early_steps is None else int(self.early_steps),
            None if self.early_kv is None else float(self.early_kv),
            None if self.late_steps is None else int(self.late_steps),
            None if self.late_kv is None else float(self.late_kv),
            self.step_video_budgets,
            self.video_token_order,
        )


@dataclass(frozen=True)
class H3OptimizationPlan:
    '''Complete composable request carried by one cloned ModelPatcher.'''

    version: int = PLAN_VERSION
    memory: MemoryRequest | None = None
    sparse: SparseRequest | None = None

    def __post_init__(self):
        if int(self.version) != PLAN_VERSION:
            raise ValueError(
                'unsupported H3 optimization plan version %r' % self.version
            )

    def with_memory(self, request: MemoryRequest):
        if not isinstance(request, MemoryRequest):
            raise TypeError('request must be MemoryRequest')
        if self.memory is not None and self.memory != request:
            raise ValueError(
                'a different H3 Memory Optimization node is already present; '
                'remove one instead of relying on node order'
            )
        return replace(self, memory=request)

    def with_sparse(self, request: SparseRequest):
        if not isinstance(request, SparseRequest):
            raise TypeError('request must be SparseRequest')
        if self.sparse is not None and self.sparse != request:
            raise ValueError(
                'a different H3 Sparse Attention node is already present; '
                'remove one instead of relying on node order'
            )
        return replace(self, sparse=request)

    @property
    def signature(self):
        return (
            int(self.version),
            None if self.memory is None else self.memory.signature,
            None if self.sparse is None else self.sparse.signature,
        )


def read_plan(model):
    options = getattr(model, 'model_options', {}) or {}
    plan = options.get(PLAN_KEY)
    if plan is None:
        return H3OptimizationPlan()
    if not isinstance(plan, H3OptimizationPlan):
        raise TypeError('%s does not contain an H3OptimizationPlan' % PLAN_KEY)
    return plan
