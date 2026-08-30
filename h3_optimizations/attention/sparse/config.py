'''Validated fixed-density Sparse Sage configuration.'''

from dataclasses import dataclass
import math

from ...plan import (
    EARLY_SCHEDULE_HOLD,
    EARLY_SCHEDULE_OPTIONS,
    EARLY_SCHEDULE_RAMP,
)

MODE_SAGE128 = 'sage128'
MODE_SAGE128_FUSED_QKV = 'sage128_fused_qkv'
IMPLEMENTED_MODES = (MODE_SAGE128, MODE_SAGE128_FUSED_QKV)
DENSITY_FIXED = 'fixed'

DENSER_EARLY_AVERAGE_EXTRA_KV = 0.12
DENSER_EARLY_MINIMUM_KV = 0.50


def _validate_budget(name, value):
    if value is None:
        return
    budget = float(value)
    if not math.isfinite(budget):
        raise ValueError('%s must be finite' % name)


@dataclass(frozen=True)
class HybridSparseConfig:
    mode: str = MODE_SAGE128
    video_budget: float = 0.5
    strict: bool = True
    density_mode: str = DENSITY_FIXED
    denser_early_late_steps: bool = False
    early_steps: int | None = None
    early_kv: float | None = None
    late_steps: int | None = None
    late_kv: float | None = None
    early_schedule: str = EARLY_SCHEDULE_HOLD
    step_video_budgets: tuple[float, ...] | None = None
    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                'sparse mode %r is unavailable; implemented modes: %s'
                % (self.mode, ', '.join(IMPLEMENTED_MODES))
            )
        _validate_budget('video_budget', self.video_budget)
        _validate_budget('early_kv', self.early_kv)
        _validate_budget('late_kv', self.late_kv)
        if self.early_schedule not in EARLY_SCHEDULE_OPTIONS:
            raise ValueError('unknown early schedule %r' % self.early_schedule)
        if self.step_video_budgets is not None:
            budgets = tuple(float(value) for value in self.step_video_budgets)
            if not budgets:
                raise ValueError('step_video_budgets must not be empty')
            for step_index, budget in enumerate(budgets):
                _validate_budget('step_video_budgets[%d]' % step_index, budget)
            object.__setattr__(self, 'step_video_budgets', budgets)
        if self.density_mode != DENSITY_FIXED:
            raise ValueError('only fixed Sparse Sage density is supported')
        values = (self.early_steps, self.early_kv, self.late_steps, self.late_kv)
        if any(value is not None for value in values):
            if not all(value is not None for value in values):
                raise ValueError('explicit early/late sparse schedule is incomplete')
            if self.denser_early_late_steps:
                raise ValueError('simple and explicit early/late schedules cannot be combined')
            for name, value in (
                ('early_steps', self.early_steps),
                ('late_steps', self.late_steps),
            ):
                if isinstance(value, bool) or int(value) != value or int(value) < 0:
                    raise ValueError('%s must be a non-negative integer' % name)
        if self.step_video_budgets is not None and (
            self.denser_early_late_steps
            or self.early_steps is not None
        ):
            raise ValueError(
                'per-step budgets cannot be combined with other sparse schedules'
            )

    @property
    def signature(self):
        return (
            self.mode,
            float(self.video_budget),
            bool(self.strict),
            self.density_mode,
            bool(self.denser_early_late_steps),
            self.early_schedule,
            None if self.early_steps is None else int(self.early_steps),
            None if self.early_kv is None else float(self.early_kv),
            None if self.late_steps is None else int(self.late_steps),
            None if self.late_kv is None else float(self.late_kv),
            self.step_video_budgets,
        )


def resolve_video_budget(config, step_index, total_steps, layer_index=None):
    budget = float(config.video_budget)
    step_index = int(step_index)
    total_steps = int(total_steps)
    if step_index < 0 or total_steps <= 0 or step_index >= total_steps:
        return budget

    if config.step_video_budgets is not None:
        if len(config.step_video_budgets) != total_steps:
            raise ValueError(
                'step_video_budgets has %d entries but the sampler has %d steps'
                % (len(config.step_video_budgets), total_steps)
            )
        return float(config.step_video_budgets[step_index])

    if config.early_steps is not None:
        in_early = step_index < int(config.early_steps)
        in_late = step_index >= total_steps - int(config.late_steps)
        early_budget = float(config.early_kv)
        if in_early and config.early_schedule == EARLY_SCHEDULE_RAMP:
            fraction = step_index / int(config.early_steps)
            early_budget = math.fsum((
                early_budget * (1.0 - fraction),
                budget * fraction,
            ))
        if in_early and in_late:
            return max(early_budget, float(config.late_kv))
        if in_early:
            return early_budget
        if in_late:
            return float(config.late_kv)
        return budget

    if not config.denser_early_late_steps:
        return budget
    peak_budget = max(budget, DENSER_EARLY_MINIMUM_KV)
    peak_extra = peak_budget - budget
    if peak_extra <= 0.0:
        return budget

    target_extra = min(
        DENSER_EARLY_AVERAGE_EXTRA_KV * total_steps,
        peak_extra * total_steps,
    )
    target_extra = max(target_extra, peak_extra)
    ramp_steps = min(
        total_steps,
        max(1, math.ceil(2.0 * target_extra / peak_extra) - 1),
    )
    if step_index >= ramp_steps:
        return budget
    if ramp_steps == 1:
        return peak_budget

    final_extra = 2.0 * target_extra / ramp_steps - peak_extra
    step_extra = peak_extra - (
        (peak_extra - final_extra) * step_index / (ramp_steps - 1)
    )
    return budget + step_extra
