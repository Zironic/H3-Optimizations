'''Opt-in stage attribution for the production QKV/attention path.

Production runs with no recorder installed, and the hook then costs two
C-level method calls per site -- no allocation, no import of torch, and no
synchronization anywhere. That is the point: the path a benchmark measures has
to be the path production runs, or the measurement describes a fork of the
code rather than the code.

A recorder is any object exposing ``stage(name)``, returning a context manager
that spans the region. ``benchmarks/h3_stage_recorder.py`` provides the CUDA
event implementation; nothing in this package creates CUDA events or
synchronizes, so installing a recorder cannot introduce a stall by itself.

Regions nest. ``qkv_producer_total`` contains ``qkv_linear``, and
``attention_total`` contains all of it. Outer regions are authoritative:
a sum of children silently omits whatever sits between them.
Reentering an active name records only its outer scope; sequential calls
with the same name remain separate samples.
'''

from __future__ import annotations

import threading


class _NullStage:
    '''A context manager that does nothing, allocated once.'''

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False

    def stage(self, name):
        del name
        return self


NULL_STAGES = _NullStage()

_STATE = threading.local()


def active():
    '''The installed recorder, or the shared no-op.'''
    return getattr(_STATE, 'recorder', NULL_STAGES)


def stage(name):
    '''Span one attribution region by name.'''
    recorder = active()
    if recorder is NULL_STAGES:
        return NULL_STAGES
    return _RecordedStage(recorder, _STATE.names, name)


class _RecordedStage:
    __slots__ = ('recorder', 'names', 'name', 'context')

    def __init__(self, recorder, names, name):
        self.recorder = recorder
        self.names = names
        self.name = name
        self.context = None

    def __enter__(self):
        if self.name in self.names:
            return None
        self.names.add(self.name)
        try:
            self.context = self.recorder.stage(self.name)
            return self.context.__enter__()
        except BaseException:
            self.context = None
            self.names.remove(self.name)
            raise

    def __exit__(self, exc_type, exc, traceback):
        if self.context is None:
            return False
        try:
            return self.context.__exit__(exc_type, exc, traceback)
        finally:
            self.context = None
            self.names.remove(self.name)


class installed:
    '''Install a recorder for the calling thread, then restore the previous.'''

    __slots__ = ('_recorder', '_previous', '_previous_names')

    def __init__(self, recorder):
        self._recorder = NULL_STAGES if recorder is None else recorder
        self._previous = None
        self._previous_names = None

    def __enter__(self):
        self._previous = active()
        self._previous_names = getattr(_STATE, 'names', None)
        _STATE.recorder = self._recorder
        _STATE.names = self._previous_names if self._previous is self._recorder else set()
        return self._recorder

    def __exit__(self, exc_type, exc, traceback):
        _STATE.recorder = self._previous
        _STATE.names = self._previous_names
        self._previous = None
        self._previous_names = None
        return False


STAGE_NAMES = (
    'block_total',
    'adaln_proj',
    'attention_total',
    'attention_residual_gate',
    'norm2_modulation',
    'norm2_modulation_quantize',
    'mlp_fc1',
    'mlp_fc1_fc2',
    'mlp_swiglu_fc2',
    'mlp_swiglu_fc2_gated',
    'final_mlp_gate',
    'qkv_producer_total',
    'qkv_linear',
    'qk_norm_rope',
    'q_activation_quant',
    'fused_q_projection',
    'anchor_projection',
    'anchor_selection',
    'producer_create',
    'routing_summary_generation',
    'full_carrier_pack',
    'qk_carrier_pack',
    'qk_pack_input_contiguous',
    'v_carrier_pack',
    'v_amax_update',
    'v_reprojection',
    'v_retention_copy',
    'carrier_finalize',
    'sparse_route',
    'v_block_route_align',
    'sparse_carrier_prepare',
    'sparse_attention_kernel',
    'dense_attention_kernel',
    'attention_out',
)
