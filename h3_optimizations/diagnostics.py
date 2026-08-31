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
    return active().stage(name)


class installed:
    '''Install a recorder for the calling thread, then restore the previous.'''

    __slots__ = ('_recorder', '_previous')

    def __init__(self, recorder):
        self._recorder = NULL_STAGES if recorder is None else recorder
        self._previous = None

    def __enter__(self):
        self._previous = active()
        _STATE.recorder = self._recorder
        return self._recorder

    def __exit__(self, exc_type, exc, traceback):
        _STATE.recorder = self._previous
        self._previous = None
        return False


STAGE_NAMES = (
    'attention_total',
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
    'sparse_carrier_prepare',
    'sparse_attention_kernel',
    'attention_out',
)
