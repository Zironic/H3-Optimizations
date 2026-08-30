"""Defer SageAttention's stock FP8-V preprocessing until after fused QKV release.

Stock ``per_channel_fp8`` allocates a full BF16 transposed/padded V temporary.
Running it inside ``prepare`` leaves that temporary live alongside the original
fused BF16 QKV projection and sets the complete-call peak before the caller can
release Q/K/V.  This compatibility layer changes only the ordering:

1. quantize Q/K;
2. copy V once into independent contiguous BF16 storage;
3. return to ``forward.py``, which deletes all fused-QKV views;
4. run stock FP8-V preprocessing and the low-level SM89 kernel in ``execute``.

The contiguous BF16 snapshot is smaller than fused QKV by 3x and also keeps the
closed Sage V preprocessor away from H3's dangerous 21,504-element sequence
stride.  No numerical algorithm changes.
"""

import logging

import torch

from . import sage_mem_eff as impl
from . import stats

_execute_prepared = impl.SM89SageMemoryEfficientBackend.execute


def _independent_contiguous_v(v):
    snapshot = v.contiguous()
    if snapshot.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
        snapshot = v.clone(memory_format=torch.contiguous_format)
    return snapshot


def _prepare(self, q, k, v, *, layer_index, transformer_options):
    batch, heads, sequence, head_dim = self._validate(q, k, v)
    q_int8, q_scale, k_int8, k_scale = self.quantizer(
        q,
        k,
        None,
        BLKQ=128,
        WARPQ=32,
        BLKK=64,
        WARPK=64,
        tensor_layout="HND",
    )

    # This is the only V allocation made while the fused QKV source is alive.
    # Store it in the existing ``v_fp8`` field until execute; the dataclass is an
    # internal carrier and remains mutable.  ``v_scale=None`` marks this state.
    v_snapshot = _independent_contiguous_v(v)

    stats.observe_sequence(sequence)
    if not self._logged:
        logging.debug(
            "[H3 attention] sage_mem_eff active: SageAttention %s, HND, "
            "per-thread int64 Q/K, deferred stock FP8 V, accumulation=%s, "
            "kernel=%s via %s",
            self.api.version,
            self.api.accumulation,
            self.api.kernel_name,
            self.api.kernel_source,
        )
        self._logged = True

    return impl.PreparedSM89(
        q_int8=q_int8,
        q_scale=q_scale,
        k_int8=k_int8,
        k_scale=k_scale,
        v_fp8=v_snapshot,
        v_scale=None,
        output_dtype=q.dtype,
        layer_index=int(layer_index),
        sequence=int(sequence),
        heads=int(heads),
        head_dim=int(head_dim),
        softmax_scale=head_dim**-0.5,
        kernel=self.api.kernel,
        kernel_name=self.api.kernel_name,
    )


def _execute(self, prepared):
    if prepared.v_scale is not None:
        return _execute_prepared(self, prepared)

    # ``forward.py`` has deleted the original q/k/v views before entering here,
    # so Sage's large transpose/pad temporary no longer overlaps fused QKV.
    v_source = prepared.v_fp8
    v_fp8, v_scale, _ = self.api.per_channel_fp8(
        v_source,
        tensor_layout="HND",
        scale_max=self.api.v_scale_max,
        smooth_v=False,
    )
    prepared.v_fp8 = None
    prepared.v_scale = None
    del v_source

    output = torch.empty(
        prepared.q_int8.shape,
        dtype=prepared.output_dtype,
        device=prepared.q_int8.device,
    )
    try:
        prepared.kernel(
            prepared.q_int8,
            prepared.k_int8,
            v_fp8,
            output,
            prepared.q_scale,
            prepared.k_scale,
            v_scale,
            1,
            0,
            3,
            prepared.softmax_scale,
            0,
        )
    except Exception as exc:
        stats.increment("kernel_errors")
        device = prepared.q_int8.device
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        raise impl.EfficientSageError(
            "sage_mem_eff kernel failed: layer=%d sequence=%d heads=%d head_dim=%d "
            "dtype=%s device=%s kernel=%s SageAttention=%s"
            % (
                prepared.layer_index,
                prepared.sequence,
                prepared.heads,
                prepared.head_dim,
                prepared.output_dtype,
                gpu,
                prepared.kernel_name,
                self.api.version,
            )
        ) from exc
    stats.increment("executed")
    return output


impl.SM89SageMemoryEfficientBackend.prepare = _prepare
impl.SM89SageMemoryEfficientBackend.execute = _execute
