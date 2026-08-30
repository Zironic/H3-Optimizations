"""Shared prepared-QKV Sage architecture infrastructure."""

from dataclasses import dataclass
import importlib.metadata
import logging
import sys

import torch

from .. import stats
from ..sage_mem_eff import EfficientSageError, max_linear_offset

SUPPORTED_SAGE_PREFIXES = ("2.2.",)
SIGNED_OFFSET_LIMIT = (1 << 31) - 1
LOG_PREFIX = "[H3 attention]"


@dataclass(frozen=True)
class KernelBinding:
    fn: object
    name: str
    source: str


@dataclass
class PreparedArchitecture:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_source: torch.Tensor
    output_dtype: torch.dtype
    layer_index: int
    sequence: int
    heads: int
    head_dim: int
    softmax_scale: float


def load_core():
    try:
        version = importlib.metadata.version("sageattention")
        import sageattention.core as core
    except Exception as exc:
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "prepared Sage backends require SageAttention 2.2.x"
        ) from exc

    if not version.startswith(SUPPORTED_SAGE_PREFIXES):
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "prepared Sage backends were validated against SageAttention "
            "2.2.x; installed version is %s" % version
        )
    return version, core


def _append_unique(items, item, label):
    if item is None:
        return
    if not any(item is existing for existing, _ in items):
        items.append((item, label))


def _collect(found, label, module):
    if module is not None:
        found.append((label, module))


def _current_family_modules(family):
    """SageAttention's present-day per-architecture module layout."""
    found = []
    if family == "sm80":
        try:
            from sageattention import sm80_compile
            _collect(found, "sageattention.sm80_compile", sm80_compile)
        except Exception:
            pass
        try:
            from sageattention import _qattn_sm80
            _collect(found, "sageattention._qattn_sm80", _qattn_sm80)
        except Exception:
            pass
        try:
            from sageattention import qattn_sm80
            _collect(found, "sageattention.qattn_sm80", qattn_sm80)
        except Exception:
            pass
    elif family == "sm89":
        try:
            from sageattention import sm89_compile
            _collect(found, "sageattention.sm89_compile", sm89_compile)
        except Exception:
            pass
        try:
            from sageattention import _qattn_sm89
            _collect(found, "sageattention._qattn_sm89", _qattn_sm89)
        except Exception:
            pass
        try:
            from sageattention import qattn_sm89
            _collect(found, "sageattention.qattn_sm89", qattn_sm89)
        except Exception:
            pass
    elif family == "sm90":
        try:
            from sageattention import sm90_compile
            _collect(found, "sageattention.sm90_compile", sm90_compile)
        except Exception:
            pass
        try:
            from sageattention import _qattn_sm90
            _collect(found, "sageattention._qattn_sm90", _qattn_sm90)
        except Exception:
            pass
        try:
            from sageattention import qattn_sm90
            _collect(found, "sageattention.qattn_sm90", qattn_sm90)
        except Exception:
            pass
    return found


def _legacy_family_modules(family):
    """The ``sage_attention`` spelling some older builds installed under."""
    found = []
    if family == "sm80":
        try:
            from sage_attention import sm80_compile
            _collect(found, "sage_attention.sm80_compile", sm80_compile)
        except Exception:
            pass
    elif family == "sm89":
        try:
            from sage_attention import sm89_compile
            _collect(found, "sage_attention.sm89_compile", sm89_compile)
        except Exception:
            pass
    elif family == "sm90":
        try:
            from sage_attention import sm90_compile
            _collect(found, "sage_attention.sm90_compile", sm90_compile)
        except Exception:
            pass
    return found


def _optional_kernel_modules(family):
    """Force-import every SageAttention submodule that might carry a kernel.

    SageAttention has moved its compiled exports between releases, so this
    probes each layout it has shipped, in preference order. The architecture
    set is finite, so the module names are written as literal imports rather
    than built from ``family``: nothing is resolved at runtime that a reader
    -- or the Registry package scanner -- cannot see statically.
    """
    found = _current_family_modules(family)
    try:
        from sageattention import _ops
        _collect(found, "sageattention._ops", _ops)
    except Exception:
        pass
    found.extend(_legacy_family_modules(family))
    try:
        from sage_attention import _ops as legacy_ops
        _collect(found, "sage_attention._ops", legacy_ops)
    except Exception:
        pass
    return found


def _candidate_surfaces(core, family, public_names):
    surfaces = []
    _append_unique(surfaces, core, "sageattention.core")
    _append_unique(
        surfaces,
        getattr(core, "%s_compile" % family, None),
        "core.%s_compile" % family,
    )

    for public_name in public_names:
        public_fn = getattr(core, public_name, None)
        globals_dict = (
            getattr(public_fn, "__globals__", {})
            if public_fn is not None
            else {}
        )
        for name, value in globals_dict.items():
            if name.startswith("__"):
                continue
            label = "%s.__globals__[%s]" % (public_name, name)
            _append_unique(surfaces, value, label)
            _append_unique(
                surfaces,
                getattr(value, "ops", None),
                label + ".ops",
            )

    for module_name, module in _optional_kernel_modules(family):
        _append_unique(surfaces, module, module_name)
        _append_unique(
            surfaces,
            getattr(module, "ops", None),
            module_name + ".ops",
        )

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(
            ("sageattention", "sage_attention")
        ):
            continue
        _append_unique(surfaces, module, module_name)
        _append_unique(
            surfaces,
            getattr(module, "ops", None),
            module_name + ".ops",
        )
    return surfaces


def _registered_ops(kernel_names):
    wanted = set(kernel_names)
    found = {}
    try:
        names = torch._C._dispatch_get_all_op_names()
    except Exception:
        return found

    for full_name in names:
        if "::" not in full_name:
            continue
        namespace, basename = full_name.split("::", 1)
        if basename not in wanted:
            continue
        try:
            op = getattr(
                getattr(torch.ops, namespace),
                basename,
            )
        except Exception:
            continue
        if callable(op):
            found.setdefault(
                basename,
                KernelBinding(
                    op,
                    basename,
                    "torch.ops.%s" % namespace,
                ),
            )
    return found


def resolve_kernel(core, family, kernel_names, public_names):
    surfaces = _candidate_surfaces(
        core,
        family,
        public_names,
    )
    dispatch = _registered_ops(kernel_names)
    for kernel_name in kernel_names:
        for surface, label in surfaces:
            try:
                kernel = getattr(surface, kernel_name)
            except Exception:
                continue
            if callable(kernel):
                return KernelBinding(
                    kernel,
                    kernel_name,
                    label,
                )
        if kernel_name in dispatch:
            return dispatch[kernel_name]

    raise EfficientSageError(
        "SageAttention 2.2.x has no callable %s kernel export; "
        "expected %s"
        % (family.upper(), ", ".join(kernel_names))
    )


def independent_contiguous(tensor):
    result = tensor.contiguous()
    if (
        result.untyped_storage().data_ptr()
        == tensor.untyped_storage().data_ptr()
    ):
        result = tensor.clone(
            memory_format=torch.contiguous_format
        )
    return result


def guard_signed_offsets(tensor):
    """Copy before stock Triton/CUDA quantizers can wrap int32 offsets."""
    if max_linear_offset(tensor) <= SIGNED_OFFSET_LIMIT:
        return tensor

    stats.increment("qk_guard_copies")
    contiguous = tensor.contiguous()
    if max_linear_offset(contiguous) > SIGNED_OFFSET_LIMIT:
        raise EfficientSageError(
            "prepared Sage cannot safely quantize a tensor with "
            "more than 2**31 addressable elements"
        )
    return contiguous


class ArchitectureBackend:
    name = "sage_mem_eff_arch"
    capabilities = frozenset()
    projected_qkv_format = None
    projected_q_tile = 1
    projected_k_tile = 1
    requires_h3_triton = False
    requires_registered_sage = True
    requires_runtime_context = False
    approximate = False
    runtime_listeners = ()

    def __init__(self, allow_cpu_for_tests=False):
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self._logged = False
        stats.increment("configured")

    def validate(self, q, k, v):
        if q.shape != k.shape or q.shape != v.shape:
            raise EfficientSageError(
                "%s requires equal Q/K/V shapes; got %s %s %s"
                % (
                    self.name,
                    tuple(q.shape),
                    tuple(k.shape),
                    tuple(v.shape),
                )
            )
        if q.ndim != 4:
            raise EfficientSageError(
                "%s expects HND rank-4 tensors" % self.name
            )
        batch, heads, sequence, head_dim = q.shape
        if batch != 1:
            raise EfficientSageError(
                "%s expects H3 attention batch 1; got %d"
                % (self.name, batch)
            )
        if head_dim != 128:
            raise EfficientSageError(
                "%s supports H3 head_dim 128; got %d"
                % (self.name, head_dim)
            )
        if q.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise EfficientSageError(
                "%s requires fp16 or bf16; got %s"
                % (self.name, q.dtype)
            )
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise EfficientSageError(
                "%s received differing Q/K/V dtypes"
                % self.name
            )
        if q.device != k.device or q.device != v.device:
            raise EfficientSageError(
                "%s received differing Q/K/V devices"
                % self.name
            )
        if (
            q.stride(-1) != 1
            or k.stride(-1) != 1
            or v.stride(-1) != 1
        ):
            raise EfficientSageError(
                "%s requires a contiguous head dimension"
                % self.name
            )

        if not self.allow_cpu_for_tests:
            if not q.is_cuda:
                raise EfficientSageError(
                    "%s requires CUDA" % self.name
                )
            capability = tuple(
                torch.cuda.get_device_capability(q.device)
            )
            if capability not in self.capabilities:
                supported = ", ".join(
                    "SM%d%d" % item
                    for item in sorted(self.capabilities)
                )
                raise EfficientSageError(
                    "%s supports %s; device capability is %d.%d"
                    % (
                        self.name,
                        supported,
                        capability[0],
                        capability[1],
                    )
                )
        return batch, heads, sequence, head_dim

    def quantize_projected_q(self, q):
        dummy_k = q[..., :1, :].contiguous()
        q_int8, q_scale, _k_int8, _k_scale = self.quantize_projected_qk(
            q,
            dummy_k,
        )
        return q_int8, q_scale

    def quantize_projected_k(self, k):
        dummy_q = k[..., :1, :].contiguous()
        _q_int8, _q_scale, k_int8, k_scale = self.quantize_projected_qk(
            dummy_q,
            k,
        )
        return k_int8, k_scale

    def prepared(
        self,
        q,
        q_int8,
        q_scale,
        k_int8,
        k_scale,
        v_source,
        *,
        layer_index,
        heads,
        sequence,
        head_dim,
    ):
        stats.observe_sequence(sequence)
        return PreparedArchitecture(
            q_int8=q_int8,
            q_scale=q_scale,
            k_int8=k_int8,
            k_scale=k_scale,
            v_source=v_source,
            output_dtype=q.dtype,
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            head_dim=int(head_dim),
            softmax_scale=head_dim**-0.5,
        )

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        del transformer_options
        if getattr(projected, "qk_format", None) != self.projected_qkv_format:
            raise EfficientSageError(
                "%s received projected Q/K format %r; expected %r"
                % (
                    self.name,
                    getattr(projected, "qk_format", None),
                    self.projected_qkv_format,
                )
            )
        if int(projected.layer_index) != int(layer_index):
            raise EfficientSageError(
                "projected Sage QKV layer %d does not match attention layer %d"
                % (projected.layer_index, layer_index)
            )
        shape = (
            1,
            int(projected.heads),
            int(projected.sequence),
            int(projected.head_dim),
        )
        if int(projected.head_dim) != 128:
            raise EfficientSageError(
                "%s supports projected head_dim 128" % self.name
            )
        if (
            tuple(projected.q_int8.shape) != shape
            or tuple(projected.k_int8.shape) != shape
            or tuple(projected.v.shape) != shape
        ):
            raise EfficientSageError(
                "%s received an invalid projected QKV shape" % self.name
            )
        if (
            projected.q_int8.dtype != torch.int8
            or projected.k_int8.dtype != torch.int8
            or projected.q_scale.dtype != torch.float32
            or projected.k_scale.dtype != torch.float32
        ):
            raise EfficientSageError(
                "%s received an invalid projected Q/K dtype" % self.name
            )
        if projected.v.dtype != projected.output_dtype or projected.v.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise EfficientSageError(
                "%s received an invalid projected V dtype" % self.name
            )
        tensors = (
            projected.q_int8,
            projected.q_scale,
            projected.k_int8,
            projected.k_scale,
            projected.v,
        )
        if any(tensor.device != projected.q_int8.device for tensor in tensors):
            raise EfficientSageError(
                "%s received projected tensors on different devices" % self.name
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise EfficientSageError(
                "%s requires contiguous projected carriers" % self.name
            )
        if not self.allow_cpu_for_tests:
            if not projected.q_int8.is_cuda:
                raise EfficientSageError(
                    "%s projected carriers require CUDA" % self.name
                )
            capability = tuple(
                torch.cuda.get_device_capability(projected.q_int8.device)
            )
            if capability not in self.capabilities:
                raise EfficientSageError(
                    "%s does not support projected carriers on SM%d%d"
                    % (self.name, capability[0], capability[1])
                )

        stats.observe_sequence(projected.sequence)
        return PreparedArchitecture(
            q_int8=projected.q_int8,
            q_scale=projected.q_scale,
            k_int8=projected.k_int8,
            k_scale=projected.k_scale,
            v_source=projected.v,
            output_dtype=projected.output_dtype,
            layer_index=int(layer_index),
            sequence=int(projected.sequence),
            heads=int(projected.heads),
            head_dim=int(projected.head_dim),
            softmax_scale=int(projected.head_dim) ** -0.5,
        )

    def log_once(self, version, detail):
        if self._logged:
            return
        logging.debug(
            "%s %s active: SageAttention %s, %s",
            LOG_PREFIX,
            self.name,
            version,
            detail,
        )
        self._logged = True

    def kernel_error(self, prepared, kernel_name, exc):
        stats.increment("kernel_errors")
        device = prepared.q_int8.device
        gpu = (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        )
        raise EfficientSageError(
            "%s kernel failed: layer=%d sequence=%d "
            "heads=%d head_dim=%d dtype=%s device=%s "
            "kernel=%s"
            % (
                self.name,
                prepared.layer_index,
                prepared.sequence,
                prepared.heads,
                prepared.head_dim,
                prepared.output_dtype,
                gpu,
                kernel_name,
            )
        ) from exc
