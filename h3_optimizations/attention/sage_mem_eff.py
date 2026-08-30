"""SM89-only memory-efficient SageAttention for MiniMax H3.

The backend reproduces the released H3 Sage path (HND, per-thread INT8 Q/K,
FP8 V, no K smoothing) while separating preparation from execution so the
caller can release the fused BF16 QKV projection before the attention kernel
starts.

SageAttention wheels are not uniform across platforms. The resolver follows
the installed public dispatcher and PyTorch's registered operators instead of
depending on a wheel-specific extension namespace.
"""

from dataclasses import dataclass
import importlib.metadata
import logging

import torch

from . import stats
from .sage_v_fp8 import TRITON_AVAILABLE, prepare_sage_v_fp8
from .triton_i64 import per_thread_int8_i64

V_OFFSET_LIMIT = (1 << 32) - 1

# Preferred order matches SageAttention's public SM89 dispatch. Every candidate
# has the same low-level ABI; only the accumulation strategy and therefore the
# FP8 V quantization range differ.
KERNEL_CANDIDATES = (
    (
        "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
        2.25,
        "fp32+fp16",
    ),
    (
        "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
        448.0,
        "fp32+fp32",
    ),
    (
        "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn",
        448.0,
        "fp32",
    ),
)


class EfficientSageError(RuntimeError):
    """The custom backend cannot run safely in the current environment."""


@dataclass(frozen=True)
class SageSM89API:
    version: str
    per_channel_fp8: object
    kernel: object
    kernel_name: str
    kernel_source: str = "injected"
    v_scale_max: float = 2.25
    accumulation: str = "fp32+fp16"


@dataclass
class PreparedSM89:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_fp8: torch.Tensor
    v_scale: torch.Tensor
    output_dtype: torch.dtype
    layer_index: int
    sequence: int
    heads: int
    head_dim: int
    softmax_scale: float
    kernel: object
    kernel_name: str


def _append_unique(modules, module):
    if module is None:
        return
    if not any(module is existing for existing in modules):
        modules.append(module)


def _sm89_export_surfaces(core):
    """Return every place a SageAttention wheel may expose SM89 kernels.

    The Python dispatcher is the stable authority: inspect the globals it
    actually references instead of depending on the wheel's extension alias.
    The core module scan covers compiled dispatchers that have no Python code.
    """
    modules = []

    public_fn = getattr(core, "sageattn_qk_int8_pv_fp8_cuda", None)
    fn_globals = getattr(public_fn, "__globals__", {}) if public_fn is not None else {}
    code = getattr(public_fn, "__code__", None)
    for name in getattr(code, "co_names", ()):
        _append_unique(modules, fn_globals.get(name))

    for value in vars(core).values():
        _append_unique(modules, value)
    return modules


def _surface_name(surface):
    return getattr(surface, "__name__", type(surface).__name__)


def _registered_sm89_kernels():
    found = {}
    wanted = {name for name, _, _ in KERNEL_CANDIDATES}
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
            kernel = getattr(getattr(torch.ops, namespace), basename)
        except Exception:
            continue
        if callable(kernel):
            found.setdefault(basename, (kernel, "torch.ops.%s" % namespace))
    return found


def _resolve_sm89_kernel(core):
    surfaces = _sm89_export_surfaces(core)
    registered = _registered_sm89_kernels()
    for kernel_name, v_scale_max, accumulation in KERNEL_CANDIDATES:
        for surface in surfaces:
            try:
                kernel = getattr(surface, kernel_name)
            except (AttributeError, RuntimeError):
                continue
            if callable(kernel):
                return (
                    kernel,
                    kernel_name,
                    _surface_name(surface),
                    v_scale_max,
                    accumulation,
                )
        if kernel_name in registered:
            kernel, source = registered[kernel_name]
            return kernel, kernel_name, source, v_scale_max, accumulation

    available = set()
    for surface in surfaces:
        try:
            names = dir(surface)
        except Exception:
            continue
        available.update(name for name in names if name.startswith("qk_int8_sv_f8"))
    available.update(registered)
    expected = ", ".join(name for name, _, _ in KERNEL_CANDIDATES)
    found = ", ".join(sorted(available)) or "none discoverable"
    raise EfficientSageError(
        "SageAttention has no supported SM89 kernel export. "
        "Expected one of: %s. Found: %s" % (expected, found)
    )


def _load_api():
    try:
        version = importlib.metadata.version("sageattention")
        import sageattention.core as core
    except Exception as exc:
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "sage_mem_eff requires SageAttention with the SM89 extension"
        ) from exc

    if not getattr(core, "SM89_ENABLED", False):
        stats.increment("compatibility_errors")
        raise EfficientSageError("SageAttention's SM89 extension is unavailable")

    per_channel_fp8 = getattr(core, "per_channel_fp8", None)
    if not callable(per_channel_fp8):
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "SageAttention has no compatible per_channel_fp8 helper"
        )

    try:
        kernel, kernel_name, source, v_scale_max, accumulation = _resolve_sm89_kernel(core)
    except EfficientSageError:
        stats.increment("compatibility_errors")
        raise

    return SageSM89API(
        version=version,
        per_channel_fp8=per_channel_fp8,
        kernel=kernel,
        kernel_name=kernel_name,
        kernel_source=source,
        v_scale_max=v_scale_max,
        accumulation=accumulation,
    )


def max_linear_offset(tensor):
    return sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))


def guard_v_stride(v):
    """Copy only when SageAttention's unsigned-32-bit V preprocessor could wrap."""
    if max_linear_offset(v) > V_OFFSET_LIMIT:
        stats.increment("v_guard_copies")
        return v.contiguous()
    return v


def first_unsafe_v_length(heads=56, head_dim=128, sequence_stride=None):
    """First HND sequence length whose maximum element offset exceeds uint32."""
    sequence_stride = sequence_stride or heads * head_dim * 3
    non_sequence_tail = (heads - 1) * head_dim + (head_dim - 1)
    last_safe_row = (V_OFFSET_LIMIT - non_sequence_tail) // sequence_stride
    return last_safe_row + 2  # row index + one for sequence length


class SM89SageMemoryEfficientBackend:
    name = "sage_mem_eff"
    projected_q_tile = 128
    projected_k_tile = 64

    def __init__(self, api=None, quantizer=None, allow_cpu_for_tests=False):
        self.api = api if api is not None else _load_api()
        self.quantizer = quantizer if quantizer is not None else per_thread_int8_i64
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        stats.increment("configured")
        self._logged = False

    def quantize_projected_qk(self, q, k):
        return self.quantizer(
            q,
            k,
            None,
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
            tensor_layout="HND",
        )

    def quantize_projected_q(self, q):
        q_int8, q_scale, _k_int8, _k_scale = self.quantize_projected_qk(
            q,
            q[..., :1, :].contiguous(),
        )
        return q_int8, q_scale

    def quantize_projected_k(self, k):
        _q_int8, _q_scale, k_int8, k_scale = self.quantize_projected_qk(
            k[..., :1, :].contiguous(),
            k,
        )
        return k_int8, k_scale

    def _validate(self, q, k, v):
        if q.shape != k.shape or q.shape != v.shape:
            raise EfficientSageError(
                "sage_mem_eff requires equal self-attention Q/K/V shapes; got %s %s %s"
                % (tuple(q.shape), tuple(k.shape), tuple(v.shape))
            )
        if q.ndim != 4:
            raise EfficientSageError("sage_mem_eff expects HND rank-4 tensors")
        batch, heads, sequence, head_dim = q.shape
        if batch != 1:
            raise EfficientSageError("released H3 expects attention batch 1; got %d" % batch)
        if head_dim != 128:
            raise EfficientSageError("sage_mem_eff supports head_dim 128; got %d" % head_dim)
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise EfficientSageError("sage_mem_eff requires fp16 or bf16; got %s" % q.dtype)
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise EfficientSageError("Q/K/V dtypes differ")
        if q.device != k.device or q.device != v.device:
            raise EfficientSageError("Q/K/V devices differ")
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            raise EfficientSageError("Q/K/V last dimension must be contiguous")
        if not self.allow_cpu_for_tests:
            if not q.is_cuda:
                raise EfficientSageError("sage_mem_eff requires CUDA")
            capability = torch.cuda.get_device_capability(q.device)
            if capability != (8, 9):
                raise EfficientSageError(
                    "sage_mem_eff is SM89-only; device capability is %d.%d" % capability
                )
        return batch, heads, sequence, head_dim

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        batch, heads, sequence, head_dim = self._validate(q, k, v)
        q_int8, q_scale, k_int8, k_scale = self.quantize_projected_qk(q, k)

        v_fp8, v_scale = prepare_sage_v_fp8(
            guard_v_stride(v),
            self.api.per_channel_fp8,
            scale_max=self.api.v_scale_max,
        )

        stats.observe_sequence(sequence)
        if not self._logged:
            logging.debug(
                "[H3 attention] sage_mem_eff active: SageAttention %s, HND, "
                "per-thread int64 Q/K, FP8 V, accumulation=%s, "
                "kernel=%s via %s",
                self.api.version,
                self.api.accumulation,
                self.api.kernel_name,
                self.api.kernel_source,
            )
            self._logged = True

        return PreparedSM89(
            q_int8=q_int8,
            q_scale=q_scale,
            k_int8=k_int8,
            k_scale=k_scale,
            v_fp8=v_fp8,
            v_scale=v_scale,
            output_dtype=q.dtype,
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            head_dim=int(head_dim),
            softmax_scale=head_dim**-0.5,
            kernel=self.api.kernel,
            kernel_name=self.api.kernel_name,
        )

    def execute(self, prepared):
        return self.execute_rectangular(
            prepared.q_int8,
            prepared.q_scale,
            prepared.k_int8,
            prepared.k_scale,
            prepared.v_fp8,
            prepared.v_scale,
            output_dtype=prepared.output_dtype,
            softmax_scale=prepared.softmax_scale,
            layer_index=prepared.layer_index,
            kernel=prepared.kernel,
            kernel_name=prepared.kernel_name,
        )

    def prepare_streamed_v(self, v):
        v_fp8, v_scale = prepare_sage_v_fp8(
            guard_v_stride(v),
            self.api.per_channel_fp8,
            scale_max=self.api.v_scale_max,
        )
        return v_fp8, v_scale

    def v_staging_parameters(self):
        return (float(self.api.v_scale_max), 64) if TRITON_AVAILABLE else None

    def execute_rectangular(
        self,
        q_int8,
        q_scale,
        k_int8,
        k_scale,
        v_carrier,
        v_scale,
        *,
        output_dtype,
        softmax_scale,
        layer_index,
        kernel=None,
        kernel_name=None,
    ):
        kernel = self.api.kernel if kernel is None else kernel
        kernel_name = self.api.kernel_name if kernel_name is None else kernel_name
        output = torch.empty(
            q_int8.shape,
            dtype=output_dtype,
            device=q_int8.device,
        )
        try:
            kernel(
                q_int8,
                k_int8,
                v_carrier,
                output,
                q_scale,
                k_scale,
                v_scale,
                1,  # HND
                0,  # non-causal
                3,  # per-thread Q/K quantization
                softmax_scale,
                0,  # no LSE return
            )
        except Exception as exc:
            stats.increment("kernel_errors")
            device = q_int8.device
            gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
            raise EfficientSageError(
                "sage_mem_eff kernel failed: layer=%d sequence=%d heads=%d head_dim=%d "
                "dtype=%s device=%s kernel=%s SageAttention=%s"
                % (
                    layer_index,
                    k_int8.shape[-2],
                    q_int8.shape[1],
                    q_int8.shape[-1],
                    output_dtype,
                    gpu,
                    kernel_name,
                    self.api.version,
                )
            ) from exc
        stats.increment("executed")
        return output
