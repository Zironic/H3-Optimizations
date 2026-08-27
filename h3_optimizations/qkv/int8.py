"""Execution-scoped ConvRot INT8 bindings for floating H3 linears."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor

from .. import diagnostics
from .formats import describe_linear, describe_weight


LAYOUT = "TensorWiseINT8Layout"
GROUP_SIZE = 256


class ConvRotINT8BindingError(RuntimeError):
    pass


class HeldConvRotINT8Linear:
    """Hold one native, effective-float, or runtime-quantized H3 linear.

    The stored checkpoint format chooses this binding, but Comfy's runtime
    weight patches are authoritative.  In native/preserve mode, if
    ``cast_bias_weight`` materializes a BF16/FP16 effective weight (for example
    because a LoRA/adapter weight_function is active), execute that effective
    value directly instead of pretending the stored ConvRot tensor is still
    the live weight.  Explicit force-quant callers set ``allow_float_conversion``
    and therefore requantize the effective value back to ConvRot-256.
    """

    def __init__(self, module, sample, *, allow_float_conversion=False):
        self.module = module
        self.sample = sample
        self.allow_float_conversion = bool(allow_float_conversion)
        self.weight = None
        self.bias = None
        self.acquired_weight = None
        self.acquired_bias = None
        self.handle = None
        self.converted_from_float = False
        self.effective_float = False

    def _release_acquired(self):
        if self.handle is not None:
            comfy.ops.uncast_bias_weight(
                self.module,
                self.acquired_weight,
                self.acquired_bias,
                self.handle,
            )
            self.handle = None
        self.acquired_weight = None
        self.acquired_bias = None

    def __enter__(self):
        if getattr(self.module, "_full_precision_mm", False):
            raise ConvRotINT8BindingError(
                "module explicitly requests full-precision matmul"
            )
        if self.sample.ndim < 2 or self.sample.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ConvRotINT8BindingError(
                "ConvRot INT8 execution requires BF16/FP16 activations"
            )

        source = describe_linear(self.module)
        weight, bias, handle = comfy.ops.cast_bias_weight(
            self.module,
            self.sample,
            offloadable=True,
            compute_dtype=self.sample.dtype,
            want_requant=True,
        )
        self.acquired_weight = weight
        self.acquired_bias = bias
        self.handle = handle
        try:
            if isinstance(weight, QuantizedTensor):
                if bias is not None:
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 conversion requires bias-free H3 linears"
                    )
                actual = describe_weight(weight, bias=bias)
                if not actual.convrot_int8_256:
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 provider received quantized layout %r"
                        % getattr(weight, "_layout_cls", None)
                    )
                self.weight = weight
                self.bias = None
            else:
                if getattr(weight, "dtype", None) not in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 conversion requires BF16/FP16 weights, got %s"
                        % getattr(weight, "dtype", None)
                    )
                if self.allow_float_conversion:
                    if bias is not None:
                        raise ConvRotINT8BindingError(
                            "ConvRot INT8 conversion requires bias-free H3 linears"
                        )
                    self.weight = QuantizedTensor.from_float(
                        weight,
                        LAYOUT,
                        scale="recalculate",
                        is_weight=True,
                        per_channel=True,
                        convrot=True,
                        convrot_groupsize=GROUP_SIZE,
                    )
                    self.bias = None
                    self.converted_from_float = True
                    self._release_acquired()
                elif source.convrot_int8_256:
                    # Runtime patches can force Comfy to materialize the
                    # effective value in compute dtype even though the stored
                    # checkpoint tensor is ConvRot INT8. Preserve that exact
                    # effective value in native mode.
                    self.weight = weight
                    self.bias = bias
                    self.effective_float = True
                else:
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 provider received a floating weight without conversion enabled"
                    )
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        self._release_acquired()
        self.weight = None
        self.bias = None
        self.sample = None
        self.effective_float = False

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def _linear(self, x, weight, bias=None):
        override = getattr(
            self.module,
            "_h3_benchmark_convrot_linear",
            None,
        )
        if override is not None and isinstance(weight, QuantizedTensor):
            if not callable(override):
                raise ConvRotINT8BindingError(
                    "benchmark ConvRot linear override is not callable"
                )
            return override(x, weight, bias)
        return F.linear(x, weight, bias)

    def linear(self, x):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        comfy.ops.run_every_op()
        return self._linear(x, self.weight, self.bias)

    def linear_range(self, x, start, end):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        start = int(start)
        end = int(end)
        if not 0 <= start < end <= int(self.weight.shape[0]):
            raise ConvRotINT8BindingError("ConvRot INT8 output slice is invalid")
        if not isinstance(self.weight, QuantizedTensor):
            bias = None if self.bias is None else self.bias[start:end]
            comfy.ops.run_every_op()
            return self._linear(x, self.weight[start:end], bias)
        if self.bias is not None:
            raise ConvRotINT8BindingError(
                "ConvRot INT8 output slicing requires a bias-free linear"
            )
        params = self.weight._params
        scale = params.scale
        if scale.numel() != 1:
            scale = scale[start:end]
        sliced = QuantizedTensor(
            self.weight._qdata[start:end],
            self.weight._layout_cls,
            replace(
                params,
                scale=scale,
                orig_shape=(end - start, int(self.weight.shape[1])),
            ),
        )
        comfy.ops.run_every_op()
        return self._linear(x, sliced, None)


class HeldConvRotINT8QKV:
    """Hold a ConvRot INT8 QKV weight across all projection chunks."""

    def __init__(self, attention, sample, *, allow_float_conversion=False):
        self.attention = attention
        self.binding = HeldConvRotINT8Linear(
            attention.qkv_proj,
            sample,
            allow_float_conversion=allow_float_conversion,
        )

    def __enter__(self):
        self.binding.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.binding.__exit__(exc_type, exc, tb)

    def _finish(self, rows, rope):
        from ..attention_forward import finish_qkv_projection, to_hnd

        with diagnostics.stage("qkv_linear"):
            projected = self.binding.linear(rows)
        with diagnostics.stage("qk_norm_rope"):
            return to_hnd(
                *finish_qkv_projection(self.attention, projected, rope)
            )

    def _finish_single_qk(self, projected, rope, norm):
        seq = int(projected.shape[0])
        projected = projected.view(
            1,
            seq,
            self.attention.heads,
            self.attention.head_dim,
        )
        if rope is None:
            return norm(projected[0])
        scale = comfy.model_management.cast_to(
            norm.weight,
            device=projected.device,
        )
        projected = F.rms_norm(
            projected,
            (self.attention.head_dim,),
            weight=scale,
            eps=norm.eps,
        )
        rot_dim = int(rope.shape[-3]) * 2
        comfy.quant_ops.ck.apply_rope_split_half1_(
            projected[..., :rot_dim],
            rope,
        )
        return projected[0]

    def project_hnd(self, x, rope_freqs, start, end):
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        return self._finish(x[start:end], rope)

    def project_q_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        with diagnostics.stage("qkv_linear"):
            q = self.binding.linear_range(x[start:end], 0, inner)
        with diagnostics.stage("qk_norm_rope"):
            q = self._finish_single_qk(q, rope, self.attention.q_norm)
        return q.transpose(0, 1).unsqueeze(0)

    def project_kv_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        with diagnostics.stage("qkv_linear"):
            projected = self.binding.linear_range(
                x[start:end],
                inner,
                inner * 3,
            )
        k, v = projected.split(inner, dim=-1)
        with diagnostics.stage("qk_norm_rope"):
            k = self._finish_single_qk(k, rope, self.attention.k_norm)
        v = v.view(
            end - start,
            self.attention.heads,
            self.attention.head_dim,
        )
        return (
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )

    def project_rows(self, x, rope_freqs, rows):
        sample_x = x.index_select(0, rows)
        sample_rope = (
            None if rope_freqs is None else rope_freqs.index_select(1, rows)
        )
        return self._finish(sample_x, sample_rope)


class HeldConvRotINT8MLP:
    """Hold runtime ConvRot INT8 fc1/fc2 weights across bounded token slabs."""

    def __init__(self, mlp, sample, *, allow_float_conversion=False):
        self.mlp = mlp
        self.sample = sample
        self.allow_float_conversion = bool(allow_float_conversion)
        self.fc1_binding = None
        self.fc2_binding = None

    def __enter__(self):
        try:
            self.fc1_binding = HeldConvRotINT8Linear(
                self.mlp.fc1,
                self.sample,
                allow_float_conversion=self.allow_float_conversion,
            )
            self.fc1_binding.__enter__()
            self.fc2_binding = HeldConvRotINT8Linear(
                self.mlp.fc2,
                self.sample,
                allow_float_conversion=self.allow_float_conversion,
            )
            self.fc2_binding.__enter__()
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        if self.fc2_binding is not None:
            self.fc2_binding.__exit__(None, None, None)
            self.fc2_binding = None
        if self.fc1_binding is not None:
            self.fc1_binding.__exit__(None, None, None)
            self.fc1_binding = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def fc1_fc2(self, x, swiglu):
        expanded = self.fc1_binding.linear(x)
        activated = swiglu(expanded)
        out = self.fc2_binding.linear(activated)
        return out, "held_convrot_int8"


class LazyConvRotINT8Linear:
    """Delay runtime quantization until the first output-projection slab."""

    def __init__(self, module):
        self.module = module
        self.binding = None

    def linear(self, x):
        if self.binding is None:
            sample = x.reshape(-1, x.shape[-1])[:1]
            self.binding = HeldConvRotINT8Linear(
                self.module,
                sample,
                allow_float_conversion=True,
            )
            self.binding.__enter__()
        return self.binding.linear(x)

    def release(self):
        if self.binding is not None:
            self.binding.__exit__(None, None, None)
            self.binding = None
