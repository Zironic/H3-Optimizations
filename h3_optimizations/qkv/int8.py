"""Execution-scoped ConvRot INT8 bindings for floating H3 linears."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

from .. import diagnostics
from .formats import describe_linear, describe_weight


LAYOUT = "TensorWiseINT8Layout"
GROUP_SIZE = 256


class ConvRotINT8BindingError(RuntimeError):
    pass


class HeldConvRotINT8Linear:
    """Hold one native or runtime-quantized ConvRot-256 linear weight."""

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
        # Kitchen quantizes the activation to INT8 inside the GEMM, so FP32 is
        # as valid an input as BF16. SM75 runs H3 with FP32 manual cast because
        # ComfyUI offers H3 no FP16 and these cards no BF16 (issue #89).
        if self.sample.ndim < 2 or self.sample.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ConvRotINT8BindingError(
                "ConvRot INT8 execution requires BF16/FP16/FP32 activations, got %s"
                % self.sample.dtype
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
            if bias is not None:
                raise ConvRotINT8BindingError(
                    "ConvRot INT8 conversion requires bias-free H3 linears"
                )
            if isinstance(weight, QuantizedTensor):
                actual = describe_weight(weight, bias=bias)
                if not actual.convrot_int8_256:
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 provider received quantized layout %r"
                        % getattr(weight, "_layout_cls", None)
                    )
                self.weight = weight
            else:
                if not self.allow_float_conversion or not source.plain_float:
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 provider received a floating weight without conversion enabled"
                    )
                if getattr(weight, "dtype", None) not in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    raise ConvRotINT8BindingError(
                        "ConvRot INT8 conversion requires BF16/FP16 weights, got %s"
                        % getattr(weight, "dtype", None)
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
                self.converted_from_float = True
                self._release_acquired()
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        self._release_acquired()
        self.weight = None
        self.bias = None
        self.sample = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def _linear(self, x, weight):
        override = getattr(
            self.module,
            "_h3_benchmark_convrot_linear",
            None,
        )
        if override is not None:
            if not callable(override):
                raise ConvRotINT8BindingError(
                    "benchmark ConvRot linear override is not callable"
                )
            return override(x, weight, self.bias)
        return F.linear(x, weight, self.bias)

    def linear(self, x):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        comfy.ops.run_every_op()
        return self._linear(x, self.weight)

    @staticmethod
    def prequantized_is_available():
        try:
            from comfy_kitchen.backends import cuda
        except ImportError:
            return False
        return callable(getattr(cuda, '_int4_linear_via_int8_values', None))

    @staticmethod
    def _prequantized_values(
        qdata,
        scale,
        weight_qdata,
        weight_scale,
        out_dtype,
    ):
        from comfy_kitchen.backends import cuda

        function = getattr(cuda, '_int4_linear_via_int8_values', None)
        if function is None:
            raise ConvRotINT8BindingError(
                'Comfy Kitchen has no prequantized INT8 GEMM entry point'
            )
        return function(
            qdata,
            weight_qdata,
            scale,
            weight_scale,
            None,
            out_dtype,
        )

    @classmethod
    def _prequantized_linear(cls, qdata, scale, weight, out_dtype):
        weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        return cls._prequantized_values(
            qdata,
            scale,
            weight_qdata,
            weight_scale,
            out_dtype,
        )

    def linear_prequantized(self, qdata, scale, out_dtype):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        comfy.ops.run_every_op()
        return self._prequantized_linear(qdata, scale, self.weight, out_dtype)

    def residual_(self, x, residual, gate):
        """``residual += gate * linear(x)`` in place, or False."""
        if (
            self.weight is None
            or self.bias is not None
            or getattr(self.module, "_h3_benchmark_convrot_linear", None) is not None
        ):
            return False
        from ..memory import fused_residual

        comfy.ops.run_every_op()
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(self.weight)
        return fused_residual.linear_residual_(
            x,
            qdata,
            scale,
            None,
            residual,
            gate,
            convrot=True,
            convrot_groupsize=GROUP_SIZE,
        )

    def linear_range(self, x, start, end):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        if self.bias is not None:
            raise ConvRotINT8BindingError(
                "ConvRot INT8 output slicing requires a bias-free linear"
            )
        start = int(start)
        end = int(end)
        if not 0 <= start < end <= int(self.weight.shape[0]):
            raise ConvRotINT8BindingError("ConvRot INT8 output slice is invalid")
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
        return self._linear(x, sliced)

    def linear_range_prequantized(self, qdata, scale, start, end, out_dtype):
        if self.weight is None:
            raise RuntimeError("ConvRot INT8 binding is not active")
        if self.bias is not None:
            raise ConvRotINT8BindingError(
                "ConvRot INT8 output slicing requires a bias-free linear"
            )
        start = int(start)
        end = int(end)
        if not 0 <= start < end <= int(self.weight.shape[0]):
            raise ConvRotINT8BindingError("ConvRot INT8 output slice is invalid")
        params = self.weight._params
        weight_scale = params.scale
        if weight_scale.numel() != 1:
            weight_scale = weight_scale[start:end]
        sliced = QuantizedTensor(
            self.weight._qdata[start:end],
            self.weight._layout_cls,
            replace(
                params,
                scale=weight_scale,
                orig_shape=(end - start, int(self.weight.shape[1])),
            ),
        )
        comfy.ops.run_every_op()
        return self._prequantized_linear(qdata, scale, sliced, out_dtype)


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

    @staticmethod
    def _carrier(rows, start, end):
        producer = getattr(rows, 'quantize_convrot256', None)
        return None if producer is None else producer(start, end)

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
        carrier = self._carrier(x, start, end)
        if carrier is not None:
            from ..attention_forward import finish_qkv_projection, to_hnd

            with diagnostics.stage("qkv_linear"):
                projected = self.binding.linear_prequantized(
                    *carrier, out_dtype=x.dtype
                )
            with diagnostics.stage("qk_norm_rope"):
                return to_hnd(
                    *finish_qkv_projection(self.attention, projected, rope)
                )
        return self._finish(x[start:end], rope)

    def project_q_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        with diagnostics.stage("qkv_linear"):
            carrier = self._carrier(x, start, end)
            q = (
                self.binding.linear_range(x[start:end], 0, inner)
                if carrier is None
                else self.binding.linear_range_prequantized(
                    *carrier, 0, inner, x.dtype
                )
            )
        with diagnostics.stage("qk_norm_rope"):
            q = self._finish_single_qk(q, rope, self.attention.q_norm)
        return q.transpose(0, 1).unsqueeze(0)

    def project_kv_hnd(self, x, rope_freqs, start, end):
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        rope = None if rope_freqs is None else rope_freqs[:, start:end]
        with diagnostics.stage("qkv_linear"):
            carrier = self._carrier(x, start, end)
            projected = (
                self.binding.linear_range(x[start:end], inner, inner * 3)
                if carrier is None
                else self.binding.linear_range_prequantized(
                    *carrier, inner, inner * 3, x.dtype
                )
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

    def project_v_hnd(self, x, rope_freqs, start, end):
        del rope_freqs
        inner = int(self.attention.heads) * int(self.attention.head_dim)
        with diagnostics.stage("qkv_linear"):
            carrier = self._carrier(x, start, end)
            v = (
                self.binding.linear_range(x[start:end], inner * 2, inner * 3)
                if carrier is None
                else self.binding.linear_range_prequantized(
                    *carrier, inner * 2, inner * 3, x.dtype
                )
            )
        v = v.view(end - start, self.attention.heads, self.attention.head_dim)
        return v.transpose(0, 1).unsqueeze(0)

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

    def fc1(self, x):
        return self.fc1_binding.linear(x)

    def _fc2_plain(self):
        if getattr(self.fc2_binding.module, "_h3_benchmark_convrot_linear", None) is not None:
            return None
        weight = self.fc2_binding.weight
        if weight is None or self.fc2_binding.bias is not None:
            return None
        return TensorWiseINT8Layout.get_plain_tensors(weight)

    def fc2_swiglu(self, expanded, native=True):
        """fc2 with SwiGLU folded into Kitchen's ConvRot activation quantizer."""
        del native
        plain = self._fc2_plain()
        if plain is None:
            from ..memory.linear import swiglu_eager  # memory imports this module

            return self.fc2_binding.linear(swiglu_eager(expanded)), "held_convrot_int8"
        comfy.ops.run_every_op()
        qdata, scale = plain
        out = comfy.quant_ops.ck.int8_linear(
            expanded,
            qdata,
            scale,
            None,
            expanded.dtype,
            convrot=True,
            convrot_groupsize=GROUP_SIZE,
            input_act="swiglu",
        )
        return out, "held_convrot_int8_fused_swiglu"

    def fc2_swiglu_residual_(self, expanded, residual, gate):
        """``residual += gate * fc2(swiglu(expanded))`` in place, or False."""
        plain = self._fc2_plain()
        if plain is None:
            return False
        from ..memory import fused_residual  # memory imports this module

        qdata, scale = plain
        return fused_residual.gated_swiglu_residual_(
            expanded,
            qdata,
            scale,
            None,
            residual,
            gate,
            convrot=True,
            convrot_groupsize=GROUP_SIZE,
        )


class LazyConvRotINT8Linear:
    """Delay runtime quantization until the first output-projection slab."""

    def __init__(self, module, *, allow_float_conversion=True):
        self.module = module
        self.allow_float_conversion = bool(allow_float_conversion)
        self.binding = None

    def _binding(self, x):
        if self.binding is None:
            sample = x.reshape(-1, x.shape[-1])[:1]
            self.binding = HeldConvRotINT8Linear(
                self.module,
                sample,
                allow_float_conversion=self.allow_float_conversion,
            )
            self.binding.__enter__()
        return self.binding

    def linear(self, x):
        self._binding(x)
        return self.binding.linear(x)

    def residual_(self, x, residual, gate):
        self._binding(x)
        return self.binding.residual_(x, residual, gate)

    def release(self):
        if self.binding is not None:
            self.binding.__exit__(None, None, None)
            self.binding = None
