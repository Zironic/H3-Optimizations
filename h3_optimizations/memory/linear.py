"""Bounded MLP execution and safe held-weight sessions."""

from dataclasses import dataclass
import itertools
import weakref

import torch
import torch.nn.functional as F

import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

from . import fused_residual


_CONVROT_MLPS = weakref.WeakValueDictionary()
_CONVROT_MLP_IDS = itertools.count(1)


class UnsafeHeldWeights(RuntimeError):
    """Two acquired weights would alias the same reusable async cast buffer."""


def swiglu_eager(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def swiglu_inplace_(x):
    """SwiGLU written into the gate half of ``x``; returns that half as a view.

    Bit-identical to :func:`swiglu_eager` without allocating a second
    activation-width tensor. ``x`` is consumed: only the returned view is valid.
    """
    width = int(x.shape[-1]) // 2
    gate = x[..., :width]
    torch.nn.functional.silu(gate, inplace=True).mul_(x[..., width:])
    return gate


def _stream_from_handle(handle):
    if isinstance(handle, tuple) and handle:
        return handle[0]
    return None


@dataclass
class AcquiredLinear:
    module: object
    weight: object
    bias: object
    handle: object
    released: bool = False

    def linear(self, x):
        return F.linear(x, self.weight, self.bias)

    def release(self):
        if self.released:
            return
        comfy.ops.uncast_bias_weight(self.module, self.weight, self.bias, self.handle)
        self.released = True


def acquire_linear(module, sample, want_requant=True):
    """Acquire one Comfy linear weight once, preserving quantized dispatch."""
    weight, bias, handle = comfy.ops.cast_bias_weight(
        module,
        sample,
        offloadable=True,
        compute_dtype=sample.dtype,
        want_requant=bool(want_requant),
    )
    return AcquiredLinear(module, weight, bias, handle)


class HeldMLP:
    """Hold fc1/fc2 across every token slab when Comfy's buffers permit it."""

    def __init__(self, mlp, sample, *, force_bf16=False):
        self.mlp = mlp
        self.sample = sample
        self.force_bf16 = bool(force_bf16)
        self.fc1_weight = None
        self.fc2_weight = None

    def _validate_bf16(self, acquired, name):
        if not self.force_bf16:
            return
        if isinstance(acquired.weight, QuantizedTensor):
            raise UnsafeHeldWeights('%s remained quantized in BF16 mode' % name)
        if getattr(acquired.weight, 'dtype', None) != torch.bfloat16:
            raise UnsafeHeldWeights('%s was not materialized as BF16' % name)
        if acquired.bias is not None and getattr(acquired.bias, 'dtype', None) != torch.bfloat16:
            raise UnsafeHeldWeights('%s bias was not materialized as BF16' % name)

    def __enter__(self):
        try:
            self.fc1_weight = acquire_linear(
                self.mlp.fc1,
                self.sample,
                want_requant=not self.force_bf16,
            )
            self._validate_bf16(self.fc1_weight, 'fc1')
            self.fc2_weight = acquire_linear(
                self.mlp.fc2,
                self.sample,
                want_requant=not self.force_bf16,
            )
            self._validate_bf16(self.fc2_weight, 'fc2')
            stream1 = _stream_from_handle(self.fc1_weight.handle)
            stream2 = _stream_from_handle(self.fc2_weight.handle)
            if stream1 is not None and stream1 is stream2:
                raise UnsafeHeldWeights(
                    "fc1 and fc2 were acquired from the same async cast stream; "
                    "the second reusable cast buffer can overwrite the first"
                )
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        if self.fc2_weight is not None:
            self.fc2_weight.release()
            self.fc2_weight = None
        if self.fc1_weight is not None:
            self.fc1_weight.release()
            self.fc1_weight = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def fc1(self, x):
        # Ordinary Comfy module forwards call this before every operation. The
        # held path bypasses the module wrapper, so preserve cancellation.
        comfy.ops.run_every_op()
        return self.fc1_weight.linear(x)

    def supports_prequantized_fc1(self):
        if self.fc1_weight.bias is not None:
            return False
        try:
            _convrot_parts(self.fc1_weight.weight, "fc1")
        except (TypeError, ValueError):
            return False
        return True

    def fc1_prequantized(self, qdata, scale, out_dtype):
        weight = self.fc1_weight.weight
        if self.fc1_weight.bias is not None:
            return None
        try:
            weight_qdata, weight_scale = _convrot_parts(weight, "fc1")
        except (TypeError, ValueError):
            return None
        from ..qkv.int8 import HeldConvRotINT8Linear

        comfy.ops.run_every_op()
        return HeldConvRotINT8Linear._prequantized_values(
            qdata,
            scale,
            weight_qdata,
            weight_scale,
            out_dtype,
        )

    def fc2_swiglu(self, expanded, native):
        weight = self.fc2_weight.weight
        bias = self.fc2_weight.bias
        if (
            native
            and isinstance(weight, QuantizedTensor)
            and getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(weight._params, "transposed", False)
        ):
            qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            out = comfy.quant_ops.ck.int8_linear(
                expanded,
                qdata,
                scale,
                bias,
                expanded.dtype,
                convrot=getattr(weight._params, "convrot", False),
                convrot_groupsize=getattr(
                    weight._params, "convrot_groupsize", 256
                ),
                input_act="swiglu",
            )
            return out, "held_tensorwise_int8_native"

        activated = swiglu_inplace_(expanded)
        return F.linear(activated, weight, bias), "held_bf16_swiglu"

    def fc2_swiglu_residual_(self, expanded, residual, gate):
        """Fold ``residual += gate * fc2(swiglu(expanded))`` into the fc2 kernel.

        Returns False, with ``residual`` untouched, unless fc2 is a TensorWise
        INT8 weight and Comfy Kitchen's fused residual kernel is usable; the
        caller then uses :meth:`fc2_swiglu` and ``addcmul_`` as before.
        """
        weight = self.fc2_weight.weight
        if (
            not isinstance(weight, QuantizedTensor)
            or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout"
            or getattr(weight._params, "transposed", False)
        ):
            return False
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        return fused_residual.gated_swiglu_residual_(
            expanded,
            qdata,
            scale,
            self.fc2_weight.bias,
            residual,
            gate,
            convrot=getattr(weight._params, "convrot", False),
            convrot_groupsize=getattr(weight._params, "convrot_groupsize", 256),
        )


def _convrot_parts(weight, name):
    if not isinstance(weight, QuantizedTensor):
        raise TypeError("%s must be a TensorWiseINT8 quantized weight" % name)
    if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
        raise TypeError("%s must use TensorWiseINT8Layout" % name)
    params = getattr(weight, "_params", None)
    if getattr(params, "transposed", False):
        raise ValueError("%s ConvRot weight must be non-transposed" % name)
    if not getattr(params, "convrot", False):
        raise ValueError("%s ConvRot metadata must set convrot=True" % name)
    if int(getattr(params, "convrot_groupsize", 0)) != 256:
        raise ValueError("%s ConvRot group size must be 256" % name)
    qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    if qdata.ndim != 2 or qdata.dtype != torch.int8:
        raise ValueError("%s ConvRot qdata must be a rank-2 INT8 tensor" % name)
    if not torch.is_tensor(scale) or scale.numel() not in (1, qdata.shape[0]):
        raise ValueError("%s ConvRot scale must be scalar or per-output-channel" % name)
    return qdata, scale


def bind_convrot_mlp(mlp):
    module_id = getattr(mlp, "_h3_optimizations_convrot_mlp_id", None)
    if module_id is None:
        module_id = next(_CONVROT_MLP_IDS)
        mlp._h3_optimizations_convrot_mlp_id = module_id
    _CONVROT_MLPS[module_id] = mlp
    return module_id


def _convrot_mlp(module_id):
    mlp = _CONVROT_MLPS.get(int(module_id))
    if mlp is None:
        raise RuntimeError("H3 ConvRot MLP is no longer active")
    return mlp


def _convrot_fc1_tiles(qdata, scale):
    half_width = qdata.shape[0] // 2
    tile_width = half_width // 2
    if scale.numel() == 1:
        scale_tiles = (scale.contiguous().clone(), scale.contiguous().clone())
    else:
        scale_tiles = tuple(
            torch.cat(
                (scale[start:stop], scale[half_width + start : half_width + stop]),
                dim=0,
            ).contiguous()
            for start, stop in ((0, tile_width), (tile_width, half_width))
        )
    tiles = tuple(
        (
            torch.cat(
                (qdata[start:stop], qdata[half_width + start : half_width + stop]),
                dim=0,
            ).contiguous(),
            scale_tiles[index],
        )
        for index, (start, stop) in enumerate(
            ((0, tile_width), (tile_width, half_width))
        )
    )
    return tiles[0][0], tiles[0][1], tiles[1][0], tiles[1][1]


def _convrot_fc2_tiles(qdata, scale):
    tile_width = qdata.shape[1] // 2
    return (
        qdata[:, :tile_width].contiguous(),
        qdata[:, tile_width:].contiguous(),
        scale.contiguous().clone(),
    )


@torch.library.custom_op("h3_optimizations::convrot_fc1_tiles", mutates_args=())
def _convrot_fc1_tiles_op(
    qdata: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _convrot_fc1_tiles(qdata, scale)


@_convrot_fc1_tiles_op.register_fake
def _convrot_fc1_tiles_fake(qdata, scale):
    tile_width = qdata.shape[0] // 4
    weight_shape = (tile_width * 2, qdata.shape[1])
    scale_shape = (
        scale.shape
        if scale.numel() == 1
        else (tile_width * 2, *scale.shape[1:])
    )
    return (
        qdata.new_empty(weight_shape),
        scale.new_empty(scale_shape),
        qdata.new_empty(weight_shape),
        scale.new_empty(scale_shape),
    )


@torch.library.custom_op("h3_optimizations::convrot_fc2_tiles", mutates_args=())
def _convrot_fc2_tiles_op(
    qdata: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _convrot_fc2_tiles(qdata, scale)


@_convrot_fc2_tiles_op.register_fake
def _convrot_fc2_tiles_fake(qdata, scale):
    tile_shape = (qdata.shape[0], qdata.shape[1] // 2)
    return qdata.new_empty(tile_shape), qdata.new_empty(tile_shape), scale.new_empty(scale.shape)


@torch.library.custom_op(
    "h3_optimizations::convrot_fc1_module",
    mutates_args=(),
    device_types="cuda",
)
def _convrot_fc1_module_op(
    sample: torch.Tensor,
    module_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mlp = _convrot_mlp(module_id)
    acquired = acquire_linear(mlp.fc1, sample)
    try:
        if acquired.bias is not None:
            raise ValueError("fc1 ConvRot weight must not have a bias")
        qdata, scale = _convrot_parts(acquired.weight, "fc1")
        if qdata.shape[0] % 2:
            raise ValueError("fc1 ConvRot output width must be divisible by two")
        half_width = int(qdata.shape[0]) // 2
        if half_width % 256 or int(qdata.shape[1]) % 256:
            raise ValueError("fc1 ConvRot dimensions must be group-size aligned")
        outputs = _convrot_fc1_tiles(qdata, scale)
        return outputs
    finally:
        acquired.release()


@_convrot_fc1_module_op.register_fake
def _convrot_fc1_module_fake(sample, module_id):
    weight = _convrot_mlp(module_id).fc1.weight
    _qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    tile_shape = (weight.shape[0] // 2, weight.shape[1])
    scale_shape = (
        scale.shape
        if scale.numel() == 1
        else (weight.shape[0] // 2, *scale.shape[1:])
    )
    return (
        sample.new_empty(tile_shape, dtype=torch.int8),
        sample.new_empty(scale_shape, dtype=torch.float32),
        sample.new_empty(tile_shape, dtype=torch.int8),
        sample.new_empty(scale_shape, dtype=torch.float32),
    )


@torch.library.custom_op(
    "h3_optimizations::convrot_fc2_module",
    mutates_args=(),
    device_types="cuda",
)
def _convrot_fc2_module_op(
    sample: torch.Tensor,
    module_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mlp = _convrot_mlp(module_id)
    acquired = acquire_linear(mlp.fc2, sample)
    try:
        if acquired.bias is not None:
            raise ValueError("fc2 ConvRot weight must not have a bias")
        qdata, scale = _convrot_parts(acquired.weight, "fc2")
        if qdata.shape[1] % 2 or (qdata.shape[1] // 2) % 256:
            raise ValueError("H3 FFN width must split into two group-aligned tiles")
        outputs = _convrot_fc2_tiles(qdata, scale)
        return outputs
    finally:
        acquired.release()


@_convrot_fc2_module_op.register_fake
def _convrot_fc2_module_fake(sample, module_id):
    weight = _convrot_mlp(module_id).fc2.weight
    _qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    tile_shape = (weight.shape[0], weight.shape[1] // 2)
    return (
        sample.new_empty(tile_shape, dtype=torch.int8),
        sample.new_empty(tile_shape, dtype=torch.int8),
        sample.new_empty(scale.shape, dtype=torch.float32),
    )


@torch.library.custom_op(
    "h3_optimizations::convrot_linear",
    mutates_args=(),
    device_types="cuda",
)
def _convrot_linear_op(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    input_act: str | None,
) -> torch.Tensor:
    kwargs = {"convrot": True, "convrot_groupsize": 256}
    if input_act is not None:
        kwargs["input_act"] = input_act
    return comfy.quant_ops.ck.int8_linear(x, qdata, scale, None, x.dtype, **kwargs)


@_convrot_linear_op.register_fake
def _convrot_linear_fake(x, qdata, scale, input_act):
    return x.new_empty((*x.shape[:-1], qdata.shape[0]))


def _convrot_linear(x, qdata, scale, input_act=None):
    return _convrot_linear_op(x, qdata, scale, input_act)


class ConvRotTwoSliceMLP:
    """Prepack two half-width ConvRot feature tiles for one H3 MLP block."""

    def __init__(self, mlp, sample, convrot_linear=None):
        self.mlp = mlp
        self.sample = sample
        self.convrot_linear = convrot_linear or _convrot_linear
        self.tiles = None

    def __enter__(self):
        if self.sample.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("mlp_chunked_convrot_2slice requires FP16 or BF16 input")
        if torch.compiler.is_compiling():
            module_id = getattr(self.mlp, "_h3_optimizations_convrot_mlp_id", None)
            if module_id is None:
                raise RuntimeError("H3 ConvRot MLP was not registered")
            fc1_weight_0, fc1_scale_0, fc1_weight_1, fc1_scale_1 = (
                _convrot_fc1_module_op(self.sample, module_id)
            )
            fc2_weight_0, fc2_weight_1, fc2_scale = _convrot_fc2_module_op(
                self.sample, module_id
            )
            self.tiles = (
                {
                    "fc1_weight": fc1_weight_0,
                    "fc1_scale": fc1_scale_0,
                    "fc2_weight": fc2_weight_0,
                    "fc2_scale": fc2_scale,
                },
                {
                    "fc1_weight": fc1_weight_1,
                    "fc1_scale": fc1_scale_1,
                    "fc2_weight": fc2_weight_1,
                    "fc2_scale": fc2_scale,
                },
            )
            return self
        fc1 = None
        fc2 = None
        try:
            fc1 = acquire_linear(self.mlp.fc1, self.sample)
            if fc1.bias is not None:
                raise ValueError("fc1 ConvRot weight must not have a bias")
            fc1_qdata, fc1_scale = _convrot_parts(fc1.weight, "fc1")
            if fc1_qdata.shape[0] % 2:
                raise ValueError("fc1 ConvRot output width must be divisible by two")
            half_width = int(fc1_qdata.shape[0]) // 2
            hidden_width = int(fc1_qdata.shape[1])
            if half_width % 256 or hidden_width % 256:
                raise ValueError("fc1 ConvRot dimensions must be group-size aligned")
            fc1_weight_0, fc1_scale_0, fc1_weight_1, fc1_scale_1 = (
                _convrot_fc1_tiles_op(fc1_qdata, fc1_scale)
            )
            fc1_tiles = (
                (fc1_weight_0, fc1_scale_0),
                (fc1_weight_1, fc1_scale_1),
            )
            fc1.release()
            fc1 = None
            fc1_qdata = None
            fc1_scale = None

            fc2 = acquire_linear(self.mlp.fc2, self.sample)
            if fc2.bias is not None:
                raise ValueError("fc2 ConvRot weight must not have a bias")
            fc2_qdata, fc2_scale = _convrot_parts(fc2.weight, "fc2")
            if fc2_qdata.shape[0] != hidden_width:
                raise ValueError("fc1/fc2 hidden dimensions are incompatible")
            if fc2_qdata.shape[1] != half_width:
                raise ValueError("fc1/fc2 dimensions are not a SwiGLU pair")
            if half_width % 2 or (half_width // 2) % 256:
                raise ValueError("H3 FFN width must split into two group-aligned tiles")
            fc2_weight_0, fc2_weight_1, fc2_scale = _convrot_fc2_tiles_op(
                fc2_qdata, fc2_scale
            )
            fc2_tiles = (fc2_weight_0, fc2_weight_1)
            self.tiles = tuple(
                {
                    "fc1_weight": fc1_tile[0],
                    "fc1_scale": fc1_tile[1],
                    "fc2_weight": fc2_tile,
                    "fc2_scale": fc2_scale,
                }
                for fc1_tile, fc2_tile in zip(fc1_tiles, fc2_tiles)
            )
            fc2.release()
            fc2 = None
            return self
        except Exception:
            if fc2 is not None:
                fc2.release()
            if fc1 is not None:
                fc1.release()
            self.release()
            raise

    def release(self):
        self.tiles = None
        self.sample = None
        self.mlp = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def fc1_swiglu(self, x):
        """Return the full SwiGLU activation for diagnostic row subsets.

        The prepacked tiles split the FFN in half, and tile ``g`` owns SwiGLU
        features ``[g * width, (g + 1) * width)``, so concatenating the tiles in
        order reproduces the natural feature layout ``fc2`` expects.
        """
        if self.tiles is None:
            raise RuntimeError("ConvRot two-slice session is not active")
        comfy.ops.run_every_op()
        parts = []
        for tile in self.tiles:
            expanded = self.convrot_linear(
                x, tile["fc1_weight"], tile["fc1_scale"], input_act=None
            )
            parts.append(swiglu_eager(expanded))
            del expanded
        return torch.cat(parts, dim=-1)

    def fc2_activation(self, activated):
        """Apply fc2 to an already-activated SwiGLU vector."""
        if self.tiles is None:
            raise RuntimeError("ConvRot two-slice session is not active")
        comfy.ops.run_every_op()
        width = int(activated.shape[-1]) // len(self.tiles)
        output = None
        for index, tile in enumerate(self.tiles):
            partial = self.convrot_linear(
                activated[..., index * width:(index + 1) * width],
                tile["fc2_weight"],
                tile["fc2_scale"],
                input_act=None,
            )
            output = partial if output is None else output.add_(partial)
        return output

    def fc1_fc2(self, x, stage_factory=None):
        if self.tiles is None:
            raise RuntimeError("ConvRot two-slice session is not active")
        comfy.ops.run_every_op()
        output = None
        for tile in self.tiles:
            if stage_factory is None:
                expanded = self.convrot_linear(
                    x, tile["fc1_weight"], tile["fc1_scale"], input_act=None
                )
            else:
                with stage_factory("mlp_fc1"):
                    expanded = self.convrot_linear(
                        x, tile["fc1_weight"], tile["fc1_scale"], input_act=None
                    )
            if stage_factory is None:
                partial = self.convrot_linear(
                    expanded,
                    tile["fc2_weight"],
                    tile["fc2_scale"],
                    input_act="swiglu",
                )
            else:
                with stage_factory("mlp_swiglu_fc2"):
                    partial = self.convrot_linear(
                        expanded,
                        tile["fc2_weight"],
                        tile["fc2_scale"],
                        input_act="swiglu",
                    )
            if output is None:
                output = partial
            else:
                output.add_(partial)
            del expanded, partial
        return output, "held_convrot_2slice"

    def fc1_fc2_prequantized(self, qdata, scale, out_dtype):
        if self.tiles is None:
            raise RuntimeError("ConvRot two-slice session is not active")
        from ..qkv.int8 import HeldConvRotINT8Linear

        comfy.ops.run_every_op()
        output = None
        for tile in self.tiles:
            expanded = HeldConvRotINT8Linear._prequantized_values(
                qdata,
                scale,
                tile["fc1_weight"],
                tile["fc1_scale"],
                out_dtype,
            )
            partial = self.convrot_linear(
                expanded,
                tile["fc2_weight"],
                tile["fc2_scale"],
                input_act="swiglu",
            )
            output = partial if output is None else output.add_(partial)
            del expanded, partial
        return output, "held_convrot_2slice_fused_norm_quantize"


def module_fc1(mlp, h):
    """Run the fallback module fc1 call for one bounded slab."""
    return mlp.fc1(h)


def module_swiglu_fc2(mlp, expanded, native):
    """Run the fallback SwiGLU/fc2 path for one bounded slab."""
    if native:
        return (
            comfy.ops.linear_input_act(mlp.fc2, expanded, "swiglu"),
            "module_native",
        )
    return mlp.fc2(swiglu_inplace_(expanded)), "module_bf16_swiglu"
