"""H3 TensorWise-INT8 QKV projection with Sparse Sage-native Q/K output."""

from dataclasses import dataclass
import itertools
import typing
import weakref

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False

from .router import KV_TILE, Q_TILE

HEAD_DIM = 128
ROT_DIM = 96
_FUSED_QKV_MODULES = weakref.WeakValueDictionary()
_FUSED_QKV_IDS = itertools.count(1)


class FusedQKVError(RuntimeError):
    pass


@dataclass
class PreparedFusedQKV:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v: torch.Tensor
    q_summary: torch.Tensor
    k_summary: torch.Tensor
    output_dtype: torch.dtype
    sequence: int
    heads: int
    head_dim: int
    layer_index: int
    smooth_k: bool = False


def sparse_fused_qkv_contract_mismatch(spec):
    if spec is None:
        return 'Sparse Sage ABI was not resolved'
    expected = (
        ('q_tile', Q_TILE),
        ('kv_tile', KV_TILE),
        ('qk_format', 'block_int8'),
        ('q_scale_layout', 'per_q_tile_float32'),
        ('k_scale_layout', 'per_kv_tile_float32'),
        ('projected_v_format', 'floating_hnd'),
        ('summary_format', 'tile_mean'),
    )
    for name, value in expected:
        if getattr(spec, name, None) != value:
            return '%s=%r does not match %r' % (
                name,
                getattr(spec, name, None),
                value,
            )
    if getattr(spec, 'v_format', None) not in ('fp16', 'fp8'):
        return 'unsupported V carrier format %r' % getattr(
            spec,
            'v_format',
            None,
        )
    if getattr(spec, 'accumulator', None) not in ('f16', 'f32'):
        return 'unsupported accumulator %r' % getattr(
            spec,
            'accumulator',
            None,
        )
    if not callable(getattr(spec, 'kernel', None)):
        return 'Sparse Sage kernel callable is unavailable'
    if getattr(spec, 'v_format', None) == 'fp8':
        fused = getattr(spec, 'fused_v_ops', None)
        if not all(
            callable(getattr(fused, name, None))
            for name in (
                'transpose_pad_permute_cuda',
                'scale_fuse_quant_cuda',
            )
        ):
            return 'Sparse Sage FP8 V preparation callables are unavailable'
    return None


def validate_prepared_fused_qkv(prepared):
    sequence = int(prepared.sequence)
    heads = int(prepared.heads)
    head_dim = int(prepared.head_dim)
    q_shape = (1, heads, sequence, head_dim)
    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    k_blocks = (sequence + KV_TILE - 1) // KV_TILE

    if head_dim != HEAD_DIM:
        raise FusedQKVError("fused H3 QKV requires head_dim 128")
    if tuple(prepared.q_int8.shape) != q_shape or tuple(prepared.k_int8.shape) != q_shape:
        raise FusedQKVError("fused H3 Q/K carrier shape is invalid")
    if prepared.q_int8.dtype != torch.int8 or prepared.k_int8.dtype != torch.int8:
        raise FusedQKVError("fused H3 Q/K carriers must be INT8")
    if tuple(prepared.v.shape) != q_shape or prepared.v.dtype != prepared.output_dtype:
        raise FusedQKVError("fused H3 V carrier is invalid")
    if tuple(prepared.q_scale.shape) != (1, heads, q_blocks):
        raise FusedQKVError("fused H3 Q scale shape is invalid")
    if tuple(prepared.k_scale.shape) != (1, heads, k_blocks):
        raise FusedQKVError("fused H3 K scale shape is invalid")
    if prepared.q_scale.dtype != torch.float32 or prepared.k_scale.dtype != torch.float32:
        raise FusedQKVError("fused H3 Q/K scales must be float32")
    if tuple(prepared.q_summary.shape) != (1, heads, q_blocks, head_dim):
        raise FusedQKVError("fused H3 Q router summary shape is invalid")
    if tuple(prepared.k_summary.shape) != (1, heads, k_blocks, head_dim):
        raise FusedQKVError("fused H3 K router summary shape is invalid")
    tensors = (
        prepared.q_int8,
        prepared.q_scale,
        prepared.k_int8,
        prepared.k_scale,
        prepared.v,
        prepared.q_summary,
        prepared.k_summary,
    )
    if any(t.device != prepared.q_int8.device for t in tensors):
        raise FusedQKVError("fused H3 QKV carrier devices differ")
    if any(not t.is_contiguous() for t in tensors):
        raise FusedQKVError("fused H3 QKV carriers must be contiguous")
    return prepared


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_qk_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        q_norm_ptr,
        k_norm_ptr,
        rope_ptr,
        q_ptr,
        q_scale_ptr,
        k_ptr,
        k_scale_ptr,
        q_summary_ptr,
        k_summary_ptr,
        sequence: tl.constexpr,
        hidden: tl.constexpr,
        heads: tl.constexpr,
        weight_stride_output: tl.constexpr,
        weight_stride_inner: tl.constexpr,
        rope_stride_seq: tl.constexpr,
        rope_stride_dim: tl.constexpr,
        rope_stride_rot: tl.constexpr,
        rope_stride_pair: tl.constexpr,
        epsilon: tl.constexpr,
        has_rope: tl.constexpr,
        KIND: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        block_m = tl.program_id(0)
        head = tl.program_id(1)

        rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_N)
        output_col = KIND * heads * BLOCK_N + head * BLOCK_N + dims
        row_mask = rows < sequence

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for start in range(0, hidden, BLOCK_K):
            inner = start + tl.arange(0, BLOCK_K)
            inner_mask = inner < hidden
            x_offsets = rows[:, None].to(tl.int64) * hidden + inner[None, :]
            w_offsets = (
                output_col[:, None].to(tl.int64) * weight_stride_output
                + inner[None, :] * weight_stride_inner
            )
            x = tl.load(
                x_ptr + x_offsets,
                mask=row_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            weight = tl.load(
                weight_ptr + w_offsets,
                mask=inner_mask[None, :],
                other=0,
            )
            accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.int32)

        row_scale = tl.load(x_scale_ptr + rows, mask=row_mask, other=0.0)
        column_scale = tl.load(weight_scale_ptr + output_col)
        value = accumulator.to(tl.float32)
        value *= row_scale[:, None] * column_scale[None, :]
        value = value.to(tl.bfloat16).to(tl.float32)

        output_offsets = (
            head.to(tl.int64) * sequence * BLOCK_N
            + rows[:, None].to(tl.int64) * BLOCK_N
            + dims[None, :]
        )
        if KIND == 0:
            norm_weight = tl.load(q_norm_ptr + dims).to(tl.float32)
        else:
            norm_weight = tl.load(k_norm_ptr + dims).to(tl.float32)
        inverse_rms = tl.rsqrt(
            tl.sum(value * value, axis=1) / BLOCK_N + epsilon
        )
        value = (value * inverse_rms[:, None] * norm_weight[None, :])
        value = value.to(tl.bfloat16).to(tl.float32)

        if has_rope:
            groups = tl.arange(0, 8)
            pairs = tl.arange(0, 16)
            grouped = tl.reshape(value, (BLOCK_M, 8, 16))
            group = dims // 16
            for rope_group in tl.static_range(0, 3):
                first = tl.sum(
                    grouped * (groups[None, :, None] == rope_group), axis=1
                )
                second = tl.sum(
                    grouped * (groups[None, :, None] == rope_group + 3), axis=1
                )
                rope_base = (
                    rope_ptr
                    + rows[:, None].to(tl.int64) * rope_stride_seq
                    + (rope_group * 16 + pairs[None, :]) * rope_stride_dim
                )
                rope_mask = row_mask[:, None]
                f00 = tl.load(rope_base, mask=rope_mask, other=0.0).to(tl.float32)
                f01 = tl.load(
                    rope_base + rope_stride_pair, mask=rope_mask, other=0.0
                ).to(tl.float32)
                f10 = tl.load(
                    rope_base + rope_stride_rot, mask=rope_mask, other=0.0
                ).to(tl.float32)
                f11 = tl.load(
                    rope_base + rope_stride_rot + rope_stride_pair,
                    mask=rope_mask,
                    other=0.0,
                ).to(tl.float32)
                rotated_first = f00 * first + f01 * second
                rotated_second = f10 * first + f11 * second
                first_full = tl.reshape(
                    tl.broadcast_to(rotated_first[:, None, :], (BLOCK_M, 8, 16)),
                    (BLOCK_M, BLOCK_N),
                )
                second_full = tl.reshape(
                    tl.broadcast_to(rotated_second[:, None, :], (BLOCK_M, 8, 16)),
                    (BLOCK_M, BLOCK_N),
                )
                value = tl.where(
                    group[None, :] == rope_group,
                    first_full,
                    tl.where(
                        group[None, :] == rope_group + 3,
                        second_full,
                        value,
                    ),
                )
            value = value.to(tl.bfloat16).to(tl.float32)

        absolute = tl.abs(value)
        if KIND == 0:
            block_scale = (
                tl.max(
                    tl.max(tl.where(row_mask[:, None], absolute, 0.0), axis=1),
                    axis=0,
                )
                / 127.0
                + 1e-7
            )
            quantized = value / block_scale
            quantized += 0.5 * tl.where(quantized >= 0, 1.0, -1.0)
            tl.store(q_ptr + output_offsets, quantized.to(tl.int8), mask=row_mask[:, None])
            tl.store(q_scale_ptr + head * tl.cdiv(sequence, BLOCK_M) + block_m, block_scale)
            count = tl.maximum(tl.minimum(sequence - block_m * BLOCK_M, BLOCK_M), 1)
            summary = tl.sum(
                tl.where(row_mask[:, None], value, 0.0), axis=0
            ) / count
            summary_offset = (
                (head * tl.cdiv(sequence, BLOCK_M) + block_m) * BLOCK_N + dims
            )
            tl.store(q_summary_ptr + summary_offset, summary)
        else:
            local_rows = tl.arange(0, BLOCK_M)
            first_mask = row_mask & (local_rows < BLOCK_M // 2)
            second_mask = row_mask & (local_rows >= BLOCK_M // 2)
            first_scale = (
                tl.max(
                    tl.max(tl.where(first_mask[:, None], absolute, 0.0), axis=1),
                    axis=0,
                )
                / 127.0
                + 1e-7
            )
            second_scale = (
                tl.max(
                    tl.max(tl.where(second_mask[:, None], absolute, 0.0), axis=1),
                    axis=0,
                )
                / 127.0
                + 1e-7
            )
            row_block_scale = tl.where(
                local_rows < BLOCK_M // 2, first_scale, second_scale
            )
            quantized = value / row_block_scale[:, None]
            quantized += 0.5 * tl.where(quantized >= 0, 1.0, -1.0)
            tl.store(k_ptr + output_offsets, quantized.to(tl.int8), mask=row_mask[:, None])

            k_blocks = tl.cdiv(sequence, BLOCK_M // 2)
            first_block = block_m * 2
            second_block = first_block + 1
            tl.store(k_scale_ptr + head * k_blocks + first_block, first_scale)
            if second_block < k_blocks:
                tl.store(k_scale_ptr + head * k_blocks + second_block, second_scale)

            first_count = tl.maximum(
                tl.minimum(sequence - block_m * BLOCK_M, BLOCK_M // 2), 1
            )
            second_count = tl.maximum(
                tl.minimum(
                    sequence - block_m * BLOCK_M - BLOCK_M // 2,
                    BLOCK_M // 2,
                ),
                1,
            )
            first_summary = tl.sum(
                tl.where(first_mask[:, None], value, 0.0), axis=0
            ) / first_count
            second_summary = tl.sum(
                tl.where(second_mask[:, None], value, 0.0), axis=0
            ) / second_count
            first_offset = (head * k_blocks + first_block) * BLOCK_N + dims
            tl.store(k_summary_ptr + first_offset, first_summary)
            if second_block < k_blocks:
                second_offset = (head * k_blocks + second_block) * BLOCK_N + dims
                tl.store(k_summary_ptr + second_offset, second_summary)


    @triton.jit
    def _fused_v_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        v_ptr,
        sequence: tl.constexpr,
        hidden: tl.constexpr,
        output_features: tl.constexpr,
        head_dim: tl.constexpr,
        weight_stride_output: tl.constexpr,
        weight_stride_inner: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        block_m = tl.program_id(0)
        block_n = tl.program_id(1)

        rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < sequence
        column_mask = columns < output_features
        weight_columns = 2 * output_features + columns

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for start in range(0, hidden, BLOCK_K):
            inner = start + tl.arange(0, BLOCK_K)
            inner_mask = inner < hidden
            x_offsets = rows[:, None].to(tl.int64) * hidden + inner[None, :]
            w_offsets = (
                weight_columns[:, None].to(tl.int64) * weight_stride_output
                + inner[None, :] * weight_stride_inner
            )
            x = tl.load(
                x_ptr + x_offsets,
                mask=row_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            weight = tl.load(
                weight_ptr + w_offsets,
                mask=column_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.int32)

        row_scale = tl.load(x_scale_ptr + rows, mask=row_mask, other=0.0)
        column_scale = tl.load(
            weight_scale_ptr + weight_columns,
            mask=column_mask,
            other=0.0,
        )
        value = accumulator.to(tl.float32)
        value *= row_scale[:, None] * column_scale[None, :]
        value = value.to(tl.bfloat16)

        heads = columns // head_dim
        dims = columns % head_dim
        output_offsets = (
            heads[None, :].to(tl.int64) * sequence * head_dim
            + rows[:, None].to(tl.int64) * head_dim
            + dims[None, :]
        )
        tl.store(
            v_ptr + output_offsets,
            value,
            mask=row_mask[:, None] & column_mask[None, :],
        )


def _plain_qkv_weight(module, x):
    import comfy.ops
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

    weight, bias, handle = comfy.ops.cast_bias_weight(
        module.qkv_proj,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        if bias is not None:
            raise FusedQKVError("fused H3 QKV does not support projection bias")
        if not isinstance(weight, QuantizedTensor):
            raise FusedQKVError("fused H3 QKV requires a quantized projection weight")
        if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
            raise FusedQKVError("fused H3 QKV requires TensorWise INT8 weights")
        params = weight._params
        if getattr(params, "transposed", False):
            raise FusedQKVError("fused H3 QKV does not support transposed weights")
        if not getattr(params, "convrot", False) or int(
            getattr(params, "convrot_groupsize", 0)
        ) != 256:
            raise FusedQKVError("fused H3 QKV requires ConvRot-256 weights")
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        return qdata, scale, handle, weight, bias
    except Exception:
        comfy.ops.uncast_bias_weight(module.qkv_proj, weight, bias, handle)
        raise


def _register_fused_qkv_module(module):
    module_id = getattr(module, "_h3_optimizations_fused_qkv_id", None)
    if module_id is None:
        module_id = next(_FUSED_QKV_IDS)
        module._h3_optimizations_fused_qkv_id = module_id
    _FUSED_QKV_MODULES[module_id] = module
    return module_id


def _fused_qkv_module(module_id):
    module = _FUSED_QKV_MODULES.get(module_id)
    if module is None:
        raise FusedQKVError("fused H3 QKV module is no longer active")
    return module


def _quantize_projection_input(x):
    try:
        from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64
    except ImportError as exc:  # pragma: no cover
        raise FusedQKVError(
            "fused H3 QKV requires Comfy Kitchen's CUDA ConvRot quantizer"
        ) from exc
    return quantize_int8_rowwise_convrot64(x, 256)


def _fused_qkv_tensor_core(
    x_int8,
    qdata,
    x_scale,
    weight_scale,
    q_norm,
    k_norm,
    rope,
    *,
    heads,
    sequence,
    hidden,
    epsilon,
    has_rope,
    rope_strides,
    output_dtype,
    q_block_k=128,
    q_num_warps=8,
    q_num_stages=3,
    k_block_k=128,
    k_num_warps=8,
    k_num_stages=3,
    v_block_m=128,
    v_block_n=256,
    v_block_k=128,
    v_num_warps=8,
    v_num_stages=3,
):
    """Run only the tensor projection and return raw carrier tensors.

    This boundary intentionally has no Comfy module objects, weight handles,
    carrier validation, routing, or dataclass construction. It is
    therefore safe for a caller to replace with a fixed-shape compiled
    callable while the production path remains eager by default.
    """
    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    k_blocks = (sequence + KV_TILE - 1) // KV_TILE
    shape = (1, heads, sequence, HEAD_DIM)
    q_int8 = torch.empty(shape, dtype=torch.int8, device=x_int8.device)
    k_int8 = torch.empty(shape, dtype=torch.int8, device=x_int8.device)
    v = torch.empty(shape, dtype=output_dtype, device=x_int8.device)
    q_scale = torch.empty((1, heads, q_blocks), dtype=torch.float32, device=x_int8.device)
    k_scale = torch.empty((1, heads, k_blocks), dtype=torch.float32, device=x_int8.device)
    q_summary = torch.empty(
        (1, heads, q_blocks, HEAD_DIM), dtype=output_dtype, device=x_int8.device
    )
    k_summary = torch.empty(
        (1, heads, k_blocks, HEAD_DIM), dtype=output_dtype, device=x_int8.device
    )

    qk_grid = (triton.cdiv(sequence, Q_TILE), heads)
    for kind, block_k, num_warps, num_stages in (
        (0, q_block_k, q_num_warps, q_num_stages),
        (1, k_block_k, k_num_warps, k_num_stages),
    ):
        _fused_qk_kernel[qk_grid](
            x_int8,
            qdata,
            x_scale,
            weight_scale,
            q_norm,
            k_norm,
            rope,
            q_int8,
            q_scale,
            k_int8,
            k_scale,
            q_summary,
            k_summary,
            sequence=sequence,
            hidden=hidden,
            heads=heads,
            weight_stride_output=qdata.stride(0),
            weight_stride_inner=qdata.stride(1),
            rope_stride_seq=rope_strides[0],
            rope_stride_dim=rope_strides[1],
            rope_stride_rot=rope_strides[2],
            rope_stride_pair=rope_strides[3],
            epsilon=epsilon,
            has_rope=has_rope,
            KIND=kind,
            BLOCK_M=Q_TILE,
            BLOCK_N=HEAD_DIM,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    v_grid = (
        triton.cdiv(sequence, v_block_m),
        triton.cdiv(heads * HEAD_DIM, v_block_n),
    )
    _fused_v_kernel[v_grid](
        x_int8,
        qdata,
        x_scale,
        weight_scale,
        v,
        sequence=sequence,
        hidden=hidden,
        output_features=heads * HEAD_DIM,
        head_dim=HEAD_DIM,
        weight_stride_output=qdata.stride(0),
        weight_stride_inner=qdata.stride(1),
        BLOCK_M=v_block_m,
        BLOCK_N=v_block_n,
        BLOCK_K=v_block_k,
        num_warps=v_num_warps,
        num_stages=v_num_stages,
    )
    return q_int8, q_scale, k_int8, k_scale, v, q_summary, k_summary


# Public alias for benchmark/test injection without exposing module internals.
fused_qkv_tensor_core = _fused_qkv_tensor_core


@torch.library.custom_op(
    "h3_optimizations::fused_qkv",
    mutates_args=(),
    device_types="cuda",
)
def fused_qkv_op(
    x: torch.Tensor,
    qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    rope: torch.Tensor,
    heads: int,
    epsilon: float,
    has_rope: bool,
    rope_strides: typing.List[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x_int8, x_scale = _quantize_projection_input(x)
    return _fused_qkv_tensor_core(
        x_int8,
        qdata,
        x_scale,
        weight_scale,
        q_norm,
        k_norm,
        rope,
        heads=heads,
        sequence=x.shape[0],
        hidden=x.shape[1],
        epsilon=epsilon,
        has_rope=has_rope,
        rope_strides=rope_strides,
        output_dtype=x.dtype,
    )


@fused_qkv_op.register_fake
def _fused_qkv_fake(
    x,
    qdata,
    weight_scale,
    q_norm,
    k_norm,
    rope,
    heads,
    epsilon,
    has_rope,
    rope_strides,
):
    sequence = x.shape[0]
    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    k_blocks = (sequence + KV_TILE - 1) // KV_TILE
    shape = (1, heads, sequence, HEAD_DIM)
    return (
        x.new_empty(shape, dtype=torch.int8),
        x.new_empty((1, heads, q_blocks), dtype=torch.float32),
        x.new_empty(shape, dtype=torch.int8),
        x.new_empty((1, heads, k_blocks), dtype=torch.float32),
        x.new_empty(shape),
        x.new_empty((1, heads, q_blocks, HEAD_DIM)),
        x.new_empty((1, heads, k_blocks, HEAD_DIM)),
    )


@torch.library.custom_op(
    "h3_optimizations::fused_qkv_module",
    mutates_args=(),
    device_types="cuda",
)
def fused_qkv_module_op(
    x: torch.Tensor,
    rope: torch.Tensor,
    module_id: int,
    heads: int,
    epsilon: float,
    has_rope: bool,
    rope_strides: typing.List[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    import comfy.model_management
    import comfy.ops

    module = _fused_qkv_module(module_id)
    qdata, weight_scale, handle, held_weight, bias = _plain_qkv_weight(module, x)
    try:
        inner = heads * HEAD_DIM
        expected_weight = (inner * 3, x.shape[1])
        if (tuple(qdata.shape) != expected_weight
                or qdata.dtype != torch.int8
                or qdata.device != x.device):
            raise FusedQKVError(
                "fused H3 QKV weight shape is %s; expected %s"
                % (tuple(qdata.shape), expected_weight)
            )
        weight_scale = weight_scale.reshape(-1).contiguous()
        if (weight_scale.numel() != inner * 3
                or weight_scale.dtype != torch.float32
                or weight_scale.device != x.device):
            raise FusedQKVError("fused H3 QKV weight scale shape is invalid")

        q_norm = comfy.model_management.cast_to(
            module.q_norm.weight, device=x.device, dtype=x.dtype
        ).contiguous()
        k_norm = comfy.model_management.cast_to(
            module.k_norm.weight, device=x.device, dtype=x.dtype
        ).contiguous()
        if (q_norm.numel() != HEAD_DIM or k_norm.numel() != HEAD_DIM
                or q_norm.dtype != x.dtype or k_norm.dtype != x.dtype):
            raise FusedQKVError("fused H3 QKV RMSNorm weights are invalid")

        x_int8, x_scale = _quantize_projection_input(x)
        carriers = _fused_qkv_tensor_core(
            x_int8,
            qdata,
            x_scale,
            weight_scale,
            q_norm,
            k_norm,
            rope,
            heads=heads,
            sequence=x.shape[0],
            hidden=x.shape[1],
            epsilon=epsilon,
            has_rope=has_rope,
            rope_strides=rope_strides,
            output_dtype=x.dtype,
        )
        return carriers
    finally:
        comfy.ops.uncast_bias_weight(module.qkv_proj, held_weight, bias, handle)


@fused_qkv_module_op.register_fake
def _fused_qkv_module_fake(
    x,
    rope,
    module_id,
    heads,
    epsilon,
    has_rope,
    rope_strides,
):
    sequence = x.shape[0]
    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    k_blocks = (sequence + KV_TILE - 1) // KV_TILE
    shape = (1, heads, sequence, HEAD_DIM)
    return (
        x.new_empty(shape, dtype=torch.int8),
        x.new_empty((1, heads, q_blocks), dtype=torch.float32),
        x.new_empty(shape, dtype=torch.int8),
        x.new_empty((1, heads, k_blocks), dtype=torch.float32),
        x.new_empty(shape),
        x.new_empty((1, heads, q_blocks, HEAD_DIM)),
        x.new_empty((1, heads, k_blocks, HEAD_DIM)),
    )


def run_fused_qkv(module, x, rope_freqs, *, layer_index, tensor_core=None):
    import comfy.model_management
    import comfy.ops

    if not TRITON_AVAILABLE:
        raise FusedQKVError("fused H3 QKV requires Triton")
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise FusedQKVError("fused H3 QKV requires a rank-2 CUDA BF16 input")
    if comfy.model_management.in_training:
        raise FusedQKVError("fused H3 QKV is inference-only")
    if module.head_dim != HEAD_DIM:
        raise FusedQKVError("fused H3 QKV requires head_dim 128")
    if rope_freqs is not None and (
        rope_freqs.ndim != 6
        or tuple(rope_freqs.shape[:3]) != (1, x.shape[0], 1)
        or int(rope_freqs.shape[-3]) * 2 != ROT_DIM
        or tuple(rope_freqs.shape[-2:]) != (2, 2)
        or rope_freqs.device != x.device
    ):
        raise FusedQKVError("fused H3 QKV requires H3's 96-wide split-half RoPE")
    if float(module.q_norm.eps) != float(module.k_norm.eps):
        raise FusedQKVError("fused H3 QKV requires matching Q/K RMSNorm epsilon")

    sequence, hidden = x.shape
    if sequence <= 0 or hidden % 256:
        raise FusedQKVError(
            "fused H3 QKV requires a non-empty ConvRot-256 hidden dimension"
        )
    heads = int(module.heads)
    inner = heads * HEAD_DIM
    expected_weight = (inner * 3, hidden)
    held_weight = None
    bias = None
    handle = None
    try:
        if rope_freqs is None:
            rope = x.new_empty((1, 1, 1, 16, 2, 2))
            rope_strides = (0, 0, 0, 0)
        else:
            rope = rope_freqs
            rope_strides = (
                rope.stride(1),
                rope.stride(3),
                rope.stride(4),
                rope.stride(5),
            )

        if tensor_core is None:
            module_id = getattr(module, "_h3_optimizations_fused_qkv_id", None)
            if module_id is None:
                raise FusedQKVError("fused H3 QKV module was not registered")
            carriers = fused_qkv_module_op(
                x,
                rope,
                module_id,
                heads,
                float(module.q_norm.eps),
                rope_freqs is not None,
                list(rope_strides),
            )
        else:
            qdata, weight_scale, handle, held_weight, bias = _plain_qkv_weight(module, x)
            if (tuple(qdata.shape) != expected_weight
                    or qdata.dtype != torch.int8
                    or qdata.device != x.device):
                raise FusedQKVError(
                    "fused H3 QKV weight shape is %s; expected %s"
                    % (tuple(qdata.shape), expected_weight)
                )
            weight_scale = weight_scale.reshape(-1).contiguous()
            if (weight_scale.numel() != inner * 3
                    or weight_scale.dtype != torch.float32
                    or weight_scale.device != x.device):
                raise FusedQKVError("fused H3 QKV weight scale shape is invalid")

            q_norm = comfy.model_management.cast_to(
                module.q_norm.weight, device=x.device, dtype=x.dtype
            ).contiguous()
            k_norm = comfy.model_management.cast_to(
                module.k_norm.weight, device=x.device, dtype=x.dtype
            ).contiguous()
            if (q_norm.numel() != HEAD_DIM or k_norm.numel() != HEAD_DIM
                    or q_norm.dtype != x.dtype or k_norm.dtype != x.dtype):
                raise FusedQKVError("fused H3 QKV RMSNorm weights are invalid")

            x_int8, x_scale = _quantize_projection_input(x)
            if (
                tuple(x_int8.shape) != tuple(x.shape)
                or x_int8.dtype != torch.int8
                or x_int8.device != x.device
                or not x_int8.is_contiguous()
                or x_scale.numel() != sequence
                or x_scale.dtype != torch.float32
                or x_scale.device != x.device
            ):
                raise FusedQKVError(
                    "Comfy Kitchen returned an invalid ConvRot activation carrier"
                )
            carriers = tensor_core(
                x_int8,
                qdata,
                x_scale.reshape(-1).contiguous(),
                weight_scale,
                q_norm,
                k_norm,
                rope,
                heads=heads,
                sequence=sequence,
                hidden=hidden,
                epsilon=float(module.q_norm.eps),
                has_rope=rope_freqs is not None,
                rope_strides=rope_strides,
                output_dtype=x.dtype,
            )
        (
            q_int8,
            q_scale,
            k_int8,
            k_scale,
            v,
            q_summary,
            k_summary,
        ) = carriers
        return validate_prepared_fused_qkv(
            PreparedFusedQKV(
                q_int8=q_int8,
                q_scale=q_scale,
                k_int8=k_int8,
                k_scale=k_scale,
                v=v,
                q_summary=q_summary,
                k_summary=k_summary,
                output_dtype=x.dtype,
                sequence=int(sequence),
                heads=heads,
                head_dim=HEAD_DIM,
                layer_index=int(layer_index),
                smooth_k=False,
            )
        )
    finally:
        if held_weight is not None:
            comfy.ops.uncast_bias_weight(module.qkv_proj, held_weight, bias, handle)


class FusedQKVProjector:
    name = "h3_fused_qkv"

    def __init__(self, tensor_core=None):
        self.tensor_core = tensor_core

    @property
    def installation_signature(self):
        if self.tensor_core is None:
            tensor_core = None
        else:
            function = getattr(self.tensor_core, "__func__", self.tensor_core)
            tensor_core = (
                getattr(function, "__module__", type(function).__module__),
                getattr(function, "__qualname__", type(function).__qualname__),
                id(function),
            )
        return (self.name, tensor_core)

    def bind(self, module):
        if self.tensor_core is None:
            return _register_fused_qkv_module(module)
        return None

    def project(self, module, x, rope_freqs, *, layer_index, transformer_options):
        return run_fused_qkv(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            tensor_core=self.tensor_core,
        )
