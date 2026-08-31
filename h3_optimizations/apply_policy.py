'''Install the streamed-QKV policy used by every public optimization node.

The core apply module deliberately stays backend/mechanism focused. This layer
makes QKV streaming policy explicit without duplicating the optimizer: it swaps
in the policy resolver once, teaches the generic Kitchen projector to stream
native FP8 weights as BF16 Q/K/V chunks, and leaves plan application in the
owning apply module.
'''

from . import apply as _base
from .attention.sparse.frost_bf16 import FrostBF16Error
from .environment import RuntimeEnvironment
from .plan import (
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_QUANT,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_FORCE_QUANT,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_FROST,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_KITCHEN_64X128,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_TRITON,
)
from .qkv.formats import describe_linear
from .qkv import policy as _qkv_policy
from .qkv import providers as _providers

_BASE_KITCHEN_PROJECTOR = _base.ChunkedKitchenQKVProjector
_BASE_MLP_RESOLVER = _base.resolve_mlp_provider
_BASE_RESOLVE_ATTENTION = _base._resolve_attention

_BACKEND_LABELS = {
    SPARSE_BACKEND_KITCHEN: 'Kitchen INT8',
    SPARSE_BACKEND_KITCHEN_64X128: 'Kitchen INT8 64x128 (experimental)',
    SPARSE_BACKEND_FROST: 'FROST BF16 (SM89)',
    SPARSE_BACKEND_SAGE: 'Sparse Sage',
    SPARSE_BACKEND_TRITON: 'BF16 Triton',
    SPARSE_BACKEND_FLEX: 'FP8 FlexAttention',
}
_BACKEND_ERRORS = (
    _base.SparseKitchenError,
    _base.SparseSageError,
    _base.TritonSparseError,
    _base.FP8FlexError,
    FrostBF16Error,
)


def _current_capability():
    '''Best-effort capability for ComfyUI's selected NVIDIA device.'''
    environment = RuntimeEnvironment.detect()
    if not environment.cuda_available or environment.capability is None:
        return None
    return tuple(int(value) for value in environment.capability)


def _probe_sparse_backend(backend, environment):
    '''Run only the backend's existing availability preflight.

    This intentionally does not install or select the backend. The probe exists
    only to make an explicit-backend failure actionable without maintaining a
    second architecture compatibility table beside the real preflight code.
    '''
    cuda_available = lambda: bool(getattr(environment, 'cuda_available', False))
    capability_getter = lambda: getattr(environment, 'capability', None)

    if backend in (SPARSE_BACKEND_KITCHEN, SPARSE_BACKEND_KITCHEN_64X128):
        q_tile, kv_tile = (
            (64, 128)
            if backend == SPARSE_BACKEND_KITCHEN_64X128
            else (_base.KITCHEN_Q_TILE, _base.KITCHEN_KV_TILE)
        )
        return _base.preflight_sparse_kitchen(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
            q_tile=q_tile,
            kv_tile=kv_tile,
        )
    if backend == SPARSE_BACKEND_SAGE:
        return _base.preflight_sparse_sage(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
        )
    if backend == SPARSE_BACKEND_TRITON:
        return _base.preflight_triton_sparse(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
        )
    if backend == SPARSE_BACKEND_FLEX:
        return _base.preflight_fp8_flex(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
            device=getattr(environment, 'device_index', None),
        )
    if backend == SPARSE_BACKEND_FROST:
        return _base.preflight_frost_bf16(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
        )
    raise ValueError('unknown sparse backend request %r' % backend)


def _available_sparse_alternatives(selected, environment):
    alternatives = []
    for backend in (
        SPARSE_BACKEND_KITCHEN,
        SPARSE_BACKEND_KITCHEN_64X128,
        SPARSE_BACKEND_FROST,
        SPARSE_BACKEND_SAGE,
        SPARSE_BACKEND_TRITON,
        SPARSE_BACKEND_FLEX,
    ):
        if backend == selected:
            continue
        try:
            _probe_sparse_backend(backend, environment)
        except Exception:
            continue
        alternatives.append(_BACKEND_LABELS[backend])
    return alternatives


def _explicit_backend_error(backend, error, environment):
    alternatives = _available_sparse_alternatives(backend, environment)
    architecture = getattr(environment, 'architecture', None)
    system = architecture or getattr(environment, 'backend', None) or 'this system'
    message = '%s is unavailable on %s: %s.' % (
        _BACKEND_LABELS.get(backend, str(backend)),
        system,
        error,
    )
    if alternatives:
        message += ' Available sparse backends detected on this system: %s.' % (
            ', '.join(alternatives)
        )
    else:
        message += ' No compatible alternative sparse backend was detected.'
    message += (
        ' Use H3 Sparse Attention instead of H3 Sparse Attention (Advanced) '
        'to select a compatible backend automatically or fall back to dense attention.'
    )
    return message


def resolve_attention(plan, model, inventory, environment):
    '''Preserve explicit backend strictness while making failures actionable.'''
    sparse = getattr(plan, 'sparse', None)
    backend = None if sparse is None else sparse.backend
    if backend in (None, SPARSE_BACKEND_AUTO):
        return _BASE_RESOLVE_ATTENTION(plan, model, inventory, environment)
    try:
        return _BASE_RESOLVE_ATTENTION(plan, model, inventory, environment)
    except _BACKEND_ERRORS as error:
        message = _explicit_backend_error(backend, error, environment)
        raise type(error)(message) from error


def resolve_qkv_provider(inventory, *, request, backend_kind, **kwargs):
    '''Apply architecture legality before the ordinary QKV preference policy.

    Turing can execute the package's ConvRot INT8 kernels with FP16 activations,
    but it cannot execute the BF16 carrier paths used by the normal floating
    Auto policy. Keep the public immutable plan as Auto; only the effective
    provider request is coerced here.

    Plain floating weights are runtime-quantized only when a direct Kitchen
    carrier is selected. Other consumers stay on upstream Comfy QKV so Turing
    receives its normal FP16/dequantized execution instead of entering one of
    H3's BF16-only bounded projectors. Existing ConvRot checkpoints retain their
    native provider. Unknown/unsupported quantized layouts likewise stay
    upstream rather than being requantized blindly.
    '''
    capability = _current_capability()
    if capability == (7, 5) and request == FUSED_QKV_AUTO:
        if getattr(inventory, 'qkv_plain_float', False):
            if backend_kind in ('comfy_kitchen_int8', 'sparse_kitchen_int8'):
                return _qkv_policy.resolve_qkv_provider(
                    inventory,
                    request=FUSED_QKV_FORCE_QUANT,
                    backend_kind=backend_kind,
                    **kwargs,
                )
            return _providers.QKVProviderResolution(
                _providers.QKV_STANDARD,
                False,
                'SM75 Auto keeps floating QKV on upstream FP16 execution '
                'unless a direct Kitchen INT8 carrier is selected',
            )
        if not getattr(inventory, 'qkv_convrot_int8_256', False):
            return _providers.QKVProviderResolution(
                _providers.QKV_STANDARD,
                False,
                'SM75 Auto leaves non-ConvRot quantized QKV on upstream Comfy '
                'execution so unsupported storage formats can dequantize safely',
            )
    return _qkv_policy.resolve_qkv_provider(
        inventory,
        request=request,
        backend_kind=backend_kind,
        **kwargs,
    )


def resolve_mlp_provider(inventory, *, request, **kwargs):
    '''Prefer ConvRot INT8 for floating Auto MLPs on NVIDIA.

    Checkpoint-native quantization remains authoritative: native FP8, W4A8 and
    ConvRot checkpoints keep their existing providers. Only plain floating H3
    MLP weights are runtime-quantized, using the Kitchen-backed ConvRot INT8
    execution path. Non-NVIDIA backends keep the ordinary provider policy.
    '''
    if (
        _current_capability() is not None
        and request == MLP_MEMORY_AUTO
        and getattr(inventory, 'mlp_plain_float', False)
    ):
        request = MLP_MEMORY_FORCE_QUANT
    return _BASE_MLP_RESOLVER(inventory, request=request, **kwargs)


class PolicyChunkedKitchenQKVProjector(_BASE_KITCHEN_PROJECTOR):
    '''Generic Kitchen carrier with checkpoint-native FP8 auto-binding.

    `fp8_projection=True` keeps its historical meaning: the policy explicitly
    authorized a floating checkpoint to be converted to FP8 as a fallback.
    When it is false and the checkpoint is already FP8, use the same held FP8
    linear without changing the public provider id. In both cases the projected
    Q/K/V chunks handed to the carrier remain BF16.
    '''

    @property
    def installation_signature(self):
        return super().installation_signature + ('native_fp8_bf16_stream',)

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        actual = describe_linear(module.qkv_proj)
        if actual.fp8 and not (
            self.force_weights_bf16
            or self.fp8_projection
            or self.convrot_int8_projection
        ):
            delegate = _BASE_KITCHEN_PROJECTOR(
                chunk_rows=self.chunk_rows,
                fp8_projection=True,
                routing_summaries=self.routing_summaries,
                q_tile=self.q_tile,
                kv_tile=self.kv_tile,
                strided_qk_input=self.strided_qk_input,
                stream_output=self.stream_output,
                streamed_q=self.streamed_q,
                v_mode=self.v_mode,
            )
            return delegate.try_project(
                module,
                x,
                rope_freqs,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
        return super().try_project(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )


# apply.py resolves these globals at execution time. Install the policy once so
# both Memory Optimization and Sparse Attention use identical priorities.
_base.resolve_qkv_provider = resolve_qkv_provider
_base.resolve_mlp_provider = resolve_mlp_provider
_base.ChunkedKitchenQKVProjector = PolicyChunkedKitchenQKVProjector
_base._resolve_attention = resolve_attention


def apply_plan(model, plan):
    return _base.apply_plan(model, plan)
