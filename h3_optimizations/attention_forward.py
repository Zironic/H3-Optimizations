'''H3 attention forward with negotiated projected-QKV fallback.'''

from weakref import WeakSet

import comfy.ldm.minimax.model as h3_model
import comfy.model_management
import comfy.quant_ops
from tqdm.auto import tqdm

from . import diagnostics
from .attention import AttentionBackendUnavailable
from .external_consumer import (
    consume_streamed_h3_qkv,
    get_streamed_h3_qkv_consumer,
)
from .normalized_rows import (
    NORM1_SOURCE_KEY,
    NormalizedRowsUnsupported,
    attention_output_buffer,
)
from .ordering_probe import has_ordering_observer, observe_attention


DENSE_KITCHEN_PREQUANTIZED = 'comfy_kitchen_int8_prequantized'
REFERENCE_KINDS = frozenset(('ref_img', 'ref_audio'))
CONDITIONING_KINDS = frozenset(('cond', 'cond_audio'))
_LOGGED_TOKEN_LAYOUTS = WeakSet()


def _tqdm_info(message):
    tqdm.write('\n'.join('[INFO] ' + line for line in message.splitlines()))


def _range_for_kind(segments, kind):
    for start, stop, segment_kind in segments:
        if segment_kind == kind:
            return start, stop
    return None


def _range_size(value):
    return 0 if value is None else int(value[1]) - int(value[0])


def _combined_range(segments):
    if not segments:
        return None
    return int(segments[0][0]), int(segments[-1][1])


def _format_tuple(value):
    if value is None:
        return 'none'
    return '(' + ','.join(str(int(item)) for item in value) + ')'


def _video_shape(layout):
    value = getattr(layout, 'video_shape', None)
    if value is not None:
        return tuple(int(item) for item in value)
    signature = getattr(layout, 'signature', ())
    if len(signature) < 4:
        return None
    return int(signature[1]), int(signature[2]) // 2, int(signature[3]) // 2


def _attention_route(backend_name):
    if backend_name and 'sparse' in backend_name.lower():
        return 'sparse'
    return 'dense'


def _attention_backend_label(backend_name):
    return {
        DENSE_KITCHEN_PREQUANTIZED: 'Comfy Kitchen INT8',
    }.get(backend_name, backend_name or 'ComfyUI attention')


def log_token_diagnostic_once(
    x,
    transformer_options,
    *,
    backend=None,
    projector=None,
    attention=None,
):
    '''Log shape-only packed-token metadata once for each ComfyUI sampling layout.'''
    options = transformer_options or {}
    layout = options.get('minimax_h3_layout')
    if layout is None or layout in _LOGGED_TOKEN_LAYOUTS:
        return
    _LOGGED_TOKEN_LAYOUTS.add(layout)

    segments = tuple(
        (int(start), int(stop), str(kind))
        for start, stop, kind in getattr(layout, 'segments', ())
    )
    text_range = _range_for_kind(segments, 'text')
    audio_range = _range_for_kind(segments, 'audio')
    video_range = _range_for_kind(segments, 'video')
    reference_segments = tuple(
        segment for segment in segments if segment[2] in REFERENCE_KINDS
    )
    conditioning_segments = tuple(
        segment for segment in segments if segment[2] in CONDITIONING_KINDS
    )
    conditioning_range = _combined_range(conditioning_segments)

    packed = int(x.shape[0])
    layout_total = getattr(layout, 'seq_len', None)
    layout_total = None if layout_total is None else int(layout_total)
    text_tokens = _range_size(text_range)
    reference_tokens = sum(stop - start for start, stop, _kind in reference_segments)
    conditioning_tokens = sum(
        stop - start for start, stop, _kind in conditioning_segments
    )
    audio_tokens = _range_size(audio_range)
    video_tokens = _range_size(video_range)
    component_total_without_conditioning = (
        text_tokens + reference_tokens + audio_tokens + video_tokens
    )
    component_total_with_conditioning = (
        component_total_without_conditioning + conditioning_tokens
    )

    backend_name = getattr(backend, 'name', None)
    if backend_name is None and attention is not None:
        backend_name = getattr(attention, '__name__', type(attention).__name__)
    streamed_q = bool(getattr(projector, 'streamed_q', False))
    if projector is not None and type(projector).__name__.startswith('Streamed'):
        streamed_q = True
    q_chunk = getattr(backend, 'query_chunk_rows', None)
    if q_chunk is None:
        q_chunk = getattr(projector, 'chunk_rows', None)

    reference_detail = ','.join(
        '%s[%d:%d]=%d' % (kind, start, stop, stop - start)
        for start, stop, kind in reference_segments
    ) or 'none'
    segment_detail = ','.join(
        '%s[%d:%d]=%d' % (kind, start, stop, stop - start)
        for start, stop, kind in segments
    ) or 'none'
    component_check = (
        'ok'
        if component_total_with_conditioning == packed == layout_total
        else ('unavailable' if layout_total is None else 'mismatch')
    )

    main_line = (
        f'[H3 Tokens] packed={packed} layout_total={layout_total} '
        f'text_tokens={text_tokens} conditioning_tokens={conditioning_tokens} '
        f'reference_tokens={reference_tokens} audio_tokens={audio_tokens} '
        f'video_tokens={video_tokens} video_shape={_format_tuple(_video_shape(layout))} '
        f'q_total={packed} route={_attention_route(backend_name)} '
        f'backend={_attention_backend_label(backend_name)} '
        f'qkv_streaming={"on" if streamed_q else "off"} q_chunk={q_chunk} '
        f'component_total_without_conditioning={component_total_without_conditioning} '
        f'component_total_with_conditioning={component_total_with_conditioning} '
        f'component_check={component_check}'
    )
    detail_line = (
        f'[H3 Tokens Detail] text_range={_format_tuple(text_range)} '
        f'conditioning_range={_format_tuple(conditioning_range)} '
        f'reference_segments={len(reference_segments)} reference_detail={reference_detail} '
        f'audio_range={_format_tuple(audio_range)} video_range={_format_tuple(video_range)} '
        f'segments={segment_detail}'
    )
    _tqdm_info(f'{main_line}\n{detail_line}')


class _AttentionOutProjectionProxy:
    def __init__(self, module, out_proj):
        self._module = module
        self.out_proj = out_proj

    def __getattr__(self, name):
        return getattr(self._module, name)


def _project_attention_output(module, out, out_projection):
    if out_projection is None:
        return module.out_proj(out)
    return out_projection.linear(out)


def finish_qkv_projection(module, projected, rope_freqs):
    seq = projected.shape[0]
    inner = module.heads * module.head_dim
    q, k, v = projected.split(inner, dim=-1)
    v = v.view(seq, module.heads, module.head_dim)

    if rope_freqs is not None:
        if comfy.model_management.in_training:
            raise RuntimeError('H3 optimized attention is inference-only')
        q = q.view(1, seq, module.heads, module.head_dim)
        k = k.view(1, seq, module.heads, module.head_dim)
        qw = comfy.model_management.cast_to(
            module.q_norm.weight,
            device=projected.device,
        )
        kw = comfy.model_management.cast_to(
            module.k_norm.weight,
            device=projected.device,
        )
        rot = rope_freqs.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q,
            k,
            rope_freqs,
            qw,
            kw,
            epsilon=module.q_norm.eps,
            rot_dim=rot,
        )
        q = q[0]
        k = k[0]
    else:
        q = module.q_norm(
            q.view(seq, module.heads, module.head_dim)
        )
        k = module.k_norm(
            k.view(seq, module.heads, module.head_dim)
        )
    return q, k, v


def project_qkv(module, x, rope_freqs):
    with diagnostics.stage('qkv_linear'):
        projected = module.qkv_proj(x)
    with diagnostics.stage('qk_norm_rope'):
        return finish_qkv_projection(module, projected, rope_freqs)


def to_hnd(q, k, v):
    return (
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
    )


def _legacy_attention(module, q, k, v, transformer_options, attention=None):
    extra = {}
    if attention is None:
        attention = h3_model.optimized_attention
        # Newer ComfyUI lets a checkpoint pick attention per block through
        # Attention.comfy_attention; optimized_attention still ranks explicit
        # overrides above it. Older cores have no such member.
        preferred = getattr(module, 'comfy_attention', None)
        if preferred is not None:
            extra['preferred_attention'] = preferred
    return attention(
        q,
        k,
        v,
        module.heads,
        mask=None,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=transformer_options,
        **extra,
    )


def _project_or_none(
    projector,
    module,
    x,
    rope_freqs,
    *,
    layer_index,
    transformer_options,
):
    callback = getattr(projector, 'try_project', None)
    if callback is None:
        callback = projector.project
    return callback(
        module,
        x,
        rope_freqs,
        layer_index=layer_index,
        transformer_options=transformer_options,
    )


def _materialize_norm1_source(x, norm1_source):
    return norm1_source.materialize() if norm1_source is not None else x


def flatten_attention_output(module, out, source):
    """Reach [batch, sequence, heads * head_dim] from whatever the kernel wrote.

    An ``nhd`` kernel output already has sequence ahead of heads, so the
    flatten is a view. An ``hnd`` output needs a transpose whose last two
    dimensions cannot merge, so ``reshape`` copies a second full-sequence BF16
    tensor -- at the production shape that is the single largest avoidable
    allocation in the block.
    """
    if out.ndim != 4:
        raise RuntimeError(
            '%s returned rank-%d output; expected HND rank 4'
            % (source, out.ndim)
        )
    # One expression for both storage layouts. Over head-major storage the
    # dimensions cannot merge and `reshape` copies; over sequence-major
    # storage the transpose lands on contiguous memory and this is a view.
    return out.transpose(1, 2).reshape(
        out.shape[0], out.shape[2], module.heads * module.head_dim
    )


def _finish_projected(module, backend, prepared, out_projection=None):
    # A streamed backend may own the full attention -> out_proj lifetime and
    # return the final hidden-size tensor directly. This is deliberately
    # opt-in so Kitchen, Triton, BF16 and legacy projected contracts stay put.
    execute_projected = getattr(backend, 'execute_projected', None)
    if execute_projected is not None:
        execution_module = (
            module
            if out_projection is None
            else _AttentionOutProjectionProxy(
                module,
                out_projection.linear,
            )
        )
        direct = execute_projected(execution_module, prepared)
        if direct is not None:
            if direct.ndim != 2:
                raise RuntimeError(
                    '%s returned rank-%d direct projected output; expected rank 2'
                    % (
                        getattr(backend, 'name', type(backend).__name__),
                        direct.ndim,
                    )
                )
            return direct

    name = getattr(backend, 'name', type(backend).__name__)
    raw = backend.execute(prepared)
    out = flatten_attention_output(module, raw, name)
    # Only the flattened view is needed from here. Under `hnd` the reshape
    # already copied, so this returns a full-sequence buffer to the allocator
    # before the projection asks for one; under `nhd` `out` aliases `raw` and
    # this is a no-op.
    del raw
    if getattr(backend, 'release_carrier_before_out_proj', False):
        release = getattr(prepared, 'release', None)
        if release is None:
            raise RuntimeError(
                '%s asked to release its carrier but %s cannot release one'
                % (name, type(prepared).__name__)
            )
        release()
    with diagnostics.stage('attention_out'):
        return _project_attention_output(
            module,
            out.squeeze(0),
            out_projection,
        )


def _finish_bf16_projected(
    module,
    backend,
    projected,
    *,
    layer_index,
    transformer_options,
    out_projection=None,
):
    """Consume chunked/native BF16 QKV without running QKV projection again."""
    q, k, v = projected.q, projected.k, projected.v
    backend_name = getattr(backend, 'name', None)

    # The dense Kitchen backend object exists here only because that is the
    # package-owned projector slot used by Memory Optimization. Preserve
    # precision must not force Kitchen attention: feed the already-projected
    # BF16 tensors to whatever attention Comfy/upstream currently selected.
    if backend is None or backend_name == DENSE_KITCHEN_PREQUANTIZED:
        raw = _legacy_attention(
            module,
            q,
            k,
            v,
            transformer_options,
        )
        source = 'existing_attention_bf16'
    else:
        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        try:
            raw = backend.execute(prepared)
        finally:
            del prepared
        source = backend_name or type(backend).__name__

    # The backend launch has consumed the BF16 inputs. Dropping the wrapper
    # here lets the stream-ordered allocator reclaim them before out_proj.
    del projected, q, k, v
    out = flatten_attention_output(module, raw, source)
    del raw
    with diagnostics.stage('attention_out'):
        return _project_attention_output(
            module,
            out.squeeze(0),
            out_projection,
        )


def _finish_streamed_dense_bf16_projected(
    module,
    projected,
    *,
    layer_index,
    transformer_options,
    out_projection=None,
):
    """Consume Q slabs against complete BF16 K/V and project each output slab."""
    output = attention_output_buffer(projected.x)
    external_consumer = get_streamed_h3_qkv_consumer(transformer_options)
    try:
        for start, end, q in projected.stream_q():
            if external_consumer is None:
                raw = _legacy_attention(
                    module,
                    q,
                    projected.k,
                    projected.v,
                    transformer_options,
                )
            else:
                raw = consume_streamed_h3_qkv(
                    external_consumer,
                    q,
                    projected.k,
                    projected.v,
                    q_start=start,
                    q_total=projected.sequence,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
            out = flatten_attention_output(
                module,
                raw,
                'streamed_dense_bf16',
            )
            del raw
            with diagnostics.stage('attention_out'):
                output[start:end].copy_(
                    _project_attention_output(
                        module,
                        out.squeeze(0),
                        out_projection,
                    )
                )
            del q, out
        return output
    finally:
        projected.release()


def make_forward(
    module,
    layer_index,
    backend=None,
    attention=None,
    projector=None,
    fallback_forward=None,
    backend_fallback_to_dense=False,
    force_out_proj_int8=False,
):
    if backend is not None and attention is not None:
        raise ValueError('pass either backend or attention, not both')
    if projector is not None and backend is None:
        raise ValueError('a fused QKV projector requires a consuming backend')
    bind_projector = getattr(projector, 'bind', None)
    if bind_projector is not None:
        bind_projector(module)

    def forward(x, rope_freqs=None, transformer_options=None):
        if force_out_proj_int8:
            from .qkv.int8 import LazyConvRotINT8Linear

        out_projection = (
            LazyConvRotINT8Linear(module.out_proj)
            if force_out_proj_int8
            else None
        )
        try:
            with diagnostics.stage('attention_total'):
                return _forward(
                    x,
                    rope_freqs,
                    transformer_options,
                    out_projection,
                )
        finally:
            if out_projection is not None:
                out_projection.release()

    def _forward(x, rope_freqs, transformer_options, out_projection):
        transformer_options = (
            transformer_options if transformer_options is not None else {}
        )
        if int(layer_index) == 0:
            log_token_diagnostic_once(
                x,
                transformer_options,
                backend=backend,
                projector=projector,
                attention=attention,
            )
        norm1_source = transformer_options.get(NORM1_SOURCE_KEY)
        if norm1_source is not None:
            # The lazy source is an implementation detail of our QKV projector.
            # Do not leak it to attention backends, observers, or foreign hooks.
            transformer_options = transformer_options.copy()
            transformer_options.pop(NORM1_SOURCE_KEY, None)

        ordering_probe = has_ordering_observer(transformer_options)
        if projector is not None and not ordering_probe:
            projection_input = norm1_source if norm1_source is not None else x
            try:
                projected = _project_or_none(
                    projector,
                    module,
                    projection_input,
                    rope_freqs,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
            except NormalizedRowsUnsupported:
                if norm1_source is None:
                    raise
                # A projector may still require a concrete tensor for one
                # internal operation. Materialize only at this boundary and
                # retry the same optimized QKV route rather than exposing the
                # lazy row object to the rest of attention.
                projection_input = norm1_source.materialize()
                projected = _project_or_none(
                    projector,
                    module,
                    projection_input,
                    rope_freqs,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
            if projected is not None:
                from .qkv.bf16 import (
                    PreparedBF16QKV,
                    PreparedStreamedDenseBF16QKV,
                )

                if isinstance(projected, PreparedStreamedDenseBF16QKV):
                    return _finish_streamed_dense_bf16_projected(
                        module,
                        projected,
                        layer_index=layer_index,
                        transformer_options=transformer_options,
                        out_projection=out_projection,
                    )

                if isinstance(projected, PreparedBF16QKV):
                    return _finish_bf16_projected(
                        module,
                        backend,
                        projected,
                        layer_index=layer_index,
                        transformer_options=transformer_options,
                        out_projection=out_projection,
                    )
                prepared = backend.prepare_projected(
                    projected,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
                del projected
                try:
                    return _finish_projected(
                        module,
                        backend,
                        prepared,
                        out_projection,
                    )
                finally:
                    del prepared
            if fallback_forward is not None:
                return fallback_forward(
                    _materialize_norm1_source(x, norm1_source),
                    rope_freqs=rope_freqs,
                    transformer_options=transformer_options,
                )

        qkv_input = _materialize_norm1_source(x, norm1_source)
        q, k, v = project_qkv(module, qkv_input, rope_freqs)
        q, k, v = to_hnd(q, k, v)
        if ordering_probe:
            observe_attention(
                layer_index,
                transformer_options,
                q,
                k,
                v,
            )
        if backend is None:
            out = _legacy_attention(
                module,
                q,
                k,
                v,
                transformer_options,
                attention=attention,
            )
        else:
            fallback_inputs_available = True
            try:
                prepared = backend.prepare(
                    q,
                    k,
                    v,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
                retain_fallback_inputs = (
                    backend_fallback_to_dense
                    and backend.requires_fallback_inputs(prepared)
                )
                if not retain_fallback_inputs:
                    del q, k, v
                    fallback_inputs_available = False
                try:
                    out_hnd = backend.execute(prepared)
                finally:
                    del prepared
            except AttentionBackendUnavailable:
                if (
                    not backend_fallback_to_dense
                    or not fallback_inputs_available
                ):
                    raise
                v = v.contiguous()
                out_hnd = _legacy_attention(
                    module,
                    q,
                    k,
                    v,
                    transformer_options,
                )
            out = flatten_attention_output(
                module,
                out_hnd,
                getattr(backend, 'name', type(backend).__name__),
            )
            del out_hnd

        with diagnostics.stage('attention_out'):
            return _project_attention_output(
                module,
                out.squeeze(0),
                out_projection,
            )

    forward._h3_optimizations_attention = True
    forward._h3_optimizations_layer_index = int(layer_index)
    forward._h3_optimizations_backend = getattr(backend, 'name', None)
    forward._h3_optimizations_projector = getattr(projector, 'name', None)
    forward._h3_optimizations_lazy_norm_source = projector is not None
    forward._h3_optimizations_force_out_proj_int8 = bool(
        force_out_proj_int8
    )
    return forward
