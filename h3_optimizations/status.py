'''Compact UI summaries for resolved H3 optimization plans.'''

from .plan import (
    EARLY_SCHEDULE_HOLD,
    EARLY_SCHEDULE_RAMP,
    PLAN_KEY,
    SPARSE_BACKEND_AUTO,
    STATUS_KEY,
)


def _model_options(model):
    return getattr(model, 'model_options', {}) or {}


def _status(model):
    transformer_options = (
        _model_options(model).get('transformer_options', {}) or {}
    )
    value = transformer_options.get(STATUS_KEY)
    return value if isinstance(value, dict) else None


def _plan(model):
    return _model_options(model).get(PLAN_KEY)


def _provider_text(section, fallback_label):
    provider = section.get('provider') or fallback_label
    reason = str(section.get('reason') or '').strip()
    return provider if not reason else '%s - %s' % (provider, reason)


def _qkv_weights_text(status):
    labels = sorted(
        set((status.get('weight_formats') or {}).get('qkv') or ())
    )
    if not labels:
        return 'unknown weights'
    lowered = [label.lower() for label in labels]
    if all('bfloat16' in label or 'bf16' in label for label in lowered):
        return 'BF16 weights'
    if all('float16' in label or 'fp16' in label for label in lowered):
        return 'FP16 weights'
    if all(
        'tensorwiseint8layout' in label and 'convrot256' in label
        for label in lowered
    ):
        return 'ConvRot-256 INT8 weights'
    if all('asymw4a8int8layout' in label or 'w4a8' in label for label in lowered):
        return 'W4A8 weights'
    if all('float8' in label or 'fp8' in label for label in lowered):
        return 'FP8 weights'
    if all('nvfp4' in label for label in lowered):
        return 'NVFP4 weights'
    if len(labels) == 1:
        return '%s weights' % labels[0]
    return 'mixed QKV weights'


def format_qkv_execution(status):
    """Describe the QKV route, marking two-pass V wherever it is active.

    The marker is applied here rather than inside each branch because more
    than one carrier family supports it now; a branch that forgot to mention
    it would read as if V were still retained in full.
    """
    text = _format_qkv_execution(status)
    qkv = status.get('fused_qkv') or {}
    if qkv.get('v_memory') == 'two_pass' and 'two-pass V' not in text:
        text = text.replace(
            'retained native Sage K/V',
            'retained native Sage K + staged native Sage V',
        ).replace(
            'retained Sparse Sage K/V',
            'retained Sparse Sage K + staged Sparse Sage V',
        )
        return '%s; two-pass V' % text
    return text


def _format_qkv_execution(status):
    qkv = status.get('fused_qkv') or {}
    provider = qkv.get('provider') or 'standard_h3_qkv'
    projector = qkv.get('projector')
    streamed_q = bool(qkv.get('streamed_q'))
    weights = _qkv_weights_text(status)
    chunk_rows = int(qkv.get('chunk_rows') or 4096)

    if provider == 'standard_h3_qkv':
        text = '%s -> standard QKV path' % weights
        reason = str(qkv.get('reason') or '').strip()
        return text if not reason else '%s (%s)' % (text, reason)

    if provider in (
        'chunked_kitchen_qkv',
        'streamed_bf16_kitchen_qkv',
        'chunked_fp8_kitchen_qkv',
        'force_convrot_int8_kitchen_qkv',
        'force_bf16_streamed_kitchen_qkv',
    ):
        projection = (
            'FP8 projection -> %d-row BF16 chunks' % chunk_rows
            if provider == 'chunked_fp8_kitchen_qkv'
            else (
                'runtime ConvRot-256 INT8 projection -> %d-row BF16 chunks'
                % chunk_rows
                if provider == 'force_convrot_int8_kitchen_qkv'
                else (
                    'forced BF16 projection -> %d-row BF16 chunks'
                    % chunk_rows
                    if provider == 'force_bf16_streamed_kitchen_qkv'
                    else '%d-row BF16 chunks' % chunk_rows
                )
            )
        )
        text = '%s -> %s -> Kitchen INT8 carrier' % (
            weights,
            projection,
        )
        details = []
        if qkv.get('output_streamed'):
            details.append('output streamed')
        return text if not details else '%s; %s' % (text, '; '.join(details))

    if provider in ('chunked_bf16_qkv', 'force_bf16_qkv'):
        prefix = 'forced BF16 projection' if provider == 'force_bf16_qkv' else 'BF16 projection'
        if streamed_q and projector == 'chunked_triton_sparse_qkv':
            return '%s -> %s; retained BF16 K/V + %d-row BF16 Q slabs -> Triton' % (
                weights,
                prefix,
                chunk_rows,
            )
        if projector == 'streamed_dense_bf16_qkv':
            return '%s -> %s in %d-row chunks; full BF16 K/V; Q and output streamed' % (
                weights,
                prefix,
                chunk_rows,
            )
        if projector == 'streamed_dense_sage_qkv':
            return '%s -> %s; retained native Sage K/V + %d-row Q/output slabs' % (
                weights,
                prefix,
                chunk_rows,
            )
        if projector == 'chunked_kitchen_dense_sage_qkv':
            return '%s -> %s in %d-row chunks -> dense Sage Q/K carrier; V retained in BF16' % (
                weights,
                prefix,
                chunk_rows,
            )
        if streamed_q and projector == 'chunked_sparse_sage_qkv':
            return '%s -> %s; retained Sparse Sage K/V + %d-row BF16 Q slabs' % (
                weights,
                prefix,
                chunk_rows,
            )
        if streamed_q and projector == 'streamed_frost_bf16_qkv':
            return '%s -> %s; retained sequence-major BF16 K/V + %d-row BF16 Q/output slabs -> FROST' % (
                weights,
                prefix,
                chunk_rows,
            )
        return '%s -> %s in %d-row chunks -> full BF16 Q/K/V' % (
            weights,
            prefix,
            chunk_rows,
        )

    if provider == 'force_fp8_qkv':
        if streamed_q:
            return '%s -> forced FP8 projection; retained Sparse Sage K/V + %d-row BF16 Q slabs' % (
                weights,
                chunk_rows,
            )
        return '%s -> forced FP8 projection in %d-row chunks -> full BF16 Q/K/V' % (
            weights,
            chunk_rows,
        )
    if provider == 'force_convrot_int8_qkv':
        if projector == 'streamed_dense_sage_qkv':
            return '%s -> runtime ConvRot-256 INT8 projection; retained native Sage K/V + %d-row Q/output slabs' % (
                weights,
                chunk_rows,
            )
        if projector == 'chunked_kitchen_dense_sage_qkv':
            return '%s -> runtime ConvRot-256 INT8 projection in %d-row chunks -> dense Sage Q/K carrier; V retained in BF16' % (
                weights,
                chunk_rows,
            )
        return '%s -> runtime ConvRot-256 INT8 projection in %d-row chunks -> full BF16 Q/K/V' % (
            weights,
            chunk_rows,
        )
    if provider == 'convrot_int8_dense_sage':
        return '%s -> dense Sage carrier; V retained in BF16' % weights
    if provider in ('convrot_int8_sparse_sage', 'chunked_fp8_sparse_sage'):
        lifetime = (
            'retained Sparse Sage K/V + %d-row BF16 Q slabs'
            if streamed_q
            else '%d-row projection -> Sparse Sage carrier; V retained in BF16'
        )
        return ('%s -> ' + lifetime) % (
            weights,
            chunk_rows,
        )
    if provider == 'chunked_triton_bf16_sparse':
        if streamed_q:
            return '%s -> retained BF16 K/V + %d-row BF16 Q slabs -> Triton' % (
                weights,
                chunk_rows,
            )
        return '%s -> %d-row projection -> Triton BF16 carrier' % (
            weights,
            chunk_rows,
        )
    if provider == 'force_convrot_int8_triton_qkv':
        return '%s -> runtime ConvRot-256 INT8 projection -> retained BF16 K/V + %d-row BF16 Q slabs -> Triton' % (
            weights,
            chunk_rows,
        )
    if provider == 'streamed_frost_bf16_qkv':
        return '%s -> retained sequence-major BF16 K/V + %d-row BF16 Q/output slabs -> FROST' % (
            weights,
            chunk_rows,
        )
    if provider == 'force_convrot_int8_frost_qkv':
        return '%s -> runtime ConvRot-256 INT8 projection -> retained sequence-major BF16 K/V + %d-row BF16 Q/output slabs -> FROST' % (
            weights,
            chunk_rows,
        )
    return '%s -> %s' % (weights, provider)


def _mark_runtime_fallback(qkv, line):
    if qkv.get('provider') in (
        'chunked_fp8_kitchen_qkv',
        'chunked_fp8_sparse_sage',
    ):
        return line + ' [selected; runtime FP8 binding may fall back to standard QKV]'
    return line


def _v_memory_notice(qkv):
    '''Report a Lower VRAM request the active attention path cannot honour.

    Kitchen INT8 and compatible Sage FP8 projectors can stage V in two passes.
    Other attention paths retain V, so staying silent would leave the readout
    implying a saving that never happened.
    '''
    if qkv.get('v_memory_requested') != 'two_pass':
        return None
    if qkv.get('v_memory') == 'two_pass':
        return None
    return (
        'Attention memory mode: Lower VRAM requested but not available on '
        'this attention path; running Standard.'
    )


def _composition_lines(status):
    composition = status.get('composition') or {}
    lines = []
    if composition.get('external_attention_preserved'):
        lines.append('Composition: preserved explicit external attention.')
    preserved = composition.get('preserved_object_patches') or {}
    attention = len(preserved.get('attention') or ())
    blocks = len(preserved.get('blocks') or ())
    final_layer = bool(preserved.get('final_layer'))
    embedding = bool(preserved.get('embedding'))
    if attention or blocks or final_layer or embedding:
        details = []
        if attention:
            details.append('%d attention' % attention)
        if blocks:
            details.append('%d block' % blocks)
        if final_layer:
            details.append('FinalLayer')
        if embedding:
            details.append('embedding _forward')
        lines.append(
            'Composition: preserved foreign object patches (%s); conflicting '
            'H3 sub-optimizations are disabled.'
            % ', '.join(details)
        )
    return lines


def format_memory_status(model):
    status = _status(model)
    if status is None:
        return 'Skipped: input model is not MiniMax H3.'

    attention = status.get('attention', {})
    qkv = status.get('fused_qkv', {})
    mlp = status.get('mlp', {})
    final_layer = status.get('final_layer') or {}
    embedding_memory = status.get('embedding_memory') or {}
    lines = [
        'Attention: %s' % (attention.get('selected') or 'preserve incoming'),
        'QKV: %s' % format_qkv_execution(status),
        'MLP: %s' % _provider_text(mlp, 'off'),
    ]
    if qkv.get('out_proj_runtime_convrot_int8'):
        lines.insert(2, 'Attention output: runtime ConvRot-256 INT8')
    if final_layer.get('chunked'):
        lines.append(
            'FinalLayer: chunked (%d-row chunks)'
            % int(final_layer.get('chunk_rows') or 4096)
        )
    lines[1] = _mark_runtime_fallback(qkv, lines[1])
    chunk_rows = mlp.get('chunk_rows')
    if chunk_rows is not None and mlp.get('provider') not in (
        None,
        'off',
        'preserve_upstream_mlp',
    ):
        mlp_index = next(
            index for index, line in enumerate(lines) if line.startswith('MLP:')
        )
        lines[mlp_index] += ' (%d-row chunks)' % int(chunk_rows)
    if embedding_memory.get('selected') == 'release':
        lines.append('Embedding memory: released before block 0')
    elif embedding_memory.get('selected') == 'stock':
        lines.append('Embedding memory: stock lifetime')
    v_memory_notice = _v_memory_notice(qkv)
    if v_memory_notice is not None:
        lines.append(v_memory_notice)
    lines.extend(_composition_lines(status))
    return '\n'.join(lines)


def format_sparse_status(model):
    status = _status(model)
    if status is None:
        return 'Skipped: input model is not MiniMax H3.'

    qkv = status.get('fused_qkv', {})
    mlp = status.get('mlp', {})
    sparse = status.get('sparse') or {}
    attention = status.get('attention', {})
    plan_sparse = getattr(_plan(model), 'sparse', None)
    budget = sparse.get('video_budget')
    if budget is None:
        budget = getattr(plan_sparse, 'video_budget', 0.0)
    backend_request = sparse.get('backend')
    if backend_request is None:
        backend_request = getattr(plan_sparse, 'backend', SPARSE_BACKEND_AUTO)

    selected = attention.get('selected') or 'normal Comfy attention'
    reason = str(attention.get('reason') or '').strip()
    if selected == 'sparse_sage':
        attention_line = 'Attention: Sparse Sage'
    elif selected == 'triton_sparse_bf16':
        attention_line = 'Attention: BF16 Triton Sparse'
    elif selected == 'flex_attention_fp8':
        attention_line = 'Attention: FP8 FlexAttention'
    elif selected == 'frost_bf16_sm89':
        attention_line = 'Attention: FROST BF16 (SM89)'
    elif selected == 'sparse_kitchen_int8':
        attention_line = 'Attention: Comfy Kitchen INT8 Sparse'
    elif selected == 'existing_dense_sparse':
        attention_line = 'Attention: Existing Dense Sparse'
    else:
        attention_line = 'Attention: %s' % selected

    lines = [
        attention_line,
        'Video token order: %s' % (
            sparse.get('video_token_order')
            or getattr(plan_sparse, 'video_token_order', 'unknown')
        ),
        'Requested video KV budget: %.1f%%' % (float(budget) * 100.0),
        'QKV: %s' % format_qkv_execution(status),
        (
            'Effective density rounds up to a whole KV-tile count at runtime; '
            'non-video context and mixed boundary tiles remain dense.'
        ),
    ]
    if backend_request != SPARSE_BACKEND_AUTO:
        lines.insert(1, 'Requested sparse backend: %s' % backend_request)
    elif selected != 'sparse_kitchen_int8' and reason:
        lines.insert(1, 'Sparse fallback: %s' % reason)
    step_budgets = sparse.get('step_video_budgets')
    if step_budgets:
        budget_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith('Requested video KV budget:')
        )
        lines.insert(
            budget_index + 1,
            'Per-step benchmark schedule: %s'
            % ', '.join('%.0f%%' % (value * 100.0) for value in step_budgets),
        )
    qkv_index = next(
        index for index, line in enumerate(lines) if line.startswith('QKV:')
    )
    if qkv.get('provider') in (
        'convrot_int8_sparse_sage',
        'chunked_fp8_sparse_sage',
        'chunked_triton_bf16_sparse',
        'force_convrot_int8_triton_qkv',
    ):
        lines[qkv_index] += ' (%d-row chunks)' % int(
            qkv.get('chunk_rows') or 4096
        )
    lines[qkv_index] = _mark_runtime_fallback(qkv, lines[qkv_index])
    if qkv.get('out_proj_runtime_convrot_int8'):
        lines.insert(qkv_index + 1, 'Attention output: runtime ConvRot-256 INT8')

    early_steps = sparse.get('early_steps')
    if early_steps is None:
        early_steps = getattr(plan_sparse, 'early_steps', None)
    if early_steps is not None:
        early_kv = sparse.get('early_kv')
        late_steps = sparse.get('late_steps')
        late_kv = sparse.get('late_kv')
        if early_kv is None:
            early_kv = getattr(plan_sparse, 'early_kv', budget)
        if late_steps is None:
            late_steps = getattr(plan_sparse, 'late_steps', 0)
        if late_kv is None:
            late_kv = getattr(plan_sparse, 'late_kv', budget)
        early_schedule = sparse.get('early_schedule')
        if early_schedule is None:
            early_schedule = getattr(
                plan_sparse,
                'early_schedule',
                EARLY_SCHEDULE_HOLD,
            )
        if int(early_steps) == 0:
            early_text = 'Early: disabled'
        elif early_schedule == EARLY_SCHEDULE_RAMP:
            early_text = (
                'Early ramp: %.1f%% -> %.1f%% KV over %d steps'
                % (
                    float(early_kv) * 100.0,
                    float(budget) * 100.0,
                    int(early_steps),
                )
            )
        else:
            early_text = (
                'Early hold: first %d steps at %.1f%% KV'
                % (int(early_steps), float(early_kv) * 100.0)
            )
        schedule_line = (
            '%s; Late: last %d steps at %.1f%% KV.'
            % (
                early_text,
                int(late_steps),
                float(late_kv) * 100.0,
            )
        )
        budget_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith('Requested video KV budget:')
        )
        lines.insert(budget_index + 1, schedule_line)
    elif sparse.get('denser_early_late_steps'):
        budget_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith('Requested video KV budget:')
        )
        lines.insert(
            budget_index + 1,
            'Early ramp: starts at 50% KV or higher; targets +12 points/step on average.',
        )

    if mlp.get('provider') not in (None, 'off'):
        lines.append(
            'MLP: %s'
            % _provider_text(mlp, 'off')
        )
    v_memory_notice = _v_memory_notice(qkv)
    if v_memory_notice is not None:
        lines.append(v_memory_notice)
    lines.extend(_composition_lines(status))
    return '\n'.join(lines)
