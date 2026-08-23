'''Compact UI summaries for resolved H3 optimization plans.'''

from .plan import PLAN_KEY, SPARSE_BACKEND_AUTO, STATUS_KEY


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


def _mark_runtime_fallback(qkv, line):
    if qkv.get('provider') in (
        'chunked_fp8_kitchen_qkv',
        'chunked_fp8_sparse_sage',
    ):
        return line + ' [selected; runtime FP8 binding may fall back to standard QKV]'
    return line


def format_memory_status(model):
    status = _status(model)
    if status is None:
        return 'Skipped: input model is not MiniMax H3.'

    attention = status.get('attention', {})
    qkv = status.get('fused_qkv', {})
    mlp = status.get('mlp', {})
    lines = [
        'Attention: %s' % (attention.get('selected') or 'preserve incoming'),
        'QKV: %s' % _provider_text(qkv, 'standard_h3_qkv'),
        'MLP: %s' % _provider_text(mlp, 'off'),
    ]
    if qkv.get('provider') in (
        'chunked_kitchen_qkv',
        'chunked_fp8_kitchen_qkv',
    ):
        lines[1] += ' (%d-row chunks, Kitchen %s)' % (
            int(qkv.get('chunk_rows') or 4096),
            qkv.get('producer_abi') or 'producer ABI unavailable',
        )
    lines[1] = _mark_runtime_fallback(qkv, lines[1])
    chunk_rows = mlp.get('chunk_rows')
    if chunk_rows is not None and mlp.get('provider') not in (
        None,
        'off',
        'preserve_upstream_mlp',
    ):
        lines[-1] += ' (%d-row chunks)' % int(chunk_rows)
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
    elif selected == 'triton_sparse_int8':
        attention_line = 'Attention: INT8 Triton Sparse'
    elif selected == 'flex_attention_fp8':
        attention_line = 'Attention: FP8 FlexAttention'
    else:
        attention_line = 'Attention: %s' % selected

    lines = [
        attention_line,
        'Requested video KV budget: %.1f%%' % (float(budget) * 100.0),
        'QKV: %s' % _provider_text(qkv, 'standard_h3_qkv'),
        (
            'Effective density rounds up to a whole KV-tile count at runtime; '
            'non-video context and mixed boundary tiles remain dense.'
        ),
    ]
    if backend_request != SPARSE_BACKEND_AUTO:
        lines.insert(1, 'Requested sparse backend: %s' % backend_request)
    elif selected != 'sparse_sage' and reason:
        lines.insert(1, 'Sparse fallback: %s' % reason)
    layer_budgets = sparse.get('layer_video_budgets')
    if layer_budgets:
        lines.insert(
            2,
            'Static layer table: %d layers, %.1f%%-%.1f%% KV'
            % (
                len(layer_budgets),
                min(layer_budgets) * 100.0,
                max(layer_budgets) * 100.0,
            ),
        )
    if qkv.get('provider') in (
        'convrot_int8_sparse_sage',
        'chunked_fp8_sparse_sage',
        'chunked_triton_int8_sparse',
    ):
        qkv_index = next(
            index for index, line in enumerate(lines) if line.startswith('QKV:')
        )
        lines[qkv_index] += ' (%d-row chunks)' % int(
            qkv.get('chunk_rows') or 4096
        )
        lines[qkv_index] = _mark_runtime_fallback(qkv, lines[qkv_index])

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
        schedule_line = (
            'Early: first %d steps at %.1f%% KV; Late: last %d steps at %.1f%% KV.'
            % (
                int(early_steps),
                float(early_kv) * 100.0,
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
            'First 2 and last 2 steps add 30 percentage points, capped at 100%.',
        )

    if mlp.get('provider') not in (None, 'off'):
        lines.append(
            'MLP: %s'
            % _provider_text(mlp, 'off')
        )
    return '\n'.join(lines)
