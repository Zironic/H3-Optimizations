'''AIMDO VBAR residency cap, with H3 block-equivalent presets.'''

import logging

import comfy.model_management
from comfy.patcher_extension import CallbacksMP
from comfy_api.latest import io, ui

from .model import get_h3_blocks, is_minimax_h3


PAGE_SIZE = 32 * 1024 * 1024
CALLBACK_KEY = 'h3_optimizations_aimdo_residency_limiter'
RESIDENCY_OPTIONS = ('0 blocks', '1 block', '2 blocks', '4 blocks', 'stock')
DEFAULT_RESIDENCY = '0 blocks'
RESIDENCY_BLOCKS = {
    '0 blocks': 0,
    '1 block': 1,
    '2 blocks': 2,
    '4 blocks': 4,
}
LOG_PREFIX = '[H3 Optimizations]'


class AIMDOResidencyLimiterError(RuntimeError):
    pass


def _pages_for_allocation(allocation):
    try:
        vbar, address, size = allocation
        base = int(vbar.base_addr)
        address = int(address)
        size = int(size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AIMDOResidencyLimiterError('invalid H3 AIMDO VBAR allocation') from exc
    if address < base or size < 0:
        raise AIMDOResidencyLimiterError('invalid H3 AIMDO VBAR allocation')
    if size == 0:
        return vbar, frozenset()
    first = (address - base) // PAGE_SIZE
    stop = (address - base + size + PAGE_SIZE - 1) // PAGE_SIZE
    return vbar, frozenset(range(first, stop))


def _block_page_counts(blocks):
    vbar = None
    counts = []
    for block in blocks:
        pages = set()
        for module in block.modules():
            allocation = getattr(module, '_v', None)
            if allocation is None:
                continue
            allocation_vbar, allocation_pages = _pages_for_allocation(allocation)
            if vbar is None:
                vbar = allocation_vbar
            elif allocation_vbar is not vbar:
                raise AIMDOResidencyLimiterError(
                    'H3 block weights span multiple AIMDO VBARs'
                )
            pages.update(allocation_pages)
        if pages:
            counts.append(len(pages))
    if vbar is None or not counts:
        raise AIMDOResidencyLimiterError(
            'H3 AIMDO limiter found no VBAR-backed block weights after load'
        )
    return vbar, tuple(counts)


def _residency_cap_pages(block_page_counts, block_equivalents):
    block_equivalents = int(block_equivalents)
    if block_equivalents < 0:
        raise ValueError('block_equivalents must be non-negative')
    return sum(sorted(block_page_counts, reverse=True)[:block_equivalents])


def _zero_residency_vbar(patcher):
    get_vbar = getattr(patcher, '_vbar_get', None)
    if callable(get_vbar):
        return get_vbar()

    # Compatibility fallback for older/test patchers that do not expose the
    # DynamicVRAM VBAR directly. This fallback remains H3-specific.
    blocks = get_h3_blocks(patcher)
    if not blocks:
        return None
    vbar, _block_page_counts_unused = _block_page_counts(blocks)
    return vbar


def _apply_residency_cap(patcher, block_equivalents):
    if not patcher.is_dynamic():
        return
    if comfy.model_management.NUM_STREAMS <= 0:
        raise AIMDOResidencyLimiterError(
            'AIMDO Residency Limiter requires async weight offloading'
        )

    block_equivalents = int(block_equivalents)
    if block_equivalents < 0:
        raise ValueError('block_equivalents must be non-negative')

    if block_equivalents == 0:
        vbar = _zero_residency_vbar(patcher)
        if vbar is None:
            return
        cap_pages = 0
    else:
        blocks = get_h3_blocks(patcher)
        if not blocks:
            return
        vbar, block_page_counts = _block_page_counts(blocks)
        cap_pages = _residency_cap_pages(block_page_counts, block_equivalents)

    native_pages = int(vbar.get_nr_pages())
    expected_pages = min(cap_pages, native_pages)

    vbar.set_watermark(cap_pages * PAGE_SIZE)
    actual_pages = int(vbar.get_watermark())
    if actual_pages != expected_pages:
        raise AIMDOResidencyLimiterError(
            'AIMDO applied watermark %d pages; expected %d'
            % (actual_pages, expected_pages)
        )

    residency = tuple(int(value) for value in vbar.get_residency())
    if len(residency) < native_pages:
        raise AIMDOResidencyLimiterError(
            'AIMDO residency covers %d of %d native pages'
            % (len(residency), native_pages)
        )
    resident_above = [
        page
        for page, value in enumerate(residency[expected_pages:native_pages], expected_pages)
        if value & 1
    ]
    if resident_above:
        raise AIMDOResidencyLimiterError(
            'AIMDO left %d VBAR page(s) resident above the limiter watermark'
            % len(resident_above)
        )

    logging.debug(
        '%s AIMDO residency limited to %d block-equivalent(s), %d pages, %d MiB',
        LOG_PREFIX,
        block_equivalents,
        expected_pages,
        expected_pages * PAGE_SIZE // (1024 * 1024),
    )


def install_aimdo_limiter(model_patcher, block_equivalents):
    model_patcher.remove_callbacks_with_key(CallbacksMP.ON_LOAD, CALLBACK_KEY)

    def apply_after_load(patcher, _device, *_args):
        _apply_residency_cap(patcher, block_equivalents)

    model_patcher.add_callback_with_key(
        CallbacksMP.ON_LOAD,
        CALLBACK_KEY,
        apply_after_load,
    )


class H3AIMDOResidencyLimiter(io.ComfyNode):
    '''Limit persistent AIMDO VBAR residency after dynamic model loading.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3AIMDOResidencyLimiter',
            display_name='H3 AIMDO Residency Limiter',
            category='H3-Optimizations/Model Patches',
            description=(
                'Controls how much model weight DynamicVRAM keeps resident in VRAM. '
                'The default 0 blocks setting works for any DynamicVRAM model; '
                '1/2/4 block-equivalent settings remain H3-specific. It is mainly '
                'useful for benchmarking, debugging AIMDO, or forcing minimal '
                'persistent model residency on tight VRAM budgets. It does not cap '
                'total GPU memory use.'
            ),
            search_aliases=[
                'AIMDO Limiter',
                'DynamicVRAM limiter',
                'H3 DynamicVRAM limiter',
                'H3 residency cap',
            ],
            inputs=[
                io.Model.Input('model'),
                io.Combo.Input(
                    'residency',
                    display_name='VBAR residency budget',
                    options=list(RESIDENCY_OPTIONS),
                    default=DEFAULT_RESIDENCY,
                    tooltip=(
                        '0 blocks is model-agnostic and keeps no DynamicVRAM VBAR '
                        'pages persistently resident. 1/2/4 blocks use H3 '
                        'block-equivalent budgets and may trade VRAM for less weight '
                        'streaming. stock leaves ComfyUI AIMDO residency management '
                        'unchanged. Numeric choices require DynamicVRAM and async '
                        'weight offloading.'
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, residency=DEFAULT_RESIDENCY):
        if residency not in RESIDENCY_OPTIONS:
            raise ValueError('unknown AIMDO residency budget %r' % residency)
        if residency != '0 blocks' and not is_minimax_h3(model):
            return io.NodeOutput(model)

        patched = model.clone()
        patched.remove_callbacks_with_key(CallbacksMP.ON_LOAD, CALLBACK_KEY)
        if residency == 'stock':
            return io.NodeOutput(
                patched,
                ui=ui.PreviewText('Stock AIMDO residency policy'),
            )

        if not patched.is_dynamic():
            return io.NodeOutput(
                patched,
                ui=ui.PreviewText(
                    'AIMDO residency limiter inactive: DynamicVRAM is not enabled'
                ),
            )

        block_equivalents = RESIDENCY_BLOCKS[residency]
        install_aimdo_limiter(patched, block_equivalents)
        preview = (
            'AIMDO residency limiter armed for zero persistent VBAR pages'
            if block_equivalents == 0
            else 'AIMDO residency limiter armed for %d block-equivalent(s)'
            % block_equivalents
        )
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(preview),
        )
