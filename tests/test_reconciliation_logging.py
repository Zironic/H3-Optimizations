'''Only the final reconciled H3 plan should be user-facing at INFO.'''

import logging

from h3_optimizations import apply as apply_module


def test_prepare_reconciliation_is_info():
    assert apply_module._reconciliation_log_level('prepare') == logging.INFO


def test_provisional_reconciliations_are_debug():
    assert apply_module._reconciliation_log_level('node') == logging.DEBUG
    assert apply_module._reconciliation_log_level('clone') == logging.DEBUG
    assert apply_module._reconciliation_log_level('future_phase') == logging.DEBUG
