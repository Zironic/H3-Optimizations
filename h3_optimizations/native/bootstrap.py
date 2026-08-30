"""Load and self-test the native backend shipped with the repository."""

from __future__ import annotations

import logging
import pathlib

from . import artifacts, loader

LOG_PREFIX = '[H3 Optimizations]'

_PACK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_BIN_DIR = _PACK_ROOT / 'native' / 'bin'
_MARKER = _BIN_DIR / 'BUILD_ID'


def installed_build_id():
    """Which release the installed binary came from, or None for a local build."""
    try:
        return _MARKER.read_text(encoding='utf-8').strip() or None
    except OSError:
        return None


def ensure_native_backend():
    """Return True when the native backend is usable. Never raises."""
    from . import selftest

    try:
        if loader.is_available():
            return _verify(selftest)

        logging.warning(
            '%s NATIVE BACKEND UNAVAILABLE - the repository has no usable '
            'binary for %s. Sparse attention will use the fallback chain. '
            'Build it locally with: cmake -S native -B native/build && '
            'cmake --build native/build --config Release',
            LOG_PREFIX, artifacts.describe_platform(),
        )
        return False
    except Exception as error:  # noqa: BLE001 - startup must not die here
        logging.warning(
            '%s NATIVE BACKEND UNAVAILABLE - unexpected error during setup: '
            '%s: %s. Sparse attention will fall back to a slower path.',
            LOG_PREFIX, type(error).__name__, error,
        )
        return False


def _verify(selftest):
    """Load, check the ABI, then prove the kernels on this actual GPU."""
    try:
        loader.load()
    except loader.NativeUnavailableError as error:
        logging.warning(
            '%s NATIVE BACKEND UNAVAILABLE - %s. Sparse attention will fall '
            'back to a slower path.',
            LOG_PREFIX, error,
        )
        return False

    if not selftest.check():
        # selftest.check already logged what failed and why.
        return False
    logging.debug('%s native backend ready (%s)', LOG_PREFIX, describe())
    return True


def describe():
    """One line for a bug report."""
    try:
        library = loader.load()
        abi = library.h3_int8_abi_version()
        encoding = loader.route_encoding()
    except loader.NativeUnavailableError as error:
        return 'unavailable: %s' % str(error).splitlines()[0]
    return 'abi=%d build=%s platform=%s route=%s' % (
        abi,
        installed_build_id() or 'local',
        artifacts.describe_platform(),
        encoding,
    )


def diagnostics():
    """A block a bug report can paste, with no GPU work beyond the cached test."""
    from . import selftest

    lines = ['native backend:']
    try:
        library = loader.load()
    except loader.NativeUnavailableError as error:
        lines.append('  status   : unavailable')
        for line in str(error).splitlines():
            lines.append('  %s' % line)
        return '\n'.join(lines)

    lines.append('  ABI      : %d (expected %d)' % (
        library.h3_int8_abi_version(), artifacts.REQUIRED_ABI
    ))
    lines.append('  build    : %s' % (installed_build_id() or 'local'))
    lines.append('  platform : %s' % artifacts.describe_platform())
    lines.append('  route    : %s' % loader.route_encoding())
    lines.append('  self-test: %s' % ('passed' if selftest.check() else 'FAILED'))
    return '\n'.join(lines)
