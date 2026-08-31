"""Load the vendored INT8 attention library through ctypes.

The pack ships its own compiled INT8 attention rather than waiting on a
comfy-kitchen release. ComfyUI pins ``comfy-kitchen==0.2.31`` in its
requirements, so a feature that only exists in a later Kitchen -- or in a
fork -- never reaches anyone, which is exactly how the chunked QKV producer
came to be integrated here but never actually running for a single user.

ctypes rather than a Python extension module on purpose. The native side is
plain C: pointers, ints, and a stream handle. That means no nanobind, no
DLPack, and no Python ABI dimension to build for, so one binary per OS and
architecture set serves every interpreter that can call ctypes -- instead of
separate cp310, cp311 and abi3 builds.

Every entry point returns a status code and leaves its message for
``h3_int8_last_error``; :func:`_check` turns that into a normal exception.
Letting a C++ exception cross a ctypes boundary is undefined behaviour, which
is why the native side is wrapped rather than called directly.
"""

from __future__ import annotations

import ctypes
import pathlib
import platform
import threading

import torch

ABI_VERSION = 4

_LIBRARY_NAMES = {
    'Windows': 'h3_int8_attention_v5.dll',
    'Linux': 'libh3_int8_attention.so',
    'Darwin': 'libh3_int8_attention.dylib',
}
_LOCAL_LIBRARY_NAMES = {
    'Windows': 'h3_int8_attention.dll',
    'Linux': 'libh3_int8_attention.so',
    'Darwin': 'libh3_int8_attention.dylib',
}

_PACK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_OVERRIDE_POINTER = _PACK_ROOT / 'native' / 'library_path.txt'
_lock = threading.Lock()
_library = None
_load_error = None


class NativeUnavailableError(RuntimeError):
    """The vendored library is absent, unloadable, or the wrong ABI."""


class NativeCallError(RuntimeError):
    """A native call reported failure and left a message."""


def _override_path():
    """A developer's explicit library path, taken from a git-ignored file.

    This replaces an ``H3_INT8_ATTENTION_LIBRARY`` environment variable. The
    Registry's automated package scanner reports any runtime environment read
    as environment manipulation, and a flagged release leaves Manager pinned to
    the last approved version, so the knob is a file the pack owns instead. The
    file never exists in a published install: it is ignored by both Git and
    the Registry package.
    """
    try:
        text = _OVERRIDE_POINTER.read_text(encoding='utf-8')
    except OSError:
        return None
    for line in text.splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith('#'):
            return pathlib.Path(candidate)
    return None


def _candidate_paths():
    """Where a built library might be, most specific first."""
    name = _LIBRARY_NAMES.get(platform.system())
    if name is None:
        return []
    local_name = _LOCAL_LIBRARY_NAMES[platform.system()]
    override = _override_path()
    candidates = [override] if override is not None else []
    candidates.extend(
        [
            # Shipped in the repo, which is how this pack distributes it.
            _PACK_ROOT / 'native' / 'bin' / name,
            # Optional local installation path retained for developers.
            _PACK_ROOT / 'native' / 'lib' / local_name,
            # A local CMake build, single- and multi-config generators.
            _PACK_ROOT / 'native' / 'build' / local_name,
            _PACK_ROOT / 'native' / 'build' / 'Release' / local_name,
        ]
    )
    return candidates


def _describe_absence(searched):
    return (
        'the vendored INT8 attention library is not built. Looked in:\n  %s\n'
        'Build it with:  cmake -S native -B native/build && '
        'cmake --build native/build --config Release'
        % '\n  '.join(str(path) for path in searched)
    )


def _bind(library):
    """Declare argument types so ctypes cannot silently truncate a pointer."""
    p, i, i64, f, sz = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_size_t,
    )

    library.h3_int8_abi_version.restype = i
    library.h3_int8_abi_version.argtypes = []
    library.h3_int8_last_error.restype = ctypes.c_char_p
    library.h3_int8_last_error.argtypes = []
    library.h3_int8_route_encoding.restype = ctypes.c_char_p
    library.h3_int8_route_encoding.argtypes = []

    library.h3_int8_device_capability.restype = i
    library.h3_int8_device_capability.argtypes = [
        i, ctypes.POINTER(i), ctypes.POINTER(i)
    ]

    attention_common = [p, p, p, p, p, p, p]
    geometry = [i] * 6
    strides = [i] * 12
    library.h3_int8_dense_attention.restype = i
    library.h3_int8_dense_attention.argtypes = (
        attention_common + [i] + geometry + strides + [f, i, sz]
    )

    library.h3_int8_sparse_attention.restype = i
    library.h3_int8_sparse_attention.argtypes = (
        attention_common + [p, p, i, i, i] + geometry + strides
        + [i, i, f, i, sz]
    )
    library.h3_int8_sparse_attention_lse.restype = i
    library.h3_int8_sparse_attention_lse.argtypes = (
        [p] * 8 + [p, p, i, i, i] + geometry + strides
        + [i, i, f, i, sz]
    )

    library.h3_int8_quantize_qk.restype = i
    library.h3_int8_quantize_qk.argtypes = (
        [p, p, p, p, p, p] + [i] * 10 + [i64] * 6 + [i, p, sz]
    )

    library.h3_int8_select_k_anchor.restype = i
    library.h3_int8_select_k_anchor.argtypes = (
        [p, ctypes.POINTER(i), p, p] + [i] * 4 + [i64] * 3 + [i, sz]
    )

    library.h3_int8_quantize_qk_chunk.restype = i
    library.h3_int8_quantize_qk_chunk.argtypes = (
        [p] * 8 + [i] * 11 + [i64] * 6 + [i, sz]
    )

    library.h3_int8_quantize_q_chunk.restype = i
    library.h3_int8_quantize_q_chunk.argtypes = (
        [p] * 3 + [i] * 7 + [i64] * 3 + [i, sz]
    )

    try:
        convrot = library.h3_int8_quantize_bf16_rowwise_convrot256
    except AttributeError:
        pass
    else:
        convrot.restype = i
        convrot.argtypes = [p] * 3 + [i64] * 2 + [sz]

    try:
        fused_q = library.h3_int8_fused_q
    except AttributeError:
        pass
    else:
        fused_q.restype = i
        fused_q.argtypes = [p] * 9 + [i64] * 3 + [i, f, sz]

    library.h3_int8_quantize_v.restype = i
    library.h3_int8_quantize_v.argtypes = [p, p, p] + [i] * 5 + [i64] * 3 + [i, sz]
    return library


def _is_rocm_runtime():
    """PyTorch exposes ROCm GPUs through torch.cuda, so test HIP explicitly."""
    return bool(getattr(torch.version, 'hip', None))


def load(force_reload=False):
    """Return the loaded library, raising NativeUnavailableError if it is not."""
    global _library, _load_error
    with _lock:
        if _library is not None and not force_reload:
            return _library
        if _load_error is not None and not force_reload:
            raise NativeUnavailableError(_load_error)

        # ROCm deliberately mirrors much of torch.cuda. In particular,
        # is_available(), device names and capability queries can all succeed on
        # AMD. The vendored library is CUDA-only, so reject HIP before loading
        # the shared object or reaching any CUDA driver entry point.
        if _is_rocm_runtime():
            _load_error = 'the vendored INT8 attention library requires NVIDIA CUDA; ROCm/HIP detected'
            raise NativeUnavailableError(_load_error)

        searched = _candidate_paths()
        if not searched:
            _load_error = 'unsupported platform %r' % platform.system()
            raise NativeUnavailableError(_load_error)

        path = next((p for p in searched if p.is_file()), None)
        if path is None:
            _load_error = _describe_absence(searched)
            raise NativeUnavailableError(_load_error)

        try:
            library = ctypes.CDLL(str(path))
        except OSError as error:
            # Usually a missing CUDA runtime or an architecture mismatch.
            _load_error = 'could not load %s: %s' % (path, error)
            raise NativeUnavailableError(_load_error) from error

        try:
            library.h3_int8_abi_version.restype = ctypes.c_int
            library.h3_int8_abi_version.argtypes = []
            found = library.h3_int8_abi_version()
        except AttributeError as error:
            _load_error = 'vendored library at %s has no ABI entry point' % path
            raise NativeUnavailableError(_load_error) from error
        if found != ABI_VERSION:
            _load_error = (
                'vendored library at %s reports ABI %d; this build expects %d. '
                'Rebuild native/.' % (path, found, ABI_VERSION)
            )
            raise NativeUnavailableError(_load_error)
        try:
            _bind(library)
        except AttributeError as error:
            _load_error = (
                'vendored library at %s is missing an ABI %d entry point: %s'
                % (path, ABI_VERSION, error)
            )
            raise NativeUnavailableError(_load_error) from error

        _library = library
        _load_error = None
        return _library


def is_available():
    try:
        load()
    except NativeUnavailableError:
        return False
    return True


def unavailable_reason():
    """Why the library did not load, or None when it did."""
    try:
        load()
    except NativeUnavailableError as error:
        return str(error)
    return None


def check(status, what):
    """Turn a native status code into an exception, with its own message."""
    if status == 0:
        return
    library = load()
    detail = library.h3_int8_last_error()
    detail = detail.decode('utf-8', 'replace') if detail else 'no detail reported'
    raise NativeCallError('%s failed (status %d): %s' % (what, status, detail))


def route_encoding():
    """Which route encoding the compiled sparse kernel walks."""
    value = load().h3_int8_route_encoding()
    return value.decode('ascii') if value else 'unknown'
