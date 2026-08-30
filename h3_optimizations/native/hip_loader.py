"""Load the experimental gfx12 Sparse Kitchen HIP library through ctypes."""

from __future__ import annotations

import ctypes
import pathlib
import platform
import threading

import torch


ABI_VERSION = 1
_LIBRARY_NAMES = {
    'Linux': 'libh3_hip_sparse_kitchen.so',
    'Windows': 'h3_hip_sparse_kitchen.dll',
}
_PACK_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_HIP_ROOT = _PACK_ROOT / 'native' / 'hip'
_OVERRIDE_POINTER = _HIP_ROOT / 'library_path.txt'
_lock = threading.Lock()
_library = None
_load_error = None


class NativeHIPUnavailableError(RuntimeError):
    pass


class NativeHIPCallError(RuntimeError):
    pass


def _override_path():
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
    name = _LIBRARY_NAMES.get(platform.system())
    if name is None:
        return []
    override = _override_path()
    candidates = [override] if override is not None else []
    candidates.extend(
        [
            _HIP_ROOT / 'bin' / name,
            _HIP_ROOT / 'lib' / name,
            _HIP_ROOT / 'build' / name,
            _HIP_ROOT / 'build' / 'Release' / name,
        ]
    )
    return candidates


def _bind(library):
    p = ctypes.c_void_p
    i = ctypes.c_int
    i64 = ctypes.c_int64
    f = ctypes.c_float
    sz = ctypes.c_size_t

    library.h3_hip_sparse_abi_version.restype = i
    library.h3_hip_sparse_abi_version.argtypes = []
    library.h3_hip_sparse_last_error.restype = ctypes.c_char_p
    library.h3_hip_sparse_last_error.argtypes = []
    library.h3_hip_sparse_route_encoding.restype = ctypes.c_char_p
    library.h3_hip_sparse_route_encoding.argtypes = []

    library.h3_hip_sparse_quantize_qk.restype = i
    library.h3_hip_sparse_quantize_qk.argtypes = (
        [p] * 7 + [i] * 5 + [i64] * 6 + [i, sz]
    )
    library.h3_hip_sparse_quantize_v.restype = i
    library.h3_hip_sparse_quantize_v.argtypes = [p] * 3 + [i] * 3 + [i64] * 2 + [i, sz]
    library.h3_hip_sparse_attention.restype = i
    library.h3_hip_sparse_attention.argtypes = (
        [p] * 9 + [i] * 6 + [i64] * 2 + [f, i, sz]
    )
    return library


def load(force_reload=False):
    global _library, _load_error
    with _lock:
        if _library is not None and not force_reload:
            return _library
        if _load_error is not None and not force_reload:
            raise NativeHIPUnavailableError(_load_error)
        if not getattr(torch.version, 'hip', None):
            _load_error = 'the experimental AMD Sparse Kitchen library requires ROCm/HIP'
            raise NativeHIPUnavailableError(_load_error)

        searched = _candidate_paths()
        path = next((candidate for candidate in searched if candidate.is_file()), None)
        if path is None:
            _load_error = (
                'the experimental AMD Sparse Kitchen library is not built. Looked in:\n  %s\n'
                'Build it with: cmake -S native/hip -B native/hip/build -G Ninja'
                % '\n  '.join(str(candidate) for candidate in searched)
            )
            raise NativeHIPUnavailableError(_load_error)
        try:
            library = ctypes.CDLL(str(path))
            library.h3_hip_sparse_abi_version.restype = ctypes.c_int
            library.h3_hip_sparse_abi_version.argtypes = []
            found = library.h3_hip_sparse_abi_version()
        except (OSError, AttributeError) as error:
            _load_error = 'could not load %s: %s' % (path, error)
            raise NativeHIPUnavailableError(_load_error) from error
        if found != ABI_VERSION:
            _load_error = (
                'HIP library at %s reports ABI %d; this source expects %d'
                % (path, found, ABI_VERSION)
            )
            raise NativeHIPUnavailableError(_load_error)
        try:
            _library = _bind(library)
        except AttributeError as error:
            _load_error = 'HIP library at %s is missing an ABI entry point: %s' % (path, error)
            raise NativeHIPUnavailableError(_load_error) from error
        _load_error = None
        return _library


def is_available():
    try:
        load()
    except NativeHIPUnavailableError:
        return False
    return True


def unavailable_reason():
    try:
        load()
    except NativeHIPUnavailableError as error:
        return str(error)
    return None


def check(status, operation):
    if status == 0:
        return
    detail = load().h3_hip_sparse_last_error()
    detail = detail.decode('utf-8', 'replace') if detail else 'no detail reported'
    raise NativeHIPCallError('%s failed (status %d): %s' % (operation, status, detail))


def route_encoding():
    value = load().h3_hip_sparse_route_encoding()
    return value.decode('ascii') if value else 'unknown'
