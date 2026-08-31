"""Identity of the native binaries committed with this source revision."""

from __future__ import annotations

import platform

# Bump when the native sources or committed binaries change. This marker is
# also what a bug report should quote.
NATIVE_BUILD = 'native-v9'

REQUIRED_ABI = 4


def platform_key():
    return (platform.system(), platform.machine())


def describe_platform():
    system, machine = platform_key()
    return '%s-%s' % (system.lower(), machine.lower())
