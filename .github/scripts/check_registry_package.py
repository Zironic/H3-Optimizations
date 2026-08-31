"""Guard the file list and contents of the published Registry package.

The Comfy Registry runs an automated scanner over every published version.
It is syntactic, so it reads ``importlib.import_module`` as obfuscated code
and ``os.environ`` as environment manipulation, and a release it flags stays
``NodeVersionStatusFlagged`` -- which leaves ComfyUI-Manager advertising the
last ``Active`` version instead. Versions 0.2.10 through 0.2.18 were all
flagged that way on benign code.

This checks the artifact that actually ships, not the repository: it applies
``.comfyignore`` the way comfy-cli does, then asserts that what survives holds
no pattern known to trip the scanner and no build-only file.

Run locally with:  python .github/scripts/check_registry_package.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import pathspec
except ImportError:
    sys.exit(
        "this check needs pathspec, the matcher comfy-cli uses for "
        ".comfyignore:  pip install pathspec"
    )

ROOT = Path(__file__).resolve().parents[2]

# Patterns the Registry scanner has flagged, or reads the same way.
BANNED = (
    (r"\bimportlib\.import_module\s*\(", "dynamic import (reads as obfuscated code)"),
    (r"\bimport_module\s*\(", "dynamic import (reads as obfuscated code)"),
    (r"\bos\.environ\b", "environment read (reads as system modification)"),
    (r"\bos\.getenv\s*\(", "environment read (reads as system modification)"),
    (r"\b__import__\s*\(", "dynamic import"),
    (r"(?<!\.)\bexec\s*\(", "dynamic code execution"),
    (r"(?<!\.)\beval\s*\(", "dynamic code execution"),
    (r"\bcompile\s*\(.*['\"]exec['\"]", "dynamic code compilation"),
    (r"\bsubprocess\b", "process spawning"),
    (r"\bsocket\.socket\s*\(", "network access"),
)

# Known, reviewed exceptions: (path, substring that must appear on the line).
# Each one is load-bearing and cannot be spelled another way without a real
# behaviour change. Keep this list short and justified.
ALLOWED = {
    # ComfyUI imports a node pack under a generated package name. The inner
    # package must still load as canonical `h3_optimizations` or relative
    # imports resolve twice under different names and class identity checks
    # fail. tests/test_package_import_identity.py pins this behaviour.
    ("__init__.py", "spec_from_file_location"),
    ("__init__.py", "module_from_spec"),
    ("__init__.py", "exec_module"),
}

REQUIRED = (
    "native/bin/BUILD_ID",
    "native/frost/h3_frost_bf16_sm89.cubin",
    "native/frost/h3_frost_bf16_sm89.symbol",
    "native/frost/LICENSE.txt",
    "native/frost/PROVENANCE",
    "native/LICENSE",
    "native/NOTICE.upstream",
    "native/third_party/cutlass/LICENSE.txt",
    "native/third_party/cutlass/PROVENANCE",
    "pyproject.toml",
    "README.md",
)

FORBIDDEN = (
    "native/frost/compile_sm89.py",
    "native/frost/Dockerfile",
    "native/frost/frost_h3.patch",
    "native/CMakeLists.txt",
    "native/selftest.json",
    "native/library_path.txt",
)


def packaged_files():
    """Every path that survives .comfyignore, as comfy-cli would compute it."""
    ignore = ROOT / ".comfyignore"
    lines = ignore.read_text(encoding="utf-8").splitlines() if ignore.exists() else []
    spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
    kept = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(".git/") or spec.match_file(relative):
            continue
        kept.append(relative)
    return kept


def main():
    files = packaged_files()
    problems = []

    present = set(files)
    for required in REQUIRED:
        if required not in present:
            problems.append("missing from the package: %s" % required)
    for forbidden in FORBIDDEN:
        if forbidden in present:
            problems.append("build-only file is being shipped: %s" % forbidden)
    for relative in files:
        if relative.startswith("native/build") and "/bin/" not in relative:
            problems.append("local build output is being shipped: %s" % relative)
        if relative.startswith("native/third_party/cutlass/include/"):
            problems.append("CUTLASS build header is being shipped: %s" % relative)

    for relative in files:
        if not relative.endswith(".py"):
            continue
        text = (ROOT / relative).read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern, reason in BANNED:
                if not re.search(pattern, line):
                    continue
                if any(
                    relative == path and token in line
                    for path, token in ALLOWED
                ):
                    continue
                problems.append(
                    "%s:%d %s\n      %s" % (relative, number, reason, line.strip())
                )
                break  # one report per line, not one per overlapping pattern

    print("Registry package: %d files" % len(files))
    if problems:
        print("\n%d problem(s):\n" % len(problems))
        for problem in problems:
            print("  - %s" % problem)
        return 1
    print("No scanner-tripping pattern or build-only file in the package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
