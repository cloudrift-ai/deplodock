"""Deplodock package init.

Patches cppyy's bundled ``libCling.so`` once on macOS so its hard-coded
MacPorts ``/opt/local/lib/libzstd.1.dylib`` dependency is rewritten to
the Homebrew path. Runtime ``DYLD_FALLBACK_LIBRARY_PATH`` would normally
fix this, but macOS SIP strips ``DYLD_*`` from ``execve``'d processes
targeting protected binaries (the venv's python is one such), so an
in-process env var workaround can't propagate to dyld. Patching the
install-name with ``install_name_tool`` is local to our venv, idempotent,
and survives across runs.
"""

from __future__ import annotations

import subprocess as _subprocess
import sys as _sys
import sysconfig as _sysconfig
from pathlib import Path as _Path


def _patch_libcling_zstd() -> None:
    if _sys.platform != "darwin":
        return
    purelib = _sysconfig.get_paths().get("purelib")
    if not purelib:
        return
    lib = _Path(purelib) / "cppyy_backend" / "lib" / "libCling.so"
    if not lib.exists():
        return
    bad = "/opt/local/lib/libzstd.1.dylib"
    good = "/opt/homebrew/lib/libzstd.1.dylib"
    if not _Path(good).exists():
        return
    try:
        deps = _subprocess.check_output(["otool", "-L", str(lib)], text=True)
    except (OSError, _subprocess.CalledProcessError):
        return
    if bad not in deps:
        return  # already patched
    _subprocess.run(
        ["install_name_tool", "-change", bad, good, str(lib)],
        check=False,
        stdout=_subprocess.DEVNULL,
        stderr=_subprocess.DEVNULL,
    )


_patch_libcling_zstd()
