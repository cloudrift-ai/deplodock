"""Thin wrapper around the GitHub ``gh`` CLI for release-asset I/O.

We piggy-back on ``gh`` (already a dev dependency in this repo) instead of
writing an HTTP client or wiring an auth flow: it handles tokens, retries,
range-resumed downloads, and per-release asset listing for free. All calls are
``subprocess.run`` against the user's installed ``gh`` — no extra Python deps.

Used by:
- ``tune --publish`` → :func:`upload_asset`, :func:`upload_manifest`
- ``tune --pull`` / autofetch → :func:`download_asset`, :func:`fetch_manifest`
- ``tune-data status`` → :func:`fetch_manifest`
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GhUnavailable(RuntimeError):
    """``gh`` is not on PATH (or the user has no GitHub auth set up)."""


def _gh() -> str:
    """Resolve the ``gh`` binary; raise :class:`GhUnavailable` when missing.

    Overridable via ``DEPLODOCK_GH_BIN`` (tests inject a fake)."""
    override = os.environ.get("DEPLODOCK_GH_BIN")
    if override:
        return override
    found = shutil.which("gh")
    if not found:
        raise GhUnavailable("`gh` CLI not found on PATH; install from https://cli.github.com/")
    return found


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    """Run ``gh`` with the given args. Returns stdout. Raises CalledProcessError
    on failure (callers catch and translate to fail-open behavior)."""
    cmd = [_gh(), *args]
    logger.debug("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
    return result.stdout


def upload_asset(repo: str, tag: str, path: Path, *, clobber: bool = True) -> None:
    """Upload ``path`` as an asset on the release tagged ``tag`` in ``repo``
    (``owner/name``). ``clobber=True`` overwrites an existing asset of the same
    name — used for the rolling ``-latest`` snapshot and the manifest."""
    args = ["release", "upload", tag, str(path), "--repo", repo]
    if clobber:
        args.append("--clobber")
    _run(args)
    logger.info("uploaded %s → %s @ %s", path.name, repo, tag)


def upload_manifest(repo: str, tag: str, manifest_path: Path) -> None:
    """Convenience: upload the ``tune-data-index.json`` asset (always clobbers)."""
    upload_asset(repo, tag, manifest_path, clobber=True)


def download_asset(repo: str, tag: str, asset_name: str, dest: Path) -> None:
    """Download a single asset from a release into ``dest`` (a file path)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "release",
        "download",
        tag,
        "--repo",
        repo,
        "--pattern",
        asset_name,
        "--output",
        str(dest),
        "--clobber",
    ]
    _run(args)
    logger.info("downloaded %s → %s", asset_name, dest)


def fetch_manifest(repo: str, tag: str, dest: Path) -> Path:
    """Download ``tune-data-index.json`` from the release tagged ``tag``.

    Returns the path written. Caller :func:`Manifest.load` on it."""
    download_asset(repo, tag, "tune-data-index.json", dest)
    return dest
