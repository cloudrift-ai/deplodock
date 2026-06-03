"""High-level publish orchestration: compress local DB → upload → update manifest.

Called by ``deplodock tune --publish`` and ``deplodock tune-data publish``.

Flow:
1. zstd-compress the local autotune DB to a per-(gpu, sha) snapshot name.
2. Fetch the current manifest (creating a fresh one when absent).
3. Upload the snapshot + the rolling ``-latest`` alias as release assets.
4. Upsert the manifest entry and re-upload the manifest itself.

All four steps go through the ``gh`` CLI via :mod:`.release`. Failures bubble
up — publish is an active user action, not autofetch's fail-open path.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from deplodock.publish import cache, release
from deplodock.publish.autofetch import _repo, installed_release
from deplodock.publish.manifest import (
    CURRENT_CACHE_KEY_VERSION,
    DbEntry,
    Manifest,
    gpu_slug,
    sha256_file,
)

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    """Short HEAD SHA (used in snapshot filenames). ``"dev"`` when not in a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "dev"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "dev"


def _compress_zst(src: Path, dest: Path) -> None:
    """zstd-compress ``src`` to ``dest``. Prefers the ``zstandard`` Python
    binding; falls back to the ``zstd`` CLI."""
    try:
        import zstandard  # type: ignore[import-not-found]

        with open(src, "rb") as fi, open(dest, "wb") as fo:
            zstandard.ZstdCompressor(level=19).copy_stream(fi, fo)
        return
    except ImportError:
        pass
    subprocess.run(["zstd", "-19", "-f", str(src), "-o", str(dest)], check=True, capture_output=True)


def _ensure_release_exists(repo: str, tag: str) -> None:
    """Best-effort: create the release if it doesn't exist (a no-op when it does)."""
    try:
        release._run(["release", "view", tag, "--repo", repo])  # noqa: SLF001
        return
    except subprocess.CalledProcessError:
        pass
    try:
        release._run(  # noqa: SLF001
            [
                "release",
                "create",
                tag,
                "--repo",
                repo,
                "--title",
                f"tune-data {tag}",
                "--notes",
                "Auto-generated tune-data release; see tune-data-index.json.",
                "--prerelease",
            ]
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("could not create release %s (%s); assuming it exists", tag, exc)


def publish_local_db(
    db_path: Path,
    gpu_name: str,
    *,
    driver_major: int = 0,
    cuda_major: int = 0,
    contributor: str = "",
    repo: str | None = None,
    release_tag: str | None = None,
    workdir: Path | None = None,
) -> DbEntry:
    """Compress ``db_path``, upload as snapshot + rolling-latest, update the
    manifest. Returns the :class:`DbEntry` that was written into the manifest.

    Pre-conditions: ``gh`` CLI installed and authed; user has write access on
    ``repo`` (default: ``DEPLODOCK_TUNE_DATA_REPO`` env var, then
    ``slonegg/deplodock``). The DB must exist."""
    if not db_path.exists():
        raise FileNotFoundError(f"local tune DB does not exist: {db_path}")
    repo = repo or _repo()
    tag = release_tag or installed_release()
    workdir = workdir or db_path.parent
    workdir.mkdir(parents=True, exist_ok=True)

    sha = _git_sha()
    slug = gpu_slug(gpu_name)
    snapshot_name = f"tune-{slug}-{sha}.db.zst"
    latest_name = f"tune-{slug}-latest.db.zst"
    snapshot = workdir / snapshot_name
    latest = workdir / latest_name
    _compress_zst(db_path, snapshot)
    # Rolling "latest" is the same bytes — copy via shutil rather than re-compressing.
    import shutil

    shutil.copyfile(snapshot, latest)

    _ensure_release_exists(repo, tag)
    release.upload_asset(repo, tag, snapshot, clobber=True)
    release.upload_asset(repo, tag, latest, clobber=True)

    # Manifest upsert. Fetch the latest if we can; otherwise start fresh.
    manifest_local = workdir / "tune-data-index.json"
    try:
        release.fetch_manifest(repo, tag, manifest_local)
        manifest = Manifest.load(manifest_local)
    except (subprocess.CalledProcessError, OSError):
        manifest = Manifest(release=tag, git_sha=sha, cache_key_version=CURRENT_CACHE_KEY_VERSION)

    # The URL the manifest records points at the snapshot — content-addressed
    # by SHA, so the cache key is stable across re-publishes. Customers
    # pulling against ``-latest`` use the rolling asset instead, but that's
    # the auto-fetch path's optimization, not the manifest's.
    asset_url = f"https://github.com/{repo}/releases/download/{tag}/{snapshot_name}"
    entry = DbEntry(
        gpu=gpu_name,
        driver_major=int(driver_major),
        cuda_major=int(cuda_major),
        url=asset_url,
        sha256=sha256_file(db_path),
        size_bytes=db_path.stat().st_size,
        published_at=datetime.now(UTC).isoformat(),
        contributor=contributor,
    )
    manifest.upsert(entry)
    manifest.save(manifest_local)
    release.upload_manifest(repo, tag, manifest_local)
    logger.info("published %s for %s @ %s", snapshot_name, gpu_name, tag)
    return entry


def pull_published_db(
    gpu_name: str,
    *,
    repo: str | None = None,
    release_tag: str | None = None,
) -> Path | None:
    """Manual counterpart of autofetch: pull the DB matching ``gpu_name`` at
    ``release_tag`` into the local cache. Returns the cached path, or ``None``
    when the manifest has no entry for that GPU.

    Used by ``deplodock tune --pull`` / ``deplodock tune-data pull`` (the
    autofetch path is non-interactive; this one prints/raises on failure)."""
    repo = repo or _repo()
    tag = release_tag or installed_release()
    manifest_local = cache.manifest_path(tag)
    release.fetch_manifest(repo, tag, manifest_local)
    manifest = Manifest.load(manifest_local)
    if manifest.redirect:
        manifest_local = cache.manifest_path(manifest.redirect)
        release.fetch_manifest(repo, manifest.redirect, manifest_local)
        manifest = Manifest.load(manifest_local)
    entry = manifest.find(gpu_name)
    if entry is None:
        logger.info("manifest has no entry for %s at %s", gpu_name, tag)
        return None
    dest = cache.db_path_for(manifest.release, entry)
    asset_name = Path(urlparse(entry.url).path).name
    raw = dest.with_name(asset_name)
    release.download_asset(repo, manifest.release, asset_name, raw)
    if asset_name.endswith(".zst"):
        from deplodock.publish.autofetch import _decompress_zst

        _decompress_zst(raw, dest)
        raw.unlink()
    else:
        raw.replace(dest)
    cache.verify_db(dest, entry.sha256)
    logger.info("pulled %s → %s", asset_name, dest)
    return dest


def status_summary() -> dict:
    """Diagnostic payload for ``deplodock tune-data status``: installed
    version, detected GPU, resolved DB, cache path, verification state."""
    from deplodock.detect import detect_local_gpus

    tag = installed_release()
    try:
        gpu_name, gpu_count = detect_local_gpus()
    except RuntimeError as exc:
        gpu_name, gpu_count = f"<undetected: {exc}>", 0
    manifest_local = cache.manifest_path(tag)
    manifest_summary: dict | str
    if manifest_local.exists():
        try:
            m = Manifest.load(manifest_local)
            manifest_summary = {
                "release": m.release,
                "git_sha": m.git_sha,
                "cache_key_version": m.cache_key_version,
                "updated_at": m.updated_at,
                "n_dbs": len(m.dbs),
            }
        except (OSError, ValueError, KeyError) as exc:
            manifest_summary = f"<unreadable: {exc}>"
    else:
        manifest_summary = "<not cached>"
    return {
        "installed_release": tag,
        "detected_gpu": gpu_name,
        "gpu_count": gpu_count,
        "cache_root": str(cache.cache_root()),
        "manifest_path": str(manifest_local),
        "manifest": manifest_summary,
        "repo": _repo(),
    }


def status_text() -> str:
    """Human-readable string form of :func:`status_summary`."""
    s = status_summary()
    lines = [
        f"installed release: {s['installed_release']}",
        f"detected GPU:      {s['detected_gpu']} (count={s['gpu_count']})",
        f"repo:              {s['repo']}",
        f"cache root:        {s['cache_root']}",
        f"manifest path:     {s['manifest_path']}",
        f"manifest:          {json.dumps(s['manifest'], default=str)}",
    ]
    return "\n".join(lines)
