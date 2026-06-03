"""Customer-side auto-pull of published tune data.

Resolves: ``installed deplodock version`` → ``manifest URL`` → ``matching DB
for detected GPU`` → ``ATTACH read-only alongside local SearchDB``.

Designed to fail open. Network / manifest / GPU mismatches log one line and
return; ``compile`` / ``run`` proceed with the local DB (and untuned
defaults). The only way to hard-fail is a sha256 mismatch on a downloaded DB
— that's data corruption, not a missing artifact.

Opt-out: ``DEPLODOCK_NO_FETCH=1`` (or ``--offline``) skips every network call.
Repo override: ``DEPLODOCK_TUNE_DATA_REPO`` (default ``slonegg/deplodock``).
"""

from __future__ import annotations

import logging
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.publish import cache, release
from deplodock.publish.manifest import CacheKeyMismatch, Manifest, check_compatible

logger = logging.getLogger(__name__)

_DEFAULT_REPO = "slonegg/deplodock"


def _no_fetch() -> bool:
    return os.environ.get("DEPLODOCK_NO_FETCH", "").strip().lower() in {"1", "true", "yes", "on"}


def _repo() -> str:
    return os.environ.get("DEPLODOCK_TUNE_DATA_REPO") or _DEFAULT_REPO


def installed_release() -> str:
    """The release tag the installed ``deplodock`` wheel corresponds to.

    ``DEPLODOCK_RELEASE`` override > ``deplodock`` PyPI version → ``v<version>``
    tag > ``"dev"`` (source checkout, no install). The env-var override wins so
    contributors and tests can pin a specific release tag independently of what
    the local wheel reports."""
    override = os.environ.get("DEPLODOCK_RELEASE")
    if override:
        return override
    try:
        return f"v{version('deplodock')}"
    except PackageNotFoundError:
        return "dev"


def _ensure_manifest(release_tag: str) -> Manifest | None:
    """Fetch (or return cached) manifest for ``release_tag``. ``None`` on any
    fetch failure (logged at debug — autofetch is best-effort)."""
    path = cache.manifest_path(release_tag)
    if cache.manifest_fresh(release_tag):
        try:
            return Manifest.load(path)
        except (OSError, ValueError) as exc:
            logger.debug("cached manifest unreadable (%s); refetching", exc)
    if _no_fetch():
        logger.debug("DEPLODOCK_NO_FETCH set; not fetching manifest")
        # Fall back to a stale cached copy if present (better than nothing).
        if path.exists():
            try:
                return Manifest.load(path)
            except (OSError, ValueError):
                return None
        return None
    try:
        release.fetch_manifest(_repo(), release_tag, path)
    except (subprocess.CalledProcessError, release.GhUnavailable) as exc:
        logger.debug("manifest fetch failed for %s: %s", release_tag, exc)
        return None
    return Manifest.load(path)


def _resolve_db_for(manifest: Manifest, gpu_name: str) -> Path | None:
    """Locate the DB for ``gpu_name``: download to cache if missing, sha256-verify."""
    entry = manifest.find(gpu_name)
    if entry is None:
        logger.info("no published tune data for %s at %s; using defaults", gpu_name, manifest.release)
        return None
    dest = cache.db_path_for(manifest.release, entry)
    if dest.exists():
        try:
            cache.verify_db(dest, entry.sha256)
            return dest
        except ValueError as exc:
            logger.warning("cached DB rejected (%s); refetching", exc)
            dest.unlink()
    if _no_fetch():
        return None
    try:
        from urllib.parse import urlparse

        asset_name = Path(urlparse(entry.url).path).name
        # Download into the same name first, then verify; supports compressed assets.
        raw = dest.with_name(asset_name)
        release.download_asset(_repo(), manifest.release, asset_name, raw)
        if asset_name.endswith(".zst"):
            _decompress_zst(raw, dest)
            raw.unlink()
        else:
            raw.replace(dest)
        cache.verify_db(dest, entry.sha256)
    except (subprocess.CalledProcessError, release.GhUnavailable, ValueError, OSError) as exc:
        # sha256 mismatch is data corruption — surface, don't silently ignore.
        if isinstance(exc, ValueError):
            raise
        logger.info("could not download %s for %s: %s", entry.url, gpu_name, exc)
        return None
    return dest


def _decompress_zst(src: Path, dest: Path) -> None:
    """Decompress a zstd asset. Tries ``zstandard`` Python module, then ``zstd`` CLI."""
    try:
        import zstandard  # type: ignore[import-not-found]

        with open(src, "rb") as fi, open(dest, "wb") as fo:
            zstandard.ZstdDecompressor().copy_stream(fi, fo)
        return
    except ImportError:
        pass
    subprocess.run(["zstd", "-d", "-f", str(src), "-o", str(dest)], check=True, capture_output=True)


def ensure_published_attached(db: SearchDB, *, gpu_name: str | None = None) -> Path | None:
    """Idempotent helper for ``compile`` / ``run`` / ``tune`` to attach the
    published DB matching this process's installed version + detected GPU.

    Returns the attached DB path on success, ``None`` when nothing was
    attached (no matching manifest entry, offline, ``gh`` unavailable, …).
    Never raises on the "missing" path — fail-open is the contract."""
    if _no_fetch() and not cache.manifest_path(installed_release()).exists():
        logger.debug("DEPLODOCK_NO_FETCH and no cached manifest; skipping autofetch")
        return None
    if gpu_name is None:
        try:
            from deplodock.detect import detect_local_gpus

            gpu_name, _ = detect_local_gpus()
        except (RuntimeError, OSError) as exc:
            logger.debug("GPU detect failed (%s); skipping autofetch", exc)
            return None
    tag = installed_release()
    manifest = _ensure_manifest(tag)
    if manifest is None:
        return None
    # Follow one level of redirect (a patch-release that aliases an older manifest).
    if manifest.redirect:
        target = _ensure_manifest(manifest.redirect)
        if target is None:
            return None
        manifest = target
    try:
        check_compatible(manifest)
    except CacheKeyMismatch as exc:
        logger.info("%s; using local tuning only", exc)
        return None
    db_path = _resolve_db_for(manifest, gpu_name)
    if db_path is None:
        return None
    db.attach_published(db_path, alias="pub")
    logger.info("attached published tune data: %s", db_path)
    return db_path
