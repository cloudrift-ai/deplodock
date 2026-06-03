"""Local cache for published tune-data DBs and manifests.

Layout::

    ~/.cache/deplodock/published/
        <release>/
            tune-data-index.json    # cached manifest (with mtime-driven TTL)
            <gpu-slug>.db           # the per-GPU DB matching this customer

All downloads are sha256-verified against the manifest entry, so a corrupt /
partial fetch is detected before ATTACH. The manifest TTL is 24h on the
default path — long enough to dodge network calls in inner dev loops, short
enough to pick up post-release contributor publishes.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from deplodock.publish.manifest import DbEntry, Manifest, sha256_file

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 24 * 3600


def cache_root() -> Path:
    """``DEPLODOCK_PUBLISHED_CACHE`` or ``~/.cache/deplodock/published``."""
    override = os.environ.get("DEPLODOCK_PUBLISHED_CACHE")
    return Path(override) if override else Path.home() / ".cache" / "deplodock" / "published"


def release_dir(release: str) -> Path:
    """Per-release cache directory: ``<cache_root>/<release>/``."""
    d = cache_root() / release
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(release: str) -> Path:
    return release_dir(release) / "tune-data-index.json"


def manifest_fresh(release: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> bool:
    """``True`` when the cached manifest exists and is younger than ``ttl_seconds``."""
    p = manifest_path(release)
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < ttl_seconds


def db_path_for(release: str, entry: DbEntry) -> Path:
    """Local path the manifest's ``entry`` would be cached at."""
    # The asset URL's tail name is the source-of-truth filename; cache mirrors
    # it minus the .zst suffix once decompressed.
    from urllib.parse import urlparse

    name = Path(urlparse(entry.url).path).name
    if name.endswith(".zst"):
        name = name[: -len(".zst")]
    return release_dir(release) / name


def verify_db(path: Path, expected_sha256: str) -> None:
    """Raise :class:`ValueError` when ``path``'s sha256 doesn't match. Used
    after a pull to detect corruption / wrong asset / tampering."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"sha256 mismatch for {path.name}: expected {expected_sha256[:12]}…, got {actual[:12]}…")


def clear_release(release: str) -> None:
    """Wipe one release's cache directory (used by ``tune-data clean-cache``)."""
    import shutil

    d = cache_root() / release
    if d.exists():
        shutil.rmtree(d)
        logger.info("cleared cache for release %s", release)


def write_cached_manifest(manifest: Manifest) -> Path:
    """Persist ``manifest`` to its release-specific cached path."""
    p = manifest_path(manifest.release)
    manifest.save(p)
    return p
