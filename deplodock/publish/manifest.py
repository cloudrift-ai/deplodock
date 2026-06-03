"""Manifest schema for published tune-data releases.

A *manifest* (``tune-data-index.json``) is one JSON release asset that points
at every per-GPU tuning DB published under that release. It's the discovery
mechanism: a customer running ``deplodock`` resolves the installed version →
manifest URL → matching DB for their detected GPU.

Pinning rule: every DB referenced by a manifest must be op-cache-key-compatible
with the code in the release. We encode that via :attr:`Manifest.cache_key_version`
— readers refuse entries with a mismatched version, so a release that changed
the lowering / fork-tree topology cannot silently serve stale tuned configs.

A patch release that did NOT change the topology can ``redirect`` to a previous
release's manifest instead of cutting fresh DBs — one tiny manifest, no
re-publishing of bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Bump when the on-disk format of the published DB (perf schema, op_cache_key
# derivation, knob serialization, …) changes in a way that makes older DBs
# unsafe to ATTACH. Customers refuse mismatched DBs with a clear error and
# fall back to local tuning.
CURRENT_CACHE_KEY_VERSION = 1


@dataclass(frozen=True)
class DbEntry:
    """One per-GPU tuning-DB asset registered in the manifest."""

    gpu: str
    driver_major: int
    cuda_major: int
    url: str
    sha256: str
    size_bytes: int
    published_at: str
    contributor: str = ""


@dataclass
class Manifest:
    """The tune-data index for one release."""

    release: str
    git_sha: str
    cache_key_version: int = CURRENT_CACHE_KEY_VERSION
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    redirect: str | None = None
    dbs: list[DbEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "release": self.release,
            "git_sha": self.git_sha,
            "cache_key_version": self.cache_key_version,
            "updated_at": self.updated_at,
            "redirect": self.redirect,
            "dbs": [asdict(d) for d in self.dbs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        return cls(
            release=data["release"],
            git_sha=data["git_sha"],
            cache_key_version=int(data.get("cache_key_version", 0)),
            updated_at=data.get("updated_at", ""),
            redirect=data.get("redirect"),
            dbs=[DbEntry(**d) for d in data.get("dbs", [])],
        )

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")

    def find(self, gpu: str) -> DbEntry | None:
        """The DB entry whose ``gpu`` matches verbatim, or ``None``."""
        for d in self.dbs:
            if d.gpu == gpu:
                return d
        return None

    def upsert(self, entry: DbEntry) -> None:
        """Replace the entry for ``entry.gpu`` if present, else append."""
        for i, d in enumerate(self.dbs):
            if d.gpu == entry.gpu:
                self.dbs[i] = entry
                return
        self.dbs.append(entry)
        self.updated_at = datetime.now(UTC).isoformat()


class CacheKeyMismatch(RuntimeError):
    """Raised when a manifest's ``cache_key_version`` does not match the code's."""


def check_compatible(manifest: Manifest) -> None:
    """Raise :class:`CacheKeyMismatch` when ``manifest.cache_key_version`` does
    not match :data:`CURRENT_CACHE_KEY_VERSION`. Caller falls back to local
    tuning."""
    if manifest.cache_key_version != CURRENT_CACHE_KEY_VERSION:
        raise CacheKeyMismatch(
            f"manifest cache_key_version={manifest.cache_key_version} "
            f"does not match installed deplodock (expected {CURRENT_CACHE_KEY_VERSION})"
        )


def sha256_file(path: Path | str) -> str:
    """Hex sha256 of ``path``. Used to key the local cache + verify pulls."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_slug(gpu_name: str) -> str:
    """Filename-safe slug for a GPU name. Stable, deterministic, lowercase."""
    return gpu_name.lower().replace("nvidia ", "").replace("amd ", "").replace(" ", "-").replace("/", "-")
