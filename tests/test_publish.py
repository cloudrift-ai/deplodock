"""Tests for the publish package: manifest schema, cache layout, autofetch flow.

These are hermetic: no actual ``gh`` calls. A fake ``gh`` binary on
``$PATH`` (injected via ``DEPLODOCK_GH_BIN``) records its arguments to a
sidecar log so we can assert which release/asset names were requested.
"""

from __future__ import annotations

import os
import stat

import pytest

from deplodock.publish import autofetch, cache
from deplodock.publish.manifest import (
    CURRENT_CACHE_KEY_VERSION,
    CacheKeyMismatch,
    DbEntry,
    Manifest,
    check_compatible,
    gpu_slug,
    sha256_file,
)

# ---------------------------------------------------------------------------
# manifest.py — schema round-trip + version gating
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    m = Manifest(release="v0.5.0", git_sha="abc1234")
    m.upsert(
        DbEntry(
            gpu="NVIDIA H100 80GB HBM3",
            driver_major=560,
            cuda_major=12,
            url="https://example.invalid/foo.db.zst",
            sha256="0" * 64,
            size_bytes=1024,
            published_at="2026-06-02T09:11:00Z",
            contributor="@slonegg",
        )
    )
    p = tmp_path / "tune-data-index.json"
    m.save(p)
    loaded = Manifest.load(p)
    assert loaded.release == "v0.5.0"
    assert loaded.cache_key_version == CURRENT_CACHE_KEY_VERSION
    assert loaded.find("NVIDIA H100 80GB HBM3") is not None
    assert loaded.find("RTX 5090") is None


def test_upsert_replaces_same_gpu():
    m = Manifest(release="v0.5.0", git_sha="abc")
    e1 = DbEntry(gpu="H100", driver_major=0, cuda_major=0, url="u1", sha256="a" * 64, size_bytes=1, published_at="t")
    e2 = DbEntry(gpu="H100", driver_major=0, cuda_major=0, url="u2", sha256="b" * 64, size_bytes=2, published_at="t")
    m.upsert(e1)
    m.upsert(e2)
    assert len(m.dbs) == 1
    assert m.dbs[0].url == "u2"


def test_check_compatible_rejects_mismatch():
    m = Manifest(release="v0.5.0", git_sha="abc", cache_key_version=CURRENT_CACHE_KEY_VERSION + 99)
    with pytest.raises(CacheKeyMismatch):
        check_compatible(m)


def test_gpu_slug_is_stable():
    assert gpu_slug("NVIDIA H100 80GB HBM3") == "h100-80gb-hbm3"
    assert gpu_slug("NVIDIA GeForce RTX 5090") == "geforce-rtx-5090"


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    # Known sha256 of "hello".
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


# ---------------------------------------------------------------------------
# cache.py — paths, TTL, verify
# ---------------------------------------------------------------------------


def test_cache_paths_isolate_under_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPLODOCK_PUBLISHED_CACHE", str(tmp_path))
    assert cache.cache_root() == tmp_path
    assert cache.manifest_path("v0.5.0") == tmp_path / "v0.5.0" / "tune-data-index.json"


def test_manifest_fresh_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPLODOCK_PUBLISHED_CACHE", str(tmp_path))
    assert not cache.manifest_fresh("v0.5.0")
    p = cache.manifest_path("v0.5.0")
    p.write_text("{}")
    assert cache.manifest_fresh("v0.5.0")
    # Backdate the file past the TTL.
    old = p.stat().st_mtime - 48 * 3600
    os.utime(p, (old, old))
    assert not cache.manifest_fresh("v0.5.0")


def test_verify_db_detects_corruption(tmp_path):
    p = tmp_path / "x.db"
    p.write_bytes(b"hello")
    cache.verify_db(p, sha256_file(p))  # passes
    with pytest.raises(ValueError):
        cache.verify_db(p, "0" * 64)


# ---------------------------------------------------------------------------
# autofetch.py — fail-open with a fake `gh` CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gh(tmp_path, monkeypatch):
    """Write a tiny shell script that records its argv to a log file and
    exits 1 (simulates a manifest miss). Returns the log path so tests can
    assert which gh calls happened."""
    log = tmp_path / "gh.log"
    script = tmp_path / "gh"
    script.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{log}"\nexit 1\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("DEPLODOCK_GH_BIN", str(script))
    monkeypatch.setenv("DEPLODOCK_PUBLISHED_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DEPLODOCK_RELEASE", "v0.0.0-test")
    return log


def test_autofetch_fails_open_on_gh_error(fake_gh, monkeypatch):
    from deplodock.compiler.pipeline.search import SearchDB

    db = SearchDB()
    # Returns None; no exception even though `gh` exits 1.
    result = autofetch.ensure_published_attached(db, gpu_name="NVIDIA H100 80GB HBM3")
    assert result is None


def test_autofetch_no_fetch_skips_network(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPLODOCK_NO_FETCH", "1")
    monkeypatch.setenv("DEPLODOCK_PUBLISHED_CACHE", str(tmp_path))
    monkeypatch.setenv("DEPLODOCK_RELEASE", "v0.0.0-test")
    from deplodock.compiler.pipeline.search import SearchDB

    db = SearchDB()
    result = autofetch.ensure_published_attached(db, gpu_name="NVIDIA H100 80GB HBM3")
    assert result is None


def test_autofetch_uses_cached_manifest_when_no_fetch(tmp_path, monkeypatch):
    """With NO_FETCH set, an already-cached manifest still gets honored —
    autofetch should attach a matching DB if its sha matches the local file."""
    monkeypatch.setenv("DEPLODOCK_NO_FETCH", "1")
    monkeypatch.setenv("DEPLODOCK_PUBLISHED_CACHE", str(tmp_path))
    monkeypatch.setenv("DEPLODOCK_RELEASE", "v0.0.0-test")

    # Build a published DB + matching manifest by hand.
    from deplodock.compiler.pipeline.search import SearchDB
    from deplodock.publish.goldens_to_db import load_goldens_into

    rel_dir = tmp_path / "v0.0.0-test"
    rel_dir.mkdir()
    pub_db = rel_dir / "h100.db"
    pdb = SearchDB(path=pub_db)
    load_goldens_into(pdb)
    pdb.close()
    entry = DbEntry(
        gpu="NVIDIA H100 80GB HBM3",
        driver_major=0,
        cuda_major=0,
        url=f"https://example.invalid/release/v0.0.0-test/{pub_db.name}",
        sha256=sha256_file(pub_db),
        size_bytes=pub_db.stat().st_size,
        published_at="2026-06-02T09:11:00Z",
    )
    m = Manifest(release="v0.0.0-test", git_sha="abc")
    m.upsert(entry)
    m.save(cache.manifest_path("v0.0.0-test"))

    db = SearchDB()
    path = autofetch.ensure_published_attached(db, gpu_name="NVIDIA H100 80GB HBM3")
    assert path == pub_db
    # The attach exposes the published DB's goldens.
    n = db._conn.execute("SELECT COUNT(*) FROM pub.golden").fetchone()[0]  # noqa: SLF001
    assert n > 0


def test_autofetch_unknown_gpu_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPLODOCK_NO_FETCH", "1")
    monkeypatch.setenv("DEPLODOCK_PUBLISHED_CACHE", str(tmp_path))
    monkeypatch.setenv("DEPLODOCK_RELEASE", "v0.0.0-test")
    m = Manifest(release="v0.0.0-test", git_sha="abc")
    cache.release_dir("v0.0.0-test")
    m.save(cache.manifest_path("v0.0.0-test"))

    from deplodock.compiler.pipeline.search import SearchDB

    db = SearchDB()
    assert autofetch.ensure_published_attached(db, gpu_name="NVIDIA RTX 9999") is None
