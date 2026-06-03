"""Publish & distribute tuning data.

Goldens + tuned configs are persisted in a uniform :class:`SearchDB` schema
(``source = 'golden' | 'tuned'``). Tuned DBs are uploaded as per-GPU GitHub
release assets keyed by ``(release, gpu)``; a manifest (a small JSON release
asset) lists what's available. Customers running locally auto-fetch the
manifest, pull the DB matching their detected GPU, and ATTACH it read-only
alongside their writeable local DB.

Submodules:
- :mod:`.goldens` — :class:`GoldenConfig` dataclasses + YAML I/O.
- :mod:`.goldens_to_db` — materialize goldens into ``SearchDB.perf``.
- :mod:`.manifest` — manifest JSON schema, load/save/validate.
- :mod:`.release` — thin wrapper around the ``gh`` CLI for asset upload/download.
- :mod:`.cache` — local cache for published DBs (sha256-verified).
- :mod:`.autofetch` — resolve installed version → manifest → matching DB, ATTACH.
"""
