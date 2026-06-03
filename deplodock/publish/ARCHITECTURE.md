# `deplodock/publish/` — publish & distribute tuning data

This package turns the per-developer SQLite tuning DB into a shared artifact: per-GPU snapshots uploaded as GitHub
release assets, a small JSON manifest pointing at them, and customer-side auto-pull keyed to the installed deplodock
version + detected local GPU.

## Why

Two problems the rest of the codebase had:

1. **Customers running deplodock locally on a supported GPU got no benefit from tuning anyone else has already done.**
   Each user re-discovered the same configs.
2. **Goldens** (the curated wins) lived in a Python file under `compiler/pipeline/search/`, while tuned `perf` rows
   lived in SQLite. Any pipeline that needed both (prior training, regression validation, "best known config" queries)
   had to read two formats.

This package solves both. Goldens migrated to `goldens/*.yaml` (reviewable, mergeable in PRs) and are materialized
into the same `SearchDB.golden` table that tuned data sits beside, so a single SQL query answers "best known
config." The publish/pull flow uploads/attaches the SQLite DB itself.

## Layout

```
deplodock/publish/
    goldens.py          # GoldenConfig / MatmulGoldenConfig + YAML load/dump
    goldens_to_db.py    # materialize goldens/*.yaml → SearchDB.golden
    manifest.py         # Manifest + DbEntry schema, cache_key_version gate, sha256
    release.py          # subprocess wrappers around `gh release upload / download`
    cache.py            # ~/.cache/deplodock/published/<release>/ layout + TTL + verify
    autofetch.py        # installed version → manifest → matching DB → ATTACH
    publish_flow.py     # `tune --publish` + `tune-data publish` orchestration

goldens/                # YAML source of truth (one file per kernel kind)
    matmul.yaml

deplodock/commands/
    tune_data.py        # `deplodock tune-data {status,pull,publish,clean-cache}`
```

`deplodock/commands/tune.py` adds three flags (`--publish`, `--pull`, `--offline`, `--contributor`) and a call into
`publish_flow.publish_local_db` after `--bench`. `compile.py` and `run.py` call
`autofetch.ensure_published_attached(db)` once after opening the local DB.

## Goldens flow

1. **Source of truth**: `goldens/<kind>.yaml` — flat list of entries. Schema mirrors the
   `MatmulGoldenConfig` dataclass: `name`, `kind`, `M/N/K`, `dtype`, `gpu_name`, `compute_cap`, `knobs`,
   `deplodock_us`, `cublas_us`. `golden = cublas_us / deplodock_us >= 0.95` (derived, not stored).
2. **Regeneration**: `scripts/find_golden_configs.py` autotunes each shape, re-benches the winner at -O3 against
   cuBLAS, and dumps to YAML via `publish.goldens.dump_goldens()`.
3. **Materialization**: `publish.goldens_to_db.load_goldens_into(db)` reads every YAML and upserts rows into the
   `golden` table on the SQLite DB. Idempotent. Pure function of the YAML on disk.
4. **Query**: callers use `db.iter_goldens(kind=, gpu_name=)` (filterable) or raw SQL against the table.

## Publish flow (`tune --publish` / `tune-data publish`)

```
local autotune.db ──zstd──► tune-<gpu>-<sha>.db.zst
                       └──► tune-<gpu>-latest.db.zst       (rolling alias)
                              │
                              ▼ gh release upload --clobber
                       GH release v<ver>
                              │
                              ▼ gh release download tune-data-index.json
                       upsert DbEntry for this gpu
                              │
                              ▼ gh release upload --clobber tune-data-index.json
```

- **Per-GPU sharding**: one DB per GPU. Avoids cross-GPU merge races, lets `--pull` grab only what's needed.
- **Snapshot + rolling latest**: every publish writes a SHA-stamped snapshot (append-only history) and overwrites
  the rolling `-latest` asset. Manifests record the snapshot URL (content-addressed by SHA, stable cache key).
- **Release tag = installed version** (`v<pip version>`), overridable via `DEPLODOCK_RELEASE`. A
  `tune-data-<sha>` style tag is intentional future room; today one tag per pip release is enough.

## Manifest format

```json
{
  "release": "v0.5.0",
  "git_sha": "25ece07e",
  "cache_key_version": 1,
  "updated_at": "2026-06-03T14:21:00Z",
  "redirect": null,
  "dbs": [
    {
      "gpu": "NVIDIA H100 80GB HBM3",
      "driver_major": 560,
      "cuda_major": 12,
      "url": "https://github.com/.../releases/download/v0.5.0/tune-h100-80gb-hbm3-25ece07e.db.zst",
      "sha256": "…",
      "size_bytes": 4192384,
      "published_at": "2026-06-02T09:11:00Z",
      "contributor": "@slonegg"
    }
  ]
}
```

- `cache_key_version` — bumped in `manifest.CURRENT_CACHE_KEY_VERSION` whenever the SQLite schema or `op_cache_key`
  derivation changes in a way that makes older DBs unsafe. Customers refuse mismatches with `CacheKeyMismatch` and
  fall back to local tuning.
- `redirect` — when set, the manifest is a thin alias to another release's manifest. Used for patch releases that did
  not move the topology — no need to re-publish DBs.
- `dbs[].url` points at the snapshot, not the rolling-latest. Once the manifest is published, the URL is permanent.

## Autofetch (`compile` / `run` / `tune --pull`)

```
ensure_published_attached(db, gpu_name=None):
    installed_release() → tag (e.g. v0.5.0)
    cache.manifest_fresh(tag)?  ──yes──► load cached
                                ──no───► gh release download tune-data-index.json
    manifest.redirect?  ──yes──► recurse
    check_compatible(manifest)  (cache_key_version)
    manifest.find(gpu_name)?  ──no──► log "no published data for X", return None
    cache.db_path_for(release, entry) exists & sha256 ok? ──yes──► attach, return
                                                          ──no───► gh release download → decompress → verify → attach
```

**Fail-open contract.** Any of: no `gh` on PATH, manifest fetch fails, no matching GPU, `DEPLODOCK_NO_FETCH=1` →
return None, log one INFO line, proceed with the local DB (and rule defaults if nothing tuned). The only hard error
is a sha256 mismatch on a downloaded DB — that's data corruption.

**Local writes shadow published.** Tune's `record_perf` only ever touches the main DB; the published DB attaches as
`pub` and is read via UNION ALL by the lookup paths. A customer who re-tunes locally and beats published gets their
config — published is a floor, not a ceiling.

## Env vars

| Var | Default | Purpose |
|---|---|---|
| `DEPLODOCK_NO_FETCH` | unset | When `1`/`true`/etc., skip every network call in autofetch |
| `DEPLODOCK_TUNE_DATA_REPO` | `slonegg/deplodock` | GH repo hosting tune-data releases (forks set their own) |
| `DEPLODOCK_RELEASE` | `v<installed-version>` | Override the release tag autofetch resolves against |
| `DEPLODOCK_PUBLISHED_CACHE` | `~/.cache/deplodock/published` | Local cache root for manifests + pulled DBs |
| `DEPLODOCK_GOLDENS_DIR` | `<repo>/goldens` | Where `load_goldens()` reads YAML from |
| `DEPLODOCK_GH_BIN` | `gh` on PATH | Override the `gh` binary (used by tests to inject a fake) |

## Testing

`tests/test_publish.py` covers manifest round-trip, version gating, cache TTL, sha256 verification, autofetch
fail-open behavior (fake `gh` binary returning failure), and the offline-with-cached-manifest path. The goldens path
is covered by `tests/compiler/test_golden_configs.py` (YAML round-trip, DB materialization idempotency, attach).
