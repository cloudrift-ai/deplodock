"""Materialize ``goldens/*.yaml`` into the ``golden`` table of a :class:`SearchDB`.

The YAML files in ``goldens/`` are the source of truth; this module makes them
queryable alongside tuned ``perf`` rows in the same SQLite database. Pure
function of the YAML on disk — re-running ``load_goldens_into(db)`` replaces
each row with the latest YAML state (idempotent under no-change, picks up
adds/edits/deletes otherwise — by default deletions are reflected via
``reset=True``)."""

from __future__ import annotations

import logging
from pathlib import Path

from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.publish.goldens import GoldenConfig, MatmulGoldenConfig, load_goldens

logger = logging.getLogger(__name__)


def _payload_for(c: GoldenConfig) -> dict:
    """Render kind-specific extra fields as a payload dict — kept opaque so
    new golden kinds can carry their own shape fields without schema growth."""
    if isinstance(c, MatmulGoldenConfig):
        return {"M": c.M, "N": c.N, "K": c.K, "dtype": c.dtype}
    raise TypeError(f"unknown golden type: {type(c).__name__}")


def _kind_for(c: GoldenConfig) -> str:
    if isinstance(c, MatmulGoldenConfig):
        return "matmul"
    raise TypeError(f"unknown golden type: {type(c).__name__}")


def load_goldens_into(db: SearchDB, *, directory: Path | None = None, reset: bool = True) -> int:
    """Read every YAML golden under ``directory`` and upsert into ``db.golden``.

    With ``reset=True`` (default), the ``golden`` table is cleared first so a
    deleted YAML entry disappears from the DB. With ``reset=False``, this is a
    pure upsert — useful when several callers contribute to the same DB.

    Returns the number of rows written."""
    configs = load_goldens(directory)
    if reset:
        db._conn.execute("DELETE FROM golden")  # noqa: SLF001 — loader is part of the same package
    for c in configs:
        db.record_golden(
            name=c.name,
            kind=_kind_for(c),
            gpu_name=c.gpu_name,
            compute_cap=c.compute_cap,
            knobs=dict(c.knobs),
            deplodock_us=c.deplodock_us,
            cublas_us=c.cublas_us,
            payload=_payload_for(c),
        )
    logger.info("Loaded %d golden config(s) into DB", len(configs))
    return len(configs)
