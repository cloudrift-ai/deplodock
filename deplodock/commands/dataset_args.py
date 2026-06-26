"""Shared ``--dataset`` CLI vocabulary for the commands that consume measurement
data (``eval`` for analysis; ``tune --dataset golden`` to tune every golden shape):
one place registers the source flags (``--dataset`` / ``--db`` / ``--kernel`` /
``--min-variants``), one helper publishes ``--prior``, and one guard fails loud on a
degenerate source. Handlers then build the actual
:class:`~deplodock.compiler.pipeline.search.data.Dataset` via its ``from_golden`` /
``from_db`` adapters — so every command selects a golden / DB dataset (and a subset)
through the same vocabulary instead of reimplementing golden filtering or opening
the DB by hand.

The source (golden vs db) is orthogonal to the analysis, but not every combination
is meaningful (a DB row has no cuBLAS reference; a golden has no kernel C identity).
:func:`require_source` lets a handler reject a degenerate combination with a
specific message rather than emit an empty table.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def add_dataset_args(parser, *, default: str, with_min_variants: bool = False) -> None:
    """Register the shared data-source flags on ``parser``. ``default`` is the
    natural source for the command (``golden`` for the golden-eval views, ``db`` for
    the regret analysis). ``with_min_variants`` adds the regret-only grouping
    threshold."""
    parser.add_argument(
        "--dataset",
        choices=["golden", "db", "nodes"],
        default=default,
        help="Measurement-data source: 'golden' (recorded golden configs), 'db' (tune DB perf rows), or 'nodes' "
        f"(tune DB search-tree node store, for `eval prior`). Default: {default}.",
    )
    parser.add_argument("--db", help="Tune DB path for --dataset db/nodes. Default: DEPLODOCK_TUNE_DB or ~/.cache/deplodock/autotune.db.")
    parser.add_argument("--kernel", help="Filter by name substring (golden name, or kernel C identifier for --dataset db).")
    if with_min_variants:
        parser.add_argument(
            "--min-variants", type=int, default=8, help="Skip kernels with fewer than this many measured variants (default: 8)."
        )


def require_source(args, allowed: set[str], msg: str) -> None:
    """Exit 2 with ``msg`` when ``--dataset`` isn't one this analysis supports."""
    if args.dataset not in allowed:
        logger.error(msg)
        sys.exit(2)


def resolve_prior_arg(args) -> None:
    """Publish ``--prior`` into the env (``DEPLODOCK_PRIOR_FILE``) so the prior loads
    from it — the single owner of the formerly-duplicated env poke."""
    if getattr(args, "prior", None):
        from deplodock import config  # noqa: PLC0415

        os.environ[config.PRIOR_FILE] = str(Path(args.prior).expanduser())
