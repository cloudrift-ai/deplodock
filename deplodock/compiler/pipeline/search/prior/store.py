"""File-backed checkpoint for the global learned prior.

A single file (``config.prior_path()``) holds ``{regime_key: blob}`` — so
distinct hardware / nvcc-flag regimes coexist — where each ``blob`` is a prior's
own serialized state (:meth:`Prior.to_bytes`). Kept out of the tune DB so the
prior is a separate, easily-shippable artifact: ``tune`` writes it, ``compile`` /
``run`` read it.
"""

from __future__ import annotations

import pickle
from pathlib import Path


def load(path: Path | str, regime_key: str) -> bytes | None:
    """The checkpointed blob for ``regime_key``, or ``None`` (missing /
    unreadable file, or no entry for this regime)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        store = pickle.loads(p.read_bytes())
    except (pickle.UnpicklingError, EOFError, OSError):
        return None
    return store.get(regime_key) if isinstance(store, dict) else None


def save(path: Path | str, regime_key: str, blob: bytes) -> None:
    """Upsert ``regime_key → blob`` into the prior file (atomic temp+rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, bytes] = {}
    if p.exists():
        try:
            loaded = pickle.loads(p.read_bytes())
            if isinstance(loaded, dict):
                store = loaded
        except (pickle.UnpicklingError, EOFError, OSError):
            store = {}
    store[regime_key] = blob
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps(store))
    tmp.replace(p)
