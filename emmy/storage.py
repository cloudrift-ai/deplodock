"""Generic JSON file I/O with numpy-array support and atomic writes.

A small utility shared by anything that needs to persist a plain-data artifact
(today: the learned tuning prior in :mod:`emmy.compiler.pipeline.search.prior`).
:func:`write_json` / :func:`read_json` round-trip nested dicts/lists of JSON
scalars **plus** ``numpy`` arrays — each array is stored as a base64 blob with
its dtype + shape (``{"__ndarray__": <b64>, "dtype": ..., "shape": ...}``), so
model weights ride inside an otherwise human-diffable JSON file rather than an
opaque pickle. Writes go through a temp file + atomic rename so a crashed write
never truncates the existing artifact.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

_NDARRAY_TAG = "__ndarray__"


def _default(o: Any) -> Any:
    """``json.dump`` fallback: encode numpy arrays as base64 + dtype/shape and
    collapse numpy scalars to native Python numbers."""
    if isinstance(o, np.ndarray):
        arr = np.ascontiguousarray(o)
        return {_NDARRAY_TAG: base64.b64encode(arr.tobytes()).decode("ascii"), "dtype": str(arr.dtype), "shape": list(arr.shape)}
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON-serializable")


def _object_hook(d: dict) -> Any:
    """Rebuild a numpy array from a ``_default``-encoded dict (pass through
    every other object)."""
    if _NDARRAY_TAG in d:
        return np.frombuffer(base64.b64decode(d[_NDARRAY_TAG]), dtype=d["dtype"]).reshape(d["shape"])
    return d


def read_json(path: Path | str) -> Any | None:
    """Parsed JSON (numpy arrays decoded), or ``None`` for a missing / corrupt /
    unreadable file."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(), object_hook=_object_hook)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def write_json(path: Path | str, obj: Any) -> None:
    """Write ``obj`` to ``path`` as JSON (numpy arrays base64-encoded), creating
    parent dirs, via a temp file + atomic rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=_default))
    tmp.replace(p)
