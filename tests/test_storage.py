"""Unit tests for :mod:`deplodock.storage` — JSON file I/O with numpy support."""

from __future__ import annotations

import numpy as np

from deplodock import storage


def test_read_missing_returns_none(tmp_path):
    assert storage.read_json(tmp_path / "nope.json") is None


def test_roundtrips_numpy_arrays_and_scalars(tmp_path):
    """Arrays (any dtype/shape) survive base64 round-trip; numpy scalars collapse
    to native numbers; nested dicts/lists pass through."""
    path = tmp_path / "obj.json"
    obj = {
        "vec": np.arange(6, dtype=np.float64).reshape(2, 3),
        "ints": np.array([1, 2, 3], dtype=np.int32),
        "scalar": np.float64(3.5),
        "meta": {"cols": ["BM", "BN"], "rows": [[{"BM": 16}, -1.2]]},
    }
    storage.write_json(path, obj)
    got = storage.read_json(path)
    assert np.array_equal(got["vec"], obj["vec"]) and got["vec"].dtype == np.float64
    assert np.array_equal(got["ints"], obj["ints"]) and got["ints"].dtype == np.int32
    assert got["scalar"] == 3.5
    assert got["meta"] == obj["meta"]


def test_read_corrupt_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert storage.read_json(path) is None


def test_write_is_atomic_no_tmp_left(tmp_path):
    path = tmp_path / "sub" / "x.json"  # parent created on write
    storage.write_json(path, {"a": 1})
    assert path.exists()
    assert not (path.parent / "x.json.tmp").exists()
