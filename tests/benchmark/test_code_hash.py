"""Tests for BenchmarkTask.compute_code_hash() determinism."""

from deplodock.planner import BenchmarkTask


def test_code_hash_deterministic():
    """Calling compute_code_hash twice returns the same value."""
    h1 = BenchmarkTask.compute_code_hash()
    h2 = BenchmarkTask.compute_code_hash()
    assert h1 == h2


def test_code_hash_is_hex_string():
    h = BenchmarkTask.compute_code_hash()
    assert len(h) == 64  # SHA256 hex digest
    assert all(c in "0123456789abcdef" for c in h)
