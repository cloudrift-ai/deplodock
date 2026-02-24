"""Tests for compute_code_hash() determinism."""

from deplodock.benchmark import compute_code_hash


def test_code_hash_deterministic():
    """Calling compute_code_hash twice returns the same value."""
    h1 = compute_code_hash()
    h2 = compute_code_hash()
    assert h1 == h2


def test_code_hash_is_hex_string():
    h = compute_code_hash()
    assert len(h) == 64  # SHA256 hex digest
    assert all(c in "0123456789abcdef" for c in h)
