"""Tests for create_run_dir() naming format."""

import re

from deplodock.benchmark import create_run_dir


def test_run_dir_created(tmp_path):
    run_dir = create_run_dir(str(tmp_path))
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path


def test_run_dir_name_format(tmp_path):
    run_dir = create_run_dir(str(tmp_path))
    # Expected: YYYY-MM-DD_HH-MM-SS_<8 hex chars>
    pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[0-9a-f]{8}$"
    assert re.match(pattern, run_dir.name)


def test_run_dir_creates_parent(tmp_path):
    nested = tmp_path / "a" / "b"
    run_dir = create_run_dir(str(nested))
    assert run_dir.exists()
