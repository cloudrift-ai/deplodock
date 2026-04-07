"""Unit tests for the staging module (git ls-files + tar)."""

import asyncio
import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from deplodock.provisioning.staging import (
    build_stage_tar,
    enumerate_staged_files,
)


def _git(repo: Path, *args):
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": "/tmp",
            "PATH": "/usr/bin:/bin",
        },
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tracked.py").write_text("print('hi')\n")
    (tmp_path / "scripts" / "untracked.py").write_text("print('new')\n")
    (tmp_path / "scripts" / "ignored.log").write_text("noise\n")
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "other.txt").write_text("outside scope\n")
    _git(tmp_path, "add", "scripts/tracked.py", ".gitignore", "other.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_enumerate_empty_stage_paths(repo):
    files = asyncio.run(enumerate_staged_files(repo, []))
    assert files == []


def test_enumerate_includes_tracked_and_untracked_excludes_ignored(repo):
    files = asyncio.run(enumerate_staged_files(repo, ["scripts"]))
    assert "scripts/tracked.py" in files
    assert "scripts/untracked.py" in files
    assert "scripts/ignored.log" not in files
    # Other path-scoped: should not include other.txt.
    assert "other.txt" not in files


def test_enumerate_dot_includes_everything_tracked(repo):
    files = asyncio.run(enumerate_staged_files(repo, ["."]))
    assert "scripts/tracked.py" in files
    assert "other.txt" in files
    assert "scripts/untracked.py" in files
    assert "scripts/ignored.log" not in files


def test_build_stage_tar_roundtrip(repo):
    files = asyncio.run(enumerate_staged_files(repo, ["scripts"]))
    tar_bytes = build_stage_tar(repo, files)
    members = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for m in tar.getmembers():
            f = tar.extractfile(m)
            if f is not None:
                members[m.name] = f.read().decode()
    assert members["scripts/tracked.py"] == "print('hi')\n"
    assert members["scripts/untracked.py"] == "print('new')\n"
    assert "scripts/ignored.log" not in members
