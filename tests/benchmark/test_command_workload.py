"""Unit tests for the command workload module (pure functions)."""

import pytest

from deplodock.benchmark.command_workload import build_substitution_map, render_command
from deplodock.planner.variant import Variant


def _variant(params):
    return Variant(params=params)


def test_build_substitution_map_flattens_dot_keys():
    v = _variant({"deploy.gpu": "NVIDIA GeForce RTX 5090", "deploy.gpu_count": 1, "marker": "a"})
    subs = build_substitution_map(v, [0], repo_dir="/tmp/repo", task_dir="/tmp/task")
    assert subs["gpu"] == "NVIDIA GeForce RTX 5090"
    assert subs["gpu_count"] == "1"
    assert subs["marker"] == "a"
    assert subs["task_dir"] == "/tmp/task"
    assert subs["repo_dir"] == "/tmp/repo"
    assert subs["gpu_device_ids"] == "0"


def test_build_substitution_map_no_repo_dir():
    v = _variant({"deploy.gpu": "x", "deploy.gpu_count": 1})
    subs = build_substitution_map(v, [0, 1], repo_dir=None, task_dir="/t")
    assert "repo_dir" not in subs
    assert subs["gpu_device_ids"] == "0,1"


def test_build_substitution_map_leaf_conflict():
    v = _variant({"deploy.gpu": "x", "deploy.gpu_count": 1, "extra.gpu": "y"})
    with pytest.raises(ValueError, match="conflicting leaf name 'gpu'"):
        build_substitution_map(v, [0], repo_dir=None, task_dir="/t")


def test_render_command_basic():
    out = render_command("echo $marker > $task_dir/out", {"marker": "a", "task_dir": "/tmp"})
    assert out == "echo a > /tmp/out"


def test_render_command_missing_var():
    with pytest.raises(ValueError, match=r"undefined variable: \$missing"):
        render_command("echo $missing", {"task_dir": "/tmp"})


def test_render_command_passes_through_shell_metachars():
    """`$(...)`, `${VAR:-default}`, `$1`, and `$$` must survive rendering."""
    # Note: `$$` is Template's escape for a literal `$`, so it renders as `$`.
    out = render_command(
        'echo $marker $(hostname) ${OTHER:-x} "$1" $$',
        {"marker": "a"},
    )
    assert out == 'echo a $(hostname) ${OTHER:-x} "$1" $'


def test_render_command_repo_dir_unavailable():
    """When staging is empty, $repo_dir is omitted from subs and triggers a friendly error."""
    v = _variant({"deploy.gpu": "x", "deploy.gpu_count": 1})
    subs = build_substitution_map(v, [0], repo_dir=None, task_dir="/t")
    with pytest.raises(ValueError, match=r"undefined variable: \$repo_dir"):
        render_command("cd $repo_dir && make", subs)
