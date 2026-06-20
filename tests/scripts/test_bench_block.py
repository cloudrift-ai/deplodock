"""CLI smoke test for scripts/bench_block.py and deplodock bench orchestration.

The full benchmark requires CUDA + a real transformer download, which isn't
viable in CI. These tests cover the CLI surface: argument parsing, module
imports, and the bench-command orchestration in dry-run mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_bench_block_help():
    """`scripts/bench_block.py --help` imports the module and renders argparse."""
    result = subprocess.run(
        [sys.executable, "scripts/bench_block.py", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Transformer block benchmark" in result.stdout
    assert "--model" in result.stdout
    assert "--dtype" in result.stdout


def test_bench_block_imports_compiler():
    """The script's Deplodock path imports the current backend/Graph API.

    Guards against API drift: ``_bench_deplodock`` walks ``compiled.inputs`` and
    ``compiled.nodes`` — if either name changes on ``Graph``, this import will
    still pass but a later attribute access would fail at benchmark time. We
    sanity-check the attributes exist on a freshly-built empty Graph here.
    """
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.graph import Graph

    g = Graph()
    assert hasattr(g, "inputs")
    assert hasattr(g, "nodes")
    # backend.compile returns a Graph with these same attributes.
    assert hasattr(CudaBackend, "compile")
    assert hasattr(CudaBackend, "run")
    assert hasattr(CudaBackend, "benchmark_async")


def test_bench_dry_run_tinyllama_block(run_cli, tmp_path):
    """`deplodock bench experiments/tinyllama-block/ --dry-run --local` works end-to-end.

    The command-style recipe expands variants, stages files, and prints the
    ``scripts/bench_block.py`` invocation that would run on the target host —
    without actually executing it.
    """
    # Isolate config so the test doesn't depend on the user's config.yaml.
    config_path = tmp_path / "config.yaml"
    config_path.write_text("benchmark:\n  local_results_dir: " + str(tmp_path / "results") + "\n")
    recipe_dir = os.path.join(str(PROJECT_ROOT), "experiments", "tinyllama-block")

    rc, stdout, stderr = run_cli(
        "bench",
        recipe_dir,
        "--dry-run",
        "--local",
        "--config",
        str(config_path),
        "--filter",
        "dtype=fp32",
        "--filter",
        "seq_len=32",
    )
    assert rc == 0, f"stderr: {stderr}\nstdout: {stdout}"
    # The filter narrows to a single variant; the rendered command must reach the inner script.
    assert "scripts/bench_block.py" in stdout
    assert "--model TinyLlama/TinyLlama-1.1B-Chat-v1.0" in stdout
    assert "--seq-len 32" in stdout
    assert "--dtype fp32" in stdout
