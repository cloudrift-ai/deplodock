"""CLI tests for ``deplodock run`` — accuracy check + ``--bench`` table.

Accuracy failures (``max_diff >= 1.0``) make ``deplodock run`` exit
non-zero, so ``rc == 0`` is the accuracy assertion.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_run_no_code_errors(run_cli):
    rc, stdout, stderr = run_cli("run")
    assert rc != 0
    # argparse complains about the missing required ``--code`` flag.
    assert "code" in (stdout + stderr).lower() or "ir" in (stdout + stderr).lower()


def test_run_code_and_ir_mutually_exclusive(run_cli, tmp_path):
    fake_ir = tmp_path / "fake.json"
    fake_ir.write_text("{}")
    rc, stdout, stderr = run_cli("run", "--code", "torch.zeros(4)", "--ir", str(fake_ir))
    assert rc != 0
    assert "mutually exclusive" in (stdout + stderr).lower()


@requires_cuda
def test_run_code_rmsnorm_accuracy(run_cli):
    rc, _, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(64)(torch.randn(1,8,64))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_accuracy(run_cli):
    rc, _, stderr = run_cli("run", "--code", "torch.matmul(torch.randn(16,32), torch.randn(32,16))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_rmsnorm_blockify(run_cli):
    """Wide hidden + ≥16 rows triggers blockify on the row axis. Regression
    test: cooperative load step must match the actual thread count, not
    BLOCK_SIZE=256, or staged-weight indices get skipped."""
    rc, _, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_softmax_blockify(run_cli):
    rc, _, stderr = run_cli("run", "--code", "torch.nn.functional.softmax(torch.randn(32,2048), dim=-1)")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_blockify(run_cli):
    rc, _, stderr = run_cli("run", "--code", "torch.matmul(torch.randn(64,128), torch.randn(128,64))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_linear_blockify(run_cli):
    rc, _, stderr = run_cli("run", "--code", "torch.nn.Linear(2048, 2048, bias=False)(torch.randn(1, 32, 2048))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_k_chunked(run_cli):
    """Matmul with K large enough to exercise the K-chunked SGEMM path
    (BK=64). Regression: the K_o outer loop is syntactically a free
    Loop (no immediate Accum) but the Init for the running accumulator
    must still land at the surrounding Tile body so it persists across
    K_o iterations."""
    rc, _, stderr = run_cli("run", "--code", "torch.matmul(torch.randn(128, 2048), torch.randn(2048, 128))")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_sdpa_k_chunked(run_cli):
    """SDPA: the per-output free loop (head_dim) wraps a reduce loop +
    a Write — its body has a Write so it is *not* a reduce-passthrough
    and the per-output accumulator must reset per iteration. Pairs
    with the matmul case above to cover both branches of the recursive
    reduce-crossing rule."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,2,32,64), torch.randn(1,2,32,64), torch.randn(1,2,32,64))",
    )
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_bench_prints_table(run_cli):
    rc, stdout, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "--bench", "--warmup", "2", "--iters", "5")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Eager PyTorch" in log
    assert "Deplodock" in log
    assert "vs Eager" in log


# --- --ir mode -------------------------------------------------------------


def _dump_ir(project_root: Path, code: str, stage: str, out_dir: Path) -> Path:
    """Run ``deplodock compile --code <code> --dump-dir`` and return the path
    to the JSON dump for the requested stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        [sys.executable, "-m", "deplodock.deplodock", "compile", "--code", code, "--dump-dir", str(out_dir), "--ir", stage],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    assert rc.returncode == 0, rc.stderr
    # Stage dumps are named ``NN_<stage>.json`` — pick the matching one.
    candidates = sorted(out_dir.glob(f"*_{stage.replace('/', '_')}*.json"))
    candidates = [p for p in candidates if not p.name.endswith(".rules.json") and not p.name.endswith(".kernels.json")]
    assert candidates, f"no dump matched stage={stage} in {out_dir}: {list(out_dir.iterdir())}"
    return candidates[-1]


@requires_cuda
def test_run_ir_loop_stage(run_cli, project_root, tmp_path):
    """``deplodock run --ir <loop.json>`` loads loop IR and runs the
    remaining tile / kernel / cuda passes, executes with random inputs."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "loop", tmp_path)
    rc, stdout, stderr = run_cli("run", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded loop IR" in log
    assert "lowering/tile" in log


@requires_cuda
def test_run_ir_tile_stage(run_cli, project_root, tmp_path):
    """Tile-IR JSON loads and runs only the kernel + cuda tail."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "tile", tmp_path)
    rc, stdout, stderr = run_cli("run", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded tile IR" in log
    assert "lowering/kernel" in log
    # tile-stage already ran lowering/tile, so it should NOT be in the tail list.
    assert "running tail passes: ['lowering/kernel'" in log


@requires_cuda
def test_run_ir_kernel_stage(run_cli, project_root, tmp_path):
    """Kernel-IR JSON loads and runs only the cuda tail."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "kernel", tmp_path)
    rc, stdout, stderr = run_cli("run", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded kernel IR" in log
    assert "running tail passes: ['lowering/cuda']" in log


@requires_cuda
def test_run_ir_cuda_stage_no_tail(run_cli, project_root, tmp_path):
    """Already-lowered cuda IR has no remaining passes."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "cuda", tmp_path)
    rc, stdout, stderr = run_cli("run", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded cuda IR" in log
    assert "running tail passes: (none)" in log


@requires_cuda
def test_run_ir_bench(run_cli, project_root, tmp_path):
    """``--bench`` with ``--ir`` prints just the deplodock latency row
    (no eager reference is available for partial-IR mode)."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "tile", tmp_path)
    rc, stdout, stderr = run_cli("run", "--ir", str(ir_path), "--bench", "--warmup", "2", "--iters", "5")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Deplodock" in log
    assert "Latency (us)" in log


@requires_cuda
def test_run_ir_seed_reproducible(run_cli, project_root, tmp_path):
    """Two runs with the same seed produce the same output mean."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "tile", tmp_path)
    runs = []
    for _ in range(2):
        rc, stdout, stderr = run_cli("run", "--ir", str(ir_path), "--seed", "42")
        assert rc == 0
        # Output line: "Output rms_norm: shape=... finite=True mean=<value>"
        for line in (stdout + stderr).splitlines():
            if "mean=" in line:
                runs.append(line.split("mean=")[-1].strip())
                break
    assert len(runs) == 2 and runs[0] == runs[1], runs


def test_run_ir_invalid_json(run_cli, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc, _, stderr = run_cli("run", "--ir", str(bad))
    assert rc != 0


def test_run_ir_missing_file(run_cli, tmp_path):
    rc, _, stderr = run_cli("run", "--ir", str(tmp_path / "does_not_exist.json"))
    assert rc != 0
