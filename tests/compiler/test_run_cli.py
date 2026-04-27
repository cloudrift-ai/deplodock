"""CLI tests for ``deplodock run`` — accuracy check + ``--bench`` table.

Accuracy failures (``max_diff >= 1.0``) make ``deplodock run`` exit
non-zero, so ``rc == 0`` is the accuracy assertion.
"""

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_run_no_code_errors(run_cli):
    rc, stdout, stderr = run_cli("run")
    assert rc != 0
    # argparse complains about the missing required ``--code`` flag.
    assert "code" in (stdout + stderr).lower()


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
def test_run_bench_prints_table(run_cli):
    rc, stdout, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "--bench", "--warmup", "2", "--iters", "5")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Eager PyTorch" in log
    assert "Deplodock" in log
    assert "vs Eager" in log
