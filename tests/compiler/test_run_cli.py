"""CLI tests for ``deplodock run`` — accuracy check + ``--bench`` table."""

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
    rc, stdout, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(64)(torch.randn(1,8,64))")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Accuracy vs eager" in log
    assert "PASS" in log


@requires_cuda
def test_run_code_matmul_accuracy(run_cli):
    rc, stdout, stderr = run_cli("run", "--code", "torch.matmul(torch.randn(16,32), torch.randn(32,16))")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Accuracy vs eager" in log
    assert "PASS" in log


@requires_cuda
def test_run_bench_prints_table(run_cli):
    rc, stdout, stderr = run_cli(
        "run", "--code", "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "--bench", "--warmup", "2", "--iters", "5"
    )
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Eager PyTorch" in log
    assert "Deplodock" in log
    assert "vs Eager" in log
