"""CLI tests for ``deplodock compile`` argument handling and ``--code``."""


def test_compile_code_torch_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))", "--ir", "torch")
    assert rc == 0, f"stderr: {stderr}"
    assert "rms_norm" in stdout
    assert "(1, 32, 2048)" in stdout


def test_compile_code_tensor_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.Linear(3, 2, bias=False)(torch.randn(4, 3))", "--ir", "tensor")
    assert rc == 0, f"stderr: {stderr}"
    assert "multiply(" in stdout
    assert "sum(" in stdout


def test_compile_code_loop_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.ReLU()(torch.randn(8))", "--ir", "loop")
    assert rc == 0, f"stderr: {stderr}"
    assert "===" in stdout
    assert "load " in stdout
    assert " = relu(" in stdout


def test_compile_code_saves_fused_graph(run_cli, tmp_path):
    """Default ``handle_compile`` path: no ``--ir``, writes fused graph to disk.

    Covers the cold-import path that skips ``_handle_compile_inspect`` and
    exercises the full decomposition→optimization→fusion pipeline (where
    ``LoopOp.__post_init__`` runs normalize_body → simplify_body on the
    first LoopOp construction).
    """
    out = tmp_path / "fused.txt"
    rc, stdout, stderr = run_cli(
        "compile",
        "--code",
        "torch.nn.RMSNorm(64)(torch.randn(1, 8, 64))",
        "--output",
        str(out),
    )
    assert rc == 0, f"stderr: {stderr}"
    assert out.exists()
    text = out.read_text()
    # Fused text dump — at least one LoopOp compute node and the RMSNorm output.
    assert "loop(" in text
    assert "outputs:" in text


def test_compile_code_functional_silu(run_cli):
    """Functional expressions (non-Module callables) should trace too."""
    rc, stdout, stderr = run_cli("compile", "--code", "F.silu(torch.randn(1,32,128))", "--ir", "tensor")
    assert rc == 0, f"stderr: {stderr}"
    assert "exp(" in stdout
    assert "reciprocal(" in stdout


def test_compile_code_functional_softmax_bakes_kwargs(run_cli):
    """Non-tensor kwargs (e.g. dim=-1) must be baked, not turned into graph inputs."""
    rc, stdout, stderr = run_cli("compile", "--code", "F.softmax(torch.randn(1,32,128), dim=-1)", "--ir", "tensor")
    assert rc == 0, f"stderr: {stderr}"
    assert "1 inputs" in stdout
    assert "maximum(" in stdout and "sum(" in stdout


def test_compile_passes_shorthand(run_cli, tmp_path):
    """'dolft' should expand to decomposition/optimization/lifting/fusion/lowering/tile."""
    out = tmp_path / "out.txt"
    rc, stdout, stderr = run_cli("compile", "-c", "F.relu(torch.randn(8))", "--passes", "dolft", "-o", str(out), "-v")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    for name in ("frontend/decomposition", "frontend/optimization", "loop/lifting", "loop/fusion", "lowering/tile"):
        assert name in log, f"missing pass {name!r} in log"
    assert "lowering/cuda" not in log


def test_compile_no_input_errors(run_cli):
    rc, stdout, stderr = run_cli("compile")
    assert rc != 0
    assert "required" in stdout + stderr


def test_compile_code_and_input_mutually_exclusive(run_cli, tmp_path):
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}")
    rc, stdout, stderr = run_cli("compile", "--code", "x", str(dummy))
    assert rc != 0
    assert "mutually exclusive" in stdout + stderr
