"""CLI tests for ``deplodock compile`` argument handling and ``--code``."""


def test_compile_code_torch_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))", "--ir", "torch")
    assert rc == 0, f"stderr: {stderr}"
    assert "rms_norm" in stdout
    assert "(1, 32, 2048)" in stdout


def test_compile_code_tensor_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.Linear(3, 2, bias=False)(torch.randn(4, 3))", "--ir", "tensor")
    assert rc == 0, f"stderr: {stderr}"
    assert "mul(" in stdout
    assert "sum(" in stdout


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
