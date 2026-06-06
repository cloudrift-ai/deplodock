"""CLI tests for ``deplodock run`` — accuracy check + ``--bench`` table.

Accuracy failures (``max_diff >= 1.0``) make ``deplodock run`` exit
non-zero, so ``rc == 0`` is the accuracy assertion.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch  # used by test_bind_inputs_preserves_int_dtype

from ..conftest import requires_cuda


def _randn(shape: str, dtype, scale: float | None = None) -> str:
    """Build a ``torch.randn(...)`` snippet for the given dtype.

    ``shape`` is a comma-joined dim list as it would appear inside the
    parens. fp16 inputs are scaled down (default 0.1) so reductions
    stay in fp16's representable range; fp32 inputs use the raw value.
    """
    if dtype.name == "f16":
        s = 0.1 if scale is None else scale
        return f"(torch.randn({shape}, dtype=torch.float16) * {s})"
    return f"torch.randn({shape})"


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


def test_run_input_and_code_mutually_exclusive(run_cli):
    rc, stdout, stderr = run_cli("run", "some/model", "--code", "torch.zeros(4)")
    assert rc != 0
    assert "mutually exclusive" in (stdout + stderr).lower()


def test_pinned_knobs_sets_and_restores_env(monkeypatch):
    """``_pinned_knobs`` pins ``DEPLODOCK_<KNOB>`` for the block, then restores the
    prior environment — removing keys that were unset, restoring preexisting ones
    (the golden-bench A/B relies on this to compile a pinned variant cleanly)."""
    import os

    from deplodock.commands.run import _pinned_knobs

    monkeypatch.delenv("DEPLODOCK_BM", raising=False)
    monkeypatch.setenv("DEPLODOCK_BN", "preexisting")
    with _pinned_knobs({"BM": 8, "BN": 32, "WARP_SPECIALIZE": False}):
        assert os.environ["DEPLODOCK_BM"] == "8"
        assert os.environ["DEPLODOCK_BN"] == "32"
        assert os.environ["DEPLODOCK_WARP_SPECIALIZE"] == "False"
    assert "DEPLODOCK_BM" not in os.environ  # was unset → removed
    assert os.environ["DEPLODOCK_BN"] == "preexisting"  # restored
    assert "DEPLODOCK_WARP_SPECIALIZE" not in os.environ


@requires_cuda
def test_run_golden_bench_shows_benched_golden_row(run_cli):
    """``run --golden NAME --bench`` compiles + benches the recorded golden (knobs
    pinned) and prints it as a ``(golden NAME)``-tagged row in the kernel table."""
    rc, stdout, stderr = run_cli("run", "--golden", "square.512", "--bench")
    assert rc == 0, f"stderr: {stderr}"
    assert "(golden square.512)" in stdout, stdout


@requires_cuda
def test_run_code_rmsnorm_accuracy(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.nn.RMSNorm(64)({_randn('1,8,64', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_rmsnorm_via_pow_neg_half(run_cli):
    """Gemma-style RMSNorm normalization uses ``torch.pow(ms, -0.5)`` (not
    ``rsqrt``); the exponent arrives as a broadcast constant. Guards the
    ``030_pow`` regression where every ``pow`` was squared — here that would
    compute ``x * (mean+eps)²`` and fail the eager comparison."""
    code = "x = torch.randn(2,64,256); torch.mul(x, torch.pow(torch.mean(torch.pow(x,2),-1,keepdim=True)+1e-6, -0.5))"
    rc, _, stderr = run_cli("run", "--code", code)
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_accuracy(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.matmul({_randn('16,32', dtype)}, {_randn('32,16', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_target_override(run_cli):
    """``--target sm_80`` gates lowering to the cp.async path (no TMA); the kernel still runs
    on the live device and must match eager, so ``rc == 0`` is the accuracy assertion."""
    rc, _, stderr = run_cli("run", "--code", "torch.matmul(torch.randn(256, 256), torch.randn(256, 256))", "--target", "sm_80")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_rmsnorm_blockify(run_cli, dtype):
    """Wide hidden + ≥16 rows triggers blockify on the row axis. Regression
    test: cooperative load step must match the actual thread count, not
    BLOCK_SIZE=256, or staged-weight indices get skipped."""
    rc, _, stderr = run_cli("run", "--code", f"torch.nn.RMSNorm(2048)({_randn('1,32,2048', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_softmax_blockify(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.nn.functional.softmax({_randn('32,2048', dtype)}, dim=-1)")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_matmul_blockify(run_cli, dtype):
    rc, _, stderr = run_cli("run", "--code", f"torch.matmul({_randn('64,128', dtype)}, {_randn('128,64', dtype)})")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
@pytest.mark.parametrize("fk", [2, 4, 8])
@pytest.mark.parametrize("br", [None, 1])
def test_run_code_rmsnorm_fk_accuracy(run_cli, monkeypatch, fk, br):
    """FK register-tiles the reduce axis into ``fk`` independent accumulators +
    a cross-accumulator fold (``plans/fk-register-tile-reductions.md``). Pin FK
    (and optionally BR=1 for the pure-serial scope) and confirm the folded
    reduction still matches eager — ``rc == 0`` is the accuracy assertion."""
    monkeypatch.setenv("DEPLODOCK_FK", str(fk))
    if br is not None:
        monkeypatch.setenv("DEPLODOCK_BR", str(br))
    rc, _, stderr = run_cli("run", "--code", "torch.nn.RMSNorm(2048)(torch.randn(4,32,2048))")
    assert rc == 0, f"stderr: {stderr}"


def test_compile_fp16_matmul_window_emits_half2(run_cli, monkeypatch):
    """Structural guard: a pinned FK window must actually rewrite the K loop into
    the half2 pack + ``__half2`` accumulate + widen-flush (catches a silent
    015_pack_fk_window regression that would fall back to fp32 accumulate). No
    CUDA device needed — just inspects the generated source.

    Pins a FULL clean (no-overhang) knob set + ``--target sm_90`` so the variant
    is fully determined: the greedy pick keys off ``compute_capability`` via
    ``score_tile_geometry``, so without the target override a GPU-less CI runner
    resolves a different capability, picks a masked non-window variant, and the
    FK-only pin falls back to FK=1 (no window). The full pin + fixed target make
    the single FK=bk window variant the only candidate on any runner."""
    monkeypatch.setenv("DEPLODOCK_KNOBS", "MMA=0,BN=16,BM=16,FM=1,FN=1,BK=4,SPLITK=1,FK=4")
    rc, stdout, stderr = run_cli(
        "compile",
        "--code",
        "torch.randn(256,256,dtype=torch.float16) @ torch.randn(256,256,dtype=torch.float16)",
        "--ir",
        "cuda",
        "--target",
        "sm_90",
    )
    assert rc == 0, f"stderr: {stderr}"
    assert "__halves2half2" in stdout, "no __half2 pack — FK window did not fire"
    assert "__low2half" in stdout and "__high2half" in stdout, "no widen+flush of the half2 window"


@requires_cuda
@pytest.mark.parametrize("fk", [2, 4, 8])
def test_run_code_fp16_matmul_window_accuracy(run_cli, monkeypatch, fk):
    """fp16 scalar matmul half2 accumulation window (``plans/fk-half2-fp16-matmul.md``):
    pin MMA off + an even FK window and confirm the windowed half2 accumulate +
    fp32 flush matches eager within fp16 tolerance — ``rc == 0`` asserts it."""
    monkeypatch.setenv("DEPLODOCK_MMA", "0")
    monkeypatch.setenv("DEPLODOCK_FK", str(fk))
    rc, _, stderr = run_cli("run", "--code", "torch.randn(256,256,dtype=torch.float16) @ torch.randn(256,256,dtype=torch.float16)")
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
@pytest.mark.parametrize("br", [None, 1])
def test_run_code_softmax_fk_accuracy(run_cli, monkeypatch, br):
    """Softmax carries both a ``max`` and a ``sum`` reduce, so FK exercises the
    ``fmaxf`` and ``+`` cross-accumulator folds together. ``rc == 0`` asserts
    the pinned-FK kernel matches eager."""
    monkeypatch.setenv("DEPLODOCK_FK", "4")
    if br is not None:
        monkeypatch.setenv("DEPLODOCK_BR", str(br))
    rc, _, stderr = run_cli("run", "--code", "torch.softmax(torch.randn(4,32,2048), dim=-1)")
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
def test_run_code_sdpa_tinyllama_per_head(run_cli):
    """Per-head SDPA at TinyLlama-block-seq=512 dimensions, mirroring the
    ``k_scaled_dot_product_attention_reduce_reduce.json`` kernel in
    ``experiments/kernel_dataset/tinyllama_block_seq512`` (M=512, K=512,
    N=64). The K=512 reduction does not fit a full smem slab once
    register-tile + double-buffer apply, so this exercises the chunked
    blockify + staging path on the per-head shape."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,1,512,64), torch.randn(1,1,512,64), torch.randn(1,1,512,64))",
    )
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_sdpa_seq1024_dynamic_smem(run_cli):
    """SDPA at seq_len=1024, 32 heads: the Q·Kᵀ kernel needs ~50 KB of
    smem after register-tile + double-buffer + bank-pad — well past the
    48 KB static cap. Pins the dynamic-smem pool path: kernel must
    declare ``extern __shared__ ... _smem_pool[]``, the launch must pass
    ``shared_mem=smem_bytes``, and ``cudaFuncSetAttribute(MaxDynamicShared
    MemorySize)`` must opt this kernel into the device's larger dynamic
    allowance."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,32,1024,64), torch.randn(1,32,1024,64), torch.randn(1,32,1024,64))",
    )
    assert rc == 0, f"stderr: {stderr}"


@requires_cuda
def test_run_code_sdpa_tinyllama_full(run_cli):
    """Full multi-head TinyLlama-block-seq=512 SDPA (1 batch × 32 heads ×
    512 × 64). Regression: the blockify + staging interaction
    over-allocated per-block smem (PTXAS rejected the kernel with
    ``uses too much shared data``, 0xc600 > 0xc000 = 49152 cap)."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1,32,512,64), torch.randn(1,32,512,64), torch.randn(1,32,512,64))",
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
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded loop IR" in log
    assert "lowering/tile" in log


@requires_cuda
def test_run_positional_json_like_ir(run_cli, project_root, tmp_path):
    """A ``.json`` passed as the positional input takes the same IR path as ``--ir``."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "loop", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    assert "Loaded loop IR" in (stdout + stderr)


@requires_cuda
def test_run_ir_tile_stage(run_cli, project_root, tmp_path):
    """Tile-IR JSON loads and runs only the kernel + cuda tail."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "tile", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
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
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    assert "Loaded kernel IR" in log
    assert "running tail passes: ['lowering/cuda']" in log


@requires_cuda
def test_run_ir_cuda_stage_no_tail(run_cli, project_root, tmp_path):
    """Already-lowered cuda IR has no remaining passes."""
    ir_path = _dump_ir(project_root, "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "cuda", tmp_path)
    rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path))
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
        rc, stdout, stderr = run_cli("run", "-v", "--ir", str(ir_path), "--seed", "42")
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


@requires_cuda
def test_run_code_dynamic_seq_len(run_cli):
    """``run --code --dynamic seq_len@x:1`` traces with torch.export's
    dynamic_shapes, compiles to a single ``int seq_len``-arg kernel,
    runs it at the canonical shape, and checks accuracy against eager."""
    rc, _, stderr = run_cli(
        "run",
        "--code",
        "torch.nn.RMSNorm(64)(torch.randn(1,8,64))",
        "--dynamic",
        "seq_len@x:1",
    )
    assert rc == 0, f"stderr: {stderr}"


# ---------------------------------------------------------------------------
# _bind_inputs: integer-dtype preservation
# ---------------------------------------------------------------------------


def test_bind_inputs_preserves_int_dtype():
    """``_bind_inputs`` must cast each torch input to the numpy dtype
    that matches the graph's declared ``Tensor.dtype`` — not blanket-
    cast to float32 as it did before integer placeholders (``input_ids``,
    ``position_ids``) became part of whole-model traces. A float32 cast
    of int64 indices would silently corrupt the embedding-lookup path."""
    import numpy as np

    from deplodock.commands.run import _bind_inputs
    from deplodock.compiler import dtype as dt
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("input_ids", (1, 8), dt.I64), node_id="input_ids")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("position_ids", (1, 8), dt.I32), node_id="position_ids")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("activations", (1, 8, 16), dt.F32), node_id="activations")
    g.inputs = ["input_ids", "position_ids", "activations"]

    class _EmptyModule:
        def named_parameters(self, remove_duplicate=True):
            return iter(())

        def named_buffers(self, remove_duplicate=True):
            return iter(())

    input_ids = torch.zeros((1, 8), dtype=torch.long)
    position_ids = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    activations = torch.randn(1, 8, 16)

    bound = _bind_inputs(g, _EmptyModule(), (input_ids, position_ids, activations), {})

    assert bound["input_ids"].dtype == np.int64
    assert bound["position_ids"].dtype == np.int32
    assert bound["activations"].dtype == np.float32
    # Values must round-trip without precision loss.
    np.testing.assert_array_equal(bound["position_ids"], np.arange(8, dtype=np.int32)[None, :])
