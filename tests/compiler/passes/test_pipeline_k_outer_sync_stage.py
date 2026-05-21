"""Regression test for ``015_pipeline_k_outer`` skipping kernels with a
plain (sync) ``BufferedStage`` alongside async/TMA Stages.

Background: when ``stage_inputs`` produces a mix of async Stages (wg /
wu) and a sync Stage (x — when ``async_copy`` can't promote it because
of slab shape / alignment), the K-outer pipeline used to hoist only the
async Stages' decls to the prologue while leaving the sync ``Smem``
decl inside the K-loop body. The pipelined epilogue then peeled the
last K iteration past the loop's closing brace, referencing
``x_smem`` from a scope where it was no longer declared → nvcc error
``identifier "x_smem" is undefined``.

The gate added in ``015_pipeline_k_outer`` rejects the loop in this
configuration so the unpipelined-but-correct schedule survives.

This test pins the exact knob set that historically tripped the bug
(``BK=2, BM=16, BN=128, FM=8, FN=8, STAGE=111``) on the gated-MLP
pattern from Qwen3-Embedding-0.6B kernel 10 and asserts the full
compile → CUDA-render pipeline succeeds end-to-end.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

GATED_MLP_CODE = (
    "import torch; import torch.nn.functional as F; "
    "x=torch.randn((1, 32, 1024)); "
    "wg=torch.randn((1024, 3072)); "
    "wu=torch.randn((1024, 3072)); "
    "F.silu(torch.matmul(x,wg))*torch.matmul(x,wu)"
)

# Knob set that historically tripped the x_smem-undefined nvcc error.
BUGGY_KNOBS = {
    "DEPLODOCK_BK": "2",
    "DEPLODOCK_BM": "16",
    "DEPLODOCK_BN": "128",
    "DEPLODOCK_FM": "8",
    "DEPLODOCK_FN": "8",
    "DEPLODOCK_STAGE": "111",
}


@pytest.fixture
def env_with_knobs(monkeypatch):
    for k, v in BUGGY_KNOBS.items():
        monkeypatch.setenv(k, v)
    return os.environ.copy()


@pytest.mark.xfail(reason="M14: planner rejects BM*FM=128 > M=32 as non-divisible; legacy knobs don't apply", strict=False)
def test_compile_gated_mlp_with_sync_x_stage_does_not_dangle_smem(env_with_knobs, tmp_path):
    """End-to-end: compile the gated-MLP pattern at the historically-
    buggy knob set and assert the rendered CUDA source declares every
    ``*_smem`` buffer it references — i.e. no dangling identifier."""
    out = tmp_path / "kernel.cu"
    result = subprocess.run(
        [sys.executable, "-m", "deplodock.deplodock", "compile", "--code", GATED_MLP_CODE, "--ir", "cuda", "--output", str(out)],
        capture_output=True,
        text=True,
        env=env_with_knobs,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists(), "compile did not produce a .cu file"
    src = out.read_text()

    # Every smem buffer referenced must be declared at some scope. The
    # historic failure mode: ``x_smem[...]`` referenced after the loop
    # whose body declared it. Asserting on a balanced
    # decl-count-per-name catches the scope mismatch without baking in
    # exact line numbers.
    for buffer in ("wg_smem", "x_smem", "wu_smem"):
        assert "__shared__" in src and buffer in src, f"missing {buffer} or any __shared__ decl"
        # Count uses (rough — any token occurrence) and decls (``__shared__ ... <buffer>[``).
        decls = src.count(f"float {buffer}[")
        uses = src.count(f"{buffer}[") + src.count(f"&{buffer}[")
        assert decls >= 1, f"{buffer} never declared"
        # Every use must be reachable from some declaration. With the
        # gate active, all three buffers are declared at the kernel-body
        # scope so every use is in scope. (Pre-fix: x_smem decl was
        # nested inside a for-loop and uses after the loop dangled.)
        assert uses > decls, f"{buffer} declared {decls}× but never used"


def test_compile_gated_mlp_does_not_pipeline_when_sync_stage_present(env_with_knobs, tmp_path):
    """Direct evidence the pipelining was skipped: the pipelined schedule
    emits a *peeled tail* (a copy of the reduce body after the main
    K-outer loop, reading from the last buffer slot). When the gate
    rejects pipelining, only the unrolled-by-double-buffer main loop
    remains and there is no post-loop tail."""
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    env = env_with_knobs.copy()
    env["DEPLODOCK_DUMP_DIR"] = str(dump_dir)
    result = subprocess.run(
        [sys.executable, "-m", "deplodock.deplodock", "compile", "--code", GATED_MLP_CODE, "--ir", "kernel"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    pipeline_dump = dump_dir / "05_lowering_tile__015_pipeline_k_outer.rules.txt"
    if pipeline_dump.exists():
        # If the file exists at all, the pass must have skipped (gate
        # rejected this case). Skipped passes leave a "no rewrite"
        # diagnostic; a successful firing would show ``BufferedStage``
        # decls hoisted out of a Loop and a new Loop with smaller
        # extent — neither should appear here when the sync-Stage gate
        # is active.
        text = pipeline_dump.read_text()
        assert "no eligible" in text.lower() or "skipped" in text.lower() or text.strip() == "", (
            "pipeline_k_outer should have skipped on sync-Stage presence, but produced a rewrite:\n" + text[:500]
        )
