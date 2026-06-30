"""Unit + end-to-end tests for ``012_fuse_sibling_register_cells``.

The unit tests build a synthetic Tile-IR fragment that mirrors what
``010_split_register_axes`` leaves behind for a masked-overhang register-
blocked matmul with an N-invariant prologue (RMSNorm-style chain), then
run the pass and assert the output has the invariants hoisted above per-
cell sub-``Cond``s. The end-to-end test pins the autotune knob bundle from
the Qwen3-Embedding-0.6B ``k_linear_mean_reduce`` variant whose original
6+ s ``cicc -O1`` compile (``BK=64, BM=1, BN=64, FM=1, FN=64``) timed out
the autotune watchdog, and confirms the rendered kernel is now small
enough to compile within budget.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Write
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.tile.ir import SerialTile, TileOp

from ..conftest import requires_cuda

# Importlib gymnastics: the rule module name starts with a digit, which is
# not a valid Python identifier, so ``from … import …`` is rejected. Use
# ``importlib.import_module`` instead.
_fuse_mod = importlib.import_module("emmy.compiler.pipeline.passes.lowering.kernel.012_fuse_sibling_register_cells")


def _ax(name: str, extent: int) -> Axis:
    return Axis(name=name, extent=Dim(extent))


def _synthetic_replicated_body(n_cells: int) -> Body:
    """Mirror the post-``010`` shape: ``n_cells`` sibling ``Cond``s, each
    containing a nested ``SerialTile`` (a5, 4) → ``SerialTile`` (a4, 8) →
    a 4-stmt N-invariant prefix (``Load x``, ``Assign v5``, ``Load wn``,
    ``Assign v6``) followed by the per-cell weight Load + multiply +
    Accum, then a per-cell ``Write`` after the loop nest. The invariant
    prefix is byte-for-byte identical across cells; only the per-cell
    tail differs (different output-N offset + different Accum target)."""
    a5 = _ax("a5", 4)
    a4 = _ax("a4", 8)
    conds: list[Cond] = []
    for i in range(n_cells):
        # Per-cell offset in N — the only thing that differs from cell 0
        # in either Load index or Accum name.
        offset = i * 8
        inner_stmts = (
            # N-invariant prefix — identical across cells.
            Load(name="in3", input="x", index=(Var("a5"), Var("a4"))),
            Assign(name="v5", op="multiply", args=("in3", "v4"), dtype=None),
            Load(name="wn_v", input="wn", index=(Var("a5"),)),
            Assign(name="v6", op="multiply", args=("wn_v", "v5"), dtype=None),
            # Per-cell tail.
            Load(name=f"wl_{i}", input="wl", index=(Var("a5"), Literal(offset, "int"))),
            Assign(name=f"v7_{i}", op="multiply", args=(f"wl_{i}", "v6"), dtype=None),
            Accum(name=f"acc_{i}", value=f"v7_{i}", op=ElementwiseImpl("add")),
        )
        body = (
            SerialTile(axis=a5, body=Body((SerialTile(axis=a4, body=Body(inner_stmts), kind="plain"),)), kind="plain"),
            Write(output="o", index=(Literal(offset, "int"),), value=f"acc_{i}"),
        )
        # Cell 0's predicate is the trivial ``1``; subsequent cells carry a
        # masked-overhang guard. The exact predicate doesn't matter for the
        # fusion test — only that they're distinct across cells.
        pred = Literal(1, "int") if i == 0 else BinaryExpr("<", Literal(offset, "int"), Literal(64, "int"))
        conds.append(Cond(cond=pred, body=Body(body)))
    return Body(tuple(conds))


def _count_kind(body: Body, *kinds) -> int:
    """Recursively count stmts of any of ``kinds`` in ``body``."""
    n = 0
    for s in body:
        if isinstance(s, kinds):
            n += 1
        for child in s.nested():
            n += _count_kind(child, *kinds)
    return n


class _FakeNode:
    """Minimal Node shim — rewrite() only reads ``.op``."""

    def __init__(self, op):
        self.op = op


def test_fuse_run_collapses_invariant_prefix():
    """Four sibling Conds with an identical N-invariant prefix collapse to a
    single shared inner-loop body; per-cell weight Load + multiply +
    Accum + Write each rewrap in their original predicate."""
    body = _synthetic_replicated_body(n_cells=4)
    root = _FakeNode(TileOp(body=body, name="k_test", knobs={}))
    rewritten = _fuse_mod.rewrite(root)
    assert rewritten is not None
    new_body = rewritten.body

    # Invariant prefix collapsed: only ONE copy of the ``Load x`` /
    # ``Load wn`` survives across the entire kernel (previously 4 copies).
    assert _count_kind(new_body, Load) == 1 + 1 + 4, (
        f"expected 6 Loads (1 ``Load x`` + 1 ``Load wn`` + 4 per-cell ``Load wl_<i>``), got {_count_kind(new_body, Load)}"
    )
    # Per-cell Accums + Writes still present once each.
    assert _count_kind(new_body, Accum) == 4
    assert _count_kind(new_body, Write) == 4


def test_singleton_cond_run_is_skipped():
    """A single ``Cond`` (no siblings) has nothing to fuse."""
    body = _synthetic_replicated_body(n_cells=1)
    root = _FakeNode(TileOp(body=body, name="k_test", knobs={}))
    with pytest.raises(Exception, match="no fusable sibling Cond runs"):
        _fuse_mod.rewrite(root)


def test_run_with_no_common_prefix_is_skipped():
    """Sibling Conds whose first inner stmt diverges (no common invariant
    prefix) leave the body unchanged — the pass refuses to emit empty
    hoisted prefixes."""
    a5 = _ax("a5", 4)
    cond_a = Cond(
        cond=Literal(1, "int"),
        body=Body((SerialTile(axis=a5, body=Body((Load(name="x_a", input="x", index=(Var("a5"),)),)), kind="plain"),)),
    )
    cond_b = Cond(
        cond=Literal(0, "int"),
        body=Body((SerialTile(axis=a5, body=Body((Load(name="x_b", input="y", index=(Var("a5"),)),)), kind="plain"),)),
    )
    root = _FakeNode(TileOp(body=Body((cond_a, cond_b)), name="k_test", knobs={}))
    with pytest.raises(Exception, match="no fusable sibling Cond runs"):
        _fuse_mod.rewrite(root)


# ---------------------------------------------------------------------------
# End-to-end on the Qwen3-Embedding-0.6B linear+mean-reduce failing variant
# ---------------------------------------------------------------------------


@requires_cuda
def test_qwen_lmhead_variant_compiles_within_budget(monkeypatch):
    """The previously-failing ``BK=64, BM=1, BN=64, FM=1, FN=64`` variant of
    ``k_linear_mean_reduce`` on Qwen3-Embedding-0.6B used to take 5–6 s
    under ``cicc -O1`` because the RMSNorm prologue duplicated 64×. With
    the new sibling-Cond fuser landed, the same source folds to a single
    body-level RMSNorm chain + 64 short per-cell guarded multiplies — the
    rendered kernel should comfortably fit in the autotune's 2 s compile
    budget. We pin a smaller M+N here so the test stays CI-friendly; the
    structural pattern is identical."""
    for key, value in {"BK": "64", "BM": "1", "BN": "64", "BR": "1", "FM": "1", "FN": "64", "SPLITK": "1", "STAGE": "1"}.items():
        monkeypatch.setenv(f"EMMY_{key}", value)

    # M=2 (tiny batch), K=1024 (RMSNorm range), N=64 × 64 + 3 = 4099 — N is
    # deliberately a non-multiple of BN·FN to trigger the masked-overhang
    # Conds whose duplicates this pass folds.
    M, K, N = 2, 1024, 4099
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (M, K)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wn", (K,)), node_id="wn")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wl", (N, K)), node_id="wl")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "wn"], output=Tensor("xn", (M, K)), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "wl"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs = ["x", "wn", "wl"]
    g.outputs = ["o"]

    rng = np.random.default_rng(seed=11)
    inputs = {
        "x": rng.standard_normal((M, K), dtype=np.float32).astype(np.float32),
        "wn": (rng.standard_normal((K,), dtype=np.float32) * 0.1).astype(np.float32),
        "wl": (rng.standard_normal((N, K), dtype=np.float32) * 0.02).astype(np.float32),
    }

    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415
    from emmy.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    ref = NumpyBackend().run(NumpyBackend().compile(g), input_data=inputs)[0].outputs["o"]

    backend = CudaBackend()
    compiled = backend.compile(g)
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    cuda_src = "\n".join(op.kernel_source for op in cuda_ops)
    n_lines = cuda_src.count("\n")
    # Pre-fix this was ~1014 lines on the layered MLP kernel (compile 5–6 s).
    # Post-fix the same shape lands at ~720 lines or less. Threshold 850
    # catches a regression with margin but doesn't crack down on small
    # post-fix codegen drift.
    assert n_lines < 850, f"rendered kernel is {n_lines} lines — regression: invariant prefix is no longer being hoisted"

    out = backend.run(compiled, input_data=inputs)[0].outputs["o"]
    np.testing.assert_allclose(out, ref, rtol=5e-2, atol=5e-3)
