"""Masked thread tiles for fused-prologue matmuls (gate B of symbolic-axis parity).

A fused prologue (SDPA P@V: softmax max/sum feeding the matmul) used to force
symbolic axes fully degenerate (``mask_ok = not prologue`` — one output
element per thread, 8-32-thread CTAs). The constraint is real only for
register tiling: per-M-row accumulators must not span register cells. THREAD
-level masking is safe — the boundary Cond wraps the whole per-row body
(prologue + matmul) inside ``SerialTile(M_r)``, and the enumerator's
``mask_f1`` restricts masked prologue rows to ``FM = FN = 1``.

The Cond placement also fixes a latent OOB: the old M-Cond wrapped only the
matmul body, so an overhang row's softmax reduces would read ``P[m, k]`` past
the buffer (static non-divisor masked tiles included).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler import target as target_mod
from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp, SoftmaxOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Cond
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.search.db import SearchDB

from ..conftest import requires_cuda

_pp = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")


@pytest.fixture(autouse=True)
def _isolated_prior(monkeypatch, tmp_path):
    monkeypatch.setenv("DEPLODOCK_PRIOR_FILE", str(tmp_path / "prior.json"))
    yield


def _softmax_matmul_graph(*, N: int = 32) -> Graph:
    """softmax → matmul with symbolic M AND symbolic K (the true P@V shape:
    ``P[seq_q, seq_k] @ V[seq_k, N]``): fuses into the fused-prologue kernel
    shape (softmax max/sum/reciprocal as prologue of the matmul). Symbolic K
    is what scopes the masking in — a symbolic-K loop never stages, so no
    collective lives under the divergent per-row guard."""
    f16 = _dt.get("f16")
    s = Dim("seq_len")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("scores", (1, s, s), f16), node_id="scores")
    g.add_node(InputOp(), [], Tensor("v", (s, N), f16), node_id="v")
    g.add_node(SoftmaxOp(axis=-1), ["scores"], Tensor("p", (1, s, s), f16), node_id="p")
    g.add_node(MatmulOp(), ["p", "v"], Tensor("o", (1, s, N), f16), node_id="o")
    g.inputs, g.outputs = ["scores", "v"], ["o"]
    return g


def _static_k_prologue_graph(*, K: int = 64, N: int = 32) -> Graph:
    """Same shape with a STATIC K: stays degenerate on its symbolic rows (its
    K pipeline stages; the masked Cond + 021 hoist would break the prologue's
    SSA ordering — see the planner comment)."""
    f16 = _dt.get("f16")
    s = Dim("seq_len")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("scores", (1, s, K), f16), node_id="scores")
    g.add_node(InputOp(), [], Tensor("v", (K, N), f16), node_id="v")
    g.add_node(SoftmaxOp(axis=-1), ["scores"], Tensor("p", (1, s, K), f16), node_id="p")
    g.add_node(MatmulOp(), ["p", "v"], Tensor("o", (1, s, N), f16), node_id="o")
    g.inputs, g.outputs = ["scores", "v"], ["o"]
    return g


def _fused_prologue_plan(graph: Graph | None = None):
    fused = Pipeline.build(LOOP_PASSES).run(graph or _softmax_matmul_graph(), ctx=Context.from_target((12, 0)), db=SearchDB())
    node = next(n for n in fused.nodes.values() if isinstance(n.op, LoopOp))
    chain, prologue = _pp._outer_free_loop_chain(node.op.body)
    assert prologue, "fixture must fuse into the fused-prologue kernel shape"
    plan = _pp._plan_kernel(node.op, Context.from_target((12, 0)), kernel_name="o", graph=fused)
    assert plan is not None
    return plan


def test_masked_prologue_rows_are_thread_tiled_f1():
    """Symbolic-M prologue kernels enumerate MASKED rows with BM > 1 (thread
    tiles — previously the whole space was the degenerate E=1 family).
    The MASKED axis's register tiling clamps to 1 (no per-row accumulator
    spans register cells along it); the unmasked static N axis keeps its F
    sweep — P@V's best configs carry FN > 1."""
    plan = _fused_prologue_plan()
    masked = [r for r in plan.params if r.get("OVERHANG")]
    assert masked, "symbolic-M prologue kernel must enumerate masked rows"
    assert any(r.get("BM", 1) > 1 for r in masked), "masked rows must include thread tiles (BM > 1)"
    m_masked = [r for r in masked if "OVERHANG" in r]
    bad = [r for r in m_masked if r.get("FM", 1) > 1]
    assert not bad, f"masked-M prologue rows must keep FM = 1, got {bad}"
    assert any(r.get("FN", 1) > 1 for r in masked), "the unmasked N axis must keep its register-tile sweep (FN > 1)"


def test_static_k_prologue_stays_degenerate():
    """The scope pin: a STATIC-K prologue kernel (fused gated-MLP class)
    keeps the degenerate symbolic-row family — its staged K pipeline can't
    coexist with the per-row guard (021's hoist would break the prologue's
    SSA ordering). Its deployment path is the structural split."""
    plan = _fused_prologue_plan(_static_k_prologue_graph())
    masked = [r for r in plan.params if r.get("OVERHANG")]
    assert not masked, f"static-K prologue kernels must not enumerate masked rows, got {masked[:3]}"
    assert all(r.get("BM", 1) == 1 for r in plan.params), "symbolic rows must stay degenerate (one element per thread)"


def test_masked_prologue_cond_encloses_prologue_reduces():
    """The boundary Cond must be an ancestor of the softmax Accums: an
    overhang row's prologue reads index the runtime-sized ``P[m, k]``, so the
    whole per-row body — prologue + matmul — is guarded as a unit."""
    plan = _fused_prologue_plan()
    row = next(r for r in plan.params if r.get("OVERHANG") and r.get("BM", 1) > 1)
    tile_op = _pp._materialize(plan, row)

    conds = tile_op.body.iter_of_type(Cond)
    assert conds, "masked variant must carry the boundary Cond"
    guarded_accums = [a for c in conds for a in c.body.iter_of_type(Accum)]
    all_accums = tile_op.body.iter_of_type(Accum)
    assert len(all_accums) >= 3, "prologue (max/sum) + matmul accums expected"
    assert len(guarded_accums) == len(all_accums), (
        f"every accum (prologue included) must sit inside the boundary Cond — {len(guarded_accums)}/{len(all_accums)} guarded"
    )


@requires_cuda
@pytest.mark.parametrize("seq", [33, 100])
def test_masked_prologue_accuracy_cuda(monkeypatch, seq):
    """Pinned masked thread tiles (BM=8, BN=32): one compiled symbolic kernel
    matches the numpy softmax @ v reference at runtime seqs that straddle the
    8-row tile."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    target_mod.set_target(None)  # live device
    monkeypatch.setenv("DEPLODOCK_BM", "8")
    monkeypatch.setenv("DEPLODOCK_BN", "32")
    rng = np.random.default_rng(0)
    N = 32
    scores = rng.standard_normal((1, seq, seq), dtype=np.float32).astype(np.float16)
    v = (rng.standard_normal((seq, N), dtype=np.float32) * 0.1).astype(np.float16)

    be = CudaBackend()
    compiled = be.compile(_softmax_matmul_graph(N=N))
    out = be.run(compiled, input_data={"scores": scores, "v": v})[0].outputs["o"].astype(np.float32)

    s32 = scores.astype(np.float32)
    e = np.exp(s32 - s32.max(axis=-1, keepdims=True))
    ref = (e / e.sum(axis=-1, keepdims=True)) @ v.astype(np.float32)
    assert out.shape == (1, seq, N)
    np.testing.assert_allclose(out, ref, atol=2e-2, rtol=0.05)
