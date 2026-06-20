"""Tests for ``tile/017_atomic_free_splitk`` — two-kernel split-K fork.

Covers the fork emission (False vs True branches), the matmul Write
rewire (atomic axis shrinks to ∅), the reduce TileOp structure,
SPLITK = 1 skip, and the matmul_add (Cond) residual interaction so
both Cond branches get rewired to the workspace.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Load, Write
from deplodock.compiler.ir.tile.ir import GridTile, SerialTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import RuleSkipped
from deplodock.compiler.tensor import Tensor

afree = importlib.import_module(
    "deplodock.compiler.pipeline.passes.lowering.tile.017_atomic_free_splitk",
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubCtx:
    """Minimal Context stand-in — the rule reads no fields off ``ctx``."""


class _StubMatch:
    """Match stub carrying the host graph (the only field 017 reads)."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph


# ---------------------------------------------------------------------------
# Fixtures — synthetic matmul TileOp bodies + host graph
# ---------------------------------------------------------------------------


_M_EXT = 128
_N_EXT = 128
_S_EXT = 4
_BM = 16
_BN = 16


def _m_idx() -> BinaryExpr:
    return Var("M_b") * Literal(_BM, "int") + Var("M_t")


def _n_idx() -> BinaryExpr:
    return Var("N_b") * Literal(_BN, "int") + Var("N_t")


def _matmul_body(*, with_residual: bool) -> Body:
    """Build a tiny matmul body: GridTile(M_b, N_b, K_s) → ThreadTile →
    [Load a, Load b, Accum, Write(out, acc)] for plain matmul; for
    matmul_add the Write is wrapped under ``Cond(K_s == 0, [Load r,
    Write(out, v=acc+r)], else_body=[Write(out, acc)])`` mirroring
    ``015_gate_splitk_residual``'s output.
    """
    M_b = Axis("M_b", _M_EXT // _BM)
    N_b = Axis("N_b", _N_EXT // _BN)
    K_s = Axis("K_s", _S_EXT)
    M_t = Axis("M_t", _BM)
    N_t = Axis("N_t", _BN)

    reduce_stmts = (
        Load(name="a", input="a", index=(_m_idx(), Var("k")), dtype=F32),
        Load(name="b", input="b", index=(Var("k"), _n_idx()), dtype=F32),
        Accum(name="acc", value="a", dtype=F32, axes=("k",)),
    )

    if with_residual:
        # Mirrors 015's output: linear-residual gated on K_s == 0, else
        # Write the bare Accum.
        epilogue = (
            Cond(
                cond=BinaryExpr("==", Var(K_s.name), Literal(0, "int")),
                body=Body(
                    (
                        Load(name="r", input="r", index=(_m_idx(), _n_idx()), dtype=F32),
                        # Reuse "acc" + "r" → Write the residual-bumped value.
                        # (No Assign — the test just needs the Write shape; the
                        # gate's exact Assign mechanics aren't under test.)
                        Write(output="out", index=(_m_idx(), _n_idx()), value="r", value_dtype=F32),
                    )
                ),
                else_body=Body((Write(output="out", index=(_m_idx(), _n_idx()), value="acc", value_dtype=F32),)),
            ),
        )
    else:
        epilogue = (Write(output="out", index=(_m_idx(), _n_idx()), value="acc", value_dtype=F32),)

    inner = ThreadTile(axes=(M_t, N_t), body=Body((*reduce_stmts, *epilogue)))
    body = GridTile(axes=(M_b, N_b, K_s), body=Body((inner,)))
    return Body((body,))


def _matmul_tileop(*, with_residual: bool = False, knobs: dict | None = None) -> TileOp:
    return TileOp(
        body=_matmul_body(with_residual=with_residual),
        name="k_matmul",
        knobs={"SPLITK": _S_EXT, "BM": _BM, "BN": _BN, **(knobs or {})},
    )


def _host_graph_with(op: TileOp) -> tuple[Graph, str]:
    """Build a host graph wiring two input nodes + the matmul TileOp.
    Returns ``(graph, matmul_node_id)``.
    """
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (_M_EXT, _N_EXT), F32), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (_N_EXT, _N_EXT), F32), node_id="b")
    inputs = ["a", "b"]
    if "r" in op.inputs:
        g.add_node(InputOp(), [], Tensor("r", (_M_EXT, _N_EXT), F32), node_id="r")
        inputs.append("r")
    nid = g.add_node(op, inputs, Tensor("out", (_M_EXT, _N_EXT), F32), node_id="out")
    g.outputs = [nid]
    return g, nid


def _rewrite(op: TileOp) -> list:
    """Run the pass against ``op`` plumbed through a stub Match + Ctx."""
    g, root_id = _host_graph_with(op)
    root = g.nodes[root_id]
    match = _StubMatch(g)
    return afree.rewrite(_StubCtx(), match, root)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_splitk_one_off_stamped(monkeypatch):
    """SPLITK = 1 — no atomic axis → records ATOMIC_FREE_SPLITK=False (the
    decision, not a skip), so the realized config keeps a uniform knob set."""
    monkeypatch.delenv("DEPLODOCK_ATOMIC_FREE_SPLITK", raising=False)
    # No K_s axis in the GridTile → atomic_axes empty → off.
    M_b = Axis("M_b", _M_EXT // _BM)
    N_b = Axis("N_b", _N_EXT // _BN)
    M_t = Axis("M_t", _BM)
    N_t = Axis("N_t", _BN)
    body = GridTile(
        axes=(M_b, N_b),
        body=Body(
            (
                ThreadTile(
                    axes=(M_t, N_t),
                    body=Body((Write(output="out", index=(_m_idx(), _n_idx()), value="acc", value_dtype=F32),)),
                ),
            )
        ),
    )
    op = TileOp(body=Body((body,)), name="k_matmul", knobs={"SPLITK": 1, "BM": _BM, "BN": _BN})
    result = _rewrite(op)
    assert isinstance(result, TileOp)
    assert result.knobs[afree.ATOMIC_FREE_SPLITK.name] is False


def test_idempotent_skip_when_knob_present(monkeypatch):
    """Re-running on a TileOp whose ``knobs`` already names the fork knob → skip."""
    monkeypatch.delenv("DEPLODOCK_ATOMIC_FREE_SPLITK", raising=False)
    op = _matmul_tileop(knobs={afree.ATOMIC_FREE_SPLITK.name: False})
    with pytest.raises(RuleSkipped, match="already decided"):
        _rewrite(op)


def test_mma_path_forks_atomic_free(monkeypatch):
    """MMA / warp tier (Step 3b): the ``is_warp`` early-out is gone — the C-fragment
    Write is still a tile-level Write at this stage, so the MMA split-K forks
    atomic-free (workspace + reduce) just like the scalar path, instead of being
    pinned to the codegen ``atomicAdd``."""
    monkeypatch.delenv("DEPLODOCK_ATOMIC_FREE_SPLITK", raising=False)
    op = _matmul_tileop(knobs={"MMA": "mma_m16n8k16_f16"})
    variants = _rewrite(op)
    assert variants is not None and len(variants) == 2
    assert isinstance(variants[0], TileOp) and variants[0].knobs[afree.ATOMIC_FREE_SPLITK.name] is False
    assert isinstance(variants[1], Graph)


def test_fork_emits_two_variants(monkeypatch):
    """Unpinned env → fork emits both False and True variants."""
    monkeypatch.delenv("DEPLODOCK_ATOMIC_FREE_SPLITK", raising=False)
    op = _matmul_tileop()
    variants = _rewrite(op)
    assert variants is not None
    assert len(variants) == 2
    # False option is a plain TileOp; True option is a Graph fragment.
    assert isinstance(variants[0], TileOp)
    assert variants[0].knobs[afree.ATOMIC_FREE_SPLITK.name] is False
    assert isinstance(variants[1], Graph)


def test_env_pin_true_only(monkeypatch):
    """Pinned True → only the atomic-free fragment is emitted."""
    monkeypatch.setenv("DEPLODOCK_ATOMIC_FREE_SPLITK", "1")
    op = _matmul_tileop()
    variants = _rewrite(op)
    assert variants is not None
    assert len(variants) == 1
    assert isinstance(variants[0], Graph)


def test_env_pin_false_only(monkeypatch):
    """Pinned False → only the legacy atomic variant is emitted."""
    monkeypatch.setenv("DEPLODOCK_ATOMIC_FREE_SPLITK", "0")
    op = _matmul_tileop()
    variants = _rewrite(op)
    assert variants is not None
    assert len(variants) == 1
    assert isinstance(variants[0], TileOp)
    assert variants[0].knobs[afree.ATOMIC_FREE_SPLITK.name] is False


def test_atomic_free_fragment_shape(monkeypatch):
    """True fragment: matmul TileOp + sibling reduce TileOp. The matmul's
    output Writes now name the workspace and carry K_s in index; the
    reduce body has GridTile(M_b_red, N_b_red) → ThreadTile → SerialTile(K_s_red)
    → Write(out)."""
    monkeypatch.setenv("DEPLODOCK_ATOMIC_FREE_SPLITK", "1")
    op = _matmul_tileop()
    (frag,) = _rewrite(op)
    assert isinstance(frag, Graph)

    # Find the two TileOp nodes in topo order.
    tile_ids = [nid for nid in frag.topological_order() if isinstance(frag.nodes[nid].op, TileOp)]
    assert len(tile_ids) == 2
    workspace_id, reduce_id = tile_ids
    assert reduce_id == frag.outputs[0]

    matmul_op: TileOp = frag.nodes[workspace_id].op
    reduce_op: TileOp = frag.nodes[reduce_id].op

    # Workspace Tensor: shape (S, M, N).
    ws_tensor = frag.nodes[workspace_id].output
    assert ws_tensor.shape == (_S_EXT, _M_EXT, _N_EXT)

    # The K_s axis name is canonicalized by ``normalize_body`` so derive it
    # from the matmul body rather than hard-coding the original "K_s".
    from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import find_split_k_axis_name

    k_s_canonical = find_split_k_axis_name(matmul_op)
    assert k_s_canonical is None  # axis is now in every Write's index → no atomic axis

    # Matmul's Writes — all targeting the workspace, K_s in index → atomic_axes empty.
    writes = matmul_op.body.coordination.writes
    assert writes
    # Every Write's leading-index free var should match the K_s name observed
    # in the ORIGINAL TileOp (before 017 fires).
    orig_k_s = find_split_k_axis_name(op)
    assert orig_k_s is not None
    for w in writes:
        assert w.output == workspace_id
        assert matmul_op.body.coordination.atomic_axes(w) == frozenset()
        assert isinstance(w.index[0], Var) and w.index[0].name == orig_k_s

    # Reduce body structure — axis names are canonicalized by ``normalize_body``,
    # so check shape (extents + nesting) rather than names.
    (grid,) = (s for s in reduce_op.body if isinstance(s, GridTile))
    expected_m_blocks = -(-_M_EXT // 16)
    expected_n_blocks = -(-_N_EXT // 16)
    assert sorted(ax.extent.as_static() for ax in grid.axes) == sorted((expected_m_blocks, expected_n_blocks))
    (thread,) = (s for s in grid.body if isinstance(s, ThreadTile))
    assert sorted(ax.extent.as_static() for ax in thread.axes) == [16, 16]
    serials = [s for s in thread.body if isinstance(s, SerialTile)]
    assert len(serials) == 1
    assert serials[0].axis.extent.as_static() == _S_EXT
    assert serials[0].unroll is True
    # Reduce Accum with op=add (SSA name canonicalized post-normalize).
    accums = [a for a in serials[0].body if isinstance(a, Accum)]
    assert len(accums) == 1 and accums[0].op.name == "add"


def test_atomic_free_with_residual_rewires_both_cond_branches(monkeypatch):
    """matmul_add post-015: a Cond wraps the residual Write (body) and
    the bare-Accum Write (else_body). 017 must rewire BOTH so neither
    leaks an atomic store to the original output."""
    from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import find_split_k_axis_name

    monkeypatch.setenv("DEPLODOCK_ATOMIC_FREE_SPLITK", "1")
    op = _matmul_tileop(with_residual=True)
    orig_k_s = find_split_k_axis_name(op)
    assert orig_k_s is not None
    (frag,) = _rewrite(op)
    workspace_id = next(nid for nid in frag.topological_order() if nid.endswith("__partial"))
    matmul_op: TileOp = frag.nodes[workspace_id].op
    writes = matmul_op.body.coordination.writes
    # Two Writes — both targeting the workspace, both with K_s in the leading index.
    assert len(writes) == 2
    for w in writes:
        assert w.output == workspace_id
        assert matmul_op.body.coordination.atomic_axes(w) == frozenset()
        assert isinstance(w.index[0], Var) and w.index[0].name == orig_k_s
