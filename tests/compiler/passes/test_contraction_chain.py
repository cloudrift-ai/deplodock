"""The carried contraction chain of the tensor-core streaming flash.

A streaming-flash nest is a ``Monoid`` carrier streaming over a *nested* QK^T
contraction. Unification 3 reads it as a **chain on a shared axis**: ``kv`` is the
dual-role hinge — free-output of the inner QK^T contraction, reduce of the outer P@V
contraction (embedded in the carrier's ``O = O·α + p·v``) and of the carrier. These
tests pin the **derived view** ``IterDag.reduction`` exposing that structure; it is a
projection of the body (computed on demand), so a non-streaming nest yields ``None``
and a flash nest yields the hinge + the inner SEMIRING contraction + the carrier.

All CPU — no CUDA, no lowering. The streaming ``LoopOp`` is built directly by the
recognizer's ``build_flash_frag`` (the same body the GPU flash tests compile)."""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Load, Monoid
from deplodock.compiler.ir.tensor.ir import ReduceOp
from deplodock.compiler.ir.tile.ir import Binding, RegisterTile
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.loop.recognize._flash import build_flash_frag
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import chain_build, seed_graph
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_M_THREAD, MAP_N_THREAD
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import legal_decomps


def _flash_loop(*, causal: bool = False, group: int = 1) -> LoopOp:
    """The fused streaming-flash ``LoopOp`` for a small ``(B,H,S,D)`` SDPA — the
    body ``iter_dag`` reads. Built straight from the recognizer so the test needs
    no torch trace."""
    B, H, S, D = 1, 2, 16, 8
    shp = tuple(Dim(d) for d in (B, H, S, D))
    out = Tensor("o", (B, H, S, D), F32)
    frag = build_flash_frag("q", "k", "v", shp, shp, shp, out, causal=causal, group=group)
    return frag.nodes["o"].op


# ``LoopOp`` canonicalizes axis / SSA names on construction (``kv`` -> ``a3``, ``s`` ->
# ``v1``), so the tests key on STRUCTURE — extent, algebra, def-use — never literal names.
_S, _D = 16, 8  # the streaming KV extent (hinge) and the head-dim (inner QK^T reduce)


def test_streaming_flash_exposes_the_chain():
    dag = iter_dag(_flash_loop())
    assert dag.reduction is not None and dag.reduction.nested, "a flash nest must be streaming"
    chain = dag.reduction
    assert chain is not None, "a streaming flash nest must expose the carried contraction chain"

    # The hinge carries the online-softmax Monoid (the streaming reduce + the outer P@V
    # contraction live here); the inner QK^T is a nested SEMIRING reduce over the head-dim.
    assert isinstance(chain.carrier, Monoid)
    assert chain.hinge.algebra is AlgebraKind.MONOID
    assert chain.hinge.extent == _S
    assert chain.inner.algebra is AlgebraKind.SEMIRING
    assert chain.inner.extent == _D
    # The inner contraction is nested directly inside the hinge (the carried chain).
    assert chain.inner.parent is not None and chain.inner.parent.axis.name == chain.hinge_name


def test_chain_hinge_is_dual_role():
    """The hinge is dual-role: the reduce of the carrier (and P@V) AND a *free output*
    of the inner QK^T contraction (it indexes K inside the QK^T body but is not the QK^T
    reduce axis)."""
    dag = iter_dag(_flash_loop())
    chain = dag.reduction
    hinge, inner = chain.hinge_name, chain.inner.loop

    # The inner contraction reduces the head-dim, not the hinge.
    assert inner.axis.name != hinge
    # ...yet the hinge indexes a Load in the inner body (K[kv, dd]) — its free output.
    inner_load_vars = {v for ld in inner.body.iter_of_type(Load) for e in ld.index for v in e.free_vars()}
    assert hinge in inner_load_vars, "the hinge must be a free output (index var) of the inner QK^T contraction"


def test_chain_score_is_the_carrier_partial():
    """The score edge is the inner contraction's result the carrier folds — the
    carrier's first partial (the INLINE edge value 1c materializes)."""
    dag = iter_dag(_flash_loop())
    chain = dag.reduction
    assert chain.score == chain.carrier.partial[0]


def test_chain_is_a_monoid_over_semiring_composition():
    """The chain is the **MONOID(SEMIRING)** composition, not a flat parse: a ``Monoid`` carrier
    composed over an inner SEMIRING :class:`Contraction`. The carried-chain invariant is the shared
    hinge — the inner contraction's free-output column IS the carrier's reduced hinge — and it is
    enforced by construction, so a malformed composition is unrepresentable."""
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import Contraction

    dag = iter_dag(_flash_loop())
    chain = dag.reduction
    assert isinstance(chain.carrier, Monoid), "the outer algebra is a MONOID carrier"
    assert isinstance(chain.inner, Contraction) and chain.inner.algebra is AlgebraKind.SEMIRING, "the inner operand is a SEMIRING"
    # The composition's invariant: the inner contraction's output column coord IS the hinge (the
    # chain link), and the score edge is the inner contraction's result (the carrier's first partial).
    assert chain.inner.out_index[-1].free_vars() == {chain.hinge_name}, "the inner contraction's output column is the hinge"
    assert chain.score == chain.inner.result == chain.carrier.partial[0]
    # The chain carries NO geometry: m/d/grid are walked off the composition at build time.
    assert not any(hasattr(chain, attr) for attr in ("m", "d", "grid", "grid_nodes", "m_axis", "d_axis"))


def test_standalone_matmul_is_a_contraction():
    """The unification: a standalone matmul is the SAME ``Contraction`` representation as the flash
    inner. ``IterDag.contractions`` yields it — one per matmul-reduce, SEMIRING, carrying the output
    ``(M, N)`` coords off the ``Write`` (not re-derived in the atom gate) — so the matmul tier and
    the flash chain share one SEMIRING-contraction abstraction and one ``contraction_atomizes`` call
    (the warp-tier behavior itself is covered end to end by the GPU matmul/flash e2e suites)."""
    from deplodock.compiler.dtype import F16
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import Contraction

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (128, 64), F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (64, 128), F16), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (128, 128), F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    lowered = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target((8, 0)))
    lo = next(n.op for n in lowered.nodes.values() if type(n.op).__name__ == "LoopOp")

    (c,) = iter_dag(lo).contractions
    assert isinstance(c, Contraction) and c.algebra is AlgebraKind.SEMIRING, "a matmul reduce is a SEMIRING Contraction"
    assert c.out_index is not None and sum(1 for e in c.out_index if e.free_vars()) >= 2, "out_index carries the matmul's (M, N)"
    assert c.result, "the contraction's result edge (the Accum SSA) is recorded"


def test_chain_free_axes_walks_the_composition():
    """The build geometry is **walked off the composition** (``chain_free_axes``), not stored on the
    chain: the query row ``m`` is the inner contraction's own free output (``out_index[-2]``), the
    head output ``d`` is the carrier's value operand's own free output, ``grid`` the shared batch /
    head axes — each read from the composition level that defines it."""
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import chain_free_axes

    dag = iter_dag(_flash_loop())
    chain = dag.reduction
    m_axis, d_axis, grid = chain_free_axes(chain, dag)
    assert m_axis.name in chain.inner.out_index[-2].free_vars(), "m is the inner contraction's output row"
    assert m_axis.name not in (chain.hinge_name, d_axis.name), "m is neither the hinge nor the P@V output"
    grid_names = {lp.axis.name for lp in grid}
    assert grid_names and m_axis.name not in grid_names and d_axis.name not in grid_names, "grid is the shared batch/head axes"


def test_chain_post_init_enforces_the_hinge_invariant():
    """A ``MonoidReduction`` whose inner output column is NOT the hinge is rejected at construction
    — the carried-chain invariant lives in the class, not in an external gate, so invalid states are
    unrepresentable."""
    import pytest

    from deplodock.compiler.ir.expr import Var
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import Contraction, MonoidReduction, chain_free_axes

    dag = iter_dag(_flash_loop())
    good = dag.reduction
    m_axis, _d, _grid = chain_free_axes(good, dag)
    # Put a non-hinge axis (the query row) in the inner's output-column coord — must raise.
    broken_inner = Contraction(node=good.inner.node, result=good.inner.result, out_index=(Var(good.hinge_name), Var(m_axis.name)))
    with pytest.raises(ValueError, match="hinge"):
        MonoidReduction(carrier=good.carrier, axis=good.hinge, inner=broken_inner)


def test_partition_free_axes_is_a_role_neutral_three_way_split():
    """The reusable partition: axes in operand A's footprint only, B's only, and the rest — in
    BOTH or in NEITHER (the shared / broadcast axes). No attention vocabulary, so any two-operand
    composition reuses it; the both-or-neither lumping is what the chain reads as ``grid``."""
    from types import SimpleNamespace

    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import partition_free_axes

    nodes = tuple(SimpleNamespace(axis=SimpleNamespace(name=n)) for n in ("m", "d", "both", "neither"))
    a_only, b_only, rest = partition_free_axes(nodes, footprint_a={"m", "both"}, footprint_b={"d", "both"})
    assert [n.axis.name for n in a_only] == ["m"]
    assert [n.axis.name for n in b_only] == ["d"]
    assert [n.axis.name for n in rest] == ["both", "neither"]


def test_causal_chain_still_exposes_the_chain():
    """A causal mask folds a masked score; the chain still reports the carrier's
    first partial (whatever the carrier folds, not a hard-coded name)."""
    dag = iter_dag(_flash_loop(causal=True))
    chain = dag.reduction
    assert chain is not None
    assert chain.score == chain.carrier.partial[0]
    assert chain.hinge.algebra is AlgebraKind.MONOID
    assert chain.inner.algebra is AlgebraKind.SEMIRING


def _reduce_loop() -> LoopOp:
    """A flat (non-streaming) ``sum``-reduce ``LoopOp`` — the MONOID counterpoint to
    the streaming flash, for the ``chain is None`` / ``inner_algebra is None`` cases."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 64)), node_id="x")
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", (4, 1)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    out = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target((12, 0)))
    return next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")


def test_plain_reduce_is_a_reduction_with_no_inner():
    """A flat (non-streaming) reduce is the SAME ``MonoidReduction`` class as flash — just with no
    inner contraction (``inner is None``). The composition is uniform; flash is the inner-present case."""
    dag = iter_dag(_reduce_loop())
    assert dag.reduction is not None and not dag.reduction.nested
    reduction = dag.reduction
    assert reduction is not None and reduction.inner is None, "a flat reduce is a MonoidReduction with no inner"


# --- 1b: classify recognizes MONOID(SEMIRING) -------------------------------------


def test_classify_streaming_is_compositional():
    """A streaming flash classifies as the compositional algebra ``MONOID(SEMIRING)``:
    the outer carrier is MONOID, the embedded P@V on the hinge is SEMIRING. Both the
    hinge and the inner QK^T axis are reduce targets."""
    dag = iter_dag(_flash_loop())
    regime = classify(dag)
    assert regime is not None
    assert regime.algebra is AlgebraKind.MONOID
    assert regime.inner_algebra is AlgebraKind.SEMIRING
    # Both contraction axes (the hinge kv stream + the inner QK^T reduce) are rewritten.
    assert {dag.reduction.hinge_name, dag.reduction.inner_name} <= regime.target_names


def test_classify_flat_monoid_is_not_compositional():
    """A flat MONOID reduce has no embedded contraction — ``inner_algebra is None``,
    so the compositional reading stays specific to the twisted (streaming) carrier."""
    regime = classify(iter_dag(_reduce_loop()))
    assert regime is not None
    assert regime.algebra is AlgebraKind.MONOID
    assert regime.inner_algebra is None


def test_legal_decomps_splits_the_hinge_under_both_traits():
    """The hinge ``kv`` split is licensed BOTH ways by the twisted carrier's traits:
    associativity (the carrier reduce → a serial / register re-bracket) and
    commutativity (the embedded P@V reduce → a THREAD partition, whose additive
    recombine is the Monoid's ``combine_states``). One ``Monoid`` carrier carries both,
    which is exactly why the shared-axis tiling (1c) is sound."""
    dag = iter_dag(_flash_loop())
    chain = dag.reduction
    carrier, kv, ext = chain.carrier, chain.hinge.loop.axis, chain.hinge.extent
    bn = 4
    placement = [Role.THREAD, Role.STAGE_INNER, Role.REGISTER]

    # Associative trait (the streaming carrier reduce): a serial BN re-bracket of kv
    # (no partition — factor[0] == 1) is licensed.
    serial = legal_decomps(carrier, kv, ext, factor_menus=[[1], [bn], [1]], placement=placement, masked=False)
    assert any(d.factors == (1, bn, 1) for d in serial)

    # Commutative trait (the embedded P@V reduce): a THREAD partition of kv
    # (factor[0] == bn — cooperative-KV / split reduce) is licensed.
    partition = legal_decomps(carrier, kv, ext, factor_menus=[[bn], [1], [1]], placement=placement, masked=False)
    assert any(d.factors == (bn, 1, 1) for d in partition)


# --- 1c: chain_build — the shared-axis reduce_decomp (the FA-2 restructuring) --------


def _build_chain(causal: bool = False):
    dag = iter_dag(_flash_loop(causal=causal))
    knobs = {MAP_N_THREAD.name: _S, MAP_M_THREAD.name: 1}
    return chain_build(seed_graph(dag, kernel_name="flash"), dag, knobs), dag


def _monoids(body):
    from deplodock.compiler.ir.stmt import Monoid as _M  # noqa: PLC0415

    return [s for s in body.iter() if isinstance(s, _M)]


def test_chain_build_realizes_the_split_carrier_as_scalar_phases():
    """The shared-axis reduce_decomp realizes the twisted carrier through the scalar
    ``ScalarCombiner`` — the same ``combine()`` the warp flash drives at the fragment tier (the split
    into a scalar **stats** fold over the score + a **register-tiled accumulation** ``O[d]`` over the
    value is generated once, inside the combiner). No ``Monoid`` survives in the block: each carried
    state (the 2 stats + the 1 accumulator) is declared once by an ``Init`` and rebound by a
    ``Reassign`` — never re-``Assign``ed (which would shadow the carried value)."""
    from deplodock.compiler.ir.kernel.ir import Reassign  # noqa: PLC0415
    from deplodock.compiler.ir.stmt import Assign, Init  # noqa: PLC0415

    tg, dag = _build_chain()
    block = tg.blocks[0]
    assert not _monoids(block.compute), "the carrier must be realized to scalar phases, not left as Monoids"
    states = set(dag.reduction.carrier.state)  # the 2 stats states (row max / denom) + the 1 accumulator (O)
    assert len(states) == 3
    inits = {s.name for s in block.compute.iter() if isinstance(s, Init)}
    reassigns = {s.name for s in block.compute.iter() if isinstance(s, Reassign)}
    assigns = {s.name for s in block.compute.iter() if isinstance(s, Assign)}
    assert states <= inits, "each carried state is declared once by an Init"
    assert states <= reassigns, "each carried state is rebound by a Reassign"
    assert not (states & assigns), "no carried state is re-Assigned (that would shadow the Init'd value)"


def test_chain_build_puts_the_pv_output_in_registers():
    """The P@V output ``d`` becomes a REGISTER domain axis (the ``O[BM, D]`` accumulator),
    so the register-replication pass shares the score across it instead of recomputing it
    per ``d`` block — the INLINE score edge."""
    tg, dag = _build_chain()
    block = tg.blocks[0]
    binding = tg.schedule.binding
    reg_axes = [a for a in block.domain if binding.get(a.name) is Binding.REGISTER and a.extent.as_static() > 1]
    assert len(reg_axes) == 1, "exactly one (non-degenerate) register axis — the P@V output d"
    assert reg_axes[0].extent.as_static() == _D


def test_chain_build_shares_the_score_across_d():
    """The score (the inner QK^T contraction) is computed ONCE per KV step, in the
    ``d``-invariant prefix — NOT inside the register tile over ``d``. So the inner
    contraction loop sits at the KV-stream scope, above any ``RegisterTile``."""
    tg, _ = _build_chain()
    block = tg.blocks[0]
    # No RegisterTile in the block compute carries the inner QK^T reduce: the score is
    # shared (the register replication keys on the d var, which the score never reads).
    for rt in (s for s in block.compute.iter() if isinstance(s, RegisterTile)):
        assert not any(getattr(s, "is_reduce", False) for s in rt.body.iter()), "the QK^T reduce must not ride the d register tile"


def test_chain_build_degenerates_to_torch_oracle_offline():
    """``chain_build`` only fires for a real carried-contraction chain; a non-chain seed
    raises rather than silently mis-lowering."""
    import pytest  # noqa: PLC0415

    dag = iter_dag(_reduce_loop())
    with pytest.raises(ValueError, match="inner contraction present"):
        chain_build(seed_graph(dag, kernel_name="r"), dag, {})
