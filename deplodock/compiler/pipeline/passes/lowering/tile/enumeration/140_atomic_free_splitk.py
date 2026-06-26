"""Atomic-free split-K — the ``partition_reduce`` combine, as a structural fork.

A cross-CTA split-K matmul (the ``K_s`` GRID
partition the reduce-decomp body move binds when ``SPLITK > 1``) combines its
per-partition partials one of two ways; this pass forks between them:

- ``NOATOMIC=False`` (the default) — each CTA ``atomicAdd``s its partial into the
  output (``K_s`` stays out of the Write index ⇒ ``escape_analysis`` emits the atomic).
  The op is left unchanged, only tagged.
- ``NOATOMIC=True`` — a **structural** split (a kernel-set change): the matmul writes
  its partial into a workspace ``partial[K_s, M, N]`` (``K_s`` prepended to the Write
  index ⇒ a plain store), and a sibling **combine kernel**
  (``_partition.additive_reduce_tilegraph``) folds the ``K_s`` axis into the original
  output. Returned as a two-node ``Graph`` fragment (matmul → workspace → reduce),
  which the engine splices; ``_is_structural_option`` classifies it structural.

This succeeds the deleted ``017_atomic_free_splitk``: same ``NOATOMIC`` BOOL fork and
Write-retarget, now over the block-DAG ``TileGraph`` (the fork runs once the matmul is
fully tiled, after ``120_stage``) instead of the legacy ``TileOp`` tower. Idempotent:
re-running on an op whose ``knobs`` already names ``NOATOMIC`` skips. The warp / MMA
tier is ``SPLITK=1`` today (no cross-CTA split — R4), so this fires only on the scalar
``SEMIRING`` matmul.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.dim import to_dim
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Accum, Body, Stmt, Write
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType, mma_atom
from deplodock.compiler.pipeline.passes.lowering._predicates import atomic_finalize_legal
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _build
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._partition import deferred_combine_tilegraph, reduce_tilegraphop
from deplodock.compiler.tensor import Tensor

PATTERN = [Pattern("root", TileGraphOp)]

# BOOL knob — the cross-CTA stage's **finalize policy** (the one tunable cell of the combine
# array; the intra-CTA folds are derived, the widths already tuned). False = the in-place
# ``Fold.ATOMIC`` finalize (codegen atomicAdd); True = the deferred ``kernel_boundary`` fold (a
# fresh combine kernel). ATOMIC's legality (additive ``Accum`` + atomicAdd dtype) is checked via
# ``_combine.atomic_finalize_legal`` in ``rewrite``; deferred is always legal. Per-shape pin via
# ``DEPLODOCK_NOATOMIC=1``.
NOATOMIC = Knob(
    "NOATOMIC",
    KnobType.BOOL,
    hints=(False, True),
    help="Replace SPLITK > 1's atomicAdd output with a workspace + sibling reduce kernel",
    aliases=("ATOMIC_FREE_SPLITK",),
    off=False,
)


def _rewrite_writes_to_workspace(stmts: tuple[Stmt, ...], *, out_name: str, workspace_name: str, k_s_name: str) -> tuple[Stmt, ...]:
    """Recurse through every nested body and redirect every ``Write`` whose ``output``
    is the kernel's output to the workspace, prepending ``Var(k_s_name)`` to its index
    (so ``K_s`` enters the index ⇒ ``atomic_axes`` shrinks to ∅ ⇒ a plain store). The
    Writes live deep (inside ``RegisterTile > SerialTile``, possibly a ``Cond``);
    recurse generically via ``Stmt.nested()`` / ``Stmt.with_bodies(...)``."""
    new_stmts: list[Stmt] = []
    for s in stmts:
        bodies = s.nested()
        if bodies:
            new_bodies = tuple(
                Body(_rewrite_writes_to_workspace(tuple(b), out_name=out_name, workspace_name=workspace_name, k_s_name=k_s_name))
                for b in bodies
            )
            if new_bodies != bodies:
                s = s.with_bodies(new_bodies)
        if isinstance(s, Write) and s.output == out_name:
            s = replace(s, output=workspace_name, index=(Var(k_s_name), *s.index))
        new_stmts.append(s)
    return tuple(new_stmts)


def _build_fragment(match, root: Node, op: TileGraphOp, k_s_name: str, splitk: int, carrier: Accum) -> Graph:
    """The ``NOATOMIC=True`` fragment: the matmul writing to ``partial[K_s, M, N]`` +
    the sibling deferred-finalize combine kernel folding ``K_s`` into the output. The combine
    is built carrier-generically (``_partition.deferred_combine_tilegraph``) — for the additive
    matmul ``carrier`` the 1-component ``Accum`` sum, the trivial instantiation of the same
    cross-partition fold a twisted ``Monoid`` (online-softmax / flash) would take."""
    out_shape = root.output.shape
    if len(out_shape) != 2 or not all(d.is_static for d in out_shape):
        raise RuleSkipped(f"atomic-free split-K expects a 2D static matmul output, got shape={out_shape}")
    m_extent, n_extent = out_shape[0].as_static(), out_shape[1].as_static()
    dtype = root.output.dtype
    out_name = root.output.name

    # Rewire the matmul body's output Writes to the workspace.
    block = op.tilegraph.blocks[0]
    new_compute = _rewrite_writes_to_workspace(
        tuple(block.compute), out_name=out_name, workspace_name=f"{root.id}__partial", k_s_name=k_s_name
    )
    if tuple(new_compute) == tuple(block.compute):
        raise RuleSkipped("no matmul output Write found to rewire")
    workspace_name = f"{root.id}__partial"
    new_tg = replace(op.tilegraph, blocks=(replace(block, compute=Body(new_compute)),))
    matmul_variant = replace(op, tilegraph=new_tg, knobs={**op.knobs, NOATOMIC.name: True})

    reduce_op = reduce_tilegraphop(
        deferred_combine_tilegraph(
            carrier,
            workspaces=(workspace_name,),
            out_name=out_name,
            s_extent=splitk,
            m_extent=m_extent,
            n_extent=n_extent,
            dtype=dtype,
            name=f"{out_name}__reduce",
        ),
        extra_knobs={NOATOMIC.name: True},
    )

    frag = Graph()
    for inp_id in dict.fromkeys(root.inputs):
        if inp_id in frag.nodes:
            continue
        inp = match.graph.nodes.get(inp_id)
        shape = inp.output.shape if inp is not None else ()
        in_dtype = inp.output.dtype if inp is not None else dtype
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, in_dtype), node_id=inp_id)
    workspace_id = frag.add_node(
        matmul_variant, list(root.inputs), Tensor(workspace_name, (to_dim(splitk), *out_shape), dtype), node_id=workspace_name
    )
    reduce_id = frag.add_node(reduce_op, [workspace_id], Tensor(out_name, out_shape, dtype), node_id=root.id)
    frag.outputs = [reduce_id]
    return frag


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.tilegraph is None or not op.tilegraph.blocks[0].domain:
        raise RuleSkipped("atomic-free split-K runs on a fully-tiled matmul (still a logical seed)")
    if op.algebra is not AlgebraKind.SEMIRING or mma_atom(op.knobs) is not None:
        raise RuleSkipped("atomic-free split-K applies to the scalar SEMIRING matmul tier")
    if NOATOMIC.name in op.knobs:
        raise RuleSkipped("NOATOMIC already decided (idempotence)")
    rk = op.knobs.get(fam.reduce_key(op.dag.k_node.loop.axis.name))
    splitk = fam.dec_reduce(rk).cta if rk is not None else 1
    if splitk <= 1:
        raise RuleSkipped("no split-K (SPLITK = 1) — atomic-free is moot")
    k_s = _build._k_s_axis(op.dag, op.knobs, op.target_names)
    if k_s is None:
        raise RuleSkipped("no split-K partition axis")

    # The cross-CTA stage's finalize policy: NOATOMIC=False is the in-place ``Fold.ATOMIC``
    # finalize, NOATOMIC=True the deferred ``kernel_boundary`` (workspace + sibling reduce kernel).
    # ATOMIC is legal only for an additive ``Accum`` over an atomicAdd-capable dtype
    # (``_predicates.atomic_finalize_legal``) — always true here (the SEMIRING matmul's combine is a
    # plain sum), but the gate is the explicit legality M5's twisted (non-additive ``Monoid``)
    # split-KV finalize will fail, falling back to deferred.
    accums = tuple(Body.coerce(op.tilegraph.blocks[0].compute).iter_of_type(Accum))
    if not accums:
        raise RuleSkipped("no reduce carrier — nothing to finalize across the split-K partitions")
    atomic_legal = atomic_finalize_legal(accums[0], root.output.dtype)

    candidates = NOATOMIC.narrow(NOATOMIC.hints)
    if not atomic_legal:
        candidates = [c for c in candidates if c is True]  # ATOMIC illegal ⇒ deferred-only (always legal)
    if not candidates:
        raise RuleSkipped("NOATOMIC pin doesn't match any candidate (or pins the illegal ATOMIC finalize)")
    variants: list = []
    for use_atomic_free in candidates:
        if use_atomic_free:
            variants.append(_build_fragment(match, root, op, k_s.name, splitk, accums[0]))
        else:
            variants.append(replace(op, knobs={**op.knobs, NOATOMIC.name: False}))
    return variants
