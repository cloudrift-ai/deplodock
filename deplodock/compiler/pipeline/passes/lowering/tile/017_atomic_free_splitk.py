"""Atomic-free split-K via a two-kernel decomposition.

Replaces the legacy SPLITK > 1 path's ``atomicAdd``-to-output with a
two-launch pair: the matmul writes its per-K_s partial into a workspace
``partial[S, M, N]`` (``K_s`` now appears in the Write index, so
``Body.coordination.atomic_axes`` returns ``∅`` and codegen emits a
plain store), and a sibling reduce ``TileOp`` sums along ``S`` into the
original output.

The choice is an ``ATOMIC_FREE_SPLITK`` BOOL fork — the autotuner picks
per shape. ``False`` keeps the legacy atomic path; ``True`` emits the
two-kernel pair as a Graph fragment.

Fires after ``015_gate_splitk_residual`` (which gates a fused linear
residual on ``K_s == 0`` — orthogonal to the atomic choice; its rewrite
leaves two Writes inside a ``Cond``, both of which 017 rewires to the
workspace) and before any staging / ring-buffer passes (020+).

The reduce ``TileOp`` is constructed pre-tiled — fixed schedule, no
exposed knobs. It loops one CTA per 16×16 output tile, one thread per
output cell, with the ``S`` axis as a fully-unrolled register-side
``SerialTile`` (no smem, no ``TreeHalve``, no cross-thread
coordination). The schedule is bandwidth-bound at the target shape
(~50 µs theoretical at 1.5 TB/s for 64 MB of partials at 2048³), and
the generic ``100_materialize_tile.py`` handles the body unchanged.

Idempotent: re-running on a TileOp whose ``knobs`` already names
``ATOMIC_FREE_SPLITK`` skips. Non-split-K TileOps stamp
``ATOMIC_FREE_SPLITK=False`` (the decision is recorded for a uniform knob set,
not skipped).

The **MMA / warp tier** (Step 3b of ``plans/atomic-free-monoid-combine.md``) forks
atomic-free too: at this tile stage the C-fragment store is still a plain
``Write(output=c)`` (the fragment ``RegStore`` is lowered later by
``kernel/005_lower_atom_tile``, which reads its output index off that Write), so
the same Write-retarget routes the C fragment into ``workspace[K_s, M, N]`` and the
additive ``Accum``-sum reduce folds it — no codegen ``atomicAdd``. The legacy
atomic arm stays available as the fork's ``False`` branch.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.dim import to_dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Combine, Cond, Init, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import GridTile, SerialTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import find_split_k_axis_name
from deplodock.compiler.tensor import Tensor

PATTERN = [Pattern("root", TileOp)]

# BOOL knob: the autotuner forks between the legacy atomicAdd path
# (False) and the two-kernel atomic-free path (True). Per-shape pin via
# ``DEPLODOCK_ATOMIC_FREE_SPLITK=1`` for deterministic A/B benches.
ATOMIC_FREE_SPLITK = Knob(
    "NOATOMIC",
    KnobType.BOOL,
    hints=(False, True),
    help="Replace SPLITK > 1's atomicAdd output with a workspace + sibling reduce kernel",
    aliases=("ATOMIC_FREE_SPLITK",),
    off=False,
)

# Fixed schedule for the reduce TileOp — bandwidth-bound at any
# realistic shape, so per-shape autotune would only re-discover this.
# 16×16 output tile per CTA, 256 threads, S serial unroll.
_BM_RED = 16
_BN_RED = 16


def _find_k_s_axis(op: TileOp, k_s_name: str) -> Axis:
    """Return the ``K_s`` ``Axis`` (for its extent) from the outer GridTile.

    The planner emits ``K_s`` as a top-level ``GridTile`` axis; pulling
    the ``Axis`` from there gives us the static extent ``S`` we need to
    shape the workspace and the reduce TileOp's serial loop.
    """
    for s in op.body:
        if isinstance(s, GridTile):
            for ax in s.axes:
                if ax.name == k_s_name:
                    return ax
    raise RuleSkipped(f"K_s axis {k_s_name!r} not found at top-level GridTile")


def _rewrite_writes_to_workspace(stmts: tuple[Stmt, ...], *, out_name: str, workspace_name: str, k_s_name: str) -> tuple[Stmt, ...]:
    """Recurse through every nested body and redirect every Write whose
    ``output`` is the kernel's output to the workspace, prepending
    ``Var(k_s_name)`` to its index.

    The Writes live deep — typically inside ``RegisterTile > SerialTile``
    for plain matmul, and inside a ``Cond`` (created by 015) for
    ``matmul_add``. Recurse generically via ``Stmt.nested()`` /
    ``Stmt.with_bodies(...)`` so both shapes get the same rewrite.

    Children first, then this level — body-order independent.
    """
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
            s = Write(
                output=workspace_name,
                index=(Var(k_s_name), *s.index),
                values=s.values,
                value_dtype=s.value_dtype,
            )
        new_stmts.append(s)
    return tuple(new_stmts)


def _build_reduce_tileop(*, workspace_name: str, out_name: str, s_extent: int, m_extent: int, n_extent: int, dtype) -> TileOp:
    """Construct a pre-tiled reduce ``TileOp``.

    Shape::

        GridTile(M_b, N_b)
          ThreadTile(M_t, N_t)
            SerialTile(K_s, unroll)
              p = Load(workspace[K_s, M_b·16 + M_t, N_b·16 + N_t])
              Accum(acc += p)
            Cond(in-bounds, [Write(out[M_b·16 + M_t, N_b·16 + N_t], acc)])

    ``M_b`` / ``N_b`` use ``ceil_div`` extents and a boundary Cond gates
    the Write so a non-divisor M or N still writes within bounds.
    Out-of-bounds entries in the workspace stay at the ``cp.zeros`` init
    value — the Load reads zero, the Accum sums zero, and the Cond
    blocks the OOB write.
    """
    bm_red = _BM_RED
    bn_red = _BN_RED
    m_blocks = -(-m_extent // bm_red)
    n_blocks = -(-n_extent // bn_red)

    # Fresh axes — the reduce kernel is a separate TileOp, so no name
    # collision with the matmul half. Names stay readable.
    K_s = Axis("K_s_red", to_dim(s_extent))
    M_b = Axis("M_b_red", to_dim(m_blocks))
    N_b = Axis("N_b_red", to_dim(n_blocks))
    M_t = Axis("M_t_red", to_dim(bm_red))
    N_t = Axis("N_t_red", to_dim(bn_red))

    m_idx = Var(M_b.name) * Literal(bm_red, "int") + Var(M_t.name)
    n_idx = Var(N_b.name) * Literal(bn_red, "int") + Var(N_t.name)

    reduce_inner: tuple[Stmt, ...] = (
        Load(name="p", input=workspace_name, index=(Var(K_s.name), m_idx, n_idx), dtype=dtype),
        Accum(name="acc", value="p", dtype=F32, axes=(K_s.name,)),
    )

    write = Write(output=out_name, index=(m_idx, n_idx), value="acc", value_dtype=dtype)
    # Boundary Cond — m_extent % bm_red == 0 keeps the predicate trivially true
    # for the divisor case (NVRTC folds it out); guards OOB writes for non-divisor.
    in_bounds = BinaryExpr("&&", BinaryExpr("<", m_idx, Literal(m_extent, "int")), BinaryExpr("<", n_idx, Literal(n_extent, "int")))
    guarded = (Cond(cond=in_bounds, body=Body((write,))),)

    body = (
        GridTile(
            axes=(M_b, N_b),
            body=Body(
                (
                    ThreadTile(
                        axes=(M_t, N_t),
                        body=Body(
                            (
                                SerialTile(axis=K_s, body=Body(reduce_inner), kind="plain", unroll=True),
                                *guarded,
                            )
                        ),
                    ),
                )
            ),
        ),
    )
    return TileOp(body=Body(body), name=f"{out_name}__reduce", knobs={ATOMIC_FREE_SPLITK.name: True})


def build_monoid_reduce_tileop(
    *,
    carrier: Combine,
    init_ops: tuple[ElementwiseImpl, ...],
    workspaces: tuple[str, ...],
    out_name: str,
    s_extent: int,
    m_extent: int,
    n_extent: int,
    dtype,
    finalize: tuple[Assign, ...] = (),
    out_value: str | None = None,
    name: str = "monoid__reduce",
) -> TileOp:
    """Carrier-general cross-partition reduce ``TileOp`` (Step 3a of
    ``plans/atomic-free-monoid-combine.md``) — the monoid sibling of
    :func:`_build_reduce_tileop`.

    Each of the carrier's ``state`` components has its own workspace buffer
    ``workspaces[i]`` (shaped ``[S, M, N]``, holding that partition's state
    component). One thread per ``(m, n)`` output cell seeds its state from
    ``identity`` (``init_ops[i]``'s op-identity per component — ``maximum`` →
    −inf, ``add`` → 0) and serially folds each ``S`` slice via the carrier's
    ``combine_states`` (the state-merges-state monoid op). After the fold, an
    optional ``finalize`` program (Assigns over the merged state — e.g. flash's
    ``res = O / l``) produces the single output value ``out_value`` (default the
    first state component); it is written to ``out_name``. The additive matmul
    case stays on :func:`_build_reduce_tileop`'s ``Accum`` sum (bit-identical);
    this is the path a non-additive carrier (flash split-KV's LSE) takes.
    """
    bm_red, bn_red = _BM_RED, _BN_RED
    m_blocks = -(-m_extent // bm_red)
    n_blocks = -(-n_extent // bn_red)
    K_s = Axis("K_s_red", to_dim(s_extent))
    M_b = Axis("M_b_red", to_dim(m_blocks))
    N_b = Axis("N_b_red", to_dim(n_blocks))
    M_t = Axis("M_t_red", to_dim(bm_red))
    N_t = Axis("N_t_red", to_dim(bn_red))
    m_idx = Var(M_b.name) * Literal(bm_red, "int") + Var(M_t.name)
    n_idx = Var(N_b.name) * Literal(bn_red, "int") + Var(N_t.name)

    # Load each partition's state component, then fold via combine_states. The
    # loaded names are the "other" state operand of as_state_merge.
    others = tuple(f"o_{i}" for i in range(len(carrier.state)))
    loads: tuple[Stmt, ...] = tuple(
        Load(name=others[i], input=workspaces[i], index=(Var(K_s.name), m_idx, n_idx), dtype=dtype) for i in range(len(workspaces))
    )
    fold = replace(carrier.as_state_merge(others), axes=(K_s.name,))
    reduce_inner = (*loads, fold)

    inits: tuple[Stmt, ...] = tuple(Init(name=st, op=init_ops[i], dtype=F32) for i, st in enumerate(carrier.state))
    written = out_value if out_value is not None else carrier.state[0]
    in_bounds = BinaryExpr("&&", BinaryExpr("<", m_idx, Literal(m_extent, "int")), BinaryExpr("<", n_idx, Literal(n_extent, "int")))
    write = Write(output=out_name, index=(m_idx, n_idx), value=written, value_dtype=dtype)
    guarded = (Cond(cond=in_bounds, body=Body((*finalize, write))),)

    body = (
        GridTile(
            axes=(M_b, N_b),
            body=Body(
                (
                    ThreadTile(
                        axes=(M_t, N_t),
                        body=Body((*inits, SerialTile(axis=K_s, body=Body(reduce_inner), kind="plain", unroll=True), *guarded)),
                    ),
                )
            ),
        ),
    )
    return TileOp(body=Body(body), name=name, knobs={ATOMIC_FREE_SPLITK.name: True})


def _build_atomic_free_fragment(match: Match, root: Node, op: TileOp, k_s_axis: Axis, out_name: str) -> Graph:
    """Build the True-branch Graph fragment: matmul writing to workspace
    + sibling reduce TileOp consuming it.

    The matmul TileOp is the original ``op`` with every output ``Write``
    rewired to a workspace name (``K_s`` prepended to the index, so
    ``atomic_axes`` shrinks to ``∅`` and codegen emits a plain store).
    The reduce is the pre-tiled TileOp from :func:`_build_reduce_tileop`.

    Fragment id layout:

    - InputOp aliases for every original matmul input (kept under their
      existing graph ids so the splice resolves them by reference).
    - matmul node: id ``f"{root.id}__partial"`` — the workspace.
    - reduce node: id ``root.id`` — collides with the still-present
      original; the splicer auto-renames to a fresh id and then promotes
      it back to ``root.id`` after the original is removed (see
      ``Graph.splice`` rename-to-friendly-name tail).
    """
    workspace_name = f"{root.id}__partial"
    s_extent = k_s_axis.extent.as_static()
    out_shape = root.output.shape
    if len(out_shape) != 2:
        raise RuleSkipped(f"atomic-free split-K expects a 2D matmul output, got shape={out_shape}")
    if not all(d.is_static for d in out_shape):
        raise RuleSkipped(f"atomic-free split-K expects static output extents, got shape={out_shape}")
    m_extent = out_shape[0].as_static()
    n_extent = out_shape[1].as_static()

    # Rewire the matmul body's output Writes to the workspace.
    new_body = _rewrite_writes_to_workspace(
        tuple(op.body),
        out_name=out_name,
        workspace_name=workspace_name,
        k_s_name=k_s_axis.name,
    )
    if new_body == tuple(op.body):
        raise RuleSkipped("no matmul output Write found to rewire")
    matmul_variant = TileOp(
        body=Body(new_body),
        name=op.name,
        knobs={**op.knobs, ATOMIC_FREE_SPLITK.name: True},
    )

    dtype = root.output.dtype
    reduce_op = _build_reduce_tileop(
        workspace_name=workspace_name,
        out_name=out_name,
        s_extent=s_extent,
        m_extent=m_extent,
        n_extent=n_extent,
        dtype=dtype,
    )

    graph = match.graph
    frag = Graph()
    for inp_id in dict.fromkeys(root.inputs):
        if inp_id in frag.nodes:
            continue
        # InputOp nodes alias the original graph node by id at splice time
        # (see ``Graph.splice``); the Tensor here is a best-effort
        # placeholder. Pull the real shape/dtype from the host graph so the
        # fragment renders + validates cleanly before the splice.
        inp = graph.nodes.get(inp_id)
        if inp is not None:
            shape = inp.output.shape
            dtype = inp.output.dtype
        else:
            shape, dtype = (), F32
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)
    workspace_id = frag.add_node(
        matmul_variant,
        list(root.inputs),
        Tensor(workspace_name, (to_dim(s_extent), *out_shape), dtype),
        node_id=workspace_name,
    )
    reduce_id = frag.add_node(
        reduce_op,
        [workspace_id],
        Tensor(root.output.name, out_shape, dtype),
        node_id=root.id,
    )
    frag.outputs = [reduce_id]
    return frag


def rewrite(ctx: Context, match: Match, root: Node) -> list[TileOp | Graph] | None:
    op: TileOp = root.op
    if ATOMIC_FREE_SPLITK.name in op.knobs:
        raise RuleSkipped("ATOMIC_FREE_SPLITK already decided (idempotence)")

    def _off() -> TileOp:
        """Record the off decision (legacy atomic / not-applicable), body
        unchanged, so the realized config keeps a uniform knob set."""
        return TileOp(body=op.body, name=op.name, knobs={**op.knobs, ATOMIC_FREE_SPLITK.name: False})

    # MMA / warp tier (Step 3b): the C-fragment store is still a tile-level
    # ``Write(output=c)`` at THIS stage — the MMA fragment lowering (RegStore) only
    # happens later at ``kernel/005_lower_atom_tile``, which reads its output index
    # off the Write. So the same Write-retarget that the scalar path uses routes
    # the warp tier's C fragment into ``workspace[K_s, M, N]`` (K_s in the index ⇒
    # ``atomic_axes = ∅`` ⇒ RegStore to the workspace, no atomicAdd), and the
    # additive ``Accum``-sum reduce kernel folds it. No ``is_warp`` early-out.
    k_s_name = find_split_k_axis_name(op)
    if k_s_name is None:
        return _off()  # no split-K (SPLITK = 1) — atomic-free is moot

    candidates = ATOMIC_FREE_SPLITK.narrow(ATOMIC_FREE_SPLITK.hints)
    if not candidates:
        raise RuleSkipped("ATOMIC_FREE_SPLITK pin doesn't match any candidate")

    # The matmul's single output buffer name lives in op.outputs (single
    # entry — TileOps in the matmul shape have one output).
    out_names = tuple(op.outputs)
    if len(out_names) != 1:
        raise RuleSkipped(f"expected a single-output matmul TileOp, got {out_names!r}")
    (out_name,) = out_names

    variants: list[TileOp | Graph] = []
    for use_atomic_free in candidates:
        if use_atomic_free:
            k_s_axis = _find_k_s_axis(op, k_s_name)
            variants.append(_build_atomic_free_fragment(match, root, op, k_s_axis, out_name))
        else:
            # False branch: structurally identical to the input, just
            # tagged with the knob so the cache key distinguishes it
            # from upstream variants and the rule is idempotent.
            variants.append(
                TileOp(
                    body=op.body,
                    name=op.name,
                    knobs={**op.knobs, ATOMIC_FREE_SPLITK.name: False},
                )
            )
    if not variants:
        raise RuleSkipped("no atomic-free split-K variant viable")
    return variants
