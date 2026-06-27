"""Cross-CTA finalize — the carrier-generic ``partition_reduce`` combine, as a structural fork.

Any reduction that splits its contraction axis across CTAs (the ``K_s`` GRID partition the
reduce-decomp / ``monoid_build`` body move binds when ``cta > 1``) combines its per-partition
partials one of two ways; this pass picks the cross-CTA combine stage's **finalize fold**,
encoded in the ``REDUCE@<axis>`` codec's ``c`` field (``c<cta>a`` = ATOMIC, ``c<cta>k`` =
deferred KERNEL — see ``_families``). The fork fires on ANY fully-tiled scalar op with
``cta > 1`` — a SEMIRING matmul (the additive 1-component carrier) OR a MONOID reduce (a plain
``Accum`` sum today; a twisted ``(m, l, O)`` flash carrier at the split-KV milestone). The
matmul is no longer special — it is the degenerate 1-component instantiation of the same
carrier-generic producer + combine:

- **ATOMIC** (``c<cta>a``, the default) — each CTA ``atomicAdd``s its partial into the
  output (``K_s`` stays out of the Write index ⇒ ``escape_analysis`` emits the atomic). The
  op is left unchanged, only the codec's finalize letter is stamped.
- **KERNEL** (``c<cta>k``) — a **structural** split (a kernel-set change): the op writes its
  partial *state* into a workspace ``partial[K_s, …]`` (``K_s`` prepended to the Write index ⇒
  a plain store), and a sibling **combine kernel** (``_partition.deferred_combine_tilegraph``,
  carrier-generic — additive ``Accum`` here) folds the ``K_s`` axis into the original output.
  Returned as a two-node ``Graph`` fragment (producer → workspace → reduce) the engine splices;
  ``_is_structural_option`` classifies it structural (by Graph-fragment type, not rule name).

The reduce-decomp / ``monoid_build`` move emits a **bare** ``c<cta>`` (finalize pending); this
pass completes it to ``a``/``k`` (idempotent: ``fam.reduce_finalize_decided`` guards re-entry).
ATOMIC's legality (additive ``Accum`` + atomicAdd dtype, ``_predicates.atomic_finalize_legal``)
narrows the offer to KERNEL-only when illegal — the gate the twisted (non-additive ``Monoid``)
split-KV finalize trips (its ``e^{Δm}`` rescale can't be an ``atomicAdd``, so attention has only
a kernel arm by construction). A ``DEPLODOCK_FINALIZE`` pin narrows it explicitly (replacing the
removed ``NOATOMIC`` knob — ``kernel`` ≡ old ``NOATOMIC=1``). The warp / MMA tier is ``cta=1``
today (R4), so this fires only on the scalar tier.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.dim import to_dim
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Monoid, Stmt, Write
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering._predicates import atomic_finalize_legal
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _build
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._partition import deferred_combine_tilegraph, reduce_tilegraphop
from deplodock.compiler.tensor import Tensor

PATTERN = [Pattern("root", TileGraphOp)]


def _with_finalize(knobs: dict, reduce_axis_name: str, finalize: str) -> dict:
    """Re-stamp the ``REDUCE@<reduce_axis>`` value with the cross-CTA ``finalize`` letter
    (``fam.ATOMIC`` / ``fam.KERNEL``), preserving serial/fold/cta/coop — the codec mutation
    that records the cross-CTA combine stage's finalize policy."""
    rk = fam.reduce_key(reduce_axis_name)
    d = fam.dec_reduce(knobs[rk])
    return {**knobs, rk: fam.enc_reduce(serial=d.serial, fold=d.fold, cta=d.cta, coop=d.coop, finalize=finalize)}


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


# --- Twisted Monoid (flash split-KV) producer: emit the partial STATE, defer the finalize ---


def _state_init_ops(compute: Body, state: tuple[str, ...]) -> tuple:
    """The per-state-component identity op (``maximum`` → −inf seed, ``add`` → 0 seed),
    read off the producer body's ``Init`` stmts (one per carried state component). The
    combine seeds each merged-state accumulator from these so the cross-partition fold
    starts at the carrier identity."""
    inits = {s.name: s.op for s in Body.coerce(compute).iter_of_type(Init)}
    missing = [st for st in state if st not in inits]
    if missing:
        raise RuleSkipped(f"cross-CTA Monoid finalize: no Init for carried state {missing}")
    return tuple(inits[st] for st in state)


def _state_finalize(compute: Body, out_name: str, state: tuple[str, ...]) -> tuple[tuple[Assign, ...], str]:
    """The producer's **finalize** — the (single) Assign feeding the output ``Write`` (flash's
    ``res = O_i / l_i``), to be DEFERRED into the combine kernel. Returns ``(finalize, out_value)``:
    the finalize Assigns (over the merged state names — they read the carrier state directly, which
    the combine reproduces) and the value the combine writes. When the output is a bare state
    component (no epilogue), ``finalize`` is empty and ``out_value`` is that component."""
    stmts = list(Body.coerce(compute).iter())
    write = next((s for s in stmts if isinstance(s, Write) and s.output == out_name), None)
    if write is None:
        raise RuleSkipped("cross-CTA Monoid finalize: no output Write to defer")
    val = write.value
    if val in state:
        return (), val
    assigns = {s.name: s for s in stmts if isinstance(s, Assign)}
    fin = assigns.get(val)
    if fin is None or any(a not in state for a in fin.args):
        raise RuleSkipped(f"cross-CTA Monoid finalize: output {val!r} is not a single Assign over the carrier state")
    return (fin,), val


def _rewrite_writes_to_state_workspaces(
    stmts: tuple[Stmt, ...], *, out_name: str, workspace: str, state: tuple[str, ...], k_s_name: str
) -> tuple[Stmt, ...]:
    """Recurse and replace every output ``Write`` with **one ``Write`` per carried state
    component** into a single **packed** workspace ``partial[K_s, c, …]`` (``K_s`` prepended and
    the component index ``c`` next ⇒ a plain store; one kernel can't write N graph buffers, so the
    state rides a component axis). The producer no longer finalizes (``O / l``) — it writes the raw
    ``(m, l, O)`` partial state, and the deferred combine folds + finalizes. The original finalize
    Assign is left dead (no consumer)."""
    new_stmts: list[Stmt] = []
    for s in stmts:
        bodies = s.nested()
        if bodies:
            new_bodies = tuple(
                Body(_rewrite_writes_to_state_workspaces(tuple(b), out_name=out_name, workspace=workspace, state=state, k_s_name=k_s_name))
                for b in bodies
            )
            if new_bodies != bodies:
                s = s.with_bodies(new_bodies)
        if isinstance(s, Write) and s.output == out_name:
            from deplodock.compiler.ir.expr import Literal  # noqa: PLC0415

            new_stmts.extend(
                replace(s, output=workspace, index=(Var(k_s_name), Literal(c, "int"), *s.index), values=(st,)) for c, st in enumerate(state)
            )
            continue
        new_stmts.append(s)
    return tuple(new_stmts)


def _build_fragment(match, root: Node, op: TileGraphOp, k_s_name: str, splitk: int, carrier) -> Graph:
    """The deferred-KERNEL (``c<cta>k``) fragment — **carrier-generic**, one builder for every
    carrier: the producer writes its partial **state** to a workspace ``partial[K_s, …]`` (``K_s``
    prepended ⇒ a plain store), and the sibling combine kernel
    (``_partition.deferred_combine_tilegraph``) folds ``K_s`` into the output.

    - **additive ``Accum``** (matmul split-K / ``sum`` split-reduce) — the 1-component carrier: the
      output Write is retargeted to ``partial[K_s, *out]`` (the ``acc`` value IS the output), the
      combine is the bit-identical ``Σ_s`` fold.
    - **twisted ``Monoid``** (flash split-KV / Flash-Decoding) — the N-component carrier: the
      finalized-output Write is replaced by N writes of the carrier's ``(m, l, O)`` state into one
      packed workspace ``partial[K_s, c, *out]`` (one kernel can't write N graph buffers), and the
      combine merges via the twisted ``combine_states`` (the ``e^{Δm}`` rescale, not a sum) and
      applies the deferred ``O / l`` finalize. The combine half is GPU-verified
      (``test_deferred_finalize_flash_attention_carrier_merges_states``)."""
    out_shape = root.output.shape
    if not out_shape or not all(d.is_static for d in out_shape):
        raise RuleSkipped(f"cross-CTA finalize expects a fully-static output shape, got shape={out_shape}")
    out_extents = tuple(d.as_static() for d in out_shape)
    dtype = root.output.dtype
    out_name = root.output.name
    workspace = f"{root.id}__partial"
    block = op.tilegraph.blocks[0]

    # Per-carrier: how the producer writes its partial state + the combine's finalize args. The
    # additive ``Accum`` is the degenerate 1-component case of the twisted ``Monoid`` state emission.
    if isinstance(carrier, Monoid):
        state = carrier.state
        new_compute = _rewrite_writes_to_state_workspaces(
            tuple(block.compute), out_name=out_name, workspace=workspace, state=state, k_s_name=k_s_name
        )
        ws_shape = (to_dim(splitk), to_dim(len(state)), *out_shape)  # one packed buffer [K_s, n_state, *out]
        finalize, out_value = _state_finalize(block.compute, out_name, state)
        combine_kw = {"init_ops": _state_init_ops(block.compute, state), "finalize": finalize, "out_value": out_value}
    else:  # additive Accum — the single-Write retarget, no init/finalize
        new_compute = _rewrite_writes_to_workspace(tuple(block.compute), out_name=out_name, workspace_name=workspace, k_s_name=k_s_name)
        ws_shape = (to_dim(splitk), *out_shape)
        combine_kw = {}
    if tuple(new_compute) == tuple(block.compute):
        raise RuleSkipped("no output Write found to rewire to the workspace")

    new_tg = replace(op.tilegraph, blocks=(replace(block, compute=Body(new_compute)),))
    producer = replace(op, tilegraph=new_tg, knobs=_with_finalize(op.knobs, op.dag.k_node.loop.axis.name, fam.KERNEL))
    combine = reduce_tilegraphop(
        deferred_combine_tilegraph(
            carrier,
            workspaces=(workspace,),
            out_name=out_name,
            s_extent=splitk,
            out_shape=out_extents,
            dtype=dtype,
            name=f"{out_name}__reduce",
            **combine_kw,
        )
    )

    frag = Graph()
    for inp_id in dict.fromkeys(root.inputs):
        if inp_id in frag.nodes:
            continue
        inp = match.graph.nodes.get(inp_id)
        shape = inp.output.shape if inp is not None else ()
        in_dtype = inp.output.dtype if inp is not None else dtype
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, in_dtype), node_id=inp_id)
    ws_id = frag.add_node(producer, list(root.inputs), Tensor(workspace, ws_shape, dtype), node_id=workspace)
    reduce_id = frag.add_node(combine, [ws_id], Tensor(out_name, out_shape, dtype), node_id=root.id)
    frag.outputs = [reduce_id]
    return frag


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.tilegraph is None or not op.tilegraph.blocks[0].domain:
        raise RuleSkipped("cross-CTA finalize runs on a fully-tiled op (still a logical seed)")
    if mma_atom(op.knobs) is not None:
        raise RuleSkipped("cross-CTA finalize is scalar-tier today (the warp / MMA tier is cta = 1)")
    if op.algebra not in (AlgebraKind.SEMIRING, AlgebraKind.MONOID):
        raise RuleSkipped("cross-CTA finalize applies to a SEMIRING matmul or a MONOID reduce")
    reduce_axis = op.dag.k_node.loop.axis.name
    rk = op.knobs.get(fam.reduce_key(reduce_axis))
    if rk is not None and fam.reduce_finalize_decided(rk):
        raise RuleSkipped("cross-CTA finalize already decided (idempotence)")
    splitk = fam.dec_reduce(rk).cta if rk is not None else 1
    if splitk <= 1:
        raise RuleSkipped("no split-K (cta = 1) — the cross-CTA finalize is moot")
    k_s = _build._k_s_axis(op.dag, op.knobs, op.target_names)
    if k_s is None:
        raise RuleSkipped("no split-K partition axis")

    # The cross-CTA combine stage's finalize fold, stamped into the codec's ``c`` field:
    # ``c<sk>a`` is the in-place ATOMIC finalize (left op-variant), ``c<sk>k`` the deferred
    # KERNEL fold (the structural workspace + combine-kernel splice). ATOMIC is legal only for
    # an additive ``Accum`` over an atomicAdd-capable dtype (``_predicates.atomic_finalize_legal``)
    # — always true here (the SEMIRING matmul's combine is a plain sum), but the gate is the
    # explicit legality the twisted (non-additive ``Monoid``) split-KV finalize will trip. A
    # ``DEPLODOCK_FINALIZE`` pin narrows the offer (the removed ``NOATOMIC`` env pin's successor).
    # The cross-partition **carrier** drives the combine (carrier-generic): a SEMIRING matmul's
    # additive ``Accum`` (read off the body), or a MONOID reduce's carrier off ``dag.reduction``
    # (an additive ``Accum`` for a plain ``sum``, a twisted ``Monoid`` for the flash ``(m, l, O)``
    # — the body ``Accum`` would be the inner QK^T score, not the carrier, so the reduction view
    # is the correct source).
    if op.algebra is AlgebraKind.MONOID and op.dag.reduction is not None:
        carrier = op.dag.reduction.carrier
    else:
        accums = tuple(Body.coerce(op.tilegraph.blocks[0].compute).iter_of_type(Accum))
        if not accums:
            raise RuleSkipped("no reduce carrier — nothing to finalize across the split-K partitions")
        carrier = accums[0]
    atomic_legal = atomic_finalize_legal(carrier, root.output.dtype)

    choices = [fam.ATOMIC, fam.KERNEL] if atomic_legal else [fam.KERNEL]
    pin = fam.pin_finalize(reduce_axis)
    if pin is not None:
        choices = [c for c in choices if c == pin]
    if not choices:
        raise RuleSkipped("DEPLODOCK_FINALIZE pin matches no legal finalize (e.g. pinned the illegal ATOMIC)")
    variants: list = []
    for finalize in choices:
        if finalize is fam.KERNEL:
            variants.append(_build_fragment(match, root, op, k_s.name, splitk, carrier))
        else:
            variants.append(replace(op, knobs=_with_finalize(op.knobs, reduce_axis, fam.ATOMIC)))
    return variants
