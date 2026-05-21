"""Cross-CTA split-K — split the K-chunk loop into a grid dimension so
multiple CTAs cooperate on each ``(M, N)`` output tile and atomic-add
their partial sums.

Runs after ``002_chunk_matmul_k`` (which produced the
``Loop(K_o, body=(Loop(K_i, reduce, ...),))`` chunk structure) and
before ``005_blockify_launch`` (so the new ``K_split`` axis enters
``Tile.axes`` and is partitioned to ``BIND_BLOCK`` like any other free
output axis).

Activation:

- ``DEPLODOCK_SPLITK`` env var > 1 — explicit user override.
- Otherwise, ``tuning.auto_splitk`` picks a value targeting ~3 waves
  per SM. Returns 1 (and the pass skips) when the M-N grid already
  fills the GPU.

Refused when:

- ``K_o.extent % splitK != 0`` (would require boundary handling).
- The trailing ``Write`` value isn't directly the matmul ``Accum`` and
  the body has more than a simple linear epilogue chain we can wrap in
  a ``Cond(K_split == 0, ...)``.

Epilogue handling:

For a body of the form ``Loop(K_o, …) ⊕ Write(out, acc)`` the rewrite
just splits ``K_o`` and marks the ``Write`` as ``reduce_op=add``.

For ``Loop(K_o, …) ⊕ <epilogue stmts producing v> ⊕ Write(out, v)``
(e.g. ``k_add_5_reduce``: down_proj + residual add), every CTA
contributes its partial sum via an ``atomic_add(out, acc)`` outside
the ``Cond``, and ``Cond(K_split == 0, atomic_add(out, residual))``
folds the epilogue value (excluding the already-accumulated ``acc``)
in once. This works when the epilogue is exactly one extra additive
term — the most common shape.

Rewrite::

    Loop(K_o:K/BK, body=(Loop(K_i:BK, reduce, B[K → K_o*BK + K_i]),))
    Write(out, idx, acc)
    →
    # K_split lifted into Tile.axes as outermost free axis (→ BLOCK)
    Loop(K_o_new:K/(splitK*BK), body=(Loop(K_i:BK, reduce, B[K_o →
            K_split*K_o_per_split + K_o_new]),))
    Write(out, idx, acc, reduce_op=add)

Output buffer must be zero-initialized (deplodock CUDA backend already
zeros allocations — see ``program.py``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, Axis, BoundAxis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Tile, Write
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile
from deplodock.compiler.tuning import BodyInfo, auto_splitk

PATTERN = [Pattern("root", TileOp)]


def _logical_output_extents(tile: Tile) -> tuple[int, ...]:
    from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD  # noqa: PLC0415

    extents: list[int] = []
    i = 0
    while i < len(tile.axes):
        ba = tile.axes[i]
        ext = int(ba.axis.extent)
        if ba.bind == BIND_BLOCK and i + 1 < len(tile.axes) and tile.axes[i + 1].bind == BIND_THREAD:
            extents.append(ext * int(tile.axes[i + 1].axis.extent))
            i += 2
            continue
        extents.append(ext)
        i += 1
    return tuple(sorted(extents, reverse=True))


_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)

SPLITK = Knob("SPLITK", KnobType.INT, hints=_SPLITK_CANDIDATES, help="Cross-CTA K-split factor (1 = no split)")


def rewrite(root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body, root.op.knobs)
    if new_body is None:
        raise RuleSkipped("rewrite helper returned no change")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(body, parent_knobs):
    idx, tile = single_tile(body)
    if tile.block_axes:
        raise RuleSkipped("Tile already partitioned — must run before 005")

    # Idempotence guard: planner stamps SPLITK_BLOCK on K_s (lifted by
    # tileify into tile.axes with BIND_BLOCK) when it does the σ-split
    # itself. In that case the σ-split is done; nothing left for 003.
    if any(ba.axis.name.endswith("_s") and ba.bind == BIND_BLOCK for ba in tile.axes):
        raise RuleSkipped("planner already did the SPLITK σ-split (K_s in tile.axes)")

    # Find the chunked matmul Loop to learn K_o.extent for the picker.
    k_outer = next((s for s in tile.body if isinstance(s, Loop) and _is_chunked_matmul(s)), None)
    if k_outer is None:
        raise RuleSkipped("no chunked matmul Loop in tile body")
    # Planner-driven path: ``000_partition_planner`` stamps SPLITK on
    # the parent LoopOp's knobs. Honor that stamp deterministically.
    # In env=0 (no planner) we fall back to the local auto_splitk pick.
    if "SPLITK" in parent_knobs:
        splitk = int(parent_knobs["SPLITK"])
    else:
        body_info = BodyInfo.of(tile.body)
        output_extents = _logical_output_extents(tile)
        thread_extents = tuple(int(ba.axis.extent) for ba in tile.axes)
        splitk = auto_splitk(output_extents, body_info, int(k_outer.axis.extent), thread_extents)
    if splitk <= 1:
        raise RuleSkipped(f"splitK={splitk} (planner / heuristic elected no split)")

    new_tile = _split_tile(tile, splitk)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _split_tile(tile: Tile, splitk: int) -> Tile | None:
    """Locate the K_o + Write pair, rewrite K_o into K_split outer +
    K_o_inner inner, lift K_split into tile.axes. Wraps any epilogue
    stmts in ``Cond(K_split == 0, ...)`` so only one CTA contributes
    them; every CTA atomic-adds its accumulator partial."""
    stmts = list(tile.body)
    k_idx = next((i for i, s in enumerate(stmts) if isinstance(s, Loop) and _is_chunked_matmul(s)), None)
    if k_idx is None:
        raise RuleSkipped("no chunked matmul Loop in tile body")

    write_idx = next((i for i, s in enumerate(stmts) if isinstance(s, Write)), None)
    if write_idx is None:
        raise RuleSkipped("no Write in tile body")
    if write_idx <= k_idx:
        raise RuleSkipped(f"Write at idx {write_idx} not after K_o at idx {k_idx}")

    k_outer = stmts[k_idx]
    K_o_extent = int(k_outer.axis.extent)
    if K_o_extent % splitk != 0:
        raise RuleSkipped(f"K_o.extent={K_o_extent} not divisible by splitK={splitk}")
    K_o_per_split = K_o_extent // splitk
    if K_o_per_split == 0:
        raise RuleSkipped(f"splitK={splitk} >= K_o.extent={K_o_extent}")

    K_split = Axis(f"{k_outer.axis.name}_split", splitk)
    K_o_new = Axis(k_outer.axis.name, K_o_per_split)
    sigma = Sigma({k_outer.axis.name: Var(K_split.name) * Literal(K_o_per_split, "int") + Var(K_o_new.name)})
    new_inner = tuple(s.rewrite(_id, sigma) for s in k_outer.body)
    new_k_loop = Loop(axis=K_o_new, body=new_inner)

    write = stmts[write_idx]
    prelude_stmts = stmts[:k_idx]
    epilogue_stmts = stmts[k_idx + 1 : write_idx]  # may be empty

    if not epilogue_stmts and write.value == _find_accum_name(new_k_loop):
        # Plain matmul — every CTA atomic-adds its acc partial.
        new_write = replace(write, reduce_op=ElementwiseImpl("add"))
        head_stmts = list(prelude_stmts) + [new_k_loop, new_write]
    else:
        acc_name = _find_accum_name(new_k_loop)
        if acc_name is None:
            raise RuleSkipped("could not locate the Accum SSA name inside K_o body")
        head_stmts = _rewrite_epilogue(prelude_stmts, write, epilogue_stmts, new_k_loop, acc_name, K_split)
        if head_stmts is None:
            raise RuleSkipped("epilogue isn't a split-K-safe shape (linear-multiplicative or acc + independent-load)")

    new_stmts = head_stmts + stmts[write_idx + 1 :]
    # Lift K_split as outermost axis with the same bind as its siblings
    # (tileify defaults them all to BIND_THREAD). 005_blockify_launch
    # binds the outermost free axes to BLOCK, putting K_split into the
    # grid where each CTA handles one K-chunk.
    bind = tile.axes[0].bind
    new_axes = (BoundAxis(axis=K_split, bind=bind),) + tile.axes
    return Tile(axes=new_axes, body=tuple(new_stmts))


def _find_accum_name(k_o_loop: Loop) -> str | None:
    """Return the SSA target of the innermost Accum in the chunked
    matmul Loop, or None if not found."""
    for s in k_o_loop.body.iter():
        if isinstance(s, Accum):
            return s.name
    return None


def _rewrite_epilogue(prelude_stmts, write, epilogue_stmts, new_k_loop, acc_name, K_split):
    """Two split-K-safe epilogue shapes are recognized:

    1. **Linear-multiplicative**: ``Write.value`` is reachable from
       ``acc`` only through ``multiply`` ops whose other operand is
       acc-independent (e.g. ``acc * silu(gate)``). Then
       ``sum_i (c * a_i) = c * sum_i a_i`` distributes — every CTA
       computes its own ``v = c * partial_acc`` and ``atomic_add``s.
       Emit: keep epilogue as-is, mark Write as ``reduce_op=add``.

    2. **Linear-additive (residual)**: ``Assign(v, add, r, acc); Write(v)``
       where ``r`` is loaded somewhere — either earlier in the
       epilogue, or hoisted into the prelude (``stmts[:k_idx]``) when
       the residual Load is K-independent and the lifter pulls it out
       above the K-loop. Then ``sum_i a_i + r`` doesn't distribute —
       only one CTA contributes ``r``. Emit: always-on
       ``atomic_add(out, acc)`` plus ``Cond(K_split == 0,
       atomic_add(out, r))``. The hoisted Load stays in the prelude
       (every CTA executes it cheaply); only the Write is conditional.

    Returns the list of head stmts (everything up to and including the
    rewritten Write/Cond), or ``None`` if neither pattern matches."""
    # Try multiplicative-only chain first (cheap, common — k_mul_8_reduce).
    if _is_linear_multiplicative_chain(epilogue_stmts, acc_name, write.value):
        new_write = replace(write, reduce_op=ElementwiseImpl("add"))
        return list(prelude_stmts) + [new_k_loop] + list(epilogue_stmts) + [new_write]

    # Fall back to the additive-residual pattern (k_add_5_reduce). Accept
    # the residual Load whether it sits in the epilogue or has been
    # hoisted into the prelude (K-independent residual loads usually are).
    residual = _extract_simple_residual(prelude_stmts, epilogue_stmts, acc_name, write.value)
    if residual is None:
        return None
    residual_name, residual_in_epilogue = residual
    cond_write = Write(output=write.output, index=write.index, value=residual_name, reduce_op=ElementwiseImpl("add"))
    always_write = Write(output=write.output, index=write.index, value=acc_name, reduce_op=ElementwiseImpl("add"))
    if residual_in_epilogue:
        # The Load lives in the epilogue position — move it into the Cond
        # body so the cond_write's referent stays in scope (and only one
        # CTA pays the load).
        residual_load = next(s for s in epilogue_stmts[:-1] if hasattr(s, "name") and s.name == residual_name)
        cond_body = (residual_load, cond_write)
    else:
        # Load was hoisted into the prelude — every CTA already has it
        # in scope; just guard the Write.
        cond_body = (cond_write,)
    cond = Cond(
        cond=BinaryExpr("==", Var(K_split.name), Literal(0, "int")),
        body=Body(cond_body),
        else_body=Body(()),
    )
    return list(prelude_stmts) + [new_k_loop, always_write, cond]


def _is_linear_multiplicative_chain(epilogue_stmts: list, acc_name: str, write_value: str) -> bool:
    """Walk the epilogue forward, propagating an ``acc_dep`` SSA-name
    set. A stmt is split-K-safe iff it's:

    - acc-independent (no deps in ``acc_dep``), OR
    - an ``Assign(v, multiply, a, b)`` where exactly one of ``a, b``
      is in ``acc_dep`` (the other is acc-independent).

    Returns True iff every epilogue stmt is safe and ``write_value`` ∈
    ``acc_dep`` (i.e. the Write's value chain back to ``acc``)."""
    from deplodock.compiler.ir.stmt import Assign  # noqa: PLC0415

    acc_dep = {acc_name}
    for s in epilogue_stmts:
        deps = set(s.deps())
        touches_acc = bool(deps & acc_dep)
        if not touches_acc:
            continue
        if not isinstance(s, Assign):
            return False
        if s.op.name != "multiply":
            return False
        if len(s.args) != 2:
            return False
        a, b = s.args
        a_dep = a in acc_dep
        b_dep = b in acc_dep
        if a_dep == b_dep:  # both deps or both indep — both indep can't happen since touches_acc
            return False
        acc_dep.add(s.name)
    return write_value in acc_dep


def _extract_simple_residual(prelude_stmts, epilogue_stmts: list, acc_name: str, write_value: str):
    """Match an additive-residual epilogue: ``Assign(v, add, ld, acc);
    Write(v)`` where ``ld`` is a ``Load`` either earlier in
    ``epilogue_stmts`` or hoisted into ``prelude_stmts``. Returns
    ``(residual_name, in_epilogue)`` or ``None`` on any deviation
    (non-add op, indirect deps, residual not actually loaded, etc.)."""
    from deplodock.compiler.ir.stmt import Assign, Load  # noqa: PLC0415

    if not epilogue_stmts:
        return None
    asn = epilogue_stmts[-1]
    if not isinstance(asn, Assign):
        return None
    if asn.name != write_value:
        return None
    if asn.op.name != "add":
        return None
    if acc_name not in asn.args or len(asn.args) != 2:
        return None
    other = next(a for a in asn.args if a != acc_name)
    # Any earlier epilogue stmts must just be the residual Load itself.
    earlier = epilogue_stmts[:-1]
    if any(not isinstance(s, Load) for s in earlier):
        return None
    if any(s.name == other for s in earlier if isinstance(s, Load)):
        return (other, True)
    # Residual Load may have been hoisted to the prelude.
    if any(isinstance(s, Load) and s.name == other for s in prelude_stmts):
        return (other, False)
    return None


def _is_chunked_matmul(loop: Loop) -> bool:
    """002's output: outer Loop whose body is exactly one inner reduce-Loop."""
    if loop.is_reduce:
        return False
    inner = [s for s in loop.body if isinstance(s, Loop)]
    if len(inner) != 1 or len(loop.body) != 1:
        return False
    return inner[0].is_reduce


def _id(name: str) -> str:
    return name
