"""Ping-pong double-buffering for K-outer staged matmul kernels.

For a Tile body shaped like ``Loop(K_outer, body=[Stage*, reduce])``,
this pass replaces each ``Stage`` with a ``BufferedStage`` carrying
``buffer_count = 2`` and a phase expression ``Var(K_outer_axis) % 2``.
The materializer then doubles the smem allocation and prepends the
phase to both the cooperative-load write and every body Load that
reads from the staged buffer.

The win comes from removing the leading ``__syncthreads`` between
prev-compute and next-load: with ping-pong, consecutive iterations
write to *different* physical smem regions, so the next iteration's
load can issue before the previous iteration's compute fully drains.
The trailing Sync (after the cooperative load, before compute reads)
is preserved.

Trigger:

- Tile body has exactly one ``Tile`` (the standard shape).
- The Tile body has a free Loop ``K_outer`` (extent ≥ 2) whose body
  contains ≥ 1 ``Stage`` and exactly one reduce ``Loop``.
- The reduce body has only ``Load`` / ``Assign`` / ``Accum`` stmts,
  defines ≥ 1 ``Accum``, and no body stmt reads any local Accum
  target's running value (rejects in-loop online-softmax-style merges
  where an Assign reads ``acc`` mid-iteration). ``register_tile`` no
  longer enforces this gate; ``014`` keeps it because its rewrite
  reorders smem visibility relative to the FMA chain and that
  reordering compounds fp32 drift on online-softmax-style bodies.
- Smem budget: ``2 × sum(slab_floats) ≤ _SMEM_BUDGET`` (default 96 KB
  to fit the per-block dynamic smem cap on Ada / Hopper consumer
  parts).

Idempotence: a Stage that's already a ``BufferedStage`` is left alone.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Body, Load, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import BufferedStage, Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import collect_invariant_names, is_matmul_k_outer, single_tile

PATTERN = [Pattern("root", TileOp)]

_BUFFER_COUNT = 2
_SMEM_BUDGET_BYTES = 96 * 1024  # 96 KB


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    new_tile_body = _process_scope(tile.body)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no K-outer matmul Loop eligible for double-buffering within smem budget")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(body: Body) -> Body:
    """Walk the body looking for K-outer Loops eligible for double-buffering.

    Tracks ``invariant_names`` — the set of SSA names defined by
    sibling stmts that precede each K-outer Loop. Cross-loop SSA
    reads that resolve into this set are loop-invariant w.r.t. the
    K-outer being considered (a sibling reduce above produced the
    name; the value is finalized before this K-outer starts), so the
    pass can safely double-buffer the loop. Reads that don't resolve
    locally *and* aren't in the invariant set are still rejected —
    those are the in-loop-online-softmax-style merges the original
    gate guarded against.
    """
    new_body: list[Stmt] = list(body)
    changed = False
    invariant_names: set[str] = set()
    for i, s in enumerate(body):
        if isinstance(s, Loop) and not s.is_reduce and _is_kouter_matmul(s, invariant_names):
            if not any(isinstance(st, BufferedStage) for st in s.body):
                stages = [st for st in s.body if isinstance(st, Stage)]
                if _BUFFER_COUNT * sum(st.smem_bytes for st in stages) <= _SMEM_BUDGET_BYTES:
                    updated = _double_buffer(s)
                    if updated is not None:
                        new_body[i] = updated
                        changed = True
        invariant_names.update(collect_invariant_names(s))
    return tuple(new_body) if changed else body


def _is_kouter_matmul(loop: Loop, invariant_names: set[str]) -> bool:
    """K-outer-matmul predicate: extent ≥ 2, ≥ 1 Stage in body, and a
    pure-matmul reduce body with no in-loop online-softmax-style merge
    (no body stmt reads a local Accum target's running value).

    Cross-loop SSA reads are allowed when the referenced names are in
    ``invariant_names`` — names defined by sibling stmts at the K-outer's
    enclosing scope, before the K-outer Loop. Those values are
    finalized before this K-outer starts, so 015's pipeline rewrite
    can keep them in registers across the steady-state without the
    synchronous-double-buffer + TMA-mbarrier deadlock the original
    gate was designed to avoid.

    Layered on the shared ``is_matmul_k_outer`` (which handles the
    base shape — non-reduce free Loop wrapping a single reduce with
    a pure-compute body that has at least one Accum) plus the
    rule-specific ``extra_gate``.
    """

    def gate(k_outer: Loop, k_inner: Loop) -> bool:
        if int(k_outer.axis.extent) < 2:
            return False
        if not any(isinstance(s, Stage) for s in k_outer.body):
            return False
        # Reject if any non-Accum stmt reads an Accum target's running
        # value (in-loop online-softmax-style merge). Driven by
        # ``Body.deps_of`` so the predicate is a one-liner: each stmt's
        # deps resolve to defining stmts; flag any that resolves to an
        # ``Accum``.
        for c in k_inner.body:
            if isinstance(c, Accum):
                continue
            if any(isinstance(s, Accum) for s in k_inner.body.deps_of(c)):
                return False
        # Cross-loop-deps gate, relaxed: reject only when an unresolved
        # dep is *not* in ``invariant_names``. Names produced by sibling
        # stmts above this K-outer (prior softmax max/sum reduces, hoisted
        # reciprocals, etc.) are loop-invariant and safe to read from
        # smem ring-buffered slabs without phase rotation.
        for c in k_inner.body:
            for d, defstmt in zip(c.deps(), k_inner.body.deps_of(c), strict=False):
                if defstmt is None and d not in invariant_names:
                    return False
        return True

    return is_matmul_k_outer(loop, extra_gate=gate)


def _double_buffer(loop: Loop) -> Loop | None:
    """Promote each ``Stage`` in the loop body to ``BufferedStage`` with
    ``buffer_count=2`` and a shared phase expression, and rewrite each
    body Load that reads from a staged buffer to prepend the phase
    index."""
    phase = Var(loop.axis.name) % Literal(_BUFFER_COUNT, "int")
    staged_names: set[str] = set()
    new_body: list[Stmt] = []
    for s in loop.body:
        if isinstance(s, Stage):
            staged_names.add(s.name)
            new_body.append(
                BufferedStage(
                    name=s.name,
                    buf=s.buf,
                    origin=s.origin,
                    axes=s.axes,
                    addressing=s.addressing,
                    pad=s.pad,
                    buffer_count=_BUFFER_COUNT,
                    phase=phase,
                )
            )
        elif isinstance(s, Loop) and s.is_reduce:
            new_body.append(dc_replace(s, body=s.body.map(_make_load_rewriter(staged_names, phase))))
        else:
            new_body.append(s)
    return dc_replace(loop, body=new_body)


def _make_load_rewriter(staged_names: set[str], phase):
    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.input in staged_names:
            return Load(name=s.name, input=s.input, index=(phase, *s.index))
        return s

    return fn
