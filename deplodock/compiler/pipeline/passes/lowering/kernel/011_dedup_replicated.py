"""Common-subexpression-eliminate the replicated body produced by 010.

After ``010_split_register_axes`` unwraps each ``RegisterTile`` and
replicates its body F× per axis, two ``Load``s or ``Assign``s from
independent N-cells that originally referenced the same N-invariant
value end up structurally identical at the same scope. Folding them
to a single occurrence is exactly what the legacy register-blocked
GEMM builder in ``010_partition_loops`` was hand-rolling out of the
Tile structure: lift the N-invariant cone, share it across F_N cells.
The blocked builder gates on five disqualifiers (FN==1, SPLITK>1, BR>1,
M-mask, fused prologue) — content-agnostic CSE here covers all of
them uniformly, plus accidental dedup opportunities the structural
classifier wouldn't catch (identical user compute, dtype-stamped
Assign equality, etc.).

Two folds, interleaved in source order per scope:

* **Load-CSE** — same ``(input, tuple(e.pretty() for e in index))`` keeps
  the first occurrence and renames downstream uses of dropped names to
  the survivor. Mirrors :func:`loop.fusion.020_dedup_loads`'s matching
  key exactly (Loop-IR analogue, same idea).
* **Assign-CSE** — same ``(op, args, dtype)`` after the running alias
  rewrite has already mapped ``args``. Has to chase Load-CSE because an
  Assign's args may name Loads that were just deduped — but since stmts
  are processed in SSA-topo order (defs precede uses), one forward sweep
  reaches a fixed point: by the time we see an Assign, every arg-defining
  Load is already aliased.

Run order: AFTER ``010_split_register_axes`` (need the replicated body)
and BEFORE ``050_vectorize_loads`` (replicas would otherwise break its
consecutive-Load detection). At this point all Loads are still scalar.

Aliases produced at an outer scope flow into nested scopes (a Load
above a Loop body remains live inside it); the per-scope ``seen``
tables do not — a Load inside a Loop body is not sibling-foldable with
an outer-scope Load, but its name lookup against the outer table is
correct when the index pretty-prints the same.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Assign, Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    new_body = _walk(root.op.body, env={}, parent_alias={})
    if new_body == root.op.body:
        raise RuleSkipped("no duplicate Loads or Assigns to dedup")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body, env: dict[tuple[str, tuple[str, ...]], str], parent_alias: dict[str, str]) -> Body:
    """Single forward pass: Load-CSE + Assign-CSE per scope, recursive
    into nested bodies. ``env`` is the Load name table propagated from
    outer scopes; ``parent_alias`` is the running SSA rename map.

    Both ``env`` and ``parent_alias`` are copied on entry so additions
    inside this scope do not leak to siblings or back upward (Python
    dict-mutation safety)."""
    local = dict(env)
    alias = dict(parent_alias)
    seen_assigns: dict[tuple, str] = {}

    def rename(n: str) -> str:
        return alias.get(n, n)

    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Load) and s.is_scalar:
            s = s.rewrite(rename)
            key = (s.input, tuple(e.pretty() for e in s.index))
            kept = local.get(key)
            if kept is not None:
                alias[s.name] = kept
                continue
            local[key] = s.name
            out.append(s)
            continue
        if isinstance(s, Assign):
            s = s.rewrite(rename)
            key = (s.op, s.args, s.dtype)
            kept = seen_assigns.get(key)
            if kept is not None:
                alias[s.name] = kept
                continue
            seen_assigns[key] = s.name
            out.append(s)
            continue
        nested = s.nested()
        if nested:
            # Rewrite the wrapper's non-body fields (Cond.cond, StridedTile
            # start/step, StageBundle.phase, Stage origins, …) through the
            # current alias, then overwrite bodies with the CSE-walked
            # versions. ``s.rewrite(rename)`` redundantly recurses into the
            # bodies too, but the result is discarded by ``with_bodies`` —
            # net cost is one extra recursive walk per nested block,
            # negligible vs the savings on large replicated towers.
            walked = tuple(_walk(b, local, alias) for b in nested)
            out.append(s.rewrite(rename).with_bodies(walked))
        else:
            out.append(s.rewrite(rename))
    return Body(out)
