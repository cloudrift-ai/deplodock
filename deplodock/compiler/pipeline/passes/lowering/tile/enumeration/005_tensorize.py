"""Tensorize pass (fork) — the warp-tier ``atomize`` atom-vs-scalar choice.

``plans/tile-ir-block-dag.md`` R4: the first warp-tier fork. It offers each
:class:`~deplodock.compiler.ir.tile.ir.Atom` the kernel admits (the gate in
``enumeration/_atom.eligible_atoms``) **plus the scalar tier** — warp variants
ranked first (``ldmatrix``/``mma.sync`` outrank scalar register tiles), scalar
last as the always-available fallback. An atom branch stamps ``MMA=<kind>`` (no
body move yet — the warp build runs once the geometry knobs are pinned, at
``009_warp_build``); the scalar branch leaves the op untouched so the scalar
passes (``010``/``020``/``030``) fire and ``040`` seals ``MMA=0``.

``DEPLODOCK_MMA`` collapses the fork: ``0`` (falsy) → scalar only (``RuleSkipped``
→ the scalar chain runs); a kind name → that atom only (pin); unset / truthy →
auto-enumerate every eligible atom + scalar. A non-matmul / ineligible kernel
has no eligible atoms, so the pass skips and the scalar chain runs.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom, mma_decode
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import eligible_atoms
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_M_THREAD, MAP_N_THREAD, TC_ATOM

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.SEMIRING or TC_ATOM.name in op.knobs or mma_atom(op.knobs) is not None:
        raise RuleSkipped("tensorize applies once, to a SEMIRING seed (warp tier not yet decided)")
    enabled, pinned = mma_decode(TC_ATOM.raw())
    if not enabled:
        raise RuleSkipped("DEPLODOCK_MMA disabled — scalar tier only")
    # An explicit scalar-tier THREAD-knob pin (``DEPLODOCK_BN`` / ``DEPLODOCK_BM``) is
    # warp-foreign — it signals scalar-tier intent — so it suppresses the
    # default-on warp offer (the user is pinning the scalar register-tile geometry).
    # An explicit ``DEPLODOCK_MMA=<kind>`` pin still wins (handled below).
    if pinned is None and (MAP_N_THREAD.raw() or MAP_M_THREAD.raw()):
        raise RuleSkipped("scalar THREAD knob pinned — scalar tier only")

    def dtype_of(buf: str):
        b = op.buffers.get(buf)
        return b.dtype if b is not None else None

    atoms = eligible_atoms(op.dag, compute_capability=ctx.compute_capability, dtype_of=dtype_of)
    if pinned is not None:
        atoms = [a for a in atoms if a.name == pinned]
    if not atoms:
        raise RuleSkipped("no eligible matmul atom — scalar tier only")

    # Warp branches first (option-0 = the fastest eligible atom), then the scalar
    # fallback — unless ``DEPLODOCK_MMA=<kind>`` pinned a single atom (no fallback).
    out = [replace(op, knobs={**op.knobs, TC_ATOM.name: a.name}) for a in atoms]
    if pinned is None:
        out.append(op)  # the scalar tier (no MMA key — the scalar chain + 040 seal it)
    return out
