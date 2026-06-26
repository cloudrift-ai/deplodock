"""Tensorize pass (fork) — the warp-tier ``atomize`` atom-vs-scalar choice.

``plans/tile-ir-block-dag.md`` R4: the first warp-tier fork. It offers each
:class:`~deplodock.compiler.ir.tile.ir.Atom` the kernel admits (the gate in
``enumeration/_atom.eligible_atoms``) **plus the scalar tier** — warp variants
ranked first (``ldmatrix``/``mma.sync`` outrank scalar register tiles), scalar
last as the always-available fallback. An atom branch stamps ``MMA=<kind>`` (no
body move yet — the warp build runs once the geometry knobs are pinned, at
``050_warp_build``); the scalar branch leaves the op untouched so the scalar
passes (``060``/``090``/``100``) fire and ``110`` seals ``MMA=0``.

``DEPLODOCK_MMA`` collapses the fork: ``0`` (falsy) → scalar only (``RuleSkipped``
→ the scalar chain runs); a kind name → that atom only (pin); unset / truthy →
auto-enumerate every eligible atom + scalar. A non-matmul / ineligible kernel
has no eligible atoms, so the pass skips and the scalar chain runs.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom, mma_decode
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import eligible_atoms

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    cell = fam.atom_key(fam.MATMUL_CELL)
    if op.algebra is not AlgebraKind.SEMIRING or cell in op.knobs or mma_atom(op.knobs) is not None:
        raise RuleSkipped("tensorize applies once, to a SEMIRING seed (warp tier not yet decided)")
    enabled, pinned = mma_decode(fam.atom_raw(fam.MATMUL_CELL))
    if not enabled:
        raise RuleSkipped("DEPLODOCK_MMA disabled — scalar tier only")
    # An explicit *legacy* scalar-tier THREAD pin (``DEPLODOCK_BN`` / ``DEPLODOCK_BM``)
    # signals scalar-tier intent — so it suppresses the default-on warp offer (the user
    # is pinning the scalar register-tile geometry). A native ``SPLIT@<axis>`` pin does
    # NOT (its tier is the cell's ``ATOM`` — pin ``ATOM=scalar`` / ``MMA=0`` for scalar).
    # An explicit ``DEPLODOCK_MMA=<kind>`` pin still wins (handled below).
    if pinned is None and (config.knob_raw("BN") or config.knob_raw("BM")):
        raise RuleSkipped("legacy scalar THREAD knob pinned — scalar tier only")

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
    out = [replace(op, knobs={**op.knobs, cell: a.name}) for a in atoms]
    if pinned is None:
        out.append(op)  # the scalar tier (no ATOM@out key yet — the scalar chain + 110 seal it)
    return out
