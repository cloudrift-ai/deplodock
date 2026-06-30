"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` carries one :class:`~.schedule.Kernel`
— the op-tree node (the *combine*, in ``ir/stmt/algebra``) paired with a kind-free
:class:`~.schedule.TileSchedule` (the *schedule* — the parallel/free axes, the reduce
partition, the grid binding). There is no per-kind kernel/schedule type: the algebra is
read structurally off the axes' :class:`~deplodock.compiler.ir.axis.AxisRole`
(``ops.axis_role``), so MAP / MONOID / SEMIRING all ride the same ``Kernel``. See
:mod:`.schedule`.

The combine lives entirely in the ``op`` wrapper (``ir/stmt/algebra``): a
:class:`~deplodock.compiler.ir.stmt.algebra.Map` whose body is the per-cell loop nest — its reduce
``Loop`` carrying the role (``AxisRole``) + the decoupled ``Carrier`` (the ⊕ algebra). The algebra
is **not stored as a node kind**; the role/carrier are read off the annotated loop where a pass
needs them (``ops.axis_role`` / ``ops.reduce_loop``). ``lower(op)`` is the body verbatim (the
carriers already dissolved into loose folds at recognition).

``op`` / ``schedule`` are **read-only projections** of ``kernel`` (preserving
``keys.op_cache_key``, ``dialect_of``, ``pretty_body``, and the materialize call
sites), so the SAME op and the SAME materializer extend across kernel kinds —
only the carrier (the ⊕) and the schedule's reduce partition change.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.tile.schedule import Kernel, Placement

#: Back-compat alias: the old two-field ``Schedule`` (``free`` / ``grid``) is now
#: :class:`~.schedule.Placement`. Kept re-exported during the transition.
Schedule = Placement


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds exactly the :class:`~.schedule.Kernel` (op-tree node + typed schedule) and a
    ``name`` — not a pre-lowered body and not a ``BodyOp``. The per-cell loop-IR body is
    generated at materialize time by ``lower(op)``, and a bare reduction's output ``Write``
    is glue generated there too (from the schedule's grid + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from
    the base :meth:`Op.populate_io` (graph edges) — no body walk.

    ``op`` and ``schedule`` are read-only properties projecting ``kernel.op`` /
    ``kernel.schedule`` (``None`` / empty :class:`~.schedule.Placement` for a placeholder
    node with no kernel). ``pretty_body`` lowers ``op`` on demand for dumps (the cache key
    lowers it likewise in ``search/keys``)."""

    kernel: Kernel | None = None
    name: str = ""

    @property
    def op(self):
        """The kernel op — a :class:`~deplodock.compiler.ir.stmt.algebra.Map` wrapping the per-cell
        annotated loop nest; ``None`` for a placeholder node carrying no kernel."""
        return self.kernel.op if self.kernel is not None else None

    @property
    def schedule(self):
        """The kernel's typed schedule; an empty :class:`~.schedule.Placement` for a
        placeholder node carrying no kernel (so ``schedule.is_mapped`` stays well-defined)."""
        return self.kernel.schedule if self.kernel is not None else Placement()

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering. Prefixes the
        atomize ``bind:`` line when the schedule carries one (the resolved operand→role
        binding, surfaced so ``compile --ir tile`` shows it above the combine)."""
        from deplodock.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        if self.op is None:
            return ""
        body = "\n".join(pretty(self.op, "    "))
        bind = getattr(self.schedule, "bind", None)
        return f"    {bind.pretty()}\n{body}" if bind is not None else body
