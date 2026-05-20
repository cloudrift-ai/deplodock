"""Loop IR — one ``LoopOp`` is one GPU kernel's worth of loop-nest compute.

After fusion, each ``LoopOp`` describes the compute for one GPU kernel as
an SSA program over a named iteration space:

    body              : Body   — SSA body: Assign | Accum | Write | Select | Loop | Load
    axes              : computed property  — iteration space walked from body's Loop tree
    reduce_axis_names : computed property  — names of axes whose Loop wraps an Accum
    loads             : computed property  — body-form Load stmts (external reads)
    accums            : computed property  — unique Accum stmts, first-use order
    input_bufs        : computed property  — distinct Load.source names (in first-use order)
    num_inputs        : computed property  — len(input_bufs)
    analyze()         : LoopMeta           — precomputed name → def / scope / reduce-axis / live-axes
    __iter__          : Iterator[Stmt]     — pre-order walk (via ``iter_body``)

Iteration is explicit via ``Loop(axis, body)`` statements. Each ``Loop``
is one iteration dimension; ``Loop.body`` runs ``axis.extent`` times.
Reduce-kind Loops fold ``Accum`` statements into the accumulator named
by each ``Accum.name``. Free-kind Loops run in parallel with no
accumulator folding. Reading top-to-bottom matches execution order.

SSA names defined by ``Assign`` / ``Select`` / ``Load`` inside a
``Loop.body`` are scoped to that body — only accumulator state crosses
Loop boundaries (via ``Accum``). Invariants (unique names, defined-
before-use, accumulator liveness) are enforced by ``LoopOp.__post_init__``
(which also runs ``normalize_body`` to canonicalize the body).

Free-function companions (used by passes that work on raw
``Body``): ``iter_body`` (pre-order generator),
``map_body`` (transformer supporting ``Stmt | None | Iterable[Stmt]``),
``Stmt.rewrite(rename_ssa, sigma)`` (per-stmt SSA / Expr rewrite).
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.stmt import (  # noqa: F401  (re-exported via __init__)
    Accum,
    Assign,
    Body,
    Cond,
    Load,
    Loop,
    Select,
    SelectBranch,
    Stmt,
    Write,
    pretty_body,
)
from deplodock.compiler.ir.stmt.ir import BodyOp

# ---------------------------------------------------------------------------
# Scope — a path of enclosing axes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scope:
    """Enclosing loop nest from outermost to innermost.

    A ``Scope`` identifies a location in a ``LoopOp`` body: the sequence
    of ``Loop`` axes one descends to reach that point. Empty = body root.
    Used by analysis passes that need to know where a named SSA value was
    defined or where a new stmt should be emitted.
    """

    enclosing: tuple[Axis, ...] = ()

    def nest(self, axis: Axis) -> Scope:
        return Scope(enclosing=self.enclosing + (axis,))


# Body Stmts (Stmt, Load, Assign, Accum, Write, Select, SelectBranch,
# Loop, Cond) and the tree-walk helpers (map_body) live in
# ``ir/stmt.py`` — they're shared across all IR layers. Imported above
# and re-exported via ``ir/loop/__init__.py``.


# ---------------------------------------------------------------------------
# LoopOp
# ---------------------------------------------------------------------------


@dataclass
class LoopOp(BodyOp):
    """One kernel's worth of computation as an SSA program over named axes.

    ``body`` is the sole stored field (inherited from :class:`BodyOp`): a
    tuple of nested ``Loop`` blocks with leaf ``Load`` / ``Assign`` /
    ``Accum`` / ``Write`` / ``Select`` statements. ``axes``, ``loads``,
    ``accums``, ``body_inputs``, and ``body_outputs`` are computed
    properties derived from that tree.
    """

    def __post_init__(self) -> None:
        from deplodock.compiler.ir.stmt import normalize_body

        # Body is a tuple subclass; coerce so ``Op(body=tuple_value)``
        # construction shape keeps working without forcing wrapping at
        # every rule's rewrite site.
        coerced = Body.coerce(self.body)
        normalized = normalize_body(coerced)
        self.body = normalized if isinstance(normalized, Body) else Body(normalized)
        _validate(self)

    @property
    def axes(self) -> tuple[Axis, ...]:
        """Iteration axes collected from the body's ``Loop`` tree.

        Pre-order traversal: outer ``Loop`` blocks (the kernel's grid) come
        first, nested inner Loops follow. Sibling Loops with the same axis
        name (softmax's two K-sweeps) contribute one entry — axes are
        deduplicated by name, keeping the first occurrence.
        """
        seen: dict[str, Axis] = {}
        for s in self:
            if isinstance(s, Loop) and s.axis.name not in seen:
                seen[s.axis.name] = s.axis
        return tuple(seen.values())

    @property
    def reduce_axis_names(self) -> frozenset[str]:
        """Names of axes whose Loop directly wraps an ``Accum``.

        Derived from body structure, not stored metadata: a Loop is a
        'reduce Loop' iff its immediate body contains an Accum.
        """
        names: set[str] = set()

        def walk(stmts: Body, innermost: str | None) -> None:
            for s in stmts:
                if isinstance(s, Accum) and innermost is not None:
                    names.add(innermost)
                elif isinstance(s, Loop):
                    walk(s.body, s.axis.name)

        walk(self.body, None)
        return frozenset(names)

    def analyze(self) -> LoopMeta:
        """One-pass summary of the body: name→def, name→scope, writes.

        Convenience for passes (e.g. the fusion splicer) that resolve SSA
        dependencies against a stable snapshot of the body tree.
        """
        defs: dict[str, Stmt] = {}
        scopes: dict[str, Scope] = {}
        reduce_axes: dict[str, Axis] = {}
        writes: list[tuple[Write, Scope]] = []

        def walk(stmts: Body, scope: Scope) -> None:
            for s in stmts:
                if isinstance(s, Loop):
                    walk(s.body, scope.nest(s.axis))
                elif isinstance(s, Cond):
                    walk(s.body, scope)
                    walk(s.else_body, scope)
                elif isinstance(s, Accum):
                    defs[s.name] = s
                    # Binding scope excludes the reduce axis (the Accum is live
                    # after its reduce Loop completes).
                    if scope.enclosing:
                        reduce_axes[s.name] = scope.enclosing[-1]
                        scopes[s.name] = Scope(enclosing=scope.enclosing[:-1])
                    else:
                        scopes[s.name] = scope
                elif isinstance(s, (Load, Assign, Select)):
                    defs[s.name] = s
                    scopes[s.name] = scope
                elif isinstance(s, Write):
                    writes.append((s, scope))

        walk(self.body, Scope())
        return LoopMeta(
            op=self,
            defs=defs,
            scopes=scopes,
            reduce_axes=reduce_axes,
            writes=tuple(writes),
            live_axes=_compute_live_axes(self.body),
        )

    def forward(self, *inputs):
        """Evaluate the kernel body via cppyy-JIT'd C++ — mirrors the other ``Op.forward`` methods.

        Each ``LoopOp`` body is rendered to plain C++ by the loop
        ``runner`` module, compiled in-process via Cling (cached by op
        identity + input shapes), and called against numpy buffers. Used
        by every backend whose graph contains ``LoopOp`` nodes (today:
        ``LoopBackend``) through the default ``Backend.run`` topo-walk
        dispatch.
        """
        import numpy as np

        from deplodock.compiler.ir.loop.runner import execute_loop_op_cpp

        out_shape = self._infer_write_shape()
        bufs = self.body_inputs
        if len(inputs) != len(bufs):
            raise ValueError(f"LoopOp.forward: expected {len(bufs)} inputs (matching input_bufs={list(bufs)}), got {len(inputs)}")
        input_arrays = {name: np.asarray(x, dtype=np.float32) for name, x in zip(bufs, inputs, strict=True)}
        return execute_loop_op_cpp(self, input_arrays, out_shape)

    def _infer_write_shape(self) -> tuple[int, ...]:
        """Derive the output buffer shape from the kernel's ``Write`` index.

        Evaluates each dim's index Expr over the full iteration space; the
        per-dim extent is ``max(value) + 1``. Handles plain ``Var(axis)`` (→
        axis extent), ``Literal(c)`` (→ 1), and affine combinations uniformly.
        Falls back to the free-axis extents when no ``Write`` is present.
        """
        import numpy as np

        writes = [s for s in self if isinstance(s, Write)]
        if not writes:
            reduce_names = self.reduce_axis_names
            return tuple(a.extent for a in self.axes if a.name not in reduce_names)
        w = writes[0]
        env: dict[str, object] = {}
        for i, a in enumerate(self.axes):
            shape = [1] * len(self.axes)
            shape[i] = int(a.extent)
            env[a.name] = np.arange(int(a.extent)).reshape(shape)
        dims: list[int] = []
        for e in w.index:
            vals = e.eval(env)
            if isinstance(vals, np.ndarray):
                dims.append(int(vals.max()) + 1)
            else:
                dims.append(int(vals) + 1)
        return tuple(dims)


# ---------------------------------------------------------------------------
# LoopMeta — analysis summary produced by ``LoopOp.analyze``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopMeta:
    """Precomputed lookups over a ``LoopOp`` body.

    - ``op``: the source ``LoopOp`` this meta was built from.
    - ``defs``: SSA name → defining ``Stmt`` (``Load`` / ``Assign`` /
      ``Select`` / ``Accum``). A ``Write`` has no SSA name and is not here.
    - ``scopes``: SSA name → binding ``Scope`` (where the value is live
      after its def). For plain stmts this is the enclosing axis chain;
      for ``Accum`` the reduce axis is excluded — the Accum binds *after*
      the reduce Loop completes.
    - ``reduce_axes``: ``Accum`` name → its reduce ``Axis`` (the tail of
      the raw enclosing chain, stripped from ``scopes``). Only present
      for ``Accum`` defs.
    - ``writes``: every ``Write`` stmt paired with the ``Scope`` it sits
      in — one entry per output, in body order.
    - ``live_axes``: SSA name → axis names transitively reachable through
      Expr subtrees (``Load.index``, ``SelectBranch.select``) while resolving
      the stmt's dep chain. For an ``Accum``, the reduce axis is excluded
      since it gets freshened at emission time.
    """

    op: LoopOp
    defs: dict[str, Stmt]
    scopes: dict[str, Scope]
    reduce_axes: dict[str, Axis]
    writes: tuple[tuple[Write, Scope], ...]
    live_axes: dict[str, frozenset[str]]


def _compute_live_axes(body: Body) -> dict[str, frozenset[str]]:
    """Axes reachable through Expr subtrees rooted at each SSA name.

    Thin wrapper over :attr:`Body.deps_closure` filtered to axis names —
    Accums already have their reduce axis subtracted by ``deps_closure``.
    """
    axes = body.axis_names
    return {name: closure & axes for name, closure in body.deps_closure.items()}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(loop: LoopOp) -> None:
    """Enforce Axis uniqueness, SSA invariants, accumulator pairing.

    Body validation recurses into ``Loop`` blocks. SSA names defined inside a
    ``Loop.body`` are scoped to that body — invisible outside — except for
    ``Accum.name`` names, which are bound post-Loop in the enclosing
    scope. Axis names are validated as a flat set for uniqueness across
    the kernel.
    """
    # LoopOp.axes uniqueness (stored axes).
    all_axis_names: set[str] = set()
    for a in loop.axes:
        if a.name in all_axis_names:
            raise ValueError(f"LoopOp.axes: duplicate axis name {a.name!r}")
        all_axis_names.add(a.name)

    # Op-consistency across repeated Updates to same target.
    # Walks the body and rejects an Accum that conflicts with an
    # earlier Accum sharing its name. ``target_ops`` is the running map.
    target_ops: dict[str, ElementwiseImpl] = {}

    def _walk(stmts: Body, defined: set[str]) -> set[str]:
        """Validate a body scope. Returns the set of ``Accum.name`` names
        that propagate to the enclosing scope after this body completes
        (they carry the accumulator's finalized value).

        Sibling Loops with the same axis name are fine — that's the softmax
        pattern (two sequential sweeps over the same axis). Nested Loops
        that re-use an outer axis name are left to backend codegen to
        disambiguate (the axis is only referenced as Var(name), and SSA
        scoping handles variable shadowing).
        """
        exported_accs: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, Assign):
                for arg in stmt.args:
                    if arg not in defined:
                        raise ValueError(f"Assign {stmt.name!r}: arg {arg!r} not defined")
                if stmt.name in defined:
                    raise ValueError(f"Assign {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Load):
                # Load is a binding site — introduces its SSA name. ``source``
                # is the producing graph node's id.
                if not isinstance(stmt.input, str) or not stmt.input:
                    raise ValueError(f"Load {stmt.name!r}: source {stmt.input!r} must be a non-empty string")
                if stmt.name in defined:
                    raise ValueError(f"Load {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Accum):
                if stmt.value not in defined and stmt.name != stmt.value:
                    raise ValueError(f"Accum: value {stmt.value!r} not defined")
                # Each Accum implicitly declares / extends an accumulator.
                # Repeated Updates to same target must share op.
                prev_op = target_ops.get(stmt.name)
                if prev_op is not None and prev_op.name != stmt.op.name:
                    raise ValueError(f"Accum {stmt.name!r}: op {stmt.op.name!r} conflicts with earlier Accum's op {prev_op.name!r}")
                target_ops[stmt.name] = stmt.op
                defined.add(stmt.name)
                exported_accs.add(stmt.name)
            elif isinstance(stmt, Select):
                for b in stmt.branches:
                    if b.value not in defined:
                        raise ValueError(f"Select {stmt.name!r}: branch value {b.value!r} not defined")
                if stmt.name in defined:
                    raise ValueError(f"Select {stmt.name!r}: name already defined")
                defined.add(stmt.name)
            elif isinstance(stmt, Write):
                if stmt.value not in defined:
                    raise ValueError(f"Write: value {stmt.value!r} not defined")
            elif isinstance(stmt, Loop):
                # SSA names defined inside Loop.body are scoped to that body,
                # except Accum.names which carry the finalized reduced
                # value out to the enclosing scope.
                inner_exports = _walk(stmt.body, set(defined))
                defined.update(inner_exports)
                exported_accs.update(inner_exports)
            elif isinstance(stmt, Cond):
                # Same scoping rules as Loop: inner Assign / Select names are
                # local; Accum names export. Validate both branches.
                inner_exports = _walk(stmt.body, set(defined))
                defined.update(inner_exports)
                exported_accs.update(inner_exports)
                inner_exports = _walk(stmt.else_body, set(defined))
                defined.update(inner_exports)
                exported_accs.update(inner_exports)
        return exported_accs

    _walk(loop.body, set())

    # Every LoopOp must Write at least one output buffer — that's its
    # observable result. Renderers, ``forward``, and ``_infer_write_shape``
    # all assume this; check at construction so callers fail loudly.
    # Runs after SSA / Accum validation so malformed-body tests get the
    # specific error the body actually has, not a no-Write surface error.
    if not loop.body_outputs:
        raise ValueError("LoopOp body has no Write")


# Tree walk helpers (map_body) live in ``ir/stmt.py`` — they
# work across all IR layers via ``Stmt.nested``. Imported above and
# re-exported via ``ir/loop/__init__.py``.
