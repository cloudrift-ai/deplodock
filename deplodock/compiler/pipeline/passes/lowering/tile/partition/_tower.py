"""Tower-building primitives shared by the legacy planner and the move
composer: the planner-internal :class:`Role` label and :func:`_wrap_tower`,
which wraps a body in the nested typed tile flavors (``GridTile`` /
``ThreadTile`` / ``RegisterTile`` / ``WarpTile`` / ``AtomTile`` /
``SerialTile``).

Extracted verbatim from ``010_partition_loops`` so the new composer can
reuse the exact tower mechanics (the legacy planner imports them back).
"""

from __future__ import annotations

import enum

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    Atom,
    AtomTile,
    GridTile,
    RegisterTile,
    SerialTile,
    ThreadTile,
    WarpTile,
)


class Role(enum.Enum):
    """Planner-internal label for ``_wrap_tower`` layers.

    Drives which tile-flavor the layer becomes when the planner builds the
    tower. Not part of the IR — never reaches downstream passes (which
    discriminate on tile-flavor type instead).

    ``WARP`` is reserved for the MMA fragment-factorization consumer plan
    (``plans/mma-fragment-factorization.md``) and the warp-specialized TMA
    refactor — ``tile/085_warp_specialize.py`` emits it today by rewriting a
    post-080 ``ThreadTile``; once M3 of the MMA plan lands the planner emits
    it directly for warp-tier matmul rows. ``ATOM`` is reserved for the MMA
    plan's hardware-atomic cell tier (``AtomTile``). ``_layer_kind_for`` /
    ``_wrap_tower`` recognise both roles so consumer plans can flip a tier
    without revisiting the tower-building mechanics.
    """

    BLOCK = "block"
    THREAD = "thread"
    REGISTER = "register"
    WARP = "warp"
    ATOM = "atom"
    STAGE_INNER = "stage_inner"
    SERIAL_OUTER = "serial_outer"
    PIPELINE = "pipeline"


def _identity_rename(name: str) -> str:
    return name


def _wrap_tower(layers: list[tuple[Axis, Role | None]], inner: tuple[Stmt, ...], *, atom: Atom | None = None) -> tuple[Stmt, ...]:
    """Wrap ``inner`` in nested typed tile flavors, innermost layer first.

    ``atom`` is the :class:`Atom` spec for an ``AtomTile`` layer (required iff
    ``layers`` contains a ``Role.ATOM`` — i.e. the warp-tier matmul tower); it
    is stamped onto the emitted ``AtomTile`` so the spec rides the IR structure.

    ``layers`` is innermost-first: ``[(K_i, STAGE_INNER), (K_o, SERIAL_OUTER)]``
    walks outer ``K_o`` outermost. Consecutive parallel-binding axes group
    into one tile (so ``[BLOCK, BLOCK, THREAD, THREAD, REGISTER]`` yields
    ``GridTile(BLOCK,BLOCK) > ThreadTile(THREAD,THREAD) > RegisterTile(REGISTER)``).
    Each serial-binding axis becomes its own ``SerialTile`` with the
    matching ``kind``.

    Role → flavor mapping:

    - ``BLOCK`` → ``GridTile.axes``. Split-K vs. regular output-partition
      is derived at codegen time from ``escape_analysis.atomic_axes``.
    - ``THREAD`` → ``ThreadTile.axes``. Cooperative-K cooperativity is
      recovered at materialize time from ``Accum.axes ∩ ThreadTile.axes``
      (see ``escape_analysis``).
    - ``REGISTER`` → ``RegisterTile.axes``.
    - ``WARP`` → ``WarpTile.axes``. Reserved for the MMA / WS-refactor
      consumer plans; no rule in this pass emits it today.
    - ``SERIAL_OUTER`` / ``STAGE_INNER`` / ``PIPELINE`` → ``SerialTile(kind=…)``.
    - Untagged (``None``) → ``SerialTile(kind="plain")``.

    **Size-1 axis filtering.** ``Loop`` IR's ``drop_size_one_free_axes``
    inlines extent-1 free Loops by substituting their var with 0. The
    planner's σ-split sometimes generates such axes (e.g. cooperative
    softmax with BN=BM=1 makes N_t / M_t extent-1 THREAD axes). Mirror
    the same drop here — except for ``BLOCK`` axes,
    which signal launch geometry to the CUDA backend and must survive
    even at extent 1 (single-CTA cooperative kernels rely on this).
    """
    inner_body = tuple(inner)
    # Drop size-1 axes that aren't launch-geometry markers; substitute
    # ``Var(axis.name) → Literal(0, "int")`` in the inner body. BLOCK axes
    # signal grid launch geometry; WARP / ATOM axes signal warp-
    # cooperative MMA codegen — both survive at extent 1 so the
    # materializer can read the cell shape / warp count off the tower.
    _STRUCTURAL_ROLES = (Role.BLOCK, Role.WARP, Role.ATOM)

    # A surviving RegisterTile (a register axis with extent > 1) must be hosted by
    # a ThreadTile — the GridTile > ThreadTile > RegisterTile contract every
    # downstream pass (``020_stage_inputs`` / ``010_split_register_axes`` via
    # ``parallel_tile_of``) relies on. If the planner picked thread=1×1 with
    # reg>1 (one thread per CTA owning the register cells), dropping the size-1
    # THREAD axes would leave GridTile > RegisterTile with no ThreadTile, which
    # ``parallel_tile_of`` can't navigate. So keep ONE degenerate THREAD axis in
    # that case (an extent-1 threadIdx binding — correct, just a 1-thread CTA).
    def _real(ax: Axis) -> bool:
        return (not ax.extent.is_static) or ax.extent.as_static() > 1

    register_present = any(role == Role.REGISTER and _real(axis) for axis, role in layers)
    thread_axes = [axis for axis, role in layers if role == Role.THREAD]
    keep_one_thread = register_present and bool(thread_axes) and not any(_real(ax) for ax in thread_axes)

    filtered: list[tuple[Axis, Role | None]] = []
    kept_thread = False
    for axis, role in layers:
        if axis.extent.is_static and axis.extent.as_static() == 1 and role not in _STRUCTURAL_ROLES:
            if role == Role.THREAD and keep_one_thread and not kept_thread:
                filtered.append((axis, role))
                kept_thread = True
                continue
            sub = Sigma({axis.name: Literal(0, "int")})
            inner_body = tuple(c.rewrite(_identity_rename, sub) for c in inner_body)
            continue
        if role == Role.THREAD:
            kept_thread = True
        filtered.append((axis, role))

    # Walk outermost-first so consecutive parallel axes group naturally.
    outermost_first = list(reversed(filtered))
    # Group: list of (group_kind, [axes], [roles])
    groups: list[tuple[str, list[Axis], list[Role | None]]] = []
    for axis, role in outermost_first:
        kind = _layer_kind_for(role)
        # Parallel kinds group consecutive same-kind axes; serial kinds always
        # start a fresh group (each serial layer is its own SerialTile).
        if groups and groups[-1][0] == kind and kind in ("grid", "thread", "register", "warp", "atom"):
            groups[-1][1].append(axis)
            groups[-1][2].append(role)
        else:
            groups.append((kind, [axis], [role]))

    # Build the tree innermost-first by wrapping ``inner`` with each group
    # in reverse order.
    current: tuple[Stmt, ...] = inner_body
    for kind, axes, roles in reversed(groups):
        if kind == "grid":
            # Split-K block axes (K_s) need no tag — codegen derives
            # atomic-add from ``escape_analysis.atomic_axes`` (block axis
            # missing from Write.index).
            current = (GridTile(axes=tuple(axes), body=Body(current)),)
        elif kind == "thread":
            # Cooperative-K thread axes (K_c) get into ``Accum.axes``
            # via the planner's σ-split; the materializer recovers
            # cooperativity from ``Accum.axes ∩ ThreadTile.axes``.
            current = (ThreadTile(axes=tuple(axes), body=Body(current)),)
        elif kind == "register":
            current = (RegisterTile(axes=tuple(axes), body=Body(current)),)
        elif kind == "warp":
            # Warp-cooperative tier (consumer plans flip a tier to
            # ``Role.WARP``). ``tile/085_warp_specialize.py`` emits it
            # today by rewriting a post-080 ThreadTile; once the MMA plan's
            # M3 lands the planner emits it directly for warp-tier matmul.
            current = (WarpTile(axes=tuple(axes), body=Body(current)),)
        elif kind == "atom":
            # Hardware-atomic cell tier (MMA fragment factorization). Marker
            # for the per-cell tensor-core extent; consumed by the MMA cell
            # materializer (``kernel/010_split_register_axes`` MMA arm), so
            # no AtomTile reaches kernel render. No rule emits this today —
            # the case wires the tower builder so M3 of
            # ``plans/mma-fragment-factorization.md`` lands without
            # revisiting ``_wrap_tower`` mechanics.
            assert atom is not None, "_wrap_tower: an ATOM layer requires the atom spec"
            current = (AtomTile(axes=tuple(axes), body=Body(current), atom=atom),)
        else:  # serial — one axis per layer
            ax = axes[0]
            role = roles[0]
            serial_kind: str = "plain"
            if role is Role.SERIAL_OUTER:
                serial_kind = "serial_outer"
            elif role is Role.STAGE_INNER:
                serial_kind = "stage_inner"
            elif role is Role.PIPELINE:
                serial_kind = "pipeline"
            current = (SerialTile(axis=ax, body=Body(current), kind=serial_kind),)
    return current


def _layer_kind_for(role: Role | None) -> str:
    if role is Role.BLOCK:
        return "grid"
    if role is Role.THREAD:
        return "thread"
    if role is Role.REGISTER:
        return "register"
    if role is Role.WARP:
        return "warp"
    if role is Role.ATOM:
        return "atom"
    return "serial"
