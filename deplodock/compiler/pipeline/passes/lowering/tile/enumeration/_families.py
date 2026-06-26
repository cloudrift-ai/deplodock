"""Algebra-native knob schema — one *move family* per DAG element.

``plans/algebra-knob-naming-schema.md``. The tile composer dispatches on the carrier
algebra (``MAP`` / ``SEMIRING`` / ``MONOID``) over an arbitrary-rank iteration DAG, so
the knob vocabulary is keyed on the DAG's own elements rather than on rank-2 GEMM
letters. Each knob is **one move applied to one DAG element**; the ``op.knobs`` key
reads ``MOVE@element``:

    SPLIT@<free-axis>     tile_axis            ``"<par>x<reg>"``
    REDUCE@<reduce-axis>  reduce_decomp        ``"s<serial>/f<fold>/c<cta>/t<coop>"``
    ATOM@<cell>           atomize              ``"scalar"`` | an ``ATOM_REGISTRY`` kind
    PLACE@<edge>          placement+transport  ``"inline"`` | ``"smem[:sync|cpasync|tma]"`` | ``"gmem"``

The *element* is the DAG element's own IR identity: a free / reduce axis's
``Axis.name`` (``a0`` / ``dd`` / ``kv``), an edge's buffer name (``xn`` / ``score``), a
cell's ``Block.name``. The schema is instantiated per kernel by walking the DAG.

Two structural unlocks over the legacy rank-2 costume (``BN``/``BM``/``BK``/``WM``/…):

- ``REDUCE`` is **per reduce axis**, so a streaming flash gets BOTH ``REDUCE@dd`` (the
  QK^T inner contraction) and ``REDUCE@kv`` (the streaming axis) — two reduce axes the
  legacy K-only ``BK``/``FK``/``SPLITK``/``BR`` cannot express.
- ``SPLIT``'s ``par`` is tier-agnostic — its binding tier (THREAD vs WARP) is read off
  the consuming cell's ``ATOM`` (scalar → thread, atom → warp), not from which knob was
  set. So legacy ``BN``/``FN`` and ``WN``/``FN`` collapse to one ``SPLIT@<n-axis>``.

The implementation reads these keys (or the IR directly) — **never** legacy GEMM-letter
names. The legacy names survive only in the ingest mapper (``_knob_legacy``, Step 2) for
backwards-compatible env pins / golden YAMLs.

**Env spelling.** ``DEPLODOCK_<MOVE>_<ELEMENT>`` (element upper-cased), e.g.
``DEPLODOCK_REDUCE_KV=s16/f1/c1/t1`` / ``DEPLODOCK_ATOM_OUT=mma_m16n8k16_f16``. A bare
``DEPLODOCK_<MOVE>`` pins every element of that family (``DEPLODOCK_ATOM=scalar``,
``DEPLODOCK_REDUCE=s16/f1/c1/t1``) — for coarse pins / tests. The env-var namespace is
``config``'s (``knob_raw``), the same one ``compiler/pipeline/knob.py`` borrows.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock import config

# --- Move families. Plain string tags; the op.knobs key is ``MOVE@element``. ---
SPLIT = "SPLIT"
REDUCE = "REDUCE"
ATOM = "ATOM"
PLACE = "PLACE"

#: The atom value naming the scalar tier (no tensor-core atomize) — the absence of an
#: ``ATOM_REGISTRY`` kind. Read by :func:`atom_kind` to recover ``None``.
SCALAR = "scalar"

#: The canonical cell name of a matmul's single atomizable output cell. A STRUCTURAL
#: constant (NOT the per-op kernel name), so ``ATOM@out`` keys identically across
#: structurally-identical matmul ops and ``op_cache_key`` stays name-independent. (Flash's
#: two cells — ``score`` / ``out`` — are named structurally by the contraction chain.)
MATMUL_CELL = "out"


def key(move: str, element: str) -> str:
    """The ``op.knobs`` key for ``move`` applied to DAG element ``element``."""
    return f"{move}@{element}"


def split_key(axis: str) -> str:
    return key(SPLIT, axis)


def reduce_key(axis: str) -> str:
    return key(REDUCE, axis)


def atom_key(cell: str) -> str:
    return key(ATOM, cell)


def place_key(edge: str) -> str:
    return key(PLACE, edge)


def split_move(k: str) -> str | None:
    """The element of a ``MOVE@element`` key, or ``None`` if ``k`` is not that move."""
    return _element_of(SPLIT, k)


def reduce_move(k: str) -> str | None:
    return _element_of(REDUCE, k)


def atom_move(k: str) -> str | None:
    return _element_of(ATOM, k)


def _element_of(move: str, k: str) -> str | None:
    prefix = f"{move}@"
    return k[len(prefix) :] if k.startswith(prefix) else None


# --- SPLIT codec: ``"<par>x<reg>"`` — the parallel-binding factor × the register-cell
# factor. The grid is the launch residual (extent / (par·reg)), never stamped. ---


def enc_split(par: int, reg: int | None = None) -> str:
    """``"<par>x<reg>"`` once both factors are chosen, or a par-only ``"<par>"``
    transitional (the thread / warp-geometry fork sets ``par``; the register fork
    completes ``reg`` — the build only reads the value once complete)."""
    return f"{par}x{reg}" if reg is not None else f"{par}"


def dec_split(raw: object) -> tuple[int, int | None]:
    """``(par, reg)`` — ``reg`` is ``None`` for a par-only transitional value."""
    s = str(raw)
    if "x" in s:
        par, _, reg = s.partition("x")
        return int(par), int(reg)
    return int(s), None


def split_complete(raw: object) -> bool:
    """True once a ``SPLIT@<axis>`` value carries its register factor (the register
    fork has run) — i.e. ``"PxR"``, not the par-only ``"P"`` transitional."""
    return "x" in str(raw)


# --- REDUCE codec: ``"s<serial>/f<fold>/c<cta>/t<coop>"`` — the four reduce-decomposition
# tower components. Which fields are legal is the carrier's traits (``associative →
# {s,f}``, ``commutative → {c,t}``); an illegal field is forced to 1 (= identity = no
# decomposition). ---


@dataclass(frozen=True)
class Decomp:
    """The four reduce-decomposition factors of one ``REDUCE@axis`` value.

    - ``serial`` — intra-CTA K-loop trip (``ext/serial`` stage-inner chunk).
    - ``fold`` — register strip-mine into independent accumulators.
    - ``cta`` — cross-CTA split (split-K).
    - ``coop`` — cooperative-thread partition (warp-shuffle / tree combine).

    ``1`` in any field is the identity (no decomposition on that lever)."""

    serial: int = 1
    fold: int = 1
    cta: int = 1
    coop: int = 1


def enc_reduce(serial: int = 1, fold: int = 1, cta: int = 1, coop: int = 1) -> str:
    return f"s{serial}/f{fold}/c{cta}/t{coop}"


def dec_reduce(raw: object) -> Decomp:
    fields: dict[str, int] = {}
    for part in str(raw).split("/"):
        part = part.strip()
        if part:
            fields[part[0]] = int(part[1:])
    return Decomp(serial=fields.get("s", 1), fold=fields.get("f", 1), cta=fields.get("c", 1), coop=fields.get("t", 1))


# --- ATOM codec: ``"scalar"`` or an ``ATOM_REGISTRY`` kind name. ---


def enc_atom(kind: str | None) -> str:
    return kind if kind else SCALAR


def atom_kind(raw: object) -> str | None:
    """The concrete tensor-core atom kind named by ``raw``, or ``None`` for the
    scalar tier (``"scalar"`` / empty)."""
    s = str(raw).strip()
    return None if not s or s.lower() == SCALAR else s


# --- Native env pins. ``DEPLODOCK_<MOVE>_<ELEMENT>`` first, bare ``DEPLODOCK_<MOVE>``
# as the all-elements fallback. ``config.knob_raw`` owns the ``DEPLODOCK_`` join. ---


def pin(move: str, element: str) -> str | None:
    """Live env pin for ``move`` on ``element`` — ``DEPLODOCK_<MOVE>_<ELEMENT>``,
    falling back to the bare ``DEPLODOCK_<MOVE>`` family pin. ``None`` when unset."""
    raw = config.knob_raw(f"{move}_{element}")
    return raw if raw is not None else config.knob_raw(move)


def split_par(dag, axis: str) -> int | None:
    """The pinned parallel-binding factor for ``SPLIT@axis`` (thread width or warp
    count — the tier is the consuming cell's ``ATOM``, not this value). Native
    ``DEPLODOCK_SPLIT_<axis>`` first, else the legacy ``BN``/``WN`` (innermost free) /
    ``BM``/``WM`` (next-out) ingest."""
    raw = pin(SPLIT, axis)
    if raw is not None:
        return dec_split(raw)[0]
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.split_par(dag, axis)


def split_reg(dag, axis: str) -> int | None:
    """The pinned register-cell factor for ``SPLIT@axis``. Native first, else the
    legacy ``FN`` (innermost free) / ``FM`` (next-out) ingest."""
    raw = pin(SPLIT, axis)
    if raw is not None:
        return dec_split(raw)[1]
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.split_reg(dag, axis)


def pin_reduce(axis: str) -> Decomp | None:
    raw = pin(REDUCE, axis)
    return dec_reduce(raw) if raw is not None else None


def reduce_fields(dag, axis: str) -> tuple[int | None, int | None, int | None, int | None]:
    """The pinned ``(serial, fold, cta, coop)`` REDUCE factors for ``axis`` — each int
    or ``None`` when unpinned (so the offer keeps its full per-field menu). A native
    ``DEPLODOCK_REDUCE_<axis>`` pin (full spec) wins; otherwise the legacy
    ``DEPLODOCK_BK``/``FK``/``SPLITK``/``BR`` ingest fills the primary reduce axis (the
    deprecation ramp — see ``_knob_legacy``)."""
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    raw = pin(REDUCE, axis)
    if raw is not None:
        d = dec_reduce(raw)
        return (d.serial, d.fold, d.cta, d.coop)
    return _knob_legacy.reduce_fields(dag, axis)


def pin_atom(cell: str) -> str | None:
    """The raw ``ATOM@cell`` env pin (``"scalar"`` / a kind / ``None`` when unset).
    Distinct from :func:`atom_kind` — a ``"scalar"`` pin is a *decision*, not unset."""
    return pin(ATOM, cell)


# --- PLACE codec: ``place[:xport]`` — the per-edge placement lattice + transport. ---
# place ∈ {inline, smem, gmem}; xport ∈ {sync, cpasync, tma} is meaningful only for smem
# and is set at the transport fork (130) after the placement fork (120) sets the place —
# the same two-phase staging as SPLIT's par→reg. ``Schedule.staged`` stays the codegen
# source of truth; ``PLACE@<edge>`` is the per-edge record/fork the passes key on.
INLINE = "inline"
SMEM = "smem"
GMEM = "gmem"


def enc_place(place: str, xport: str | None = None) -> str:
    return f"{place}:{xport}" if xport else place


def dec_place(raw: object) -> tuple[str, str | None]:
    place, _, xport = str(raw).partition(":")
    return place, (xport or None)


def place_of(raw: object) -> str:
    return dec_place(raw)[0]


def place_xport(raw: object) -> str | None:
    return dec_place(raw)[1]


def place_decided(knobs: dict) -> bool:
    """True once the placement fork (120) has stamped a ``PLACE@<edge>`` for this op —
    the idempotence guard so it runs once."""
    return any(k.startswith("PLACE@") for k in knobs)


def smem_edges_without_xport(knobs: dict) -> list[str]:
    """The ``PLACE@<edge>`` keys placed in smem but with no transport chosen yet — the
    read-sites the transport fork (130) promotes. Empty once transport is decided (or
    nothing is staged)."""
    return [k for k, v in knobs.items() if k.startswith("PLACE@") and place_of(v) == SMEM and place_xport(v) is None]


def pin_place_mask(n: int) -> int | None:
    """A pinned staging mask over ``n`` ranked edges, or ``None`` (auto-enumerate).
    Native bare ``DEPLODOCK_PLACE`` (``smem`` → stage all, ``gmem``/``inline`` → none),
    else the legacy ``DEPLODOCK_STAGE`` bitmask ingest."""
    raw = pin(PLACE, "")  # bare DEPLODOCK_PLACE family pin
    if raw is not None:
        return (1 << n) - 1 if place_of(raw) == SMEM else 0
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.stage_mask(n)


def pin_xport() -> bool | None:
    """Pinned transport for staged edges: native bare ``DEPLODOCK_PLACE`` carrying a
    ``:tma`` / ``:sync`` xport, else the legacy ``DEPLODOCK_TMA`` ingest. ``None`` =
    auto."""
    raw = pin(PLACE, "")
    if raw is not None:
        xport = place_xport(raw)
        return None if xport is None else xport == "tma"
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.tma_pin()


def atom_raw(cell: str) -> str | None:
    """The raw atom control for ``cell`` — native ``DEPLODOCK_ATOM_<cell>`` / bare
    ``DEPLODOCK_ATOM`` first, else the legacy ``DEPLODOCK_MMA`` ingest. The string
    ``mma_decode`` interprets (``scalar`` / auto / a kind name)."""
    raw = pin(ATOM, cell)
    if raw is not None:
        return raw
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.atom_raw()
