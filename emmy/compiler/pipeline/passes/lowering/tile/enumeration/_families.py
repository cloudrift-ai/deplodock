"""Algebra-native knob schema — one *move family* per DAG element.

The tile composer dispatches on the carrier
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

**Env spelling.** ``EMMY_<MOVE>_<ELEMENT>`` (element upper-cased), e.g.
``EMMY_REDUCE_KV=s16/f1/c1/t1`` / ``EMMY_ATOM_OUT=mma_m16n8k16_f16``. A bare
``EMMY_<MOVE>`` pins every element of that family (``EMMY_ATOM=scalar``,
``EMMY_REDUCE=s16/f1/c1/t1``) — for coarse pins / tests. The env-var namespace is
``config``'s (``knob_raw``), the same one ``compiler/pipeline/knob.py`` borrows.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmy import config

# --- Move families. Plain string tags; the op.knobs key is ``MOVE@element``. ---
SPLIT = "SPLIT"
REDUCE = "REDUCE"
ATOM = "ATOM"
PLACE = "PLACE"

#: The atom value naming the scalar tier (no tensor-core atomize) — the absence of an
#: ``ATOM_REGISTRY`` kind.
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
# decomposition). The ``c`` (cross-CTA / split-K) field additionally carries the cross-CTA
# **combine stage**'s finalize fold as a trailing letter (``c<cta>a`` = ATOMIC in-place,
# ``c<cta>k`` = deferred KERNEL-boundary) — the ``CombineStage(width=cta, fold, kernel_boundary)``
# of the cross-CTA level. A bare ``c<cta>`` (no letter) is the pre-decision transient the
# reduce-decomp emits; ``150_cross_cta_finalize`` completes it to ``a``/``k``. ``c1`` carries
# no finalize (no cross-CTA stage). ---

#: The cross-CTA finalize folds (the ``c`` field's trailing letter ↔ the policy name).
ATOMIC = "atomic"
KERNEL = "kernel"
_FINALIZE_CODE = {ATOMIC: "a", KERNEL: "k"}
_FINALIZE_NAME = {"a": ATOMIC, "k": KERNEL}


@dataclass(frozen=True)
class Decomp:
    """The four reduce-decomposition factors of one ``REDUCE@axis`` value, plus the
    cross-CTA finalize.

    - ``serial`` — intra-CTA K-loop trip (``ext/serial`` stage-inner chunk).
    - ``fold`` — register strip-mine into independent accumulators.
    - ``cta`` — cross-CTA split (split-K).
    - ``coop`` — cooperative-thread partition (warp-shuffle / tree combine).
    - ``finalize`` — the cross-CTA stage's combine fold: ``"atomic"`` (in-place ``atomicAdd``)
      or ``"kernel"`` (deferred combine kernel). Meaningful only when ``cta > 1``; defaults to
      ``"atomic"`` (the historical split-K default).

    ``1`` in any factor is the identity (no decomposition on that lever)."""

    serial: int = 1
    fold: int = 1
    cta: int = 1
    coop: int = 1
    finalize: str = ATOMIC


def enc_reduce(serial: int = 1, fold: int = 1, cta: int = 1, coop: int = 1, finalize: str | None = None) -> str:
    """Encode a ``REDUCE@axis`` value. ``finalize`` (``"atomic"`` / ``"kernel"``) stamps the
    cross-CTA ``c`` field's trailing letter; ``None`` (the default) emits a **bare** ``c<cta>``
    — the pre-decision transient the reduce-decomp move emits before ``140`` picks the finalize.
    ``cta <= 1`` never carries a letter (no cross-CTA stage)."""
    c = f"c{cta}" if cta <= 1 or finalize is None else f"c{cta}{_FINALIZE_CODE[finalize]}"
    return f"s{serial}/f{fold}/{c}/t{coop}"


def dec_reduce(raw: object) -> Decomp:
    """Decode a ``REDUCE@axis`` value, parsing the ``c`` field's optional finalize letter
    (``a``/``k``). A bare ``c<cta>`` (no letter) decodes ``finalize="atomic"`` — the semantic
    default — so a consumer reading a pre-decision or legacy value still gets a well-defined
    combine (use :func:`reduce_finalize_decided` to tell bare from an explicit ``a``)."""
    fields: dict[str, int] = {}
    finalize = ATOMIC
    for part in str(raw).split("/"):
        part = part.strip()
        if not part:
            continue
        tag, val = part[0], part[1:]
        if tag == "c":
            letter = val[-1:] if val[-1:] in _FINALIZE_NAME else ""
            num = val[: -len(letter)] if letter else val
            fields["c"] = int(num) if num else 1
            if letter and fields["c"] > 1:
                finalize = _FINALIZE_NAME[letter]
        else:
            fields[tag] = int(val)
    return Decomp(serial=fields.get("s", 1), fold=fields.get("f", 1), cta=fields.get("c", 1), coop=fields.get("t", 1), finalize=finalize)


def reduce_finalize_decided(raw: object) -> bool:
    """True once the ``c`` field carries an explicit finalize letter (``a``/``k``) — the
    idempotence guard ``140`` reads (bare ``c<cta>`` means the finalize is still pending)."""
    for part in str(raw).split("/"):
        part = part.strip()
        if part[:1] == "c":
            return part[-1:] in _FINALIZE_NAME
    return False


# --- Native env pins. ``EMMY_<MOVE>_<ELEMENT>`` first, bare ``EMMY_<MOVE>``
# as the all-elements fallback. ``config.knob_raw`` owns the ``EMMY_`` join. ---


def pin(move: str, element: str) -> str | None:
    """Live env pin for ``move`` on ``element`` — ``EMMY_<MOVE>_<ELEMENT>``,
    falling back to the bare ``EMMY_<MOVE>`` family pin. ``None`` when unset."""
    raw = config.knob_raw(f"{move}_{element}")
    return raw if raw is not None else config.knob_raw(move)


def split_par(dag, axis: str) -> int | None:
    """The pinned parallel-binding factor for ``SPLIT@axis`` (thread width or warp
    count — the tier is the consuming cell's ``ATOM``, not this value). Native
    ``EMMY_SPLIT_<axis>`` first, else the legacy ``BN``/``WN`` (innermost free) /
    ``BM``/``WM`` (next-out) ingest."""
    raw = pin(SPLIT, axis)
    if raw is not None:
        return dec_split(raw)[0]
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.split_par(dag, axis)


def split_reg(dag, axis: str) -> int | None:
    """The pinned register-cell factor for ``SPLIT@axis``. Native first, else the
    legacy ``FN`` (innermost free) / ``FM`` (next-out) ingest."""
    raw = pin(SPLIT, axis)
    if raw is not None:
        return dec_split(raw)[1]
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.split_reg(dag, axis)


def reduce_fields(dag, axis: str) -> tuple[int | None, int | None, int | None, int | None]:
    """The pinned ``(serial, fold, cta, coop)`` REDUCE factors for ``axis`` — each int
    or ``None`` when unpinned (so the offer keeps its full per-field menu). A native
    ``EMMY_REDUCE_<axis>`` pin (full spec) wins; otherwise the legacy
    ``EMMY_BK``/``FK``/``SPLITK``/``BR`` ingest fills the primary reduce axis (the
    deprecation ramp — see ``_knob_legacy``)."""
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    raw = pin(REDUCE, axis)
    if raw is not None:
        # A **partial** native pin (only the fields PRESENT in the string) leaves the rest free —
        # the native replacement for the legacy single-field ``BK``/``FK``/``SPLITK``/``BR`` pins.
        # ``EMMY_REDUCE=t128`` pins only ``coop`` (the offer still picks ``serial`` so the
        # reduce extent is covered); ``c2k`` pins only ``cta`` (+ the finalize, read by
        # ``pin_finalize``). A full ``s/f/c/t`` spec pins everything (a recorded golden).
        present = {p.strip()[0] for p in str(raw).split("/") if p.strip()}
        d = dec_reduce(raw)
        return (
            d.serial if "s" in present else None,
            d.fold if "f" in present else None,
            d.cta if "c" in present else None,
            d.coop if "t" in present else None,
        )
    return _knob_legacy.reduce_fields(dag, axis)


def pin_finalize(axis: str) -> str | None:
    """The pinned cross-CTA finalize for ``axis`` — ``"atomic"`` / ``"kernel"``, or ``None``
    (auto). The env override ``140`` honors when narrowing its finalize offer (the codec's
    ``c``-suffix fork). Sourced, in order, from:

    1. the native ``REDUCE@<axis>`` codec pin's ``c``-letter (``EMMY_REDUCE_<axis>`` /
       bare ``EMMY_REDUCE`` carrying e.g. ``c2k`` / ``c2a``) — the finalize lives IN the
       reduce codec, so one native knob owns both the split-K width and its finalize;
    2. a standalone ``EMMY_FINALIZE_<axis>`` / bare ``EMMY_FINALIZE`` convenience pin.

    Replaces the removed ``EMMY_NOATOMIC`` pin (``kernel`` ≡ the old ``NOATOMIC=1``)."""
    rraw = pin(REDUCE, axis)
    if rraw is not None and reduce_finalize_decided(rraw):
        return dec_reduce(rraw).finalize
    raw = config.knob_raw(f"FINALIZE_{axis.upper()}")
    if raw is None:
        raw = config.knob_raw("FINALIZE")
    if raw is None:
        return None
    r = raw.strip().lower()
    if r in ("kernel", "deferred", "k", "1", "true", "noatomic"):
        return KERNEL
    if r in ("atomic", "a", "0", "false"):
        return ATOMIC
    return None


# --- PLACE codec: ``place[:xport]`` — the per-edge placement lattice + transport. ---
# place ∈ {inline, smem, gmem}; xport ∈ {sync, cpasync, tma} is meaningful only for smem
# and is set at the transport fork (130) after the placement fork (120) sets the place —
# the same two-phase staging as SPLIT's par→reg. ``Schedule.staged`` stays the codegen
# source of truth; ``PLACE@<edge>`` is the per-edge record/fork the passes key on.
INLINE = "inline"
SMEM = "smem"
GMEM = "gmem"
#: The structural materialize-to-gmem placement of a demoted-matmul cone — distinct from
#: ``gmem`` (an operand read uncached, op-variant) so the value self-describes the
#: kernel-set-changing decision: ``gmem`` on an operand read-site is "read direct", ``cut``
#: on the cone is "materialize to a gmem intermediate kernel".
CUT = "cut"
#: The canonical structural element of the demoted-cone cut decision (like ``ATOM@out``'s
#: ``out`` — one cut per demoted matmul today, width-1; per-cone-edge is the additive
#: follow-up). Distinct from any operand / score element, so the structural recognizer keys
#: on ``PLACE@cone`` unambiguously.
CONE = "cone"


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
    Native bare ``EMMY_PLACE`` (``smem`` → stage all, ``gmem``/``inline`` → none),
    else the legacy ``EMMY_STAGE`` bitmask ingest."""
    raw = pin(PLACE, "")  # bare EMMY_PLACE family pin
    if raw is not None:
        return (1 << n) - 1 if place_of(raw) == SMEM else 0
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.stage_mask(n)


def cone_key() -> str:
    """The ``PLACE@cone`` key — the demoted-cone keep-vs-cut decision."""
    return place_key(CONE)


def pin_cut() -> bool | None:
    """Pinned keep-vs-cut for the demoted cone: native ``EMMY_PLACE_CONE`` (``cut`` /
    ``inline``), else legacy ``EMMY_CUT`` / ``EMMY_SPLIT_CONE``. ``True`` = cut,
    ``False`` = keep, ``None`` = auto."""
    raw = pin(PLACE, CONE)
    if raw is not None:
        return place_of(raw) == CUT
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.cut_pin()


def pin_inline_chain() -> bool:
    """The opt-in for the FA-2 shared-score restructuring — the score edge placed
    ``inline`` (register fragment). Native bare ``EMMY_PLACE=inline``, else the legacy
    ``EMMY_CHAIN`` ingest."""
    raw = pin(PLACE, "")
    if raw is not None:
        return place_of(raw) == INLINE
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.chain_pin()


def pin_xport() -> bool | None:
    """Pinned transport for staged edges: native bare ``EMMY_PLACE`` carrying a
    ``:tma`` / ``:sync`` xport, else the legacy ``EMMY_TMA`` ingest. ``None`` =
    auto."""
    raw = pin(PLACE, "")
    if raw is not None:
        xport = place_xport(raw)
        return None if xport is None else xport == "tma"
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.tma_pin()


def atom_raw(cell: str) -> str | None:
    """The raw atom control for ``cell`` — native ``EMMY_ATOM_<cell>`` / bare
    ``EMMY_ATOM`` first, else the legacy ``EMMY_MMA`` ingest. The string
    ``mma_decode`` interprets (``scalar`` / auto / a kind name)."""
    raw = pin(ATOM, cell)
    if raw is not None:
        return raw
    from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _knob_legacy  # noqa: PLC0415

    return _knob_legacy.atom_raw()
