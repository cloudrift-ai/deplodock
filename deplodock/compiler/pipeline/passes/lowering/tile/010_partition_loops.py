"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain. Stamps ``Role`` tags on body Loops;
``001_launch_geometry`` and other downstream rules read the tags and skip
their own decisions.

Output axes split as ``A → A_b·(T·R) + A_t·R + A_r`` (T = BN or BM, R = FN or
FM). K splits as ``K → K_s·(K_o·br·bk) + K_o·(br·bk) + K_i·br + K_c`` (K_s for
SPLITK > 1, K_c for cooperative-K BR > 1). Resulting nesting:

    K_s BLOCK (split-K) → M_b BLOCK → N_b BLOCK → M_t THREAD →
      N_t THREAD → K_c THREAD (coop-K, innermost = fastest threadIdx bits) →
      M_r REGISTER → N_r REGISTER →
        prelude → K_o SERIAL_OUTER → K_i STAGE_INNER (reduce σ(body)) →
        (helper-driven cross-thread combine when K_c is cooperative) →
        post-K tower → Write

Example transformation (matmul A[M=64,K=32] @ B[K=32,N=64] with BN=BM=16,
FM=FN=1, BK=16, SPLITK=1, BR=1):

    Input LoopOp body:
        for m in 0..64:
            for n in 0..64:
                Init(acc)
                for k in 0..32 reduce:
                    a = load A[m, k]; b = load B[k, n]
                    Accum(acc, a*b)
                Write(C[m, n], acc)

    Output (σ_outer rewrites m, n; σ_k rewrites k; tower wraps):
        for m_b in 0..4 BLOCK:
            for n_b in 0..4 BLOCK:
                for m_t in 0..16 THREAD:
                    for n_t in 0..16 THREAD:
                        for m_r in 0..1 REGISTER:    # inlined by normalize_body
                            for n_r in 0..1 REGISTER:    # inlined
                                Init(acc)
                                for k_o in 0..2 SERIAL_OUTER:
                                    for k_i in 0..16 STAGE_INNER reduce:
                                        a = load A[m_b·16 + m_t, k_o·16 + k_i]
                                        b = load B[k_o·16 + k_i, n_b·16 + n_t]
                                        Accum(acc, a*b)
                                Write(C[m_b·16 + m_t, n_b·16 + n_t], acc)

Warp-tier (MMA) variant — emitted by ``_build_split_body_warp`` when the
planner picks a warp-tier knob row (only fires when
``DEPLODOCK_MMA=1`` and ``is_atom_eligible`` passes; see
``plans/mma-fragment-factorization.md``). The output-axis factorization
gains a WARP tier between BLOCK and REGISTER and an ATOM tier inside
REGISTER carrying the hardware cell shape:

    N → N_b·(W_n·F_n·A_n) + N_w·(F_n·A_n) + N_r·A_n + N_a
          BLOCK    WARP             REGISTER          ATOM

K stride per ``K_i`` step is the atom's K dim (e.g. 16 for
``mma_m16n8k16_f16``) rather than 1 — each ``K_i`` iteration consumes
``atom_k`` K-elements via one ``mma.sync`` instruction.

Example (matmul A[M=128,K=128] @ B[K=128,N=128] with WN=WM=2, FM=FN=2,
BK=2, SPLITK=1, ATOM_KIND="mma_m16n8k16_f16" → cell shape (16,8,16),
``F16`` operands, ``F32`` accumulator):

    Output (σ_outer maps m/n through 4 tiers; σ_k stride = atom_k = 16;
    M_a / N_a Vars stay OUT of σ — the fragment-lane offset is owned by
    the mma.sync instruction, not the body indices):
        for m_b in 0..2 BLOCK:
            for n_b in 0..4 BLOCK:
                for m_w in 0..2 WARP:
                    for n_w in 0..2 WARP:
                        for m_r in 0..2 REGISTER:
                            for n_r in 0..2 REGISTER:
                                for m_a in 0..16 ATOM:    # structural marker;
                                    for n_a in 0..8 ATOM:     # absent from σ
                                        Init(acc)
                                        for k_o in 0..4 SERIAL_OUTER:
                                            for k_i in 0..2 STAGE_INNER reduce:
                                                a = load A[m_b·64 + m_w·32 + m_r·16, k_o·32 + k_i·16]
                                                b = load B[k_o·32 + k_i·16, n_b·32 + n_w·16 + n_r·8]
                                                Accum(acc, a*b)
                                        Write(C[m_b·64 + m_w·32 + m_r·16, n_b·32 + n_w·16 + n_r·8], acc)

The ``AtomTile(m_a, n_a)`` wrapper + its enclosed matmul-cell body
(``Init + K_o/K_i reduce + Write``) is consumed in the kernel pass
chain by ``kernel/005_lower_atom_tile``, which pattern-matches this
shape and rewrites it into an Mma* fragment chain
(``MmaFragment`` × 3 + ``MmaFill`` + per-K_i ``MmaLoad`` × 2 +
``MmaSync`` + final ``MmaStore``). The ``RegisterTile`` wrapper stays;
``kernel/010_split_register_axes`` replicates the Mma* chain per
``(m_r, n_r)`` cell, giving each cell its own fragment SSA names
(``c_frag_0_0`` / ``c_frag_0_1`` / …). After both kernel passes, the
``mma.sync`` instructions fire ``WM·WN·FM·FN·K_o·K_i`` times per CTA on
this shape.

For cooperative-K reduce (e.g. sum K=512 with BR=256, BK=2), K_c appears as
the INNERMOST THREAD axis and σ_k extends to ``k = k_o·512 + k_i·256 + k_c``;
the materializer emits the cross-thread combine after the reduce subtree
based on the escape-analysis helper (``Body.coordination`` reading
``Accum.axes ∩ ThreadTile.axes``). With BN·BM > 1 (strided-cooperative
rows — free-axis threads alongside the K lanes) the combine is a SEGMENTED
warp shuffle over each row's aligned BR-lane group, so the enumerator clips
those rows' BR to powers of two ≤ warp_size.

**Flash-style kernels** (an online-softmax ``Monoid`` reduce — plans/
atomic-free-monoid-combine.md Step 4) route here too: the KV (Monoid) reduce is
the cooperative-parallelization axis, NOT the nested score dot-product (a
matmul-reduce over head_dim that stays serial inside each KV step). When a
(static-extent) Monoid reduce is present the matmul branch is skipped, so the KV
axis splits across the CTA's threads and the per-thread partial ``(m, l, O)``
states merge via the monoid combine (``MonoidWarpShuffle`` / ``MonoidTreeHalve``
over the carrier's ``combine_states``). A symbolic (masked) KV stays serial — the
overhang-key → identity masking for a Monoid is a follow-up.

For the fused-prologue matmul (SDPA P@V — softmax max/sum/reciprocal sitting
as siblings of an inner output Loop that holds the actual matmul), the chain
walker extends through the inner Loop (``_classify_fused_prologue``), stashes
the sibling reduces + assigns as ``KernelShape.prologue``, and the body builder
places the σ-rewritten prologue inside the ``M_r`` REGISTER scope but outside
the ``N_r`` register tower. SPLITK clamps to 1 in this branch — the prologue
feeds the matmul, so each K_s CTA would consume a partial softmax stat.

Pointwise collapses to BM = FM = FN = 1; extent-1 sub-axes get inlined by
normalize_body. SPLITK + Write atomicity is derived at codegen time from
``escape_analysis.atomic_axes`` (Write index vs enclosing block axes).
Cooperative-K combine emission similarly happens in the materializer from
the helper's ``accum_cooperative_axes``.

Priority keys: matmul prefers high cells/thread (amortize K-loop overhead);
pointwise prefers low cells/thread (memory-bandwidth bound); cooperative
reduce prefers warp-sized-or-larger BR. All target ~256 threads/CTA.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import (
    ATOM_REGISTRY,
    Atom,
    AtomTile,
    GridTile,
    RegisterTile,
    SerialTile,
    ThreadTile,
    TileOp,
    WarpTile,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.fork import Fork, Level, build_fork_tree
from deplodock.compiler.pipeline.knob import is_warp
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import (
    BM,
    BN,
    BR,
    FM,
    FN,
    MMA,
    WM,
    WN,
    enumerate_cartesian,
    mma_mode,
)
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, segmentable_k_extent
from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import has_nonlinear_post_reduce_epilogue


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


PATTERN = [Pattern("root", LoopOp)]


@dataclass(frozen=True)
class KernelShape:
    """Per-LoopOp shape info that stays constant across every knob-row
    variant of a single kernel: the output axis Loops (innermost-N, optional
    M, extra outer chain), the K reduce Loop (None for pointwise), and the
    set of axis names ``_replace_k_loops`` should rewrite (collected once
    upfront by ``_plan_kernel`` instead of re-classified per variant).
    """

    outer_n: Loop
    outer_m: Loop | None
    extra_outer: tuple[Loop, ...]
    k_loop: Loop | None
    target_names: frozenset[str]
    # Sibling stmts of ``outer_n`` at the inner level where the chain stopped.
    # Non-empty only for the fused-prologue matmul (softmax max/sum/reciprocal
    # → SDPA P@V); empty for plain matmul / pointwise / cooperative-reduce.
    prologue: tuple[Stmt, ...] = ()


@dataclass(frozen=True)
class _Plan:
    """Cheap-to-build planning state shared across every variant of one
    LoopOp. Holds the inputs ``_materialize`` needs to build any single
    TileOp on demand: the per-LoopOp ``shape``, the leading non-Loop body
    stmts, the LoopOp's carry-forward knobs, the rendered kernel name, and
    the enumerated knob rows sorted by score. The planner runs every
    classification + enumeration step once and stops short of body
    construction — ``_materialize(plan, params)`` runs the expensive
    ``_build_split_body`` + ``TileOp.__post_init__`` work for one
    variant only, called lazily from the chosen Fork leaf's ``expand``.
    """

    shape: KernelShape
    leading: tuple[Stmt, ...]
    # The LoopOp's carry-forward knobs, including its ``S_*`` structural
    # feature identity — merged under every scored row so structurally
    # identical kernels score identically.
    base_knobs: dict
    kernel_name: str
    params: tuple[dict, ...]


def rewrite(ctx: Context, root: Node, match) -> Graph | None | TileOp | Fork:
    """Emit one hierarchical Fork tree over knob bundles:
    ``MMA → BR → (BM,BN) → (WM,WN) → (FM,FN) → TileOp leaf`` — each leaf
    carries its COMPLETE knob row (incl. ``BK`` / ``SPLITK`` / ``FK`` /
    ``OVERHANG``), the DB-matchable variant identity.

    Both tiers live in the one tree: the root ``MMA`` level keys the warp
    rows by atom kind while scalar rows return an empty key and SKIP the
    level (their ``BR``-and-below subtree splices up as siblings of the
    atom branches — no ``MMA`` knob ever pins a scalar path), and the
    builder's single-value collapse erases tier-foreign levels — warp rows
    carry ``br = bm = bn = 1`` so the ``BR`` / ``(BM,BN)`` levels vanish
    inside a warp subtree, scalar rows carry ``wm = wn = 1`` so ``(WM,WN)``
    vanishes inside the scalar subtree. Warp-vs-scalar ordering is
    score-driven like every other sibling decision (an explicit
    ``DEPLODOCK_MMA=<kind>`` pin is authoritative — ``_plan_kernel`` drops
    the scalar tier so score can't sidestep the pin).

    Single-variant kernels short-circuit to a bare ``TileOp`` (the engine
    applies it inline without a fork point). Most matmul kernels have BR
    fixed at 1, so they actually emit a 3-level scalar tree.

    Variant *materialization* (``_build_split_body`` + ``TileOp(...)``
    which runs the body-normalize pipeline) is deferred to the leaf
    Fork's ``expand`` thunk, so greedy compile only builds the one
    chosen variant per LoopOp. ``_plan_kernel`` runs the cheap up-front
    classification + enumeration and produces a ``_Plan`` with bare
    knob rows; sibling ranking is the search policy's job (the learned
    prior over the row knobs — Forks carry no score), so the planner
    never materializes or scores a variant itself.

    ``build_fork_tree`` constructs branch Forks lazily — a whole-model
    compile builds O(path) Forks per kernel instead of one per enumerated
    variant."""
    loop_op: LoopOp = root.op
    # Name was stamped onto the LoopOp by ``loop/stamp/010_stamp_loop_names``
    # (the last loop-dialect pass), so we just forward it onto the TileOp.
    kernel_name = loop_op.name
    # Idempotence is structural: once the planner has built a TileOp, the
    # rule pattern (LoopOp) no longer matches.
    plan = _plan_kernel(loop_op, ctx, kernel_name=kernel_name, graph=match.graph)
    if plan is None:
        raise RuleSkipped("kernel shape not handled by planner (or already planned)")

    if len(plan.params) == 1:
        return _materialize(plan, plan.params[0])

    return build_fork_tree(
        params=plan.params,
        levels=[
            Level((MMA.name,), lambda p: (p["MMA"],) if is_warp(p) else ()),
            Level((BR.name,), lambda p: (p.get("BR", 1),)),
            Level((BM.name, BN.name), lambda p: (p.get("BM", 1), p.get("BN", 1))),
            Level((WM.name, WN.name), lambda p: (p.get("WM", 1), p.get("WN", 1))),
            Level((FM.name, FN.name), lambda p: (p["FM"], p["FN"])),
        ],
        materialize=lambda p: _materialize(plan, p),
    )


def _split_leading_non_loops(body) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Split body into ``(leading non-Loop stmts, rest)``. Mirrors
    ``001_launch_geometry``'s prefix handling."""
    leading: list[Stmt] = []
    rest = tuple(body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]
    return tuple(leading), rest


def _outer_free_loop_chain(body) -> tuple[tuple[Loop, ...], tuple[Stmt, ...]]:
    """Walk the outer single-stmt chain of untagged free Loops, outermost-first.

    Returns ``(chain, prologue)``. ``prologue`` is empty for the plain
    matmul / pointwise / cooperative-reduce paths. The fused-prologue
    matmul (SDPA P@V — softmax max/sum/reciprocal sitting as siblings
    of the head_dim Loop) is detected via :func:`_classify_fused_prologue`:
    when the chain stops because the body has siblings split as
    ``[reduces..., assigns..., one non-reduce Loop containing a Write]``,
    pull the inner Loop into the chain (so ``outer_n`` becomes the buried
    output-N axis) and surface the siblings as ``prologue``."""
    _, rest = _split_leading_non_loops(body)
    out: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        out.append(cur[0])
        cur = tuple(cur[0].body)

    prologue, inner_loop = _classify_fused_prologue(cur)
    if inner_loop is not None:
        out.append(inner_loop)
        return tuple(out), prologue
    return tuple(out), ()


def _classify_fused_prologue(stmts: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], Loop | None]:
    """Match the fused-prologue matmul shape: ``[reduce Loops..., leaf
    stmts..., exactly one non-reduce Loop whose body transitively contains
    a matmul-shape reduce]``. Returns ``(prologue, inner_loop)`` on
    match, ``((), None)`` otherwise.

    The matmul-reduce requirement on the inner Loop is what keeps this
    helper from over-matching non-matmul kernels with sibling shapes:
    RMSNorm / softmax / mean follow ``[reduce..., assigns, Loop(post-
    pointwise)]`` too, but their post-pointwise Loop has no matmul-shape
    reduce — those fall through to the cooperative-K path the planner
    already handles.

    Bail conditions: a second non-reduce Loop at this level, an
    unexpected wrapper (``StridedLoop``), or a naked Write."""
    prologue: list[Stmt] = []
    inner: Loop | None = None
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce:
            prologue.append(s)
        elif isinstance(s, Loop) and not s.is_reduce and inner is None and _contains_matmul_reduce(s):
            inner = s
        elif isinstance(s, (Loop, StridedLoop)):
            return (), None
        elif isinstance(s, Write):
            return (), None
        else:
            prologue.append(s)
    return (tuple(prologue), inner) if inner is not None else ((), None)


def _contains_matmul_reduce(stmt: Stmt) -> bool:
    """True if ``stmt`` is or transitively contains a matmul-shape reduce
    Loop (``is_matmul_reduce``: reduce body has ≥ 2 K-indexed Loads + an
    Accum)."""
    if isinstance(stmt, Loop) and stmt.is_reduce and is_matmul_reduce(stmt):
        return True
    for sub in stmt.nested():
        for c in sub:
            if _contains_matmul_reduce(c):
                return True
    return False


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
    filtered: list[tuple[Axis, Role | None]] = []
    for axis, role in layers:
        if axis.extent.is_static and axis.extent.as_static() == 1 and role not in _STRUCTURAL_ROLES:
            sub = Sigma({axis.name: Literal(0, "int")})
            inner_body = tuple(c.rewrite(_identity_rename, sub) for c in inner_body)
            continue
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


def _is_fp16_matmul(matmul_reduces: list, graph: Graph | None) -> bool:
    """True iff every K-indexed operand ``Load`` in every matmul-reduce loop
    resolves to an ``F16`` source buffer. Loop-IR Loads don't carry a dtype
    until ``kernel/030_stamp_types``, so the operand dtype comes off
    ``graph.nodes[buf].output.dtype`` (mirrors ``_atom._mma_eligible_factory``).
    Gates the fp16 half2 accumulation window — fp32 / bf16 / mixed matmuls keep
    the scalar fp32-accumulate path."""
    if graph is None or not matmul_reduces:
        return False
    for k_loop in matmul_reduces:
        K_name = k_loop.axis.name
        k_loads = [ld for ld in k_loop.body.iter_of_type(Load) if K_name in {v for e in ld.index for v in e.free_vars()}]
        if not k_loads:
            return False
        for ld in k_loads:
            node = graph.nodes.get(ld.input)
            if node is None or node.output.dtype != F16:
                return False
    return True


def _coop_reduce_ext(ext) -> int:
    """Cooperative-reduce eligibility extent: the static reduce extent, or the
    ``Dim`` hint for a SYMBOLIC reduce axis (the masked cooperative reduce —
    tiled at the hint, ceil-div ``K_o``, with the partial last tile boundary-
    masked to the Accum identity). ``0`` when symbolic with no hint, so the
    ``>= warp_size`` gate falls through to the pointwise path."""
    return ext.as_static() if ext.is_static else (ext.hint or 0)


def _plan_kernel(loop_op: LoopOp, ctx: Context, *, kernel_name: str = "", graph: Graph | None = None) -> _Plan | None:
    """Unified σ-split planning for matmul, pointwise, and cooperative-reduce
    kernels. Returns a :class:`_Plan` whose ``params`` enumerates every
    candidate knob row but doesn't materialize any TileOp — the
    expensive ``_build_split_body`` + ``TileOp.__post_init__`` work is
    deferred to :func:`_materialize`, invoked from the chosen Fork leaf's
    ``expand`` thunk in the ``build_fork_tree`` tree.

    Detection is predicate-driven: ``is_matmul_reduce`` (≥ 2 K-indexed Loads +
    Accum) picks the matmul knob set; any other reduce with extent ≥ warp_size
    picks the cooperative-K set; no qualifying reduce falls to pointwise.

    **Phantom outer axis for chain-less kernels.** Body shapes without an
    outer free-Loop chain — e.g. global reductions ``[Loop(k, reduce),
    Write(o[0])]`` — would otherwise have no axis to lift into
    ``GridTile`` / ``ThreadTile``. We synthesize a size-1 ``__phantom__``
    axis and wrap the LoopOp body in one outer free Loop so the rest of
    the partitioner runs unchanged. ``_wrap_tower``'s size-1-axis filter
    drops the phantom before it reaches the IR.
    """
    chain, prologue = _outer_free_loop_chain(loop_op.body)
    if not chain:
        # Synthesize a single size-1 outer Loop so the rest of the
        # partitioner has an axis to lift. We can't actually wrap it in a
        # new LoopOp — ``LoopOp.__post_init__`` runs ``drop_size_one_free_axes``
        # which would inline the phantom back out. Build the phantom Loop
        # as a chain entry directly; downstream consumers (``shape.outer_n``,
        # ``_split_leading_non_loops``) only need the Loop object and its
        # body, not a containing LoopOp.
        phantom_axis = Axis("__phantom__", 1)
        phantom = Loop(axis=phantom_axis, body=Body(loop_op.body))
        chain = (phantom,)
        prologue = ()

    outer_n: Loop = chain[-1]
    outer_m: Loop | None = chain[-2] if len(chain) >= 2 else None
    extra_outer: tuple[Loop, ...] = chain[:-2] if outer_m is not None else chain[:-1]
    # Symbolic free axes (Dim("seq_len") etc.) use their hint as the expected
    # size and always emit a MASKED (overhang) tile: the enumerator picks tile
    # sizes for the hint, ``_build_split_body`` ceil-divs the symbolic extent
    # for the grid, and a boundary Cond gates lanes past the runtime value.
    # ``hint`` is set at trace from ``--seq-len`` (assumed present on any
    # symbolic axis the planner reaches).
    n_symbolic = not outer_n.axis.extent.is_static
    m_symbolic = outer_m is not None and not outer_m.axis.extent.is_static
    E_N = outer_n.axis.extent.hint if n_symbolic else outer_n.axis.extent.as_static()
    E_M = (outer_m.axis.extent.hint if m_symbolic else outer_m.axis.extent.as_static()) if outer_m is not None else 1

    # Single walk: classify body + collect every axis name _replace_k_loops
    # should rewrite. ``target_names`` survives σ_outer (only axis NAMES are
    # used downstream, not Loop identity — names don't change under σ).
    all_loops: tuple[Loop, ...] = outer_n.body.iter_of_type(Loop)
    # Route on each reduce loop's bottom-up algebra tag (Part C1a of
    # plans/algebraic-carrier-analysis.md) instead of re-deriving the archetype:
    # ``SEMIRING`` → matmul-output tiling; everything else → cooperative-reduce.
    reduce_loops = [lp for lp in all_loops if lp.is_reduce]
    matmul_reduces = [lp for lp in reduce_loops if lp.algebra_kind is AlgebraKind.SEMIRING]
    nonmatmul_reduces = [lp for lp in reduce_loops if lp.algebra_kind is not AlgebraKind.SEMIRING]
    # Flash-style kernel (online-softmax ``TWISTED_MONOID`` reduce): the KV reduce
    # is the cooperative-parallelization axis (Step 4 of
    # plans/atomic-free-monoid-combine.md), NOT the nested score dot-product (a
    # matmul-reduce over head_dim that stays serial inside each KV step). When a
    # twisted monoid is present, route to the cooperative-reduce path so the KV
    # axis splits across the CTA's threads (Step 2's monoid combine) instead of
    # the matmul-output tiling that leaves KV serial. ``commutative`` (split-KV
    # legality) is True for the LSE monoid. Restricted to a STATIC KV extent: a
    # SYMBOLIC (masked) KV needs the overhang keys folded to the monoid identity
    # (score → −inf so exp → 0), which the ``Accum``-only ``_mask_reduce_accums``
    # doesn't do for a Monoid — masked symbolic flash stays on its existing
    # serial-KV path (a follow-up).
    combine_reduces = [lp for lp in nonmatmul_reduces if lp.axis.extent.is_static and lp.algebra_kind is AlgebraKind.TWISTED_MONOID]

    k_loop: Loop | None
    target_names: frozenset[str]
    param_combos: list[dict]
    if matmul_reduces and not combine_reduces:
        if outer_m is None:
            return None
        k_loop = matmul_reduces[0]
        # Symbolic M / N is allowed: planner forces BM/BN/FM/FN=1 so each output
        # element runs on its own CTA with a serial K loop. Symbolic K also runs
        # via the same path (BK=SPLITK=BR=1 — whole K stays as one serial
        # iteration inside the per-output-element CTA). Both are inefficient
        # but correct; perf follow-up uses strided cooperative threads.
        k_symbolic = not k_loop.axis.extent.is_static
        E_K = 1 if k_symbolic else k_loop.axis.extent.as_static()
        # Warp-tier masked K (SDPA P@V's symbolic seq_len): the warp enumeration
        # below tiles K at its hint and zero-fills the final partial K slab, so
        # it needs the hint extent (the scalar path keeps E_K=1 — a single
        # serial reduce over the runtime extent). A masked-K warp tile only
        # deploys for a clean matmul (no fused prologue): the softmax stats of
        # an SDPA prologue can't share the zero-filled slab, so a fused-prologue
        # symbolic-K matmul stays scalar and its deployment path is the split.
        E_K_warp = k_loop.axis.extent.hint if k_symbolic else E_K
        # target_names unions over outer_n.body (the matmul K reduces) and
        # the prologue (softmax max/sum reduces sharing the matmul K extent,
        # axis-name-unified by unify_sibling_reduce_axes upstream). For
        # plain matmul / matmul_add / gated_mlp the prologue is empty and
        # this collapses to ``{lp.axis.name for lp in matmul_reduces}``.
        prologue_reduces = tuple(lp for lp in Body(prologue).iter_of_type(Loop) if lp.is_reduce and lp.axis.extent == k_loop.axis.extent)
        target_names = frozenset((*(lp.axis.name for lp in matmul_reduces), *(lp.axis.name for lp in prologue_reduces)))
        # SPLITK > 1 only works when each Write's atomic-add is mathematically
        # equivalent to the unsplit reduce. SPLITK is forced off for two
        # patterns:
        #
        # 1. Non-linear post-reduce combines like ``silu(acc_g) * acc_u``
        #    (gated_mlp) or softmax (sdpa). Conservative proxy: any matmul-
        #    reduce loop with > 1 Accum fuses multiple K-sums into one
        #    output cell — must be non-linear (else fusion would have
        #    merged them upstream).
        # 2. A non-empty prologue (SDPA P@V) — softmax max/sum/reciprocal
        #    are *consumed* by the matmul reduce, so each K_s CTA's
        #    partial softmax stat would feed a partial matmul.
        #
        # A post-reduce *linear* epilogue (``matmul_add``: ``Load(r)`` +
        # ``Assign(v = acc + r)`` → ``Write(v)``) IS allowed at SPLITK > 1.
        # ``_build_split_body`` rewrites the body so the residual add is
        # hoisted into a ``Cond(K_s == 0)`` — the K_s == 0 CTA atomic-adds
        # ``acc + r`` while every other K_s CTA atomic-adds just ``acc``,
        # so ``sum_i acc_i + r = c·sum_k a_k + r``. Non-linear epilogues
        # (e.g. ``v = acc * r``) fall through here and force splitk=1.
        multi_accum = any(sum(1 for s in lp.body if isinstance(s, Accum)) > 1 for lp in matmul_reduces)
        has_nonlinear_epilogue = has_nonlinear_post_reduce_epilogue(outer_n.body)
        force_splitk_one = multi_accum or bool(prologue) or has_nonlinear_epilogue
        # A fused prologue (SDPA P@V: softmax max/sum) carries a per-M-row
        # reduction whose accumulators must reset per register cell — masking
        # with FM/FN > 1 would register-tile the row and share one accumulator
        # across cells, which is wrong. THREAD-level masking is safe for the
        # SYMBOLIC-K prologue class (SDPA P@V — K = seq_len): the boundary
        # Cond wraps the whole per-row body (prologue + matmul, placed inside
        # SerialTile(M_r) by the prologue branch of ``_build_split_body``)
        # and ``mask_f1`` clamps the MASKED axis's register tiling to 1 (the
        # unmasked axis keeps its F sweep — a static-N register cell never
        # shares a row accumulator) — no shared accumulators, and nothing stages (a
        # symbolic K never builds a slab), so no collective lives under the
        # divergent guard. A STATIC-K prologue kernel (fused gated-MLP)
        # stays degenerate on its symbolic rows: its K pipeline stages, and
        # ``021_hoist_staged_loads_above_mask`` would hoist the staged
        # matmul above the Cond while the prologue chain it consumes
        # (rsqrt of the row stats) stays guarded below — an SSA-ordering
        # break (undefined value at render). Its deployment path is the
        # structural split (``005_split_demoted``, which now offers on
        # symbolic rows); masked staged prologues are follow-up work.
        prologue_mask_ok = (not prologue) or k_symbolic
        mask_f1 = bool(prologue)
        # fp16 half2 window (FK on the scalar matmul path): enabled only when
        # every K-indexed operand Load is fp16 and there's no fused prologue
        # (SDPA P@V's softmax stats would interleave with the windowed flush).
        # The window trades a bounded fp16-accumulation error for 2× packed
        # ``__hfma2`` throughput — see ``plans/fk-half2-fp16-matmul.md``.
        fp16_window = (not prologue) and not k_symbolic and _is_fp16_matmul(matmul_reduces, graph)
        param_combos = enumerate_cartesian(
            E_M=E_M if (prologue_mask_ok or not m_symbolic) else 1,
            E_N=E_N if (prologue_mask_ok or not n_symbolic) else 1,
            E_K=E_K,
            ctx=ctx,
            priority_mode="matmul",
            force_splitk_one=force_splitk_one,
            m_axis_name=outer_m.axis.name if outer_m is not None else None,
            n_axis_name=outer_n.axis.name,
            m_forced_mask=m_symbolic and prologue_mask_ok,
            n_forced_mask=n_symbolic and prologue_mask_ok,
            mask_f1=mask_f1,
            fp16_window=fp16_window,
        )
        # Warp-tier MMA: only when the MMA knob is enabled (default;
        # ``DEPLODOCK_MMA=0`` for scalar-only, ``DEPLODOCK_MMA=<kind>``
        # to enable + pin one atom kind — see ``_enumeration.mma_mode``),
        # no prologue (M9 extension), static K, and the per-kind
        # eligibility predicate fires. Symbolic M and/or N are admitted as
        # MASKED warp tiles (tiled at the hint; ceil-div grid + per-element
        # store guard — the M9 path): the enumerator stamps ``OVERHANG``
        # via the forced-mask flags, the builder emits the boundary Cond,
        # and a symbolic-N output resolves its ldm from the runtime kernel
        # arg at render (``_resolve_ldm``). Symbolic K (flash-style
        # attention) is out of scope. Each eligible kind gets one
        # warp-tier knob row per (WN, WM, FM, FN, BK, SPLITK); we
        # concatenate them onto the scalar param list — the fork tree in
        # :func:`rewrite` splits the rows by type and emits sibling
        # subtrees with disjoint level schemas.
        from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible  # noqa: PLC0415

        mma_on, pinned_atom = mma_mode()
        # Warp tier admits a symbolic (masked) K, but only for a CLEAN matmul
        # (the ``not prologue`` gate below): the K reduce is tiled at the hint
        # and the partial final slab is zero-filled in smem (``_stage_expand``),
        # so the mma accumulates zero past the runtime seq_len. A fused-prologue
        # matmul (SDPA P@V before the demoted-matmul split) keeps K scalar — its
        # softmax stats can't co-exist with the zero-filled slab; its warp-tier
        # path is the structural split, which hands the consumer a clean
        # symbolic-K matmul that reaches here.
        if mma_on and graph is not None and not prologue:
            eligible = tuple(atom for atom in ATOM_REGISTRY.values() if is_atom_eligible(atom, loop_op, ctx, graph=graph))
            # The s16816 ``mma.sync`` + ``ldmatrix`` path is the sole
            # tensor-core family (WMMA was removed; the swizzled slab beats
            # it). It auto-enumerates alongside the scalar register-tile tier
            # on **sm_80+** — ``is_atom_eligible`` gates ``min_cc=(8, 0)``, so
            # pre-Ampere never sees it. sm_90+ gets the swizzled-TMA fast path;
            # sm_80-89 stages the operands through cp.async (the ``ldmatrix``
            # loads carry the ``.shared`` state-space qualifier so ptxas keeps
            # the smem offset as-is instead of folding a spurious
            # generic->shared conversion — without it ``LDSM`` faults on Ada).
            # The greedy/DB-less picker and the autotuner choose mma.sync vs
            # scalar per shape. Shapes mma.sync can't serve fall to scalar:
            # non-divisible extents (filtered by ``is_atom_eligible``) and
            # single-warp tiles (pruned below — ``020_stage_inputs`` skips
            # staging at one warp and ldmatrix is smem→register only, so an
            # unstaged AtomTile can't lower). A ``DEPLODOCK_MMA=<kind>``
            # pin forces the kind at any size / arch (bring-up + A-B
            # benching).
            if eligible:
                warp_combos = enumerate_cartesian(
                    E_M=E_M,
                    E_N=E_N,
                    E_K=E_K_warp,
                    ctx=ctx,
                    priority_mode=("matmul", "warp"),
                    force_splitk_one=force_splitk_one,
                    atoms=eligible,
                    m_axis_name=outer_m.axis.name if outer_m is not None else None,
                    n_axis_name=outer_n.axis.name,
                    m_forced_mask=m_symbolic,
                    n_forced_mask=n_symbolic,
                    k_axis_name=k_loop.axis.name,
                    k_forced_mask=k_symbolic,
                )
                # Drop single-warp (WM·WN == 1) variants: ldmatrix is
                # smem→register only, so the atom REQUIRES staged operands, but
                # ``020_stage_inputs`` skips staging when the CTA is one warp
                # (``n_thread <= warp_size``) — an unstaged AtomTile would crash
                # at render. The scalar tier covers those tiny tiles. Single-warp
                # is never the perf pick anyway, so pruning is free.
                warp_combos = [p for p in warp_combos if p["WM"] * p["WN"] != 1]
                if pinned_atom is not None and warp_combos:
                    # An explicit ``DEPLODOCK_MMA=<kind>`` pin is authoritative
                    # (mirrors ``Knob.narrow``): drop the scalar tier so the
                    # score-driven sibling order can't sidestep the pin. A
                    # structurally unworkable pin (no surviving warp rows)
                    # keeps the scalar fallback.
                    param_combos = warp_combos
                else:
                    param_combos = [*warp_combos, *param_combos]
    elif (coop_reduces := combine_reduces or nonmatmul_reduces) and _coop_reduce_ext(coop_reduces[0].axis.extent) >= ctx.warp_size:
        # Cooperative-K: with BN=BM=1 the combine spans the whole CTA (any
        # BR); with free-axis threads alongside (BN·BM > 1, the strided-
        # cooperative form) the enumerator clips BR to powers of two ≤
        # warp_size and the materializer emits a SEGMENTED warp-shuffle
        # combine per row (``_combine.cooperative_combine_geometry`` —
        # K_c is the innermost THREAD layer, see the tower below).
        # E_K ≥ warp_size: smaller reduces don't justify a warp-shuffle.
        # target_names includes both K-reduce axes AND per-K post-pointwise
        # axes (non-reduce free Loops sharing E_K), since both get rewritten.
        #
        # SPLITK is restricted to 1 here: cross-CTA reduce for cooperative-K
        # would need atomic accumulation of the partial sums (the per-CTA
        # Monoid only reduces *within* a CTA), plus a barrier before the
        # post-reduce pointwise epilogue reads the final value. Neither is
        # wired up today — the K_s=0 CTA would race with K_s>0 CTAs that
        # are still writing partial sums, and only K_s=0 writes the output
        # using its own (half-data) reduction. Forcing SPLITK=1 keeps the
        # search space honest.
        # Prefer the Monoid (flash KV) reduce as the cooperative axis when present,
        # else the first plain non-matmul reduce (softmax / norm).
        k_loop = coop_reduces[0]
        k_red_symbolic = not k_loop.axis.extent.is_static
        E_K = _coop_reduce_ext(k_loop.axis.extent)  # static extent, or the Dim hint for a symbolic reduce
        # ``is_static`` guards the ``as_static()`` read: a symbolic free Loop
        # in the body (e.g. a split producer's row sweep next to a static
        # row-stat reduce) is never a K-target, not a crash. For a SYMBOLIC
        # reduce axis (masked cooperative reduce) the K-targets are the loops
        # sharing that symbolic extent — matched by ``Dim`` identity, not a
        # static value.
        if k_red_symbolic:
            target_names = frozenset(lp.axis.name for lp in all_loops if lp.axis.extent == k_loop.axis.extent and not is_matmul_reduce(lp))
        else:
            target_names = frozenset(
                lp.axis.name
                for lp in all_loops
                if lp.axis.extent.is_static and lp.axis.extent.as_static() == E_K and not is_matmul_reduce(lp)
            )
        # NOTE: this is the symbolic-*free*-axis case. A symbolic *reduce* axis
        # (the softmax-producer key sweep) is now handled above — ``_coop_reduce_ext``
        # gates it in at the Dim hint, the K_o ceil-divs the symbolic extent, and
        # ``_mask_reduce_accums`` masks the partial last tile (clamp the K read,
        # fold the Accum identity past seq_len) WITHOUT wrapping the Accum in a
        # Cond, so ``is_reduce`` + the cross-thread combine stay intact.
        #
        # A SYMBOLIC free axis stays degenerate (E=1, no forced mask): bound
        # whole-to-grid, correct at any seq_len. A masked register-tile
        # (FN>1) would wrap the reduce body in the boundary Cond and hide it
        # from the cooperative-reduce + smem-staging passes, breaking the
        # cross-thread combine — and the masked BR=1 thread-tile family
        # (rows-per-CTA at the hint with the boundary guard) is DEFERRED for
        # the same 021-hoist SSA interaction that keeps static-K
        # fused-prologue matmuls degenerate (a reduce kernel's staged row
        # sweep plus its post-reduce output loop consume the rsqrt chain
        # between them; see the prologue_mask_ok comment in the matmul
        # branch — 021 refuses such lifts defensively).
        #
        # Thread-level parallelism on a symbolic-row kernel instead comes
        # from its STATIC free axes (strided-cooperative rows): BN/BM bind
        # static rows as threads alongside the K_c cooperative lanes, so
        # e.g. a per-head RMSNorm with symbolic seq deploys
        # ``grid=(seq·heads/BN), CTA=BN×BR`` rather than the v1 8-thread
        # degenerate CTA. No masking involved — static rows are divisor-
        # checked, the symbolic axis keeps its exact symbolic grid.
        param_combos = enumerate_cartesian(
            E_M=1 if m_symbolic else E_M,
            E_N=1 if n_symbolic else E_N,
            E_K=E_K,
            ctx=ctx,
            priority_mode="reduce",
        )
    else:
        # Pointwise — no qualifying reduce.
        k_loop = None
        target_names = frozenset()
        param_combos = enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=1,
            ctx=ctx,
            priority_mode="pointwise",
            m_axis_name=outer_m.axis.name if outer_m is not None else None,
            n_axis_name=outer_n.axis.name,
            m_forced_mask=m_symbolic,
            n_forced_mask=n_symbolic,
        )

    shape = KernelShape(
        outer_n=outer_n,
        outer_m=outer_m,
        extra_outer=extra_outer,
        k_loop=k_loop,
        target_names=target_names,
        prologue=prologue,
    )
    leading, _ = _split_leading_non_loops(loop_op.body)
    if not param_combos:
        return None
    plan = _Plan(
        shape=shape,
        leading=leading,
        base_knobs=dict(loop_op.knobs),
        kernel_name=kernel_name,
        params=tuple(param_combos),
    )
    return plan


def _materialize(plan: _Plan, params: dict) -> TileOp:
    """Build one ``TileOp`` for a single knob row against the
    planner's pre-computed shape. The expensive bits — ``_build_split_body``
    and ``TileOp.__post_init__`` (which runs ``normalize_body`` over the
    fresh body) — happen here, lazily, from the chosen Fork leaf's
    ``expand`` thunk.

    The ``RuleSkipped`` catch is shape-determined (raised only when
    ``_replace_k_loops`` can't find any K-axis Loops in the body, which
    depends on ``shape.target_names`` matching the body's axis names, not
    on params). If it ever fires here, the shape was misclassified at
    plan time — the assertion surfaces the bug instead of silently
    dropping a leaf.
    """
    try:
        chain_body = _build_split_body(plan.shape, params)
    except RuleSkipped as exc:
        raise AssertionError(f"shape-level RuleSkipped fired at materialize time for kernel {plan.kernel_name!r}: {exc}") from exc
    knobs = {**plan.base_knobs, **params}
    # Drop leading stmts whose SSA name is *also* defined inside ``chain_body``.
    # A pre-loop invariant (e.g. ``v0 = reciprocal(1024)``) used only by a
    # post-reduce stmt gets pulled into the thread scope by the body builder as
    # that stmt's dependency; prepending it here too would put two defs of the
    # same name in nested scopes — invalid CUDA ("v0 already declared"). The
    # outer copy is dead (nothing at the outer scope uses it), so it's safe to
    # drop. Leading stmts genuinely used at the outer scope aren't re-defined
    # inside, so they're kept.
    inner_defs = {name for s in Body.coerce(chain_body).iter() for name in s.defines()}
    leading = tuple(s for s in plan.leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=leading + chain_body, name=plan.kernel_name, knobs=knobs)


def _build_split_body(shape: KernelShape, params: dict) -> tuple[Stmt, ...]:
    """σ-split ``shape.outer_n``'s body and wrap in the output
    BLOCK/THREAD/REGISTER tower. ``shape.outer_m`` / ``shape.k_loop`` are
    None for 1D pointwise / non-reduce kernels.

    K_s and K_c (when present) are shared across all reduces in the kernel;
    K_o / K_i are per-K-Loop, built inside ``_replace_k_loops``. SPLITK
    atomic-Write is deferred to ``001_launch_geometry``.

    Fused-prologue matmul (``shape.prologue`` non-empty, SDPA P@V): the
    prologue runs inside the ``M_r`` REGISTER scope but *outside* the
    ``N_r`` register tower — softmax stats are per-seq-q-row (per M_r),
    independent of the output N. σ_outer + σ_k are applied uniformly so
    the prologue's M-axis references resolve to the in-scope ``M_r``.

    The warp-tier MMA builder lands here in M3 of
    ``plans/mma-fragment-factorization.md``. At M1 no warp-tier row
    ever reaches this function (the warp enumerator returns []) — the
    ``atom is not None`` dispatch routes warp-tier rows to
    :func:`_build_split_body_warp` (M3); the rest of this function handles
    the scalar tier.
    """
    if is_warp(params):
        return _build_split_body_warp(shape, params)
    sigma_map: dict[str, object] = {}

    # source_axis: every sub-axis points back to the original Axis it was
    # carved out of (= ``shape.outer_n.axis`` for N, etc.). Downstream
    # passes use this to group surrounding axes by source identity (e.g.
    # the MMA factorization plan's BLOCK·GROUP·CELL·ATOM enumeration along
    # each output axis) without name-suffix string matching.
    overhang = frozenset(params.get("OVERHANG", ()))
    N_axis = shape.outer_n.axis
    N_name = N_axis.name
    N_src = N_axis.source_axis or N_axis
    n_bnfn = params["BN"] * params["FN"]
    # ``n_bound`` is the masked-boundary Cond RHS (an Expr): ``Literal(E_N)`` for
    # a static overhang axis, the symbolic ``Var`` for a hint-driven one, None
    # when N isn't masked.
    n_bound: object | None = None
    # Masked tiles (N in overhang): ceil-div so the boundary CTA covers the
    # partial last tile, and ``n_bound`` carries the Cond RHS so the materializer
    # gates boundary lanes (``if decoded < n_bound``). ``Dim.ceil_div`` is one
    # formula for both regimes — it folds to the integer ceil for a static extent
    # and builds the composite ceil-div Expr (launch-resolver-evaluated) for a
    # symbolic one. A clean (non-masked) tile floor-divides; the degenerate
    # symbolic case (bn=fn=1) leaves the whole axis on the GridTile (``ext // 1``).
    # ``real_extent`` (the static pre-ceil bound) only applies when the extent is
    # statically known; a symbolic masked tile relies solely on ``n_bound``.
    n_masked = N_name in overhang
    N_b = Axis(
        f"{N_name}_b",
        N_axis.extent.ceil_div(n_bnfn) if n_masked else N_axis.extent // n_bnfn,
        source_axis=N_src,
        real_extent=N_axis.extent.as_static() if n_masked and N_axis.extent.is_static else None,
    )
    if n_masked:
        n_bound = N_axis.extent.expr
    N_t = Axis(f"{N_name}_t", params["BN"], source_axis=N_src)
    N_r = Axis(f"{N_name}_r", params["FN"], source_axis=N_src)
    # N register-tile decode — two layouts:
    #
    #   blocked      ``N = N_b·(BN·FN) + N_t·FN + N_r``  (thread-major)
    #   interleaved  ``N = N_b·(BN·FN) + N_r·BN + N_t``  (thread-minor)
    #
    # blocked keeps a thread's FN cells in contiguous columns. On a clean-
    # divisor tile the FN cells share one K-loop, so those contiguous
    # per-thread loads vectorize and the warp fully uses each cache line —
    # blocked wins, and it's also what staged-smem reads want (conflict-free
    # after permute_lane_accesses). A MASKED tile (N not a multiple of BN·FN
    # — e.g. the lm_head vocab=151669 projection) instead wraps each cell in
    # its own ``if (col < N)`` guard, splitting the FN cells into separate
    # K-loops: no cross-cell vectorization, and consecutive threads land
    # FN·elem bytes apart → fully uncoalesced global weight loads (~25×
    # wasted bandwidth; 94 ms → 3.5 ms on Qwen3 lm_head). Interleaved strides
    # the cell by BN so consecutive threads map to consecutive columns — the
    # warp reads BN contiguous columns per step (coalesced) despite the
    # per-cell guards. The Stage-fill (if any) and the register read share
    # this σ, so the column order stays self-consistent either way.
    n_block = Var(N_b.name) * Literal(params["BN"] * params["FN"], "int")
    if N_name in overhang:
        sigma_map[N_name] = n_block + Var(N_r.name) * Literal(params["BN"], "int") + Var(N_t.name)
    else:
        sigma_map[N_name] = n_block + Var(N_t.name) * Literal(params["FN"], "int") + Var(N_r.name)

    M_b = M_t = M_r = None
    m_bound: object | None = None
    if shape.outer_m is not None:
        M_axis = shape.outer_m.axis
        M_name = M_axis.name
        M_src = M_axis.source_axis or M_axis
        m_bnfm = params["BM"] * params["FM"]
        # Mirror of the N-axis split above: ``Dim.ceil_div`` unifies the static
        # and symbolic masked ceil-div, a clean tile floor-divides, ``real_extent``
        # is the static pre-ceil bound (None when symbolic).
        m_masked = M_name in overhang
        M_b = Axis(
            f"{M_name}_b",
            M_axis.extent.ceil_div(m_bnfm) if m_masked else M_axis.extent // m_bnfm,
            source_axis=M_src,
            real_extent=M_axis.extent.as_static() if m_masked and M_axis.extent.is_static else None,
        )
        if m_masked:
            m_bound = M_axis.extent.expr
        M_t = Axis(f"{M_name}_t", params["BM"], source_axis=M_src)
        M_r = Axis(f"{M_name}_r", params["FM"], source_axis=M_src)
        sigma_map[M_name] = (
            Var(M_b.name) * Literal(params["BM"] * params["FM"], "int") + Var(M_t.name) * Literal(params["FM"], "int") + Var(M_r.name)
        )

    sigma_outer = Sigma(sigma_map)

    # K axes: K_s / K_c are kernel-wide (single SPLITK / single cooperative
    # thread direction); K_o / K_i / K_f are per-K-Loop, built inside
    # _replace_k_loops. K_f (the FK multiple-accumulator register strip-mine of
    # the K serial loop) only exists for non-matmul reduces with ``FK > 1``.
    K_s = K_c = K_f = None
    K_o_ext: object = 0
    K_src: Axis | None = None
    # The fp16-matmul half2 window keeps the FK=1 fp32 K factorization here — the
    # window length is the stage chunk ``bk`` and the entire two-level
    # (fp16 window + fp32 master) rewrite is done downstream in
    # ``kernel/075_pack_fk_window`` off the stamped ``FK`` knob. Only the reduce
    # strip-mine consumes ``fk`` in the factorization / K_f tile below.
    fk = 1 if "FKWIN" in params else params["FK"]
    k_masked = False
    k_bound: object = None
    if shape.k_loop is not None:
        K_axis = shape.k_loop.axis
        K_name = K_axis.name
        K_src = K_axis.source_axis or K_axis
        k_div = params["SPLITK"] * params["BR"] * params["BK"] * fk
        # K_o serial bound. A masked cooperative reduce over a SYMBOLIC K (k_div >
        # 1) ceil-divs the symbolic extent so ``K_o`` covers the partial last tile
        # at any runtime size; the final overhang step is wrapped in
        # ``Cond(decoded_K < seq_len)`` (in ``_replace_k_loops``) so it skips (no
        # spurious fold, no OOB read/write). ``Dim.ceil_div`` is one formula for
        # both regimes — folds to the int ceil for a static extent, builds the
        # launch-resolver-evaluated Expr for a symbolic one. Static K and the
        # degenerate symbolic-serial case (BR=BK=SPLITK=fk=1, ``ext // 1``)
        # floor-divide and stay unmasked.
        k_masked = not K_axis.extent.is_static and k_div > 1
        K_o_ext = K_axis.extent.ceil_div(k_div) if k_masked else K_axis.extent // k_div
        if k_masked:
            k_bound = K_axis.extent.expr
        K_s = Axis(f"{K_name}_s", params["SPLITK"], source_axis=K_src) if params["SPLITK"] > 1 else None
        K_c = Axis(f"{K_name}_c", params["BR"], source_axis=K_src) if params["BR"] > 1 else None
        K_f = Axis(f"{K_name}_f", fk, source_axis=K_src) if fk > 1 else None

    # σ-rewrite outer_n's body (M/N axes), then replace every K-iter Loop with
    # a K_o · K_i tower. Both paths use shared canonical K_o / K_i names so
    # 020_stage_inputs row-cache can merge structurally-equivalent Loads.
    inner_after_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in shape.outer_n.body)

    if shape.k_loop is not None:
        new_inner, n_replaced = _replace_k_loops(
            inner_after_outer,
            target_names=shape.target_names,
            K_canonical_name=shape.k_loop.axis.name,
            K_s=K_s,
            K_c=K_c,
            K_f=K_f,
            br=params["BR"],
            bk=params["BK"],
            fk=fk,
            K_o_ext=K_o_ext,
            k_masked=k_masked,
            k_bound=k_bound,
        )
        if n_replaced == 0:
            raise RuleSkipped("K reduce not found in body")
        # SPLITK > 1 with a linear residual epilogue (matmul_add) is gated
        # post-planner by ``015_gate_splitk_residual``, which finds the
        # K_s axis in the wrapped GridTile and hoists the linear epilogue
        # under ``Cond(K_s == 0)`` so the residual is added exactly once
        # across the K_s CTAs. We only have to keep ``force_splitk_one``
        # at enumeration time so non-linear epilogues don't even reach
        # the gate pass.
    else:
        new_inner = inner_after_outer

    # Masked tiles: when ceil-div has rounded a block-axis extent up past the
    # axis bound, wrap the σ-rewritten body in ``Cond(decoded_axis_coord <
    # bound)``. ``bound`` is the static ``real_extent`` (lm_head vocab) or the
    # symbolic ``Var`` (hint-driven dynamic axis — resolved to the runtime
    # value at launch). The Cond sits INSIDE the register tower (N_r is in
    # scope for the predicate). The replicator in ``010_split_register_axes``
    # handles Conds whose predicate depends on the replicated axis by
    # σ-substituting per replica so each replicated body gets a partly-
    # constant-folded predicate (NVRTC drops always-true copies).
    if n_bound is not None:
        n_pred = sigma_outer.reduce(Var(N_name), SimplifyCtx({}))
        new_inner = (Cond(cond=BinaryExpr("<", n_pred, n_bound), body=Body(new_inner)),)
    if m_bound is not None and not shape.prologue:
        # Prologue kernels emit the M-Cond around the whole per-row body
        # (prologue + matmul tower) inside SerialTile(M_r) below — the
        # softmax max/sum loops index ``P[m, k]`` of a possibly
        # runtime-sized buffer, so an overhang row's prologue reads must be
        # guarded too, not just the matmul body.
        m_pred = sigma_outer.reduce(Var(M_name), SimplifyCtx({}))
        new_inner = (Cond(cond=BinaryExpr("<", m_pred, m_bound), body=Body(new_inner)),)

    # σ-rewrite + K-replace the prologue when present. The prologue sits
    # inside M_r (so M_r is in scope for σ_outer's M-axis mapping) and
    # outside N_r (softmax stats are N-invariant).
    prologue_rewritten: tuple[Stmt, ...] = ()
    if shape.prologue:
        prologue_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in shape.prologue)
        if shape.k_loop is not None:
            prologue_rewritten, _ = _replace_k_loops(
                prologue_outer,
                target_names=shape.target_names,
                K_canonical_name=shape.k_loop.axis.name,
                K_s=K_s,
                K_c=K_c,
                K_f=K_f,
                br=params["BR"],
                bk=params["BK"],
                fk=fk,
                K_o_ext=K_o_ext,
                k_masked=k_masked,
                k_bound=k_bound,
            )
        else:
            prologue_rewritten = prologue_outer

    # Wrap tower, innermost first. extent-1 layers (e.g. M_t / N_t under v1
    # cooperative BN=BM=1) are inlined later by normalize_body.
    #
    # Without prologue: REGISTER layers (N_r, M_r) join the same tower as
    # THREAD/BLOCK, wrapped innermost-first by _wrap_tower.
    #
    # With prologue: N_r wraps the matmul body alone; the prologue stmts
    # are prepended at the M_r scope, so the M_r Loop body becomes
    # ``[prologue..., N_r tower]``. Outer layers (THREAD/BLOCK/SPLITK)
    # then wrap that combined body. ``011_dedup_replicated`` later folds
    # the per-cell ``Load`` / ``Assign`` duplicates the replicator emits
    # so the lowered kernel matches what the deleted register-blocked
    # builder used to produce structurally.
    if shape.prologue:
        matmul_tower = _wrap_tower([(N_r, Role.REGISTER)], new_inner)
        body_inside_mr = prologue_rewritten + matmul_tower
        if m_bound is not None:
            # Masked M on a prologue kernel: the boundary Cond wraps the WHOLE
            # per-row body — prologue reduces (softmax max/sum over the
            # runtime-sized ``P[m, k]``) plus the matmul tower — as a unit.
            # FM = 1 on masked-M rows (``mask_f1``) keeps one row per
            # thread, so no per-row accumulator spans register cells, and no
            # collectives live inside (prologue forces SPLITK=1, BR=1) — the
            # divergence is benign; staged loads are lifted back out by
            # ``021_hoist_staged_loads_above_mask``.
            m_pred = sigma_outer.reduce(Var(M_name), SimplifyCtx({}))
            body_inside_mr = (Cond(cond=BinaryExpr("<", m_pred, m_bound), body=Body(body_inside_mr)),)
        if M_r is not None:
            # M_r stays serial (not RegisterTile) — the SDPA prologue
            # (softmax max/sum/reciprocal) computes once per output row
            # and must NOT be replicated per register cell. Only N_r is
            # the register-tile axis here.
            body_after_register = (SerialTile(axis=M_r, body=Body(body_inside_mr), kind="plain"),)
        else:
            body_after_register = body_inside_mr
        outer_layers: list[tuple[Axis, Role | None]] = [(N_t, Role.THREAD)]
        if M_t is not None:
            outer_layers.append((M_t, Role.THREAD))
        if K_c is not None:
            outer_layers.append((K_c, Role.THREAD))  # cooperative-K stride
        outer_layers.append((N_b, Role.BLOCK))
        if M_b is not None:
            outer_layers.append((M_b, Role.BLOCK))
        if K_s is not None:
            outer_layers.append((K_s, Role.BLOCK))  # split-K
        outer_layers.extend((lp.axis, Role.BLOCK) for lp in reversed(shape.extra_outer))
        return _wrap_tower(outer_layers, body_after_register)

    layers: list[tuple[Axis, Role | None]] = []
    layers.append((N_r, Role.REGISTER))
    if M_r is not None:
        layers.append((M_r, Role.REGISTER))
    if K_c is not None:
        # Cooperative-K stride — INNERMOST thread layer (fastest-varying
        # threadIdx bits), so when free-axis threads coexist (BN·BM > 1,
        # the strided-cooperative form) each row's BR lanes form a
        # contiguous aligned intra-warp segment — required by the
        # segmented-shuffle combine (``_combine.cooperative_combine_geometry``)
        # — and consecutive lanes read consecutive K elements (σ_k gives
        # K_c stride 1: coalesced). Invisible to the BN=BM=1 form, where
        # K_c is the only surviving THREAD layer either way.
        layers.append((K_c, Role.THREAD))
    layers.append((N_t, Role.THREAD))
    if M_t is not None:
        layers.append((M_t, Role.THREAD))
    layers.append((N_b, Role.BLOCK))
    if M_b is not None:
        layers.append((M_b, Role.BLOCK))
    if K_s is not None:
        layers.append((K_s, Role.BLOCK))  # split-K
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(shape.extra_outer))
    return _wrap_tower(layers, new_inner)


def _build_split_body_warp(shape: KernelShape, params: dict) -> tuple[Stmt, ...]:
    """σ-split + tower wrap for the warp-tier MMA matmul path.

    Mirrors :func:`_build_split_body` but with the 4-level output-axis
    factorization ``A_b · (W · F · A_atom) + A_w · (F · A_atom) + A_r ·
    A_atom + A_a`` (BLOCK > WARP > REGISTER > ATOM) and the
    ``atom_k``-strided K iteration. The body inside the AtomTile retains
    its Load+Accum shape; the MMA cell materializer
    (``kernel/010_split_register_axes`` MMA arm, M5) sees the AtomTile +
    the matmul-reduce body and rewrites it into the ``MmaFragment`` +
    ``MmaLoad`` + ``MmaSync`` chain.

    No prologue support (matches the warp enumerator gate); cooperative-K
    is gated off by construction (``BR=1`` on warp-tier rows).

    Masked tiles (the M9 path): an axis in the row's ``OVERHANG`` —
    stamped by the enumerator for a symbolic (forced-mask) axis, or a
    static non-divisor one — gets a ceil-div block axis and the body is
    wrapped in a boundary ``Cond(σ(axis) < bound)``, exactly like the
    scalar builder. The Cond gates the atom tile's BASE coordinate only
    (``M_a`` / ``N_a`` are not in σ); ``kernel/005_lower_atom_tile``
    classifies the predicate and stamps per-element guards onto the
    ``RegStore`` for tiles straddling the bound, and
    ``021_hoist_staged_loads_above_mask`` lifts the K-pipeline above the
    Cond (clamped slab fill, unguarded ldmatrix/mma.sync on the
    hint-sized slab). The predicate LHS is built with ``sigma_outer
    .apply`` — pure substitution, the same path the Write index takes —
    so 005's struct-equality classification holds.

    The ``M_a`` / ``N_a`` Vars *do not appear* in ``σ_outer`` — the
    in-fragment lane offset is owned by the mma.sync instruction, not the
    body indices. The materializer reads ``AtomTile.axes`` extents to
    know the fragment shape and emits one ``MmaSync`` per (M_r, N_r)
    cell.
    """
    atom = ATOM_REGISTRY[str(params["MMA"])]
    atom_m, atom_n, atom_k = atom.shape

    sigma_map: dict[str, object] = {}
    overhang = frozenset(params.get("OVERHANG", ()))

    def _block_axis(axis: Axis, name: str, per_block: int, src: Axis) -> tuple[Axis, object | None]:
        """The BLOCK-tier axis for one output dim + its mask bound (None when
        unmasked). Mirrors the scalar builder's static/masked/symbolic
        branches: masked static → ceil-div with ``real_extent``; masked
        symbolic → composite ceil-div Expr over the runtime extent (the
        launch resolver evals it; the bound is the symbolic Var)."""
        if axis.extent.is_static:
            ext = axis.extent.as_static()
            if name in overhang:
                return Axis(f"{name}_b", -(-ext // per_block), source_axis=src, real_extent=ext), Literal(ext, "int")
            return Axis(f"{name}_b", ext // per_block, source_axis=src), None
        if name in overhang:
            return Axis(f"{name}_b", (axis.extent + (per_block - 1)) // per_block, source_axis=src), axis.extent.expr
        raise RuleSkipped(f"warp-tier axis {name!r} is symbolic but not masked — enumerator/builder out of sync")

    # ---- N axis: A_b · (W_n · F_n · A_n) + A_w · (F_n · A_n) + A_r · A_n + A_a
    N_axis = shape.outer_n.axis
    N_name = N_axis.name
    N_src = N_axis.source_axis or N_axis
    n_per_block = params["WN"] * params["FN"] * atom_n
    N_b, n_bound = _block_axis(N_axis, N_name, n_per_block, N_src)
    N_w = Axis(f"{N_name}_w", params["WN"], source_axis=N_src)
    N_r = Axis(f"{N_name}_r", params["FN"], source_axis=N_src)
    N_a = Axis(f"{N_name}_a", atom_n, source_axis=N_src)
    sigma_map[N_name] = (
        Var(N_b.name) * Literal(n_per_block, "int")
        + Var(N_w.name) * Literal(params["FN"] * atom_n, "int")
        + Var(N_r.name) * Literal(atom_n, "int")
    )

    # ---- M axis (symmetric to N).
    assert shape.outer_m is not None, "warp-tier matmul requires an outer M axis"
    M_axis = shape.outer_m.axis
    M_name = M_axis.name
    M_src = M_axis.source_axis or M_axis
    m_per_block = params["WM"] * params["FM"] * atom_m
    M_b, m_bound = _block_axis(M_axis, M_name, m_per_block, M_src)
    M_w = Axis(f"{M_name}_w", params["WM"], source_axis=M_src)
    M_r = Axis(f"{M_name}_r", params["FM"], source_axis=M_src)
    M_a = Axis(f"{M_name}_a", atom_m, source_axis=M_src)
    sigma_map[M_name] = (
        Var(M_b.name) * Literal(m_per_block, "int")
        + Var(M_w.name) * Literal(params["FM"] * atom_m, "int")
        + Var(M_r.name) * Literal(atom_m, "int")
    )

    sigma_outer = Sigma(sigma_map)

    # ---- K axis: K_s · K_o · bk · atom_k + K_o · bk · atom_k + K_i · atom_k.
    assert shape.k_loop is not None, "warp-tier matmul requires a K reduce"
    K_axis = shape.k_loop.axis
    K_src = K_axis.source_axis or K_axis
    k_masked = not K_axis.extent.is_static
    # Segmented-K: if a matmul operand reads K folded as (strided-outer ×
    # contiguous-inner ``% C``) — the o_proj attn-out / P@V transpose layout —
    # pin the staged inner run to the contiguous extent ``C`` (``bk·atom_k ==
    # C``) so ``K_o`` lands on the segment boundary. The σ then makes the index
    # delinearization fold (range-aware simplify) to a clean ``[K_o, …, K_i]``
    # read: the matmul reaches the mma tier reading gmem directly, no transpose
    # producer. Overrides the tuned ``BK`` for that operand only.
    bk = params["BK"]
    seg_c = next(
        (c for ld in shape.k_loop.body.iter_of_type(Load) if (c := segmentable_k_extent(ld, K_axis.name)) is not None),
        None,
    )
    if seg_c is not None and seg_c % atom_k == 0:
        bk = seg_c // atom_k
    k_per_block = params["SPLITK"] * bk * atom_k
    # Masked K (SDPA P@V's symbolic seq_len): tile at the hint, ceil-div the
    # runtime extent for the K_o serial bound. The final partial K slab is
    # zero-filled in smem (``_stage_expand``), so the mma accumulates zero past
    # the runtime seq_len. The warp enumerator forces SPLITK=1 for a masked K, so
    # this ceil-div ``Dim`` never feeds the (SPLITK>1) K_s σ term — only the K_o
    # serial-loop bound, which renders the ceil-div Expr. A static K floor-divides
    # (``Dim.ceil_div`` would fold identically when divisible, but the static warp
    # tile is always a clean divisor). ``Dim.ceil_div`` keeps the one formula.
    K_o_ext: object = K_axis.extent.ceil_div(k_per_block) if k_masked else K_axis.extent // k_per_block
    K_s = Axis(f"{K_axis.name}_s", params["SPLITK"], source_axis=K_src) if params["SPLITK"] > 1 else None

    # σ-rewrite the matmul body's outer axes (M/N), then expand the K reduce
    # into a K_o > K_i serial tower with K_i stride = atom_k.
    inner_after_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in shape.outer_n.body)
    new_inner, n_replaced = _replace_k_loops(
        inner_after_outer,
        target_names=shape.target_names,
        K_canonical_name=K_axis.name,
        K_s=K_s,
        K_c=None,
        br=1,
        bk=bk,
        K_o_ext=K_o_ext,
        atom_k=atom_k,
    )
    if n_replaced == 0:
        raise RuleSkipped("K reduce not found in body")

    # Masked tiles: wrap the cell body (K tower + Write) in the boundary
    # Cond, INSIDE the AtomTile — `021` lifts the K-pipeline back out so the
    # cooperative slab fill runs unguarded for every thread, leaving
    # ``Cond > Write`` for ``005_lower_atom_tile`` to classify into RegStore
    # guards. ``sigma_outer.apply`` (pure substitution, no simplify) keeps
    # the predicate LHS struct-equal to the σ-rewritten Write index.
    if n_bound is not None:
        new_inner = (Cond(cond=BinaryExpr("<", sigma_outer.apply(Var(N_name)), n_bound), body=Body(new_inner)),)
    if m_bound is not None:
        new_inner = (Cond(cond=BinaryExpr("<", sigma_outer.apply(Var(M_name)), m_bound), body=Body(new_inner)),)

    # ---- Tower: AtomTile(M_a, N_a) > RegisterTile(M_r, N_r) >
    # WarpTile(M_w, N_w) > GridTile(M_b, N_b, K_s?). Layers innermost-first.
    layers: list[tuple[Axis, Role | None]] = []
    layers.append((N_a, Role.ATOM))
    layers.append((M_a, Role.ATOM))
    layers.append((N_r, Role.REGISTER))
    layers.append((M_r, Role.REGISTER))
    layers.append((N_w, Role.WARP))
    layers.append((M_w, Role.WARP))
    layers.append((N_b, Role.BLOCK))
    layers.append((M_b, Role.BLOCK))
    if K_s is not None:
        layers.append((K_s, Role.BLOCK))
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(shape.extra_outer))
    return _wrap_tower(layers, new_inner, atom=atom)


def _mask_reduce_accums(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask each ``Accum``'s input value to the reduce identity past the
    boundary (masked cooperative reduce over a symbolic K). For each
    ``Accum(name, value=V, op)`` insert, just before it, ``Init(V_kid, op)``
    (the op's neutral element) and ``Select(V_km, [V if pred, else V_kid])``,
    then fold ``V_km`` instead of ``V``. The Accum stays a direct child of the
    reduce loop (``is_reduce`` + the cross-thread combine intact); an overhang K
    step folds the identity (no contribution). The Load was already index-
    clamped by the caller so the read stays in-bounds."""
    out: list[Stmt] = []
    for c in body:
        if isinstance(c, Accum):
            ident = f"{c.value}_kid"
            masked = f"{c.value}_km"
            # The cooperative-reduce stats (softmax max/sum, RMSNorm) accumulate in
            # f32; the identity carrier declares f32 so downstream dtype passes
            # (030_stamp_types / 040_demote) see a concrete type, not the Accum's
            # None loop-IR dtype.
            out.append(Init(name=ident, op=c.op, dtype=c.dtype or F32))
            out.append(
                Select(
                    name=masked,
                    branches=(SelectBranch(value=c.value, select=pred), SelectBranch(value=ident, select=Literal(1, "int"))),
                )
            )
            out.append(replace(c, value=masked))
        else:
            out.append(c)
    return tuple(out)


def _replace_k_loops(
    stmts: tuple[Stmt, ...],
    *,
    target_names: frozenset[str],
    K_canonical_name: str,
    K_s: Axis | None,
    K_c: Axis | None,
    K_f: Axis | None = None,
    br: int,
    bk: int,
    fk: int = 1,
    K_o_ext: int,
    atom_k: int = 1,
    k_masked: bool = False,
    k_bound: object = None,
) -> tuple[tuple[Stmt, ...], int]:
    """Replace every ``Loop`` whose axis name is in ``target_names`` with a
    ``Loop(K_o, SERIAL_OUTER, Loop(K_i, STAGE_INNER, σ(body)))`` tower.
    Returns ``(new_stmts, n_replaced)``.

    ``target_names`` is built once in ``_split_kernel_fully``: the set of
    K-iteration axes that should be rewritten (matmul-shape reduces, or for
    cooperative-K both the reduces AND the per-K post-pointwise loops sharing
    the same K extent). ``Loop.is_reduce`` is derived from body Accum
    presence, so K_i inherits the right status automatically.

    ``K_f`` / ``fk`` (the FK multiple-accumulator strip-mine; ``None`` / ``1``
    when unused): when present the innermost reduce body is wrapped in a
    ``RegisterTile((K_f,), reduce=s.is_reduce)`` placed *inside* ``K_i``, so
    ``010_split_register_axes`` replicates the body into FK independent
    accumulators. The σ then carries the ``K_f`` term and ``K_o`` strides by
    ``br·bk·fk`` (see :func:`_build_k_sigma`). Both the reduce loop AND any
    per-K post-pointwise loop sharing this K extent get the K_f tile so the
    factorization covers the full K range — the post-pointwise tile carries
    ``reduce=False`` (replicate-and-drop, no fold; FK-unrolled writes).

    Non-reduce match + SPLITK > 1: wrap the K_o tower in ``Cond(K_s == 0)`` —
    every K_s CTA would otherwise re-execute the post. Reduce-K skips the
    Cond; launch_geometry's atomic-Write rewrite handles cross-CTA SPLITK."""
    out: list[Stmt] = []
    n_replaced = 0
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            K_name = s.axis.name
            K_src = s.axis.source_axis or s.axis
            K_o = Axis(f"{K_canonical_name}_o", K_o_ext, source_axis=K_src)
            K_i = Axis(f"{K_canonical_name}_i", bk, source_axis=K_src)
            sigma_k = _build_k_sigma(K_name, K_s, K_o, K_c, K_i, K_f, K_o_ext, br, bk, fk, atom_k=atom_k)
            # Masked cooperative reduce over a SYMBOLIC K (ceil-div K_o): the final
            # K_o tile overruns the runtime extent. Two shapes, by whether the
            # matched loop reduces:
            #
            #  - REDUCE loop: keep the ``Accum`` a DIRECT child (so ``Loop.is_reduce``
            #    stays True and the cross-thread combine engages — wrapping it in a
            #    ``Cond`` would flip ``is_reduce`` off). Instead clamp the K index for
            #    a safe in-bounds read (``min(decoded, seq_len-1)``, via the σ) and
            #    mask each Accum's input VALUE to the op identity past the bound
            #    (``Select(decoded < seq_len ? value : identity)``) so the overhang
            #    folds a no-op. This is the scalar-reduce twin of the masked-K mma's
            #    clamp-read + zero-fill (``_stage_expand``).
            #  - non-reduce (post-pointwise Write) loop: no Accum to protect, so the
            #    simple ``Cond(decoded < seq_len)`` skips the OOB store outright.
            if k_masked and k_bound is not None:
                decoded_k = sigma_k.apply(Var(K_name))
                pred = BinaryExpr("<", decoded_k, k_bound)
                if s.is_reduce:
                    clamp = TernaryExpr(cond=pred, if_true=decoded_k, if_false=BinaryExpr("-", k_bound, Literal(1, "int")))
                    body_c = tuple(c.rewrite(_identity_rename, Sigma({K_name: clamp})) for c in s.body)
                    new_body = _mask_reduce_accums(body_c, pred)
                else:
                    body_u = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
                    new_body = (Cond(cond=pred, body=Body(body_u)),)
            else:
                new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
            # FK strip-mine: the register tile sits innermost (inside K_i) so the
            # FK cells are the unrolled, ILP-exposed dimension. ``reduce=True``
            # marks it as a reduce-AXIS (K_f) tile — distinguishing it from the
            # FM/FN output tile in ``010_split_register_axes`` so the
            # cross-accumulator fold + role-driven knob stamping fire correctly.
            # The post-pointwise loop's K_f tile carries the same flag but wraps
            # no Accum, so the fold is a no-op there (FK-unrolled writes).
            if K_f is not None:
                new_body = (RegisterTile(axes=(K_f,), body=Body(new_body), reduce=True),)
            tower = _wrap_tower([(K_i, Role.STAGE_INNER), (K_o, Role.SERIAL_OUTER)], new_body)
            if not s.is_reduce and K_s is not None:
                out.append(
                    Cond(
                        cond=BinaryExpr("==", Var(K_s.name), Literal(0, "int")),
                        body=tower,
                        else_body=(),
                    )
                )
            else:
                out.extend(tower)
            n_replaced += 1
            continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, r = _replace_k_loops(
                s.body,
                target_names=target_names,
                K_canonical_name=K_canonical_name,
                K_s=K_s,
                K_c=K_c,
                K_f=K_f,
                br=br,
                bk=bk,
                fk=fk,
                K_o_ext=K_o_ext,
                atom_k=atom_k,
                k_masked=k_masked,
                k_bound=k_bound,
            )
            if r:
                out.append(replace(s, body=inner))
                n_replaced += r
                continue
        if isinstance(s, Cond):
            inner_b, rb = _replace_k_loops(
                s.body,
                target_names=target_names,
                K_canonical_name=K_canonical_name,
                K_s=K_s,
                K_c=K_c,
                K_f=K_f,
                br=br,
                bk=bk,
                fk=fk,
                K_o_ext=K_o_ext,
                atom_k=atom_k,
                k_masked=k_masked,
                k_bound=k_bound,
            )
            inner_e, re_ = _replace_k_loops(
                s.else_body,
                target_names=target_names,
                K_canonical_name=K_canonical_name,
                K_s=K_s,
                K_c=K_c,
                K_f=K_f,
                br=br,
                bk=bk,
                fk=fk,
                K_o_ext=K_o_ext,
                atom_k=atom_k,
                k_masked=k_masked,
                k_bound=k_bound,
            )
            if rb or re_:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                n_replaced += rb + re_
                continue
        out.append(s)
    return tuple(out), n_replaced


def _build_k_sigma(
    K_name: str,
    K_s: Axis | None,
    K_o: Axis,
    K_c: Axis | None,
    K_i: Axis,
    K_f: Axis | None,
    K_o_ext: int | Dim,
    br: int,
    bk: int,
    fk: int = 1,
    atom_k: int = 1,
) -> Sigma:
    """σ for ``K = K_s·(K_o_ext·br·bk·fk·atom_k) + K_o·(br·bk·fk·atom_k) +
    K_f·(br·bk·atom_k) + K_i·br·atom_k + K_c·atom_k`` (in the most general
    form). K_s / K_c / K_f terms collapse when those axes are None
    (SPLITK=1 / BR=1 / FK=1); when K_c is absent, K_i loses its ``·br``
    stride. ``K_f`` (the FK multiple-accumulator strip-mine of the K serial
    loop) inserts a register-tile step of stride ``br·bk·atom_k`` between
    K_o and K_i, and multiplies the K_o / K_s strides by ``fk`` so the
    factorization still covers the full K extent. ``atom_k`` (the mma.sync
    K dimension; default 1 for scalar) extends every per-K-step factor —
    one K_i step covers ``atom_k`` K-elements when MMA is in play, so
    the inner body's K-indexed Loads see ``K_o·bk·atom_k + K_i·atom_k``
    instead of ``K_o·bk + K_i``."""
    inner_expr = Var(K_o.name) * Literal(br * bk * fk * atom_k, "int")
    if K_f is not None:
        inner_expr = inner_expr + Var(K_f.name) * Literal(br * bk * atom_k, "int")
    if K_c is not None:
        if atom_k > 1:
            inner_expr = inner_expr + Var(K_i.name) * Literal(br * atom_k, "int") + Var(K_c.name) * Literal(atom_k, "int")
        else:
            inner_expr = inner_expr + Var(K_i.name) * Literal(br, "int") + Var(K_c.name)
    elif atom_k > 1:
        inner_expr = inner_expr + Var(K_i.name) * Literal(atom_k, "int")
    else:
        inner_expr = inner_expr + Var(K_i.name)
    if K_s is not None:
        # The K_s (SPLITK) stride needs a concrete int K_o_ext. SPLITK > 1 only
        # pairs with a static K (symbolic-K paths force SPLITK=1), so the now-
        # unified ``Dim`` K_o_ext is always a static-folded literal here.
        k_o_ext = K_o_ext.as_static() if isinstance(K_o_ext, Dim) else K_o_ext
        inner_expr = Var(K_s.name) * Literal(k_o_ext * br * bk * fk * atom_k, "int") + inner_expr
    return Sigma({K_name: inner_expr})
