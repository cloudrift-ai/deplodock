"""Op provenance — map fused kernels back to the original frontend (PyTorch) ops.

A node carries one hint, ``prov`` (key :data:`PROV`), a JSON-safe map::

    {origin_id: {"kind": <op-class-name>, "pieces": [piece_id, ...]}}

``origin_id`` is the trace-time node id of an original frontend op (e.g.
``rms_norm_0``); ``pieces`` are the node ids of the primitives that op
decomposed into and that this node embodies. The piece set grows as
decomposition expands an op into many primitives (mint, :func:`mint`) and as
fusion merges primitives from one origin into a single kernel (aggregate,
:func:`union`). A kernel's coverage of an origin is ``len(its pieces)`` over the
union of that origin's pieces across the whole graph (:func:`totals` /
:func:`coverage`) — so ``i/N`` stays correct under CSE and recursive
decomposition instead of freezing ``N`` at the first split.

Pure attribution metadata: it rides on ``Node.hints``, which structural and
autotune-cache keys already skip, so it never affects codegen identity. Stored
with ``list`` (never ``set``) piece collections so it round-trips through
``Graph.to_dict`` / JSON unchanged.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph, Node
    from deplodock.compiler.ir.loop import LoopOp

PROV = "prov"

# Boundary sentinels compute nothing, so no kernel ever "implements" them —
# they never carry provenance and never count toward coverage.
_SKIP = ("InputOp", "ConstantOp")


def is_boundary(op: object) -> bool:
    """Whether ``op`` is a boundary sentinel that never carries provenance."""
    return type(op).__name__ in _SKIP


def get(node: Node) -> dict:
    """Return ``node``'s prov map (an empty dict if unset).

    May return the live hint dict — treat as read-only; mutate via :func:`put`.
    """
    return node.hints.get(PROV) or {}


def put(node: Node, prov: dict) -> None:
    """Store ``prov`` on ``node``, dropping the hint entirely when empty.

    Boundary sentinels never carry provenance (see :data:`_SKIP`): storing onto
    one clears any stale hint instead — e.g. prov copied by ``Graph.splice``'s
    generic hint merge when a fold collapses a compute op into a constant."""
    if prov and not is_boundary(node.op):
        node.hints.set(PROV, prov)
    else:
        node.hints.remove(PROV)


def origins(prov: dict) -> dict[str, str]:
    """``{origin_id: kind}`` for every origin in ``prov``."""
    return {oid: entry["kind"] for oid, entry in prov.items()}


def union(*provs: dict) -> dict:
    """Per-origin merge of several prov maps: union the piece lists, keep the
    kind. The aggregate path (fusion / lifting / optimization folds) uses this
    so a merged kernel accumulates every piece its inputs carried."""
    out: dict = {}
    for prov in provs:
        for oid, entry in prov.items():
            if oid in out:
                out[oid]["pieces"] = sorted(set(out[oid]["pieces"]) | set(entry["pieces"]))
            else:
                out[oid] = {"kind": entry["kind"], "pieces": sorted(set(entry["pieces"]))}
    return out


def mint(origins_kind: dict[str, str], piece_id: str) -> dict:
    """Build a prov where ``piece_id`` is a fresh single piece of each origin in
    ``origins_kind`` (``{origin_id: kind}``). The mint path (decomposition) uses
    this so each new fragment node becomes a distinct piece of the op it
    expands."""
    return {oid: {"kind": kind, "pieces": [piece_id]} for oid, kind in origins_kind.items()}


def seed(graph: Graph) -> None:
    """Idempotently stamp every compute node lacking prov with itself as a
    single-piece origin: ``{node.id: {"kind": <op>, "pieces": [node.id]}}``.

    Run once at pipeline entry. Nodes that already carry prov (a graph reloaded
    mid-pipeline) keep it; boundary sentinels are skipped (see :data:`_SKIP`)."""
    for nid, node in graph.nodes.items():
        if is_boundary(node.op) or node.hints.has(PROV):
            continue
        node.hints.set(PROV, {nid: {"kind": type(node.op).__name__, "pieces": [nid]}})


def totals(graph: Graph) -> dict[str, set[str]]:
    """``{origin_id: all piece ids}`` unioned across every live node — the
    denominator for ``i/N`` coverage."""
    out: dict[str, set[str]] = {}
    for node in graph.nodes.values():
        for oid, entry in get(node).items():
            out.setdefault(oid, set()).update(entry["pieces"])
    return out


def propagate(
    graph: Graph,
    *,
    consumed_prov: dict[str, dict],
    new_compute_ids: list[str],
    new_by_old: dict[str, str],
    output_map: dict[str, str],
    mint_pieces: bool,
) -> None:
    """Thread provenance across one ``Graph.splice``.

    ``consumed_prov`` is ``{consumed_node_id: prov}`` snapshotted before the
    consumed nodes were removed; ``new_compute_ids`` are the graph ids of the
    freshly-added non-boundary fragment nodes; ``new_by_old`` maps each
    redirected consumed node to its fragment output.

    - **mint** (``mint_pieces=True``, decomposition): each new fragment node
      becomes a fresh piece of the consumed origins — one op expanding into
      many distinct primitives.
    - **aggregate** (otherwise — fusion / lifting / optimization folds): each
      fragment output inherits its own consumed node's pieces unioned with the
      ``shared`` pieces of every *dissolved* consumed node (those not in
      ``output_map`` — e.g. a producer inlined into all its consumers), so no
      origin is dropped at a multi-output splice.

    A fragment output that is a boundary sentinel (e.g. a fold collapsing a
    transpose into its parameter ``ConstantOp``) gets its prov scrubbed instead:
    the splice's generic hint merge copied the consumed node's hints — prov
    included — onto it, and a boundary must never carry provenance (its pieces
    would inflate :func:`totals` and make every other kernel of the origin read
    as partial coverage)."""
    for new_out in new_by_old.values():
        node = graph.nodes[new_out]
        if is_boundary(node.op):
            node.hints.remove(PROV)
    if mint_pieces:
        origins_kind = origins(union(*consumed_prov.values())) if consumed_prov else {}
        for nid in new_compute_ids:
            put(graph.nodes[nid], mint(origins_kind, nid))
        return

    shared = union(*[p for cid, p in consumed_prov.items() if cid not in output_map])
    for old_id, new_out in new_by_old.items():
        put(graph.nodes[new_out], union(consumed_prov.get(old_id, {}), shared))
    outputs = set(new_by_old.values())
    for nid in new_compute_ids:
        if nid not in outputs:
            put(graph.nodes[nid], union(shared))


def coverage(node_prov: dict, all_totals: dict[str, set[str]]) -> dict[str, tuple[int, int, bool]]:
    """Per-origin ``(have, total, full)`` for one node given graph-wide
    :func:`totals`. ``full`` means every piece of that origin lives in this
    node — i.e. the kernel realizes the whole original op."""
    out: dict[str, tuple[int, int, bool]] = {}
    for oid, entry in node_prov.items():
        have = len(set(entry["pieces"]))
        total = len(all_totals.get(oid, set(entry["pieces"])))
        out[oid] = (have, total, have >= total)
    return out


# Generic glue ops carry no descriptive name — a kernel made only of these
# falls back to the node-id name. When a meaningful op (rms_norm / linear /
# sdpa / …) is also present, the glue is dropped from the label.
_GENERIC_KINDS = frozenset({"ElementwiseOp", "ReduceOp", "ScanOp", "IndexMapOp", "GatherOp", "ScatterOp"})

# Frontend layout/plumbing ops label a kernel only when no strong op is present:
# an attention kernel that also absorbs RoPE's cat/slice/unsqueeze plumbing
# stays ``k_sdpa_…``, while a standalone copy kernel (e.g. a cat feeding a graph
# output) still reads ``k_cat_…`` instead of the node-id fallback.
_WEAK_KINDS = frozenset({"TransposeOp", "ReshapeOp", "UnsqueezeOp", "CatOp", "SliceOp"})


def _humanize_kind(kind: str) -> str:
    """``RmsNormOp`` → ``rms_norm``, ``SdpaOp`` → ``sdpa`` (CamelCase→snake, drop ``Op``)."""
    stem = kind[:-2] if kind.endswith("Op") else kind
    return re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()


def _dedup_tokens(name: str) -> str:
    """Drop consecutive duplicate ``_``-separated tokens.

    ``softmax_softmax_max`` → ``softmax_max``; ``rms_rms_norm`` → ``rms_norm``.
    Preserves order; only collapses adjacent duplicates so structurally
    distinct repeats (``add_mul_add``) survive.
    """
    out: list[str] = []
    for tok in name.split("_"):
        if not tok or (out and out[-1] == tok):
            continue
        out.append(tok)
    return "_".join(out) if out else name


def name_for(loop: LoopOp, base_name: str, node_prov: dict, all_totals: dict[str, set[str]]) -> str:
    """Name the kernel after the original ops it implements (op provenance).

    A kernel that fully realizes exactly one meaningful op gets ``k_<op>_<h>``
    (e.g. ``k_rms_norm_3f2a1b``); a partial one keeps the ``_<reduce|pointwise>``
    qualifier so the reduce half is told apart from the pointwise tail. Multiple
    meaningful ops join dominant-first — sorted by descending piece count (the op
    the kernel mostly implements leads, e.g. ``k_sdpa_linear_...``), ties broken
    lexically — so the label is independent of fusion merge order. Layout ops
    (:data:`_WEAK_KINDS`) label the kernel only when no strong op is present;
    with no provenance (or only glue ops) it falls back to the node-id name.

    ``<h>`` is a short structural-body hash: prov labels are *not* unique (two
    rms_norms, or SDPA's two distinct reduce kernels, share a label), but the
    backend dispatches kernels by name and the launch dict holds one source per
    name. The hash makes structurally-identical kernels share a name (so they
    dedup to one compilation) and distinct kernels differ (so a duplicate label
    never makes one launch reuse another's code). The node-id fallback is
    already unique, so it needs no hash.

    Local import for ``Accum`` avoids a top-of-module cycle through
    ``ir.loop`` (which imports ``provenance`` for graph utilities)."""
    from deplodock.compiler.ir.stmt import Accum

    suffix = "reduce" if any(isinstance(s, Accum) for s in loop) else "pointwise"
    strong = [oid for oid, e in node_prov.items() if e["kind"] not in _GENERIC_KINDS | _WEAK_KINDS]
    meaningful = strong or [oid for oid, e in node_prov.items() if e["kind"] in _WEAK_KINDS]
    if not meaningful:
        return f"k_{_dedup_tokens(base_name)}_{suffix}"

    meaningful.sort(key=lambda oid: (-len(set(node_prov[oid]["pieces"])), _humanize_kind(node_prov[oid]["kind"])))
    labels: list[str] = []
    for oid in meaningful:
        lbl = _humanize_kind(node_prov[oid]["kind"])
        if lbl not in labels:
            labels.append(lbl)
    joined = _dedup_tokens("_".join(labels))
    # ``structural_key`` is a pretty-printed body string; hash it to a short
    # alphanumeric token (valid in a C identifier; identical bodies → same token).
    h = hashlib.sha1(loop.body.structural_key().encode()).hexdigest()[:6]
    cov = coverage(node_prov, all_totals)
    if len(meaningful) == 1 and cov[meaningful[0]][2]:  # single op, fully covered
        return f"k_{joined}_{h}"
    return f"k_{joined}_{suffix}_{h}"
