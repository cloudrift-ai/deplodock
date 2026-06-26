"""Diagnostics join keys — every golden ↔ measured-data join goes through
``ShapeKey`` (``from_matmul`` on the golden side, ``from_s_features`` on the op
side) with the ``_matmul_sig`` histogram gate.

The load-bearing regression here is the dead ``S_n_mma`` marker: the stamp pass
runs at fusion end, before the tile tier emits ``Mma`` stmts, so ``S_n_mma`` is
0.0 on every stamped row. Gating on it made ``_golden_coverage`` permanently
empty, mislabeled every matmul in ``reachability``, and silently dropped every
fp16 golden from ``golden_prior_eval``'s rank join (the hidden fp16 lockout).
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search.golden import goldens_by_name
from deplodock.compiler.pipeline.search.prior import diagnostics


def _sig(free_prod, reduce_max, *, fp32=True, matmul=True):
    """An op-group signature (sorted ``S_*`` items) as ``Dataset.group_by_op``
    keys it — a matmul histogram (product → reduce-add, 2 distinct inputs) or a
    reduce-shaped one. ``S_n_mma`` is stamped 0.0, as on every real row."""
    d = {
        "S_ext_free_prod": float(free_prod),
        "S_ext_free_max": float(free_prod),
        "S_ext_reduce_max": float(reduce_max),
        "S_n_mma": 0.0,
        "S_reduce_add": 1.0,
        "S_n_distinct_input": 2.0 if matmul else 1.0,
        ("S_dtype_f32" if fp32 else "S_dtype_f16"): 2.0,
    }
    if matmul:
        d["S_pw_multiply"] = 1.0
    return tuple(sorted(d.items()))


def test_op_label_recognizes_stamped_matmuls():
    """Matmuls label as ``matmul`` from the histogram (S_n_mma is always 0.0);
    a plain reduce keeps its label."""
    assert diagnostics._op_label(_sig(512 * 512, 512)).startswith("matmul")
    assert diagnostics._op_label(_sig(512, 4096, matmul=False)).startswith("reduce")


def test_golden_coverage_counts_dtype_twins_separately():
    """Coverage joins on ShapeKey: an fp32 group at square.512's extents covers the
    fp32 golden only — its ``.fp16`` twin needs its own group; a reduce-shaped group
    at the same extents counts nothing. (The old ``S_n_mma > 0`` gate made coverage
    permanently 0/N.)"""
    fp = 512 * 512
    base = diagnostics._golden_coverage({})[0]
    assert base == 0
    only_fp32 = diagnostics._golden_coverage({_sig(fp, 512): []})[0]
    assert only_fp32 == 1  # square.512, not square.512.fp16
    both = diagnostics._golden_coverage({_sig(fp, 512): [], _sig(fp, 512, fp32=False): []})[0]
    assert both == 2
    collider = diagnostics._golden_coverage({_sig(fp, 512, matmul=False): []})[0]
    assert collider == 0  # non-matmul group at matmul extents doesn't count


def test_golden_coverage_splits_dynamic_twin(monkeypatch):
    """A ``.dynM`` golden is covered only by a symbolic-marked op group
    (``S_ext_n_symbolic_axis > 0``, ``free_prod`` excluding the symbolic axis —
    what the 992 stamp emits for a masked-tile kernel); its static twin's group
    does not satisfy it, and vice versa."""
    from deplodock.compiler.pipeline.search import golden as gmod

    dyn_g = gmod.MatmulGoldenConfig(
        name="square.512.dynM",
        M=512,
        N=512,
        K=512,
        knobs={"BM": 8},
        deplodock_us=10.0,
        cublas_us=11.0,
        dynamic={"seq_len": {"input": "x0", "axis": 0}},
    )
    monkeypatch.setattr(gmod, "GOLDEN_CONFIGS", [dyn_g])

    static_group = _sig(512 * 512, 512)  # the static twin's stamped extents
    assert diagnostics._golden_coverage({static_group: []}) == (0, 1)

    sym = dict(_sig(512, 512))  # free_prod = N only; symbolic axis flagged
    sym["S_ext_n_symbolic_axis"] = 1.0
    assert diagnostics._golden_coverage({tuple(sorted(sym.items())): []}) == (1, 1)


class _FakePrior:
    """Constant-score prior with a hand-built reservoir (``golden_prior_eval``
    only needs ``_dataset`` for the op-group index and ``mean_score``)."""

    def __init__(self, rows):
        self._dataset = rows

    def mean_score(self, _feats):
        return 0.0


def test_golden_prior_eval_joins_fp16_goldens():
    """An fp16-tuned op group must rank its ``.fp16`` golden (under the old
    ``S_n_mma``-keyed join the op side was always ``False``, so fp16 goldens
    silently dropped while fp32 goldens could steal the fp16 group's histogram)."""
    g16 = goldens_by_name("square.512.fp16")[0]
    rows = [({**dict(_sig(g16.M * g16.N, g16.K, fp32=False)), "WM": 4}, 5.0)]
    out = diagnostics.golden_prior_eval(_FakePrior(rows), kernel_filter="square.512")
    assert "square.512.fp16" in out and "rank" in out
    # The fp32 square.512 has no tuned group here → a per-shape SKIPPED line, not silence.
    skip_lines = [line for line in out.splitlines() if "SKIPPED" in line]
    assert any(line.strip().startswith("square.512 ") and "no tuned rows" in line for line in skip_lines)


def test_golden_prior_eval_warns_per_unjoinable_shape():
    """Shapes with no tuned rows must each print a SKIPPED line — the silent drop
    hid the fp16 lockout in the 2026-06-12 sweep."""
    out = diagnostics.golden_prior_eval(_FakePrior([]), kernel_filter="square.512")
    skip_lines = [line for line in out.splitlines() if "SKIPPED" in line and "no tuned rows" in line]
    # square.512 and its .fp16 twin (one line per recorded config name, deduped by name set)
    assert {line.split()[0] for line in skip_lines} >= {"square.512", "square.512.fp16"}


# ---------------------------------------------------------------------------
# node-store fork sibling-ranking
# ---------------------------------------------------------------------------


class _BMPrior:
    """A prior whose predicted latency is ``sign * features['BM']`` — so ``sign=+1``
    predicts smaller BM as faster, ``sign=-1`` reverses the order."""

    fitted = True

    def __init__(self, sign: float = 1.0):
        self._sign = sign

    def mean_score(self, feats) -> float:
        return self._sign * float(feats.get("BM", 0.0))


def _child(node_key, bm, value_us, *, parent="P"):
    from deplodock.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    return NodeRow(node_key, parent, "ctx", "mm", {"BM": bm}, value_us, 2)


def _fork_nodes():
    """A root P with three children whose value-of-position falls as BM falls."""
    from deplodock.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    return [NodeRow("P", None, "ctx", "mm", {}, 1.0, 1), _child("c8", 8, 1.0), _child("c32", 32, 2.0), _child("c64", 64, 3.0)]


def test_node_sibling_ranking_recovers_best_and_order():
    """The prior orders the fork's children by value-of-position: top-1 hit (its
    predicted-best child IS the true-best) and Spearman +1."""
    n_forks, top1, rho, n_children = diagnostics.node_sibling_ranking(_BMPrior(), _fork_nodes())
    assert (n_forks, n_children) == (1, 3)
    assert top1 == 1.0 and rho == 1.0


def test_node_sibling_ranking_penalizes_reversed_order():
    """A prior that ranks the children backwards misses top-1 and scores Spearman -1."""
    _, top1, rho, _ = diagnostics.node_sibling_ranking(_BMPrior(sign=-1.0), _fork_nodes())
    assert top1 == 0.0  # predicted-best (BM=64) is the slowest child
    assert rho == -1.0


def test_node_sibling_ranking_ignores_singletons_and_top_forks():
    """A top fork (parent None) and a single-child parent are not multi-child forks,
    so there is nothing to rank."""
    from deplodock.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    nodes = [NodeRow("P", None, "ctx", "mm", {}, 1.0, 1), _child("only", 8, 1.0)]
    assert diagnostics.node_sibling_ranking(_BMPrior(), nodes) is None


def test_node_report_combines_fork_and_leaf_sections():
    """``node_report`` renders both the fork sibling-ranking and the leaf reachability
    over the node store."""
    from deplodock.compiler.pipeline.search.db import NodeRow  # noqa: PLC0415

    s = {"S_reduce_add": 1.0, "S_pw_multiply": 1.0, "S_n_distinct_input": 2.0, "S_ext_free_max": 512.0}
    nodes = [
        NodeRow("P", None, "ctx", "mm", s, 1.0, 1),
        NodeRow("c8", "P", "ctx", "mm", {**s, "BM": 8}, 1.0, 2),
        NodeRow("c64", "P", "ctx", "mm", {**s, "BM": 64}, 3.0, 2),
    ]
    text = diagnostics.node_report(_BMPrior(), nodes)
    assert "node store: 3 nodes" in text
    assert "fork sibling-ranking" in text
    assert "leaf reachability over node store" in text
