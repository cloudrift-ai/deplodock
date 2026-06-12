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
    assert "square.512.fp16" in out
    # The fp32 square.512 has no tuned group here → no rank line for it.
    assert not any(line.strip().startswith("square.512 ") for line in out.splitlines())
