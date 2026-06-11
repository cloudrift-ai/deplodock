"""``Sample`` — one measured-or-recorded ``(config, latency, identity)`` row,
the common currency over all three measurement-data sources.

A golden config, a tune-DB ``perf`` row, and a learned-prior reservoir row are all
the same thing once normalized: a tunable-knob dict, a measured latency, a
structural identity, and (for golden) a reference latency. ``Sample`` is that
normal form. The split into ``knobs`` (tunable) / ``context`` (``H_*``) /
``s_features`` (``S_*``) is by key prefix and therefore lossless — :meth:`all_knobs`
re-merges them to the exact original dict, and :meth:`features` runs the single
featurizer (:func:`knob.knob_features`) on that merge, so a ``Sample`` reproduces
the feature vector each source built inline today.

Featurization fidelity (the load-bearing invariant): a trained ``CatBoostPrior``
regresses on the full ``S_*`` histogram stamped by ``992_stamp_structural_features``.
DB / prior rows carry that histogram inline already; golden rows only know
``(M,N,K)``, so the full histogram is derived by compiling the snippet
(:func:`compiled_s_features`, cached) and passed as ``compile_s_feats=True`` —
*only* when a learned prior is the consumer. The cheap arithmetic
:meth:`ShapeKey.s_features_arith` suffices for the cold ``AnalyticPrior`` (it reads
only ``D_*`` derived from the extents) and for grouping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.knob import CTX_PREFIX, STRUCT_PREFIX
from deplodock.compiler.pipeline.search.data.shape import ShapeKey

# The C identifier of a CUDA kernel, parsed from ``cuda_op.pretty`` — the grouping
# key for the per-knob regret analysis. Kept here so the DB-row adapter and the
# regret grouping share one source. Anchored on the ``__global__`` entry point
# (``__launch_bounds__`` sits between it and ``void``) — MMA/TMA kernel sources
# open with ``__device__`` helper preludes (``dpl_ldmatrix_x4``, ``mbarrier_init``),
# so a bare first-``void`` match would name the helper, collapsing distinct kernels
# into one leaderboard bucket and hiding them from ``--kernel`` filters.
KERNEL_NAME_RE = re.compile(r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\(")


def _split_by_prefix(knobs: dict) -> tuple[dict, dict, dict]:
    """Split a stamped knob dict into ``(tunable, context H_*, structural S_*)`` by
    key prefix. Disjoint prefixes → re-merging is lossless."""
    ctx = {k: v for k, v in knobs.items() if k.startswith(CTX_PREFIX)}
    s = {k: v for k, v in knobs.items() if k.startswith(STRUCT_PREFIX)}
    tunable = {k: v for k, v in knobs.items() if not k.startswith((CTX_PREFIX, STRUCT_PREFIX))}
    return tunable, ctx, s


@lru_cache(maxsize=256)
def compiled_s_features(M: int, N: int, K: int, dtype: str, compute_cap: tuple[int, int]) -> tuple[tuple[str, float], ...]:
    """The full ``S_*`` structural histogram for a matmul shape — by compiling its
    snippet to the loop dialect (where ``992_stamp_structural_features`` runs) and
    scraping the stamped ``S_*`` knobs off the nodes. This is what a trained
    ``CatBoostPrior`` was fit on for that shape; the cheap arithmetic extents are a
    subset of it. Cached (the compile is ~seconds) and returned as sorted items so
    the result is hashable / frozen-friendly. Heavy imports are deferred to here so
    importing this module stays torch-free."""
    from deplodock.commands.trace import graph_from_code  # noqa: PLC0415
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.golden import matmul_snippet  # noqa: PLC0415

    graph, _, _ = graph_from_code(matmul_snippet(M, N, K, dtype))
    compiled = Pipeline.build(LOOP_PASSES).run(graph)  # loop dialect — S_* stamped, no codegen
    s_feats: dict[str, float] = {}
    for n in compiled.nodes.values():
        s_feats.update({k: v for k, v in (getattr(n.op, "knobs", {}) or {}).items() if k.startswith(STRUCT_PREFIX)})
    return tuple(sorted(s_feats.items()))


@dataclass(frozen=True)
class Sample:
    """One ``(config, latency, identity)`` row, normalized across sources.

    ``knobs`` holds *only* tunable knobs (``S_*`` / ``H_*`` live in ``s_full`` /
    ``context``); ``shape`` is the arithmetic identity; ``ref_us`` is the cuBLAS /
    torch reference (golden only, ``None`` elsewhere); ``pretty`` / ``name`` carry
    the kernel C identifier for DB rows; ``snippet`` reproduces a golden's torch
    expression. ``source`` ∈ ``{"golden", "db", "prior"}`` marks provenance for the
    orthogonality fail-fast (``dataset_args.require_source``)."""

    knobs: dict
    latency_us: float
    shape: ShapeKey | None = None
    name: str | None = None
    dtype: str | None = None
    ref_us: float | None = None
    context: dict = field(default_factory=dict)
    pretty: str | None = None
    snippet: str | None = None
    source: str = "db"
    s_full: dict | None = None  # full compiled S_* histogram when known (db/prior rows, or golden compile_s_feats)
    error: str | None = None  # bench_fail failure text (db rows only; None on ok rows)

    def s_features(self) -> dict[str, float]:
        """The ``S_*`` features this sample featurizes on: the full stamped histogram
        when known, else the cheap arithmetic extents, else nothing."""
        if self.s_full is not None:
            return self.s_full
        return self.shape.s_features_arith() if self.shape is not None else {}

    def all_knobs(self) -> dict:
        """The original stamped dict — ``context ∪ s_features ∪ knobs``. For a DB
        row this re-merges to exactly the recorded ``perf.knobs``; the per-knob
        regret analysis iterates this so its output is unchanged."""
        return {**self.context, **self.s_features(), **self.knobs}

    def features(self) -> dict[str, float]:
        """The flat numeric feature vector the priors regress on — the single
        featurizer over the merged dict. Merge order ``context, s_*, knobs`` matches
        the inline construction the eval / prior code used (knobs win on collision,
        though the prefixes are disjoint)."""
        return knob.knob_features(self.all_knobs())

    @classmethod
    def from_golden(cls, cfg, *, compile_s_feats: bool = False) -> Sample:
        """A golden ``MatmulGoldenConfig`` as a ``Sample``. ``compile_s_feats``
        derives the full ``S_*`` histogram (for the learned-prior featurization);
        leave it off for the cold-analytic / grouping / bench paths."""
        from deplodock.compiler.context import Context  # noqa: PLC0415

        tunable, _ctx, _s = _split_by_prefix(cfg.knobs)
        s_full = dict(compiled_s_features(cfg.M, cfg.N, cfg.K, cfg.dtype, cfg.compute_cap)) if compile_s_feats else None
        return cls(
            knobs=tunable,
            latency_us=cfg.deplodock_us,
            shape=ShapeKey.from_matmul(cfg.M, cfg.N, cfg.K, cfg.dtype),
            name=cfg.name,
            dtype=cfg.dtype,
            ref_us=cfg.cublas_us,
            context=Context.from_target(cfg.compute_cap).features(),
            snippet=cfg.snippet(),
            source="golden",
            s_full=s_full,
        )

    @classmethod
    def from_perf_sample(cls, ps) -> Sample:
        """A tune-DB ``perf ⋈ cuda_op`` row (:class:`db.PerfSample`) as a ``Sample``.
        Splits the recorded knob dict by prefix; the kernel C identifier (for
        per-knob regret grouping) is parsed from ``cuda_op.pretty``."""
        tunable, ctx, s = _split_by_prefix(ps.knobs)
        m = KERNEL_NAME_RE.search(ps.pretty or "")
        return cls(
            knobs=tunable,
            latency_us=ps.latency_us,
            name=m.group(1) if m else None,
            context=ctx,
            pretty=ps.pretty,
            source="db",
            s_full=s,
            error=ps.error,
        )

    @classmethod
    def from_prior_row(cls, knobs: dict, latency_us: float) -> Sample:
        """A learned-prior reservoir row ``(stamped_knobs, latency)`` as a ``Sample``.
        The reservoir dicts already carry ``S_*`` / ``H_*`` inline (stamped by the
        live pipeline), so the split + re-merge is lossless for grouping / scoring."""
        tunable, ctx, s = _split_by_prefix(knobs)
        return cls(knobs=tunable, latency_us=latency_us, context=ctx, source="prior", s_full=s)
