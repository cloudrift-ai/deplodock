"""``ShapeKey`` — the cheap arithmetic structural identity of a tiled op.

A kernel's full structural signature is the ``S_*`` histogram the
``992_stamp_structural_features`` pass stamps (op counts, dtypes, loop depths,
extents). For grouping, the cold-start ``AnalyticPrior`` ranking, and matching a
golden config to a kernel, only the *extents* matter — and those are derivable
arithmetically from a matmul's ``(M, N, K)`` with no compile. ``ShapeKey`` is that
arithmetic handle: ``(free_prod, reduce_max, is_warp)``, the same triple the prior
diagnostics and the per-kernel golden A/B already match on.

It deliberately carries only the extent keys. A *trained* ``CatBoostPrior`` regresses
on the full ``S_*`` histogram, so the full set is derived (by compiling the snippet)
and cached on the :class:`~deplodock.compiler.pipeline.search.data.sample.Sample`,
not here — see that module's ``compile_s_feats`` path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeKey:
    """Arithmetic extent identity of a matmul-shaped op.

    ``free_prod`` is the product of the **static** output free dims (``M*N``; a
    symbolic axis is excluded, mirroring the ``992`` stamp — see :meth:`from_matmul`),
    ``reduce_max`` the reduce extent (``K``), ``is_warp`` whether it lowers on the
    tensor-core warp tier (any non-fp32 matmul), ``is_dyn`` whether an axis is
    symbolic — the split that keeps a dynamic golden and its static twin apart,
    exactly as ``is_warp`` keeps the fp32/fp16 twins apart. Hashable, so it keys
    ``Dataset`` groupings."""

    free_prod: int
    reduce_max: int
    is_warp: bool
    is_dyn: bool = False

    @classmethod
    def from_matmul(cls, M: int, N: int, K: int, dtype: str, *, dynamic: bool = False) -> ShapeKey:
        """The shape of an ``(M, K) @ (K, N)`` matmul. fp32 lowers on the scalar
        thread tier; fp16 / bf16 on the warp (MMA) tier. ``dynamic`` marks the M
        axis symbolic (the only symbolic-axis golden form today): the key then
        MIRRORS what ``992_stamp_structural_features`` puts on the op — symbolic
        axes are **excluded** from the extent products (``free_prod = N``, not the
        hint-sized ``M*N``) and flagged via ``S_ext_n_symbolic_axis`` — because
        the stamped histogram is the only identity the op side has (it doesn't
        know the hint), so a hint-sized golden key would never join it."""
        return cls(free_prod=N if dynamic else M * N, reduce_max=K, is_warp=dtype != "fp32", is_dyn=dynamic)

    @classmethod
    def from_s_features(cls, s: dict) -> ShapeKey:
        """The key of a stamped ``S_*`` histogram — an op-group signature from
        ``Dataset.group_by_op``, a prior-reservoir row's knobs, or a ``CudaOp.knobs``
        dict carrying the stamped features. The op-side twin of :meth:`from_matmul`:
        every golden ↔ measured-data join must build BOTH sides through these two
        constructors, so a new key dimension (e.g. the planned symbolic-axis flag)
        lands in one place instead of per join site.

        ``is_warp`` derives from the operand-dtype multiset (``S_dtype_f32``), NOT
        ``S_n_mma``: the stamp pass runs at fusion end, before the tile tier emits
        ``Mma`` stmts, so ``S_n_mma`` is 0.0 on every stamped row — keying on it
        merged the fp32/fp16 twins (and silently dropped fp16 goldens from the
        diagnostics joins), the bug class this single constructor exists to
        prevent."""
        return cls(
            free_prod=int(s.get("S_ext_free_prod", 0)),
            reduce_max=int(s.get("S_ext_reduce_max", 0)),
            is_warp=not s.get("S_dtype_f32", 0),
            is_dyn=s.get("S_ext_n_symbolic_axis", 0) > 0,
        )

    def s_features_arith(self) -> dict[str, float]:
        """The extent ``S_*`` features derivable without compiling — the exact set
        the cold ``AnalyticPrior`` reads (``analytic._analytic_scorer`` builds the
        same dict). For a matmul the reduce axis is a single contraction, so
        ``S_ext_reduce_prod == S_ext_reduce_max == K``. A dynamic key adds the
        ``S_ext_n_symbolic_axis`` flag (and its ``free_prod`` already excludes the
        symbolic axis), mirroring the stamped histogram so the arith fallback
        featurizes like the rows a DB-trained prior saw."""
        out = {
            "S_ext_free_prod": float(self.free_prod),
            "S_ext_reduce_prod": float(self.reduce_max),
            "S_ext_reduce_max": float(self.reduce_max),
        }
        if self.is_dyn:
            out["S_ext_n_symbolic_axis"] = 1.0
        return out
