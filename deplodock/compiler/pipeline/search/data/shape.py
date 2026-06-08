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

    ``free_prod`` is the output free-dim product (``M*N``), ``reduce_max`` the
    reduce extent (``K``), ``is_warp`` whether it lowers on the tensor-core warp
    tier (any non-fp32 matmul). Hashable, so it keys ``Dataset`` groupings."""

    free_prod: int
    reduce_max: int
    is_warp: bool

    @classmethod
    def from_matmul(cls, M: int, N: int, K: int, dtype: str) -> ShapeKey:
        """The shape of an ``(M, K) @ (K, N)`` matmul. fp32 lowers on the scalar
        thread tier; fp16 / bf16 on the warp (MMA) tier."""
        return cls(free_prod=M * N, reduce_max=K, is_warp=dtype != "fp32")

    def s_features_arith(self) -> dict[str, float]:
        """The extent ``S_*`` features derivable without compiling — the exact set
        the cold ``AnalyticPrior`` reads (``analytic._analytic_scorer`` builds the
        same dict). For a matmul the reduce axis is a single contraction, so
        ``S_ext_reduce_prod == S_ext_reduce_max == K``."""
        return {
            "S_ext_free_prod": float(self.free_prod),
            "S_ext_reduce_prod": float(self.reduce_max),
            "S_ext_reduce_max": float(self.reduce_max),
        }
