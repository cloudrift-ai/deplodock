"""Per-compilation context passed through the rewrite pipeline.

A ``Context`` carries target-derived state (``compute_capability``) and
any future per-compilation knobs (tuning params, flags) that rules need
to read. Built once per ``run_pipeline`` invocation and threaded through
the engine; rules that take a ``ctx`` parameter receive it via the same
dispatcher binding that supplies ``graph`` / ``match`` / ``root``.

This replaces process-global access patterns like the ``_OVERRIDE``
module state in :mod:`deplodock.compiler.target`: explicit-by-design
keeps tests simple (no monkey-patching), keeps compilations independent
(no cross-thread state), and makes rule dependencies discoverable from
the signature alone.
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-block static-smem cap. Hard hardware limit since Maxwell (sm_50);
# anything declared as ``__shared__`` at compile time must fit.
# Universal across every arch we target.
STATIC_SMEM_CAP = 48 * 1024

# Per-block dynamic-smem opt-in cap by compute capability. NVIDIA assigns
# different sm_XX numbers to datacenter vs consumer SKUs within the same
# arch family (sm_80 A100 vs sm_86 RTX 30xx; sm_90 H100 vs sm_120 RTX
# 50xx), so this is keyed on cc, not "arch generation". Values from
# ``cudaDevAttrMaxSharedMemoryPerBlockOptin``.
_MAX_DYNAMIC_SMEM_BY_CC: dict[tuple[int, int], int] = {
    (7, 5): 64 * 1024,
    (8, 0): 163 * 1024,
    (8, 6): 99 * 1024,
    (8, 9): 99 * 1024,
    (9, 0): 227 * 1024,
    (10, 0): 227 * 1024,
    (12, 0): 99 * 1024,
}


def _max_dynamic_smem_for(cc: tuple[int, int]) -> int:
    """Per-block dynamic-smem opt-in cap for ``cc``. Falls back to
    ``STATIC_SMEM_CAP`` for unknown / no-hardware caps so callers that
    use it as a budget naturally degrade to the static-only ceiling."""
    return _MAX_DYNAMIC_SMEM_BY_CC.get(cc, STATIC_SMEM_CAP)


@dataclass(frozen=True)
class Context:
    """Per-compilation state visible to rules.

    Build with :meth:`from_target` (CLI / autotuner) or :meth:`probe`
    (live device). Frozen so a ``Context`` can't be mutated mid-pipeline
    — replace via :func:`dataclasses.replace` if a phase needs to alter
    a field.
    """

    compute_capability: tuple[int, int]
    static_smem_cap: int = STATIC_SMEM_CAP
    max_dynamic_smem: int = STATIC_SMEM_CAP  # overridden by from_target/probe
    # Hardware-universal per-CTA thread cap (CUDA compute capability ≥ 2.0).
    # Used by ``KernelOp.validate`` to filter autotune variants whose launch
    # geometry would be rejected by the driver before the kernel ever runs.
    max_threads_per_cta: int = 1024
    # Hardware warp width — 32 on every NVIDIA arch we target. Carried on
    # ``Context`` so cooperative-reduce gating (``000_partition_planner``)
    # and warp-shuffle dispatch (``001_materialize_tile``) read a single
    # source of truth instead of redefining the constant module-locally.
    warp_size: int = 32
    # Identifies which backend's perf rows this compile should consult
    # for DB-driven decisions (``GreedySearch`` looks up ``perf`` by
    # ``(context_key, op_key, backend)``). Defaults to ``"cuda"`` — the
    # canonical autotune target. ``run_autotune`` replaces this when a
    # live :class:`Backend` is supplied.
    backend_name: str = "cuda"

    @classmethod
    def from_target(cls, cap: tuple[int, int]) -> Context:
        return cls(compute_capability=cap, max_dynamic_smem=_max_dynamic_smem_for(cap))

    def structural_key(self) -> str:
        """Implements :class:`deplodock.compiler.structural.Structural`.

        Folds in only codegen-affecting fields. ``compute_capability``
        gates hardware-feature passes (TMA, cp.async, dynamic smem cap);
        anything derived from it (``max_dynamic_smem``) is implied. As
        non-derived knobs land (forced TMA on/off, splitk overrides),
        extend this method explicitly — keep ambient I/O fields out so
        the autotuning cache survives debug-flag flips.
        """
        from deplodock.compiler.structural import digest  # noqa: PLC0415

        return digest("Context", self.compute_capability)

    @classmethod
    def probe(cls) -> Context:
        """Build by probing the live CUDA device. Falls back to (0, 0) if
        cupy is unavailable — callers treat that as "no hardware feature
        support" (rules gating on capability self-skip via ``RuleSkipped``)."""
        from deplodock.compiler.target import compute_capability  # noqa: PLC0415

        cap = compute_capability()
        return cls(compute_capability=cap, max_dynamic_smem=_max_dynamic_smem_for(cap))
