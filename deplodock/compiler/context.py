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


@dataclass(frozen=True)
class Context:
    """Per-compilation state visible to rules.

    Build with :meth:`from_target` (CLI / autotuner) or :meth:`probe`
    (live device). Frozen so a ``Context`` can't be mutated mid-pipeline
    — replace via :func:`dataclasses.replace` if a phase needs to alter
    a field.
    """

    compute_capability: tuple[int, int]

    @classmethod
    def from_target(cls, cap: tuple[int, int]) -> Context:
        return cls(compute_capability=cap)

    @classmethod
    def probe(cls) -> Context:
        """Build by probing the live CUDA device. Falls back to (0, 0) if
        cupy is unavailable — callers treat that as "no hardware feature
        support" (rules gating on capability self-skip via ``RuleSkipped``)."""
        from deplodock.compiler.target import compute_capability  # noqa: PLC0415

        return cls(compute_capability=compute_capability())
