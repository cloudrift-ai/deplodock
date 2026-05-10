"""Structural identity protocol — one convention for every type whose
instances we want to compare or dedup by their structure rather than by
Python identity / field equality.

Implementers return a hex sha256 digest folded over the bits of state
that affect downstream behavior (codegen output, dataflow semantics,
compilation result). Bits that are name-only / advisory / ambient I/O
are deliberately excluded — see each implementer's docstring for the
exact include/exclude list.

Implementers today:

- :class:`deplodock.compiler.graph.Graph` — Merkle digest over op kinds,
  body structure, output shapes/dtypes, input wiring; excludes node ids,
  Tensor names, and Hints.
- :class:`deplodock.compiler.ir.stmt.body.Body` — canonicalized body
  rendering with SSA / axis / commutative-arg / external-buffer names
  normalized away.
- :class:`deplodock.compiler.context.Context` — codegen-affecting
  compilation knobs (compute capability today; tuning overrides as they
  land). Excludes ambient I/O fields (dump dirs, verbosity).

The cache layer in the autotuning loop keys candidates by these digests,
so adding a field to an implementer is an explicit decision: include it
in the digest only if it changes generated code or dataflow semantics.

``digest(*parts)`` is the canonical fold helper — pass a mix of strings,
ints, bytes, and pre-computed child digests; the helper canonicalizes
via ``repr`` and returns a hex sha256 string. Composite implementers use
it to fold child digests with their own discriminating fields.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Structural(Protocol):
    """Anything whose structural identity is comparable as a hex digest.

    Two instances that should be treated as equivalent for caching /
    dedup purposes return the same string from :meth:`structural_key`;
    two instances that differ in any codegen- or dataflow-relevant way
    return different strings.
    """

    def structural_key(self) -> str: ...


def digest(*parts: object) -> str:
    """Fold ``parts`` into a hex sha256 digest. Each part is rendered
    via ``repr`` and joined; pass child digests (already strings),
    primitive fields (ints, tuples, names), or any ``repr``-stable
    object. Order is significant — callers control canonicalization."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()
