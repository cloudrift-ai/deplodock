"""``Rule`` — one loaded rewrite rule: pattern + rewrite function +
binding metadata for the engine's signature dispatcher.

Lives in its own module so :mod:`pipeline.pattern` (which holds
``Pipeline``, the engine-wide rule layout) and :mod:`pipeline.engine`
(which loads rules off disk and runs their rewrites) share one
definition without cycling through each other. Rules are loaded by
``engine._load_rule`` from the ``passes/`` directory tree; tests can
construct a one-rule ``Pipeline`` via :meth:`Pipeline.from_pattern`
which uses a no-rewrite ``Rule`` to drive pure pattern matching.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.base import Op
    from deplodock.compiler.pipeline.pattern import Pattern


@dataclass(frozen=True)
class Rule:
    """One rewrite rule loaded from a ``passes/<dir>/NNN_<name>.py``
    module.

    * ``name`` — the file stem (engine display + dump filenames).
    * ``pattern`` — the chain-match pattern the rule fires on.
    * ``rewrite`` — the rule's ``rewrite`` function. ``None`` for the
      no-rewrite stubs :meth:`Pipeline.from_pattern` builds for
      pattern-matching-only callers.
    * ``param_names`` — captured at load time so the dispatcher can
      bind each rewrite param via signature inspection. The binding
      rules (kept here so docstring + dataclass live together):

      - ``graph`` — the current ``Graph``
      - ``match`` — the full ``Match`` (escape hatch)
      - ``root`` — ``graph.nodes[match.root_node_id]``
      - ``out`` — ``root.output``
      - ``ctx`` — the engine's ``Context``
      - any ``Pattern.name`` declared in ``pattern`` — that pattern
        entry's matched ``Node``
      - anything else — bound positionally to the input ``Node`` at
        slot ``i``, ``None`` past the input count or for deleted
        source nodes.
    """

    name: str
    pattern: list[Pattern]
    rewrite: Callable[..., Graph | Op | None] | None = None
    param_names: tuple[str, ...] = field(default_factory=tuple)


__all__ = ["Rule"]
