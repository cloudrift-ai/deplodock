"""Pattern-based rewrite engine — value types + driver.

Public surface (re-exported from sibling modules so callers can
``from deplodock.compiler.pipeline.engine import X`` for any of them):

- :mod:`.pipeline` — ``Pattern`` / ``Match`` / ``Rule`` /
  ``RuleSkipped`` / ``Pass`` / ``Pipeline``. Bundled together because
  they form a tight chain (``Pipeline → Pass → Rule → Pattern``, plus
  ``Match`` carrying ``Rule``).
- :mod:`.driver` — rule loader, rewrite dispatcher, search loop, and
  the public ``run_pipeline`` / ``run_autotune`` entry points.

Rule modules under ``passes/`` declare ``PATTERN = [Pattern(...), ...]``
and a ``rewrite(...)`` function whose return type discriminates the
rewrite flavor:

* ``Graph`` — functional fragment, spliced in place of the match.
* ``Op`` — in-place rebind of ``root.op`` (id, inputs, hints kept).
* ``list[Graph | Op]`` — autotuning fork: engine applies option 0
  inline and pushes one ``Candidate`` per remaining option onto the
  search queue.

Raise ``RuleSkipped`` to decline a match.
"""

from deplodock.compiler.pipeline.engine.driver import (
    _build_rewrite_kwargs,
    _strip_rule_prefix,
    run_autotune,
    run_pipeline,
)
from deplodock.compiler.pipeline.engine.pipeline import (
    Match,
    Pass,
    Pattern,
    Pipeline,
    Rule,
    RuleSkipped,
)

__all__ = [
    "Match",
    "Pass",
    "Pattern",
    "Pipeline",
    "Rule",
    "RuleSkipped",
    "_build_rewrite_kwargs",
    "_strip_rule_prefix",
    "run_autotune",
    "run_pipeline",
]
