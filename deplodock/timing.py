"""Phase timing collector for deploy/bench.

``PhaseTimer`` records named phase durations (seconds). It is threaded through the
async deploy/bench call chain by mutation, so we never change ``run_deploy``'s
``bool`` return type — callers that want timing pass a timer in and read it back
after; callers that don't get a discarded throwaway timer.
"""

import logging
import time
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)

# ── Phase names (stable snake_case keys, shared across orchestrator, JSON, table) ──

# Provisioning (per VM, shared across a group's tasks)
PHASE_VM_PROVISION = "vm_provision"
PHASE_REMOTE_PROVISION = "remote_provision"

# Deploy (per task)
PHASE_IMAGE_PULL = "image_pull"
PHASE_MODEL_DOWNLOAD = "model_download"
PHASE_MODEL_LOAD_AND_WARMUP = "model_load_and_warmup"  # compose up --wait until /health
PHASE_SMOKE_TEST = "smoke_test"

# Optional sub-phases parsed best-effort from container logs (absent when unmatched).
# They live *inside* model_load_and_warmup, so they are excluded from the total.
PHASE_WEIGHTS_LOAD = "weights_load"
PHASE_CUDA_GRAPH = "cuda_graph_capture"

# Benchmark / teardown (per task)
PHASE_BENCHMARK = "benchmark"
PHASE_TEARDOWN = "teardown"
PHASE_COMMAND = "command"  # command-recipe coarse wall-clock

# Sub-phases excluded from total() (they are a breakdown of model_load_and_warmup).
SUBPHASES = frozenset({PHASE_WEIGHTS_LOAD, PHASE_CUDA_GRAPH})

# Canonical render order for the .txt section and console table.
PHASE_ORDER = [
    PHASE_VM_PROVISION,
    PHASE_REMOTE_PROVISION,
    PHASE_IMAGE_PULL,
    PHASE_MODEL_DOWNLOAD,
    PHASE_MODEL_LOAD_AND_WARMUP,
    PHASE_WEIGHTS_LOAD,
    PHASE_CUDA_GRAPH,
    PHASE_SMOKE_TEST,
    PHASE_BENCHMARK,
    PHASE_TEARDOWN,
    PHASE_COMMAND,
]


class PhaseTimer:
    """Ordered collector of phase durations (seconds), mutated in place.

    Durations accumulate (``+=``): recording the same phase twice adds to it, so a
    per-group provisioning duration can be seeded into each task's timer cleanly.
    """

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}

    def record(self, name: str, seconds: float, *, log: bool = True) -> None:
        self.phases[name] = self.phases.get(name, 0.0) + seconds
        if log:
            logger.info(f"[timing] {name}: {seconds:.1f}s")

    @contextmanager
    def measure(self, name: str):
        """Sync context manager. Records elapsed even if the body raises."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - t0)

    @asynccontextmanager
    async def ameasure(self, name: str):
        """Async context manager. Records elapsed even if the body raises."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - t0)

    def total(self) -> float:
        """Sum of top-level phases (sub-phases excluded to avoid double-counting)."""
        return sum(v for k, v in self.phases.items() if k not in SUBPHASES)

    def as_dict(self) -> dict[str, float]:
        """Rounded copy for serialization, with a ``total`` key appended."""
        out = {k: round(v, 2) for k, v in self.phases.items()}
        out["total"] = round(self.total(), 2)
        return out

    def format_table(self) -> str:
        """Aligned plain-text breakdown (sub-phases indented), total last."""
        return format_timing(self.as_dict())


def format_timing(timing: dict[str, float]) -> str:
    """Render a timing dict (phase -> seconds, may include ``total``) as aligned text.

    Phases are shown in :data:`PHASE_ORDER`; sub-phases are indented to signal they
    are a breakdown of ``model_load_and_warmup``. ``total`` is rendered last.
    """
    keys = [k for k in PHASE_ORDER if k in timing]
    width = max((len(k) for k in keys), default=0)
    if "total" in timing:
        width = max(width, len("total"))
    lines = []
    for k in keys:
        indent = "  " if k in SUBPHASES else ""
        label = (indent + k).ljust(width + 2)
        lines.append(f"{label}{timing[k]:>8.1f}s")
    if "total" in timing:
        lines.append(f"{'total'.ljust(width + 2)}{timing['total']:>8.1f}s")
    return "\n".join(lines)
