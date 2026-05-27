"""Live single-line progress bar for ``deplodock tune`` (default verbosity, tty only).

The two-level tune evaluates a small number of outer fusion terminals (one today); each terminal's tuned op leaves
are its post-fusion kernels (``LoopOp``s), tuned independently by the inner search. We measure progress as
**completed vs total op leaves** across the registered terminals: the op counter advances once per kernel, while
the live tail (current kernel · variant knobs · this variant's perf · running best) updates per benched variant.

When ``enabled`` is False every method is a no-op, so ``handle_tune`` can construct one unconditionally and let it
decide whether to draw (disabled under ``-v`` / ``-q`` / a non-tty stream). Drawing uses ``\\r\\033[K`` to clear and
rewrite a single line on stderr, matching the convention that all ``[tune] …`` status goes to stderr.
"""

from __future__ import annotations

import shutil
import sys


class TuneProgress:
    def __init__(self, *, enabled: bool, stream=None, bar_width: int = 16) -> None:
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.bar_width = bar_width
        self.total = 0
        self.done = 0
        self._drawn = False

    def start_terminal(self, n_ops: int) -> None:
        """Register one outer terminal's tunable-op count (accumulating denominator)."""
        if not self.enabled:
            return
        self.total += n_ops
        self._redraw()

    def op_start(self, name: str) -> None:
        if not self.enabled:
            return
        self._redraw(tail=f"{name}  …")

    def variant(self, kernel: str, knobs_label: str, *, median_us: float | None, status: str, idx: int, best_us: float | None) -> None:
        """Update the live tail for the variant just benched within ``kernel``."""
        if not self.enabled:
            return
        perf = f"{median_us:.1f}us" if median_us is not None and status == "ok" else (status or "—")
        tail = f"{kernel} #{idx + 1}  {knobs_label}  {perf}"
        if best_us is not None and best_us != float("inf"):
            tail += f" (best {best_us:.1f}us)"
        self._redraw(tail=tail)

    def op_done(self, name: str) -> None:
        if not self.enabled:
            return
        self.done += 1
        self._redraw(tail=f"{name}  done")

    def close(self) -> None:
        """Finalize the bar with a trailing newline so following output starts clean. Idempotent."""
        if self.enabled and self._drawn:
            self.stream.write("\n")
            self.stream.flush()
            self._drawn = False

    def _bar(self) -> str:
        if self.total <= 0:
            return "░" * self.bar_width
        filled = max(0, min(self.bar_width, round(self.bar_width * self.done / self.total)))
        return "█" * filled + "░" * (self.bar_width - filled)

    def _redraw(self, tail: str = "") -> None:
        line = f"[tune] [{self._bar()}] {self.done}/{self.total} ops"
        if tail:
            line += f" · {tail}"
        # Truncate to the terminal width so a long variant label can't wrap onto a
        # second line (which would defeat the \r overwrite and leave smeared rows).
        cols = shutil.get_terminal_size(fallback=(120, 24)).columns
        if len(line) >= cols:
            line = line[: max(0, cols - 1)]
        self.stream.write("\r\033[K" + line)
        self.stream.flush()
        self._drawn = True
