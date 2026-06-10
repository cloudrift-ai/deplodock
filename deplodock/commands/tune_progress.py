"""Live single-line progress bar for ``deplodock tune`` (default verbosity, tty only).

The two-level tune evaluates a small number of outer fusion terminals (one today); each terminal's tuned op leaves
are its post-fusion kernels (``LoopOp``s), tuned independently by the inner search. We measure progress as
**completed vs total op leaves** across the registered terminals: the op counter advances once per kernel, while
the live tail (current kernel · this variant's perf · running best · variant knobs) updates per benched variant.
The current latency is fixed-width and the variable-length knob string sits last, so the prefix up to the knobs
stays put as the per-variant latency changes (only a new best, which is rare, shifts the trailing part) — the
bar / counter / kernel / current timing stay steady instead of flickering.

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
        self._variants = 0  # benched variants for the current op (visible activity within a single-op tune)
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
        self._variants = 0
        self._redraw(tail=f"{name}  …")

    def variant(self, kernel: str, knobs_label: str, *, median_us: float | None, status: str, best_us: float | None) -> None:
        """Update the live tail for the variant just benched within ``kernel``.

        Layout ``<kernel> <current> (best <best>) <knobs>``: the *current* latency is
        fixed-width (it changes every variant, so a fixed field keeps everything after
        it from shifting), and the variable-length knob string sits last where its churn
        can't move the bar / counter / kernel / current timing. Only a new best (rare)
        nudges the trailing part."""
        if not self.enabled:
            return
        self._variants += 1
        cur = f"{median_us:8.1f}us" if (median_us is not None and status == "ok") else f"{status or '—':>10}"
        best = f"{best_us:.1f}us" if (best_us is not None and best_us != float("inf")) else "—"
        self._redraw(tail=f"{kernel} #{self._variants} {cur} (best {best})  {knobs_label}")

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
