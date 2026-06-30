"""Live single-line progress bar for ``emmy tune`` (default verbosity, tty only).

The two-level tune evaluates a small number of outer fusion terminals (one today); each terminal's tuned op leaves
are its post-fusion kernels (``LoopOp``s), tuned independently by the inner search. We measure progress as
**completed vs total op leaves** across the registered terminals: the op counter advances once per kernel, while
the live tail shows each *currently in-flight* kernel (current variant's perf · running best · knobs). Single-GPU
tuning runs one kernel at a time → one tail entry; multi-GPU (``--gpus``) runs one per device → the tail joins the
active kernels with `` | `` so the line tracks every slot at once. Per active kernel the current latency is
fixed-width (it changes every variant, so a fixed field keeps the rest from shifting).

The ``slot`` key identifies the in-flight kernel (the inner driver passes each op's ``op_idx``); at most
``#GPUs`` are active between ``op_start`` and ``op_done``, so the tail never grows past the device count.

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
        self._variants: dict[int, int] = {}  # slot → benched-variant count for its current op
        self._tails: dict[int, str] = {}  # slot → live tail for its in-flight kernel (insertion-ordered)
        self._drawn = False

    def start_terminal(self, n_ops: int) -> None:
        """Register one outer terminal's tunable-op count (accumulating denominator)."""
        if not self.enabled:
            return
        self.total += n_ops
        self._redraw()

    def op_start(self, name: str, *, slot: int = 0) -> None:
        if not self.enabled:
            return
        self._variants[slot] = 0
        self._tails[slot] = f"{name}  …"
        self._redraw()

    def variant(
        self,
        kernel: str,
        knobs_label: str,
        *,
        median_us: float | None,
        status: str,
        best_us: float | None,
        slot: int = 0,
    ) -> None:
        """Update the live tail for the variant just benched within ``kernel`` on ``slot``.

        Layout ``<kernel> #<n> <current> (best <best>) <knobs>``: the *current* latency is
        fixed-width (it changes every variant, so a fixed field keeps everything after it
        from shifting), and the variable-length knob string sits last."""
        if not self.enabled:
            return
        n = self._variants.get(slot, 0) + 1
        self._variants[slot] = n
        cur = f"{median_us:8.1f}us" if (median_us is not None and status == "ok") else f"{status or '—':>10}"
        best = f"{best_us:.1f}us" if (best_us is not None and best_us != float("inf")) else "—"
        self._tails[slot] = f"{kernel} #{n} {cur} (best {best})  {knobs_label}"
        self._redraw()

    def op_done(self, name: str, *, slot: int = 0) -> None:
        if not self.enabled:
            return
        self.done += 1
        self._tails.pop(slot, None)
        self._variants.pop(slot, None)
        self._redraw()

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

    def _redraw(self) -> None:
        line = f"[tune] [{self._bar()}] {self.done}/{self.total} ops"
        tail = " | ".join(self._tails[s] for s in self._tails)
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
