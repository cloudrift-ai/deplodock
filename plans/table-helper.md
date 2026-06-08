# Shared console-table helper + shrink knob tables

## Context

The CLI hand-rolls every table with f-string format specs. The genuinely hard part — ANSI-aware width (a
`\033[31m…\033[0m` cell breaks `str.ljust`) — is solved in only two spots (`knobfmt.align_knob_columns`,
`eval._cell`); everything else hardcodes widths (`"-"*90`, `<44`, `[:42]`) and the plain latency table is duplicated
three times with different widths. Separately, the knob tables waste width by repeating the knob name in every cell
(`BM=8/8 BN=32/32 …`).

This change adds **one** small ANSI-aware table helper and routes the knob and latency tables through it. Knob names
move from each cell into a shared header row (cells become `8/8`), which both shrinks the tables and is what the helper
naturally produces (every column has a name = its header). `align_knob_columns` and `eval._cell`/`_aligned_knob_cells`
are deleted.

Outcome: one source of truth for column width / alignment / color; narrower knob tables; the three latency tables
collapse to one call each.

## Design

New module **`deplodock/commands/table.py`** (replaces `knobfmt.py`, which is deleted):

- Color constants `TTY, GREEN, YELLOW, RED, RESET` (moved verbatim from `knobfmt.py`; tty-gated).
- `Col(name, align)` — `align` is `"l"` or `"r"`.
- A **cell** is a plain `str` or a `(text, color)` tuple. Width is measured by `len(text)` (visible length), never the
  colored body — the trick `align_knob_columns`/`_cell` already use. Colored body is `f"{color}{text}{RESET}"` only when
  `color` is truthy (`""` → plain), padded by `width - len(text)`.
- `col_widths(columns, rows) -> list[int]` — `width[i] = max(len(col[i].name), max visible cell len in column i)`. Sizing
  to the header name is exactly why the old aligner couldn't host the header (it only sized to cells, so `SPLITK` wouldn't
  fit over `1`).
- `render_table(columns, rows, *, rule=False, indent="", gap="  ") -> list[str]` — built on `col_widths`. Returns header
  line, optional `---` rule, then one line per row. Right-align pads before the body, left-align after; each line
  `rstrip`-ed; `indent` prefixed to every line. **All number/percent/`x`/`K`/`us` formatting stays at call sites** — cells
  are final strings.
- `knob_columns(rows: list[dict[str, tuple[str, bool]]]) -> tuple[list[Col], list[list[cell]]]` — replaces
  `align_knob_columns`. Union of knob names sorted by `knob_sort_key` (`compiler/pipeline/knob.py`) → one **right-aligned**
  `Col` per knob (name = header). Per row: `(value_text, RED if red else "")` where present, `""` where absent. Callers
  build the `{name: (value_text, red)}` dicts with **no `NAME=` prefix** in `value_text`.

## Call-site rewrites

**`deplodock/commands/eval.py`** — golden tables `_emit_analytic_eval` / `_emit_prior_golden_check`:
- Build leading `Col`s (`kernel` L; `m/t` L colored via `_ratio_color`; `analytic` also `rank` L, `pool` L) + leading
  cells, then append `knob_columns(...)`; `render_table(..., indent="  ", rule=False)`.
- Keep the existing `("row"/"err")` `entries` interleave: render only the non-err rows, print the header line, then
  re-walk `entries` pulling data lines for `"row"` / printing err text for `"err"`. **Slice header by the `rule` flag**
  (no rule here → header is line `[0]`). **Err-row prefix indent must come from the rendered kernel-column width** (include
  err kernel names in the width calc) — not the old `nw` — else err rows misalign.
- Move the descriptive `knobs (found/golden; red = mismatch)` text to a caption line above the table.
- Delete `_cell` and `_aligned_knob_cells`; repoint color imports (`_GREEN/_RED/_RESET/_YELLOW`) to `table`. Keep
  `_ratio_color` (now feeds the `(text, color)` slot).

**`deplodock/commands/run.py`** — `_print_kernel_stats`:
- Leading `Col`s (`Kernel` L; `us %  grid block smem regs occ` all R) + `knob_columns`. `TOTAL` becomes a final row with
  blank trailing knob cells (its `us` cell sizes the column like any other; blanks rstrip away).
- Drop hardcoded `<44` / `[:42]` / `"-"*90`; let `render_table` compute widths (confirm `golden NAME` labels are no longer
  truncated). Caption `knobs (greedy pick)` above. Repoint the local `align_knob_columns` import to
  `table.knob_columns`.
- `_print_table` (Backend/Latency/vs Eager), the lightweight deplodock-only bench (968-971): plain
  `render_table(rule=True)`, numeric cols right-aligned. Replaces `"-"*48` / `"-"*38`.

**`deplodock/commands/tune.py`** — `_print_per_kernel_table`: plain `render_table(rule=True)`; replaces `"-"*84`.

**`deplodock/commands/eval.py`** — stats tables:
- `_emit_regret_table`, `_emit_interaction_matrix`: `render_table(rule=...)`, **`indent=""`** (the regret test depends on
  rows starting `"\nBIG "`; the interaction first `Col` name stays literally `"K1\\K2"`). Accept ragged per-column widths
  in the matrix (vs today's uniform 10-wide grid) — more compact, only the corner string is asserted.

## Out of scope (recommended deviation — confirm)

- **`_emit_registry`** (the knob *schema* table, ~`eval.py:188-213`): its `help` column wraps via `textwrap.wrap` with a
  hanging indent — a multi-line-cell layout the single-line row helper doesn't model. It uses no color and shares no logic
  with the others, so migrating it adds the only leaky special case for zero dedup benefit. **Recommend leaving it as-is.**
  (You selected "migrate stats tables for consistency"; this is the one table where that costs more than it gives. Say the
  word and I'll fold it in with manual continuation lines.)
- `scripts/bench_block.py` — left alone (you selected only run.py + tune.py for latency tables).

## Docs / tests

- Delete `knobfmt.py`. Update **CLAUDE.md:117** (`commands/knobfmt.align_knob_columns` → `commands/table`). No
  `commands/ARCHITECTURE.md` mention of knobfmt/these tables (grep-confirmed) — nothing to update there; re-check on edit.
- Replace `tests/compiler/cli/test_eval.py::test_align_knob_columns_orders_and_aligns` with a `table` unit test:
  `render_table`/`knob_columns` puts names in the header, values-only in cells, canonical column order, ANSI-aware width,
  and correct left/right padding (assert visible alignment with a colored cell). Verify the existing eval-knobs tests
  (`"\nBIG "` regret row start, `"K1\\K2"` matrix corner, `"knob interaction"`) still hold.

## Critical files

- `deplodock/commands/table.py` (new), `deplodock/commands/knobfmt.py` (delete)
- `deplodock/commands/eval.py`, `deplodock/commands/run.py`, `deplodock/commands/tune.py`
- `deplodock/compiler/pipeline/knob.py` (`knob_sort_key` — read only, reused)
- `tests/compiler/cli/test_eval.py`, `CLAUDE.md`

## Verification

1. `./venv/bin/pytest tests/compiler/cli/test_eval.py tests/compiler/cli/test_run.py -p no:randomly` — knob-order /
   eval-knobs / latency-table tests pass (new `table` unit test included).
2. `make lint` (then `make format` if needed).
3. Eyeball the rendering — knob names in the header, values-only cells, columns aligned, mismatches red on a tty:
   - `deplodock eval analytic` and `deplodock eval golden` (knob value tables).
   - `deplodock eval knobs` against a tune DB (regret table + interaction matrix).
   - `deplodock run --code "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))" --bench` (per-kernel + latency tables; pipe
     to a file to confirm plain output has no stray ANSI and stays aligned).
4. `make test` for the full suite.
