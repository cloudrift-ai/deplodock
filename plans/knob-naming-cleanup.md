# Knob naming + env-var consistency cleanup

## Background

The compiler has a clean *intended* design for tunable parameters: a `Knob` descriptor (`compiler/pipeline/knob.py`) is
the canonical schema for one tuning dimension — its name (also the `DEPLODOCK_<NAME>` env-var key), type, autotune
`hints`, and `help`. Descriptors are declared as module-level constants in the rule that owns them, auto-collected by
`registry()` (walks loaded modules for `Knob` instances), env-pinnable via `Knob.narrow`, and *all* `os.environ` access
for `DEPLODOCK_*` routes through `config.py`. Most lowering knobs follow this. The cleanup below targets the places that
don't — knob-namespace vars that bypass the descriptor system, duplicated name strings, divergent `config.py` idioms,
inconsistent knob names, and stale references.

An audit on 2026-05-30 (branch `feature/mma-perf-closures`) found six categories of issue. This plan addresses them in
milestone order, lowest-risk / highest-value first. Renames touch the public `DEPLODOCK_*` env namespace, so they land
last and update docstrings + `tuning.py` docs in lockstep.

## Current state (audit findings)

1. **Knob-namespace vars that bypass the `Knob` system.** Three `DEPLODOCK_*` vars are real codegen-affecting knobs but
   are hardcoded as bespoke getters in `config.py` instead of `Knob` descriptors in their owning rule:

   | Var | Getter | Owning rule | Candidate set |
   |---|---|---|---|
   | `DEPLODOCK_MMA` | `config.mma_enabled()` | `lowering/tile/010_partition_loops.py` | on/off |
   | `DEPLODOCK_TMA_SWIZZLE` | `config.tma_swizzle_enabled()` | `compiler/tuning.py` (`_tma_swizzle_enabled`) | on/off |
   | `DEPLODOCK_GROUP_M` | `config.group_m()` | `lowering/tile/025_swizzle_blocks.py` | `(1,2,4,8,16)` |

   Because they aren't descriptors they don't appear in `deplodock knobs`, aren't `format_tuning_knobs`-rendered, can't
   be forked/autotuned (only env-pinned), and `GROUP_M`'s allowed-value validation is duplicated in `config.py` rather
   than expressed as `hints`. `DEPLODOCK_MMA` is more defensible as a genuine global feature flag (it gates whether MMA
   variants are *emitted at all*, separate from the `ATOM_KIND` fork that picks between emitted variants) — but it still
   must not read via a raw string literal (see #3).

2. **`tuning.py` duplicates matmul knob names as bare strings.** `compiler/tuning.py` reads the matmul tile knobs via
   `_knob_int("BN", ...)`, `"BM"`, `"FM"`, `"FN"`, `"BK"`, `"SPLITK"` — string literals — while the canonical `Knob`
   descriptors for those exact names already live in `_enumeration.py` (`BN = Knob("BN", ...)`). Two parallel
   definitions of the same namespace: the schema (hints/help) is in `_enumeration.py`, the heuristic-default path
   re-spells the names by hand. A rename in `_enumeration.py` would silently leave `tuning.py` reading a dead var. The
   two lists are *already* out of sync — `tuning.py` lacks the newer warp-tier knobs (`BR`, `WN`, `WM`, `ATOM_KIND`).

3. **`config.py` internal idiom divergence.** The three knob-namespace getters each read the namespace differently:
   `mma_enabled` uses a hardcoded literal `_bool("DEPLODOCK_MMA", ...)` (defeats `knob_var`); `tma_swizzle_enabled` uses
   `_bool(knob_var("tma_swizzle"))` (lowercase arg); `group_m` uses `knob_raw("GROUP_M")` (uppercase arg). Functionally
   `knob_var` uppercases, so the latter two are equivalent — but it reads as three conventions in one file.

4. **`Knob` name inconsistencies.**
   - `USE_` prefix applied unevenly: `USE_TMA` / `USE_ASYNC_COPY` carry it; equivalent pass-on/off markers `PAD_SMEM`,
     `PIPELINE_STAGES`, `VECTORIZE_LOADS`, `INTERLEAVE_LOADS`, `PERMUTE_LANES`, `HOIST_COMPUTE`, `ATOMIC_FREE_SPLITK`,
     `WS` do not.
   - `WS` is a cryptic 2-letter abbreviation for warp-specialize while every peer spells out the verb. It is also
     `KnobType.BOOL` yet declares integer `hints=(0, 1)` instead of `(False, True)` like every other BOOL knob — a real
     type inconsistency (hints feed the tuner as ints for one knob, bools for the rest).
   - `BUFCNT` is abbreviated while the closely related `PIPELINE_STAGES` (both control multi-buffering depth) is spelled
     out under a different convention.
   - `ATOM_KIND` vs `ATOMIC_FREE_SPLITK` share an `ATOM`/`ATOMIC` prefix but are unrelated (atom *tile kind* vs
     *atomic-free* split-K) — easy to confuse when scanning `DEPLODOCK_KNOBS`.
   - The matmul tile family (`BN/BM/BK/BR/WN/WM/FM/FN/SPLITK`) is terse but internally coherent — leave it.

5. **BOOL `hints` ordering.** Greedy uses `hints[0]` as its default pick, so order encodes default-on vs default-off:
   `(False, True)` for `HOIST_COMPUTE`/`PAD_SMEM`/`ATOMIC_FREE_SPLITK` (default-off) vs `(True, False)` for
   `PIPELINE_STAGES`/`USE_ASYNC_COPY`/`USE_TMA` (default-on). That part is intentional. But the convention isn't
   self-documenting and `WS`'s `(0, 1)` is an outlier — worth a confirming pass that each default is the intended one.

6. **Stale env-var references in comments.** `DEPLODOCK_MMA_STAGE_PROBE` (`100_materialize_tile.py:172`) and
   `DEPLODOCK_LOG_DEDUP` (`011_dedup_replicated.py:12`) appear *only* in comments — there is no actual env read for
   either. Same for the already-removed `DEPLODOCK_AFFINE_COLLAPSE` and `DEPLODOCK_WIDE_FM_FN`. Dead references that
   imply behavior that no longer exists.

## Goal

Every `DEPLODOCK_*` knob-namespace var is either (a) a registry `Knob` descriptor in its owning rule, or (b) a
deliberate non-knob config var documented as such. No bare knob-name string literals outside the descriptors. One
`config.py` idiom for reading the knob namespace. Consistent, self-documenting knob names. No stale env references.

## Milestones

Each milestone is one commit on a single feature branch, after `make test` + `make lint`. Verify codegen-affecting
changes with focused examples (matmul / softmax / RMSNorm / SDPA via `deplodock run --code`), not the full suite.

### M1 — `config.mma_enabled` reads through `knob_var` (no rename, no behavior change)

Change `_bool("DEPLODOCK_MMA", default=True)` → `_bool(knob_var("MMA"), default=True)`. Pure idiom fix; the env var
spelling is unchanged. Lowest-risk item; lands first to settle the `config.py` convention. Decide and apply one casing
for `knob_var`/`knob_raw` args across `mma_enabled` / `tma_swizzle_enabled` / `group_m` (recommend uppercase, matching
the `DEPLODOCK_<NAME>` form users actually type).

### M2 — `tuning.py` reads matmul knobs through the `_enumeration.py` descriptors

Replace the bare-string `_knob_int("BN", ...)` calls with reads keyed off the canonical descriptors. Two options:

- **Option A (preferred):** add a small `Knob.read_int(default)` helper on the descriptor (`config.int_env(self.env,
  default)`) and call `BN.read_int(def_bn)` etc., importing the constants from `_enumeration.py`. Kills the duplicate
  names; a rename now fails loudly at import.
- **Option B (minimal):** keep `_knob_int` but pass `BN.name` instead of `"BN"`. Still couples to the descriptor without
  the new method.

Watch for an import cycle (`tuning.py` ↔ `_enumeration.py`); if one appears, the helper lives on `Knob` (already
imports only `config`) and `tuning.py` imports the constants, not vice-versa. This is the highest-value structural fix —
it removes the silent-drift hazard between the two parallel name lists.

### M3 — Promote `GROUP_M` and `TMA_SWIZZLE` to real `Knob` descriptors

- `GROUP_M`: declare `GROUP_M = Knob("GROUP_M", KnobType.INT, hints=(1,2,4,8,16), help=...)` in
  `025_swizzle_blocks.py`. Move the allowed-value validation out of `config.group_m()`; either keep a thin
  `config.group_m()` that delegates to the descriptor (`GROUP_M.read_int(8)` + validate) or inline the read in the rule.
  Preserve the `1 = disable swizzle` escape hatch and the `ValueError`-on-garbage behavior.
- `TMA_SWIZZLE`: declare `TMA_SWIZZLE = Knob("TMA_SWIZZLE", KnobType.BOOL, hints=(False, True), help=...)` in its owning
  rule (currently gated from `compiler/tuning.py:_tma_swizzle_enabled`). Decide owning module — likely `050_use_tma.py`
  or `tuning.py` if it stays heuristic-side.

After this, both appear in `deplodock knobs` and become candidates for forking/autotune. Decide per-knob whether to
actually wire them into a fork (`GROUP_M` has a natural candidate set; `TMA_SWIZZLE` is on/off) or leave them
registry-visible but manual-only for now. Keep `DEPLODOCK_MMA` as a documented global feature flag (not a per-op knob),
but ensure it reads via `knob_var` (done in M1).

### M4 — Delete stale env-var references in comments

Remove the `DEPLODOCK_MMA_STAGE_PROBE` comment in `100_materialize_tile.py`, the `DEPLODOCK_LOG_DEDUP` comment in
`011_dedup_replicated.py`, and any lingering `DEPLODOCK_AFFINE_COLLAPSE` / `DEPLODOCK_WIDE_FM_FN` mentions that describe
removed behavior. Pure doc hygiene; can ride with any other milestone.

### M5 — Fix `WS` hints type, then rename for clarity (env-namespace change — lands last)

- First, change `WS` hints `(0, 1)` → `(False, True)` to match every other BOOL knob (no env-spelling change).
- Then rename `WS` → `WARP_SPECIALIZE` (`DEPLODOCK_WS` → `DEPLODOCK_WARP_SPECIALIZE`). Update the descriptor, the
  `RuleSkipped` message ("DEPLODOCK_WS env pin removed all WS choices"), the docstring at `085_warp_specialize.py:74`,
  and any `plans/`/`tests/` references. Grep `DEPLODOCK_WS` and `\bWS\b` repo-wide first.

### M6 — Settle the `USE_` prefix convention (env-namespace change — optional, lands last)

Pick one convention for pass-on/off marker knobs and apply it. Two coherent choices:

- **Drop `USE_`:** `USE_TMA` → `TMA`, `USE_ASYNC_COPY` → `ASYNC_COPY`. Shorter; matches the majority (`PAD_SMEM`,
  `PIPELINE_STAGES`, …). Note `TMA` would then sit beside the `MMA` global flag — fine, both are feature gates.
- **Keep `USE_`** only where it disambiguates a verb from a noun, document the rule in `STYLE.md`.

Also consider `BUFCNT` → `BUFFER_COUNT` (or `RING_DEPTH`) to match the spelled-out convention and sit naturally beside
`PIPELINE_STAGES`. Each rename updates the descriptor, docstrings, `RuleSkipped`/`ValueError` messages, `tuning.py`
docs, `plans/`, and `tests/`. M6 is the most invasive and least functionally important; do it only if the team wants
the namespace tidy. Skippable.

## Risks / gotchas

- **Env-namespace renames (M5/M6) are user-visible.** Anyone with `DEPLODOCK_WS=…` in a script or `DEPLODOCK_KNOBS`
  breaks silently (the old name becomes a no-op — `Knob.narrow` only honors the registered name). Grep `tests/`,
  `plans/`, `Makefile`, and recipe dirs before renaming. No back-compat alias unless the team wants one.
- **`registry()` dedups by first-seen name.** Two descriptors with the same `name` collapse — when adding `GROUP_M` /
  `TMA_SWIZZLE`, make sure the name isn't already claimed elsewhere.
- **Import cycles in M2.** `tuning.py` importing `_enumeration.py` (a lowering pass) may cycle; keep the read helper on
  `Knob` (config-only deps) and have `tuning.py` import the leaf constants.
- **`config.group_m()` callers.** `025_swizzle_blocks.py` and `tuning.py` call it; preserve the `==1` disable path and
  the validation semantics when relocating.
- Follow the repo feedback: tight A/B verification on focused examples, not full sweeps; one branch with milestone
  commits, no separate PRs per milestone.
