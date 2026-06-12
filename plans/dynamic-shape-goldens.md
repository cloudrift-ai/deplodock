# Plan: dynamic-shape goldens — tune/A/B the golden shapes as symbolic-axis kernels

Extend the golden workflow (`tune --dataset golden`, `run --bench --golden`, the `eval` views, the `tune-golden`
skill) so a golden shape can be compiled as a dynamically-shaped kernel: same kernel + inputs, but one axis traced
symbolic (`--dynamic NAME@INPUT:AXIS`), deployed as a masked tile, benched at the `Dim` hint.

The heavy machinery already exists — `--dynamic` tracing (`commands/compile.py:383` `_resolve_dynamic_shapes`),
hint-sized benching (`commands/run.py:1281` `_collect_sym_env`, `run.py:1322` `_hint_sized_inputs`), masked-tile
lowering (thread tier, masked warp MMA, `005_split_demoted` symbolic-row offers), and per-kernel `.torch.json`
reproducers that keep symbolic dims. What's missing is making a symbolic shape a first-class golden: the schema,
the target plumbing, and the search/prior keying all assume integer M/N/K.

## 1. Golden schema + snippet generation (`search/golden.py`)

`MatmulGoldenConfig` stores static `M/N/K` ints (golden.py:111-114); `matmul_snippet()` (golden.py:58-70) emits
`torch.matmul(torch.randn(M,K), torch.randn(K,N))`.

- Add an optional YAML field marking the symbolic axis, e.g. `dynamic: {seq_len: {input: x, axis: 0}}`. The existing
  `M: 512` doubles as the **hint** (matching `DEFAULT_SEQ_HINT` semantics, `compiler/dim.py:105`).
- The snippet stays the same hint-shaped code; the config additionally carries the `--dynamic seq_len@x:0`-style
  spec for the tracer.
- **Distinct names** (`q_proj.s512.dynM`), following the `.fp16` twin precedent: a masked-tile kernel at hint 512 is
  a different deployment artifact than the static-512 kernel (boundary guards, degenerate/masked tiers, different
  variant space) — separate golden, own knobs + latency, never merged with its static twin.
- Loader (`golden.py:162` `_load_goldens`) parses the block; schema tests (`tests/compiler/test_golden_configs.py`)
  validate it (axis int, input name, hint ≥ 1, matmul-only for now).

## 2. Plumbing through tune and run (`commands/tune.py`, `commands/run.py`)

- `_tune_targets()` returns `(label, code, input)` tuples (tune.py:192-219) — extend to carry the dynamic spec; set
  `args.dynamic` in `_tune_one()` before `load_or_trace()` (compile.py:360). The existing `_resolve_dynamic_shapes`
  path does the rest (torch.export `dynamic_shapes`, SymInt propagation).
- `_bench_golden_variants()` (run.py:432-457) re-traces each golden from its snippet — apply the golden's own
  dynamic spec to the re-trace. `resolve_golden_arg()` (compile.py:131) rejects a CLI `--dynamic` next to `--golden`
  (the spec is part of the config, same way `--ir` rejects it).
- Benching needs nothing new: symbolic graphs already bench at the hint with the `benched at seq_len=…` note
  (run.py:1364 `_symbolic_bench_note`); cuBLAS reference numbers are hint-shaped either way, so the recorded
  `deplodock_us` / `cublas_us` stay an apples-to-apples pair *at the hint*.

## 3. Search/prior keying — the real design work

Symbolic extents leak into machinery that expects ints:

- **`ShapeKey`** (`search/data/shape.py:22-49`) computes `free_prod = M*N` arithmetically; a `Dim` makes that an
  `Expr` and `s_features_arith()` breaks. Fix: compute arith features **at the hint** and add a split feature (an
  `S_dyn` flag / per-axis symbolic marker) so symbolic and static twins never share a shape key — exactly how
  `S_dtype_f32` keeps the fp32/fp16 twins apart today.
- **`Sample.from_golden()`** (`search/data/sample.py:118-137`): for symbolic goldens force `compile_s_feats=True`
  (the arith fallback can't run), so the `S_*` histogram comes from the actually-compiled masked-tile graph — the
  same signature a DB-trained prior saw.
- **`op_cache_key` / perf rows** (`search/keys.py:31-55`) are likely fine as-is: they digest kernel source + launch
  geometry, and a masked kernel's source differs (the `int seq_len` runtime arg, the boundary guard, `Expr` grid).
  Add a test asserting static-vs-symbolic twins never collide rather than rewriting the key.
- **Prior featurization**: hint-sized `S_ext_*` plus the symbolic flag is enough for the learned prior to start.
  The `AnalyticPrior` needs the flag too — the masked tier's cost profile (guard overhead, no staged prologue on
  symbolic rows) differs from static at the same hint; refit via `scripts/golden_knob_heuristics.py` once a few
  symbolic goldens are recorded.

## 4. Eval views and the skill

`eval golden / analytic / prior` follow from `Sample.from_golden` once §3 lands. Fix on the way (both bit the
2026-06-12 sweep, `plans/golden-sweep-rtx5090-findings.md`):

- `eval prior --dataset golden` silently drops shapes with no tuned rows (hid the fp16 lockout) — print a per-shape
  warning instead.
- `run --golden`'s name list should include the new dynamic goldens like any other, so the `tune-golden` skill
  sweeps them with zero workflow changes beyond new YAML entries.

## Sequencing caveats

- **Fix the fp16 TMA+WARPSPEC launch crash first** (finding 1 of the 2026-06-12 sweep). Symbolic-M matmuls route
  through the masked warp-tier MMA — the code the crash lives next to — so dynamic fp16 goldens would inherit it
  and the tuner would again be locked out of the class it's supposed to record.
- **Symbolic M only** to start (the seq axis): that's what the masked thread tier, masked warp MMA, and the
  `005_split_demoted` symbolic-row offers support today. Symbolic K (flash-style) is explicitly future work — keep
  it out of the schema until the lowering exists.
- Open question: is a dynamic golden's recorded latency *only* the hint number, or do entries get a small
  `eval_sizes: [128, 512, 2048]` list so the A/B can catch knobs that win at the hint but fall off at other runtime
  sizes? Start single-hint (matches how tune/compile already rank symbolic variants); add multi-size validation
  only if a real regression slips through.

## Order of work

1. Schema + snippet + loader + schema tests (§1) — small, self-contained.
2. Target/A-B plumbing (§2) — small; gated by §1.
3. Keying + featurization (§3) — the careful part; copy the `S_dtype_f32` twin-split pattern.
4. Eval-view fixes (§4), seed a first batch of `.dynM` goldens (square.512/1024 + the qwen s512 projections), run
   the `tune-golden` skill over them, and record.
