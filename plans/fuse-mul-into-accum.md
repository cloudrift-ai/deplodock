# Fuse `Assign(multiply) + Accum(add)` into an inline FMA — Kernel-IR codegen pass

**Branch:** `feature/fuse-mul-into-accum`
**Origin:** `plans/matmul-cublas-gap-2026-05-30.md` § "Hand-optimized kernel variants — measured 2026-05-30"
**Effort:** ~1 day, low risk

## Problem

The matmul kernel (and any reduce whose body is `acc <- add(acc, mul(a, b))`) currently lowers
to a two-statement CUDA pattern:

```c
float v0  = in0 * in1;
float v1  = in0 * in2;
...
float v103 = in26 * in29;
acc0  += v0;
acc1  += v1;
...
acc103 += v103;
```

Manual A/B (`plans/matmul-cublas-gap-2026-05-30.md` § hand-optimized variants) showed that
rewriting the body to:

```c
acc0  += in0 * in1;
acc1  += in0 * in2;
...
acc103 += in26 * in29;
```

is **0.6 percentage points faster** on 2048³ fp32 matmul (`97.4 % → 98.0 % of cuBLAS`), measured
over 10 rounds × 1000 iters round-robin. ptxas evidently does not always fold the two-step form
back into a single FFMA when the SSA cluster carries 104 live `vN` intermediates; the inline
form gives it an unambiguous FMA pattern and removes the register-pressure detour.

The win is **structural** — applies to every kernel lowered through this path, not just 2048³,
not just matmul. A reduce whose body is a multiply-add pair is the inner loop of matmul, SDPA
score×value, RMSNorm sum-of-squares, and any dot-product reduction.

## Where the IR pattern lives

The canonical matmul cell (`deplodock/compiler/pipeline/passes/lowering/tile/_atom.py:111`) is:

```
[Load a, Load b, Assign(name=v, op=multiply, args=(a, b)), Accum(name=acc, op=add, value=v)]
```

After `095_interleave_loads.py` sinks loads next to their consumers and
`100_materialize_tile.py` lowers the TileOp body into concrete Kernel-IR statements, the body
contains a flat sequence of these `Assign + Accum` pairs (104 of them for the golden tile).

The CUDA renderer (`deplodock/compiler/ir/stmt/leaves.py:483` for `Assign`, `:501` for `Accum`)
emits each stmt in isolation — `Assign` becomes `float v = a * b;` and `Accum` becomes
`acc += v;`. There is no peephole that joins them.

## Design

### IR change — one optional field on `Accum`

In `deplodock/compiler/ir/stmt/leaves.py`, extend `Accum`:

```python
@dataclass(frozen=True)
class Accum(Stmt):
    name: str
    value: str
    op: ElementwiseImpl = field(default_factory=lambda: ElementwiseImpl("add"))
    dtype: DataType | None = None
    axes: tuple[str, ...] = ()
    # NEW: when set, render emits "acc <op>= factor[0] * factor[1];" and ignores `value`.
    # The fuse-mul-into-accum pass populates this for FMA-able matmul cells. None preserves
    # the existing `acc <op>= value;` rendering.
    factor: tuple[str, str] | None = None
```

`Accum.deps()` returns `self.factor or (self.value,)` so dependency tracking continues to work
(the renamer / dead-code passes see the operand names, not the dropped intermediate).

`Accum.render()` gains one branch — when `factor` is set, emit
`f"{pad}{name} += {factor[0]} * {factor[1]};"` (with the dtype-conversion logic threading
through both operands).

### New pass — `kernel/120_fuse_mul_into_accum.py`

Runs after `110_drop_redundant_syncs.py` (the last Kernel-IR pass before render). Scans each
`Body` for the peephole:

```
Assign(name=v, op=multiply, args=(a, b), dtype=D)
Accum(name=acc, op=add, value=v, factor=None)        ← adjacent or separated only by Assigns
                                                       not referencing v
```

…where `v` is referenced **exactly once** in the enclosing body (the `Accum`'s `value`). The
rewrite drops the `Assign` and replaces the `Accum` with `Accum(..., factor=(a, b))`. Other
fields (dtype, axes) are preserved.

Eligibility (all required):

- `Assign.op.name == "multiply"` and `len(Assign.args) == 2`
- `Accum.op.name == "add"` (the only FMA-able combine on FP32; `multiply`/`min`/`max` don't
  fuse into FMA on Blackwell SIMT)
- The `Assign.name` is consumed exactly once in the body — by this `Accum.value`
- The two stmts may be separated by other stmts, **as long as none of them reference
  `Assign.name`** and none redefine `a` or `b` (handles `095_interleave_loads.py`'s sinking,
  which can put the matching `Accum` several lines below the `Assign`)
- `Accum.dtype` and `Assign.dtype` are compatible (if both set, must agree; if only Assign is
  set, propagate to Accum's render path)

### Where it runs

New file `deplodock/compiler/pipeline/passes/lowering/kernel/120_fuse_mul_into_accum.py`.
Registered in `KERNEL_PASSES` / `LOWERING_PASSES` after `110_drop_redundant_syncs`. No knob —
this is unconditional: never increases instruction count, never increases register pressure,
and is structurally correct whenever the use-count invariant holds.

## Milestones (single branch, commit after each `make test` passes)

Per `feedback_single_branch_milestones`: one branch, milestone commits after `make test` —
no separate PRs.

**M1 — `Accum.factor` field + render branch**

- Add `factor: tuple[str, str] | None = None` to `Accum`.
- Update `Accum.deps()`, `Accum.pretty()`, `Accum.render()` to handle the new field.
- Update any equality / hashing tests that enumerate `Accum` fields.
- No fusion pass yet — field is unused, all existing tests pass unchanged.

**M2 — Fusion pass**

- Add `kernel/120_fuse_mul_into_accum.py` with the peephole described above.
- Register in `KERNEL_PASSES`.
- Add unit tests in `tests/compiler/pipeline/passes/lowering/kernel/test_fuse_mul_into_accum.py`:
  - Trivial case: one mul + one accum → one fused accum
  - Multi-use Assign: skipped (no fusion)
  - Wrong op (subtract / divide): skipped
  - Separated by unrelated stmts: still fuses
  - Mul + non-add Accum: skipped
  - Dtype mismatch path

**M3 — End-to-end matmul validation**

- Run `deplodock run -c "torch.matmul(torch.randn(2048,2048),torch.randn(2048,2048))" --bench`
  with `--warmup 50 --iters 200`, capture before/after kernel latency from at least 3
  separate invocations. Expect ~0.6 pp improvement (from `~96 %` to `~96.5 %` of cuBLAS in
  deplodock's bench methodology; the standalone harness measures the same delta as
  `97.4 % → 98.0 %`).
- Re-render and diff `--ir cuda` to confirm the `v = a*b; acc += v` pairs are gone.
- Re-run TinyLlama block accuracy test from `tests/perf/` (or `scripts/bench_block.py`).

**M4 — Broader shape sweep**

- Run on three shapes: 2048³ (the article's golden), `128×16384×128` (split-K shape), and
  Qwen3-Embedding-0.6B at `--layer 0` with `--seq-len 32`. Latency should be equal or better;
  accuracy unchanged.
- If any shape regresses, gate the pass behind a knob (unlikely — register pressure can only
  drop with one fewer SSA name per cell).

**M5 — Docs**

- Update `deplodock/compiler/ARCHITECTURE.md` (kernel pass list) with the new pass.
- Update `deplodock/compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md` if one exists.
- Note in `plans/matmul-cublas-gap-2026-05-30.md` that the V3 finding has shipped.

## Validation checklist (per `CLAUDE.md` § "Before committing")

After every milestone:

1. `make test` — all 600+ tests pass
2. `make lint` (or `make format` then `make lint`)
3. Update `ARCHITECTURE.md` files in directories touched

After M3 / M4 specifically:

4. Capture before/after `deplodock --bench` numbers on 2048³ in the commit message
5. Sanity-check the generated CUDA via `deplodock compile … --ir cuda` — visual diff against
   the golden to confirm the pattern actually changed

## Risks and edge cases

- **Single-use invariant must be exact.** If `Assign.name` is consumed twice — once by the
  Accum and once by some Write/Store — fusion would duplicate the multiply. The single-use
  check is non-negotiable. Add a body-wide reference count, not just an adjacency check.
- **Operand aliasing across the gap.** The pass tolerates other stmts between the Assign and
  the Accum as long as they neither read `Assign.name` nor write to `Assign.args` (the two
  source operands). For matmul this is the common case after `095_interleave_loads.py` sinks
  one Assign past two unrelated Assigns. Track operand defs through the gap and bail if any
  Assign in the gap redefines `a` or `b`.
- **Dtype conversions.** If `Accum.dtype != Assign.dtype` (e.g. fp16 multiply into fp32
  accumulator), the renderer must thread the existing `convert(value, value_dt, acc_dt)`
  logic over both factors. Mirror the current `value` conversion path for `factor[0]` and
  `factor[1]` separately; both factors share `Assign.dtype` so it's one conversion to apply
  to both.
- **`Assign` with > 2 args.** `multiply` is binary; the fast path checks `len(args) == 2`.
  Bail otherwise (defensive; not currently emitted).
- **`Accum.op != "add"`.** FMA only exists for `mul + add`. `multiply`-accumulators (running
  product) don't fuse into anything useful. Skip.
- **Hot-path renderer cost.** The fusion is one extra pass over each Body once at lowering
  time, O(n) in stmt count, runs after `100_materialize_tile`. Negligible compile-time cost.
- **Tile-IR `Accum` round-trip.** `Accum` exists in `tile/ir.py` as well as `kernel/ir.py`
  (both wrap the same dataclass). Adding a new field there shouldn't affect Tile-IR passes
  since the field is `None` by default until the new pass runs.

## Out-of-scope (separate follow-ups)

- **Inlining beyond mul→add.** A general `assign + use` inliner would benefit other patterns
  (Loads that feed exactly one Op). Not in this plan; the FMA fusion is the proven win.
- **Three-operand FFMA from `acc = a*b + c`.** ptxas already generates this when the IR
  carries it explicitly; the matmul cell doesn't have a `+c` term per FFMA — the `+ c` is the
  accumulator's prior value, which is exactly what FMA's third operand is.
- **PTX-level `.reuse` hints.** Out of scope. The remaining 2 pp gap to cuBLAS after this
  fix is SASS-level operand-reuse density. Closing it needs inline PTX or a SASS post-pass
  (`project_tma_perf_findings.md`, `project_cublas_gap_2026-05-30.md`).
- **CTA clusters + DSMEM / persistent CTA + Stream-K / triple-buffer at BK=16.** All listed
  in `plans/matmul-cublas-gap-2026-05-30.md` § "Five new optimization kinds to feature". Each
  is its own multi-day plan.

## Expected outcome

- 0.6 pp improvement on 2048³ fp32 matmul (97.4 → 98.0 % of cuBLAS in standalone harness;
  ~96 → ~96.6 % under `deplodock --bench` methodology)
- Equal or better on every other shape that lowers through the same matmul cell pattern
- One new pass (~80 lines), one optional field on `Accum`, no knob, no behavior change for
  non-matmul reductions
- Removes the entire `vN = aN * bN;` intermediate cluster from the rendered CUDA — the inner
  loop body shrinks by ~50 % in line count
