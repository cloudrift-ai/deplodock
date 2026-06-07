# Golden-set tune / bench sweep — report (2026-06-07)

End-to-end sweep over the 23 `GOLDEN_CONFIGS` matmul shapes: for each shape `tune --golden NAME` then
`run --bench --golden NAME` (after-tune bench), then a final `run --bench --golden NAME` for every shape once all are
tuned. The first config is tuned with `--clean` so the learned prior is rebuilt from scratch over the sweep. Driver +
analyzer live under `/tmp/golden_sweep/` (`driver.py`, `analyze.py`).

Two comparison bases are reported:

- **after-tune** — each shape's greedy pick right after *its own* tune (per-shape freshly-fit prior). Shows what the
  tuner can find for a shape in isolation.
- **final** — the deployable greedy pick after the whole set is tuned (the number a user actually gets from
  `run` / `compile`). This is the one that matters for deployment.

## Headline

The codegen fix + regenerated goldens + heuristic adjustment moved the sweep from "mostly broken" to "half at-or-better,
no crashes":

| run | crashes | BETTER | SAME-DIFF | SAME-MATCH | WORSE | regressions |
|-----|--------:|-------:|----------:|-----------:|------:|------------:|
| before codegen fix      | 2 | 0 | 3 | 0 | 18 | 15 |
| after fix (prev sweep)  | 0 | 0 | 2 | 0 | 21 | 16 |
| **this sweep (final)**  | **0** | **2** | **6** | **1** | **14** | **12** |

(categories: BETTER = greedy >3% faster than golden; SAME-DIFF = within 3% but different knobs; SAME-MATCH =
reproduces a golden; WORSE = >3% slower / couldn't reach golden; regressions = final >3% slower than after-tune.)

## Fix that unblocked this: duplicate thread-axis codegen bug

`rename_ssa_sequential` canonicalized axis names from every binding scope **except**
`WarpSpecialize.consumer_thread_axes`. After `085_warp_specialize` removes the consumer `ThreadTile`, those thread
coords live only as free Vars in the consumer body + that off-body field. An unstaged coord (e.g. the N-thread index of
a matmul whose B operand isn't TMA-staged, so it's absent from every StageBundle cache axis) kept an uncontrolled name
and could collide with a renamed cache axis — emitting two `int a3` in one scope → `nvcc` error
(`"a3" has already been declared`). The learned-prior greedy was selecting a warp-specialized fp32 config for
`square.2048` / `square.4096`, so both failed to compile.

Fix (commit `9d2853ed`): record `consumer_thread_axes` in the rename pass so each thread coord claims its own canonical
slot. Regression test: `tests/compiler/ir/stmt/test_rename_ws_thread_axes.py` (reproduces the `['a3','a3']` collapse
pre-fix). Full compiler suite green.

## This sweep — final re-bench (deployable greedy vs golden)

**BETTER (2)** — genuine deployable wins, ~20% faster than the recorded golden:

| shape | greedy | golden | gain | knobs |
|-------|-------:|-------:|-----:|-------|
| `square.1024`           | 55.3 µs | 68.9 | +19.7% | `BM=8 BN=32 BK=64 FM=8 FN=4 FK=1 SPLITK=1 RING=2 STAGE=11` |
| `qwen3_06b.q_proj.s512` | 55.4 µs | 70.0 | +20.9% | (same) |

**SAME-DIFF (6)** — at parity, prior consistently lands on a uniform
`BM=8 BN=32 BK=64 FM=4 FN=2 SPLITK=2 RING=2` config: `gate_up_proj.s32`, `kv_proj.s128`, `gate_up_proj.s128`,
`kv_proj.s512`, `o_proj.s512`, `down_proj.s512`. **SAME-MATCH (1):** `square.512` reproduces its golden.

**WORSE (14)** — the dominant remaining gap is **fp16 tensor-core selection**: the prior picks a ~2–3× slower config for
every fp16 square:

| shape | greedy | golden | slowdown |
|-------|-------:|-------:|---------:|
| `square.512.fp16`  | 17.9 µs | 6.1   | **+193%** |
| `square.1024.fp16` | 31.5 µs | 18.5  | +70% |
| `square.2048.fp16` | 131.8 µs | 107.7 | +22% |
| `square.4096.fp16` | 840.4 µs | 738.8 | +14% |

Other WORSE: fp32 `square.2048` (+38%) / `square.4096` (+22%), and several qwen `s32`/`s128` shapes (+24–56%).

**Regressions (12)** — final vs after-tune drift is now mild (mostly +3–46%, vs +50–190% in the previous sweep), e.g.
`square.512` 12.3→14.3, `down_proj.s128` 26.9→39.2. The catastrophic prior collapse from earlier sweeps is gone.

## Golden updates applied

The 2 BETTER configs were recorded (commit replaces `square.1024` and `qwen3_06b.q_proj.s512` in `golden.py`):
deplodock 69.2/69.6 → **55.3/55.4 µs** at ratio 0.96 (now golden). The `cublas_us` reference is reused (config-
independent torch SGEMM). The 6 SAME-DIFF variants were **not** added — they're parity-only and the warm-prior pick that
produces them isn't reproducible by the isolated `find_golden_configs.py` search (see below), so they'd be churn.

### Note on the regenerator

`scripts/find_golden_configs.py` was repaired (commit `28d6dcee`): it pointed at a nonexistent `golden_configs.py`,
dumped the `S_`/`H_` feature columns into recorded knobs, and clobbered the set. It now targets `golden.py`, strips
feature + control knobs, and merges (better→replace, parity+diff→add, worse→keep) so a weak tune can't degrade a golden.

**Important limitation:** its isolated per-shape search (cold DB, `-O1` ranking) consistently finds **worse** configs
than the warm-prior greedy pick — e.g. for `square.1024` it lands on ~117 µs while the deployed greedy is 55 µs. So the
authoritative regen reproduces almost none of the sweep's wins; the wins only surface through the warm-DB greedy path.
That gap (isolated search ≪ warm greedy) is itself a tuner-quality signal worth chasing.

## Next targets (highest leverage first)

1. **fp16 tensor-core mis-pick** — the prior avoids the fast `mma_m16n8k16_f16` config on every fp16 square (up to
   +193%). Investigate the warp-tier scoring / enumeration for fp16.
2. **Isolated-search vs warm-greedy gap** — `find_golden_configs.py`'s cold `-O1` search can't reach configs the warm
   greedy deploys. Closing this would let the authoritative regen capture wins directly.
3. **fp32 large squares** (`2048`/`4096`, +20–38%) and several qwen `s32`/`s128` shapes remain off golden.
