# Golden-set sweep — finding (2026-06-07, post fp16 warp-tier work)

Re-ran the full tune/bench sweep over the 23 `GOLDEN_CONFIGS` shapes after the fp16 changes
(`8f0bcad9 fp16: emit WARPSPEC=True first for the warp tier; add opt-in ε-greedy explore` + dead-scorer removals).
Driver + analyzer under `/tmp/golden_sweep/` (`driver.py`, `analyze.py`). Comparison = final deployable greedy pick
(prior-free option-0, what `run` / `compile` emit) vs the recorded golden, benched live the same run.

**Verdict: the fp16 fix worked, but it introduced a single fp32 emission-order regression that now dominates the
counts.**

| basis | crashes | BETTER | SAME-DIFF | SAME-MATCH | WORSE | regressions |
|-------|--------:|-------:|----------:|-----------:|------:|------------:|
| prev sweep (2026-06-07) | 0 | 2 | 6 | 1 | 14 | 12 |
| **this sweep**          | 0 | 0 | 3 | 0 | **20** | 15 |

## fp16 — fixed ✅ (was the #1 target, up to +193%)

3 of 4 fp16 squares now land on the fast warp-tier `mma_m16n8k16_f16` config (`WM=1 WN=4`) at parity:

| shape | before | now | golden |
|-------|-------:|----:|-------:|
| `square.512.fp16`  | 17.9 µs (+193%) | **6.1 (parity)**     | 6.1 |
| `square.1024.fp16` | 31.5 (+70%)     | **16.4 (exact match)** | 16.4 |
| `square.4096.fp16` | 840 (+14%)      | **662 (−1.5%)**      | 652 |
| `square.2048.fp16` | 131.8 (+22%)    | 114.2 (−22.7%)       | 93.1 |

The greedy now emits `WM=1 WN=4 MMA=mma_m16n8k16_f16` first — exactly the golden warp-tier config. Only `2048.fp16`
still lags (−22.7%).

## fp32 + qwen — regressed ❌ (one ordering bug)

Every fp32/qwen shape now emits **`STAGE=10 RING=3/4`** instead of the golden **`STAGE=11 RING=2`**. `STAGE=10` is the
non-pipelined staging, so the greedy pick runs 1.5–3× slower:

| shape | greedy | golden | delta |
|-------|-------:|-------:|------:|
| `square.512`            | `RING=4 STAGE=10` → 20.5 µs  | `RING=2 STAGE=11` → 12.3 | was SAME-MATCH → **WORSE** |
| `square.1024`           | `RING=4 STAGE=10` → 82.1     | `STAGE=11` → 49.7        | was BETTER → **WORSE** |
| `square.2048`           | `RING=4 STAGE=10` → 798.8    | `STAGE=11` → 271.4       | −38% → **−194%** |
| `qwen3_06b.o_proj.s512` | `RING=3 STAGE=10` → 155.2    | `STAGE=11` → 59.6        | **−160%** |

11/11 fp32 shapes checked show the same `STAGE=10`-first flip; only `square.4096` kept `STAGE=11` (−7%).

## Diagnosis

The fp16 warp-tier reordering (WS=1-first / enumeration changes) inadvertently flipped the **scalar fp32 STAGE/RING
emission order**, so the prior-free option-0 greedy emits `STAGE=10 RING=4` before `STAGE=11 RING=2`. It is a single
ordering regression, not a tuner-capability loss — the golden `STAGE=11` configs still exist; greedy just stopped
emitting them first. The worse summary counts mask a genuine fp16 win.

## Next step

Restore `STAGE=11 RING=2`-first for the thread-tier (scalar fp32) emission order **without** disturbing the fp16
warp-tier ordering, and add a regression guard so the two orderings can't trade off again.
