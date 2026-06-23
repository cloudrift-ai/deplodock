# Tile lowering — enumeration + assembly over the block-DAG IR

The tile phase lowers each fused `LoopOp` to a kernel-ready `TileOp` in **two passes** over the block-DAG Tile IR
(`ir/tile/ir.py`), following `plans/tile-ir-block-dag.md`:

- **`enumeration/`** (`010_enumerate`) — `LoopOp` → a generative `Fork` tree whose leaves are `TileGraphOp`s (a chosen
  `Schedule`'s `TileGraph`). This is the **search**: every variant is a point in the move/schedule space.
- **`assembly/`** (`010_assemble`) — `TileGraphOp` → `TileOp` (the materialized tower the `lowering/kernel` passes
  lower to `KernelOp`). This is **one deterministic step** — every scheduling decision already lives on the
  `Schedule`, so there is no search here.

```
LoopOp ──010_enumerate──▶ Fork tree of TileGraphOp ──(search resolves a leaf)──▶ TileGraphOp ──010_assemble──▶ TileOp
```

## The block-DAG Tile IR (`ir/tile/ir.py`)

One file, two comment-block sections:

- **ENUMERATION** — the *invariant algorithm* + the *variant* `Schedule` the composer searches: `Block`
  (`name + domain + compute`), `Buffer`, `Edge`, `Schedule`, `TileGraph`. The derived projections
  (`Block.reads`/`writes`/`carrier`/`atom`, `TileGraph.edges`) are **computed on demand, never stored** — the same
  discipline as `Loop.algebra_kind`, so they can't drift and don't enter `op_cache_key`. `TileGraphOp` wraps a chosen
  `TileGraph` as a graph node (the enumeration-pass output; `op_cache_key` keys it on `TileGraph.structural_key` +
  knobs).
- **MATERIALIZED** — what `assemble` emits: `TileOp` + the typed tile flavors (`GridTile` / `ThreadTile` /
  `RegisterTile` / `WarpTile` / `AtomTile` / `SerialTile`) + `StageBundle` / `Source` / `WarpSpecialize` / `AsyncWait` /
  `Atom`. (Slated for removal once `assemble` emits `KernelOp` directly.)

## INVARIANT: a purely algebraic moveset, no specializations

**The composer never dispatches on a named shape** (matmul / pointwise / attention / RMSNorm / gated-MLP). It
dispatches on the reduce axes' **carrier algebra** (`ir/algebra.py::AlgebraKind`) — `MAP` / `SEMIRING` / `MONOID` /
`TWISTED_MONOID` — read off the body by `_tree.classify`. A "matmul" is just a `SEMIRING`; "pointwise" a `MAP`;
"RMSNorm" / "softmax" a `MONOID`; "flash attention" a `TWISTED_MONOID`. Adding a model architecture is **never** a new
branch — it is the same move set on its algebra.

Every move is gated by a **carrier trait**, not a shape (`_moves.legal_decomps`):

- **`tile_axis`** on a free (`PARALLEL` / `MAP`) axis — split it into GRID / THREAD / REGISTER. Always legal (no
  carrier, nothing to recombine).
- **the reduce decomposition** on a contraction axis — `associative` → split at all; `commutative` → a partition
  factor (split-K across CTAs / cooperative across threads); `has_identity` → mask a non-divisible / symbolic axis
  with the carrier identity. The recombine is derived (`carrier.combine_partials`); the hardware realization
  (atomic / shuffle / tree / mma) is a downstream cost choice keyed off placement.

The only algebra-*conditioned* heuristic is a **ranking** cost model, not a code path: the free-axis register menu is
bandwidth-biased for a `MAP` nest (`map_reg_offers`) and compute/ILP-biased for a reduce regime (`reduce_reg_offers`).
This is the tile-phase instance of the global rule in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — *No
shape-specific pattern matching.*

## `enumeration/` — the move composer

| Module | Role |
| ------ | ---- |
| `010_enumerate.py` | The pass: `iter_dag` → `build_partition` → a `Fork` tree (or a single `TileGraphOp`). |
| `_iterdag.py`      | `iter_dag` — the derived iteration-DAG view (axes tagged `PARALLEL` / `REDUCE` + carrier). |
| `_moves.py`        | `Budget` + `legal_decomps` + the offers (`thread_offers`, `map_reg_offers`, `reduce_offers`, `reduce_reg_offers`) + knob deltas. |
| `_knobs.py`        | The knob schema (`BN`/`BM`/`BK`/`FK`/… + the composer aliases `MAP_*` / `RED_*` / `TC_*`). |
| `_build.py`        | `build_dag` (free-axis tile + optional reduce re-bracket → `TileGraph`) + `lower` (→ `TileGraphOp`). |
| `_tree.py`         | The one uniform `Fork` tree; `classify` → `_Regime(algebra=AlgebraKind)`; `build_partition` dispatches on the `AlgebraKind`. |

`build_partition` runs the SAME tree for every regime: a `MAP` nest enters at the free-axis root (thread → register);
a `SEMIRING` reduce enters at the reduce root (reduce → thread → register). One `build_dag` serves both — the reduce
re-bracket only fires when `target_names` (the contraction axes) is non-empty.

## `assembly/` — `assemble`

| Module | Role |
| ------ | ---- |
| `010_assemble.py` | The pass: `TileGraphOp` → `TileOp` via `assemble_block`. |
| `_assemble.py`    | `assemble_block` — reconstruct the binding tower from `Schedule.binding` (ATOM/REGISTER/WARP/THREAD/GRID tiers) over the block's σ-rewritten `compute`. |
| `_tower.py`       | `_wrap_tower` — the shared innermost-first tower-building primitive (`Role` → tile flavor). |

## Coverage

Built today: **`MAP`** + scalar **`SEMIRING`** (including masked / symbolic free axes, split-K, and the `FK`
strip-mine). Recognized by `classify` but not yet built (they re-enter the same uniform tree once their decomposition
placement + assembly land): `MONOID` cooperative-reduce, `TWISTED_MONOID` fused flash, and the `SEMIRING` warp-tier
(tensor-core MMA). See `plans/tile-ir-block-dag.md` and `plans/algebra-licensed-decomposition-moves.md`.
