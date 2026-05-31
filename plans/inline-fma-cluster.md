# Inline-PTX FMA cluster — close the `.reuse` density gap to cuBLAS

**Branch:** `feature/inline-fma-cluster`
**Origin:** `plans/matmul-cublas-gap-2026-05-30.md` § "What's actually left on the table",
discussions 2026-05-30 on `.reuse` mechanism, Triton's SASS-friendly PTX emission, and
where to put cluster assembly in the pass pipeline.
**Depends on:** `plans/fuse-mul-into-accum.md` (V3) is recommended but not required — the
assembly pass recognizes both the fused `Accum(factor=(a, b))` form and the legacy
two-step `Assign + Accum`. V3 makes the assembly cleaner.
**Effort:** ~2 weeks
**Risk:** medium — inline PTX is brittle and per-arch sensitive. Mitigated by the
three-pass architecture: the assembly is at tile-IR (where matmul structure is still
explicit), the emitter is at the bottom and operates on a structured op rather than a
flat pattern match.

## Status — 2026-05-31: implemented M1–M3, M4 gate FAILED, parked

Shipped on `feature/inline-fma-cluster` (commits M1 `cc31557b`, M2 `4ea860a5`, M3 `e29aa525`):

- **M1** — `FmaCluster` kernel-IR node (the tile-IR `FmaClusterTile` was dropped; see below).
- **M2** — assembly moved to the **kernel-IR phase** (`lowering/kernel/120_assemble_fma_clusters`,
  post-materialize). Operating on the already-flattened per-thread cell let us collapse to a single
  node and made the round-trip trivial — better than the planned tile/099 placement, which would
  have had to coordinate SSA names with `split_register_axes`. Detector is conservative (clean A×B
  outer product over f32 buffers only; masked / fp16 / one-operand-global cells skip).
- **M3** — `FmaCluster.render` emits the operand `Load`s as plain C + one `asm volatile` of
  `fm*fn` `fma.rn.f32` in `B_INNER` order (FFMA-only-asm variant; loads stay C). Accuracy verified:
  512³ `max_diff=1.1e-4`, 2048³ (golden 26×4 cell, 256 clusters) `6.4e-4`, both PASS vs eager.

- **M4 — SASS `.reuse` gate: FAILED.** On sm_120 (RTX 5090), 2048³ fp32 (the golden 26×4 B_INNER
  cell, which fires natively — fp32 SGEMM is FFMA on this card, *not* HMMA):
  - `.reuse`/FFMA: **0.762 (off) → 0.777 (on)** — gate needs ≥ 0.95; the 26×4 shape predicts 0.96.
  - 2048³ wall-clock: **279 µs on vs 279 µs off** (eager/cuBLAS 269 µs); both 0.96× cuBLAS. **No gain.**
  - Cause: ptxas does not honor the inline-asm operand-port ordering — it reallocates registers and
    commutes the FFMA `a`/`b` operands (multiply commutes), so our Rb pinning is gone before the
    post-scheduling `.reuse` peephole runs. The plan's heavier fallbacks (loads-in-asm, `+&f`,
    explicit `%rN`) don't fix the root cause; the only real fix (hand-placed `.reuse` via a SASS
    assembler) is blocked on absent sm_120 tooling (`project_tma_perf_findings`).

**Decision:** knob `FMA_CLUSTER` flipped to **off by default** (opt in `DEPLODOCK_FMA_CLUSTER=1`). The
machinery is correct + accuracy-verified and kept behind the knob as an experiment / readability
switch. M5–M11 are moot until the gate passes. The larger SGEMM-vs-cuBLAS opportunity is the HMMA
per-thread address-generation gap (`project_cublas_gap_ncu_2026-05-31`), not fp32 FFMA `.reuse`.

## Problem

The remaining 4 % gap to cuBLAS at 2048³ fp32 is **register-file port pressure** in the FFMA
cluster, measured directly via the SASS `.reuse` modifier density:

| metric | cuBLAS | golden (ours) | gap |
|:---|---:|---:|---:|
| `.reuse` hints per FFMA | 1.12 | **0.76** | −32 % |
| `dispatch_stall` (warps/cycle) | 0.22 | **0.49** | +123 % |
| Executed IPC | 3.00 | **2.48** | −17 % |
| FMA pipe active | 62.9 % | 55.5 % | −7 pp |

`.reuse` is a SASS-only modifier (not exposed in PTX) that tells the operand collector to
keep a source register in a small bypass latch so the *next* instruction reading the same
register in the same source port can skip the register-file read. Each hit saves an RF port
cycle. With ~210 M FFMAs per 2048³ kernel, the missing 0.36 reuses-per-FFMA cost us ~75 M
extra RF reads — which translates almost exactly to the measured IPC delta.

The reason ptxas inserts only 0.76 reuses per FFMA instead of cuBLAS's 1.12:

- **Register allocator runs before `.reuse` peephole.** Allocation optimizes spills, not
  reuse opportunities. Two values that *could* share a port-aligned reuse pattern often
  land in different registers (or the same register but different ports).
- **Per-source-port matching.** `.reuse` fires only when the same value sits in the *same*
  source port (Ra, Rb, or Rc) on consecutive instructions. ptxas's allocator has no
  port-affinity constraint.
- **Conservative peephole.** ptxas's `.reuse` insertion is post-scheduling; any intervening
  LDS / ALU op evicts the latch and breaks the chain. The heuristic only marks `.reuse`
  when the *immediate* next instruction is guaranteed to hit.

## The intervention

Bypass ptxas's allocator decisions for the matmul cell by emitting the inner-loop FFMA
cluster as **inline PTX `asm volatile` blocks** with explicit operand constraints. The
operand ordering inside the asm string pins each source value to a specific port across the
whole cluster, so ptxas's `.reuse` peephole trivially fires.

This is the same mechanism Triton uses for its `tt.dot` lowering. We adopt it as a focused
matmul-specific pass rather than introducing a Triton-style layout system; the layout
framework pays off when many tensor-core paths share the discipline (HMMA / BMMA / WGMMA /
sparse), and we're not there yet.

### Why operand ordering controls `.reuse`

In PTX, the order of source operands in an FFMA encodes which physical port each lives in:

```ptx
fma.rn.f32 %dst, %a, %b, %c;
//                ^   ^   ^
//                Ra  Rb  Rc
```

If two consecutive FFMAs share an operand and it sits in the same port both times, ptxas's
peephole adds `.reuse` to the first instance and the second skips the RF read:

```ptx
fma.rn.f32 %0, %in_a0, %in_b0, %0;    // in_b0 in port Rb
fma.rn.f32 %1, %in_a1, %in_b0, %1;    // in_b0 still in port Rb → .reuse fires
fma.rn.f32 %2, %in_a2, %in_b0, %2;    // .reuse fires again
```

For the matmul cell, the outer-product structure is naturally reuse-friendly: each B operand
feeds `FM` consecutive FFMAs (one per A row) and each A operand feeds `FN` consecutive
FFMAs (one per B column). The peak achievable density is **`max(FM, FN)` / `(FM·FN)` =
1/min(FM, FN)`** reuses per FFMA — for our `FM=26, FN=4` tile that's 1/4 = 0.25 from a
single-axis traversal. cuBLAS gets 1.12 by alternating which operand is in the latch on
*every* FFMA — both Ra and Rb participate via deliberate operand reordering. We aim for
the same.

## Design — three-pass architecture with structured ops

The cluster is a **first-class IR node at two levels**: `FmaClusterTile` at tile-IR
(pre-threading, with axis-shape semantics) and `FmaCluster` at kernel-IR (post-threading,
with per-thread SSA names). Three passes compose:

```
Tile-IR passes (existing 010-090)
    │
    ▼
099_assemble_fma_clusters.py        ← NEW
    │   walk tile bodies; when a ReduceTile body matches the matmul outer-product
    │   shape, replace it with FmaClusterTile carrying (fm, fn, bk, smem indexes, …)
    ▼
100_materialize_tile.py             ← MODIFIED (small extension)
    │   - FmaClusterTile → per-thread FmaCluster (kernel-IR), fresh SSA names
    │   - everything else: existing unfolding behavior unchanged
    ▼
Kernel-IR passes (existing 110, 120-V3 fuse-mul-into-accum)
    │
    ▼
130_emit_inline_fma_cluster.py      ← NEW
    │   lower FmaCluster to a single inline-PTX asm block; the operand-ordering
    │   policy is consulted to maximize port-aligned reuse
    ▼
Render to CUDA
```

Three small changes, each with one responsibility. Detection is centralized in the
assembly pass (at the IR level where matmul structure is still explicit); lowering is
mechanical (no pattern matching); emission operates on a structured op rather than a flat
sequence of stmts.

### The two IR nodes

**`FmaClusterTile`** (tile-IR, pre-threading) lives in `deplodock/compiler/ir/tile/ir.py`:

```python
@dataclass(frozen=True)
class FmaClusterTile(TileStmt):
    """Outer-product FFMA cluster, lifted from a matmul-cell tile body.

    Threading dimensions (FM register × FN register × BK reduce) are still implicit —
    `100_materialize_tile.py` unfolds them per thread when lowering to `FmaCluster`.
    """
    fm:       int                    # output rows per thread (register-tile M)
    fn:       int                    # output cols per thread (register-tile N)
    bk:       int                    # K-reduce extent inside the cluster
    a_axis:   Axis                   # M dimension axis
    b_axis:   Axis                   # N dimension axis
    k_axis:   Axis                   # the reduce axis
    a_smem:   str                    # source smem buffer for A operands
    b_smem:   str                    # source smem buffer for B operands
    a_index:  IndexMap               # (m, k) → a_smem offset
    b_index:  IndexMap               # (k, n) → b_smem offset
    acc_base: str                    # output accumulator base name
    dtype:    DataType = F32
```

**`FmaCluster`** (kernel-IR, post-threading) lives in `deplodock/compiler/ir/kernel/ir.py`:

```python
@dataclass(frozen=True)
class FmaCluster(Stmt):
    """Per-thread FFMA cluster. Lowered by `130_emit_inline_fma_cluster.py` to
    a single inline-PTX asm block with port-aligned operand ordering.
    """
    a_names:   tuple[str, ...]       # FM A-operand SSA names (LDS dests)
    b_names:   tuple[str, ...]       # FN B-operand SSA names (LDS dests)
    acc_names: tuple[str, ...]       # FM*FN accumulator SSA names (in/out)
    a_addr:    Expr                  # smem address expression for A loads
    b_addr:    Expr                  # smem address expression for B loads
    a_vec:     int = 1               # LDS vector width for A (1/2/4)
    b_vec:     int = 1               # LDS vector width for B
    dtype:     DataType = F32
    policy:    str = "B_INNER"       # set by reorder pass or by the assembly pass
```

`acc_names` indexes as `acc_names[m * fn + n]` — the canonical row-major flatten of the
outer-product cell. Both nodes are frozen dataclasses per `feedback_stmt_hashable`. Their
`Stmt.render()` methods raise (they're lowered, not rendered) — emission is the dedicated
pass's job.

### Pass 1: `tile/099_assemble_fma_clusters.py`

Walks each `TileOp` body. For every `ReduceTile(k_axis, body=...)` whose body matches the
canonical matmul-cell shape, replace it with an `FmaClusterTile` carrying the extracted
parameters. Skip otherwise — non-matmul tile bodies pass through untouched.

The shape we recognize:

```
ReduceTile(k_axis) {
    body = [
        Load(a_smem, indexed_by=[m_axis, k_axis], → a_name)
        Load(b_smem, indexed_by=[k_axis, n_axis], → b_name)
        Assign(v = multiply(a_name, b_name))   # legacy two-step form, OR
        Accum(acc, += v)                        # post-V3 form: Accum(factor=(a_name, b_name))
    ]
}
```

— wrapped by register-tile axes `m_axis` (extent `FM`) and `n_axis` (extent `FN`). The
detector reads `FM, FN` from the surrounding `RegisterTile`s, extracts `BK` from
`k_axis.extent`, lifts the smem index maps from the `Load`s, and emits one
`FmaClusterTile` in place of the entire `ReduceTile` body.

This is much easier to match at tile-IR than at kernel-IR because the structure is
explicit: `ReduceTile` is one statement, `m_axis` and `n_axis` are named, the index maps
are first-class objects. No flat-sequence pattern matching, no order-sensitivity. Pseudo-code:

```python
def assemble_fma_clusters(op: TileOp) -> TileOp:
    def visit(body: Body) -> Body:
        new_stmts = []
        for stmt in body:
            if isinstance(stmt, ReduceTile) and _is_matmul_cell(stmt):
                fm, fn = _surrounding_register_extents(stmt)
                cluster = FmaClusterTile(
                    fm=fm, fn=fn, bk=stmt.axis.extent,
                    a_axis=_axis_by_role(stmt, "M"),
                    b_axis=_axis_by_role(stmt, "N"),
                    k_axis=stmt.axis,
                    a_smem=_load_buffer(stmt, "A"),
                    b_smem=_load_buffer(stmt, "B"),
                    a_index=_load_index(stmt, "A"),
                    b_index=_load_index(stmt, "B"),
                    acc_base=_accum_target(stmt),
                )
                new_stmts.append(cluster)
            elif hasattr(stmt, "body"):
                new_stmts.append(stmt.with_body(visit(stmt.body)))
            else:
                new_stmts.append(stmt)
        return Body(tuple(new_stmts))
    return TileOp(body=visit(op.body), **op.fields_minus_body)
```

`_is_matmul_cell` is the dataflow check — ~40 lines, all local to the `ReduceTile`. Both
the legacy `Assign + Accum` form and the post-V3 fused `Accum(factor=...)` form match.

### Pass 2: `tile/100_materialize_tile.py` (extension)

Materialize already walks tile statements and lowers each. The extension adds one case
for `FmaClusterTile`:

```python
def _lower_fma_cluster_tile(node: FmaClusterTile, ctx: ThreadContext) -> FmaCluster:
    """Lower the tile-level cluster to a per-thread cluster with fresh SSA names."""
    a_names = tuple(ctx.fresh(f"a{i}")   for i in range(node.fm))
    b_names = tuple(ctx.fresh(f"b{j}")   for j in range(node.fn))
    acc_names = tuple(ctx.fresh(f"acc{m * node.fn + n}")
                      for m in range(node.fm) for n in range(node.fn))
    return FmaCluster(
        a_names=a_names, b_names=b_names, acc_names=acc_names,
        a_addr=node.a_index.substitute(ctx.tid_subst),
        b_addr=node.b_index.substitute(ctx.tid_subst),
        a_vec=_vector_width(node.fm, node.dtype),
        b_vec=_vector_width(node.fn, node.dtype),
        dtype=node.dtype,
    )
```

One method, ~15 lines. Compare to materialize's current matmul-cell handling, which
unfolds the same body into 30 Loads + 104 Accums via the generic unfolding path. The
generic path is unchanged for non-cluster bodies; `FmaClusterTile` short-circuits into
this single method.

### Pass 3: `kernel/130_emit_inline_fma_cluster.py`

Lowers `FmaCluster` to a single inline-PTX `asm volatile` block. The emitter reads the
cluster fields directly — no pattern reconstruction:

- For each `a_names[m]`: emit an output constraint `"=&f"` and reference it as `%<idx>`
- For each `b_names[j]`: same
- For each `acc_names[m * fn + n]`: emit an in/out constraint `"+f"` and reference
- Group consecutive A operands into `ld.shared.v4.f32` (LDS.128) blocks per `a_vec`
- Group consecutive B operands similarly per `b_vec`
- Emit `fm * fn` FFMAs in the order chosen by `cluster.policy`

The emitter is mechanical because the cluster's structure is in its fields. Adding a new
policy (M3 below) is a code-only change in this pass — no IR change.

### Operand-ordering policy

The pass picks an emission order that maximizes consecutive same-port operand reuse. Two
viable policies:

**Policy `B_INNER`** (column-major over the (m, n) grid):

```
for n in 0..FN:
    for m in 0..FM:
        emit fma acc[m,n], a[m], b[n], acc[m,n]
```

`b[n]` stays in port Rb for `FM` consecutive FFMAs → `FM − 1` reuse hits per column.
Total reuses = `FN · (FM − 1)` = `FN·FM − FN` = `104 − 4` = 100, density `100/104 ≈ 0.96`.

**Policy `INTERLEAVED`** (alternate which port gets reuse — mimic cuBLAS):

```
for k_idx in 0..FM·FN:
    (m, n) = (...interleaved walk...)
    # arrange so b[n] reuses for runs of size FM, then a[m] reuses for runs of size FN
```

The interleaved walk chooses (m, n) sequences such that both port latches stay valid as
long as possible. Concretely, the cuBLAS pattern alternates strips of:

```
fma acc[m,n],   a[m],   b[n],   acc[m,n]   # Rb latch: b[n]
fma acc[m+1,n], a[m+1], b[n],   acc[m+1,n] # Rb still b[n] → .reuse
fma acc[m+1,n+1], a[m+1], b[n+1], acc[m+1,n+1] # Ra still a[m+1] → .reuse
fma acc[m,n+1], a[m],   b[n+1], acc[m,n+1] # Rb still b[n+1] → .reuse
                                            # zig-zag continues
```

A zig-zag walk over the (m, n) grid keeps **both** ports alive between consecutive FFMAs,
delivering density > 1.0 — matching cuBLAS's 1.12.

Start with **`B_INNER`** in M1 (simpler, gets ~96 % density vs cuBLAS's ~112 %); add
**`INTERLEAVED`** in M3 if M1's measured win falls short of the IPC ceiling.

### Vectorized loads

Group consecutive A-row loads into `ld.shared.v4.f32` (LDS.128) — same vectorization the C
emitter already does, but the inline form has explicit register constraints that prevent
ptxas's allocator from breaking the vector into scalars. With `FM=26` A loads, that's 6
LDS.128 instructions covering 24 floats plus one LDS.64 for the remaining 2. With `FN=4`
B loads, one LDS.128 covers them all.

### What the emitted asm block looks like

For the golden `FM=26, FN=4, BK=32` tile, one K-iter body becomes a single block:

```c
asm volatile(
    // ---- 26 A loads as 6×LDS.128 + 1×LDS.64 ----
    "ld.shared.v4.f32 {%0,  %1,  %2,  %3},  [%112];\n"
    "ld.shared.v4.f32 {%4,  %5,  %6,  %7},  [%112 + 16];\n"
    "ld.shared.v4.f32 {%8,  %9,  %10, %11}, [%112 + 32];\n"
    "ld.shared.v4.f32 {%12, %13, %14, %15}, [%112 + 48];\n"
    "ld.shared.v4.f32 {%16, %17, %18, %19}, [%112 + 64];\n"
    "ld.shared.v4.f32 {%20, %21, %22, %23}, [%112 + 80];\n"
    "ld.shared.v2.f32 {%24, %25},           [%112 + 96];\n"
    // ---- 4 B loads as 1×LDS.128 ----
    "ld.shared.v4.f32 {%26, %27, %28, %29}, [%113];\n"
    // ---- 104 FFMAs in B_INNER (or INTERLEAVED) order ----
    "fma.rn.f32 %30,  %0,  %26, %30;\n"
    "fma.rn.f32 %31,  %1,  %26, %31;\n"
    "fma.rn.f32 %32,  %2,  %26, %32;\n"
    // ...26 FFMAs sharing %26 in port Rb...
    "fma.rn.f32 %56,  %0,  %27, %56;\n"
    "fma.rn.f32 %57,  %1,  %27, %57;\n"
    // ...26 FFMAs sharing %27...
    // ...similar for %28, %29...
    "fma.rn.f32 %133, %25, %29, %133;\n"
    : // outputs (early-clobber LDS dests + in/out accumulators)
      "=&f"(a0),  "=&f"(a1),  /* ... */ "=&f"(a25),
      "=&f"(b0),  "=&f"(b1),  "=&f"(b2), "=&f"(b3),
      "+f"(acc0), "+f"(acc1), /* ... */ "+f"(acc103)
    : // inputs (smem addresses)
      "r"(a_smem_addr), "r"(b_smem_addr)
    : "memory"
);
```

ptxas allocates physical registers for each `%N` slot. The operand-ordering policy ensures
`%26` (= `b0`) sits in port Rb for all 26 FFMAs that use it, so `.reuse` fires on each. Same
for `%27`, `%28`, `%29`. Reuse density = 100/104 ≈ 0.96 with `B_INNER`.

### Pipeline placement summary

| pass | level | runs after | runs before |
|:---|:---|:---|:---|
| existing tile passes (010-090) | tile-IR | — | the new assembly pass |
| **`099_assemble_fma_clusters` (new)** | tile-IR | 090_mark_unroll | 100_materialize_tile |
| `100_materialize_tile` (extended) | tile→kernel | 099 | existing kernel passes |
| `110_drop_redundant_syncs` | kernel-IR | 100 | 120 |
| `120_fuse_mul_into_accum` (V3) | kernel-IR | 110 | 130 |
| **`130_emit_inline_fma_cluster` (new)** | kernel-IR | 120 | (render) |
| render to CUDA source | — | 130 | — |

The assembly pass runs late in the tile-IR pipeline (after staging, ring-buffering,
pipelining, and unroll-marking) so it sees the cluster in its final tile-level shape.
The emit pass runs last in the kernel-IR pipeline so it operates on the post-V3 fused
form when V3 is enabled.

### Knob: `FMA_CLUSTER` (default `True`) — for readability, not for perf

The pass is gated on the `FMA_CLUSTER` knob (default `True`). The knob exists permanently,
not as a temporary feature flag — its purpose is **readability**. When investigating a
kernel via `deplodock compile … --ir kernel` or `--ir cuda`, the inline-PTX `asm volatile`
block is opaque to a human reader: 30 LDS + 104 FFMAs collapsed into one PTX string that
takes a SASS dump to understand. Setting `DEPLODOCK_KNOBS=FMA_CLUSTER=0` keeps the flat
`Load + Accum` form in the IR all the way to render, producing the C body that's already
familiar from the article. The optimized inline-PTX form is what ships in production runs;
the readable form is one knob flip away for debugging or article material.

Concretely:

- `FMA_CLUSTER=1` (default): `099_assemble_fma_clusters.py` rewrites; `materialize_tile`
  lowers `FmaClusterTile → FmaCluster`; `130_emit_inline_fma_cluster.py` emits inline PTX.
- `FMA_CLUSTER=0`: `099_assemble_fma_clusters.py` skips entirely; the cluster goes through
  the existing generic unfolding path; render produces the same flat C body as today's
  golden kernel.

This dual-mode design also makes the pass easier to ship: the off path is the existing
behavior (zero risk), the on path is the new behavior (gated by the knob). Bisection
against any regression is one env-var flip.

## Milestones (single branch, commit after each `make test` passes)

Per `feedback_single_branch_milestones`. The three-pass architecture lets us ship the
infrastructure before the lowering, so each milestone is independently testable.

**M1 — IR nodes (no behavior change)**

- Add `FmaClusterTile` to `deplodock/compiler/ir/tile/ir.py`. Frozen dataclass per
  `feedback_stmt_hashable`.
- Add `FmaCluster` to `deplodock/compiler/ir/kernel/ir.py`. Frozen dataclass.
- Both `.render()` raise `NotImplementedError` for now — they're lowered before reaching
  render in later milestones.
- IR tests: round-trip through JSON dump/load, hashing, equality.
- **Success criterion**: `make test` green. Existing kernels unaffected (the new nodes
  are never constructed by any current pass).

**M2 — Assembly pass `tile/099_assemble_fma_clusters.py`**

- Implement the matmul-cell detector and rewrite. Recognizes both legacy
  `Assign(multiply) + Accum(add)` and post-V3 fused `Accum(factor=...)` forms.
- Knob `FMA_CLUSTER = Knob("FMA_CLUSTER", KnobType.BOOL, default=True)`.
  **The knob is permanent, not a temporary feature flag** — its purpose is
  *readability*: when investigating a kernel via `deplodock compile … --ir cuda` or
  `--ir kernel`, setting `FMA_CLUSTER=0` keeps the flat `Load + Accum` form so the
  body reads as straightforward C. The optimized inline-PTX form is what ships in
  production runs (the default), but the readable form is one knob flip away.
- During M2 itself the knob still gates the actual rewrite — M2 ships with the
  round-trip placeholder lowering, so even with `FMA_CLUSTER=1` the kernel runs
  identically (assembly + round-trip = no-op). The default of `True` becomes
  *effective* in M3 when the real lowering ships.
- Materialize_tile still needs to handle `FmaClusterTile` — but for M2 we only need a
  *placeholder* lowering that re-expands it back to the original `Loads + Accums` form
  (round-trip). This lets M2 ship independently of M3.
- Unit tests:
  - Canonical matmul cell → `FmaClusterTile` with correct `(fm, fn, bk)`
  - Pre-V3 form → matches via `Assign + Accum` shape
  - Post-V3 form → matches via `Accum(factor=...)` shape
  - Non-matmul tile body → no rewrite
  - SDPA-prologue body → no rewrite (different load index structure)
  - The round-trip placeholder lowering preserves accuracy on TinyLlama block
- **Success criterion**: `FMA_CLUSTER=1` compiles end-to-end via the round-trip
  placeholder; accuracy and perf unchanged from `FMA_CLUSTER=0`.

**M3 — Real materialize extension + inline-PTX emitter `kernel/130_emit_inline_fma_cluster.py`**

- Replace the round-trip placeholder in `materialize_tile.py` with the real
  `FmaClusterTile → FmaCluster` lowering (~15-line method described in Design).
- Implement `130_emit_inline_fma_cluster.py`. Emits one `asm volatile` block per
  `FmaCluster` with the `B_INNER` operand-ordering policy.
- Unit tests:
  - Render produces compilable inline-PTX asm
  - For a 26×4 cluster: the asm contains exactly 6 `ld.shared.v4.f32` for A, 1 for B,
    and 104 `fma.rn.f32` in B-inner order
  - Accumulator constraints are `+f` (in/out), operand constraints are `=&f`
    (early-clobber output)
- **Success criterion**: kernel compiles with `FMA_CLUSTER=1`, accuracy unchanged on
  TinyLlama block and 2048³ matmul.

**M4 — SASS verification**

- Compile the golden 2048³ tile with `FMA_CLUSTER=0` and `=1`. Dump SASS via
  `cuobjdump --dump-sass`. Count `.reuse` annotations on FFMA operands in both. Expect:
  - Off: ~0.76 reuses/FFMA (matches existing measurement)
  - On (`B_INNER`): ≥ 0.95 reuses/FFMA
- If density doesn't move, ptxas's allocator isn't honoring the operand discipline
  (`%N` slots got reshuffled). Inspect the SASS to confirm `%N` → physical-register
  mapping; tighten constraints if needed (e.g. `+&f` over `+f`, or explicit register
  number references via `:r`).
- **Success criterion**: SASS shows ≥ 0.95 `.reuse`/FFMA at 2048³. This is the gate —
  without the SASS moving, no perf measurement makes sense.

**M5 — Perf measurement and decision on `INTERLEAVED` policy**

- Bench 2048³ with `FMA_CLUSTER=0` vs `=1` under the harness methodology from
  `plans/matmul-cublas-gap-2026-05-30.md` § "Hand-optimized kernel variants": 10 rounds
  × 1000 iters round-robin. Compare to cuBLAS.
- If `B_INNER` delivers ≥ 95 % of the predicted gain (≥ 4 % wall-clock, landing
  ≤ 265 µs ≈ 100 % of cuBLAS): ship `B_INNER`, skip `INTERLEAVED`.
- Otherwise: add the `INTERLEAVED` policy as a second branch in
  `130_emit_inline_fma_cluster.py` (one new `if policy == "INTERLEAVED"` block in the
  same emitter; no IR change). Re-measure. Pick the winner.
- Pick the winning policy as the default that `099_assemble_fma_clusters.py` stamps onto
  new clusters.
- **Success criterion**: best policy ≥ 99 % of cuBLAS at 2048³ in deplodock `--bench`
  methodology (≤ 268 µs).

**M6 — Multi-shape validation**

- Run on the shape matrix from `plans/persistent-cta-streamk.md` M7: 1024³, 2048³, 4096³,
  8192³, 16384³. Plus TinyLlama block and Qwen3-Embedding-0.6B `--layer 0`.
- Expect: 2048³ wins, larger shapes equal or marginally win (since they're already at
  similar IPC), no shape regresses. Smaller shapes (1024³ and below) may show smaller
  wins because per-CTA work is too small to amortize the FFMA cluster setup.
- **Success criterion**: no regression on any shape; ≥ 4 % win on 2048³ and 4096³.

**M7 — Per-arch validation**

- Compile and bench on three target architectures via `--target`:
  - `sm_80` (Ampere — original `.reuse` semantics)
  - `sm_90` (Hopper — different RF banking)
  - `sm_120` (Blackwell — current target)
- Verify accuracy on each (`--target sm_NN --bench` runs the kernel on the live GPU but
  with the target's codegen path).
- If any arch regresses, add an arch-specific operand-order template (mirror sm_120's,
  perturbed). The policy field on `FmaCluster` is the carrier — add per-arch policy
  selection in the assembly pass.
- **Success criterion**: no regression on any of the three architectures.

**M8 — Autotune integration**

- The knob is already default-`True` (M2). M8 confirms M3–M7 validated that default and
  no per-shape exception is needed.
- Add `FMA_CLUSTER` to the autotune fork search at canonical matmul-cell shapes. Should
  rarely fork off because the on-state never regresses, but the fork lets the tuner
  verify per shape and surface any pathological case.
- Document the knob in the user-facing docs as a **readability switch**, not a
  performance toggle: "Set `DEPLODOCK_KNOBS=FMA_CLUSTER=0` to disable the inline-PTX
  matmul cluster and emit the body as plain C — useful when stepping through the kernel
  to understand the FFMA structure."
- **Success criterion**: autotune on the standard shape matrix never picks
  `FMA_CLUSTER=0` (default holds in every measured shape).

**M9 — Combined Stream-K + inline-FMA measurement (if both shipped)**

- If `plans/persistent-cta-streamk.md` has also shipped, run the combined kernel on
  2048³ and confirm the orthogonal multiplication holds: golden 275 µs × 0.94 (Stream-K)
  × IPC ratio ≈ 230–245 µs target.
- **Success criterion**: combined kernel ≥ 105 % of cuBLAS at 2048³.

**M10 — Optional: extend assembly pass to SDPA score-times-value**

- The SDPA prologue has a similar outer-product FFMA cluster but with different load
  index structure (one operand is a per-row score vector, the other a value-tile slice).
- Extend `099_assemble_fma_clusters.py` to recognize the SDPA shape and emit
  `FmaClusterTile` (or a sibling `SdpaClusterTile` if the structure differs enough to
  warrant its own node — TBD by inspection).
- The lowering and emitter stay the same; only the detector grows.
- **Success criterion**: SDPA kernel in TinyLlama / Qwen layer benchmarks shows a
  measurable IPC improvement.

**M11 — Docs**

- Update `deplodock/compiler/ARCHITECTURE.md` (pass list at both tile-IR and kernel-IR
  levels) and any relevant child `ARCHITECTURE.md` files.
- Update `plans/matmul-cublas-gap-2026-05-30.md` § "What's actually left on the table"
  with the measured outcome.
- Add a new article section if the user wants: "Inline PTX: pinning operand-port
  assignment to match cuBLAS's `.reuse` density." Show one K-iter body diff before /
  after, plus the side-by-side SASS `.reuse`-density comparison.

## Validation checklist (per `CLAUDE.md` § "Before committing")

After every milestone:

1. `make test`
2. `make lint`
3. Update `ARCHITECTURE.md` in any directory touched

After M2, M3, M4, M5 specifically:

4. Capture SASS `.reuse` density (M2) or `--bench` latency (M3+) in the commit message.
5. Per `feedback_perf_eval_scope`: tight A/B per milestone, isolated DB, no full sweeps
   unless asked. M4 is the *one* shape sweep.
6. Re-check `--ir cuda` after M1 — confirm the inline asm block is what we expected.

## Risks and edge cases

- **ptxas ignores our register-port discipline.** ptxas's allocator decides which physical
  register goes into each `%N` slot. If it picks an allocation that doesn't preserve port
  affinity across FFMAs, the `.reuse` peephole won't fire and the pass delivers zero gain.
  Mitigation: M2 directly inspects SASS density. If it doesn't move, either constrain
  registers more tightly (e.g. via the `:r` constraint with explicit register numbers via
  `%r10`-style references — fragile) or accept the limitation.
- **`asm volatile` blows out compile time.** The 30 + 104 = 134-instruction block × 32-iter
  inner-loop unroll = 4288 PTX instructions per asm block per K-chunk. With 64 K-chunks the
  emitted PTX is large. ptxas compile-time may rise from ~1 s to ~10 s per kernel.
  Mitigation: emit one asm block per K-iter, not per K-chunk; let ptxas handle the K-iter
  loop. This keeps blocks at 134 instructions each.
- **Inline asm prevents some ptxas optimizations.** Specifically, ptxas can't sink loads
  across asm boundaries. The pipelining pass (`080_pipeline_stages.py`) is at tile-IR level
  and already handles cross-K-chunk pipelining, so this is OK for the K-chunk-outer level.
  But within the inner K-iter, the asm block forces all 30 LDS to issue at the top of the
  block — which is exactly what we want for reuse density, but eliminates any opportunity
  for ptxas to schedule them later.
- **Per-architecture register banking.** sm_80 vs sm_90 vs sm_120 have slightly different
  register file bank layouts and reuse cache capacities. The operand-ordering policy may
  need per-arch variants. M5 validates this.
- **Per-tile-shape templates.** The asm block is specific to `(FM, FN)`. Each shape needs
  a different template (different operand counts, different orderings). Mitigation: the pass
  builds the asm string programmatically from `FM` and `FN`, not from hardcoded templates.
  One generator handles the whole knob space.
- **Debugging inline PTX.** When something goes wrong (wrong accumulator clobbered, wrong
  register width), the error surfaces deep in ptxas's diagnostics with line numbers in the
  asm block, not the Python pass. Mitigation: emit a `// pass-generated, do not edit`
  comment with the source op IDs so dumps stay traceable.
- **Numerical drift.** Inline PTX should be numerically identical to the C-emitted FFMA
  (both are `fma.rn.f32`). Verify on TinyLlama block test which catches accumulation order
  drift in fp16/fp32 mixed paths.
- **Conflict with V3 fuse-mul-into-accum.** If V3 hasn't shipped yet, this pass operates on
  the unfused `Assign + Accum` pattern, which has 2× more SSA names per cell. Detection
  logic and asm emission must handle both forms. Simplest: gate this pass on V3 having
  shipped (the dependency declared above).
- **Tile shapes outside the matmul cell.** SDPA-prologue and chunked-reduce kernels also
  have FFMA-heavy bodies but with different structures (e.g., softmax × value has a
  per-row reduce, not an outer product). The pattern matcher rejects these (no canonical
  cell). They get no benefit but no regression either.

## Out-of-scope (separate follow-ups)

- **Inline PTX for non-matmul kernels.** SDPA's score-times-value has its own reuse pattern
  (each value vector multiplied by a row of scores). Generalizing the inline-PTX approach
  requires a different pattern matcher and a different ordering policy. Defer until matmul
  ships and is measured.
- **Full layout system.** A Triton-style `#dot_op` / `#fma` layout framework with
  explicit thread-to-register mapping would generalize this pass to every tensor-core path
  (HMMA, BMMA, WGMMA, sparse, FP8). Roughly a quarter of refactor work. Defer until we have
  ≥ 3 paths that would benefit.
- **Hand-tuned cross-cluster reuse.** cuBLAS's 1.12 reuses/FFMA exceeds what a
  single-cluster policy can deliver (capped at `1.0` with single-port reuse, ~`1.12` with
  the zig-zag `INTERLEAVED` walk). To go past `1.0` requires reusing operands across the
  cluster boundary (last FFMA of one K-iter → first FFMA of the next). Possible but
  complicates pipelining; skip until measured ceiling justifies it.
- **PTX-level `.reuse` annotations.** Not exposed in PTX (per the
  2026-05-30 discussion). No path here.
- **Cubin post-processing / SASS assembler.** Would let us write SASS directly with
  hand-placed `.reuse`. Blocked on absence of sm_120 SASS assembler tooling per
  `project_tma_perf_findings.md`. If/when CuAssembler or similar gains sm_120 support, this
  becomes the most direct way to close the remaining gap to cuBLAS's 1.12 density.

## Expected outcome

Combining ncu-measured baselines and the `B_INNER` policy's predicted density:

| metric | golden (today) | this pass (predicted) | cuBLAS |
|:---|---:|---:|---:|
| `.reuse` / FFMA | 0.76 | 0.95–1.00 | 1.12 |
| `dispatch_stall` | 0.49 | ~0.30 | 0.22 |
| IPC | 2.48 | ~2.85 | 3.00 |
| FMA pipe active | 55.5 % | ~59 % | 62.9 % |
| 2048³ latency | 275 µs | ~245 µs | 265 µs |
| % of cuBLAS time | 96 % | **~108 %** | 100 % |

If `INTERLEAVED` ships (M3), density could reach 1.05–1.10 and latency another 2–3 % lower
(~238 µs ≈ 111 % of cuBLAS time, i.e. 11 % faster than cuBLAS).

**Combined with Stream-K** (orthogonal multiplication, per the 2026-05-30 discussion):

```
T_base                                                = 275 µs
T_stream_k        (no tail)                          = 259 µs   (-5.9 %)
T_inline_fma     (.reuse density 0.76 → 0.96)       = 245 µs   (-10.9 %)
T_both           (compound)                          = 231 µs   = 87 % of cuBLAS
                                                              = ~15 % faster than cuBLAS
```

The 231 µs ceiling lines up with what tuned Triton kernels achieve at similar shapes —
within a percent or two of the theoretical FFMA-pipe roofline.

The honest hedge: ptxas may not honor the operand discipline as cleanly as we hope (M2 is
the gate), and the K-cluster's compile-time and SASS-size impact may force template
simplifications. Realistic delivery is more likely **5–8 % wall-clock improvement** on
2048³ standalone, **12–15 %** combined with Stream-K — landing at 245 / 233 µs respectively,
both crossing 100 % of cuBLAS.

This is the lever that takes deplodock from "competitive with cuBLAS" to **"faster than
cuBLAS"** on SGEMM. The Stream-K plan handles wave-fill; this plan handles per-cycle issue
density. Together they close the last identified bottlenecks in the ncu profile and there
is no measured stall row left that's significantly above cuBLAS's.
