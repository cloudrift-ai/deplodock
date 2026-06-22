# Monoid-DAG carrier annotation — softmax/layernorm/Welford as composed monoids, not new regimes

**Branch:** `feature/move-composer` (companion to `plans/move-composer-axis-walk-scheduler.md`)
**Status:** brainstorming → planned. Enrich the algebraic annotation so a loop scope can carry a **DAG of same-axis
carriers + maps with dependency edges**, instead of a single `AlgebraKind` tag per reduce loop. The payoff: unfused
softmax / LayerNorm / RMSNorm / Welford become *compositions* of `MONOID` + `MAP` transforms over a shared axis —
schedulable by the existing per-axis cooperative-reduce + map transforms — and fused-vs-unfused (the `TWISTED_MONOID`
collapse) becomes a **search choice over one representation**, not a separate recognized regime.

## The framing insight

Today `AlgebraKind` (`compiler/ir/algebra.py`) tags each reduce loop with one kind: `MAP` / `MONOID` / `SEMIRING` /
`TWISTED_MONOID` / `SCAN`. Softmax does not fit any single tag — but it does not *need* the heavy twisted machinery
either. Decompose softmax over `K`:

- `m = max_k x_k` — a max-monoid reduce
- `s = Σ_k exp(x_k − m)` — an add-monoid reduce whose body reads `m`
- `y_k = exp(x_k − m) / s` — a map, reads `m` and `s`

So the **unfused** form is two same-axis `MONOID` reduces with a read-after-reduce edge between them, plus a `MAP`
epilogue. Every piece is a monoid or a map; the whole is a **monoid-DAG over one axis** — not a single monoid, and not
a twisted monoid. The `TWISTED_MONOID` kind only appears when you insist on doing it in **one pass**.

Push further: the `max` pass is not algebraically necessary — it is an fp-stability guard. Ignoring overflow, softmax
is literally **one add-monoid (Σ exp) + one map (divide)**. The max-subtraction is a rescale bolted on for safety.
Flash / online softmax is precisely the trick of carrying that rescale *inside* the fold to keep stability without a
second pass — and that rescale is the "twist" (the `action` in the `(init, action, fold)` carrier interface sketched in
`plans/algebraic-carrier-analysis.md` C1b–C5).

Three realizations of the **same algebraic content**:

| realization     | algebra                       | passes over K | cost                          |
|-----------------|-------------------------------|---------------|-------------------------------|
| unsafe          | 1 MONOID + MAP                | 1             | overflows in fp16             |
| safe, unfused   | 2 MONOID (max, sum) + MAP     | 3             | reads x 3× (or holds it live) |
| safe, fused     | 1 TWISTED_MONOID + MAP        | 1             | the `action` rescale          |

**The punchline:** a twisted monoid is a monoid-DAG collapsed into one pass via a correction. `action = identity` ⇒
plain monoid; `action = rescale` ⇒ the twist. Welford (mean → M2), argmax (max → index), and LayerNorm (mean → var)
factor the same way: unfused they are plain monoid-DAGs; the streaming combine is the fused twist.

## What to build — annotate the DAG, not more kinds

Do **not** add new `AlgebraKind` values. Enrich the carrier annotation so a loop scope can express a small DAG:

```
CarrierNode  = (carrier, role ∈ {MONOID, MAP}, reads: frozenset[str])   # reads name the prior nodes' outputs
CarrierDAG   = ordered list of CarrierNode over a shared axis, edges = read-after-reduce dependencies
```

`role` and the per-node algebra come from the existing carrier traits (`ReduceCarrier.associative` etc.,
`ir/stmt/base.py`); the new content is the **dependency edges** between same-axis nodes. With the DAG in hand:

- **Unfused realization** (default, no twisted machinery): the move-composer walk
  (`plans/move-composer-axis-walk-scheduler.md`) emits one cooperative-reduce per `MONOID` node and a map per `MAP`
  node, all **sharing the K-axis tiling**, with a barrier between dependent nodes. This is exactly the "multi-same-axis
  MONOID + epilogue + second-pass map" shape, and it composes from transforms that already exist
  (`_replace_k_coop` + the free-axis path in `_assemble`).
- **Fused realization** (optional Fork branch): when the DAG matches a known fusible twist (`action` known), collapse it
  into a single `TWISTED_MONOID` pass — the flash path. Cost (memory traffic vs registers/occupancy) decides, so this
  lives in the **search** (`partition/tree.py`), not in the annotation.

## Worked example — softmax over K

The whole mechanism on one kernel: safe softmax over axis `k` (extent `K`) for rows `n`, end to end.

### (a) The fused graph coming in

After decomposition/fusion the kernel is three sibling loops over the same axis inside the row loop:

```
LoopOp softmax(x: [N, K]) -> y: [N, K]
  Loop n (N):                         # parallel row axis
    Loop k1 (K) reduce:               # ── pass A
      Load  t = x[n, k1]
      Accum m = max(m, t)             #    carrier: max-monoid  → output m
    Loop k2 (K) reduce:               # ── pass B  (body reads m)
      Load  t = x[n, k2]
      Assign e = exp(t - m)
      Accum s = add(s, e)             #    carrier: add-monoid  → output s
    Loop k3 (K):                      # ── pass C  (map, reads m and s)
      Load  t = x[n, k3]
      Assign e = exp(t - m)
      Assign y = e / s
      Write y[n, k3] = y
```

Today this dies at the envelope checks: `lift_coop_reduce` bails on `len(reduce_loops) != 1` (`skeleton.py:147`) and on
the epilogue (`:174`), so the whole kernel falls to the legacy planner.

### (b) What the analyzer derives — the CarrierDAG

The walk reads the three same-axis loops and derives the DAG (computed, not stamped — like `algebra_kind`):

```
axis k, in scope of row n:
  A: MONOID(max),  reads {x}        → m
  B: MONOID(add),  reads {x, m}     edge A→B   (read-after-reduce)
  C: MAP,          reads {x, m, s}  edges A→C, B→C
```

The two facts beyond the old single-kind tag — and the only genuinely new content — are the **edges** (`A→B`, `A→C`,
`B→C`) and the **reduced-then-broadcast** outputs (`m`, `s` are reduce results consumed by a later same-axis pass).

### (c) Unfused realization — the default the walk emits

`n` → grid; all three nodes share **one** K-axis cooperative tiling (`BR` lanes per row, the existing `_replace_k_coop`
transform). Edges become barriers; broadcast facts become a lane broadcast after each combine:

```
CTA owns row n,  BR lanes (lane = K_c):
  # ── node A: max-monoid
  m_p = -inf
  for k_o in range(K/BR):  m_p = max(m_p, x[n, k_o*BR + lane])
  m = combine_max(m_p over BR lanes)        # segmented warp shuffle
  broadcast m to all lanes                   # ← reduced-then-broadcast
  barrier                                    # ← edge A→B

  # ── node B: add-monoid  (reads m)
  s_p = 0
  for k_o in range(K/BR):  s_p += exp(x[n, k_o*BR + lane] - m)
  s = combine_add(s_p over BR lanes)
  broadcast s
  barrier                                    # ← edge B→C

  # ── node C: map  (reads m, s)
  for k_o in range(K/BR):
    y[n, k_o*BR + lane] = exp(x[n, k_o*BR + lane] - m) / s
```

Every piece is built from transforms that **already exist** — `_replace_k_coop` for A and B, the free-axis/map path of
`_assemble` for C. The only new lowering is "barrier + broadcast between dependent nodes," driven straight off the DAG
edges. No `TWISTED_MONOID`, no new regime.

### (d) Fused alternative — same DAG, a Fork branch

When the DAG matches a known fusible twist, the search can offer the single-pass collapse (online softmax). The
annotation is unchanged; only the realization differs:

```
  m = -inf;  s = 0
  for k_o in range(K/BR):
    xi    = x[n, k_o*BR + lane]
    m_new = max(m, xi)
    s     = s * exp(m - m_new) + exp(xi - m_new)   # ← the "action": rescale s
    m     = m_new
  (m, s) = twisted_combine over BR lanes            # coupled (m,s) fold
  # then node C map, as above
```

One pass over K instead of three. The `s * exp(m - m_new)` factor is exactly the `action`; for a plain monoid that
factor is `1` (`action = identity`), which is *why* unfused A and B are plain monoids and this fused form is twisted.

### (e) The trivial case degrades cleanly

**RMSNorm** is the same machinery with two nodes removed: `{A: add-monoid(Σx²), C: map(rsqrt-scale, reads A)}` — one
edge, one barrier, one broadcast. The phase-2 target is literally softmax with B and the max-subtraction deleted.

## Why this belongs in the algebra layer (and what does not)

Consistent with `plans/move-composer-axis-walk-scheduler.md`'s three-layer split:

- **Facts (the DAG: carriers + dependency edges)** are derivable bottom-up from the body, single-valued, and shared by
  every downstream pass → annotation layer. This is the genuinely new content.
- **The fuse-or-not decision** is one-of-many under a cost tradeoff → stays in the Fork search. The annotation must be
  able to *represent both* realizations; it must not *pick* one.

So the enrichment is purely on the fact side — but it carries two scheduling-relevant facts the single-kind tag does
not, and these are the real cost of the move:

1. **Read-after-reduce edges** → a barrier between dependent cooperative combines (`s`'s pass cannot start until `m`'s
   combine + broadcast completes). A `__syncthreads`-class constraint the schedule must honor.
2. **Reduced-then-broadcast scalars** (`m`, `s` must reach every lane after the combine) → the annotation marks which
   reduce outputs are consumed by a later same-axis pass, so the lowering emits the broadcast.

## Coverage this unlocks (one mechanism, many shapes)

- **RMSNorm** — sum-of-squares `MONOID` + `rsqrt`-scale `MAP` epilogue (the `skeleton.py:174` deferred case; the first
  target in the companion plan). Single monoid node + map node, no dependency twist.
- **LayerNorm** — mean `MONOID` → variance `MONOID` (reads mean) → normalize `MAP`. Two nodes + edge + map.
- **Softmax (safe, unfused)** — max `MONOID` → sum-exp `MONOID` (reads max) → divide `MAP`. The canonical DAG above.
- **Welford / argmax** — same monoid-DAG shape; their fused combine is the twisted realization.

All four are *compositions* over the same DAG mechanism rather than four envelopes — and each one's fused single-pass
form is the corresponding `TWISTED_MONOID` Fork branch, reachable later without a new recognizer.

## How this generalizes to symbolic axes

The DAG mechanism is *orthogonal* to whether an axis is static or symbolic — and the carrier-DAG annotation actually
makes the symbolic-K reduce **more correct by construction**, because the masking fill value it needs is already in the
algebra.

**Symbolic K (the reduce axis).** A masked-K reduce tiles K at the hint (`DEFAULT_SEQ_HINT`) and zero-fills the final
partial slab so the fold accumulates the identity past the runtime extent (today: `(k < seq_len) ? v : 0`, pinned to
the SYNC transport — a clamped *duplicate* would corrupt the reduction). The crucial generalization: **the fill value
is the carrier's monoid identity**, and the DAG node already names its carrier, so the fill is just
`ElementwiseImpl.identity` (`ir/elementwise.py:120`, `_IDENTITY` at `:70`). For the softmax DAG that resolves per node
automatically:

- **node A** (max-monoid) fills past-extent lanes with `maximum`'s identity `-1e30` — so the masked lanes never raise
  the row max;
- **node B** (add-monoid) fills with `add`'s identity `0.0` — so the masked lanes contribute nothing to `Σ exp`;
- **node C** (map) gets a per-element store guard `if (k < seq_len)` — the masked-tile boundary `Cond` the free-axis
  path already emits.

So the hard-coded `? v : 0` in the current masked-K path becomes `? v : carrier.identity`, read off the DAG. The
single new fact "this reduce is symbolic ⇒ fill with the node's identity" covers *every* monoid node uniformly — no
per-shape special-casing. (`add`'s identity happens to be `0`, which is why the existing single-carrier path could hard-
code it; a max-monoid reduce would silently mis-compile under that hard-code, and the DAG fixes that.)

**The fused twist over symbolic K.** The online-softmax single-pass form (d) is even cleaner: fill a masked element's
`xi` with `-1e30` (the max identity) and *both* carry updates degenerate to identity in one stroke —
`m_new = max(m, -1e30) = m` and `exp(xi - m_new) = 0`, so `s` is unchanged. One fill value, the max identity, neutralizes
the whole coupled `(m, s)` update past the runtime extent. The masked-K → SYNC-transport pin from the static plan
carries over unchanged.

**Symbolic N (the parallel row axis).** Orthogonal to the reduction: a symbolic row axis becomes a ceil-div masked grid
with a per-row store guard, exactly the strided-cooperative-rows treatment (static lanes thread-bind alongside the `BR`
cooperative lanes → a `BN×BR` CTA instead of a degenerate one). The DAG nodes all share the row tiling, so symbolic-N
support is entirely in the free-axis split and never touches the carrier edges/barriers/broadcasts — those are per-row
scalars regardless of how many rows exist. A symbolic-seq per-head q/k-norm (an RMSNorm DAG with a symbolic row axis) is
the concrete case this lights up.

**Net.** Symbolic support is *additive* over the static DAG: (1) reduce nodes fill with `carrier.identity` instead of a
hard-coded `0`; (2) the map epilogue and symbolic rows get the masked-tile store guard the free-axis path already emits;
(3) the masked-K SYNC pin is inherited. The barrier/broadcast structure from the DAG edges is identical static or
symbolic — it operates on per-row reduced scalars, which don't depend on the runtime extent.

## Phasing & the byte-identical gate

Land behind the same discipline as the carrier-analysis and axis-walk plans: per-kernel `deplodock compile` compare
under a fixed `PYTHONHASHSEED`, green `make test`.

1. **DAG analyzer.** Build the bottom-up `CarrierDAG` extraction (carriers + read-after-reduce edges) as a derived
   read over the body — no stamped field, mirroring `Loop.algebra_kind`. A single-`MONOID`-node DAG must reproduce
   today's `MONOID` classification exactly (byte-identical dispatch).
2. **Barrier + broadcast lowering.** Teach the cooperative-reduce realization to emit a barrier between dependent nodes
   and a broadcast for reduced-then-consumed scalars. Validate on RMSNorm (one node + map — trivial DAG, exercises the
   epilogue path) against eager accuracy + `run --bench`.
3. **Multi-node DAG → unfused LayerNorm / softmax.** Drop the single-reduce envelope; the walk emits N cooperative
   reduces sharing the K tiling with barriers. First genuinely new multi-reduce coverage; validate accuracy vs eager.
4. **Symbolic axes.** Swap the hard-coded masked-K `? v : 0` fill for `? v : carrier.identity` (read off each DAG
   node), and let the free-axis path's masked-tile store guard cover the map epilogue + symbolic rows. Validate on a
   symbolic-seq RMSNorm / q-k-norm against a hint-shaped torch reference (the `--dynamic seq_len@x:1` path). Additive
   over phases 2–3 — no change to the barrier/broadcast structure.
5. **Fused twist as a Fork branch.** When the DAG matches a known `action`, offer the single-pass `TWISTED_MONOID`
   collapse as a search alternative (lands *with* the MMA-flash carrier work; behind the `action = identity`
   byte-identical gate from `plans/algebraic-carrier-analysis.md`). Inherits the symbolic-K identity-fill from phase 4
   (one `-1e30` fill neutralizes the coupled `(m, s)` update past the runtime extent).

## Hard constraints

- **No new `AlgebraKind` values.** The enrichment is structural (a DAG of existing carriers), not a new tag.
- **Annotation represents, search decides.** Fuse-vs-unfused never moves into the analyzer; it stays a Fork choice.
- **Derived, not stamped.** The `CarrierDAG` is a computed read over the body (like `algebra_kind`), so it cannot
  contradict the body and adds zero serialization / `op_cache_key` surface.
- **Byte-identical on covered shapes.** Phase 1 must not change dispatch for any kernel already handled.

## Open questions

- **Where is the DAG attached?** Per reduce *loop* it is too local (the edges cross sibling loops over the same axis);
  per `LoopOp` it is too coarse. Likely keyed by shared *axis name* across the sibling reduce loops in one scope — needs
  a concrete home in the IR walk.
- **Barrier cost vs the twist.** Multi-pass keeps `x` live across passes (re-read from gmem, or held in smem/regs); the
  per-pass barrier + re-traffic is exactly what the fused twist eliminates. The Fork must price both — does the prior
  have features for "pass count × K traffic"?
- **Partial-DAG matching for the twist.** Recognizing that a given monoid-DAG *is* a known fusible twist (online
  softmax, Welford) is itself a pattern match — does that recognizer live in the analyzer (annotate "fusible-as X") or
  in the Fork move generator? Leaning toward the move generator, to keep the analyzer decision-free.
- **Identity-fill for carriers without a registered identity.** The symbolic-K fill assumes every monoid node has an
  `ElementwiseImpl.identity` (`_IDENTITY` at `ir/elementwise.py:70`). A carrier whose op is associative but has no
  registered neutral element (none today, but `divide`/`subtract` are deliberately absent) can't be masked this way —
  such a node must keep its reduce static, or the symbolic case bails to legacy. Confirm the gate.
