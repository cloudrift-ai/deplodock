# Pipeline Architecture

Pattern-based rewrite engine + pass directories + dump hooks.

## Modules

```
pipeline/
├── pipeline.py    # Pattern, Match, Rule, Pass, Pipeline (frozen layout), Run (per-run state + engine loop)
├── fork.py        # Fork interface + OptionFork / ThunkFork; Level + build_fork_tree lazy knob-cartesian tree builder
├── knobs.py       # format_tuning_knobs: render real knobs (drop pass-marker booleans) for tune output
├── search/        # Autotune state: Candidate, Search policies, SearchDB + SearchTree
│   ├── candidate.py  # Candidate / LazyCandidate / Cursor data classes
│   ├── policy/       # Search ABC (base.py) + TuningSearch (mcts.py, tune) + greedy_decide (greedy.py, the Run.resolve pick for compile/run); both rank via the Prior
│   ├── db.py         # SearchDB SQLite store: op inventory + lowering edges + perf (per-variant replay cache); open_readonly + iter_perf_samples (perf ⋈ cuda_op) back the data layer
│   ├── data/         # harmonized read-view over the 3 sources (golden / DB perf / prior reservoir): Sample (one normalized row + the single knob_features path; golden rows carry the config's `--dynamic` specs in `.dynamic`), Dataset (from_golden/from_db/from_prior + group_by_op/group_by_kernel_name), ShapeKey (arithmetic S_* identity AND the single golden↔measured join key: `from_matmul` / `MatmulGoldenConfig.shape_key()` build the golden side, `from_s_features` the stamped-op side — dtype flag from `S_dtype_f32`, never `S_n_mma`, which is 0 on every stamped row; `is_dyn` splits a symbolic-axis golden from its static twin, mirroring the 992 stamp's symbolic-excluded extent products + `S_ext_n_symbolic_axis` flag; all diagnostics joins + run's golden A/B kernel matching key through it)
│   ├── keys.py       # op_cache_key / dialect_of / source_chain
│   ├── slice.py      # single_node_graph: isolate one finalized kernel into a standalone graph
│   ├── two_level.py  # two-level tuner: outer structural MCTS + inner separable per-op reward
│   ├── golden.py     # GoldenConfig + Matmul/Reduce/Pointwise subclasses: autotuned knobs per shape (matmul fp32/fp16, cooperative reduce, pointwise) — the AnalyticPrior fit's ground truth across all kernel regimes. A matmul golden may mark its M axis symbolic (YAML `dynamic: {NAME: {input, axis}}`, M doubling as the Dim hint, `.dynM` name suffix): the shape then compiles/benches as a masked-tile kernel via its own `--dynamic` spec (`dynamic_specs()`), a separate deployment artifact from its static twin — never merged. Data lives in goldens/<gpu>.yaml
│   ├── prior/        # the ONE ranking path: Prior ABC + AnalyticPrior (cold heuristic) + CatBoostPrior (learned) composed behind FallbackPrior (load_prior)
│   └── analytic.py  # golden-config eval harness (evaluate_golden / pick_matmul): ranks a shape's enumeration via a Prior (AnalyticPrior by default) — drives eval analytic / eval prior (weights fit by scripts/golden_knob_heuristics.py)
│ # SearchTree (in-memory MCTS state) lives in policy/mcts.py — MCTS is the only policy that reads it.
├── dump.py        # CompilerDump + on_pass dispatch
├── rule_diff.py   # Per-rule unified-diff renderer for ``compile -vv`` output
└── passes/        # pass-authoring invariants (no shape-specific pattern matching) → passes/ARCHITECTURE.md
    ├── frontend/
    │   ├── decomposition/  # frontend ops → tensor-IR primitives
    │   └── optimization/   # IndexMap fusion before lift-to-loop
    ├── loop/
    │   ├── lifting/        # tensor ops → trivial LoopOp nodes
    │   └── fusion/         # fuse fan-out indexmaps into all consumers, merge adjacent LoopOp pairs (splice), then stamp name + structural-feature (`S_*`) knobs
    └── lowering/
        ├── tile/           # LoopOp → TileOp (tileify + scheduling rules)
        ├── kernel/         # TileOp → KernelOp (materialize scheduling)
        └── cuda/           # KernelOp → CudaOp (render source string)
```

## Engine (`engine.py`)

### Chain matcher

A `Pattern(name, op_type, constraints={})` matches one node by op type
plus optional `node.op` field equality. A pattern list matches a chain:
the seed matches `pattern[0]`, its sole consumer matches `pattern[1]`,
and so on. Multi-node patterns only fire when each intermediate node
has exactly one consumer.

`match_pattern(graph, pattern) → list[Match]` walks every topo-ordered
seed; overlaps between matches are allowed (the rewriter exits after
the first successful rewrite per iteration so overlap is just
candidate enumeration).

`Match.nodes: dict[str, Node]` maps each pattern entry's name to the
matched `Node`. `Match.consumed` and `Match.output` are overridable by
the rewrite function to control which nodes the splicer removes and
which node's edges get rewired.

### Rule module convention

Every file named `NNN_<name>.py` under a pass directory is a rule:

```python
PATTERN = [Pattern("root", SomeOp), ...]   # required
def rewrite(ctx: Context, graph: Graph, match: Match) -> Graph | Op | list[Graph | Op]:
    ...
```

The dispatcher binds parameters by name. Reserved names: `graph`,
`match`, `root`, `out`, `ctx`. Pattern names from `PATTERN` bind to
matched `Node` objects. Anything else binds positionally to
`root.inputs[i]`. Take only what you need — `ctx` is optional.

**Returning a list = autotune fork.** A rule that's unsure which parameter to use returns the
alternatives as a list, in any order — the engine spawns one `LazyCandidate` per option (sharing the
parent's graph snapshot) and hands them ALL to a `Search` policy, which ranks them via a `Prior`
(`search/prior/`): `TuningSearch` (`tune`) by PUCT, `greedy_decide` (`compile`/`run`, via `Run.resolve`) by
`Prior.pick` — measured -O3 reservoir evidence first (`evidence_pick`: the candidate prefix-consistent with the
fastest `H_opt=3` row of the same op, value-of-position semantics), the `mean_score` argmin when no candidate
has evidence. There is ONE ranking path — the `Prior` is the hand-coded `AnalyticPrior` cold (a real heuristic
*score* over `knob.knob_features`, not emission order; a separate `_W_A_DYN` weight set ranks symbolic-axis
masked-tile kernels, selected on the stamped `S_ext_n_symbolic_axis`) and the learned `CatBoostPrior` once
trained, composed behind `FallbackPrior` (`load_prior`). A single-shot compile picks the analytic argmin cold;
a `tune` sweep explores every fork. `FallbackPrior` splits its two surfaces once the learned half is fitted:
`mean_score` / `mean_scores` / `pick` (deploy + eval) are pure-learned + evidence, but `score` — the MCTS
*selection* signal — tilts the learned µs by the analytic's dimensionless ranking multiplier
(`learned · analytic**W`, `W = config.analytic_tilt`, neutral 1.0), so PUCT still explores a region the cold
heuristic prices well but the data-poor learned model buries (the fp16 small-`BK` warp tiles — golden-sweep
finding 1). (DB-best replay `_best_fork` and the static `score_of` prior were removed;
the old `_priority_*` enumeration sort that ranked the cold path by emission order is gone — the
`AnalyticPrior` ranks it now. Greedy stays prior-only: the -O3 evidence ships inside the prior's checkpointed
reservoir, not the DB.)
Single-option returns (or bare `Graph` / `Op`) are the deterministic case — no fork.

**Lazy hierarchical forks via `Fork`.** `Fork` is an interface (`pipeline/fork.py`): `knobs` (the knob delta
the fork pins), `is_leaf`, and `expand()` (the next level of options — more Forks, concrete `Op`/`Graph`
leaves, or a mix; exactly `[option]` for a leaf). The search loop pops a Fork-pending `LazyCandidate`,
invokes `expand()` to materialize the children, pushes them back, and continues; cursor advance only fires
when the lineage resolves to a concrete leaf. Lets a rule expose a hierarchy of decisions lazily — only the
subtrees the search actually walks into get materialized. `Fork.knobs` is the knob delta a fork pins — the
variant identity the perf DB and the prior key on — read without expanding. Forks carry NO score:
ranking is the policy's job (the `Prior` over `fork.knobs` — `AnalyticPrior` cold, `CatBoostPrior` trained),
never grouping order.
Implementations hold their producer's state as data: `OptionFork` (a concrete `Op`/`Graph` leaf, built by
`LazyCandidate.from_option`), `ThunkFork` (generic flat forks — `expand_fn(knobs)` a function of the fork's
own knob delta, so siblings share one function instead of per-instance capture lambdas; used by
`085_warp_specialize`), and the tree builder's branch / leaf node classes.

The partition planner (`lowering/tile/010_partition_loops`) routes each reduce loop by its bottom-up **algebra
tag** (`Loop.algebra_kind`, `ir/algebra.py`) rather than re-deriving the archetype: `SEMIRING` → matmul-output
tiling; a static `TWISTED_MONOID` (flash's online-softmax `Monoid`) → the cooperative-reduce KV split; any other
reduce → cooperative-K / pointwise. The tag reads the carrier already in the body, so the routing inputs come
from the analyzer, not a fresh structural match (Part C1a of `plans/algebraic-carrier-analysis.md`). It then emits
one hierarchical Fork tree over both tiers:
`MMA → BR → (BM,BN) → (WM,WN) → (FM,FN) → TileOp` leaf — each leaf carrying its COMPLETE knob row
(incl. `BK` / `SPLITK` / `FK` / `OVERHANG`, which live in no level), the DB-matchable variant identity. The root
`MMA` level keys warp rows by atom kind; scalar rows return an empty key and skip the level (their subtree
splices up as siblings of the atom branches — no `MMA` knob ever pins a scalar path), and the builder's
single-value collapse erases tier-foreign levels (warp rows carry `br = bm = bn = 0`, scalar rows
`wm = wn = 0` — the OFF sentinels), so a pure-scalar kernel's tree is exactly the classic
`BR → (BM,BN) → (FM,FN)` over full-row leaves. Warp-vs-scalar ranking is the `Prior`'s job (siblings emit in
enumeration construction order — the `AnalyticPrior` ranks them cold, the `CatBoostPrior` once trained); an
explicit `DEPLODOCK_MMA=<kind>` pin is authoritative (the planner drops the scalar tier so the search can't
sidestep it). No variant is scored or materialized to rank it — the prior featurizes the row knobs directly
(`knob.knob_features`).

**Hierarchical move composer (opt-in, `DEPLODOCK_MOVE_COMPOSER`).** A from-scratch reimplementation of the
partition stage as a stack of algebraically-justified *moves* lives under `lowering/tile/partition/`
(`walk` → `skeleton` → `moves` + `budget` → `tree` → `materialize`, sharing the extracted `_tower._wrap_tower`).
The front-end is **one nest walk** (`walk.py::walk_nest`), not three template-matchers: it walks the loop nest once
and tags it by each reduce loop's `Loop.algebra_kind` (`MAP`→pointwise, `SEMIRING`→matmul, `MONOID`→cooperative
reduce; `TWISTED_MONOID`→legacy), producing the regime skeleton. Because non-reduce statements *ride* instead of
being rejected by a rigid envelope, a `MONOID` reduce followed by a scalar **epilogue** + a second-pass **map**
loop composes with no new regime — that is how **RMSNorm / softmax** are covered (the reduce(s) and the
different-named map loop are cooperative-split together, keyed by extent via `target_names`; the warp/tree combine
is reused from `kernel/100` via `Accum.axes` σ-propagation). **Symbolic free axes** (dynamic seq_len as rows / M /
N) compose too — masked as a ceil-div grid + `< extent` store guard, scalar tier only (the warp path is
clean-tile); a symbolic **K** cooperative reduce tiles at the hint and fills the masked final tile with the
**carrier identity** (`Init(op)`: `0` for add, `-inf` for max — the monoid-DAG mechanism). A matmul **MAP epilogue**
(QK^T scale, matmul_add) rides the output tile (SPLITK forced to 1); a **multi-accumulator** matmul (gated MLP)
schedules on the cooperative multi-Accum path. With `DEPLODOCK_FLASH=1` the composer also covers the **fused SDPA
flash** nest (`TWISTED_MONOID`): `walk_nest` → `build_flash_tile` tiles the free output axes and serial-transforms
the streaming KV + nested QK^T reduces while the `FlashCombine` carrier renders its own online-softmax rescale
(scalar tier; the tensor-core P@V tier is future work). See `plans/move-composer-axis-walk-scheduler.md`,
`plans/monoid-dag-carrier-annotation.md`, `plans/online-softmax-flash-attention.md`.
Unlike
the legacy planner — which enumerates a flat knob cartesian and groups it post-hoc via `build_fork_tree` — the
composer **generates** the Fork tree move-by-move (`tree.py`): the root offers legal thread tiles, each branch
offers the register tiles legal *given that thread tile* (incremental budget pruning), and each leaf
materializes the same `TileOp` tower. It uses a **greenfield knob vocabulary** (`partition/knobs.py`:
`MAP_N_THREAD` / `MAP_N_REG` / `MAP_M_THREAD` / `MAP_M_REG`, …), so the goldens / prior retrain on the new keys.
`010_partition_loops.rewrite` dispatches to `partition.compose.try_compose` when the flag is set and falls
through to the legacy planner for any regime the composer doesn't yet cover. **Covered so far: pointwise (`MAP`)
and plain matmul (`SEMIRING`, static K, no split-K / cooperative-K / fused prologue) — both scalar and tensor-core
tiers.** The matmul adds a `TileSerial` move re-bracketing K into a `K_o` (serial-outer) / `K_i` (stage-inner)
tower (scalar: optional `RED_FK` strip-mine; warp: `atom_k`-strided), and a `Tensorize` move gated on
`_atom.is_atom_eligible` that emits the BLOCK>WARP>REGISTER>ATOM tower for an eligible (fp16/bf16) matmul. The
generative tree's matmul root is the `Tensorize` choice — warp subtrees (one per eligible atom) plus the scalar
fallback — then warp→reg→bk (or reduce→thread→register for scalar). The warp tier reuses `011_lower_atom_cell` to
fold the canonical cell into `Mma` (absorbing 011 fully is deferred), and **stamps the legacy `MMA` knob on the
deployed warp tile** so the knob-driven downstream passes (`020_stage_inputs`, `005_lower_atom_tile`, `is_warp`)
take the tensor-core path — the greenfield `TC_*` knobs are the search vocabulary, `MMA` is the downstream bridge
until those passes are greenfielded in Phase 4. **Split-K** (`RED_SPLITK` → a `K_s` grid axis, codegen-derived
atomic-add) and **whole-CTA cooperative-reduce** (`MONOID`, K ≥ warp_size: free rows → grid, `COOP_BR` threads on
the `K_c` THREAD axis, the warp/tree combine reused from `kernel/100` via `Accum.axes` σ-propagation) also land.
Greenfield search dims are env-pinnable (`DEPLODOCK_<KNOB>`, `moves._pin`). Deferred to legacy fallthrough:
strided-cooperative rows, the matmul_add residual gate (015), reduce epilogues, masked/symbolic K, and
flash/SDPA-prologue. See `plans/melodic-giggling-gem.md`.

**Demoted-matmul split (`SPLIT_CONE`, rule `005_split_demoted`).** A fused computed-operand cone (the
gated-MLP norm prologue, an elementwise scale) keeps a matmul off the warp tier — `ldmatrix` feeds fragments
from staged smem and a computed A operand has no buffer to stage, so `is_atom_eligible` never passes on the
fused body. Rule `005_split_demoted` (its own pass unit, running before partition so the body is still
un-tiled) offers a structural fork: `[keep-fused Op ({SPLIT_CONE: False} stamped), OptionFork(Graph whose
kernels carry {SPLIT_CONE: True})]`. The cut is ONE rule with no per-shape cases: each multiply operand is
independently a stageable plain Load (K in one index dim — stays put), a K-FOLDED Load (K across several
dims, the collapsed reshape/transpose o_proj attn-out read; `020_stage_inputs` can only stage a single-K-dim
slab, so the warp tier is structurally unreachable — a degenerate cone whose only member is the Load itself,
its producer the contiguizing copy), or a computed cone, and every distinct cone becomes a producer
materializing an `xn` intermediate over exactly the axes it reads — `xn[rows read…, k]` for a row cone (the
A operand), `xn[rows read…, k, n]` for an N-reading cone (the B operand; N innermost, so K lands
second-to-last and the consumer's B load carries the canonical layout the cell tagger / stager serve even
when the original access was transposed `[n, k]`). Producers carry their cone's prologue deps (the norm's
row-stat reduce, P@V's softmax stats) and materialize at the cone's uniform CELL-leaf dtype (prologue/lead
loads only feed row stats — an f32 mean-count scalar must not block an f16 norm chain); the consumer
loads each `xn` under the cone root's SSA name, so the multiply and epilogue are untouched. Cones compare
by VALUE, not SSA name: fusion inlines a shared chain once per consuming matmul (the gated-MLP norm feeds
gate AND up as two structurally identical chains), so the cell is value-numbered and same-class roots share
one `xn`. A MULTI-accum K loop (gate+up sharing the reduce) additionally extracts each accum's matmul into
its own clean single-matmul gemm producer (`__mm0`, `__mm1`, …) writing `mm_i[rows…, n]` at f32 (the
accumulator's own precision) — the mma cell gate admits exactly one matmul per K loop, so the fused pair
could never reach the warp tier — and the consumer becomes the pointwise combine: each K loop replaced by
Loads re-reading the `mm_i` buffers under the accums' SSA names, the epilogue (SiLU·up) untouched. The
familiar shapes are instances: norm→linear / scale→matmul = one row cone beside a Load; SDPA P@V = one row
cone with prologue deps; rotary QK^T = a row cone + an N cone (the GQA `head / 2` shared-KV read stays a
leading dim, duplicated across the sharing heads); a weight-side scale = one N cone beside a Load; the
gated MLP = one value-shared row cone + two extracted gemms + the combine; o_proj's collapsed attn-out =
one degenerate Load cone beside a Load. All pieces
re-enter the planner as ordinary LoopOps with their own fork trees. The offer is gated ONLY on the cut's
well-formedness (`_split_demoted.try_split_demoted` bails on multiple K loops, no computed or K-folded
operand, a K-invariant Load operand, distinct-class cones sharing stmts, escaping cone values,
mixed-dtype cell leaves, symbolic K or N extents (symbolic ROW axes are admitted — the cut re-emits rows
verbatim and the `xn` / `mm_i` buffers carry the symbolic Dim), more than one N-reading cone — two
`(…, K, N)` buffers would re-do the matmul's own volume — or, multi-accum only, a cell stmt no gemm claims) —
deliberately NOT on a predicted tier for the clean gemm: profitability is the search's question (an earlier
eligibility-simulating gate immediately drifted from what the cell tagger accepts). The two-level tuner owns
the offer as an **outer structural fork** (`plans/structural-forks-in-two-level.md` step 2): keep-vs-split
branches the outer tree, each side's kernels are tuned in first-class per-op slices, and the Σ-per-op
terminal rewards compare the kernel sets (017's sub-partition splice still sums inside the op's slice);
greedy deploys the split only via the trained prior's structural pricing (see `Pipeline.run` above) — never
cold. The split decision lives in the tile phase, not as a fusion guard: by partition time
the fused body is final, so the demotion is visible order-independently (decomposed chains assemble the
matmul multiply-last — no standalone node is ever eligible mid-fusion). The `op.knobs` stamp is the
considered-vs-declined idiom (`020_stage_inputs`'s declined `STAGE` row, `search/keys.py`): it is
simultaneously the rule's idempotence guard (without it a multi-site batch re-offers combinatorially in fork
children — measured; a node may still be offered once per sibling branch of an earlier fork point, the
intended cross-product of independent decisions), the learned prior's training signal (absent = never
offered → NaN-filled; `False`/`True` = the decision, riding every perf row), and the `op_cache_key`
separation that keeps each decision state distinct from its parent in the search tree. The stamp is
deterministic per offer site, so identical kernels across graphs stamp identically and keep sharing perf
rows.

**Post-split re-fusion (rules `006_merge_split_glue` – `009_stamp_structural_features`).** The
split's glue kernels — `xn` materializations, the pointwise combine — are launch-latency-floor at deploy
size (the Qwen3 layer-0 findings measured ~23 of 48 per-launch µs in glue at 1–23% DRAM), so right after
`005_split_demoted` the tile head re-runs the loop-fusion *mechanism* on the still-untiled `Graph[LoopOp]`:
`006_merge_split_glue` wraps `loop/fusion`'s splice plumbing (`_helpers.build_merged_op` /
`wrap_merge_fragment`) under a split-preserving guard set, and `007`/`008`/`009` are thin re-export aliases
of `020_dedup_loads` / `991_stamp_loop_names` / `992_stamp_structural_features` so merged kernels get
deduped bodies, names, and fresh `S_*` features before partition fixes their `op_cache_key`. The flagship
merge folds the gated-MLP combine into one extracted gemm's epilogue (the existing
`classify_fragment_epilogue` / `RegEpilogue` fold lowers it; the fold also resolves prologue-resident scalar
leaf loads — a real trace's f32 constants the splicer parks at the TileOp root — via
`005_lower_atom_tile._collect_outer_loads`, keeping the fold in sync with the Loop-IR gate that admitted
them). The guards: fire only when a matched op carries `SPLIT_CONE: True` (inert in the loop tier and on
keep-fused branches); never merge a node already carrying the `tile.split_glue` **node hint** (one-level
contract — the full pipeline gives the rule ONE LoopOp batch per scan while the outer head loops to
quiescence, and second-order merges firing only in the latter would split `op_cache_key`s between outer
search and greedy replay; the marker rides `Node.hints` like provenance, NOT `op.knobs`, because every knobs
key becomes a prior training feature and this is plumbing, not a decision to learn); never inline
into a consumer that reads the producer inside a reduce loop (re-polluting the K cell = re-demotion); never
merge two `Accum`-bearing bodies (would rebuild the multi-accum kernel the split cut apart); the base
blowup + broadcast-materialization economics; and never trade away atom eligibility (`is_atom_eligible` on
both constituents vs the merged op — the real gate, no simulation). The base multi-load-of-reduce-heavy
guard is deliberately dropped here: the post-split shapes that read a reduce producer through two Loads
(RoPE reading the normed row) are exactly the target merges, and the splicer dedups the row stats to one
emission. The merged op is re-stamped `SPLIT_CONE: True` (005's idempotence — without it the rule re-offers
a split of the merged kernel and split→merge→re-split never terminates) and forwards a constituent's
`source` for the outer search's Σ attribution. Unconditional cleanup, NOT a fork — a single deterministic
`Graph` rewrite adds no outer-tree nodes. Known
v1 gap: the deployed-graph attn@V→contiguize-copy backward merge (o_proj's `xn`) is rejected by
`splice_graph` itself (σ-solve of the 4-D SDPA write vs the collapsed read), so that copy stays a launch.

Binding tiers the planner emits today: `Role.BLOCK` (→ `GridTile`),
`Role.THREAD` (→ `ThreadTile`), `Role.REGISTER` (→ `RegisterTile`).
`Role.WARP` (→ `WarpTile`) and `Role.ATOM` (→ `AtomTile`, the
hardware-atomic MMA cell tier) are wired through `_layer_kind_for` /
`_wrap_tower` but no rule in this pass emits either today — the MMA
fragment-factorization consumer plan (`plans/mma-fragment-factorization.md`)
will flip these tiers when its M3 ships, without revisiting the tower
mechanics. M1 of that plan landed the `AtomTile` flavor + the (then-empty)
atom registry — now `ir/tile/ir.py`'s `ATOM_REGISTRY` — + the warp-tier variant row in `_enumeration`
(a plain knob dict like every row: warp tier carries `{WN, WM, FM, FN, BK, SPLITK, MMA}` and is
discriminated by the `MMA` key; the `Atom` spec is `ATOM_REGISTRY[row["MMA"]]`). `085_warp_specialize` already emits `WarpTile(role)` (one
role axis = total CTA warps) wrapping `WarpSpecialize` directly,
bypassing the planner's tower builder — its role split is structural
(`Cond(role < n_producer_warps, …)`), not the σ-shifted extended
`ThreadTile` the pre-refactor shape used. The materializer drops a
`ThreadTile(tid_offset=n_producer_threads, …)` inside the consumer
branch so the original consumer thread axes decode against
`threadIdx.x - n_producer_threads`. The pass handles **both** consumer
tiers: a scalar `ThreadTile` (pointwise / cooperative-reduce) and the
warp-tier MMA tower's existing `WarpTile` (WM×WN warp coords). For the
warp tier it consumes the planner-emitted `WarpTile` directly — no fresh
tier synthesis — and stamps `WarpSpecialize.consumer_is_warp=True`, so the
materializer wraps the consumer in a `WarpTile(tid_offset)` decode
(`warp_id = (threadIdx.x − n_producer_threads) / 32`) and scales every
consumer `bar.sync` participant count by 32 (warp axes count warps, not
threads). `005_lower_atom_tile` harvests the producer body's hoisted
`StageBundle` Sources before lowering the consumer `AtomTile` so the MMA
Loads resolve their smem addressing across the producer/consumer split.
Validated `max_diff=0` across 256²–2048²; ~no latency change vs WS=0 on
GeForce s16816 (its cuBLAS gap is the SASS mma schedule below the IR, not
producer/consumer overlap), so the autotuner normally picks WS=0 — the warp
arm is for parity / investigation and Hopper-class parts.

The tree-building algorithm itself (group params by per-level knob keys, collapse single-key levels, skip
empty-key levels, defer leaf materialization to `expand()`) lives in `pipeline/fork.py` (next to the
`Fork` interface and its flat implementations) as the
reusable `Level` + `build_fork_tree` pair — `partition_loops` supplies the `Level`s + `materialize=`
callable and returns the result. Nodes are real classes holding data, not closures: every
`_Branch` / `_Leaf` references one shared `_Tree` (levels + materialize). The
builder hands back the lazy ROOT `_Branch` and nothing else exists yet: a branch's `expand()` builds its
children on demand in grouping order (ranking is the search's job — Forks carry no score). Future rules with
multi-level knob-cartesian forks should reuse the builder; one-shot flat forks (e.g.
`lowering/tile/085_warp_specialize`'s `WS={0,1}` 2-element `ThunkFork` list) stay inline.

**FN > 1 lowering.** The partition planner always emits the per-cell
shape — one `RegisterTile(N_r)` wrapping the whole
`{Init, K-reduce, Write}` body, regardless of FN / SPLITK / BR / prologue
shape:

    M_r REGISTER:
      RegisterTile(N_r):
        Init(acc)
        K_o SERIAL_OUTER:
          K_i STAGE_INNER (reduce):
            <body>                               # M-axis Loads, prologue,
                                                 # Load b, Accum — replicated
                                                 # per cell by the Kernel-IR
                                                 # replicator
        Write(C, acc)

The Kernel-IR replicator (`lowering/kernel/010_split_register_axes`)
walks the body and per-cell duplicates only statements that transitively
depend on the register axis (dep-tracked via Expr free vars + SSA
def-use); N-invariant statements (e.g. Load `a[m, k]`, the RMSNorm
prologue chain) stay single-copy. `dedup_replicated`
(`lowering/kernel/011_dedup_replicated`) then runs as content-agnostic
defense in depth: structurally identical Loads and Assigns left over
after replication CSE-fold into one — the same effect the deleted
register-blocked GEMM builder used to get from its hand-written
N-invariant-cone partition (see `plans/obsolete-blocked-gemm-builder.md`).

Leaf Fork `expand` thunks call `_materialize(plan, params)` lazily —
`_build_split_body` + `TileOp.__post_init__` (which runs the full
12-pass `normalize_body`) only fire for the variant the search actually
resolves. In greedy `deplodock compile`, that's one variant per LoopOp. The enumeration is large — the
widened FM/FN candidate set yields ~40k+ knob rows per matmul-class kernel (not the "couple hundred" the
original lazy split assumed) — so the finished `_Plan` rides its LoopOp as op metadata
(`loop_op.meta["plan"] = (cache_key, plan)`): ops are shared by reference across graph copies and
`single_node_graph` tune slices, so re-planning the same op object short-circuits entirely (classification
and enumeration included). The stamp is validated against `_plan_cache_key` — the carry-forward knobs,
which include the `S_*` structural features stamped by `loop/fusion/992_stamp_structural_features` (a stmt/op
histogram + loop extents + operand dtypes, so a structure or dtype change is an identity change), + the
hardware context + the live `DEPLODOCK_<KNOB>` pin snapshot (pins fold into enumeration via `Knob.narrow`)
— so a pin / ctx / structure change invalidates it. The same `S_*` knobs ride every variant's knob dict, so
`knob.knob_features` turns each row into the prior's feature vector and structurally identical
kernels — the same layer repeated through a whole model — featurize alike and share the prior's rows.

Variant ranking is a single `Prior` over `knob.knob_features` (`search/prior/`): greedy picks via
`Prior.pick` (measured -O3 reservoir evidence first, `mean_score` argmin otherwise), MCTS ranks the PUCT
frontier. The `Prior` is the hand-coded `AnalyticPrior` cold (a fixed linear model over the engineered `D_*`
geometry / occupancy features — the cold path is ranked by a real heuristic *score*, not emission order; the
masked tier rides its own `_W_A_DYN` weight set keyed on `S_ext_n_symbolic_axis`) and the learned
`CatBoostPrior` once trained, composed behind `FallbackPrior` (`load_prior`). One gated term sits OUTSIDE the
fit weights — `AnalyticPrior.score` rewards `NOATOMIC` (atomic-free split-K) once the split count `SPLITK`
reaches `atomic_free_split_threshold` (default 4) and penalizes it below, via the `af_on · (±1)` interaction a
plain linear weight can't express (the workspace + reduce wins on wide splits, the `atomicAdd` fast-path on
narrow ones — see `plans/atomic-free-monoid-combine.md`). Hardcoded `__init__` params, not fit; the
`CatBoostPrior` takes over once real atomic-vs-free `H_opt=3` rows exist. There is ONE ranking path: the old per-variant `Op.lazy_score` /
`TileOp.score_tile_geometry` formula, the `Fork.score` / `Search.score_of` plumbing, AND the `_priority_*`
enumeration sort that ranked the cold path were all removed — nothing materializes or scores a TileOp just to
rank it; the prior featurizes the row knobs directly.

**Idempotence requirement.** Every rule MUST be idempotent on its own
output. The engine re-runs the entire pipeline on each popped candidate
from pass 0; rules whose output is already in the graph must `RuleSkipped`
or have a pattern that no longer matches. Most rules satisfy this
implicitly via op-type changes (`LoopOp` → `TileOp`); the rest have
explicit `raise RuleSkipped("already X")` guards. Without idempotence,
re-runs would double-apply and corrupt graph state.

The return type discriminates the rewrite flavor:

- **Functional** — returns a `Graph` fragment, spliced in place of
  `match.output` (defaults to `match.root_node_id`); fragment `InputOp`
  nodes reference existing graph nodes by id, non-Input nodes get fresh
  ids.
- **In-place** — returns an `Op`. The engine assigns it to `root.op`
  directly, preserving the node id, inputs list, output Tensor, and
  hints. Used by the lowering rules because `KernelOp.arg_order` /
  `CudaOp.arg_order` embed the original node id as the output buffer
  name, so a fresh id from splicing would break the generated kernel's
  buffer binding.

Raise `RuleSkipped(reason)` to decline a match — the engine logs the
reason at DEBUG and moves on.

**Knob-stamp invariant.** Every emitted variant carries an **explicit value for
every declared knob** — no realized leaf has an absent knob. This is enforced
declaratively: each `Knob` declares an `off` value (its "unused / declined"
sentinel — `TMA.off=False`, `RING.off=1`, the tier-foreign `WM.off=WN.off=0`,
`MMA.off="0"`, `BM.off=BN.off=BR.off=FK.off=0`), and the pipeline fills any of a
pass's declared knobs the variant left unspecified at the **pass boundary**
(`Cursor.advance` → `_off_fill_pass`, via `knob.apply_off_defaults`) — covering a
pass that acted, declined, was skipped, or returned no variants alike. `Pass.load`
discovers a pass's declared knobs by scanning its rule modules' `vars()` (so an
imported knob, like the planner's `_enumeration` tier set, counts). Scoping the
fill to the just-finished pass avoids prematurely stamping a *later* pass's knob
(which would trip that pass's `if KNOB.name in op.knobs` idempotency guard). The
partition planner additionally OFF-fills its tier knobs **at enumeration** (in
`_enumeration`'s two impls) so the sentinel rides the variant identity from the
fork-tree keys + score input through to `TileOp.knobs` and the DB — the value the
prior is *queried* on during greedy descent matches the value it's *trained* on.

The reason it matters: the learned prior (`knob.knob_features`) NaN-fills absent
feature columns. With explicit OFF values, NaN now means **only** "not-yet-decided"
(a partial fork prefix during descent), distinct from "decided: unused" (an OFF
value on a complete leaf) — the prior no longer conflates a tier-foreign knob with
an undecided one and is no longer dragged onto degenerate all-default configs. A
knob with no `off` (the `_UNSET` default — universal knobs like `BN`/`BM` always
set by their pass) is never auto-filled. `BINMASK` `STAGE` keeps stamping its own
width-correct all-zero off mask (a static OFF can't encode the per-kernel width).
Tier discrimination is value-based throughout (`knob.is_warp` / `knob.mma_atom`,
since a scalar leaf now carries the truthy *string* `MMA="0"`). Verified
end-to-end by `tests/compiler/passes/test_knob_stamp_invariant.py`.

A rewrite that *returns* an op which fails `Op.validate(ctx)` (e.g. a
`100_materialize_tile` `KernelOp` whose smem exceeds `ctx.max_dynamic_smem`)
is filtered by `Candidate.try_rewrite` — correct as **fork pruning** (sibling
branches carry other tile shapes) but fatal in a single-path greedy compile,
where it leaves the node un-lowered. `Pipeline.run` installs a `rejections` sink on the
`Run` that records each such drop `(node, pass, reason)`;
after the terminal settles, `_raise_on_unlowered` raises a loud `LoweringError`
naming any still-un-lowered node (its op is still a `LoopOp`/`TileOp`) instead
of letting it leak to `CudaBackend` as a cryptic `non-CudaOp` `TypeError`. The
sink is absent under `tune`, so the fork-pruning path stays silent and a
validate-dropped branch is a graceful dead end.

A rewrite that *raises* mid-lowering (a deterministic lowering pass hitting an un-lowerable shape it can't
represent — e.g. `100_materialize_tile`'s single-Write hoisted-compute materializer on a sibling-cell-fused slab,
which raises `LoweringError` from `_stage_expand.compute_phase_info`) is the same kind of dead end, but for an
*exception* rather than a validate-filter. Greedy `resolve` lets it propagate loudly; under `tune`, `Run.drive`
catches it per-candidate, drops that candidate's subtree (the `pop()` already decremented its `live` count, so this
is bookkeeping-identical to a dead-end terminal), logs `[tune] dropped un-lowerable candidate (…)`, and bumps
`Run._dropped_candidates` (reported in the terminal-count line). Without this, one search-only un-lowerable fork
aborted the whole tune.

Files starting with `_` (e.g. `_broadcast.py`) are **not** loaded as
rules — they're shared helpers for the pass's rule modules.

### Drivers

`Pipeline.build(passes)` wraps a pass list; the resulting object exposes the
compile entry points (`run` / `tune`), each driving one of the `Run` engine
loops (`drive` for exploration, `resolve` for deterministic resolution):

- `Pipeline.run(graph, *, backend=None, db=None) -> Graph` — single-shot greedy compile: a deterministic
  resolution (`Run.resolve`, below) with the greedy pick (`policy/greedy.greedy_decide`), NOT a search — no
  frontier, no tree, no benching. The decide flattens each fork point to its complete leaves
  (`fork.flatten_leaves`) and picks the `Prior`'s `mean_scores` argmin — `AnalyticPrior` cold, `CatBoostPrior`
  once trained; option-0 (first leaf, emission order) only if the prior fails to load entirely. The input graph
  is copied once per attempt and resolved in place — no per-fork copies (a whole-model compile used to pay one
  full graph copy per fork point for sibling snapshots it immediately dropped). **Structural options are
  priced, never raw-scored**: the per-op prior prices one kernel's knob row, so its
  score for a multi-kernel `Graph` splice (the `SPLIT_CONE` split, 017's atomic-free
  combine) is noise. With the *trained* prior loaded, `greedy_decide`'s `_pick_structural`
  prices each side properly — a nested `resolve` per kernel over a `lowering/tile`-only pipeline (CPU, no
  backend) and a trace query: the kernel's price is the `score` of its slice-resolve's `Decision` at the
  partition fork; the structural side is the Σ over its fragment's kernels,
  memoized per `op_cache_key` — and the cheaper kernel set wins, so an unpinned
  compile deploys the splits `tune` measured best. Cold (analytic / no prior), or
  when a side is unpriceable (a pre-tiled combine has no partition fork), the
  structural leaf is filtered as before — a cold compile never changes kernel sets.
  Retries are decide-wrappers over a deterministic re-resolve (every other choice replays identically — cheap
  non-chronological backtracking, no snapshots or undo log): if a structural pick leaves a fragment kernel
  un-lowered (`validate(ctx)` rejection — "did this resolution take a structural pick" is a trace query, `any
  Decision with chosen_kind == "graph"`), the retry retires structural picks wholesale
  (`price_structural=False`) and re-resolves down the keep-fused branch before falling back to tile
  blocklisting (`blocked=`).
  `tune_async` explores structural forks regardless; an env pin makes the
  Graph the rule's only option, which applies inline and never reaches a decide. `Pipeline.run` is the
  deterministic greedy compile only — it does not bench (the benching tune path is `tune_async`).
- `async Pipeline.tune_async(graph, *, search, backend=None, db=None)` — the (async-only) autotune
  sweep; the sync `Pipeline.tune` is gone. Pass a `TuningSearch(patience=, ucb_c=)`; the async generator
  yields one terminal `Candidate` per fully-explored rollout.
  `tune_async` benches each terminal via `await _bench_terminal_async` (writes
  per-kernel `perf` / `lowering` / inventory rows, returns the aggregate
  `PerfStats`), then calls `search.observe(stats, status)`. With
  `backend=None` the bench is stubbed to `latency_us=1.0` and nothing
  is persisted, so a backend-less sweep never overwrites tuned
  `best_median_us` rows with the stub.
- `Run.drive(graph) -> Iterator[tuple[token, Candidate]]` — the exploration engine loop (`tune`).
  `Run` is the per-run state object (`pipeline` + `ctx` + `search` + `db` + `backend` + `dump` +
  `rejections`): `Pipeline` stays a frozen, shareable pass layout while every run-scoped sink and service
  lives on the Run, reached from engine-adjacent code through the candidate (`cand.run.dump`,
  `cand.run.rejections`, `cand.ctx`). `drive` seeds the root candidate, then per iteration pops a
  `LazyCandidate`, resolves it, runs one rule batch (`Run._step`, shared with `resolve`), pushes successors
  under the pop's token. Selection is
  `TuningSearch`'s job (PUCT over the learned prior). (The DB-best replay path `_best_fork` and the `best=` push argument were nuked — see "no longer drives
  selection" above; the perf DB still *records* every bench as the prior's training data.) Each fork push is
  classified by effect at the spawn site, where the raw option list is concrete: any `Graph`-splicing option
  (a kernel-set change — `tile/005_split_demoted`'s split, `tile/017_atomic_free_splitk`'s combine) marks the
  push `structural=True`; `Op` rebinds and the partition planner's branch Forks are op-variant (`False`).
  The flag rides `Search.push(structural=)` so policies can treat kernel-set decisions specially (see
  `plans/structural-forks-in-two-level.md`).
- `Run.resolve(graph, decide) -> (Graph, list[Decision])` — the deterministic-resolution counterpart of `drive`
  (`plans/resolve-trace-driver.md`). Both entry points share one rule-batch body (`Run._step`: matching, inline
  single-option applies, cursor advance, the structural-decision replay), but `resolve` is a fold, not a search: ONE
  live graph mutated in place (no `LazyCandidate` sibling snapshots, no per-fork graph copies — the terminal IS the
  seeded graph object), and at each undecided fork a `decide` callback gets a `ForkPoint` (the `Match`, the raw
  options exactly as the rule emitted them — lazy fork trees unexpanded, the pre-decision root op, ctx) and returns
  the option to apply (a concrete `Op`/`Graph` or a leaf `Fork`; a decide that wants complete tile rows flattens
  branch Forks itself). The returned trace — one `Decision(rule_name, node_id, chosen_kind, knob_delta, score,
  n_options)` per decided fork, `score` being the decide's own annotation on the `ForkPoint` — is the resolution's
  only process-state output: "did this compile take a structural pick", "what did the partition fork predict for
  this kernel" are trace queries, never accumulated policy attributes. Inline replays of an already-decided offer
  site don't trace (they are reads of the first decision, not decisions).

### The keying map: two identities

Everything the search stores or replays is keyed by one of TWO identities — when adding a cache or a
table, pick one; don't invent a third:

- **Variant identity = `(context, knobs)`** — anything *predictive or replayable*. The `S_*` structural
  features (`loop/fusion/992_stamp_structural_features`: a stmt/op histogram + loop extents + operand dtypes) make
  the merged knob dict a COMPLETE identity, so a prior is a pure function of it: the `perf` row keys on the
  realized op digest, the planner's op-metadata plan stamp keys `(ctx fields, frozenset(merged knobs))` plus
  the `DEPLODOCK_*` pin snapshot (pins are context-side: environment that gates enumeration). The *learned*
  prior is exactly `score(features(ctx, knobs))`: the structural facts (op histogram, extents, dtypes) are
  already in the knob dict, so `knob.knob_features` turns it straight into the model feature vector (the
  `S_*` knobs pass through; tuning knobs encode by type, `MMA` expands to atom props). See the learned-prior
  section below.
- **Measurement identity = `(ctx.structural_key, op_cache_key)`** — ground truth about *materialized
  leaves*: `perf` rows (the per-variant replay cache), op inventory (`loop_op`/`tile_op`/`kernel_op`/`cuda_op`), and
  two-level dedup. The structural `child_key` on `lowering` rows is measurement linkage (it joins the
  inventory), NOT a replay key.

### Search persistence: on-disk inventory vs in-memory MCTS

The autotune state is split across two cooperating modules:

- **`SearchDB`** (`search/db.py`) — SQLite store partitioned into six
  tables: `loop_op`, `tile_op`, `kernel_op`, `cuda_op` (one row per op
  encountered along any lowering chain, keyed by `op_cache_key`), a
  `lowering` edge table (one row per rewrite hop carrying the knob
  delta the rule stamped at that hop plus a best-median upsert — the
  chain `best_per_op_time` walks to resolve a pre-final op's measured
  cost — loop→loop source hops are skipped: those are
  structural/decision hops, and a one-best-child row would let a
  multi-kernel decomposition's parent resolve through ONE fragment
  kernel's median), and a backend-partitioned `perf` table carrying
  full stats (`latency_us_{median,min,max,mean,variance}`,
  `n_samples`, `backend`, `status`, `knobs`). Selection statistic is
  the median.
- **`SearchTree`** (`search/policy/mcts.py`) — pure-Python in-memory
  MCTS state, colocated with `TuningSearch` because MCTS is the only
  policy that reads it. Each tree node wraps a `LazyCandidate`; nodes
  carry `visits` and `best_reward` (max reward over the subtree's
  measured leaves), plus a `live` counter that filters out subtrees
  whose frontier has been fully drained. Lineage is TOKEN-THREADED, not
  call-order-dependent: `pop()` returns `(token, candidate)` (the token
  IS the `SearchNode`), the engine pushes children with `parent=token`
  and observes the terminal's measurement with the same token, so the
  tree stays correct however the engine interleaves pops / pushes /
  observes. Rebuilt fresh each process; cached `perf` rows in the DB
  ensure no re-bench on warm starts. Greedy compiles build no tree at
  all — they don't go through a `Search` (see `Run.resolve`).

`_bench_terminal_async` (over the shared `_TerminalBench`) is the only path that knows about all
four parts (graph, DB, tree-through-`search.observe`, backend). It
short-circuits when every `CudaOp` in the graph already has a `perf`
row for the current `(context_key, backend)` — no GPU bench, stats
reconstructed from the DB. Otherwise it does one
`await backend.benchmark_async(...)` call, walks `Op.source` once to record op
inventory + lowering edges + the `perf` row per kernel, and returns
the aggregate `PerfStats` (summed across kernels) for the search to
score.

## Tuning workflow

The autotune loop selects one tile-lowering variant per CudaOp by repeatedly running the lowering pipeline with
different knob choices at each fork point, benching the produced kernels, and steering subsequent rollouts toward the
configurations that produced the lowest measured latency.

### Two-level search: outer structural MCTS + inner separable per-op tuning

`deplodock tune` does **not** run one MCTS over the whole graph. The pipeline applies rules sequentially, so two very
different kinds of fork — **op-variant** forks (tile / pad / stage choices for one kernel) and **structural** forks
(which kernels exist: fusion grouping, the demoted-matmul split) — would nest and cross-product under one global
patience. That cross-product is what starved deep ops (the bottleneck kernel exhausted patience before an SDPA P@V
kernel reached its good tile). The two kinds have opposite structure, so `search/two_level.py` splits them on the
fork's *effect* (the spawn-site `Op`-rebind / `Graph`-splice classification — `plans/structural-forks-in-two-level.md`):

- **Outer search** (`run_two_level_tune`) drives the graph-changing passes — `frontend` + `loop` plus the
  pre-partition head of `lowering/tile` (`outer_pipeline()`: `005_split_demoted`'s keep-vs-split offer followed by the
  non-forking `006`–`009` post-split re-fusion aliases, which change kernel sets but never branch the tree). A
  **terminal** is the state where the cursor reaches `partition_loops` with every structural fork resolved — every op
  post-fusion and structurally final, split producers/consumers included as real `LoopOp` nodes. Each terminal is a
  candidate fused graph; its **reward** is `1 / Σ best-per-op time` from the inner search, backpropagated by the
  reused `TuningSearch` — keep-vs-split is an outer-terminal comparison, the natural cost model for a kernel-set
  decision. Structurally identical offer sites within one trajectory take the same side: `Run.drive` replays the
  first decision read off the trajectory's own graph (`_replay_structural_decision` — any op carrying the fork's
  decision knobs whose `Op.source` chain contains an op structurally identical to the offer; the stamped knob values
  pick the matching option), so the outer tree stays linear in *unique* kernels instead of `2^sites` with no
  side-table state threaded through resolves; a terminal whose ops are all known is a pure DB read. **Fusion
  itself is still deterministic** (no rule emits a multi-option *fusion* fork — see `autotune_no_graph_forks`); a graph
  with no structural offers yields one terminal and this reduces to "tune each op once, sum, assemble". The outer uses
  a `Run` directly (manual `observe`) since its reward comes from the inner tuning, not `_bench_terminal_async`. The global
  prior also drives the outer PUCT (the outer `TuningSearch` carries `prior_model` + ctx `base_knobs`): each terminal
  emits one **composed Σ row** per structural decision it realized — features `{ctx, pre-decision op knobs, decision
  delta}`, label = the Σ of that side's per-kernel bests (`_decomposition_rows`) — attributed through the `Op.source`
  decomposition links `Candidate.apply` stamps on loop-dialect lowering splices (005 sets it explicitly on its
  keep-fused rebind too, since `replace` would copy the pre-decision op's own source past the offer site;
  `_rename_buf_in_op` preserves `source` through the splice id-promotion). The row's feature shape is exactly what the
  outer's `_node_knobs` produces at the fork's siblings (`LazyCandidate.resolved_knobs` keeps a resolved ancestor's
  delta visible to its descendants — without it the structural branch's continuation would score as a knob-less
  generic row against its fully-knobed unresolved sibling), so a warm re-tune descends the predicted-cheaper kernel
  set first instead of emission order. Composed rows are derived value-of-position estimates that *order
  exploration*; greedy's deploy decision keeps the sharper compositional probe (complete-row predictions per kernel).
- **Inner search** (`_inner_reward_async`) tunes each finalized kernel **independently** in its own single-node slice
  (`single_node_graph`, `search/slice.py`) with a plain `TuningSearch` over `LOWERING_PASSES` only (`tile → kernel →
  cuda`). The slice keeps the root kernel + its leaf-op closure and turns every other kernel-input into a synthetic
  `InputOp`; the root op is shared **by reference**, so its body — and thus `op_cache_key` — is byte-for-byte the
  full-graph op's. Lowering-only (never re-running `loop/fusion`) is what keeps that body untouched. Because the inner
  tree holds one op, MCTS explores only that op's forks with `patience` as the op's own budget — `Σ_k n_k` benches,
  never the product. **Leaves are deduped by `op_cache_key`** before iteration: 24 RMSNorm LoopOps across 24 layers
  collapse to one work unit, and the outer `total_us` accumulates `best * multiplicity` so the reward stays
  multiplicity-weighted (bit-for-bit the same as per-node iteration). `OpResult.multiplicity` carries the count;
  positions = `sum(r.multiplicity for r in reward.per_op)`. The progress denominator is the deduped count, so
  Qwen3-Embedding-0.6B's ~14 unique kernels show as 14/14 not 14/337.

**Separability + the structural handoff.** Op-variant forks are separable: every multi-option fork is an in-place `Op`
rebind that leaves the graph unchanged, so whole-graph time is `Σ_k t_k`. Results key structurally (`op_cache_key` =
name-invariant body+knobs digest), so a kernel tuned in its slice transfers to the assembled graph unchanged **and** is
shared across outer terminals — two fusion candidates sharing an identical op reuse its tuning (a DB hit), so fusion
search only pays for the ops that differ. After the best fusion is picked, the assembled `Graph[CudaOp]` (greedy replay
of the DB-best forks over the original graph) is benched **once** for the real in-context whole-graph latency; comparing
it to the `Σ` estimate is the **separability check** — a gap exposes L2 / clock / launch coupling the isolated benches
can't see (in practice <2% for the small graphs above).

**Always re-run, replay from the cache.** The inner search runs for **every** op on every pass — it is never skipped on
prior effort. Replay is cheap, not gated: each benched terminal hits the per-variant `perf` cache (`_TerminalBench`),
so an already-measured variant is served from the DB with no GPU bench. An identical re-run (same prior) re-walks the
same deterministic trajectory → every terminal is a cache hit → zero benches and the same total — idempotent without a
gate. But the global learned prior keeps changing (it refits across ops and runs), so the **same patience** can steer the
MCTS down a *different* trajectory; re-running lets it reach and bench the genuinely-new variants the improved prior
surfaces, replaying the rest for free. (The old `op_effort` "skip already-tuned" gate is gone — it skipped the whole op,
which suppressed exactly that prior-driven re-exploration.) The inner search records the **best whole-slice total**
(`Σ` over the slice's CudaOps, so a split-K main + combine both count) under the LoopOp key via `record_perf`;
`best_per_op_time` prefers that direct row and otherwise walks the `lowering` chain down to the `cuda` terminal.

**Per-kernel GPU parallelism (`--gpus N` / `--devices 0,1,2`).** Because the inner search tunes each unique kernel
independently, the per-op loop fans out across GPUs. The whole tuner is async-only: `run_two_level_tune` (async; the
sync CLI bridges with `asyncio.run` in `handle_tune`) `await`s `_inner_reward_async` per outer terminal, which runs
one coroutine per unique kernel over an `asyncio.Queue` of `len(pool)` device-pinned `CudaBackend`s — each pops a
backend, drives its op's whole inner search via `Pipeline.tune_async` (whose only `await` is the per-terminal bench,
`_bench_terminal_async`), then returns the backend. So `len(pool)` benches run at once, one per GPU. **True single-thread asyncio**: every Python
statement (lowering, DB writes, prior `add_rows` / `maybe_refit` / `checkpoint`) runs on the one event-loop thread
and yields only at the bench `await`, so the shared `db` / `prior` need no locks — the in-flight refit is atomic
between awaits. Each op seeds its `TuningSearch` by `seed + op_idx` (execution-order-independent) and the reward is
a commutative `Σ`, so the per-op DB bests and `total_us` are byte-identical regardless of slot count; only the
learned `prior.json` varies run-to-run (rows arrive in completion order). The **default single-GPU** path is a
one-slot pool whose coroutines acquire the lone worker in `op_idx` order → strictly sequential, identical to the
old serial loop. A backend pins its async worker to a physical GPU via the child spawn env
(`CUDA_VISIBLE_DEVICES`, plus a per-device `DEPLODOCK_GPU_LOCK` suffix so workers don't serialise on one lock) —
never mutating the parent `os.environ`. Parallelism is bounded by the unique-kernel count; devices must be
homogeneous (the tune keys every perf row on one probed `ctx`). Each terminal's `asyncio.run` SIGKILLs its workers
on exit (their subprocess transports bind to that loop); the backend objects persist and respawn on the next
terminal. See `plans/let-s-make-a-plan-glimmering-mist.md`.

**Driving the loop.** `deplodock tune <model_or_ir | --code EXPR>` probes a `Context`, opens the tuning database
(default `~/.cache/deplodock/autotune.db`, overridable via `DEPLODOCK_TUNE_DB`), and calls `run_two_level_tune(...)`.
On completion it prints one `done: N fused terminal(s) in Xs` line — the deployable numbers come from the optional
`--bench` step below. The DB accumulates rows across runs; re-running resumes from the cached state.

On default verbosity (and a tty) a `commands/tune_progress.TuneProgress` draws a live single-line bar — completed/total
tuned op leaves plus a `<kernel> <current us> (best <best us>) <knobs>` tail. The current latency is fixed-width and the
variable-length `pipeline.variant_label` knob string sits last, so the prefix up to the knobs stays put as the
per-variant latency changes (only a new best, which is rare, shifts the trailing part — no flicker). It is threaded as an optional `progress=` through `run_two_level_tune` → `_inner_reward_async`
(duck-typed, so the search package keeps no dependency on `commands/`): one op leaf ticked per kernel, the tail updated
per benched variant (read off `TuningSearch.last_stats`). Under `--gpus N` the tail keys by a per-op `slot` and joins
every in-flight kernel with ` | ` (one per device); single-GPU shows the one active kernel as before. `-v` disables the bar (the per-`[tune]` INFO lines show
progress instead); `-q` is quiet (errors only). `--bench` re-benches the tuned winner at **-O3** (deployable, not the -O1 ranking pass) after tuning —
the assembled full model **against the real torch module** (eager / `torch.compile` / Deplodock, via the bundle
plumbed from `load_or_trace` → `commands/run.bench_full_model_real`; a symbolic graph benches the torch side on
hint-tiled inputs — `_hint_sized_inputs` grows each symbolic input axis to its `Dim` hint, matching the hint-sized
synthetic inputs the deplodock side resolves to, and the table notes `benched at seq_len=… (symbolic hint)`) and
each kernel's `.torch.json` provenance
reproducer (re-lowered greedily so the tuned forks are picked) vs eager / `torch.compile` / Deplodock via
`commands/run.bench_lowered_vs_torch`, printing
full-model + per-kernel tables and (when a dump dir is set) an HTML chart at `<dump-dir>/kernels.html`. Every bench —
the tuning sweep, the full-model table, and the per-kernel rows — times under **CUDA graph capture** by default (pure
GPU time): the torch side replays the frontend graph op-by-op and would otherwise be dispatch-bound, with the GPU
starving between aten launches; deplodock's cupy launch loop has the same exposure for small kernels. Capture is
all-or-nothing per comparison: if any backend fails to capture, that bench retries fully uncaptured and the table
prints a fallback note. Each `perf` row records whether its measurement was captured (the `captured` column); on write,
a captured measurement supersedes a wall-semantics one for the same key regardless of median (never the reverse), so
old rows keep serving replay and prior training and upgrade in place as re-tunes measure them captured. Recorded
goldens keep their original numbers until the next `tune-golden` re-record. See the `capture_graphs` section in
`backend/cuda/ARCHITECTURE.md`.

**Search dynamics.** Each level reuses the **same** SP-MCTS (`search/policy/mcts.py`) — outer over structural forks, inner
over one op's forks — with max-Q normalized UCB1:

- **Selection** is PUCT (`_select`): `Q_norm(c) + ucb_c · P(c) · sqrt(N_parent+1)/(1+N_c)`, where
  `Q_norm = child.best_reward / global_best_reward`, `reward = 1 / median_us`, and `P` is the softmax over the learned
  `CatBoostPrior`'s scores of the sibling set. The prior is the sole signal — greedy, the static
  `TileOp.score` tiebreak, and the `+∞`-unvisited UCB rule are all gone (see the learned-prior section).
- **Expansion** is implicit: `Run.drive` pops a node and runs one rule batch; every fork pushes one new child per
  alternative. The tree mirrors the graph's fork lineage.
- **Simulation** is the actual `await backend.benchmark_async(...)` call on the terminal — for the inner search that is
  one real GPU run of a single-kernel slice per leaf.
- **Backprop** walks the popped candidate's `parent` chain up to the root, updating `visits` and `best_reward` so future
  UCB1 calls see the new max-Q.
- **Patience** counts terminals visited *since the last new global best*; when it exceeds `patience` (`--patience N`,
  default 50), `TuningSearch.stop_reason` is set and that level's `Pipeline.tune_async` / `Run.drive` exits. The inner
  search records `∞` effort when it instead drains its tree (no patience stop).

**Learned prior (`search/prior/`).** ONE global `CatBoostPrior` across every kernel, GPU and nvcc setting — not per-op,
not partitioned by regime. Op structure (`S_*`) and the host/hardware regime (`H_*` — GPU compute capability + nvcc opt
level, from `Context.features`) are **features in every row**, not a cache key. Training signal is **value-of-position**:
real benches exist only at leaves, but the prior ranks partial-knob siblings at every fork level, so the label for any
node is the best (min) median latency µs over its benched descendants (`1/best_reward`, the max-Q `record_terminal`
maintains on `SearchNode.best_reward`) — the prior regresses on **latency**, and the reward conversion (`1/û`) lives in the
MCTS `_select` loop, not the stored data. `TuningSearch._collect_rows` walks the live tree and emits `(knobs, label)` for every node with
a benched descendant (leaves **and** branches). A directly-benched **leaf** uses its `realized_knobs` — the FULL config
read off the resolved graph's op in `observe` (so knobs stamped at deterministic, non-forking lowering steps —
`FK`/`BK`/`SPLITK`/`STAGE`/… — are captured, not just the `BR`/`FM`/`FN` that come from multi-option forks). A **branch**
has no realized knobs of its own, so it falls back to `_node_knobs` (its partial `fork.knobs` prefix under the op's
`S_*`/`H_*` base), carrying the value-of-position label. `knob.knob_features` vectorizes. (Before this, `_collect_rows`
used only the fork prefix for every node, so the prior was blind to every deterministically-stamped knob — e.g. it never
saw `FK`, the dominant knob for a reduction, and greedy stayed on `FK=1`.)

Why CatBoost (chosen by `scripts/prior_bakeoff.py` over a multi-op tuning dataset): the model's greedy pick must not run
off to a degenerate corner. A linear model (the former `BayesianRidgePrior`) is monotone in every knob, so its optimum is
always a corner of the candidate box — the `BR=1` blow-up (4us → 232us / invalid kernels). Any **bounded** tree ensemble
is off-manifold-safe (an un-benched extreme inherits the nearest leaf's value), and among them CatBoost also generalizes
to an *untuned* op near-perfectly (leave-one-op-out pick ratio ~1.0 vs xgb/lgbm 1.18, rf 1.31) thanks to ordered
boosting + oblivious trees. So one global CatBoost prior is good enough on a new op that it is **not refit within an op's
own search** — it is a fixed model per run.

The dataset is bounded + batched (`base.Prior`): each tuned op's value-of-position rows stream into a reservoir-sampled
dataset capped at `MAX_ROWS` (100k — Algorithm R keeps a uniform sample of all rows ever seen, across runs), and the
model refits (`maybe_refit`) on a **dataset-size-tiered cadence** (`REFIT_SCHEDULE`) — frequently while data-poor, then
coarsening: every 100 rows up to 1k, every 1k up to 10k, every 10k from there on — then checkpoints. So the model warms
up fast on the first op (~10 refits inside the first ~1k rows) and settles to ~once per op once large. End-of-run does a
`maybe_refit(force=True)` so even a small tune that never crossed a tier still ends with a fitted model (above `min_rows`). The checkpoint is
a JSON file (`config.prior_path()`, `~/.cache/deplodock/prior.json`) holding the CatBoost `cbm` blob (base64) + the
dataset, via `deplodock.storage`; `tune` writes it, `compile` / `run` read it.

**-O3 deployable samples.** The sweep compiles at `-Xcicc -O1` (fast, but a *ranking* signal — it ties configs that
differ at -O3, e.g. a reduction's `FK` or a warp tile's `WARPSPEC`). So whenever a bench lands **within
`DEPLODOCK_O3_TOL` (default 15%, `config.o3_tol`) of the best -O1 so far** — flagged `TuningSearch.last_o3_worthy`, a
band *wider* than a strict new best so near-tied contenders all qualify — the engine re-benches it at `-Xcicc -O3`
(`_rebench_o3`) and `observe_o3` records an extra row with the same realized knobs tagged `H_opt=3` (the deployable
regime). Each config is re-benched at most once (`_o3_done`, keyed on a value-stringified knob signature). The `H_*`
feature lets the -O1 (broad) and -O3 (near-best) rows coexist; `compile` / `run` run at -O3 (`H_opt=3`) so greedy ranks
by the deployable rows and reaches the true optimum. Widening from winner-only to a tolerance band is what lets configs
that tie at -O1 but diverge at -O3 (the fp16 warp WARPSPEC / occupancy split) each get an -O3 truth sample — see
`plans/golden-sweep-report.md`. The
`nvcc_flags` override rides the bench request to the worker (`config.nvcc_flags_override`), so only winners pay the -O3
recompile and the cubin cache keys on the flags.

How the prior enters selection — **PUCT is the only rule** (`_select`): the prior is the *sole* signal; greedy-tiebreak and
the `+∞`-unvisited UCB rule are gone.

    score(c) = Q(c) + c · P(c) · √(N_parent+1) / (1+N_c)

`Q = best_reward/global_best` (0 if unvisited); `P` is the prior's **predicted reward** on the same scale — the prior
predicts latency `û(c)`, which `_select` converts to reward (`1/û`) and normalizes by the same `global_best` as `Q` (no
softmax); `c = --ucb-c`. A confidently-slow sibling (large `û` → small `P`) gets a tiny exploration term → it is
deprioritized instead of force-benched (no forced breadth). The prior is ALWAYS consulted — the `FallbackPrior` returns
the learned `CatBoostPrior`'s prediction once trained and the `AnalyticPrior`'s heuristic cold (only a non-positive score
falls back to a uniform `P = 1`). The enumeration is itself ordered by the `AnalyticPrior` (`_prior_order`), so the cold
MCTS front-loads good variants and a single `tune` pass reaches the prior-best within patience. The end-of-run sanity
block (silly-pick rate warmup-vs-post, self-calibration) prints once for the global prior.

**Greedy uses the prior too — and flattens.** `Pipeline.run`'s `greedy_decide` (the `Run.resolve` decide for
`compile` / `run`) lazy-loads
the global `Prior` via `load_prior` (the `FallbackPrior` over `CatBoostPrior` + `AnalyticPrior`). The lazy fork tree is an
**MCTS** structure — it stages knob choices across levels (`BR` → `BM/BN` → `FM/FN`) so MCTS pays one node per pop.
Greedy must NOT walk it level-by-level: a branch carries only a *partial* tile, and `knob.knob_features` can't compute the
tile's area / occupancy until `FM/FN` are pinned, so the prior is **blind at the `BM/BN` choice** and defaults to `BN=16`
for every shape (it also defaulted the warp-vs-scalar tier by emission order, not the prior). Instead greedy **flattens**
each fork point to its complete leaves — `fork.flatten_leaves` expands branches depth-first (cheap; only knob dicts,
materialization stays deferred to the one chosen leaf) — and picks the lowest `Prior.mean_scores` over the full
`{H_*, S_*, complete-knob-row}` vector the prior trained on, in **one batched `predict`**. The pick equals scoring the
flat candidate set, invariant to the tree's level order. Cold (no trained `CatBoostPrior`) the `AnalyticPrior` ranks
(including the positive `MMA_tier` warp-preference that replaced the old warp-first emission order); only if `load_prior`
returns nothing does it take option-0 (the first leaf). (Greedy benches nothing, so it can only *use* a prior, never
train one — and it is not a `Search` at all: a deterministic resolution has no frontier, so its process facts live on
the returned `Decision` trace, never on policy-object state.)

**Greedy validity fallback.** The prior ranks by *predicted latency*, which can rank a tile that fails `validate(ctx)`
(smem / thread budget) first — `tune` benches-and-skips it, but greedy benches nothing. So when a deterministic compile
leaves a node un-lowered (its only lowering rejected at `validate`), `Pipeline.run` blocklists that tile's
`tile_identity` (its planner knobs) and **re-resolves**: `greedy_decide(blocked=…)` drops the matching leaf from the
flattened set and picks the next-best (the valid runner-up is usually ranked right below). Bounded by
`_MAX_GREEDY_RETRIES` (each retry blocks ≥1 fresh tile or stops). Only the offending leaf is dropped — its full-row
`tile_identity` never matches a different tile, so no other candidate is pruned. When the retry budget exhausts with the
node still un-lowered (a *learned* prior can rank many over-budget tiles above the first in-budget one — e.g. a prior
trained on big square matmuls extrapolating a >smem-cap tile onto a tiny projection, which crashed the golden sweep at
`qwen3_06b.q_proj.s32`), `Pipeline.run` takes one last **option-0 (emission-order) resolve** (`greedy_decide(prior=None)`):
it ignores the prior, and the planner emits a budget-safe tile first, so it lowers whenever any in-budget tile exists.
Only when even option-0 overflows (the rule genuinely has no in-budget option) does `_raise_on_unlowered` fire the loud
`LoweringError` — the single-option guardrail is preserved.

**Reading the result.** `_bench_terminal_async` writes one `perf` row per CudaOp per `(context_key, backend)` keyed on
`op_cache_key`, plus a `lowering` edge per rewrite hop carrying the knob delta the rule stamped (and the inner search
adds the whole-slice total under the LoopOp key) — the bench record / training data. A subsequent `deplodock compile` /
`deplodock run` does NOT replay these DB forks (the greedy DB→fork replay was removed with the learned prior); instead
`greedy_decide` picks each fork from the global `Prior` (`FallbackPrior`: learned `CatBoostPrior` once trained, else the
`AnalyticPrior`'s `mean_score` argmin — lowest predicted latency) — see "Greedy uses the prior too" above.
`run_two_level_tune` assembles its final graph the same way.

**Stub backend.** With `backend=None`, `Pipeline.tune_async`'s `_bench_terminal_async` short-circuits to `latency_us=1.0` and
persists nothing (`Pipeline.run` skips benching entirely without a backend) — so a GPU-less compile or sweep never
clobbers tuned rows with a stub.

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK`),
candidate `hints` (advisory — the rule still validates structural fit), and a short help string. Rules declare them as
module-level constants and stamp values into `TileOp.knobs` dicts; the autotuner reads those dicts back as the per-hop
knob delta in the `lowering` table. The registry (`knob.registry()`) auto-collects every `Knob` instance in every loaded
rule module — no manual registration.

**Pinning knobs from the environment.** Two equivalent forms:

- **Per-knob:** `DEPLODOCK_<NAME>=<value>` (e.g. `DEPLODOCK_BK=32`). Read by the rule that owns the knob (via
  `Knob.narrow`) or by `compiler/tuning.py`'s heuristics. The `DEPLODOCK_<NAME>` env-var key is built by
  `config.knob_var` and the value read via `config.knob_raw` / `config.int_env` — `deplodock/config.py` is the single
  owner of `os.environ` for all `DEPLODOCK_*` vars; `knob.py` keeps the `Knob` descriptor's per-type decode.
- **Aggregate:** `DEPLODOCK_KNOBS="K1=V1,K2=V2,..."` (e.g. `DEPLODOCK_KNOBS="BK=2,BM=16,BN=128,FM=8,FN=8,STAGE=111"`).
  Parsed once at `knob.py` import via `apply_knobs_env()`, which splats each entry into the corresponding
  `DEPLODOCK_<K>` env var (via `config.set_knob(..., overwrite=False)`) so all the per-knob readers pick it up
  uniformly. An explicit per-knob var wins over the aggregate (so `DEPLODOCK_BK=4 DEPLODOCK_KNOBS="BK=2,BM=16"` ends up
  with BK=4, BM=16).

Pinning replaces tuner choice: the rule sees the env value and emits exactly that variant instead of forking. Useful for
reproducing a tune-time variant from CI logs, A/B-comparing two configs, or pinning a known-good config in a Makefile
recipe.

Pinning is **authoritative** — an env value outside the knob's hint tuple is honored, not silently dropped. `Knob.narrow`
returns `(pinned,)` regardless of hint membership; downstream structural gates (divisibility, threads-per-CTA budget,
TMA eligibility, …) still apply, so a structurally invalid pin yields an empty enumeration and the per-call-site
fallback (`_enumeration._run(apply_pins=False)`) takes over. This lets a tile shape that the planner wouldn't reach
on its own — e.g. the article's BM=8, FM=26, fat 208×128 matmul tile — be explored manually, while peer kernels with
incompatible divisibility still get a sensible default.

**Registered knobs.** All knobs in `passes/lowering/tile/*.py`:

| Knob                 | Type      | Owning rule                   | What it controls                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------------|-----------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `BK`                 | INT       | `010_partition_loops`         | Per-stage K-chunk size for matmul reductions; intra-CTA K-loop trip count = `K / BK`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `SPLITK`             | INT       | `010_partition_loops`         | Cross-CTA K-split factor for matmul; `1` = no split. Multiplies CTA count, requires a final combine.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `BN`                 | INT       | `010_partition_loops`         | CTA innermost THREAD-axis width (the column tile each warp covers).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `BM`                 | INT       | `010_partition_loops`         | CTA outer THREAD-axis width (matmul only — the row tile each warp covers).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `STAGE`              | BINMASK   | `020_stage_inputs`            | Bitmask over ranked candidate buffers — char `i` = stage buffer `i`. Selected buffers fold into one `StageBundle` with per-source `Source` entries.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `FM`                 | INT       | `010_partition_loops`         | Register-tile factor along the matmul M (output row) axis; per-thread cell-grid height.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `FN`                 | INT       | `010_partition_loops`         | Register-tile factor along the matmul N (output column) axis; per-thread cell-grid width. The planner emits one outer `RegisterTile(N_r)` around `{Init, K-reduce, Write}`; the Kernel-IR replicator + `dedup_replicated` pass produce the textbook blocked-GEMM shape (N-invariant Loads kept single-copy, N-dependent Accums replicated).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `BR`                 | INT       | `010_partition_loops`         | Cooperative-K thread count (1 = pure serial chunked reduce); BR > 1 routes through the cooperative reduce path with cross-thread combine. With BN=BM=1 the combine spans the whole CTA (any BR: warp shuffle / hierarchical / smem tree-halve by size); alongside free-axis threads (BN·BM > 1, strided-cooperative rows) BR clips to powers of two ≤ warp_size and the combine is a SEGMENTED warp shuffle over each row's BR lanes — K_c is the innermost THREAD axis, see `_combine.cooperative_combine_geometry`.                                                                                                                                                                                                                                                                                                                                                                                       |
| `FK`                 | INT       | `010_partition_loops`         | Reduce-axis multiple-accumulator factor (non-matmul reduces). Strip-mines the per-thread K serial loop into `FK` independent accumulators (a `RegisterTile(K_f, reduce=True)` inside `K_i`) for ILP; `010_split_register_axes` replicates the wrapped `Accum` into `acc_0..acc_{FK-1}` and appends a cross-accumulator tree-fold after the K serial loops, so the materializer/combine see one `acc`. Swept only as a divisor of the per-thread K-chunk extent, capped by `FK·FM·FN ≤ _MAX_CELLS_PER_THREAD`. **fp16 scalar matmul** reuses `FK` as the half2 accumulation-window length (= even `bk`): the planner keeps the FK=1 fp32 structure + stamps `FKWIN`, and `kernel/015_pack_fk_window` rewrites the window K loop into `__hfma2` packed multiply-adds over a `__half2` accumulator with a widen+horizontal-sum flush into the fp32 master each stage — bounded fp16 error for 2× packed throughput. `FK=1` (and fp32/bf16/MMA) is byte-identical to the pre-FK planner (it ranks first in the greedy tiebreak). See `plans/fk-register-tile-reductions.md` and `plans/fk-half2-fp16-matmul.md`. |
| `WN`                 | INT       | `010_partition_loops`         | CTA innermost WARP count along the matmul output N axis (warp-tier MMA tiles only).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `WM`                 | INT       | `010_partition_loops`         | CTA outer WARP count along the matmul output M axis (warp-tier MMA tiles only).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `S_*`                | FLOAT     | `loop/fusion/992_stamp_structural_features` | The LoopOp's structural features (`ir/features.py:structure_features`): a flat `S_`-prefixed dict — stmt/op histogram (`S_n_load`, `S_pw_*`, `S_reduce_*`, …) + loop extents (`S_ext_*`) + operand dtype multiset (`S_dtype_*`). Not tunable — identity facts that make a knob dict a complete variant identity (the learned prior's feature vector). Stamped last in the loop dialect so every downstream keying (op_cache_key, tune DB, plan-stamp) sees one consistent knob set; `knob_features` turns the whole knob dict into the model feature vector. Skipped by `format_tuning_knobs` (facts, not tuning decisions). |
| `MMA`                | STR       | `010_partition_loops`         | Three-way control for warp-tier MMA (tensor-core) matmul enumeration: falsy (`0`/`false`/…) forces the scalar-only path (debug / fallback); truthy (`1`/`true`/…) or unset (the default) auto-enumerates every eligible atom kind; any other value names an atom kind (e.g. `mma_m16n8k16_f16`) — enable **and** pin that kind, incl. the force-at-any-arch pin-only path. `DEPLODOCK_ATOM_KIND` is its env **alias** (`Knob.aliases` — either spelling works; the primary `DEPLODOCK_MMA` wins when both are set). Not an autotune fork: the tuner picks warp-vs-scalar through the `ATOM_KIND` sibling subtree. Eligibility (`tile/_atom.is_atom_eligible`) mirrors what the cell tagger can classify **by construction**: gate and tagger call the ONE A/B classifier (`tile/_atom.classify_matmul_operands` — K-in-last ⇒ A / K-in-first ⇒ B, plus a positional fallback for K in a *middle* index dim: a load whose single K dim sits after every other var-carrying dim ⇒ A, before every one ⇒ B, e.g. the SDPA cone-split's 4-D V slab `(0, k, 0, n)`). A transposed-B matmul (Q@K^T, Linear's raw [N, K] weight: both loads K-in-last) is recovered from the output coordinates (`classify_matmul_operands(..., out_index=)`: the operand sharing the M/row output var is A, the N/col one is B) and reaches the tensor-core tier with B read gmem-direct via `dpl_mma_load_b_gmem_trans` — `[N, K]` is the native `mma.row.col` col-major B, so no `ldmatrix.trans` (carried on `Mma.b_trans`; `020_stage_inputs` leaves the transposed-B operand unstaged, the staged ldmatrix-no-trans path being future work). The `SPLIT_CONE` two-producer cut can alternatively re-materialize such a B at [K, N] for a canonical consumer. Declared in `_enumeration.py`, decoded by `mma_mode()`; sits in `_PLANNER_KNOBS` so the enumeration-memo pin snapshot covers it (alias included, via `Knob.raw`).                                                                                                                                                                                                                                                                                                                          |
| `HOIST_COMPUTE`      | BOOL      | `030_hoist_invariant_compute` | False (default) → inline-fuse Stage; True → single transport Stage + a `StageBundle.compute` phase. Autotune fork.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `PAD_SMEM`           | BOOL      | `070_pad_smem`                | True → apply per-source ``+1`` smem pad to break bank conflicts; False → leave the slab dense. Autotune fork.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `GROUP_M`            | INT       | `025_swizzle_blocks`          | L2-friendly CTA-swizzle row-group size (Triton/CUTLASS convention). Default `8`; `1` is the global escape hatch (row-major decode). Stamped on the outer matmul GridTile's `swizzle_group_m` field; the renderer emits a Triton-canonical `blockIdx.x` remap so groups of `GROUP_M` CTAs walk down M before stepping N, sharing A's row tile in L2. Self-disabling on tiny / tall-skinny matmuls via the runtime `min(GROUP_M, num_m - first_m)` clamp.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `RING`               | INT       | `040_use_ring_buffers`        | Ring-buffer depth (and pipeline stages) for BUFFERED/ASYNC/TMA staged K-outer loops. `2` = classic double-buffer; `3`/`4` = CUTLASS-style multistage (pruned when the per-stage smem × N exceeds the cap). The greedy default orders the surviving variants by occupancy — front-loading the deepest depth that still keeps **2 CTA-blocks/SM** resident (`2 × depth × per-stage ≤ cap`), since past that the kernel drops to 1 block/SM and runs slower (measured 2048² fp16: 128×128 depth-3 = 115 µs vs depth-4 = 136 µs). This reorder fires **only for single-`StageBundle` kernels** (a pure GEMM, where the ring slab is the whole dynamic-smem footprint so the keeps-2 test is exact); a fused multi-bundle kernel (SDPA's QK+P@V) carries an intermediate cross-bundle workspace that dominates the materialized smem and is invisible to the ring-byte budget, so it keeps the shallow-first default (depth-2, always downstream-valid) — the autotuner still explores its deeper rings. `DEPLODOCK_BUFFER_COUNT` is its env **alias** (`Knob.aliases` — either spelling works).                                                                                                          |
| `TMA`                | BOOL      | `050_use_tma`                 | Promote BUFFERED/ASYNC bundles to TMA. `1` = force (hard-fail on ineligibility), `0` = skip the pass. Default on for Hopper+. A symbolic *innermost* gmem dim is normally declined (its runtime extent gives an unaligned above-inner stride → `cuTensorMapEncodeTiled` `CUresult=1`), but a symbolic inner that is a **provable multiple of the 16 B alignment unit** is accepted (`_inner_stride_aligned`) — the demoted symbolic-N B operand `xnb[…, K, N]` is padded to exactly that (`_split_demoted._pad_inner_for_tma`, inner rounded up to a 64-multiple) so the dynamic rotary QK^T reaches TMA + warp-spec like its static twin. **Masked-K** (symbolic *reduce*) sources reach TMA too: the reduce overhang must read 0, which TMA's hardware OOB zero-fill delivers on the **middle-K** B operand (V, allocated at the real `seq_len`, so its descriptor globalDim is `seq_len` and coords past it zero-fill) — binding every overhang product to 0 regardless of the (padded) A operand's overhang, which the zero-init-reused scratch keeps finite. `040_use_ring_buffers` rings a masked-K bundle only when the shared `050.tile_reaches_tma` predicate confirms the whole tile is TMA-eligible, so a masked-K bundle is never stranded on cp.async / a synchronous double-buffer (neither can zero the partial-K slab) — it stays SYNC with the `_stage_expand` ternary otherwise. The symbolic-K_o ring drains correctly at every runtime `K_o` down to 1 (validated seq ∈ {16…700}); the deployable P@V is the demoted softmax-prob cone + clean symbolic-K gemm. Beyond the per-`Source` shape/alignment checks, three gates decline shapes whose failure would only surface on the device: every collapsed per-dim box extent must be ≤ 256 (`cuTensorMapEncodeTiled`'s `boxDim` limit — an oversized box, e.g. the scalar matmul's `BM·FM` M box, compiles fine and dies at launch with `CUresult=1`; the runtime encoder in `backend/cuda/_tma.py` double-checks and names the offending dim); the bundle's `serial_outer` K loop must not be nested inside a serial loop with trip count > 1 (the materializer inits the ring mbarriers once at kernel entry, so a re-entered pipeline starts at stale slot parity and deadlocks — the Qwen3 `k_linear_mean_reduce` FM=2 hang; cp.async has no cross-iteration phase state, so the fallback handles re-entry fine); and a double-buffered (`buffer_count > 1`) NONE-swizzle bundle's per-slot box footprint must be a 128 B multiple **at the true element width** (`strict_slot_align`) — the materializer's slot pad sizes its 128 B threshold off the fp32 `BYTES_PER_ELEM`, so a pure-reduction fp16 slab whose box is a single 32-elem axis (64 B) read as already-aligned, stayed unpadded, and the second ring slot landed at a 64 B offset → `cp.async.bulk.tensor` `CUDA_ERROR_MISALIGNED_ADDRESS` device hang (the #244 `k_linear_mean_reduce` wedge; matmul slabs collapse `BK·BN·FN` into a ≥128 B box and stay on TMA, swizzled/mma slabs align via their swizzle atom, single-slot bundles sit at the aligned base — none set `strict_slot_align`). All repro'd + locked in by `tests/compiler/passes/test_use_tma_gates.py` + `test_tma_smem_alignment.py` (compile-only) and `test_knob_pinning.py` (static/dynamic accuracy); see `plans/qwen3-embedding-layer0-tune-findings.md` and `plans/qwen3-embedding-layer0-static-vs-dynamic-tune-findings.md`.                                                |
| `ASYNC_COPY`         | BOOL      | `060_use_async_copy`          | Promote double-buffered (BUFFERED) bundles to cp.async (ASYNC). `0` = keep the synchronous double-buffer. Default on for sm_80+.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `PIPELINE_STAGES`    | BOOL      | `080_pipeline_stages`         | Software-pipeline async-staged K-outer loops into prologue/main/epilogue. `0` = keep the depth-1 staged loop.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `WARPSPEC`           | BOOL      | `085_warp_specialize`         | Warp-specialize TMA staging: producer warp(s) issue TMA, consumer warps wait + reduce. Autotune fork on depth-2 TMA rings (so `040_use_ring_buffers` front-loads `RING=2` on the warp tier to keep it eligible). Eligibility also requires the bundle be **reachable by the producer split** — `_split_by_role` recurses only through `serial_outer` / `RegisterTile` / `AtomTile`, so a bundle under any other wrapper (e.g. the fused linear+mean kernel's `SerialTile(kind='plain')` fragment loop) would strand the TMA issues in the consumer branch and deadlock every consumer `mbarrier.wait` (the Qwen3 `k_linear_mean_reduce` hang — `plans/qwen3-embedding-tune-hung-kernel.md`); such shapes stamp WS=False. Both consumer tiers: scalar `ThreadTile` (pointwise / coop-reduce) and the warp-tier MMA tower's `WarpTile` (`consumer_is_warp`). On the **64×64 4-warp** fp16 mma.sync tile WS=1 is the measured win (≈17%: 94 µs vs 115 µs at 2048²) and both greedy and the tuner now pick it; it was ~neutral at the old 128×128 tile, where the gap was mma-schedule-bound. The WS=1 fork is **emitted first** for the warp tier (option-0), the deterministic tie-break the cold picker takes when the prior ties WS=0/WS=1 (the `AnalyticPrior` has no WARPSPEC feature), so it deploys the win cold instead of taking WS=0 and never benching WS=1 (the fp16 cliff in `plans/golden-sweep-report.md`). `DEPLODOCK_WARP_SPECIALIZE` is its env **alias** (`Knob.aliases` — either spelling works).                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `NOATOMIC`           | BOOL      | `017_atomic_free_splitk`      | Replace `SPLITK > 1`'s atomicAdd output with a workspace + sibling reduce kernel (deterministic accumulation). `DEPLODOCK_ATOMIC_FREE_SPLITK` is its env **alias** (`Knob.aliases` — either spelling works).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `SPLIT_CONE`         | BOOL      | `005_split_demoted`           | Split a demoted matmul's computed multiply-operand cone(s) into producer kernel(s) + the clean gemm — one producer per cone, each materialized over exactly the axes it reads, an N-reading cone at […, K, N] so the consumer keeps the canonical B layout (see the demoted-matmul split section above). Stamped on `op.knobs` at offer sites only — `False` = considered-and-declined, `True` = every split kernel — the rule's idempotence guard and the prior's training signal (absent = never offered). Deliberately declares no `off=` value: `_off_fill_pass` would stamp an off-default onto every knob-bearing TileOp at the pass boundary, erasing the absent-vs-declined distinction. `DEPLODOCK_SPLIT_CONE=1/0` pins the branch.                                                                                                |
| `FLASH`              | BOOL      | `loop/recognize/010_recognize_flash` | Fuse SDPA into a single streaming online-softmax kernel (the `Monoid` carrier) instead of the score-materializing `010_sdpa` decomposition — tiles the KV (reduce) axis, never materializes the `[S_q, S_k]` score matrix. Recognition is a **Loop-IR pattern-recognition** pass (`loop/recognize`, see that pass's own section) that runs AFTER the entire `loop/fusion` fixpoint settles — NOT interleaved — with NO decomposition-stage change. After fusion a non-causal SDPA is two `LoopOp`s — the scaled scores `X = (Σ_dd Q·K)·scale` and the softmax-then-P@V kernel `Σ_kv softmax(X)·V` (rowmax + rowsum-of-exp + normalized P@V in one body — the online-softmax pattern in one place). The pass anchors on that softmax-P@V kernel (tell-tale `maximum` rowmax Accum + `exp` + a P@V sum feeding the output), reads `X` / `V` off its body, and recovers the score two ways: **synthetic** (`build_flash_frag`) when `X`'s producer is a clean scaled-QK (Q / K read as plain Loads, disambiguated by index — the QK operand whose seq index matches the score's row/M axis is Q), and **recovered** (`build_flash_recovered`) when the producer is fused (RoPE / GQA index / scale / mask inline → Q/K are computed SSA values, not Loads — **real decoder/embedding layers**): the producer's score body is inlined wholesale (RoPE rides along) and the consumer's V-load / output indices are recovered (the GQA `head // group` index + any o_proj reshape ride along). The scores kernel orphans and is removed. One independent streaming softmax per output element — the **scalar tier**; the score is recomputed per output dim `d`, so on a real layer the fused kernel is *correct but currently slower than the split path* (≈5× on Qwen3-Embedding layer 0 — the recompute), which the **tensor-core P@V tier** (Part C, future) fixes by computing the score once and carrying `O` as a register fragment. Masking and GQA are recovered **structurally** from the fused body (no frontend provenance): the score feeding the rowmax `Accum` is the bare score Load (no mask), `add(score, Select(kv ≤ m))` (**causal** — the lifted causal `IndexMapOp` bias → the per-element `causal=True` nest), or `add(score, Load(mask))` (an explicit broadcast additive bias, the HF `(1,1,S,S)` whole-model mask → a per-`(m,kv)` mask Load in the nest); the **GQA** group is the `q_heads // kv_heads` shape ratio, deployed as a `head // group` K/V index (no materialized broadcast). Detecting the mask is a correctness requirement, not an optimization — a masked SDPA that matched the anchor but built an unmasked nest is silently wrong. Scope: static OR dynamic (symbolic `seq_len` on Q/K/V dim -2 — one cached kernel carrying `int seq_len` serves every runtime size, the symbol landing on both the masked-row M and the symbolic reduce), causal / non-causal, optional additive mask, GQA, and RoPE-fused producers (so flash now deploys on Qwen3-Embedding layer 0). Anything ineligible (symbolic non-seq, non-broadcastable mask, indivisible heads, an unrecoverable producer, or `FLASH` off) leaves the `010_sdpa` decomposition untouched, so the default is unchanged. Read today from the `DEPLODOCK_FLASH=1` env pin; the two-level `OptionFork` offer + `AnalyticPrior` cold-start term are a follow-up. See `plans/online-softmax-flash-attention.md` and `plans/masked-gqa-mma-flash-attention.md`. |

`BINMASK` parsing accepts a binary string (`"101"` = bits 0 and 2 set, char `i` = bit `i`), the keywords `"all"` / `"none"`,
or a decimal / `0x`-hex int clamped to the candidate width. `format_tuning_knobs` drops `BOOL` knobs from the rendered
`knobs=` line — they're treated as pass-presence markers, not values.

`HOIST_COMPUTE` is an autotune fork: `030_hoist_invariant_compute` emits both variants per fusable cone in a fixed
order (inline-fuse first as the greedy default — smaller smem, works on every architecture). Honors
`DEPLODOCK_HOIST_COMPUTE` for one-off pinning. `PAD_SMEM` follows the same shape in `070_pad_smem`: both polarities
fire whenever any source has a fixable conflict; the greedy run picks pad-on first. Honors `DEPLODOCK_PAD_SMEM` for
one-off pinning.

One smem pad is **not** an autotune fork: the **masked-K MMA slab** alignment pad, stamped intrinsically on the
`Source` at creation by `020_stage_inputs._masked_k_mma_pad`. A symbolic-reduce (`kmask`) operand staged for a warp
matmul — the SDPA P@V softmaxed `P` — lands in a flat `[…, M, K]` smem slab read by `ldmatrix.x4`; when the M-row
stride (the innermost block-scaled alloc-extent) is a 128 B multiple, the ldmatrix M-row lanes all alias one bank
(the 3.67M-load-conflict storm in the Qwen3-Embedding dynamic P@V — NCU 38 µs @ 26 % occ). `070_pad_smem` can't fix
it: `kmask` pins masked-K to the SYNC transport (which `070` skips) and the source is block-stamped (which `070` also
skips, since its `+1` pad breaks ldmatrix's 16 B alignment). So `020` pads the innermost cache dim by one 16 B
ldmatrix chunk (`16 // elem_bytes` elements) — stepping the stride off the alias while keeping every row 16 B aligned.
It's intrinsic (not a fork) because it's a near-strict win with no misalignment penalty, so greedy deploys it without a
re-tune (`070` then self-skips the already-padded source); the result is numerically transparent and drops the P@V
consumer's conflicts ~3.67M → ~1000 (NCU 38 → 27 µs). A flat `[M][K]` slab can't reach the static path's 0-conflict
floor — that needs its swizzle-atom-wide K-subtile relayout — so this is the deployable flat-slab fix. See
`plans/fused-symbolic-pv-smem-staged.md`.

## Pass directories

Pass files are numerically prefixed so `sorted()` pickup is
deterministic. Pick a fresh prefix when adding a rule; the pass loader
ignores the prefix itself — it's only for ordering readability.

| Pass                       | What rules do                                                        |
|----------------------------|----------------------------------------------------------------------|
| `frontend/decomposition/`  | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused ops like `rms_norm`/`layer_norm`/`softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s. Each rule emits broadcast-explicit IR via `_broadcast.broadcast_to`. |
| `frontend/optimization/`   | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map — prevents trivial layout kernels from blocking fusion. |
| `loop/lifting/`            | `lift_*` rules wrap each surviving tensor primitive (elementwise / reduce / indexmap / gather) in a trivial one-op `LoopOp`. |
| `loop/fusion/`             | `split_shared_indexmap` (runs first) fuses a pure-indexmap `LoopOp` that fans out to ≥2 consumers into **all** of them in one rewrite — it inlines the producer's body into each consumer (reusing `splice_loop_ops`) and dissolves the producer via a single multi-output `Graph.splice` (`output={consumer_id: fused_id}`); a consumer the splicer can't take falls back to a private copy. Then `merge_loop_ops` splices the remaining adjacent single-consumer `LoopOp` pairs via `ir/loop/splicer.py::splice_graph`. The split is what lets the scalar-constant broadcasts torch.export folds into mask/RoPE scaffolding fold into their consumers (inlined as `float x = 0.0f;` literals) instead of surviving as standalone copy kernels (full Qwen3-Embedding-0.6B: 394 → 337 CUDA kernels). `dedup_loads` then drops identical `(input, index)` Loads within each fused body. Naming + structural stamping moved out to the `loop/stamp` pass (below) so they run once after both fusion and recognition. Shared `is_pure_indexmap` / `rename_write_output` helpers live in `_helpers.py`. |
| `loop/recognize/`          | **Pattern recognizers** — rewrite a generic fused op-cluster into a specialized fused kernel, AFTER the `loop/fusion` fixpoint has fully settled (not interleaved — see `recognize_flash`'s docstring for why the placement is load-bearing: an interleaved firing would catch a half-fused score producer and re-materialize the RoPE'd `qk_ew` product). `recognize_flash` is the first: it folds a softmax-then-P@V SDPA into one streaming online-softmax flash `LoopOp` (the `FLASH` knob; `_flash.py` holds the knob + nest builders). Future recognizers (other attention variants, fused-norm patterns) live here. |
| `loop/stamp/`              | `stamp_loop_names` stamps `LoopOp.name` via `provenance.name_for` (e.g. `k_rms_norm_3f2a1b`); `stamp_structural_features` stamps the `S_*` features. Runs last in the loop dialect — after both `loop/fusion` and `loop/recognize` — so every kernel (fused or pattern-recognized) is named / stamped against its final body; the Tile dialect forwards the name onto each emitted `TileOp` (and every dialect below copies it through). The Tile dialect re-runs both (its `008` / `009` aliases) after split-driven re-fusion. |
| `lowering/tile/`           | Tile-IR structural passes — Stage formation, transport (cp.async / TMA), double-buffering, pipelining, smem padding. Order: `partition_loops` → `lower_atom_cell` (MMA-only: rewrites the warp-tier matmul cell into tensor-core form right after partitioning — reading the `Atom` spec off the enclosing `AtomTile.atom` (stamped there by `partition_loops`, no `ATOM_KIND` knob lookup), the `Assign(multiply) + Accum` collapses into a single `Mma` (`c += a @ b`) that carries that `Atom` spec (cell shape + operand dtypes) and names its A (M×K) / B (K×N) operand `Load`s by SSA value. The operand loads stay **plain** — the `Mma` is the sole tensor-core marker. Both flow through every staging pass as ordinary IR (the loads stage like any `Load`; the `Mma` keeps its reduce loop `is_reduce`), so the cell carries its tensor-core intent through the whole tile chain. The final lowering to the `ldmatrix` + `mma.sync` kernel chain is `kernel/005_lower_atom_tile`, which recovers each operand's role from the `Mma`. Idempotent / scalar TileOps skip; see `plans/mma-fragment-factorization.md`) → `gate_splitk_residual` → `stage_inputs` → `hoist_staged_loads_above_mask` (lifts a masked-tile boundary `Cond(decoded < bound, ...)` from `010_partition_loops`: any K-pipeline stmt — `StageBundle` itself, plus `SerialTile` / `StridedTile` whose subtree carries one — is hoisted ABOVE the Cond so the cooperative load fires on every CTA thread (TMA's elected issuer / cp.async's full-CTA fan-out would otherwise be gated out). Un-staged gmem Loads in the hoisted body whose index references a gated var are wrapped in an inner `Cond(predicate, body=cone)` covering their forward SSA cone; every hoisted `Source` gets `gmem_extents` stamped (static ints or the symbolic dim's `Expr`) so the slab fill clamps its gmem read to the runtime buffer bounds. Refuses the lift when a hoisted pipeline reads a name defined by a stmt staying inside the Cond (the fused-prologue shape — hoisting would order the consumer above its definition). Skips `==` Conds (the SPLITK invariant-compute guard) and bare Conds with no staged transport. Deterministic, no knob — split out of `020_stage_inputs` so the staging walk is uniform and the Cond-shape rewrite is focused) → `swizzle_blocks` (default-on
L2-friendly CTA swizzle for matmul-priority TileOps — stamps `GridTile.swizzle_group_m = DEPLODOCK_GROUP_M`,
default 8, so the renderer emits a Triton-canonical `blockIdx.x` remap; identifies matmul kernels via
`TileOp.knobs` rather than the axis-suffix convention because the body-normalizer renames axes by the time
the pass runs) → `unify_sibling_stages` (drops a `StageBundle` Source whose `buf` was already staged by a
prior sibling scope and reverts its consumer Loads back to gmem — keeps the fused RMSNorm + linear `x_smem`
single-allocation invariant when the matmul-side K_i, now visible as a reduce through transparent
`RegisterTile` wrappers, would otherwise re-stage `x`) → `hoist_invariant_compute` → `use_ring_buffers` →
`use_tma` → `use_async_copy` → `pad_smem` → `pipeline_stages` → `mark_unroll`. Coordination (split-K atomic-writes, cooperative-K Monoid emission, broadcast-write guards) is no longer a separate pass: the materializer / Kernel-IR render derives those decisions from `ir/tile/escape_analysis.py` queries against the tile body. Cooperativity is derived from `Accum.axes ∩ ThreadTile.axes`; atomic writes from enclosing `GridTile.axes` vs `Write.index`. `015_gate_splitk_residual` reuses the same `Body.coordination.atomic_axes` signal to identify the split-K block axis without any axis-naming convention or role tag — when SPLITK > 1, it wraps a `matmul_add`-shape linear residual epilogue under `Cond(K_s == 0, ...)` so the residual is atomic-added exactly once across the K_s CTAs (rewrite + predicates live in sibling `_splitk_residual.py`, shared with `010_partition_loops`'s `force_splitk_one` enumeration-time gate). The partition planner's knob globals + per-mode candidate tuples + the pruned `(BN, BM, FM, FN, BK, SPLITK, BR)` cartesian generator + per-mode priority/score functions live in sibling `_enumeration.py` — `010_partition_loops.py` imports the `enumerate_cartesian` entry point; rows are plain knob dicts, so tests can hit `_enumeration` directly without routing through `_plan_kernel`. `split_register_axes` / `permute_lane_accesses` used to live here but moved to `lowering/kernel/` once dtype-aware analytical passes consolidated there (see `plans/stamp-ssa-dtypes-and-reorder.md`); they still pattern-match `TileOp` because they run pre-materialize. |
| `lowering/kernel/`         | Pre-materialize dtype-aware analytical passes plus the final `TileOp → KernelOp` lowering. Order: `lower_atom_tile` (MMA-only: lowers the tensor-core matmul cell — plain operand `Load`s + an `Mma`, carried in from `tile/011_lower_atom_cell` through the staging passes — to the `"mma_sync"` s16816 kernel chain: `RegFragment` decls + per-reduce `LdmatrixLoad a`+`LdmatrixLoad b`+`MmaSyncPtx` + final `RegStore` (the `ldmatrix` + `mma.sync.aligned` register-array path). Operands are matched per reduce site via the `Mma` (which names its A/B operands by SSA value); the fragment SSA names are seeded once from the FIRST reduce site (stable across prologue/inner/epilogue for the per-cell replicator); each `LdmatrixLoad.src_index` is rebuilt per cache-axis (each `Var * block`) with `ldm` from the inner source dim's slab stride by re-harvesting the live `Source`s. `ldmatrix` is smem→register only, so each operand's transport is picked by whether an enclosing `StageBundle` staged it: staged → `LdmatrixLoad` (smem); unstaged → `LdmatrixLoad(staged=False)`, which renders a gmem-direct fragment load (`dpl_mma_load_{a,b}_gmem`, replicating the m16n8k16 lane→element map without ldmatrix). So an MMA tile whose operands the staging passes declined to stage (e.g. slabs over the smem budget) still compiles — slower than the staged path — instead of raising, and the planner needn't avoid emitting it. A masked warp tile (symbolic M/N, `OVERHANG`-stamped) carries a boundary `Cond` that only gates the atom tile's BASE coordinate, so `_boundary_guards` classifies its predicate against the cell Write's M/N coordinate exprs and stamps per-element row/col guards onto the `RegStore` (row guards keep the vectorized pair stores; a col guard splits to per-element scalar stores; guarded epilogue gmem reads move inside the element's check); an unstaged gated-axis operand takes the clamped gmem-direct helper (`LdmatrixLoad.gmem_guard` → `dpl_mma_load_a_gmem_mclamp` / `_b_gmem_nclamp`), and a symbolic output inner extent resolves `ldm` from the runtime kernel arg. A fused **pointwise epilogue** (residual adds, bias / scale broadcasts, activation chains — anything in the backward slice from the Write to the accumulator) is folded into the store, CUTLASS epilogue-visitor style: `_scan_epilogue` strips the scalar Loads + Assigns (whose accumulator SSA name doesn't exist on the fragment path) and the `RegStore` evaluates the chain per fragment element in f32 at the element's own (row, col) — leaf operands load with per-dim `m`/`n`/`fixed` roles at each buffer's own stride (transposed / broadcast operands included), ops render via the scalar `op_to_expr` translation. A coord-predicated **`Select`** (the causal attention mask `(n<=m) ? mask_zero : mask_fill`) folds too: its branch values must be leaf Loads and its predicate must reference only the M/N output coords, so the store renders it as a per-element ternary — `_scan_epilogue` rewrites the predicate's M/N coordinate expressions to `__M__`/`__N__` placeholders the `RegStore` substitutes with each fragment element's own (row, col). This is what lets the (RoPE-split) QK^T scores matmul reach tensor cores. Eligibility is the NEGATIVE rule shared with the planner gate (`tile/_atom.classify_fragment_epilogue` — the slice folds unless it has an ineligible op/dependency: accumulator consumed mid-reduce, multiple accumulators, multiple/vector Writes, escaping values, non-Load/non-coord-Select leaves, in-kernel-produced or non-f32-convertible leaf buffers, ops without a rendering, or leaf index dims the lane arithmetic can't reproduce); blocked shapes gate to the scalar tier. Unlocked the Qwen3 down_proj+residual fusion's tensor cores — 29 → 8 µs, `plans/qwen3-embedding-layer0-tune-findings.md` finding 3. Strips the `AtomTile` wrapper. The `rewrite` entry point and its lowering helpers all live in this one module. Scalar TileOps skip; see `plans/mma-fragment-factorization.md` and `plans/mma-smem-staging.md`) → `split_register_axes` (replicates REGISTER-tagged bodies per-cell, with dep-tracked single-copy preservation of axis-invariant statements — for MMA kernels, replicates the Mma* chain per (M_r, N_r) cell, threading per-cell fragment SSA renames via the `Mma*.rewrite.register` handlers) → `dedup_replicated` (content-agnostic CSE: structurally identical Loads / Assigns left over after replication fold into one — the same shape the deleted blocked-GEMM builder used to produce by hand-partitioning N-invariant cones; see `plans/obsolete-blocked-gemm-builder.md`) → `place_inits` (places explicit `Init` Stmts at correct accumulator scope — descends into a `WarpTile`-wrapped `WarpSpecialize` to land the Init at the **consumer_body head**, above the consumer K loop and inside the role split; placing it higher would let the renderer's default per-loop init fire inside the loop and reset the accumulator every K chunk. A `Cond` wrapping a `Write` (the masked-boundary output store of a register tile — `if (coord < N) out[...] = acc`, emitted for non-divisible extents) is a per-iteration output escape just like a bare `Write`, so the crossable-reduce check treats it as non-crossable and the Init lands inside the register-M loop; without that the mask hid the escape and accumulators leaked across register-tile rows) → `stamp_types` (single body walk populating `Load.dtype` / `Assign.dtype` / `Write.value_dtype` / `Source.dtype` from `graph.nodes[buf].output.dtype`; also forces fp32 for overflow-prone ops — a square `multiply(a, a)` or any `pow` — so RMSNorm's mean-of-squares of large fp16 activations (e.g. Gemma's q/k pre-norm ±200s, whose square exceeds fp16's 65504) computes in fp32 like torch's `.float()`, rather than overflowing to inf → garbage reduction; distinct-arg `multiply` (matmul) stays fp16) → `demote_to_write_dtype` (folds f16-only chains feeding f16 Writes) → `vectorize_loads` (widens consecutive scalar Loads into LDS.128 / `__half2`) → `permute_lane_accesses` (chunks the N register tile into LDS.128-sized strips to remove bank conflicts on `FN > V`; skipped for MMA — `ldmatrix` handles its own swizzling) → `pack_fp16_pairs` (pairs scalar `__half` Inits/Accums into `__half2`; skipped for MMA — the C fragment IS the accumulator) → `vectorize_stores` (widens consecutive scalar Writes) → `flatten_wrap_stages` (flattens wrap-body `Stage(... body=[consumer])` into `[Stage(empty), *consumer]` so the materializer walks producer scaffolding then consumer siblings) → `materialize_tile` (purely-mechanical Tile → Kernel lowering; Smem decls read `Source.dtype` directly, and swizzled TMA operand slabs align to their full swizzle atom (`8 × swizzle_width` B: B128→1024, B64→512, B32→256) so the coordinate-only `ldmatrix` XOR matches the hardware's absolute-address swizzle (non-swizzled TMA stays at the 128 B box recommendation); its emit logic lives in sibling `_`-prefixed helper modules `_stage_expand` / `_combine` / `_tma_groups`, which the pass loader skips) → `drop_redundant_syncs` (Kernel-IR peephole collapsing back-to-back / leading `Sync`s at the tile-body level). All passes through `flatten_wrap_stages` pattern-match `TileOp`; `materialize_tile` consumes `TileOp` and produces the `KernelOp`; `drop_redundant_syncs` rewrites `KernelOp → KernelOp`. |
| `lowering/cuda/`           | `lower_kernelop` renders the `KernelOp` body to a `__global__` source string (via `ir/kernel/render.py::render_kernelop`) and mutates the node's op to `CudaOp` in place. |

See `ir/ARCHITECTURE.md` for what each IR dialect looks like.

## Dump hooks (`dump.py`)

`CompilerDump.on_pass(idx, pass_name, graph)` dumps the post-pass graph
uniformly for every pass: `NN_<pass_name>.{json,txt,dot}` (+
`NN_<pass_name>.kernels.txt` if any node has a non-empty
`pretty_body()`). Slashes in the pass name are flattened to
underscores, so `lowering/cuda` dumps as e.g. `06_lowering_cuda.*`. The
pre-pipeline input graph is dumped separately as `00_input.*` via
`dump.dump_input_graph(graph)` from the caller.

The uniform strategy means adding a new pass automatically gets dumped
— no per-pass registration needed. Rendering the kernel-level IRs
(loop/tile/kernel/cuda) lives in `format_kernels(graph)`, which calls
each op's own `pretty_body()`. Node ids accumulate `merged_` per
fusion step and a leading `lift_` from lifting; `_canonical_node_id`
collapses both for display, and the rendered body is rewritten with
the same map so `Load`/`Write` references match.

Per compute kernel, `_dump_per_kernel` also writes `<prefix>.kernels/<kname>.json` — a standalone sub-graph (kernel +
its `InputOp`/`ConstantOp` producers) loadable via `deplodock run --ir`. When op provenance is present (see
`compiler/provenance.py`), it additionally writes `<kname>.torch.json` + `<kname>.torch.txt`: the **original Torch ops**
that kernel implements, sliced from the pristine pre-decomposition graph stashed in `dump_input_graph`, with an `i/N`
coverage header per origin (full vs partial). Because the slice is taken from the original graph by origin id, it is
always made of whole Torch ops — runnable via `deplodock run --ir <kname>.torch.json --bench` to reproduce accuracy /
latency vs torch for exactly those ops.

## Per-rule diff output (`rule_diff.py`)

At `compile -vv` (DEBUG log level) the engine emits one block per rule application: a unified diff between the
matched subgraph before the rewrite and the rewritten fragment, bracketed by `>>> <pass>:NNN_rulename` and
`<<< <pass>:NNN_rulename` markers on their own lines. The `<pass>` prefix is the single-letter shorthand from
`PASS_SHORTHAND` (`d`=decomposition, `o`=optimization, `l`=lifting, `f`=fusion, `t`=tile, `k`=kernel, `c`=cuda) —
the same letters the CLI accepts in `--passes dolft`. Skipped rules (`RuleSkipped` exception) collapse to a
one-liner `--- <pass>:NNN_rulename skipped at <root>: <reason>`.

`PASS_SHORTHAND` is the single source of truth: `commands/compile.py` imports it to build its `--passes` shortcut
expander, so the CLI flag and the `-vv` marker prefix can never drift.

The bracketing makes per-rule and per-pass slicing trivial — `awk '/^>>> t:005/,/^<<< t:005/'` extracts a
single rule's diff and `awk '/^>>> t:/,/^<<< t:/'` extracts the entire tile-lowering pass. ANSI color (`+` green, `-` red, `@@` cyan) is applied only inside the diff body, so the markers
stay plain ASCII and `awk` matches reliably even on colored output. Color follows
`compile --color {auto,always,never}` (default `auto`: tty-aware, honors `NO_COLOR`); diff context and a
fallback line cap are tuned via `--diff-context N` and `--diff-max-lines N`. `RuleRenderConfig` is set once from
`commands/compile.py:handle_compile` via `rule_diff.set_config()`.

The structured `.rules.json` dump under `DEPLODOCK_DUMP_DIR` is unaffected — the diff is purely a presentation
layer for the human-readable `_rule_texts` channel.

## Fragment splice (`engine._apply_replacement`)

1. Walk the fragment in topo order. `InputOp` nodes forward their id to
   the existing graph node (external reference); non-Input nodes are
   added to the graph with fresh ids.
2. `replace_node(match.output or match.root_node_id, new_output)`
   rewires all consumers (and `graph.outputs` slots) from the old
   output to the fragment's output id.
3. Merge hints from every consumed node into the new output.
4. Remove consumed nodes and run `_remove_orphans` to drop any now-
   dangling constants/inputs.

## Invariants

- A rule module must not reach into the engine's internals; its
  interface is `PATTERN` + `rewrite(graph, match)`.
- `pipeline/` imports from `ir/` but never from `backend/`. Lowering
  rules produce IR; actually executing that IR is the backend's job.
