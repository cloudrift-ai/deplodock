# Visualizing the Tile IR block DAG — with AntV G6 v5

A fancy, interactive view that renders a `TileGraph` (see `plans/tile-ir-block-dag.md`) as a **layered dataflow graph
with the Schedule drawn on top**: the algorithm topology, the derived projections, and the scheduling annotations in one
explorable picture. Built on **AntV G6 v5**, a vanilla-JS graph engine loaded as a single CDN UMD bundle — no build
step, same deployment shape as the existing ECharts pages. Primary users: compiler debugging (what did the composer
actually schedule?), `assemble` validation (DAG + schedule beside the emitted tower), and teaching the IR.

## Engine decision: AntV G6 v5

The view needs four things a statistical-chart engine doesn't have: a **layered DAG layout**, **rich HTML node cards**,
**nested + overlapping grouping** (launch / scope / cohort), and **interactivity** (pan/zoom, expand a block's body,
toggle the schedule overlay). Online survey of the 2026 options against those needs:

| Engine | Layered DAG layout | Node content | Grouping | Build step | Verdict |
|---|---|---|---|---|---|
| **AntV G6 v5** | built-in `antv-dagre` | **real HTML nodes** (`type:'html'`, full CSS) | **Combos** (nested) + **BubbleSets** (overlapping sets) | none — UMD CDN, vanilla JS | **chosen** |
| Apache ECharts | none (`graph` is force/circular/manual) | symbol + label | none (fake with markArea) | none | keep for bar charts only |
| React Flow (xyflow) | none — bolt on dagre/ELK | fanciest (DOM/React) | sub-flows | **React + bundler** | clashes with the one-HTML-file convention |
| Cytoscape.js | dagre/elk extension | canvas (HTML via extension) | compound nodes | none | nodes not HTML; cards are harder |
| Graphviz (wasm) | best static layout | SVG tables | clusters | none | static; weak interactivity |

G6 is the only option that is simultaneously **fancy + interactive**, **HTML-rendered nodes**, has a **built-in layered
layout** (no separate layout lib to wire), offers **both nested Combos and overlapping BubbleSets** (which exactly fits
the launch ⊃ scope nesting *and* the cohort grouping that crosses it), and ships as a **single vanilla-JS CDN script** —
matching the repo's "emit one self-contained HTML file, screenshot via Playwright" pattern. React Flow renders the
fanciest DOM nodes but drags in React + a bundler, which the repo's HTML-from-CDN convention has no place for; revisit
only if deplodock ever adopts a JS build pipeline. ECharts stays the engine for the latency/bench bar charts.

G6 v5 confirmed capabilities used below: HTML node (`node.type='html'`, `style.innerHTML:(d)=>htmlString`), layouts
incl. `antv-dagre`/`dagre`/ELK, Combos (Rect/Circle/Custom), renderers Canvas/SVG/WebGL, and plugins **Minimap,
Toolbar, Legend, Tooltip, Fullscreen, Contextmenu, GridLine, BubbleSets, Hull, Watermark**. UMD bundle ~0.96 MB, global
`window.G6`, on cdnjs / jsDelivr.

## What we draw — the three strata, mapped to G6 primitives

The whole value is showing all three IR strata at once, and letting you **toggle the schedule off** to see the bare
algorithm DAG. Top-down layout: kernel **inputs** at the top, output `Write`s at the bottom, blocks layered by
topological rank (`antv-dagre`, `rankdir: 'TB'`).

- **Block → an HTML-node card.** `node.type='html'`; `innerHTML(d)` returns a styled `<div>` card:
  - header: block `name` + derived badges — `carrier` kind (SEMIRING/MONOID/TWISTED_MONOID), `atom` (tensor-core),
    `mask` (symbolic-K ⚠);
  - body: the one-line `compute` summary (click to expand the full pretty body — Tier 2);
  - footer: the `domain` axis chips, each tinted by `Schedule.binding` (GRID/SERIAL/THREAD/WARP/REGISTER/ATOM);
  - card fill/border by `Schedule.role` when warp-specialized.
- **Derived edge → a G6 edge** (`type:'cubic-vertical'`), styled by the read's Schedule: `stroke` by `transport`
  (gmem-direct = muted; SYNC/CPASYNC/TMA = three legend colors), `lineDash` when `distance>0` (pipelined),
  `endArrow:true`, `labelText` = `buffer · CPASYNC Δ1 d2`.
- **Launch group → a Rect Combo** ("launch N = one kernel"). Atomic-free split-K renders two combos joined by the
  `partial` buffer edge crossing the boundary — the clearest "this is two kernels."
- **Scope / placement → nested Combos** mirroring `Block.scope`, so the fused-prologue case literally shows
  `SerialTile(M_scope) ⊃ { prologue ; RegisterTile(N_reg) ⊃ matmul }` — `hoist` becomes visible.
- **Cohort → the BubbleSets plugin** — a set-membership bubble around the staged reads that retire under one barrier.
  BubbleSets draws around arbitrary node sets *even when they overlap the combo boxes*, which is exactly why the prior
  "three groupings can't all be boxes" problem disappears: launch ⊃ scope are nested combos, cohort is a bubble overlay.

Color discipline (the repo's standing rule — no inline hex): `binding` → `PALETTE_1` (cool), `transport`/`cohort` →
`PALETTE_2` (warm), card chrome from `THEMES[theme]` CSS variables. The two color dimensions stay disjoint — what the
dual-palette design is for.

### The block card (HTML-node `innerHTML`)

```js
function card(d) {                                   // d = the projected block
  const chip = (ax) => `<span class="chip b-${ax.binding}">${ax.name}</span>`;
  const badge = (t, v) => v ? `<span class="badge ${t}">${v}</span>` : '';
  return `<div class="blk role-${d.role||'none'}">
    <div class="hd">${d.name}${badge('carrier', d.carrier)}${badge('atom', d.atom)}${badge('mask', d.mask)}</div>
    <div class="op">${d.compute}</div>
    <div class="axes">${d.domain.map(chip).join('')}</div>
  </div>`;
}
```

The card CSS pulls `--fg`/`--surface`/`--muted` from the theme variables `page.render_html` already injects, and the
`.b-grid/.b-thread/.b-register/...` chip classes map to `PALETTE_1` entries. `node.style.size` is computed per block
from the axis count so `antv-dagre` knows the box extents.

### G6 setup (illustrative; confirm option names against G6 v5 docs)

```js
const graph = new G6.Graph({
  container: 'app',
  autoFit: 'view',
  data: VIEW,                                         // { nodes, edges, combos } from the JSON projection
  node:  { type: 'html', style: { size: (d)=>d.size, innerHTML: (d)=>card(d.data) } },
  edge:  { type: 'cubic-vertical',
           style: { stroke: (d)=>TRANSPORT[d.data.staged] ?? THEME.muted,
                    lineDash: (d)=> d.data.distance ? [6,4] : null,
                    endArrow: true, labelText: (d)=>d.data.label } },
  combo: { type: 'rect', style: { labelText: (d)=>d.data.title, lineDash: [4,4] } },
  layout: { type: 'antv-dagre', rankdir: 'TB', nodesep: 28, ranksep: 64 },
  behaviors: ['zoom-canvas', 'drag-canvas', 'drag-element', 'collapse-expand'],
  plugins: ['minimap', 'fullscreen', 'grid-line',
            { type: 'toolbar', getItems: () => FIT_ZOOM_EXPORT_TOGGLE },
            { type: 'legend', /* binding + transport swatches */ },
            { type: 'tooltip', getContent: (e, items) => hoverDetail(items) },   // edge → staging detail; node → full body
            { type: 'bubble-sets', key: 'cohorts', members: COHORT_NODE_IDS }],
});
graph.render().then(() => { window.graphReady = true; });   // signal the PNG renderer
```

## Data model — one deterministic JSON projection

A pure function `tile_graph_to_view(tg: TileGraph) -> dict` (in the compiler, beside the IR) flattens the algorithm +
derived views + schedule into render-agnostic JSON that G6 consumes directly (engine never leaks into Python):

```jsonc
{
  "name": "k_matmul",
  "axes":   [{ "name": "M_b", "extent": 4, "binding": "grid" }, ...],
  "nodes":  [{ "id": "mm", "data": { "name": "mm", "compute": "acc += A·B; C = acc", "compute_full": "<pretty body>",
                "domain": [{"name":"M_b","binding":"grid"}, ...], "carrier": "semiring(+)",
                "atom": "mma_m16n8k16", "mask": null, "role": null }, "combo": "launch:0" }],
  "edges":  [{ "source": "A", "target": "mm",
               "data": { "label": "A · sync", "staged": "sync", "distance": 0, "ring_depth": 1, "cohort": 0 } }],
  "combos": [{ "id": "launch:0", "data": { "title": "launch 0 — kernel" } }],
  "cohorts": { "0": ["A", "mm", "B"] }
}
```

`edges` / `carrier` / `atom` come from the IR's **derived properties** (`Block.reads`, `Block.carrier`,
`TileGraph.edges`) — so the JSON *is* the derived state, no second source of truth. Keep it deterministic (sorted keys,
stable node/axis order) so goldens are stable. It doubles as the `NN_tilegraph.json` dump artifact and the future input
to a `deplodock compare` graph-diff (highlight which Schedule entries changed between two tunes).

## Integration

- **`deplodock/visualize/graph.py`** — `TileGraphView` + `render_tile_graph(view, *, theme) -> str`. Builds the G6
  `VIEW` object + the card/CSS + the init script, wraps via `page.render_html(body_html=…, scripts_js=…, extra_css=…)`.
  Export from `visualize/__init__`.
- **Generalize the page shell.** `page.render_html` hardcodes the ECharts `<script>` (`page.py:64`). Add
  `head_scripts: tuple[str, ...] = (ECHARTS_CDN,)`; the graph page passes `(G6_CDN,)` instead. `G6_CDN =
  "https://cdn.jsdelivr.net/npm/@antv/g6@5/dist/g6.min.js"` (global `window.G6`). One shell, two libs.
- **PNG export.** `image.render`'s `.png`/`.pdf` paths are generic Playwright screenshots and already work on any HTML
  (G6 html-nodes are DOM, captured fine). Just teach the wait to accept `window.graphReady` (set after
  `graph.render()`) alongside `echartsReady`. The `.svg` path stays ECharts-only; for vector export use G6's own
  toolbar "export image" or its SVG renderer.
- **CLI** — `deplodock viz <model_or_ir> [--png] [--no-schedule]` (sibling of `compile`/`inspect`): trace+compile to a
  `TileGraph`, project, write `tilegraph.html` (+ `tilegraph.json`, + optional `.png`). `--ir <file>` renders a dump;
  `--no-schedule` emits the bare algorithm DAG.
- **Dump dir** — `DEPLODOCK_DUMP_DIR` writes `NN_tilegraph.{json,html}` per compiled kernel (the `kernels.html`
  pattern), plus an `index.html` linking them for a whole-model trace (one page per kernel keeps each graph small).

During migration (before the real `TileGraph` exists), point the projector at the composer's **reference schedule** (the
DAG + Schedule it emits for `assemble`), so the view is useful from step 1 of the IR migration and helps confirm
`assemble` matches the picture.

## Interactive from the start (the "fancy right away")

Phase 1 ships these, not a static stub: `antv-dagre` layered layout; HTML-node cards with binding chips + derived
badges; transport/distance edge styling; launch Combos; **Minimap**, **Toolbar** (fit / zoom / export-image),
**Legend** (binding + transport), **Tooltip** (hover edge → staging detail, hover node → summary), **Fullscreen**,
**GridLine** background; and `zoom-canvas` / `drag-canvas` / `drag-element` behaviors.

## Phasing

1. **Projection + the interactive G6 page** — `tile_graph_to_view` + `graph.py` + `deplodock viz` + dump artifact, with
   the full Phase-1 interaction list above. → verify: golden the JSON; smoke the HTML includes the G6 CDN + block names
   + the binding legend; eyeball one matmul, one split-K (two launch combos), one fused-prologue (nested scope combos).
2. **Grouping depth + export + toggles** — BubbleSets cohorts, nested scope Combos, the **schedule-overlay on/off**
   toolbar toggle (re-style to the bare algorithm), **collapse-expand** combos, click-a-node → expand `compute_full`,
   and `--png` wiring (`graphReady`). Docs: `visualize/ARCHITECTURE.md` + `CLAUDE.md` (`deplodock viz`, dump artifacts).
3. **Move animation** — drive G6's data-update transitions to animate a move: render before, apply one Schedule edit
   (`stage` / `retime` / `partition_reduce`), `graph.setData(after)` → watch the edges/combos re-decorate. Turns the
   view into a scheduler explorer and a teaching aid for "scheduling = annotation."

## Risks / open

- **Bundle + offline.** ~0.96 MB UMD from CDN (parity with ECharts ~1 MB); needs network at view time. If offline is
  wanted, vendor `g6.min.js` under the `visualize` extra (where Playwright already lives) — defer unless asked.
- **HTML-node sizing.** `antv-dagre` needs node box extents; html nodes provide them via `style.size`, so compute a
  size per card from its axis-chip count (or a fixed-width card with wrapping). Verify the layout reads html-node sizes.
- **Renderer for crisp export.** Default Canvas + DOM html-nodes screenshots fine via Playwright; switch G6 to the SVG
  renderer if vector/zoomable export or selectable edge text is wanted (html-nodes stay DOM either way).
- **Three groupings.** launch ⊃ scope render as nested Combos; cohort renders as a BubbleSets overlay (crosses freely).
  Make the active grouping a Tier-2 toggle so a dense graph isn't over-boxed.
- **API drift.** G6 v5 option names above are from the current docs; confirm `antv-dagre` / `bubble-sets` / html-node
  `innerHTML` signatures against the pinned `@antv/g6@5` version when implementing.

## Tests + docs

- `tests/test_visualize.py` — golden `tile_graph_to_view` on a small fixture graph (deterministic JSON); assert
  `render_tile_graph` HTML includes the G6 CDN, the block names, the binding legend, a `launch` combo when ≥2 launch
  groups, and a BubbleSets `members` list when a cohort exists.
- Update `deplodock/visualize/ARCHITECTURE.md` (new `graph.py`, the `head_scripts` shell param, the G6 CDN + UMD
  global, the `graphReady` flag) and `CLAUDE.md` (`deplodock viz` + the `NN_tilegraph.*` dump artifacts).

## References

- AntV G6 — graph engine, layouts, combos, renderers, plugins: https://g6.antv.antgroup.com/en
- G6 v5 HTML node (`type:'html'`, `innerHTML`): https://g6.antv.antgroup.com/en/manual/element/node/html
- G6 v5 feature overview (layouts incl. antv-dagre, combos, BubbleSets, plugins): https://g6.antv.antgroup.com/en/manual/whats-new/feature
- G6 on jsDelivr / cdnjs (UMD CDN): https://www.jsdelivr.com/package/npm/@antv/g6
- React Flow layouting (why it needs an external layout lib) — considered, not chosen: https://reactflow.dev/learn/layouting/layouting
- Cytoscape.js — considered, not chosen: https://js.cytoscape.org/
