"""Bank-conflict visualizer driven by real Tile-IR ``Stage``s.

Takes one or more IR JSON paths and renders one column per input. Each
column contains a card per ``(Stage, body-Load)`` pair showing the warp's
per-lane smem bank at one inner-loop iteration.

Simulation:
:func:`deplodock.compiler.diagnostics.bank_conflicts.simulate_graph`
compiles the kernel, instruments it at the kernel-lowering stage, and runs one CTA
to record actual smem addresses per lane. This script is a thin CLI +
ECharts emitter. **Requires CUDA** — compile and launch must succeed.

Workflow — produce IRs via the compiler with whatever pass gates you
want, then feed the dumped JSONs in::

    DEPLODOCK_DUMP_DIR=/tmp/dump_a deplodock compile MODEL --layer 0
    DEPLODOCK_DISABLE_CHUNK_REDUCE=1 \\
        DEPLODOCK_DUMP_DIR=/tmp/dump_b deplodock compile MODEL --layer 0
    python scripts/visualize_bank_conflicts.py \\
        -i /tmp/dump_a/14_lowering_tile.json:baseline \\
        -i /tmp/dump_b/14_lowering_tile.json:no_chunk_reduce \\
        --out /tmp/diff.html

Any post-tile-pass dump works; pick the one whose Stages you want to
inspect (typically the final tile-stage dump).
"""

from __future__ import annotations

import argparse
import json
import os

from deplodock.compiler.diagnostics.bank_conflicts import BankConflictResult, simulate_graph
from deplodock.compiler.graph import Graph
from deplodock.visualize.page import render_html


def graph_from_ir_path(path: str) -> Graph:
    with open(path) as f:
        return Graph.from_dict(json.load(f))


def _serialize(panels: list[BankConflictResult], all_panels_for_union: list[BankConflictResult] | None = None) -> list[dict]:
    """Serialize ``panels`` to JSON-friendly dicts.

    ``all_panels_for_union`` is the un-load-filtered list used to compute
    the per-Stage union ladder. When ``--load`` filters the visible
    cards, the ladder still shows the full Stage's footprint (i.e. cells
    touched by every body Load of the Stage, not just the focused one).
    Falls back to ``panels`` if not provided.
    """
    union_source = all_panels_for_union if all_panels_for_union is not None else panels
    stage_union: dict[tuple[str, str], dict[tuple[int, int], list[tuple]]] = {}
    stage_conflict_loads: dict[tuple[str, str], dict[tuple[int, int], set[str]]] = {}
    for p in union_source:
        key = (p.tile_op_name, p.stage_name)
        u = stage_union.setdefault(key, {})
        for cell, pairs in p.full_sweep_touched.items():
            for k, lane in pairs:
                u.setdefault(cell, []).append((p.load_name, k, lane))
        cmap = stage_conflict_loads.setdefault(key, {})
        for cell in p.full_sweep_conflict_cells:
            cmap.setdefault(cell, set()).add(p.load_name)
    out: list[dict] = []
    for p in panels:
        # Per-lane address-color index: same address → same color, regardless
        # of bank. Cells of the SAME color stacked in one bank column are
        # the actual conflict (different lanes serialized to one bank with
        # different addresses); cells of DIFFERENT colors stacked = fine
        # (each lane hit a different address, no within-warp serialization).
        # Wait — corrected: same color = same addr = broadcast = fine.
        # Different colors stacked in one column = different addrs → conflict.
        unique_addrs = sorted(set(p.lane_addrs))
        addr_to_idx = {a: i for i, a in enumerate(unique_addrs)}
        lane_addr_idx = [addr_to_idx[a] for a in p.lane_addrs]
        # Smem layout ladder: cell(r, c) → bank id under the padded row
        # stride. Including the pad columns so the +1-shift is visible.
        # Cap rows to keep the diagram readable (most ladders are clear in
        # the first 16-32 rows).
        pad_cols = p.pad[1] if len(p.pad) > 1 else 0
        pad_rows = p.pad[0] if len(p.pad) > 0 else 0
        layout_rows = min(p.rows + pad_rows, 64)
        layout_cols = p.cols + pad_cols
        row_stride = p.cols + pad_cols
        smem_layout = [[(r * row_stride + c) % 32 for c in range(layout_cols)] for r in range(layout_rows)]
        # Touched at the *current* k_iter (the one ``simulate`` was
        # called with) — drawn with a white outline overlay.
        touched_now: dict[tuple[int, int], list[int]] = {}
        for lane, addr in enumerate(p.lane_addrs):
            r, c = divmod(addr, row_stride)
            if 0 <= r < layout_rows and 0 <= c < layout_cols:
                touched_now.setdefault((r, c), []).append(lane)
        # Per-Stage UNION across every body Load of the Stage.
        union = stage_union.get((p.tile_op_name, p.stage_name), {})
        sweep_per_cell: dict[tuple[int, int], list[list]] = {}
        for (r, c), entries in union.items():
            if not (0 <= r < layout_rows and 0 <= c < layout_cols):
                continue
            sweep_per_cell[(r, c)] = [[ln, k, lane] for (ln, k, lane) in entries]
        all_cells = sorted(set(touched_now) | set(sweep_per_cell))
        conflict_loads = stage_conflict_loads.get((p.tile_op_name, p.stage_name), {})
        touched_entries = [
            {
                "r": r,
                "c": c,
                "lanes_now": touched_now.get((r, c), []),
                # sweep: list of [load_name, k_iter, lane]
                "sweep": sweep_per_cell.get((r, c), []),
                # Loads whose LDS conflicts when touching this cell.
                "conflict_loads": sorted(conflict_loads.get((r, c), set())),
            }
            for (r, c) in all_cells
        ]
        out.append(
            {
                "panel_title": f"{p.stage_name} ← {p.buf}",
                "buf_short": p.buf,
                "formula": (f"{p.stage_class}({p.rows}×{p.cols})  " + (f"pad={p.pad}" if p.pad and any(p.pad) else "no pad")),
                "lane_banks": p.lane_banks,
                "lane_addrs": list(p.lane_addrs),
                "lane_addr_idx": lane_addr_idx,
                "n_unique_addrs": len(unique_addrs),
                "counts": p.counts,
                "distinct_addrs": p.distinct_addrs,
                "max_way": p.max_way,
                "raw_max_way": p.raw_max_way,
                "conflict_events": p.conflict_events,
                "lds128_events": p.lds128_events,
                "vec_group_size": p.vec_group_size,
                "avg_way": p.avg_way,
                "rows": p.rows,
                "cols": p.cols,
                "pad": list(p.pad),
                "smem_bytes": p.smem_bytes,
                "layout": {
                    "rows": layout_rows,
                    "cols": layout_cols,
                    "data_rows": p.rows,
                    "data_cols": p.cols,
                    "row_stride": row_stride,
                    "banks": smem_layout,
                    "touched": touched_entries,
                    "index_expr": ", ".join(p.index_repr),
                    "load_name": p.load_name,
                },
                "notes": [],
            }
        )
    return out


EXTRA_CSS = """
  *{box-sizing:border-box;}
  /* Center the column grid + shared legend horizontally; shrink to
     content width so unused horizontal space sits outside the page. */
  .page{display:flex;flex-direction:column;align-items:center;margin:0;padding:0;}
  /* Single page-grid: N auto-sized columns (one per IR variant);
     shared titles + legends span all columns. justify-content:center
     so the whole block stays centered when the iframe is wider. */
  .page-grid{display:grid;gap:8px 20px;justify-content:center;align-items:start;}
  .page-grid > .col-head,
  .page-grid > .matrix,
  .page-grid > .hist,
  .page-grid > .ladder{justify-self:center;}
  .page-grid > .section-title{grid-column:1 / -1;}
  .page-grid > .ladder,
  .page-grid > .ladder-sub{grid-column:1 / -1;justify-self:center;}
  .page-grid > .hist-legend-shared,
  .page-grid > .bank-legend-shared{grid-column:1 / -1;justify-self:center;}
  .ladder-sub{font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);
    margin:14px 0 4px;text-align:center;}
  .ladder-sub .label{color:var(--label-accent);font-weight:600;}
  .section-title{font-size:16px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
    color:var(--fg);margin:22px 0 8px;text-align:left;}
  .col-head{text-align:center;font-size:14px;text-transform:uppercase;letter-spacing:.16em;
    color:var(--muted);padding:6px 0 8px;border-bottom:1px solid var(--rule);min-width:200px;}
  .col-head .label{color:var(--label-accent);font-weight:600;}
  .card-title{font-size:14px;font-weight:600;letter-spacing:-.01em;}
  .card-formula{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:12px;color:var(--muted);margin-top:4px;}
  .card-notes{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:11px;color:var(--muted);margin-top:6px;line-height:1.55;}
  .card-verdict{display:inline-flex;align-items:center;gap:6px;margin-top:10px;padding:4px 10px;
    border-radius:999px;font-size:12px;font-weight:600;background:rgba(255,255,255,.04);}
  .dot{width:6px;height:6px;border-radius:50%;}
  .v-ok{color:#3ddc84;} .v-ok .dot{background:#3ddc84;}
  .v-warn{color:#ffb454;} .v-warn .dot{background:#ffb454;}
  .v-bad{color:#ff5c7a;} .v-bad .dot{background:#ff5c7a;}
  /* Square punchcard: 32 banks × 32 lanes is intrinsically square
     data. Capping width keeps cells visibly square instead of
     squashed into wide rectangles when the container is stretched. */
  .matrix{width:100%;max-width:440px;aspect-ratio:1/1;height:auto;margin:0 auto 12px;}
  /* Histogram shares the bank axis with the punchcard above —
     match its max-width so banks line up vertically. */
  .hist{width:100%;max-width:440px;height:110px;margin:2px auto 0;}
  .ladder{margin:6px auto 0;}
  .ladder-title{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);
    margin-top:12px;margin-bottom:6px;}
  .bank-legend{display:grid;grid-template-columns:repeat(8,auto);gap:6px 18px;margin-top:10px;
    width:fit-content;font-family:'JetBrains Mono',ui-monospace,monospace;font-size:14px;color:var(--muted);}
  .bank-legend span{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;}
  .bank-legend i{width:13px;height:13px;border-radius:2px;display:inline-block;flex-shrink:0;}
  .bank-legend-shared{margin-top:18px;font-size:14px;}
  .hist-legend{display:flex;flex-wrap:wrap;gap:6px 22px;margin:10px auto 0;
    width:fit-content;font-size:14px;color:var(--muted);}
  .hist-legend span{display:inline-flex;align-items:center;gap:7px;}
  .hist-legend i{width:14px;height:14px;border-radius:2px;display:inline-block;}
  .hist-legend-shared{margin-top:14px;}
  .empty{color:var(--muted);font-size:12px;padding:32px 16px;text-align:center;
    background:rgba(255,255,255,.02);border-radius:10px;border:1px dashed rgba(255,255,255,.06);}
  .legend{display:flex;gap:18px;margin-top:28px;color:var(--muted);font-size:12px;}
  .legend span{display:inline-flex;align-items:center;gap:8px;}
  .legend i{width:10px;height:10px;border-radius:3px;display:inline-block;}
"""

BODY_HTML = """
  <div class="page">
    <div class="page-grid" id="page-grid"></div>
  </div>
"""

# PALETTE_1 — used for the smem-layout ladder ("color = bank id").
# PALETTE_2 — used for the lane×bank punch card ("color = address index").
# Both are 32-color qualitative palettes from deplodock.visualize.theme;
# kept visually disjoint so a reader never confuses one plot for the other.
SCRIPTS_JS = r"""
const WARP=32, BANKS=32;
const palette={empty:THEME.empty,ok:STATUS.ok,warn:STATUS.warn,bad:STATUS.bad};
const cellColor=c=>c===0?palette.empty:c===1?palette.ok:c<=4?palette.warn:palette.bad;
const verdict=m=>m>4?'v-bad':m>1?'v-warn':'v-ok';

// Layout: one page-grid with N columns (one per IR variant). Rows
// alternate between shared titles (spanning all columns, left-justified)
// and per-column plot cells. We render rows of HTML strings into the
// grid in the right order.
const root = document.getElementById('page-grid');
// Fixed-width columns so the two punchcards/histograms render at the
// same size regardless of how long the col-head labels are.
root.style.gridTemplateColumns = `repeat(${PAYLOAD.columns.length}, 440px)`;

// 1) Shared punchcard title (above the column headings).
const punTitle = document.createElement('div');
punTitle.className = 'section-title';
punTitle.textContent = 'bank access punchcard — access per (lane, bank)';
root.appendChild(punTitle);

// 2) Column headings.
PAYLOAD.columns.forEach(col => {
  const head = document.createElement('div');
  head.className = 'col-head';
  head.innerHTML = `<span class="label">${col.label}</span>`;
  root.appendChild(head);
});

// 3) Punchcards row.
PAYLOAD.columns.forEach((col, ci) => {
  if (!col.panels.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'no Stages found';
    root.appendChild(empty);
  } else {
    const matrix = document.createElement('div');
    matrix.className = 'matrix';
    matrix.id = `m_c${ci}_p0`;
    root.appendChild(matrix);
  }
});

// 4) Histograms row.
PAYLOAD.columns.forEach((col, ci) => {
  if (!col.panels.length) {
    root.appendChild(document.createElement('div'));
    return;
  }
  const hist = document.createElement('div');
  hist.className = 'hist';
  hist.id = `h_c${ci}_p0`;
  root.appendChild(hist);
});

// 5) Shared severity legend (spans all columns, centered).
const sevLegend = document.createElement('div');
sevLegend.className = 'hist-legend hist-legend-shared';
sevLegend.innerHTML = `
  <span><i style="background:#3ddc84"></i>1 lane (no conflict)</span>
  <span><i style="background:#ffb454"></i>2–4 lanes (mild)</span>
  <span><i style="background:#ff5c7a"></i>&gt;4 lanes (heavy)</span>
`;
root.appendChild(sevLegend);

// 6) Shared ladder title.
const ladTitle = document.createElement('div');
ladTitle.className = 'section-title';
ladTitle.textContent = 'smem layout — bank per (row, col)';
root.appendChild(ladTitle);

// 7) Ladders — stacked vertically, each spanning all columns. Each
// ladder gets its own sub-heading (the variant label) so the reader
// can tell them apart without the side-by-side layout.
PAYLOAD.columns.forEach((col, ci) => {
  if (!col.panels.length) return;
  const p = col.panels[0];
  const sub = document.createElement('div');
  sub.className = 'ladder-sub';
  sub.innerHTML = `<span class="label">${col.label}</span>`;
  root.appendChild(sub);
  const ladder = document.createElement('div');
  ladder.className = 'ladder';
  ladder.id = `l_c${ci}_p0`;
  // Stacked layout doubles the available real estate per ladder.
  const lay = p.layout;
  const axisW = 56, axisH = 32, maxH = 900, maxW = 960;
  const cellH = Math.max(6, Math.min(10, Math.floor((maxH - axisH) / lay.rows)));
  const cellW = Math.max(cellH, Math.min(3 * cellH, Math.floor((maxW - axisW) / lay.cols)));
  ladder.style.width = `${lay.cols * cellW + axisW}px`;
  ladder.style.height = `${lay.rows * cellH + axisH}px`;
  root.appendChild(ladder);
});

// 8) Shared bank legend.
const bankLegendEl = document.createElement('div');
bankLegendEl.className = 'bank-legend bank-legend-shared';
bankLegendEl.id = 'shared-bank-legend';
root.appendChild(bankLegendEl);

// Now render plots for each column's first panel.
PAYLOAD.columns.forEach((col, ci) => {
  if (!col.panels.length) return;
  col.panels.forEach((p, pi) => {
    if (pi > 0) return;  // one panel per column expected
    const id = `c${ci}_p${pi}`;

    const m=echarts.init(document.getElementById(`m_${id}`),null,{renderer:'canvas'});
    const md=[];
    for(let l=0;l<WARP;l++) for(let b=0;b<BANKS;b++)
      md.push({value:[b,l,0],itemStyle:{color:palette.empty}});
    p.lane_banks.forEach((bank,lane)=>{
      // Color cells by ADDRESS index. Same color = same address (broadcast).
      // Different colors stacked in one bank column = different addresses
      // serialized = real bank conflict.
      const addrIdx = p.lane_addr_idx[lane];
      const color = PALETTE_2[addrIdx % PALETTE_2.length];
      md.push({
        value:[bank,lane,1],
        itemStyle:{color: color, shadowBlur:6, shadowColor:color+'66'},
        addrIdx: addrIdx,
      });
    });
    m.setOption({
      backgroundColor:'transparent',
      tooltip:{backgroundColor:THEME.tooltipBg,borderColor:THEME.axisLine,textStyle:{color:THEME.tooltipText,fontSize:12},
        formatter:pt=>{const [b,l,c]=pt.value;
          if (c === 0) return `bank ${b}<br/><span style="color:#6b7280">no lane</span>`;
          const addr = p.lane_addrs[l];
          const dist = p.distinct_addrs[b];
          const verdict = dist === 1 ? '<span style="color:#3ddc84">broadcast — 0 events</span>'
                       : dist <= 4 ? `<span style="color:#ffb454">${dist}-way conflict</span>`
                                   : `<span style="color:#ff5c7a">${dist}-way conflict</span>`;
          return `lane <b>${l}</b> → bank <b>${b}</b>, addr <b>${addr}</b><br/>${verdict}`;
        }},
      grid:{left:30,right:8,top:8,bottom:22},
      xAxis:{type:'category',data:[...Array(BANKS).keys()],
        axisLine:{show:false},axisTick:{show:false},
        axisLabel:{color:'#6b7280',fontSize:12,interval:3,showMaxLabel:true,showMinLabel:true}},
      yAxis:{type:'category',data:[...Array(WARP).keys()],inverse:true,
        axisLine:{show:false},axisTick:{show:false},
        axisLabel:{color:'#6b7280',fontSize:12,interval:3,showMaxLabel:true,showMinLabel:true}},
      series:[{type:'heatmap',data:md,progressive:0,
        itemStyle:{borderRadius:2},
        emphasis:{itemStyle:{borderColor:'#fff',borderWidth:1.5}},
        animationDuration:500,animationEasing:'cubicOut'}]});

    const h=echarts.init(document.getElementById(`h_${id}`),null,{renderer:'canvas'});
    h.setOption({
      backgroundColor:'transparent',
      tooltip:{backgroundColor:THEME.tooltipBg,borderColor:THEME.axisLine,textStyle:{color:THEME.tooltipText,fontSize:12},
        formatter:pt=>`bank <b>${pt.name}</b><br/>${pt.value} lane(s)`},
      grid:{left:38,right:8,top:6,bottom:22},
      xAxis:{type:'category',data:[...Array(BANKS).keys()],
        axisLine:{show:false},axisTick:{show:false},
        axisLabel:{color:'#6b7280',fontSize:12,interval:3,showMaxLabel:true,showMinLabel:true}},
      yAxis:{type:'value',min:0,max:8,splitLine:{lineStyle:{color:THEME.splitLine}},
        axisLabel:{color:'#6b7280',fontSize:12},axisLine:{show:false},axisTick:{show:false}},
      series:[{type:'bar',data:p.distinct_addrs.map(c=>({value:c,itemStyle:{
        color:{type:'linear',x:0,y:0,x2:0,y2:1,
          colorStops:[{offset:0,color:cellColor(c)},{offset:1,color:cellColor(c)+'55'}]},
        borderRadius:[3,3,0,0]}})),barWidth:'70%',
        animationDuration:600,animationEasing:'cubicOut'}]});
    // Smem-layout ladder: each cell colored by its bank id (0..31)
    // using the same address palette mod 32. Cells the warp actually
    // touched at this k_iter get a white outline overlay so the
    // connection between layout and access pattern is visible.
    const lay = p.layout;
    const lEl = document.getElementById(`l_${id}`);
    const ldr = echarts.init(lEl, null, {renderer:'canvas'});
    const ldrData = [];
    // touchedNow: cells the 32 lanes of the focused Load access AT THE
    // CURRENT k_iter (32 cells, one per lane). These are the only cells
    // fully opaque — making the conflict reading direct: two opaque
    // cells sharing a bank color = same bank hit with different
    // addresses = conflict for this LDS.
    const touchedNow = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.lanes_now]));
    const sweep = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.sweep]));
    const focusLoad = lay.load_name;
    const conflictLoadsByCell = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.conflict_loads || []]));
    for (let r = 0; r < lay.rows; r++) {
      for (let c = 0; c < lay.cols; c++) {
        const bank = lay.banks[r][c];
        const isPad = (c >= lay.data_cols) || (r >= lay.data_rows);
        const k = `${r},${c}`;
        const isNow = (touchedNow.get(k) || []).length > 0;
        const color = isPad ? THEME.padCell : PALETTE_1[bank % PALETTE_1.length];
        const op = isNow ? THEME.opFocus : THEME.opFaint;
        const style = {color: color, opacity: op};
        if (isNow) {
          style.borderColor = THEME.focusBorder;
          style.borderWidth = 1.5;
        }
        ldrData.push({value:[c, r, bank], itemStyle: style});
      }
    }
    ldr.setOption({
      backgroundColor:'transparent',
      tooltip:{
        backgroundColor:'#0e1014', borderColor:'#2a2d33',
        textStyle:{color:'#e8eaed', fontSize:12},
        formatter: pt => {
          const [c, r, bank] = pt.value;
          const k = `${r},${c}`;
          const padTag = (c >= lay.data_cols || r >= lay.data_rows)
            ? '<br/><span style="color:#6b7280">padding (allocated, never accessed)</span>' : '';
          const sweepPairs = sweep.get(k) || [];
          const isNow = (touchedNow.get(k) || []).length > 0;
          const head = `row=<b>${r}</b>, col=<b>${c}</b>, bank <b>${bank}</b>, addr <b>${r * lay.row_stride + c}</b>`;
          if (!sweepPairs.length) {
            return head + padTag + `<br/><span style="color:#6b7280">never read by warp 0 (other warps own this row)</span>`;
          }
          const loadNames = [...new Set(sweepPairs.map(([ln, _k, _l]) => ln))].sort();
          const loadSection = loadNames.length
            ? `<br/>${loadNames.map(ln => `<code style="color:#3ddc84">${ln}</code>`).join('<br/>')}`
            : '';
          const conflicts = conflictLoadsByCell.get(k) || [];
          let conflictTag = '';
          if (conflicts.length) {
            const inFocus = conflicts.includes(focusLoad);
            const others = conflicts.filter(l => l !== focusLoad);
            if (inFocus) {
              conflictTag = `<br/><span style="color:#ff5c7a">⚠ conflict in <b>${focusLoad}</b>'s LDS at this k_iter</span>`;
              if (others.length) {
                conflictTag += `<br/><span style="color:#6b7280">(also: ${others.join(', ')})</span>`;
              }
            } else {
              conflictTag = `<br/><span style="color:#ffb454">⚠ conflict in: ${conflicts.join(', ')}</span>`;
            }
          }
          return head + loadSection + conflictTag + padTag;
        },
      },
      grid:{left:30, right:14, top:6, bottom:18},
      xAxis:{
        type:'category', data:[...Array(lay.cols).keys()],
        axisLine:{show:false}, axisTick:{show:false},
        axisLabel:{color:'#6b7280', fontSize:11,
          interval: Math.max(0, Math.floor(lay.cols/8) - 1),
          showMaxLabel: true, showMinLabel: true},
      },
      yAxis:{
        type:'category', data:[...Array(lay.rows).keys()], inverse:true,
        axisLine:{show:false}, axisTick:{show:false},
        axisLabel:{color:'#6b7280', fontSize:11,
          interval: Math.max(0, Math.floor(lay.rows/8) - 1),
          showMaxLabel: true, showMinLabel: true},
      },
      series:[{
        type:'heatmap', data: ldrData, progressive: 0,
        itemStyle:{borderRadius:1},
        animationDuration:400, animationEasing:'cubicOut',
      }],
    });
    window.addEventListener('resize',()=>{m.resize();h.resize();ldr.resize();});
  });
});

// One shared bank-color legend below all columns. Renders 8×4 chips
// for every bank that appears in any panel's layout.
(() => {
  const banksUsed = new Set();
  PAYLOAD.columns.forEach(col => col.panels.forEach(p => {
    for (let r = 0; r < p.layout.rows; r++) {
      for (let c = 0; c < p.layout.cols; c++) banksUsed.add(p.layout.banks[r][c]);
    }
  }));
  const host = document.getElementById('shared-bank-legend');
  if (!host) return;
  [...banksUsed].sort((a,b)=>a-b).forEach(bank => {
    const chip = document.createElement('span');
    chip.innerHTML = `<i style="background:${PALETTE_1[bank % PALETTE_1.length]}"></i>bank ${bank}`;
    host.appendChild(chip);
  });
})();
"""


def emit_html(columns: list[dict], out_path: str, theme: str = "dark", *, transparent: bool = True) -> str:
    payload_js = f"const PAYLOAD = {json.dumps({'columns': columns})};\n"
    html = render_html(
        body_html=BODY_HTML,
        scripts_js=payload_js + SCRIPTS_JS,
        theme=theme,
        title="smem bank conflicts (IR-driven)",
        extra_css=EXTRA_CSS,
        transparent=transparent,
    )
    with open(out_path, "w") as f:
        f.write(html)
    return html


def _parse_ir_arg(arg: str) -> tuple[str, str]:
    if ":" in arg:
        path, label = arg.rsplit(":", 1)
        return path, label
    return arg, os.path.basename(arg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--ir",
        action="append",
        default=[],
        required=True,
        metavar="PATH[:LABEL]",
        help="IR JSON dump (post-tile-pass). Repeat for side-by-side columns.",
    )
    p.add_argument("--stage", action="append", default=[], help="filter Stage names (repeatable)")
    p.add_argument("--load", action="append", default=[], help="filter Load SSA names (repeatable, e.g. in3)")
    p.add_argument("--list", action="store_true", help="print available (kernel, stage, load) probes for the first IR and exit")
    p.add_argument("--k-iter", type=int, default=0)
    p.add_argument("--warp-id", type=int, default=0)
    p.add_argument("--out", default="/tmp/bank_conflicts.html")
    p.add_argument("--theme", choices=("dark", "light"), default="dark")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="Also render to image (format from suffix: .png/.jpg/.webp/.pdf/.svg). Repeatable. Requires '.[visualize]' extra.",
    )
    p.add_argument("--transparent", dest="transparent", action="store_true", default=True)
    p.add_argument("--no-transparent", dest="transparent", action="store_false")
    args = p.parse_args()

    stage_filter = set(args.stage) if args.stage else None
    load_filter = set(args.load) if args.load else None

    if args.list:
        g = graph_from_ir_path(_parse_ir_arg(args.ir[0])[0])
        results = simulate_graph(g, stage_filter, args.k_iter, args.warp_id, load_filter)
        print(f"{'kernel':32}  {'stage':18}  {'load':6}  {'max_way':>7}  {'LDS.32':>7}  {'LDS.128':>8}  {'vec':>3}  index")
        print("-" * 120)
        for r in sorted(results, key=lambda x: (x.tile_op_name, x.stage_name, x.load_name)):
            idx = ", ".join(r.index_repr)
            print(
                f"{r.tile_op_name:32}  {r.stage_name:18}  {r.load_name:6}  "
                f"{r.max_way:>7}  {r.conflict_events:>7}  {r.lds128_events:>8}  {r.vec_group_size:>3}  ({idx})"
            )
        return

    columns: list[dict] = []
    for arg in args.ir:
        path, label = _parse_ir_arg(arg)
        g = graph_from_ir_path(path)
        panels = simulate_graph(g, stage_filter, args.k_iter, args.warp_id, load_filter)
        union_panels = simulate_graph(g, stage_filter, args.k_iter, args.warp_id, None) if load_filter is not None else panels
        scalar_total = sum(p.conflict_events for p in panels)
        vec_total = sum(p.lds128_events for p in panels)
        print(f"{label}: {len(panels)} probes, ΣLDS.32={scalar_total}  ΣLDS.128={vec_total}")
        columns.append(
            {
                "label": label,
                "panels": _serialize(panels, all_panels_for_union=union_panels),
            }
        )

    html = emit_html(columns, args.out, theme=args.theme, transparent=args.transparent)
    print(f"saved {args.out}")

    for image_path in args.image:
        from deplodock.visualize.image import render as render_image

        render_image(html, image_path, transparent=args.transparent)
        print(f"saved {image_path}")


if __name__ == "__main__":
    main()
