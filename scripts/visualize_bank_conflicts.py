"""Bank-conflict visualizer driven by real Tile-IR ``Stage``s.

Takes one or more IR JSON paths and renders one column per input. Each
column contains a card per ``(Stage, body-Load)`` pair showing the warp's
per-lane smem bank at one inner-loop iteration. Simulation lives in
``deplodock.compiler.diagnostics.bank_conflicts``; this script is a thin
CLI + ECharts emitter.

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
    # Build per-Stage union: (tile_op_name, stage_name) → {(r,c) → list of
    # (load_name, k_iter, lane, subst_idx_tuple)} merged across all body
    # Loads of the Stage. Also a per-cell map of Loads whose LDS conflicts
    # when touching that cell (so the tooltip can flag conflicts on the
    # focused Load specifically).
    stage_union: dict[tuple[str, str], dict[tuple[int, int], list[tuple]]] = {}
    stage_conflict_loads: dict[tuple[str, str], dict[tuple[int, int], set[str]]] = {}
    for p in union_source:
        key = (p.tile_op_name, p.stage_name)
        u = stage_union.setdefault(key, {})
        for cell, pairs in p.full_sweep_touched.items():
            subst = p.full_sweep_subst_idx.get(cell, ())
            for k, lane in pairs:
                u.setdefault(cell, []).append((p.load_name, k, lane, subst))
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
        # Per-Stage UNION across every body Load of the Stage: cell →
        # list of (load_name, k_iter, lane, subst_idx_tuple). The
        # ladder shows the union (so it answers "across the K loop,
        # which cells does this Stage's body ever read"); the top
        # heatmap stays per-Load (bank conflicts are per-LDS).
        union = stage_union.get((p.tile_op_name, p.stage_name), {})
        sweep_per_cell: dict[tuple[int, int], list[list]] = {}
        subst_per_cell: dict[tuple[int, int], dict[str, tuple]] = {}
        for (r, c), entries in union.items():
            if not (0 <= r < layout_rows and 0 <= c < layout_cols):
                continue
            sweep_per_cell[(r, c)] = [[ln, k, lane] for (ln, k, lane, _s) in entries]
            # Keep one substituted form per Load that hits the cell.
            for ln, _k, _l, s in entries:
                subst_per_cell.setdefault((r, c), {}).setdefault(ln, tuple(s))
        all_cells = sorted(set(touched_now) | set(sweep_per_cell))
        conflict_loads = stage_conflict_loads.get((p.tile_op_name, p.stage_name), {})
        touched_entries = [
            {
                "r": r,
                "c": c,
                "lanes_now": touched_now.get((r, c), []),
                # sweep: list of [load_name, k_iter, lane]
                "sweep": sweep_per_cell.get((r, c), []),
                # subst_by_load: {load_name: [substituted_index_string, ...]}
                "subst_by_load": {ln: list(s) for ln, s in subst_per_cell.get((r, c), {}).items()},
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


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>smem bank conflicts (IR-driven)</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root { --bg:transparent; --panel:transparent; --panel-2:transparent; --fg:#e8eaed; --muted:#6b7280; }
  *{box-sizing:border-box;}
  html,body{margin:0;background:transparent;color:var(--fg);
    font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
  .page{max-width:100%;margin:0;padding:0;}
  .columns{display:grid;grid-template-columns:repeat(__NCOL__,1fr);gap:8px;}
  .col-head{text-align:center;font-size:11px;text-transform:uppercase;letter-spacing:.16em;
    color:var(--muted);padding:4px 0 6px;margin-bottom:6px;border-bottom:1px solid rgba(255,255,255,.06);}
  .col-head .label{color:#7dd3fc;font-weight:600;}
  .card{background:transparent;border:none;border-radius:0;padding:0;box-shadow:none;margin-bottom:0;}
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
  .matrix{width:100%;max-width:320px;aspect-ratio:1/1;height:auto;margin:0 auto 10px;}
  /* Histogram shares the bank axis with the punchcard above —
     match its max-width so banks line up vertically. */
  .hist{width:100%;max-width:320px;height:80px;margin:2px auto 0;}
  .ladder{margin:4px auto 0;}
  .ladder-title{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);
    margin-top:10px;margin-bottom:4px;}
  .bank-legend{display:grid;grid-template-columns:repeat(8,1fr);gap:3px 6px;margin-top:8px;
    font-family:'JetBrains Mono',ui-monospace,monospace;font-size:9px;color:var(--muted);}
  .bank-legend span{display:inline-flex;align-items:center;gap:4px;white-space:nowrap;}
  .bank-legend i{width:8px;height:8px;border-radius:2px;display:inline-block;flex-shrink:0;}
  .bank-legend-shared{margin-top:14px;font-size:10px;}
  .hist-legend{display:flex;flex-wrap:wrap;gap:4px 12px;margin:6px auto 0;
    width:100%;max-width:320px;font-size:10px;color:var(--muted);}
  .hist-legend span{display:inline-flex;align-items:center;gap:5px;}
  .hist-legend i{width:9px;height:9px;border-radius:2px;display:inline-block;}
  .empty{color:var(--muted);font-size:12px;padding:32px 16px;text-align:center;
    background:rgba(255,255,255,.02);border-radius:10px;border:1px dashed rgba(255,255,255,.06);}
  .legend{display:flex;gap:18px;margin-top:28px;color:var(--muted);font-size:12px;}
  .legend span{display:inline-flex;align-items:center;gap:8px;}
  .legend i{width:10px;height:10px;border-radius:3px;display:inline-block;}
</style>
</head>
<body>
  <div class="page">
    <div class="columns" id="columns"></div>
    <div class="bank-legend bank-legend-shared" id="shared-bank-legend"></div>
  </div>
<script>
const PAYLOAD = __PAYLOAD__;
const WARP=32, BANKS=32;
const palette={empty:'#2a2d33',ok:'#3ddc84',warn:'#ffb454',bad:'#ff5c7a'};
const cellColor=c=>c===0?palette.empty:c===1?palette.ok:c<=4?palette.warn:palette.bad;
const verdict=m=>m>4?'v-bad':m>1?'v-warn':'v-ok';
// Two distinct palettes — keep "color = bank" (smem layout) and
// "color = address" (punch card) visually independent so the reader
// doesn't conflate them.
//
// BANK_PALETTE: 16-color rainbow used in the smem-layout ladder. Each
// hue maps to a bank id (mod 16 — banks 0..15 use the first 16 hues,
// banks 16..31 reuse them).
const BANK_PALETTE=['#7dd3fc','#3ddc84','#ffb454','#ff5c7a','#c084fc','#fcd34d','#67e8f9','#fb923c',
                    '#a3e635','#f472b6','#60a5fa','#34d399','#fde047','#fb7185','#818cf8','#facc15'];
// ADDR_PALETTE: warm-only palette (yellows, oranges, reds, browns) used
// in the lane×bank punch card. A warp typically has 1-4 distinct
// addresses per Load, so a short palette is fine. Warm-only avoids
// overlap with the cool/rainbow BANK_PALETTE — the reader can tell at
// a glance which plot a color belongs to.
const ADDR_PALETTE=['#fde047','#fb923c','#ef4444','#a3a3a3','#facc15','#fdba74','#dc2626','#78350f'];

const root=document.getElementById('columns');
PAYLOAD.columns.forEach((col,ci)=>{
  const colEl=document.createElement('div');
  colEl.innerHTML=`<div class="col-head"><span class="label">${col.label}</span></div>`;
  // Append BEFORE init so document.getElementById can find the chart hosts.
  root.appendChild(colEl);
  if(col.panels.length===0){
    const e=document.createElement('div');e.className='empty';e.textContent='no Stages found';
    colEl.appendChild(e);
  }
  col.panels.forEach((p,pi)=>{
    const id=`c${ci}_p${pi}`;
    const card=document.createElement('div');card.className='card';
    card.innerHTML=`
      <div class="ladder-title" style="margin-top:0">bank access punchcard — ${p.buf_short}</div>
      <div class="matrix" id="m_${id}"></div>
      <div class="hist" id="h_${id}"></div>
      <div class="hist-legend">
        <span><i style="background:#3ddc84"></i>1 lane (no conflict)</span>
        <span><i style="background:#ffb454"></i>2–4 lanes (mild)</span>
        <span><i style="background:#ff5c7a"></i>&gt;4 lanes (heavy)</span>
      </div>
      <div class="ladder-title">smem layout — bank per (row, col)</div>
      <div class="ladder" id="l_${id}" style="${(()=>{
        // 3:1 wide rectangles for ladder cells (cell_w = 3 × cell_h).
        // Container fills column; height = rows × cell_h + axis. Cell
        // height is computed to keep total ≤ 360 px while ensuring the
        // 3× wider cells don't exceed 100% column width.
        const axisH = 24, maxH = 360;
        const N = Math.max(2, Math.min(6, Math.floor((maxH - axisH) / p.layout.rows)));
        const h = p.layout.rows * N + axisH;
        return `width:100%; height:${h}px`;
      })()}"></div>
      ${p.notes.length ? `<div class="card-notes">${p.notes.join('<br/>')}</div>` : ''}`;
    colEl.appendChild(card);

    const m=echarts.init(document.getElementById(`m_${id}`),null,{renderer:'canvas'});
    const md=[];
    for(let l=0;l<WARP;l++) for(let b=0;b<BANKS;b++)
      md.push({value:[b,l,0],itemStyle:{color:palette.empty}});
    p.lane_banks.forEach((bank,lane)=>{
      // Color cells by ADDRESS index. Same color = same address (broadcast).
      // Different colors stacked in one bank column = different addresses
      // serialized = real bank conflict.
      const addrIdx = p.lane_addr_idx[lane];
      const color = ADDR_PALETTE[addrIdx % ADDR_PALETTE.length];
      md.push({
        value:[bank,lane,1],
        itemStyle:{color: color, shadowBlur:6, shadowColor:color+'66'},
        addrIdx: addrIdx,
      });
    });
    m.setOption({
      backgroundColor:'transparent',
      tooltip:{backgroundColor:'#0e1014',borderColor:'#2a2d33',textStyle:{color:'#e8eaed',fontSize:12},
        formatter:pt=>{const [b,l,c]=pt.value;
          if (c === 0) return `bank ${b}<br/><span style="color:#6b7280">no lane</span>`;
          const ai = pt.data.addrIdx;
          const dist = p.distinct_addrs[b];
          const verdict = dist === 1 ? '<span style="color:#3ddc84">broadcast — 0 events</span>'
                       : dist <= 4 ? `<span style="color:#ffb454">${dist}-way conflict</span>`
                                   : `<span style="color:#ff5c7a">${dist}-way conflict</span>`;
          return `lane <b>${l}</b> → bank <b>${b}</b>, addr-color <b>#${ai}</b><br/>${verdict}`;
        }},
      grid:{left:38,right:8,top:8,bottom:42},
      xAxis:{type:'category',data:[...Array(BANKS).keys()],name:'bank',nameLocation:'middle',nameGap:22,
        nameTextStyle:{color:'#6b7280',fontSize:11},axisLine:{lineStyle:{color:'#2a2d33'}},
        axisTick:{show:false},axisLabel:{color:'#6b7280',fontSize:10,interval:3,showMaxLabel:true,showMinLabel:true}},
      yAxis:{type:'category',data:[...Array(WARP).keys()],inverse:true,name:'lane',
        nameLocation:'middle',nameGap:28,nameTextStyle:{color:'#6b7280',fontSize:11},
        axisLine:{lineStyle:{color:'#2a2d33'}},axisTick:{show:false},
        axisLabel:{color:'#6b7280',fontSize:10,interval:3,showMaxLabel:true,showMinLabel:true}},
      series:[{type:'heatmap',data:md,progressive:0,
        itemStyle:{borderRadius:2,borderColor:'#161a21',borderWidth:1},
        emphasis:{itemStyle:{borderColor:'#fff',borderWidth:1.5}},
        animationDuration:500,animationEasing:'cubicOut'}]});

    const h=echarts.init(document.getElementById(`h_${id}`),null,{renderer:'canvas'});
    h.setOption({
      backgroundColor:'transparent',
      tooltip:{backgroundColor:'#0e1014',borderColor:'#2a2d33',textStyle:{color:'#e8eaed',fontSize:12},
        formatter:pt=>`bank <b>${pt.name}</b><br/>${pt.value} lane(s)`},
      grid:{left:38,right:8,top:6,bottom:22},
      xAxis:{type:'category',data:[...Array(BANKS).keys()],
        axisLine:{lineStyle:{color:'#2a2d33'}},axisTick:{show:false},
        axisLabel:{color:'#6b7280',fontSize:10,interval:3,showMaxLabel:true,showMinLabel:true}},
      yAxis:{type:'value',min:0,max:8,splitLine:{lineStyle:{color:'#1c212b'}},
        axisLabel:{color:'#6b7280',fontSize:10},axisLine:{show:false},axisTick:{show:false}},
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
    // touchedNow: cells the warp accessed AT THE CURRENT k_iter (white outline)
    // sweep: cells the warp accessed across the full inner-loop sweep
    //   (tooltip shows the full provenance per cell)
    const touchedNow = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.lanes_now]));
    const sweep = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.sweep]));
    const substByLoad = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.subst_by_load]));
    // Cells touched by the focused Load (the one this card represents)
    // across the whole K sweep. Highlighted with a white outline so
    // the reader can spot when two highlighted cells in the same bank
    // column share a color = conflict for that LDS.
    const focusLoad = lay.load_name;
    const focusCells = new Set(
      lay.touched.filter(t => t.subst_by_load && t.subst_by_load[focusLoad]).map(t => `${t.r},${t.c}`)
    );
    const conflictLoadsByCell = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.conflict_loads || []]));
    for (let r = 0; r < lay.rows; r++) {
      for (let c = 0; c < lay.cols; c++) {
        const bank = lay.banks[r][c];
        const isPad = (c >= lay.data_cols) || (r >= lay.data_rows);
        const k = `${r},${c}`;
        const reachable = sweep.has(k);
        const isFocus = focusCells.has(k);
        const color = isPad ? '#3a3f48' : BANK_PALETTE[bank % BANK_PALETTE.length];
        // Three opacity levels: focused-Load cells (top), other
        // reachable cells (middle), padding/unreachable (bottom).
        // Reading rule: two cells with the high-opacity rows in the
        // same column sharing a color → conflict for that LDS.
        const op = isPad ? 0.18 : (isFocus ? 0.95 : (reachable ? 0.32 : 0.18));
        ldrData.push({
          value:[c, r, bank],
          itemStyle:{color: color, opacity: op},
        });
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
          // One substituted form per Load that hits this cell.
          const substMap = substByLoad.get(k) || {};
          const substLines = Object.entries(substMap).sort().map(
            ([ln, subst]) => `<code style="color:#3ddc84">${ln}[${subst.join(', ')}]</code>`
          );
          const loadSection = substLines.length ? `<br/>${substLines.join('<br/>')}` : '';
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
        axisLine:{lineStyle:{color:'#2a2d33'}}, axisTick:{show:false},
        axisLabel:{color:'#6b7280', fontSize:9,
          interval: Math.max(0, Math.floor(lay.cols/8) - 1),
          showMaxLabel: true, showMinLabel: true},
      },
      yAxis:{
        type:'category', data:[...Array(lay.rows).keys()], inverse:true,
        axisLine:{lineStyle:{color:'#2a2d33'}}, axisTick:{show:false},
        axisLabel:{color:'#6b7280', fontSize:9,
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
    chip.innerHTML = `<i style="background:${BANK_PALETTE[bank % BANK_PALETTE.length]}"></i>bank ${bank}`;
    host.appendChild(chip);
  });
})();
</script>
</body>
</html>
"""


def emit_html(columns: list[dict], out_path: str) -> None:
    n = max(1, len(columns))
    html = (
        HTML.replace("__MAXW__", str(min(2200, 480 * n + 80)))
        .replace("__NCOL__", str(n))
        .replace("__PAYLOAD__", json.dumps({"columns": columns}))
    )
    with open(out_path, "w") as f:
        f.write(html)


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
        # Run twice when ``--load`` is set: once unfiltered (for the
        # per-Stage union ladder) and once filtered (for the visible
        # cards).
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

    emit_html(columns, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
