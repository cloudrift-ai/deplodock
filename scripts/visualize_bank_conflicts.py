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


def _serialize(panels: list[BankConflictResult]) -> list[dict]:
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
        # Mark cells the warp actually touched in this LDS (row, col) →
        # list of lanes that read the cell. Drawn with a white outline
        # overlay on top of the ladder; tooltip shows the lanes (and they
        # collapse to one entry when broadcast).
        touched_map: dict[tuple[int, int], list[int]] = {}
        for lane, addr in enumerate(p.lane_addrs):
            r, c = divmod(addr, row_stride)
            if 0 <= r < layout_rows and 0 <= c < layout_cols:
                touched_map.setdefault((r, c), []).append(lane)
        touched_entries = [{"r": r, "c": c, "lanes": lanes} for (r, c), lanes in sorted(touched_map.items())]
        out.append(
            {
                "panel_title": f"{p.stage_name} ← {p.buf}",
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
                "notes": [
                    f"index = ({', '.join(p.index_repr)})",
                    f"enclosing axes = {', '.join(p.enclosing_axes) or '(none)'}",
                    (f"LDS.32 events: {p.conflict_events}  ·  LDS.128 events: {p.lds128_events}  (vec={p.vec_group_size})"),
                ],
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
  :root { --bg:#0e1014; --panel:#161a21; --panel-2:#1c212b; --fg:#e8eaed; --muted:#6b7280; }
  *{box-sizing:border-box;}
  html,body{margin:0;background:var(--bg);color:var(--fg);
    font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
  .page{max-width:__MAXW__px;margin:0 auto;padding:48px 32px;}
  .eyebrow{text-transform:uppercase;letter-spacing:.18em;font-size:11px;color:var(--muted);margin-bottom:8px;}
  h1{font-size:28px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em;}
  .subtitle{color:var(--muted);font-size:14px;margin-bottom:36px;}
  .columns{display:grid;grid-template-columns:repeat(__NCOL__,1fr);gap:22px;}
  .col-head{text-align:center;font-size:12px;text-transform:uppercase;letter-spacing:.18em;
    color:var(--muted);padding:8px 0;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,.06);}
  .col-head .label{color:#7dd3fc;font-weight:600;}
  .card{background:linear-gradient(180deg,var(--panel),var(--panel-2));
    border:1px solid rgba(255,255,255,.04);border-radius:14px;padding:18px 16px 12px;
    box-shadow:0 1px 0 rgba(255,255,255,.03) inset,0 8px 24px rgba(0,0,0,.35);margin-bottom:16px;}
  .card-title{font-size:14px;font-weight:600;letter-spacing:-.01em;}
  .card-formula{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:12px;color:var(--muted);margin-top:4px;}
  .card-notes{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:11px;color:var(--muted);margin-top:6px;line-height:1.55;}
  .card-verdict{display:inline-flex;align-items:center;gap:6px;margin-top:10px;padding:4px 10px;
    border-radius:999px;font-size:12px;font-weight:600;background:rgba(255,255,255,.04);}
  .dot{width:6px;height:6px;border-radius:50%;}
  .v-ok{color:#3ddc84;} .v-ok .dot{background:#3ddc84;}
  .v-warn{color:#ffb454;} .v-warn .dot{background:#ffb454;}
  .v-bad{color:#ff5c7a;} .v-bad .dot{background:#ff5c7a;}
  .matrix{width:100%;height:240px;}
  .hist{width:100%;height:100px;margin-top:4px;}
  .ladder{width:100%;margin-top:8px;}
  .ladder-title{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);
    margin-top:14px;margin-bottom:6px;}
  .empty{color:var(--muted);font-size:12px;padding:32px 16px;text-align:center;
    background:rgba(255,255,255,.02);border-radius:10px;border:1px dashed rgba(255,255,255,.06);}
  .legend{display:flex;gap:18px;margin-top:28px;color:var(--muted);font-size:12px;}
  .legend span{display:inline-flex;align-items:center;gap:8px;}
  .legend i{width:10px;height:10px;border-radius:3px;display:inline-block;}
</style>
</head>
<body>
  <div class="page">
    <div class="eyebrow">tile-IR bank-conflict probe</div>
    <h1>smem bank conflicts</h1>
    <div class="subtitle">__SUBTITLE__</div>
    <div class="columns" id="columns"></div>
    <div class="legend">
      <span><i style="background:#3ddc84"></i> 1 lane (no conflict)</span>
      <span><i style="background:#ffb454"></i> 2–4 lanes (mild)</span>
      <span><i style="background:#ff5c7a"></i> &gt;4 lanes (heavy)</span>
      <span><i style="background:#2a2d33"></i> 0 lanes</span>
    </div>
  </div>
<script>
const PAYLOAD = __PAYLOAD__;
const WARP=32, BANKS=32;
const palette={empty:'#2a2d33',ok:'#3ddc84',warn:'#ffb454',bad:'#ff5c7a'};
const cellColor=c=>c===0?palette.empty:c===1?palette.ok:c<=4?palette.warn:palette.bad;
const verdict=m=>m>4?'v-bad':m>1?'v-warn':'v-ok';
// Categorical palette for "color = address". Picked for high contrast on
// dark bg + reasonable distinguishability. Cycles modulo length when a
// panel has more unique addresses than colors.
const ADDR_PALETTE=['#7dd3fc','#3ddc84','#ffb454','#ff5c7a','#c084fc','#fcd34d','#67e8f9','#fb923c',
                    '#a3e635','#f472b6','#60a5fa','#34d399','#fde047','#fb7185','#818cf8','#facc15'];

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
      <div class="card-title">${p.panel_title}</div>
      <div class="card-formula">${p.formula}</div>
      <span class="card-verdict ${verdict(p.max_way)}"><span class="dot"></span>
        max-way ${p.max_way} · avg ${p.avg_way.toFixed(1)} lane/bank</span>
      <div class="matrix" id="m_${id}"></div>
      <div class="hist" id="h_${id}"></div>
      <div class="ladder-title">smem layout — bank per (row,col), accessed cells outlined</div>
      <div class="ladder" id="l_${id}" style="height:${Math.min(360, 8 + p.layout.rows * 6)}px"></div>
      <div class="card-notes">${p.notes.join('<br/>')}</div>`;
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
      grid:{left:38,right:8,top:8,bottom:28},
      xAxis:{type:'category',data:[...Array(BANKS).keys()],name:'bank',nameLocation:'middle',nameGap:22,
        nameTextStyle:{color:'#6b7280',fontSize:11},axisLine:{lineStyle:{color:'#2a2d33'}},
        axisTick:{show:false},axisLabel:{color:'#6b7280',fontSize:10,interval:3}},
      yAxis:{type:'category',data:[...Array(WARP).keys()],inverse:true,name:'lane',
        nameLocation:'middle',nameGap:28,nameTextStyle:{color:'#6b7280',fontSize:11},
        axisLine:{lineStyle:{color:'#2a2d33'}},axisTick:{show:false},
        axisLabel:{color:'#6b7280',fontSize:10,interval:3}},
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
        axisLabel:{color:'#6b7280',fontSize:10,interval:3}},
      yAxis:{type:'value',splitLine:{lineStyle:{color:'#1c212b'}},
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
    const touchedMap = new Map(lay.touched.map(t => [`${t.r},${t.c}`, t.lanes]));
    for (let r = 0; r < lay.rows; r++) {
      for (let c = 0; c < lay.cols; c++) {
        const bank = lay.banks[r][c];
        const isPad = (c >= lay.data_cols) || (r >= lay.data_rows);
        const isTouched = touchedMap.has(`${r},${c}`);
        const color = isPad ? '#3a3f48' : ADDR_PALETTE[bank % ADDR_PALETTE.length];
        ldrData.push({
          value:[c, r, bank],
          itemStyle:{
            color: color,
            opacity: isPad ? 0.45 : 1.0,
            borderColor: isTouched ? '#ffffff' : 'transparent',
            borderWidth: isTouched ? 1.5 : 0,
          },
        });
      }
    }
    // Compact a sorted list of ints into a "0-15, 17, 22-25" form for
    // the tooltip — readable when 16 lanes broadcast one cell.
    const compactRanges = arr => {
      if (!arr.length) return '';
      const parts = []; let s = arr[0], e = arr[0];
      for (let i = 1; i < arr.length; i++) {
        if (arr[i] === e + 1) { e = arr[i]; }
        else { parts.push(s === e ? `${s}` : `${s}-${e}`); s = arr[i]; e = arr[i]; }
      }
      parts.push(s === e ? `${s}` : `${s}-${e}`);
      return parts.join(', ');
    };
    ldr.setOption({
      backgroundColor:'transparent',
      tooltip:{
        backgroundColor:'#0e1014', borderColor:'#2a2d33',
        textStyle:{color:'#e8eaed', fontSize:12},
        formatter: pt => {
          const [c, r, bank] = pt.value;
          const lanes = touchedMap.get(`${r},${c}`);
          const padTag = (c >= lay.data_cols || r >= lay.data_rows)
            ? '<br/><span style="color:#6b7280">padding</span>' : '';
          let access = '';
          if (lanes) {
            access =
              `<br/><span style="color:#fff">★ accessed by warp</span>` +
              `<br/>load <code style="color:#7dd3fc">${lay.load_name}[${lay.index_expr}]</code>` +
              `<br/>lanes: <b>${compactRanges(lanes)}</b> (${lanes.length}×)`;
          }
          return `row=<b>${r}</b>, col=<b>${c}</b><br/>bank <b>${bank}</b>${access}${padTag}`;
        },
      },
      grid:{left:38, right:8, top:6, bottom:24},
      xAxis:{
        type:'category', data:[...Array(lay.cols).keys()],
        name:'col', nameLocation:'middle', nameGap:18,
        nameTextStyle:{color:'#6b7280', fontSize:10},
        axisLine:{lineStyle:{color:'#2a2d33'}}, axisTick:{show:false},
        axisLabel:{color:'#6b7280', fontSize:9, interval: Math.max(0, Math.floor(lay.cols/8) - 1)},
      },
      yAxis:{
        type:'category', data:[...Array(lay.rows).keys()], inverse:true,
        name:'row', nameLocation:'middle', nameGap:24,
        nameTextStyle:{color:'#6b7280', fontSize:10},
        axisLine:{lineStyle:{color:'#2a2d33'}}, axisTick:{show:false},
        axisLabel:{color:'#6b7280', fontSize:9, interval: Math.max(0, Math.floor(lay.rows/8) - 1)},
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
</script>
</body>
</html>
"""


def emit_html(columns: list[dict], subtitle: str, out_path: str) -> None:
    n = max(1, len(columns))
    html = (
        HTML.replace("__MAXW__", str(min(2200, 480 * n + 80)))
        .replace("__NCOL__", str(n))
        .replace("__SUBTITLE__", subtitle)
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
        panels = simulate_graph(g, stage_filter, args.k_iter, args.warp_id, load_filter)
        scalar_total = sum(p.conflict_events for p in panels)
        vec_total = sum(p.lds128_events for p in panels)
        print(f"{label}: {len(panels)} probes, ΣLDS.32={scalar_total}  ΣLDS.128={vec_total}")
        columns.append(
            {
                "label": f"{label} · ΣLDS.32={scalar_total}  ΣLDS.128={vec_total}",
                "panels": _serialize(panels),
            }
        )

    subtitle = f"k_iter={args.k_iter} · warp={args.warp_id} · {len(args.ir)} input IR(s)"
    emit_html(columns, subtitle, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
