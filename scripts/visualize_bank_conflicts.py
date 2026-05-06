"""Bank-conflict visualizer driven by real Tile-IR ``Stage``s.

Each input contributes one column. Within a column, every ``Stage`` ×
``Load`` pair gets a card showing the warp's per-lane smem bank at one
fixed inner-loop iteration. The simulation lives in
``deplodock.compiler.diagnostics.bank_conflicts``; this script is a
thin CLI + ECharts emitter.

Modes:

  --ir PATH[:LABEL] (repeatable)
      Load JSON dumps produced under ``DEPLODOCK_DUMP_DIR`` (any
      post-tile-pass dump that contains ``TileOp`` nodes — typically
      ``07_*_stage_inputs.json`` or later).

  --chunk-reduce-diff [--seq-len N]
      Run the tile pipeline live on an SDPA(B=1, H=8, seq=N, d=64)
      graph twice — with and without ``006_chunk_reduce`` enabled
      (gated via ``DEPLODOCK_DISABLE_CHUNK_REDUCE``) — and compare.
"""

from __future__ import annotations

import argparse
import json
import os

from deplodock.compiler.diagnostics.bank_conflicts import BankConflictResult, simulate_graph
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import SdpaOp
from deplodock.compiler.pipeline import TILE_PASSES, run_pipeline


def graph_from_ir_path(path: str) -> Graph:
    with open(path) as f:
        return Graph.from_dict(json.load(f))


def graph_from_sdpa(seq_len: int, disable_chunk_reduce: bool) -> Graph:
    g = Graph()
    for name in ("q", "k", "v"):
        g.add_node(InputOp(), [], Tensor(name, (1, 8, seq_len, 64)), node_id=name)
    g.add_node(SdpaOp(), ["q", "k", "v"], Tensor("o", (1, 8, seq_len, 64)), node_id="o")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["o"]
    prev = os.environ.get("DEPLODOCK_DISABLE_CHUNK_REDUCE")
    os.environ["DEPLODOCK_DISABLE_CHUNK_REDUCE"] = "1" if disable_chunk_reduce else "0"
    try:
        run_pipeline(g, TILE_PASSES)
    finally:
        if prev is None:
            del os.environ["DEPLODOCK_DISABLE_CHUNK_REDUCE"]
        else:
            os.environ["DEPLODOCK_DISABLE_CHUNK_REDUCE"] = prev
    return g


def _serialize(panels: list[BankConflictResult]) -> list[dict]:
    return [
        {
            "panel_title": f"{p.stage_name} ← {p.buf}",
            "formula": (f"{p.stage_class}({p.rows}×{p.cols})  " + (f"pad={p.pad}" if p.pad and any(p.pad) else "no pad")),
            "lane_banks": p.lane_banks,
            "counts": p.counts,
            "max_way": p.max_way,
            "raw_max_way": p.raw_max_way,
            "conflict_events": p.conflict_events,
            "avg_way": p.avg_way,
            "rows": p.rows,
            "cols": p.cols,
            "pad": list(p.pad),
            "smem_bytes": p.smem_bytes,
            "notes": [
                f"index = ({', '.join(p.index_repr)})",
                f"enclosing axes = {', '.join(p.enclosing_axes) or '(none)'}",
            ],
        }
        for p in panels
    ]


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

const root=document.getElementById('columns');
PAYLOAD.columns.forEach((col,ci)=>{
  const colEl=document.createElement('div');
  colEl.innerHTML=`<div class="col-head"><span class="label">${col.label}</span></div>`;
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
      <div class="card-notes">${p.notes.join('<br/>')}</div>`;
    colEl.appendChild(card);

    const m=echarts.init(document.getElementById(`m_${id}`),null,{renderer:'canvas'});
    const md=[];
    for(let l=0;l<WARP;l++) for(let b=0;b<BANKS;b++)
      md.push({value:[b,l,0],itemStyle:{color:palette.empty}});
    p.lane_banks.forEach((bank,lane)=>{
      const c=p.counts[bank];
      md.push({value:[bank,lane,c],itemStyle:{color:cellColor(c),shadowBlur:6,shadowColor:cellColor(c)+'88'}});
    });
    m.setOption({
      backgroundColor:'transparent',
      tooltip:{backgroundColor:'#0e1014',borderColor:'#2a2d33',textStyle:{color:'#e8eaed',fontSize:12},
        formatter:pt=>{const [b,l,c]=pt.value;
          return c===0?`bank ${b}<br/><span style="color:#6b7280">no lane</span>`
                      :`lane <b>${l}</b> → bank <b>${b}</b><br/>contention: <b>${c}</b> lane(s)`;}},
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
      series:[{type:'bar',data:p.counts.map(c=>({value:c,itemStyle:{
        color:{type:'linear',x:0,y:0,x2:0,y2:1,
          colorStops:[{offset:0,color:cellColor(c)},{offset:1,color:cellColor(c)+'55'}]},
        borderRadius:[3,3,0,0]}})),barWidth:'70%',
        animationDuration:600,animationEasing:'cubicOut'}]});
    window.addEventListener('resize',()=>{m.resize();h.resize();});
  });
  root.appendChild(colEl);
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
    p.add_argument("--ir", action="append", default=[], help="PATH[:LABEL] — repeatable")
    p.add_argument("--chunk-reduce-diff", action="store_true", help="Run pipeline live with/without chunk_reduce on SDPA")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--stage", action="append", default=[], help="filter Stage names (repeatable)")
    p.add_argument("--k-iter", type=int, default=0)
    p.add_argument("--warp-id", type=int, default=0)
    p.add_argument("--out", default="/tmp/bank_conflicts.html")
    args = p.parse_args()

    stage_filter = set(args.stage) if args.stage else None
    columns: list[dict] = []

    if args.chunk_reduce_diff:
        for label, disabled in (("chunk_reduce ENABLED", False), ("chunk_reduce DISABLED", True)):
            print(f"running pipeline: {label}...")
            g = graph_from_sdpa(args.seq_len, disable_chunk_reduce=disabled)
            panels = simulate_graph(g, stage_filter, args.k_iter, args.warp_id)
            print(f"  → {len(panels)} stage probe(s)")
            columns.append({"label": label, "panels": _serialize(panels)})
        subtitle = f"SDPA(B=1, H=8, seq={args.seq_len}, d=64) · k_iter={args.k_iter} · warp={args.warp_id}"
    elif args.ir:
        for arg in args.ir:
            path, label = _parse_ir_arg(arg)
            g = graph_from_ir_path(path)
            panels = simulate_graph(g, stage_filter, args.k_iter, args.warp_id)
            print(f"{label}: {len(panels)} stage probe(s)")
            columns.append({"label": label, "panels": _serialize(panels)})
        subtitle = f"k_iter={args.k_iter} · warp={args.warp_id} · {len(args.ir)} input IR(s)"
    else:
        p.error("provide --ir PATH (repeatable) or --chunk-reduce-diff")

    emit_html(columns, subtitle, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
