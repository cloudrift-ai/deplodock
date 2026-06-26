"""fp16 matmul half2 accumulation window (``FK`` knob) — pack the K window.

The fp16 scalar matmul accumulates every product into an fp32 register master
(``__half2float(a) * __half2float(b)`` summed in f32). That leaves the GPU's 2×
packed-fp16 throughput on the table — ``__hfma2`` does two fp16 MACs per
instruction, but pure fp16 accumulation over a full K loop loses too much
precision. The ``FK`` window bounds the
rounding error: accumulate one **stage chunk** of K (the window, length
``FK = bk``, even) in a ``__half2`` register, then widen + horizontal-sum +
flush into the persistent fp32 master once per window.

This pass realizes the window from the FK=1 fp32 structure the planner emits
(it leaves the planner / staging / pipelining path untouched). For each
**window loop** — the innermost K reduce ``SerialTile`` carrying the matmul
``Accum``s (its extent is the window length ``bk``) — it:

  1. strip-mines the loop by 2 (trip count ``bk/2``);
  2. for each scalar operand ``Load``, emits the two adjacent-K copies + a
     ``Pack`` into ``__half2`` (``Pack`` already converts f16 operands with no
     cost);
  3. rewrites the ``multiply`` to ``F16x2`` and accumulates into a per-cell
     ``__half2`` window accumulator ``hacc`` (renders ``hacc += a2 * b2`` →
     nvcc fuses to ``__hfma2``);
  4. after the loop, **flushes**: ``Unpack(hacc) → w_lo, w_hi`` (the even/odd-K
     partial sums of the window), ``s = f32(w_lo) + f32(w_hi)`` (horizontal
     combine), ``facc += s`` into the original fp32 master.

Runs at 015 — after ``010_split_register_axes`` (so each FM/FN output cell
already owns its own master ``Accum``) and **before** ``020_place_inits`` /
``030_stamp_types`` / ``050_vectorize_loads``, so the loop body is still scalar
(no vectorized Loads to split) and ``place_inits`` keeps the master ``facc``
Init at ThreadTile scope while leaving the per-window ``hacc`` Init this pass
emits inside K_o untouched (the ``preplaced`` guard). Fires only on the
``FKWIN``-stamped fp16-matmul window variant — the reduce-axis FK (which also
stamps ``FK > 1`` but is realized by the fold in ``010_split_register_axes``)
and fp32 / bf16 / MMA kernels carry no ``FKWIN`` so the pass no-ops; ``FK=1`` is
byte-identical to today.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F16, F32, F16x2
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Pack, Stmt, Unpack
from deplodock.compiler.ir.tile.ir import GridTile, RegisterTile, SerialTile, StageBundle, StridedTile, ThreadTile, TileOp, WarpTile
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    top = root.op
    # ``FKWIN`` is stamped only on the fp16-matmul half2-window variant (the
    # planner verified fp16 operands). The reduce-axis FK also stamps ``FK > 1``
    # but NOT ``FKWIN`` — it's realized by the fold in 010_split_register_axes,
    # so this pass must not touch it.
    if int(top.knobs.get("FKWIN", 1)) <= 1:
        raise RuleSkipped("FK window not enabled")
    counter = [0]
    new_body, did = _walk(top.body, counter)
    if not did:
        raise RuleSkipped("no fp16 matmul window loop found")
    return TileOp(body=new_body, name=top.name, knobs=dict(top.knobs))


def _walk(body: Body, counter: list[int]) -> tuple[Body, bool]:
    """Recurse outer→in. A ``SerialTile`` whose *immediate* body carries
    ``Accum``s is a window loop — transform it; otherwise descend so peeled /
    nested window loops (cp.async pipelining replicates the K_i body) are all
    rewritten."""
    out: list[Stmt] = []
    did = False
    for s in body:
        if isinstance(s, SerialTile) and _is_window_loop(s):
            out.extend(_pack_window(s, counter))
            did = True
        elif isinstance(s, StageBundle):
            # The window K loop (the staged smem→register reduce) lives inside the
            # bundle's ``body`` at this stage — the bundle is only expanded to inline
            # Sync + cooperative loads later (kernel/100_materialize_tile). Recurse
            # into both bundle bodies so the window fires on a staged fp16 matmul.
            nb, db = _walk(s.body, counter)
            comp = s.compute if s.compute is not None else None
            nc, dc = _walk(comp, counter) if comp is not None else (comp, False)
            out.append(s.with_bodies((nc if comp is not None else Body(()), nb)))
            did = did or db or dc
        elif isinstance(s, (SerialTile, StridedTile, RegisterTile, GridTile, ThreadTile, WarpTile)):
            nb, d = _walk(s.body, counter)
            out.append(s.with_bodies((nb,)))
            did = did or d
        elif isinstance(s, Cond):
            nb, db = _walk(s.body, counter)
            ne, de = _walk(s.else_body, counter)
            out.append(Cond(cond=s.cond, body=nb, else_body=ne))
            did = did or db or de
        else:
            out.append(s)
    return Body(tuple(out)), did


def _is_window_loop(s: SerialTile) -> bool:
    """A window loop is an even-extent reduce ``SerialTile`` whose immediate
    body holds the matmul cell(s): each ``Accum`` reads a ``multiply`` of two
    fp16 operand ``Load``s. Guards against firing on a non-fp16 reduce or the
    outer K_o loop (whose direct body has no ``Accum``)."""
    accums = [x for x in s.body if isinstance(x, Accum)]
    if not accums or not s.axis.extent.is_static or s.axis.extent.as_static() % 2 != 0:
        return False
    muls = {x.name: x for x in s.body if isinstance(x, Assign) and x.op.semiring_product}
    loads = {x.name: x for x in s.body if isinstance(x, Load)}
    for acc in accums:
        mul = muls.get(acc.value)
        # The accumulated value must be a multiply of two fp16 Loads.
        if mul is None or len(mul.args) != 2:
            return False
        for a in mul.args:
            ld = loads.get(a)
            if ld is None or (ld.dtype is not None and ld.dtype != F16):
                return False
    return True


def _rename_one(old: str, new: str):
    def _f(name: str) -> str:
        return new if name == old else name

    return _f


def _pack_window(loop: SerialTile, counter: list[int]) -> list[Stmt]:
    """Strip-mine a window loop by 2 + pack into ``__hfma2``; emit the flush."""
    k = loop.axis
    fk = k.extent.as_static()
    kh = Axis(f"{k.name}p", fk // 2, source_axis=k.source_axis or k)
    two_kh = Literal(2, "int") * Var(kh.name)
    sig_lo = Sigma({k.name: two_kh})
    sig_hi = Sigma({k.name: two_kh + Literal(1, "int")})

    packed: dict[str, str] = {}  # scalar SSA name -> its __half2 pack name
    flush_map: list[tuple[str, str, object, object]] = []  # (master facc, window hacc, axes, op)
    new_body: list[Stmt] = []
    cid = counter[0]
    counter[0] += 1

    for st in loop.body:
        if isinstance(st, Load) and st.is_scalar:
            x = st.name
            lo, hi = f"{x}__lo", f"{x}__hi"
            new_body.append(st.rewrite(_rename_one(x, lo), sig_lo))
            new_body.append(st.rewrite(_rename_one(x, hi), sig_hi))
            pn = f"{x}__p{cid}"
            new_body.append(Pack(name=pn, low=lo, high=hi, dtype=F16x2))
            packed[x] = pn
        elif isinstance(st, Assign) and st.op.semiring_product:
            args = tuple(packed.get(a, a) for a in st.args)
            pn = f"{st.name}__p{cid}"
            new_body.append(Assign(name=pn, op=st.op, args=args, dtype=F16x2))
            packed[st.name] = pn
        elif isinstance(st, Accum):
            hacc = f"{st.name}__h{cid}"
            new_body.append(Accum(name=hacc, value=packed.get(st.value, st.value), op=st.op, dtype=F16x2, axes=st.axes))
            flush_map.append((st.name, hacc, st.axes, st.op))
        else:
            # Pass through anything else (defensive — _is_window_loop already
            # constrained the body to Load/multiply/Accum).
            new_body.append(st)

    # Init the per-window ``hacc`` right here (inside K_o, before the window) so
    # it resets each stage; the explicit Init makes ``020_place_inits`` suppress
    # its own (it would otherwise hoist it above K_o and accumulate fp16 over the
    # whole K — defeating the bounded-error window). The fp32 master ``facc``
    # keeps its place_inits-placed ThreadTile-scope Init (persists across K_o).
    out: list[Stmt] = [Init(name=hacc, op=acc_op, dtype=F16x2) for _facc, hacc, _axes, acc_op in flush_map]
    out.append(SerialTile(axis=kh, body=Body(tuple(new_body)), kind=loop.kind))
    for facc, hacc, axes, _op in flush_map:
        wlo, whi, s_name = f"{hacc}__wlo", f"{hacc}__whi", f"{hacc}__s"
        out.append(Unpack(low_name=wlo, high_name=whi, value=hacc, lane_dtype=F16))
        out.append(Assign(name=s_name, op="add", args=(wlo, whi), dtype=F32))
        out.append(Accum(name=facc, value=s_name, op="add", dtype=F32, axes=axes))
    return out
