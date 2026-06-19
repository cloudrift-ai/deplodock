"""Flash-attention decomposition of ``SdpaOp`` — the streaming online-softmax nest.

Recognizes attention at ``SdpaOp`` (before the generic ``010_sdpa`` score-matrix
decomposition) and, when the ``FLASH`` structural-fork knob is on, emits a single
fused ``LoopOp`` that tiles the KV (reduce) axis and never materializes the
``[S_q, S_k]`` score matrix. The online-softmax recurrence rides the
``FlashCombine`` carrier (``plans/online-softmax-flash-attention.md``).

The nest (one independent streaming softmax per output element ``(…, m, d)`` — a
correct, if redundant, scalar form; the tensor-core P@V tier is future work)::

    for *batch, m (query rows), d (value dim):       # free / grid
      Init (m_i = -inf, l_i = 0, O_i = 0)            # running (max, denom, out)
      for kv in 0..S_k:                              # streaming reduce
        Init sacc = 0
        for dd in 0..head_dim: sacc += Q[…,m,dd]·K[…,kv,dd]   # score reduce
        s = sacc · scale
        FlashCombine((m_i,l_i,O_i), (s, V[…,kv,d]))  # the LSE rescale
      out[…,m,d] = O_i / l_i

Scope (so far): static OR dynamic (symbolic ``seq_len``) shapes, causal or
non-causal, no explicit additive mask, no GQA. The single dynamic axis (Q/K/V
dim -2) rides through as the actual ``Dim('seq_len')`` instance, landing on BOTH
the query (masked-row M) and KV (reduce) positions, so one cached kernel carrying
``int seq_len`` serves every runtime size. Causal masks the score per element
(``kv ≤ m`` keeps it, else −inf); the tile-skip-above-the-diagonal optimization
belongs to the future tensor-core tier. Anything else raises ``RuleSkipped`` so
``010_sdpa`` handles it. The ``FLASH``
knob is read from the env pin (``DEPLODOCK_FLASH=1`` / ``DEPLODOCK_KNOBS=FLASH=1``)
— the structural-fork offer + analytic cold-start wiring is a follow-up.
"""

from __future__ import annotations

import math

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.frontend.ir import SdpaOp
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, FlashCombine, Init, Load, Loop, Select, SelectBranch, Write
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", SdpaOp)]

# Structural fork: deploy the fused streaming nest, or fall through to the
# score-materializing 010_sdpa path. Declared like 005_split_demoted's
# SPLIT_CONE; auto-registered by knob.registry()'s module walk. (The two-level
# OptionFork offer + analytic cold-start term land in a follow-up — for now the
# env pin drives it so the nest is GPU-verifiable without changing defaults.)
FLASH = Knob("FLASH", KnobType.BOOL, hints=(True, False), help="Fuse SDPA into the streaming online-softmax flash nest")


def _flash_enabled() -> bool:
    raw = FLASH.raw()
    return raw is not None and FLASH.parse(raw)


def _static(d) -> int | None:
    """The static extent of a ``Dim``, or ``None`` when symbolic."""
    return d.as_static() if d.is_static else None


def rewrite(match: Match, root: Node, inp_q: Node, inp_k: Node, inp_v: Node, inp_mask: Node | None, out: Tensor) -> Graph | None:
    if not _flash_enabled():
        raise RuleSkipped("FLASH knob off — generic 010_sdpa handles this SDPA")
    if inp_mask is not None:
        raise RuleSkipped("flash: explicit additive mask not yet supported — fall through to 010_sdpa")
    causal = bool(root.op.is_causal)

    q, k, v = inp_q.output.shape, inp_k.output.shape, inp_v.output.shape
    if len(q) < 2 or len(k) < 2 or len(v) < 2:
        raise RuleSkipped(f"flash: need rank>=2 Q/K/V, got {len(q)}/{len(k)}/{len(v)}")
    # Batch dims + head_dim + value-dim must be static (only the seq axis may be
    # symbolic — that is the single dynamic axis attention carries). The seq axis
    # (Q/K/V dim -2) rides through as the actual symbolic ``Dim`` instance, so the
    # masked-row (M) + symbolic-reduce (K) machinery threads one ``seq_len`` symbol.
    batch = [_static(d) for d in q[:-2]]
    if any(b is None for b in batch):
        raise RuleSkipped("flash: symbolic batch/head dims unsupported — only the seq axis may be dynamic")
    if [_static(d) for d in k[:-2]] != batch or [_static(d) for d in v[:-2]] != batch:
        raise RuleSkipped("flash: GQA / mismatched batch dims — fall through to 010_sdpa")
    head_dim, d_v = _static(q[-1]), _static(v[-1])
    if head_dim is None or d_v is None:
        raise RuleSkipped("flash: symbolic head_dim / value-dim unsupported")
    if _static(k[-1]) != head_dim:
        raise RuleSkipped(f"flash: K head_dim {_static(k[-1])} != Q head_dim {head_dim}")
    s_q_dim, s_k_dim = q[-2], k[-2]  # Dim instances — static int or symbolic seq_len
    if v[-2] != s_k_dim:
        raise RuleSkipped(f"flash: V seq {v[-2]} != K seq {s_k_dim}")

    scale = 1.0 / math.sqrt(head_dim)
    body = _flash_loop_body(inp_q.id, inp_k.id, inp_v.id, "_flash_scale", batch, s_q_dim, s_k_dim, head_dim, d_v, out.name, causal=causal)
    nest = _wrap_free_axes(body, batch, s_q_dim, d_v)

    frag = open_fragment(match.graph, [inp_q, inp_k, inp_v])
    inputs = [inp_q.id, inp_k.id, inp_v.id, "_flash_scale"]
    frag.add_node(
        op=ConstantOp(name="_flash_scale", value=scale), inputs=[], output=Tensor("_flash_scale", (1,), out.dtype), node_id="_flash_scale"
    )
    if causal:
        # -inf bias for masked (key-after-query) positions: exp(-inf)=0, so a
        # masked score contributes nothing to the streaming softmax / output.
        frag.add_node(
            op=ConstantOp(name="_flash_ninf", value=-1e30), inputs=[], output=Tensor("_flash_ninf", (1,), out.dtype), node_id="_flash_ninf"
        )
        inputs.append("_flash_ninf")
    frag.add_node(
        op=LoopOp(body=nest),
        inputs=inputs,
        output=Tensor(out.name, out.shape, out.dtype),
        node_id=out.name,
    )
    frag.outputs = [out.name]
    return frag


def _batch_vars(n: int) -> tuple[Var, ...]:
    return tuple(Var(f"b{i}") for i in range(n))


def _flash_loop_body(
    q_buf: str,
    k_buf: str,
    v_buf: str,
    scale_buf: str,
    batch: list[int],
    s_q: Dim,
    s_k: Dim,
    head_dim: int,
    d_v: int,
    out_buf: str,
    *,
    causal: bool = False,
) -> tuple:
    """The per-output-element ``(…, m, d)`` body: streaming KV reduce + finalize."""
    bvars = _batch_vars(len(batch))
    q_idx = (*bvars, Var("m"), Var("dd"))
    k_idx = (*bvars, Var("kv"), Var("dd"))
    v_idx = (*bvars, Var("kv"), Var("d"))
    out_idx = (*bvars, Var("m"), Var("d"))

    score_reduce = Loop(
        axis=Axis(name="dd", extent=Dim(head_dim)),
        body=(
            Load(name="q_e", input=q_buf, index=q_idx),
            Load(name="k_e", input=k_buf, index=k_idx),
            Assign(name="qk", op="multiply", args=("q_e", "k_e")),
            Accum(name="sacc", value="qk", op=ElementwiseImpl("add")),
        ),
    )
    score: tuple = (Assign(name="s", op="multiply", args=("sacc", "scale_c")),)
    score_name = "s"
    prologue: tuple = ()
    if causal:
        # Causal mask: keep the score where key ≤ query (kv ≤ m), else −inf.
        prologue = (Load(name="ninf_c", input="_flash_ninf", index=()),)
        score += (
            Select(
                name="s_masked",
                branches=(
                    SelectBranch(value="s", select=BinaryExpr("<=", Var("kv"), Var("m"))),
                    SelectBranch(value="ninf_c", select=Literal(1, "int")),
                ),
            ),
        )
        score_name = "s_masked"
    kv_body = (
        Init(name="sacc", op=ElementwiseImpl("add"), dtype="f32"),
        score_reduce,
        *score,
        Load(name="v_e", input=v_buf, index=v_idx),
        FlashCombine(state=("m_i", "l_i", "O_i"), partial=(score_name, "v_e"), axes=("kv",)),
    )
    return (
        Load(name="scale_c", input=scale_buf, index=()),
        *prologue,
        Init(name="m_i", op=ElementwiseImpl("maximum"), dtype="f32"),
        Init(name="l_i", op=ElementwiseImpl("add"), dtype="f32"),
        Init(name="O_i", op=ElementwiseImpl("add"), dtype="f32"),
        Loop(axis=Axis(name="kv", extent=s_k), body=kv_body),
        Assign(name="res", op="divide", args=("O_i", "l_i")),
        Write(output=out_buf, index=out_idx, value="res"),
    )


def _wrap_free_axes(body: tuple, batch: list[int], s_q: Dim, d_v: int) -> tuple:
    """Wrap the per-element body in the free grid loops: batch…, query m, value d."""
    nest: tuple = (Loop(axis=Axis(name="m", extent=s_q), body=(Loop(axis=Axis(name="d", extent=Dim(d_v)), body=body),)),)
    for i in reversed(range(len(batch))):
        nest = (Loop(axis=Axis(name=f"b{i}", extent=Dim(batch[i])), body=nest),)
    return nest
