"""Flash-attention shared helper — the ``FLASH`` knob, eligibility predicate, and
the streaming online-softmax ``LoopOp`` nest builder.

Flash recognition is a **loop-lifting** pass (``015_lift_sdpa_flash``): the
``LoopOp`` is constructed in the loop dialect, not at tensor-IR decomposition.
For the intact ``SdpaOp`` to survive to the lifting stage, the generic
``frontend/decomposition/010_sdpa`` rule **defers** (``RuleSkipped``) whenever
flash will handle the op — both sites consult the same :func:`flash_enabled` +
:func:`flash_shape_eligible` here so the decision can't drift. This module lives
under ``loop/lifting`` (the LoopOp owner) and is imported by both the lifting
pass and the decomposition deferral; it is the sole owner of the ``FLASH`` knob.

The nest fuses scaled-dot-product attention into ONE kernel that tiles the KV
(reduce) axis and never materializes the ``[S_q, S_k]`` score matrix. It runs one
independent streaming softmax per output element ``(…, m, d)`` — a correct, if
redundant, scalar form; the tensor-core P@V tier is future work
(``plans/online-softmax-flash-attention.md``)::

    for *batch, m (query rows), d (value dim):       # free / grid
      Init (m_i = -inf, l_i = 0, O_i = 0)            # running (max, denom, out)
      for kv in 0..S_k:                              # streaming reduce
        Init sacc = 0
        for dd in 0..head_dim: sacc += Q[…,m,dd]·K[…,kv,dd]   # score reduce
        s = sacc · scale
        FlashCombine((m_i,l_i,O_i), (s, V[…,kv,d]))  # the LSE rescale
      out[…,m,d] = O_i / l_i

Scope: static OR dynamic (symbolic ``seq_len`` on Q/K/V dim -2 — one cached kernel
carrying ``int seq_len`` serves every runtime size, the symbol landing on BOTH the
masked-row M and the symbolic reduce), causal or non-causal (causal masks the
score per element, ``kv ≤ m`` else −inf — tile-skip is a tensor-core-tier
follow-up), no explicit additive mask, no GQA. Read from the ``DEPLODOCK_FLASH=1``
env pin; the two-level ``OptionFork`` offer + ``AnalyticPrior`` cold-start are a
follow-up.
"""

from __future__ import annotations

import math

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, FlashCombine, Init, Load, Loop, Select, SelectBranch, Write
from deplodock.compiler.pipeline.knob import Knob, KnobType

# Structural fork: deploy the fused streaming nest, or fall through to the
# score-materializing 010_sdpa path. Declared like 005_split_demoted's SPLIT_CONE;
# auto-registered by knob.registry()'s module walk.
FLASH = Knob("FLASH", KnobType.BOOL, hints=(True, False), help="Fuse SDPA into the streaming online-softmax flash nest")


def flash_enabled() -> bool:
    raw = FLASH.raw()
    return raw is not None and FLASH.parse(raw)


def _static(d) -> int | None:
    """The static extent of a ``Dim``, or ``None`` when symbolic."""
    return d.as_static() if d.is_static else None


def flash_shape_eligible(q_shape: tuple, k_shape: tuple, v_shape: tuple, *, has_mask: bool) -> bool:
    """True iff the flash nest can serve this SDPA — static (only the seq axis may
    be symbolic), no explicit additive mask, no GQA. The decomposition deferral
    and the lifting rule MUST agree on this, so both call it."""
    if has_mask:
        return False
    if len(q_shape) < 2 or len(k_shape) < 2 or len(v_shape) < 2:
        return False
    batch = [_static(d) for d in q_shape[:-2]]
    if any(b is None for b in batch):
        return False  # symbolic batch / head — only the seq axis may be dynamic
    if [_static(d) for d in k_shape[:-2]] != batch or [_static(d) for d in v_shape[:-2]] != batch:
        return False  # GQA / mismatched batch dims
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    if head_dim is None or d_v is None:
        return False  # symbolic head_dim / value-dim
    if _static(k_shape[-1]) != head_dim:
        return False
    return v_shape[-2] == k_shape[-2]  # V seq must match K seq


def build_flash_frag(
    q_id: str, k_id: str, v_id: str, q_shape: tuple, k_shape: tuple, v_shape: tuple, out: Tensor, *, causal: bool
) -> Graph:
    """Build the fragment graph holding the fused flash ``LoopOp`` (+ its scale /
    -inf constants). The caller guarantees :func:`flash_shape_eligible`."""
    batch = [_static(d) for d in q_shape[:-2]]
    head_dim, d_v = _static(q_shape[-1]), _static(v_shape[-1])
    s_q_dim, s_k_dim = q_shape[-2], k_shape[-2]  # Dim instances — static int or symbolic seq_len
    scale = 1.0 / math.sqrt(head_dim)

    body = _flash_loop_body(q_id, k_id, v_id, "_flash_scale", batch, s_q_dim, s_k_dim, head_dim, d_v, out.name, causal=causal)
    nest = _wrap_free_axes(body, batch, s_q_dim, d_v)

    frag = Graph()
    for nid, shp in ((q_id, q_shape), (k_id, k_shape), (v_id, v_shape)):
        frag.add_node(op=InputOp(), inputs=[], output=Tensor(nid, shp, out.dtype), node_id=nid)
    inputs = [q_id, k_id, v_id, "_flash_scale"]
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
    frag.add_node(op=LoopOp(body=nest), inputs=inputs, output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
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
    """The per-output-element ``(…, m, d)`` body: streaming KV reduce + finalize.

    The score ``s = Σ_dd Q·K`` is an inner reduce nested in the KV streaming
    reduce, so ``Init(sacc)`` is pre-placed at the KV-body scope (reset per step):
    the streaming ``FlashCombine`` loop is a per-iteration boundary the default
    Init-placement would otherwise cross."""
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
