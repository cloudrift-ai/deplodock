"""Tests that ``006_chunk_reduce`` chunks non-matmul reduces so
``007_stage_inputs`` can fit candidate slabs in its smem budget.

Two angles:

- **Firing tests** on whole frontend graphs via ``TILE_PASSES`` —
  confirms the rule fires (or doesn't) on representative shapes and
  that ``stage_inputs`` then admits Stages.
- **Scoped unit tests** that drive ``_qualifies`` / ``_chunk_loop``
  directly — covers idempotence, the matmul-shape guard, and the
  fan-in requirement without standing up the full pipeline.
"""

from __future__ import annotations

import importlib

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.frontend.ir import SdpaOp
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop
from deplodock.compiler.pipeline import TILE_PASSES, run_pipeline

_mod = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.006_chunk_reduce")
_qualifies = _mod._qualifies
_chunk_loop = _mod._chunk_loop


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- firing tests ----------------------------------------------------


def test_sdpa_seq512_fires_chunk_reduce(recording_dump):
    """SDPA at seq_len=512 produces a fused softmax + V-projection
    kernel whose softmax reduces sit at K=512. ``002_split_matmul_k``
    only chunks the V-projection (matmul-shaped); the two softmax
    reduces stay at K=512 and ``stage_inputs`` would bail (32 KB slab)
    without ``006_chunk_reduce``."""
    g = Graph()
    _input(g, "q", (1, 8, 512, 64))
    _input(g, "k", (1, 8, 512, 64))
    _input(g, "v", (1, 8, 512, 64))
    g.add_node(op=SdpaOp(), inputs=["q", "k", "v"], output=Tensor("o", (1, 8, 512, 64)), node_id="o")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["o"]

    run_pipeline(g, TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "chunk_reduce" in fired, fired
    # And staging should now fire on the chunked kernel.
    assert "stage_inputs" in fired, fired


def test_sdpa_seq128_does_not_need_chunk_reduce(recording_dump):
    """At seq_len=128 the candidate slab is 16×128×4 = 8 KB — already
    within the per-slab cap. ``stage_inputs`` admits Stages directly,
    no chunking needed (the rule should bail with RuleSkipped)."""
    g = Graph()
    _input(g, "q", (1, 8, 128, 64))
    _input(g, "k", (1, 8, 128, 64))
    _input(g, "v", (1, 8, 128, 64))
    g.add_node(op=SdpaOp(), inputs=["q", "k", "v"], output=Tensor("o", (1, 8, 128, 64)), node_id="o")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["o"]

    run_pipeline(g, TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    # Staging should fire (slab fits without chunking).
    assert "stage_inputs" in fired, fired


# --- scoped _qualifies / _chunk_loop tests --------------------------


def _t_axes(*names_and_extents) -> tuple[Axis, ...]:
    return tuple(Axis(n, e) for n, e in names_and_extents)


def _softmax_max_reduce(K: int = 512) -> Loop:
    """``Loop(K, body=(Load attn[a0, a1_i, K], Accum(max)))`` — ``a1_i``
    is the post-blockify thread-tile var so the slab includes a 16-wide
    cache axis (matches what ``005_blockify_launch`` produces)."""
    return Loop(
        axis=Axis("k", K),
        body=(
            Load(name="v", input="attn", index=(Var("a0"), Var("a1_i"), Var("k"))),
            Accum(name="acc", value="v", op="max"),
        ),
    )


def test_qualifies_softmax_with_fanin():
    """One thread axis (``a3_i``) is absent from the Load index → fan-in
    exists, qualifies for chunking."""
    loop = _softmax_max_reduce()
    thread_axes = _t_axes(("a1_i", 16), ("a3_i", 16))
    assert _qualifies(loop, thread_axes) is True


def test_qualifies_no_fanin_skipped():
    """Every thread axis appears in the Load index → no fan-in,
    chunking would not help staging."""
    loop = Loop(
        axis=Axis("k", 512),
        body=(
            Load(name="v", input="attn", index=(Var("a1_i"), Var("a3_i"), Var("k"))),
            Accum(name="acc", value="v", op="add"),
        ),
    )
    thread_axes = _t_axes(("a1_i", 16), ("a3_i", 16))
    assert _qualifies(loop, thread_axes) is False


def test_qualifies_matmul_shape_skipped():
    """Two distinct buffers K-indexed + Accum → matmul-shaped, deferred
    to ``002_split_matmul_k``."""
    loop = Loop(
        axis=Axis("k", 512),
        body=(
            Load(name="a", input="A", index=(Var("a0"), Var("k"))),
            Load(name="b", input="B", index=(Var("k"), Var("a1_i"))),
            Assign(name="m", op="multiply", args=("a", "b")),
            Accum(name="acc", value="m", op="add"),
        ),
    )
    thread_axes = _t_axes(("a1_i", 16), ("a3_i", 16))
    assert _qualifies(loop, thread_axes) is False


def test_qualifies_no_k_indexed_load_skipped():
    """The Load doesn't reference ``k`` → no benefit from chunking
    (nothing to stage along K)."""
    loop = Loop(
        axis=Axis("k", 512),
        body=(
            Load(name="c", input="bias", index=(Var("a0"),)),
            Accum(name="acc", value="c", op="add"),
        ),
    )
    thread_axes = _t_axes(
        ("a1_i", 16),
    )
    assert _qualifies(loop, thread_axes) is False


def test_qualifies_slab_already_fits_skipped():
    """K small enough that the candidate slab is already within the
    16 KB cap → ``007_stage_inputs`` admits a Stage without chunking,
    so we skip (avoid bloating IR / enabling premature TMA)."""
    loop = _softmax_max_reduce(K=64)  # 16 × 64 × 4 = 4 KB ≤ 16 KB
    thread_axes = _t_axes(("a1_i", 16), ("a3_i", 16))
    assert _qualifies(loop, thread_axes) is False


def test_qualifies_already_chunked_skipped():
    """Idempotence — a reduce whose immediate body is itself a reduce
    Loop has already been chunked; re-firing would create a 3-level
    nest. We guard by inspecting top-level body Loops."""
    inner = _softmax_max_reduce(K=64)
    rechunk_target = Loop(axis=Axis("k", 64), body=(inner,))
    thread_axes = _t_axes(("a1_i", 16), ("a3_i", 16))
    assert _qualifies(rechunk_target, thread_axes) is False


def test_chunk_loop_picks_largest_divisor_under_budget():
    """K=512 with a 16-wide thread axis appearing in the load → largest
    BK with 16×BK×4 ≤ 8 KB is 128. Outer extent = 512/128 = 4."""
    loop = _softmax_max_reduce(K=512)
    thread_axes = _t_axes(("a1_i", 16), ("a3_i", 16))
    chunked = _chunk_loop(loop, thread_axes)
    assert chunked is not None
    assert chunked.axis.name == "k_o"
    assert int(chunked.axis.extent) == 4
    inner = chunked.body[0]
    assert isinstance(inner, Loop)
    assert inner.axis.name == "k_i"
    assert int(inner.axis.extent) == 128


def test_chunk_loop_picks_smaller_bk_for_wider_thread_axis():
    """K=512 with a 64-wide thread axis appearing in the load → BK=128
    would bust the slab cap (64×128×4 = 32 KB). Picker must pick BK=32
    so the slab is 64×32×4 = 8 KB ≤ headroom."""
    loop = _softmax_max_reduce(K=512)
    thread_axes = _t_axes(("a1_i", 64), ("a3_i", 64))
    chunked = _chunk_loop(loop, thread_axes)
    assert chunked is not None
    inner = chunked.body[0]
    assert int(inner.axis.extent) == 32
    assert int(chunked.axis.extent) == 16


def test_chunk_loop_non_divisible_returns_none():
    """K=500 has no divisor in the candidate list → returns None."""
    loop = Loop(
        axis=Axis("k", 500),
        body=(
            Load(name="v", input="attn", index=(Var("a0"), Var("k"))),
            Accum(name="acc", value="v", op="add"),
        ),
    )
    thread_axes = _t_axes(
        ("a1_i", 16),
    )
    assert _chunk_loop(loop, thread_axes) is None


def test_chunk_loop_rewrites_index_substitution():
    """Inner body's Load index should reference ``k_o*BK + k_i`` instead
    of ``k`` after chunking."""
    loop = _softmax_max_reduce(K=512)
    thread_axes = _t_axes(("a1_i", 16), ("a3_i", 16))
    chunked = _chunk_loop(loop, thread_axes)
    inner = chunked.body[0]
    load = next(s for s in inner.body if isinstance(s, Load))
    # Index dim 2 (``k`` originally) should now be the substituted expr.
    idx_vars = {v for v in load.index[2].free_vars()}
    assert idx_vars == {"k_o", "k_i"}
