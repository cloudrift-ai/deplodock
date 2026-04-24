"""Standalone tests for ``splice_loop_ops``.

Each test builds producer/consumer ``LoopOp``s by hand and splices them
directly, asserting on the merged body structure. This skips the usual
frontend → tensor-IR → loop-IR pipeline so failures point at the splicer
itself rather than some upstream lowering quirk.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopOp,
    Write,
    iter_body,
    splice_loop_ops,
    splice_loops,
)
from deplodock.compiler.ir.tensor.ir import ElementwiseOp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ops_by_name(op: LoopOp) -> dict[str, object]:
    """Flatten body, index Assigns/Loads/Accums by their SSA name."""
    out: dict[str, object] = {}
    for s in iter_body(op.body):
        name = getattr(s, "name", None)
        if name is not None:
            out[name] = s
    return out


def _count_kind(op: LoopOp, cls: type) -> int:
    return sum(1 for s in iter_body(op.body) if isinstance(s, cls))


def _elementwise_fns(op: LoopOp) -> list[str]:
    return [s.op.name for s in iter_body(op.body) if isinstance(s, Assign)]


# ---------------------------------------------------------------------------
# Fixtures — shared axes
# ---------------------------------------------------------------------------


A0 = Axis("a0", 4)
A1 = Axis("a1", 8)
K = Axis("k", 16)


# ---------------------------------------------------------------------------
# Basic pipeline: pointwise producer → pointwise consumer
# ---------------------------------------------------------------------------


def test_pointwise_chain():
    """producer: y = exp(x)  ;  consumer: z = add(y, y)."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", source=0, index=(Var("a0"),)),
                    Assign(name="y", op=ElementwiseOp("exp"), args=("x",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", source=0, index=(Var("a0"),)),
                    Assign(name="z", op=ElementwiseOp("add"), args=("yv", "yv")),
                    Write(output=0, index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source=0)
    assert merged is not None
    fns = _elementwise_fns(merged)
    # exp from producer + copy alias for the Load + add from consumer.
    assert "exp" in fns
    assert "add" in fns
    # Producer's Load (consumer had no other inputs) survives as source 0.
    loads = [s for s in iter_body(merged.body) if isinstance(s, Load)]
    assert len(loads) == 1
    assert loads[0].source == 0
    # Single Write, referencing the final add.
    writes = [s for s in iter_body(merged.body) if isinstance(s, Write)]
    assert len(writes) == 1


# ---------------------------------------------------------------------------
# Shared intermediate: consumer uses producer output twice at same index → one emission
# ---------------------------------------------------------------------------


def test_shared_intermediate_deduped():
    """Two consumer refs to the same producer value under identity σ share one binding."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", source=0, index=(Var("a0"),)),
                    Assign(name="y", op=ElementwiseOp("exp"), args=("x",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    # Consumer loads producer twice at the same index — both solve σ = {a0: a0}.
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="ya", source=0, index=(Var("a0"),)),
                    Load(name="yb", source=0, index=(Var("a0"),)),
                    Assign(name="z", op=ElementwiseOp("add"), args=("ya", "yb")),
                    Write(output=0, index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source=0)
    assert merged is not None
    # Only one exp in the merged body (producer chain materializes once).
    assert _elementwise_fns(merged).count("exp") == 1


# ---------------------------------------------------------------------------
# Different σs: two consumer loads at different indices → two emissions
# ---------------------------------------------------------------------------


def test_different_indices_emit_twice():
    """Consumer loads a 2D producer with two different index permutations —
    the two σs are distinct, so the producer's chain materializes twice."""
    # Producer axes (a0, a1); writes y[a0, a1] = exp(x[a0, a1]).
    A1_same = Axis("a1", 4)  # square so the transposed load is well-typed.
    A0_same = Axis("a0", 4)
    producer = LoopOp(
        body=(
            Loop(
                axis=A0_same,
                body=(
                    Loop(
                        axis=A1_same,
                        body=(
                            Load(name="x", source=0, index=(Var("a0"), Var("a1"))),
                            Assign(name="y", op=ElementwiseOp("exp"), args=("x",)),
                            Write(output=0, index=(Var("a0"), Var("a1")), value="y"),
                        ),
                    ),
                ),
            ),
        ),
    )
    # Consumer reads the producer at (a0, a1) and (a1, a0) — transposed.
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0_same,
                body=(
                    Loop(
                        axis=A1_same,
                        body=(
                            Load(name="ya", source=0, index=(Var("a0"), Var("a1"))),
                            Load(name="yb", source=0, index=(Var("a1"), Var("a0"))),
                            Assign(name="z", op=ElementwiseOp("add"), args=("ya", "yb")),
                            Write(output=0, index=(Var("a0"), Var("a1")), value="z"),
                        ),
                    ),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source=0)
    assert merged is not None
    # Two distinct σs ({a0:a0,a1:a1}, {a0:a1,a1:a0}) → exp materializes twice.
    assert _elementwise_fns(merged).count("exp") == 2


# ---------------------------------------------------------------------------
# Reduction producer: sum over k, consumer is pointwise on the result
# ---------------------------------------------------------------------------


def test_reduction_producer():
    """Producer: s = sum_k x[a0,k]. Consumer: y = exp(s[a0])."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", source=0, index=(Var("a0"), Var("k"))),
                            Accum(name="s", value="x", op=ElementwiseOp("add")),
                        ),
                    ),
                    Write(output=0, index=(Var("a0"),), value="s"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="sv", source=0, index=(Var("a0"),)),
                    Assign(name="y", op=ElementwiseOp("exp"), args=("sv",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source=0)
    assert merged is not None
    # The Accum survives in the merged body.
    assert _count_kind(merged, Accum) == 1
    # Elementwise chain includes the producer-load copy + the consumer's exp.
    assert "exp" in _elementwise_fns(merged)


# ---------------------------------------------------------------------------
# Consumer has extra input: source-remapping puts consumer inputs after producer's
# ---------------------------------------------------------------------------


def test_consumer_extra_input_source_remap():
    """Consumer has producer at source 0 and an unrelated buffer at source 1;
    after splice, that unrelated buffer should load from merged source 1
    (producer's input is at 0)."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", source=0, index=(Var("a0"),)),
                    Assign(name="y", op=ElementwiseOp("exp"), args=("x",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", source=0, index=(Var("a0"),)),  # from producer
                    Load(name="b", source=1, index=(Var("a0"),)),  # unrelated
                    Assign(name="z", op=ElementwiseOp("add"), args=("yv", "b")),
                    Write(output=0, index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source=0)
    assert merged is not None
    loads = {s.name: s for s in iter_body(merged.body) if isinstance(s, Load)}
    # Producer's Load survives at source 0; consumer's unrelated Load shifts to 1.
    assert any(ld.source == 0 for ld in loads.values())
    assert any(ld.source == 1 for ld in loads.values())
    assert merged.num_inputs == 2


# ---------------------------------------------------------------------------
# Literal in producer write index: no σ-binding needed, still splices
# ---------------------------------------------------------------------------


def test_live_axes_computed():
    """``LoopMeta.live_axes`` captures axes transitively used through each
    dep's Expr subtrees. For Accum, the reduce axis is excluded.

    Normalization canonicalizes SSA and axis names (``Load→in0``,
    ``Accum→acc0``; free axes first then reduce axes), so we assert on
    ``meta.scopes`` (now the post-reduce binding scope for Accum) and
    ``meta.reduce_axes`` rather than pre-normalize names.
    """
    op = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Loop(
                        axis=K,
                        body=(
                            Load(name="x", source=0, index=(Var("a0"), Var("k"))),
                            Accum(name="s", value="x", op=ElementwiseOp("add")),
                        ),
                    ),
                    Write(output=0, index=(Var("a0"),), value="s"),
                ),
            ),
        ),
    )
    meta = op.analyze()
    [(load_name, load_scope)] = [(n, s) for n, s in meta.scopes.items() if isinstance(meta.defs[n], Load)]
    [(acc_name, acc_scope)] = [(n, s) for n, s in meta.scopes.items() if isinstance(meta.defs[n], Accum)]
    # Load sees both loop axes, Accum binds post-reduce so its scope is shorter.
    assert {a.name for a in load_scope.enclosing} == meta.live_axes[load_name]
    assert acc_name in meta.reduce_axes
    reduce_axis = meta.reduce_axes[acc_name]
    assert reduce_axis.name not in meta.live_axes[acc_name]
    # Accum's binding scope = load's enclosing minus the reduce axis.
    assert meta.scopes[acc_name].enclosing == tuple(a for a in load_scope.enclosing if a.name != reduce_axis.name)


# ---------------------------------------------------------------------------
# N-way chain fusion via splice_loop_chain
# ---------------------------------------------------------------------------


def test_chain_three_loops():
    """Fuse a → b → c in one splice call. ``a`` is pointwise exp, ``b`` negates
    the result, ``c`` adds a bias. The merged kernel should contain exp, neg,
    add — with one Load for x (a's input) and one for bias (c's extra input)."""
    a = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", source=0, index=(Var("a0"),)),
                    Assign(name="y", op=ElementwiseOp("exp"), args=("x",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    b = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="av", source=0, index=(Var("a0"),)),  # reads a
                    Assign(name="y", op=ElementwiseOp("negative"), args=("av",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    c = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="bv", source=0, index=(Var("a0"),)),  # reads b
                    Load(name="bias", source=1, index=(Var("a0"),)),  # external
                    Assign(name="y", op=ElementwiseOp("add"), args=("bv", "bias")),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )

    merged = splice_loops(
        loops={"a": a, "b": b, "c": c},
        splice_edges={("b", 0): ("a", 0), ("c", 0): ("b", 0)},
        input_remap={
            ("a", 0): 0,  # a's x → merged slot 0
            ("c", 1): 1,  # c's bias → merged slot 1
        },
    )
    assert merged is not None
    fns = _elementwise_fns(merged)
    assert "exp" in fns
    assert "negative" in fns
    assert "add" in fns
    # Two external Loads remain (x, bias); the splice edges collapsed into copy aliases.
    loads = [s for s in iter_body(merged.body) if isinstance(s, Load)]
    assert len(loads) == 2
    assert {ld.source for ld in loads} == {0, 1}
    assert merged.num_inputs == 2


# ---------------------------------------------------------------------------
# Multi-output support
# ---------------------------------------------------------------------------


def test_multi_output_root_preserves_all_writes():
    """A root whose body contains Writes at multiple output indices should
    seed each of them — the merged kernel carries all Writes."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", source=0, index=(Var("a0"),)),
                    Assign(name="y", op=ElementwiseOp("exp"), args=("x",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        ),
    )
    # Consumer produces two outputs: a negation and an absolute value — both
    # reading the producer.
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", source=0, index=(Var("a0"),)),
                    Assign(name="z0", op=ElementwiseOp("negative"), args=("yv",)),
                    Assign(name="z1", op=ElementwiseOp("abs"), args=("yv",)),
                    Write(output=0, index=(Var("a0"),), value="z0"),
                    Write(output=1, index=(Var("a0"),), value="z1"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source=0)
    assert merged is not None
    writes = [s for s in iter_body(merged.body) if isinstance(s, Write)]
    assert {w.output for w in writes} == {0, 1}
    fns = _elementwise_fns(merged)
    # Producer chain materialized once (shared σ); both consumer ops present.
    assert fns.count("exp") == 1
    assert "negative" in fns
    assert "abs" in fns


def test_multi_output_splice_target():
    """A splice target with two outputs (output=0 and output=1) — distinct
    splice edges read each output. The target's expression chain reconstructs
    separately for each output via the standard worklist walk."""
    # Target: y0 = exp(x), y1 = neg(x). Two outputs.
    target = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", source=0, index=(Var("a0"),)),
                    Assign(name="y0", op=ElementwiseOp("exp"), args=("x",)),
                    Assign(name="y1", op=ElementwiseOp("negative"), args=("x",)),
                    Write(output=0, index=(Var("a0"),), value="y0"),
                    Write(output=1, index=(Var("a0"),), value="y1"),
                ),
            ),
        ),
    )
    # Sink: z = add(target_out0, target_out1). Two consumer input slots both
    # pointing at the target — slot 0 reads output 0 (exp), slot 1 reads
    # output 1 (neg).
    sink = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="a", source=0, index=(Var("a0"),)),
                    Load(name="b", source=1, index=(Var("a0"),)),
                    Assign(name="z", op=ElementwiseOp("add"), args=("a", "b")),
                    Write(output=0, index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loops(
        loops={"target": target, "sink": sink},
        splice_edges={
            ("sink", 0): ("target", 0),  # sink's slot 0 ← target's output 0 (exp)
            ("sink", 1): ("target", 1),  # sink's slot 1 ← target's output 1 (neg)
        },
        input_remap={
            ("target", 0): 0,  # target's x → merged slot 0
        },
    )
    assert merged is not None
    fns = _elementwise_fns(merged)
    # Both target ops appear in the merged body once each.
    assert fns.count("exp") == 1
    assert fns.count("negative") == 1
    assert "add" in fns
    # Only one external Load (target's x); the splice edges replaced sink's Loads.
    loads = [s for s in iter_body(merged.body) if isinstance(s, Load)]
    assert len(loads) == 1
    assert merged.num_inputs == 1


def test_literal_producer_write_index():
    """Producer writes at (a0, 0); consumer reads at (a0, 0). The Literal dim
    contributes no σ binding and shouldn't block splicing."""
    producer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="x", source=0, index=(Var("a0"),)),
                    Assign(name="y", op=ElementwiseOp("exp"), args=("x",)),
                    Write(output=0, index=(Var("a0"), Literal(0)), value="y"),
                ),
            ),
        ),
    )
    consumer = LoopOp(
        body=(
            Loop(
                axis=A0,
                body=(
                    Load(name="yv", source=0, index=(Var("a0"), Literal(0))),
                    Assign(name="z", op=ElementwiseOp("negative"), args=("yv",)),
                    Write(output=0, index=(Var("a0"),), value="z"),
                ),
            ),
        ),
    )

    merged = splice_loop_ops(producer, consumer, source=0)
    assert merged is not None
    assert "exp" in _elementwise_fns(merged)
    assert "negative" in _elementwise_fns(merged)
