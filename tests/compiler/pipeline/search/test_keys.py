"""``op_cache_key`` — a static matmul and its symbolic-axis (masked-tile) twin
must never share a key, at any stage.

No key rewrite was needed for the symbolic work: the loop/tile/kernel key digests
the body structure (a symbolic loop extent differs structurally from a static
one) and the cuda key digests rendered source + launch geometry (the ``int
seq_len`` runtime arg, the boundary guard, the ``Expr`` grid). These tests pin
that invariant so a future key normalization can't silently merge the twins —
they are different deployment artifacts with different variant spaces.
"""

from __future__ import annotations

from deplodock.commands.trace import graph_from_code
from deplodock.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.search.keys import op_cache_key
from deplodock.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

SNIPPET = "torch.matmul(torch.randn(64,64), torch.randn(64,64))"


def _keys(dynamic: bool, passes) -> set[str]:
    shapes = build_torch_dynamic_shapes(parse_position_specs(["seq_len@x0:0"])) if dynamic else None
    graph, _, _ = graph_from_code(SNIPPET, dynamic_shapes=shapes)
    compiled = Pipeline.build(passes).run(graph)
    keys = {op_cache_key(n.op) for n in compiled.nodes.values()}
    keys.discard(None)
    assert keys, "no kernel-bearing ops produced a key"
    return keys


def test_static_and_symbolic_twins_never_collide_loop_stage():
    assert _keys(False, LOOP_PASSES).isdisjoint(_keys(True, LOOP_PASSES))


def test_static_and_symbolic_twins_never_collide_cuda_stage():
    assert _keys(False, CUDA_PASSES).isdisjoint(_keys(True, CUDA_PASSES))
