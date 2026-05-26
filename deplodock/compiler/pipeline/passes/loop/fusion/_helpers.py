"""Shared helpers for the ``loop/fusion`` rules.

Lives in a ``_``-prefixed module so the pass loader skips it (only
``NNN_<name>.py`` files are loaded as rules — see ``Pass.load``). Both
``010_merge_loop_ops`` and ``005_split_shared_indexmap`` import from here,
so the pure-indexmap predicate and the Write-output renamer stay defined
once.
"""

from __future__ import annotations

from deplodock.compiler.ir.loop import Accum, Assign, Load, LoopOp, Write


def has_data_dependent_load(loop_op: LoopOp) -> bool:
    """True if any ``Load``'s *index* references an SSA name defined inside the
    body — i.e. a gather (``weight[(int)in0, ...]`` where ``in0 = load ids``).

    The splicer carries SSA dependencies a stmt declares via ``Stmt.deps()``,
    but a ``Load``'s ``deps()`` is empty even when its index reads another
    stmt's SSA (the gather index lives in the index Exprs, not a ``deps()``
    slot). Inlining such a producer therefore drops the index-defining Load
    and lets the index name collide with a same-named consumer SSA — silently
    turning the data-dependent row into whatever the consumer's ``in0`` holds
    (e.g. an embedding reads a constant row instead of ``input_ids[pos]``).
    Until the splicer threads index-SSA deps end-to-end, fusion leaves these
    producers as their own (correct) kernel.
    """
    defined = {n for s in loop_op.body.iter() for n in s.defines()}
    return any(e.free_vars() & defined for ld in loop_op.body.loads if isinstance(ld, Load) for e in ld.index)


def is_pure_indexmap(loop_op: LoopOp) -> bool:
    """Body contains only Loops / Loads / Writes / Selects — no compute
    (``Assign``) or ``Accum``.

    Such a kernel is an ``IndexMapOp`` lifted into Loop IR: broadcast,
    transpose, reshape, slice, cat. Its content is pure coord rewriting +
    copying. Fusing a non-indexmap producer (one with real compute)
    *into* such a consumer forces the producer's body to land inside
    the indexmap's iteration space — materializing any broadcast the
    indexmap was expressing lazily.
    """
    for s in loop_op.body.iter():
        if isinstance(s, (Assign, Accum)):
            return False
    return True


def rename_write_output(op: LoopOp, *, old: str, new: str) -> LoopOp:
    """Return ``op`` with every ``Write`` whose ``output == old`` rewritten
    to ``output=new`` (recursively descends into nested Loops). Used by
    fusion to align a spliced/duplicated root's Writes with its new graph
    node id (buf names == node ids)."""

    def fn(s):
        if isinstance(s, Write) and s.output == old:
            return Write(output=new, index=s.index, value=s.value)
        return s

    return LoopOp(body=op.body.map(fn))
