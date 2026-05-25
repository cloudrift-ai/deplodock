"""Demote elementwise Assigns to a narrower dtype when their result
flows to a narrower-dtype Write (or to another demoted Assign).

Pattern the pass closes (from softmax over fp16):

    __half acc_b = ...;
    float acc1_b = ...;
    float v2 = 1.0f / acc1_b;
    __half in2 = x_smem[a];
    __half v3 = in2 - acc_b;
    __half v4 = hexp(v3);
    float v5 = v2 * __half2float(v4);    // <-- f32 compute
    out[a] = __float2half(v5);           // <-- demoted at store

After demotion of ``v5`` to f16, the multiply uses native fp16 and the
store happens without a final conversion:

    __half v5 = __float2half(v2) * v4;
    out[a] = v5;

In the user's words: the pass detects ``half → float → ... → float →
half`` sequences and demotes the float-compute middle to half.

## Algorithm

Backward dataflow over the KernelOp body:

1. Build a def map ``name → defining_stmt`` from a deep ``Body.iter()``.
2. Seed a worklist with the ``value`` of every ``Write`` whose output
   buffer is fp16.
3. For each seeded name, if it's defined by an ``Assign`` whose op has
   a native fp16 form, mark that Assign demoted. Recurse to the
   Assign's args: any arg defined by another native-fp16-form Assign
   that we haven't visited becomes a candidate. Args defined by
   ``Load`` / ``Accum`` / non-native-fp16 ``Assign`` are stopping
   points — the renderer inserts the conversion at the consumer's use
   site via ``ctx.target.convert``.
4. Walk the body via ``Body.map`` and stamp ``Assign.dtype = F16`` on
   every Assign in the demote set.

## Why this is safe

- ``Accum`` is not crossed — its dtype is the freeze point chosen by
  ``020_place_inits``; changing it would change reduction semantics.
- ``Load`` is not crossed either — the source buffer's dtype is the
  ground truth; Load.render already declares the local in that dtype.
- Ops without a native fp16 form aren't demoted; they stay in fp32
  via the existing promote-and-demote path in ``Assign.render``.
- The demoted Assign's mixed-dtype args (e.g. one f32 reciprocal feeding
  a f16 multiply) are handled by ``_args_at_dtype`` in the renderer:
  each f32 arg gets a per-use ``__float2half`` wrap.

## When the pass doesn't fire

- Output is fp32 → no fp16-demanding Writes.
- The chain feeding the Write contains an op without a native fp16
  form, AND that op is the immediate predecessor of the Write — the
  pass demotes nothing in that case (raising ``RuleSkipped``).
- The body is already demoted (idempotent — the pass skips Assigns
  with ``self.dtype is not None``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget
from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

# Single shared target instance — the pass only consults its
# ``has_native_op`` predicate, which is stateless.
_TARGET = CudaRenderTarget()


def rewrite(match: Match, root: Node) -> Graph | None:
    top: TileOp = root.op
    body = top.body

    # Output buffer dtypes — read off ``TileOp.outputs`` (same shape as
    # the legacy ``KernelOp.outputs``: matcher-populated graph Tensors).
    # Fall back to ``root.output`` for the op's own node-output.
    out_dtypes: dict[str, str] = {n: t.dtype.name for n, t in top.outputs.items()}
    out_dtypes.setdefault(root.id, root.output.dtype.name)

    # ssa-name → defining Stmt (deep traversal).
    defs: dict[str, Stmt] = {}
    for s in body.iter():
        for n in s.defines():
            defs[n] = s

    # Step 1 (forward): the set of SSA names that already carry an f16
    # value end-to-end — Loads with stamped f16 dtype, Accums / Inits
    # with ``dtype = F16``, and any Assign whose op has a native f16
    # form AND at least one arg is itself an f16 carrier. Iterates to
    # fixed point.
    fp16_carrier: set[str] = _seed_fp16_carriers(body)
    changed = True
    while changed:
        changed = False
        for s in body.iter():
            if not isinstance(s, Assign):
                continue
            if s.name in fp16_carrier:
                continue
            if s.dtype is not None:  # already stamped — its dtype is the source of truth
                if s.dtype.name == "f16":
                    fp16_carrier.add(s.name)
                    changed = True
                continue
            if not _TARGET.has_native_op(s.op.name, "f16"):
                continue
            if any(a in fp16_carrier for a in s.args):
                fp16_carrier.add(s.name)
                changed = True

    # Step 2 (backward): names that transitively feed a Write whose
    # output buffer is f16.
    feeds_f16_write: set[str] = set()
    queue: list[str] = []
    for s in body.iter():
        if isinstance(s, Write) and out_dtypes.get(s.output, "f32") == "f16":
            # ``s.value`` asserts scalar; pre-materialize Writes are
            # always scalar (vectorize_stores runs later).
            if s.value not in feeds_f16_write:
                feeds_f16_write.add(s.value)
                queue.append(s.value)
    while queue:
        name = queue.pop()
        defining = defs.get(name)
        if isinstance(defining, Assign):
            for arg in defining.args:
                if arg not in feeds_f16_write:
                    feeds_f16_write.add(arg)
                    queue.append(arg)

    # Initial demote set: Assigns both carrying an f16 value (forward)
    # AND feeding an f16 Write (backward). Excludes Assigns already
    # stamped.
    demote_set: set[str] = {n for n in (fp16_carrier & feeds_f16_write) if isinstance(defs.get(n), Assign) and defs[n].dtype is None}

    # Step 3 (extension): an Assign whose result is only consumed by
    # already-demoted Assigns or by f16 Writes can also be demoted,
    # *even when* none of its args is an f16 carrier. This catches
    # ``v2 = reciprocal(acc1_b)`` in softmax — acc1_b is f32 from the
    # sum accumulator, but v2's only use is the f16-demoted ``v5 =
    # multiply(v2, v4)``. Demoting v2 lets the renderer compute the
    # reciprocal in f32 and convert once at the result rather than
    # leaving an extra ``__float2half(v2)`` at the v5 use site.
    uses = _build_uses(body)
    changed = True
    while changed:
        changed = False
        for s in body.iter():
            if not isinstance(s, Assign):
                continue
            if s.name in demote_set:
                continue
            if s.dtype is not None:
                continue
            if not _TARGET.has_native_op(s.op.name, "f16"):
                continue
            if _all_uses_demoted(s.name, uses, demote_set, out_dtypes):
                demote_set.add(s.name)
                changed = True

    if not demote_set:
        raise RuleSkipped("no demotable Assigns")

    # Step 4: rebuild the body with demoted Assigns.
    def _maybe_demote(s: Stmt) -> Stmt:
        if isinstance(s, Assign) and s.name in demote_set and s.dtype is None:
            return replace(s, dtype=F16)
        return s

    new_body: Body = body.map(_maybe_demote)
    if new_body == body:
        raise RuleSkipped("no change after demotion (already stamped)")

    return TileOp(body=new_body, name=top.name)


def _build_uses(body: Body) -> dict[str, list[Stmt]]:
    """Use chain for ``Assign`` / ``Write`` / ``Accum`` consumers.
    For each SSA name, list every Stmt that reads it as an arg/value."""
    uses: dict[str, list[Stmt]] = {}
    for s in body.iter():
        deps: list[str] = []
        if isinstance(s, Assign):
            deps = list(s.args)
        elif isinstance(s, Write):
            deps = list(s.values)
        elif isinstance(s, Accum):
            deps = [s.value]
        for d in deps:
            uses.setdefault(d, []).append(s)
    return uses


def _all_uses_demoted(name: str, uses: dict[str, list[Stmt]], demote_set: set[str], out_dtypes: dict[str, str]) -> bool:
    """An Assign ``name`` can be safely demoted if every consumer is
    already in the demote set (so it'll consume ``name`` in fp16) or
    is a Write to an fp16 buffer (where the demoted value goes
    directly to the store). Accum consumers block demotion — changing
    the accumulator's input dtype changes reduction semantics."""
    use_list = uses.get(name, [])
    if not use_list:
        return False
    for u in use_list:
        if isinstance(u, Write):
            if out_dtypes.get(u.output, "f32") != "f16":
                return False
        elif isinstance(u, Assign):
            if u.name not in demote_set:
                return False
        else:
            # Accum or other — conservative.
            return False
    return True


def _seed_fp16_carriers(body: Body) -> set[str]:
    """Initial fp16 carriers — SSA names whose value naturally lives in
    fp16 from the moment they're defined:

    - ``Load`` whose stamped ``dtype.name == "f16"`` (either reading
      from an fp16 graph buffer or from an fp16-stamped Stage source).
      ``030_stamp_types`` populates ``Load.dtype`` from either side
      before this pass runs.
    - ``Accum`` / ``Init`` with ``dtype == F16``.

    Pre-materialize there are no ``WarpShuffle`` / ``TreeHalve`` /
    ``Smem`` Stmts to consider — those land at materialize time.
    Pre-materialize Loads are always scalar (vectorize_loads runs
    later), so ``s.names`` has length 1.
    """
    carriers: set[str] = set()
    for s in body.iter():
        if isinstance(s, Load):
            if s.dtype is not None and s.dtype.name == "f16":
                carriers.update(s.names)
        elif isinstance(s, Accum):
            if (s.dtype or F32).name == "f16":
                carriers.add(s.name)
        elif isinstance(s, Init):
            if s.dtype.name == "f16":
                carriers.add(s.name)
    return carriers
