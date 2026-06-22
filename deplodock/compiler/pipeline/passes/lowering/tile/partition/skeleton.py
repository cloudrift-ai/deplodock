"""Hardware-free algebraic skeletons + the shared nest helpers.

One skeleton per regime the composer covers — pointwise (`MAP`), matmul
(`SEMIRING`), cooperative reduce (`MONOID`) — each naming the innermost free
axis `N`, the next-out one `M`, the extra outer free loops, and (for reduces)
the `K` axis. The recognition that fills them lives in `walk.py` (`walk_nest`);
this module holds only the dataclasses + the free-axis-chain helpers they share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.stmt import Loop, Stmt


@dataclass(frozen=True)
class MapAxis:
    """One free (map) axis to tile. ``extent`` is the static size, or the
    ``Dim`` hint for a symbolic axis (``symbolic=True``)."""

    loop: Loop
    symbolic: bool
    extent: int


@dataclass(frozen=True)
class PointwiseSkeleton:
    """Pointwise kernel shape: free axes + the body to tile."""

    inner_n: MapAxis
    outer_m: MapAxis | None
    extra_outer: tuple[Loop, ...]
    inner_body: tuple[Stmt, ...]
    leading: tuple[Stmt, ...]


@dataclass(frozen=True)
class FlashSkeleton:
    """Fused flash-attention nest (`DEPLODOCK_FLASH=1`): free output axes
    (q-rows / head-dim / head) + a `TWISTED_MONOID` streaming KV reduce whose
    body holds a nested `SEMIRING` QK^T reduce and the `FlashCombine` carrier
    (which renders its own online-softmax rescale). The composer tiles the free
    axes and serial-transforms both reduces; the carriers lower themselves. This
    is the scalar (redundant) nest — one streaming softmax per output element;
    the tensor-core P@V tier is future work.

    A symbolic ``seq_len`` lands on BOTH the free q-rows axis (masked tile, the
    same path pointwise uses) and the streaming KV reduce. The KV reduce streams
    masked: ``k_bounds`` maps each symbolic streaming-axis name to its runtime
    boundary ``Expr``, and ``build_flash_tile`` ceil-divides that axis' serial
    loop and guards each step with ``Cond(k < bound)`` — the TWISTED_MONOID
    identity is "skip the fold" (m/l/O unchanged), so an out-of-range key
    contributes nothing. The nested QK^T reduce (head_dim) stays static."""

    inner_n: MapAxis
    outer_m: MapAxis | None
    extra_outer: tuple[Loop, ...]
    target_names: frozenset[str]
    inner_body: tuple[Stmt, ...]
    leading: tuple[Stmt, ...]
    # Symbolic streaming-axis name → runtime boundary ``Expr`` (empty = all static).
    k_bounds: dict = field(default_factory=dict)


def _split_leading_non_loops(body: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    leading: list[Stmt] = []
    rest = tuple(body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]
    return tuple(leading), rest


def _map_axis(loop: Loop) -> MapAxis:
    ext = loop.axis.extent
    if ext.is_static:
        return MapAxis(loop=loop, symbolic=False, extent=ext.as_static())
    return MapAxis(loop=loop, symbolic=True, extent=ext.hint or 0)


@dataclass(frozen=True)
class MatmulSkeleton:
    """Plain scalar-matmul shape: free output axes ``M`` (outer) / ``N`` (inner)
    + one ``SEMIRING`` reduce axis ``K``, with the canonical body
    ``[Loop(k, [Load a, Load b, Assign(multiply), Accum]), Write(o[m,n])]``.

    Phase 2a covers the scalar tier with no split-K / cooperative-K / fused
    prologue and a static ``K`` extent; everything else returns ``None`` from
    :func:`lift_matmul` (legacy fallthrough)."""

    inner_n: MapAxis
    outer_m: MapAxis
    extra_outer: tuple[Loop, ...]
    k_loop: Loop
    k_name: str
    k_extent: int
    inner_body: tuple[Stmt, ...]
    leading: tuple[Stmt, ...]
    # Every K-extent contraction loop name to split — usually just ``{k_name}``,
    # but a multi-accumulator matmul (gated MLP / SwiGLU: gate + up matmuls
    # sharing K with a fused epilogue between them) has several same-K reduces.
    target_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CoopReduceSkeleton:
    """Plain associative reduce (`MONOID`) over a static K axis ≥ warp_size, with
    free output rows. Phase 3b covers the whole-CTA cooperative form: each CTA
    reduces one row's K across `BR` threads, then a warp/tree combine folds the
    partials. The canonical body is `[Loop(k, [Load, Accum]), Write(o[rows])]`."""

    inner_n: MapAxis
    outer_m: MapAxis | None
    extra_outer: tuple[Loop, ...]
    k_loop: Loop
    k_name: str
    k_extent: int  # tiling extent: static size, or the Dim hint for a symbolic K
    inner_body: tuple[Stmt, ...]
    leading: tuple[Stmt, ...]
    # Every K-extent loop name to cooperatively split — the reduce(s) PLUS any
    # second-pass map loop (RMSNorm normalize, softmax exp). Keyed by the K
    # ``Dim`` (so static and symbolic match uniformly), since the map loop
    # carries a different axis name than the reduce (only sibling *reduces* are
    # unified upstream). Defaults to ``{k_name}`` for the plain reduce.
    target_names: frozenset[str] = frozenset()
    # The symbolic K boundary ``Expr`` (runtime ``seq_len``) when K is symbolic,
    # else ``None``. A masked-K reduce tiles at the hint (``k_extent``) and fills
    # past ``k_bound`` with each carrier's identity (``_mask_reduce_accums``).
    k_bound: object = None
