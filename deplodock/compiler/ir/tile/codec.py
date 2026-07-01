"""Generic codec engine — the single ser/de source of truth for the tile-schedule knob codecs.

Every tunable schedule codec (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC``) is one string of
``/``-separated *nodes*. A node is ``TOKEN`` optionally followed by a value and, after a ``:``, a
comma-separated list of params (themselves nodes — the recursive case used by ``WSPEC`` roles).
This module owns the grammar (:func:`desugar` + :func:`decode` + :func:`encode`); each IR codec
dataclass (``ReducePlan`` / ``TilePlan`` / ``Stage`` / ``WarpSpec``) declares a
:class:`Schema` of :class:`Field`\\s and routes its ``parse`` / ``spell`` through here, keeping only
its own semantics (``combine`` / ``tile_m`` / ``block_threads`` / ``is_async`` / validation).

Grammar, after the :func:`desugar` rewrite ``TOKEN<v>`` → ``TOKEN:<v>`` and ``TOKEN<v>:rest`` →
``TOKEN:<v>,rest`` (the glued value is the field's own positional value)::

    codec := node ("/" node)*
    node  := TOKEN (":" param ("," param)*)?
    param := tuple | node
    tuple := int ("x" int)*

Delimiters: ``x`` = tuple element, ``,`` = sibling param within a level, ``:`` = descend, ``/`` =
top-level field separator. Params bind **by token, order-free**; the single bare-tuple param binds
to the field's own value. ``INT`` and ``PAIR`` collapse into one ``TUPLE`` (arity 1..n); a ``FLAG``
is a node with no value; a ``CHOICE`` is a bare token whose identity is the value (the ``STAGE``
transport); a ``NAME`` resolves an external registry (the warp atom). The one non-uniform value
codec is the ``REDUCE`` ``g<n>[a|k]`` finalize letter (:class:`Field.suffix`) — a trailing enum
glued to an int, kept inside the value rather than spelled as a ``,``-param so the wire format
stays byte-identical.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any

from deplodock.compiler.ir.atom import AtomKind, atom_for

_TOKEN_RE = re.compile(r"^([A-Za-z]+)(.*)$")


def _codec_width(num: str, *, tok: str, codec: str) -> int:
    """Parse a codec field's positive-integer width, rejecting empty / non-numeric / ``< 1``
    values with a clear message. A ``1`` width is the legal identity (the level is off); only
    ``0`` / negatives / non-digits are rejected. The ``codec`` name rides the message so a bad
    pin names the knob it came from (``bad REDUCE token ...`` / ``bad WARP token ...``)."""
    if not num.isdigit() or int(num) < 1:
        raise ValueError(f"bad {codec} token {tok!r}: expected a positive integer width, got {num!r}")
    return int(num)


class FieldKind(enum.Enum):
    """The five leaf shapes a codec field can take."""

    TUPLE = "tuple"  # int(xint)* — arity-1 scalar or arity-n pair (n/f/w/b/r/k/d/p)
    NAME = "name"  # a registry name resolved to an object (the warp atom, a:<atom>)
    CHOICE = "choice"  # a bare token that IS the value (the STAGE transport sync|cp|tma)
    FLAG = "flag"  # a valueless bare token → bool (STAGE ring)
    GROUP = "group"  # a value plus named sub-fields (the WSPEC roles, p<np>[:<param>,...])


class Emit(enum.Enum):
    """When :func:`encode` renders a field (the byte-identity lever — mirrors the old ``spell``)."""

    ALWAYS = "always"  # render unconditionally (STAGE d/transport, warp a/w/f)
    NONDEFAULT = "nondefault"  # render iff the value differs from the field default (g/b/r/k/n/f/p)
    TRUE = "true"  # render iff truthy (the ring flag)


@dataclass(frozen=True)
class Field:
    """One node slot in a codec :class:`Schema`."""

    token: str
    kind: FieldKind
    arity: int = 1  # TUPLE: number of int elements (1 = scalar int value, ≥2 = tuple value)
    emit: Emit = Emit.NONDEFAULT
    suppress_trailing: bool = True  # TUPLE arity≥2: drop trailing ``x<m>`` when it equals 1
    required: bool = False  # decode raises if the node is absent (the warp atom)
    suffix: tuple[tuple[str, str], ...] = ()  # TUPLE arity-1 enum letter: ((codec, canonical), ...)
    suffix_default: str = ""  # the canonical suffix when the letter is omitted
    choices: tuple[tuple[str, str], ...] = ()  # CHOICE: ((codec-token, canonical), ...)
    default: Any = None  # CHOICE: the canonical default; others derive via :func:`_default`
    params: tuple[Field, ...] = ()  # GROUP sub-fields

    @property
    def is_bare(self) -> bool:
        """True for the no-leading-letter-token kinds matched whole (CHOICE / FLAG)."""
        return self.kind in (FieldKind.CHOICE, FieldKind.FLAG)


@dataclass(frozen=True)
class Schema:
    """A codec's field list + its name (rides parse errors) + the ``expect`` hint."""

    name: str
    fields: tuple[Field, ...]
    expect: str = ""

    def by_token(self) -> dict[str, Field]:
        return {f.token: f for f in self.fields}


def _default(field: Field) -> Any:
    """The value of an absent field (decode fills these so the IR glue reads every token)."""
    if field.kind is FieldKind.TUPLE:
        if field.suffix:
            return (1, field.suffix_default)
        return 1 if field.arity == 1 else tuple([1] * field.arity)
    if field.kind is FieldKind.FLAG:
        return False
    if field.kind is FieldKind.CHOICE:
        return field.default
    if field.kind is FieldKind.NAME:
        return None
    return None  # GROUP absent → None


def desugar(node: str) -> str:
    """Rewrite a glued node into canonical ``TOKEN:value[,params]`` form.

    ``w2x2`` → ``w:2x2``; ``g2a`` → ``g:2a``; ``p2:q8`` → ``p:2,q8``. A node with no glued value
    (``cp`` / ``ring`` / ``a:mma...`` — already colon-form) is returned unchanged."""
    m = _TOKEN_RE.match(node)
    if not m:
        return node
    token, rest = m.group(1), m.group(2)
    if rest and rest[0].isdigit():
        value, _, tail = rest.partition(":")
        return f"{token}:{value}" + (f",{tail}" if tail else "")
    return node


def _split_top(s: str) -> list[str]:
    return [t.strip() for t in s.split("/") if t.strip()]


def _decode_tuple(field: Field, body: str, *, tok: str, codec: str) -> Any:
    """Decode an ``int(xint)*`` value (peeling a suffix letter first when the field has one)."""
    if field.suffix:
        suf = dict(field.suffix)
        tag = field.suffix_default
        if body and body[-1] in suf:
            tag = suf[body[-1]]
            body = body[:-1]
        return (_codec_width(body, tok=tok, codec=codec), tag)
    parts = body.split("x") if body else [""]
    if len(parts) > field.arity:
        raise ValueError(f"bad {codec} token {tok!r}: expected ≤{field.arity} dims, got {len(parts)}")
    nums = [_codec_width(p, tok=tok, codec=codec) for p in parts]
    nums += [1] * (field.arity - len(nums))
    return nums[0] if field.arity == 1 else tuple(nums)


def _decode_value(field: Field, body: str, *, tok: str, schema: Schema) -> Any:
    if field.kind is FieldKind.TUPLE:
        return _decode_tuple(field, body, tok=tok, codec=schema.name)
    if field.kind is FieldKind.NAME:
        return atom_for(body)  # raises ValueError on an unknown name
    if field.kind is FieldKind.GROUP:
        return _decode_group(field, body, tok=tok, schema=schema)
    raise ValueError(f"bad {schema.name} token {tok!r} ({schema.expect})")


def _decode_group(field: Field, body: str, *, tok: str, schema: Schema) -> dict[str, Any]:
    """Decode a GROUP body ``2,q8`` into ``{"": <own value>, <sub-token>: <sub value>, ...}``,
    binding the single bare-tuple positional to the field's own value and named params by token."""
    sub_by_token = {p.token: p for p in field.params}
    out: dict[str, Any] = {"": 1 if field.arity == 1 else tuple([1] * field.arity)}
    out.update({p.token: _default(p) for p in field.params})
    for raw in body.split(",") if body else []:
        part = raw.strip()
        if not part:
            continue
        if part[0].isdigit():  # the positional own-value tuple
            out[""] = _decode_tuple(field, part, tok=tok, codec=schema.name)
            continue
        sub = desugar(part)
        sub_tok, _, sub_body = sub.partition(":")
        pf = sub_by_token.get(sub_tok)
        if pf is None:
            raise ValueError(f"bad {schema.name} role param {part!r} on {tok!r} ({schema.expect})")
        out[sub_tok] = _decode_value(pf, sub_body, tok=part, schema=schema)
    return out


def _match_bare(schema: Schema, node: str) -> Field | None:
    """Match a whole bare node against a FLAG token or a CHOICE codec-token."""
    for f in schema.fields:
        if f.kind is FieldKind.FLAG and node == f.token:
            return f
        if f.kind is FieldKind.CHOICE and node in dict(f.choices):
            return f
    return None


def decode(schema: Schema, spec: str | None) -> dict[str, Any]:
    """Decode a codec string into ``{token: value}`` (defaults filled for absent fields).

    Raises ``ValueError`` — and only ``ValueError`` — on any malformed input (the featurizer
    degrades on ``ValueError``); the message names ``schema.name`` so a bad pin names its knob."""
    result: dict[str, Any] = {f.token: _default(f) for f in schema.fields}
    present: set[str] = set()
    by_token = schema.by_token()
    for node in _split_top((spec or "").strip()):
        bare = _match_bare(schema, node)
        if bare is not None:
            result[bare.token] = True if bare.kind is FieldKind.FLAG else dict(bare.choices)[node]
            present.add(bare.token)
            continue
        token, _, body = desugar(node).partition(":")
        field = by_token.get(token)
        if field is None or field.is_bare:
            raise ValueError(f"bad {schema.name} token {node!r} ({schema.expect})")
        result[field.token] = _decode_value(field, body, tok=node, schema=schema)
        present.add(field.token)
    for f in schema.fields:
        if f.required and f.token not in present:
            raise ValueError(f"{schema.name} codec {spec!r} names no {f.token} ({schema.expect})")
    return result


def _is_default(field: Field, value: Any) -> bool:
    if field.kind is FieldKind.TUPLE:
        if field.suffix:
            return value[0] == 1  # the width drives presence; the suffix rides an emitted token
        return value == 1 if field.arity == 1 else all(v == 1 for v in value)
    if field.kind is FieldKind.CHOICE:
        return value == field.default
    return not value


def _render_tuple(field: Field, value: Any) -> str:
    if field.suffix:
        width, tag = value
        canonical_to_codec = {canonical: codec for codec, canonical in field.suffix}
        return f"{width}{canonical_to_codec[tag]}"
    if field.arity == 1:
        return str(value)
    nums = list(value)
    if field.suppress_trailing:
        while len(nums) > 1 and nums[-1] == 1:
            nums.pop()
    return "x".join(str(n) for n in nums)


def _encode_field(field: Field, value: Any) -> str | None:
    if field.emit is Emit.TRUE:
        return field.token if value else None
    if field.emit is Emit.NONDEFAULT and _is_default(field, value):
        return None
    if field.kind is FieldKind.FLAG:
        return field.token if value else None
    if field.kind is FieldKind.CHOICE:
        return dict((t, c) for c, t in field.choices)[value]
    if field.kind is FieldKind.NAME:
        return f"{field.token}:{value.name}"
    if field.kind is FieldKind.GROUP:
        return _encode_group(field, value)
    return f"{field.token}{_render_tuple(field, value)}"


def _encode_group(field: Field, value: dict[str, Any]) -> str:
    """Render ``p<own>`` (glued, no params) or ``p<own>:<param>,...`` — the positional value is
    glued to the token; emitted named params follow after a ``:`` (matching the ``WSPEC`` spelling)."""
    own = _render_tuple(field, value.get("", 1 if field.arity == 1 else tuple([1] * field.arity)))
    params = [frag for p in field.params if (frag := _encode_field(p, value.get(p.token, _default(p)))) is not None]
    return f"{field.token}{own}" + (":" + ",".join(params) if params else "")


def encode(schema: Schema, values: dict[str, Any]) -> str:
    """Encode a ``{token: value}`` map into the canonical codec string (fields in schema order)."""
    toks: list[str] = []
    for field in schema.fields:
        frag = _encode_field(field, values.get(field.token, _default(field)))
        if frag is not None:
            toks.append(frag)
    return "/".join(toks)


def field_default(field: Field) -> Any:
    """The value of an absent ``field`` — the public handle on the engine's default (used to drop
    default-valued params when building a structured codec object, so they don't spell back)."""
    return _default(field)


__all__ = [
    "AtomKind",
    "Emit",
    "Field",
    "FieldKind",
    "Schema",
    "_codec_width",
    "decode",
    "desugar",
    "encode",
    "field_default",
]
