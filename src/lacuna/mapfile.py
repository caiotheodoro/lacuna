"""A map file: how to read each log as a witness, written down once, in TOML.

The estimator needs, for every action a log recorded, what the action was, which
plane it went through, and (where the log has one) an id it shares with the
other logs of that plane. A map file says where those are in each log. It holds
paths and constants only: no expression is evaluated and no code is named except
a registered environment's normaliser (`lacuna.envs.registry.ENVS`).

    version = 1
    planes = ["a", "b"]
    env = "backoffice"            # optional: turns (tool, arguments) into an action

    [[witness]]
    name = "spans"                # several blocks may share a name: one log per plane
    source = "spans"              # the --in name of the file; defaults to `name`
    format = "otlp-spans"         # otlp-spans | otlp-logs | jsonl
    pipeline = "sdk"              # what delivers it; `lint` counts distinct ones
    plane = { path = ["resource", "service.name"], map = { "agent-a" = "a" } }
    select = [{ path = ["attributes", "mcp.method.name"], equals = "tools/call" }]
    ok = [{ path = ["attributes", "is_error"], equals = false }]
    session = ["attributes", "session"]
    tool = ["attributes", "tool.name"]         # with `env`: tool and arguments
    args = ["attributes", "tool.arguments"]    # an object, or a JSON string of one
    id = [["trace_id"], ["attributes", "request.id"]]
    seq = ["attributes", "request.id"]

A path is a list of keys, because a key may hold dots. Without `env`, a block
gives `op`, `resource` and `value` paths and the map lists `actions` (with `env`, the
env's own actions are the default). A block with no `id` attaches by what the action
was (`by_id = false`): a state diff. `format = "otlp-logs"` with `body = "json"`
parses each record's body, so paths may start with `body`.

What a block does with its file, in order. A line that is not a JSON object is
not a record: it is counted as `unreadable_lines`. A span delivered twice is one
record: the copies are counted as `duplicates`. Every record left is counted under
exactly one of: `not_selected` (`select` and `has` say whether it is this
witness's at all), `plane_not_mapped`, `failed` (`ok`), `no_session`,
`bad_arguments`, `not_an_action`, or `kept`.

Two things a test compares strictly. Equality is by type: `true` is not 1, though
200 equals 200.0. And a session, an id part, an op, a resource or a value that is
not a string or a number is not usable, and the record is counted as lacking it.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lacuna.core.records import Obs, WitnessSpec
from lacuna.ingest import jsonl, otlp

FORMATS = ("otlp-spans", "otlp-logs", "jsonl")
# More witnesses on one plane than this and the estimator's model space is too
# large to search: 2 million models at seven.
MAX_WITNESSES = 5
Path_ = tuple[str, ...]
_MISSING = object()


class MapError(ValueError):
    """The map file, or what it was given to read, cannot be followed. The message says why."""


@dataclass(frozen=True)
class Check:
    path: Path_
    equals: Any


@dataclass(frozen=True)
class Block:
    name: str
    source: str
    format: str
    pipeline: str | None
    plane: str | None  # a constant plane, or
    plane_path: Path_ | None  # where to read it, through
    plane_map: dict[str, str]  # this table
    select: tuple[Check, ...]
    has: tuple[Path_, ...]
    ok: tuple[Check, ...]
    session: Path_ | None
    tool: Path_ | None
    args: Path_ | None
    op: Path_ | None
    resource: Path_ | None
    value: Path_ | None
    id: tuple[Path_, ...]
    seq: Path_ | None
    body_json: bool

    @property
    def by_id(self) -> bool:
        return bool(self.id)

    @property
    def sees(self) -> frozenset[str]:
        return frozenset([self.plane] if self.plane else self.plane_map.values())


@dataclass(frozen=True)
class Map:
    planes: tuple[str, ...]
    env: str | None
    actions: frozenset[str]
    blocks: tuple[Block, ...]

    def specs(self) -> tuple[WitnessSpec, ...]:
        """One spec per witness name, in the order the names first appear."""
        sees: dict[str, set[str]] = {}
        by_id: dict[str, bool] = {}
        for b in self.blocks:
            sees.setdefault(b.name, set()).update(b.sees)
            by_id[b.name] = b.by_id
        return tuple(WitnessSpec(n, frozenset(s), by_id[n]) for n, s in sees.items())

    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(b.source for b in self.blocks))

    def witnesses_of(self, plane: str) -> list[str]:
        return list(dict.fromkeys(b.name for b in self.blocks if plane in b.sees))


@dataclass
class Counts:
    """What happened to one block's file. `read` is the sum of the seven after it."""

    unreadable_lines: int = 0
    duplicates: int = 0
    read: int = 0
    not_selected: int = 0
    plane_not_mapped: int = 0
    failed: int = 0
    no_session: int = 0
    bad_arguments: int = 0
    not_an_action: int = 0
    kept: int = 0


# Parsing ----------------------------------------------------------------------------


def _path(v: object, where: str) -> Path_:
    if isinstance(v, str) and v:
        return (v,)
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return tuple(v)
    raise MapError(f"{where}: a path is a key or a list of keys, not {v!r}")


def _paths(v: object, where: str) -> tuple[Path_, ...]:
    if not isinstance(v, list):
        raise MapError(f"{where}: a list of paths, not {v!r}")
    return tuple(_path(x, f"{where}[{i}]") for i, x in enumerate(v))


def _checks(v: object, where: str) -> tuple[Check, ...]:
    if v is None:
        return ()
    if not isinstance(v, list):
        raise MapError(f"{where}: a list of {{ path, equals }} tables")
    out = []
    for i, t in enumerate(v):
        if not isinstance(t, dict) or set(t) != {"path", "equals"}:
            raise MapError(f"{where}[{i}]: needs exactly `path` and `equals`")
        out.append(Check(_path(t["path"], f"{where}[{i}].path"), t["equals"]))
    return tuple(out)


def _text(d: dict, key: str, where: str, default: str | None = None) -> str | None:
    v = d.get(key, default)
    if v is not None and (not isinstance(v, str) or not v):
        raise MapError(f"{where}.{key}: a non-empty string, not {v!r}")
    return v


_BLOCK_KEYS = {
    "name", "source", "format", "pipeline", "plane", "select", "has", "ok", "session", "tool",
    "args", "op", "resource", "value", "id", "seq", "body",
}  # fmt: skip


def _block(d: object, i: int, planes: tuple[str, ...], env: str | None) -> Block:
    where = f"witness[{i}]"
    if not isinstance(d, dict):
        raise MapError(f"{where}: a table, not {d!r}")
    name = _text(d, "name", where)
    if name is None:
        raise MapError(f"{where}: needs a name")
    where = f"witness[{i}] ({name})"
    extra = sorted(set(d) - _BLOCK_KEYS)
    if extra:
        raise MapError(f"{where}: unknown keys {extra}")
    fmt = d.get("format")
    if not isinstance(fmt, str) or fmt not in FORMATS:
        raise MapError(f"{where}: format must be one of {list(FORMATS)}, not {fmt!r}")
    body = d.get("body")
    if not (body is None or body == "json") or (body and fmt != "otlp-logs"):
        raise MapError(f'{where}: `body = "json"` is the only body option, for otlp-logs')

    plane, plane_path, plane_map = None, None, {}
    p = d.get("plane")
    if isinstance(p, str):
        plane = p
    elif isinstance(p, dict) and set(p) == {"path", "map"} and isinstance(p["map"], dict):
        plane_path = _path(p["path"], f"{where}.plane.path")
        plane_map = dict(p["map"])
        if not plane_map or not all(isinstance(v, str) for v in plane_map.values()):
            raise MapError(f"{where}.plane.map: a non-empty table of plane names")
    else:
        raise MapError(f"{where}: plane is a plane's name or {{ path, map }}")
    unknown = sorted(({plane} if plane else set(plane_map.values())) - set(planes))
    if unknown:
        raise MapError(f"{where}: planes {unknown} are not in `planes`")

    def opt(key: str) -> Path_ | None:
        return _path(d[key], f"{where}.{key}") if key in d else None

    tool, args = opt("tool"), opt("args")
    op, resource, value = opt("op"), opt("resource"), opt("value")
    if (tool is None) == (op is None):
        raise MapError(f"{where}: give `tool` and `args`, or `op`, `resource` and `value`")
    if tool is not None and (args is None or env is None):
        raise MapError(f"{where}: `tool` needs `args`, and the map needs `env` to read them")
    if op is not None and (resource is None or value is None):
        raise MapError(f"{where}: `op` needs `resource` and `value`")
    return Block(
        name=name,
        source=_text(d, "source", where, name),
        format=fmt,
        pipeline=_text(d, "pipeline", where),
        plane=plane,
        plane_path=plane_path,
        plane_map=plane_map,
        select=_checks(d.get("select"), f"{where}.select"),
        has=_paths(d.get("has", []), f"{where}.has"),
        ok=_checks(d.get("ok"), f"{where}.ok"),
        session=opt("session"),
        tool=tool,
        args=args,
        op=op,
        resource=resource,
        value=value,
        id=_paths(d.get("id", []), f"{where}.id"),
        seq=opt("seq"),
        body_json=body == "json",
    )


def parse(text: str) -> Map:
    try:
        d = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise MapError(f"not TOML: {e}") from None
    if type(d.get("version")) is not int or d["version"] != 1:
        raise MapError("`version = 1` is the only version this reads")
    extra = sorted(set(d) - {"version", "planes", "env", "actions", "witness"})
    if extra:
        raise MapError(f"unknown top-level keys {extra}")
    planes = d.get("planes")
    if (
        not isinstance(planes, list)
        or not planes
        or not all(isinstance(p, str) and p for p in planes)
    ):
        raise MapError("`planes` is a non-empty list of names")
    if len(set(planes)) != len(planes):
        raise MapError("a plane is named twice")
    env = d.get("env")
    actions = d.get("actions")
    if env is not None:
        from lacuna.envs.registry import ENVS

        if not isinstance(env, str) or env not in ENVS:
            raise MapError(f"env {env!r} is not one of {sorted(ENVS)}")
        if actions is None:
            # An env names its own actions; a block that reads ops directly uses them.
            actions = list(ENVS[env].action_tools)
    actions = [] if actions is None else actions
    if not isinstance(actions, list) or not all(isinstance(a, str) for a in actions):
        raise MapError("`actions` is a list of op names")
    raw = d.get("witness")
    if not isinstance(raw, list) or not raw:
        raise MapError("no [[witness]] block")
    blocks = tuple(_block(b, i, tuple(planes), env) for i, b in enumerate(raw))
    by_id: dict[str, bool] = {}
    for b in blocks:
        if by_id.setdefault(b.name, b.by_id) != b.by_id:
            raise MapError(f"witness {b.name}: some blocks have an id and some do not")
        if b.op is not None and not actions:
            raise MapError(f"witness {b.name}: `op` paths need the map's `actions` list")
    return Map(tuple(planes), env, frozenset(actions), blocks)


def load(path: Path) -> Map:
    try:
        return parse(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        raise MapError(f"{path} is not UTF-8 text") from None


# Reading ----------------------------------------------------------------------------


def _get(record: object, path: Path_) -> object:
    for key in path:
        if not isinstance(record, dict) or key not in record:
            return _MISSING
        record = record[key]
    return record


def _number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _scalar(v: object) -> bool:
    """A string or a number: something an action, a session or an id part can be."""
    return isinstance(v, str) or _number(v)


def _same(a: object, b: object) -> bool:
    """Equal, and not by Python's `True == 1`. A number equals an equal number."""
    if a is _MISSING:
        return False
    if _number(a) and _number(b):
        return a == b
    return type(a) is type(b) and a == b


def _records(block: Block, lines: list[str]) -> tuple[list[dict], int, int]:
    """(records, lines that are not JSON objects, records dropped as second copies)."""
    filled = [x for x in lines if x.strip()]
    if block.format == "jsonl":
        out = jsonl.read(filled)
        return out, len(filled) - len(out), 0
    requests = []
    for x in filled:
        try:
            d = json.loads(x)
        except ValueError:  # a JSONDecodeError, or a number too long to read
            continue
        if isinstance(d, dict):
            requests.append(x)
    bad = len(filled) - len(requests)
    if block.format == "otlp-spans":
        out = otlp.read_spans(requests)
        return out, bad, otlp.count_spans(requests) - len(out)
    out = otlp.read_logs(requests)
    if block.body_json:
        for rec in out:
            parsed = jsonl.read([rec["body"]]) if isinstance(rec["body"], str) else []
            rec["body"] = parsed[0] if parsed else None
    return out, bad, 0


def _action(m: Map, block: Block, rec: dict) -> tuple[str, str, str] | str:
    """(op, resource, value), or the name of the reason there is none."""
    if block.op is not None:
        got = [_get(rec, p) for p in (block.op, block.resource, block.value)]
        if not all(_scalar(g) for g in got) or got[0] not in m.actions:
            return "not_an_action"
        return str(got[0]), str(got[1]), str(got[2])
    from lacuna.envs.registry import ENVS

    env = ENVS[m.env]
    tool, args = _get(rec, block.tool), _get(rec, block.args)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    if not isinstance(tool, str) or tool not in env.action_tools:
        return "not_an_action"
    if not isinstance(args, dict):
        return "bad_arguments"
    try:
        return env.normalise(tool, args) or "bad_arguments"
    except (ValueError, TypeError, KeyError, ArithmeticError):
        return "bad_arguments"  # arguments the env's own normaliser cannot read


def read_block(m: Map, block: Block, lines: Iterable[str]) -> tuple[list[Obs], Counts]:
    out: list[Obs] = []
    c = Counts()
    try:
        recs, c.unreadable_lines, c.duplicates = _records(block, list(lines))
    except (ValueError, TypeError, AttributeError, KeyError, RecursionError) as e:
        raise MapError(
            f"source {block.source!r} is not {block.format}: {type(e).__name__}: {e}"
        ) from None
    for rec in recs:
        c.read += 1
        if not all(_same(_get(rec, t.path), t.equals) for t in block.select) or any(
            _get(rec, p) is _MISSING for p in block.has
        ):
            c.not_selected += 1
            continue
        plane = block.plane
        if plane is None:
            raw = _get(rec, block.plane_path)
            plane = block.plane_map.get(raw) if isinstance(raw, str) else None
        if plane is None:
            c.plane_not_mapped += 1
            continue
        if not all(_same(_get(rec, t.path), t.equals) for t in block.ok):
            c.failed += 1
            continue
        session = _get(rec, block.session) if block.session else None
        if block.session and (not _scalar(session) or session == ""):
            c.no_session += 1
            continue
        action = _action(m, block, rec)
        if isinstance(action, str):
            setattr(c, action, getattr(c, action) + 1)
            continue
        parts = [_get(rec, p) for p in block.id]
        link: Hashable | None = None
        if parts and all(_scalar(p) and p != "" for p in parts):
            link = tuple(parts)
        seq = _get(rec, block.seq) if block.seq else None
        out.append(
            Obs(
                block.name,
                *action,
                stratum=plane,
                link_id=link,
                seq=seq if _scalar(seq) else None,
                session=session,
            )  # fmt: skip
        )
        c.kept += 1
    return out, c


def key(o: Obs) -> Hashable:
    """What one action was, inside its session."""
    return (o.session, o.op, o.resource, o.value)


def _order(o: Obs) -> tuple:
    """Session, plane, then the call's number: whole numbers in order, anything else
    after them by its text. The rule of `lacuna.rig.witnesses`, kept the same."""
    if isinstance(o.seq, int):
        rank: tuple = (0, o.seq, "")
    else:
        rank = (1, 0, "" if o.seq is None else str(o.seq))
    return (str(o.session), str(o.stratum), *rank)


def check_planes(m: Map) -> None:
    """Refuse a map no table can be built from: the two errors `lint` is about and one
    the estimator cannot afford."""
    for plane in m.planes:
        n = len(m.witnesses_of(plane))
        if n < 2:
            raise MapError(f"plane {plane}: {n} of the map's witnesses can see it; needs two")
        if n > MAX_WITNESSES:
            raise MapError(f"plane {plane}: {n} witnesses; more than {MAX_WITNESSES} is too many")


def records(m: Map, sources: dict[str, list[str]]) -> tuple[list[Obs], list[dict]]:
    """Every witness record the map finds in the sources, and what each block did.

    Records of id witnesses come first, in call order inside a session and plane, so
    that a record with no id attaches to the earliest action that fits; then the rest
    in file order.
    """
    check_planes(m)
    missing = sorted(set(m.sources()) - set(sources))
    if missing:
        raise MapError(f"no input for sources {missing}")
    linked: list[Obs] = []
    attached: list[Obs] = []
    report = []
    for i, block in enumerate(m.blocks):
        obs, c = read_block(m, block, sources[block.source])
        (linked if block.by_id else attached).extend(obs)
        report.append({"witness": block.name, "block": i, "source": block.source} | vars(c))
    linked.sort(key=_order)
    return linked + attached, report


# Lint -----------------------------------------------------------------------------


def lint(m: Map) -> tuple[list[str], list[str]]:
    """(errors, warnings) about the map as a plan for an audit, read without any log.

    It reads what the map says, not what is true of the deployment. An error is
    something no log can fix: a plane fewer than two witnesses can see, a plane
    whose witnesses all arrive by one pipeline (as the map names them; a witness with
    no `pipeline` counts as its own), a plane with more witnesses than the estimator
    can search, or a witness with no id whose rows two blocks would both read.
    """
    errors, warnings = [], []
    for plane in m.planes:
        seeing = [b for b in m.blocks if plane in b.sees]
        names = m.witnesses_of(plane)
        pipelines = {b.pipeline or f"(witness {b.name})" for b in seeing}
        if len(names) < 2:
            errors.append(
                f"plane {plane}: {len(names)} of the map's witnesses can see it ({names}); "
                "needs two"
            )
        elif len(pipelines) < 2:
            shared = sorted(pipelines)[0]
            errors.append(
                f"plane {plane}: its witnesses {names} all arrive by one pipeline ({shared})"
            )
        if len(names) > MAX_WITNESSES:
            errors.append(
                f"plane {plane}: {len(names)} witnesses; more than {MAX_WITNESSES} is too many"
            )
        with_id = {b.name for b in seeing if b.by_id}
        if seeing and len(with_id) == len(names):
            warnings.append(
                f"plane {plane}: every witness is a log with an id; none is a record of the "
                "state, so what all logs lost together leaves no trace"
            )
        if len(with_id) == 1:
            warnings.append(f"plane {plane}: one witness has an id, so nothing is linked by it")
        by_source: dict[str, set[str]] = {}
        for b in seeing:
            by_source.setdefault(b.source, set()).add(b.name)
        for source, sharing in sorted(by_source.items()):
            if len(sharing) > 1:
                warnings.append(
                    f"plane {plane}: witnesses {sorted(sharing)} are read from one file "
                    f"({source}); what loses one of them is likely to lose the other"
                )
        twice = {
            (b.name, b.source)
            for i, b in enumerate(seeing)
            for other in seeing[i + 1 :]
            if not b.by_id and (b.name, b.source) == (other.name, other.source)
        }
        for name, source in sorted(twice):
            errors.append(
                f"plane {plane}: witness {name} has no id and two of its blocks read {source} "
                "for this plane, so a row both select is counted twice"
            )
    for i, b in enumerate(m.blocks):
        where = f"witness[{i}] ({b.name})"
        if b.by_id and not b.ok:
            warnings.append(f"{where}: no `ok` test, so a failed call counts as an action")
        if b.session is None:
            warnings.append(f"{where}: no `session`, so like actions of two sessions are one")
        if b.pipeline is None:
            warnings.append(f"{where}: no `pipeline` named; it is counted as independent")
    return errors, warnings
