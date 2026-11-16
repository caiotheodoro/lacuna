"""`lacuna`: count what every log missed, from the logs and a map file.

    lacuna lint     --map m.toml [--json]
    lacuna table    --map m.toml --in spans=spans.jsonl --in env=env.jsonl [--json]
    lacuna estimate --map m.toml --in spans=spans.jsonl --in env=env.jsonl [--json]

`lint` reads the map alone. It checks what the map says: that every plane has two
witnesses, arriving by two pipelines as the map names them. It cannot check that the
names are true, that the planes are every route an action can take, or that the
witnesses lose records independently. It exits 1 on an error.

`table` prints, per plane, how many actions each combination of witnesses recorded,
with the floors: what all witnesses hold together (`union_count`, which is also the
lower bound `union_bound`) and what the map's first witness holds (`trust`).

`estimate` adds an estimate of how many actions happened, per plane and in total, and
names the estimator it used.

Which estimator. Where every plane has two logs linked by an id and one witness with no
`id`, and every block gives a session, it is `ll_sa_hom` (`lacuna.estimators.sa_hom`),
the estimator study 5 registered. The tool reads the witness with no id as a state
diff and cannot check that it is one. The planes' tables are summed. The point is
Lincoln-Petersen between two lists, "either log" and the state diff; the interval is
that model's, widened by how much the tables of single sessions disagree. Two things
stop it. Link ids used by more than one session of a plane are refused (exit 2): the
plane's table and its sessions' tables would not add up. And a plane with actions and
none in both a log and the state diff leaves the total with no upper bound, because
the sum would lend that plane the state diff's view of the others. For any other
layout the estimator is `strat_bic`, as before, and the output says that it is not the
one study 5 tested.

A point outside its own interval (the point is a real number, the interval's ends are
whole) is reported with that end moved to the point, and marked
(`interval_extended_to_point`).

What an estimate assumes, and it cannot check any of it: that the witnesses miss
actions independently of each other, and that every record is a real action. Where
two witnesses lose the same records, or a state diff hides the same actions the logs
lost, the estimate is too low. Where records are not real actions, or one witness
sees only a part of the plane, it can be too high. The interval is the estimator's
own (its 2.5% to 97.5% points); it was not calibrated on your logs.

`ll_sa_hom` lets its two logs lose the same records. For it the first assumption is
about the state diff against the logs: it takes the state diff to show the actions the
logs lost as often as it shows the rest. Where the state diff shows no action the logs
missed, its estimate is the union, whatever was lost. It sums the planes, so planes
that differ in what they lose or in what the state diff hides bias it, and nothing here
checks for that.

What was tested, and no more than this (`docs/PREREG-STUDY5.md`; `docs/DECISIONS.md`
D92 and D93). `ll_sa_hom`, on one rig on one machine: two planes, two relays that both
planes share, both killed together once in a run and started again 15 to 45 s later,
synthetic tickets, sessions recorded once and replayed. The registered criterion on
coverage was met on the runs left in the test: 277 of the 300 planned, the integrity
check having set 23 aside (`artifacts/robustness/study5-verdicts.json`). With those 23
counted as not covered, the same rule reads the figure as "not shown" (the same file's
`P1_not_in_test_as_uncovered`, D93); the registration gives that figure no criterion.
The bounds behind both take runs to be independent, and runs share sessions. An
interval with no upper end would have counted as covering; no stressed run had one.
The study scored the total over the two planes, never a plane's own line. On one
unstressed run the registered code gave no estimate; `lacuna.estimators.sa_hom` gives
one there from a closed form added after the study, and an estimate made that way is
marked (`closed_form`). No criterion was registered for another layout, another kind
of loss, or an agent that forges ids.

Exit status: 0 done, 1 `lint` found an error, 2 the map or an input cannot be read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from lacuna import mapfile
from lacuna.core.records import Obs
from lacuna.core.tables import build
from lacuna.estimators import Estimate
from lacuna.estimators.sa_hom import NAME as SA_HOM
from lacuna.estimators.sa_hom import sa_hom
from lacuna.estimators.stratified import strat_bic
from lacuna.histories import keep_witnesses

ASSUMES = (
    "This assumes the witnesses miss actions independently and every record is a real "
    "action.\nWitnesses that lose the same records make it too low; it can also be too "
    "high.\nThe interval is the estimator's own and was not calibrated on these logs."
)
# Which estimator made the numbers, whether study 5 tested it, and what the tool took
# from the map without being able to check it.
TESTED = (
    f"Estimator: {SA_HOM}, the one study 5 tested, on a map with two id-linked logs and "
    "one witness without an id per plane.\nThe tool reads that witness as a state diff "
    "and cannot check that it is one.\nStudy 5 was one rig and scored the total over "
    "planes, not a plane's own line."
)
NOT_TESTED = (
    "Estimator: strat_bic, which is not the one study 5 tested: that one needs two "
    "id-linked logs and one witness without an id on every plane, each with a session."
)
CLOSED_FORM = (
    "the fit study 5 registered did not converge on this table; this is the same model "
    "in closed form, added after the study"
)
HELP = {
    "lint": "check what a map says: two witnesses a plane, by two pipelines as it names them",
    "table": "per plane, how many actions each combination of witnesses recorded",
    "estimate": "the tables, and an estimate of how many actions happened. It names the "
    "estimator and says whether it is the one study 5 tested. " + ASSUMES,
}


def _sources(pairs: list[str]) -> dict[str, list[str]]:
    out = {}
    for pair in pairs:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise mapfile.MapError(f"--in takes NAME=PATH, not {pair!r}")
        if name in out:
            raise mapfile.MapError(f"--in names {name!r} twice")
        try:
            # Split on newlines only: a log line may hold other separators in a string.
            out[name] = Path(path).read_text(encoding="utf-8").split("\n")
        except UnicodeDecodeError:
            raise mapfile.MapError(f"{path} is not UTF-8 text") from None
        except OSError as e:
            raise mapfile.MapError(f"cannot read {path}: {e.strerror}") from None
    return out


def _cells(names: tuple[str, ...], table: np.ndarray) -> dict[str, int]:
    """A table as {"spans+env": 12, ...}: who recorded an action -> how many actions."""
    out = {}
    for h in range(1, len(table)):
        who = "+".join(n for i, n in enumerate(names) if h >> i & 1)
        out[who] = int(table[h])
    return out


def _num(v: float) -> float | None:
    return round(v, 1) if math.isfinite(v) else None


def tables(m: mapfile.Map, sources: dict[str, list[str]]) -> dict:
    obs, report = mapfile.records(m, sources)
    built = build(obs, m.specs(), mapfile.key)
    first = m.blocks[0].name
    planes = {}
    for plane, (names, table) in built.items():
        union = int(table.sum())
        seen = {
            n: int(sum(c for h, c in enumerate(table) if h >> i & 1)) for i, n in enumerate(names)
        }
        planes[plane] = {
            "witnesses": list(names),
            "recorded_by": _cells(names, table),
            # The floors. The union is a count and a lower bound; `trust` is every
            # action the map's first witness recorded, where that witness sees the plane.
            "union_count": union,
            "union_bound": {"at_least": union, "at_most": None},
            "trust": {"witness": first, "count": seen.get(first)},
            "recorded_nothing": [n for n in names if not seen[n]],
        }
    return {"planes": planes, "blocks": report, "_built": built, "_obs": obs}


def _estimate(e: Estimate) -> dict:
    """An estimate as the tool reports it.

    An estimator's point is a real number and its interval search is over whole numbers,
    so the interval can stop short of the point (46 to 46 around 46.1). An interval that
    leaves out its own point is not reported: the end is moved to the point and marked.
    """
    lo, hi = e.lo, e.hi
    extended = math.isfinite(e.point) and not lo <= e.point <= hi
    if extended:
        lo, hi = min(lo, e.point), max(hi, e.point)
    out = {
        "actions": _num(e.point),
        "interval_low": _num(lo),
        "interval_high": _num(hi),
        "bounded": bool(e.bounded and math.isfinite(e.point)),
    }
    if extended:
        out["interval_extended_to_point"] = True
    if e.detail.get("closed_form"):
        out["closed_form"] = True  # see CLOSED_FORM
    return out


def _why_unbounded(union: int, detail: dict) -> str:
    if union == 0:
        return "no witness recorded an action"
    if detail.get("unidentified"):
        return "its witnesses recorded the same actions, or only one recorded any"
    return "too few actions were recorded by more than one witness to say how many all missed"


def _why_no_ceiling(table: np.ndarray) -> str:
    """`_why_unbounded` for `ll_sa_hom`, whose table has the state diff at bit 2."""
    if table.sum() == 0:
        return "no witness recorded an action"
    if table[5:].sum() == 0:
        return "no action is in both a log and the state diff"
    return "too few actions are in both a log and the state diff to say how many all missed"


Tables = dict[str, tuple[tuple[str, ...], np.ndarray]]


def study5_tables(
    m: mapfile.Map, built: Tables, obs: list[Obs]
) -> tuple[dict[str, np.ndarray], list[tuple[str, np.ndarray]]] | None:
    """What study 5 handed `ll_sa_hom`, or None where the map's layout is not the study's.

    The layout: every plane has exactly two witnesses with an id and one without, and
    every block gives a session, because the interval is scaled by how much sessions
    disagree. The witness without an id is read as a state diff; a map cannot say that
    it is one. Returned: each plane's table with the two id witnesses at bits 0 and 1
    and the state diff at bit 2, and the same table for every session and plane any
    witness recorded on, as (plane, table), by session then plane. This is
    `plane_tables` and `session_tables` of `scripts/study5_lib.py`.

    A plane's table links records by id over the whole plane and a session's table inside
    the session. They add up only when no id is used by two sessions; where they do not,
    the estimator would be handed a count and a dispersion of two different tables, and
    this raises `mapfile.MapError`.
    """
    if any(b.session is None for b in m.blocks):
        return None
    by_id = {s.name: s.by_id for s in m.specs()}
    order = {}
    for plane, (names, _) in built.items():
        logs = [i for i, n in enumerate(names) if by_id[n]]
        state = [i for i, n in enumerate(names) if not by_id[n]]
        if (len(logs), len(state)) != (2, 1):
            return None
        order[plane] = (*logs, *state)
    per_plane = {p: keep_witnesses(t, order[p]) for p, (_, t) in built.items()}
    by_session: dict[str, list[Obs]] = {}
    for o in obs:
        by_session.setdefault(str(o.session), []).append(o)
    rows = []
    for session in sorted(by_session):
        for plane, (_, table) in build(by_session[session], m.specs(), mapfile.key).items():
            if table.sum():
                rows.append((plane, keep_witnesses(table, order[plane])))
    for plane, table in per_plane.items():
        in_sessions = sum((t for p, t in rows if p == plane), np.zeros(8, dtype=np.int64))
        if not np.array_equal(in_sessions, table):
            raise mapfile.MapError(
                f"plane {plane}: link ids repeat across sessions, so its sessions' tables do "
                "not add up to its table; an id must be unique across the plane (the map's "
                "`id` takes several paths: add the session or a trace id to it)"
            )
    return per_plane, rows


def estimate(m: mapfile.Map, sources: dict[str, list[str]]) -> dict:
    out = tables(m, sources)
    built = out["_built"]
    study5 = study5_tables(m, built, out["_obs"])
    if study5 is None:
        per_plane = {p: t for p, (_, t) in built.items()}
        live = {p: tuple(range(len(names))) for p, (names, _) in built.items()}
        for plane, table in per_plane.items():
            e = strat_bic({plane: table}, live={plane: live[plane]})
            block = out["planes"][plane]
            block["estimate"] = _estimate(e)
            if not block["estimate"]["bounded"]:
                block["estimate"]["why_unbounded"] = _why_unbounded(block["union_count"], e.detail)
        total = _estimate(strat_bic(per_plane, live=live))
    else:
        per_plane, rows = study5
        for plane, table in per_plane.items():
            block = out["planes"][plane]
            block["estimate"] = _estimate(sa_hom(table, [t for p, t in rows if p == plane]))
            if not block["estimate"]["bounded"]:
                block["estimate"]["why_unbounded"] = _why_no_ceiling(table)
        # As `scripts/study5_lib.estimates` has it: the planes' tables summed.
        pooled = np.sum(list(per_plane.values()), axis=0).astype(np.int64)
        pooled[0] = 0
        # Summing lends every plane the state diff's view of the others. A plane with
        # actions and none in both a log and the state diff has no such view of its own,
        # so the sum has no ceiling either. No run of study 5 had such a plane.
        dark = [p for p, t in per_plane.items() if t.sum() and not t[5:].sum()]
        if dark:
            n = int(pooled.sum())
            total = _estimate(Estimate(SA_HOM, n, math.inf, float(n), math.inf))
            where = f"plane {dark[0]}" if len(dark) == 1 else f"planes {', '.join(dark)}"
            total["why_unbounded"] = f"on {where} no action is in both a log and the state diff"
        else:
            total = _estimate(sa_hom(pooled, [t for _, t in rows]))
            if not total["bounded"]:
                total["why_unbounded"] = _why_no_ceiling(pooled)
    union = int(sum(t.sum() for t in per_plane.values()))
    out["total"] = total | {
        "union_count": union,
        "union_bound": {"at_least": union, "at_most": None},
    }
    out["estimator"] = {
        "name": "strat_bic" if study5 is None else SA_HOM,
        "study5_tested": study5 is not None,
        "says": NOT_TESTED if study5 is None else TESTED,
    }
    out["assumes"] = ASSUMES
    return out


def _line(e: dict, union: int) -> str:
    if not e["bounded"]:
        return f"estimate: at least {union} (what the logs hold), no upper bound"
    return (
        f"estimate: {e['actions']:g} actions "
        f"(interval {e['interval_low']:g} to {e['interval_high']:g})"
    )


def _print(out: dict) -> None:
    for plane, p in out["planes"].items():
        trust = p["trust"]
        floor = f"union_count {p['union_count']} (union_bound: at least {p['union_count']})"
        if trust["count"] is not None:
            floor += f", trust {trust['witness']} {trust['count']}"
        print(f"plane {plane}: {floor}")
        for who, n in p["recorded_by"].items():
            if n:
                print(f"  {n:6d}  {who}")
        for name in p["recorded_nothing"]:
            print(f"  {name} recorded nothing on this plane")
        if "estimate" in p:
            print(f"  {_line(p['estimate'], p['union_count'])}")
            if "why_unbounded" in p["estimate"]:
                print(f"  no upper bound: {p['estimate']['why_unbounded']}")
            if p["estimate"].get("closed_form"):
                print(f"  {CLOSED_FORM}")
    for b in out["blocks"]:
        did = {k: v for k, v in b.items() if k not in ("witness", "block", "source") and v}
        print(f"{b['witness']} (block {b['block']}, {b['source']}): {did or 'no record read'}")
        if not b["read"]:
            print("  warning: this block read no record; is the file in the format the map says?")
        if b["unreadable_lines"]:
            print(f"  warning: {b['unreadable_lines']} lines are not JSON objects (a cut line?)")
    if "total" in out:
        t = out["total"]
        print(f"all planes: union_count {t['union_count']}; {_line(t, t['union_count'])}")
        if "why_unbounded" in t:
            print(f"  no upper bound: {t['why_unbounded']}")
        if t.get("closed_form"):
            print(f"  {CLOSED_FORM}")
        print(out["estimator"]["says"])
        print(ASSUMES)


def _lint(m: mapfile.Map, as_json: bool) -> int:
    errors, warnings = mapfile.lint(m)
    if as_json:
        print(json.dumps({"errors": errors, "warnings": warnings}, indent=1))
        return 1 if errors else 0
    for line in errors:
        print(f"error: {line}")
    for line in warnings:
        print(f"warning: {line}")
    if errors:
        print(f"errors: {len(errors)}. No count can be made with this map as it is.")
    elif warnings:
        print(f"no error, warnings: {len(warnings)}")
    else:
        print("ok: two witnesses on every plane, by two pipelines as the map names them")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lacuna", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, text in HELP.items():
        s = sub.add_parser(name, help=text.splitlines()[0], description=text)
        s.add_argument("--map", type=Path, required=True, help="the map file (TOML)")
        s.add_argument("--json", action="store_true", help="print JSON")
        if name != "lint":
            s.add_argument("--in", dest="inputs", action="append", default=[], metavar="NAME=PATH",
                           help="a log file, under the source name the map uses")  # fmt: skip
    a = p.parse_args(argv)

    try:
        m = mapfile.load(a.map)
        if a.command == "lint":
            return _lint(m, a.json)
        sources = _sources(a.inputs)
        out = tables(m, sources) if a.command == "table" else estimate(m, sources)
    except (mapfile.MapError, OSError) as e:
        print(f"lacuna: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # a log this tool cannot digest must not look like a result
        print(f"lacuna: cannot work with these inputs: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for private in ("_built", "_obs"):
        out.pop(private)
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        _print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
