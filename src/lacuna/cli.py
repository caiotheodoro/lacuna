"""`lacuna`: count what every log missed, from the logs and a map file.

    lacuna lint     --map m.toml
    lacuna table    --map m.toml --in spans=spans.jsonl --in env=env.jsonl
    lacuna estimate --map m.toml --in spans=spans.jsonl --in env=env.jsonl [--json]

`lint` reads the map alone and says whether an audit built on it can work: it exits 1
if some plane has fewer than two independent recorders. `table` prints, per plane, how
many actions each combination of witnesses recorded. `estimate` adds the estimate of
how many actions happened, with its interval, beside the floors: what the first
witness alone holds, and what all witnesses together hold.

An estimate is only as good as the map: two witnesses that lose records together,
or a state diff that hides the same actions the logs lost, give a number that is too
low and an interval that does not say so.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from lacuna import mapfile
from lacuna.core.tables import build
from lacuna.estimators.stratified import strat_bic


def _sources(pairs: list[str]) -> dict[str, list[str]]:
    out = {}
    for pair in pairs:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise mapfile.MapError(f"--in takes NAME=PATH, not {pair!r}")
        if name in out:
            raise mapfile.MapError(f"--in names {name!r} twice")
        try:
            out[name] = Path(path).read_text().splitlines()
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
    planes = {}
    for plane, (names, table) in built.items():
        planes[plane] = {
            "witnesses": list(names),
            "recorded_by": _cells(names, table),
            # The floors: every action some witness holds, and the first witness alone.
            "union_count": int(table.sum()),
            f"trust_{names[0]}": int(sum(n for h, n in enumerate(table) if h & 1)),
        }
    return {"planes": planes, "blocks": report, "_built": built}


def estimate(m: mapfile.Map, sources: dict[str, list[str]]) -> dict:
    out = tables(m, sources)
    built = out["_built"]
    per_plane = {p: t for p, (_, t) in built.items()}
    live = {p: tuple(range(len(names))) for p, (names, _) in built.items()}
    for plane, table in per_plane.items():
        e = strat_bic({plane: table}, live={plane: live[plane]})
        out["planes"][plane]["estimate"] = _estimate(e)
    total = strat_bic(per_plane, live=live)
    out["total"] = _estimate(total) | {
        "union_count": int(sum(t.sum() for t in per_plane.values())),
        "silent_planes": total.detail.get("silent", 0),
    }
    return out


def _estimate(e) -> dict:
    return {
        "actions": _num(e.point),
        "at_least": _num(e.lo),
        "at_most": _num(e.hi),
        "bounded": bool(e.bounded and math.isfinite(e.point)),
    }


def _print_tables(out: dict) -> None:
    for plane, p in out["planes"].items():
        print(f"plane {plane}: {p['union_count']} actions in some log")
        for who, n in p["recorded_by"].items():
            if n:
                print(f"  {n:6d}  {who}")
        if "estimate" in p:
            print(f"  {_line(p['estimate'])}")
    for b in out["blocks"]:
        dropped = {k: v for k, v in b.items() if k not in ("witness", "block", "source") and v}
        print(f"{b['witness']} (block {b['block']}, {b['source']}): {dropped}")


def _line(e: dict) -> str:
    if not e["bounded"]:
        return f"estimate: at least {e['at_least']:g}, no upper bound"
    return (
        f"estimate: {e['actions']:g} actions (95% interval {e['at_least']:g} to {e['at_most']:g})"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lacuna", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name, text in (
        ("lint", "check a map file: can an audit built on it work?"),
        ("table", "per plane, how many actions each combination of witnesses recorded"),
        ("estimate", "the tables, and how many actions happened"),
    ):
        s = sub.add_parser(name, help=text, description=text)
        s.add_argument("--map", type=Path, required=True, help="the map file (TOML)")
        s.add_argument("--json", action="store_true", help="print JSON")
        if name != "lint":
            s.add_argument("--in", dest="inputs", action="append", default=[], metavar="NAME=PATH",
                           help="a log file, under the source name the map uses")  # fmt: skip
    a = p.parse_args(argv)

    try:
        m = mapfile.load(a.map)
        if a.command == "lint":
            errors, warnings = mapfile.lint(m)
            if a.json:
                print(json.dumps({"errors": errors, "warnings": warnings}, indent=1))
            else:
                for line in errors:
                    print(f"error: {line}")
                for line in warnings:
                    print(f"warning: {line}")
                print(
                    "ok" if not errors else f"{len(errors)} planes cannot be audited with this map"
                )
            return 1 if errors else 0
        sources = _sources(a.inputs)
        out = tables(m, sources) if a.command == "table" else estimate(m, sources)
    except (mapfile.MapError, OSError) as e:
        print(f"lacuna: {e}", file=sys.stderr)
        return 2
    out.pop("_built")
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    _print_tables(out)
    if "total" in out:
        t = out["total"]
        print(f"all planes: {t['union_count']} actions in some log; {_line(t)}")
        if t["silent_planes"]:
            print(f"{t['silent_planes']} planes have a witness that recorded nothing")
        if not t["bounded"]:
            print(
                "no upper bound: in some plane the witnesses recorded exactly the same "
                "actions, or too few differ, so nothing says what all of them missed"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
