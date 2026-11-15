"""From a rig run's files to `Obs` records, per-plane tables and a run summary.

Three witnesses per plane:

- `spans`: the originator's own spans, as the backend collector received them;
- `access`: the receiver's access-log lines, as the backend collector received
  them (the gateway's on plane A, the MCP front's on plane B);
- `env`: the end state and the sink, attributed to a plane by last writer.

`spans` and `access` share a link id per call, (trace id, JSON-RPC id), both
stamped by the client library (AGENTS.md rule 4, `lacuna.rig.replay`). `env`
has no id and attaches by signature inside a session.

A record is an action if its tool is one of the env's action tools and the call
did not fail. What "failed" means is read from each log's own field.

`summarise` also reads two things no witness reads: the ledger, to say how many
actions there were, and the receivers' complete access-log files, to say how
much each pipeline delivered. Neither reaches a table. Nothing here estimates.
"""

from __future__ import annotations

import datetime
import json
import math
from collections import Counter
from collections.abc import Hashable
from pathlib import Path

from lacuna.core.records import Obs, WitnessSpec
from lacuna.core.tables import build
from lacuna.envs.base import Env
from lacuna.envs.registry import BACKOFFICE
from lacuna.ingest import jsonl, otlp
from lacuna.linkage import signature

SPANS, ACCESS, ENV = "spans", "access", "env"
PLANES = frozenset({"a", "b"})
SPECS = (
    WitnessSpec(SPANS, PLANES),
    WitnessSpec(ACCESS, PLANES),
    WitnessSpec(ENV, PLANES, by_id=False),
)
# Which originator uses which plane, and which access-log file is which plane's.
# This is the deployer's knowledge of the topology, not something a record claims.
SERVICE_PLANE = {"agent-a": "a", "agent-b": "b"}
LOG_PLANE = {"gateway.jsonl": "a", "front-b.jsonl": "b"}
# The service whose stdout each access-log file is (`scripts/rig_run.py` follows these).
LOG_SERVICE = {"gateway.jsonl": "agentgateway", "front-b.jsonl": "mcp-b"}


def key(o: Obs) -> Hashable:
    """One action's signature inside its session."""
    return (o.session, *signature(o.op, o.resource, o.value))


def _obs(witness: str, env: Env, plane: str, call: dict) -> Obs | None:
    """`call` is one tool call read from a log: tool, args, ok, session, trace_id, rpc_id.

    A record with no session is not made into an action: its signature could be any
    session's, and it would count twice. `load` counts what is set aside this way.
    """
    if not call["ok"] or call["tool"] not in env.action_tools or not call["session"]:
        return None
    if not isinstance(call["args"], dict):
        return None
    norm = env.normalise(call["tool"], call["args"])
    if norm is None:
        return None
    link = None
    if call["trace_id"] and call["rpc_id"] is not None:
        link = (call["trace_id"], call["rpc_id"])
    return Obs(witness, *norm, stratum=plane, link_id=link, ts=call.get("ts"),
               seq=call["rpc_id"], session=call["session"])  # fmt: skip


def span_calls(spans: list[dict]) -> list[tuple[str, dict]]:
    """(plane, call) for every tool-call span of a known originator."""
    out = []
    for sp in spans:
        a = sp["attributes"]
        plane = SERVICE_PLANE.get(sp["resource"].get("service.name"))
        if plane is None or a.get("mcp.method.name") != "tools/call":
            continue
        try:
            args = json.loads(a.get("gen_ai.tool.call.arguments") or "null")
        except json.JSONDecodeError:
            args = None
        out.append(
            (
                plane,
                {
                    "tool": a.get("gen_ai.tool.name"),
                    "args": args,
                    # A span that ended without the flag is a call whose reply never came.
                    "ok": a.get("lacuna.tool.is_error") is False,
                    "session": a.get("lacuna.session"),
                    "trace_id": sp["trace_id"],
                    "rpc_id": a.get("jsonrpc.request.id"),
                    "ts": sp["start_ns"] / 1e9,
                },
            )
        )
    return out


def _unix(stamp: object) -> float | None:
    """An RFC 3339 time as the gateway writes it, in seconds, or None."""
    try:
        return datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def gateway_calls(lines: list[dict]) -> list[dict]:
    """Tool calls in agentgateway's access log, with the rig's added fields."""
    return [
        {
            "tool": d.get("gen_ai.tool.name"),
            "args": d.get("tool_args"),
            "ok": d.get("http.status") == 200 and d.get("tool_is_error") is False,
            "session": d.get("session"),
            "trace_id": d.get("trace.id"),
            "rpc_id": d.get("rpc_id"),
            "ts": _unix(d["time"]) if "time" in d else None,
        }
        for d in lines
        if d.get("scope") == "request" and d.get("mcp.method.name") == "tools/call"
    ]


def front_calls(lines: list[dict]) -> list[dict]:
    """Tool calls in the MCP front's access log (`lacuna.rig.front`)."""
    return [
        {k: d.get(k) for k in ("tool", "args", "session", "trace_id", "rpc_id", "ts")}
        | {"ok": d.get("ok") is True}
        for d in lines
        if "tool" in d and "rpc_id" in d
    ]


def access_calls(file_name: str, lines: list[dict]) -> list[dict]:
    return gateway_calls(lines) if file_name == "gateway.jsonl" else front_calls(lines)


def env_obs(rows: list[dict]) -> list[Obs]:
    return [
        Obs(
            ENV,
            r["op"],
            r["resource"],
            r["value"],
            stratum=(r["principal"] or "").removeprefix("plane-") or None,
            session=r["session"],
        )  # fmt: skip
        for r in rows
    ]


def _links(calls: list[dict]) -> set[tuple]:
    return {
        (c["trace_id"], c["rpc_id"]) for c in calls if c["trace_id"] and c["rpc_id"] is not None
    }


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def load(run_dir: Path, env: Env = BACKOFFICE) -> dict:
    """Everything `summarise` works from: witness records, and the two references."""
    spans = span_calls(otlp.read_spans(_lines(run_dir / "witnesses" / "spans.jsonl")))
    delivered: dict[str, list[dict]] = {name: [] for name in LOG_PLANE}
    for rec in otlp.read_logs(_lines(run_dir / "witnesses" / "logs.jsonl")):
        name = rec["attributes"].get("log.file.name")
        if name in delivered and isinstance(rec["body"], str):
            delivered[name].extend(jsonl.read([rec["body"]]))
    access = {name: access_calls(name, lines) for name, lines in delivered.items()}
    written = {
        name: access_calls(name, jsonl.read(_lines(run_dir / "logs" / name))) for name in LOG_PLANE
    }
    tagged = [(SPANS, plane, c) for plane, c in spans]
    for name, calls in access.items():
        tagged += [(ACCESS, LOG_PLANE[name], c) for c in calls]
    obs = [o for w, plane, c in tagged if (o := _obs(w, env, plane, c))]
    # Call order inside a session and plane, whichever file a record came from. The
    # table attaches an env row to the earliest action that fits, so file order would
    # hand every tie to the witness read first.
    obs.sort(key=lambda o: (str(o.session), o.stratum, o.seq if o.seq is not None else math.inf))
    obs += env_obs(jsonl.read(_lines(run_dir / "witnesses" / "env.jsonl")))
    no_session = Counter(
        plane
        for _, plane, c in tagged
        if c["ok"] and c["tool"] in env.action_tools and not c["session"]
    )
    return {
        "obs": obs,
        "records_without_session": no_session,
        "span_calls": spans,
        "access_calls": access,
        "written_calls": written,
        "ledger": jsonl.read(_lines(run_dir / "truth" / "ledger.jsonl")),
    }


def _recall(records: list[Obs], truth: Counter) -> tuple[int, int]:
    """(actions of `truth` this witness has a record for, records that match no action)."""
    seen: Counter = Counter()
    linked: set = set()
    for o in records:
        if o.link_id is not None:
            if o.link_id in linked:
                continue  # one action logged twice is one action
            linked.add(o.link_id)
        seen[key(o)] += 1
    hit = sum(min(n, truth[k]) for k, n in seen.items())
    return hit, sum(seen.values()) - hit


def summarise(run_dir: Path, env: Env = BACKOFFICE) -> dict:
    d = load(run_dir, env)
    tables = build(d["obs"], SPECS, key)
    planes = {}
    for plane, (names, table) in tables.items():
        truth = Counter(
            (e["session"], *signature(e["op"], e["resource"], e["value"]))
            for e in d["ledger"]
            if e["principal"] == f"plane-{plane}"
        )
        n = sum(truth.values())
        file_name = next(f for f, p in LOG_PLANE.items() if p == plane)
        written = _links(d["written_calls"][file_name])
        delivered = _links(d["access_calls"][file_name])
        span_links = _links([c for p, c in d["span_calls"] if p == plane])
        witnesses = {}
        for w in names:
            mine = [o for o in d["obs"] if o.witness == w and o.stratum == plane]
            hit, phantom = _recall(mine, truth)
            witnesses[w] = {
                "records": len(mine),
                "actions_seen": hit,
                "matching_no_action": phantom,
                "missed_share": round(1 - hit / n, 4) if n else None,
            }
        # link id -> key, per id witness: do two logs that share an id tell one story?
        keys = {
            w: {o.link_id: key(o) for o in d["obs"]
                if o.witness == w and o.stratum == plane and o.link_id is not None}
            for w in (SPANS, ACCESS)
        }  # fmt: skip
        pairs = keys[SPANS].keys() & keys[ACCESS].keys()
        same = sum(1 for i in pairs if keys[SPANS][i] == keys[ACCESS][i])
        either = keys[SPANS].keys() | keys[ACCESS].keys()
        spans_bit = 1 << names.index(SPANS)
        planes[plane] = {
            "actions_in_ledger": n,
            "witness_names": list(names),
            "table": [int(x) for x in table],
            # The floors (AGENTS.md rule 5), from the table alone.
            "union_count": int(table.sum()),
            "union_bound": {"at_least": int(table.sum()), "at_most": None},
            "trust_trace": int(sum(n for i, n in enumerate(table) if i & spans_bit)),
            # Scored against the ledger; no table reads these.
            "witnesses": witnesses,
            "records_without_session": d["records_without_session"][plane],
            # Of the calls the receiver wrote down, how many each pipeline delivered.
            "calls_written_by_receiver": len(written),
            "access_lines_delivered": len(delivered & written),
            "spans_delivered": len(span_links & written),
            # Two different things. `agreement`: of the actions both id witnesses
            # recorded under one id, the share where they name the same action; a
            # linking fault lowers it. `coverage`: of the ids either has, the share both
            # have; loss in either pipeline lowers it.
            "link": {
                "pairs": len(pairs),
                "pairs_same_key": same,
                "agreement": round(same / len(pairs), 4) if pairs else None,
                "coverage": round(len(pairs) / len(either), 4) if either else None,
            },
        }
    total = sum(p["actions_in_ledger"] for p in planes.values())
    union = sum(p["union_count"] for p in planes.values())
    return {
        "planes": planes,
        "headline": {
            "actions_in_ledger": total,
            "union_count": union,
            "union_missed_share": round(1 - union / total, 4) if total else None,
            **{
                f"plane_{p}": {
                    "actions": v["actions_in_ledger"],
                    "spans_missed": v["witnesses"][SPANS]["missed_share"],
                    "access_missed": v["witnesses"][ACCESS]["missed_share"],
                    "env_missed": v["witnesses"][ENV]["missed_share"],
                    "link_agreement": v["link"]["agreement"],
                    "link_coverage": v["link"]["coverage"],
                }
                for p, v in planes.items()
            },
        },
    }
