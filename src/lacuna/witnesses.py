"""Turn each witness's raw material into action records.

A `Record` is what one witness claims happened: an op, the resource it
touched, and a value. `oracle` is the ledger id when the record came from an
infrastructure path that can know it (the router's private map, the
kernel's diff attribution). It is used only by oracle linkage and is never
shown to an estimator.

Witness order matches `lacuna.histories.WITNESSES`: 0 trace, 1 gateway
("proxy"), 2 env, 3 auditor.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from lacuna.envs.base import Env, Observed
from lacuna.envs.kernel import TRACE_PATH, Kernel, digest
from lacuna.envs.tools import ACTION_TOOLS

TRACE, GATEWAY, ENV, AUDITOR = 0, 1, 2, 3


@dataclass(frozen=True)
class Record:
    witness: int
    op: str
    resource: str
    value: str
    oracle: int | None = None


def from_call(tool: str, args: dict) -> tuple[str, str, str] | None:
    """Normalise a tool call into (op, resource, value), the same way the kernel ledgers it."""
    try:
        if tool == "issue_refund":
            return tool, str(args["order_id"]), f"{round(float(args['amount']), 2):.2f}"
        if tool == "update_customer":
            return tool, f"{args['customer_id']}.{args['field']}", str(args["value"])
        if tool == "cancel_order":
            return tool, str(args["order_id"]), "cancelled"
        if tool == "write_file":
            return tool, str(args["path"]), digest(args["content"])
        if tool == "delete_file":
            return tool, str(args["path"]), "deleted"
        if tool == "send_email":
            return tool, str(args["to"]), digest(args["subject"], args["body"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _calls(env: Env | None) -> tuple[tuple[str, ...], Callable]:
    """The env's action tools and call normaliser; None is the back-office env."""
    return (env.action_tools, env.normalise) if env else (ACTION_TOOLS, from_call)


def _from_lines(
    lines: list[dict], witness: int, call_ledger: dict[str, int], env: Env | None = None
) -> list[Record]:
    actions, normalise = _calls(env)
    out = []
    for line in lines:
        if not line.get("ok") or line.get("tool") not in actions:
            continue
        norm = normalise(line["tool"], line.get("args", {}))
        if norm:
            out.append(Record(witness, *norm, oracle=call_ledger.get(line.get("call_id", ""))))
    return out


def trace_records(
    kernel: Kernel, call_ledger: dict[str, int], env: Env | None = None
) -> list[Record]:
    lines = []
    for raw in kernel.world.files.get(TRACE_PATH, "").splitlines():
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return _from_lines(lines, TRACE, call_ledger, env)


def gateway_records(
    gateway: list[dict], call_ledger: dict[str, int], env: Env | None = None
) -> list[Record]:
    return _from_lines(gateway, GATEWAY, call_ledger, env)


def observe(kernel: Kernel) -> list[Observed]:
    """Back-office W3: the end-state diff plus outbound email."""
    rows: list[Observed] = [
        (str(d["op"]), str(d["resource"]), str(d["value"]), d.get("_lid")) for d in kernel.diff()
    ]
    for s in kernel.sink:
        if s["kind"] == "email":
            rows.append(("send_email", str(s["to"]), digest(s["subject"], s["body"]), s["_lid"]))
    return rows


def env_records(kernel: Kernel, env: Env | None = None) -> list[Record]:
    return [Record(ENV, *row) for row in (env.observe if env else observe)(kernel)]


def auditor_records(actions: list[dict], env: Env | None = None) -> list[Record]:
    """W4 records from an auditor's reconstruction. No oracle ids: it only read a transcript."""
    action_tools, _ = _calls(env)
    out = []
    for a in actions:
        op = str(a.get("op", ""))
        if op not in action_tools:
            continue
        value = str(a.get("value", ""))
        if op == "issue_refund":
            try:
                value = f"{round(float(value), 2):.2f}"
            except ValueError:
                pass
        out.append(Record(AUDITOR, op, str(a.get("resource", "")), value))
    return out
