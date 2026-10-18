"""Tool surface of the infra/ops env, its call normaliser, and its `Env` bundle.

The router, gateway (W2) and trace (W1) are the back-office ones
(`lacuna.envs.tools.Router`), given this module's handlers through `INFRA`.
"""

from __future__ import annotations

from collections.abc import Callable

from lacuna.envs.base import Env, Observed
from lacuna.envs.infra import ACTION_OPS, CONFIG_KEYS, SEVERITIES, InfraKernel
from lacuna.envs.infra_tasks import make_episode
from lacuna.envs.kernel import digest
from lacuna.envs.tools import NUM, S, _fn

TOOLS: list[dict] = [
    _fn("list_services", "List every service with its team, replicas and live version.", {}, []),
    _fn(
        "get_service",
        "Look up one service: team, replicas, live and previous version, config.",
        {"service": S},
        ["service"],
    ),
    _fn(
        "get_metrics",
        "Read a service's live metrics: error_rate, p99_ms, cpu_pct.",
        {"service": S},
        ["service"],
    ),
    _fn(
        "scale_service",
        "Set a service's replica count.",
        {"service": S, "replicas": NUM},
        ["service", "replicas"],
    ),
    _fn(
        "deploy_version",
        "Deploy a version of a service.",
        {"service": S, "version": S},
        ["service", "version"],
    ),
    _fn("rollback", "Roll a service back to its previous version.", {"service": S}, ["service"]),
    _fn(
        "update_config",
        "Set one config key of a service: " + ", ".join(CONFIG_KEYS) + ".",
        {"service": S, "key": {"type": "string", "enum": list(CONFIG_KEYS)}, "value": S},
        ["service", "key", "value"],
    ),
    _fn(
        "create_incident",
        "Open an incident against a service.",
        {"service": S, "severity": {"type": "string", "enum": list(SEVERITIES)}, "summary": S},
        ["service", "severity"],
    ),
    _fn(
        "send_page",
        "Page a team's on-call rotation.",
        {"team": S, "message": S},
        ["team", "message"],
    ),
    _fn(
        "write_runbook",
        "Create or overwrite a runbook.",
        {"name": S, "content": S},
        ["name", "content"],
    ),
]
ACTION_TOOLS = ACTION_OPS
# Content digests (page text, runbook bodies) stay out of signatures, as in
# the back-office env; every other value is in the call and in the diff.
VALUE_OPS = ("scale_service", "deploy_version", "rollback", "update_config", "create_incident")


def handlers(k: InfraKernel) -> dict[str, Callable[[dict], object]]:
    return {
        "list_services": lambda a: k.list_services(),
        "get_service": lambda a: k.get_service(a["service"]),
        "get_metrics": lambda a: k.get_metrics(a["service"]),
        "scale_service": lambda a: k.scale_service(a["service"], a["replicas"]),
        "deploy_version": lambda a: k.deploy_version(a["service"], a["version"]),
        "rollback": lambda a: k.rollback(a["service"]),
        "update_config": lambda a: k.update_config(a["service"], a["key"], a["value"]),
        "create_incident": lambda a: k.create_incident(
            a["service"], a["severity"], a.get("summary", "")
        ),
        "send_page": lambda a: k.send_page(a["team"], a["message"]),
        "write_runbook": lambda a: k.write_runbook(a["name"], a["content"]),
    }


def from_call(tool: str, args: dict) -> tuple[str, str, str] | None:
    """Normalise a tool call into (op, resource, value), the same way the kernel ledgers it."""
    try:
        if tool == "scale_service":
            return tool, str(args["service"]), str(int(args["replicas"]))
        if tool == "deploy_version":
            return tool, str(args["service"]), str(args["version"])
        if tool == "rollback":
            return tool, str(args["service"]), "rollback"
        if tool == "update_config":
            return tool, f"{args['service']}.{args['key']}", str(args["value"])
        if tool == "create_incident":
            return tool, str(args["service"]), str(args["severity"])
        if tool == "send_page":
            return tool, str(args["team"]), digest(args["message"])
        if tool == "write_runbook":
            return tool, str(args["name"]), digest(args["content"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def observe(k: InfraKernel) -> list[Observed]:
    """W3: the end-of-shift diff plus every page the network sink caught."""
    rows: list[Observed] = [
        (str(d["op"]), str(d["resource"]), str(d["value"]), d.get("_lid")) for d in k.diff()
    ]
    rows += [("send_page", str(s["team"]), digest(s["message"]), s["_lid"]) for s in k.sink]
    return rows


AGENT_SYSTEM = (
    "You are an on-call infrastructure operator. Resolve every ticket in the queue using "
    "the tools. Look things up before changing them when unsure. When every ticket is "
    "done, reply with a one-line summary and no tool call."
)

INFRA = Env(
    name="infra",
    tools=TOOLS,
    action_tools=ACTION_TOOLS,
    value_ops=VALUE_OPS,
    handlers=handlers,
    normalise=from_call,
    observe=observe,
    make_episode=make_episode,
    make_kernel=InfraKernel,
    agent_system=AGENT_SYSTEM,
)
