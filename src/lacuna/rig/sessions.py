"""What a replayed session is, and scripted ones for runs that need no model.

A session is one kernel world worked on from several planes at once: the
orchestrator takes some tickets through the gateway, the sub-agent takes the
others straight to the MCP server. One `Session` object is one plane's share:
the tool calls it makes, in order, with the pause before each.

The file holds requests, not outcomes. Planes interleave differently on every
run, so a call that succeeded when recorded can fail when replayed; the truth of
a run is that run's ledger.

`scripted_sessions` deals the tickets of a scripted plan to the planes in turn.
It is for smoke runs and tests. Study sessions are recorded from model runs and
written in the same format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from lacuna.agents.scripted import ScriptedAgent
from lacuna.envs.base import Env
from lacuna.envs.tasks import Episode


@dataclass(frozen=True)
class Step:
    ticket: int
    tool: str
    args: dict
    gap_s: float  # pause before the call


@dataclass(frozen=True)
class Session:
    session: str
    seed: int
    n_tickets: int
    plane: str
    steps: tuple[Step, ...]


def _read_for(args: dict) -> tuple[str, dict] | None:
    """The lookup an agent makes before acting on an order or a customer."""
    if "order_id" in args:
        return "get_order", {"order_id": args["order_id"]}
    if "customer_id" in args:
        return "get_customer", {"customer_id": args["customer_id"]}
    return None


def scripted_sessions(
    seed: int,
    n_tickets: int,
    planes: tuple[str, ...],
    pace_s: float = 0.0,
    reads: bool = True,
    env: Env | None = None,
) -> list[Session]:
    """One `Session` per plane for episode `seed`; ticket `i` goes to plane `i % len(planes)`.

    The actions are exactly `ScriptedAgent.plan` under the generator `[seed, 1]`.
    Gaps are `pace_s` times a uniform draw on [0.5, 1.5], from `[seed, 2]`.
    """
    from lacuna.envs.tasks import make_episode

    ep = (env.make_episode if env else make_episode)(seed, n_tickets)
    agent = ScriptedAgent(np.random.default_rng([seed, 1]))
    pause = np.random.default_rng([seed, 2])
    steps: dict[str, list[Step]] = {p: [] for p in planes}
    for i, ticket in enumerate(ep.tickets):
        calls = agent.plan(Episode(ep.seed, ep.world, [ticket]))
        read = _read_for(calls[0][1]) if reads and calls else None
        for tool, args in ([read] if read else []) + calls:
            gap = float(round(pace_s * pause.uniform(0.5, 1.5), 3))
            steps[planes[i % len(planes)]].append(Step(i, tool, args, gap))
    return [Session(f"s{seed}", seed, n_tickets, p, tuple(steps[p])) for p in planes]


def dump(sessions: list[Session]) -> str:
    return json.dumps([asdict(s) for s in sessions], indent=1) + "\n"


def load(text: str) -> list[Session]:
    return [
        Session(
            d["session"],
            int(d["seed"]),
            int(d["n_tickets"]),
            d["plane"],
            tuple(
                Step(int(s["ticket"]), s["tool"], s["args"], float(s["gap_s"])) for s in d["steps"]
            ),
        )
        for d in json.loads(text)
    ]
