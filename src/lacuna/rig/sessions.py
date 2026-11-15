"""What a replayed session is, and scripted ones for runs that need no model.

A session is one kernel world worked on from several planes at once: the
orchestrator takes some tickets through the gateway, the sub-agent takes the
others straight to the MCP server. One `Session` object is one plane's share:
the tool calls it makes, in order, with the pause before each.

The file holds requests, not outcomes. Planes interleave differently on every
run, so a call that succeeded when recorded can fail when replayed; the truth of
a run is that run's ledger.

A step's gap is the time since the plane's previous step was due, or since the
session's start for its first. `timed` writes the gaps of every plane from one
clock, so that replayed to a schedule the planes make their calls in the order
the session had: a follow-up handled on one plane finds what its opener left on
the other.

`scripted_sessions` deals the tickets of a scripted plan to the planes in turn.
It is for smoke runs and tests. Study sessions are recorded from model runs
(`lacuna.rig.record`) and written in the same format.
"""

from __future__ import annotations

import json
import random
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
    timeline: bool = False,
) -> list[Session]:
    """One `Session` per plane for episode `seed`; ticket `i` goes to plane `i % len(planes)`.

    The actions are exactly `ScriptedAgent.plan` under the generator `[seed, 1]`.
    Gaps are `pace_s` times a uniform draw on [0.5, 1.5], from `[seed, 2]`. By default
    each plane paces itself and the planes run side by side; with `timeline` the
    session has one clock and the tickets are worked in queue order (`timed`).
    """
    from lacuna.envs.tasks import make_episode

    ep = (env.make_episode if env else make_episode)(seed, n_tickets)
    agent = ScriptedAgent(np.random.default_rng([seed, 1]))
    pause = np.random.default_rng([seed, 2])
    steps: dict[str, list[Step]] = {p: [] for p in planes}
    ordered: list[tuple[str, int, str, dict]] = []
    for i, ticket in enumerate(ep.tickets):
        calls = agent.plan(Episode(ep.seed, ep.world, [ticket]))
        read = _read_for(calls[0][1]) if reads and calls else None
        for tool, args in ([read] if read else []) + calls:
            if timeline:
                ordered.append((planes[i % len(planes)], i, tool, args))
                continue
            gap = float(round(pace_s * pause.uniform(0.5, 1.5), 3))
            steps[planes[i % len(planes)]].append(Step(i, tool, args, gap))
    if timeline:
        return timed(f"s{seed}", seed, n_tickets, ordered, pace_s, planes)
    return [Session(f"s{seed}", seed, n_tickets, p, tuple(steps[p])) for p in planes]


def timed(
    session: str,
    seed: int,
    n_tickets: int,
    calls: list[tuple[str, int, str, dict]],
    pace_s: float,
    planes: tuple[str, ...],
) -> list[Session]:
    """`calls` are (plane, ticket, tool, args) in the order the session made them.

    Consecutive calls are `pace_s` times a uniform draw on [0.5, 1.5] apart, from
    `[seed, 2]`, on one clock for the whole session. Each plane's gaps are read off
    that clock.
    """
    pause = np.random.default_rng([seed, 2])
    clock = 0.0
    last = {p: 0.0 for p in planes}
    steps: dict[str, list[Step]] = {p: [] for p in planes}
    for plane, ticket, tool, args in calls:
        clock = round(clock + pace_s * float(pause.uniform(0.5, 1.5)), 3)
        steps[plane].append(Step(int(ticket), tool, args, round(clock - last[plane], 3)))
        last[plane] = clock
    return [Session(session, seed, n_tickets, p, tuple(steps[p])) for p in planes]


def offsets(session: Session) -> list[float]:
    """When each step is due, in seconds after the session's start."""
    out: list[float] = []
    t = 0.0
    for st in session.steps:
        t = round(t + st.gap_s, 3)
        out.append(t)
    return out


def draw(pool: list[Session], n: int, seed: int) -> list[Session]:
    """`n` sessions of a pool, every plane of each, chosen by `seed` without replacement.

    A pool with `n` sessions or fewer is returned whole. The order is the pool's.
    """
    names = sorted({s.session for s in pool})
    if n < len(names):
        names = random.Random(f"draw/{seed}").sample(names, n)
    keep = set(names)
    return [s for s in pool if s.session in keep]


def session_delay(session: str, stagger_s: float) -> float:
    """How long after the run's start a session starts. The same for all its planes."""
    return random.Random(f"start/{session}").uniform(0.0, stagger_s)


def planned_end(sessions: list[Session], stagger_s: float) -> float:
    """When the last call of a run is due, in seconds after its start, if none is late."""
    return max(
        (session_delay(s.session, stagger_s) + offsets(s)[-1] for s in sessions if s.steps),
        default=0.0,
    )


def clashes(sessions: list[Session]) -> list[str]:
    """What stops a list of sessions from being one run: empty if nothing does.

    A session is named once per plane, and one name means one seed and one queue.
    Two pools recorded from the same seed break both, and replayed together they
    would act on one kernel world as if they were one session.
    """
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    world: dict[str, tuple[int, int]] = {}
    for s in sessions:
        if (s.session, s.plane) in seen:
            out.append(f"session {s.session} has two shares for plane {s.plane}")
        seen.add((s.session, s.plane))
        if world.setdefault(s.session, (s.seed, s.n_tickets)) != (s.seed, s.n_tickets):
            out.append(f"session {s.session} is named with two seeds or two queue lengths")
    return sorted(set(out))


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
