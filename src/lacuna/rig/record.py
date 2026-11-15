"""Record one session from real models: two conversations over one kernel.

A session's queue is dealt to the planes in turn. Each plane is one conversation
with its own model. Tickets are handed out one at a time in queue order, each to
the plane it was dealt to, and that plane's agent works it to the end before the
next ticket goes out. So a follow-up finds what its opener left, whichever plane
did the opener, and the order of the session's calls is the order they are
replayed in (`lacuna.rig.sessions.timed`).

Calls go through `KernelService`, the code the rig's kernel runs, so the world a
model sees while it is recorded is the world its calls will meet in a run. What
is kept is every request the models made, in order: failed calls, lookups and
calls with arguments that would not parse included. Outcomes are kept beside
them for the pool's own description and are not replayed; a run's truth is that
run's ledger.

`complete` is any callable with `ChatClient.complete`'s shape, so this module
needs no HTTP client and the tests need no model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from lacuna.envs.base import Env
from lacuna.envs.service import KernelService
from lacuna.rig.sessions import Session, timed

# (messages, tools, tag) -> the assistant message
Complete = Callable[..., dict]

# Not the back-office prompt of the earlier studies: that one hands over a whole queue.
RECORD_SYSTEM = (
    "You are a back-office support agent. You share a ticket queue with another agent, so "
    "a record may have been changed since you last saw it. Tickets arrive one at a time. "
    "Resolve each ticket using the tools. Look things up before changing them when unsure. "
    "When the ticket is done, reply with one line and no tool call."
)


@dataclass
class Recorded:
    session: str
    seed: int
    n_tickets: int
    planes: tuple[str, ...]
    calls: list[tuple[str, int, str, dict]]  # (plane, ticket, tool, args), in order
    outcomes: list[bool]  # whether each call succeeded when recorded
    transcripts: dict[str, list[dict]]  # per plane, without the system prompt
    ledger: list[dict]  # the recording's own ledger, with principals
    rows: list[dict]  # what the end state and the sink showed after recording
    unfinished: list[int]  # tickets whose agent was still calling tools at the turn limit

    def sessions(self, pace_s: float) -> list[Session]:
        return timed(self.session, self.seed, self.n_tickets, self.calls, pace_s, self.planes)


def _args(raw: object) -> dict:
    """A tool call's arguments as the model wrote them, or {} if they are not an object."""
    try:
        args = json.loads(raw or "{}") if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def record_session(
    seed: int,
    n_tickets: int,
    env: Env,
    complete: dict[str, Complete],
    planes: tuple[str, ...] = ("a", "b"),
    max_turns: int = 12,
) -> Recorded:
    """`complete[plane]` answers for that plane's model. `max_turns` is per ticket."""
    models = {p: complete[p] for p in planes}
    keys = {p: f"record-{p}" for p in planes}
    service = KernelService(env, {keys[p]: f"plane-{p}" for p in planes})
    session = f"s{seed}"
    service.open(session, seed, n_tickets)
    episode = env.make_episode(seed, n_tickets)
    messages: dict[str, list[dict]] = {
        p: [{"role": "system", "content": RECORD_SYSTEM}] for p in planes
    }
    calls: list[tuple[str, int, str, dict]] = []
    outcomes: list[bool] = []
    unfinished: list[int] = []
    for i, ticket in enumerate(episode.tickets):
        plane = planes[i % len(planes)]
        msgs = messages[plane]
        msgs.append({"role": "user", "content": f"Ticket {i + 1}: {ticket.text}"})
        for turn in range(max_turns):
            reply = models[plane](msgs, env.tools, tag=f"rec:{seed}:{plane}:{i}:{turn}")
            msg = {"role": "assistant", "content": reply.get("content")}
            if "reasoning_content" in reply:
                # Passed back as returned, by key presence (`lacuna.agents.llm`, D34).
                msg["reasoning_content"] = reply["reasoning_content"]
            tool_calls = reply.get("tool_calls") or []
            if tool_calls:
                msg["tool_calls"] = tool_calls
            msgs.append(msg)
            if not tool_calls:
                break
            for tc in tool_calls:
                name = str(tc["function"]["name"])
                args = _args(tc["function"].get("arguments"))
                ok, text = service.execute(session, keys[plane], name, args)
                calls.append((plane, i, name, args))
                outcomes.append(ok)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
        else:
            unfinished.append(i)
    return Recorded(
        session=session,
        seed=seed,
        n_tickets=n_tickets,
        planes=planes,
        calls=calls,
        outcomes=outcomes,
        transcripts={p: messages[p][1:] for p in planes},
        ledger=service.ledger(session),
        rows=service.observe(session),
        unfinished=unfinished,
    )
