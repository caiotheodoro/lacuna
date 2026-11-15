"""Recording a session: two conversations over one kernel, ticket by ticket."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lacuna.envs.kernel import Kernel
from lacuna.envs.registry import BACKOFFICE_CHURN as ENV
from lacuna.envs.tools import Router
from lacuna.rig.record import RECORD_SYSTEM, record_session
from lacuna.rig.sessions import offsets


class PlanFollower:
    """Stands in for a model: for each ticket it is handed, it makes the ticket's planned
    calls one per turn, then answers in words. It finds the ticket by its text."""

    def __init__(self, ep, extra: dict[int, list[dict]] | None = None):
        self.ep = ep
        self.extra = extra or {}  # ticket index -> raw tool calls to make first
        self.seen: list[tuple[list[dict], list[dict] | None, str]] = []
        self.n = 0

    def complete(
        self, messages: list[dict], tools: list[dict] | None = None, tag: str = ""
    ) -> dict:
        self.seen.append(([dict(m) for m in messages], tools, tag))
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        text = messages[last_user]["content"]
        index = next(i for i, t in enumerate(self.ep.tickets) if text.endswith(t.text))
        done = sum(1 for m in messages[last_user:] if m["role"] == "tool")
        raw = self.extra.get(index, [])
        if done < len(raw):
            return {"content": None, "tool_calls": [raw[done]]}
        plan = self.ep.tickets[index].plan
        k = done - len(raw)
        if k >= len(plan):
            return {"content": f"Ticket {index + 1} done."}
        self.n += 1
        tool, args = plan[k]
        call = {"id": f"t{self.n}", "type": "function"}
        fn = {"name": tool, "arguments": json.dumps(args)}
        return {"content": None, "tool_calls": [call | {"function": fn}]}


def _raw(i: int, name: str, arguments: str) -> dict:
    return {"id": f"x{i}", "type": "function", "function": {"name": name, "arguments": arguments}}


def _record(seed: int = 4, n: int = 10, **kw):
    ep = ENV.make_episode(seed, n)
    a, b = PlanFollower(ep, kw.pop("extra_a", None)), PlanFollower(ep, kw.pop("extra_b", None))
    rec = record_session(seed, n, ENV, {"a": a.complete, "b": b.complete}, **kw)
    return ep, a, b, rec


def test_the_calls_are_every_tickets_plan_in_queue_order_on_the_plane_it_was_dealt_to():
    ep, _, _, rec = _record()
    want = [
        (("a", "b")[i % 2], i, tool, args)
        for i, t in enumerate(ep.tickets)
        for tool, args in t.plan
    ]
    assert rec.calls == want
    assert rec.outcomes == [True] * len(want)
    assert rec.session == "s4" and rec.seed == 4 and rec.n_tickets == 10
    assert rec.unfinished == []


def test_both_conversations_act_on_one_world_and_the_ledger_names_who_did_what():
    ep, _, _, rec = _record()
    # The same plans through one in-process router give the same ledger.
    kernel = Kernel(ENV.make_episode(4, 10).world)
    router = Router(kernel, np.random.default_rng(0), 0.0, 0.0)
    for i, (tool, args) in enumerate(c for t in ep.tickets for c in t.plan):
        router.execute(f"c{i}", tool, args)
    got = [(e["op"], e["resource"], e["value"]) for e in rec.ledger]
    assert got == [(e.op, e.resource, e.value) for e in kernel.ledger]
    by = {"plane-a": 0, "plane-b": 0}
    for e in rec.ledger:
        by[e["principal"]] += 1
    assert by["plane-a"] > 0 and by["plane-b"] > 0
    actions = [(p, tool) for p, _, tool, _ in rec.calls if tool in ENV.action_tools]
    assert [f"plane-{p}" for p, _ in actions] == [e["principal"] for e in rec.ledger]
    assert len(rec.rows) < len(rec.ledger)  # the end state hides some of it


def test_each_plane_sees_only_its_own_tickets_one_at_a_time_and_its_own_history():
    ep, a, b, rec = _record()
    for plane, follower in (("a", a), ("b", b)):
        for messages, tools, tag in follower.seen:
            assert messages[0] == {"role": "system", "content": RECORD_SYSTEM}
            assert tools is ENV.tools
            assert tag.startswith(f"rec:4:{plane}:")
            users = [m["content"] for m in messages if m["role"] == "user"]
            mine = [t.text for i, t in enumerate(ep.tickets) if ("a", "b")[i % 2] == plane]
            assert [u.split(": ", 1)[1] for u in users] == mine[: len(users)]
        final = rec.transcripts[plane]
        assert final[0]["role"] == "user"  # the system prompt is not stored per session
        tool_msgs = [m for m in final if m["role"] == "tool"]
        assert len(tool_msgs) == sum(1 for p, *_ in rec.calls if p == plane)


def test_a_call_that_fails_or_cannot_be_parsed_is_kept_as_a_request():
    extra = {
        0: [
            _raw(0, "cancel_order", '{"order_id": "nope"}'),
            _raw(1, "update_customer", "{not json"),
            _raw(2, "no_such_tool", "[1, 2]"),
        ]
    }
    ep, _, _, rec = _record(extra_a=extra)
    head = rec.calls[:3]
    assert head == [
        ("a", 0, "cancel_order", {"order_id": "nope"}),
        ("a", 0, "update_customer", {}),
        ("a", 0, "no_such_tool", {}),
    ]
    assert rec.outcomes[:3] == [False, False, False]
    assert all(rec.outcomes[3:])
    # Failed calls are not actions: the ledger is the plans' alone.
    assert len(rec.ledger) == sum(
        1 for t in ep.tickets for tool, _ in t.plan if tool in ENV.action_tools
    )


def test_a_ticket_that_runs_out_of_turns_is_named_and_the_session_goes_on():
    ep, _, _, rec = _record(max_turns=1)
    # One turn per ticket: the first planned call is made and the reply in words never comes.
    assert rec.unfinished == list(range(len(ep.tickets)))
    assert [c[1] for c in rec.calls] == list(range(len(ep.tickets)))


def test_reasoning_is_passed_back_as_the_provider_returned_it():
    ep = ENV.make_episode(4, 10)
    follower = PlanFollower(ep)

    def thinking(messages, tools=None, tag=""):
        return {**follower.complete(messages, tools, tag), "reasoning_content": ""}

    rec = record_session(4, 10, ENV, {"a": thinking, "b": thinking})
    assistants = [m for m in rec.transcripts["a"] if m["role"] == "assistant"]
    assert assistants and all(m["reasoning_content"] == "" for m in assistants)
    _, _, _, plain = _record()
    assert all("reasoning_content" not in m for m in plain.transcripts["a"])


def test_the_session_file_replays_the_recording_in_its_order():
    ep, _, _, rec = _record()
    sessions = rec.sessions(pace_s=1.0)
    assert [s.plane for s in sessions] == ["a", "b"]
    at = sorted(
        (t, s.plane, st.ticket, st.tool, st.args)
        for s in sessions
        for t, st in zip(offsets(s), s.steps, strict=True)
    )
    assert [row[1:] for row in at] == rec.calls
    assert all(s.session == "s4" and s.seed == 4 and s.n_tickets == 10 for s in sessions)


def test_a_model_for_every_plane_is_required():
    ep = ENV.make_episode(4, 10)
    with pytest.raises(KeyError):
        record_session(4, 10, ENV, {"a": PlanFollower(ep).complete})
