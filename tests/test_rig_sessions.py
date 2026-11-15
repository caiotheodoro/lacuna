"""Sessions for the rig: a scripted plan dealt over planes, and the file they travel in."""

from __future__ import annotations

import numpy as np

from lacuna.agents.scripted import ScriptedAgent
from lacuna.envs.registry import BACKOFFICE
from lacuna.envs.tasks import make_episode
from lacuna.rig.sessions import Session, Step, dump, load, scripted_sessions


def _actions(sessions: list[Session]) -> list[tuple[int, str, dict]]:
    return [
        (st.ticket, st.tool, st.args)
        for s in sessions
        for st in s.steps
        if st.tool in BACKOFFICE.action_tools
    ]


def test_the_actions_are_the_scripted_plan_and_nothing_is_dropped():
    for seed in (0, 5, 12):
        sessions = scripted_sessions(seed, 7, ("a", "b"))
        want = ScriptedAgent(np.random.default_rng([seed, 1])).plan(make_episode(seed, 7))
        got = sorted(_actions(sessions), key=lambda t: t[0])  # stable: order within a ticket
        assert [(tool, args) for _, tool, args in got] == want


def test_tickets_are_dealt_to_planes_in_turn():
    sessions = scripted_sessions(3, 7, ("a", "b"))
    assert [s.plane for s in sessions] == ["a", "b"]
    assert all(s.session == "s3" and s.seed == 3 and s.n_tickets == 7 for s in sessions)
    for i, s in enumerate(sessions):
        assert s.steps and {st.ticket % 2 for st in s.steps} == {i}
    three = scripted_sessions(3, 7, ("a", "b", "c"))
    assert [{st.ticket % 3 for st in s.steps} for s in three] == [{0}, {1}, {2}]


def test_reads_come_before_a_ticket_and_are_not_actions():
    sessions = scripted_sessions(3, 7, ("a", "b"))
    reads = [st for s in sessions for st in s.steps if st.tool not in BACKOFFICE.action_tools]
    assert reads and {st.tool for st in reads} <= {"get_order", "get_customer"}
    bare = scripted_sessions(3, 7, ("a", "b"), reads=False)
    assert _actions(bare) == _actions(sessions)
    assert all(st.tool in BACKOFFICE.action_tools for s in bare for st in s.steps)
    for s in sessions:
        first = {}
        for st in s.steps:
            first.setdefault(st.ticket, st)
        # Every ticket that names an order or a customer opens with a read of it.
        for st in first.values():
            assert st.tool in ("get_order", "get_customer", "write_file")


def test_gaps_follow_the_pace_and_the_seed():
    a = scripted_sessions(3, 7, ("a", "b"), pace_s=2.0)
    b = scripted_sessions(3, 7, ("a", "b"), pace_s=2.0)
    assert a == b
    gaps = [st.gap_s for s in a for st in s.steps]
    assert all(1.0 <= g <= 3.0 for g in gaps) and len(set(gaps)) > 1
    assert scripted_sessions(4, 7, ("a", "b"), pace_s=2.0) != a
    still = scripted_sessions(3, 7, ("a",), pace_s=0.0)
    assert all(st.gap_s == 0.0 for s in still for st in s.steps)


def test_the_file_round_trips():
    sessions = scripted_sessions(0, 7, ("a", "b")) + scripted_sessions(1, 7, ("a", "b"))
    assert load(dump(sessions)) == sessions
    one = Session("s9", 9, 7, "a", (Step(0, "cancel_order", {"order_id": "O1"}, 0.5),))
    assert load(dump([one])) == [one]
