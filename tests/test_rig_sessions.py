"""Sessions for the rig: a scripted plan dealt over planes, and the file they travel in."""

from __future__ import annotations

import numpy as np

from lacuna.agents.scripted import ScriptedAgent
from lacuna.envs.registry import BACKOFFICE, BACKOFFICE_CHURN
from lacuna.envs.tasks import make_episode
from lacuna.rig.sessions import (
    Session,
    Step,
    draw,
    dump,
    load,
    offsets,
    scripted_sessions,
    timed,
)


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


CALLS = [
    ("a", 0, "get_order", {"order_id": "O1"}),
    ("a", 0, "cancel_order", {"order_id": "O1"}),
    ("b", 1, "write_file", {"path": "/x", "content": "1"}),
    ("a", 2, "delete_file", {"path": "/x"}),
    ("b", 3, "write_file", {"path": "/y", "content": "2"}),
    ("b", 3, "write_file", {"path": "/y", "content": "3"}),
]


def _merged(sessions: list[Session]) -> list[tuple[str, int, str, dict]]:
    """Every plane's steps on the session's one clock, earliest first."""
    at = [
        (t, s.plane, st.ticket, st.tool, st.args)
        for s in sessions
        for t, st in zip(offsets(s), s.steps, strict=True)
    ]
    assert len({t for t, *_ in at}) == len(at)  # no two calls at one instant
    return [row[1:] for row in sorted(at, key=lambda row: row[0])]


def test_timed_puts_every_plane_on_one_clock_in_the_order_given():
    sessions = timed("s5", 5, 4, CALLS, pace_s=2.0, planes=("a", "b"))
    assert [s.plane for s in sessions] == ["a", "b"]
    assert all(s.session == "s5" and s.seed == 5 and s.n_tickets == 4 for s in sessions)
    assert [len(s.steps) for s in sessions] == [3, 3]
    assert _merged(sessions) == CALLS
    # Consecutive calls of the session are one pace apart, whoever makes them.
    times = sorted(t for s in sessions for t in offsets(s))
    steps = [b - a for a, b in zip([0.0, *times], times, strict=False)]
    assert all(1.0 <= d <= 3.0 for d in steps) and len({round(d, 3) for d in steps}) > 1
    assert timed("s5", 5, 4, CALLS, pace_s=2.0, planes=("a", "b")) == sessions
    assert timed("s6", 6, 4, CALLS, pace_s=2.0, planes=("a", "b")) != sessions


def test_timed_keeps_a_plane_that_makes_no_call_and_refuses_an_unknown_one():
    only_a = [c for c in CALLS if c[0] == "a"]
    sessions = timed("s5", 5, 4, only_a, pace_s=1.0, planes=("a", "b"))
    assert [(s.plane, len(s.steps)) for s in sessions] == [("a", 3), ("b", 0)]
    try:
        timed("s5", 5, 4, [("c", 0, "get_order", {})], pace_s=1.0, planes=("a", "b"))
    except KeyError:
        pass
    else:
        raise AssertionError("a call on a plane the session does not have was accepted")


def test_offsets_are_the_running_sum_of_the_gaps():
    s = Session(
        "s", 0, 1, "a", (Step(0, "t", {}, 0.5), Step(0, "t", {}, 0.25), Step(0, "t", {}, 1.0))
    )
    assert offsets(s) == [0.5, 0.75, 1.75]
    assert offsets(Session("s", 0, 1, "a", ())) == []


def test_scripted_sessions_on_one_clock_follow_the_queue():
    for seed in (0, 4, 9):
        sessions = scripted_sessions(
            seed, 10, ("a", "b"), pace_s=1.0, env=BACKOFFICE_CHURN, timeline=True
        )
        merged = _merged(sessions)
        tickets = [ticket for _, ticket, _, _ in merged]
        assert tickets == sorted(tickets)
        assert all(plane == ("a", "b")[ticket % 2] for plane, ticket, _, _ in merged)
        ep = BACKOFFICE_CHURN.make_episode(seed, 10)
        want = ScriptedAgent(np.random.default_rng([seed, 1])).plan(ep)
        got = [(tool, args) for _, _, tool, args in merged if tool in BACKOFFICE.action_tools]
        assert got == want


def test_the_default_is_still_a_clock_per_plane():
    a = scripted_sessions(3, 7, ("a", "b"), pace_s=2.0)
    assert all(1.0 <= st.gap_s <= 3.0 for s in a for st in s.steps)


def test_a_draw_takes_whole_sessions_by_seed_without_replacement():
    pool = [s for seed in range(12) for s in scripted_sessions(seed, 7, ("a", "b"))]
    got = draw(pool, 5, 3)
    names = [s.session for s in got]
    assert len(set(names)) == 5 and len(got) == 10
    assert all(names.count(n) == 2 for n in set(names))
    assert got == draw(pool, 5, 3)
    assert [s for s in pool if s in got] == got  # the pool's order
    assert len({tuple(s.session for s in draw(pool, 5, k)) for k in range(20)}) > 10
    assert draw(pool, 12, 0) == pool and draw(pool, 99, 0) == pool
