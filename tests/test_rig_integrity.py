"""A run that went wrong in the plumbing must say so, not pass as a run with loss."""

from __future__ import annotations

from lacuna.rig.integrity import killed_for_good, problems
from lacuna.rig.manifest import tree_hash
from lacuna.rig.stress import Event, plan

GOOD = {
    "schedule": [],
    "events_done": [],
    "exit_codes": {"agent-a": 0, "agent-b": 0},
    "reports": {"agent-a": {}, "agent-b": {}},
    "file_sizes": {"spans.jsonl": 10, "logs.jsonl": 10, "env.jsonl": 10},
    "not_running": [],
    "follower_lines": {"gateway.jsonl": 5, "front-b.jsonl": 4},
    "full_lines": {"gateway.jsonl": 5, "front-b.jsonl": 4},
    "planned_calls": {"a": 5, "b": 4},
    "written_calls": {"a": 5, "b": 4},
    "agent_plane": {"agent-a": "a", "agent-b": "b"},
}


def _with(**over) -> list[str]:
    return problems(**{**GOOD, **over})


def test_a_clean_run_has_no_problems():
    assert _with() == []


def test_each_kind_of_broken_plumbing_is_named():
    assert "agent-a exited 1" in _with(exit_codes={"agent-a": 1, "agent-b": 0})[0]
    assert "s3" in _with(reports={"agent-a": {"s3": "boom"}, "agent-b": {}})[0]
    assert "no report" in _with(reports={"agent-a": None, "agent-b": {}})[0]
    assert "logs.jsonl" in _with(file_sizes={**GOOD["file_sizes"], "logs.jsonl": 0})[0]
    assert "otelcol-node" in _with(not_running=["otelcol-node"])[0]
    assert "front-b.jsonl" in _with(follower_lines={"gateway.jsonl": 5, "front-b.jsonl": 3})[0]
    assert "plane a" in _with(written_calls={"a": 4, "b": 4})[0]
    assert len(_with(not_running=["backend"], exit_codes={"agent-a": 2, "agent-b": 3})) == 3


def test_a_stress_event_that_did_not_happen_is_a_problem():
    schedule = plan("gw-outage", 0)
    done = [{**vars(e), "ok": True} for e in schedule]
    assert _with(schedule=schedule, events_done=done) == []
    assert "start otelcol-gw" in _with(schedule=schedule, events_done=[done[0]])[0]
    failed = [done[0], {**done[1], "ok": False}]
    assert "start otelcol-gw" in _with(schedule=schedule, events_done=failed)[0]


def test_a_killed_originator_is_expected_to_die_and_to_leave_calls_unmade():
    schedule = plan("agent-kill", 0)
    assert killed_for_good(schedule) == {"agent-b"}
    assert killed_for_good(plan("gw-outage", 0)) == set()
    assert killed_for_good(
        [Event(1, "kill", "x"), Event(2, "start", "x"), Event(3, "kill", "x")]
    ) == {"x"}
    done = [{**vars(e), "ok": True} for e in schedule]
    killed = {
        "schedule": schedule,
        "events_done": done,
        "exit_codes": {"agent-a": 0, "agent-b": 137},
        "reports": {"agent-a": {}, "agent-b": None},
        "written_calls": {"a": 5, "b": 2},
    }
    assert _with(**killed) == []
    # It must really have been killed, and the other plane is still held to its plan.
    assert "agent-b exited 0" in _with(**{**killed, "exit_codes": {"agent-a": 0, "agent-b": 0}})[0]
    assert "plane a" in _with(**{**killed, "written_calls": {"a": 4, "b": 2}})[0]


def test_tree_hash_follows_content_and_names_not_order(tmp_path):
    (tmp_path / "pkg").mkdir()
    a, b = tmp_path / "pkg" / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    h = tree_hash(tmp_path, [a, b])
    assert h == tree_hash(tmp_path, [b, a]) and len(h) == 64
    a.write_text("x = 2\n")
    assert tree_hash(tmp_path, [a, b]) != h
    a.write_text("x = 1\n")
    a.rename(tmp_path / "pkg" / "c.py")
    assert tree_hash(tmp_path, [tmp_path / "pkg" / "c.py", b]) != h
