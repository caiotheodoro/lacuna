"""A run that went wrong in the plumbing must say so, not pass as a run with loss."""

from __future__ import annotations

from lacuna.rig.integrity import file_problems, killed_for_good, problems
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


def test_both_planes_are_held_to_their_plan_in_both_directions():
    assert "plane b" in _with(written_calls={"a": 5, "b": 3})[0]
    assert "plane b" in _with(written_calls={"a": 5, "b": 6})[0]  # a retried call is not a call
    assert "plane a" in _with(written_calls={"b": 4})[0]


def test_a_missing_or_short_copy_of_an_access_log_is_a_problem_either_way():
    assert "gateway.jsonl" in _with(follower_lines={"front-b.jsonl": 4})[0]
    assert "gateway.jsonl" in _with(follower_lines={"gateway.jsonl": 6, "front-b.jsonl": 4})[0]
    assert "gateway.jsonl" in _with(follower_lines={"gateway.jsonl": 0, "front-b.jsonl": 4})[0]


def test_killed_for_good_reads_the_schedule_in_time_order_whatever_order_it_is_given():
    late_kill = [Event(3, "kill", "x"), Event(2, "start", "x"), Event(1, "kill", "x")]
    assert killed_for_good(late_kill) == {"x"}
    back_up = [Event(2, "start", "x"), Event(1, "kill", "x")]
    assert killed_for_good(back_up) == set()


def test_a_service_killed_for_good_is_not_expected_to_be_running():
    schedule = [Event(1.0, "kill", "otelcol-node")]
    done = [{**vars(e), "ok": True} for e in schedule]
    assert _with(schedule=schedule, events_done=done, not_running=["otelcol-node"]) == []
    assert "backend" in _with(schedule=schedule, events_done=done, not_running=["backend"])[0]


def _plane(**over) -> dict:
    base = {
        "actions_in_ledger": 10,
        "receiver_ok_actions": 10,
        "calls_written_by_receiver": 16,
        "access_lines_delivered": 16,
        "records_without_session": 0,
        "witnesses": {
            "spans": {"matching_no_action": 0},
            "access": {"matching_no_action": 0},
            "env": {"matching_no_action": 0},
        },
    }
    return {**base, **over}


def test_what_the_files_alone_can_show_to_be_wrong():
    good = {"planes": {"a": _plane(), "b": _plane()}}
    assert file_problems(good, dead_planes=set()) == []

    def bad(**over) -> list[str]:
        return file_problems({"planes": {"a": _plane(), "b": _plane(**over)}}, dead_planes=set())

    assert "session" in bad(records_without_session=2)[0]
    phantom = {**_plane()["witnesses"], "spans": {"matching_no_action": 1}}
    assert "spans" in bad(witnesses=phantom)[0] and "plane b" in bad(witnesses=phantom)[0]
    # The log relay delivering nothing at all for a plane is a broken path, not loss.
    assert "delivered no" in bad(access_lines_delivered=0)[0]
    assert bad(access_lines_delivered=0, calls_written_by_receiver=0) == []
    # The receiver and the kernel are both unmodified plumbing and must agree.
    assert "receiver" in bad(receiver_ok_actions=9)[0]
    assert "receiver" in bad(receiver_ok_actions=11)[0]


def test_a_killed_originator_may_leave_an_action_its_receiver_never_confirmed():
    cut = {"planes": {"a": _plane(), "b": _plane(receiver_ok_actions=9)}}
    assert file_problems(cut, dead_planes={"b"}) == []
    assert "plane b" in file_problems(cut, dead_planes=set())[0]
    # The other way round is never legitimate: a success the ledger does not hold.
    more = {"planes": {"a": _plane(), "b": _plane(receiver_ok_actions=11)}}
    assert "plane b" in file_problems(more, dead_planes={"b"})[0]


def test_a_run_whose_calls_were_late_or_whose_clocks_disagree_is_a_problem():
    on_time = {"agent-a": {"start_lag_s": 0.0, "max_lag_s": 0.25}, "agent-b": None}
    assert _with(timing=on_time, clock_offset_s=0.5) == []
    assert _with(timing=None, clock_offset_s=None) == []  # a run made before these existed
    late = {"agent-a": {"start_lag_s": 0.0, "max_lag_s": 0.251}, "agent-b": on_time["agent-a"]}
    (problem,) = _with(timing=late)
    assert "agent-a made a call 0.251 s late" in problem
    assert _with(timing=late, max_lag_s=0.3) == []
    slow = {"agent-b": {"start_lag_s": 0.004, "max_lag_s": 0.0}}
    assert "agent-b was ready 0.004 s after" in _with(timing=slow)[0]
    assert "clock" in _with(clock_offset_s=0.501)[0] and "clock" in _with(clock_offset_s=-0.6)[0]
    # A killed originator leaves no report and is not asked for one.
    schedule = plan("agent-kill", 0)
    done = [{**vars(e), "ok": True} for e in schedule]
    killed = {"schedule": schedule, "events_done": done,
              "exit_codes": {"agent-a": 0, "agent-b": 137}, "reports": {"agent-a": {}}}  # fmt: skip
    worst = {"agent-b": {"start_lag_s": 9.0, "max_lag_s": 9.0}}
    assert _with(**killed, timing=worst) == []
