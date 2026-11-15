"""scripts/study5_pilot_summary.py: totals and splits over rounds, from artifacts alone."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import study5_pilot_summary as ps  # noqa: E402
from test_study5_lib import rec  # noqa: E402


def outage(kill: float, length: float, service: str = "otelcol-gw") -> list[dict]:
    return [
        {"at_s": kill, "action": "kill", "service": service},
        {"at_s": round(kill + length, 3), "action": "start", "service": service},
    ]


def run(name: str, condition: str, schedule: list[dict], x=(100, 90, 110), n: int = 100, **kw):
    est = dict(
        ll_sa_hom=x, chapman_te=x, strat_bic=x, ll_bic=x, ll_bic_hom=x, ll_bic_boot=x,
        ll_average=x, chapman_sa=(70, 65, 75), seq_gap=(95, 95, 95),
        seq_gap_tail=kw.pop("seq", (95, 94, 96)), union_count=(92, 92, 92),
        union_bound=(92, 92, float("inf")), trust_trace=(60, 60, 60),
    )  # fmt: skip
    base = rec(n, missed=kw.pop("missed", 0.05), **est)
    return base | {
        "run": name, "condition": condition, "schedule": schedule, "duration_s": 120.0,
        "union_count": 92, "tables": {"a": [0, 1, 2, 3, 4, 5, 6, 7], "b": [0] * 8},
        "sessions_sha256": name, "commit": "abcdef012345", "dirty": False,
        "actions_by_plane": {"a": n // 2, "b": n - n // 2}, "timing": None,
        "containers_clock_minus_driver_s": None,
    } | kw  # fmt: skip


def rig_run(name: str, schedule: list[dict], given_up: dict, errors=(3, 3, 0)) -> dict:
    names = ("transient_errors", "name_resolution_errors", "connection_refused")
    kinds = dict(zip(names, errors, strict=True))
    lost = {"spans": {"calls": 4, "first": 10.0, "last": 20.0, "written_between": 4}}
    return {
        "run": name, "schedule": schedule, "span_batches_given_up": given_up,
        "sdk_export_messages": {a: dict(kinds) for a in given_up},
        "planes": {p: {"lost_calls_written_at_s": lost} for p in ("a", "b")},
    }  # fmt: skip


def test_an_outage_length_and_who_was_killed_for_good_come_from_the_schedule():
    r = run("r", "x", outage(10.0, 21.5) + [{"at_s": 30.0, "action": "kill", "service": "agent-b"}])
    assert ps.outage_s(r) == 21.5
    assert ps.outage_s(r, "otelcol-node") is None
    assert ps.killed(r) == ["agent-b"]
    assert ps.outage_s(run("q", "none", [])) is None and ps.killed(run("q", "none", [])) == []


def test_a_tally_counts_coverage_direction_width_and_the_paired_comparisons():
    hit = run("a", "c", [], x=(99, 90, 110), seq=(95, 94, 96))
    high = run("b", "c", [], x=(112, 104, 120), seq=(99, 98, 100))
    wide = run("c", "c", [], x=(float("inf"), 90, float("inf")))
    t = ps.tally([hit, hit, high, wide], "ll_sa_hom")
    assert (t["runs"], t["covered"], t["unbounded"]) == (4, 3, 1)  # an unbounded one covers
    assert (t["over"], t["under"]) == (1, 2)
    assert t["median_abs_error"] == 1.0 and t["mean_error"] == pytest.approx(10 / 3, abs=1e-3)
    assert t["median_width"] == 20.0  # over the three bounded intervals
    assert t["wilson"] == [round(v, 4) for v in ps.lib.wilson(3, 4)]
    assert t["closer_than"]["seq_gap_tail"]["wins"] == 2  # high loses, wide loses
    assert t["closer_than"]["union_count"]["wins"] == 2
    assert "closer_than" not in ps.tally([hit], "seq_gap_tail")


def test_the_proposed_runs_are_one_condition_after_one_round_split_by_outage_length():
    r1 = {"runs": [run("p1", "node-restart", outage(5, 30)), run("n1", "none", [])],
          "runs_set_aside": []}  # fmt: skip
    lengths = (20.0, 22.0, 27.0, 40.0)
    restarts = [run(f"p2-{i}", "node-restart", outage(5, x)) for i, x in enumerate(lengths)]
    blip = run("b2", "node-blip", outage(5, 18.0), missed=0.01)
    r2 = {"runs": [*restarts, blip], "runs_set_aside": [{"run": "z"}]}
    rep = {"runs": [run("p2-0", "node-restart", outage(5, 20.0))] * 2}
    rig = {"runs": [rig_run("p2-0", outage(5, 20.0), {"agent-a": 1, "agent-b": 1})]}
    out = ps.summarise([r1, r2], [rep, rep], rig, "node-restart", after=1, short_s=25.0)
    assert out["runs"] == 7 and out["unstressed"]["runs"] == 1 and out["stressed"]["runs"] == 6
    assert out["stressed_round_1"]["runs"] == 1
    assert [r["set_aside"] for r in out["rounds"]] == [0, 1]
    assert out["rounds"][0]["commits"] == ["abcdef0"] and out["rounds"][0]["dirty"] == [False]
    p = out["proposed_after_round"]
    # Round 1's run of the same condition is not among them, nor is the blip.
    assert (p["all"]["runs"], p["short"]["runs"], p["long"]["runs"]) == (4, 2, 2)
    assert p["same_split_between_s"] == [22.0, 27.0]
    assert set(out["by_condition_and_round"]) == {
        "node-restart (1)", "none (1)", "node-restart (2)", "node-blip (2)",
    }  # fmt: skip
    # Short outages of any condition: the two short restarts and the blip.
    assert out["short_outages_any_condition"]["runs"] == 3
    assert (
        out["by_condition_and_round"]["node-blip (2)"]["in_no_log"]["runs_under_exercised_min"] == 1
    )
    assert out["stressed"]["seq_gap_tail_below_union"] == 0
    assert len(out["repeats"]) == 2  # one entry per repeated schedule
    assert out["repeats"][0]["one_schedule"] and out["repeats"][0]["largest_cell_difference"] == 0
    assert out["timing"] == {"runs_with_a_report": 0, "max_lag_s": None,
                             "containers_clock_minus_driver_s": {}}  # fmt: skip


def test_runs_with_a_killed_originator_are_kept_out_of_the_end_state_share_and_listed():
    kill = [{"at_s": 30.0, "action": "kill", "service": "agent-a"}]
    killed = run("k", "node-down-kill", outage(5, 30) + kill, x=(120, 105, 140))
    killed["missed_share"]["a"]["env"] = 0.9
    plain = run("p", "node-down", outage(5, 30))
    rep = {"runs": [plain]}
    out = ps.summarise([{"runs": [killed, plain], "runs_set_aside": []}], [rep], {"runs": []},
                       "node-down", 0, 25.0)  # fmt: skip
    assert out["end_state_hides_share_no_kill"] == {"runs": 1, "min": 0.4, "max": 0.4}
    (row,) = out["killed_originator"]["runs"]
    assert row["run"] == "k" and row["killed"] == ["agent-a"] and row["ll_sa_hom_error"] == 20.0


def test_batches_are_counted_per_live_originator_against_the_span_relays_outage():
    both = lambda n, m=None: {"agent-a": n, "agent-b": n if m is None else m}  # noqa: E731
    kill = [{"at_s": 30.0, "action": "kill", "service": "agent-b"}]
    runs = [
        rig_run("a", outage(5, 18.0), both(1)),
        rig_run("a-again", outage(5, 18.0), both(1)),  # the same schedule a second time
        rig_run("b", outage(9, 21.5), both(1, 2), errors=(5, 4, 1)),
        rig_run("c", outage(5, 30.0) + kill, both(2, 0)),  # agent-b was killed: left out
        rig_run("d", [], both(0)),  # no outage: left out
        rig_run("e", outage(5, 12.0, "otelcol-node"), both(0)),  # not the span relay
    ]
    b = ps.bands(runs)
    assert b["by_count"] == {
        "1": {"outage_s": [18.0, 21.5], "originator_runs": 5, "schedules": 2},
        "2": {"outage_s": [21.5, 30.0], "originator_runs": 2, "schedules": 2},
    }
    assert b["originator_runs"] == 7
    assert b["originators_disagree"] == [
        {"run": "b", "outage_s": 21.5, "given_up": {"agent-a": 1, "agent-b": 2}}
    ]
    assert ps.sdk_errors(runs) == {
        "transient_errors": 40, "name_resolution_errors": 38, "connection_refused": 2,
    }  # fmt: skip


def test_the_thresholds_are_the_criterias_at_a_given_number_of_runs():
    t = ps.thresholds(300)
    assert (t["runs"], t["covered_for_P1"], t["wins_for_P2"]) == (300, 281, 168)
    assert ps.thresholds(1000)["covered_for_P1"] / 1000 < 281 / 300  # more runs, a lower rate
    # How often 281 of 300 is reached, if runs were independent: rarely at 0.92, mostly
    # at 0.95. A rate at the floor itself almost never passes.
    p1 = t["P1_reached_if_true_coverage"]
    assert p1["0.9"] < 0.02 and 0.1 < p1["0.92"] < 0.25 and 0.8 < p1["0.95"] < 0.95
    assert list(p1.values()) == sorted(p1.values())
    assert t["P2_reached_if_true_win_rate"]["0.5"] < 0.03
    assert ps.tail(0, 10, 0.3) == 1.0 and ps.tail(10, 10, 0.5) == 0.5**10


def test_a_restart_close_to_the_runs_end_is_listed():
    late = run("late", "x", outage(90.0, 28.5))  # back at 118.5 of 120
    early = run("early", "x", outage(10.0, 28.5))
    assert [r["run"] for r in ps.near_end([late, early, run("n", "none", [])], 3.0)] == ["late"]


def test_the_proposed_runs_are_also_read_by_band_with_the_runs_not_covered_listed():
    def lost(n: int, shown: int) -> dict:
        return {"actions": n, "shown_by_end_state": shown, "rest": 100 - n, "rest_shown": 60}

    parts = [[20, 8], [20, 10], [20, 12], [20, 14], [20, 16]]
    made = [(15.0, 4, False), (19.5, 30, True), (24.0, 20, True), (40.0, 35, True)]
    runs = [
        run(f"p-{i}", "node-restart", outage(5, x), x=(100, 90, 110) if ok else (120, 110, 130),
            span_relay_outage_s=x, lost_by_both=lost(m, m // 2), shown_by_position=parts,
            pool="ds")
        for i, (x, m, ok) in enumerate(made)
    ]  # fmt: skip
    rig = {"runs": []}
    out = ps.summarise([{"runs": runs, "runs_set_aside": []}], [], rig, "node-restart", 0, 25.0)
    e = out["exploratory"]
    assert e["runs"] == 4 and e["span_relay_outage_s"] == [15.0, 19.5, 24.0, 40.0]
    assert e["by_outage_s"]["0 to 20"]["runs"] == 2
    assert e["by_outage_s"]["0 to 20"]["covered"]["ll_sa_hom"] == 1
    assert e["by_outage_s"]["28 or more"]["covered"]["ll_sa_hom"] == 1
    assert e["by_lost_by_both"]["0 to 12"]["covered"]["ll_sa_hom"] == 0
    assert e["by_lost_by_both"]["12 or more"]["covered"]["ll_sa_hom"] == 3
    assert e["not_covered"] == [
        {"run": "p-0", "pool": "ds", "span_relay_outage_s": 15.0, "actions": 100,
         "point": 120.0, "lo": 110.0, "hi": 130.0, "lost_by_both": lost(4, 2)}
    ]  # fmt: skip
    first, last = e["shown_by_position"]["all"][0], e["shown_by_position"]["all"][-1]
    assert first == {"actions": 80, "shown": 32, "share": 0.4}
    assert last == {"actions": 80, "shown": 64, "share": 0.8}
    assert set(e["shown_by_position"]["by_outage_s"]) == {"0 to 20", "20 to 28", "28 or more"}
    assert e["shown_by_position"]["by_outage_s"]["0 to 20"][0]["actions"] == 40
    # Runs scored before these fields existed give an empty description, not an error.
    old = [run("o-0", "node-restart", outage(5, 30.0))]
    e = ps.summarise([{"runs": old, "runs_set_aside": []}], [], rig, "node-restart", 0, 25.0)
    assert e["exploratory"]["shown_by_position"]["all"] == []
    assert e["exploratory"]["by_lost_by_both"]["0 to 12"]["runs"] == 0
