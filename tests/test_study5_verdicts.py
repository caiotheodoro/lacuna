"""scripts/study5_verdicts.py: the criteria applied to an analysis artifact."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import study5_verdicts as sv  # noqa: E402
from test_study5_lib import rec  # noqa: E402


def run(condition: str, **est) -> dict:
    return rec(100, **est) | {"condition": condition}


def artifact(runs: list[dict], made: dict[str, int] | None = None) -> dict:
    conditions = {r["condition"] for r in runs}
    return {
        "runs": runs,
        "runs_set_aside": [{"run": "x", "why": "crashed"}],
        "by_condition": {
            c: {"gates": {"runs_made": (made or {}).get(c, sum(r["condition"] == c for r in runs))}}
            for c in conditions
        },
    }


GOOD = dict(x=(99, 90, 110), chapman_sa=(70, 65, 75), seq_gap_tail=(90, 90, 90),
            union_count=(92, 92, 92), union_bound=(92, 92, float("inf")),
            trust_trace=(60, 60, 60))  # fmt: skip


def test_the_verdict_reads_one_condition_and_reports_the_rest_beside_it():
    runs = [run("s1", **GOOD)] * 60 + [run("s0", **GOOD | {"x": (100, 98, 104)})] * 10
    out = sv.build(artifact(runs), "x", "s1", "s0", "chapman_sa", 50)
    assert out["condition"] == "s1" and out["min_runs"] == 50 and out["runs_set_aside"] == 1
    assert out["primary"]["P1"] == "pass" and out["primary"]["P2"] == "pass"
    assert out["primary"]["coverage"]["runs"] == 60
    # Without the end state the estimate is short of the truth in every run.
    assert out["Q3_without_env"]["P1"] == "fail" and out["Q3_without_env"]["P2"] == "fail"
    assert set(out["floors"]) == {"union_count", "union_bound", "trust_trace"}
    assert out["floors"]["union_count"]["covered"] == 0
    assert out["control"]["condition"] == "s0"
    assert out["control"]["coverage"]["runs"] == 10
    assert out["control"]["false_alarms"] == {"runs": 10, "interval_above_truth": 0}


def test_the_verdict_carries_the_exploratory_breakdown_of_the_primary_and_no_criterion_on_it():
    lost = {"actions": 20, "shown_by_end_state": 9, "rest": 80, "rest_shown": 50}
    few = lost | {"actions": 3, "rest": 97}
    hit = run("s1", **GOOD) | {"span_relay_outage_s": 30.0, "lost_by_both": lost}
    miss = run("s1", **GOOD | {"x": (120, 110, 130)}) | {
        "span_relay_outage_s": 16.0, "lost_by_both": few, "run": "m",
    }  # fmt: skip
    quiet = run("s0", **GOOD) | {"span_relay_outage_s": None, "lost_by_both": few}
    e = sv.build(artifact([hit] * 5 + [miss, quiet]), "x", "s1", "s0", "chapman_sa", 1)[
        "exploratory"
    ]
    # The primary condition's runs only, and the primary estimate only.
    assert e["by_outage_s"]["28 or more"] == {
        "runs": 5, "covered": {"x": 5},
        "lost_by_both": {"actions": 100, "shown_by_end_state": 45, "rest": 400, "rest_shown": 250},
    }  # fmt: skip
    assert e["by_outage_s"]["0 to 20"]["covered"] == {"x": 0}
    assert e["by_lost_by_both"]["0 to 12"]["runs"] == 1
    assert [r["run"] for r in e["not_covered"]["x"]] == ["m"]
    assert e["not_covered"]["x"][0]["outside_by"] == 10.0
    assert not any("pass" in str(v) or "fail" in str(v) for v in e.values())


def test_runs_set_aside_still_count_as_made_for_the_underpowered_gate():
    runs = [run("s1", **GOOD)] * 60
    out = sv.build(artifact(runs, {"s1": 75}), "x", "s1", None, "chapman_sa", 61)
    assert out["primary"]["gates"]["runs_made"] == 75
    assert out["primary"]["gates"]["runs_passing_integrity"] == 60
    assert out["primary"]["P1"] == "underpowered" and "control" not in out


def test_a_control_that_invents_actions_is_counted():
    loud = run("s0", **GOOD | {"x": (110, 104, 118)})
    out = sv.build(
        artifact([run("s1", **GOOD)] * 60 + [loud] * 4), "x", "s1", "s0", "chapman_sa", 50
    )
    assert out["control"]["false_alarms"] == {"runs": 4, "interval_above_truth": 4}


def test_an_estimator_the_analysis_does_not_hold_is_refused():
    with pytest.raises(KeyError, match="nope"):
        sv.build(artifact([run("s1", **GOOD)]), "nope", "s1", None, "chapman_sa", 1)


def test_an_analysis_made_without_a_registration_is_refused_unless_called_a_pilot(
    tmp_path, monkeypatch
):
    import json

    art = artifact([run("s1", **GOOD)] * 60)
    path, out = tmp_path / "a.json", tmp_path / "v.json"
    argv = ["study5_verdicts.py", "--analysis", str(path), "--primary", "x",
            "--condition", "s1", "--min-runs", "50", "--out", str(out)]  # fmt: skip
    path.write_text(json.dumps(art | {"registered": None}))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="not the study's"):
        sv.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", [*argv, "--pilot"])
    sv.main()
    assert json.loads(out.read_text())["registered"] is None
    out.unlink()
    path.write_text(json.dumps(art | {"registered": "registered.json"}))
    monkeypatch.setattr(sys, "argv", argv)
    sv.main()
    v = json.loads(out.read_text())
    assert v["registered"] == "registered.json" and v["primary"]["P1"] == "pass"
    assert v["primary"]["P1_reading"] == "pass"


def test_a_boundary_condition_is_counted_like_the_primary_one_and_judged_by_nothing():
    short = run("s3", **GOOD | {"x": (80, 78, 82)})  # the interval misses
    runs = [run("s1", **GOOD)] * 60 + [short] * 12 + [run("s3", **GOOD)] * 8
    out = sv.build(artifact(runs), "x", "s1", None, "chapman_sa", 50, boundary="s3")
    edge = out["boundary"]
    assert set(edge) == {"condition", "runs", "coverage", "closer_than", "mean_union_missed_share"}
    assert (edge["condition"], edge["runs"]) == ("s3", 20)
    assert edge["coverage"]["covered"] == 8 and edge["mean_union_missed_share"] == 0.05
    # Counts against each baseline, and nothing that reads as a verdict.
    assert set(edge["closer_than"]) == {"seq_gap_tail", "union_count"}
    for row in edge["closer_than"].values():
        assert row == {"runs": 20, "wins": 8, "ties": 0, "win_rate": 0.4}
    assert out["primary"]["P1"] == "pass"  # the boundary's misses are not the primary's
    assert "boundary" not in sv.build(artifact(runs), "x", "s1", None, "chapman_sa", 50)


def test_a_boundary_that_is_not_one_is_refused():
    runs = [run("s1", **GOOD)] * 60 + [run("s0", **GOOD)] * 5 + [run("s3", **GOOD)] * 5
    for bad, word in (("s1", "primary or the control"), ("s0", "primary or the control"),
                      ("s9", "no run of the boundary condition 's9'")):  # fmt: skip
        with pytest.raises(KeyError, match=word):
            sv.build(artifact(runs), "x", "s1", "s0", "chapman_sa", 50, boundary=bad)
    lacking = [run("s1", **GOOD)] * 60 + [run("s3", union_count=(9, 9, 9))]
    with pytest.raises(KeyError, match="no estimate named 'x'"):
        sv.build(artifact(lacking), "x", "s1", None, "chapman_sa", 50, boundary="s3")


def test_the_command_reports_a_boundary_and_exits_on_a_wrong_one(tmp_path, monkeypatch):
    import json

    runs = [run("s1", **GOOD)] * 60 + [run("s3", **GOOD)] * 6
    path, out = tmp_path / "a.json", tmp_path / "v.json"
    path.write_text(json.dumps(artifact(runs) | {"registered": "registered.json"}))
    argv = ["study5_verdicts.py", "--analysis", str(path), "--primary", "x", "--condition", "s1",
            "--min-runs", "50", "--out", str(out)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", [*argv, "--boundary", "s3"])
    sv.main()
    assert json.loads(out.read_text())["boundary"]["runs"] == 6
    out.unlink()
    monkeypatch.setattr(sys, "argv", [*argv, "--boundary", "s7"])
    with pytest.raises(SystemExit, match="cannot give a verdict: no run of the boundary"):
        sv.main()
    assert not out.exists()
