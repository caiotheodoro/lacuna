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
