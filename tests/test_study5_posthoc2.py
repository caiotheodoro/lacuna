"""scripts/study5_posthoc2.py: two more descriptions made after the verdicts, on small inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402
import study5_posthoc2 as post  # noqa: E402

# Cell i counts the actions seen by exactly the logs whose bit is set: spans 1, access 2,
# end state 4. Cell 0 is nobody's.
A = [0, 7, 4, 55, 18, 11, 6, 72]
B = [0, 3, 1, 41, 19, 9, 0, 83]


def _run(tables: dict, actions: int, condition: str = "s1") -> dict:
    return {"run": "r", "condition": condition, "tables": tables, "actions": actions}


def _two_list(both: int, tele_only: int, env_only: int, actions: int) -> dict:
    """A run whose one plane holds exactly these two-list cells."""
    return _run({"a": [0, tele_only, 0, 0, env_only, both, 0, 0]}, actions)


def test_the_shares_are_the_counts_divided_and_nothing_else():
    lost = {"actions": 40, "shown_by_end_state": 10, "rest": 200, "rest_shown": 150, "no_link": 0}
    a = post.pooled_shown(lost, runs=7, not_described=0)
    assert a["lost_by_both_telemetry_logs"] == {"actions": 40, "shown": 10, "share": 0.25}
    assert a["the_rest"] == {"actions": 200, "shown": 150, "share": 0.75}
    assert (a["runs"], a["runs_not_described"], a["no_link"]) == (7, 0, 0)
    assert "not a test" in a["fence"] and "pooled" in a["fence"]


def test_a_pool_with_no_lost_action_has_no_share():
    lost = {"actions": 0, "shown_by_end_state": 0, "rest": 9, "rest_shown": 3, "no_link": 0}
    assert post.pooled_shown(lost, 1, 0)["lost_by_both_telemetry_logs"]["share"] is None


def test_the_cells_are_the_planes_summed_and_the_telemetry_taken_as_one_list():
    # By hand: both = 11+6+72 + 9+0+83, telemetry only = 7+4+55 + 3+1+41, end state only = 18+19.
    assert post.two_list_cells({"a": A, "b": B}) == (181, 111, 37)
    pooled = np.array(A) + np.array(B)
    collapsed = lib.collapse_telemetry(pooled)
    assert post.two_list_cells({"a": A, "b": B}) == tuple(int(collapsed[i]) for i in (3, 1, 2))
    # The registered estimator's own missing cell on that table.
    z0 = lib.sa_hom(pooled.astype(np.int64), []).detail["z0"]
    assert z0 == pytest.approx(37 * 111 / 181)


def test_an_action_nobody_saw_is_not_a_cell():
    assert post.two_list_cells({"a": [5, 1, 0, 0, 2, 3, 0, 0]}) == (3, 1, 2)


def test_the_implied_odds_ratio_is_one_when_the_two_list_count_is_the_true_count():
    # 30 in both, 10 and 6 in one each: the two-list count adds 10 * 6 / 30 = 2.
    assert post.implied_odds_ratio(30, 10, 6, 30 + 10 + 6 + 2) == 1.0


def test_the_implied_odds_ratio_by_hand():
    # Unseen 343 - 329 = 14; 181 * 14 / (111 * 37) = 2534 / 4107.
    assert post.implied_odds_ratio(181, 111, 37, 343) == pytest.approx(2534 / 4107)
    # Twice the unseen actions the two-list count adds: 2.
    assert post.implied_odds_ratio(30, 10, 6, 50) == 2.0
    # The union already exact: 0.
    assert post.implied_odds_ratio(30, 10, 6, 46) == 0.0


def test_it_is_not_defined_when_a_list_saw_nothing_alone():
    assert post.implied_odds_ratio(30, 0, 6, 40) is None
    assert post.implied_odds_ratio(30, 10, 0, 40) is None


def test_the_summary_over_runs_by_hand():
    runs = [
        _two_list(30, 10, 6, 46),  # 0
        _two_list(30, 10, 6, 47),  # 0.5
        _two_list(30, 10, 6, 48),  # 1: within both factors
        _two_list(30, 10, 6, 50),  # 2
        _two_list(40, 10, 5, 56),  # unseen 1; 40 / 50 = 0.8: within both, at the edge of 1.25
        _two_list(40, 10, 5, 57),  # unseen 2; 1.6: within neither
        _two_list(60, 10, 8, 80),  # unseen 2; 120 / 80 = 1.5: within 1.5, at its edge
        _two_list(30, 0, 6, 40),  # not defined
        _two_list(30, 10, 6, 48) | {"condition": "s2"},  # another condition: not read
    ]
    d = post.dependence_needed([r for r in runs if r["condition"] == "s1"])
    assert (d["runs"], d["defined"], d["not_defined"]) == (8, 7, 1)
    # Sorted: 0, 0.5, 0.8, 1, 1.5, 1.6, 2. Quartiles that include both ends.
    assert d["implied_odds_ratio"] == {"q1": 0.65, "median": 1.0, "q3": 1.55}
    assert d["within_a_factor_of_1"] == [
        {"factor": 1.25, "from": 0.8, "to": 1.25, "runs": 2},
        {"factor": 1.5, "from": 0.6667, "to": 1.5, "runs": 3},
    ]
    assert d["union_already_the_count"] == 1
    assert "none of" in d["what_it_is_not"]


def test_no_defined_run_gives_no_quartile():
    d = post.dependence_needed([_two_list(30, 0, 6, 40)])
    assert d["defined"] == 0 and d["implied_odds_ratio"] is None


def test_the_cells_give_the_registered_point_in_every_run_of_the_test():
    """The task's check on three runs, made on all of them: the pooled cells reproduce the
    missing cell the registered estimator wrote down, and the count it saw."""
    confirm = json.loads((ROOT / "artifacts" / "study5" / "confirm.json").read_text())
    runs = [r for r in confirm["runs"] if r["condition"] == "s1"]
    assert len(runs) == 277
    for r in runs:
        both, tele, env = post.two_list_cells(r["tables"])
        assert both + tele + env == r["union_count"]
        e = r["estimates"]["ll_sa_hom"]
        assert e["detail"]["z0"] == pytest.approx(env * tele / both, abs=1e-9)
        assert e["point"] == pytest.approx(both + tele + env + env * tele / both, abs=5e-4)


def test_the_pooled_counts_are_the_runs_summed():
    confirm = json.loads((ROOT / "artifacts" / "study5" / "confirm.json").read_text())
    verdicts = json.loads((ROOT / "artifacts" / "robustness" / "study5-verdicts.json").read_text())
    runs = [r for r in confirm["runs"] if r["condition"] == "s1"]
    for key in lib.LOST_KEYS:
        assert verdicts["exploratory"]["lost_by_both"][key] == sum(
            r["lost_by_both"][key] for r in runs
        )


def test_the_artifact_says_it_is_post_hoc_and_is_what_the_script_writes(tmp_path):
    confirm = json.loads((ROOT / "artifacts" / "study5" / "confirm.json").read_text())
    verdicts = json.loads((ROOT / "artifacts" / "robustness" / "study5-verdicts.json").read_text())
    post.main(["--out", str(tmp_path / "p.json")])
    made = json.loads((tmp_path / "p.json").read_text())
    assert made == post.summarise(confirm, verdicts)
    assert made["note"].startswith("Post hoc") and "D107" in made["note"]
    assert "no registered criterion" in made["note"]
    assert made["dependence_each_run_would_need"]["runs"] == 277
    committed = ROOT / "artifacts" / "study5" / "posthoc2.json"
    assert json.loads(committed.read_text()) == made
