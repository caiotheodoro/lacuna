"""scripts/study5_posthoc.py: two descriptions made after the verdicts, on small inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402
import study5_posthoc as post  # noqa: E402


def _run(actions: int, union: int, gap: float, mine: float, bounded: bool = True) -> dict:
    return {
        "condition": "s1",
        "actions": actions,
        "union_count": union,
        "estimates": {
            "seq_gap_tail": {"point": gap},
            "ll_sa_hom": {"abs_error": abs(mine - actions), "bounded": bounded},
        },
    }


def test_the_lower_end_at_whole_runs_is_the_registered_wilson():
    for k, n in ((269, 277), (18, 20), (5, 9)):
        assert abs(post.wilson_lower(k / n, n) - lib.wilson(k, n)[0]) < 1e-12


def test_more_dependence_lowers_the_bound_and_the_largest_effect_is_the_last_that_keeps_it():
    d = post.design_effect(269, 277)
    los = [r["wilson_lo"] for r in d["by_design_effect"]]
    assert los == sorted(los, reverse=True) and los[0] == round(lib.wilson(269, 277)[0], 4)
    top = d["largest_design_effect_at_or_above_the_bar"]
    assert post.wilson_lower(269 / 277, 277 / top) >= 0.90
    assert post.wilson_lower(269 / 277, 277 / (top + 0.02)) < 0.90


def test_a_share_already_under_the_bar_has_no_design_effect_that_keeps_it():
    assert post.design_effect(240, 277)["largest_design_effect_at_or_above_the_bar"] is None


def test_the_baseline_is_the_larger_of_the_gap_count_and_the_union_and_ties_go_against():
    runs = [
        _run(100, union=90, gap=95.0, mine=99.0),  # baseline 95, off 5; mine off 1: win
        _run(100, union=98, gap=92.0, mine=97.0),  # baseline 98, off 2; mine off 3: loss
        _run(100, union=100, gap=96.0, mine=100.0),  # baseline exact, both off 0: tie
        _run(100, union=90, gap=95.0, mine=100.0, bounded=False),  # no upper end: loss
    ]
    g = post.gap_with_end_state(runs, "ll_sa_hom")
    assert (g["runs"], g["wins"], g["ties"]) == (4, 1, 1)
    assert g["runs_where_the_baseline_is_the_count"] == 1
    assert g["lower_97_5"] == round(lib.exact_lower(1, 4), 4)


def test_the_artifact_says_it_is_post_hoc_and_is_what_the_script_writes(tmp_path):
    confirm = json.loads((ROOT / "artifacts" / "study5" / "confirm.json").read_text())
    post.main(["--out", str(tmp_path / "p.json")])
    made = json.loads((tmp_path / "p.json").read_text())
    assert made == post.summarise(confirm)
    assert made["note"].startswith("Post hoc") and "no registered criterion" in made["note"]
    committed = ROOT / "artifacts" / "study5" / "posthoc.json"
    assert json.loads(committed.read_text()) == made
