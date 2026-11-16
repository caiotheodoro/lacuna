"""Every number the README states must come from MEASUREMENT_CARD.json.

AGENTS.md rule 2. Four figures were once typed from the wrong artifact
(D26); this is what stops that reaching a reader.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import claims_lib
import pytest

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "MEASUREMENT_CARD.json"
README = ROOT / "README.md"
S5_VERDICTS = ROOT / "artifacts" / "robustness" / "study5-verdicts.json"
S5_ANALYSIS = ROOT / "artifacts" / "study5" / "confirm.json"

sys.path.insert(0, str(ROOT / "scripts"))

import make_card  # noqa: E402
import study5_lib  # noqa: E402

# Figures that are not measurements: table dimensions, section numbers,
# years, model names, and the nominal level the whole project is stated at.
EXEMPT = {"0", "1", "2", "3", "4", "95", "100", "2026", "4.1", "0.0.0"}


def _mask(body: str) -> str:
    """`body` with what names a thing blanked out, every other character where it was."""
    for names in (
        r"`[^`]*`",  # code spans name files, not findings
        r"\bD\d+\b",  # D17 is a decision reference, not a measurement
        r"\b[Ss]tudy \d+\b",  # so is "study 5"
    ):
        body = re.sub(names, lambda m: " " * len(m.group()), body)
    return body


def _numbers(text: str) -> set[str]:
    # Only inside the Status section, which is where results are quoted.
    body = _mask(text.split("## Status", 1)[-1].split("## Run", 1)[0])
    return {m for m in re.findall(r"\d+\.\d+|\d+", body)} - EXEMPT


def _card_values(obj, out: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            _card_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _card_values(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(str(obj))
        # "undercounting by 93.63" states the magnitude of a bias of -93.63.
        out.add(str(abs(obj)))
        out.add(f"{abs(float(obj)):.1f}")
        out.add(f"{abs(float(obj)):.2f}")
        out.add(str(round(float(obj))))
        out.add(f"{float(obj) * 100:.0f}")  # a rate quoted as a percentage
        out.add(f"{float(obj):.1f}")
        out.add(f"{float(obj):.2f}")
    return out


@pytest.mark.skipif(not CARD.exists(), reason="card not built yet")
def test_readme_numbers_are_all_in_the_card():
    allowed = _card_values(json.loads(CARD.read_text()), set())
    # Thresholds and design constants from the frozen pre-registration are
    # stated facts, not measurements.
    prereg = (ROOT / "docs" / "PRE-REGISTRATION.md").read_text()
    allowed |= set(re.findall(r"\d+(?:\.\d+)?", prereg))
    missing = sorted(n for n in _numbers(README.read_text()) if n not in allowed)
    assert not missing, (
        f"README states figures absent from MEASUREMENT_CARD.json: {missing}. "
        "Either the card is stale (rerun scripts/make_card.py) or the claim is unsupported."
    )


@pytest.mark.skipif(not CARD.exists(), reason="card not built yet")
def test_card_names_its_sources():
    card = json.loads(CARD.read_text())
    for section in ("exploratory", "auditor"):
        assert card[section].get("source"), f"{section} does not name the artifact it came from"
    assert card["limitation"]


# --- key-aware gate ---------------------------------------------------------
# The test above passes a number found anywhere in the card. This one binds
# each Status number to the artifact cell it was read from (tests/claims_map.json).


def _status_numbers() -> tuple[str, dict[int, str]]:
    """The squashed Status text and {offset: number} for each figure needing a source."""
    body = claims_lib.squash(README.read_text().split("## Status", 1)[-1].split("## Run", 1)[0])
    return body, {
        m.start(): m.group()
        for m in re.finditer(r"\d+\.\d+|\d+", _mask(body))
        if m.group() not in EXEMPT
    }


def _claims() -> list[dict]:
    return json.loads(claims_lib.MAP.read_text())["readme_status"]


def test_every_status_number_is_mapped_to_an_artifact_value():
    body, numbers = _status_numbers()
    problems = claims_lib.unmapped_and_failing(body, numbers, _claims())
    assert not problems, "README Status numbers not backed by their mapped artifact cell:\n" + (
        "\n".join(problems)
    )


def test_a_number_from_the_wrong_cell_is_caught():
    # 0.946 appears in the card for several cells; the old gate accepted all of
    # them. Pointing a claim at another cell must fail even when the value
    # exists elsewhere.
    body, numbers = _status_numbers()
    claims = [dict(c) for c in _claims()]
    claim = next(c for c in claims if "[[0.893]] over 3000" in c["context"])
    claim["path"] = ["runs", "confirm", "fin_mid/oracle", "ll_bic", "coverage"]  # 0.948
    problems = claims_lib.unmapped_and_failing(body, numbers, claims)
    assert any("0.893" in p and "0.948" in p for p in problems), problems


def test_a_new_unmapped_number_is_caught():
    body, numbers = _status_numbers()
    pos = len(body) + len(" covers ")
    problems = claims_lib.unmapped_and_failing(
        body + " covers 0.777 here", {**numbers, pos: "0.777"}, _claims()
    )
    assert any("0.777" in p and "no claim" in p for p in problems), problems


def test_a_study_is_named_by_its_number_and_any_other_5_is_a_figure():
    found = re.findall(r"\d+", _mask("Study 5 and study 4, off by 5 actions in `s5-s0-021` (D92)"))
    assert found == ["5"]


# --- study 5 ----------------------------------------------------------------
# The block is built from the verdict and the analysis artifacts and nothing else
# (docs/PREREG-STUDY5.md; D92, corrected by D93).


@pytest.fixture(scope="module")
def s5() -> tuple[dict, dict]:
    return json.loads(S5_VERDICTS.read_text()), json.loads(S5_ANALYSIS.read_text())


@pytest.fixture(scope="module")
def block(s5) -> dict:
    return make_card.study5_block(*s5)


def _h3(x: float) -> float:
    return float(claims_lib.rounded(x, "0.000"))


def _same_coverage(got: dict, cell: dict) -> None:
    for key in ("covered", "runs", "unbounded", "bounded", "covered_bounded"):
        assert got[key] == cell[key], key
    assert got["wilson95"] == [_h3(cell["wilson_lo"]), _h3(cell["wilson_hi"])]
    assert got["median_abs_error"] == _h3(cell["median_abs_error"])
    assert got["mean_error"] == _h3(cell["mean_error"])


def test_the_committed_card_holds_the_study_5_block_the_artifacts_give(s5, block):
    v, _ = s5
    assert json.loads(CARD.read_text())["study5"] == json.loads(json.dumps(block))
    assert block["source"] == (
        "artifacts/robustness/study5-verdicts.json, artifacts/study5/confirm.json"
    )
    assert block["preregistration"] == "docs/PREREG-STUDY5.md"
    # The verdict was written from this analysis and no other.
    assert hashlib.sha256(S5_ANALYSIS.read_bytes()).hexdigest() == v["analysis"]["sha256"]


def test_study_5_runs_planned_in_the_test_and_set_aside_add_up(s5, block):
    v, a = s5
    assert set(block["runs"]) == {"s0", "s1", "s2"}
    for cond, runs in block["runs"].items():
        assert runs["planned"] == a["planned_by_condition"][cond]
        assert runs["in_test"] == sum(r["condition"] == cond for r in a["runs"])
        assert runs["set_aside"] == sum(r["condition"] == cond for r in a["runs_set_aside"])
        assert runs["set_aside_by_kind"] == v["runs_set_aside_by"][cond]
        assert runs["planned"] == runs["in_test"] + runs["set_aside"]
    assert block["runs_set_aside"] == v["runs_set_aside"]


def test_study_5_gates(s5, block):
    g = s5[0]["primary"]["gates"]
    got = block["gates"]
    assert got["min_runs"] == s5[0]["min_runs"]
    assert got["runs_passing_integrity"] == g["runs_passing_integrity"]
    assert (got["underpowered"], got["not_exercised"]) == (False, False)
    assert got["mean_share_in_no_log"] == _h3(g["mean_union_missed_share"])
    assert got["mean_telemetry_missed_share"] == {
        w: _h3(x) for w, x in g["mean_telemetry_missed_share"].items()
    }


def test_study_5_primary_on_s1_with_the_runs_not_in_the_test_beside_it(s5, block):
    v, a = s5
    prim = block["s1"]["ll_sa_hom"]
    _same_coverage(prim, v["primary"]["coverage"])
    assert prim["verdict"] == {
        "P1": "pass",
        "P2": "pass",
        "P1_reading": "pass",
        "P2_reading": "pass",
        "P1_clauses": {"coverage": "pass", "unbounded": "pass"},
        "P2_against": {"seq_gap_tail": "pass", "union_count": "pass"},
    }
    for baseline, cell in v["primary"]["paired"].items():
        got = prim["paired"][baseline]
        assert (got["wins"], got["ties"], got["runs"]) == (cell["wins"], cell["ties"], cell["runs"])
        assert got["lower_97_5"] == _h3(cell["lower_97_5"])
        assert got["passes"] is cell["passes"]
    low = block["s1"]["P1_not_in_test_as_uncovered"]
    reg = v["P1_not_in_test_as_uncovered"]
    assert (low["covered"], low["runs"]) == (reg["covered"], reg["runs"])
    assert (low["planned"], low["not_in_test"]) == (reg["planned"], reg["not_in_test"])
    assert low["wilson95"] == [_h3(x) for x in reg["wilson"]]
    # No criterion of its own; read by the registered rule its interval straddles the bar.
    assert low["wilson95"][0] < make_card.COVERAGE_BAR <= low["wilson95"][1]
    assert low["reading_by_the_registered_rule"] == "not shown"
    assert (
        block["s1"]["runs_where_the_union_is_exact"]
        == (v["primary"]["gates"]["runs_with_nothing_in_no_log"])
    )
    assert block["s1"]["runs_where_the_point_is_the_union"] == sum(
        r["estimates"]["ll_sa_hom"]["point"] == r["union_count"]
        for r in a["runs"]
        if r["condition"] == "s1"
    )
    assert block["s1"]["interval_above_truth"] == v["false_alarms"]["interval_above_truth"]


def test_the_coverage_reading_is_the_registered_rule(s5):
    v, _ = s5
    assert make_card.COVERAGE_BAR == study5_lib.COVERAGE_FLOOR
    for key in ("primary", "Q3_without_env"):
        cov = v[key]["coverage"]
        assert (
            make_card.coverage_reading(cov["wilson_lo"], cov["wilson_hi"])
            == v[key]["P1_clauses"]["coverage"]
        )
    assert make_card.coverage_reading(0.90, 0.95) == "pass"
    assert make_card.coverage_reading(0.85, 0.90) == "not shown"
    assert make_card.coverage_reading(0.80, 0.899) == "missed"


def test_study_5_without_the_end_state_floors_and_the_estimators_beside(s5, block):
    v, a = s5
    q3 = block["s1"]["without_end_state"]
    assert q3["estimator"] == "chapman_sa"
    _same_coverage(q3, v["Q3_without_env"]["coverage"])
    assert q3["verdict"]["P1_clauses"] == {"coverage": "missed", "unbounded": "pass"}
    assert q3["verdict"]["P2_against"] == {"seq_gap_tail": "missed", "union_count": "missed"}
    assert (
        q3["paired"]["seq_gap_tail"]["wins"]
        == v["Q3_without_env"]["paired"]["seq_gap_tail"]["wins"]
    )
    assert set(block["s1"]["floors"]) == {"union_count", "union_bound", "trust_trace"}
    for name, cell in v["floors"].items():
        _same_coverage(block["s1"]["floors"][name], cell)
    assert set(block["s1"]["beside"]) == {"seq_gap_tail", "strat_bic"}
    for name, got in block["s1"]["beside"].items():
        _same_coverage(got, a["by_condition"]["s1"]["estimators"][name])


def test_study_5_control_names_the_run_with_no_point_and_s2_is_descriptive(s5, block):
    v, a = s5
    ctl = block["control"]
    assert ctl["condition"] == "s0"
    _same_coverage(ctl["ll_sa_hom"], v["control"]["coverage"])
    assert ctl["ll_sa_hom"]["runs_with_a_point"] == v["control"]["coverage"]["runs_with_a_point"]
    assert ctl["interval_above_truth"] == v["control"]["false_alarms"]["interval_above_truth"]
    (lost,) = ctl["runs_with_no_point"]
    assert lost["run"] == "s5-s0-021"
    assert lost["actions"] == lost["union_count"] and lost["share_in_no_log"] == 0.0
    assert lost["estimators_with_no_point"] == ["ll_sa_hom", "ll_sa_hom_sig"]
    assert lost["counts_as_covering"] is True
    assert ctl["defect"]["status"] == "open"
    s2 = block["s2"]
    assert "no criterion" in s2["note"] and "no prediction" in s2["note"]
    _same_coverage(s2["ll_sa_hom"], a["by_condition"]["s2"]["estimators"]["ll_sa_hom"])
    assert "verdict" not in s2


def test_study_5_predictions_say_which_held(block):
    held = {k: p["held"] for k, p in block["predictions"].items()}
    assert held == {
        "s1_exercised": True,
        "P1_coverage": False,  # predicted not met; the clause passes on the runs left
        "P1_unbounded": True,
        "P2_seq_gap_tail": True,
        "P2_union_count": True,
        "Q3": True,
        "control": True,
        "s2": None,  # none was made
    }
    assert "not shown" in block["predictions"]["P1_coverage"]["note"]
    assert all(p["source"] == "docs/PREREG-STUDY5.md" for p in block["predictions"].values())


def test_study_5_predictions_follow_the_verdict_and_are_void_when_a_gate_fires(s5):
    v, a = s5
    short = copy.deepcopy(v)
    short["primary"]["P1_clauses"]["coverage"] = "not shown"
    assert make_card.study5_block(short, a)["predictions"]["P1_coverage"]["held"] is True
    gated = copy.deepcopy(v)
    for key in ("primary", "Q3_without_env"):
        gated[key]["gates"]["underpowered"] = True
        gated[key] |= {"P1": "underpowered", "P2": "underpowered", "P1_reading": "underpowered"}
        gated[key] |= {"P2_reading": "underpowered", "P1_clauses": None, "P2_against": None}
    void = make_card.study5_block(gated, a)["predictions"]
    for key in ("P1_coverage", "P1_unbounded", "P2_seq_gap_tail", "P2_union_count", "Q3"):
        assert void[key]["held"] is None, key
    assert void["control"]["held"] is True


def test_study_5_block_carries_no_table_by_band_pool_or_outage_length(block):
    # The registration bars any claim from them, so the card, which is what the README
    # may quote, does not hold them.
    text = json.dumps(block)
    for key in ("by_outage_s", "by_lost_by_both", "by_pool", "lost_by_both"):
        assert f'"{key}"' not in text


def _study5_status_text() -> tuple[str, str]:
    status = README.read_text().split("## Status", 1)[1].split("## Quickstart", 1)[0]
    row = next(x for x in status.splitlines() if x.startswith("|") and "PREREG-STUDY5" in x)
    return row, row + status.split("\nStudy 5 ", 1)[1].split("\nTerms:", 1)[0]


def test_the_study_5_pass_is_never_quoted_without_the_runs_set_aside():
    row, _ = _study5_status_text()
    start = row.index("269 of the 277")
    rest = row[start:]
    # The same sentence or the next.
    assert rest.count(". ") >= 1
    following = ". ".join(rest.split(". ")[:2])
    for needed in ("23 of the 300 planned", "set aside on integrity", "269 of 300", "not shown"):
        assert needed in following, needed
    assert "was wrong" in row


def test_the_study_5_text_leans_no_reading_and_states_the_defect_as_open():
    _, text = _study5_status_text()
    for barred in (r"\bnear", "consistent with", "close to nominal", "calibrat"):
        assert not re.search(barred, text, re.I), barred
    assert "open defect" in text.lower() and "s5-s0-021" in text
