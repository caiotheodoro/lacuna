"""The diff behind `make repro-headline`, on small inputs (the make target itself replays
the pool and takes minutes)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repro_headline", ROOT / "scripts/repro_headline.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

OLD = {"configs": {"c/signature": {"estimators": {"ll_bic": {"coverage": 0.94, "bias": None}}}}}


def test_equal_artifacts_match():
    assert rh.diff_numbers(OLD, OLD) == []


def test_a_changed_coverage_is_reported_with_its_path():
    new = {"configs": {"c/signature": {"estimators": {"ll_bic": {"coverage": 0.93, "bias": None}}}}}
    (bad,) = rh.diff_numbers(new, OLD)
    assert "ll_bic/coverage" in bad and rh.estimator_of(bad) == "ll_bic"


def test_a_missing_field_is_a_mismatch_and_a_new_one_is_not():
    est = {"coverage": 0.94, "bias": None, "extra": 1}
    new = {"configs": {"c/signature": {"estimators": {"ll_bic": est}}}}
    assert rh.diff_numbers(new, OLD) == []
    del est["coverage"]
    assert any("coverage" in m for m in rh.diff_numbers(new, OLD))


def _path(cfg, lk, est, field):
    return f"/configs/{cfg}/{lk}/estimators/{est}/{field}: 1 != 2"


def test_exactly_the_eight_known_cells_are_split_off():
    assert len(rh.DRIFT) == 8 and {c for c, *_ in rh.DRIFT} == {"fin_mid_rho100"}
    known = [_path(c, lk, e, f) for c, lk, e, f in sorted(rh.DRIFT)]
    failures, drift = rh.split_drift(known)
    assert failures == [] and drift == known


def test_the_same_estimators_elsewhere_or_in_other_fields_still_fail():
    bad = [
        _path("fin_mid", "oracle", "ll_average", "bias"),  # other config
        _path("fin_mid_rho100", "oracle", "ll_average", "coverage"),  # other field
        _path("fin_mid_rho100", "signature", "strat_lc_gated", "bias"),  # other field
        _path("fin_mid_rho100", "oracle", "strat_lc_gated", "detects_gap"),  # other linkage
        _path("fin_mid_rho100", "oracle", "ll_bic", "coverage"),
        "/configs/fin_mid_rho100/oracle/estimators/ll_average: missing from the rerun",
        "hypotheses/hypotheses/H1 recovery/passed: False != True",
    ]
    failures, drift = rh.split_drift(bad)
    assert drift == [] and failures == bad
