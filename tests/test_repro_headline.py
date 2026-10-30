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


def test_known_drift_is_split_off_but_ll_bic_never_is():
    bad = [
        "/configs/a/estimators/ll_average/bias: 1 != 2",
        "/configs/a/estimators/strat_lc_gated/detects_gap: 0 != 0.01",
        "/configs/a/estimators/ll_bic/coverage: 0.9 != 0.94",
        "hypotheses/hypotheses/H1 recovery/passed: False != True",
    ]
    failures, drift = rh.split_drift(bad)
    assert len(drift) == 2
    assert failures == bad[2:]
