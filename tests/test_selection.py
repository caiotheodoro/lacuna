"""The selection-aware bootstrap interval (src/lacuna/estimators/selection.py)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lacuna.estimators import loglinear, selection

TABLE = np.array([0, 31, 22, 40, 17, 28, 19, 55])
HET = np.array([0, 22, 3, 24, 6, 16, 9, 108])


def test_select_point_is_the_bic_winner_of_best_bic():
    pairs, point = selection.select_point(TABLE)
    ref = loglinear.best_bic(TABLE)
    assert [list(p) for p in pairs] == ref.detail["pairs"]
    assert point == pytest.approx(ref.point, rel=1e-9)


def test_fitted_full_table_has_the_missing_cell_and_sums_to_nhat():
    pairs, point = selection.select_point(TABLE)
    mu = selection.fitted_full_table(TABLE, pairs)
    assert mu[0] == pytest.approx(point - TABLE.sum(), rel=1e-8)
    assert mu.sum() == pytest.approx(point, rel=1e-8)


def test_interval_is_finite_and_contains_the_point():
    for fn_name in ("ll_bic_boot", "ll_bic_boot_pct"):
        e = selection.bootstrap_bic(TABLE, reps=100, seed=1)[fn_name]
        assert math.isfinite(e.lo) and math.isfinite(e.hi)
        assert e.lo >= TABLE.sum()
        assert e.lo <= e.point <= e.hi


def test_deterministic_given_seed_and_sensitive_to_it():
    a = selection.bootstrap_bic(TABLE, reps=60, seed=7)["ll_bic_boot"]
    b = selection.bootstrap_bic(TABLE, reps=60, seed=7)["ll_bic_boot"]
    c = selection.bootstrap_bic(TABLE, reps=60, seed=8)["ll_bic_boot"]
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert (a.lo, a.hi) != (c.lo, c.hi)


def test_selection_is_redone_inside_each_resample():
    # On a heterogeneous table the winner changes across resamples; the draws
    # must therefore be spread over more than one model's N-hat.
    pairs, point = selection.select_point(HET)
    mu = selection.fitted_full_table(HET, pairs)
    rng = np.random.default_rng(3)
    winners = set()
    for _ in range(60):
        tab = rng.multinomial(int(round(mu.sum())), mu / mu.sum())
        tab[0] = 0
        winners.add(selection.select_point(tab)[0])
    assert len(winners) > 1


def test_unbounded_winner_gives_unbounded_interval():
    # Two witnesses never overlap with the third: z0 diverges.
    t = np.array([0, 30, 25, 0, 20, 0, 0, 0])
    e = selection.bootstrap_bic(t, reps=20, seed=0)["ll_bic_boot"]
    assert e.lo == t.sum() and not e.bounded and math.isinf(e.point)


def test_estimate_wrapper_matches_dict_entry():
    e = selection.bic_boot(TABLE, reps=40, seed=5)
    d = selection.bootstrap_bic(TABLE, reps=40, seed=5)["ll_bic_boot"]
    assert (e.point, e.lo, e.hi) == (d.point, d.lo, d.hi)


def test_saturated_world_names_and_validity():
    d = selection.bootstrap_bic(TABLE, reps=60, seed=2, world="saturated")
    assert set(d) == {"ll_bic_boot_sat", "ll_bic_boot_sat_pct"}
    assert math.isfinite(d["ll_bic_boot_sat"].hi) and d["ll_bic_boot_sat"].lo >= TABLE.sum()
    with pytest.raises(ValueError):
        selection.bootstrap_bic(TABLE, reps=5, world="nope")


SELECTION_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts/robustness/selection-coverage.json"
)


def _est(art, cfg, name):
    return art["configs"][cfg]["estimators"][name]


def test_committed_artifact_reports_coverage_with_mcse():
    art = json.loads(SELECTION_ARTIFACT.read_text())
    assert art["reps"] >= 200 and art["boot_reps"] >= 100
    assert any(k.startswith("confirm/fin_mid") for k in art["configs"])
    for cfg in art["configs"].values():
        for m in ("ll_bic", "ll_average", "ll_bic_boot", "ll_bic_boot_pct"):
            e = cfg["estimators"][m]
            assert 0 <= e["coverage"] <= 1 and e["coverage_mcse"] >= 0 and "detects_gap" in e


def test_committed_artifact_pins_the_numbers_the_docs_quote():
    art = json.loads(SELECTION_ARTIFACT.read_text())
    conf = "confirm/fin_mid/oracle"
    assert _est(art, conf, "ll_bic")["coverage"] == pytest.approx(0.938, abs=5e-4)
    assert _est(art, conf, "ll_bic_boot")["coverage"] == pytest.approx(0.9425, abs=1e-9)
    assert _est(art, conf, "ll_bic")["detects_gap"] == 1.0
    # Selection-aware does not repair a wrong model class: coverage stays far below nominal.
    het = "sim/l0_heterogeneous"
    assert _est(art, het, "ll_bic")["coverage"] == pytest.approx(0.3075, abs=1e-9)
    assert _est(art, het, "ll_bic_boot")["coverage"] < 0.4
    # and the bootstrap interval gives up gap detection.
    for cfg in ("sim/l0_independent", het, "sim/l0_siloed", "sim/l1_hides_from_trace"):
        assert (
            _est(art, cfg, "ll_bic_boot")["detects_gap"] < _est(art, cfg, "ll_bic")["detects_gap"]
        )


def test_the_script_default_reproduces_the_committed_artifact(tmp_path):
    # The artifact was run with 400 outer reps; the script's default must be that.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import selection_bootstrap as sb

    art = json.loads(SELECTION_ARTIFACT.read_text())
    assert art["reps"] == 400 and sb.DEFAULT_REPS == 400
    assert art["boot_reps"] == sb.DEFAULT_BOOT_REPS == 300
    # A new artifact records how it was produced.
    out = tmp_path / "smoke.json"
    argv = ["--reps", "2", "--boot-reps", "10", "--sim", "l0_independent", "--no-confirm"]
    argv += ["--workers", "1", "--out", str(out)]
    r = subprocess.run(
        [sys.executable, sb.__file__, *argv],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(sb.__file__).parents[1] / "src")},
    )
    assert r.returncode == 0, r.stderr[-300:]
    rec = json.loads(out.read_text())
    assert rec["argv"] == argv and rec["reps"] == 2
    assert rec["parameters"] == {
        "workers": 1,
        "sim_configs": ["l0_independent"],
        "confirm": False,
    }
