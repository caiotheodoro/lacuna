"""The R cross-check (docs/DECISIONS.md D1): our log-linear fits against Rcapture.

The committed artifact (artifacts/robustness/r-crosscheck.json) is produced by
scripts/r_crosscheck.py inside a pinned rocker/r-ver image. The first block of
tests reads it and needs no R and no Docker. The live test re-runs a small set
of tables through Docker and is skipped when no Docker daemon is reachable.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import r_crosscheck as rc  # noqa: E402

from lacuna.estimators import loglinear as ll  # noqa: E402

ARTIFACT = ROOT / "artifacts" / "robustness" / "r-crosscheck.json"


@pytest.fixture(scope="module")
def artifact():
    return json.loads(ARTIFACT.read_text())


# --- committed artifact ------------------------------------------------------


def test_point_estimates_agree_with_rcapture_to_tolerance(artifact):
    g = artifact["summary"]["gated"]
    assert g["max_rel_point_vs_rcapture"] < rc.TOL_REL
    assert g["closedp_t_Mt_max_rel"] < rc.TOL_REL


def test_profile_interval_agrees_with_independent_r_reference(artifact):
    # Same profile likelihood, evaluated by stats::glm.fit and root-found at 1e-10.
    g = artifact["summary"]["gated"]
    assert g["max_rel_point_vs_r_glm"] < rc.TOL_REL
    assert g["max_rel_profile_vs_r_glm"] < rc.TOL_REL


def test_bic_ranking_agrees(artifact):
    s = artifact["summary"]
    assert s["gated"]["max_abs_delta_bic"] < 1e-3
    assert s["bic_pick_agreement"] == s["datasets"]


def test_rcapture_interval_gap_is_its_stirling_term(artifact):
    # closedpCI.t's endpoints differ from ours by up to several units. Its own
    # (N, loglik) curve equals our profile plus the Stirling correction in its
    # source, to rounding; it does not equal our profile with the exact term.
    r = artifact["summary"]["rcapture_profile_ci"]
    assert r["stirling_max_abs_residual_vs_rcapture_ct"] < 1e-5
    assert (
        r["stirling_max_abs_residual_vs_exact_ct"]
        > 100 * r["stirling_max_abs_residual_vs_rcapture_ct"]
    )


def test_artifact_is_not_vacuous(artifact):
    s = artifact["summary"]
    assert s["regular_fits"] >= 100
    assert artifact["versions"]["Rcapture"] and artifact["versions"]["dga"] != "absent"
    assert {x["source"] for x in artifact["datasets"]} >= {"sim", "confirm", "edge"}
    assert len(artifact["dga"]) == s["datasets"]
    for row in s["max_diff_by_model"].values():
        for field in ("point", "x_lo", "x_hi"):
            assert row[field] is not None


def test_every_boundary_fit_is_listed_not_dropped(artifact):
    fits = sum(len(x["models"]) for x in artifact["datasets"])
    assert fits == artifact["summary"]["regular_fits"] + len(artifact["summary"]["boundary_fits"])


# --- helpers, no Docker ------------------------------------------------------


def test_rcapture_names():
    names = [rc.rcapture_name(m) for m in ll.model_space(3)]
    assert names == [
        "[1,2,3]",
        "[12,3]",
        "[13,2]",
        "[23,1]",
        "[12,13]",
        "[12,23]",
        "[13,23]",
        "[12,13,23]",
    ]


def test_continuous_profile_independence_matches_closed_form():
    # Independence has a closed-form profile: capture rates n_j / N.
    counts = np.array([0, 22, 3, 24, 6, 16, 9, 108])
    n = int(counts.sum())
    nj = [int(sum(c for h, c in enumerate(counts) if h and h >> j & 1)) for j in range(3)]

    def closed(big_n):
        return (
            math.lgamma(big_n + 1)
            - math.lgamma(big_n - n + 1)
            + sum(x * math.log(x / big_n) + (big_n - x) * math.log(1 - x / big_n) for x in nj)
        )

    prof = ll.Profile(counts, ())
    for big_n in (188.4, 189.0, 190.5, 195.0):
        assert prof(big_n) == pytest.approx(closed(big_n), abs=1e-9)
    got = rc.continuous_profile(counts, ())
    grid = np.linspace(n, n + 10, 20001)
    vals = np.array([closed(x) for x in grid])
    inside = grid[vals >= vals.max() - ll.CHI2_1_95 / 2]
    assert got["prof"] == pytest.approx(grid[vals.argmax()], abs=1e-3)
    assert got["lo"] == pytest.approx(inside.min(), abs=1e-3)
    assert got["hi"] == pytest.approx(inside.max(), abs=1e-3)


def test_rcapture_correction_term_is_exact_for_small_tables_and_stirling_above():
    assert rc.ct_rcapture(60.0, 50) == pytest.approx(rc.ct_exact(60.0, 50))
    # Large N, large n0: close. Small n0: far.
    assert abs(rc.ct_rcapture(1000.0, 500) - rc.ct_exact(1000.0, 500)) < 1e-3
    assert abs(rc.ct_rcapture(188.01, 188) - rc.ct_exact(188.01, 188)) > 1.0


# --- live --------------------------------------------------------------------


@pytest.mark.skipif(
    not rc.have_docker(), reason="no Docker daemon; the committed artifact is checked above"
)
def test_live_rcapture_agrees_on_a_few_tables():
    datasets = [
        {"name": "t/closed-form", "source": "t", "counts": [0, 31, 22, 40, 17, 28, 19, 55]},
        {"name": "t/het", "source": "t", "counts": [0, 22, 3, 24, 6, 16, 9, 108]},
        {"name": "t/mid", "source": "t", "counts": [0, 58, 61, 56, 134, 146, 160, 140]},
    ]
    result = rc.run(datasets, with_dga=False)
    g = result["summary"]["gated"]
    assert result["summary"]["regular_fits"] >= 20
    assert g["max_rel_point_vs_rcapture"] < rc.TOL_REL
    assert g["max_rel_profile_vs_r_glm"] < rc.TOL_REL
    assert g["max_abs_delta_bic"] < 1e-3
    assert result["versions"]["Rcapture"] == "1.4.4"
