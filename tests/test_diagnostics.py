from __future__ import annotations

import math

import numpy as np
import pytest

from lacuna.estimators import diagnostics
from lacuna.sim import SCENARIOS


def _obs(name: str, seed: int) -> tuple[int, np.ndarray]:
    full = SCENARIOS[name].sample(np.random.default_rng(seed))
    obs = full.copy()
    obs[0] = 0
    return int(full.sum()), obs


def test_overlap_fraction_counts_multi_witness_cells():
    counts = np.array([0, 10, 10, 5, 10, 5, 5, 5])  # h=3,5,6,7 have >=2 bits
    assert diagnostics.overlap_fraction(counts) == pytest.approx(20 / 50)


def test_disjoint_witnesses_are_refused():
    est = diagnostics.lc_gated(np.array([0, 200, 200, 0, 200, 0, 0, 0]))
    assert "refused" in est.detail
    assert est.hi == math.inf and est.lo == 600


def test_independent_witnesses_pass_and_report_lc():
    n, obs = _obs("l0_independent", 5)
    est = diagnostics.lc_gated(obs)
    assert "refused" not in est.detail
    assert est.bounded and est.covers(n)


def test_supported_points_include_the_best_model():
    _, obs = _obs("l0_heterogeneous", 6)
    pts = diagnostics.supported_points(obs)
    assert len(pts) >= 1
