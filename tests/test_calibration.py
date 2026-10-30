"""End-to-end calibration: does the shipped interval cover at its nominal rate?

The unit tests check pieces of the estimator. These run the whole path a
published coverage number takes (simulator -> `coverage.run_once` -> estimator ->
`covers` -> `summarise`) on the L0 independent configuration, the one every
log-linear model is designed for, and check the result against nominal 95%
given its Monte Carlo error. A change that quietly breaks the interval (a
wrong profile cutoff, an off-by-one in the observed cells) shows up here and
nowhere in a test of a single function.

Only `ll_bic` and `union_count` are run: the latent-class estimator in the
full sweep takes minutes and is not under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lacuna import coverage
from lacuna.estimators import baselines, loglinear

# The first scenario in scripts/sim_coverage.py: N=200 actions, three independent logs.
SCENARIO = "l0_independent"
REPS = 200
SEED = 2026
NOMINAL = 0.95
LL = "ll_bic@trace+proxy+env"
UNION = "union_count@trace+proxy+env"


@pytest.fixture(scope="module")
def summary() -> dict:
    """coverage.summarise() per estimator over REPS replicates, via the real run_once."""
    saved = (coverage.PER_SET, coverage.ANCHOR_GAMMAS)
    coverage.PER_SET = {"ll_bic": loglinear.best_bic, "union_count": baselines.union_count}
    coverage.ANCHOR_GAMMAS = ()
    try:
        s = coverage.SCENARIOS[SCENARIO]
        rng = np.random.default_rng([SEED, 0])
        rows: dict[str, list] = {}
        for _ in range(REPS):
            n_true, ests = coverage.run_once(s, rng)
            for name, e in ests.items():
                rows.setdefault(name, []).append((n_true, e))
    finally:
        coverage.PER_SET, coverage.ANCHOR_GAMMAS = saved
    out = {k: coverage.summarise(v) for k, v in rows.items()}
    return out


def test_ll_bic_covers_near_nominal_at_l0_independent(summary):
    c = summary[LL]["coverage"]
    mcse = math.sqrt(NOMINAL * (1 - NOMINAL) / REPS)
    # Two-sided, and both sides can fail: not below 0.85, nor more than four Monte Carlo errors
    # under nominal, and not above 0.99 (nominal + 2.6 MCSE), which an interval wide enough to
    # cover every table (coverage 1.0) would be. "+- 4 MCSE" alone cannot catch that:
    # nominal + 4 MCSE is above 1.
    lo = max(0.85, NOMINAL - 4 * mcse)
    hi = 0.99
    assert lo <= c <= hi, f"ll_bic covers {c} at {SCENARIO}, outside [{lo:.3f}, {hi}]"
    assert summary[LL]["unbounded_rate"] == 0.0
    # An interval that always covers by being huge would also pass; require it informative.
    assert summary[LL]["mean_width"] < 0.25 * 200


def test_union_count_does_not_cover_when_something_is_hidden(summary):
    # The floor never exceeds N and the hidden mass at this configuration averages ~3.7,
    # so its "interval" misses N in nearly every replicate. Coverage near 0 is the
    # correct negative control: a gate that passes everything would pass this too.
    assert summary[UNION]["coverage"] <= 0.10
    assert summary[UNION]["bias"] < 0
