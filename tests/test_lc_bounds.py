from __future__ import annotations

import numpy as np
import pytest

from lacuna.estimators import bounds, latent_class, loglinear
from lacuna.sim import SCENARIOS


def _observed(name: str, seed: int) -> tuple[int, np.ndarray]:
    full = SCENARIOS[name].sample(np.random.default_rng(seed))
    obs = full.copy()
    obs[0] = 0
    return int(full.sum()), obs


def test_one_class_is_the_independence_model():
    _, obs = _observed("l0_independent", 1)
    one = latent_class.latent_class(obs, k_max=1)
    ind = loglinear.independence(obs)
    # Bayesian (Beta(1,1), 1/N) against MLE: same model, a few actions apart.
    assert one.point == pytest.approx(ind.point, abs=3)


def test_latent_class_is_deterministic_per_table():
    _, obs = _observed("l0_heterogeneous", 2)
    a = latent_class.sample_posterior(obs, burn=200, draws=300)
    b = latent_class.sample_posterior(obs, burn=200, draws=300)
    np.testing.assert_array_equal(a, b)


def test_negative_binomial_step_matches_the_one_over_n_posterior():
    # With one class and capture rates fixed, n0 | rest is NegBin(n, 1 - p0).
    # Check the parameterisation numpy uses against the closed form mean.
    rng = np.random.default_rng(0)
    n, p0 = 150, 0.2
    draws = rng.negative_binomial(n, 1 - p0, size=200_000)
    assert draws.mean() == pytest.approx(n * p0 / (1 - p0), rel=0.01)


def test_anchor_table_collapses_everyone_else():
    counts = np.array([0, 5, 7, 11, 13, 17, 19, 23])  # bits: trace, proxy, env
    t = bounds.anchor_table(counts, anchor=2)
    # env only = h4; rest only = h1,h2,h3; both = h5,h6,h7.
    assert t.tolist() == [0, 13, 5 + 7 + 11, 17 + 19 + 23]


def test_gamma_one_is_lincoln_petersen_of_anchor_against_rest():
    counts = np.array([0, 5, 7, 11, 13, 17, 19, 23])
    est = bounds.anchor_bounds(counts, gamma=1.0)
    t = bounds.anchor_table(counts)
    assert est.point == pytest.approx(t[1:].sum() + t[1] * t[2] / t[3])
    assert est.lo <= est.point <= est.hi


@pytest.mark.parametrize("gamma", [1.0, 2.5, 6.0])
def test_offset_profile_peaks_at_the_closed_form(gamma):
    t = np.array([0, 40, 300, 450])
    top = t[1:].sum() + gamma * t[1] * t[2] / t[3]
    mle, _ = bounds._psi_profile(t, gamma).maximise(top)
    assert abs(mle - top) < 3


def test_upper_end_grows_with_gamma_and_lower_end_does_not_move():
    _, obs = _observed("l2_env_blind", 3)
    ests = [bounds.anchor_bounds(obs, gamma=g) for g in (1, 2, 4, 8)]
    assert len({e.lo for e in ests}) == 1
    his = [e.hi for e in ests]
    assert his == sorted(his) and his[0] < his[-1]


def test_gamma_below_one_is_refused():
    with pytest.raises(ValueError):
        bounds.anchor_bounds(np.array([0, 5, 7, 11, 13, 17, 19, 23]), gamma=0.5)


def test_anchor_out_of_range_is_an_error_not_a_finding():
    with pytest.raises(ValueError):
        bounds.anchor_bounds(np.array([0, 30, 40, 20]))
