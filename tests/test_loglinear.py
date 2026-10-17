"""The estimator against answers that are known without it.

There is no R on the build machine, so Rcapture is not the reference yet
(docs/DECISIONS.md D1). These are stronger anyway where they apply: closed
forms that any correct log-linear fit must reproduce exactly, and statsmodels
for the IRLS itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lacuna.estimators import loglinear as ll


def test_two_witness_independence_is_lincoln_petersen():
    # h=1 seen by W0 only, h=2 by W1 only, h=3 by both.
    counts = np.array([0, 40, 25, 60])
    n1, n2, m = 40 + 60, 25 + 60, 60
    est = ll.independence(counts)
    assert est.point == pytest.approx(n1 * n2 / m, rel=1e-9)


def test_three_witness_all_pairs_matches_closed_form():
    # With all three pairs the model is saturated on the 7 observed cells and
    # the missing cell is z111 z100 z010 z001 / (z110 z101 z011).
    counts = np.array([0, 31, 22, 40, 17, 28, 19, 55])
    z100, z010, z110, z001, z101, z011, z111 = (counts[h] for h in (1, 2, 3, 4, 5, 6, 7))
    want = z111 * z100 * z010 * z001 / (z110 * z101 * z011)
    fit = ll.fit_conditional(counts, ll.all_pairs(3))
    assert fit.z0 == pytest.approx(want, rel=1e-8)


def test_model_space_excludes_unidentifiable_models():
    assert ll.model_space(2) == [()]
    assert len(ll.model_space(3)) == 8
    assert len(ll.model_space(4)) == 64


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_irls_matches_statsmodels(seed):
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(seed)
    counts = np.concatenate([[0], rng.integers(0, 60, size=15)])
    pairs = ((0, 1), (2, 3), (1, 3))
    x = ll.design(4, pairs, range(1, 16))
    y = counts[1:].astype(float)
    ours = ll.fit_poisson(x, y)
    ref = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    assert ours.converged
    assert ours.loglik == pytest.approx(ref.llf, rel=1e-8)
    np.testing.assert_allclose(ours.mu, ref.mu, rtol=1e-6)


@pytest.mark.parametrize("counts", [[0, 40, 25, 60], [0, 13, 7, 5], [0, 100, 80, 30]])
def test_two_witness_profile_is_the_product_binomial_likelihood(counts):
    # Independence with K=2 profiles to a closed form: capture rates n_j / N.
    counts = np.array(counts)
    n1, n2, n = counts[1] + counts[3], counts[2] + counts[3], counts[1:].sum()

    def ref(big_n):
        out = math.lgamma(big_n + 1) - math.lgamma(big_n - n + 1)
        for nj in (n1, n2):
            q = nj / big_n
            out += nj * math.log(q) + (big_n - nj) * math.log(1 - q)
        return out

    prof = ll.Profile(counts, ())
    for big_n in range(n + 1, 3 * n):
        assert prof(big_n) - prof(n + 1) == pytest.approx(ref(big_n) - ref(n + 1), abs=1e-8)


def test_profile_interval_brackets_point_and_unconditional_mle():
    counts = np.array([0, 50, 30, 45, 20, 35, 15, 60])
    for pairs in ll.model_space(3):
        est = ll._single(counts, pairs, "x")
        if not est.bounded:
            continue
        mle, _ = ll.Profile(counts, pairs).maximise(est.point)
        assert est.lo <= min(mle, est.point) <= max(mle, est.point) <= est.hi


def test_no_overlap_is_unbounded_not_a_number():
    # Two witnesses that never saw the same action: Lincoln-Petersen divides by 0.
    counts = np.array([0, 40, 25, 0])
    est = ll.independence(counts)
    assert est.hi == math.inf
    assert est.covers(10_000)


def test_maximise_stops_at_the_cap_when_likelihood_keeps_rising():
    counts = np.array([0, 40, 25, 0])
    prof = ll.Profile(counts, ())
    mle, _ = prof.maximise(70)
    assert mle == ll.CAP_RATIO * 65
    assert len(prof._cache) < 200  # exponential search, not a walk to the cap


def test_average_without_overlap_is_unbounded():
    est = ll.averaged(np.array([0, 40, 25, 0]))
    assert est.hi == math.inf
    assert est.detail["unbounded_weight"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "obs",
    [
        # From the first sweep (l0_auditor_copies_trace, true N=200). A warm
        # start underflowed mu to 0 and NaN reached the mixture: points 280, 3930.
        [0, 5, 1, 11, 1, 8, 2, 13, 0, 20, 0, 38, 0, 33, 0, 67],
        [0, 6, 1, 11, 1, 4, 1, 13, 0, 16, 0, 56, 0, 36, 0, 54],
    ],
)
def test_average_survives_sparse_k4_tables(obs):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = ll.averaged(np.array(obs))
    assert est.lo <= 200 <= est.hi
    assert est.hi - est.lo < 20


def test_average_weights_sum_to_one_and_interval_is_ordered():
    counts = np.array([0, 50, 30, 45, 20, 35, 15, 60])
    est = ll.averaged(counts)
    assert sum(est.detail["weights"]) == pytest.approx(1.0)
    assert est.n_observed <= est.lo <= est.point <= est.hi


def test_long_tailed_model_keeps_its_shape_in_the_average():
    # Review finding: a model peaking at 38 with a 95% interval (36, 69) was
    # sent wholly to +inf because its 1e-5 interval reached the cap.
    est = ll.averaged(np.array([0, 11, 1, 2, 9, 6, 6, 1]))
    assert est.point < 60
    assert 0 <= est.detail["unbounded_weight"] < 1


def test_profile_refuses_an_unconverged_fit(monkeypatch):
    real = ll._irls

    def stalled(x, y, off, beta, max_iter):
        return real(x, y, off, beta, 1)

    monkeypatch.setattr(ll, "_irls", stalled)
    with pytest.raises(ll.ConvergenceError):
        ll.Profile(np.array([0, 50, 30, 45, 20, 35, 15, 60]), ((0, 1),))(300)
    assert not ll.fit_conditional(np.array([0, 50, 30, 45, 20, 35, 15, 60]), ((0, 1),)).converged
