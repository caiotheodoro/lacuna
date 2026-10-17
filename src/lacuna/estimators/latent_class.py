"""Bayesian latent-class capture-recapture (Manrique-Vallier 2016, "LCMCR").

Log-linear models handle dependence one named interaction at a time. They do
not handle heterogeneity, where some actions are hard for every witness (a
write that is reverted before the diff runs, egress that looks like a read).
The first sweep showed exactly this: `ll_bic` covers 0.30 on
l0_heterogeneous. Latent classes are the textbook answer: each action belongs
to an unobserved class, and within a class the witnesses are independent.
Marginally they are positively dependent, which is the heterogeneity.

The model, with a truncated stick-breaking prior over K* classes:

    z_i ~ Cat(pi),  pi ~ SB(alpha),  alpha ~ Gamma(a, b)
    x_ij | z_i = c ~ Bern(lambda_jc),  lambda_jc ~ Beta(1, 1)
    p(N) ∝ 1/N

The Gibbs sampler works on the 2**K cell counts, not on individual actions,
so its cost does not grow with N. The never-seen count is augmented each
sweep: under the 1/N prior it is NegBin(n, 1 - p0), where p0 is the current
probability that no witness sees an action.

Identifiability warning, kept here so nobody reads the numbers without it:
with K=3 witnesses, latent-class models with more than one class are not
identified from the observed cells (Aleshin-Guendel 2020). The posterior is
then prior-driven in the directions the data cannot see. That shows up as
width, and the coverage sweep measures whether the width is honest.
"""

from __future__ import annotations

import math

import numpy as np

from lacuna.estimators import Estimate
from lacuna.estimators.loglinear import CAP_RATIO
from lacuna.histories import bits, n_witnesses, observed_total

EPS = 1e-12


def sample_posterior(
    counts: np.ndarray,
    k_max: int = 10,
    burn: int = 1000,
    draws: int = 2000,
    a_alpha: float = 0.25,
    b_alpha: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """Posterior draws of N. The rng is keyed on the table, so equal tables give equal draws."""
    k = n_witnesses(counts)
    x = np.array([bits(h, k) for h in range(2**k)])  # cells x witnesses
    y = np.asarray(counts, dtype=np.int64).copy()
    y[0] = 0
    n = int(y.sum())
    rng = np.random.default_rng([seed, *y.tolist()])

    lam = rng.beta(1.0, 1.0, size=(k_max, k))  # classes x witnesses
    pi = np.full(k_max, 1.0 / k_max)
    alpha = 1.0
    n0 = 0
    out = np.empty(draws, dtype=np.int64)

    for it in range(burn + draws):
        # P(cell | class), then each cell's count split across classes.
        logp = x @ np.log(lam).T + (1 - x) @ np.log1p(-lam).T + np.log(pi)
        w = np.exp(logp - logp.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        y[0] = n0
        m = rng.multinomial(y, w)  # cells x classes
        nk = m.sum(axis=0)

        seen = x.T @ m  # witnesses x classes
        lam = np.clip(rng.beta(1.0 + seen.T, 1.0 + (nk - seen).T), EPS, 1 - EPS)

        later = np.cumsum(nk[::-1])[::-1] - nk  # actions in classes after c
        v = np.clip(rng.beta(1.0 + nk[:-1], alpha + later[:-1]), EPS, 1 - EPS)
        left = np.concatenate([[1.0], np.cumprod(1 - v)])
        pi = np.concatenate([v, [1.0]]) * left
        alpha = rng.gamma(a_alpha + k_max - 1, 1.0 / (b_alpha - np.log1p(-v).sum()))

        p0 = float(np.sum(pi * np.prod(1 - lam, axis=1)))
        n0 = int(rng.negative_binomial(n, max(1 - p0, EPS)))
        if it >= burn:
            out[it - burn] = n + n0
    return out


def latent_class(counts: np.ndarray, level: float = 0.95, **kw) -> Estimate:
    n = observed_total(counts)
    if n == 0:
        return Estimate("lc", 0, math.inf, 0.0, math.inf, {"unbounded": True})
    draws = sample_posterior(counts, **kw)
    tail = (1 - level) / 2
    lo, med, hi = np.quantile(draws, [tail, 0.5, 1 - tail])
    detail = {"draws": len(draws)}
    if hi > CAP_RATIO * n:
        return Estimate("lc", n, float(med), float(lo), math.inf, {**detail, "unbounded": True})
    return Estimate("lc", n, float(med), float(lo), float(hi), detail)
