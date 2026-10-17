"""Log-linear multiple-systems estimation (Fienberg 1972; Cormack 1989).

The observed 2**K - 1 cells are fitted with a Poisson log-linear model: main
effects per witness plus a chosen set of pairwise interactions. The never-seen
cell is the model's intercept, so N = n + exp(b0). Which interactions are in
the model is the whole game: an interaction is a dependence the model allows
for, and a dependence left out biases N in the direction of its sign.

The K-way interaction is never fitted. It is not identifiable from the
observed cells, and pretending otherwise is how MSE produces confident
nonsense (Aleshin-Guendel, Sadinle & Wakefield 2024). Leaving it out is an
assumption, so it is written down here and in the pre-registration.

Intervals are profile likelihood on the full multinomial (Cormack 1992, what
Rcapture reports), not Wald on log(z0): Wald intervals are symmetric on a
scale where the likelihood is not, and undercover at exactly the small
overlaps an adversary produces.

IRLS is written out here instead of calling statsmodels so that a coverage
sweep (thousands of tables times dozens of models times a profile search) fits
in a laptop minute. tests/test_loglinear.py checks it against statsmodels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from lacuna.estimators import Estimate
from lacuna.histories import bits, n_witnesses, observed_total

CHI2_1_95 = 3.841458820694124
# A profile that has not dropped by the cutoff before N = CAP_RATIO * n is
# reported as unbounded above. 20x the observed count is far past any
# plausible miss rate; a likelihood still flat there is saying "no idea".
CAP_RATIO = 20

Pairs = tuple[tuple[int, int], ...]


def all_pairs(k: int) -> Pairs:
    return tuple(combinations(range(k), 2))


def model_space(k: int) -> list[Pairs]:
    """Every subset of pairwise interactions that the observed cells can identify."""
    pairs = all_pairs(k)
    out = []
    for r in range(len(pairs) + 1):
        for subset in combinations(pairs, r):
            if 1 + k + len(subset) <= 2**k - 1:
                out.append(subset)
    return out


def design(k: int, pairs: Pairs, rows) -> np.ndarray:
    out = []
    for h in rows:
        x = bits(h, k)
        out.append([1.0, *x, *(x[a] * x[b] for a, b in pairs)])
    return np.array(out)


@dataclass(frozen=True)
class PoissonFit:
    beta: np.ndarray
    mu: np.ndarray
    loglik: float
    converged: bool


def _xlogy(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m)
    pos = y > 0
    out[pos] = y[pos] * np.log(m[pos])
    return out


def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
    return float(2 * np.sum(_xlogy(y, np.where(y > 0, y, 1.0)) - _xlogy(y, mu) - (y - mu)))


MU_FLOOR = 1e-300


class ConvergenceError(FloatingPointError):
    """A fit that did not converge. Its likelihood is not a likelihood, so do not use it."""


def fit_poisson(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray | None = None,
    max_iter: int = 100,
    offset: np.ndarray | None = None,
) -> PoissonFit:
    """Poisson GLM, log link, IRLS. Converges on deviance like R's glm.

    A cell whose MLE is on the boundary (a zero count with its own parameter)
    sends one coefficient towards -inf; the deviance still converges, which
    is why the stopping rule is on deviance and not on beta.

    `offset` fixes part of the linear predictor: it is how a dependence the
    data cannot estimate is set by assumption (estimators/bounds.py).

    A warm start that fails is retried cold. Before this, a warm start far
    from the optimum could underflow mu to exactly 0, the working response
    went NaN, and the NaN rode silently into a model average.
    """
    y = np.asarray(y, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    if beta is not None:
        fit = _irls(x, y, off, beta, max_iter)
        if fit.converged:
            return fit
    return _irls(x, y, off, None, max_iter)


def _irls(
    x: np.ndarray, y: np.ndarray, off: np.ndarray, beta: np.ndarray | None, max_iter: int
) -> PoissonFit:
    if beta is None:
        mu = y + 0.5
        eta = np.log(mu)
        b = np.zeros(x.shape[1])
    else:
        b = beta
        eta = x @ b + off
        if not np.all(np.isfinite(eta)) or eta.max() > 700:
            return PoissonFit(b, np.full(len(y), np.nan), math.nan, False)
        mu = np.maximum(np.exp(eta), MU_FLOOR)
    dev_old = math.inf
    converged = False
    for _ in range(max_iter):
        z = eta - off + (y - mu) / mu
        xtw = x.T * mu
        a, r = xtw @ x, xtw @ z
        try:
            b_new = np.linalg.solve(a, r)
        except np.linalg.LinAlgError:
            b_new = np.linalg.lstsq(a, r, rcond=None)[0]
        eta_new = x @ b_new + off
        if not np.all(np.isfinite(eta_new)) or eta_new.max() > 700:
            break
        b, eta = b_new, eta_new
        mu = np.maximum(np.exp(eta), MU_FLOOR)
        dev = _deviance(y, mu)
        if abs(dev - dev_old) < 1e-10 * (abs(dev) + 0.1):
            converged = True
            break
        dev_old = dev
    loglik = float(np.sum(_xlogy(y, mu) - mu) - sum(math.lgamma(v + 1) for v in y))
    return PoissonFit(b, mu, loglik, converged)


@dataclass(frozen=True)
class ConditionalFit:
    pairs: Pairs
    n: int
    z0: float  # estimated never-seen count, exp(intercept)
    bic: float
    converged: bool


def fit_conditional(counts: np.ndarray, pairs: Pairs) -> ConditionalFit:
    """Fit on the observed cells only. The intercept is log of the missing cell."""
    k = n_witnesses(counts)
    x = design(k, pairs, range(1, 2**k))
    if x.shape[1] > 2**k - 1:
        raise ValueError(f"{len(pairs)} interactions are not identifiable with K={k}")
    y = np.asarray(counts[1:], dtype=float)
    n = int(y.sum())
    fit = fit_poisson(x, y)
    if not fit.converged:
        # A fit that stopped early has b=0-ish coefficients and would report
        # z0 = 1 with a plausible BIC. Make it unusable instead.
        return ConditionalFit(pairs, n, math.nan, math.inf, False)
    z0 = math.exp(fit.beta[0]) if fit.beta[0] < 700 else math.inf
    bic = -2 * fit.loglik + x.shape[1] * math.log(max(n, 1))
    return ConditionalFit(pairs, n, z0, bic, True)


class Profile:
    """Profile log-likelihood of N for one model, on the full multinomial.

    l(N) = log N! - log (N-n)! + sum_h z_h log(mu_h / N), where mu is the
    Poisson log-linear fit to the complete table with z_0 = N - n (its total
    is N, so mu / N is the multinomial MLE under the model). Constants that do
    not depend on N are dropped; only differences in l are ever used.
    """

    def __init__(self, counts: np.ndarray, pairs: Pairs, offset: np.ndarray | None = None):
        self.k = n_witnesses(counts)
        self.pairs = pairs
        self.y = np.asarray(counts, dtype=float).copy()
        self.n = observed_total(counts)
        self.x = design(self.k, pairs, range(2**self.k))
        self.offset = offset
        self._beta: np.ndarray | None = None
        self._cache: dict[float, float] = {}

    def __call__(self, big_n: float) -> float:
        if big_n in self._cache:
            return self._cache[big_n]
        y = self.y.copy()
        y[0] = big_n - self.n
        fit = fit_poisson(self.x, y, self._beta, offset=self.offset)
        if not fit.converged:
            raise ConvergenceError(f"IRLS did not converge at N={big_n}")
        self._beta = fit.beta
        ll = (
            math.lgamma(big_n + 1)
            - math.lgamma(big_n - self.n + 1)
            + float(np.sum(_xlogy(y, fit.mu / big_n)))
        )
        if not math.isfinite(ll):
            raise FloatingPointError(f"profile log-likelihood not finite at N={big_n}")
        self._cache[big_n] = ll
        return ll

    def maximise(self, start: float) -> tuple[int, float]:
        """Integer argmax of l on [n, cap], from a start near it; l is unimodal in N.

        Exponential search away from the start, then ternary search inside the
        bracket. A likelihood still rising at the cap returns the cap, which
        callers read as "unbounded" through `interval`.
        """
        cap = CAP_RATIO * max(self.n, 1)
        x = min(max(self.n, int(round(start))), cap)
        if x + 1 <= cap and self(x + 1) > self(x):
            d = 1
        elif x - 1 >= self.n and self(x - 1) > self(x):
            d = -1
        else:
            return x, self(x)
        step, prev = 1, x
        while True:
            nxt = min(max(self.n, prev + d * step), cap)
            if nxt == prev:
                return prev, self(prev)
            if self(nxt) <= self(prev):
                break
            prev, step = nxt, step * 2
        a, b = sorted((prev - d * (step // 2), nxt))
        a, b = max(a, self.n), min(b, cap)
        while b - a > 2:
            m1, m2 = a + (b - a) // 3, b - (b - a) // 3
            if self(m1) < self(m2):
                a = m1
            else:
                b = m2
        best = max(range(a, b + 1), key=self)
        return best, self(best)

    def interval(self, mle: int, mle_ll: float, chi2: float = CHI2_1_95) -> tuple[float, float]:
        cut = mle_ll - chi2 / 2
        cap = CAP_RATIO * max(self.n, 1)

        if self(self.n) >= cut:
            lo = float(self.n)
        else:
            a, b = self.n, mle  # l(a) < cut <= l(b)
            while b - a > 1:
                m = (a + b) // 2
                a, b = (a, m) if self(m) >= cut else (m, b)
            lo = float(b)

        step = max(1, mle - self.n)
        a, b = mle, mle + step
        while self(b) >= cut:
            if b > cap:
                return lo, math.inf
            a, b = b, b + 2 * (b - a)
        while b - a > 1:
            m = (a + b) // 2
            a, b = (m, b) if self(m) >= cut else (a, m)
        return lo, float(a)


def _single(counts: np.ndarray, pairs: Pairs, name: str, extra: dict | None = None) -> Estimate:
    n = observed_total(counts)
    cond = fit_conditional(counts, pairs)
    detail = {"pairs": [list(p) for p in pairs], "z0": cond.z0, **(extra or {})}
    if n == 0 or not math.isfinite(cond.z0) or n + cond.z0 > CAP_RATIO * max(n, 1):
        return Estimate(name, n, math.inf, float(n), math.inf, {**detail, "unbounded": True})
    prof = Profile(counts, pairs)
    try:
        mle, mle_ll = prof.maximise(n + cond.z0)
        lo, hi = prof.interval(mle, mle_ll)
    except ConvergenceError:
        return Estimate(name, n, math.inf, float(n), math.inf, {**detail, "nonconverged": True})
    return Estimate(name, n, n + cond.z0, lo, hi, detail)


def independence(counts: np.ndarray) -> Estimate:
    """No interactions: every witness independent of every other. The naive MSE."""
    return _single(counts, (), "ll_independence")


def best_bic(counts: np.ndarray) -> Estimate:
    """The single model with the lowest BIC, interval as if it had been chosen in advance."""
    k = n_witnesses(counts)
    fits = [fit_conditional(counts, m) for m in model_space(k)]
    best = min(fits, key=lambda f: f.bic)
    return _single(counts, best.pairs, "ll_bic", {"bic": best.bic})


def averaged(
    counts: np.ndarray, occam: float = 1e-3, grid_points: int = 300, level: float = 0.95
) -> Estimate:
    """BIC-weighted mixture of per-model profile likelihoods, read as posteriors.

    Each model's profile likelihood over N is normalised into a density (a flat
    prior on N), the densities are mixed with BIC weights, and the interval
    is the central `level` mass of the mixture. Model uncertainty then widens
    the interval instead of being thrown away by picking one model, which is
    the point of averaging (Madigan & York 1997 do this exactly with
    decomposable models and a hyper-Dirichlet prior; this is the cheap version,
    noted in docs/DECISIONS.md).

    Models below `occam` times the best weight are dropped (Occam's window),
    and so are models whose fit did not converge.

    Mass beyond the cap is one atom at +inf, and all three numbers are read
    from that one mixture. Per model, the atom is the likelihood's tail past
    the cap, extrapolated from its slope at the cap (an exponential tail
    integrates in closed form). A model whose likelihood is not falling at the
    cap has an improper tail, and its whole weight goes to the atom. A model
    that merely has a long tail keeps its shape and loses only that tail.
    """
    k = n_witnesses(counts)
    n = observed_total(counts)
    fits = [f for f in (fit_conditional(counts, m) for m in model_space(k)) if f.converged]
    bic = np.array([f.bic for f in fits])
    w = np.exp(-(bic - bic.min()) / 2)
    keep = [i for i in range(len(fits)) if w[i] >= occam * w.max()]
    if n == 0 or not keep:
        return Estimate("ll_average", n, math.inf, float(n), math.inf, {"unbounded": True})

    # Grid wide enough to hold every kept model's mass: each model's 1e-5
    # likelihood-ratio interval (chi2 ~ 23), unioned. Models that reach the
    # cap get a coarse extension out to it.
    cap = float(CAP_RATIO * n)
    profiles, weights, models = [], [], []
    lo_g, hi_g, reaches_cap = math.inf, -math.inf, False
    for i in keep:
        f = fits[i]
        prof = Profile(counts, f.pairs)
        start = n + f.z0 if math.isfinite(f.z0) and n + f.z0 <= cap else cap
        try:
            mle, mle_ll = prof.maximise(start)
            a, b = prof.interval(mle, mle_ll, chi2=23.0)
        except ConvergenceError:
            continue
        profiles.append(prof)
        weights.append(w[i])
        models.append([list(p) for p in f.pairs])
        lo_g = min(lo_g, a)
        if math.isfinite(b):
            hi_g = max(hi_g, b)
        else:
            reaches_cap = True
    if not profiles:
        return Estimate("ll_average", n, math.inf, float(n), math.inf, {"unbounded": True})
    hi_g = max(hi_g, lo_g + 1)
    w_keep = np.array(weights) / sum(weights)

    if hi_g - lo_g <= grid_points:
        grid = np.arange(lo_g, hi_g + 1)
    else:
        grid = np.unique(np.round(np.linspace(lo_g, hi_g, grid_points)))
    if reaches_cap:
        grid = np.unique(np.concatenate([grid, np.round(np.linspace(hi_g, cap, grid_points))]))
    # Trapezoid widths so a coarse grid does not over-weight its dense end.
    width = np.gradient(grid) if len(grid) > 1 else np.ones(1)

    mix = np.zeros(len(grid))
    atom = 0.0
    for wi, prof in zip(w_keep, profiles, strict=True):
        try:
            ll = np.array([prof(float(g)) for g in grid])
        except ConvergenceError:
            atom += wi  # cannot place its mass, so it says nothing finite
            continue
        top = ll.max()
        dens = np.exp(ll - top) * width
        tail = 0.0
        if reaches_cap and len(grid) > 1:
            slope = (ll[-1] - ll[-2]) / (grid[-1] - grid[-2])
            tail = math.inf if slope >= 0 else math.exp(ll[-1] - top) / -slope
        if math.isinf(tail):
            atom += wi
            continue
        total = dens.sum() + tail
        mix += wi * dens / total
        atom += wi * tail / total

    level_tail = (1 - level) / 2
    cdf = np.cumsum(mix)

    def q(p: float) -> float:
        if p > cdf[-1]:
            return math.inf
        return float(grid[min(int(np.searchsorted(cdf, p)), len(grid) - 1)])

    detail = {
        "models": models,
        "weights": [float(v) for v in w_keep],
        "unbounded_weight": round(atom, 6),
    }
    return Estimate("ll_average", n, q(0.5), q(level_tail), q(1 - level_tail), detail)
