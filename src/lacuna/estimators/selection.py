"""A selection-aware interval for `ll_bic`: bootstrap with the BIC choice inside.

`loglinear.best_bic` picks the lowest-BIC model and then profiles the
likelihood of that model as if it had been chosen before seeing the data. The
interval conditions on the selection, so any uncertainty about which
interactions belong in the model (and the interactions move N-hat by tens of
units, see the L0 heterogeneous scenario) is left out.

This interval puts the selection inside the resampling loop. A parametric
bootstrap generates full tables from the fitted BIC-selected model at its own
N-hat (cell 0 included, the unseen cell is just another cell), hides cell 0,
reruns the entire procedure (all 8 models, BIC choice, N-hat = n + z0 of the
winner) and records the error N-hat* - N*. The choice of model therefore
varies from resample to resample, and its variability is in the spread.

The resampling world is a choice. `world="selected"` generates from the model
BIC picked, so it repeats the selection's own premise. `world="saturated"`
generates from the all-pairwise (no three-way) model, the richest the observed
cells identify, so resamples contain interactions that the selected model
dropped and the selection can miss them again; this is the world that tests
what selection costs. It falls back to the selected world when the saturated
fit is unbounded. Names carry a `_sat` suffix for the saturated world.

Two intervals come from the same draws, and both are reported by
`bootstrap_bic`:

- percentile: the 2.5% and 97.5% quantiles of N-hat*. Simple, but it is an
  interval for where N-hat falls, centred on N-hat, not on N.
- basic (pivotal): N-hat minus the 97.5% and 2.5% quantiles of N-hat* - N*.
  Uses the bootstrap distribution of the error, which is the quantity an
  interval for N must invert. This is the one returned as the headline
  `ll_bic_boot`; percentile is `ll_bic_boot_pct`.

Not bootstrap-t: with 7 observed cells the standard error of a selected model's
N-hat is itself a function of the selection, so a studentised version needs an
inner bootstrap per resample (200 x 200 fits) for a gain this sample size cannot
resolve. The lower end is floored at n (the observed count is a hard bound), and
an upper end is infinite when more than 2.5% of resamples have no finite N-hat.

What it costs and what it does not fix (artifacts/robustness/selection-coverage.json,
400 outer reps):

- It loses gap detection. The interval is wider and its lower end is floored at n, so
  "the lower end exceeds the union count" fires less often than for `ll_bic`
  (l0_independent 0.32 -> 0.11, l0_heterogeneous 0.38 -> 0.0025, l0_siloed 0.13 -> 0.05,
  l1_hides_from_trace 0.93 -> 0.74; confirm/fin_mid stays at 1.00).
- It does not fix model-class failure. Every resample is drawn from a fitted
  pairwise log-linear model, so when the truth lies outside the class (l0_heterogeneous,
  coverage 0.31 for `ll_bic`) the bootstrap world is wrong too and coverage stays near
  0.33. It repairs the part of the error that comes from choosing among the models, not
  from the family itself.
"""

from __future__ import annotations

import math

import numpy as np

from lacuna.estimators import Estimate, loglinear
from lacuna.histories import n_witnesses, observed_total

DEFAULT_REPS = 200


def select_point(counts: np.ndarray) -> tuple[loglinear.Pairs, float]:
    """The BIC-selected model and its N-hat = n + z0 (inf if the winner is unbounded)."""
    k = n_witnesses(counts)
    n = observed_total(counts)
    fits = [loglinear.fit_conditional(counts, m) for m in loglinear.model_space(k)]
    usable = [f for f in fits if f.converged]
    if n == 0 or not usable:
        return (), math.inf
    best = min(usable, key=lambda f: f.bic)
    point = n + best.z0
    if not math.isfinite(point) or point > loglinear.CAP_RATIO * n:
        return best.pairs, math.inf
    return best.pairs, point


def fitted_full_table(counts: np.ndarray, pairs: loglinear.Pairs) -> np.ndarray:
    """Expected counts for all 2**K cells under `pairs`, cell 0 = z0.

    Not rescaled. The Poisson fit has an intercept, so the fitted observed cells sum to the
    observed total and the table sums to n + z0 (the N-hat of this model) as a property of
    the fit, not by construction.
    """
    k = n_witnesses(counts)
    x_obs = loglinear.design(k, pairs, range(1, 2**k))
    fit = loglinear.fit_poisson(x_obs, np.asarray(counts[1:], dtype=float))
    x_all = loglinear.design(k, pairs, range(2**k))
    return np.exp(x_all @ fit.beta)


def bootstrap_draws(
    counts: np.ndarray, reps: int, rng: np.random.Generator, world: str = "selected"
) -> tuple[float, np.ndarray, np.ndarray]:
    """N-hat and, per resample, N-hat* and the resample's true N* (selection redone each time)."""
    counts = np.asarray(counts, dtype=np.int64)
    pairs, point = select_point(counts)
    if not math.isfinite(point):
        return point, np.full(reps, np.inf), np.full(reps, np.nan)
    mu = fitted_full_table(counts, pairs)
    if world == "saturated":
        k = n_witnesses(counts)
        sat = fitted_full_table(counts, loglinear.all_pairs(k))
        n = observed_total(counts)
        if np.all(np.isfinite(sat)) and n < sat.sum() <= loglinear.CAP_RATIO * n:
            mu = sat
    elif world != "selected":
        raise ValueError(f"unknown world {world!r}")
    n_star = max(int(round(mu.sum())), observed_total(counts))
    p = mu / mu.sum()
    est = np.empty(reps)
    for b in range(reps):
        tab = rng.multinomial(n_star, p)
        tab[0] = 0
        est[b] = select_point(tab)[1]
    return point, est, np.full(reps, float(n_star))


def bootstrap_bic(
    counts: np.ndarray,
    reps: int = DEFAULT_REPS,
    seed: int = 0,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
    world: str = "selected",
) -> dict[str, Estimate]:
    """`ll_bic_boot` (basic) and `ll_bic_boot_pct` (percentile) from one set of draws.

    With world="saturated" the names are `ll_bic_boot_sat` and `ll_bic_boot_sat_pct`.
    """
    counts = np.asarray(counts, dtype=np.int64)
    n = observed_total(counts)
    rng = rng if rng is not None else np.random.default_rng(seed)
    sfx = "_sat" if world == "saturated" else ""
    basic_name, pct_name = f"ll_bic_boot{sfx}", f"ll_bic_boot{sfx}_pct"
    point, est, n_star = bootstrap_draws(counts, reps, rng, world)
    if not math.isfinite(point):
        return {
            name: Estimate(name, n, math.inf, float(n), math.inf, {"unbounded": True})
            for name in (basic_name, pct_name)
        }
    a = (1 - level) / 2
    s = np.sort(est)
    # Order statistics, infinities included: the top slice of the draws is allowed to be +inf.
    pct_lo = float(s[max(0, int(math.floor(a * reps)))])
    pct_hi = float(s[min(reps - 1, int(math.ceil((1 - a) * reps)) - 1)])
    err = np.sort(est - n_star)  # inf - finite = inf; no NaN because n_star is finite here
    e_lo = float(err[max(0, int(math.floor(a * reps)))])
    e_hi = float(err[min(reps - 1, int(math.ceil((1 - a) * reps)) - 1)])
    basic_lo, basic_hi = point - e_hi, point - e_lo
    detail = {
        "reps": reps,
        "unbounded_draws": int(np.sum(~np.isfinite(est))),
        "selected_pairs": [list(p) for p in select_point(counts)[0]],
        "world": world,
    }
    return {
        basic_name: Estimate(
            basic_name,
            n,
            point,
            max(float(n), basic_lo) if math.isfinite(basic_lo) else float(n),
            basic_hi if math.isfinite(basic_hi) else math.inf,
            {**detail, "method": "basic"},
        ),
        pct_name: Estimate(
            pct_name,
            n,
            point,
            max(float(n), pct_lo),
            pct_hi if math.isfinite(pct_hi) else math.inf,
            {**detail, "method": "percentile"},
        ),
    }


def bic_boot(counts: np.ndarray, reps: int = DEFAULT_REPS, seed: int = 0) -> Estimate:
    """The headline selection-aware interval (basic bootstrap), deterministic given `seed`."""
    return bootstrap_bic(counts, reps=reps, seed=seed)["ll_bic_boot"]
