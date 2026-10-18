"""Stratified log-linear estimation: one BIC-best model per stratum, convolved.

When the drop rate depends on something every captured record carries (its
op), the missingness is MAR given that covariate and a pooled model is
misspecified, while each stratum on its own has the constant-rate,
independent-drop structure the log-linear family assumes. `strat_bic` fits
each stratum separately and adds the strata up in distribution, not in
interval endpoints (docs/MNAR-ROBUSTNESS.md).

Per stratum:
1. Witnesses that saw nothing in it are dropped (the `financial` env view
   makes W3 structurally dead in non-financial strata).
2. The stratum is *identified* only with at least two live witnesses and
   some discordance among them (docs/DECISIONS.md D17). Otherwise nothing in
   the data speaks to what the live witnesses jointly missed.
3. The BIC-best log-linear model's profile likelihood over N_s is normalised
   into a density (a flat prior on N_s >= n_s). Past the search cap
   (CAP_RATIO * n_s) the likelihood is extrapolated as a power law, which is
   its true large-N shape: the two-witness likelihood falls like N**-m, m the
   doubly-seen count. An exponent at or below 1 is an improper posterior.
4. The densities are convolved, and the total's 2.5/50/97.5% points are
   reported.

A stratum that is unidentified, or whose density cannot be normalised (no
converged model, a point estimate past the cap, an improper tail, a profile
that will not converge), is not in the convolution. It still adds its
observed count to the lower end, which stays a valid floor; the point and the
upper end of the total go to infinity.

An empty stratum (nothing observed) is skipped. An auditor cannot know an op
exists that no witness ever recorded, and under oracle linkage the stratum
list comes from the ledger, which an estimator must not read.

This replaces the Bonferroni sum of per-stratum intervals (`strat_lc_gated`,
D17/D27). Summing endpoints at level 1 - alpha/m spends coverage m times;
convolving densities spends it once, on the total.
"""

from __future__ import annotations

import math

import numpy as np

from lacuna.estimators import Estimate
from lacuna.estimators.loglinear import CAP_RATIO, Profile, fit_conditional, model_space
from lacuna.histories import keep_witnesses, n_witnesses, observed_total

# Likelihood-ratio drop (chi2 scale) past which a stratum's density is treated
# as zero: exp(-30/2) is 3e-7 of the peak.
SUPPORT_CHI2 = 30.0
# Profile evaluations per stratum. A wider support is evaluated on a geometric
# grid and interpolated onto the integers, where the profile is smooth.
GRID_POINTS = 300
# Convolutions longer than this go through the FFT.
DIRECT_CONV_MAX = 4_000_000


def live_witnesses(table: np.ndarray, k: int) -> tuple[int, ...]:
    """Witnesses that saw at least one action in this table."""
    return tuple(w for w in range(k) if any(table[h] for h in range(1, 2**k) if h >> w & 1))


def has_discordance(table: np.ndarray, live: tuple[int, ...]) -> bool:
    """True if some action was seen by a proper subset of the live witnesses.

    With no discordance every live witness saw exactly the same actions, so
    the data carry no information about what they jointly missed. A model
    fitted there reports the observed count with a thin margin, which looks
    confident and is not (the reviewer's perfect-overlap case: a table whose
    only non-zero cell is the all-live one returns 20 [20, 22]).
    """
    full = sum(1 << w for w in live)
    return any(table[h] for h in range(1, len(table)) if h != full and h & full)


def stratum_density(counts: np.ndarray) -> tuple[int, np.ndarray, float] | None:
    """(offset, pmf, tail): the flat-prior density of N for one stratum.

    pmf[i] is the mass at N = offset + i; `tail` is the mass past the cap,
    which sits at +inf for the purposes of the total. pmf.sum() + tail == 1.
    None when the density cannot be normalised (see module docstring).
    """
    k = n_witnesses(counts)
    n = observed_total(counts)
    fits = [f for f in (fit_conditional(counts, m) for m in model_space(k)) if f.converged]
    if n == 0 or not fits:
        return None
    best = min(fits, key=lambda f: f.bic)
    cap = CAP_RATIO * n
    if not math.isfinite(best.z0) or n + best.z0 > cap:
        return None
    prof = Profile(counts, best.pairs)
    try:
        mle, mle_ll = prof.maximise(n + best.z0)
        lo, hi = prof.interval(mle, mle_ll, chi2=SUPPORT_CHI2)
        reaches_cap = not math.isfinite(hi)
        lo_i, hi_i = int(lo), cap if reaches_cap else int(hi)
        hi_i = max(hi_i, lo_i + 1)
        if hi_i - lo_i + 1 <= GRID_POINTS:
            grid = np.arange(lo_i, hi_i + 1, dtype=float)
        else:
            grid = np.unique(np.round(np.geomspace(max(lo_i, 1), hi_i, GRID_POINTS)))
            grid = np.unique(np.concatenate([[float(lo_i)], grid, [float(mle)]]))
        ll = np.array([prof(float(g)) for g in grid])
    except FloatingPointError:  # ConvergenceError is one
        return None

    support = np.arange(lo_i, hi_i + 1, dtype=float)
    ll_int = ll if len(grid) == len(support) else np.interp(support, grid, ll)
    top = float(ll_int.max())
    dens = np.exp(ll_int - top)
    tail = 0.0
    if reaches_cap:
        # Power-law extrapolation from the last two grid points: l = a - s log N.
        s = -(ll[-1] - ll[-2]) / (math.log(grid[-1]) - math.log(grid[-2]))
        if s <= 1:
            return None
        tail = float(dens[-1]) * support[-1] / (s - 1)
    total = float(dens.sum()) + tail
    return lo_i, dens / total, tail / total


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) * len(b) <= DIRECT_CONV_MAX:
        return np.convolve(a, b)
    size = len(a) + len(b) - 1
    out = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)
    return np.clip(out, 0.0, None)


def convolve_densities(parts: list[tuple[int, np.ndarray]]) -> tuple[int, np.ndarray]:
    """Density of the sum of independent integer variables, each (offset, pmf)."""
    offset, pmf = 0, np.ones(1)
    for off, p in parts:
        offset += off
        pmf = _convolve(pmf, p)
    return offset, pmf


def quantile(offset: int, pmf: np.ndarray, p: float) -> float:
    """Smallest N with P(total <= N) >= p; +inf when the finite mass is short of p."""
    cdf = np.cumsum(pmf)
    if p > cdf[-1] + 1e-12:
        return math.inf
    return float(offset + min(int(np.searchsorted(cdf, p - 1e-12)), len(pmf) - 1))


def strat_bic(per_op_tables: dict[str, np.ndarray], level: float = 0.95) -> Estimate:
    n_total = floor = 0
    parts: list[tuple[int, np.ndarray]] = []
    tails: list[float] = []
    counts = {"strata": 0, "identified": 0, "unidentified": 0, "unbounded": 0, "empty": 0}
    for table in per_op_tables.values():
        n = observed_total(table)
        if n == 0:
            counts["empty"] += 1
            continue
        counts["strata"] += 1
        n_total += n
        live = live_witnesses(table, n_witnesses(table))
        if len(live) < 2 or not has_discordance(table, live):
            counts["unidentified"] += 1
            floor += n
            continue
        dens = stratum_density(keep_witnesses(table, live))
        if dens is None:
            counts["unbounded"] += 1
            floor += n
            continue
        counts["identified"] += 1
        off, pmf, tail = dens
        parts.append((off, pmf))
        tails.append(tail)

    if not counts["strata"]:
        return Estimate("strat_bic", 0, math.inf, 0.0, math.inf, counts)
    offset, pmf = convolve_densities(parts)
    offset += floor
    tail_p = (1 - level) / 2
    lo, point, hi = (quantile(offset, pmf, p) for p in (tail_p, 0.5, 1 - tail_p))
    if not math.isfinite(lo):
        # Almost all the identified strata's mass is past their caps.
        lo = float(n_total)
    if floor:
        point = hi = math.inf
    detail = {
        **counts,
        "unbounded_mass": round(1 - float(np.prod([1 - t for t in tails])), 6),
    }
    return Estimate("strat_bic", n_total, point, lo, hi, detail)


def tier_bic(episode_tables, n_tiers: int = 6) -> Estimate:
    """`strat_bic` over tiers of episodes ranked by how much the logs saw (D53).

    When session quality varies and hits every log at once, a pooled table mixes
    good and bad sessions and no log-linear model separates them. Episodes are
    observed, so they can be grouped: rank them by the number of distinct actions
    any log recorded, split into `n_tiers` near-equal tiers, and estimate each
    tier on its own. Tier boundaries come from observed counts only.
    """
    tables = [np.asarray(t, dtype=np.int64) for t in episode_tables]
    seen = np.array([int(t[1:].sum()) for t in tables])
    order = np.argsort(seen, kind="stable")
    groups = [g for g in np.array_split(order, n_tiers) if len(g)]
    tiers = {i: np.sum([tables[j] for j in g], axis=0) for i, g in enumerate(groups)}
    for t in tiers.values():
        t[0] = 0
    est = strat_bic(tiers)
    return Estimate("tier_bic", est.n_observed, est.point, est.lo, est.hi, est.detail)
