"""When to refuse to report a count: an identifiability gate in front of `lc`.

The second sweep showed `lc` covering 0.054 on l0_siloed while reporting a
finite upper end 86% of the time. When witnesses barely overlap, models that
fit the observed cells equally well disagree about N without limit
(Aleshin-Guendel, Sadinle & Wakefield 2024). The latent-class prior then picks
one answer, and the posterior looks confident about it.

The gate uses the log-linear family as a witness against the latent-class
answer. A model is "well supported" if it converged and its BIC is within
`delta_bic` of the best (6 is "strong evidence" on the Kass & Raftery scale).
Then:

1. If any well-supported model is unbounded, the data allow N to be
   anything. Refuse.
2. If `lc`'s interval does not contain every well-supported model's point,
   the answer depends on which plausible model you believe, and `lc` is
   hiding that. Refuse.

A refusal reports `union_bound`: at least n, and nothing more. The gate's
cost is its refusal rate on scenarios where N *is* identified, so the sweep
reports both sides.

The overlap fraction (share of seen actions caught by two or more witnesses)
is reported for reading, not used to decide. A threshold on it would be a
tuned number; the two rules above are not.
"""

from __future__ import annotations

import math

import numpy as np

from lacuna.estimators import Estimate
from lacuna.estimators.latent_class import latent_class
from lacuna.estimators.loglinear import CAP_RATIO, fit_conditional, model_space
from lacuna.histories import n_witnesses, observed_total

DELTA_BIC = 6.0


def overlap_fraction(counts: np.ndarray) -> float:
    n = observed_total(counts)
    multi = sum(int(c) for h, c in enumerate(counts) if h and bin(h).count("1") >= 2)
    return multi / n if n else 0.0


def supported_points(counts: np.ndarray, delta_bic: float = DELTA_BIC) -> list[float]:
    """N-hat of every converged log-linear model within delta_bic of the best."""
    k = n_witnesses(counts)
    n = observed_total(counts)
    fits = [f for f in (fit_conditional(counts, m) for m in model_space(k)) if f.converged]
    if not fits:
        return [math.inf]
    best = min(f.bic for f in fits)
    out = []
    for f in fits:
        if f.bic - best <= delta_bic:
            point = n + f.z0
            out.append(point if math.isfinite(point) and point <= CAP_RATIO * n else math.inf)
    return out


def lc_gated(counts: np.ndarray, delta_bic: float = DELTA_BIC, **lc_kw) -> Estimate:
    n = observed_total(counts)
    points = supported_points(counts, delta_bic)
    detail = {
        "overlap_fraction": round(overlap_fraction(counts), 4),
        "supported_points": [p if math.isfinite(p) else "inf" for p in points],
    }
    if any(not math.isfinite(p) for p in points):
        return _refuse(n, {**detail, "refused": "a well-supported model is unbounded"})
    est = latent_class(counts, **lc_kw)
    if not all(est.lo <= p <= est.hi for p in points):
        return _refuse(
            n, {**detail, "refused": "well-supported models fall outside the lc interval"}
        )
    return Estimate("lc_gated", n, est.point, est.lo, est.hi, {**detail, **est.detail})


def _refuse(n: int, detail: dict) -> Estimate:
    return Estimate("lc_gated", n, math.inf, float(n), math.inf, detail)
