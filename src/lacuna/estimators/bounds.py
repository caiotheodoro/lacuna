"""Sensitivity bounds for the adaptive agent (L2), anchored on one witness.

Against an agent that correlates its omissions across witnesses, N is not
identified from the observed cells. No model choice fixes that
(docs/SIM-FINDINGS.md, l2_env_blind). What can still be said is conditional:
"if the anchor's misses and the others' misses are dependent by at most Γ,
then N is at most U(Γ)".

Collapse the witnesses to two lists: the anchor (W3, the env diff, by
default) and the union of everyone else. Write ψ for their odds ratio. Then

    z_00 = ψ · z_10 · z_01 / z_11

ψ = 1 is Lincoln-Petersen of the anchor against the rest. That is a point
estimate, and it is exact if the anchor is independent of the agent's hiding
policy (Qin et al.'s interception server, if it worked, would be such an
anchor). ψ > 1 means the anchor tends to miss what the others miss: the
agent's hidden actions are also the ones the env diff is blind to.

The assumption is ψ ∈ [1, Γ]: hiding never makes the anchor *more* likely to
see an action. The interval runs from the profile lower end at ψ = 1 to the
profile upper end at ψ = Γ. Γ is a dial the auditor reports a curve over,
in the spirit of Rosenbaum's sensitivity analysis. It is not something to
estimate: the data cannot see it.
"""

from __future__ import annotations

import math

import numpy as np

from lacuna.estimators import Estimate
from lacuna.estimators.loglinear import CAP_RATIO, ConvergenceError, Profile
from lacuna.histories import n_witnesses, observed_total

ENV = 2


def anchor_table(counts: np.ndarray, anchor: int = ENV) -> np.ndarray:
    """2-list table: bit 0 = anchor saw it, bit 1 = any other witness saw it."""
    k = n_witnesses(counts)
    if not 0 <= anchor < k:
        # Without this, anchor=2 on a K=2 table never sets the bit and comes
        # back "unbounded", which reads like a finding instead of a typo.
        raise ValueError(f"anchor {anchor} out of range for K={k}")
    out = np.zeros(4, dtype=np.int64)
    for h, c in enumerate(counts):
        if h == 0:
            continue
        a = (h >> anchor) & 1
        rest = int((h & ~(1 << anchor)) != 0)
        out[a | (rest << 1)] += c
    return out


def _psi_profile(table: np.ndarray, psi: float) -> Profile:
    offset = np.array([0.0, 0.0, 0.0, math.log(psi)])  # on the both-seen cell
    return Profile(table, (), offset=offset)


def anchor_bounds(counts: np.ndarray, gamma: float = 1.0, anchor: int = ENV) -> Estimate:
    if gamma < 1:
        raise ValueError("gamma < 1 would assume hiding helps the anchor see")
    name = f"anchor_g{gamma:g}"
    t = anchor_table(counts, anchor)
    n = observed_total(t)
    z10, z01, z11 = int(t[1]), int(t[2]), int(t[3])
    detail = {"anchor": anchor, "gamma": gamma, "table": t.tolist()}
    if z11 == 0 or n == 0:
        return Estimate(name, n, math.inf, float(n), math.inf, {**detail, "unbounded": True})

    point = n + z10 * z01 / z11
    top = n + gamma * z10 * z01 / z11
    try:
        lo_prof = _psi_profile(t, 1.0)
        mle, mle_ll = lo_prof.maximise(point)
        lo, _ = lo_prof.interval(mle, mle_ll)
        if top > CAP_RATIO * n:
            return Estimate(name, n, point, lo, math.inf, {**detail, "unbounded": True})
        hi_prof = _psi_profile(t, gamma)
        mle, mle_ll = hi_prof.maximise(top)
        _, hi = hi_prof.interval(mle, mle_ll)
    except ConvergenceError:
        return Estimate(name, n, point, float(n), math.inf, {**detail, "nonconverged": True})
    return Estimate(name, n, point, lo, hi, {**detail, "upper_point": top})
