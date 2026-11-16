"""`ll_sa_hom`: the estimator study 5 registered, where the tool can import it.

The input is one table over three witnesses in this order: two logs linked by an id
(bits 0 and 1; `spans` and `access` on the study's rig) and one record of the state
with no id (bit 2; `env`), summed over planes, and the same table per session and
plane.

The model has one interaction, between the two logs: they may lose the same records.
The state is taken to miss actions independently of both. Its missing cell has a closed
form, Lincoln-Petersen between "either log" and the state. The interval is that model's
profile likelihood, its cutoff scaled by how much the per-session tables disagree
(`loglinear.homogeneity_dispersion`, D51). It is unbounded when no action is in both
lists, when the point passes `loglinear.CAP_RATIO` times the count seen, or when the
profile's upper end does.

This is `scripts/study5_lib.sa_hom` (frozen with the registration,
`docs/PREREG-STUDY5.md`): same model, same interval, same guards, same `detail`.
Wherever that code gives an estimate, this gives the same one
(`tests/test_sa_hom.py`).

One difference, made after the study and so post hoc. The registered code fits the
model by iteration (`loglinear.fit_poisson`) at every N the profile visits. On a table
with empty cells the fit is on the boundary: the empty cells' fitted counts go to zero,
the weighted normal equations lose rank as they do, and whether the iteration stops
before a solve of them returns noise is decided by rounding. When it loses, the
registered code reports no estimate (`detail["nonconverged"]`); it did on one
unstressed run of study 5, `s5-s0-021`, where no log had lost anything. This model
does not need the iteration: the two logs together and the state are two independent
factors, so the fitted table is the product of their margins over N. Where the
iteration fails, the profile is recomputed from that closed form and
`detail["closed_form"]` says so. It is the same likelihood, so nothing else changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from lacuna.estimators import Estimate, loglinear

NAME = "ll_sa_hom"
# The two logs may depend on each other; the state depends on neither.
TELEMETRY_PAIR: loglinear.Pairs = ((0, 1),)


class ClosedFormProfile(loglinear.Profile):
    """`loglinear.Profile` for `TELEMETRY_PAIR` on three witnesses, with no iteration.

    With z_0 = N - n in the table, the model's fitted count for a cell is (actions with
    that record in the two logs) x (actions with that record in the state) / N. A cell
    that holds an action has both margins above zero, so the likelihood is finite at
    every N from n up, empty cells or not. The searches are the parent's.
    """

    def __call__(self, big_n: float) -> float:
        if big_n in self._cache:
            return self._cache[big_n]
        y = self.y.copy()
        y[0] = big_n - self.n
        logs = y[:4] + y[4:]  # by what the two logs hold: neither, first, second, both
        mu = np.concatenate([logs * y[:4].sum(), logs * y[4:].sum()]) / big_n
        ll = (
            math.lgamma(big_n + 1)
            - math.lgamma(big_n - self.n + 1)
            + float(np.sum(loglinear._xlogy(y, mu / big_n)))
        )
        self._cache[big_n] = ll
        return ll


def _interval(prof: loglinear.Profile, start: float, phi: float) -> tuple[float, float]:
    mle, mle_ll = prof.maximise(start)
    return prof.interval(mle, mle_ll, chi2=loglinear.CHI2_1_95 * phi)


def sa_hom(pooled: np.ndarray, per_session: Sequence[np.ndarray], name: str = NAME) -> Estimate:
    """One model named in advance, its interval widened by how much sessions disagree.

    No model is selected, so a sparse table cannot tip the choice to a model whose
    missing cell is unstable.
    """
    if len(pooled) != 8:
        raise ValueError(f"{NAME} needs a table over three witnesses, not {len(pooled)} cells")
    phi = loglinear.homogeneity_dispersion(list(per_session)) if len(per_session) else 1.0
    n = int(pooled[1:].sum())
    env_only = int(pooled[4])
    tele_only = int(pooled[1] + pooled[2] + pooled[3])
    both = int(pooled[5] + pooled[6] + pooled[7])
    detail = {"pairs": [list(p) for p in TELEMETRY_PAIR], "phi": phi}
    unbounded = Estimate(name, n, math.inf, float(n), math.inf, detail | {"unbounded": 1})
    if both == 0:
        # No action is in both lists (or there are no actions): nothing ties the state
        # to the logs, and the missing cell is 0/0 or infinite.
        return unbounded
    z0 = env_only * tele_only / both
    detail["z0"] = z0
    if n + z0 > loglinear.CAP_RATIO * n:
        # Past the cap every log-linear estimate here is reported as "no idea".
        return unbounded
    try:
        lo, hi = _interval(loglinear.Profile(pooled, TELEMETRY_PAIR), n + z0, phi)
    except FloatingPointError:  # ConvergenceError is one
        # Where the registered code stops with `nonconverged`. See the module docstring.
        lo, hi = _interval(ClosedFormProfile(pooled, TELEMETRY_PAIR), n + z0, phi)
        detail["closed_form"] = 1
    return Estimate(name, n, n + z0, lo, hi, detail)
