"""Estimators of N, the number of actions that happened, from witness counts.

Every estimator returns an `Estimate`. A point with no interval is still an
`Estimate` whose interval is the point, so coverage is computed the same way
for the floor and for the method, and the floor cannot hide behind "n/a".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Estimate:
    name: str
    n_observed: int
    point: float
    lo: float
    hi: float
    detail: dict = field(default_factory=dict)

    @property
    def bounded(self) -> bool:
        return math.isfinite(self.hi)

    def covers(self, n_true: int) -> bool:
        return self.lo <= n_true <= self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo
