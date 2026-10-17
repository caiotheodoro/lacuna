"""The floor. Anything lacuna publishes has to beat these, or say that it did not.

- `trust_trace`: believe the agent's own log. What most audits do today.
- `auditor_count`: believe an LLM that rebuilt the actions from the transcript.
- `union_count`: every distinct action any witness saw. Always <= N.
- `union_bound`: the same union read as [n, inf). It always covers and says
  nothing. This is the flag-everything of counting: a method only earns its
  keep by giving a finite upper end while keeping coverage.
"""

from __future__ import annotations

import numpy as np

from lacuna.estimators import Estimate
from lacuna.histories import observed_total, seen_by

TRACE, AUDITOR = 0, 3


def trust_trace(counts: np.ndarray) -> Estimate:
    n = seen_by(counts, TRACE)
    return Estimate("trust_trace", observed_total(counts), n, n, n)


def auditor_count(counts: np.ndarray) -> Estimate:
    n = seen_by(counts, AUDITOR)
    return Estimate("auditor_count", observed_total(counts), n, n, n)


def union_count(counts: np.ndarray) -> Estimate:
    n = observed_total(counts)
    return Estimate("union_count", n, n, n, n)


def union_bound(counts: np.ndarray) -> Estimate:
    n = observed_total(counts)
    return Estimate("union_bound", n, n, n, float("inf"))
