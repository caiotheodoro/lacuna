"""Synthetic capture histories with a known N and a known dependence structure.

Before any agent runs, the estimator has to earn its place on tables where the
truth is a constant, not a measurement. Every scenario here is the same
generative model with different knobs:

- each action belongs to a latent type (easy to log, hard to log, hidden);
- within a type, the history probability is log-linear over the K capture
  bits: main effects per witness plus a log odds ratio per witness pair.

Types are how heterogeneity enters (every witness finds hard actions hard,
which makes the lists positively dependent without any witness copying
another). Pair terms are direct dependence (the auditor reads the agent's own
transcript, so it misses what the trace misses). A type with a -inf shift on
some witnesses is an agent that keeps those actions away from them entirely:
the adversary of the threat ladder, as data.

Sampling is exact: for K <= 4 the 2**K history probabilities are enumerated
and drawn from a multinomial, so there is no MCMC to trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lacuna.histories import bits

NEG_INF = float("-inf")


@dataclass(frozen=True)
class ActionType:
    weight: float
    shift: tuple[float, ...]  # added to each witness's logit; -inf = never seen by it


@dataclass(frozen=True)
class Scenario:
    name: str
    level: str  # threat ladder rung: L0, L1, L2
    n_actions: int
    logit: tuple[float, ...]  # per-witness main effect
    pairs: dict[tuple[int, int], float] = field(default_factory=dict)  # log odds ratios
    types: tuple[ActionType, ...] = ()
    note: str = ""

    @property
    def k(self) -> int:
        return len(self.logit)

    def _types(self) -> tuple[ActionType, ...]:
        return self.types or (ActionType(1.0, (0.0,) * self.k),)

    def history_probs(self, t: ActionType) -> np.ndarray:
        """P(history) for one action type. Entry 0 is the never-seen cell."""
        k = self.k
        score = np.empty(2**k)
        for h in range(2**k):
            x = bits(h, k)
            s = 0.0
            for j in range(k):
                if x[j]:
                    s += self.logit[j] + t.shift[j]
            for (a, b), gamma in self.pairs.items():
                if x[a] and x[b]:
                    s += gamma
            score[h] = s
        score -= score[np.isfinite(score)].max()
        p = np.exp(score)
        return p / p.sum()

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Full table including the true cell 0. Callers must not leak it."""
        types = self._types()
        w = np.array([t.weight for t in types], dtype=float)
        per_type = rng.multinomial(self.n_actions, w / w.sum())
        counts = np.zeros(2**self.k, dtype=np.int64)
        for t, n_t in zip(types, per_type, strict=True):
            counts += rng.multinomial(n_t, self.history_probs(t))
        return counts


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


# Capture rates are loose guesses at the witnesses in PLAN.md: the agent's own
# trace mostly works (L0 loss is sampling and async drops), the proxy misses
# what goes around it, the env diff misses transient writes.
P_TRACE, P_PROXY, P_ENV = 0.85, 0.70, 0.60
BASE3 = (_logit(P_TRACE), _logit(P_PROXY), _logit(P_ENV))

SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario(
            "l0_independent",
            "L0",
            200,
            BASE3,
            note="Three independent witnesses. Every log-linear model should cover here.",
        ),
        Scenario(
            "l0_heterogeneous",
            "L0",
            200,
            BASE3,
            types=(
                ActionType(0.7, (1.0, 1.0, 1.0)),
                ActionType(0.3, (-1.5, -1.5, -1.5)),
            ),
            note="Hard actions are hard for every witness: positive dependence by type.",
        ),
        Scenario(
            "l0_auditor_copies_trace",
            "L0",
            200,
            (*BASE3, _logit(0.15)),
            pairs={(0, 3): 3.5},
            note="W4 rebuilds actions from the agent's transcript, so it inherits W1's misses "
            "(H3).",
        ),
        Scenario(
            "l0_siloed",
            "L0",
            200,
            BASE3,
            types=(
                ActionType(0.3, (0.0, NEG_INF, NEG_INF)),
                ActionType(0.3, (NEG_INF, 0.0, NEG_INF)),
                ActionType(0.3, (NEG_INF, NEG_INF, 0.0)),
                ActionType(0.1, (0.0, 0.0, 0.0)),
            ),
            note="Each witness covers its own kind of action and they barely overlap. "
            "N is effectively unknown; any finite interval here should be suspect.",
        ),
        Scenario(
            "l1_hides_from_trace",
            "L1",
            200,
            BASE3,
            types=(
                ActionType(0.85, (0.0, 0.0, 0.0)),
                ActionType(0.15, (NEG_INF, 0.0, 0.0)),
            ),
            note="Prompted to hide: 15% of actions never reach the agent's own trace.",
        ),
        Scenario(
            "l2_bypasses_proxy",
            "L2",
            200,
            BASE3,
            types=(
                ActionType(0.85, (0.0, 0.0, 0.0)),
                ActionType(0.15, (NEG_INF, NEG_INF, 0.0)),
            ),
            note="Adaptive: hidden actions skip trace and proxy (code-exec side channel); "
            "the env diff is still independent of the policy.",
        ),
        Scenario(
            "l2_env_blind",
            "L2",
            200,
            BASE3,
            types=(
                ActionType(0.85, (0.0, 0.0, 0.0)),
                ActionType(0.15, (NEG_INF, NEG_INF, -2.0)),
            ),
            note="Adaptive and transient: hidden actions also mostly evade the env diff. "
            "No witness is independent of the policy; point estimates should fail (H2).",
        ),
    )
}
