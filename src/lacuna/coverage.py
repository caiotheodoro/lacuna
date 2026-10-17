"""Coverage sweep: run every estimator on many simulated tables, score against truth.

This is the week-4 gate in docs/PLAN.md. If the log-linear estimators cannot
cover at L0 on tables they were designed for, nothing downstream is worth
plumbing, and the project ships the benchmark alone.

Coverage is reported with its Monte Carlo standard error, because "0.93" from
500 replicates is 0.93 +- 0.011, and a nominal-95% claim lives or dies inside
that margin.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from lacuna.estimators import Estimate, baselines, bounds, diagnostics, latent_class, loglinear
from lacuna.histories import WITNESSES, keep_witnesses
from lacuna.sim import SCENARIOS, Scenario

PER_SET: dict[str, Callable[[np.ndarray], Estimate]] = {
    "union_bound": baselines.union_bound,
    "union_count": baselines.union_count,
    "ll_independence": loglinear.independence,
    "ll_bic": loglinear.best_bic,
    "ll_average": loglinear.averaged,
    "lc": latent_class.latent_class,
    "lc_gated": diagnostics.lc_gated,
}
# Sensitivity dial for the env-anchored bounds. Only run on the hard-witness
# set, where the env diff is witness index 2.
ANCHOR_GAMMAS = (1.0, 2.0, 4.0, 8.0)


def witness_sets(s: Scenario) -> list[tuple[int, ...]]:
    sets = [(0, 1, 2)]
    if s.k == 4:
        sets.append((0, 1, 2, 3))
    return sets


def label(keep: tuple[int, ...]) -> str:
    return "+".join(WITNESSES[w] for w in keep)


def run_once(s: Scenario, rng: np.random.Generator) -> tuple[int, dict[str, Estimate]]:
    full = s.sample(rng)
    n_true = int(full.sum())
    observed = full.copy()
    observed[0] = 0  # the only line that knows the truth; nothing below sees it

    out = {"trust_trace": baselines.trust_trace(observed)}
    if s.k == 4:
        out["auditor_count"] = baselines.auditor_count(observed)
    for keep in witness_sets(s):
        table = keep_witnesses(observed, keep)
        for name, fn in PER_SET.items():
            out[f"{name}@{label(keep)}"] = fn(table)
        if keep == (0, 1, 2):
            for g in ANCHOR_GAMMAS:
                e = bounds.anchor_bounds(table, gamma=g, anchor=2)
                out[f"{e.name}@{label(keep)}"] = e
    return n_true, out


def summarise(rows: list[tuple[int, Estimate]]) -> dict:
    reps = len(rows)
    cover = sum(e.covers(n) for n, e in rows) / reps
    finite = [(n, e) for n, e in rows if math.isfinite(e.point)]
    bounded = [e for _, e in rows if e.bounded]
    err = np.array([e.point - n for n, e in finite], dtype=float)
    return {
        "coverage": round(cover, 4),
        "coverage_mcse": round(math.sqrt(max(cover * (1 - cover), 1e-12) / reps), 4),
        "unbounded_rate": round(1 - len(bounded) / reps, 4),
        "mean_width": round(float(np.mean([e.width for e in bounded])), 2) if bounded else None,
        "bias": round(float(err.mean()), 2) if len(err) else None,
        "rmse": round(float(np.sqrt((err**2).mean())), 2) if len(err) else None,
        "finite_points": len(finite),
    }


def sweep(names: list[str] | None, reps: int, seed: int) -> dict:
    results = {}
    for i, name in enumerate(names or list(SCENARIOS)):
        s = SCENARIOS[name]
        # One child stream per scenario, so adding a scenario does not
        # reshuffle the draws of the ones before it.
        rng = np.random.default_rng([seed, i])
        per_est: dict[str, list[tuple[int, Estimate]]] = {}
        for _ in range(reps):
            n_true, ests = run_once(s, rng)
            for est_name, e in ests.items():
                per_est.setdefault(est_name, []).append((n_true, e))
        results[name] = {
            "level": s.level,
            "note": s.note,
            "n_actions": s.n_actions,
            "estimators": {k: summarise(v) for k, v in per_est.items()},
        }
    return {"reps": reps, "seed": seed, "nominal": 0.95, "scenarios": results}
