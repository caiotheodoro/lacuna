"""Shared replay loop for the robustness scripts (D44).

`scripts/thin_episodes.py` runs every estimator on a fixed pool of
configurations. The robustness checks need many loss seeds, configurations
outside the pools, and a few estimators only, so they share this smaller
loop. It thins recorded episodes exactly as `thin_episodes.py` does (episodes
resampled with replacement, then `thin`, then `tables`); nothing here calls a
model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

from lacuna.estimators import loglinear
from lacuna.estimators.stratified import strat_bic
from lacuna.histories import keep_witnesses
from lacuna.thinning import Config, tables, thin
from lacuna.witnesses import Record

# Each takes the pooled 3-witness table, the per-op tables and the per-episode
# tables; most use only the first.
ESTIMATORS: dict[str, Callable] = {
    "ll_independence": lambda t3, p3, ts=None: loglinear.independence(t3),
    "ll_bic": lambda t3, p3, ts=None: loglinear.best_bic(t3),
    "strat_bic": lambda t3, p3, ts=None: strat_bic(p3),
    "ll_bic_qp": lambda t3, p3, ts=None: loglinear.quasi_bic(t3),
    "ll_independence_qp": lambda t3, p3, ts=None: loglinear.quasi_independence(t3),
    "ll_qbic_hom": lambda t3, p3, ts=None: loglinear.quasi_bic_episodes(ts),
    "ll_bic_hom": lambda t3, p3, ts=None: loglinear.bic_episodes(ts),
}


def replicates(
    episodes: list,
    cfg: Config | None,
    linkage: str,
    reps: int,
    seed: int,
    estimators: Iterable[str] = ("ll_independence", "ll_bic"),
    thinner: Callable[[list[Record], np.random.Generator], list[Record]] | None = None,
    n_episodes: int | None = None,
) -> list[dict]:
    """One row per replicate: true N, the linked union, and each estimate.

    `n_episodes` resamples that many episodes per replicate instead of the
    pool's size, to study small audits.
    """
    rng = np.random.default_rng(seed)
    names = tuple(estimators)
    k = n_episodes or len(episodes)
    rows = []
    for _ in range(reps):
        idx = rng.integers(0, len(episodes), k)
        sample = [episodes[i] for i in idx]
        n = sum(len(led) for led, _ in sample)
        parts, per_op = [], {}
        for led, recs in sample:
            kept = thinner(recs, rng) if thinner else thin(recs, cfg, rng)
            t4, po = tables(led, kept, linkage)
            parts.append(t4)
            for op, t in po.items():
                per_op[op] = per_op.get(op, 0) + t
        ts = [keep_witnesses(t, (0, 1, 2)) for t in parts]
        t3 = np.sum(ts, axis=0)
        p3 = {op: keep_witnesses(t, (0, 1, 2)) for op, t in per_op.items()}
        row = {"n": n, "union": int(t3[1:].sum())}
        for name in names:
            row[name] = ESTIMATORS[name](t3, p3, ts)
        rows.append(row)
    return rows


def summarise(rows: list[dict], name: str) -> dict:
    """Coverage (with MCSE), gap detection, refusal rate, bias, width, hidden.

    Bias is over finite points and width over bounded intervals, as in
    `thin_episodes.summarise`; coverage counts an unbounded interval as
    covering when its lower end is at most N.
    """
    k = len(rows)
    ests = [r[name] for r in rows]
    cover = float(np.mean([e.covers(r["n"]) for e, r in zip(ests, rows, strict=True)]))
    finite = [(e, r) for e, r in zip(ests, rows, strict=True) if math.isfinite(e.hi)]
    pts = [e.point - r["n"] for e, r in zip(ests, rows, strict=True) if math.isfinite(e.point)]
    return {
        "reps": k,
        "coverage": round(cover, 3),
        "coverage_mcse": round(math.sqrt(max(cover * (1 - cover), 1e-12) / k), 4),
        "detects_gap": round(
            float(np.mean([e.lo > r["union"] for e, r in zip(ests, rows, strict=True)])), 3
        ),
        "unbounded_rate": round(1 - len(finite) / k, 3),
        "n_finite": len(finite),
        "coverage_among_finite": (
            round(float(np.mean([e.covers(r["n"]) for e, r in finite])), 3) if finite else None
        ),
        "bias": round(float(np.mean(pts)), 2) if pts else None,
        "mean_width": round(float(np.mean([e.hi - e.lo for e, _ in finite])), 1)
        if finite
        else None,
        "hidden": round(float(np.mean([r["n"] - r["union"] for r in rows])), 2),
    }
