#!/usr/bin/env python3
"""Post-hoc mechanism checks for docs/MNAR-ROBUSTNESS.md, run after the four runs.

1. Which log-linear model BIC picks per op stratum, and whether the stratum
   ends up unbounded (`full_mid_hidden`, `fin_mid_mnar`; confirm episodes).
2. Which side `strat_bic` misses on (`fin_sampled`, `fin_sampled_rho100`;
   long-30 episodes, episodes resampled).

These were not predicted; they explain outcomes already seen. Writes
artifacts/thinning/mnar-diagnostics.json and prints it as markdown.

Usage: PYTHONPATH=src uv run python scripts/mnar_diagnostics.py [--out PATH]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from thin_episodes import load

from lacuna.estimators.loglinear import fit_conditional, model_space
from lacuna.estimators.stratified import (
    has_discordance,
    live_witnesses,
    strat_bic,
    stratum_density,
)
from lacuna.histories import keep_witnesses
from lacuna.thinning import CONFIGS, MNAR, MNAR_HIDDEN, tables, thin

ROOT = Path(__file__).resolve().parents[1]
LINKAGE = "signature"
OUT = ROOT / "artifacts" / "thinning" / "mnar-diagnostics.json"


def per_op(sample, cfg, rng) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for led, recs in sample:
        _, po = tables(led, thin(recs, cfg, rng), LINKAGE)
        for op, t in po.items():
            out[op] = out.get(op, 0) + keep_witnesses(t, (0, 1, 2))
    return out


def model_picks(run: str, cfg, reps: int, seed: int) -> dict:
    eps = load(run)
    rng = np.random.default_rng(seed)
    picks: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _ in range(reps):
        for op, t in per_op(eps, cfg, rng).items():
            live = live_witnesses(t, 3)
            if len(live) < 2 or not has_discordance(t, live):
                picks[op]["unidentified"] += 1
                continue
            tt = keep_witnesses(t, live)
            fits = [fit_conditional(tt, m) for m in model_space(len(live))]
            fits = [f for f in fits if f.converged]
            best = min(fits, key=lambda f: f.bic)
            name = "+".join(f"W{a + 1}:W{b + 1}" for a, b in best.pairs) or "independence"
            unbounded = stratum_density(tt) is None
            picks[op][f"{name}{' (unbounded)' if unbounded else ''}"] += 1
    return {op: dict(c.most_common()) for op, c in sorted(picks.items())}


def miss_sides(run: str, cfg, reps: int, seed: int) -> dict:
    eps = load(run)
    rng = np.random.default_rng(seed)
    c = collections.Counter()
    for _ in range(reps):
        idx = rng.integers(0, len(eps), len(eps))
        sample = [eps[i] for i in idx]
        n_true = sum(len(led) for led, _ in sample)
        e = strat_bic(per_op(sample, cfg, rng))
        c["lo above N" if e.lo > n_true else "hi below N" if e.hi < n_true else "covers"] += 1
        c["unbounded"] += not e.bounded
    return dict(c)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=OUT, help="artifact to write (default: %(default)s)"
    )
    args = ap.parse_args(argv)
    out = {
        "linkage": LINKAGE,
        "model_picks": {
            "reps": 20,
            "seed": 99,
            "run": "confirm",
            "episodes_resampled": False,
            "configs": {
                cfg.name: model_picks("confirm", cfg, 20, 99) for cfg in (MNAR_HIDDEN[0], MNAR[0])
            },
        },
        "miss_sides": {
            "reps": 40,
            "seed": 7,
            "run": "long-30",
            "episodes_resampled": True,
            "configs": {
                cfg.name: miss_sides("long-30", cfg, 40, 7) for cfg in (CONFIGS[3], CONFIGS[5])
            },
        },
    }
    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")

    mp = out["model_picks"]
    print(
        f"Source: `artifacts/thinning/mnar-diagnostics.json`, BIC-best model per stratum, "
        f"{mp['reps']} replicates of the {mp['run']} episodes, seed {mp['seed']}, {LINKAGE} "
        "linkage.\n"
    )
    print("| config | stratum | BIC-best model (count) |")
    print("|---|---|---|")
    for cfg, ops in mp["configs"].items():
        for op, picks in ops.items():
            cell = ", ".join(f"{k} ({v})" for k, v in picks.items())
            print(f"| `{cfg}` | `{op}` | {cell} |")
    ms = out["miss_sides"]
    print(
        f"\nSource: same file, `strat_bic` misses by side, {ms['reps']} replicates of the "
        f"{ms['run']} episodes resampled, seed {ms['seed']}.\n"
    )
    print("| config | covers | lo above N | hi below N | unbounded |")
    print("|---|---|---|---|---|")
    for cfg, c in ms["configs"].items():
        print(
            f"| `{cfg}` | {c.get('covers', 0)} | {c.get('lo above N', 0)} "
            f"| {c.get('hi below N', 0)} | {c.get('unbounded', 0)} |"
        )


if __name__ == "__main__":
    main()
