#!/usr/bin/env python3
"""What does BIC selection cost `ll_bic`'s interval? Coverage with and without it.

Usage:
  uv run python scripts/selection_bootstrap.py                       # defaults (400 reps): ~8 min
  uv run python scripts/selection_bootstrap.py --reps 40 --boot-reps 100 --out /tmp/smoke.json

`ll_bic` profiles the likelihood of the model BIC picked, as if the model had
been fixed in advance. `ll_bic_boot` / `ll_bic_boot_pct`
(src/lacuna/estimators/selection.py) redo the selection inside every
bootstrap resample. `ll_average` mixes the models instead of picking one.

Two settings, three scores per method (coverage with its Monte Carlo SE,
mean width over bounded intervals, gap detection = share whose lower end
exceeds the union count, the claim an auditor acts on):

  sim      L0/L1 scenarios of lacuna.sim, 3 hard witnesses, fresh table per rep
  confirm  the `confirm` episodes at the registered loss config fin_mid
           (HELDOUT pool, oracle linkage), episode bootstrap as in
           scripts/thin_episodes.py --bootstrap-episodes: resample episodes,
           thin, count, estimate

Reps are independent, each seeded by (seed, config, rep), so the result does
not depend on --workers. The nested bootstrap has --boot-reps inner draws
per outer rep, so its endpoints carry inner Monte Carlo noise of order
1/sqrt(boot_reps) in their quantile ranks; the coverage MCSE is the outer one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thin_episodes import load, summarise  # noqa: E402

from lacuna.estimators import baselines, loglinear, selection  # noqa: E402
from lacuna.histories import keep_witnesses  # noqa: E402
from lacuna.sim import SCENARIOS  # noqa: E402
from lacuna.thinning import HELDOUT, tables, thin  # noqa: E402

OUT = ROOT / "artifacts" / "robustness" / "selection-coverage.json"
DEFAULT_REPS = 400  # what artifacts/robustness/selection-coverage.json used
DEFAULT_BOOT_REPS = 300
SIM_CONFIGS = (
    "l0_independent",
    "l0_heterogeneous",
    "l0_auditor_copies_trace",
    "l0_siloed",
    "l1_hides_from_trace",
)
CONFIRM_RUN, CONFIRM_CFG = "confirm", "fin_mid"
METHODS = (
    "ll_bic",
    "ll_average",
    "ll_bic_boot",
    "ll_bic_boot_pct",
    "ll_bic_boot_sat",
    "ll_bic_boot_sat_pct",
)

_EPISODES = None


def _episodes():
    global _EPISODES
    if _EPISODES is None:
        _EPISODES = load(CONFIRM_RUN)
    return _EPISODES


def estimate(t3: np.ndarray, boot_reps: int, rng: np.random.Generator) -> dict:
    out = {
        "union_count": baselines.union_count(t3),
        "ll_bic": loglinear.best_bic(t3),
        "ll_average": loglinear.averaged(t3),
    }
    out.update(selection.bootstrap_bic(t3, reps=boot_reps, rng=rng))
    out.update(selection.bootstrap_bic(t3, reps=boot_reps, rng=rng, world="saturated"))
    return out


def one_rep(job: tuple) -> tuple[int, dict]:
    kind, name, ci, rep, seed, boot_reps = job
    rng = np.random.default_rng([seed, ci, rep])
    if kind == "sim":
        s = SCENARIOS[name]
        full = s.sample(rng)
        n_true = int(full.sum())
        full[0] = 0  # the only line that knows the truth; nothing below sees it
        t3 = keep_witnesses(full, (0, 1, 2))
    else:
        cfg = next(c for c in HELDOUT if c.name == CONFIRM_CFG)
        eps = _episodes()
        sample = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        n_true = sum(len(led) for led, _ in sample)
        t4 = np.sum(
            [tables(led, thin(recs, cfg, rng), "oracle")[0] for led, recs in sample], axis=0
        )
        t3 = keep_witnesses(t4, (0, 1, 2))
    return n_true, estimate(t3, boot_reps, rng)


def paired(rows: list[tuple[int, dict]], a: str, b: str) -> dict:
    """Coverage(a) - coverage(b) on the same tables, with the paired MCSE."""
    d = np.array([float(r[a].covers(n)) - float(r[b].covers(n)) for n, r in rows])
    return {
        "diff": round(float(d.mean()), 4),
        "mcse": round(float(d.std(ddof=1) / math.sqrt(len(d))), 4),
        "only_a_covers": int((d > 0).sum()),
        "only_b_covers": int((d < 0).sum()),
    }


def width_ratio(rows: list[tuple[int, dict]], a: str, b: str) -> float | None:
    """Median over tables where both are bounded of width(a) / width(b)."""
    r = [
        x[a].width / x[b].width for _, x in rows if x[a].bounded and x[b].bounded and x[b].width > 0
    ]
    return round(float(np.median(r)), 3) if r else None


def run_config(pool, kind: str, name: str, ci: int, reps: int, seed: int, boot_reps: int) -> dict:
    jobs = [(kind, name, ci, rep, seed, boot_reps) for rep in range(reps)]
    t0 = time.perf_counter()
    rows = pool.map(one_rep, jobs, chunksize=1)
    summary = summarise(rows)
    out = {
        "kind": kind,
        "reps": reps,
        "n_true_mean": round(float(np.mean([n for n, _ in rows])), 1),
        "estimators": {m: summary[m] for m in METHODS},
        "paired_coverage_vs_ll_bic": {m: paired(rows, m, "ll_bic") for m in METHODS[1:]},
        "median_width_ratio_vs_ll_bic": {m: width_ratio(rows, m, "ll_bic") for m in METHODS[1:]},
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    print(
        f"{kind}/{name}: "
        + ", ".join(f"{m} {summary[m]['coverage']:.3f}/{summary[m]['mean_width']}" for m in METHODS)
        + f"  ({out['runtime_s']}s)",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS, help="outer replicates per config")
    ap.add_argument(
        "--boot-reps",
        type=int,
        default=DEFAULT_BOOT_REPS,
        help="inner bootstrap draws per replicate",
    )
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--sim", action="append", help="repeatable; default all L0/L1 scenarios")
    ap.add_argument("--no-confirm", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite an artifact")

    t0 = time.perf_counter()
    result = {
        "reps": args.reps,
        "boot_reps": args.boot_reps,
        "seed": args.seed,
        "argv": sys.argv[1:],
        "parameters": {
            "workers": args.workers,
            "sim_configs": list(args.sim or SIM_CONFIGS),
            "confirm": not args.no_confirm,
        },
        "nominal": 0.95,
        "interval_method": {
            "ll_bic": "profile likelihood of the BIC-selected model (conditional on selection)",
            "ll_average": "BIC-weighted mixture of per-model profile likelihoods",
            "ll_bic_boot": (
                "nested parametric bootstrap from the BIC-selected model, "
                "selection redone per resample; basic (pivotal)"
            ),
            "ll_bic_boot_pct": "same draws, percentile",
            "ll_bic_boot_sat": (
                "as ll_bic_boot but resamples drawn from the all-pairwise (no 3-way) model"
            ),
            "ll_bic_boot_sat_pct": "same draws, percentile",
        },
        "configs": {},
    }
    with Pool(args.workers) as pool:
        for ci, name in enumerate(args.sim or SIM_CONFIGS):
            result["configs"][f"sim/{name}"] = {
                "level": SCENARIOS[name].level,
                **run_config(pool, "sim", name, ci, args.reps, args.seed, args.boot_reps),
            }
        if not args.no_confirm:
            ci = 100
            result["configs"][f"{CONFIRM_RUN}/{CONFIRM_CFG}/oracle"] = run_config(
                pool, "confirm", CONFIRM_CFG, ci, args.reps, args.seed, args.boot_reps
            )
    result["runtime_s"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} in {result['runtime_s']}s")


if __name__ == "__main__":
    main()
