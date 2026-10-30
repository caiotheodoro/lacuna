#!/usr/bin/env python3
"""Does a quasi-Poisson interval fix burst loss without breaking anything else? (D50)

D49 found that loss in blocks of consecutive actions leaves the point estimate
nearly unbiased but the profile interval too narrow. The refuter's fix scales
the likelihood-ratio cutoff by the Pearson dispersion of the independence fit
(`loglinear.quasi_bic`, `quasi_independence`). This scores plain and scaled
intervals on bursts, on the per-record loss of the held-out configurations,
on coupled loss, and on action-dependent loss, so the cost of the fix (width
where nothing is wrong) is measured beside its benefit.

Post hoc.

Usage: uv run python scripts/overdispersion.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from burst_loss import burst_thinner
from mnar_dose import op_keyed
from replay import replicates, summarise
from thin_episodes import load

from lacuna.thinning import HELDOUT, POOLS

ROOT = Path(__file__).resolve().parents[1]
EST = ("ll_bic", "ll_bic_qp", "ll_independence", "ll_independence_qp")
SEEDS = range(50_000, 50_010)
OUT = ROOT / "artifacts" / "robustness" / "overdispersion.json"


@lru_cache(None)
def episodes(run: str) -> list:
    return load(run)


def cell(job: tuple) -> dict:
    key, run, kind, arg, linkage = job
    rows = []
    for s in SEEDS:
        if kind == "burst":
            width, joint, view = arg
            rows += replicates(
                episodes(run), None, linkage, 100, s, EST, thinner=burst_thinner(width, joint, view)
            )
        else:
            rows += replicates(episodes(run), arg, linkage, 100, s, EST)
    phis = [r["ll_bic_qp"].detail.get("phi", 1.0) for r in rows]
    return {
        "key": key,
        "mean_phi": round(sum(phis) / len(phis), 2),
        **{e: summarise(rows, e) for e in EST},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=OUT, help="artifact to write (default: %(default)s)"
    )
    args = ap.parse_args(argv)
    held = {c.name: c for c in HELDOUT}
    infra = {c.name: c for c in POOLS["transfer"]}
    jobs = []
    for run, view in (("confirm", "financial"), ("long-30", "financial"), ("transfer", "full")):
        for w in (1, 3, 6, 12):
            jobs.append(
                (f"burst/{run}/{view}/indep/w{w}", run, "burst", (w, False, view), "oracle")
            )
    for w in (1, 12):
        jobs.append(
            (
                f"burst/confirm/financial/joint/w{w}",
                "confirm",
                "burst",
                (w, True, "financial"),
                "oracle",
            )
        )
    for lk in ("oracle", "signature"):
        jobs.append((f"heldout/confirm/fin_mid/{lk}", "confirm", "config", held["fin_mid"], lk))
    jobs.append(
        (
            "heldout/confirm/fin_mid_rho100/oracle",
            "confirm",
            "config",
            held["fin_mid_rho100"],
            "oracle",
        )
    )
    jobs.append(
        (
            "heldout/transfer/transfer_mid/oracle",
            "transfer",
            "config",
            infra["transfer_mid"],
            "oracle",
        )
    )
    jobs.append(
        (
            "heldout/transfer/transfer_mid_rho100/oracle",
            "transfer",
            "config",
            infra["transfer_mid_rho100"],
            "oracle",
        )
    )
    for m in (1.1, 1.2):
        jobs.append((f"mnar/confirm/op_x{m:g}/oracle", "confirm", "config", op_keyed(m), "oracle"))
    with Pool(12) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    for c in cells:
        print(
            c["key"],
            "phi",
            c["mean_phi"],
            *(f"{e}={c[e]['coverage']}/{c[e]['mean_width']}" for e in EST),
            flush=True,
        )
    out = {
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "reps_per_seed": 100,
        "cells": {c.pop("key"): c for c in cells},
    }
    dst = args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
