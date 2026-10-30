#!/usr/bin/env python3
"""An interval that survives bursts and coupling together (D51).

D50 found no single quasi-Poisson variant handled both: dispersion from the
independence fit counts real dependence as dispersion, and dispersion from the
selected model lets spurious interactions absorb the bursts. The refuter's fix
(`loglinear.quasi_bic_episodes`, `ll_qbic_hom`) takes the dispersion from
variation between episodes, selects the model by QBIC on that scale, and
scales the profile cutoff by it. This scores it against the plain and D50
intervals on per-record, coupled, burst, joint-burst and action-dependent
loss, and on small audits.

Post hoc.

Usage: uv run python scripts/episode_dispersion.py
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

from lacuna.thinning import HELDOUT, POOLS, Config

ROOT = Path(__file__).resolve().parents[1]
EST = ("ll_bic", "ll_bic_qp", "ll_independence_qp", "ll_qbic_hom")
SEEDS = range(60_000, 60_010)
OUT = ROOT / "artifacts" / "robustness" / "episode-dispersion.json"


@lru_cache(None)
def episodes(run: str) -> list:
    return load(run)


def cell(job: tuple) -> dict:
    key, run, kind, arg, linkage, k = job
    rows = []
    for s in SEEDS:
        if kind == "burst":
            width, joint, view = arg
            rows += replicates(
                episodes(run),
                None,
                linkage,
                100,
                s,
                EST,
                thinner=burst_thinner(width, joint, view),
                n_episodes=k,
            )
        else:
            rows += replicates(episodes(run), arg, linkage, 100, s, EST, n_episodes=k)
    phis = [r["ll_qbic_hom"].detail.get("phi", 1.0) for r in rows]
    return {
        "key": key,
        "mean_phi_hom": round(sum(phis) / len(phis), 2),
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
    rho50 = Config("fin_mid_rho50", 0.5, 0.5, "financial", rho=0.5)
    jobs = [
        ("perrecord/confirm/fin_mid/oracle", "confirm", "config", held["fin_mid"], "oracle", None),
        (
            "perrecord/confirm/fin_mid/signature",
            "confirm",
            "config",
            held["fin_mid"],
            "signature",
            None,
        ),
        (
            "perrecord/transfer/transfer_mid/oracle",
            "transfer",
            "config",
            infra["transfer_mid"],
            "oracle",
            None,
        ),
        (
            "perrecord/transfer/transfer_mid/signature",
            "transfer",
            "config",
            infra["transfer_mid"],
            "signature",
            None,
        ),
        ("coupled/confirm/fin_mid_rho50/oracle", "confirm", "config", rho50, "oracle", None),
        (
            "coupled/confirm/fin_mid_rho100/oracle",
            "confirm",
            "config",
            held["fin_mid_rho100"],
            "oracle",
            None,
        ),
        (
            "coupled/transfer/transfer_mid_rho100/oracle",
            "transfer",
            "config",
            infra["transfer_mid_rho100"],
            "oracle",
            None,
        ),
        ("mnar/confirm/op_x1.2/oracle", "confirm", "config", op_keyed(1.2), "oracle", None),
    ]
    for run, view in (("confirm", "financial"), ("long-30", "financial"), ("transfer", "full")):
        for w in (3, 12):
            jobs.append(
                (f"burst/{run}/{view}/indep/w{w}", run, "burst", (w, False, view), "oracle", None)
            )
    jobs.append(
        (
            "burst/confirm/financial/joint/w12",
            "confirm",
            "burst",
            (12, True, "financial"),
            "oracle",
            None,
        )
    )
    jobs.append(
        (
            "burst/transfer/full/indep/w12/signature",
            "transfer",
            "burst",
            (12, False, "full"),
            "signature",
            None,
        )
    )
    for k in (4, 8):
        jobs.append(
            (f"small/confirm/fin_mid/k{k}", "confirm", "config", held["fin_mid"], "oracle", k)
        )
        jobs.append(
            (
                f"small/confirm/burst_w12/k{k}",
                "confirm",
                "burst",
                (12, False, "financial"),
                "oracle",
                k,
            )
        )
    with Pool(12) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    for c in cells:
        print(
            c["key"],
            "phi_hom",
            c["mean_phi_hom"],
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
