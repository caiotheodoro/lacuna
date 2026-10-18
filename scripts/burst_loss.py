#!/usr/bin/env python3
"""Loss that arrives in bursts, under exact matching (D49).

Every loss model so far drops records one by one. Outages, sampling windows
and load spikes drop runs of consecutive actions. This drops each block of
`width` consecutive ledger ids with probability 0.5 (width 1 is the paper's
per-record loss), independently per log or jointly (one outage for both),
and scores the estimators under oracle linkage, so linkage plays no part.

Exploratory, after the confirmatory run.

Usage: uv run python scripts/burst_loss.py
"""

from __future__ import annotations

import json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from replay import replicates, summarise
from thin_episodes import load

from lacuna.thinning import ENV_VIEWS
from lacuna.witnesses import ENV, GATEWAY, TRACE

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (1, 2, 3, 6, 12)
P_DROP = 0.5
SEEDS = range(40_000, 40_020)
EST = ("ll_independence", "ll_bic")


@lru_cache(None)
def episodes(run: str) -> list:
    return load(run)


def burst_thinner(width: int, joint: bool, view: str):
    allowed = ENV_VIEWS[view]

    def thinner(recs, rng):
        ids = sorted({r.oracle for r in recs if r.oracle is not None})
        pos = {o: i for i, o in enumerate(ids)}
        n_blocks = (len(ids) + width - 1) // width

        def blocks():
            return np.repeat(rng.random(n_blocks) < P_DROP, width)[: len(ids)]

        drop_gw = blocks()
        drop_tr = drop_gw if joint else blocks()
        out = []
        for r in recs:
            if r.witness == GATEWAY and drop_gw[pos[r.oracle]]:
                continue
            if r.witness == TRACE and drop_tr[pos[r.oracle]]:
                continue
            if r.witness == ENV and allowed is not None and r.op not in allowed:
                continue
            out.append(r)
        return out

    return thinner


def cell(args: tuple) -> dict:
    run, width, joint, view = args
    rows = []
    for s in SEEDS:
        rows += replicates(
            episodes(run), None, "oracle", 100, s, EST, thinner=burst_thinner(width, joint, view)
        )
    return {
        "run": run,
        "width": width,
        "joint": joint,
        "env_view": view,
        **{e: summarise(rows, e) for e in EST},
    }


def main() -> None:
    jobs = [
        (run, w, joint, view)
        for run, views in (
            ("confirm", ("financial", "full")),
            ("long-30", ("financial",)),
            ("transfer", ("full",)),
        )
        for view in views
        for joint in (False, True)
        for w in WIDTHS
    ]
    with Pool(12) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    for c in cells:
        print(
            c["run"],
            c["env_view"],
            "joint" if c["joint"] else "indep",
            c["width"],
            c["ll_bic"]["coverage"],
            c["ll_independence"]["coverage"],
            flush=True,
        )
    out = {
        "p_drop": P_DROP,
        "linkage": "oracle",
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "reps_per_seed": 100,
        "cells": {
            f"{c['run']}/{c['env_view']}/{'joint' if c['joint'] else 'indep'}/w{c['width']}": c
            for c in cells
        },
    }
    dst = ROOT / "artifacts" / "robustness" / "burst-loss.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
