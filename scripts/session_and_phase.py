#!/usr/bin/env python3
"""Where the dispersion-scaled interval still fails, and what the failures are (D53).

Three checks from the D52 refuter:

1. Phase structure. On the long-episode pools the env-blind operations come
   in runs (the start of a long queue is mostly non-financial work), so
   blocks of consecutive lost lines fall on exactly the actions the env diff
   cannot see. Shuffling the order of actions inside each episode before
   dropping blocks removes that, and plain `ll_bic` recovers.
2. Whole-session outages: each log is down for a whole episode with
   probability 0.5. `ll_bic` was already nominal there; the dispersion
   correction widens by several times and stops detecting the gap.
3. Session quality shared by gateway, trace and env diff. Pooled intervals
   fail; ranking episodes by how much the logs saw and stratifying on tiers
   (`stratified.tier_bic`), or latent classes (`lc_gated`), recover part.

Exact matching throughout. Post hoc.

Usage: uv run python scripts/session_and_phase.py [--out PATH]
"""

from __future__ import annotations

import argparse
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
SEEDS = range(80_000, 80_010)
SEEDS_SLOW = range(81_000, 81_004)
OUT = ROOT / "artifacts" / "robustness" / "session-and-phase.json"


@lru_cache(None)
def episodes(run: str) -> list:
    return load(run)


def block_thinner(width: int, view: str, shuffle: bool):
    allowed = ENV_VIEWS[view]

    def thinner(recs, rng):
        ids = sorted({r.oracle for r in recs if r.oracle is not None})
        rank = rng.permutation(len(ids)) if shuffle else np.arange(len(ids))
        pos = {o: int(rank[i]) for i, o in enumerate(ids)}
        n_blocks = (len(ids) + width - 1) // width
        drop_gw = np.repeat(rng.random(n_blocks) < 0.5, width)[: len(ids)]
        drop_tr = np.repeat(rng.random(n_blocks) < 0.5, width)[: len(ids)]
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


def session_outage(joint: bool, view: str):
    allowed = ENV_VIEWS[view]

    def thinner(recs, rng):
        gw_down = rng.random() < 0.5
        tr_down = gw_down if joint else rng.random() < 0.5
        out = []
        for r in recs:
            if r.witness == GATEWAY and gw_down:
                continue
            if r.witness == TRACE and tr_down:
                continue
            if r.witness == ENV and allowed is not None and r.op not in allowed:
                continue
            out.append(r)
        return out

    return thinner


def shared_quality(view: str):
    allowed = ENV_VIEWS[view]

    def thinner(recs, rng):
        q = rng.uniform(0.1, 0.9)
        out = []
        for r in recs:
            if r.witness == ENV and allowed is not None and r.op not in allowed:
                continue
            if rng.random() < q:
                continue
            out.append(r)
        return out

    return thinner


def build(spec: tuple):
    kind, *args = spec
    return {"block": block_thinner, "outage": session_outage, "shared": shared_quality}[kind](*args)


def cell(job: tuple) -> dict:
    key, run, spec, est, seeds = job
    rows = []
    for s in seeds:
        rows += replicates(episodes(run), None, "oracle", 100, s, est, thinner=build(spec))
    return {"key": key, **{e: summarise(rows, e) for e in est}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=OUT, help="artifact to write (default: %(default)s)"
    )
    args = ap.parse_args(argv)
    base = ("ll_bic", "ll_bic_hom")
    jobs = []
    for run in ("confirm", "long-60", "long-120"):
        for shuffle in (False, True):
            tag = "shuffled" if shuffle else "actual"
            jobs.append(
                (
                    f"phase/{run}/{tag}",
                    run,
                    ("block", 12, "financial", shuffle),
                    base,
                    SEEDS,
                )
            )
    for run, view in (("confirm", "financial"), ("long-60", "financial"), ("transfer", "full")):
        for joint in (False, True):
            tag = "joint" if joint else "indep"
            jobs.append(
                (
                    f"outage/{run}/{tag}",
                    run,
                    ("outage", joint, view),
                    base,
                    SEEDS,
                )
            )
    slow = ("ll_bic", "ll_bic_hom", "lc_gated", "tier_bic")
    for run in ("confirm", "transfer"):
        jobs.append((f"shared/{run}", run, ("shared", "full"), slow, SEEDS_SLOW))
    with Pool(12) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    for c in cells:
        print(
            c["key"],
            *(
                f"{e}={c[e]['coverage']}/{c[e]['mean_width']}/g{c[e]['detects_gap']}"
                for e in c
                if e != "key"
            ),
            flush=True,
        )
    out = {
        "linkage": "oracle",
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "seeds_slow": [SEEDS_SLOW.start, SEEDS_SLOW.stop - 1],
        "reps_per_seed": 100,
        "cells": {c.pop("key"): c for c in cells},
    }
    dst = args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
