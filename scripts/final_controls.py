#!/usr/bin/env python3
"""Controls that pin down the burst, outage and shared-quality results (D54).

1. Bursts: which property of the recorded episodes makes block loss fail?
   Blocks of 12 consecutive actions are dropped per log with probability 0.5,
   and the action order or the env diff's visibility is altered first:
   `actual`, `shuffle` (order permuted), `reverse`, `quarter` (shuffled only
   within each quarter of the episode), `visperm` (order kept, but which
   actions the env diff sees is permuted across the episode).
2. Whole-session outages in which the env diff can fail too, not only the
   gateway and trace.
3. `tier_bic` beyond the loss it was built for, and its tier count.

Exact matching. Post hoc.

Usage: uv run python scripts/final_controls.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import replay
from replay import replicates, summarise
from thin_episodes import load

from lacuna.estimators.stratified import tier_bic
from lacuna.thinning import ENV_VIEWS, HELDOUT, MNAR
from lacuna.witnesses import ENV, GATEWAY, TRACE

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(95_000, 95_008)
SEEDS_SLOW = range(96_000, 96_004)
OUT = ROOT / "artifacts" / "robustness" / "final-controls.json"
for _k in (4, 8, 10):
    replay.ESTIMATORS[f"tier_bic_{_k}"] = (
        lambda kk: lambda t3, p3, ts=None: tier_bic(ts, n_tiers=kk)
    )(_k)


@lru_cache(None)
def episodes(run: str) -> list:
    return load(run)


def burst_control(mode: str, view: str, width: int = 12):
    allowed = ENV_VIEWS[view]

    def thinner(recs, rng):
        op_of = {r.oracle: r.op for r in recs if r.oracle is not None and r.witness != ENV}
        ids = sorted({r.oracle for r in recs if r.oracle is not None})
        n = len(ids)
        rank = np.arange(n)
        visible = {o: allowed is None or op_of.get(o) in allowed for o in ids}
        if mode == "shuffle":
            rank = rng.permutation(n)
        elif mode == "reverse":
            rank = rank[::-1].copy()
        elif mode == "quarter":
            quarter = np.array([4 * i // max(n, 1) for i in range(n)])
            for q in range(4):
                idx = np.flatnonzero(quarter == q)
                rank[idx] = rng.permutation(idx)
        elif mode == "visperm":
            flags = [visible[o] for o in ids]
            perm = rng.permutation(n)
            visible = {o: flags[perm[i]] for i, o in enumerate(ids)}
        pos = {o: int(rank[i]) for i, o in enumerate(ids)}
        n_blocks = (n + width - 1) // width
        drop_gw = np.repeat(rng.random(n_blocks) < 0.5, width)[:n]
        drop_tr = np.repeat(rng.random(n_blocks) < 0.5, width)[:n]
        out = []
        for r in recs:
            if r.witness == GATEWAY and drop_gw[pos[r.oracle]]:
                continue
            if r.witness == TRACE and drop_tr[pos[r.oracle]]:
                continue
            if r.witness == ENV and not visible.get(r.oracle, True):
                continue
            out.append(r)
        return out

    return thinner


def outage(mode: str, view: str):
    """Per episode: which of (gateway, trace, env diff) are down for the whole session."""
    allowed = ENV_VIEWS[view]

    def states(rng):
        def coin(p=0.5):
            return rng.random() < p

        if mode == "gateway_trace":
            return coin(), coin(), False
        if mode == "all_three_independent":
            return coin(), coin(), coin()
        if mode == "gateway_with_diff":
            g = coin()
            return g, coin(), g
        if mode == "cascade":
            g = coin()
            t = coin(0.9 if g else 0.1)
            return g, t, coin(0.9 if t else 0.1)
        if mode == "bad_session":
            p = 0.8 if coin() else 0.2
            return coin(p), coin(p), coin(p)
        if mode == "all_three_together":
            g = coin()
            return g, g, g
        raise ValueError(mode)

    def thinner(recs, rng):
        g, t, e = states(rng)
        out = []
        for r in recs:
            if r.witness == GATEWAY and g:
                continue
            if r.witness == TRACE and t:
                continue
            if r.witness == ENV and (e or (allowed is not None and r.op not in allowed)):
                continue
            out.append(r)
        return out

    return thinner


def shared_quality(view: str):
    allowed = ENV_VIEWS[view]

    def thinner(recs, rng):
        q = rng.uniform(0.1, 0.9)
        return [
            r
            for r in recs
            if not (r.witness == ENV and allowed is not None and r.op not in allowed)
            and not rng.random() < q
        ]

    return thinner


def build(spec: tuple):
    kind, *args = spec
    return {"burst": burst_control, "outage": outage, "shared": shared_quality}[kind](*args)


def cell(job: tuple) -> dict:
    key, run, spec, est, seeds = job
    cfgs = {c.name: c for c in HELDOUT + MNAR}
    rows = []
    for s in seeds:
        if spec[0] == "config":
            rows += replicates(episodes(run), cfgs[spec[1]], "oracle", 100, s, est)
        else:
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
    for run, view in (
        ("confirm", "financial"),
        ("long-60", "financial"),
        ("long-120", "financial"),
        ("transfer", "full"),
    ):
        for mode in ("actual", "shuffle", "reverse", "quarter", "visperm"):
            jobs.append((f"burst/{run}/{mode}", run, ("burst", mode, view), base, SEEDS))
    for run, view in (("confirm", "financial"), ("transfer", "full")):
        for mode in (
            "gateway_trace",
            "all_three_independent",
            "gateway_with_diff",
            "cascade",
            "bad_session",
            "all_three_together",
        ):
            jobs.append((f"outage/{run}/{mode}", run, ("outage", mode, view), base, SEEDS))
    tiers = ("ll_bic", "tier_bic")
    for name in ("fin_mid", "fin_mid_rho100", "fin_mid_mnar"):
        jobs.append((f"tier/confirm/{name}", "confirm", ("config", name), tiers, SEEDS_SLOW))
    jobs.append(
        ("tier/confirm/burst", "confirm", ("burst", "actual", "financial"), tiers, SEEDS_SLOW)
    )
    jobs.append(("tier/transfer/burst", "transfer", ("burst", "actual", "full"), tiers, SEEDS_SLOW))
    sweep = ("tier_bic_4", "tier_bic", "tier_bic_8", "tier_bic_10")
    for run in ("confirm", "transfer"):
        jobs.append((f"tiercount/{run}", run, ("shared", "full"), sweep, SEEDS_SLOW))
    with Pool(12) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    for c in cells:
        print(
            c["key"],
            *(
                f"{e}={c[e]['coverage']}/w{c[e]['mean_width']}/b{c[e]['bias']}"
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
