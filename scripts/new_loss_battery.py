#!/usr/bin/env python3
"""Defects in the lines a log wrote: duplicates, phantoms, reordering, drift.

The loss taxonomy so far only removes records (per-record, coupled, exclusive,
op-keyed, hidden-key, bursts, outages, session quality, phase). Real logs also
fail the other way. This applies, after `thin`, four opt-in faults from
`lacuna.thinning` and reads `ll_bic` under four linkages:

- `dup`: a line written twice by one witness (a retry), per witness.
- `phantom`: a line for an action that never happened (a failed call logged as
  success, or a forged line), per witness, as a `clone` of a real record's
  signature or a `forged` one nothing else shares.
- `reorder`: adjacent lines swap, or each line's position is jittered, per
  witness. Signature and oracle linkage never read order (their tables are
  proved order-blind in tests/test_new_loss.py); call-id linkage reads it only
  to break ties; the last-writer arm reads the merged log order.
- `drift`: the gateway and trace loss rate rises linearly over the episode
  (`DriftConfig`), with the same mean.

Linkages (scripts/linkage_arms.py): `oracle`, `signature`, `callid` (trace and
gateway linked by call id, W3 by signature) and `callid_lastwriter`. All four
read the same thinned and corrupted records, so the comparison is paired.

Seed conventions follow linkage_arms.py (30_000.. precise, 31_000.. quick;
`thin` draws are identical, so the unfaulted rows reproduce
artifacts/robustness/linkage-arms.json). Corruption draws come from their own
generator, seeded from the same seed, so they never shift the loss draws.
Exploratory, post hoc, exact matching.

Usage: uv run python scripts/new_loss_battery.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from linkage_arms import table3
from replay import ESTIMATORS, summarise
from thin_episodes import load

from lacuna.thinning import HELDOUT, POOLS, Config, Corruption, DriftConfig, corrupt, thin

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("oracle", "signature", "callid", "callid_lastwriter")
EST = ("ll_independence", "ll_bic")
PRECISE = range(30_000, 30_020)
QUICK = range(31_000, 31_010)
REPS = 100
WITNESS_SETS = {
    "trace": (0,),
    "gateway": (1,),
    "env": (2,),
    "all": (0, 1, 2),
}


def per_witness(rate: float, ws: tuple[int, ...]) -> tuple[float, float, float]:
    return tuple(rate if w in ws else 0.0 for w in range(3))  # type: ignore[return-value]


def mechanisms() -> dict[str, tuple[Corruption, float]]:
    """name -> (corruption, drift). Rates are per line the witness wrote."""
    out: dict[str, tuple[Corruption, float]] = {"none": (Corruption(), 0.0)}
    for wname, ws in WITNESS_SETS.items():
        for rate in (0.05, 0.15, 0.30):
            out[f"dup/{wname}/{rate:g}"] = (Corruption(dup=per_witness(rate, ws)), 0.0)
        for kind in ("clone", "forged"):
            for rate in (0.02, 0.05, 0.10):
                out[f"phantom_{kind}/{wname}/{rate:g}"] = (
                    Corruption(phantom=per_witness(rate, ws), phantom_kind=kind),
                    0.0,
                )
    out["reorder/swap0.1"] = (Corruption(swap_p=0.1), 0.0)
    out["reorder/swap0.5"] = (Corruption(swap_p=0.5), 0.0)
    out["reorder/jitter3"] = (Corruption(jitter=3.0), 0.0)
    out["reorder/jitter10"] = (Corruption(jitter=10.0), 0.0)
    out["reorder/swap0.5+jitter10"] = (Corruption(swap_p=0.5, jitter=10.0), 0.0)
    out["drift/0.4"] = (Corruption(), 0.4)
    out["drift/0.8"] = (Corruption(), 0.8)
    return out


def bases() -> list[tuple[str, Config]]:
    """(run, config): the two registered cells, the same cell with W3 live on
    confirm, and a nothing-hidden control (exclusive loss, so any detected gap
    is false), as in linkage_arms.py."""
    fin_mid = next(c for c in HELDOUT if c.name == "fin_mid")
    full_mid = Config("full_mid", 0.5, 0.5, "full")
    transfer_mid = next(c for c in POOLS["transfer"] if c.name == "transfer_mid")
    anti = Config("anti0.4_full", 0.4, 0.4, "full", anti=True)
    return [
        ("confirm", fin_mid),
        ("transfer", transfer_mid),
        ("confirm", full_mid),
        ("confirm", anti),
        ("transfer", anti),
    ]


# The control gets a thinner grid: it answers one question (false gaps).
CONTROL_MECHANISMS = (
    "none",
    "dup/all/0.15",
    "phantom_clone/all/0.05",
    "phantom_forged/all/0.05",
    "reorder/swap0.5+jitter10",
    "drift/0.8",
)


def seeds_for(cfg: Config, mech: str) -> range:
    """Unfaulted registered cells run at the precise seeds, to reproduce
    linkage-arms.json; everything else at the quick seeds."""
    return PRECISE if mech == "none" and not cfg.anti and cfg.name != "full_mid" else QUICK


@lru_cache(None)
def episodes(run: str) -> list:
    return load(run)


def cell(job: tuple) -> dict:
    run, cfg, seeds, mech, reps = job
    corr, drift = mechanisms()[mech]
    thin_cfg = (
        DriftConfig(**{**cfg.__dict__, "drift": drift}) if drift else cfg
    )  # fresh config, same fields plus drift
    eps = episodes(run)
    rows: dict[str, list[dict]] = {a: [] for a in ARMS}
    for s in seeds:
        rng = np.random.default_rng(s)
        crng = np.random.default_rng([777, s])
        for _ in range(reps):
            idx = rng.integers(0, len(eps), len(eps))
            sample = [eps[i] for i in idx]
            n = sum(len(led) for led, _ in sample)
            t3 = {a: np.zeros(8, dtype=np.int64) for a in ARMS}
            for led, recs in sample:
                kept = corrupt(thin(recs, thin_cfg, rng), corr, crng, cfg.env_view)
                for a in ARMS:
                    t3[a] += table3(led, kept, a)
            for a in ARMS:
                row = {"n": n, "union": int(t3[a][1:].sum())}
                for e in EST:
                    row[e] = ESTIMATORS[e](t3[a], {})
                rows[a].append(row)
    return {
        "run": run,
        "config": cfg.name,
        "mechanism": mech,
        "arms": {a: {e: summarise(rows[a], e) for e in EST} for a in ARMS},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 seeds x 5 reps, writes nothing")
    args = ap.parse_args()
    t0 = time.perf_counter()
    jobs = []
    for run, cfg in bases():
        for m in CONTROL_MECHANISMS if cfg.anti else mechanisms():
            sd = seeds_for(cfg, m)
            sd, reps = (list(sd)[:2], 5) if args.smoke else (sd, REPS)
            jobs.append((run, cfg, sd, m, reps))
    jobs.sort(key=lambda j: -len(j[2]))
    with Pool(12) as pool:
        cells = []
        for c in pool.imap_unordered(cell, jobs, chunksize=1):
            cells.append(c)
            b = c["arms"]["oracle"]["ll_bic"]
            print(
                f"{len(cells)}/{len(jobs)} {c['run']}/{c['config']}/{c['mechanism']} "
                f"oracle cov {b['coverage']} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    assert all(not math.isnan(c["arms"][a]["ll_bic"]["coverage"]) for c in cells for a in ARMS)
    if args.smoke:
        return
    cells.sort(key=lambda c: (c["run"], c["config"], c["mechanism"]))
    out = {
        "arms": list(ARMS),
        "estimators": list(EST),
        "seeds": {
            "precise": [PRECISE.start, PRECISE.stop - 1],
            "quick": [QUICK.start, QUICK.stop - 1],
            "note": "mechanism 'none' on fin_mid and transfer_mid uses precise, all else quick",
            "corruption": "default_rng([777, seed])",
        },
        "reps_per_seed": REPS,
        "rates": "per line the witness wrote; phantom rates are phantoms per written line",
        "cells": {f"{c['run']}/{c['config']}/{c['mechanism']}": c["arms"] for c in cells},
    }
    dst = ROOT / "artifacts" / "robustness" / "new-loss.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dst} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
