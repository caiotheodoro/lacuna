#!/usr/bin/env python3
"""Three intervals against the loss patterns that broke each one (D52).

`ll_bic` (plain), `ll_qbic_hom` (D51: QBIC selection and cutoff on the
between-episode dispersion) and `ll_bic_hom` (D52: plain BIC selection, cutoff
scaled by that dispersion). The cells are the ones the D51 refuter used to
break `ll_qbic_hom`: another pool (long-60), session-level variation in
capture rates that needs no widening, session quality shared by all three
logs, pooled environments, and low-loss full-view bursts where gap detection
matters. Exact matching throughout.

Post hoc.

Usage: uv run python scripts/interval_comparison.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from burst_loss import burst_thinner
from replay import replicates, summarise
from thin_episodes import load

from lacuna.thinning import ENV_VIEWS, HELDOUT
from lacuna.witnesses import ENV, GATEWAY, TRACE

ROOT = Path(__file__).resolve().parents[1]
EST = ("ll_bic", "ll_qbic_hom", "ll_bic_hom")
SEEDS = range(70_000, 70_010)
OUT = ROOT / "artifacts" / "robustness" / "interval-comparison.json"


@lru_cache(None)
def episodes(runs: tuple) -> list:
    out = []
    for r in runs:
        out += load(r)
    return out


def session_thinner(mode: str, view: str):
    """Each episode draws its own drop rates from U(0.1, 0.9), then loses per record.

    `gateway`: only the gateway's rate varies (trace fixed at 0.5).
    `both`: gateway and trace draw independent rates.
    `shared`: one session quality hits gateway, trace and env diff alike.
    """
    allowed = ENV_VIEWS[view]

    def thinner(recs, rng):
        u1, u2 = rng.uniform(0.1, 0.9, 2)
        pg, pt, pe = {"gateway": (u1, 0.5, 0.0), "both": (u1, u2, 0.0), "shared": (u1, u1, u1)}[
            mode
        ]
        out = []
        for r in recs:
            if r.witness == GATEWAY and rng.random() < pg:
                continue
            if r.witness == TRACE and rng.random() < pt:
                continue
            if r.witness == ENV:
                if allowed is not None and r.op not in allowed:
                    continue
                if rng.random() < pe:
                    continue
            out.append(r)
        return out

    return thinner


def cell(job: tuple) -> dict:
    key, runs, kind, arg = job
    eps = episodes(runs)
    rows = []
    for s in SEEDS:
        if kind == "config":
            rows += replicates(eps, arg, "oracle", 100, s, EST)
        elif kind == "burst":
            rows += replicates(eps, None, "oracle", 100, s, EST, thinner=burst_thinner(*arg))
        else:
            rows += replicates(eps, None, "oracle", 100, s, EST, thinner=session_thinner(*arg))
    phis = [r["ll_bic_hom"].detail.get("phi", 1.0) for r in rows]
    return {
        "key": key,
        "mean_phi_ep": round(sum(phis) / len(phis), 2),
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
    from lacuna.thinning import Config

    full_mid = Config("full_mid", 0.5, 0.5, "full")
    jobs = [
        ("perrecord/confirm", ("confirm",), "config", held["fin_mid"]),
        ("perrecord/long-60", ("long-60",), "config", held["fin_mid"]),
        ("perrecord/confirm/full", ("confirm",), "config", full_mid),
        ("perrecord/backoffice+infra/full", ("confirm", "transfer"), "config", full_mid),
        ("coupled/confirm/rho100", ("confirm",), "config", held["fin_mid_rho100"]),
        ("burst/confirm/w12", ("confirm",), "burst", (12, False, "financial")),
        ("burst/long-60/w12", ("long-60",), "burst", (12, False, "financial")),
        ("burst/transfer/w12", ("transfer",), "burst", (12, False, "full")),
        ("joint/confirm/w12", ("confirm",), "burst", (12, True, "financial")),
        ("joint/long-60/w12", ("long-60",), "burst", (12, True, "financial")),
        ("burst/confirm/full/w12", ("confirm",), "burst", (12, False, "full")),
        ("session/gateway/confirm", ("confirm",), "session", ("gateway", "financial")),
        ("session/both/confirm", ("confirm",), "session", ("both", "financial")),
        ("session/shared/confirm/full", ("confirm",), "session", ("shared", "full")),
        ("session/shared/transfer/full", ("transfer",), "session", ("shared", "full")),
    ]
    with Pool(12) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    for c in cells:
        print(
            c["key"],
            "phi",
            c["mean_phi_ep"],
            *(f"{e}={c[e]['coverage']}/{c[e]['mean_width']}/g{c[e]['detects_gap']}" for e in EST),
            flush=True,
        )
    out = {
        "linkage": "oracle",
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "reps_per_seed": 100,
        "cells": {c.pop("key"): c for c in cells},
    }
    dst = args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
