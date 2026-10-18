#!/usr/bin/env python3
"""False gaps across drop rates, env views and episode pools (D44).

D42 measured false gaps at one exclusive-loss rate (0.4), on one pool, at one
seed. This sweeps the rate and the env view over four pools, under both
linkages, plus three nulls where nothing is hidden and losses are independent
(one log complete). Nothing is hidden in any cell by construction, so any
`detects_gap` is a false positive.

Exploratory, after the confirmatory run.

Usage: uv run python scripts/falsegap_battery.py
"""

from __future__ import annotations

import json
from pathlib import Path

from replay import replicates, summarise
from thin_episodes import load

from lacuna.thinning import Config

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("confirm", "confirm-plus-supp", "long-30", "long-60")
REPS = 100
SEED = 777


def configs() -> list[Config]:
    out = [
        Config(f"anti{p:g}_{env}", p, p, env, anti=True)
        for p in (0.2, 0.3, 0.4, 0.5)
        for env in ("full", "financial")
    ]
    out += [
        Config("gateway_complete_financial", 0.0, 0.5, "financial"),
        Config("gateway_complete_full", 0.0, 0.5, "full"),
        Config("trace_complete_financial", 0.5, 0.0, "financial"),
    ]
    return out


def main() -> None:
    out: dict = {"reps": REPS, "seed": SEED, "pools": {}}
    for run in POOLS:
        episodes = load(run)
        out["pools"][run] = {"n": sum(len(led) for led, _ in episodes), "configs": {}}
        for cfg in configs():
            for linkage in ("oracle", "signature"):
                rows = replicates(episodes, cfg, linkage, REPS, SEED)
                cell = {e: summarise(rows, e) for e in ("ll_independence", "ll_bic")}
                out["pools"][run]["configs"][f"{cfg.name}/{linkage}"] = cell
                print(run, cfg.name, linkage, "bic gap", cell["ll_bic"]["detects_gap"], flush=True)
    dst = ROOT / "artifacts" / "robustness" / "falsegap-battery.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
