#!/usr/bin/env python3
"""H1's coverage at higher precision, by linkage, with BIC's model choice (D44).

The registered verdict is one loss seed of 100 replicates (MCSE about 0.024).
This reruns the two held-out configurations at 30 fresh loss seeds of 100
replicates each, under both linkages with the same random numbers, on the
registered 21-episode pool and the 24-episode pool with the lost episodes
restored. It also records which interactions `ll_bic` selected, and its
coverage and bias given each choice.

Post hoc. It cannot change the registered verdict (hypotheses.json).

Usage: uv run python scripts/precision.py [--reps N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from replay import replicates, summarise
from thin_episodes import load

from lacuna.thinning import HELDOUT

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("confirm", "confirm-plus-supp")
SEEDS = range(10_000, 10_030)  # disjoint from 4242 and 1-3 (D35)
REPS = 100
OUT = ROOT / "artifacts" / "robustness" / "precision.json"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=OUT, help="artifact to write (default: %(default)s)"
    )
    ap.add_argument("--reps", type=int, default=REPS, help="replicates per loss seed")
    args = ap.parse_args(argv)
    reps = args.reps
    out: dict = {"seeds": [SEEDS.start, SEEDS.stop - 1], "reps_per_seed": reps, "runs": {}}
    for run in RUNS:
        episodes = load(run)
        out["runs"][run] = {}
        for cfg in HELDOUT:
            for linkage in ("oracle", "signature"):
                rows = []
                for s in SEEDS:
                    rows += replicates(episodes, cfg, linkage, reps, s)
                choice: dict = defaultdict(list)
                for r in rows:
                    pairs = "+".join(f"W{a + 1}W{b + 1}" for a, b in r["ll_bic"].detail["pairs"])
                    choice[pairs or "independence"].append(r)
                cell = {
                    "ll_independence": summarise(rows, "ll_independence"),
                    "ll_bic": summarise(rows, "ll_bic"),
                    "ll_bic_selected": {
                        k: {"share": round(len(v) / len(rows), 3), **summarise(v, "ll_bic")}
                        for k, v in sorted(choice.items(), key=lambda kv: -len(kv[1]))
                    },
                }
                out["runs"][run][f"{cfg.name}/{linkage}"] = cell
                print(
                    run,
                    cfg.name,
                    linkage,
                    "bic",
                    cell["ll_bic"]["coverage"],
                    "ind",
                    cell["ll_independence"]["coverage"],
                    {k: v["share"] for k, v in cell["ll_bic_selected"].items()},
                    flush=True,
                )
    dst = args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
