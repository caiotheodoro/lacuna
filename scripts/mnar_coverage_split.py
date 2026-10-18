#!/usr/bin/env python3
"""Split strat_bic's coverage under hidden-key MNAR into refusals and finite intervals.

`artifacts/thinning/confirm-mnar-hidden.json` stores only aggregates: coverage
0.51 with 0.52 of replicates unbounded. That does not say whether the finite
intervals ever cover. This replays the same replicates (same RNG stream as
scripts/thin_episodes.py, which no estimator draws from) and counts both.

Usage: uv run python scripts/mnar_coverage_split.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from lacuna.estimators.stratified import strat_bic
from lacuna.histories import keep_witnesses
from lacuna.thinning import POOLS, tables, thin

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "thinning" / "confirm-mnar-hidden.json"


def main() -> None:
    from thin_episodes import load

    src = json.loads(SOURCE.read_text())
    episodes = load(src["run"])
    pool = POOLS[src["pool"]]
    out = {"source": SOURCE.name, "seed": src["seed"], "reps": src["reps"], "configs": {}}
    for ci, cfg in enumerate(pool):
        linkage = "signature"
        rng = np.random.default_rng([src["seed"], ci, linkage == "signature"])
        counts = {"unbounded_cover": 0, "unbounded_miss": 0, "finite_cover": 0, "finite_miss": 0}
        for _ in range(src["reps"]):
            idx = rng.integers(0, len(episodes), len(episodes))
            sample = [episodes[i] for i in idx]
            n = sum(len(led) for led, _ in sample)
            per_op_all: dict = {}
            for led, recs in sample:
                _, per_op = tables(led, thin(recs, cfg, rng), linkage)
                for op, t in per_op.items():
                    per_op_all[op] = per_op_all.get(op, 0) + t
            e = strat_bic({op: keep_witnesses(t, (0, 1, 2)) for op, t in per_op_all.items()})
            kind = "unbounded" if math.isinf(e.hi) else "finite"
            counts[f"{kind}_{'cover' if e.covers(n) else 'miss'}"] += 1
        # Must reproduce the stored aggregates, or this is not the same run.
        stored = src["configs"][f"{cfg.name}/{linkage}"]["estimators"]["strat_bic"]
        cov = (counts["unbounded_cover"] + counts["finite_cover"]) / src["reps"]
        unb = (counts["unbounded_cover"] + counts["unbounded_miss"]) / src["reps"]
        assert round(cov, 4) == stored["coverage"], (cov, stored["coverage"])
        assert round(unb, 4) == stored["unbounded_rate"], (unb, stored["unbounded_rate"])
        counts["unbounded"] = counts["unbounded_cover"] + counts["unbounded_miss"]
        counts["finite"] = counts["finite_cover"] + counts["finite_miss"]
        out["configs"][f"{cfg.name}/{linkage}"] = counts
        print(cfg.name, counts, flush=True)
    dst = ROOT / "artifacts" / "mnar-coverage-split.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
