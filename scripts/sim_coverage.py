#!/usr/bin/env python3
"""Run the simulator coverage sweep and write artifacts/sim_coverage.json.

Usage:
  uv run python scripts/sim_coverage.py --reps 500 --seed 2026
  uv run python scripts/sim_coverage.py --scenario l0_independent --reps 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lacuna.coverage import sweep

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "sim_coverage.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scenario", action="append", help="repeatable; default all")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t0 = time.perf_counter()
    result = sweep(args.scenario, args.reps, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    for name, sc in result["scenarios"].items():
        print(f"\n{name} [{sc['level']}] N={sc['n_actions']}")
        head = ("estimator", "cover", "±", "unb", "width", "bias", "rmse")
        print(f"  {head[0]:38} " + " ".join(f"{h:>6}" for h in head[1:]))
        for est, m in sc["estimators"].items():
            width = "-" if m["mean_width"] is None else f"{m['mean_width']:.1f}"
            bias = "-" if m["bias"] is None else f"{m['bias']:.1f}"
            rmse = "-" if m["rmse"] is None else f"{m['rmse']:.1f}"
            print(
                f"  {est:38} {m['coverage']:6.3f} {m['coverage_mcse']:6.3f} "
                f"{m['unbounded_rate']:5.2f} {width:>7} {bias:>7} {rmse:>7}"
            )
    print(f"\nwrote {args.out} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
