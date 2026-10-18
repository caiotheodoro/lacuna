#!/usr/bin/env python3
"""Evaluate T1-T4 exactly as frozen in docs/PREREG-TRANSFER.md.

The thresholds are literals here because they were fixed before the data
existed. Changing one is a decision entry, not an edit. They are the
back-office H1-H4 thresholds, moved to the infra env's transfer pool.

Usage: uv run python scripts/test_transfer.py --run transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "transfer_mid"
COUPLED = "transfer_mid_rho100"
LINKAGE = "signature"  # the headline linkage: what an auditor can compute


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="transfer")
    args = ap.parse_args()
    r = json.loads((ROOT / "artifacts" / "thinning" / f"{args.run}.json").read_text())
    if r.get("pool") != "transfer":
        raise SystemExit(f"{args.run} was not run on the transfer pool.")
    if not r.get("bootstrap_episodes"):
        raise SystemExit(f"{args.run} did not resample episodes; the prereg requires it.")
    if r.get("seed") != 4242 or r.get("reps") != 100:
        raise SystemExit(f"{args.run} is not seed 4242 at 100 replicates, as registered.")

    def est(cfg: str, name: str) -> dict:
        return r["configs"][f"{cfg}/{LINKAGE}"]["estimators"][name]

    bic, ind = est(PRIMARY, "ll_bic"), est(PRIMARY, "ll_independence")
    bic_c, ind_c = est(COUPLED, "ll_bic"), est(COUPLED, "ll_independence")
    union = est(PRIMARY, "union_count")

    ind_drop = ind["coverage"] - ind_c["coverage"]
    bic_drop = bic["coverage"] - bic_c["coverage"]
    gap_lead = bic["detects_gap"] - union["detects_gap"]

    results = [
        ("T1 recovery", bic["coverage"] >= 0.90, f"ll_bic coverage {bic['coverage']:.3f} >= 0.90"),
        (
            "T2 informativeness",
            bic["detects_gap"] > 0.90,
            f"ll_bic detects_gap {bic['detects_gap']:.2f} > 0.90",
        ),
        (
            "T3 dependence",
            ind_drop >= 0.10 and bic_drop < 0.05,
            f"ll_independence drops {ind_drop:+.3f} (need >= 0.10), "
            f"ll_bic drops {bic_drop:+.3f} (need < 0.05)",
        ),
        ("T4 floor", gap_lead >= 0.50, f"detects_gap lead over union_count {gap_lead:.2f} >= 0.50"),
    ]

    print(
        f"run={args.run} pool={r['pool']} episodes={r['episodes']} N={r['n_true']} "
        f"reps={r['reps']} linkage={LINKAGE}"
    )
    for cfg in (PRIMARY, COUPLED):
        print(
            f"{cfg}: hidden from all three = "
            f"{r['configs'][f'{cfg}/{LINKAGE}']['mean_unseen_by_w1w2w3']}"
        )
    print()
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:20} {detail}")

    out = {
        "run": args.run,
        "pool": r["pool"],
        "linkage": LINKAGE,
        "episodes": r["episodes"],
        "n_true": r["n_true"],
        "hypotheses": {n: {"passed": bool(ok), "detail": d} for n, ok, d in results},
        "all_passed": all(ok for _, ok, _ in results),
    }
    # Its own file: artifacts/hypotheses.json is the back-office H1-H4 result.
    dst = ROOT / "artifacts" / "transfer_hypotheses.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
