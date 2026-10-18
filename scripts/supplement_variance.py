#!/usr/bin/env python3
"""Rebuild artifacts/supplement-variance.json from the per-seed thinning runs (D35, D36).

It was first computed in a one-off snippet. A figure the paper quotes has to
come from a script, so this is that script. Per pool and loss seed it records
ll_bic coverage, bias and gap detection at fin_mid, and the H3 contrast: how
much each estimator's coverage falls from fin_mid to fin_mid_rho100.

Usage: uv run python scripts/supplement_variance.py
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIN = ROOT / "artifacts" / "thinning"
SEEDS = (4242, 1, 2, 3)
H1_BAR = 0.90  # docs/PRE-REGISTRATION.md, H1
POOLS = {
    "21 eps (registered)": ("confirm.json", "var-confirm-s{}.json"),
    "24 eps (supplement)": ("confirm-plus-supp.json", "var-supp-s{}.json"),
}


def est(d: dict, cfg: str, name: str) -> dict:
    return d["configs"][f"{cfg}/signature"]["estimators"][name]


def main() -> None:
    out = {}
    for pool, (base, pattern) in POOLS.items():
        rows = []
        for seed in SEEDS:
            f = base if seed == 4242 else pattern.format(seed)
            d = json.loads((THIN / f).read_text())
            bic, bic_c = est(d, "fin_mid", "ll_bic"), est(d, "fin_mid_rho100", "ll_bic")
            ind, ind_c = (
                est(d, "fin_mid", "ll_independence"),
                est(d, "fin_mid_rho100", "ll_independence"),
            )
            rows.append(
                {
                    "seed": seed,
                    "coverage": bic["coverage"],
                    "bias": bic["bias"],
                    "detects_gap": bic["detects_gap"],
                    "h3_independence_drop": round(ind["coverage"] - ind_c["coverage"], 3),
                    "h3_bic_drop": round(bic["coverage"] - bic_c["coverage"], 3),
                }
            )
        cov = [r["coverage"] for r in rows]
        mean = st.mean(cov)
        mcse = math.sqrt(mean * (1 - mean) / (100 * len(SEEDS)))
        first = cov[0]
        out[pool] = {
            "seeds": list(SEEDS),
            "coverage": cov,
            "bias": [r["bias"] for r in rows],
            "detects_gap": [r["detects_gap"] for r in rows],
            "h3_independence_drop": [r["h3_independence_drop"] for r in rows],
            "h3_bic_drop": [r["h3_bic_drop"] for r in rows],
            "mean_coverage": round(mean, 3),
            "sd_coverage": round(st.stdev(cov), 3),
            # MCSE of a mean over len(SEEDS) x 100 replicates.
            "mean_mcse": round(mcse, 3),
            # Distance from the frozen H1 bar, in Monte Carlo errors: the
            # registered seed alone, and the four-seed mean.
            "seed_4242_margin_mcse": round(
                (first - H1_BAR) / math.sqrt(first * (1 - first) / 100), 1
            ),
            "mean_margin_mcse": round((mean - H1_BAR) / mcse, 1),
        }
    dst = ROOT / "artifacts" / "supplement-variance.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    for pool, r in out.items():
        print(pool, r["coverage"], "mean", r["mean_coverage"], "mcse", r["mean_mcse"])
        print(
            "   gap",
            r["detects_gap"],
            "h3 ind",
            r["h3_independence_drop"],
            "h3 bic",
            r["h3_bic_drop"],
        )


if __name__ == "__main__":
    main()
