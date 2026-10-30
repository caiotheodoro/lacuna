#!/usr/bin/env python3
"""Emit the two headline tables of docs/ENV-FINDINGS.md from the artifact.

Every number in that document comes from here or from scripts/thin_table.py.
Hand-typing them once produced four wrong figures (AGENTS.md rule 2).

Usage: python3 scripts/headline_tables.py long-30.bootstrap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOSS = [
    ("light", "light"),
    ("fin_light", "fin_light"),
    ("sampled", "sampled"),
    ("fin_sampled", "fin_sampled"),
]
RHO = [("fin_sampled", "none"), ("fin_sampled_rho50", "half"), ("fin_sampled_rho100", "full")]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run", nargs="?", default="long-30.bootstrap", help="artifacts/thinning/<run>.json"
    )
    args = ap.parse_args(argv)
    r = json.loads((ROOT / "artifacts" / "thinning" / f"{args.run}.json").read_text())
    cfgs = r["configs"]

    print(
        "| hidden from all 3 | `union_count` coverage | `ll_independence` coverage "
        "| width | detects gap |"
    )
    print("|---|---|---|---|---|")
    for cfg, label in LOSS:
        d = cfgs[f"{cfg}/oracle"]
        u, m = d["estimators"]["union_count"], d["estimators"]["ll_independence"]
        print(
            f"| {d['mean_unseen_by_w1w2w3']} (`{label}`) | {u['coverage']:.3f} "
            f"| {m['coverage']:.3f} | {m['mean_width']} | {m['detects_gap']:.2f} |"
        )

    print("\n| coupling | `ll_independence` coverage | its bias | `ll_bic` coverage | its bias |")
    print("|---|---|---|---|---|")
    for cfg, label in RHO:
        d = cfgs[f"{cfg}/oracle"]
        a, b = d["estimators"]["ll_independence"], d["estimators"]["ll_bic"]
        print(
            f"| {label} | {a['coverage']:.3f} | {a['bias']:+} | {b['coverage']:.3f} "
            f"| {b['bias']:+} |"
        )


if __name__ == "__main__":
    main()
