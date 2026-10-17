#!/usr/bin/env python3
"""Render artifacts/thinning/<run>.json as the markdown table in docs/ENV-FINDINGS.md.

Docs quote this output; numbers are never hand-typed (AGENTS.md rule 2).

Usage:
  python3 scripts/thin_table.py long-30 > /tmp/t.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "trust_trace",
    "union_count",
    "union_bound",
    "ll_independence",
    "ll_bic",
    "ll_average",
    "anchor_g1",
    "anchor_g4",
    "lc_gated",
    "strat_lc_gated",
]


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "long-30"
    linkage = sys.argv[2] if len(sys.argv) > 2 else "oracle"
    r = json.loads((ROOT / "artifacts" / "thinning" / f"{run}.json").read_text())
    kind = (
        "episodes resampled with replacement per replicate"
        if r.get("bootstrap_episodes")
        else "the same episodes every replicate"
    )
    print(
        f"Source: `artifacts/thinning/{run}.json` — {r['episodes']} episodes, "
        f"N={r['n_true']}, {r['reps']} loss replicates per configuration, "
        f"{kind}, seed {r['seed']}, {linkage} linkage, nominal 95%.\n"
    )
    for cfg, d in r["configs"].items():
        name, lk = cfg.split("/")
        if lk != linkage:
            continue
        c = d["config"]
        print(
            f"**`{name}`** — gateway drops {c['p_gateway_drop']}, trace drops "
            f"{c['p_trace_drop']}, env view `{c['env_view']}`. "
            f"Mean actions hidden from all three witnesses: "
            f"**{d['mean_unseen_by_w1w2w3']}**.\n"
        )
        print(
            "| estimator | coverage ± MCSE | detects gap | unbounded | mean width | bias | RMSE |"
        )
        print("|---|---|---|---|---|---|---|")
        for k in ORDER:
            m = d["estimators"].get(k)
            if not m:
                continue
            w = "—" if m["mean_width"] is None else m["mean_width"]
            print(
                f"| `{k}` | {m['coverage']:.3f} ± {m['coverage_mcse']:.3f} "
                f"| {m.get('detects_gap', '—')} | {m['unbounded_rate']:.2f} "
                f"| {w} | {m['bias']} | {m['rmse']} |"
            )
        print()


if __name__ == "__main__":
    main()
