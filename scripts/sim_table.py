#!/usr/bin/env python3
"""Print artifacts/sim_coverage.json as the markdown table in docs/SIM-FINDINGS.md.

Docs quote this output, never hand-typed numbers (AGENTS.md rule 2).

Usage:
  python3 scripts/sim_table.py > /tmp/table.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "artifacts" / "sim_coverage.json"
SKIP = ("union_bound", "union_count@trace+proxy+env+auditor")


def main() -> None:
    r = json.loads((Path(sys.argv[1]) if len(sys.argv) > 1 else SRC).read_text())
    print(f"Source: `artifacts/sim_coverage.json`, reps={r['reps']}, seed={r['seed']}.\n")
    print("| scenario | estimator | coverage ± MCSE | unbounded | mean width | bias | RMSE |")
    print("|---|---|---|---|---|---|---|")
    for s, sc in r["scenarios"].items():
        for e, m in sorted(sc["estimators"].items()):
            if e.startswith(SKIP):
                continue
            w = "—" if m["mean_width"] is None else m["mean_width"]
            print(
                f"| {s} | `{e}` | {m['coverage']:.3f} ± {m['coverage_mcse']:.3f} "
                f"| {m['unbounded_rate']:.2f} | {w} | {m['bias']} | {m['rmse']} |"
            )


if __name__ == "__main__":
    main()
