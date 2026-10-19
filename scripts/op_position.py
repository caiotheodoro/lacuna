#!/usr/bin/env python3
"""Where in an episode the env-blind actions fall (D53).

Under the financial env view, W3 sees only refunds, cancellations and outbound
mail. If those cluster at one end of an episode, blocks of consecutive lost
lines land on the actions W3 cannot see. This records, per pool, the share of
each quarter of an episode (by ledger position) that the financial view
covers.

Usage: uv run python scripts/op_position.py
"""

from __future__ import annotations

import json
from pathlib import Path

from thin_episodes import load

from lacuna.thinning import ENV_VIEWS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("confirm", "long-30", "long-60", "long-120")


def main() -> None:
    visible = set(ENV_VIEWS["financial"])
    out = {}
    for run in RUNS:
        seen = [0, 0, 0, 0]
        total = [0, 0, 0, 0]
        for led, _ in load(run):
            n = len(led)
            for i, e in enumerate(led):
                q = min(3, 4 * i // max(n, 1))
                total[q] += 1
                seen[q] += e.op in visible
        out[run] = {
            "actions_per_episode": round(sum(total) / len(load(run)), 1),
            "w3_visible_share_by_quarter": [
                round(s / t, 3) if t else None for s, t in zip(seen, total, strict=True)
            ],
        }
        print(run, out[run], flush=True)
    dst = ROOT / "artifacts" / "robustness" / "op-position.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
