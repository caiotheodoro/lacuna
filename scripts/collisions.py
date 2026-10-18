#!/usr/bin/env python3
"""How often effect signatures collide, and how complete the raw logs are (D44).

Signature linkage cannot tell apart two actions with the same signature in
one episode. This counts, per run, the ledger entries that share a signature
with another entry in the same episode, by operation. It also counts, before
any thinning, how many ledger ids each witness recorded, and how many gateway
lines (tool calls of any kind, reads and errors included) the agent made per
action: the denominator a numbered gateway would actually number.

Usage: uv run python scripts/collisions.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lacuna.envs.tools import ACTION_TOOLS
from lacuna.linkage import signature, signature_table
from lacuna.witnesses import ENV, GATEWAY, TRACE

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("confirm", "confirm-plus-supp", "long-30", "long-60")


def main() -> None:
    from thin_episodes import load

    out = {}
    for run in RUNS:
        n = calls = action_calls = shared = 0
        shared_by_op: Counter = Counter()
        n_by_op: Counter = Counter()
        seen = {TRACE: 0, GATEWAY: 0, ENV: 0}
        merged = 0
        for line in (ROOT / "artifacts" / "episodes" / run / "episodes.jsonl").open():
            e = json.loads(line)
            for m in e.get("transcript") or []:
                for tc in m.get("tool_calls") or []:
                    calls += 1
                    action_calls += (tc.get("function") or {}).get("name") in ACTION_TOOLS
        for led, recs in load(run):
            n += len(led)
            ids = {x.id for x in led}
            for w in seen:
                seen[w] += len(ids & {r.oracle for r in recs if r.witness == w})
            sigs = Counter(signature(x.op, x.resource, x.value) for x in led)
            for x in led:
                n_by_op[x.op] += 1
                if sigs[signature(x.op, x.resource, x.value)] > 1:
                    shared += 1
                    shared_by_op[x.op] += 1
            t = signature_table([r for r in recs if r.witness < 3], 3)
            merged += len(led) - int(t[1:].sum())
        out[run] = {
            "n": n,
            "sharing_a_signature": shared,
            "sharing_share": round(shared / n, 3),
            "sharing_by_op": {
                op: {
                    "n": n_by_op[op],
                    "sharing": shared_by_op[op],
                    "share": round(shared_by_op[op] / n_by_op[op], 3),
                }
                for op in sorted(n_by_op)
            },
            "unthinned_ids_seen": {
                "trace": seen[TRACE],
                "gateway": seen[GATEWAY],
                "env": seen[ENV],
            },
            "merged_by_signature_unthinned": merged,
            "tool_calls": calls,
            "action_tool_calls": action_calls,
            "actions_per_gateway_line": round(n / calls, 3) if calls else None,
        }
        print(run, out[run]["sharing_share"], out[run]["unthinned_ids_seen"], calls, flush=True)
    dst = ROOT / "artifacts" / "robustness" / "collisions.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
