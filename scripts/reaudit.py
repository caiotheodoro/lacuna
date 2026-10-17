#!/usr/bin/env python3
"""Re-score W4 on stored transcripts, without paying for the agent again.

The auditor is the only witness that can be swapped after the fact: the other
three are produced by infrastructure during the run. Episodes recorded since
D14 store their transcript, so auditor variants are comparable on identical
agent behaviour.

Usage:
  uv run python scripts/reaudit.py --run narr-30 --view narration --budget 2
  uv run python scripts/reaudit.py --run narr-30 --view full --budget 2

Writes artifacts/reaudit/<run>.<view>.json with per-episode recall and
precision against the ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lacuna.envs.kernel import LedgerEntry
from lacuna.linkage import oracle_table
from lacuna.witnesses import AUDITOR, Record, auditor_records

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--view", choices=["full", "narration"], default="narration")
    ap.add_argument("--budget", type=float, default=2.0)
    args = ap.parse_args()

    import os
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_episodes import load_dotenv

    from lacuna.agents.llm import ChatClient, LLMAuditor

    load_dotenv(ROOT / ".env")
    client = ChatClient(
        base_url=os.environ["LACUNA_BASE_URL"],
        api_key=os.environ["LACUNA_API_KEY"],
        model=os.environ["LACUNA_MODEL"],
        usd_per_m_in=float(os.environ["LACUNA_USD_PER_M_IN"]),
        usd_per_m_out=float(os.environ["LACUNA_USD_PER_M_OUT"]),
        budget_usd=args.budget,
    )
    auditor = LLMAuditor(client, view=args.view)

    src = ROOT / "artifacts" / "episodes" / args.run / "episodes.jsonl"
    rows, stopped = [], ""
    for line in src.open():
        e = json.loads(line)
        if not e.get("transcript"):
            raise SystemExit(f"{args.run} has no stored transcripts; re-run it (D14).")
        client.headers = {"x-opencode-session": f"lacuna-reaudit-{args.run}-{e['seed']}"}
        try:
            actions = auditor.reconstruct(e["transcript"], tag=str(e["seed"]))
        except Exception as exc:  # budget or network: keep what is done
            stopped = f"{type(exc).__name__}: {exc}"
            break
        ledger = [LedgerEntry(**x) for x in e["ledger"]]
        # Keep the non-W4 records, swap in this view's W4.
        recs = [Record(**r) for r in e["records"] if r["witness"] != AUDITOR]
        recs += auditor_records(actions)
        t, st = oracle_table(ledger, recs, 4)
        matched = sum(int(c) for h, c in enumerate(t) if h >> AUDITOR & 1)
        reported = len(auditor_records(actions))
        rows.append(
            {
                "seed": e["seed"],
                "n_true": len(ledger),
                "reported": reported,
                "matched": matched,
                "recall": round(matched / len(ledger), 4) if ledger else None,
                "precision": round(matched / reported, 4) if reported else None,
                "false_positives": st["auditor_false_positives"],
            }
        )
        print(f"seed {e['seed']}: recall {rows[-1]['recall']} ${client.spent:.4f}", flush=True)

    rec = [r["recall"] for r in rows if r["recall"] is not None]
    out = {
        "run": args.run,
        "view": args.view,
        "model": os.environ["LACUNA_MODEL"],
        "episodes": len(rows),
        "stopped": stopped,
        "recall_mean": round(float(np.mean(rec)), 4) if rec else None,
        "recall_min": round(float(np.min(rec)), 4) if rec else None,
        "total_false_positives": sum(r["false_positives"] for r in rows),
        "usd": round(client.spent, 6),
        "per_episode": rows,
    }
    dst = ROOT / "artifacts" / "reaudit" / f"{args.run}.{args.view}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"\n{args.view}: recall mean={out['recall_mean']} min={out['recall_min']} "
        f"FP={out['total_false_positives']} ${out['usd']}"
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
