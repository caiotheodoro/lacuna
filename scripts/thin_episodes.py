#!/usr/bin/env python3
"""Replay recorded episodes under logging-loss configurations and score every estimator.

Usage:
  uv run python scripts/thin_episodes.py --run long-30 --reps 100 --seed 4242

Reads artifacts/episodes/<run>/episodes.jsonl (recorded with zero faults) and
writes artifacts/thinning/<run>.json. Coverage is over fault realisations,
conditional on the agent behaviour recorded in that run (docs/DECISIONS.md D12).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from lacuna.envs.kernel import LedgerEntry
from lacuna.thinning import CONFIGS, HELDOUT, MNAR, estimate_all, tables, thin
from lacuna.witnesses import AUDITOR, Record

ROOT = Path(__file__).resolve().parents[1]


def load(run: str) -> list[tuple[list[LedgerEntry], list[Record]]]:
    out = []
    for line in (ROOT / "artifacts" / "episodes" / run / "episodes.jsonl").open():
        e = json.loads(line)
        out.append(
            (
                [LedgerEntry(**x) for x in e["ledger"]],
                [Record(**r) for r in e["records"]],
            )
        )
    return out


def auditor_recall(episodes) -> list[float]:
    """Per-episode share of ledger actions the W4 auditor reported (oracle-matched)."""
    from lacuna.linkage import oracle_table

    out = []
    for ledger, records in episodes:
        if not ledger:
            continue
        t, _ = oracle_table(ledger, records, 4)
        seen = sum(int(c) for h, c in enumerate(t) if h >> AUDITOR & 1)
        out.append(seen / len(ledger))
    return out


def summarise(rows: list[tuple[int, dict]]) -> dict:
    """Coverage and friends, plus whether the interval is informative.

    Coverage alone cannot separate a useful answer from a wide one: the
    trivial bound [n, inf) covers every time and says nothing. `detects_gap`
    is the share of replicates whose lower end exceeds the union of all
    witnesses, which is the operational claim an auditor needs - "your logs
    are missing actions, with 95% confidence" - and which the trivial bound
    never makes.
    """
    names = rows[0][1].keys()
    out = {}
    for name in names:
        ests = [(n, r[name]) for n, r in rows]
        unions = [r["union_count"].point for _, r in rows]
        reps = len(ests)
        cover = sum(e.covers(n) for n, e in ests) / reps
        bounded = [e for _, e in ests if e.bounded]
        err = [e.point - n for n, e in ests if math.isfinite(e.point)]
        out[name] = {
            "coverage": round(cover, 4),
            "coverage_mcse": round(math.sqrt(max(cover * (1 - cover), 1e-12) / reps), 4),
            "unbounded_rate": round(1 - len(bounded) / reps, 4),
            "mean_width": round(float(np.mean([e.width for e in bounded])), 2) if bounded else None,
            "bias": round(float(np.mean(err)), 2) if err else None,
            "rmse": round(float(np.sqrt(np.mean(np.square(err)))), 2) if err else None,
            "detects_gap": round(
                sum(e.lo > u for (_, e), u in zip(ests, unions, strict=True)) / reps, 4
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4242)
    # Without this, the reported interval covers only logging-fault variation
    # and treats these 15 episodes as the population. Resampling episodes with
    # replacement adds the between-episode variation a new shift would bring
    # (the cluster bootstrap assay uses over environments).
    ap.add_argument("--bootstrap-episodes", action="store_true")
    # Resample k episodes instead of all of them, to see how interval width
    # scales with study size. Implies --bootstrap-episodes.
    ap.add_argument("--n-episodes", type=int, default=0)
    # The confirmatory hypotheses are tested only on the held-out
    # configurations (docs/PRE-REGISTRATION.md, D28).
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--mnar", action="store_true")
    args = ap.parse_args()

    episodes = load(args.run)
    ledger = [e for led, _ in episodes for e in led]
    n_true = len(ledger)
    recall = auditor_recall(episodes)
    result = {
        "run": args.run,
        "episodes": len(episodes),
        "n_true": n_true,
        "reps": args.reps,
        "pool": "mnar" if args.mnar else ("heldout" if args.heldout else "exploratory"),
        "bootstrap_episodes": bool(args.bootstrap_episodes or args.n_episodes),
        "n_episodes_sampled": args.n_episodes or len(episodes),
        "seed": args.seed,
        "auditor_recall": {
            "mean": round(float(np.mean(recall)), 4),
            "min": round(float(np.min(recall)), 4),
            "per_episode": [round(r, 4) for r in recall],
        },
        "configs": {},
    }
    t0 = time.perf_counter()
    pool = MNAR if args.mnar else (HELDOUT if args.heldout else CONFIGS)
    for ci, cfg in enumerate(pool):
        for linkage in ("oracle", "signature"):
            rng = np.random.default_rng([args.seed, ci, linkage == "signature"])
            rows, missing = [], []
            for _ in range(args.reps):
                if args.bootstrap_episodes or args.n_episodes:
                    k = args.n_episodes or len(episodes)
                    idx = rng.integers(0, len(episodes), k)
                    sample = [episodes[i] for i in idx]
                else:
                    sample = episodes
                n_rep = sum(len(led) for led, _ in sample)
                t4_parts, per_op_all = [], {}
                for led, recs in sample:
                    t4, per_op = tables(led, thin(recs, cfg, rng), linkage)
                    t4_parts.append(t4)
                    for op, t in per_op.items():
                        per_op_all[op] = per_op_all.get(op, 0) + t
                table4 = np.sum(t4_parts, axis=0)
                n_true_rep = n_rep
                # Unseen by the three hard witnesses. The four-witness figure
                # is always 0 while W4 is degenerate (D14), so it measures
                # nothing; this is the quantity the @3 estimators face.
                seen3 = sum(int(c) for h, c in enumerate(table4) if h & 0b111)
                missing.append(n_true_rep - seen3)
                rows.append((n_true_rep, estimate_all(table4, per_op_all)))
            result["configs"][f"{cfg.name}/{linkage}"] = {
                "config": cfg.__dict__,
                "mean_unseen_by_w1w2w3": round(float(np.mean(missing)), 2),
                "estimators": summarise(rows),
            }
            print(f"{cfg.name}/{linkage}: done ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = ROOT / "artifacts" / "thinning" / f"{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
