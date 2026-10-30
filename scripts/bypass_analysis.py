"""Frozen analysis for the `backoffice_bypass` study: how the estimators fare when W1 and W2 are
blind to the same actions.

In `backoffice_bypass` an action sent through `run_script` is ledgered and seen by W3 like a
direct call, but the harness trace (W1) and the gateway (W2) log only the `run_script` call, so
both miss every action inside it. This replays the recorded episodes of one run under
logging-loss configurations, scores the estimators under three linkage arms, and describes the
run's bypass accounting. It calls no model.

Replicates. Each replicate resamples the run's episodes with replacement (the
`--bootstrap-episodes` convention of scripts/thin_episodes.py), thins each with the loss
configuration, and builds the three-witness table under each arm from the same thinned draw:

- `callid`: trace and gateway share the provider's tool-call id; W3 is attached by effect
  signature (`lacuna.linkage.callid_table`, as in scripts/linkage_arms.py).
- `signature`: all three logs linked by effect signature, the pre-registered linkage.
- `oracle`: the ledger id, which no auditor has.

Loss configurations: `fin_mid` (0.50/0.50, W3 sees the financial ops), `full_mid` (the same
rates, W3 sees everything) and `fin_mid_rho100` (gateway and trace drops perfectly coupled).
Estimators: ll_bic, ll_independence, union_count, Chapman for each witness pair, and
seq_gap_tail on the gateway counter. The counter numbers the gateway's action records only, so
a bypassed action takes no number (as in the env, where it has no gateway action record).

Bypass accounting (`descriptive`, from the recorded complete logs, before any loss):

- bypassed action: a ledger action with no W2 (gateway) action record. This is checked against
  the scripts' own replies in the transcript (`n_bypassed_from_scripts`).
- bypass share: bypassed actions / N, N the ledger count.
- capture: per witness, ledger actions with at least one action record from it (oracle link).
- `bypassed_w3_share`: share of the bypassed actions that have a W3 record.

Usage:
  uv run python scripts/bypass_analysis.py --run bypass-scripted-30

Deterministic per seed: seed s uses `numpy.random.default_rng(s)`, and every config and arm
sees the same stream. Writes artifacts/robustness/bypass-<run>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np

from lacuna.envs import bypass
from lacuna.envs.kernel import LedgerEntry
from lacuna.estimators import Estimate, baselines, loglinear
from lacuna.histories import keep_witnesses
from lacuna.linkage import callid_table, signature
from lacuna.thinning import HELDOUT, Config, tables, thin
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 98_000  # disjoint from every seed range in scripts/ (96_000 is final_controls.py)
SEEDS = 30
REPS_PER_SEED = 100
CONFIGS = (
    HELDOUT[0],
    Config("full_mid", 0.50, 0.50, "full"),
    HELDOUT[1],
)
ARMS = ("callid", "signature", "oracle")
ESTIMATORS = (
    "ll_bic",
    "ll_independence",
    "union_count",
    "chapman_w1w2",
    "chapman_w1w3",
    "chapman_w2w3",
    "seq_gap_tail",
)
Z = 1.96


class Ep(NamedTuple):
    ledger: list[LedgerEntry]
    records: list[Record]
    transcript: list[dict]


@lru_cache(None)
def load_episodes(run: str) -> list[Ep]:
    out = []
    path = ROOT / "artifacts" / "episodes" / run / "episodes.jsonl"
    for line in path.open():
        e = json.loads(line)
        out.append(
            Ep(
                [LedgerEntry(**x) for x in e["ledger"]],
                [Record(**r) for r in e["records"]],
                e["transcript"],
            )
        )
    return out


# --- the bypass accounting ---------------------------------------------------------------


def bypassed_ids(ledger, records: list[Record]) -> set[int]:
    """Ledger ids of actions with no gateway (W2) action record, before any loss."""
    via_gateway = {r.oracle for r in records if r.witness == GATEWAY}
    ids = [e["id"] if isinstance(e, dict) else e.id for e in ledger]
    return {i for i in ids if i not in via_gateway}


def describe(run: str) -> dict:
    eps = load_episodes(run)
    n = sum(len(e.ledger) for e in eps)
    seen = {TRACE: 0, GATEWAY: 0, ENV: 0}
    bypassed = bypassed_seen_by_w3 = bypassed_seen_by_w1 = direct_seen_by_w3 = 0
    n_scripts = from_scripts = 0
    by_op: dict[str, int] = {}
    for e in eps:
        ids = {x.id for x in e.ledger}
        by_w = {w: {r.oracle for r in e.records if r.witness == w} & ids for w in seen}
        for w in seen:
            seen[w] += len(by_w[w])
        b = bypassed_ids(e.ledger, e.records)
        bypassed += len(b)
        bypassed_seen_by_w3 += len(b & by_w[ENV])
        bypassed_seen_by_w1 += len(b & by_w[TRACE])
        direct_seen_by_w3 += len((ids - b) & by_w[ENV])
        for x in e.ledger:
            if x.id in b:
                by_op[x.op] = by_op.get(x.op, 0) + 1
        n_scripts += bypass.script_calls(e.transcript)
        from_scripts += bypass.bypassed_actions(e.transcript)
    summary_path = ROOT / "artifacts" / "episodes" / run / "summary.json"
    faults = (
        json.loads(summary_path.read_text()).get("fault_rates") if summary_path.exists() else None
    )
    direct = n - bypassed
    return {
        "episodes": len(eps),
        "n_true": n,
        "n_bypassed": bypassed,
        "n_bypassed_from_scripts": from_scripts,
        "bypass_definition": "ledger actions with no W2 (gateway) action record before any loss",
        "bypassed_share": round(bypassed / n, 4) if n else None,
        "n_script_calls": n_scripts,
        "capture": {"trace": seen[TRACE], "gateway": seen[GATEWAY], "env": seen[ENV]},
        "capture_share": {
            "trace": round(seen[TRACE] / n, 4),
            "gateway": round(seen[GATEWAY] / n, 4),
            "env": round(seen[ENV] / n, 4),
        },
        "bypassed_seen_by_w3": bypassed_seen_by_w3,
        "bypassed_w3_share": round(bypassed_seen_by_w3 / bypassed, 4) if bypassed else None,
        "bypassed_seen_by_w1": bypassed_seen_by_w1,
        "direct_w3_share": round(direct_seen_by_w3 / direct, 4) if direct else None,
        "bypassed_by_op": dict(sorted(by_op.items())),
        "recorded_fault_rates": faults,
        "complete_logs": bool(faults) and not any(faults.values()),
        "accounting_consistent": bypassed == from_scripts,
    }


# --- one replicate -----------------------------------------------------------------------


def gateway_numbers(records: list[Record]) -> dict[int, int]:
    """Gateway counter of each action: its place among the gateway's action records.

    A bypassed action has no gateway action record, so it takes no number.
    """
    return baselines.gateway_seq(records)


def arm_table(led, kept: list[Record], arm: str) -> np.ndarray:
    """Three-witness table (cell 0 zeroed) of one thinned episode under a linkage arm."""
    if arm == "callid":
        t = callid_table(kept, lambda r: signature(r.op, r.resource, r.value))
    else:
        t = keep_witnesses(tables(led, kept, arm)[0], (0, 1, 2))
    t = np.array(t, dtype=np.int64)
    t[0] = 0
    return t


def estimate_all(t3: np.ndarray, obs, union: int) -> dict[str, Estimate]:
    return {
        "ll_bic": loglinear.best_bic(t3),
        "ll_independence": loglinear.independence(t3),
        "union_count": baselines.union_count(t3),
        "chapman_w1w2": baselines.chapman_w1w2(t3),
        "chapman_w1w3": baselines.chapman_w1w3(t3),
        "chapman_w2w3": baselines.chapman_w2w3(t3),
        "seq_gap_tail": baselines.seq_gap_tail(obs, union),
    }


def run_seed(run: str, cfg: Config, seed: int, reps: int) -> list[dict]:
    """`reps` replicates for one loss seed: true N, each arm's union and every estimate."""
    eps = load_episodes(run)
    seqs = [gateway_numbers(e.records) for e in eps]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        idx = rng.integers(0, len(eps), len(eps))
        parts: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
        obs = []
        n = 0
        for i in idx:
            e = eps[i]
            n += len(e.ledger)
            kept = thin(e.records, cfg, rng)
            obs.append(baselines.seq_obs(kept, seqs[i]))
            for a in ARMS:
                parts[a].append(arm_table(e.ledger, kept, a))
        t3 = {a: np.sum(parts[a], axis=0) for a in ARMS}
        union = {a: int(t3[a][1:].sum()) for a in ARMS}
        rows.append(
            {
                "n": n,
                "union": union,
                "est": {a: estimate_all(t3[a], obs, union[a]) for a in ARMS},
            }
        )
    return rows


# --- summaries ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(ests: list[Estimate], ns: list[int], unions: list[int]) -> dict:
    """Coverage with its MCSE and Wilson interval, width, bias, rmse, gap detection.

    Coverage counts an unbounded interval as covering when its lower end is at most N. Bias and
    rmse are over finite points, width over bounded intervals, as in scripts/replay.py.
    """
    reps = len(ests)
    hit = sum(e.covers(n) for e, n in zip(ests, ns, strict=True))
    cover = hit / reps
    bounded = [e for e in ests if e.bounded]
    err = [e.point - n for e, n in zip(ests, ns, strict=True) if math.isfinite(e.point)]
    lo, hi = wilson(hit, reps)
    detects = sum(e.lo > u for e, u in zip(ests, unions, strict=True)) / reps
    return {
        "reps": reps,
        "coverage": round(cover, 4),
        "coverage_mcse": round(math.sqrt(max(cover * (1 - cover), 1e-12) / reps), 4),
        "coverage_wilson95": [round(lo, 4), round(hi, 4)],
        "unbounded_rate": round(1 - len(bounded) / reps, 4),
        "mean_width": round(float(np.mean([e.width for e in bounded])), 2) if bounded else None,
        "bias": round(float(np.mean(err)), 2) if err else None,
        "rmse": round(float(np.sqrt(np.mean(np.square(err)))), 2) if err else None,
        "detects_gap": round(detects, 4),
    }


def _job(args: tuple) -> tuple[str, int, list[dict]]:
    run, cfg_name, seed, reps = args
    cfg = next(c for c in CONFIGS if c.name == cfg_name)
    return cfg_name, seed, run_seed(run, cfg, seed, reps)


def analyse(run: str, seed0: int, seeds: int, reps: int, processes: int) -> dict:
    jobs = [(run, c.name, s, reps) for c in CONFIGS for s in range(seed0, seed0 + seeds)]
    if processes > 1:
        with Pool(processes) as pool:
            done = pool.map(_job, jobs, chunksize=1)
    else:
        done = [_job(j) for j in jobs]
    cells = {}
    for cfg in CONFIGS:
        rows = [r for name, _, rs in done if name == cfg.name for r in rs]  # seed order
        for arm in ARMS:
            ns = [r["n"] for r in rows]
            unions = [r["union"][arm] for r in rows]
            cells[f"{cfg.name}/{arm}"] = {
                "config": asdict(cfg),
                "arm": arm,
                "replicates": len(rows),
                "mean_n_true": round(float(np.mean(ns)), 2),
                "mean_union": round(float(np.mean(unions)), 2),
                "mean_hidden": round(
                    float(np.mean([n - u for n, u in zip(ns, unions, strict=True)])), 2
                ),
                "estimators": {
                    name: summarise([r["est"][arm][name] for r in rows], ns, unions)
                    for name in ESTIMATORS
                },
            }
    return cells


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, help="artifacts/episodes/<run>/episodes.jsonl")
    ap.add_argument(
        "--seed0", type=int, default=SEED0, help="first loss seed (default %(default)s)"
    )
    ap.add_argument("--seeds", type=int, default=SEEDS, help="number of loss seeds")
    ap.add_argument("--reps", type=int, default=REPS_PER_SEED, help="replicates per seed")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: artifacts/robustness/bypass-<run>.json")
    return ap


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    run_dir = ROOT / "artifacts" / "episodes" / args.run
    cells = analyse(args.run, args.seed0, args.seeds, args.reps, args.processes)
    out = {
        "script": "scripts/bypass_analysis.py",
        "argv": argv,
        "run": args.run,
        "episodes_sha256": hashlib.sha256((run_dir / "episodes.jsonl").read_bytes()).hexdigest(),
        "params": {
            "seed0": args.seed0,
            "seeds": args.seeds,
            "seed_range": [args.seed0, args.seed0 + args.seeds - 1],
            "reps_per_seed": args.reps,
            "replicates": args.seeds * args.reps,
            "bootstrap_episodes": True,
            "rng": "numpy.random.default_rng(seed) per (config, seed); one stream shared by "
            "all arms of a config and by all configs",
            "configs": [asdict(c) for c in CONFIGS],
            "arms": list(ARMS),
            "estimators": list(ESTIMATORS),
            "arm_definitions": {
                "callid": "trace and gateway linked by the shared tool-call id; W3 attached by "
                "effect signature (lacuna.linkage.callid_table)",
                "signature": "all three logs linked by effect signature",
                "oracle": "ledger id",
            },
            "seq_gap_tail": "gateway counter over the gateway's action records only; bypassed "
            "actions take no number",
        },
        "descriptive": describe(args.run),
        "cells": cells,
    }
    dst = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts" / "robustness" / f"bypass-analysis-{args.run}.json"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    for key, c in cells.items():
        b = c["estimators"]["ll_bic"]
        print(key, b["coverage"], b["mean_width"], b["bias"], flush=True)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
