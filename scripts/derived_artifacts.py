#!/usr/bin/env python3
"""Recompute, into artifacts/derived.json, the figures that so far lived only in prose.

The auditor recall table (D18, D20), the worst-case estimator comparison (D27),
the power curve (D27) and the attrition diagnostic (D31) were computed in
one-off shell snippets. A number the paper quotes has to trace to an artifact,
so this script is the single place they come from.

Usage: uv run python scripts/derived_artifacts.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thin_episodes import load  # noqa: E402

from lacuna.envs.kernel import LedgerEntry  # noqa: E402
from lacuna.envs.tasks import make_episode  # noqa: E402
from lacuna.linkage import oracle_table  # noqa: E402
from lacuna.thinning import CONFIGS, estimate_all, tables, thin  # noqa: E402
from lacuna.witnesses import AUDITOR, Record  # noqa: E402

ART = ROOT / "artifacts"


def auditor_recall() -> dict:
    out = {}
    runs = [("pilot-l0", 7), ("long-30", 30), ("long-60", 60), ("long-120", 120), ("narr-30", 30)]
    for run, tickets in runs:
        path = ART / "episodes" / run / "episodes.jsonl"
        summary = json.loads((ART / "episodes" / run / "summary.json").read_text())
        recall, fp, n, calls = [], 0, [], []
        for line in path.open():
            e = json.loads(line)
            led = [LedgerEntry(**x) for x in e["ledger"]]
            recs = [Record(**r) for r in e["records"]]
            t, st = oracle_table(led, recs, 4)
            recall.append(sum(int(c) for h, c in enumerate(t) if h >> AUDITOR & 1) / len(led))
            fp += st["auditor_false_positives"]
            n.append(len(led))
            calls.append(e["n_calls"])
        out[run] = {
            "tickets": tickets,
            "episodes": len(recall),
            "auditor_view": summary.get("auditor_view", "full"),
            "model": summary.get("model"),
            "actions_per_episode": round(float(np.mean(n)), 1),
            "calls_per_episode": round(float(np.mean(calls)), 1),
            "actions_per_ticket": round(float(np.mean(n)) / tickets, 2),
            "recall_mean": round(float(np.mean(recall)), 4),
            "recall_min": round(float(np.min(recall)), 4),
            "false_positives": fp,
            "n_true": int(sum(n)),
        }
    return out


def worst_case() -> dict:
    b = json.loads((ART / "thinning" / "long-30.bootstrap.json").read_text())
    cfgs = [c for c in b["configs"] if c.endswith("/oracle")]
    names = [
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
    out = {"source": "artifacts/thinning/long-30.bootstrap.json", "configs": len(cfgs)}
    for k in names:
        rows = [(b["configs"][c]["estimators"][k], c.split("/")[0]) for c in cfgs]
        worst = min(rows, key=lambda r: r[0]["coverage"])
        widths = [r[0]["mean_width"] for r in rows if r[0]["mean_width"] is not None]
        out[k] = {
            "worst_coverage": worst[0]["coverage"],
            "worst_config": worst[1],
            "mean_detects_gap": round(float(np.mean([r[0]["detects_gap"] for r in rows])), 2),
            "mean_width": round(float(np.mean(widths)), 1) if widths else None,
        }
    return out


def power_curve(reps: int = 60, seed: int = 7) -> dict:
    eps = load("long-30")
    cfg = next(c for c in CONFIGS if c.name == "fin_sampled")
    rows = []
    for k in (4, 8, 15, 30):
        # Reseeded per study size, as the D27 run was (one process per size),
        # so this artifact reproduces the logged curve exactly.
        rng = np.random.default_rng(seed)
        widths, covs, ns = [], [], []
        for _ in range(reps):
            sample = [eps[i] for i in rng.integers(0, len(eps), k)]
            n = sum(len(led) for led, _ in sample)
            t4, per_op = [], {}
            for led, recs in sample:
                a, b = tables(led, thin(recs, cfg, rng), "oracle")
                t4.append(a)
                for op, v in b.items():
                    per_op[op] = per_op.get(op, 0) + v
            e = estimate_all(np.sum(t4, axis=0), per_op)["ll_bic"]
            widths.append(e.width)
            covs.append(e.covers(n))
            ns.append(n)
        rows.append(
            {
                "episodes": k,
                "n_mean": round(float(np.mean(ns))),
                "width_mean": round(float(np.mean(widths)), 2),
                "relative_width": round(float(np.mean(widths) / np.mean(ns)), 3),
                "coverage": round(float(np.mean(covs)), 3),
                "coverage_mcse": round(
                    float(np.sqrt(max(np.mean(covs) * (1 - np.mean(covs)), 1e-12) / reps)), 3
                ),
            }
        )
    # Relative width against 1/sqrt(N): the constant c in width/N = c/sqrt(N).
    c = [r["relative_width"] * np.sqrt(r["n_mean"]) for r in rows]
    return {
        "config": "fin_sampled",
        "estimator": "ll_bic",
        "reps": reps,
        "seed": seed,
        "rows": rows,
        "c_mean": round(float(np.mean(c)), 2),
    }


def attrition() -> dict:
    s = json.loads((ART / "episodes" / "confirm" / "summary.json").read_text())
    failed = [f["seed"] for f in s["failed_episodes"]]
    done = [
        json.loads(line)["seed"]
        for line in (ART / "episodes" / "confirm" / "episodes.jsonl").open()
    ]
    nonfin = {"update_customer", "write_file"}

    def share(seed: int) -> float:
        ops = Counter(t.plan[0][0] for t in make_episode(seed, 30).tickets)
        return sum(v for k, v in ops.items() if k in nonfin) / sum(ops.values())

    return {
        "failed_seeds": failed,
        "completed": len(done),
        "nonfinancial_share_failed": round(float(np.mean([share(x) for x in failed])), 3),
        "nonfinancial_share_completed": round(float(np.mean([share(x) for x in done])), 3),
    }


def env_blind_psi() -> dict:
    """Anchor odds ratio of l2_env_blind in closed form from the scenario's parameters (D7)."""
    from lacuna.sim import SCENARIOS

    sc = SCENARIOS["l2_env_blind"]
    vis, hid = (t.weight for t in sc.types)
    pv = sc.history_probs(sc.types[0])
    ph = sc.history_probs(sc.types[1])

    # Collapse to anchor (env, bit 2) against the union of the rest (bits 0-1).
    def cells(p):
        z = {"00": 0.0, "10": 0.0, "01": 0.0, "11": 0.0}
        for h, v in enumerate(p):
            a = (h >> 2) & 1
            rest = int((h & 0b011) != 0)
            z[f"{a}{rest}"] += v
        return z

    zv, zh = cells(pv), cells(ph)
    z = {k: vis * zv[k] + hid * zh[k] for k in zv}
    psi = z["00"] * z["11"] / (z["10"] * z["01"])
    return {"scenario": "l2_env_blind", "psi": round(float(psi), 2)}


def op_mix() -> dict:
    out = {}
    for run in ("long-30", "confirm"):
        c = Counter()
        for line in (ART / "episodes" / run / "episodes.jsonl").open():
            c.update(x["op"] for x in json.loads(line)["ledger"])
        nonfin = c["update_customer"] + c["write_file"]
        out[run] = {"ops": dict(c), "n_true": sum(c.values()), "nonfinancial": nonfin}
    # D19's arithmetic: non-financial actions x trace loss x gateway loss.
    out["long-30"]["predicted_hidden_fin_sampled"] = round(
        out["long-30"]["nonfinancial"] * 0.3 * 0.9, 1
    )
    return out


def spend() -> dict:
    runs = {}
    for r in sorted((ART / "episodes").glob("*/receipt.json")):
        runs[r.parent.name] = round(json.loads(r.read_text())["total_usd"], 4)
    return {"runs": runs, "total_usd": round(sum(runs.values()), 2)}


def main() -> None:
    out = {
        "env_blind_psi": env_blind_psi(),
        "op_mix": op_mix(),
        "spend": spend(),
        "auditor_recall": auditor_recall(),
        "worst_case": worst_case(),
        "power_curve": power_curve(),
        "attrition": attrition(),
    }
    dst = ART / "derived.json"
    dst.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {dst}")
    for r in out["power_curve"]["rows"]:
        print("power", r)
    print("c_mean", out["power_curve"]["c_mean"])
    print("attrition", out["attrition"])


if __name__ == "__main__":
    main()
