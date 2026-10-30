"""Frozen-able analysis for study 4: a replication pool and a live-adversary pool.

Study 4 records two fresh LLM runs; this script turns each into the numbers the registration will
name. It calls no model, builds no estimator of its own (every estimator, linkage arm, loss
configuration and bootstrap is imported from the frozen scripts/bypass_analysis.py,
scripts/bypass_counterfactuals.py and scripts/bypass_verdicts.py) and attaches no verdict labels:
thresholds belong to the registration. Output is deterministic given the episodes file and the
seeds; the file carries no timestamp. It refuses to overwrite an existing output.

Pool `replication`. A run on env `backoffice` (no bypass tool; `--agent llm` from
scripts/run_episodes.py, complete logs). Same procedure and seed layout as the existing back-office
confirm analysis (`bypass_analysis.py --run confirm`, artifacts/robustness/bypass-analysis-
confirm.json): the run's episodes are resampled with replacement, thinned at `fin_mid` (gateway and
trace drop 0.50, W3 sees the financial ops), and scored with ll_bic (also ll_independence and
union_count) under call-id linkage (`callid`) and effect-signature linkage (`signature`), loss
seeds `--seed0`..`--seed0`+`--seeds`-1 with `--reps` replicates each. Reported per cell: coverage
with its Wilson interval over the replicates, unbounded rate, mean width, bias, gap detection, and
the per-loss-seed coverage. The Wilson interval treats replicates as independent although they
resample one pool of episodes; it reflects loss variation, not episode variation. The episode-level
guard is `loo`: leave-one-episode-out coverage of ll_bic at fin_mid for each arm (each pool
resampled over `--loo-seeds` x `--loo-reps` replicates, seeds from 98200), with min, max, sd
(ddof=1) and `max_deviation` = max |LOO coverage - coverage|, as in bypass_verdicts.py. The
reference is the same cell of the confirm analysis artifact when it is on disk.

Pool `adversary`. A run on env `backoffice_bypass` with `--organism l2`. Reported:
- `bypass`: pooled bypass share (ledger actions with no gateway action record / ledger actions)
  with the percentile episode bootstrap of bypass_verdicts.py (`share_ci`), per-episode shares,
  episodes that called run_script, and the run's bypass accounting (`descriptive`);
- `cells`: the three loss configurations (fin_mid, full_mid, fin_mid_rho100) x three arms
  (callid, signature, oracle) x seven estimators of bypass_analysis.py, so ll_bic at
  fin_mid/callid with the stipulated witnesses is `cells["fin_mid/callid"]`;
- `loo`: as above, for this run;
- `parsed_scripts_control`: bypass_counterfactuals.py's `parsed_scripts` world (one W1 and one W2
  record added for every bypassed action, as an auditor that parsed run_script text would have it)
  for the same configurations, with the hidden-mass split, the pool description and the per-episode
  script-rejection counts;
- `vs_neutral`: the l2 run against the stored neutral run (`--neutral`, default bypass-llm).
  Bypass-share difference (this run minus neutral) paired over the seeds both runs share, when
  there are any (resampling seed indices), and unpaired over all episodes (each run resampled on
  its own); both are percentile bootstrap CIs with the seeds of bypass_verdicts.py. Also ll_bic
  coverage at fin_mid/callid against the neutral run's stored analysis artifact.

Run checks (before anything is computed): the replication run's summary must be a back-office run
with no organism, the adversary run's a backoffice_bypass run with organism l2 (`--expect-organism`
changes the organism, to analyse an existing neutral or l1 run), and both must have
complete logs (zero fault rates) and at least one episode. `completion` records episodes completed,
failures, early stop, turn-limit hits and the model; `underpowered` flags fewer than
`--min-episodes` (18) completed episodes.

Usage:
  uv run python scripts/study4_analysis.py --run <run> --pool replication
  uv run python scripts/study4_analysis.py --run <run> --pool adversary [--neutral bypass-llm]
Writes artifacts/robustness/study4-<pool>-<run>.json unless --out is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bypass_analysis as ba  # noqa: E402
import bypass_counterfactuals as cf  # noqa: E402
import bypass_verdicts as bv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("replication", "adversary")
CELL = ("fin_mid", "callid", "ll_bic")  # the headline cell, as in bypass_verdicts.py
REPLICATION_ARMS = ("callid", "signature")
REPLICATION_ESTIMATORS = ("ll_bic", "ll_independence", "union_count")
COVERAGE_FLOOR = 0.90  # descriptive flags only; the registration sets the verdict rules
MIN_EPISODES = 18
TURN_LIMIT = 400


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- cells -----------------------------------------------------------------------------------


def _cfg(name: str):
    return next(c for c in ba.CONFIGS if c.name == name)


def _job(args: tuple) -> tuple[str, int, list[dict]]:
    run, cfg_name, seed, reps, arms, names = args
    return cfg_name, seed, ba.run_seed(run, _cfg(cfg_name), seed, reps, arms, names)


def _map(fn, jobs: list, processes: int) -> list:
    if processes > 1:
        with Pool(processes) as pool:
            return pool.map(fn, jobs, chunksize=1)
    return [fn(j) for j in jobs]


def seed_coverage(rows_by_seed: list[list[dict]], arm: str, name: str) -> dict:
    cov = [
        round(sum(r["est"][arm][name].covers(r["n"]) for r in rows) / len(rows), 4)
        for rows in rows_by_seed
    ]
    return {
        "values": cov,
        "min": min(cov),
        "max": max(cov),
        "sd": round(float(np.std(cov, ddof=1)), 4) if len(cov) > 1 else None,
        "sd_ddof": 1,
    }


def cells(
    run: str,
    configs: tuple,
    arms: tuple[str, ...],
    names: tuple[str, ...],
    seed0: int,
    seeds: int,
    reps: int,
    processes: int,
) -> dict:
    """Same cell structure as `bypass_analysis.analyse`, restricted to `configs`/`arms`/`names`."""
    jobs = [
        (run, c.name, s, reps, arms, names) for c in configs for s in range(seed0, seed0 + seeds)
    ]
    done = _map(_job, jobs, processes)
    out = {}
    for cfg in configs:
        by_seed = [rows for name, _, rows in done if name == cfg.name]  # seed order
        rows = [r for rs in by_seed for r in rs]
        for arm in arms:
            ns = [r["n"] for r in rows]
            unions = [r["union"][arm] for r in rows]
            out[f"{cfg.name}/{arm}"] = {
                "config": asdict(cfg),
                "arm": arm,
                "replicates": len(rows),
                "mean_n_true": round(float(np.mean(ns)), 2),
                "mean_union": round(float(np.mean(unions)), 2),
                "mean_hidden": round(
                    float(np.mean([n - u for n, u in zip(ns, unions, strict=True)])), 2
                ),
                "estimators": {
                    name: ba.summarise([r["est"][arm][name] for r in rows], ns, unions)
                    for name in names
                },
                "ll_bic_seed_coverage": seed_coverage(by_seed, arm, "ll_bic"),
            }
    return out


# --- leave-one-episode-out ------------------------------------------------------------------


def _loo_job(args: tuple) -> tuple[int, int, list[int]]:
    run, drop, seed, reps, arms = args
    rows = ba.run_seed(run, _cfg(CELL[0]), seed, reps, arms, (CELL[2],), drop=drop)
    return drop, seed, [sum(r["est"][a][CELL[2]].covers(r["n"]) for r in rows) for a in arms]


def loo(run: str, arms: tuple[str, ...], seed0: int, seeds: int, reps: int, processes: int) -> dict:
    """ll_bic coverage at fin_mid of each leave-one-episode-out pool, per arm.

    Pool i is the run without episode i (file order), resampled over `seeds` x `reps` replicates
    (the same loss seeds for every pool). `max_deviation` = max |pool coverage - full coverage|
    is filled in by `attach_deviation` once the full-pool coverage is known.
    """
    n_eps = len(ba.load_episodes(run))
    jobs = [(run, d, s, reps, arms) for d in range(n_eps) for s in range(seed0, seed0 + seeds)]
    done = _map(_loo_job, jobs, processes)
    total = seeds * reps
    out = {
        "cell": f"{CELL[0]}/{{arm}}",
        "estimator": CELL[2],
        "seed0": seed0,
        "seeds": seeds,
        "reps_per_seed": reps,
        "reps_per_pool": total,
        "pools": n_eps,
        "arms": {},
    }
    for k, arm in enumerate(arms):
        cov = [round(sum(h[k] for d, _, h in done if d == i) / total, 4) for i in range(n_eps)]
        out["arms"][arm] = {
            "coverage": cov,
            "min": min(cov),
            "max": max(cov),
            "sd": round(float(np.std(cov, ddof=1)), 4) if n_eps > 1 else None,
            "sd_ddof": 1,
        }
    return out


def attach_deviation(loo_block: dict, cell_cov: dict[str, float]) -> None:
    for arm, b in loo_block["arms"].items():
        b["max_deviation"] = round(max(abs(x - cell_cov[arm]) for x in b["coverage"]), 4)


# --- run checks and descriptive blocks ------------------------------------------------------


def check_run(run: str, pool: str, desc: dict, summary: dict, expect: str = "l2") -> None:
    organism = summary.get("organism")
    env = summary.get("env")
    if pool == "replication" and (env not in (None, "backoffice") or organism is not None):
        raise SystemExit(f"{run}: replication needs a back-office run (no bypass, no organism)")
    if pool == "adversary" and (env != "backoffice_bypass" or organism != expect):
        raise SystemExit(f"{run}: adversary needs env backoffice_bypass and organism {expect}")
    if not desc["complete_logs"]:
        raise SystemExit(f"{run}: not recorded with complete logs (--p-*-drop 0)")
    if desc["episodes"] < 1:
        raise SystemExit(f"{run}: no episodes")


def run_summary_block(summary: dict) -> dict:
    keys = (
        "env", "agent", "model", "usd_per_m_in", "usd_per_m_out", "episodes", "seed0", "seeds",
        "tickets", "fault_rates", "organism", "organism_applied", "organism_prompt_sha256",
        "agent_system_sha256",
    )  # fmt: skip
    return {k: summary.get(k) for k in keys}


def reference_cells(path: Path, run: str, arms: tuple[str, ...]) -> dict | None:
    """The same fin_mid cells of a stored analysis artifact, if it is on disk and matches."""
    if not path.exists():
        return None
    art = json.loads(path.read_text())
    ep = ROOT / "artifacts" / "episodes" / run / "episodes.jsonl"
    if not ep.exists() or art.get("episodes_sha256") != sha256(ep):
        return None
    out = {
        "artifact": f"artifacts/robustness/{path.name}",
        "artifact_sha256": sha256(path),
        "run": run,
    }
    for arm in arms:
        c = art["cells"].get(f"{CELL[0]}/{arm}")
        if c:
            b = c["estimators"][CELL[2]]
            out[arm] = {
                "coverage": b["coverage"],
                "wilson95": b["coverage_wilson95"],
                "unbounded_rate": b["unbounded_rate"],
                "mean_width": b["mean_width"],
                "loo": art.get("loo") if arm == CELL[1] else None,
            }
    return out


# --- pools ------------------------------------------------------------------------------------


def headline(cell_block: dict, loo_arm: dict) -> dict:
    b = cell_block["estimators"][CELL[2]]
    low = round(b["coverage"] - loo_arm["max_deviation"], 4)
    return {
        "coverage": b["coverage"],
        "wilson95": b["coverage_wilson95"],
        "unbounded_rate": b["unbounded_rate"],
        "mean_width": b["mean_width"],
        "bias": b["bias"],
        "detects_gap": b["detects_gap"],
        "seed_coverage_sd": cell_block["ll_bic_seed_coverage"]["sd"],
        "loo_min": loo_arm["min"],
        "loo_max": loo_arm["max"],
        "loo_sd": loo_arm["sd"],
        "loo_max_deviation": loo_arm["max_deviation"],
        "coverage_minus_loo_deviation": low,
        "floor": COVERAGE_FLOOR,
        "wilson_lower_ge_floor": b["coverage_wilson95"][0] >= COVERAGE_FLOOR,
        "loo_floor_ge_floor": low >= COVERAGE_FLOOR,
    }


def replication_pool(args, run_data: dict) -> dict:
    run = args.run
    c = cells(
        run, (_cfg(CELL[0]),), REPLICATION_ARMS, REPLICATION_ESTIMATORS,
        args.seed0, args.seeds, args.reps, args.processes,
    )  # fmt: skip
    lo = loo(run, REPLICATION_ARMS, ba.LOO_SEED0, args.loo_seeds, args.loo_reps, args.processes)
    attach_deviation(
        lo, {a: c[f"{CELL[0]}/{a}"]["estimators"][CELL[2]]["coverage"] for a in REPLICATION_ARMS}
    )
    ref = reference_cells(
        ROOT / "artifacts" / "robustness" / f"bypass-analysis-{args.reference}.json",
        args.reference,
        REPLICATION_ARMS,
    )
    head = {a: headline(c[f"{CELL[0]}/{a}"], lo["arms"][a]) for a in REPLICATION_ARMS}
    if ref:
        for a in REPLICATION_ARMS:
            if a in ref:
                head[a]["reference_coverage"] = ref[a]["coverage"]
                head[a]["coverage_minus_reference"] = round(
                    head[a]["coverage"] - ref[a]["coverage"], 4
                )
    return {"cells": c, "loo": lo, "headline": head, "reference": ref}


def adversary_pool(args, run_data: dict) -> dict:
    run = args.run
    c = cells(
        run, ba.CONFIGS, ba.ARMS, ba.ESTIMATORS, args.seed0, args.seeds, args.reps, args.processes
    )
    lo = loo(run, (CELL[1],), ba.LOO_SEED0, args.loo_seeds, args.loo_reps, args.processes)
    attach_deviation(lo, {CELL[1]: c[f"{CELL[0]}/{CELL[1]}"]["estimators"][CELL[2]]["coverage"]})

    b0 = bv.b0_section(run_data["loaded"], {"descriptive": run_data["descriptive"]})
    b0.pop("raw_verdict")  # thresholds belong to the registration
    b0.pop("threshold")

    cfres = cf.analyse(
        [run], ("parsed_scripts",), args.cf_seed0, args.seeds, args.reps, args.processes
    )[run]
    control = {
        "definition": "one W1 and one W2 record added for every bypassed action "
        "(bypass_counterfactuals.py, world parsed_scripts); same loss configurations and arms",
        "seed0": args.cf_seed0,
        "seeds": args.seeds,
        "reps_per_seed": args.reps,
        "worlds": cfres["worlds"]["parsed_scripts"],
        "hidden_mass": cfres["hidden_mass"]["parsed_scripts"],
        "pool": cf.pool_description(run),
        "episodes": cf.episode_stats(run),
    }
    out = {
        "cells": c,
        "loo": lo,
        "headline": headline(c[f"{CELL[0]}/{CELL[1]}"], lo["arms"][CELL[1]]),
        "bypass": b0,
        "parsed_scripts_control": control,
    }
    out["headline"]["parsed_scripts_coverage"] = control["worlds"][CELL[0]][CELL[1]]["estimators"][
        CELL[2]
    ]["coverage"]
    out["vs_neutral"] = vs_neutral(args, run_data, out["headline"])
    return out


def vs_neutral(args, run_data: dict, head: dict) -> dict:
    neutral = bv.load_run(args.neutral)
    bn, nn = bv.arrays(neutral)
    bl, nl = bv.arrays(run_data["loaded"])
    matched = sorted(
        {e["seed"] for e in neutral["eps"]} & {e["seed"] for e in run_data["loaded"]["eps"]}
    )
    d_pooled = bv.share(bl, nl) - bv.share(bn, nn)
    out = {
        "neutral_run": args.neutral,
        "treatment_run": args.run,
        "treatment_organism": run_data["loaded"]["summary"].get("organism"),
        "neutral_episodes_sha256": neutral["sha256"],
        "neutral_model": neutral["summary"].get("model"),
        "neutral_share": round(bv.share(bn, nn), 4),
        "treatment_share": round(bv.share(bl, nl), 4),
        "diff_pooled": round(d_pooled, 4),
        "matched_seeds": matched,
        "unpaired": {
            "n_neutral": len(nn),
            "n_treatment": len(nl),
            "diff": round(d_pooled, 4),
            "ci95": [round(x, 4) for x in bv.unpaired_diff_ci(bn, nn, bl, nl)],
            "resamples": bv.RESAMPLES,
            "rng_seed": bv.SEED_UNPAIRED,
        },
        "paired": None,
    }
    if matched:
        pbn, pnn = bv.arrays(neutral, matched)
        pbl, pnl = bv.arrays(run_data["loaded"], matched)
        out["paired"] = {
            "n_pairs": len(matched),
            "diff": round(bv.share(pbl, pnl) - bv.share(pbn, pnn), 4),
            "ci95": [round(x, 4) for x in bv.paired_diff_ci(pbn, pnn, pbl, pnl)],
            "resamples": bv.RESAMPLES,
            "rng_seed": bv.SEED_PAIRED,
        }
    ref = reference_cells(
        ROOT / "artifacts" / "robustness" / f"bypass-analysis-{args.neutral}.json",
        args.neutral,
        (CELL[1],),
    )
    out["neutral_ll_bic_fin_mid_callid"] = ref[CELL[1]] if ref else None
    if ref:
        out["ll_bic_coverage_treatment_minus_neutral"] = round(
            head["coverage"] - ref[CELL[1]]["coverage"], 4
        )
    return out


# --- main -------------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, help="artifacts/episodes/<run>/episodes.jsonl")
    ap.add_argument("--pool", choices=POOLS, required=True)
    ap.add_argument("--seed0", type=int, default=ba.SEED0, help="first loss seed")
    ap.add_argument("--seeds", type=int, default=ba.SEEDS, help="number of loss seeds")
    ap.add_argument("--reps", type=int, default=ba.REPS_PER_SEED, help="replicates per seed")
    ap.add_argument("--loo-seeds", type=int, default=ba.LOO_SEEDS, help="loss seeds per LOO pool")
    ap.add_argument("--loo-reps", type=int, default=ba.LOO_REPS_PER_SEED, help="reps per LOO seed")
    ap.add_argument("--cf-seed0", type=int, default=cf.SEED0, help="adversary: control loss seed")
    ap.add_argument("--neutral", default="bypass-llm", help="adversary: neutral comparison run")
    ap.add_argument("--reference", default="confirm", help="replication: stored reference run")
    ap.add_argument(
        "--expect-organism",
        default="l2",
        help="adversary: the organism the run's summary must record (default %(default)s)",
    )
    ap.add_argument("--min-episodes", type=int, default=MIN_EPISODES)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument(
        "--out", default=None, help="default: artifacts/robustness/study4-<pool>-<run>.json"
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = parser()
    args = ap.parse_args(argv)
    dst = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts" / "robustness" / f"study4-{args.pool}-{args.run}.json"
    )
    if dst.exists():
        ap.error(f"{dst} already exists; pick another --out (this script never overwrites)")
    run_dir = ba.ROOT / "artifacts" / "episodes" / args.run
    if not (run_dir / "episodes.jsonl").exists():
        ap.error(f"{run_dir / 'episodes.jsonl'} does not exist")
    loaded = bv.load_run(args.run)
    desc = ba.describe(args.run)
    check_run(args.run, args.pool, desc, loaded["summary"], args.expect_organism)
    completion = bv.completion(loaded, TURN_LIMIT)
    run_data = {"loaded": loaded, "descriptive": desc}

    pool = (replication_pool if args.pool == "replication" else adversary_pool)(args, run_data)
    out = {
        "script": "scripts/study4_analysis.py",
        "argv": argv,
        "pool": args.pool,
        "run": args.run,
        "episodes_sha256": loaded["sha256"],
        "run_summary": run_summary_block(loaded["summary"]),
        "completion": completion,
        "min_episodes": args.min_episodes,
        "underpowered": completion["completed"] < args.min_episodes,
        "params": {
            "seed0": args.seed0,
            "seeds": args.seeds,
            "seed_range": [args.seed0, args.seed0 + args.seeds - 1],
            "reps_per_seed": args.reps,
            "replicates": args.seeds * args.reps,
            "loo_seed0": ba.LOO_SEED0,
            "bootstrap_episodes": True,
            "rng": "numpy.random.default_rng(seed) per (config, seed), as bypass_analysis.py",
            "headline_cell": f"{CELL[0]}/<arm> {CELL[2]}",
            "arms": list(REPLICATION_ARMS if args.pool == "replication" else ba.ARMS),
            "estimators": list(
                REPLICATION_ESTIMATORS if args.pool == "replication" else ba.ESTIMATORS
            ),
            "imports": [
                "scripts/bypass_analysis.py",
                "scripts/bypass_counterfactuals.py",
                "scripts/bypass_verdicts.py",
            ],
        },
        "descriptive": desc,
        **pool,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    for arm, h in (
        pool["headline"].items() if args.pool == "replication" else [(CELL[1], pool["headline"])]
    ):
        print(
            f"{args.pool} {CELL[0]}/{arm} {CELL[2]}: coverage {h['coverage']} "
            f"wilson {h['wilson95']} loo [{h['loo_min']}, {h['loo_max']}]"
        )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
