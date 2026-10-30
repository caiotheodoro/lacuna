"""POST HOC counterfactuals for the bypass study. Nothing here is pre-registered.

docs/PREREG-BYPASS.md (D63) failed B1: ll_bic at fin_mid under call-id linkage covered 0.759 on
the neutral LLM run, which sent most of its actions through `run_script`. A refuter then found
(a) under fin_mid most of the hidden actions are `update_customer` calls inside scripts, which
no witness sees (W1 and W2 log one `run_script` line, and the financial W3 view cannot see
`update_customer`), so their capture probability is zero, and (b) a replay that gives the logs a
record for every bypassed action restores coverage. This script makes both checkable from a
committed file. It calls no model and reuses scripts/bypass_analysis.py (frozen) by import.

Worlds (what the witnesses recorded), replayed under the same loss configurations and arms:

- `as_registered`: the recorded logs, unchanged. At the registered seeds and replicates it
  reproduces the cells of artifacts/robustness/bypass-analysis-<run>.json exactly.
- `parsed_scripts`: as an auditor that parsed `run_script` text would have it. For every
  bypassed action (a ledger action with no gateway action record) add one W1 trace record and
  one W2 gateway record, each carrying the ledger entry's (op, resource, value) as a direct
  call's record does (`witnesses.from_call` and the ledger normalise a call the same way, and
  on these runs every direct W1/W2 record equals its ledger entry field for field), and the
  ledger id as `oracle`. The oracle of a W1/W2 record is the ledger id looked up from the line's
  call id, so grouping by it is grouping by call id; here each action gets its own id, standing
  for "<script call id>#<ordinal>" (`Router.script_ledger`). Scripts the interpreter rejected
  ran no action, so they add nothing. The extra records are appended after the recorded ones,
  so the loss draws differ from `as_registered` from the first extra record on (same seeds,
  different stream); that is part of the contrast, not a pairing.
- `w3_full_view_on_bypassed` (fin_mid only; at full_mid it is `as_registered`): the logs as
  recorded, but W3 also sees the bypassed actions the financial view hides. After thinning, the
  recorded W3 records of bypassed actions of an op outside the view are put back. The loss
  draws are those of `as_registered`. This isolates the zero-capture-probability part.

Hidden-mass decomposition (`hidden_mass`). A hidden action is a ledger action none of W1, W2, W3
kept (oracle link). Each is classed by the recorded data, before any loss, under the config's
W3 view: `bypassed_w3_invisible` (bypassed, and W3 has no record the view lets through: zero
capture probability while the logs are blind), `bypassed_w3_visible` (bypassed, W3 can see it)
and `direct`. Means per replicate, by op. `pool` gives the same classes as counts in the run.

Episodes (`episodes`): per episode, the bypass share (bypassed / ledger actions), the run_script
calls and, from the scripts' own replies in the transcript, how many were rejected whole
(`syntax`: "script does not parse"; `invalid`: any other validation error; `too_long`) and how
many lines ran and failed. `paired` summarises seeds 9500-9511, run in both organisms.

Usage:
  uv run python scripts/bypass_counterfactuals.py

Deterministic per seed: seed s uses `numpy.random.default_rng(s)`. Writes
artifacts/robustness/bypass-counterfactuals.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bypass_analysis as ba  # noqa: E402

from lacuna.thinning import ENV_VIEWS, thin  # noqa: E402
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record  # noqa: E402

ROOT = ba.ROOT
SEED0 = 98_500  # disjoint from every seed range of bypass_analysis.py
SEEDS = 30
REPS_PER_SEED = 100
SANITY_RUN = "confirm"
SANITY_SEEDS = 10
RUNS = ("bypass-llm", "bypass-l1")
WORLDS = ("as_registered", "parsed_scripts", "w3_full_view_on_bypassed")
WORLD_CONFIGS = {
    "as_registered": ("fin_mid", "full_mid"),
    "parsed_scripts": ("fin_mid", "full_mid"),
    "w3_full_view_on_bypassed": ("fin_mid",),
}
ESTIMATORS = ("ll_bic", "ll_independence", "chapman_w1w3", "union_count")
CLASSES = ("bypassed_w3_invisible", "bypassed_w3_visible", "direct")
PAIRED_SEEDS = tuple(range(9500, 9512))
CONFIG_BY_NAME = {c.name: c for c in ba.CONFIGS}


class World(NamedTuple):
    ep: ba.Ep
    bypassed: frozenset
    restore: tuple  # W3 records of bypassed actions, put back after thinning if the view drops them


@lru_cache(None)
def worlds(run: str, world: str) -> list[World]:
    """The run's episodes as the witnesses would have recorded them in `world`."""
    out = []
    for e in ba.load_episodes(run):
        b = frozenset(ba.bypassed_ids(e.ledger, e.records))
        if world == "parsed_scripts":
            extra = [
                Record(w, x.op, x.resource, x.value, x.id)
                for x in e.ledger
                if x.id in b
                for w in (TRACE, GATEWAY)
            ]
            out.append(World(ba.Ep(e.ledger, e.records + extra, e.transcript), b, ()))
        elif world == "w3_full_view_on_bypassed":
            keep = tuple(r for r in e.records if r.witness == ENV and r.oracle in b)
            out.append(World(e, b, keep))
        elif world == "as_registered":
            out.append(World(e, b, ()))
        else:
            raise ValueError(f"unknown world {world!r}; choose from {WORLDS}")
    return out


@lru_cache(None)
def classes(run: str, cfg_name: str) -> list[dict[int, str]]:
    """Class of every ledger action of each episode, from the recorded data and the W3 view."""
    view = ENV_VIEWS[CONFIG_BY_NAME[cfg_name].env_view]
    out = []
    for w in worlds(run, "as_registered"):
        visible = {
            r.oracle for r in w.ep.records if r.witness == ENV and (view is None or r.op in view)
        }
        out.append(
            {
                x.id: (
                    "direct"
                    if x.id not in w.bypassed
                    else "bypassed_w3_visible"
                    if x.id in visible
                    else "bypassed_w3_invisible"
                )
                for x in w.ep.ledger
            }
        )
    return out


def run_seed(run: str, world: str, cfg, seed: int, reps: int) -> list[dict]:
    """`reps` replicates of one loss seed in `world`: true N, each arm's union and estimates,
    and the hidden actions by (class, op). Same resampling and thinning as
    `bypass_analysis.run_seed`."""
    ws = worlds(run, world)
    cls = classes(run, cfg.name)
    view = ENV_VIEWS[cfg.env_view]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        idx = rng.integers(0, len(ws), len(ws))
        parts: dict[str, list[np.ndarray]] = {a: [] for a in ba.ARMS}
        hidden: Counter = Counter()
        n = 0
        for i in idx:
            w = ws[i]
            n += len(w.ep.ledger)
            kept = thin(w.ep.records, cfg, rng)
            if w.restore and view is not None:
                kept = kept + [r for r in w.restore if r.op not in view]
            seen = {r.oracle for r in kept if r.witness in (TRACE, GATEWAY, ENV)}
            for x in w.ep.ledger:
                if x.id not in seen:
                    hidden[(cls[i][x.id], x.op)] += 1
            for a in ba.ARMS:
                parts[a].append(ba.arm_table(w.ep.ledger, kept, a))
        t3 = {a: np.sum(parts[a], axis=0) for a in ba.ARMS}
        union = {a: int(t3[a][1:].sum()) for a in ba.ARMS}
        rows.append(
            {
                "n": n,
                "union": union,
                "hidden": hidden,
                "est": {a: ba.estimate_all(t3[a], None, union[a], ESTIMATORS) for a in ba.ARMS},
            }
        )
    return rows


def _job(args: tuple) -> tuple[str, str, str, int, list[dict]]:
    run, world, cfg_name, seed, reps = args
    return run, world, cfg_name, seed, run_seed(run, world, CONFIG_BY_NAME[cfg_name], seed, reps)


def _cell(rows: list[dict], cfg, arm: str) -> dict:
    ns = [r["n"] for r in rows]
    unions = [r["union"][arm] for r in rows]
    return {
        "config": asdict(cfg),
        "arm": arm,
        "replicates": len(rows),
        "mean_n_true": round(float(np.mean(ns)), 2),
        "mean_union": round(float(np.mean(unions)), 2),
        "mean_hidden": round(float(np.mean([n - u for n, u in zip(ns, unions, strict=True)])), 2),
        "estimators": {
            name: ba.summarise([r["est"][arm][name] for r in rows], ns, unions)
            for name in ESTIMATORS
        },
    }


def _hidden(rows: list[dict]) -> dict:
    """Mean hidden actions per replicate (oracle: ledger actions no witness kept), by class, op."""
    tot: Counter = Counter()
    for r in rows:
        tot.update(r["hidden"])
    reps = len(rows)
    by_class = {c: {} for c in CLASSES}
    for (c, op), k in sorted(tot.items()):
        by_class[c][op] = round(k / reps, 2)
    return {
        "mean_hidden": round(sum(tot.values()) / reps, 2),
        "mean_hidden_by_class": {c: round(sum(v.values()), 2) for c, v in by_class.items()},
        "mean_hidden_by_class_op": by_class,
    }


def analyse(runs, worlds_, seed0: int, seeds: int, reps: int, processes: int) -> dict:
    jobs = [
        (run, world, cfg_name, s, reps)
        for run in runs
        for world in worlds_
        for cfg_name in WORLD_CONFIGS[world]
        for s in range(seed0, seed0 + seeds)
    ]
    if processes > 1:
        with Pool(processes) as pool:
            done = pool.map(_job, jobs, chunksize=1)
    else:
        done = [_job(j) for j in jobs]
    out: dict = {}
    for run in runs:
        res = out.setdefault(run, {"worlds": {}, "hidden_mass": {}})
        for world in worlds_:
            for cfg_name in WORLD_CONFIGS[world]:
                rows = [
                    r
                    for rn, w, c, _, rs in done
                    if (rn, w, c) == (run, world, cfg_name)
                    for r in rs
                ]  # seed order
                cfg = CONFIG_BY_NAME[cfg_name]
                res["worlds"].setdefault(world, {})[cfg_name] = {
                    arm: _cell(rows, cfg, arm) for arm in ba.ARMS
                }
                res["hidden_mass"].setdefault(world, {})[cfg_name] = _hidden(rows)
    return out


# --- the recorded pool and the episodes ----------------------------------------------------


def pool_description(run: str) -> dict:
    """Counts in the recorded run: bypassed actions by op and W3 visibility, per config view."""
    eps = ba.load_episodes(run)
    n = sum(len(e.ledger) for e in eps)
    out: dict = {
        "n_true": n,
        "n_extra_records_parsed_scripts": sum(
            len(p.ep.records) - len(o.ep.records)
            for p, o in zip(
                worlds(run, "parsed_scripts"), worlds(run, "as_registered"), strict=True
            )
        ),
    }
    for cfg_name in ("fin_mid", "full_mid"):
        by: dict = {c: Counter() for c in CLASSES}
        for e, cl in zip(eps, classes(run, cfg_name), strict=True):
            for x in e.ledger:
                by[cl[x.id]][x.op] += 1
        out[cfg_name] = {
            "by_class": {c: sum(v.values()) for c, v in by.items()},
            "by_class_op": {c: dict(sorted(v.items())) for c, v in by.items()},
        }
    return out


def _script_replies(transcript: list[dict]) -> list[str]:
    ids = {
        tc["id"]
        for m in transcript
        for tc in m.get("tool_calls") or []
        if tc["function"]["name"] == "run_script"
    }
    return [m["content"] for m in transcript if m["role"] == "tool" and m["tool_call_id"] in ids]


def reject_kind(content: str) -> str | None:
    """How the interpreter rejected a whole script, None if it ran (a `results` reply)."""
    try:
        j = json.loads(content)
    except json.JSONDecodeError:
        return "invalid"
    if isinstance(j, dict) and "results" in j:
        return None
    err = str(j.get("error", "")) if isinstance(j, dict) else ""
    if err.startswith("script does not parse"):
        return "syntax"
    if err.startswith(("more than", "script longer")):
        return "too_long"
    return "invalid"


def episode_stats(run: str) -> list[dict]:
    path = ROOT / "artifacts" / "episodes" / run / "episodes.jsonl"
    seeds = [json.loads(line)["seed"] for line in path.open()]
    out = []
    for seed, e in zip(seeds, ba.load_episodes(run), strict=True):
        b = ba.bypassed_ids(e.ledger, e.records)
        kinds = Counter(reject_kind(c) for c in _script_replies(e.transcript))
        ran = kinds.pop(None, 0)
        line_errors = 0
        for c in _script_replies(e.transcript):
            if reject_kind(c) is None:
                line_errors += sum(r.get("status") == "error" for r in json.loads(c)["results"])
        calls = ran + sum(kinds.values())
        rejected = sum(kinds.values())
        out.append(
            {
                "seed": seed,
                "paired": seed in PAIRED_SEEDS,
                "n_ledger": len(e.ledger),
                "n_bypassed": len(b),
                "bypass_share": round(len(b) / len(e.ledger), 4) if e.ledger else None,
                "n_script_calls": calls,
                "n_scripts_ran": ran,
                "n_scripts_rejected": rejected,
                "rejected_by_kind": {k: kinds.get(k, 0) for k in ("syntax", "invalid", "too_long")},
                "n_lines_failed_in_scripts_that_ran": line_errors,
                "all_scripts_rejected": bool(calls) and rejected == calls,
            }
        )
    return out


def paired_summary(neutral: list[dict], l1: list[dict]) -> dict:
    nb = {e["seed"]: e for e in neutral if e["paired"]}
    lb = {e["seed"]: e for e in l1 if e["paired"]}
    seeds = sorted(set(nb) & set(lb))
    eff = {s: round(lb[s]["bypass_share"] - nb[s]["bypass_share"], 4) for s in seeds}

    def mean(xs):
        xs = list(xs)
        return round(float(np.mean(xs)), 4) if xs else None

    def pooled(side, ss):
        n = sum(side[s]["n_ledger"] for s in ss)
        return round(sum(side[s]["n_bypassed"] for s in ss) / n, 4) if n else None

    out = {
        "seeds": seeds,
        "neutral_share": {s: nb[s]["bypass_share"] for s in seeds},
        "l1_share": {s: lb[s]["bypass_share"] for s in seeds},
        "l1_minus_neutral": eff,
        "mean_l1_minus_neutral": mean(eff.values()),
        "mean_neutral_share": mean(nb[s]["bypass_share"] for s in seeds),
        "mean_l1_share": mean(lb[s]["bypass_share"] for s in seeds),
        "pooled_neutral_share": pooled(nb, seeds),
        "pooled_l1_share": pooled(lb, seeds),
        "neutral_all_scripts_rejected": [s for s in seeds if nb[s]["all_scripts_rejected"]],
        "l1_all_scripts_rejected": [s for s in seeds if lb[s]["all_scripts_rejected"]],
    }
    for drop in ((9503, 9509), (9503,), (9509,)):
        rest = [s for s in seeds if s not in drop]
        key = "without_" + "_".join(map(str, drop))
        out[key] = {
            "seeds": rest,
            "mean_l1_minus_neutral": mean(eff[s] for s in rest),
            "mean_neutral_share": mean(nb[s]["bypass_share"] for s in rest),
            "mean_l1_share": mean(lb[s]["bypass_share"] for s in rest),
        }
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", default=list(RUNS), help="runs under study")
    ap.add_argument(
        "--seed0", type=int, default=SEED0, help="first loss seed (default %(default)s)"
    )
    ap.add_argument("--seeds", type=int, default=SEEDS, help="number of loss seeds")
    ap.add_argument("--reps", type=int, default=REPS_PER_SEED, help="replicates per seed")
    ap.add_argument(
        "--sanity-run", default=SANITY_RUN, help="no-bypass run; '' skips it (default %(default)s)"
    )
    ap.add_argument("--sanity-seeds", type=int, default=SANITY_SEEDS)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument(
        "--out", default=None, help="default: artifacts/robustness/bypass-counterfactuals.json"
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    runs = list(args.runs)
    res = analyse(runs, WORLDS, args.seed0, args.seeds, args.reps, args.processes)
    out: dict = {
        "script": "scripts/bypass_counterfactuals.py",
        "argv": argv,
        "post_hoc": True,
        "status": "post hoc and exploratory: not pre-registered, run after B1 failed "
        "(docs/PREREG-BYPASS.md, D63)",
        "params": {
            "seed0": args.seed0,
            "seeds": args.seeds,
            "seed_range": [args.seed0, args.seed0 + args.seeds - 1],
            "reps_per_seed": args.reps,
            "replicates": args.seeds * args.reps,
            "bootstrap_episodes": True,
            "rng": "numpy.random.default_rng(seed) per (run, world, config, seed); one stream "
            "shared by all arms",
            "worlds": list(WORLDS),
            "world_configs": {w: list(c) for w, c in WORLD_CONFIGS.items()},
            "arms": list(ba.ARMS),
            "estimators": list(ESTIMATORS),
            "analysis_module": "scripts/bypass_analysis.py (imported, unchanged)",
            "world_definitions": {
                "as_registered": "recorded logs unchanged",
                "parsed_scripts": "one W1 trace and one W2 gateway record per bypassed action: "
                "the ledger entry's (op, resource, value), oracle = ledger id (the call id of "
                "its script plus ordinal); appended after the recorded records",
                "w3_full_view_on_bypassed": "recorded logs; after thinning, W3 records of "
                "bypassed actions outside the config's W3 view are put back (fin_mid only)",
            },
            "hidden_definition": "ledger action none of W1, W2, W3 kept (oracle link)",
            "classes": list(CLASSES),
        },
        "runs": {},
    }
    for run in runs:
        run_dir = ROOT / "artifacts" / "episodes" / run
        out["runs"][run] = {
            "episodes_sha256": hashlib.sha256(
                (run_dir / "episodes.jsonl").read_bytes()
            ).hexdigest(),
            "pool": pool_description(run),
            **res[run],
        }
    if args.sanity_run:
        s = analyse(
            [args.sanity_run],
            ("as_registered", "parsed_scripts"),
            args.seed0,
            args.sanity_seeds,
            args.reps,
            args.processes,
        )
        run_dir = ROOT / "artifacts" / "episodes" / args.sanity_run
        out["no_bypass_sanity"] = {
            "run": args.sanity_run,
            "seeds": args.sanity_seeds,
            "seed_range": [args.seed0, args.seed0 + args.sanity_seeds - 1],
            "episodes_sha256": hashlib.sha256(
                (run_dir / "episodes.jsonl").read_bytes()
            ).hexdigest(),
            "pool": pool_description(args.sanity_run),
            **s[args.sanity_run],
        }
    ep = {run: episode_stats(run) for run in RUNS if (ROOT / "artifacts/episodes" / run).is_dir()}
    out["episodes"] = ep
    if len(ep) == 2:
        out["paired"] = paired_summary(ep["bypass-llm"], ep["bypass-l1"])
    dst = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts" / "robustness" / "bypass-counterfactuals.json"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    for run in runs:
        for world, by_cfg in out["runs"][run]["worlds"].items():
            for cfg_name, arms in by_cfg.items():
                print(
                    run,
                    world,
                    cfg_name,
                    {a: arms[a]["estimators"]["ll_bic"]["coverage"] for a in arms},
                )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
