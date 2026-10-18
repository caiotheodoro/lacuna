#!/usr/bin/env python3
"""Robustness checks on the infra transfer result. Post hoc: none of this can
change the registered T1-T4 verdicts (artifacts/transfer_hypotheses.json),
and none of it calls a model.

  uv run python scripts/transfer_robustness.py precision
  uv run python scripts/transfer_robustness.py mechanism
  uv run python scripts/transfer_robustness.py density
  uv run python scripts/transfer_robustness.py falsegap
  uv run python scripts/transfer_robustness.py behaviour
  uv run python scripts/transfer_robustness.py render     # markdown for docs/TRANSFER.md

Each writes artifacts/robustness/transfer-<name>.json.

- precision: T1's configurations at 20 fresh loss seeds of 100 replicates,
  both linkages on the same random numbers (scripts/replay.py), on the paid
  run and the 24-episode scripted run.
- mechanism: a linkage ablation. Records are grouped by a key and paired in
  order within it, as signature linkage does; the key is the signature for
  some records and the ledger id (exact matching) for others. Which records
  get exact matching is chosen by how their signature group splits across
  W3 (all seen, all missed, mixed), by ticket kind, or by op. Every variant
  sees the same thinned records in each replicate.
- density: T1 on free scripted episodes at other ticket counts and fleet
  sizes, to see where the registered verdict sits.
- falsegap: per-op mixes of signature and exact linkage under `anti_full`
  (nothing hidden), on the infra run and, for contrast, on the back-office
  confirmatory run under `anti_fin`.
- behaviour: how many of the paid agent's actions the scripted plans contain,
  and the shape of the provider replies around the failed episode.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from replay import replicates, summarise
from thin_episodes import load

from lacuna.agents.scripted import ScriptedAgent, ScriptedAuditor
from lacuna.envs.infra import ACTION_OPS
from lacuna.envs.infra_tasks import SERVICES, make_episode
from lacuna.envs.registry import ENVS
from lacuna.episode import run_episode
from lacuna.estimators import loglinear
from lacuna.linkage import VALUE_OPS, signature
from lacuna.thinning import POOLS, thin
from lacuna.witnesses import ENV, Record

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "robustness"
INFRA = ENVS["infra"]
TRANSFER = {c.name: c for c in POOLS["transfer"]}
FALSEPOS = {c.name: c for c in POOLS["falsepos"]}
RUNS = ("transfer", "transfer-scripted-24")
REPS = 100
# Loss seeds, disjoint from 4242 (registered) and from main's robustness seeds.
PRECISION_SEEDS = range(12_000, 12_020)
MECHANISM_SEEDS = range(13_000, 13_010)
FALSEGAP_SEEDS = range(14_000, 14_005)
DENSITY_SEEDS = range(15_000, 15_002)
DENSITY_EPISODE_SEED0 = 77_000
UNDO_KINDS = ("deploy-fail", "loadtest", "debug")
TEAMS = ("core", "money", "shop", "data", "ops")


def write(name: str, out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"transfer-{name}.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dst}")


def episodes_with_seeds(run: str) -> list[tuple]:
    seeds = [
        json.loads(line)["seed"]
        for line in (ROOT / "artifacts/episodes" / run / "episodes.jsonl").open()
    ]
    return [(led, recs, s) for (led, recs), s in zip(load(run), seeds, strict=True)]


# Keyed linkage ---------------------------------------------------------------

Key = Callable[[int, int, Record], object]


def table3(records: list[Record], key: Callable[[Record], object]) -> np.ndarray:
    """W1-W3 table: records grouped by `key`, paired in order within a group.

    With `key` the effect signature this is `signature_table`; with the ledger
    id it is the oracle table's observed cells.
    """
    per: dict = defaultdict(Counter)
    for r in records:
        if r.witness < 3:
            per[key(r)][r.witness] += 1
    t = np.zeros(8, dtype=np.int64)
    for counts in per.values():
        for j in range(max(counts.values())):
            t[sum(1 << w for w, c in counts.items() if j < c)] += 1
    return t


def sig(k: int, i: int, r: Record) -> object:
    return signature(r.op, r.resource, r.value)


def exact(k: int, i: int, r: Record) -> object:
    # k is the position in the bootstrap sample, so a resampled episode's
    # actions stay distinct from its copy's.
    return ("id", k, r.oracle)


def exact_if(pred: Callable[[int, Record], bool]) -> Key:
    return lambda k, i, r: exact(k, i, r) if pred(i, r) else sig(k, i, r)


def keyed_replicates(episodes: list, cfg, variants: dict[str, Key], reps: int, seed: int) -> list:
    """Same random numbers as replay.replicates: resample, then thin in order."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        idx = rng.integers(0, len(episodes), len(episodes))
        n = 0
        tabs = {v: np.zeros(8, dtype=np.int64) for v in variants}
        for k, i in enumerate(idx):
            led, recs = episodes[i][0], episodes[i][1]
            n += len(led)
            kept = thin(recs, cfg, rng)
            for v, fn in variants.items():
                tabs[v] += table3(kept, functools.partial(fn, k, int(i)))
        row: dict = {"n": n}
        for v, t in tabs.items():
            row[v] = (t, loglinear.best_bic(t))
        rows.append(row)
    return rows


def keyed_summary(rows: list, v: str) -> dict:
    flat = [{"n": r["n"], "union": int(r[v][0][1:].sum()), v: r[v][1]} for r in rows]
    s = summarise(flat, v)
    s["mean_union"] = round(float(np.mean([f["union"] for f in flat])), 2)
    s["mean_w1w2_overlap"] = round(
        float(np.mean([r[v][0][0b011] + r[v][0][0b111] for r in rows])), 2
    )
    return s


def check_matches_replay(episodes: list, cfg) -> None:
    """The keyed loop must reproduce replay.replicates exactly for both linkages."""
    two = [(e[0], e[1]) for e in episodes]
    keyed = keyed_replicates(episodes, cfg, {"sig": sig, "exact": exact}, 3, 1)
    for linkage, v in (("signature", "sig"), ("oracle", "exact")):
        ref = replicates(two, cfg, linkage, 3, 1, estimators=("ll_bic",))
        for a, b in zip(ref, keyed, strict=True):
            e = b[v][1]
            assert (a["ll_bic"].point, a["ll_bic"].lo, a["ll_bic"].hi) == (e.point, e.lo, e.hi)
            assert a["union"] == int(b[v][0][1:].sum())


# Classes per ledger id ------------------------------------------------------------


def group_class(led, recs) -> dict[int, str]:
    """How each action's signature group splits across W3 (unthinned, full view)."""
    seen = {r.oracle for r in recs if r.witness == ENV}
    groups: dict = defaultdict(list)
    for x in led:
        groups[signature(x.op, x.resource, x.value)].append(x.id)
    out = {}
    for ids in groups.values():
        n_seen = sum(i in seen for i in ids)
        if len(ids) == 1:
            c = "single"
        elif n_seen == 0:
            c = "all-invisible"
        elif n_seen == len(ids):
            c = "all-visible"
        else:
            c = "mixed"
        out.update(dict.fromkeys(ids, c))
    return out


def kind_of(plan) -> str:
    tools = [t for t, _ in plan]
    if "deploy_version" in tools:
        return "deploy-fail" if "rollback" in tools else "deploy-ok"
    if "create_incident" in tools:
        return "incident"
    if tools == ["rollback"]:
        return "rollback"
    if "write_runbook" in tools:
        return "runbook"
    if tools.count("scale_service") == 2:
        return "loadtest"
    if "scale_service" in tools:
        return "scale"
    keys = [a["key"] for t, a in plan if t == "update_config"]
    if "log_level" in keys:
        return "debug"
    return "tune-twice" if len(keys) == 2 else "tune-once"


def ticket_kinds(led, seed: int, tickets: int) -> dict[int, str]:
    """Ticket kind of each ledger entry: the first unused plan action with its signature."""
    pool: dict = defaultdict(list)
    for t in make_episode(seed, tickets).tickets:
        kind = kind_of(t.plan)
        for tool, args in t.plan:
            if tool in ACTION_OPS:
                pool[signature(*INFRA.normalise(tool, args))].append(kind)
    out = {}
    for x in sorted(led, key=lambda e: e.id):
        bucket = pool.get(signature(x.op, x.resource, x.value))
        out[x.id] = bucket.pop(0) if bucket else "unmatched"
    return out


def tickets_of(run: str) -> int:
    return json.loads((ROOT / "artifacts/episodes" / run / "summary.json").read_text())["tickets"]


def mechanism_variants(gcls: list[dict], kinds: list[dict]) -> dict[str, Key]:
    digest_ops = set(VALUE_OPS) | {"send_page", "write_runbook"}

    def group_in(*classes: str) -> Key:
        return exact_if(lambda i, r: gcls[i][r.oracle] in classes)

    return {
        "signature": sig,
        "exact (oracle)": exact,
        "exact for all-W3-invisible groups": group_in("all-invisible"),
        "exact for mixed groups": group_in("mixed"),
        "exact for all-W3-visible groups": group_in("all-visible"),
        "exact for all-W3-invisible and mixed groups": group_in("all-invisible", "mixed"),
        "exact for undo tickets (deploy-fail, loadtest, debug)": exact_if(
            lambda i, r: kinds[i][r.oracle] in UNDO_KINDS
        ),
        "exact for every other ticket": exact_if(lambda i, r: kinds[i][r.oracle] not in UNDO_KINDS),
        "signature plus content digest for send_page, write_runbook": lambda k, i, r: (
            r.op,
            r.resource,
            r.value if r.op in digest_ops else "",
        ),
    }


def signature_only_for(ops: set[str]) -> Key:
    return exact_if(lambda i, r: r.op not in ops)


def falsegap_variants(run: str, ops: list[str]) -> dict[str, Key]:
    variants: dict[str, Key] = {"exact (oracle)": exact, "signature": sig}
    for op in ops:
        variants[f"signature for {op} only"] = signature_only_for({op})
    if run == "transfer":
        visible = {"send_page", "create_incident"}
        variants["signature for send_page and create_incident only"] = signature_only_for(visible)
        variants["signature for every op but send_page and create_incident"] = signature_only_for(
            set(ops) - visible
        )
    else:
        variants["signature for every op but send_email"] = signature_only_for(
            set(ops) - {"send_email"}
        )
    return variants


# Subcommands ---------------------------------------------------------------------


def precision() -> None:
    out: dict = {
        "seeds": [PRECISION_SEEDS.start, PRECISION_SEEDS.stop - 1],
        "reps_per_seed": REPS,
        "runs": {},
    }
    for run in RUNS:
        episodes = load(run)
        out["runs"][run] = {}
        for cname in ("transfer_mid", "transfer_mid_rho100"):
            for linkage in ("oracle", "signature"):
                rows, per_seed = [], []
                for s in PRECISION_SEEDS:
                    r = replicates(episodes, TRANSFER[cname], linkage, REPS, s)
                    per_seed.append(summarise(r, "ll_bic")["coverage"])
                    rows += r
                cell = {
                    "ll_bic": summarise(rows, "ll_bic"),
                    "ll_independence": summarise(rows, "ll_independence"),
                    "ll_bic_coverage_per_seed": per_seed,
                    "ll_bic_coverage_seed_range": [min(per_seed), max(per_seed)],
                }
                out["runs"][run][f"{cname}/{linkage}"] = cell
                print(
                    run,
                    cname,
                    linkage,
                    cell["ll_bic"]["coverage"],
                    cell["ll_bic_coverage_seed_range"],
                    flush=True,
                )
    write("precision", out)


def mechanism() -> None:
    cfg = TRANSFER["transfer_mid"]
    out: dict = {
        "config": cfg.name,
        "seeds": [MECHANISM_SEEDS.start, MECHANISM_SEEDS.stop - 1],
        "reps_per_seed": REPS,
        "runs": {},
    }
    for run in RUNS:
        episodes = episodes_with_seeds(run)
        check_matches_replay(episodes, cfg)
        tickets = tickets_of(run)
        gcls = [group_class(led, recs) for led, recs, _ in episodes]
        kinds = [ticket_kinds(led, s, tickets) for led, _, s in episodes]
        variants = mechanism_variants(gcls, kinds)
        rows = []
        for s in MECHANISM_SEEDS:
            rows += keyed_replicates(episodes, cfg, variants, REPS, s)
        res = {v: keyed_summary(rows, v) for v in variants}
        sg, ex = res["signature"], res["exact (oracle)"]
        obs = float(
            np.mean([r["signature"][0][1:].sum() - r["exact (oracle)"][0][1:].sum() for r in rows])
        )
        gap = sg["bias"] - ex["bias"]
        group_sizes = Counter(c for g in gcls for c in g.values())
        kind_sizes = Counter(k for m in kinds for k in m.values())
        out["runs"][run] = {
            "variants": res,
            "decomposition": {
                "bias_signature_minus_oracle": round(gap, 2),
                "observed_count_lost_to_merging": round(obs, 2),
                "share_from_observed_count": round(obs / gap, 3),
                "share_from_model_extrapolation": round(1 - obs / gap, 3),
                "w1w2_overlap_signature": sg["mean_w1w2_overlap"],
                "w1w2_overlap_oracle": ex["mean_w1w2_overlap"],
            },
            "actions_by_group_class": dict(sorted(group_sizes.items())),
            "actions_by_ticket_kind": dict(sorted(kind_sizes.items())),
        }
        for v, s in res.items():
            print(run, f"{v:60s}", s["coverage"], s["bias"], flush=True)
        print(run, out["runs"][run]["decomposition"], flush=True)
    write("mechanism", out)


def fleet(n: int) -> dict[str, str]:
    return (
        dict(SERVICES)
        if n == len(SERVICES)
        else {f"svc{i}": TEAMS[i % len(TEAMS)] for i in range(n)}
    )


def density() -> None:
    cfg = TRANSFER["transfer_mid"]
    grid = ((10, 10), (10, 20), (10, 30), (10, 60), (30, 30), (30, 60))
    out: dict = {
        "config": cfg.name,
        "episodes_per_cell": 24,
        "episode_seed0": DENSITY_EPISODE_SEED0,
        "loss_seeds": [DENSITY_SEEDS.start, DENSITY_SEEDS.stop - 1],
        "reps_per_seed": REPS,
        "cells": {},
    }
    for n_services, n_tickets in grid:
        env = dataclasses.replace(
            INFRA, make_episode=functools.partial(make_episode, fleet=fleet(n_services))
        )
        episodes = []
        for j in range(24):
            s = DENSITY_EPISODE_SEED0 + j
            res = run_episode(
                s,
                ScriptedAgent(np.random.default_rng(s)),
                ScriptedAuditor(np.random.default_rng(1), env=env),
                p_gateway_drop=0.0,
                p_trace_drop=0.0,
                n_tickets=n_tickets,
                env=env,
            )
            episodes.append((res.ledger, res.records))
        n = sum(len(led) for led, _ in episodes)
        inv = shared_inv = 0
        for led, recs in episodes:
            cls = group_class(led, recs)
            seen = {r.oracle for r in recs if r.witness == ENV}
            inv += sum(x.id not in seen for x in led)
            shared_inv += sum(
                cls[x.id] in ("all-invisible", "mixed") and x.id not in seen for x in led
            )
        cell: dict = {
            "services": n_services,
            "tickets": n_tickets,
            "n": n,
            "w3_invisible_share": round(inv / n, 3),
            "invisible_sharing_a_signature_share": round(shared_inv / inv, 3),
        }
        for linkage in ("signature", "oracle"):
            rows = []
            for s in DENSITY_SEEDS:
                rows += replicates(episodes, cfg, linkage, REPS, s, estimators=("ll_bic",))
            cell[linkage] = summarise(rows, "ll_bic")
        out["cells"][f"{n_services}x{n_tickets}"] = cell
        print(
            n_services,
            n_tickets,
            cell["signature"]["coverage"],
            cell["oracle"]["coverage"],
            flush=True,
        )
    write("density", out)


def falsegap() -> None:
    out: dict = {
        "seeds": [FALSEGAP_SEEDS.start, FALSEGAP_SEEDS.stop - 1],
        "reps_per_seed": REPS,
        "runs": {},
    }
    for run, cname in (("transfer", "anti_full"), ("confirm", "anti_fin")):
        episodes = episodes_with_seeds(run)
        ops = sorted({x.op for led, _, _ in episodes for x in led})
        variants = falsegap_variants(run, ops)
        rows = []
        for s in FALSEGAP_SEEDS:
            rows += keyed_replicates(episodes, FALSEPOS[cname], variants, REPS, s)
        res = {v: keyed_summary(rows, v) for v in variants}
        out["runs"][run] = {"config": cname, "variants": res}
        for v, s in res.items():
            print(run, cname, f"{v:55s}", "gap", s["detects_gap"], "bias", s["bias"], flush=True)
    # Counterfactual: the same infra episodes with W3 handed a record of every
    # action it missed, so it has no blind spot. Pairing on pages is unchanged.
    complete = []
    for led, recs, seed in episodes_with_seeds("transfer"):
        seen = {r.oracle for r in recs if r.witness == ENV}
        extra = [Record(ENV, x.op, x.resource, x.value, x.id) for x in led if x.id not in seen]
        complete.append((led, recs + extra, seed))
    variants = {"exact (oracle)": exact, "signature": sig}
    rows = []
    for s in FALSEGAP_SEEDS:
        rows += keyed_replicates(complete, FALSEPOS["anti_full"], variants, REPS, s)
    res = {v: keyed_summary(rows, v) for v in variants}
    out["runs"]["transfer, W3 complete (counterfactual)"] = {"config": "anti_full", "variants": res}
    for v, s in res.items():
        print("W3 complete", f"{v:55s}", "gap", s["detects_gap"], "bias", s["bias"], flush=True)
    write("falsegap", out)


def behaviour() -> None:
    out: dict = {"runs": {}}
    for run in RUNS:
        tickets = tickets_of(run)
        n = matched = 0
        for led, _, s in episodes_with_seeds(run):
            kinds = ticket_kinds(led, s, tickets)
            n += len(led)
            matched += sum(k != "unmatched" for k in kinds.values())
        out["runs"][run] = {"actions": n, "in_the_scripted_plans": matched}
    summary = json.loads((ROOT / "artifacts/episodes/transfer/summary.json").read_text())
    receipt = json.loads((ROOT / "artifacts/episodes/transfer/receipt.json").read_text())
    shapes = Counter(c.get("rc", "unrecorded") for c in receipt["calls"])
    failed = []
    for f in summary["failed_episodes"]:
        turns = (f.get("diag") or {}).get("assistant_turns", [])
        failed.append(
            {
                "seed": f["seed"],
                "error": f["error"],
                "assistant_turns_sent": len(turns),
                "turns_without_reasoning_key": [t["i"] for t in turns if t["rc"] == "absent"],
                "turns_with_empty_reasoning": [t["i"] for t in turns if t["rc"] == "empty"],
                "last_three_turns": turns[-3:],
            }
        )
    out["provider"] = {
        "completed_calls": len(receipt["calls"]),
        "completed_calls_without_reasoning_key": shapes.get("absent", 0),
        "completed_calls_with_empty_reasoning": shapes.get("empty", 0),
        "failed_episodes": failed,
    }
    print(json.dumps(out, indent=2)[:1500])
    write("behaviour", out)


def _load(name: str) -> dict:
    return json.loads((OUT / f"transfer-{name}.json").read_text())


def render() -> None:
    """Markdown tables for docs/TRANSFER.md, from the artifacts only (AGENTS.md rule 2)."""
    d = _load("precision")
    print(
        f"Source: `artifacts/robustness/transfer-precision.json` — loss seeds "
        f"{d['seeds'][0]}-{d['seeds'][1]}, {d['reps_per_seed']} replicates each, both linkages "
        "on the same random numbers.\n"
    )
    print(
        "| run | config | linkage | `ll_bic` coverage ± MCSE | per-seed range | bias "
        "| mean width | unbounded | `ll_independence` coverage |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for run, cells in d["runs"].items():
        for key, c in cells.items():
            cfg, lk = key.split("/")
            b, i = c["ll_bic"], c["ll_independence"]
            lo, hi = c["ll_bic_coverage_seed_range"]
            print(
                f"| `{run}` | `{cfg}` | {lk} | {b['coverage']:.3f} ± {b['coverage_mcse']:.3f} "
                f"| {lo:.2f}-{hi:.2f} | {b['bias']} | {b['mean_width']} | {b['unbounded_rate']} "
                f"| {i['coverage']:.3f} |"
            )
    print()
    m = _load("mechanism")
    runs = list(m["runs"])
    print(
        f"Source: `artifacts/robustness/transfer-mechanism.json` — `{m['config']}`, loss seeds "
        f"{m['seeds'][0]}-{m['seeds'][1]}, {m['reps_per_seed']} replicates each; every variant "
        "sees the same thinned records.\n"
    )
    print("| linkage variant | " + " | ".join(f"`{r}` coverage (bias)" for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for v in m["runs"][runs[0]]["variants"]:
        cells = [m["runs"][r]["variants"][v] for r in runs]
        print(f"| {v} | " + " | ".join(f"{c['coverage']:.3f} ({c['bias']})" for c in cells) + " |")
    print()
    print("| | " + " | ".join(f"`{r}`" for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    rows = (
        ("bias, signature minus oracle", "bias_signature_minus_oracle"),
        ("of which observed actions lost to merged groups", "observed_count_lost_to_merging"),
        ("share from the lost observed count", "share_from_observed_count"),
        ("share from the model's extrapolation", "share_from_model_extrapolation"),
        ("trace-and-gateway overlap, signature", "w1w2_overlap_signature"),
        ("trace-and-gateway overlap, oracle", "w1w2_overlap_oracle"),
    )
    for label, key in rows:
        print(
            f"| {label} | "
            + " | ".join(str(m["runs"][r]["decomposition"][key]) for r in runs)
            + " |"
        )
    for label, key in (("actions by signature group", "actions_by_group_class"),):
        for r in runs:
            print(
                f"\n{label}, `{r}`: " + ", ".join(f"{k} {v}" for k, v in m["runs"][r][key].items())
            )
    print()
    d = _load("density")
    print(
        f"Source: `artifacts/robustness/transfer-density.json` — free scripted episodes, "
        f"{d['episodes_per_cell']} per cell from seed {d['episode_seed0']}, `{d['config']}`, loss "
        f"seeds {d['loss_seeds'][0]}-{d['loss_seeds'][1]} x {d['reps_per_seed']} replicates.\n"
    )
    print(
        "| services | tickets per episode | N | W3-invisible share | of those, sharing a signature "
        "| `ll_bic` signature coverage (bias) | `ll_bic` oracle coverage (bias) |"
    )
    print("|---|---|---|---|---|---|---|")
    for c in d["cells"].values():
        sg, orc = c["signature"], c["oracle"]
        print(
            f"| {c['services']} | {c['tickets']} | {c['n']} | {c['w3_invisible_share']} "
            f"| {c['invisible_sharing_a_signature_share']} | {sg['coverage']:.3f} ({sg['bias']}) "
            f"| {orc['coverage']:.3f} ({orc['bias']}) |"
        )
    print()
    f = _load("falsegap")
    print(
        f"Source: `artifacts/robustness/transfer-falsegap.json` — loss seeds "
        f"{f['seeds'][0]}-{f['seeds'][1]}, {f['reps_per_seed']} replicates each. Nothing is "
        "hidden under `anti`, so any detected gap is false.\n"
    )
    print(
        "| episodes | config | linkage | `ll_bic` detects gap | coverage | bias "
        "| unseen in the table |"
    )
    print("|---|---|---|---|---|---|---|")
    for run, r in f["runs"].items():
        for v, c in r["variants"].items():
            print(
                f"| {run} | `{r['config']}` | {v} | {c['detects_gap']:.2f} | {c['coverage']:.3f} "
                f"| {c['bias']} | {c['hidden']} |"
            )
    print()
    b = _load("behaviour")
    print("Source: `artifacts/robustness/transfer-behaviour.json`.\n")
    for run, r in b["runs"].items():
        print(
            f"- `{run}`: {r['in_the_scripted_plans']} of {r['actions']} actions "
            "are in the scripted plans."
        )
    pv = b["provider"]
    print(
        f"- provider: {pv['completed_calls_without_reasoning_key']} of {pv['completed_calls']} "
        "completed calls returned no `reasoning_content` key."
    )
    for e in pv["failed_episodes"]:
        last = ", ".join(f"i={t['i']} {t['rc']}" for t in e["last_three_turns"])
        print(
            f"- seed {e['seed']}: {e['assistant_turns_sent']} assistant turns sent; "
            "no reasoning key "
            f"at {e['turns_without_reasoning_key']}; last three turns: {last}."
        )


def main() -> None:
    cmds = {
        "precision": precision,
        "mechanism": mechanism,
        "density": density,
        "falsegap": falsegap,
        "behaviour": behaviour,
        "render": render,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
