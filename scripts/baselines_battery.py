#!/usr/bin/env python3
"""The cheaper checks the paper did not compare, next to ll_bic.

Three alternatives to fitting a log-linear model (src/lacuna/estimators/baselines.py):

- F9a Chapman, a two-witness Lincoln-Petersen with its standard variance
  interval, for W1xW2, W1xW3 and W2xW3.
- F9b A sequence-number check on the gateway: the highest number seen counts
  the actions it routed. Every action in these environments routes through it,
  so it is exact whenever an episode's last gateway record survives. Also the
  variant where a share b of actions bypasses the gateway, and a tail-corrected
  version (`seq_gap_tail`).
- F9c Canaries: the auditor plants k known actions per episode with the real op
  mix, they are thinned by the same loss draw, and N = union / (1 - P(missed by
  all)) with P read off the canaries.

Run on the `confirm` pool, resampling its episodes with replacement per
replicate and thinning with the same loss configurations and seed convention as
the other robustness scripts (replay.replicates: 100 replicates per seed, one
seed list for every cell, so cells share random numbers). Oracle and signature
linkage come from the same thinned draw. Canaries are scored on their own
thinned draw of the same replicate, the real actions' ll_bic on that draw
included (`ll_bic@canary_world`) so the comparison with the canary estimate is
paired.

Gateway records carry no sequence field; the number of an action is its
position in the complete gateway log. That assumes the gateway's counter is
never lost, only the records it numbered (the generous reading).

Exploratory, after the confirmatory run. Writes artifacts/robustness/baselines.json.

Usage: uv run python scripts/baselines_battery.py [--out PATH] [--seeds N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from burst_loss import burst_thinner
from mnar_dose import op_keyed
from replay import ESTIMATORS, summarise
from session_and_phase import shared_quality
from thin_episodes import load

from lacuna.estimators import Estimate, baselines
from lacuna.histories import keep_witnesses, observed_total
from lacuna.thinning import HELDOUT, MNAR_HIDDEN, Config, tables, thin
from lacuna.witnesses import GATEWAY, Record

ROOT = Path(__file__).resolve().parents[1]
RUN = "confirm"
SEED0 = 94_000  # disjoint from every seed range in scripts/ and from 1-3, 4242
REPS_PER_SEED = 100
LINKAGES = ("oracle", "signature")
K_CANARIES = 5
FIN_LIGHT = Config("fin_light", 0.05, 0.10, "financial")

Thinner = Callable[[list[Record], np.random.Generator], list[Record]]


@lru_cache(None)
def episodes() -> list:
    return load(RUN)


def all_ops() -> tuple[str, ...]:
    return tuple(sorted({e.op for led, _ in episodes() for e in led}))


@dataclass(frozen=True)
class Cell:
    name: str
    note: str
    thinner: Callable[[], Thinner]
    bypass: float = 0.0
    k: int = K_CANARIES
    # name -> kwargs for plant_canaries; empty (or k == 0) skips the canary arm.
    canary_variants: dict = field(default_factory=lambda: {"episode_mix": {}})


def _cfg(cfg: Config) -> Callable[[], Thinner]:
    return lambda: lambda recs, rng: thin(recs, cfg, rng)


def cells() -> list[Cell]:
    uniform = {"ops": all_ops()}
    out = [
        Cell("per_record", "fin_mid: 0.5/0.5, financial view, independent drops", _cfg(HELDOUT[0])),
        Cell(
            "per_record_light",
            "fin_light: gateway 0.05, trace 0.10, financial view",
            _cfg(FIN_LIGHT),
        ),
        Cell(
            "coupled_rho100",
            "fin_mid_rho100: gateway and trace drops perfectly coupled",
            _cfg(HELDOUT[1]),
        ),
        Cell(
            "mnar_op_x1.2",
            "op-keyed MNAR: update_customer, write_file dropped at 1.2x (mnar_dose.op_keyed)",
            _cfg(op_keyed(1.2)),
            canary_variants={"episode_mix": {}, "uniform_mix": uniform},
        ),
        Cell(
            "mnar_op_x1.8",
            "op-keyed MNAR at 1.8x, the paper's MNAR point",
            _cfg(op_keyed(1.8)),
            canary_variants={"episode_mix": {}, "uniform_mix": uniform},
        ),
        Cell(
            "hidden_key_x1.8",
            "id-keyed MNAR (every 3rd action lost 1.8x, W3 blind), full view; canaries in or "
            "outside the key",
            _cfg(MNAR_HIDDEN[0]),
            canary_variants={"inside_key": {}, "outside_key": {"interleave": False}},
        ),
        Cell(
            "burst_w12",
            "blocks of 12 consecutive actions dropped per log w.p. 0.5 (burst_loss.burst_thinner)",
            lambda: burst_thinner(12, False, "financial"),
        ),
        Cell(
            "shared_quality_fin",
            "one session quality q~U(0.1,0.9) drops every log at rate q, financial view",
            lambda: shared_quality("financial"),
        ),
        Cell(
            "shared_quality_full",
            "as above with W3's full view (session_and_phase / final_controls convention)",
            lambda: shared_quality("full"),
        ),
    ]
    for k in (1, 2, 10, 20):
        out.append(
            Cell(
                f"per_record_k{k}",
                f"per_record with {k} canaries per episode",
                _cfg(HELDOUT[0]),
                k=k,
            )
        )
    for b in (0.0, 0.05, 0.2):
        for cfg in (HELDOUT[0], FIN_LIGHT):
            out.append(
                Cell(
                    f"bypass_b{b:g}/{cfg.name}",
                    f"per-record loss {cfg.name}; each action bypasses the gateway w.p. {b:g}",
                    _cfg(cfg),
                    bypass=b,
                    canary_variants={},
                )
            )
    return out


def one_rep(cell: Cell, rng: np.random.Generator) -> dict[str, list[dict]]:
    eps = episodes()
    sample = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
    n = sum(len(led) for led, _ in sample)
    thinner = cell.thinner()
    parts = {lk: [] for lk in LINKAGES}
    per_op = {lk: {} for lk in LINKAGES}
    obs = []
    canary_arm = cell.k > 0 and bool(cell.canary_variants)
    cparts = {(v, lk): [] for v in cell.canary_variants for lk in LINKAGES}
    chist = {v: [] for v in cell.canary_variants}
    for led, recs in sample:
        bypass = frozenset(e.id for e in led if rng.random() < cell.bypass) if cell.bypass else ()
        seq = baselines.gateway_seq(recs, bypass)
        routed = [r for r in recs if not (r.witness == GATEWAY and r.oracle in bypass)]
        kept = thinner(routed, rng)
        obs.append(baselines.seq_obs(kept, seq))
        for lk in LINKAGES:
            t4, po = tables(led, kept, lk)
            parts[lk].append(t4)
            for op, t in po.items():
                per_op[lk][op] = per_op[lk].get(op, 0) + t
        if canary_arm:
            for v, kw in cell.canary_variants.items():
                led2, recs2, cids = baselines.plant_canaries(led, recs, cell.k, rng, **kw)
                kept2 = thinner(recs2, rng)
                real_led = [e for e in led2 if e.id not in cids]
                real = baselines.strip_canaries(kept2, cids)
                chist[v] += baselines.canary_histories(kept2, cids)
                for lk in LINKAGES:
                    cparts[v, lk].append(tables(real_led, real, lk)[0])

    rows: dict[str, list[dict]] = {}
    for lk in LINKAGES:
        t3 = keep_witnesses(np.sum(parts[lk], axis=0), (0, 1, 2))
        p3 = {op: keep_witnesses(t, (0, 1, 2)) for op, t in per_op[lk].items()}
        ts = [keep_witnesses(t, (0, 1, 2)) for t in parts[lk]]
        union = observed_total(t3)
        row: dict = {"n": n, "union": union}
        row["trust_trace"] = baselines.trust_trace(t3)
        row["union_count"] = baselines.union_count(t3)
        row["union_bound"] = baselines.union_bound(t3)
        for name in ("ll_independence", "ll_bic"):
            row[name] = ESTIMATORS[name](t3, p3, ts)
        for f in (baselines.chapman_w1w2, baselines.chapman_w1w3, baselines.chapman_w2w3):
            e = f(t3)
            row[e.name] = e
        for f in (baselines.seq_gap, baselines.seq_gap_lb, baselines.seq_gap_tail):
            e = f(obs, union)
            row[e.name] = e
        rows[f"{lk}/plain"] = [row]
        if canary_arm:
            for v in cell.canary_variants:
                c3 = keep_witnesses(np.sum(cparts[v, lk], axis=0), (0, 1, 2))
                cp3 = {}
                # per-op tables are not needed by ll_bic; pass an empty dict.
                crow: dict = {"n": n, "union": observed_total(c3)}
                crow["ll_bic@canary_world"] = ESTIMATORS["ll_bic"](c3, cp3)
                for joint in (True, False):
                    e = baselines.canary_estimate(crow["union"], chist[v], joint=joint)
                    crow[f"{e.name}@{v}"] = e
                rows[f"{lk}/canary_{v}"] = [crow]
    return rows


def run_cell(args: tuple[str, int]) -> dict:
    name, seeds = args
    cell = next(c for c in cells() if c.name == name)
    t0 = time.perf_counter()
    pooled: dict[str, list[dict]] = {}
    for s in range(SEED0, SEED0 + seeds):
        rng = np.random.default_rng(s)
        for _ in range(REPS_PER_SEED):
            for key, rs in one_rep(cell, rng).items():
                pooled.setdefault(key, []).extend(rs)
    out: dict = {
        "note": cell.note,
        "bypass": cell.bypass,
        "k_canaries": cell.k if cell.canary_variants else 0,
        "reps": seeds * REPS_PER_SEED,
        "seconds": round(time.perf_counter() - t0, 1),
        "linkage": {},
    }
    for lk in LINKAGES:
        est: dict = {}
        for key, rows in pooled.items():
            if not key.startswith(lk + "/"):
                continue
            for name_ in rows[0]:
                if isinstance(rows[0][name_], Estimate):
                    est[name_] = summarise(rows, name_)
        plain = pooled[f"{lk}/plain"]
        out["linkage"][lk] = {
            "mean_n_true": round(float(np.mean([r["n"] for r in plain])), 1),
            "mean_union": round(float(np.mean([r["union"] for r in plain])), 1),
            "estimators": est,
        }
    return {"name": name, **out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "robustness" / "baselines.json"))
    ap.add_argument("--seeds", type=int, default=5, help="seeds of 100 replicates per cell")
    ap.add_argument("--only", default="", help="comma-separated cell names (smoke runs)")
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    names = [c.name for c in cells() if not args.only or c.name in args.only.split(",")]
    with Pool(args.processes) as pool:
        results = pool.map(run_cell, [(n, args.seeds) for n in names], chunksize=1)
    for r in results:
        o = r["linkage"]["oracle"]["estimators"]
        print(
            r["name"],
            f"{r['seconds']}s",
            *(
                f"{k}={o[k]['coverage']}/w{o[k]['mean_width']}/b{o[k]['bias']}"
                for k in o
                if k.startswith(("ll_bic", "chapman_w1w2", "seq_gap", "canary_joint"))
            ),
            flush=True,
        )
    out = {
        "run": RUN,
        "reps_per_seed": REPS_PER_SEED,
        "seeds": [SEED0, SEED0 + args.seeds - 1],
        "episodes_resampled_with_replacement": True,
        "k_canaries_default": K_CANARIES,
        "assumptions": {
            "seq": "gateway sequence number = position in the complete gateway log; the counter "
            "itself is never lost, only the records it numbered",
            "canary": "canaries are exchangeable with real actions: same op mix, thinned by the "
            "same loss process; every canary has a record for W1, W2 and W3 before loss "
            "(W3 view still applies)",
            "bypass": "an action that bypasses the gateway has no gateway record and no number",
        },
        "cells": {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in results},
    }
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
