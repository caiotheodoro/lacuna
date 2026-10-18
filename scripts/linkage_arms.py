#!/usr/bin/env python3
"""Which record linkage an auditor should use: five arms against exact matching (D49-D51).

The pre-registered linkage is the effect signature (op, resource, and the value
for a fixed set of ops). D44-D48 showed it fails. This compares it with arms an
auditor can also compute from the logs:

- `signature`: as pre-registered (`lacuna.linkage.signature`).
- `digest`: the signature with the value kept for every op. Email and file
  bodies are already stored as content digests in every witness record; the
  pre-registered signature drops them, a choice made for the LLM auditor
  (W4), which was dropped in D20.
- `callid`: trace and gateway lines share the provider's tool-call id, so they
  are linked exactly by it; the env diff (W3) carries no call id and is linked
  to the result by signature, in order, preferring actions seen by both logs.
- `callid_digest`: the same, with W3 linked by the digest signature.
- `oracle`: the ledger id, which no auditor has.

The call id is the provider's, written into both logs by the harness. An agent
that writes its own trace could forge it; that is not tested here.

Usage: uv run python scripts/linkage_arms.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from replay import ESTIMATORS, summarise
from thin_episodes import load

from lacuna.histories import keep_witnesses
from lacuna.linkage import signature
from lacuna.thinning import Config, tables, thin
from lacuna.witnesses import ENV, GATEWAY, TRACE, Record

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("signature", "digest", "callid", "callid_digest", "callid_lastwriter", "oracle")
OVERWRITE = frozenset(
    {"update_customer", "write_file", "scale_service", "update_config", "write_runbook"}
)
EST = ("ll_independence", "ll_bic")


def digest_signature(r: Record) -> tuple[str, str, str]:
    return r.op, r.resource, r.value


def sig_table(records: list[Record], key) -> np.ndarray:
    """In-order pairing within a key, as `linkage.signature_table` does."""
    per: dict = defaultdict(lambda: [0, 0, 0])
    for r in records:
        if r.witness < 3:
            per[key(r)][r.witness] += 1
    t = np.zeros(8, dtype=np.int64)
    for counts in per.values():
        for j in range(max(counts)):
            t[sum(1 << w for w in range(3) if j < counts[w])] += 1
    return t


def callid_table(records: list[Record], key) -> np.ndarray:
    """Trace and gateway linked by call id; W3 attached by `key`, in order.

    `Record.oracle` on a trace or gateway record is the ledger id the harness
    looked up from that line's call id (`witnesses.py`), so grouping by it is
    grouping by call id. W3 records are never grouped by it.
    """
    calls: dict = defaultdict(int)
    key_of: dict = {}
    for r in records:
        if r.witness in (TRACE, GATEWAY):
            calls[r.oracle] |= 1 << r.witness
            key_of[r.oracle] = key(r)
    by_key: dict = defaultdict(list)
    for cid, bits in calls.items():
        by_key[key_of[cid]].append(bits)
    w3: dict = defaultdict(int)
    for r in records:
        if r.witness == ENV:
            w3[key(r)] += 1
    t = np.zeros(8, dtype=np.int64)
    for k in set(by_key) | set(w3):
        seen = sorted(by_key.get(k, []), key=lambda b: -bin(b).count("1"))
        c3 = w3.get(k, 0)
        for j, bits in enumerate(seen):
            t[bits | (4 if j < c3 else 0)] += 1
        t[4] += max(0, c3 - len(seen))
    return t


def merged_order(records: list[Record]) -> dict:
    """Position of each call in the merge of the gateway and trace line orders.

    Each log lists its lines in the order it wrote them; a call both logs kept
    appears in both and ties the two sequences together. Calls only one log
    kept go in at their place in that log, gateway first on a tie. The call id
    is used only to tell calls apart; `Record.oracle` stands in for it here,
    and its numeric value (the ledger order) is never used (D51).
    """
    gw = [r.oracle for r in records if r.witness == GATEWAY]
    tr = [r.oracle for r in records if r.witness == TRACE]
    shared = set(gw) & set(tr)
    order: dict = {}
    i = j = 0
    while i < len(gw) or j < len(tr):
        if i < len(gw) and gw[i] in order:
            i += 1
        elif j < len(tr) and tr[j] in order:
            j += 1
        elif i < len(gw) and gw[i] not in shared:
            order[gw[i]] = len(order)
            i += 1
        elif j < len(tr) and tr[j] not in shared:
            order[tr[j]] = len(order)
            j += 1
        else:
            c = gw[i] if i < len(gw) else tr[j]
            order[c] = len(order)
    return order


def lastwriter_table(records: list[Record]) -> np.ndarray:
    """Call-id linking for W1/W2; W3 attached by the last-writer rule (see module doc)."""
    pos = merged_order(records)
    calls: dict = defaultdict(int)
    info: dict = {}
    for r in records:
        if r.witness in (TRACE, GATEWAY):
            calls[r.oracle] |= 1 << r.witness
            info[r.oracle] = (r.op, r.resource, r.value)
    by_key: dict = defaultdict(list)
    for cid, bits in calls.items():
        op, res, val = info[cid]
        by_key[(op, res)].append((cid, val, bits))
    w3: dict = defaultdict(list)
    for r in records:
        if r.witness == ENV:
            w3[(r.op, r.resource)].append(r.value)
    t = np.zeros(8, dtype=np.int64)
    for key in set(by_key) | set(w3):
        logged = sorted(by_key.get(key, []), key=lambda c: pos[c[0]])  # merged log order
        vals = w3.get(key, [])
        if key[0] in OVERWRITE and len(vals) <= 1:
            attached = None
            if vals:
                if logged and logged[-1][1] == vals[0]:
                    attached = len(logged) - 1
                else:
                    t[4] += 1
            for j, (_, _, bits) in enumerate(logged):
                t[bits | (4 if j == attached else 0)] += 1
        else:
            need = Counter(vals)
            for _, v, bits in reversed(logged):
                if need[v] > 0:
                    need[v] -= 1
                    t[bits | 4] += 1
                else:
                    t[bits] += 1
            t[4] += sum(need.values())
    return t


def table3(led, kept: list[Record], arm: str) -> np.ndarray:
    if arm in ("signature", "oracle"):
        t4, _ = tables(led, kept, arm)
        t = keep_witnesses(t4, (0, 1, 2))
    elif arm == "digest":
        t = sig_table(kept, digest_signature)
    elif arm == "callid":
        t = callid_table(kept, lambda r: signature(r.op, r.resource, r.value))
    elif arm == "callid_lastwriter":
        t = lastwriter_table(kept)
    else:
        t = callid_table(kept, digest_signature)
    t = np.array(t, dtype=np.int64)
    t[0] = 0
    return t


@lru_cache(None)
def episodes(run: str) -> list:
    return load(run)


def cell(args: tuple) -> dict:
    run, cfg, arm, seeds, reps = args
    eps = episodes(run)
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for _ in range(reps):
            idx = rng.integers(0, len(eps), len(eps))
            sample = [eps[i] for i in idx]
            n = sum(len(led) for led, _ in sample)
            t3 = np.sum([table3(led, thin(recs, cfg, rng), arm) for led, recs in sample], axis=0)
            row = {"n": n, "union": int(t3[1:].sum())}
            for e in EST:
                row[e] = ESTIMATORS[e](t3, {})
            rows.append(row)
    return {"run": run, "config": cfg.name, "arm": arm, **{e: summarise(rows, e) for e in EST}}


def unthinned_union(run: str, arm: str) -> int:
    """Distinct actions the three complete logs show under this arm: N when linkage is exact."""
    return sum(int(table3(led, recs, arm)[1:].sum()) for led, recs in episodes(run))


def main() -> None:
    from lacuna.thinning import HELDOUT, POOLS

    precise = range(30_000, 30_020)
    quick = range(31_000, 31_010)
    transfer = {c.name: c for c in POOLS["transfer"]}
    jobs = []
    for run in ("confirm", "confirm-plus-supp"):
        for cfg in HELDOUT:
            jobs += [(run, cfg, a, precise, 100) for a in ARMS]
    for p in (0.2, 0.3, 0.4, 0.5):
        for view in ("full", "financial"):
            cfg = Config(f"anti{p:g}_{view}", p, p, view, anti=True)
            jobs += [("confirm", cfg, a, quick, 100) for a in ARMS]
    # The registered held-out cells use the financial view, which hides the
    # overwrite ops from W3; these show W3 attachment where W3 sees them.
    for name, pg, pt in (("full_mid", 0.5, 0.5), ("full_sampled", 0.9, 0.3)):
        cfg = Config(name, pg, pt, "full")
        for run in ("confirm", "long-30"):
            jobs += [(run, cfg, a, precise, 100) for a in ARMS]
    for name in ("transfer_mid", "transfer_mid_rho100"):
        jobs += [("transfer", transfer[name], a, precise, 100) for a in ARMS]
    for p in (0.2, 0.3, 0.4, 0.5):
        cfg = Config(f"anti{p:g}_full", p, p, "full", anti=True)
        jobs += [("transfer", cfg, a, quick, 100) for a in ARMS]
    with Pool(12) as pool:
        cells = pool.map(cell, jobs, chunksize=1)
    out: dict = {
        "arms": list(ARMS),
        "seeds": {
            "precise": [precise.start, precise.stop - 1],
            "quick": [quick.start, quick.stop - 1],
        },
        "reps_per_seed": 100,
        "unthinned_union": {
            run: {
                "n": sum(len(led) for led, _ in episodes(run)),
                **{a: unthinned_union(run, a) for a in ARMS},
            }
            for run in ("confirm", "confirm-plus-supp", "transfer")
        },
        "cells": {f"{c['run']}/{c['config']}/{c['arm']}": c for c in cells},
    }
    for c in cells:
        b = c["ll_bic"]
        print(
            c["run"], c["config"], c["arm"], b["coverage"], b["detects_gap"], b["bias"], flush=True
        )
    assert all(not math.isnan(c["ll_bic"]["coverage"]) for c in cells)
    dst = ROOT / "artifacts" / "robustness" / "linkage-arms.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
