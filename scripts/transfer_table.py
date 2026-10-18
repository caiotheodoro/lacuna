#!/usr/bin/env python3
"""Tables for docs/PREREG-TRANSFER.md and docs/TRANSFER.md, from artifacts only.

  uv run python scripts/transfer_table.py sanity transfer-scripted
  uv run python scripts/transfer_table.py sanity confirm financial   # back-office, for contrast
  uv run python scripts/transfer_table.py results transfer

`sanity <run>` reads artifacts/episodes/<run>/episodes.jsonl and counts, per
episode and per op, the ledger actions the end-state diff and sink (W3) did
not see (oracle attribution). With the env view `full`, W3 sees every other
action, so an action is hidden from all three witnesses only if it is
W3-invisible and both logs lost it. The expected hidden count per
configuration follows from the pool's drop rates: p_gateway * p_trace for
independent loss, min(p_gateway, p_trace) for fully coupled loss.

It also counts actions whose effect signature (lacuna.linkage.signature) is
shared with another action of the same episode. Signature linkage pairs such
records in order, which overstates the overlap between witnesses once loss
is applied; where W3 also missed them, nothing corrects it.

`results <name>` renders artifacts/thinning/<name>.json for the transfer pool.
Numbers are never hand-typed (AGENTS.md rule 2).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from lacuna.linkage import signature
from lacuna.thinning import ENV_VIEWS, POOLS

ROOT = Path(__file__).resolve().parents[1]
ENV = 2
HEADLINE = ["trust_trace", "union_count", "union_bound", "ll_independence", "ll_bic"]
ALSO = ["ll_average", "anchor_g1", "anchor_g4", "lc_gated", "strat_lc_gated"]


def p_hidden(cfg) -> float:
    """P(both logs drop an action) under the pool's loss model."""
    pg, pt = cfg.p_gateway_drop, cfg.p_trace_drop
    return cfg.rho * min(pg, pt) + (1 - cfg.rho) * pg * pt


def sanity(run: str, view: str = "full") -> None:
    """`view` restricts W3 as `lacuna.thinning.ENV_VIEWS` does (back-office: `financial`)."""
    ops_seen = ENV_VIEWS[view]
    path = ROOT / "artifacts" / "episodes" / run / "episodes.jsonl"
    rows, ops_all, ops_inv = [], Counter(), Counter()
    ops_shared, ops_shared_inv = Counter(), Counter()
    for line in path.open():
        e = json.loads(line)
        seen = {
            r["oracle"]
            for r in e["records"]
            if r["witness"] == ENV and (ops_seen is None or r["op"] in ops_seen)
        }
        inv = [x for x in e["ledger"] if x["id"] not in seen]
        ops_all.update(x["op"] for x in e["ledger"])
        ops_inv.update(x["op"] for x in inv)
        sig = Counter(signature(x["op"], x["resource"], x["value"]) for x in e["ledger"])
        shared = [x for x in e["ledger"] if sig[signature(x["op"], x["resource"], x["value"])] > 1]
        ops_shared.update(x["op"] for x in shared)
        ops_shared_inv.update(x["op"] for x in shared if x["id"] not in seen)
        rows.append((e["seed"], len(e["ledger"]), len(inv), e["n_calls"], e["n_tool_errors"]))
    n, inv = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(
        f"Source: `artifacts/episodes/{run}/episodes.jsonl` — {len(rows)} episodes, "
        f"W3 env view `{view}`.\n"
    )
    print("| seed | actions N | W3-invisible | share | tool calls | tool errors |")
    print("|---|---|---|---|---|---|")
    for seed, ni, ii, calls, errs in rows:
        print(f"| {seed} | {ni} | {ii} | {ii / ni:.3f} | {calls} | {errs} |")
    print(f"| **all** | **{n}** | **{inv}** | **{inv / n:.3f}** | | |\n")
    print("| op | actions | W3-invisible | share invisible | shared sig | shared, invisible |")
    print("|---|---|---|---|---|---|")
    for op, c in sorted(ops_all.items()):
        print(
            f"| `{op}` | {c} | {ops_inv[op]} | {ops_inv[op] / c:.3f} "
            f"| {ops_shared[op]} | {ops_shared_inv[op]} |"
        )
    sh, shi = sum(ops_shared.values()), sum(ops_shared_inv.values())
    print(f"| **all** | **{n}** | **{inv}** | **{inv / n:.3f}** | **{sh}** | **{shi}** |")
    print()
    if view != "full":
        return
    per_ep = inv / len(rows)
    print(
        "| config | P(both logs drop) | expected hidden from all 3, per episode | at 24 episodes |"
    )
    print("|---|---|---|---|")
    for cfg in POOLS["transfer"]:
        p = p_hidden(cfg)
        print(f"| `{cfg.name}` | {p:.3f} | {p * per_ep:.2f} | {24 * p * per_ep:.1f} |")


def results(name: str) -> None:
    r = json.loads((ROOT / "artifacts" / "thinning" / f"{name}.json").read_text())
    if r.get("pool") != "transfer":
        raise SystemExit(f"{name} was not run on the transfer pool")
    print(
        f"Source: `artifacts/thinning/{name}.json` — run `{r['run']}`, {r['episodes']} episodes, "
        f"N={r['n_true']}, {r['reps']} loss replicates per configuration, episodes resampled "
        f"with replacement per replicate, seed {r['seed']}, nominal 95%.\n"
    )
    for linkage in ("signature", "oracle"):
        print(f"**{linkage} linkage**\n")
        print(
            "| config | hidden from all 3 | estimator | coverage ± MCSE | detects gap "
            "| unbounded | mean width | bias | RMSE |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for cfg in POOLS["transfer"]:
            d = r["configs"][f"{cfg.name}/{linkage}"]
            names = HEADLINE + (ALSO if linkage == "signature" else [])
            for i, k in enumerate(names):
                m = d["estimators"][k]
                w = "—" if m["mean_width"] is None else m["mean_width"]
                head = f"`{cfg.name}` | {d['mean_unseen_by_w1w2w3']}" if i == 0 else " | "
                print(
                    f"| {head} | `{k}` | {m['coverage']:.3f} ± {m['coverage_mcse']:.3f} "
                    f"| {m['detects_gap']} | {m['unbounded_rate']:.2f} | {w} "
                    f"| {m['bias']} | {m['rmse']} |"
                )
        print()


def main() -> None:
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("sanity", "results"):
        raise SystemExit(__doc__)
    {"sanity": sanity, "results": results}[sys.argv[1]](*sys.argv[2:])


if __name__ == "__main__":
    main()
