#!/usr/bin/env python3
"""Tables for docs/PREREG-TRANSFER.md and docs/TRANSFER.md, from artifacts only.

  uv run python scripts/transfer_table.py sanity transfer-scripted
  uv run python scripts/transfer_table.py sanity confirm financial   # back-office, for contrast
  uv run python scripts/transfer_table.py results transfer
  uv run python scripts/transfer_table.py predictions transfer
  uv run python scripts/transfer_table.py plan transfer
  uv run python scripts/transfer_table.py falsepos transfer-falsepos

`falsepos <name>` renders a `falsepos` pool run (docs/DECISIONS.md D41, D42):
the true hidden count is 0 under `anti`, so detects_gap is a false-positive
rate there.

`plan <run>` compares, per op, the actions the tickets' scripted plans call
for (regenerated from each episode's seed) with the actions the agent took.

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

`predictions <run>` scores the predictions frozen in docs/PREREG-TRANSFER.md
against artifacts/episodes/<run> and artifacts/thinning/<run>.json. The
predicted ranges are literals copied from the pre-registration.
Numbers are never hand-typed (AGENTS.md rule 2).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from lacuna.envs.infra import ACTION_OPS
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
        f"| config | P(both logs drop) | expected hidden from all 3, per episode "
        f"| at {len(rows)} episodes |"
    )
    print("|---|---|---|---|")
    for cfg in POOLS["transfer"]:
        p = p_hidden(cfg)
        print(f"| `{cfg.name}` | {p:.3f} | {p * per_ep:.2f} | {p * inv:.1f} |")


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


def _totals(run: str) -> dict:
    """Actions, W3-invisible actions, and those of them sharing a signature."""
    n = inv = shared_inv = eps = 0
    for line in (ROOT / "artifacts" / "episodes" / run / "episodes.jsonl").open():
        e = json.loads(line)
        seen = {r["oracle"] for r in e["records"] if r["witness"] == ENV}
        sig = Counter(signature(x["op"], x["resource"], x["value"]) for x in e["ledger"])
        for x in e["ledger"]:
            n += 1
            if x["id"] not in seen:
                inv += 1
                shared_inv += sig[signature(x["op"], x["resource"], x["value"])] > 1
        eps += 1
    return {"n": n, "inv": inv, "shared_inv": shared_inv, "episodes": eps}


def predictions(run: str) -> None:
    t = _totals(run)
    r = json.loads((ROOT / "artifacts" / "thinning" / f"{run}.json").read_text())

    def est(cfg: str, linkage: str, name: str) -> dict:
        return r["configs"][f"{cfg}/{linkage}"]["estimators"][name]

    def hidden(cfg: str) -> float:
        return r["configs"][f"{cfg}/oracle"]["mean_unseen_by_w1w2w3"]

    mid, rho = "transfer_mid", "transfer_mid_rho100"
    bic = est(mid, "signature", "ll_bic")
    bic_rho = est(rho, "signature", "ll_bic")
    ind, ind_rho = (
        est(mid, "signature", "ll_independence"),
        est(rho, "signature", "ll_independence"),
    )
    union = est(mid, "signature", "union_count")
    # P(both logs drop) is 0.25 at transfer_mid: the MCAR expectation of the
    # hidden count given this run's W3-invisible actions.
    expected_mid = 0.25 * t["inv"]
    rows = [
        (
            "1",
            "actions per episode",
            "38 to 52",
            t["n"] / t["episodes"],
            38 <= t["n"] / t["episodes"] <= 52,
        ),
        (
            "1",
            "W3-invisible share",
            "0.40 to 0.55",
            t["inv"] / t["n"],
            0.40 <= t["inv"] / t["n"] <= 0.55,
        ),
        (
            "2",
            "hidden from all 3 at mid, oracle",
            "95 to 160",
            hidden(mid),
            95 <= hidden(mid) <= 160,
        ),
        (
            "2",
            "  ... over 0.25 x W3-invisible",
            "close to 1 (read: 0.9 to 1.1)",
            hidden(mid) / expected_mid,
            0.9 <= hidden(mid) / expected_mid <= 1.1,
        ),
        (
            "2",
            "hidden at rho100 over hidden at mid",
            "about 2 (read: 1.8 to 2.2)",
            hidden(rho) / hidden(mid),
            1.8 <= hidden(rho) / hidden(mid) <= 2.2,
        ),
        (
            "3",
            "ll_bic coverage at mid, signature",
            "0.20 to 0.75",
            bic["coverage"],
            0.20 <= bic["coverage"] <= 0.75,
        ),
        ("3", "ll_bic bias at mid, signature", "negative", bic["bias"], bic["bias"] < 0),
        (
            "3",
            "ll_bic coverage at mid, oracle",
            ">= 0.90",
            est(mid, "oracle", "ll_bic")["coverage"],
            est(mid, "oracle", "ll_bic")["coverage"] >= 0.90,
        ),
        (
            "4",
            "ll_bic detects_gap at mid",
            ">= 0.95",
            bic["detects_gap"],
            bic["detects_gap"] >= 0.95,
        ),
        (
            "5",
            "ll_independence coverage at rho100",
            "about 0 (read: <= 0.05)",
            ind_rho["coverage"],
            ind_rho["coverage"] <= 0.05,
        ),
        (
            "5",
            "ll_bic coverage fall, mid to rho100",
            "negative",
            bic["coverage"] - bic_rho["coverage"],
            bic["coverage"] - bic_rho["coverage"] < 0,
        ),
        (
            "5",
            "ll_independence coverage fall",
            ">= 0.10 (T3)",
            ind["coverage"] - ind_rho["coverage"],
            ind["coverage"] - ind_rho["coverage"] >= 0.10,
        ),
        (
            "6",
            "detects_gap lead over union_count",
            ">= 0.95",
            bic["detects_gap"] - union["detects_gap"],
            bic["detects_gap"] - union["detects_gap"] >= 0.95,
        ),
        (
            "7",
            "W3-invisible sharing a signature",
            "0.15 to 0.40",
            t["shared_inv"] / t["inv"],
            0.15 <= t["shared_inv"] / t["inv"] <= 0.40,
        ),
        ("8", "union_count coverage at mid", "0.000", union["coverage"], union["coverage"] == 0),
        (
            "8",
            "union_count coverage at rho100",
            "0.000",
            est(rho, "signature", "union_count")["coverage"],
            est(rho, "signature", "union_count")["coverage"] == 0,
        ),
    ]
    # Rows that could not have failed: MCAR loss makes the hidden count 0.25
    # (or 0.5 when coupled) times the W3-invisible count in expectation, and
    # union_count covers 0 whenever anything is hidden. Shown, not counted.
    not_predictions = {
        "  ... over 0.25 x W3-invisible": "identity under MCAR loss",
        "hidden at rho100 over hidden at mid": "identity under MCAR loss",
        "union_count coverage at mid": "true by construction",
        "union_count coverage at rho100": "true by construction",
    }
    print(
        f"Source: `artifacts/episodes/{run}/episodes.jsonl` ({t['episodes']} episodes, "
        f"N={t['n']}) and `artifacts/thinning/{run}.json`. Ranges from docs/PREREG-TRANSFER.md. "
        "A reading in parentheses turns a verbal prediction into a band; the bands were "
        "added after the freeze, in commit 0748430.\n"
    )
    print("| # | quantity | predicted | outcome | held |")
    print("|---|---|---|---|---|")
    counted = held = 0
    for k, q, pred, out, ok in rows:
        if q in not_predictions:
            verdict = f"not a test: {not_predictions[q]}"
        else:
            counted, held = counted + 1, held + bool(ok)
            verdict = "yes" if ok else "**no**"
        print(f"| {k} | {q} | {pred} | {out:.3f} | {verdict} |")
    print(f"\n{held} of {counted} rows that could have failed held.")


def plan(run: str) -> None:
    from lacuna.envs.infra_tasks import make_episode

    planned, done = Counter(), Counter()
    tickets = None
    for line in (ROOT / "artifacts" / "episodes" / run / "episodes.jsonl").open():
        e = json.loads(line)
        tickets = (
            tickets
            or json.loads((ROOT / "artifacts" / "episodes" / run / "summary.json").read_text())[
                "tickets"
            ]
        )
        ep = make_episode(e["seed"], tickets)
        actions = set(ACTION_OPS)
        planned.update(tool for t in ep.tickets for tool, _ in t.plan if tool in actions)
        done.update(x["op"] for x in e["ledger"])
    print(f"Source: `artifacts/episodes/{run}/episodes.jsonl`; plans from `make_episode(seed)`.\n")
    print("| op | in the scripted plans | taken by the agent | ratio |")
    print("|---|---|---|---|")
    for op in sorted(set(planned) | set(done)):
        ratio = f"{done[op] / planned[op]:.3f}" if planned[op] else "—"
        print(f"| `{op}` | {planned[op]} | {done[op]} | {ratio} |")
    tp, td = sum(planned.values()), sum(done.values())
    print(f"| **all** | **{tp}** | **{td}** | **{td / tp:.3f}** |")


def falsepos(name: str) -> None:
    r = json.loads((ROOT / "artifacts" / "thinning" / f"{name}.json").read_text())
    if r.get("pool") != "falsepos":
        raise SystemExit(f"{name} was not run on the falsepos pool")
    print(
        f"Source: `artifacts/thinning/{name}.json` — run `{r['run']}`, {r['episodes']} episodes, "
        f"N={r['n_true']}, {r['reps']} replicates, episodes resampled, seed {r['seed']}.\n"
    )
    print("| config | linkage | hidden from all 3 | estimator | detects gap | coverage | bias |")
    print("|---|---|---|---|---|---|---|")
    for key, d in r["configs"].items():
        cfg, linkage = key.split("/")
        for i, k in enumerate(("ll_independence", "ll_bic", "strat_bic", "lc_gated")):
            m = d["estimators"].get(k)
            if m is None:
                continue
            head = f"`{cfg}` | {linkage} | {d['mean_unseen_by_w1w2w3']}" if i == 0 else " | |"
            print(f"| {head} | `{k}` | {m['detects_gap']} | {m['coverage']:.3f} | {m['bias']} |")


def main() -> None:
    cmds = {
        "sanity": sanity,
        "results": results,
        "predictions": predictions,
        "plan": plan,
        "falsepos": falsepos,
    }
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]](*sys.argv[2:])


if __name__ == "__main__":
    main()
