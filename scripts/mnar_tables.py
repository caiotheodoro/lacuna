#!/usr/bin/env python3
"""Tables and prediction checks for docs/MNAR-ROBUSTNESS.md, from the artifacts.

Every number in that document comes from this script (AGENTS.md rule 2).
The predictions below were committed, in the doc and here, before any of the
four runs existed.

Usage:
  python3 scripts/mnar_tables.py tables [signature|oracle]
  python3 scripts/mnar_tables.py checks [signature|oracle]
  python3 scripts/mnar_tables.py repro
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIN = ROOT / "artifacts" / "thinning"

# New artifact, and the artifact the same run wrote before strat_bic existed.
RUNS = [
    ("confirm-mnar-strat", "confirm-mnar"),
    ("confirm-mnar-hidden", None),
    ("confirm-heldout-strat", "confirm"),
    ("long-30-strat", "long-30.bootstrap"),
]
SHOW = ["union_count", "ll_bic", "lc_gated", "strat_lc_gated", "strat_bic"]
LINK = "signature"  # the headline linkage (D28)


def load(run: str) -> dict:
    return json.loads((THIN / f"{run}.json").read_text())


def fmt(v, spec: str = ".3f") -> str:
    if v is None:
        return "—"
    return format(v, spec)


def tables(linkage: str) -> None:
    for run, _ in RUNS:
        path = THIN / f"{run}.json"
        if not path.exists():
            print(f"`{run}.json`: not run yet.\n")
            continue
        r = load(run)
        print(
            f"Source: `artifacts/thinning/{run}.json` (pool `{r['pool']}`, run `{r['run']}`), "
            f"{r['episodes']} episodes, N={r['n_true']}, {r['reps']} replicates, "
            f"episodes resampled: {r['bootstrap_episodes']}, seed {r['seed']}, "
            f"{linkage} linkage.\n"
        )
        print("| config | hidden | estimator | coverage | detects gap | unbounded | width | bias |")
        print("|---|---|---|---|---|---|---|---|")
        for key, d in r["configs"].items():
            name, lk = key.split("/")
            if lk != linkage:
                continue
            first = True
            for est in SHOW:
                m = d["estimators"].get(est)
                if m is None:
                    continue
                head = f"`{name}` | {d['mean_unseen_by_w1w2w3']}" if first else " | "
                first = False
                bias = "—" if m["bias"] is None else f"{m['bias']:+.2f}"
                print(
                    f"| {head} | `{est}` | {fmt(m['coverage'])} | {fmt(m['detects_gap'], '.2f')} "
                    f"| {fmt(m['unbounded_rate'], '.2f')} | {fmt(m['mean_width'], '.1f')} "
                    f"| {bias} |"
                )
        print()


def _m(run: str, cfg: str, est: str) -> dict:
    return load(run)["configs"][f"{cfg}/{LINK}"]["estimators"][est]


def _hidden(run: str, cfg: str) -> float:
    return load(run)["configs"][f"{cfg}/{LINK}"]["mean_unseen_by_w1w2w3"]


def _cmp(v, op: str, x: float) -> bool:
    if v is None:
        return False
    return {"<=": v <= x, ">=": v >= x, "<": v < x}[op]


def _rule(run, cfg, est, metric, op, x):
    def check():
        v = _m(run, cfg, est)[metric]
        return f"{metric} {fmt(v, '.3f' if metric != 'bias' else '+.2f')}", _cmp(v, op, x)

    return check


def _all(*rules):
    def check():
        outs = [r() for r in rules]
        return "; ".join(o for o, _ in outs), all(ok for _, ok in outs)

    return check


def _between(run, cfg, est, metric, a, b):
    def check():
        v = _m(run, cfg, est)[metric]
        return f"{metric} {fmt(v, '.3f')}", v is not None and a <= v <= b

    return check


def _bias_or_unbounded(run, cfg, est, x):
    """Bias at or below x where finite; a run with no finite point passes vacuously."""

    def check():
        m = _m(run, cfg, est)
        if m["bias"] is None:
            return "no finite point", True
        return f"bias {m['bias']:+.2f}", m["bias"] <= x

    return check


def _width_ratio(run, cfg, est, ref, x):
    def check():
        a, b = _m(run, cfg, est)["mean_width"], _m(run, cfg, ref)["mean_width"]
        if a is None:
            return "never bounded", True
        return f"width {a:.1f} vs {b:.1f}", a >= x * b

    return check


def _hidden_near(run, cfg, x):
    def check():
        v = _hidden(run, cfg)
        return f"hidden {v}", abs(v - x) <= 0.1 * x

    return check


M, H, R, L = "confirm-mnar-strat", "confirm-mnar-hidden", "confirm-heldout-strat", "long-30-strat"
PREDICTIONS = [
    ("P1", "fin_mid_mnar: strat_bic covers",
        _rule(M, "fin_mid_mnar", "strat_bic", "coverage", ">=", 0.93)),
    ("P2", "fin_mid_mnar: strat_bic mostly unbounded",
        _rule(M, "fin_mid_mnar", "strat_bic", "unbounded_rate", ">=", 0.80)),
    ("P3", "fin_mid_mnar: strat_bic detects gap in [0.45, 0.85]",
        _between(M, "fin_mid_mnar", "strat_bic", "detects_gap", 0.45, 0.85)),
    ("P4", "fin_mid_mnar_rho100: strat_bic refuses", _all(
        _rule(M, "fin_mid_mnar_rho100", "strat_bic", "unbounded_rate", ">=", 0.99),
        _rule(M, "fin_mid_mnar_rho100", "strat_bic", "coverage", ">=", 0.99),
        _rule(M, "fin_mid_mnar_rho100", "strat_bic", "detects_gap", "<=", 0.05))),
    ("P5", "full_mid_hidden: strat_bic fails", _all(
        _rule(H, "full_mid_hidden", "strat_bic", "coverage", "<=", 0.50),
        _bias_or_unbounded(H, "full_mid_hidden", "strat_bic", -100))),
    ("P6", "full_mid_hidden: ll_bic undercovers, biased low", _all(
        _rule(H, "full_mid_hidden", "ll_bic", "coverage", "<=", 0.85),
        _between(H, "full_mid_hidden", "ll_bic", "bias", -160, -20))),
    ("P7", "full_mid_hidden_rho100: both confidently wrong", _all(
        _rule(H, "full_mid_hidden_rho100", "strat_bic", "coverage", "<=", 0.10),
        _rule(H, "full_mid_hidden_rho100", "strat_bic", "unbounded_rate", "<=", 0.10),
        _rule(H, "full_mid_hidden_rho100", "strat_bic", "bias", "<=", -150),
        _rule(H, "full_mid_hidden_rho100", "ll_bic", "coverage", "<=", 0.10),
        _rule(H, "full_mid_hidden_rho100", "ll_bic", "bias", "<=", -150))),
    ("P8", "hidden mass matches the design arithmetic (233.0, 262.2) within 10%", _all(
        _hidden_near(H, "full_mid_hidden", 233.0),
        _hidden_near(H, "full_mid_hidden_rho100", 262.2))),
    ("P9", "fin_mid: strat_bic covers, wider than ll_bic, sometimes unbounded", _all(
        _rule(R, "fin_mid", "strat_bic", "coverage", ">=", 0.93),
        _width_ratio(R, "fin_mid", "strat_bic", "ll_bic", 1.5),
        _between(R, "fin_mid", "strat_bic", "unbounded_rate", 0.05, 0.50))),
    ("P10", "fin_mid_rho100: strat_bic refuses, ll_bic holds", _all(
        _rule(R, "fin_mid_rho100", "strat_bic", "unbounded_rate", ">=", 0.99),
        _rule(R, "fin_mid_rho100", "strat_bic", "detects_gap", "<=", 0.05),
        _rule(R, "fin_mid_rho100", "ll_bic", "coverage", ">=", 0.90))),
    ("P11", "light, sampled, fin_light: strat_bic covers >= 0.85", _all(
        *(_rule(L, c, "strat_bic", "coverage", ">=", 0.85)
          for c in ("light", "sampled", "fin_light")))),
    ("P12", "fin_sampled: strat_bic covers, mostly unbounded", _all(
        _rule(L, "fin_sampled", "strat_bic", "coverage", ">=", 0.93),
        _rule(L, "fin_sampled", "strat_bic", "unbounded_rate", ">=", 0.50))),
    ("P13", "fin_sampled_rho100: strat_bic undercovers low, ll_bic holds", _all(
        _rule(L, "fin_sampled_rho100", "strat_bic", "coverage", "<=", 0.80),
        _bias_or_unbounded(L, "fin_sampled_rho100", "strat_bic", 0.0),
        _rule(L, "fin_sampled_rho100", "ll_bic", "coverage", ">=", 0.90))),
    ("P14", "fin_sampled_rho50: strat_bic biased low where finite",
        _bias_or_unbounded(L, "fin_sampled_rho50", "strat_bic", 0.0)),
]  # fmt: skip


def checks() -> None:
    print(f"{LINK.capitalize()} linkage. Held/failed is computed here, not judged by hand.\n")
    print("| id | prediction | observed | outcome |")
    print("|---|---|---|---|")
    for pid, text, check in PREDICTIONS:
        try:
            obs, ok = check()
        except FileNotFoundError:
            obs, ok = "not run", None
        verdict = "—" if ok is None else ("held" if ok else "**failed**")
        print(f"| {pid} | {text} | {obs} | {verdict} |")


def repro() -> None:
    """P0: estimators that existed before strat_bic reproduce exactly on the same seed."""
    print("| new artifact | reference | estimators compared | max abs difference |")
    print("|---|---|---|---|")
    for run, ref in RUNS:
        if ref is None or not (THIN / f"{run}.json").exists():
            continue
        new, old = load(run), load(ref)
        worst, names = 0.0, set()
        for key, d in old["configs"].items():
            if key not in new["configs"]:
                continue
            a, b = d["mean_unseen_by_w1w2w3"], new["configs"][key]["mean_unseen_by_w1w2w3"]
            worst = max(worst, abs(a - b))
            for est, m in d["estimators"].items():
                names.add(est)
                n = new["configs"][key]["estimators"][est]
                for field, v in m.items():
                    w = n.get(field)
                    if v is None or w is None:
                        worst = max(worst, 0.0 if v == w else math.inf)
                    else:
                        worst = max(worst, abs(v - w))
        print(f"| `{run}` | `{ref}` | {len(names)} | {worst} |")


def main(argv: list[str] | None = None) -> None:
    global LINK
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd")
    for name in ("tables", "checks"):
        sp = sub.add_parser(name)
        sp.add_argument("linkage", nargs="?", default=LINK, choices=["signature", "oracle"])
    sub.add_parser("repro")
    args = ap.parse_args(argv)
    cmd = args.cmd or "tables"
    if cmd == "tables":
        tables(getattr(args, "linkage", LINK))
    elif cmd == "checks":
        LINK = args.linkage
        checks()
    else:
        repro()


if __name__ == "__main__":
    main()
