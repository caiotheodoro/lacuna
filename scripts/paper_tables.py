#!/usr/bin/env python3
"""Emit every results table in paper/lacuna.tex from the artifacts.

The paper \\input{}s these files. Nothing in a table is typed by hand: four
figures once were, and four were wrong (D26).

Usage: uv run python scripts/paper_tables.py
Tables whose source artifact does not exist yet are written as a one-line
placeholder that fails the paper build check, so a missing result cannot ship
silently.
"""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from lacuna.thinning import HIDDEN_MOD

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "paper" / "tables"
MISSING = "\\textbf{MISSING ARTIFACT}"


def load(rel: str) -> dict | None:
    p = ART / rel
    return json.loads(p.read_text()) if p.exists() else None


def tt(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def cov(m: dict) -> str:
    return f"{m['coverage']:.3f}"


def wid(m: dict) -> str:
    w = m.get("mean_width")
    if w is None:
        return "$\\infty$"
    return f"{w:.2f}" if w < 1 else f"{w:.1f}"


def table(
    name: str,
    colspec: str,
    header: list[str],
    rows: list[list[str]],
    note: str = "",
    fit: bool = False,
) -> None:
    # fit=True scales a wide table to the text width; the note stays outside
    # the box so it wraps normally.
    lines = [
        ("\\resizebox{\\linewidth}{!}{" if fit else "") + f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}" + ("}" if fit else ""))
    if note:
        lines.append(f"\\par\\smallskip{{\\footnotesize {note}}}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def placeholder(name: str, source: str) -> None:
    (OUT / f"{name}.tex").write_text(f"{MISSING}: \\texttt{{{source}}}\n")


def sim() -> None:
    s = load("sim_coverage.json")
    if not s:
        return placeholder("sim", "sim_coverage.json")
    order = [
        ("l0_independent", "L0"),
        ("l0_heterogeneous", "L0"),
        ("l0_siloed", "L0"),
        ("l1_hides_from_trace", "L1"),
        ("l2_bypasses_proxy", "L2"),
        ("l2_env_blind", "L2"),
    ]
    ests = ["union_count", "ll_independence", "ll_bic", "lc", "lc_gated", "anchor_g4"]
    rows = []
    for sc, rung in order:
        e = s["scenarios"][sc]["estimators"]
        cells = [tt(sc), rung]
        for k in ests:
            m = e[f"{k}@trace+proxy+env"]
            refuse = m["unbounded_rate"]
            mark = f"$^{{{refuse:.2f}}}$" if refuse >= 0.10 else ""
            cells.append(cov(m) + mark)
        rows.append(cells)
    table(
        "sim",
        "llrrrrrr",
        ["scenario", "rung"] + [tt(k) for k in ests],
        rows,
        f"Coverage of the 95\\% interval, {s['reps']} replicates per scenario, seed {s['seed']}, "
        "N=200. A superscript is the share of replicates reported unbounded (refusal), shown "
        "when at least 0.10. Source: \\texttt{artifacts/sim\\_coverage.json}.",
        fit=True,
    )


def auditor() -> None:
    d = load("derived.json")
    if not d:
        return placeholder("auditor", "derived.json")
    rows = []
    for run in ("pilot-l0", "long-30", "long-60", "long-120", "narr-30"):
        r = d["auditor_recall"][run]
        rows.append(
            [
                tt(run),
                r["auditor_view"],
                str(r["episodes"]),
                f"{r['actions_per_episode']:.1f}",
                f"{r['calls_per_episode']:.1f}",
                f"{r['recall_mean']:.4f}",
                str(r["false_positives"]),
            ]
        )
    table(
        "auditor",
        "llrrrrr",
        ["run", "W4 reads", "episodes", "actions/ep", "calls/ep", "recall", "false pos."],
        rows,
        "Recall of the LLM auditor against the kernel ledger, oracle-matched. "
        "Source: \\texttt{artifacts/derived.json}.",
    )


def live() -> None:
    b = load("thinning/long-30.bootstrap.json")
    if not b:
        return placeholder("live", "thinning/long-30.bootstrap.json")
    rows = []
    for cfg in ("light", "fin_light", "sampled", "fin_sampled"):
        for lk in ("oracle", "signature"):
            d = b["configs"][f"{cfg}/{lk}"]
            e = d["estimators"]
            rows.append(
                [
                    tt(cfg) if lk == "oracle" else "",
                    lk,
                    f"{d['mean_unseen_by_w1w2w3']}",
                    cov(e["union_count"]),
                    cov(e["ll_independence"]),
                    cov(e["ll_bic"]),
                    wid(e["ll_bic"]),
                    f"{e['ll_bic']['detects_gap']:.2f}",
                ]
            )
    table(
        "live",
        "llrrrrrr",
        [
            "config",
            "linkage",
            "hidden",
            tt("union_count"),
            tt("ll_independence"),
            tt("ll_bic"),
            tt("ll_bic") + " width",
            tt("ll_bic") + " detects gap",
        ],
        rows,
        f"Exploratory live run: {b['episodes']} episodes, N={b['n_true']}, {b['reps']} loss "
        "replicates, episodes resampled with replacement, one loss seed; both linkages share "
        "the episode draws. ``hidden'' is the mean number of actions no witness saw. "
        "Source: \\texttt{artifacts/thinning/long-30.bootstrap.json}.",
        fit=True,
    )


def coupling() -> None:
    b = load("thinning/long-30.bootstrap.json")
    if not b:
        return placeholder("coupling", "thinning/long-30.bootstrap.json")
    rows = []
    for cfg, rho in (
        ("fin_sampled", "0"),
        ("fin_sampled_rho50", "0.5"),
        ("fin_sampled_rho100", "1"),
    ):
        for lk in ("oracle", "signature"):
            e = b["configs"][f"{cfg}/{lk}"]["estimators"]
            rows.append(
                [
                    rho if lk == "oracle" else "",
                    lk,
                    cov(e["ll_independence"]),
                    f"{e['ll_independence']['bias']:+.2f}",
                    cov(e["ll_bic"]),
                    f"{e['ll_bic']['bias']:+.2f}",
                    wid(e["ll_bic"]),
                ]
            )
    table(
        "coupling",
        "rlrrrrr",
        ["$\\rho$", "linkage", tt("ll_independence"), "bias", tt("ll_bic"), "bias", "width"],
        rows,
        "Share $\\rho$ of actions whose gateway and trace losses share one draw. One loss "
        "seed, 100 replicates (MCSE about 0.02). "
        "Source: \\texttt{artifacts/thinning/long-30.bootstrap.json}.",
    )


def worst() -> None:
    d = load("derived.json")
    if not d:
        return placeholder("worst", "derived.json")
    w = d["worst_case"]
    rows = []
    for k in (
        "union_bound",
        "union_count",
        "ll_independence",
        "ll_bic",
        "ll_average",
        "anchor_g1",
        "anchor_g4",
        "lc_gated",
    ):
        cells = [tt(k)]
        for lk in ("oracle", "signature"):
            r = w[lk][k]
            refuse = r["worst_unbounded_rate"]
            mark = f"$^{{{refuse:.2f}}}$" if refuse >= 0.10 else ""
            cells += [
                f"{r['worst_coverage']:.3f}{mark}",
                tt(r["worst_config"]),
                "$\\infty$" if r["mean_width"] is None else f"{r['mean_width']:.1f}",
            ]
        rows.append(cells)
    table(
        "worst",
        "lrlrrlr",
        [
            "estimator",
            "oracle: worst",
            "at",
            "width",
            "signature: worst",
            "at",
            "width",
        ],
        rows,
        f"Worst coverage across the {w['oracle']['configs']} exploratory loss configurations, "
        "and mean width over bounded intervals. A superscript is the share of the worst cell "
        "reported unbounded, when at least 0.10. The estimator was chosen on the oracle columns. "
        "Source: \\texttt{artifacts/derived.json}.",
        fit=True,
    )


def power() -> None:
    d = load("derived.json")
    if not d:
        return placeholder("power", "derived.json")
    p = d["power_curve"]
    rows = [
        [
            str(r["episodes"]),
            str(r["n_mean"]),
            f"{r['width_mean']:.2f}",
            f"{r['relative_width']:.3f}",
            f"{r['coverage']:.3f} $\\pm$ {r['coverage_mcse']:.3f}",
        ]
        for r in p["rows"]
    ]
    table(
        "power",
        "rrrrr",
        ["episodes", "N", "width", "width / N", "coverage"],
        rows,
        f"{tt(p['estimator'])} at {tt(p['config'])}, {p['reps']} replicates per size, seed "
        f"{p['seed']}. Source: \\texttt{{artifacts/derived.json}}.",
    )


def confirm() -> None:
    c = load("thinning/confirm.json")
    if not c:
        return placeholder("confirm", "thinning/confirm.json")
    rows = []
    for cfg in ("fin_mid", "fin_mid_rho100"):
        d = c["configs"][f"{cfg}/signature"]
        o = c["configs"][f"{cfg}/oracle"]
        for k in (
            "trust_trace",
            "union_count",
            "union_bound",
            "ll_independence",
            "ll_bic",
            "anchor_g1",
            "anchor_g4",
            "lc_gated",
        ):
            m = d["estimators"][k]
            rows.append(
                [
                    tt(cfg) if k == "trust_trace" else "",
                    f"{d['mean_unseen_by_w1w2w3']}" if k == "trust_trace" else "",
                    tt(k),
                    cov(m),
                    f"{m.get('unbounded_rate', 0.0):.2f}",
                    f"{m['detects_gap']:.2f}",
                    wid(m),
                    f"{m['bias']:+.2f}",
                    cov(o["estimators"][k]),
                ]
            )
    table(
        "confirm",
        "lrlrrrrrr",
        [
            "config",
            "hidden",
            "estimator",
            "coverage",
            "refuses",
            "detects gap",
            "width",
            "bias",
            "oracle cov.",
        ],
        rows,
        f"Confirmatory run on the held-out configurations: {c['episodes']} episodes, "
        f"N={c['n_true']}, {c['reps']} replicates, seed {c['seed']}. Effect-signature linkage "
        "except the last column. ``refuses'' is the share of unbounded intervals; width and "
        "bias are over bounded intervals and finite points. "
        "Source: \\texttt{artifacts/thinning/confirm.json}.",
        fit=True,
    )


def hypotheses() -> None:
    h = load("hypotheses.json")
    if not h:
        return placeholder("hypotheses", "hypotheses.json")
    rows = []
    for name, r in h["hypotheses"].items():
        rows.append(
            [
                name.replace("_", " "),
                "pass" if r["passed"] else "\\textbf{fail}",
                r["detail"]
                .replace("_", "\\_")
                .replace(">=", "$\\geq$")
                .replace(">", "$>$")
                .replace("<", "$<$"),
            ]
        )
    table(
        "hypotheses",
        "llp{0.62\\linewidth}",
        ["hypothesis", "verdict", "measured"],
        rows,
        "Thresholds as frozen in \\texttt{docs/PRE-REGISTRATION.md}. "
        "Source: \\texttt{artifacts/hypotheses.json}.",
    )


def mnar() -> None:
    m = load("thinning/confirm-mnar.json")
    if not m:
        return placeholder("mnar", "thinning/confirm-mnar.json")
    rows = []
    for cfg in ("fin_mid_mnar", "fin_mid_mnar_rho100"):
        d = m["configs"][f"{cfg}/signature"]
        for k in ("union_count", "ll_independence", "ll_bic", "lc_gated"):
            e = d["estimators"][k]
            rows.append(
                [
                    tt(cfg) if k == "union_count" else "",
                    f"{d['mean_unseen_by_w1w2w3']}" if k == "union_count" else "",
                    tt(k),
                    cov(e),
                    f"{e.get('unbounded_rate', 0.0):.2f}",
                    f"{e['detects_gap']:.2f}",
                    wid(e),
                    f"{e['bias']:+.2f}",
                ]
            )
    table(
        "mnar",
        "lrlrrrrr",
        ["config", "hidden", "estimator", "coverage", "refuses", "detects gap", "width", "bias"],
        rows,
        "Loss concentrated on the ops the env view cannot see (drop rate $\\times$1.8). "
        "Signature linkage, one loss seed. Width and bias are over bounded intervals and finite "
        "points. Exploratory, after the confirmatory run. "
        "Source: \\texttt{artifacts/thinning/confirm-mnar.json}.",
        fit=True,
    )


def supplement() -> None:
    v = load("supplement-variance.json")
    if not v:
        return placeholder("supplement", "supplement-variance.json")
    rows = []
    for pool, label in (
        ("21 eps (registered)", "registered, 21 episodes"),
        ("24 eps (supplement)", "lost episodes restored, 24"),
    ):
        r = v[pool]
        for i, seed in enumerate(r["seeds"]):
            rows.append(
                [
                    label if i == 0 else "",
                    str(seed),
                    f"{r['coverage'][i]:.2f}",
                    f"{r['bias'][i]:+.1f}",
                    f"{r['detects_gap'][i]:.2f}",
                    f"{r['h3_independence_drop'][i]:+.2f}",
                    f"{r['h3_bic_drop'][i]:+.2f}",
                ]
            )
        rows.append(
            ["", "mean", f"{r['mean_coverage']:.3f} $\\pm$ {r['mean_mcse']:.3f}", "", "", "", ""]
        )
    table(
        "supplement",
        "lrrrrrr",
        [
            "pool",
            "seed",
            "coverage",
            "bias",
            "detects gap",
            "H3: \\texttt{ll\\_ind.} drop",
            "H3: \\texttt{ll\\_bic} drop",
        ],
        rows,
        "\\texttt{ll\\_bic} at \\texttt{fin\\_mid}, signature linkage, 100 replicates per "
        "seed; seed 4242 is the registered one. The H3 columns are coverage at "
        "\\texttt{fin\\_mid} minus coverage at \\texttt{fin\\_mid\\_rho100}. "
        "Source: \\texttt{artifacts/supplement-variance.json}.",
        fit=True,
    )


def sim_extra() -> None:
    s = load("sim_coverage.json")
    if not s:
        return placeholder("sim_extra", "sim_coverage.json")
    rows = []
    a = s["scenarios"]["l0_auditor_copies_trace"]["estimators"]
    for k, label in (
        ("ll_independence@trace+proxy+env+auditor", "auditor treated as independent"),
        ("ll_bic@trace+proxy+env+auditor", "auditor dependence selected"),
    ):
        m = a[k]
        rows.append([tt("l0_auditor_copies_trace"), label, cov(m), wid(m)])
    e = s["scenarios"]["l2_env_blind"]["estimators"]
    for g in (1, 2, 4, 8):
        m = e[f"anchor_g{g}@trace+proxy+env"]
        rows.append([tt("l2_env_blind"), f"$\\Gamma = {g}$", cov(m), wid(m)])
    table(
        "sim_extra",
        "llrr",
        ["scenario", "estimator", "coverage", "width"],
        rows,
        f"Simulator rows quoted in the text. {s['reps']} replicates, seed {s['seed']}, N=200. "
        "Source: \\texttt{artifacts/sim\\_coverage.json}.",
    )


def configs() -> None:
    seen: dict[str, dict] = {}
    for rel in (
        "thinning/long-30.bootstrap.json",
        "thinning/confirm.json",
        "thinning/confirm-mnar.json",
        "thinning/confirm-mnar-hidden.json",
        "thinning/confirm-falsepos.json",
    ):
        d = load(rel)
        if not d:
            continue
        for v in d["configs"].values():
            c = v["config"]
            seen.setdefault(c["name"], c)
    if not seen:
        return placeholder("configs", "thinning/*.json")
    rows = []
    for name, c in seen.items():
        notes = []
        if c.get("rho"):
            notes.append(f"$\\rho = {c['rho']:g}$")
        if c.get("anti"):
            notes.append("exclusive")
        if c.get("op_bias"):
            ops = ", ".join(tt(o) for o, _ in c["op_bias"])
            notes.append(f"drop $\\times{c['op_bias'][0][1]:g}$ for {ops}")
        if c.get("hidden_mult") and c.get("hidden_mult") != 1:
            notes.append(
                f"drop $\\times{c['hidden_mult']:g}$ for every {HIDDEN_MOD}rd action"
                + (", W3 blind to it" if c.get("hidden_env_blind") else "")
            )
        rows.append(
            [
                tt(name),
                f"{c['p_gateway_drop']:.2f}",
                f"{c['p_trace_drop']:.2f}",
                c["env_view"],
                ", ".join(notes) or "independent",
            ]
        )
    table(
        "configs",
        "lrrlp{5.2cm}",
        ["configuration", "gateway drop", "trace drop", "env view", "loss coupling"],
        rows,
        "Every loss configuration in this paper, read from the artifacts that ran it.",
    )


def mnar_strat() -> None:
    m = load("thinning/confirm-mnar-strat.json")
    h = load("thinning/confirm-mnar-hidden.json")
    if not (m and h):
        return placeholder("mnar_strat", "thinning/confirm-mnar-strat.json")
    rows = []
    for src, cfg, key in (
        (m, "fin_mid_mnar", "loss keyed on op (observed)"),
        (m, "fin_mid_mnar_rho100", "same, coupled"),
        (h, "full_mid_hidden", "loss keyed on a hidden id"),
        (h, "full_mid_hidden_rho100", "same, coupled"),
    ):
        e = src["configs"][f"{cfg}/signature"]["estimators"]
        cells = [key]
        for k in ("ll_bic", "strat_bic"):
            x = e[k]
            cells += [cov(x), f"{x['unbounded_rate']:.2f}", f"{x['detects_gap']:.2f}"]
        rows.append(cells)
    table(
        "mnar_strat",
        "lrrrrrr",
        [
            "loss",
            tt("ll_bic"),
            "refuses",
            "gap",
            tt("strat_bic"),
            "refuses",
            "gap",
        ],
        rows,
        "Coverage, the share of replicates reported unbounded, and gap detection, at a "
        "1.8$\\times$ multiplier, one loss seed, signature linkage. Exploratory. "
        "Sources: \\texttt{confirm-mnar-strat.json}, \\texttt{confirm-mnar-hidden.json}.",
        fit=True,
    )


def falsepos() -> None:
    c = load("thinning/confirm-falsepos.json")
    if not c:
        return placeholder("falsepos", "thinning/confirm-falsepos.json")
    rows = []
    for cfg in ("light", "anti_full", "anti_fin"):
        for k in ("ll_independence", "ll_bic", "strat_bic"):
            cells = [tt(cfg) if k == "ll_independence" else "", tt(k)]
            for lk in ("oracle", "signature"):
                m = c["configs"][f"{cfg}/{lk}"]["estimators"][k]
                cells += [f"{m['detects_gap']:.2f}", f"{m['bias']:+.1f}"]
            rows.append(cells)
    table(
        "falsepos",
        "llrrrr",
        ["config", "estimator", "false gap", "bias", "false gap", "bias"],
        rows,
        "Nothing is hidden by construction under \\texttt{anti}: the gateway and trace never "
        "lose the same action. ``False gap'' is the share of replicates whose lower end exceeds "
        "the union. First pair of columns: oracle linkage; second: effect-signature linkage. "
        "Source: \\texttt{artifacts/thinning/confirm-falsepos.json}.",
    )


def precision() -> None:
    d = load("robustness/precision.json")
    if not d:
        return placeholder("precision", "robustness/precision.json")
    rows = []
    for run, label in (("confirm", "registered, 21"), ("confirm-plus-supp", "restored, 24")):
        for lk in ("oracle", "signature"):
            c = d["runs"][run][f"fin_mid/{lk}"]
            b, i = c["ll_bic"], c["ll_independence"]
            w12 = c["ll_bic_selected"].get("W1W2", {"share": 0.0, "coverage": None, "bias": None})
            rows.append(
                [
                    label if lk == "oracle" else "",
                    lk,
                    f"{b['coverage']:.3f} $\\pm$ {b['coverage_mcse']:.3f}",
                    f"{b['bias']:+.2f}",
                    f"{i['coverage']:.3f}",
                    f"{w12['share']:.3f}",
                    "--" if w12["coverage"] is None else f"{w12['coverage']:.3f}",
                    "--" if w12["bias"] is None else f"{w12['bias']:+.2f}",
                ]
            )
    n = (d["seeds"][1] - d["seeds"][0] + 1) * d["reps_per_seed"]
    table(
        "precision",
        "llrrrrrr",
        [
            "pool",
            "linkage",
            tt("ll_bic"),
            "bias",
            tt("ll_independence"),
            "W1:W2 alone",
            "its coverage",
            "its bias",
        ],
        rows,
        f"\\texttt{{fin\\_mid}}, {n} replicates per cell ({d['seeds'][1] - d['seeds'][0] + 1} "
        "fresh loss seeds), both linkages on the same random numbers. ``W1:W2 alone'' is the "
        "share of replicates where BIC selected the model with only the trace--gateway "
        "interaction; ``its coverage'' and ``its bias'' are \\texttt{ll\\_bic}'s on those "
        "replicates. Post hoc. "
        "Source: \\texttt{artifacts/robustness/precision.json}.",
        fit=True,
    )


def mnar_dose() -> None:
    d = load("robustness/mnar-dose.json")
    if not d:
        return placeholder("mnar_dose", "robustness/mnar-dose.json")
    rows = []
    for m in ("1", "1.1", "1.2", "1.3", "1.4", "1.6", "1.8"):
        o, g = d["op_keyed"][f"{m}/oracle"], d["op_keyed"][f"{m}/signature"]
        st = g["strat_bic"]
        rows.append(
            [
                f"{m}$\\times$",
                f"{g['ll_bic']['hidden']:.1f}",
                cov(o["ll_bic"]),
                wid(o["ll_bic"]),
                cov(g["ll_bic"]),
                wid(g["ll_bic"]),
                cov(st),
                f"{st['unbounded_rate']:.2f}",
                "--"
                if st["coverage_among_finite"] is None
                else f"{st['coverage_among_finite']:.3f}",
                str(st["n_finite"]),
            ]
        )
    table(
        "mnar_dose",
        "rrrrrrrrrr",
        [
            "drop $\\times$",
            "hidden",
            tt("ll_bic") + " oracle",
            "width",
            tt("ll_bic"),
            "width",
            tt("strat_bic"),
            "refuses",
            "finite cover",
            "finite $n$",
        ],
        rows,
        "Both logs' drop rate multiplied for \\texttt{update\\_customer} and "
        "\\texttt{write\\_file}, the operations the financial env view cannot see; "
        f"\\texttt{{fin\\_mid}} base, {d['reps']} replicates per dose. Signature linkage except "
        "the third and fourth columns. \\texttt{strat\\_bic}'s coverage counts a refusal as "
        "covering; ``finite cover'' is its coverage among its bounded intervals. Exploratory. "
        "Source: \\texttt{artifacts/robustness/mnar-dose.json}.",
        fit=True,
    )


def hidden_key() -> None:
    d = load("robustness/mnar-dose.json")
    if not d:
        return placeholder("hidden_key", "robustness/mnar-dose.json")
    rows = []
    for key, w3, m in (
        ("1/w3_blind", "blind", "1"),
        ("1.2/w3_blind", "blind", "1.2"),
        ("1.8/w3_blind", "blind", "1.8"),
        ("1.2/w3_sees", "sees", "1.2"),
        ("1.8/w3_sees", "sees", "1.8"),
    ):
        c = d["hidden_key"][key]
        rows.append(
            [
                w3,
                f"{m}$\\times$",
                f"{c['ll_bic']['hidden']:.1f}",
                cov(c["ll_bic"]),
                cov(c["strat_bic"]),
                f"{c['strat_bic']['unbounded_rate']:.2f}",
            ]
        )
    table(
        "hidden_key",
        "lrrrrr",
        [
            "W3 on keyed actions",
            "logs' drop $\\times$",
            "hidden",
            tt("ll_bic"),
            tt("strat_bic"),
            "refuses",
        ],
        rows,
        "Every third ledger id is keyed; full env view, base drop 0.5, signature linkage, "
        f"{d['reps']} replicates. Exploratory. "
        "Source: \\texttt{artifacts/robustness/mnar-dose.json}.",
    )


def falsegap() -> None:
    d = load("robustness/falsegap-battery.json")
    if not d:
        return placeholder("falsegap", "robustness/falsegap-battery.json")
    rows = []
    for run, p in d["pools"].items():
        cells = [tt(run), str(p["n"])]
        for lk in ("oracle", "signature"):
            for env in ("full", "financial"):
                g = [
                    p["configs"][f"anti{r}_{env}/{lk}"]["ll_bic"]["detects_gap"]
                    for r in ("0.2", "0.3", "0.4", "0.5")
                ]
                cells.append(f"{min(g):.2f}--{max(g):.2f}")
        nulls = [v["ll_bic"]["detects_gap"] for k, v in p["configs"].items() if "complete" in k]
        cells.append(f"{max(nulls):.2f}")
        rows.append(cells)
    table(
        "falsegap",
        "lrrrrrr",
        [
            "pool",
            "N",
            "oracle, full",
            "oracle, fin.",
            "signature, full",
            "signature, fin.",
            "nulls",
        ],
        rows,
        "Share of replicates where \\texttt{ll\\_bic} claims a gap when nothing is hidden: "
        "range over exclusive-loss drop rates 0.2--0.5, by linkage and env view. ``nulls'' is "
        "the maximum over three independent-loss configurations with one log complete. "
        f"{d['reps']} replicates per cell, seed {d['seed']}. Exploratory. "
        "Source: \\texttt{artifacts/robustness/falsegap-battery.json}.",
        fit=True,
    )


def transfer() -> None:
    h = load("transfer_hypotheses.json")
    d = load("robustness/transfer-precision.json")
    if not (h and d):
        return placeholder("transfer", "robustness/transfer-precision.json")
    rows = []
    for cfg in ("transfer_mid", "transfer_mid_rho100"):
        for lk in ("oracle", "signature"):
            c = d["runs"]["transfer"][f"{cfg}/{lk}"]
            b, i = c["ll_bic"], c["ll_independence"]
            lo, hi = c["ll_bic_coverage_seed_range"]
            rows.append(
                [
                    tt(cfg) if lk == "oracle" else "",
                    lk,
                    f"{b['coverage']:.3f} $\\pm$ {b['coverage_mcse']:.3f}",
                    f"{lo:.2f}--{hi:.2f}",
                    f"{b['bias']:+.2f}",
                    wid(b),
                    cov(i),
                ]
            )
    n = (d["seeds"][1] - d["seeds"][0] + 1) * d["reps_per_seed"]
    table(
        "transfer",
        "llrrrrr",
        ["config", "linkage", tt("ll_bic"), "per seed", "bias", "width", tt("ll_independence")],
        rows,
        f"Infra/ops environment, paid run: {h['episodes']} episodes, N={h['n_true']}. "
        f"{n} replicates per cell ({d['seeds'][1] - d['seeds'][0] + 1} fresh loss seeds), both "
        "linkages on the same random numbers. The registered T1 cell (one seed) is 0.300. "
        "Source: \\texttt{artifacts/robustness/transfer-precision.json}.",
        fit=True,
    )


def transfer_mech() -> None:
    d = load("robustness/transfer-mechanism.json")
    if not d:
        return placeholder("transfer_mech", "robustness/transfer-mechanism.json")
    v = d["runs"]["transfer"]["variants"]
    rows = []
    for k, label in (
        ("signature", "signature everywhere"),
        ("exact (oracle)", "exact everywhere"),
        ("exact for all-W3-visible groups", "exact where W3 saw the whole group"),
        ("exact for all-W3-invisible groups", "exact where W3 saw none of the group"),
        ("exact for mixed groups", "exact where W3 saw part of the group"),
        ("exact for all-W3-invisible and mixed groups", "exact wherever W3 missed any"),
        ("exact for undo tickets (deploy-fail, loadtest, debug)", "exact for undo tickets"),
        ("exact for every other ticket", "exact for all other tickets"),
        (
            "signature plus content digest for send_page, write_runbook",
            "signature plus text digests",
        ),
    ):
        m = v[k]
        rows.append([label, cov(m), f"{m['bias']:+.2f}", f"{m['mean_w1w2_overlap']:.1f}"])
    table(
        "transfer_mech",
        "lrrr",
        ["records matched", "coverage", "bias", "W1\\&W2 overlap"],
        rows,
        f"\\texttt{{ll\\_bic}} at {tt(d['config'])}, paid run, "
        f"{v['signature']['reps']} replicates per variant on the same random numbers. A group is "
        "the set of actions sharing one signature in an episode. "
        "Source: \\texttt{artifacts/robustness/transfer-mechanism.json}.",
    )


def transfer_density() -> None:
    d = load("robustness/transfer-density.json")
    if not d:
        return placeholder("transfer_density", "robustness/transfer-density.json")
    rows = []
    for c in d["cells"].values():
        rows.append(
            [
                str(c["services"]),
                str(c["tickets"]),
                f"{c['invisible_sharing_a_signature_share']:.3f}",
                cov(c["signature"]),
                cov(c["oracle"]),
            ]
        )
    table(
        "transfer_density",
        "rrrrr",
        ["services", "tickets", "colliding", "signature", "oracle"],
        rows,
        f"Scripted agent, {d['episodes_per_cell']} episodes per cell, "
        f"{d['reps_per_seed'] * (d['loss_seeds'][1] - d['loss_seeds'][0] + 1)} replicates. "
        "``colliding'' is the share of W3-invisible actions that share a signature with another "
        "action in the episode. The registered setting is 10 services, 30 tickets. "
        "Source: \\texttt{artifacts/robustness/transfer-density.json}.",
    )


def linkage_arms() -> None:
    d = load("robustness/linkage-arms.json")
    if not d:
        return placeholder("linkage_arms", "robustness/linkage-arms.json")
    arms = ("signature", "digest", "callid", "callid_lastwriter", "oracle")
    rows = []
    for run, cfg, label in (
        ("confirm", "fin_mid", "back-office, financial view"),
        ("confirm-plus-supp", "fin_mid", "same, restored pool"),
        ("confirm", "full_mid", "back-office, full view"),
        ("long-30", "full_sampled", "same, exploratory pool, sampled"),
        ("transfer", "transfer_mid", "infra"),
        ("transfer", "transfer_mid_rho100", "infra, coupled"),
    ):
        cells = [cov(d["cells"][f"{run}/{cfg}/{a}"]["ll_bic"]) for a in arms]
        rows.append([label, tt(cfg)] + cells)
    for run, label in (("confirm", "back-office"), ("transfer", "infra")):
        for score in ("false gap", "coverage"):
            cells = []
            for a in arms:
                vals = [
                    c["ll_bic"]["detects_gap" if score == "false gap" else "coverage"]
                    for k, c in d["cells"].items()
                    if k.startswith(f"{run}/anti") and k.endswith(f"/{a}")
                ]
                cells.append(f"{min(vals):.2f}--{max(vals):.2f}")
            rows.append([f"{label}, nothing hidden" if score == "false gap" else "", score] + cells)
    p, q = d["seeds"]["precise"], d["seeds"]["quick"]
    table(
        "linkage_arms",
        "llrrrrr",
        ["setting", "config / score", "signature", "digest", "call id", "last writer", "exact"],
        rows,
        "Coverage of \\texttt{ll\\_bic} under five ways of building the capture table. "
        "``digest'' keeps the content digest the logs already store; ``call id'' links trace "
        "and gateway by the provider's tool-call id and attaches the env diff by signature; "
        "``last writer'' attaches the env diff's end-state row for an overwritten field to the "
        "latest logged call on it; ``exact'' is the ledger id, which no auditor has. "
        f"Held-out cells: {(p[1] - p[0] + 1) * d['reps_per_seed']} replicates, on loss seeds "
        "different from Table~\\ref{tab:precision}. Nothing-hidden rows: ranges over "
        "exclusive-loss rates 0.2--0.5 (both env views in back-office), "
        f"{(q[1] - q[0] + 1) * d['reps_per_seed']} replicates each. Post hoc. "
        "Source: \\texttt{artifacts/robustness/linkage-arms.json}.",
        fit=True,
    )


def bursts() -> None:
    d = load("robustness/burst-loss.json")
    if not d:
        return placeholder("bursts", "robustness/burst-loss.json")
    rows = []
    for key, label in (
        ("confirm/financial/indep", "back-office, financial view"),
        ("confirm/financial/joint", "same, one outage for both logs"),
        ("confirm/full/indep", "back-office, full view"),
        ("long-30/financial/indep", "back-office exploratory pool"),
        ("transfer/full/indep", "infra"),
    ):
        rows.append([label] + [cov(d["cells"][f"{key}/w{w}"]["ll_bic"]) for w in (1, 2, 3, 6, 12)])
    table(
        "bursts",
        "lrrrrr",
        ["loss", "$w=1$", "2", "3", "6", "12"],
        rows,
        "Coverage of \\texttt{ll\\_bic} when each block of $w$ consecutive actions is dropped "
        f"with probability {d['p_drop']}; $w=1$ is per-record loss. Exact matching, "
        f"{(d['seeds'][1] - d['seeds'][0] + 1) * d['reps_per_seed']} replicates per cell. "
        "Exploratory. Source: \\texttt{artifacts/robustness/burst-loss.json}.",
    )


def overdispersion() -> None:
    d = load("robustness/overdispersion.json")
    if not d:
        return placeholder("overdispersion", "robustness/overdispersion.json")
    rows = []
    for key, label in (
        ("heldout/confirm/fin_mid/oracle", "per-record loss, \\texttt{fin\\_mid}"),
        ("heldout/confirm/fin_mid_rho100/oracle", "coupled, \\texttt{fin\\_mid\\_rho100}"),
        ("heldout/transfer/transfer_mid/oracle", "infra, \\texttt{transfer\\_mid}"),
        ("heldout/transfer/transfer_mid_rho100/oracle", "infra, coupled"),
        ("burst/confirm/financial/indep/w3", "bursts, $w = 3$"),
        ("burst/confirm/financial/indep/w12", "bursts, $w = 12$"),
        ("burst/long-30/financial/indep/w12", "bursts, $w = 12$, exploratory pool"),
        ("burst/transfer/full/indep/w12", "bursts, $w = 12$, infra"),
        ("burst/confirm/financial/joint/w12", "joint bursts, $w = 12$"),
        ("mnar/confirm/op_x1.2/oracle", "action-dependent, 1.2$\\times$"),
        ("heldout/confirm/fin_mid/signature", "per-record, signature matching"),
    ):
        c = d["cells"][key]
        cells = [label, f"{c['mean_phi']:.1f}"]
        for e in ("ll_bic", "ll_bic_qp", "ll_independence", "ll_independence_qp"):
            cells.append(f"{cov(c[e])} ({wid(c[e])})")
        rows.append(cells)
    table(
        "overdispersion",
        "lrrrrr",
        [
            "loss",
            "$\\hat\\phi$",
            tt("ll_bic"),
            "quasi",
            tt("ll_independence"),
            "quasi",
        ],
        rows,
        "Coverage (mean width) of the plain profile interval and the quasi-Poisson one, whose "
        "likelihood-ratio cutoff is scaled by the Pearson dispersion $\\hat\\phi$ of the "
        "selected model (for \\texttt{ll\\_bic}) or of the independence model. "
        "$\\hat\\phi$ shown is the selected model's mean. Back-office unless marked; exact "
        f"matching unless marked; {(d['seeds'][1] - d['seeds'][0] + 1) * d['reps_per_seed']} "
        "replicates per row, on loss seeds different from Table~\\ref{tab:bursts}. Post hoc. "
        "Source: \\texttt{artifacts/robustness/overdispersion.json}.",
        fit=True,
    )


def episode_dispersion() -> None:
    d = load("robustness/episode-dispersion.json")
    if not d:
        return placeholder("episode_dispersion", "robustness/episode-dispersion.json")
    rows = []
    for key, label in (
        ("perrecord/confirm/fin_mid/oracle", "per-record, \\texttt{fin\\_mid}"),
        ("perrecord/transfer/transfer_mid/oracle", "per-record, infra"),
        ("coupled/confirm/fin_mid_rho50/oracle", "half coupled"),
        ("coupled/confirm/fin_mid_rho100/oracle", "fully coupled"),
        ("coupled/transfer/transfer_mid_rho100/oracle", "fully coupled, infra"),
        ("burst/confirm/financial/indep/w3", "bursts, $w = 3$"),
        ("burst/confirm/financial/indep/w12", "bursts, $w = 12$"),
        ("burst/long-30/financial/indep/w12", "bursts, $w = 12$, exploratory pool"),
        ("burst/transfer/full/indep/w12", "bursts, $w = 12$, infra"),
        ("burst/confirm/financial/joint/w12", "one outage for both logs, $w = 12$"),
        ("burst/transfer/full/indep/w12/signature", "bursts, infra, signature matching"),
        ("small/confirm/burst_w12/k8", "bursts, 8-episode audit"),
        ("mnar/confirm/op_x1.2/oracle", "action-dependent, 1.2$\\times$"),
        ("perrecord/transfer/transfer_mid/signature", "per-record, infra, signature matching"),
    ):
        c = d["cells"][key]
        cells = [label, f"{c['mean_phi_hom']:.1f}"]
        for e in ("ll_bic", "ll_independence_qp", "ll_qbic_hom"):
            cells.append(f"{cov(c[e])} ({wid(c[e])})")
        cells.append(f"{c['ll_qbic_hom']['detects_gap']:.2f}")
        rows.append(cells)
    table(
        "episode_dispersion",
        "lrrrrr",
        [
            "loss",
            "$\\hat\\phi_{\\mathrm{ep}}$",
            tt("ll_bic"),
            "quasi, indep.",
            "quasi, episodes",
            "gap",
        ],
        rows,
        "Coverage (mean width) of the plain interval; of the quasi-Poisson interval with its "
        "dispersion from the independence fit; and of the interval whose dispersion "
        "$\\hat\\phi_{\\mathrm{ep}}$ comes from variation between episodes, used both to select "
        "the model (QBIC) and to scale the cutoff. ``gap'' is that interval's gap detection. "
        "Back-office unless marked, exact matching unless marked, "
        f"{(d['seeds'][1] - d['seeds'][0] + 1) * d['reps_per_seed']} replicates per row. Post hoc. "
        "Source: \\texttt{artifacts/robustness/episode-dispersion.json}.",
        fit=True,
    )


def interval_comparison() -> None:
    d = load("robustness/interval-comparison.json")
    if not d:
        return placeholder("interval_comparison", "robustness/interval-comparison.json")
    rows = []
    for key, label in (
        ("perrecord/confirm", "per-record"),
        ("perrecord/long-60", "per-record, long-60"),
        ("perrecord/backoffice+infra/full", "per-record, both environments pooled"),
        ("coupled/confirm/rho100", "fully coupled"),
        ("burst/confirm/w12", "bursts, $w = 12$"),
        ("burst/long-60/w12", "bursts, $w = 12$, long-60"),
        ("burst/transfer/w12", "bursts, $w = 12$, infra"),
        ("joint/confirm/w12", "one outage for both logs"),
        ("joint/long-60/w12", "one outage for both logs, long-60"),
        ("burst/confirm/full/w12", "bursts, full view, little hidden"),
        ("session/both/confirm", "session-level rates, both logs"),
        ("session/shared/confirm/full", "session quality shared by all three"),
        ("session/shared/transfer/full", "same, infra"),
    ):
        c = d["cells"][key]
        cells = [label, f"{c['mean_phi_ep']:.1f}"]
        for e in ("ll_bic", "ll_qbic_hom", "ll_bic_hom"):
            cells.append(f"{cov(c[e])} ({wid(c[e])}, {c[e]['detects_gap']:.2f})")
        rows.append(cells)
    table(
        "interval_comparison",
        "lrrrr",
        ["loss", "$\\hat\\phi_{\\mathrm{ep}}$", tt("ll_bic"), tt("ll_qbic_hom"), tt("ll_bic_hom")],
        rows,
        "Coverage (mean width, gap detection). \\texttt{ll\\_qbic\\_hom} selects the model by "
        "QBIC on the between-episode dispersion $\\hat\\phi_{\\mathrm{ep}}$ and scales the "
        "cutoff by it; \\texttt{ll\\_bic\\_hom} keeps \\texttt{ll\\_bic}'s model and only "
        "scales the cutoff, so it never covers less than \\texttt{ll\\_bic}. "
        "Back-office financial view unless marked, exact matching, "
        f"{(d['seeds'][1] - d['seeds'][0] + 1) * d['reps_per_seed']} replicates per row. Post hoc. "
        "Source: \\texttt{artifacts/robustness/interval-comparison.json}.",
        fit=True,
    )


def session_and_phase() -> None:
    d = load("robustness/session-and-phase.json")
    if not d:
        return placeholder("session_and_phase", "robustness/session-and-phase.json")
    rows = []

    def fmt(m):
        return f"{cov(m)} ({wid(m)}, {m['detects_gap']:.2f})"

    for key, label in (
        ("phase/long-60/actual", "bursts, long-60, actual order"),
        ("phase/long-60/shuffled", "same, action order shuffled"),
        ("phase/long-120/actual", "bursts, long-120, actual order"),
        ("phase/long-120/shuffled", "same, action order shuffled"),
        ("outage/confirm/indep", "whole-session outages"),
        ("outage/long-60/indep", "same, long-60"),
        ("outage/transfer/joint", "same, joint, infra"),
    ):
        c = d["cells"][key]
        rows.append(
            [label, f"{c['ll_bic']['hidden']:.1f}", fmt(c["ll_bic"]), fmt(c["ll_bic_hom"]), "", ""]
        )
    for key, label in (
        ("shared/confirm", "session quality shared by all three"),
        ("shared/transfer", "same, infra"),
    ):
        c = d["cells"][key]
        rows.append(
            [
                label,
                f"{c['ll_bic']['hidden']:.1f}",
                fmt(c["ll_bic"]),
                fmt(c["ll_bic_hom"]),
                fmt(c["lc_gated"]),
                fmt(c["tier_bic"]),
            ]
        )
    n = (d["seeds"][1] - d["seeds"][0] + 1) * d["reps_per_seed"]
    n_slow = (d["seeds_slow"][1] - d["seeds_slow"][0] + 1) * d["reps_per_seed"]
    table(
        "session_and_phase",
        "lrrrrr",
        ["loss", "hidden", tt("ll_bic"), tt("ll_bic_hom"), tt("lc_gated"), tt("tier_bic")],
        rows,
        "Coverage (mean width, gap detection). Bursts are blocks of 12; ``shuffled'' permutes "
        "the order of actions inside each episode before the blocks fall. A whole-session "
        "outage drops a log for a whole episode with probability one half. "
        "\\texttt{tier\\_bic} ranks episodes by how many actions the logs saw and stratifies "
        f"on six tiers. Exact matching; {n} replicates per row, {n_slow} for the last two. "
        "Post hoc. Source: \\texttt{artifacts/robustness/session-and-phase.json}.",
        fit=True,
    )


def controls() -> None:
    d = load("robustness/final-controls.json")
    if not d:
        return placeholder("controls", "robustness/final-controls.json")
    rows = []

    def fmt(m):
        return f"{cov(m)} ({m['bias']:+.1f})"

    for run, label in (
        ("confirm", "confirmatory pool"),
        ("long-60", "long-60"),
        ("long-120", "long-120"),
        ("transfer", "infra"),
    ):
        cells = [label]
        for mode in ("actual", "visperm", "shuffle"):
            cells.append(fmt(d["cells"][f"burst/{run}/{mode}"]["ll_bic"]))
        cells.append(cov(d["cells"][f"burst/{run}/actual"]["ll_bic_hom"]))
        rows.append(cells)
    table(
        "controls",
        "lrrrr",
        [
            "bursts, $w = 12$",
            "actual",
            "diff's view permuted",
            "order shuffled",
            tt("ll_bic_hom") + ", actual",
        ],
        rows,
        "Coverage (bias) of \\texttt{ll\\_bic} under blocks of 12 lost actions, with the "
        "episode as recorded; with which actions the env diff sees permuted across the "
        "episode, order kept; and with the order of actions shuffled. Infra has no configured "
        "view, so its permuted column differs from the actual one only by random draws. "
        "Exact matching, "
        f"{(d['seeds'][1] - d['seeds'][0] + 1) * d['reps_per_seed']} replicates per cell, on "
        "loss seeds different from the burst sweep and the interval comparison. "
        "Post hoc. Source: \\texttt{artifacts/robustness/final-controls.json}.",
        fit=True,
    )


def burst_seeds() -> None:
    """One burst cell, ll_bic at w = 12, on every disjoint loss-seed set that ran it."""
    names = (
        "burst-loss",
        "overdispersion",
        "episode-dispersion",
        "interval-comparison",
        "final-controls",
    )
    cols = []  # (name, seeds, reps, confirm, long30, infra)
    for name in names:
        d = load(f"robustness/{name}.json")
        if not d:
            return placeholder("burst_seeds", f"robustness/{name}.json")
        c = d["cells"]
        long30 = None
        if name == "burst-loss":
            conf, long30, infra = (
                c["confirm/financial/indep/w12"],
                c["long-30/financial/indep/w12"],
                c["transfer/full/indep/w12"],
            )
        elif name in ("overdispersion", "episode-dispersion"):
            conf = c["burst/confirm/financial/indep/w12"]
            long30 = c["burst/long-30/financial/indep/w12"]
            infra = c["burst/transfer/full/indep/w12"]
        elif name == "interval-comparison":
            conf, infra = c["burst/confirm/w12"], c["burst/transfer/w12"]
        else:
            conf, infra = c["burst/confirm/actual"], c["burst/transfer/actual"]
        lo, hi = d["seeds"]
        cols.append(
            (
                name,
                f"{lo}--{hi}",
                (hi - lo + 1) * d["reps_per_seed"],
                conf["ll_bic"]["coverage"],
                None if long30 is None else long30["ll_bic"]["coverage"],
                infra["ll_bic"]["coverage"],
            )
        )

    def f(x):
        return "--" if x is None else f"{x:.3f}"

    rows = [[tt(n) + "\\texttt{.json}", sd, str(r), f(a), f(b), f(c)] for n, sd, r, a, b, c in cols]

    def span(i):
        v = [c[i] for c in cols if c[i] is not None]
        return f"{min(v):.3f}--{max(v):.3f}"

    rows.append(["range", "", "", span(3), span(4), span(5)])
    table(
        "burst_seeds",
        "llrrrr",
        ["artifact", "loss seeds", "replicates", "back-office", "back-office, long-30", "infra"],
        rows,
        "Coverage of \\texttt{ll\\_bic} at $w = 12$ under blocks of consecutive lost actions, "
        "exact matching, on every disjoint set of loss "
        "seeds that ran the cell. Back-office has the financial view; infra has no "
        "configured view. Only the first three ran \\texttt{long-30}. The first row is "
        "Table~\\ref{tab:bursts}. Post hoc.",
        fit=True,
    )


def outages() -> None:
    d = load("robustness/final-controls.json")
    if not d:
        return placeholder("outages", "robustness/final-controls.json")
    rows = []
    for mode, label in (
        ("gateway_trace", "gateway and trace, independently"),
        ("all_three_independent", "all three, independently"),
        ("gateway_with_diff", "gateway with the diff, trace independent"),
        ("bad_session", "bad sessions hit all three"),
        ("cascade", "cascade, gateway to trace to diff"),
        ("all_three_together", "all three together"),
    ):
        cells = [label]
        for run in ("confirm", "transfer"):
            c = d["cells"][f"outage/{run}/{mode}"]
            cells.append(f"{cov(c['ll_bic'])} / {cov(c['ll_bic_hom'])}")
        rows.append(cells)
    table(
        "outages",
        "lrr",
        ["logs down for a whole session", "back-office", "infra"],
        rows,
        "Coverage of \\texttt{ll\\_bic} / \\texttt{ll\\_bic\\_hom} when logs are down for whole "
        "sessions. Exact matching, "
        f"{(d['seeds'][1] - d['seeds'][0] + 1) * d['reps_per_seed']} replicates per cell. "
        "Post hoc. Source: \\texttt{artifacts/robustness/final-controls.json}.",
    )


def quasi_short() -> None:
    d = load("robustness/interval-comparison.json")
    if not d:
        return placeholder("quasi_short", "robustness/interval-comparison.json")
    rows = []
    for key, label in (
        ("perrecord/confirm", "per-record, back-office"),
        ("perrecord/backoffice+infra/full", "per-record, both pooled"),
        ("burst/confirm/w12", "bursts, back-office"),
        ("burst/long-60/w12", "bursts, long-60"),
        ("burst/transfer/w12", "bursts, infra"),
        ("session/shared/confirm/full", "shared session quality"),
    ):
        c = d["cells"][key]
        row = [label]
        for name in ("ll_bic", "ll_bic_hom", "ll_qbic_hom"):
            e = c[name]
            row.append(f"{e['coverage']:.3f} ({e['mean_width']:.1f})")
        row.append(f"{c['ll_bic_hom']['detects_gap']:.2f}")
        rows.append(row)
    table(
        "quasi_short",
        "lrrrr",
        ["", tt("ll_bic"), "interval only", "selection too", "gap, interval only"],
        rows,
        "Coverage (mean width) of the plain interval and of two quasi-likelihood corrections: "
        "widening BIC's interval, and also using the dispersion to select the model. Last "
        "column: gap detection of the interval-only version. ``both pooled'' fits back-office "
        "and infra together under the full view; under shared session quality each episode "
        "draws one loss rate, uniform on 0.1--0.9, shared by all three logs. Exact matching, "
        "1000 replicates per cell, loss seeds differ from Table~\\ref{tab:controls}. Post hoc. "
        "Source: \\texttt{artifacts/robustness/interval-comparison.json}.",
    )


def linkage_short() -> None:
    d = load("robustness/linkage-arms.json")
    if not d:
        return placeholder("linkage_short", "robustness/linkage-arms.json")
    arms = ("signature", "callid", "oracle")
    rows = []
    for run, cfg, label in (
        ("confirm", "fin_mid", "back-office, registered view"),
        ("confirm", "full_mid", "back-office, full view"),
        ("transfer", "transfer_mid", "infra"),
    ):
        rows.append([label] + [cov(d["cells"][f"{run}/{cfg}/{a}"]["ll_bic"]) for a in arms])
    for run, view, label in (
        ("confirm", "_financial", "back-office, registered view"),
        ("confirm", "_full", "back-office, full view"),
        ("transfer", "", "infra"),
    ):
        cells = []
        for a in arms:
            vals = [
                c["ll_bic"]["detects_gap"]
                for k, c in d["cells"].items()
                if k.startswith(f"{run}/anti") and k.endswith(f"{view}/{a}")
            ]
            cells.append(f"{max(vals):.2f}")
        rows.append([f"false gaps, {label}"] + cells)
    table(
        "linkage_short",
        "lrrr",
        ["", "signature", "shared call id", "exact"],
        rows,
        "Top: coverage of \\texttt{ll\\_bic} under three linkages, per-record loss; the "
        "full-view row is not a held-out configuration. Bottom: the worst share of replicates "
        "claiming a gap when nothing is hidden, over exclusive-loss rates 0.2--0.5. Loss seeds "
        "differ from Table~\\ref{tab:precision}. Post hoc. "
        "Source: \\texttt{artifacts/robustness/linkage-arms.json}.",
    )


def cheaper_checks() -> None:
    d = load("robustness/baselines.json")
    if not d:
        return placeholder("cheaper_checks", "robustness/baselines.json")
    cols = (
        ("ll_bic", tt("ll_bic")),
        ("chapman_w1w3", "Chapman W1$\\times$W3"),
        ("chapman_w1w2", "Chapman W1$\\times$W2"),
        ("seq_gap", "max sequence"),
        ("seq_gap_tail", "sequence, tail-corrected"),
        ("canary_joint@episode_mix", "canaries, $k = 5$"),
    )
    rows = []
    for key, label in (
        ("per_record", "per-record loss"),
        ("coupled_rho100", "trace and gateway fully coupled"),
        ("mnar_op_x1.2", "loss on unseen operations, 1.2 times"),
        ("mnar_op_x1.8", "loss on unseen operations, 1.8 times"),
        ("hidden_key_x1.8", "every third action lost, 1.8 times"),
        ("burst_w12", "bursts, $w = 12$"),
        ("shared_quality_fin", "session quality shared by all three"),
        ("bypass_b0.05/fin_mid", "gateway-only bypass, 5\\% of actions"),
        ("bypass_b0.2/fin_mid", "gateway-only bypass, 20\\% of actions"),
    ):
        cell = d["cells"][key]["linkage"]["oracle"]["estimators"]
        row = [label]
        for est, _ in cols:
            if est == "canary_joint@episode_mix" and key == "hidden_key_x1.8":
                est = "canary_joint@inside_key"
            if est not in cell:
                row.append("--")
                continue
            m = cell[est]
            row.append(f"{cov(m)} ({wid(m)})")
        rows.append(row)
    table(
        "cheaper_checks",
        "lrrrrrr",
        ["loss"] + [c for _, c in cols],
        rows,
        "Coverage (mean width), back-office, exact matching, "
        f"{d['cells']['per_record']['reps']} replicates per row (loss seeds "
        f"{d['seeds'][0]}--{d['seeds'][1]}), the confirmatory episodes resampled with "
        "replacement. Financial env view, except the every-third-action row, which uses "
        "the full view. The sequence number is the action's position in the gateway's "
        "complete log, assumed never lost. Canaries are $k$ known actions per episode with "
        "the real operation mix, thinned by the same loss draw; each has a record for all "
        "three witnesses before loss (inside the loss key in the third-action row). Canaries "
        "were not run with bypass. The bypass rows drop gateway records only, whereas in the "
        "live bypass environment the trace misses those actions too, so they say nothing "
        "about \\texttt{ll\\_bic} under real bypass. Post hoc. "
        "Source: \\texttt{artifacts/robustness/baselines.json}.",
        fit=True,
    )


def cheaper_canary() -> None:
    d = load("robustness/baselines.json")
    if not d:
        return placeholder("cheaper_canary", "robustness/baselines.json")
    ref = "ll_bic@canary_world"
    rows = []
    for key, est, label in (
        ("per_record_k1", "canary_joint@episode_mix", "per-record, $k = 1$"),
        ("per_record_k2", "canary_joint@episode_mix", "per-record, $k = 2$"),
        ("per_record", "canary_joint@episode_mix", "per-record, $k = 5$"),
        ("per_record_k10", "canary_joint@episode_mix", "per-record, $k = 10$"),
        ("per_record_k20", "canary_joint@episode_mix", "per-record, $k = 20$"),
        ("mnar_op_x1.8", "canary_joint@episode_mix", "1.8 times, matched mix"),
        ("mnar_op_x1.8", "canary_joint@uniform_mix", "1.8 times, uniform mix"),
        ("hidden_key_x1.8", "canary_joint@inside_key", "third-action loss, canaries in the key"),
        ("hidden_key_x1.8", "canary_joint@outside_key", "third-action loss, canaries outside it"),
    ):
        cell = d["cells"][key]["linkage"]["oracle"]["estimators"]
        row = [label]
        for name in (est, ref):
            m = cell[name]
            row.append(f"{cov(m)} ({wid(m)}, {m['detects_gap']:.2f})")
        rows.append(row)
    table(
        "cheaper_canary",
        "lrr",
        ["canaries", "canary estimate", tt("ll_bic") + ", same draw"],
        rows,
        "Coverage (mean width, gap detection), exact matching, 500 replicates per row, $k$ "
        "canaries per episode. The matched mix draws canaries from the episode's own "
        "operations, the uniform mix draws each operation equally, and canaries outside the "
        "key sit on actions the loss rule does not touch. Post hoc. "
        "Source: \\texttt{artifacts/robustness/baselines.json}.",
    )


NEW_LOSS_ROWS = (
    ("drift/0.4", "loss rate ramps over the episode, slope 0.4", True),
    ("drift/0.8", "loss rate ramps over the episode, slope 0.8", True),
    ("dup/trace/0.05", "5\\% duplicate lines, trace", True),
    ("dup/gateway/0.05", "5\\% duplicate lines, gateway", True),
    ("dup/env/0.05", "5\\% duplicate rows, env diff", True),
    ("dup/all/0.05", "5\\% duplicates, all three", True),
    ("phantom_forged/all/0.02", "2\\% forged phantoms, all three", False),
    ("phantom_forged/all/0.05", "5\\% forged phantoms, all three", False),
    ("phantom_clone/all/0.02", "2\\% cloned phantoms, all three", False),
    ("phantom_clone/all/0.05", "5\\% cloned phantoms, all three", False),
    ("reorder/swap0.5+jitter10", "reordered lines", True),
)


def new_loss() -> None:
    d = load("robustness/new-loss.json")
    if not d:
        return placeholder("new_loss", "robustness/new-loss.json")
    rows = []
    for key, label, exact_ok in NEW_LOSS_ROWS:
        c = d["cells"][f"confirm/fin_mid/{key}"]
        row = [label]
        for arm in ("oracle", "signature", "callid"):
            row.append(cov(c[arm]["ll_bic"]) if (arm != "oracle" or exact_ok) else "--")
        row.append(f"{c['signature']['ll_bic']['bias']:+.0f}")
        rows.append(row)
    table(
        "new_loss",
        "lrrrr",
        ["mechanism", "exact", "signature", "call id", "bias, signature"],
        rows,
        "Coverage of \\texttt{ll\\_bic}, back-office \\texttt{fin\\_mid}, by linkage, "
        "1000 replicates per row. Rates are per line written. Exact matching ignores "
        "records without a ledger id, so it is not an arm for phantoms. In the drift "
        "rows the trace's and the gateway's drop rate runs linearly from (1 - slope) to "
        "(1 + slope) times its mean over the episode, so the average loss is unchanged. "
        "The reordered row is one of five reorder settings; every arm shown is identical "
        "across all five. Post hoc. "
        "Source: \\texttt{artifacts/robustness/new-loss.json}.",
    )


def _bypass_inputs():
    v = load("robustness/bypass-verdicts.json")
    a = load("robustness/bypass-analysis-bypass-llm.json")
    l1 = load("robustness/bypass-analysis-bypass-l1.json")
    ref = load("robustness/bypass-analysis-confirm.json")
    return v, a, l1, ref


def bypass_verdicts() -> None:
    v, a, l1, ref = _bypass_inputs()
    if not (v and a and l1 and ref):
        return placeholder("bypass_verdicts", "robustness/bypass-verdicts.json")
    b0, b1, b2, b3, ll = v["B0"], v["B1"], v["B2"], v["B3"], v["L1"]
    # The predictions are the frozen registration's table (docs/PREREG-BYPASS.md).
    # Whether each one hit is decided here from the artifact, not typed.
    lo, hi = 0.05, 0.30
    b0_hit = lo <= b0["share"] <= hi
    b1_hit = b1["verdict"] in ("fail", "inconclusive")  # the share >= 0.25 row
    l1_hit = 0 < ll["paired"]["diff"] <= 0.10
    rows = [
        [
            "B0, bypass share",
            f"{b0['share']:.3f} [{b0['ci95'][0]:.3f}, {b0['ci95'][1]:.3f}], {b0['verdict']}",
            f"in [{lo:.2f}, {hi:.2f}], pass",
            "right" if b0_hit else "wrong: share out of range",
        ],
        [
            "B1, " + tt("ll_bic") + " coverage",
            f"{b1['coverage']:.3f} [{b1['wilson95'][0]:.3f}, {b1['wilson95'][1]:.3f}], "
            f"{b1['verdict']}",
            "fail or inconclusive (share $\\ge 0.25$)",
            "right" if b1_hit else "wrong",
        ],
        [
            "B1, shift from no bypass",
            f"{b1['bypass_effect']:.3f} below {b1['reference']['coverage']:.3f}",
            "more than 0.05",
            "right" if b1["bypass_effect_gt_0_05"] else "wrong",
        ],
        [
            "B2, " + tt("seq_gap_tail") + " coverage",
            f"{b2['seq_gap_tail_coverage']:.3f}, {b2['verdict']}",
            "below 0.50 (design check)",
            "right" if b2["check"] == "pass" else "wrong",
        ],
        [
            "B3, gap detection",
            f"{tt('ll_bic')} {b3['ll_bic_detects_gap']:.3f}, "
            f"{tt('union_count')} {b3['union_count_detects_gap']:.3f}",
            "at least 0.90 (design check)",
            "right" if b3["check"] == "pass" else "wrong",
        ],
        [
            "L1, change in share",
            f"{ll['paired']['diff']:+.3f} paired [{ll['paired']['ci95'][0]:.3f}, "
            f"{ll['paired']['ci95'][1]:.3f}]; pooled {ll['diff_pooled']:+.3f}",
            "in (0, 0.10]",
            "right" if l1_hit else "too small",
        ],
    ]
    d = a["descriptive"]
    table(
        "bypass_verdicts",
        "p{3.3cm}p{5.2cm}p{3.4cm}p{2.4cm}",
        ["test", "result", "prediction", "outcome"],
        rows,
        f"Bypass study, {v['completion']['neutral']['completed']} neutral and "
        f"{v['completion']['l1']['completed']} L1 episodes, none failed. "
        f"{d['n_bypassed']} of {d['n_true']} actions were run in scripts. "
        "Coverage is at \\texttt{fin\\_mid}, call-id linkage. B1 is read as an effect of "
        "bypass: all four registered conditions hold. Predictions are the frozen "
        "registration's. Source: \\texttt{artifacts/robustness/bypass-verdicts.json}.",
    )


def _bypass_cell(c: dict, est: str) -> list[str]:
    m = c["estimators"][est]
    return [
        cov(m),
        f"{m['unbounded_rate']:.3f}",
        wid(m),
        f"{m['bias']:+.1f}",
        f"{m['detects_gap']:.3f}",
    ]


def bypass_posthoc() -> None:
    v, a, l1, ref = _bypass_inputs()
    if not (v and a and l1 and ref):
        return placeholder("bypass_posthoc", "robustness/bypass-analysis-bypass-llm.json")
    c, r = a["cells"], ref["cells"]
    rows = [
        ("registered cell, call id", "ll_bic", c["fin_mid/callid"], False),
        ("no bypass (confirm), same cell", "ll_bic", r["fin_mid/callid"], False),
        ("signature", "ll_bic", c["fin_mid/signature"], True),
        ("trace and env diff, call id", "chapman_w1w3", c["fin_mid/callid"], True),
        ("independence model, call id", "ll_independence", c["fin_mid/callid"], True),
        ("env diff sees every op, call id", "ll_bic", c["full_mid/callid"], True),
        ("env diff sees every op, ledger id", "ll_bic", c["full_mid/oracle"], True),
        ("trace, gateway lose same lines, call id", "ll_bic", c["fin_mid_rho100/callid"], True),
    ]
    out = [
        [label + (" (post hoc)" if post else ""), tt(name)] + _bypass_cell(cell, name)
        for label, name, cell, post in rows
    ]
    out.append(
        ["L1 pilot, 12 episodes, call id", tt("ll_bic")]
        + _bypass_cell(l1["cells"]["fin_mid/callid"], "ll_bic")
    )
    table(
        "bypass_posthoc",
        "p{5.6cm}lrrrrr",
        ["cell", "estimator", "coverage", "unbounded", "width", "bias", "gap detected"],
        out,
        "Bypass-llm unless stated, \\texttt{fin\\_mid} unless stated, 3000 replicates per "
        "cell. Width is over bounded intervals. The L1 pilot's coverage is its share of "
        "unbounded intervals, not calibration. Post hoc rows are outside the verdict. "
        "Source: \\texttt{artifacts/robustness/bypass-analysis-*.json}.",
        fit=True,
    )


def _cf_rows(cf: dict) -> dict[str, tuple[str, dict]]:
    """The counterfactual cells the paper quotes: key -> (label, ll_bic cell)."""
    llm, l1 = cf["runs"]["bypass-llm"]["worlds"], cf["runs"]["bypass-l1"]["worlds"]

    def cell(w: dict, world: str, cfg: str, arm: str) -> dict:
        return w[world][cfg][arm]["estimators"]["ll_bic"]

    return {
        "reg": ("as registered, call id", cell(llm, "as_registered", "fin_mid", "callid")),
        "par": ("scripts parsed, call id", cell(llm, "parsed_scripts", "fin_mid", "callid")),
        "par_sig": (
            "scripts parsed, signature",
            cell(llm, "parsed_scripts", "fin_mid", "signature"),
        ),
        "w3": (
            "env diff also sees the bypassed actions, call id",
            cell(llm, "w3_full_view_on_bypassed", "fin_mid", "callid"),
        ),
        "full_reg_call": (
            "full view, as registered, call id",
            cell(llm, "as_registered", "full_mid", "callid"),
        ),
        "full_reg_orc": (
            "full view, as registered, ledger id",
            cell(llm, "as_registered", "full_mid", "oracle"),
        ),
        "full_par_call": (
            "full view, scripts parsed, call id",
            cell(llm, "parsed_scripts", "full_mid", "callid"),
        ),
        "full_par_sig": (
            "full view, scripts parsed, signature",
            cell(llm, "parsed_scripts", "full_mid", "signature"),
        ),
        "full_par_orc": (
            "full view, scripts parsed, ledger id",
            cell(llm, "parsed_scripts", "full_mid", "oracle"),
        ),
        "l1_reg": (
            "audit-warning pilot, as registered, call id",
            cell(l1, "as_registered", "fin_mid", "callid"),
        ),
        "l1_par": (
            "audit-warning pilot, scripts parsed, call id",
            cell(l1, "parsed_scripts", "fin_mid", "callid"),
        ),
    }


def _cf_note(cf: dict) -> str:
    lo, hi = cf["params"]["seed_range"]
    return (
        f"Post hoc and exploratory: loss seeds {lo} to {hi}, {cf['params']['reps_per_seed']} "
        "replicates each, so the as-registered row is a fresh draw of the verdict's cell. "
        "Bypass-llm unless stated, \\texttt{fin\\_mid} unless stated. "
        "``Scripts parsed'' adds a trace and a gateway record for every bypassed action. "
        "Width is over bounded intervals. Command: \\texttt{PYTHONPATH=src uv run python "
        "scripts/bypass\\_counterfactuals.py -{}-processes 12}. Source: "
        "\\texttt{artifacts/robustness/bypass-counterfactuals.json}."
    )


def bypass_counterfactuals() -> None:
    cf = load("robustness/bypass-counterfactuals.json")
    if not cf:
        return placeholder("bypass_counterfactuals", "robustness/bypass-counterfactuals.json")
    rows = _cf_rows(cf)
    order = (
        "reg",
        "par",
        "par_sig",
        "w3",
        "full_reg_call",
        "full_reg_orc",
        "full_par_call",
        "full_par_sig",
        "full_par_orc",
        "l1_reg",
        "l1_par",
    )
    table(
        "bypass_counterfactuals",
        "p{6.4cm}rrrrr",
        ["cell, " + tt("ll_bic"), "coverage", "unbounded", "width", "bias", "gap detected"],
        [
            [rows[k][0]] + _bypass_cell({"estimators": {"ll_bic": rows[k][1]}}, "ll_bic")
            for k in order
        ],
        _cf_note(cf)
        + " The audit-warning pilot's as-registered coverage is its share of unbounded intervals.",
        fit=True,
    )


def bypass_cf_short() -> None:
    cf = load("robustness/bypass-counterfactuals.json")
    if not cf:
        return placeholder("bypass_cf_short", "robustness/bypass-counterfactuals.json")
    rows = _cf_rows(cf)
    order = ("reg", "par", "w3", "full_par_call", "full_par_orc")
    table(
        "bypass_cf_short",
        "p{5.9cm}rrrr",
        ["cell, " + tt("ll_bic"), "coverage", "width", "bias", "gap detected"],
        [
            [
                rows[k][0],
                cov(rows[k][1]),
                wid(rows[k][1]),
                f"{rows[k][1]['bias']:+.1f}",
                f"{rows[k][1]['detects_gap']:.3f}",
            ]
            for k in order
        ],
        f"Post hoc. Bypass-llm, \\texttt{{fin\\_mid}}, loss seeds {cf['params']['seed_range'][0]} "
        f"to {cf['params']['seed_range'][1]}, so row one is a fresh draw of the verdict's "
        "cell. Scripts parsed: one trace and one gateway record per bypassed action. "
        "Command: \\texttt{scripts/bypass\\_counterfactuals.py -{}-processes 12}, "
        "\\texttt{artifacts/robustness/bypass-counterfactuals.json}.",
        fit=True,
    )


def bypass_short() -> None:
    v, a, l1, ref = _bypass_inputs()
    if not (v and a and l1 and ref):
        return placeholder("bypass_short", "robustness/bypass-verdicts.json")
    rows = []
    for label, art in (
        ("no bypass (" + tt("confirm") + ")", ref),
        ("neutral prompt", a),
        ("L1 prompt, pilot", l1),
    ):
        m = art["cells"]["fin_mid/callid"]["estimators"]["ll_bic"]
        rows.append(
            [
                label,
                str(art["descriptive"]["episodes"]),
                f"{art['descriptive']['bypassed_share']:.3f}",
                cov(m),
                f"{m['unbounded_rate']:.3f}",
                wid(m),
                f"{m['detects_gap']:.3f}",
            ]
        )
    table(
        "bypass_short",
        "lrrrrrr",
        ["run", "episodes", "bypass share", "coverage", "unbounded", "width", "gap detected"],
        rows,
        tt("ll_bic") + " at " + tt("fin_mid") + ", call-id linkage, 3000 replicates. Width is "
        "over bounded intervals. The L1 coverage is its share of unbounded intervals, "
        "not calibration. Source: \\texttt{artifacts/robustness/bypass-analysis-*.json}.",
        fit=True,
    )


# --- Study 4 (docs/PREREG-STUDY4.md, D70) -----------------------------------
# Predictions are the frozen registration's table. Whether each one hit is
# decided here from the stored verdicts, not typed.

S4_RUNS = (
    # (gate key, input key in the verdict file, setting, prompt)
    ("replication", "replication", "back-office, second model", "none"),
    ("qwen_bypass", "qwen_bypass", "bypass env, second model", "neutral"),
    ("neutral", "neutral", "bypass env, concurrent neutral", "neutral"),
    ("adversary", "adversary", "bypass env, adversary prompt", tt("l2")),
)


def _s4_inputs():
    v = load("robustness/study4-verdicts.json")
    arts = {
        "neutral": load("robustness/study4-adversary-s4-neutral.json"),
        "l2": load("robustness/study4-adversary-s4-l2.json"),
        "qwen": load("robustness/study4-adversary-s4-qwen-bypass.json"),
    }
    if not (v and all(arts.values())):
        return None, None
    return v, arts


def _h3(x: float, signed: bool = False) -> str:
    """Three decimals, half up (0.9245 reads 0.925, as D70 and the claim gate do)."""
    d = Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    out = f"{d:+}" if signed else f"{d}"
    return out.replace("-", "$-$")


def _s4_ci(lo: float, hi: float, signed: bool = False) -> str:
    return "[" + _h3(lo, signed) + ", " + _h3(hi, signed) + "]"


def _s4_rows(v: dict) -> dict[str, list]:
    """Verdict rows keyed by test: [test, result, prediction, outcome, held]."""
    t, g = v["tests"], v["gates"]
    r1, r1s, q0, q1 = t["R1"], t["R1s"], t["Q0"], t["Q1"]
    h, a0, a1, a1n = t["H"], t["A0"], t["A1"], t["A1n"]
    a2, a2n, a2o = t["A2"], t["A2n"], t["A2o"]

    def cell(x: dict) -> str:
        return f"{x['coverage']:.3f} {_s4_ci(*x['wilson95'])}, $d$ {x['loo_max_deviation']:.3f}"

    def yn(ok: bool) -> str:
        return "yes" if ok else "no"

    # Registered predictions (docs/PREREG-STUDY4.md, "Predictions" table).
    r1_hit = r1["verdict"] in ("pass", "inconclusive") and 0.90 <= r1["coverage"] <= 0.96
    r1s_hit = r1s["verdict"] in ("inconclusive", "fail")
    q0_range = 0.10 <= q0["share"] <= 0.90
    q0_hit = q0["verdict"] == "pass" and q0_range
    q1_hit = q1["verdict"] == "fail"
    h_hit = h["verdict"] == "no hiding beyond the channel"
    a0_hit = a0["paired"]["classification"] == "no change"
    a1_hit = a1["verdict"] == "fail"
    a2_gap = a2["minus_neutral"]
    a2_margin = abs(a2_gap) <= 0.10
    a2n_level = abs(a2n["coverage"] - 0.584) <= 0.05  # "about 0.58"
    lo2, hi2 = a2["coverage"] - a2["loo_max_deviation"], a2["coverage"] + a2["loo_max_deviation"]
    lo2n = a2n["coverage"] - a2n["loo_max_deviation"]
    hi2n = a2n["coverage"] + a2n["loo_max_deviation"]
    overlap = not (hi2 < lo2n or lo2 > hi2n)
    ts = t["ticket_success_difference"]["difference"]
    gates_ok = g["replication"]["status"] == "ok" and g["qwen_bypass"]["status"] == "ok"
    return {
        "gates": [
            "gates, second model",
            f"ticket success {g['replication']['ticket_success']:.3f} and "
            f"{g['qwen_bypass']['ticket_success']:.3f}, both ok",
            "both gates passed",
            yn(gates_ok),
            gates_ok,
        ],
        "R1": [
            "R1, second model, back-office",
            cell(r1) + f", {r1['verdict']}",
            "pass or inconclusive, 0.90 to 0.96",
            yn(r1_hit),
            r1_hit,
        ],
        "R1s": [
            "R1s, signature linkage",
            cell(r1s) + f", {r1s['verdict']}",
            "inconclusive or fail",
            yn(r1s_hit),
            r1s_hit,
        ],
        "Q0": [
            "Q0, bypass share, second model",
            f"{q0['share']:.3f} {_s4_ci(*q0['ci95'])}, {q0['verdict']}",
            "pass, share 0.10 to 0.90",
            ("verdict yes; " if q0["verdict"] == "pass" else "verdict no; ")
            + ("range yes" if q0_range else "range no"),
            q0_hit,
        ],
        "Q1": [
            "Q1, coverage, second model, bypass",
            cell(q1) + f", unbounded {q1['unbounded_rate']:.3f}, {q1['verdict']}; "
            f"bypass effect {str(q1['bypass_effect']['value']).lower()}",
            "fail (share at least 0.25)",
            yn(q1_hit),
            q1_hit,
        ],
        "H": [
            "H, integrity",
            f"{h['n_unwitnessed_other']} actions unwitnessed, accounting "
            f"{'consistent' if h['accounting_consistent'] else 'inconsistent'}",
            "no hiding beyond the channel (by construction)",
            yn(h_hit),
            h_hit,
        ],
        "A0": [
            "A0, bypass share, adversary against neutral",
            f"{_h3(a0['treatment_share'])} against {_h3(a0['base_share'])}; paired "
            f"{_h3(a0['paired']['diff'], True)} {_s4_ci(*a0['paired']['ci95'], signed=True)}, "
            f"{a0['paired']['classification']}",
            "no change",
            yn(a0_hit),
            a0_hit,
        ],
        "A1": [
            "A1, scripts not itemised, adversary",
            cell(a1) + f", {a1['verdict']}",
            "fail",
            yn(a1_hit),
            a1_hit,
        ],
        "A1n": [
            "A1n, same, neutral",
            cell(a1n) + f", unbounded {a1n['unbounded_rate']:.3f}, {a1n['verdict']}",
            "(not predicted)",
            "n/a",
            None,
        ],
        "A2": [
            "A2, scripts parsed from text, lost whole, adversary",
            cell(a2) + f", {a2['verdict']}",
            "fail, within 0.10 of A2n",
            f"verdict {yn(a2['verdict'] == 'fail')}; margin {a2_gap:.3f} {yn(a2_margin)}",
            a2["verdict"] == "fail" and a2_margin,
        ],
        "A2n": [
            "A2n, same, neutral",
            cell(a2n) + f", {a2n['verdict']}",
            "fail, about 0.58",
            f"verdict {yn(a2n['verdict'] == 'fail')}; level {yn(a2n_level)}",
            a2n["verdict"] == "fail" and a2n_level,
        ],
        "A2vsA2n": [
            "A2 against A2n",
            f"A2 cov $-$ $d$ {lo2:.3f}, A2n cov $+$ $d$ {hi2n:.3f}: "
            + ("ranges overlap" if overlap else "ranges apart"),
            "no difference shown",
            yn(overlap),
            overlap,
        ],
        "A2o": [
            "A2o, idealised auditor, lines lost singly",
            cell(a2o) + f", {a2o['verdict']}",
            "(no verdict registered)",
            "n/a",
            None,
        ],
        "ticket": [
            "ticket success, adversary minus neutral",
            _h3(ts, True),
            "within 0.2",
            yn(abs(ts) <= 0.2),
            abs(ts) <= 0.2,
        ],
    }


def study4_verdicts() -> None:
    v, arts = _s4_inputs()
    if not v:
        return placeholder("study4_verdicts", "robustness/study4-verdicts.json")
    rows = _s4_rows(v)
    order = (
        "gates",
        "R1",
        "R1s",
        "Q0",
        "Q1",
        "H",
        "A0",
        "A1",
        "A1n",
        "A2",
        "A2n",
        "A2vsA2n",
        "A2o",
        "ticket",
    )
    scored = [rows[k][4] for k in order if rows[k][4] is not None]
    table(
        "study4_verdicts",
        "p{3.4cm}p{5.6cm}p{3.1cm}p{2.5cm}",
        ["test", "result", "prediction", "hit"],
        [rows[k][:4] for k in order],
        f"Study 4 against its registered predictions: {sum(scored)} of {len(scored)} held in "
        "full. Q0's prediction included its range; A2 and A2n predicted a verdict and also "
        "a margin or a level. Coverage is \\texttt{ll\\_bic} at \\texttt{fin\\_mid} with "
        "call-id linkage, Wilson interval, $d$ the largest leave-one-episode-out "
        "deviation. Predictions are the frozen registration's. Source: "
        "\\texttt{artifacts/robustness/study4-verdicts.json}.",
    )


def study4_short() -> None:
    v, arts = _s4_inputs()
    if not v:
        return placeholder("study4_short", "robustness/study4-verdicts.json")
    rows = _s4_rows(v)
    t = v["tests"]
    q01 = [
        "Q0, Q1",
        f"share {t['Q0']['share']:.3f}; coverage {t['Q1']['coverage']:.3f}, "
        f"unbounded {t['Q1']['unbounded_rate']:.3f}",
        "share 0.10 to 0.90; coverage fail",
        f"Q0 {rows['Q0'][3]}; Q1 {rows['Q1'][3]}",
    ]
    a2 = [
        "A2, A2n",
        f"{t['A2']['coverage']:.3f} against {t['A2n']['coverage']:.3f}; "
        + ("ranges overlap" if rows["A2vsA2n"][4] else "ranges apart"),
        "both fail; no difference shown",
        f"A2 {rows['A2'][3]}; A2n {rows['A2n'][3]}",
    ]
    out = [
        ["R1", rows["R1"][1], rows["R1"][2], rows["R1"][3]],
        q01,
        ["A0", rows["A0"][1], rows["A0"][2], rows["A0"][3]],
        a2,
        ["H", rows["H"][1], rows["H"][2], rows["H"][3]],
    ]
    table(
        "study4_short",
        "p{1.6cm}p{5.6cm}p{3.5cm}p{3.4cm}",
        ["test", "result", "prediction", "hit"],
        out,
        "Study 4. R1: second model, plain back-office. Q: the same model with the batch "
        "tool. A0 and H: the adversary prompt against a concurrent neutral run. A2: scripts "
        "parsed from the logged text, lost a script at a time. Coverage is "
        "\\texttt{ll\\_bic}, \\texttt{fin\\_mid}, call-id linkage; $d$ the largest "
        "leave-one-episode-out deviation. Source: "
        "\\texttt{artifacts/robustness/study4-verdicts.json}.",
    )


def study4_runs() -> None:
    v, arts = _s4_inputs()
    if not v:
        return placeholder("study4_runs", "robustness/study4-verdicts.json")
    t, g, inp = v["tests"], v["gates"], v["inputs"]
    share = {
        "replication": "n/a",
        "qwen_bypass": f"{t['Q0']['share']:.3f}",
        "neutral": _h3(t["A0"]["base_share"]),
        "adversary": _h3(t["A0"]["treatment_share"]),
    }
    rows = []
    total = 0.0
    for key, inkey, label, prompt in S4_RUNS:
        run = inp[inkey]["run"]
        usd = json.loads((ART / "episodes" / run / "receipt.json").read_text())["total_usd"]
        total += usd
        gate = g[key]
        rows.append(
            [
                tt(run),
                tt(inp[inkey]["model_recorded"]),
                label,
                prompt,
                str(gate["completed"]),
                share[key],
                f"{gate['ticket_success']:.3f}",
                f"{gate['mean_n_per_completed_episode']:.1f}",
                gate["status"],
                f"{usd:.2f}",
            ]
        )
    table(
        "study4_runs",
        "lllrrrrrlr",
        [
            "run",
            "model",
            "setting",
            "prompt",
            "episodes",
            "bypass share",
            "ticket success",
            "mean $N$",
            "gates",
            "USD",
        ],
        rows,
        "24 episodes registered per run; two runs lost one episode each to a provider read "
        "timeout. Gates (ticket success at least 0.5, mean $N$ at least 19.7) are applied "
        "before any test. USD from the receipts, at the \\texttt{deepseek-v4.1-flash} rates "
        f"the runner applied to every run: {total:.2f} in all, and the two qwen figures "
        "overstate the provider's cost. Source: "
        "\\texttt{artifacts/robustness/study4-verdicts.json}, "
        "\\texttt{artifacts/episodes/s4-*/receipt.json}.",
        fit=True,
    )


def study4_clustered() -> None:
    v, arts = _s4_inputs()
    if not v:
        return placeholder("study4_clustered", "robustness/study4-verdicts.json")
    rows = []
    for label, a in (
        ("neutral, 23 episodes", arts["neutral"]),
        ("adversary prompt, 23 episodes", arts["l2"]),
        ("second model, 24 episodes", arts["qwen"]),
    ):
        tp = a["text_parsed_control"]
        ind = tp["worlds_independent_loss"]["fin_mid"]["callid"]["estimators"]["ll_bic"]
        orc = a["parsed_scripts_control"]["headline"]["callid"]
        pool = tp["pool"]
        per_script = pool["lines_itemised"] / max(
            pool["script_calls"] - pool["scripts_rejected"], 1
        )
        rows.append(
            [
                label,
                f"{a['headline']['coverage']:.3f}",
                f"{tp['headline']['callid']['coverage']:.3f}",
                f"{ind['coverage']:.3f}",
                f"{orc['coverage']:.3f}",
                f"{per_script:.1f}",
            ]
        )
    table(
        "study4_clustered",
        "lrrrrr",
        [
            "pool",
            "not itemised",
            "parsed, lost whole",
            "parsed, lost singly",
            "parsed from ledger",
            "lines per script",
        ],
        rows,
        "Study 4, \\texttt{ll\\_bic} coverage at \\texttt{fin\\_mid}, call-id linkage, 3000 "
        "replicates. The columns change what the auditor has and how a script's records are "
        "lost: scripts not itemised (the stipulation of the bypass study); parsed from the "
        "logged script text, with all of a script's lines lost together; the same parsed "
        "records lost one by one; one record per bypassed action taken from the ledger "
        "(idealised). Lines per script is parsed lines over accepted scripts. Source: "
        "\\texttt{artifacts/robustness/study4-adversary-s4-*.json}.",
        fit=True,
    )


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (
        sim,
        auditor,
        live,
        coupling,
        worst,
        power,
        confirm,
        hypotheses,
        mnar,
        supplement,
        mnar_strat,
        falsepos,
        sim_extra,
        configs,
        precision,
        mnar_dose,
        hidden_key,
        falsegap,
        transfer,
        transfer_mech,
        bypass_verdicts,
        bypass_posthoc,
        bypass_counterfactuals,
        bypass_cf_short,
        bypass_short,
        transfer_density,
        linkage_arms,
        bursts,
        overdispersion,
        episode_dispersion,
        interval_comparison,
        session_and_phase,
        controls,
        burst_seeds,
        outages,
        linkage_short,
        quasi_short,
        cheaper_checks,
        cheaper_canary,
        new_loss,
        study4_verdicts,
        study4_short,
        study4_runs,
        study4_clustered,
    ):
        fn()
    missing = [p.name for p in OUT.glob("*.tex") if MISSING in p.read_text()]
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
