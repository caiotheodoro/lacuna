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

import json
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
        f"{(d['seeds'][1] - d['seeds'][0] + 1) * d['reps_per_seed']} replicates per cell. "
        "Post hoc. Source: \\texttt{artifacts/robustness/final-controls.json}.",
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
        "Coverage (mean width) of the plain interval and of two quasi-likelihood corrections "
        "with the dispersion estimated across episodes: widening the interval of BIC's model, "
        "and also using the dispersion to select the model. Last column: how often the "
        "interval-only version still detects the gap. ``both pooled'' fits back-office and "
        "infra episodes together under the full view; under shared session quality each "
        "episode draws one loss rate, uniform on 0.1--0.9, that all three logs share. "
        "Exact matching, 1000 replicates per "
        "cell, on loss seeds different from Table~\\ref{tab:controls}. Post hoc. Source: "
        "\\texttt{artifacts/robustness/interval-comparison.json}.",
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
        "registered-view and infra rows are held-out configurations, the full-view row is not. "
        "Bottom: the worst share of replicates claiming a gap when nothing is hidden, over "
        "exclusive-loss rates 0.2--0.5. These cells use loss seeds different from "
        "Table~\\ref{tab:precision}. "
        "Post hoc. Source: \\texttt{artifacts/robustness/linkage-arms.json}.",
    )


def main() -> None:
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
        transfer_density,
        linkage_arms,
        bursts,
        overdispersion,
        episode_dispersion,
        interval_comparison,
        session_and_phase,
        controls,
        outages,
        linkage_short,
        quasi_short,
    ):
        fn()
    missing = [p.name for p in OUT.glob("*.tex") if MISSING in p.read_text()]
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
