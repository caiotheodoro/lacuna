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
    return "$\\infty$" if w is None else f"{w:.1f}"


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
        d = b["configs"][f"{cfg}/oracle"]
        e = d["estimators"]
        rows.append(
            [
                tt(cfg),
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
        "lrrrrrr",
        [
            "config",
            "hidden",
            tt("union_count"),
            tt("ll_independence"),
            tt("ll_bic"),
            "width",
            "detects gap",
        ],
        rows,
        f"Exploratory live run: {b['episodes']} episodes, N={b['n_true']}, {b['reps']} loss "
        "replicates, episodes resampled with replacement. ``hidden'' is the mean number of "
        "actions no witness saw. Source: \\texttt{artifacts/thinning/long-30.bootstrap.json}.",
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
        e = b["configs"][f"{cfg}/oracle"]["estimators"]
        rows.append(
            [
                rho,
                cov(e["ll_independence"]),
                f"{e['ll_independence']['bias']:+.2f}",
                cov(e["ll_bic"]),
                f"{e['ll_bic']['bias']:+.2f}",
            ]
        )
    table(
        "coupling",
        "rrrrr",
        ["$\\rho$", tt("ll_independence"), "bias", tt("ll_bic"), "bias"],
        rows,
        "Share $\\rho$ of actions whose gateway and trace losses share one draw. "
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
        r = w[k]
        rows.append(
            [
                tt(k),
                f"{r['worst_coverage']:.3f}",
                tt(r["worst_config"]),
                f"{r['mean_detects_gap']:.2f}",
                "$\\infty$" if r["mean_width"] is None else f"{r['mean_width']:.1f}",
            ]
        )
    table(
        "worst",
        "lrlrr",
        ["estimator", "worst coverage", "at", "mean detects gap", "mean width"],
        rows,
        f"Across the {w['configs']} exploratory loss configurations. "
        "Source: \\texttt{artifacts/derived.json}.",
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
        for k in (
            "union_count",
            "union_bound",
            "ll_independence",
            "ll_bic",
            "anchor_g4",
            "lc_gated",
        ):
            m = d["estimators"][k]
            rows.append(
                [
                    tt(cfg) if k == "union_count" else "",
                    f"{d['mean_unseen_by_w1w2w3']}" if k == "union_count" else "",
                    tt(k),
                    cov(m),
                    f"{m['detects_gap']:.2f}",
                    wid(m),
                    f"{m['bias']:+.2f}",
                ]
            )
    table(
        "confirm",
        "lrlrrrr",
        ["config", "hidden", "estimator", "coverage", "detects gap", "width", "bias"],
        rows,
        f"Confirmatory run on the held-out configurations: {c['episodes']} episodes, "
        f"N={c['n_true']}, effect-signature linkage, {c['reps']} replicates. "
        "Source: \\texttt{artifacts/thinning/confirm.json}.",
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
                    f"{e['detects_gap']:.2f}",
                    wid(e),
                    f"{e['bias']:+.2f}",
                ]
            )
    table(
        "mnar",
        "lrlrrrr",
        ["config", "hidden", "estimator", "coverage", "detects gap", "width", "bias"],
        rows,
        "Loss concentrated on the ops the env view cannot see (drop rate $\\times$1.8). "
        "Exploratory, after the confirmatory run. "
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
            cells += [cov(x), f"{x['unbounded_rate']:.2f}"]
        rows.append(cells)
    table(
        "mnar_strat",
        "lrrrr",
        ["loss", "\\texttt{ll\\_bic}", "refuses", "\\texttt{strat\\_bic}", "refuses"],
        rows,
        "Coverage, and the share of replicates reported unbounded. Exploratory. "
        "Sources: \\texttt{confirm-mnar-strat.json}, \\texttt{confirm-mnar-hidden.json}.",
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
    ):
        fn()
    missing = [p.name for p in OUT.glob("*.tex") if MISSING in p.read_text()]
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
