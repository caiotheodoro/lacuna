#!/usr/bin/env python3
"""Render the paper's figures from the artifacts. Vector PDF, no hand-typed data.

Usage: uv run --group paper python scripts/paper_figures.py

  fig_mnar_dose.pdf  coverage of ll_bic against the dose of action-dependent loss
                     (artifacts/robustness/mnar-dose.json), exact and signature matching
  fig_bursts.pdf     coverage and mean width of ll_bic against the burst size w
                     (artifacts/robustness/burst-loss.json)
  fig_combined.pdf   the coverage panels of both side by side, for the workshop paper
  fig_study5.pdf     study 5, the registered runs of the primary condition: each run's
                     ll_sa_hom interval against the true count, and the signed error of the
                     primary estimate, the estimate without the end state, the sequence
                     baseline and two floors (artifacts/study5/confirm.json,
                     artifacts/robustness/study5-verdicts.json)

The burst sweep holds ll_bic and ll_independence only. ll_bic_hom is evaluated at
w = 12 in interval-comparison.json and is not plotted: it is a different seed set.
Colours are the Okabe-Ito set; every series also has its own marker and line style,
so the figures read in grayscale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    # Before the matplotlib import, so --help works without the paper group and runs nothing.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "robustness"
OUT = ROOT / "paper" / "figures"

BLUE, VERMILLION, ORANGE, SKY, BLACK = "#0072B2", "#D55E00", "#E69F00", "#56B4E9", "#000000"
BAR, NOMINAL = 0.90, 0.95

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    }
)


def _load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def _guides(ax) -> None:
    """The 0.90 bar the registrations set and the nominal 0.95, in their own legend."""
    bar = ax.axhline(BAR, color="0.35", lw=0.8, ls=(0, (5, 2)), zorder=1, label="_bar")
    nom = ax.axhline(NOMINAL, color="0.6", lw=0.8, ls=(0, (1, 1.5)), zorder=1, label="_nominal")
    series = ax.legend(loc="lower left", frameon=False, borderaxespad=0.2)
    ax.add_artist(series)
    ax.legend(
        handles=[bar, nom],
        labels=["0.90 bar", "0.95 nominal"],
        loc="lower right",
        frameon=False,
        borderaxespad=0.2,
    )


def dose_panel(ax) -> None:
    d = _load("mnar-dose.json")["op_keyed"]
    doses = sorted({float(k.split("/")[0]) for k in d})

    def series(linkage: str, key: str) -> list[float]:
        return [d[f"{x:g}/{linkage}"]["ll_bic"][key] for x in doses]

    for linkage, label, colour, marker, ls in (
        ("oracle", "exact matching", BLUE, "o", "-"),
        ("signature", "signature matching", VERMILLION, "s", "--"),
    ):
        ax.errorbar(
            doses,
            series(linkage, "coverage"),
            yerr=[1.96 * e for e in series(linkage, "coverage_mcse")],
            label=label,
            color=colour,
            marker=marker,
            ls=ls,
            capsize=1.5,
            elinewidth=0.6,
            zorder=3,
        )
    _guides(ax)
    ax.set_xlabel("drop rate on the actions the env diff cannot see ($\\times$ base)")
    ax.set_ylabel("coverage of ll_bic")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(doses[0] - 0.03, doses[-1] + 0.03)
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{x:g}" for x in doses])


BURST_SERIES = (
    ("confirm/financial/indep", "back-office, financial view", BLACK, "o", "-"),
    ("confirm/full/indep", "back-office, full view", SKY, "^", "-."),
    ("long-30/financial/indep", "back-office, long episodes", ORANGE, "D", ":"),
    ("transfer/full/indep", "infra", VERMILLION, "s", "--"),
)
WIDTHS = (1, 2, 3, 6, 12)


def burst_panel(ax, what: str) -> None:
    cells = _load("burst-loss.json")["cells"]
    for key, label, colour, marker, ls in BURST_SERIES:
        ys = [cells[f"{key}/w{w}"]["ll_bic"][what] for w in WIDTHS]
        ax.plot(WIDTHS, ys, label=label, color=colour, marker=marker, ls=ls, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.minorticks_off()
    ax.set_xlabel("block size $w$ (consecutive actions dropped together)")
    if what == "coverage":
        _guides(ax)
        ax.set_ylabel("coverage of ll_bic")
        ax.set_ylim(0.4, 1.0)
    else:
        ax.set_ylabel("mean interval width (actions)")
        ax.set_ylim(0, None)


STUDY5_ERRORS = (
    ("ll_sa_hom", "ll_sa_hom (primary)"),
    ("chapman_sa", "chapman_sa (no end state)"),
    ("seq_gap_tail", "seq_gap_tail"),
    ("union_count", "union_count"),
    ("trust_trace", "trust_trace"),
)


def _study5_runs() -> tuple[str, list[dict]]:
    """The registered primary condition and its runs left, in the artifact's order."""
    verdict = _load("study5-verdicts.json")
    confirm = json.loads((ROOT / "artifacts" / "study5" / "confirm.json").read_text())
    cond, primary = verdict["condition"], verdict["primary"]["primary"]
    runs = [r for r in confirm["runs"] if r["condition"] == cond]
    cov = verdict["primary"]["coverage"]
    assert len(runs) == cov["runs"], (len(runs), cov["runs"])
    assert sum(r["estimates"][primary]["covers"] for r in runs) == cov["covered"]
    assert cov["unbounded"] == 0, "the interval panel draws bounded intervals only"
    return primary, runs


def study5_interval_panel(ax) -> None:
    primary, runs = _study5_runs()
    est = sorted((r["estimates"][primary] | {"n": r["actions"]} for r in runs), key=_point_error)
    for covers, colour, lw, label in (
        (True, "0.55", 0.5, "covers the true count"),
        (False, VERMILLION, 1.1, "does not"),
    ):
        xs = [i for i, e in enumerate(est) if e["covers"] is covers]
        ax.vlines(
            xs,
            [est[i]["lo"] - est[i]["n"] for i in xs],
            [est[i]["hi"] - est[i]["n"] for i in xs],
            color=colour,
            lw=lw,
            label=f"{label} ({len(xs)})",
            zorder=3 if not covers else 2,
        )
    ax.plot(
        range(len(est)),
        [_point_error(e) for e in est],
        color=BLACK,
        lw=0.8,
        zorder=4,
        label="point",
    )
    ax.axhline(0, color="0.2", lw=0.6, ls=(0, (5, 2)), zorder=1)
    ax.set_xlabel(f"the {len(est)} runs left, ordered by the point's error")
    ax.set_ylabel("interval minus true count (actions)")
    ax.set_xlim(-2, len(est) + 1)
    ax.set_ylim(None, 1.45 * max(e["hi"] - e["n"] for e in est))  # room for the legend
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.2, handlelength=1.2)


def _point_error(e: dict) -> float:
    return e["point"] - e["n"]


def study5_error_panel(ax) -> None:
    _, runs = _study5_runs()
    data = [[r["estimates"][name]["error"] for r in runs] for name, _ in STUDY5_ERRORS]
    ax.boxplot(
        data[::-1],
        orientation="horizontal",
        widths=0.55,
        whis=(0, 100),
        medianprops={"color": VERMILLION, "lw": 1.2},
        boxprops={"lw": 0.7},
        whiskerprops={"lw": 0.7},
        capprops={"lw": 0.7},
    )
    ax.set_yticks(range(1, len(STUDY5_ERRORS) + 1))
    ax.set_yticklabels([label for _, label in STUDY5_ERRORS][::-1])
    ax.axvline(0, color="0.2", lw=0.6, ls=(0, (5, 2)), zorder=1)
    ax.set_xlabel("point minus true count (actions)")


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", metadata={"CreationDate": None, "Creator": None})
    plt.close(fig)


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.7), layout="constrained")
    dose_panel(ax)
    _save(fig, "fig_mnar_dose.pdf")

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.6), layout="constrained")
    burst_panel(a, "coverage")
    burst_panel(b, "mean_width")
    a.set_title("(a) coverage", loc="left")
    b.set_title("(b) mean width", loc="left")
    _save(fig, "fig_bursts.pdf")

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.5), layout="constrained")
    dose_panel(a)
    burst_panel(b, "coverage")
    a.set_title("(a) action-dependent loss", loc="left")
    b.set_title("(b) loss in blocks", loc="left")
    a.set_xlabel("drop rate on diff-blind actions ($\\times$ base)")
    _save(fig, "fig_combined.pdf")

    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.5, 1.9), layout="constrained", gridspec_kw={"width_ratios": (1.25, 1)}
    )
    study5_interval_panel(a)
    study5_error_panel(b)
    a.set_title("(a) the registered interval, run by run", loc="left")
    b.set_title("(b) error of each count", loc="left")
    _save(fig, "fig_study5.pdf")
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
