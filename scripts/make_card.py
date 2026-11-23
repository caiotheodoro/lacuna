#!/usr/bin/env python3
"""Write MEASUREMENT_CARD.json: every number the README is allowed to state.

`tests/test_published_claims.py` fails when the README quotes a figure this
card does not contain, which is how AGENTS.md rule 2 is enforced.

Usage: uv run python scripts/make_card.py [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "MEASUREMENT_CARD.json"


def _round(x, n=3):
    return round(x, n) if isinstance(x, (int, float)) else x


def h3(x: float) -> float:
    # Half up, as the claim gate reads a quoted figure (0.9245 is 0.925).
    return float(Decimal(repr(float(x))).quantize(Decimal("0.001"), ROUND_HALF_UP))


# --- study 5 ----------------------------------------------------------------
# Figures are from the two artifacts. What is typed here is the registration's own text
# (docs/PREREG-STUDY5.md, frozen): its bar, its predictions, its scope and its caveats.
S5_PREREG = "docs/PREREG-STUDY5.md"
COVERAGE_BAR = 0.90  # P1's bar on the Wilson lower end; COVERAGE_FLOOR in study5_lib.py
PREDICTED_SHARE_IN_NO_LOG = (0.03, 0.09)


def coverage_reading(lo: float, hi: float) -> str:
    """How the registration reads a coverage interval against the bar."""
    return "pass" if lo >= COVERAGE_BAR else "missed" if hi < COVERAGE_BAR else "not shown"


def _finite(x) -> bool:
    # The artifacts write an infinite point as the string "inf".
    return isinstance(x, (int, float)) and math.isfinite(x)


def _s5_coverage(c: dict) -> dict:
    return {
        "covered": c["covered"],
        "runs": c["runs"],
        "coverage": h3(c["coverage"]),
        "wilson95": [h3(c["wilson_lo"]), h3(c["wilson_hi"])],
        "unbounded": c["unbounded"],
        "bounded": c["bounded"],
        "covered_bounded": c["covered_bounded"],
        "runs_with_a_point": c["runs_with_a_point"],
        "median_abs_error": h3(c["median_abs_error"]),
        "mean_error": h3(c["mean_error"]),
    }


def _s5_tested(t: dict) -> dict:
    """One estimator put to P1 and P2: its coverage, the paired results and the labels."""
    return {
        **_s5_coverage(t["coverage"]),
        "paired": {
            b: {
                "wins": p["wins"],
                "ties": p["ties"],
                "runs": p["runs"],
                "win_rate": h3(p["win_rate"]),
                "lower_97_5": h3(p["lower_97_5"]),
                "passes": p["passes"],
            }
            for b, p in t["paired"].items()
        },
        "verdict": {
            k: t[k] for k in ("P1", "P2", "P1_reading", "P2_reading", "P1_clauses", "P2_against")
        },
    }


def _s5_predictions(v: dict) -> dict:
    """Each registered prediction and whether it held; None is void or never made."""
    prim, q3 = v["primary"], v["Q3_without_env"]
    g = prim["gates"]
    void = g["underpowered"] or g["not_exercised"]
    clauses, against = prim["P1_clauses"] or {}, prim["P2_against"] or {}
    share = g["mean_union_missed_share"]
    lo, hi = PREDICTED_SHARE_IN_NO_LOG
    above = v["control"]["false_alarms"]["interval_above_truth"]

    def said(predicted: str, result, held: bool | None, **more) -> dict:
        return {"predicted": predicted, "result": result, "held": held, "source": S5_PREREG, **more}

    def tested(predicted: str, result, held: bool) -> dict:
        return said(predicted, result, None if void else held)

    return {
        "s1_exercised": said(
            f"s1 is exercised: mean share of actions in no log between {lo} and {hi}",
            h3(share),
            False if g["not_exercised"] else None if g["underpowered"] else lo <= share <= hi,
        ),
        "P1_coverage": tested(
            "not met: the Wilson lower end of coverage on the s1 runs left is under 0.90",
            clauses.get("coverage"),
            clauses.get("coverage") != "pass",
        )
        | {
            "note": "Wrong by the registration's own rule: the clause passes on the runs left. "
            "With every planned s1 run not in the test counted as not covered the same "
            "clause reads not shown (P1_not_in_test_as_uncovered; D93)."
        },
        "P1_unbounded": tested(
            "under 0.05 of intervals unbounded, so P1 does not fail on that clause",
            clauses.get("unbounded"),
            clauses.get("unbounded") == "pass",
        ),
        "P2_seq_gap_tail": tested(
            "pass", against.get("seq_gap_tail"), against.get("seq_gap_tail") == "pass"
        ),
        "P2_union_count": tested(
            "pass", against.get("union_count"), against.get("union_count") == "pass"
        ),
        "Q3": tested('"missed" on P1', q3["P1_reading"], q3["P1_reading"] == "missed"),
        "control": said("no interval above the truth", above, above == 0),
        "s2": said("none", None, None),
    }


def study5_block(v: dict, a: dict) -> dict:
    """The card's study-5 block from the verdict artifact `v` and the analysis `a`."""
    cond, control = v["condition"], v["control"]["condition"]
    prim = v["primary"]
    g = prim["gates"]
    by = a["by_condition"]
    in_test = {c: [r for r in a["runs"] if r["condition"] == c] for c in by}
    low = v["P1_not_in_test_as_uncovered"]
    no_point = [
        {
            "run": r["run"],
            "actions": r["actions"],
            "union_count": r["union_count"],
            "share_in_no_log": r["union_missed_share"],
            "lo": r["estimates"][prim["primary"]]["lo"],
            "nonconverged": r["estimates"][prim["primary"]]["detail"].get("nonconverged"),
            "counts_as_covering": r["estimates"][prim["primary"]]["covers"],
            "estimators_with_no_point": sorted(
                name for name, e in r["estimates"].items() if not _finite(e["point"])
            ),
        }
        for r in in_test[control]
        if not _finite(r["estimates"][prim["primary"]]["point"])
    ]
    outages = [r["span_relay_outage_s"] for r in in_test[cond]]
    return {
        "source": "artifacts/robustness/study5-verdicts.json, artifacts/study5/confirm.json",
        "preregistration": S5_PREREG,
        "decisions": "D92, corrected by D93",
        "analysis_sha256": v["analysis"]["sha256"],
        "primary": prim["primary"],
        "condition": cond,
        "scope": (
            "If a trusted counter exists at the source or at the sink, use it. This study is "
            "the case where none does: several originators that share no counter, and logs "
            "that lose records on their own through real telemetry components. One rig, with "
            "outages of 15 to 45 s in the tested condition."
        ),
        "caveats": [
            "Every verdict is on the runs left; the number of planned runs not in the test "
            "goes wherever one is quoted (runs, P1_not_in_test_as_uncovered).",
            "The Wilson and exact bounds treat runs as independent, and the runs of a pool "
            "replay shared sessions.",
            "One rig: one layout and one outage shape per condition.",
            "An interval with no upper end counts toward the unbounded share and counts as "
            "covering unless its lower end is above the truth (unbounded, covered_bounded).",
            "Against union_count the paired comparison shows that the amount added is the "
            "right order, not that it is accurate; the informative comparison is the one "
            "with seq_gap_tail.",
            "The runs set aside were bunched in time and the cause is not established (D93).",
            "No claim for or against the estimator by outage length, pool, band or time "
            "window: those tables are in the two artifacts, are exploratory, and are not "
            "carried here.",
        ],
        "runs": {
            c: {
                "planned": a["planned_by_condition"][c],
                "in_test": by[c]["gates"]["runs_passing_integrity"],
                "set_aside": sum(v["runs_set_aside_by"].get(c, {}).values()),
                "set_aside_by_kind": v["runs_set_aside_by"].get(c, {}),
            }
            for c in sorted(by)
        },
        "runs_planned": a["planned_runs"],
        "runs_set_aside": v["runs_set_aside"],
        "gates": {
            "min_runs": v["min_runs"],
            "runs_passing_integrity": g["runs_passing_integrity"],
            "underpowered": g["underpowered"],
            "mean_share_in_no_log": h3(g["mean_union_missed_share"]),
            "mean_telemetry_missed_share": {
                w: h3(x) for w, x in g["mean_telemetry_missed_share"].items()
            },
            "not_exercised": g["not_exercised"],
        },
        cond: {
            prim["primary"]: _s5_tested(prim),
            "P1_not_in_test_as_uncovered": {
                "planned": low["planned"],
                "not_in_test": low["not_in_test"],
                "covered": low["covered"],
                "runs": low["runs"],
                "wilson95": [h3(x) for x in low["wilson"]],
                "reading_by_the_registered_rule": coverage_reading(*low["wilson"]),
                "note": "Every planned run not in the test counted as not covered. The "
                "registration puts it beside the criterion with no criterion of its own.",
            },
            "runs_where_the_union_is_exact": g["runs_with_nothing_in_no_log"],
            "runs_where_the_point_is_the_union": sum(
                r["estimates"][prim["primary"]]["point"] == r["union_count"] for r in in_test[cond]
            ),
            "interval_above_truth": v["false_alarms"]["interval_above_truth"],
            "not_covered": v["exploratory"]["not_covered"][prim["primary"]],
            "span_relay_outage_s": {"min": min(outages), "max": max(outages)},
            "without_end_state": {
                "estimator": v["Q3_without_env"]["primary"],
                **_s5_tested(v["Q3_without_env"]),
            },
            "floors": {name: _s5_coverage(c) for name, c in v["floors"].items()},
            "beside": {
                name: _s5_coverage(by[cond]["estimators"][name])
                for name in ("seq_gap_tail", "strat_bic", "chapman_te")
            },
        },
        "control": {
            "condition": control,
            prim["primary"]: _s5_coverage(v["control"]["coverage"]),
            "interval_above_truth": v["control"]["false_alarms"]["interval_above_truth"],
            "runs_with_no_point": no_point,
            "defect": {
                "status": "fixed in the library, post hoc",
                "what": "On an unstressed run where no log lost anything the registered primary "
                "estimator did not converge and gave an infinite point with no upper end. By "
                "the registered rule that interval counts as covering; it changes no verdict.",
                "since": "docs/DECISIONS.md D94: src/lacuna/estimators/sa_hom.py falls back to "
                "the model's closed form where the registered fit raises. Written after the "
                "verdicts were read; the registered artifacts were not regenerated and every "
                "figure in this block is the registered code's.",
            },
        },
        "s2": {
            "note": "Descriptive: no criterion and no prediction.",
            prim["primary"]: _s5_coverage(by["s2"]["estimators"][prim["primary"]]),
        },
        "predictions": _s5_predictions(v),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT, help="card to write (default: %(default)s)")
    args = ap.parse_args(argv)
    boot = json.loads((ART / "thinning" / "long-30.bootstrap.json").read_text())
    card: dict = {
        "protocol": "1.0.0",
        "frozen_preregistration": "docs/PRE-REGISTRATION.md",
        "exploratory": {},
        "confirmatory": {},
        "limitation": (
            "Studies 1 to 4: one agent family, two environments and a batch-tool variant of one "
            "(backoffice_bypass), benign agents and post-hoc logging loss only. The bypass "
            "environment stipulates a channel no witness parses; under its registered view "
            "most of the hidden actions are bypassed customer updates that no witness can "
            "record, so the registered cell asks the estimator to extrapolate to actions with "
            "zero capture probability. No live adversarial run: the audit-warning prompt "
            "(named L1 in the bypass registration, not the simulator's L1 rung) is a single "
            "pilot prompt on 12 paired seeds, not an adaptive adversary, and every simulator "
            "L2 number is from the parametric simulator. Canary and sequence-number baselines "
            "were compared post hoc on the confirm pool only "
            "(artifacts/robustness/baselines.json); canaries were never run with bypass. "
            "Study 5 loses records through real telemetry components and simulates none, in "
            "one rig: one layout, one outage shape per condition, synthetic tickets, recorded "
            "sessions of two model families from one provider, no adversary. Its verdicts are "
            "on the runs left after some were set aside (study5.runs)."
        ),
    }

    # Exploratory headline: the hardest exploratory configuration.
    d = boot["configs"]["fin_sampled/oracle"]
    e = d["estimators"]
    card["exploratory"] = {
        "source": "artifacts/thinning/long-30.bootstrap.json",
        "episodes": boot["episodes"],
        "n_true": boot["n_true"],
        "reps": boot["reps"],
        "hidden_from_all_three": d["mean_unseen_by_w1w2w3"],
        "ll_bic_coverage": e["ll_bic"]["coverage"],
        "ll_independence_coverage": e["ll_independence"]["coverage"],
        "ll_independence_detects_gap": e["ll_independence"]["detects_gap"],
        "ll_independence_width": e["ll_independence"]["mean_width"],
        "union_count_coverage": e["union_count"]["coverage"],
        "union_bound_detects_gap": e["union_bound"]["detects_gap"],
        "rho100_ll_independence_coverage": boot["configs"]["fin_sampled_rho100/oracle"][
            "estimators"
        ]["ll_independence"]["coverage"],
        "rho100_ll_bic_coverage": boot["configs"]["fin_sampled_rho100/oracle"]["estimators"][
            "ll_bic"
        ]["coverage"],
    }

    # Auditor measurement (D18, D20).
    card["auditor"] = {
        "source": "artifacts/episodes/*/episodes.jsonl",
        "full_transcript_recall": 1.0,
        "narration_recall": 0.0,
        "episodes_full": 50,
        "episodes_narration": 10,
        "false_positives": 0,
    }

    conf = ART / "thinning" / "confirm.json"
    if conf.exists():
        c = json.loads(conf.read_text())
        out = {
            "source": "artifacts/thinning/confirm.json",
            "episodes": c["episodes"],
            "n_true": c["n_true"],
            "pool": c.get("pool"),
        }
        for cfg in ("fin_mid", "fin_mid_rho100"):
            for lk in ("oracle", "signature"):
                key = f"{cfg}/{lk}"
                if key in c["configs"]:
                    est = c["configs"][key]["estimators"]
                    out[key] = {
                        "hidden": c["configs"][key]["mean_unseen_by_w1w2w3"],
                        **{
                            k: {
                                "coverage": est[k]["coverage"],
                                "detects_gap": est[k]["detects_gap"],
                                "width": est[k]["mean_width"],
                                "bias": est[k]["bias"],
                                "rmse": est[k]["rmse"],
                            }
                            for k in ("ll_bic", "ll_independence", "union_count", "union_bound")
                        },
                    }
        card["confirmatory"] = out

    mnar = ART / "thinning" / "confirm-mnar.json"
    if mnar.exists():
        m = json.loads(mnar.read_text())
        card["mnar"] = {"source": "artifacts/thinning/confirm-mnar.json"}
        for cfg in ("fin_mid_mnar", "fin_mid_mnar_rho100"):
            d = m["configs"][f"{cfg}/signature"]
            card["mnar"][cfg] = {
                "hidden": d["mean_unseen_by_w1w2w3"],
                **{
                    k: {
                        "coverage": d["estimators"][k]["coverage"],
                        "detects_gap": d["estimators"][k]["detects_gap"],
                        "width": d["estimators"][k]["mean_width"],
                        "bias": d["estimators"][k]["bias"],
                    }
                    for k in ("ll_bic", "union_count", "lc_gated")
                },
            }

    sv = ART / "supplement-variance.json"
    hs = ART / "hypotheses-supplement.json"
    if sv.exists() and hs.exists():
        v = json.loads(sv.read_text())
        h = json.loads(hs.read_text())
        card["supplement"] = {
            "source": "artifacts/supplement-variance.json, artifacts/hypotheses-supplement.json",
            "episodes": h["episodes"],
            "n_true": h["n_true"],
            "h1_passed": h["hypotheses"]["H1 recovery"]["passed"],
            "registered_mean_coverage": v["21 eps (registered)"]["mean_coverage"],
            "registered_coverage_by_seed": v["21 eps (registered)"]["coverage"],
            "restored_mean_coverage": v["24 eps (supplement)"]["mean_coverage"],
            "restored_coverage_by_seed": v["24 eps (supplement)"]["coverage"],
        }

    ms = ART / "thinning" / "confirm-mnar-strat.json"
    if ms.exists():
        m = json.loads(ms.read_text())
        card["mnar_strat"] = {"source": "artifacts/thinning/confirm-mnar-strat.json"}
        for cfg in ("fin_mid_mnar", "fin_mid_mnar_rho100"):
            e = m["configs"][f"{cfg}/signature"]["estimators"]["strat_bic"]
            card["mnar_strat"][cfg] = {
                "coverage": e["coverage"],
                "unbounded_rate": e["unbounded_rate"],
                "detects_gap": e["detects_gap"],
            }
        mh = json.loads((ART / "thinning" / "confirm-mnar-hidden.json").read_text())
        for cfg in ("full_mid_hidden", "full_mid_hidden_rho100"):
            ests = mh["configs"][f"{cfg}/signature"]["estimators"]
            card["mnar_strat"][cfg] = {
                k: {"coverage": ests[k]["coverage"], "bias": ests[k]["bias"]}
                for k in ("ll_bic", "strat_bic")
            }

    # Post-hoc robustness (D44): precision by linkage, MNAR dose, false gaps.
    rob = ART / "robustness"
    if (rob / "precision.json").exists():
        p = json.loads((rob / "precision.json").read_text())
        card["precision"] = {"source": "artifacts/robustness/precision.json"}
        for run in ("confirm", "confirm-plus-supp"):
            for lk in ("oracle", "signature"):
                c = p["runs"][run][f"fin_mid/{lk}"]
                card["precision"][f"{run}/fin_mid/{lk}"] = {
                    "reps": c["ll_bic"]["reps"],
                    "ll_bic_coverage": c["ll_bic"]["coverage"],
                    "ll_bic_mcse": c["ll_bic"]["coverage_mcse"],
                    "ll_independence_coverage": c["ll_independence"]["coverage"],
                    "w1w2_selected": c["ll_bic_selected"].get("W1W2", {}).get("share", 0.0),
                }
    if (rob / "mnar-dose.json").exists():
        d = json.loads((rob / "mnar-dose.json").read_text())
        card["mnar_dose"] = {
            "source": "artifacts/robustness/mnar-dose.json",
            **{
                f"op_x{m}": {
                    "multiplier": float(m),
                    "ll_bic_coverage": d["op_keyed"][f"{m}/signature"]["ll_bic"]["coverage"],
                    "ll_bic_oracle_coverage": d["op_keyed"][f"{m}/oracle"]["ll_bic"]["coverage"],
                    "strat_bic_coverage": d["op_keyed"][f"{m}/signature"]["strat_bic"]["coverage"],
                    "strat_bic_unbounded": d["op_keyed"][f"{m}/signature"]["strat_bic"][
                        "unbounded_rate"
                    ],
                }
                for m in ("1", "1.1", "1.2", "1.4", "1.8")
            },
        }
    if (rob / "falsegap-battery.json").exists():
        f = json.loads((rob / "falsegap-battery.json").read_text())
        gaps = {
            (lk, env): [
                pool["configs"][f"anti{r}_{env}/{lk}"]["ll_bic"]["detects_gap"]
                for pool in f["pools"].values()
                for r in ("0.2", "0.3", "0.4", "0.5")
            ]
            for lk in ("oracle", "signature")
            for env in ("full", "financial")
        }
        card["falsegap"] = {
            "source": "artifacts/robustness/falsegap-battery.json",
            **{f"{lk}/{env}": {"min": min(v), "max": max(v)} for (lk, env), v in gaps.items()},
        }

    tp = rob / "transfer-precision.json"
    th = ART / "transfer_hypotheses.json"
    if tp.exists() and th.exists():
        t = json.loads(tp.read_text())["runs"]["transfer"]
        h = json.loads(th.read_text())
        card["transfer"] = {
            "source": "artifacts/transfer_hypotheses.json, artifacts/robustness/transfer-*.json",
            "episodes": h["episodes"],
            "n_true": h["n_true"],
            "passed": {k: v["passed"] for k, v in h["hypotheses"].items()},
            "t1_registered_coverage": json.loads((ART / "thinning" / "transfer.json").read_text())[
                "configs"
            ]["transfer_mid/signature"]["estimators"]["ll_bic"]["coverage"],
            "ll_bic_signature_coverage": t["transfer_mid/signature"]["ll_bic"]["coverage"],
            "ll_bic_oracle_coverage": t["transfer_mid/oracle"]["ll_bic"]["coverage"],
        }

    bv = rob / "bypass-verdicts.json"
    ba = rob / "bypass-analysis-bypass-llm.json"
    bl = rob / "bypass-analysis-bypass-l1.json"
    if bv.exists() and ba.exists() and bl.exists():
        v = json.loads(bv.read_text())
        a = json.loads(ba.read_text())
        l1 = json.loads(bl.read_text())
        est = a["cells"]["fin_mid/callid"]["estimators"]
        sig = a["cells"]["fin_mid/signature"]["estimators"]
        spend = {
            run: json.loads((ART / "episodes" / run / "receipt.json").read_text())["total_usd"]
            for run in ("bypass-llm", "bypass-l1")
        }
        card["bypass"] = {
            "source": "artifacts/robustness/bypass-verdicts.json, "
            "artifacts/robustness/bypass-analysis-*.json",
            "preregistration": "docs/PREREG-BYPASS.md",
            "episodes": v["completion"]["neutral"]["completed"],
            "n_true": a["descriptive"]["n_true"],
            "n_bypassed": a["descriptive"]["n_bypassed"],
            "bypass_share": _round(v["B0"]["share"]),
            "bypass_share_ci95": [_round(x) for x in v["B0"]["ci95"]],
            "b0_verdict": v["B0"]["verdict"],
            "b1_verdict": v["B1"]["verdict"],
            "ll_bic_coverage": v["B1"]["coverage"],
            "ll_bic_wilson95": [_round(x) for x in v["B1"]["wilson95"]],
            "ll_bic_unbounded_rate": v["B1"]["unbounded_rate"],
            "ll_bic_width": v["B1"]["mean_width"],
            "reference_coverage": v["B1"]["reference"]["coverage"],
            "reference_width": v["B1"]["reference"]["mean_width"],
            # The width ratio is absolute (333.6 / 39.0). Against N (1283 here, 828 in the
            # reference pool) the ratio is smaller; the pools differ, so both are shown.
            "width_ratio_abs": _round(
                v["B1"]["mean_width"] / v["B1"]["reference"]["mean_width"], 2
            ),
            "width_ratio_rel_n": _round(
                (v["B1"]["mean_width"] / a["descriptive"]["n_true"])
                / (
                    v["B1"]["reference"]["mean_width"]
                    / json.loads((rob / "bypass-analysis-confirm.json").read_text())["descriptive"][
                        "n_true"
                    ]
                ),
                2,
            ),
            "bypass_effect": _round(v["B1"]["bypass_effect"]),
            "seq_gap_tail_coverage": v["B2"]["seq_gap_tail_coverage"],
            "ll_bic_detects_gap": v["B3"]["ll_bic_detects_gap"],
            "union_count_detects_gap": v["B3"]["union_count_detects_gap"],
            "l1_share": v["L1"]["l1_share"],
            "l1_diff_pooled": _round(v["L1"]["diff_pooled"]),
            "l1_diff_paired": _round(v["L1"]["paired"]["diff"]),
            "l1_diff_paired_ci95": [_round(x) for x in v["L1"]["paired"]["ci95"]],
            "l1_ll_bic_coverage": v["L1"]["l1_ll_bic_fin_mid_callid"]["coverage"],
            "l1_ll_bic_coverage_note": (
                "the audit-warning pilot's 0.996 is its share of unbounded intervals, not "
                "calibration"
            ),
            "l1_ll_bic_unbounded_rate": v["L1"]["l1_ll_bic_fin_mid_callid"]["unbounded_rate"],
            "l1_ll_bic_detects_gap": l1["cells"]["fin_mid/callid"]["estimators"]["ll_bic"][
                "detects_gap"
            ],
            "post_hoc": {
                "ll_bic_signature_coverage": _round(sig["ll_bic"]["coverage"]),
                "chapman_w1w3_coverage": _round(est["chapman_w1w3"]["coverage"]),
                "chapman_w1w3_width": est["chapman_w1w3"]["mean_width"],
                "ll_independence_coverage": _round(est["ll_independence"]["coverage"]),
                "full_mid_callid_coverage": a["cells"]["full_mid/callid"]["estimators"]["ll_bic"][
                    "coverage"
                ],
                "full_mid_oracle_coverage": a["cells"]["full_mid/oracle"]["estimators"]["ll_bic"][
                    "coverage"
                ],
                "fin_mid_rho100_coverage": a["cells"]["fin_mid_rho100/callid"]["estimators"][
                    "ll_bic"
                ]["coverage"],
                # The best post hoc coverage of any estimator in any cell of the neutral run,
                # and what the L1 pilot's two best cells cover and at what width (N = 603).
                "best_coverage": max(
                    _round(m["coverage"])
                    for c in a["cells"].values()
                    for name, m in c["estimators"].items()
                    if name in ("ll_bic", "ll_independence", "chapman_w1w3", "chapman_w2w3")
                ),
                "l1_chapman_w1w3_rho100_coverage": l1["cells"]["fin_mid_rho100/callid"][
                    "estimators"
                ]["chapman_w1w3"]["coverage"],
                "l1_chapman_w1w3_rho100_width": l1["cells"]["fin_mid_rho100/callid"]["estimators"][
                    "chapman_w1w3"
                ]["mean_width"],
                "l1_ll_independence_rho100_coverage": l1["cells"]["fin_mid_rho100/callid"][
                    "estimators"
                ]["ll_independence"]["coverage"],
                "l1_ll_independence_rho100_width": l1["cells"]["fin_mid_rho100/callid"][
                    "estimators"
                ]["ll_independence"]["mean_width"],
                "l1_n_true": l1["descriptive"]["n_true"],
            },
            "spend_usd": {k: round(x, 4) for k, x in spend.items()},
            "spend_total_usd": round(sum(spend.values()), 2),
        }

    bc = rob / "bypass-counterfactuals.json"
    if bc.exists() and "bypass" in card:
        cf = json.loads(bc.read_text())
        run = cf["runs"]["bypass-llm"]

        def ll(world: str, cfg: str, arm: str, run_name: str = "bypass-llm") -> dict:
            return cf["runs"][run_name]["worlds"][world][cfg][arm]["estimators"]["ll_bic"]

        hm = run["hidden_mass"]["as_registered"]["fin_mid"]
        eps = cf["episodes"]
        kinds = ("syntax", "invalid", "too_long")
        pr = cf["paired"]
        neutral = [pr["neutral_share"][str(s)] for s in pr["seeds"]]
        card["bypass"]["post_hoc_counterfactuals"] = {
            "source": "artifacts/robustness/bypass-counterfactuals.json "
            "(scripts/bypass_counterfactuals.py --processes 12)",
            "status": cf["status"],
            "loss_seeds": cf["params"]["seed_range"],
            "reps_per_seed": cf["params"]["reps_per_seed"],
            "as_registered_fin_mid_callid_coverage": ll("as_registered", "fin_mid", "callid")[
                "coverage"
            ],
            "as_registered_fin_mid_callid_width": ll("as_registered", "fin_mid", "callid")[
                "mean_width"
            ],
            "as_registered_fin_mid_callid_bias": ll("as_registered", "fin_mid", "callid")["bias"],
            "hidden_mean_per_replicate": hm["mean_hidden"],
            "hidden_bypassed_unrecorded_per_replicate": hm["mean_hidden_by_class"][
                "bypassed_w3_invisible"
            ],
            "hidden_bypassed_unrecorded_share": _round(
                hm["mean_hidden_by_class"]["bypassed_w3_invisible"] / hm["mean_hidden"]
            ),
            "hidden_bypassed_unrecorded_percent": round(
                100 * hm["mean_hidden_by_class"]["bypassed_w3_invisible"] / hm["mean_hidden"],
                1,
            ),
            "hidden_bypassed_unrecorded_ops": sorted(
                hm["mean_hidden_by_class_op"]["bypassed_w3_invisible"]
            ),
            "bypassed_visible_to_registered_env_diff": run["pool"]["fin_mid"]["by_class"][
                "bypassed_w3_visible"
            ],
            "bypassed_invisible_to_registered_env_diff": run["pool"]["fin_mid"]["by_class"][
                "bypassed_w3_invisible"
            ],
            "parsed_fin_mid_callid_coverage": ll("parsed_scripts", "fin_mid", "callid")["coverage"],
            "parsed_fin_mid_callid_wilson95": ll("parsed_scripts", "fin_mid", "callid")[
                "coverage_wilson95"
            ],
            "parsed_fin_mid_callid_width": ll("parsed_scripts", "fin_mid", "callid")["mean_width"],
            "parsed_fin_mid_callid_bias": ll("parsed_scripts", "fin_mid", "callid")["bias"],
            "parsed_fin_mid_signature_coverage": ll("parsed_scripts", "fin_mid", "signature")[
                "coverage"
            ],
            "as_registered_full_mid_callid_coverage": ll("as_registered", "full_mid", "callid")[
                "coverage"
            ],
            "as_registered_full_mid_oracle_coverage": ll("as_registered", "full_mid", "oracle")[
                "coverage"
            ],
            "parsed_full_mid_callid_coverage": ll("parsed_scripts", "full_mid", "callid")[
                "coverage"
            ],
            "parsed_full_mid_signature_coverage": ll("parsed_scripts", "full_mid", "signature")[
                "coverage"
            ],
            "parsed_full_mid_oracle_coverage": ll("parsed_scripts", "full_mid", "oracle")[
                "coverage"
            ],
            "w3_full_view_on_bypassed_fin_mid_callid_coverage": ll(
                "w3_full_view_on_bypassed", "fin_mid", "callid"
            )["coverage"],
            "w3_full_view_on_bypassed_fin_mid_callid_bias": ll(
                "w3_full_view_on_bypassed", "fin_mid", "callid"
            )["bias"],
            "l1_as_registered_fin_mid_callid_coverage": ll(
                "as_registered", "fin_mid", "callid", "bypass-l1"
            )["coverage"],
            "l1_as_registered_fin_mid_callid_unbounded_rate": ll(
                "as_registered", "fin_mid", "callid", "bypass-l1"
            )["unbounded_rate"],
            "l1_parsed_fin_mid_callid_coverage": ll(
                "parsed_scripts", "fin_mid", "callid", "bypass-l1"
            )["coverage"],
            "script_calls": {
                name: {
                    "calls": sum(e["n_script_calls"] for e in eps[run_name]),
                    "rejected_whole": sum(e["n_scripts_rejected"] for e in eps[run_name]),
                    **{k: sum(e["rejected_by_kind"][k] for e in eps[run_name]) for k in kinds},
                }
                for name, run_name in (("neutral", "bypass-llm"), ("audit_warning", "bypass-l1"))
            },
            "paired_audit_warning": {
                "pairs": len(pr["seeds"]),
                "neutral_pooled_share_12_seeds": _round(pr["pooled_neutral_share"]),
                "audit_warning_pooled_share_12_seeds": _round(pr["pooled_l1_share"]),
                "neutral_seeds_with_share_0.96_to_0.98": sum(0.96 <= x < 0.985 for x in neutral),
                "neutral_all_scripts_rejected_seeds": pr["neutral_all_scripts_rejected"],
                "neutral_share_seed_9503": pr["neutral_share"]["9503"],
                "neutral_share_seed_9509": _round(pr["neutral_share"]["9509"], 2),
                # Mean of per-episode share differences (the registered paired +0.121 weights
                # by actions).
                "mean_episode_diff": _round(pr["mean_l1_minus_neutral"]),
                "mean_episode_diff_without_9503_9509": _round(
                    pr["without_9503_9509"]["mean_l1_minus_neutral"]
                ),
            },
        }

    if "post_hoc_counterfactuals" in card.get("bypass", {}):
        cfc = card["bypass"]["post_hoc_counterfactuals"]
        for k, v in list(cfc.items()):
            if k.endswith("_coverage") or k.endswith("_unbounded_rate"):
                cfc[k] = _round(v)
        ph = card["bypass"]["post_hoc"]
        for k in ("l1_chapman_w1w3_rho100_coverage", "l1_ll_independence_rho100_coverage"):
            ph[k] = _round(ph[k])
        # Scripted harness checks (D61): 10% and 30% of actions through run_script.
        sc = {}
        for n in ("10", "30"):
            f = rob / f"bypass-analysis-bypass-scripted-{n}.json"
            if f.exists():
                a_s = json.loads(f.read_text())
                sc[f"share_{n}"] = _round(a_s["descriptive"]["bypassed_share"], 4)
                sc[f"ll_bic_coverage_{n}"] = _round(
                    a_s["cells"]["fin_mid/callid"]["estimators"]["ll_bic"]["coverage"]
                )
        cfc["scripted_checks"] = sc
        eps_llm = json.loads(bc.read_text())["episodes"]["bypass-llm"]
        cfc["paired_audit_warning"]["scripts_rejected_seed_9503"] = next(
            e["n_scripts_rejected"] for e in eps_llm if e["seed"] == 9503
        )

    s4 = rob / "study4-verdicts.json"
    s4l2 = rob / "study4-adversary-s4-l2.json"
    if s4.exists() and s4l2.exists():
        t = json.loads(s4.read_text())["tests"]
        l2 = json.loads(s4l2.read_text())
        pools = {}
        for key, run in (
            ("neutral", "s4-neutral"),
            ("adversary", "s4-l2"),
            ("second_model", "s4-qwen-bypass"),
        ):
            a_ = json.loads((rob / f"study4-adversary-{run}.json").read_text())
            tp = a_["text_parsed_control"]
            singly = tp["worlds_independent_loss"]["fin_mid"]["callid"]["estimators"]["ll_bic"]
            pools[key] = {
                "scripts_not_itemised_coverage": h3(a_["headline"]["coverage"]),
                "parsed_clustered_loss_coverage": h3(tp["headline"]["callid"]["coverage"]),
                "same_records_lost_singly_coverage": h3(singly["coverage"]),
            }
        card["study4"] = {
            "source": "artifacts/robustness/study4-verdicts.json, "
            "artifacts/robustness/study4-adversary-s4-*.json",
            "second_model_backoffice_coverage": h3(t["R1"]["coverage"]),
            "second_model_bypass_share": h3(t["Q0"]["share"]),
            "second_model_bypass_coverage": h3(t["Q1"]["coverage"]),
            "adversary_bypass_share": h3(t["A0"]["treatment_share"]),
            "neutral_bypass_share": h3(t["A0"]["base_share"]),
            "clustered_loss_by_pool": {
                "note": "ll_bic at fin_mid, call-id linkage; compare within a pool, never "
                "across pools",
                **pools,
            },
            "adversary_episodes_completed": l2["completion"]["completed"],
            "adversary_episodes_with_share_zero": l2["behaviour"]["episodes_with_share_zero"],
            "neutral_episodes_with_share_zero": json.loads(
                (rob / "study4-adversary-s4-neutral.json").read_text()
            )["behaviour"]["episodes_with_share_zero"],
            "adversary_episodes_with_keyword_marker_not_refusal": l2["behaviour"][
                "episodes_with_refusal_marker"
            ],
            "keyword_marker_note": "A keyword regex that fires on ordinary task language; one "
            "episode that complied is flagged and another that complied is not. Not a "
            "measure of refusal and no evidence (D73).",
        }

    s5v = rob / "study5-verdicts.json"
    s5a = ART / "study5" / "confirm.json"
    if s5v.exists() and s5a.exists():
        v5 = json.loads(s5v.read_text())
        got = hashlib.sha256(s5a.read_bytes()).hexdigest()
        if got != v5["analysis"]["sha256"]:
            raise SystemExit(f"{s5v} was not written from this {s5a} (sha256 {got})")
        card["study5"] = study5_block(v5, json.loads(s5a.read_text()))
        s5p = ART / "study5" / "posthoc.json"
        if s5p.exists():
            # Made after the verdicts were read (D98): beside the block, never inside it.
            ph = json.loads(s5p.read_text())
            dep = ph["dependence_between_runs"]
            gap = ph["gap_baseline_that_reads_the_end_state"]
            card["study5_post_hoc"] = {
                "source": "artifacts/study5/posthoc.json (scripts/study5_posthoc.py)",
                "note": ph["note"],
                "largest_design_effect_at_or_above_the_bar": dep[
                    "largest_design_effect_at_or_above_the_bar"
                ],
                "design_effect_estimated": False,
                "gap_baseline_that_reads_the_end_state": {
                    "baseline": gap["baseline"],
                    "runs": gap["runs"],
                    "wins": gap["wins"],
                    "ties": gap["ties"],
                    "lower_97_5": _round(gap["lower_97_5"]),
                },
            }

        s5p3 = ART / "study5" / "posthoc3.json"
        if s5p.exists() and s5p3.exists():
            # Made later still (D111, D112): the dependence estimated, the runs set aside
            # described. Beside the block, never inside it.
            p3 = json.loads(s5p3.read_text())
            miss = p3["dependence_between_runs"]["by_outcome"]["miss"]
            aside = p3["runs_set_aside"]
            log, est = aside["beside_the_power_log"], aside["estimator"]
            scored = est["with_the_runs_set_aside_as_scored"]["no_record_as_uncovered"]
            battery = log["windows"]["on_battery_until_external_power"]
            card["study5_post_hoc"]["design_effect_estimated"] = True
            card["study5_post_hoc_later"] = {
                "source": "artifacts/study5/posthoc3.json (scripts/study5_posthoc3.py)",
                "note": p3["note"],
                "dependence_between_runs": {
                    "outcome": "whether a run's interval missed the true count",
                    "rho": _round(miss["rho"]),
                    "p_one_sided": _round(miss["p_one_sided"]),
                    "design_effect": miss["design_effect"],
                    "design_effect_upper_95": miss["design_effect_upper_95"],
                },
                "runs_set_aside": {
                    "set_aside_all_conditions": log["set_aside"],
                    "long": log["long"]["runs"],
                    "long_before_the_log_begins": log["long"]["before_the_log_begins"],
                    "long_beside_a_sleep": log["long"]["beside_a_sleep"],
                    "short": log["by_kind"]["short"],
                    "short_on_battery_until_external_power": battery["set_aside_short"],
                    "runs_on_battery_until_external_power": battery["runs"],
                    "s1_set_aside": est["set_aside"],
                    "s1_with_a_full_record": est["with_a_full_record"],
                    "s1_with_no_record": len(est["with_no_record"]),
                    "s1_covered_as_scored": est["covered"],
                    "covered_with_no_record_as_uncovered": scored["covered"],
                    "planned": scored["runs"],
                    "wilson": [_round(x) for x in scored["wilson"]],
                },
            }

    dj = ART / "derived.json"
    if dj.exists():
        card["spend"] = {
            "source": "artifacts/derived.json (sum of artifacts/episodes/*/receipt.json)",
            "total_usd": json.loads(dj.read_text())["spend"]["total_usd"],
            "study_usd": json.loads(dj.read_text())["spend"]["study_usd"],
            "diagnostic_usd": json.loads(dj.read_text())["spend"]["diagnostic_usd"],
        }

    la = rob / "linkage-arms.json"
    if la.exists():
        d = json.loads(la.read_text())["cells"]
        card["linkage_arms"] = {
            "source": "artifacts/robustness/linkage-arms.json",
            **{
                f"{run}/{cfg}": {
                    a: d[f"{run}/{cfg}/{a}"]["ll_bic"]["coverage"]
                    for a in ("signature", "digest", "callid", "callid_lastwriter", "oracle")
                }
                for run, cfg in (
                    ("confirm", "fin_mid"),
                    ("confirm", "full_mid"),
                    ("transfer", "transfer_mid"),
                    ("transfer", "transfer_mid_rho100"),
                )
            },
        }
    bl = rob / "burst-loss.json"
    if bl.exists():
        d = json.loads(bl.read_text())["cells"]
        card["bursts"] = {
            "source": "artifacts/robustness/burst-loss.json",
            "widths": [1, 12],
            **{
                k: {
                    "w1": d[f"{k}/w1"]["ll_bic"]["coverage"],
                    "w12": d[f"{k}/w12"]["ll_bic"]["coverage"],
                }
                for k in (
                    "confirm/financial/indep",
                    "long-30/financial/indep",
                    "transfer/full/indep",
                )
            },
        }
    od = rob / "overdispersion.json"
    if od.exists():
        d = json.loads(od.read_text())["cells"]
        card["overdispersion"] = {
            "source": "artifacts/robustness/overdispersion.json",
            **{
                k: {e: d[k][e]["coverage"] for e in ("ll_bic", "ll_bic_qp", "ll_independence_qp")}
                for k in (
                    "burst/confirm/financial/indep/w12",
                    "burst/long-30/financial/indep/w12",
                    "burst/transfer/full/indep/w12",
                    "heldout/confirm/fin_mid_rho100/oracle",
                )
            },
        }

    ed = rob / "episode-dispersion.json"
    if ed.exists():
        d = json.loads(ed.read_text())["cells"]
        card["episode_dispersion"] = {
            "source": "artifacts/robustness/episode-dispersion.json",
            "note": "ll_qbic_hom, superseded by ll_bic_hom (D52): it collapses under shared "
            "session quality",
            **{
                k: d[k]["ll_qbic_hom"]["coverage"]
                for k in (
                    "perrecord/confirm/fin_mid/oracle",
                    "coupled/confirm/fin_mid_rho50/oracle",
                    "coupled/confirm/fin_mid_rho100/oracle",
                    "burst/confirm/financial/indep/w3",
                    "burst/transfer/full/indep/w12",
                    "burst/confirm/financial/joint/w12",
                    "mnar/confirm/op_x1.2/oracle",
                )
            },
        }

    ic = rob / "interval-comparison.json"
    if ic.exists():
        d = json.loads(ic.read_text())["cells"]
        card["interval_comparison"] = {
            "source": "artifacts/robustness/interval-comparison.json",
            **{
                k: {e: d[k][e]["coverage"] for e in ("ll_bic", "ll_qbic_hom", "ll_bic_hom")}
                for k in (
                    "perrecord/confirm",
                    "burst/confirm/w12",
                    "burst/long-60/w12",
                    "burst/transfer/w12",
                    "joint/confirm/w12",
                    "session/shared/confirm/full",
                    "session/shared/transfer/full",
                )
            },
        }

    sp = rob / "session-and-phase.json"
    if sp.exists():
        d = json.loads(sp.read_text())["cells"]
        card["session_and_phase"] = {
            "source": "artifacts/robustness/session-and-phase.json",
            "phase": {
                k: {e: d[k][e]["coverage"] for e in ("ll_bic", "ll_bic_hom")}
                for k in (
                    "phase/long-60/actual",
                    "phase/long-60/shuffled",
                    "phase/long-120/actual",
                    "phase/long-120/shuffled",
                )
            },
            "shared": {
                k: {
                    e: d[k][e]["coverage"] for e in ("ll_bic", "ll_bic_hom", "lc_gated", "tier_bic")
                }
                for k in ("shared/confirm", "shared/transfer")
            },
        }

    fc = rob / "final-controls.json"
    if fc.exists():
        d = json.loads(fc.read_text())["cells"]
        card["final_controls"] = {
            "source": "artifacts/robustness/final-controls.json",
            "bursts": {
                k: {"ll_bic": d[k]["ll_bic"]["coverage"], "bias": d[k]["ll_bic"]["bias"]}
                for k in d
                if k.startswith("burst/")
            },
            "outages": {
                k: {e: d[k][e]["coverage"] for e in ("ll_bic", "ll_bic_hom")}
                for k in d
                if k.startswith("outage/")
            },
            "tier_bic": {k: d[k]["tier_bic"]["coverage"] for k in d if k.startswith("tier/")},
        }

    bs = rob / "baselines.json"
    if bs.exists():
        b = json.loads(bs.read_text())
        cells = b["cells"]

        def pick(cell: str, est: str) -> dict:
            m = cells[cell]["linkage"]["oracle"]["estimators"][est]
            return {k: m[k] for k in ("coverage", "mean_width", "detects_gap", "bias")}

        card["baselines"] = {
            "source": "artifacts/robustness/baselines.json",
            "note": "exact matching; bypass_b* cells drop gateway records only (gateway-only "
            "bypass), so they say nothing about ll_bic under real bypass",
            "reps": cells["per_record"]["reps"],
            **{
                cell: {
                    est: pick(cell, est)
                    for est in ("ll_bic", "seq_gap", "seq_gap_tail", "chapman_w1w2", "chapman_w1w3")
                }
                for cell in (
                    "per_record",
                    "coupled_rho100",
                    "mnar_op_x1.2",
                    "mnar_op_x1.8",
                    "hidden_key_x1.8",
                    "burst_w12",
                    "shared_quality_fin",
                    "bypass_b0.05/fin_mid",
                    "bypass_b0.2/fin_mid",
                )
            },
            "canaries": {
                f"{cell}/{variant}": pick(cell, f"canary_joint@{variant}")
                for cell, variants in (
                    ("per_record", ("episode_mix",)),
                    ("coupled_rho100", ("episode_mix",)),
                    ("mnar_op_x1.2", ("episode_mix", "uniform_mix")),
                    ("mnar_op_x1.8", ("episode_mix", "uniform_mix")),
                    ("hidden_key_x1.8", ("inside_key", "outside_key")),
                    ("burst_w12", ("episode_mix",)),
                    ("shared_quality_fin", ("episode_mix",)),
                )
                for variant in variants
            },
        }
    nlj = rob / "new-loss.json"
    if nlj.exists():
        n = json.loads(nlj.read_text())["cells"]
        card["new_loss"] = {
            "source": "artifacts/robustness/new-loss.json",
            "note": "ll_bic at confirm fin_mid; exact matching ignores records without a "
            "ledger id, so it is not an arm for phantoms",
            **{
                key: {arm: n[f"confirm/fin_mid/{key}"][arm]["ll_bic"] for arm in arms}
                for key, arms in (
                    ("drift/0.4", ("oracle", "signature")),
                    ("drift/0.8", ("oracle", "signature")),
                    ("dup/all/0.05", ("oracle", "signature", "callid")),
                    ("phantom_forged/all/0.05", ("signature", "callid")),
                    ("phantom_clone/all/0.05", ("signature", "callid")),
                )
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({len(json.dumps(card))} bytes)")


if __name__ == "__main__":
    main()
