#!/usr/bin/env python3
"""Write MEASUREMENT_CARD.json: every number the README is allowed to state.

`tests/test_published_claims.py` fails when the README quotes a figure this
card does not contain, which is how AGENTS.md rule 2 is enforced.

Usage: uv run python scripts/make_card.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "MEASUREMENT_CARD.json"


def _round(x, n=3):
    return round(x, n) if isinstance(x, (int, float)) else x


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
            "One agent family, two environments and a batch-tool variant of one "
            "(backoffice_bypass), benign agents and post-hoc logging loss only. The bypass "
            "environment stipulates a channel no witness parses; under its registered view "
            "most of the hidden actions are bypassed customer updates that no witness can "
            "record, so the registered cell asks the estimator to extrapolate to actions with "
            "zero capture probability. No live adversarial run: the audit-warning prompt "
            "(named L1 in the bypass registration, not the simulator's L1 rung) is a single "
            "pilot prompt on 12 paired seeds, not an adaptive adversary, and every simulator "
            "L2 number is from the parametric simulator. Canary and sequence-number baselines "
            "were compared post hoc on the confirm pool only "
            "(artifacts/robustness/baselines.json); canaries were never run with bypass."
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
        from decimal import ROUND_HALF_UP, Decimal

        def h3(x: float) -> float:
            # Half up, as the claim gate reads a quoted figure (0.9245 is 0.925).
            return float(Decimal(repr(float(x))).quantize(Decimal("0.001"), ROUND_HALF_UP))

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
