#!/usr/bin/env python3
"""Write MEASUREMENT_CARD.json: every number the README is allowed to state.

`tests/test_published_claims.py` fails when the README quotes a figure this
card does not contain, which is how AGENTS.md rule 2 is enforced.

Usage: uv run python scripts/make_card.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _round(x, n=3):
    return round(x, n) if isinstance(x, (int, float)) else x


def main() -> None:
    boot = json.loads((ART / "thinning" / "long-30.bootstrap.json").read_text())
    card: dict = {
        "protocol": "1.0.0",
        "frozen_preregistration": "docs/PRE-REGISTRATION.md",
        "exploratory": {},
        "confirmatory": {},
        "limitation": (
            "One agent family, two environments, benign agents and post-hoc logging loss "
            "only. No live adversarial run; every L1/L2 number is from the parametric "
            "simulator. Canary and sequence-number baselines were not compared."
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

    (ROOT / "MEASUREMENT_CARD.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    print(f"wrote MEASUREMENT_CARD.json ({len(json.dumps(card))} bytes)")


if __name__ == "__main__":
    main()
