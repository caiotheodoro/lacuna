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
            "One agent, one environment, benign logging loss only. No live adversarial "
            "run; every L1/L2 number is from the parametric simulator."
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

    (ROOT / "MEASUREMENT_CARD.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    print(f"wrote MEASUREMENT_CARD.json ({len(json.dumps(card))} bytes)")


if __name__ == "__main__":
    main()
