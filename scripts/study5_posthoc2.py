"""Two more descriptions of study 5 made after its verdicts were read. Post hoc; D107.

Neither was registered, neither has a criterion, and both were chosen knowing the
results. They read `artifacts/study5/confirm.json` and
`artifacts/robustness/study5-verdicts.json` and nothing else.

1. The pooled figure on the estimator's assumption. The estimate takes what both
   telemetry logs lost to be as visible in the end state as the rest. The verdict
   artifact holds, in a table the registration labels exploratory, how many actions both
   telemetry logs lost over the `s1` runs and how many of those the end state shows, and
   the same two counts for every other action. Here the two shares are computed from
   those four counts. The registration fences that table: it is published and used for
   no claim for or against the estimate by band, pool or outage length. The shares are
   pooled over runs that replay shared sessions; they are an average, say nothing about
   any one run, and are not a test of the assumption.

2. The dependence each run would have needed. The registered point is the two-list
   estimate between "any telemetry" and the end state. With x11 actions in both lists,
   x10 in telemetry only, x01 in the end state only and N the true count, the cell
   neither list holds is N - (x11 + x10 + x01), and the odds ratio between the two lists
   that makes the two-list count equal N is x11 * (N - union) / (x10 * x01). Independence
   is 1. It is defined when x10 and x01 are both positive. This describes how far each
   run sat from independence. It is none of the methods of the dependence-sensitivity
   papers the paper cites, and nothing is computed from those.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONFIRM = ROOT / "artifacts" / "study5" / "confirm.json"
VERDICTS = ROOT / "artifacts" / "robustness" / "study5-verdicts.json"
OUT = ROOT / "artifacts" / "study5" / "posthoc2.json"
FACTORS = (1.25, 1.5)
NOTE = (
    "Post hoc (docs/DECISIONS.md D107): computed after the verdicts of study 5 were read, "
    "with no registered criterion. No verdict depends on it."
)
FENCE = (
    "From a table the registration labels exploratory: published, and used for no claim "
    "for or against the estimate by band, pool or outage length. The counts are pooled "
    "over runs that replay shared sessions. The shares are an average, say nothing about "
    "any one run, and are not a test of the assumption."
)
NOT_A_METHOD = (
    "A description of how far each run sat from independence between the two lists. It "
    "is none of the methods of the dependence-sensitivity papers the paper cites, and "
    "nothing is computed from them."
)


def _share(shown: int, actions: int) -> float | None:
    return round(shown / actions, 4) if actions else None


def pooled_shown(lost: dict, runs: int, not_described: int) -> dict:
    """The share the end state shows of what both telemetry logs lost, and of the rest."""
    return {
        "source": "artifacts/robustness/study5-verdicts.json: exploratory.lost_by_both",
        "runs": runs,
        "runs_not_described": not_described,
        "lost_by_both_telemetry_logs": {
            "actions": lost["actions"],
            "shown": lost["shown_by_end_state"],
            "share": _share(lost["shown_by_end_state"], lost["actions"]),
        },
        "the_rest": {
            "actions": lost["rest"],
            "shown": lost["rest_shown"],
            "share": _share(lost["rest_shown"], lost["rest"]),
        },
        "no_link": lost["no_link"],
        "fence": FENCE,
    }


def two_list_cells(tables: dict[str, list[int]]) -> tuple[int, int, int]:
    """(both, telemetry only, end state only) of a run, pooled as `sa_hom` pools them.

    The planes' tables are summed. Cell i of a table counts the actions seen by exactly
    the logs whose bit is set (spans 1, access 2, end state 4), and cell 0 is not read.
    """
    pooled = [sum(t[i] for t in tables.values()) for i in range(8)]
    env_only = pooled[4]
    tele_only = pooled[1] + pooled[2] + pooled[3]
    both = pooled[5] + pooled[6] + pooled[7]
    return both, tele_only, env_only


def implied_odds_ratio(both: int, tele_only: int, env_only: int, actions: int) -> float | None:
    """The odds ratio between the two lists at which the two-list count is the true one."""
    if tele_only <= 0 or env_only <= 0:
        return None
    unseen = actions - (both + tele_only + env_only)
    return both * unseen / (tele_only * env_only)


def dependence_needed(runs: list[dict]) -> dict:
    """Over the runs: how many have an implied odds ratio, its quartiles, and how many
    fall within each factor of 1 (both ends counted)."""
    ratios, exact = [], 0
    for r in runs:
        both, tele_only, env_only = two_list_cells(r["tables"])
        exact += r["actions"] == both + tele_only + env_only
        ratio = implied_odds_ratio(both, tele_only, env_only, r["actions"])
        if ratio is not None:
            ratios.append(ratio)
    quartiles = None
    if ratios:
        q1, median, q3 = (
            statistics.quantiles(ratios, n=4, method="inclusive") if len(ratios) > 1 else ratios * 3
        )
        quartiles = {"q1": round(q1, 4), "median": round(median, 4), "q3": round(q3, 4)}
    return {
        "lists": ["any telemetry (spans or access)", "end state"],
        "formula": "x11 * (N - x11 - x10 - x01) / (x10 * x01); independence is 1",
        "runs": len(runs),
        "defined": len(ratios),
        "not_defined": len(runs) - len(ratios),
        "defined_when": "x10 and x01 are both positive",
        "implied_odds_ratio": quartiles,
        "quartiles": "statistics.quantiles, n=4, method inclusive",
        "within_a_factor_of_1": [
            {
                "factor": f,
                "from": round(1 / f, 4),
                "to": f,
                "runs": sum(1 <= ratio * f and ratio <= f for ratio in ratios),
            }
            for f in FACTORS
        ],
        # A run whose union is already the true count has an implied odds ratio of 0.
        "union_already_the_count": exact,
        "what_it_is_not": NOT_A_METHOD,
    }


def summarise(confirm: dict, verdicts: dict, condition: str = "s1") -> dict:
    assert verdicts["condition"] == condition, "the verdict artifact is of another condition"
    runs = [r for r in confirm["runs"] if r["condition"] == condition]
    exploratory = verdicts["exploratory"]
    return {
        "note": NOTE,
        "condition": condition,
        "end_state_shows_what_both_telemetry_logs_lost": pooled_shown(
            exploratory["lost_by_both"], len(runs), exploratory["runs_not_described"]
        ),
        "dependence_each_run_would_need": dependence_needed(runs),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--analysis", type=Path, default=CONFIRM)
    p.add_argument("--verdicts", type=Path, default=VERDICTS)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    out = summarise(json.loads(a.analysis.read_text()), json.loads(a.verdicts.read_text()))
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    s = out["end_state_shows_what_both_telemetry_logs_lost"]
    d = out["dependence_each_run_would_need"]
    print(
        f"end state shows {s['lost_by_both_telemetry_logs']['share']} of what both logs lost "
        f"and {s['the_rest']['share']} of the rest; implied odds ratio defined in "
        f"{d['defined']} of {d['runs']} runs, quartiles {d['implied_odds_ratio']} -> {a.out}"
    )


if __name__ == "__main__":
    main()
