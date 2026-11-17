"""Two descriptions of study 5 made after its verdicts were read. Post hoc; D98.

Neither was registered, neither has a criterion, and both were chosen knowing the
results. They read `artifacts/study5/confirm.json` and nothing else.

1. How much dependence between runs the pass can bear. The registered bound treats runs
   as independent, and they replay sessions drawn from three pools of 72. The dependence
   is not estimated here: the runs give no way to. What is computed is the Wilson lower
   end of the same share at an effective number of runs n / d, for a range of design
   effects d, and the largest d at which it is still at or above the bar.

2. A baseline that reads the end state. `seq_gap_tail` counts gaps in request ids on
   delivered telemetry and is never shown the end state. `max(seq_gap_tail, union_count)`
   is: both are counts of actions known to have happened or been numbered, and the union
   includes what only the end state shows. The registered pairing rule is applied to it
   (a tie counts against the estimate).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402

CONFIRM = ROOT / "artifacts" / "study5" / "confirm.json"
OUT = ROOT / "artifacts" / "study5" / "posthoc.json"
DESIGN_EFFECTS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
NOTE = (
    "Post hoc (docs/DECISIONS.md D98): computed after the verdicts of study 5 were read, "
    "with no registered criterion. No verdict depends on it."
)


def wilson_lower(share: float, n: float, z: float = lib.Z) -> float:
    """The Wilson lower end for a share over `n` runs, where `n` need not be whole."""
    if n <= 0:
        return 0.0
    centre = share + z * z / (2 * n)
    spread = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n)


def design_effect(covered: int, runs: int, floor: float = lib.COVERAGE_FLOOR) -> dict:
    """The lower end at n / d for each d, and the largest d (to 0.01) that keeps the bar."""
    share = covered / runs
    rows = [
        {
            "design_effect": d,
            "effective_runs": round(runs / d, 1),
            "wilson_lo": round(wilson_lower(share, runs / d), 4),
        }
        for d in DESIGN_EFFECTS
    ]
    largest = None
    d = 1.0
    while d <= runs and wilson_lower(share, runs / d) >= floor:
        largest = round(d, 2)
        d += 0.01
    return {
        "covered": covered,
        "runs": runs,
        "bar": floor,
        "by_design_effect": rows,
        "largest_design_effect_at_or_above_the_bar": largest,
        "not_estimated": "the design effect itself; the runs give no way to estimate it",
    }


def gap_with_end_state(runs: list[dict], primary: str) -> dict:
    """The estimate against max(seq_gap_tail, union_count), by the registered pairing rule."""
    wins = ties = exact = 0
    base_err, mine_err = [], []
    for r in runs:
        base = max(r["estimates"]["seq_gap_tail"]["point"], r["union_count"])
        b = abs(base - r["actions"])
        e = r["estimates"][primary]
        m = e["abs_error"] if e["bounded"] else math.inf
        base_err.append(b)
        mine_err.append(m)
        exact += b == 0
        wins += m < b
        ties += m == b
    n = len(runs)
    return {
        "baseline": "max(seq_gap_tail, union_count)",
        "runs": n,
        "wins": wins,
        "ties": ties,
        "win_rate": round(wins / n, 4),
        "lower_97_5": round(lib.exact_lower(wins, n), 4),
        "upper_97_5": round(lib.exact_upper(wins, n), 4),
        "median_abs_error": {
            primary: round(statistics.median(mine_err), 3),
            "baseline": round(statistics.median(base_err), 3),
        },
        "runs_where_the_baseline_is_the_count": exact,
    }


def summarise(confirm: dict, condition: str = "s1", primary: str = "ll_sa_hom") -> dict:
    runs = [r for r in confirm["runs"] if r["condition"] == condition]
    cov = confirm["by_condition"][condition]["estimators"][primary]
    return {
        "note": NOTE,
        "condition": condition,
        "primary": primary,
        "dependence_between_runs": design_effect(cov["covered"], cov["runs"]),
        "gap_baseline_that_reads_the_end_state": gap_with_end_state(runs, primary),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--analysis", type=Path, default=CONFIRM)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    out = summarise(json.loads(a.analysis.read_text()))
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    d, g = out["dependence_between_runs"], out["gap_baseline_that_reads_the_end_state"]
    print(
        f"bar kept up to a design effect of {d['largest_design_effect_at_or_above_the_bar']}; "
        f"against {g['baseline']}: {g['wins']} wins, {g['ties']} ties of {g['runs']} "
        f"-> {a.out}"
    )


if __name__ == "__main__":
    main()
