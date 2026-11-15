#!/usr/bin/env python3
"""Apply study 5's criteria to an analysis artifact. Calls no model, starts no container.

    PYTHONPATH=src uv run python scripts/study5_verdicts.py \\
        --analysis artifacts/study5/confirm.json --primary ll_sa_hom \\
        --condition s1 --control s0 --boundary s3 \\
        --out artifacts/robustness/study5-verdicts.json

The criteria are the design's (`docs/DESIGN-STUDY5.md`), in `scripts/study5_lib.py`:

- gates: fewer runs passing integrity than `--min-runs` is "underpowered"; a mean
  share of actions in no log under 2%, or both telemetry logs losing under 1% on
  average, is "not exercised";
- P1: the Wilson lower end of the primary estimator's coverage over runs is at or
  above 0.90, and fewer than 5% of its intervals are unbounded;
- P2: against each of `seq_gap_tail` and `union_count`, a paired sign test on absolute
  error over runs, ties and unbounded estimates counted against the estimator; the
  exact one-sided 97.5% lower bound on the win rate must be above 0.5 for both;
- Q3: the same two criteria for the estimator that has no end state (`--no-env`);
- control: on the unstressed runs, how often the primary interval lies wholly above
  the truth;
- boundary (`--boundary`): a condition the registration expects the estimator to fall
  short on, reported with the same counts and no criterion.

The registration fixes `--primary`, `--condition`, `--control`, `--boundary` and
`--min-runs`. An
analysis made without a registration is refused unless `--pilot` is given, and then the
output's `registered` is null: what it holds is not a registered verdict.

`P1` and `P2` are pass or fail by the criteria. `P1_reading` and `P2_reading` split a
fail into "missed" (the interval excludes meeting the criterion) and "not shown".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402


def build(
    artifact: dict,
    primary: str,
    condition: str,
    control: str | None,
    no_env: str,
    min_runs: int,
    boundary: str | None = None,
) -> dict:
    runs = artifact["runs"]
    mine = [r for r in runs if r["condition"] == condition]
    made = artifact["by_condition"].get(condition, {}).get("gates", {}).get("runs_made", len(mine))
    for name in (primary, no_env):
        if mine and name not in mine[0]["estimates"]:
            raise KeyError(f"no estimate named {name!r} in the analysis")
    main_v = lib.verdict(mine, primary, made, min_runs)
    out = {
        "condition": condition,
        "min_runs": min_runs,
        "runs_set_aside": len(artifact.get("runs_set_aside", [])),
        "primary": main_v,
        "Q3_without_env": lib.verdict(mine, no_env, made, min_runs),
        "floors": {name: lib.coverage(mine, name) for name in lib.FLOORS} if mine else {},
        # No criterion: the bands were drawn after the pilot's misses were seen.
        "exploratory": lib.breakdown(mine, [primary]),
    }
    if boundary is not None:
        # A condition registered as one the estimator is expected to fall short on.
        # Counts only: nothing here is a criterion, so no bound and no pass or fail.
        if boundary in (condition, control):
            raise KeyError(f"the boundary condition {boundary!r} is the primary or the control")
        edge = [r for r in runs if r["condition"] == boundary]
        if not edge:
            raise KeyError(f"no run of the boundary condition {boundary!r} in the analysis")
        if primary not in edge[0]["estimates"]:
            raise KeyError(f"no estimate named {primary!r} in the analysis")
        counts = ("runs", "wins", "ties", "win_rate")
        out["boundary"] = {
            "condition": boundary,
            "runs": len(edge),
            "coverage": lib.coverage(edge, primary),
            "closer_than": {
                b: {k: v for k, v in lib.paired(edge, primary, b).items() if k in counts}
                for b in lib.BASELINES
            },
            "mean_union_missed_share": lib.gates(edge, len(edge), 0)["mean_union_missed_share"],
        }
    if control is not None:
        quiet = [r for r in runs if r["condition"] == control]
        out["control"] = {
            "condition": control,
            "coverage": lib.coverage(quiet, primary) if quiet else None,
            "false_alarms": lib.false_alarms(quiet, primary),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--primary", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--control", default=None)
    p.add_argument("--boundary", default=None, help="a condition reported with no criterion")
    p.add_argument("--no-env", default="chapman_sa")
    p.add_argument("--min-runs", type=int, default=270)
    p.add_argument(
        "--pilot", action="store_true",
        help="accept an analysis made without a registration; the output says so",
    )  # fmt: skip
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    raw = a.analysis.read_bytes()
    artifact = json.loads(raw)
    if not artifact.get("registered") and not a.pilot:
        sys.exit("this analysis was made without --registered: it is not the study's (--pilot)")
    try:
        out = build(artifact, a.primary, a.condition, a.control, a.no_env, a.min_runs, a.boundary)
    except KeyError as e:
        sys.exit(f"cannot give a verdict: {e.args[0]}")
    out["registered"] = artifact.get("registered")
    out["analysis"] = {"path": a.analysis.name, "sha256": hashlib.sha256(raw).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    q3 = out["Q3_without_env"]
    print(
        f"P1 {out['primary']['P1']}  P2 {out['primary']['P2']}  "
        f"Q3 {q3['P1']}/{q3['P2']}  -> {a.out}"
    )


if __name__ == "__main__":
    main()
