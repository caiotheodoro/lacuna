#!/usr/bin/env python3
"""Every figure the pilot's write-up quotes across rounds, in one artifact. Reads artifacts only.

    PYTHONPATH=src uv run python scripts/study5_pilot_summary.py \\
        --round artifacts/study5/pilot-round1.json --round artifacts/study5/pilot-round2.json \\
        --round artifacts/study5/pilot-round3.json --round artifacts/study5/pilot-round4.json \\
        --repeats artifacts/study5/pilot-repeats.json \\
        --repeats artifacts/study5/pilot-repeats-band.json --rig artifacts/study5/pilot-rig.json \\
        --proposed node-restart --after-round 1 --out artifacts/study5/pilot-summary.json

`--round` files are `scripts/study5_analysis.py` artifacts, in the order the rounds were
made. `--rig` is `scripts/rig_collect.py`'s. Nothing is estimated here: the runs are
already scored, and this counts, sums and splits what the round files hold, so that a
document quoting a total over rounds can name a file that holds it (AGENTS.md rule 2).

`--proposed` and `--after-round` pick out the runs of one condition made after a given
round: the runs of the condition the write-up proposes, made after the estimator it
proposes existed.

This is a description of a pilot. No criterion is applied and nothing here is a verdict.
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

CANDIDATES = ("ll_sa_hom", "chapman_te", "strat_bic", "ll_bic", "ll_bic_hom", "ll_bic_boot",
              "ll_average")  # fmt: skip
OTHERS = ("chapman_sa", "seq_gap", "seq_gap_tail", "union_count", "union_bound", "trust_trace")
SPAN_RELAY = "otelcol-gw"


def err(run: dict, name: str) -> float:
    """Signed error of a point, or None-like infinity when there is no point."""
    p = run["estimates"][name]["point"]
    return float("inf") if p == "inf" else float(p) - run["actions"]


def width(run: dict, name: str) -> float | None:
    e = run["estimates"][name]
    return None if "inf" in (e["lo"], e["hi"]) else float(e["hi"]) - float(e["lo"])


def outage_s(run: dict, service: str = SPAN_RELAY) -> float | None:
    """How long `service` was down, from the run's schedule; None if it was not."""
    at = {e["action"]: e["at_s"] for e in run["schedule"] if e["service"] == service}
    return round(at["start"] - at["kill"], 3) if {"kill", "start"} <= set(at) else None


def killed(run: dict) -> list[str]:
    """Services the schedule kills and never restarts."""
    last: dict[str, str] = {}
    for e in sorted(run["schedule"], key=lambda e: e["at_s"]):
        last[e["service"]] = e["action"]
    return sorted(s for s, a in last.items() if a == "kill")


def tally(runs: list[dict], name: str) -> dict:
    """One estimator over a set of runs: coverage, error, and the paired comparisons."""
    n = len(runs)
    covered = sum(r["estimates"][name]["covers"] for r in runs)
    errors = [e for e in (err(r, name) for r in runs) if e != float("inf")]
    widths = [w for w in (width(r, name) for r in runs) if w is not None]
    lo, hi = lib.wilson(covered, n)
    out = {
        "runs": n,
        "covered": covered,
        "wilson": [round(lo, 4), round(hi, 4)],
        "unbounded": sum(not r["estimates"][name]["bounded"] for r in runs),
        "median_abs_error": round(statistics.median(abs(e) for e in errors), 3) if errors else None,
        "mean_error": round(statistics.fmean(errors), 3) if errors else None,
        "sd_error": round(statistics.stdev(errors), 3) if len(errors) > 1 else None,
        "over": sum(e > 0 for e in errors),
        "under": sum(e < 0 for e in errors),
        "median_width": round(statistics.median(widths), 3) if widths else None,
    }
    if name not in lib.BASELINES:
        out["closer_than"] = {}
        for b in lib.BASELINES:
            pair = lib.paired(runs, name, b)
            out["closer_than"][b] = {k: pair[k] for k in ("wins", "ties", "lower_97_5")}
    return out


def block(runs: list[dict]) -> dict:
    """A set of runs described: what was lost, and every estimator's tally."""
    if not runs:
        return {"runs": 0}
    missed = [r["union_missed_share"] for r in runs]
    names = [n for n in (*CANDIDATES, *OTHERS) if n in runs[0]["estimates"]]
    return {
        "runs": len(runs),
        "actions": [min(r["actions"] for r in runs), max(r["actions"] for r in runs)],
        "duration_s": [min(r["duration_s"] for r in runs), max(r["duration_s"] for r in runs)],
        "in_no_log": {
            "mean": round(statistics.fmean(missed), 4),
            "sd": round(statistics.stdev(missed), 4) if len(missed) > 1 else None,
            "min": round(min(missed), 4),
            "max": round(max(missed), 4),
            "runs_under_exercised_min": sum(m < lib.EXERCISED_MIN for m in missed),
        },
        "missed_mean": {
            w: round(statistics.fmean(v[w] for r in runs for v in r["missed_share"].values()), 4)
            for w in ("spans", "access", "env")
        },
        "estimators": {name: tally(runs, name) for name in names},
        # The sequence baseline has no floor of its own: how often it is under the union.
        "seq_gap_tail_below_union": sum(
            float(r["estimates"]["seq_gap_tail"]["point"]) < r["union_count"] for r in runs
        ),
    }


def bands(rig_runs: list[dict]) -> dict:
    """Span batches given up per originator against the span relay's outage length.

    Killed originators and runs with no outage of the span relay are left out. One
    schedule made several times counts every time; `schedules` counts it once.
    """
    by: dict[int, list[tuple[float, str]]] = {}
    split_runs = []
    for r in rig_runs:
        length = outage_s(r)
        if length is None:
            continue
        dead = set(killed(r))
        counts = {a: n for a, n in r["span_batches_given_up"].items() if a not in dead}
        for n in counts.values():
            by.setdefault(n, []).append((length, json.dumps(r["schedule"], sort_keys=True)))
        if len(set(counts.values())) > 1:
            split_runs.append({"run": r["run"], "outage_s": length, "given_up": counts})
    return {
        "by_count": {
            str(n): {
                "outage_s": [min(x for x, _ in v), max(x for x, _ in v)],
                "originator_runs": len(v),
                "schedules": len({s for _, s in v}),
            }
            for n, v in sorted(by.items())
        },
        "originator_runs": sum(len(v) for v in by.values()),
        # Runs where the two originators saw one outage and gave up different counts.
        "originators_disagree": split_runs,
    }


def sdk_errors(rig_runs: list[dict]) -> dict:
    total: dict[str, int] = {}
    for r in rig_runs:
        for per in r["sdk_export_messages"].values():
            for kind, n in per.items():
                total[kind] = total.get(kind, 0) + n
    return total


def near_end(runs: list[dict], within_s: float) -> list[dict]:
    """Runs whose last restart came within `within_s` of the run's end, or after it."""
    out = []
    for r in runs:
        starts = [e["at_s"] for e in r["schedule"] if e["action"] == "start"]
        if starts and max(starts) > r["duration_s"] - within_s:
            row = {"run": r["run"], "last_restart_s": max(starts)}
            out.append(row | {"duration_s": r["duration_s"]})
    return out


def repeats(runs: list[dict], rig_runs: list[dict]) -> dict:
    rig = {r["run"]: r for r in rig_runs}
    cells = [max(abs(x - y) for a in runs for b in runs for p in a["tables"]
                 for x, y in zip(a["tables"][p], b["tables"][p], strict=True))]  # fmt: skip
    return {
        "runs": [r["run"] for r in runs],
        "one_schedule": len({json.dumps(r["schedule"], sort_keys=True) for r in runs}) == 1,
        "one_set_of_sessions": len({r["sessions_sha256"] for r in runs}) == 1,
        "span_relay_outage_s": outage_s(runs[0]),
        "actions": sorted({r["actions"] for r in runs}),
        "union_counts": [r["union_count"] for r in runs],
        "largest_cell_difference": cells[0],
        "batches_given_up": [
            rig[r["run"]]["span_batches_given_up"] for r in runs if r["run"] in rig
        ],
        # When the receiver wrote the last call whose span was lost, per plane.
        "last_lost_span_call_s": {
            p: [
                rig[r["run"]]["planes"][p]["lost_calls_written_at_s"]["spans"].get("last")
                for r in runs
                if r["run"] in rig
            ]
            for p in ("a", "b")
        },  # fmt: skip
    }


def tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X binomial(n, p)."""
    return 1.0 - math.fsum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k))


def thresholds(n: int) -> dict:
    """The fewest covered runs for P1 and the fewest wins for P2, of `n` runs, and how
    likely each is to be reached if runs were independent with a given true rate."""
    covered = next(k for k in range(n + 1) if lib.wilson(k, n)[0] >= lib.COVERAGE_FLOOR)
    wins = next(k for k in range(n + 1) if lib.exact_lower(k, n) > lib.WIN_RATE_FLOOR)
    return {
        "runs": n,
        "covered_for_P1": covered,
        "wins_for_P2": wins,
        "P1_reached_if_true_coverage": {
            str(p): round(tail(covered, n, p), 3) for p in (0.90, 0.92, 0.94, 0.95, 0.96, 0.98)
        },
        "P2_reached_if_true_win_rate": {
            str(p): round(tail(wins, n, p), 3) for p in (0.5, 0.55, 0.6, 0.7, 0.8)
        },
    }


def summarise(
    rounds: list[dict], reps: list[dict], rig: dict, proposed: str, after: int, short_s: float
) -> dict:
    runs = [r | {"round": i + 1} for i, d in enumerate(rounds) for r in d["runs"]]
    stressed = [r for r in runs if r["condition"] != "none"]
    conditions = sorted({(r["condition"], r["round"]) for r in runs})
    fresh = [r for r in stressed if r["condition"] == proposed and r["round"] > after]
    short = [r for r in fresh if outage_s(r) < short_s]
    lengths = sorted(outage_s(r) for r in fresh)
    no_kill = [r for r in runs if not killed(r)]
    env = [v["env"] for r in no_kill for v in r["missed_share"].values()]
    timed = [r for r in runs if r.get("timing")]
    lags = [t["max_lag_s"] for r in timed for t in r["timing"].values() if t]
    return {
        "rounds": [
            {
                "round": i + 1,
                "runs": len(d["runs"]),
                "set_aside": len(d["runs_set_aside"]),
                "commits": sorted({str(r["commit"])[:7] for r in d["runs"]}),
                "dirty": sorted({bool(r["dirty"]) for r in d["runs"]}),
            }
            for i, d in enumerate(rounds)
        ],
        "runs": len(runs),
        "by_condition_and_round": {
            f"{c} ({k})": block([r for r in runs if (r["condition"], r["round"]) == (c, k)])
            for c, k in conditions
        },
        "unstressed": block([r for r in runs if r["condition"] == "none"]),
        "stressed": block(stressed),
        "stressed_round_1": block([r for r in stressed if r["round"] == 1]),
        "proposed_after_round": {
            "condition": proposed,
            "after_round": after,
            "all": block(fresh),
            "short_s": short_s,
            "short": block(short),
            "long": block([r for r in fresh if outage_s(r) >= short_s]),
            # The split point is not pinned by the data: any value between these two
            # neighbouring outage lengths gives the same two groups.
            "same_split_between_s": [
                max((x for x in lengths if x < short_s), default=None),
                min((x for x in lengths if x >= short_s), default=None),
            ],
        },
        "short_outages_any_condition": block(
            [
                r
                for r in stressed
                if outage_s(r) is not None
                and outage_s(r) < short_s
                and not killed(r)
                and r["condition"] != "relays"
            ]
        ),  # fmt: skip
        "end_state_hides_share_no_kill": {"runs": len(no_kill), "min": min(env), "max": max(env)},
        "killed_originator": {
            "runs": [
                {
                    "run": r["run"],
                    "actions": r["actions"],
                    "killed": killed(r),
                    "actions_by_plane": r["actions_by_plane"],
                    "ll_sa_hom_error": round(err(r, "ll_sa_hom"), 3),
                }
                for r in stressed
                if any(s.startswith("agent-") for s in killed(r))
            ],
        },  # fmt: skip
        "batches": bands(rig["runs"]),
        "sdk_export_errors": sdk_errors(rig["runs"]),
        "rig_runs": len(rig["runs"]),
        "restart_within_3_s_of_end": near_end(runs, 3.0),
        # One entry per schedule that was made more than once.
        "repeats": [repeats(rep["runs"], rig["runs"]) for rep in reps],
        "timing": {
            "runs_with_a_report": len(timed),
            "max_lag_s": [min(lags), max(lags)] if lags else None,
            "containers_clock_minus_driver_s": {
                str(k): sorted(
                    r["containers_clock_minus_driver_s"] for r in timed if r["round"] == k
                )
                for k in sorted({r["round"] for r in timed})
            },
        },
        "thresholds": [thresholds(n) for n in (270, 300, 1000)],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--round", type=Path, action="append", required=True, dest="rounds")
    p.add_argument("--repeats", type=Path, action="append", required=True)
    p.add_argument("--rig", type=Path, required=True)
    p.add_argument("--proposed", required=True, help="the condition the write-up proposes")
    p.add_argument("--after-round", type=int, required=True)
    p.add_argument("--short-s", type=float, default=25.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    load = lambda f: json.loads(f.read_text())  # noqa: E731
    out = summarise([load(f) for f in a.rounds], [load(f) for f in a.repeats], load(a.rig),
                    a.proposed, a.after_round, a.short_s)  # fmt: skip
    out["inputs"] = [f.name for f in (*a.rounds, *a.repeats, a.rig)]
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    f = out["proposed_after_round"]["all"]["estimators"]["ll_sa_hom"]
    print(f"{out['runs']} runs; {a.proposed} after round {a.after_round}: ll_sa_hom covers "
          f"{f['covered']}/{f['runs']}; wrote {a.out}")  # fmt: skip


if __name__ == "__main__":
    main()
