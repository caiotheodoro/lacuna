"""Two more things computed on study 5 after its verdicts were read. Post hoc; D111, D112.

Neither was registered, neither has a criterion, and no verdict depends on either.
`docs/DECISIONS.md` D111 says what each is, and was committed before either was run.

A. The dependence between runs that replay shared sessions, estimated. A run of a pool
   replays 24 of its 72 sessions, so two runs of a pool share some. If a run's outcome is
   its pool's mean, plus a term for each session it replays, plus noise of its own, then
   the covariance of two runs' outcomes is tau^2 * k / 24, with k the sessions they
   share. The products of pairs of centred outcomes are regressed on k / 24 with an
   intercept for each pool; the slope is tau^2, and rho is the slope over the pooled
   variance within a pool. The design effect on a mean over runs is
   1 + rho * (the mean over runs of the sum of k / 24 over the other runs of its pool).
   Inference is by permuting outcomes among the runs of a pool. The model is an
   assumption. A pool's pairs are used to count shared sessions and for nothing else.

B. The runs set aside on integrity.
   1. Each run's start, lateness and clock offset, beside the times the machine that made
      the runs slept, woke and changed its power source, from its own power log
      (`artifacts/study5/host/pmset-2026-11-16.txt`). Times and counts of runs.
   2. The registered estimator, by the frozen `study5_lib.run_record`, on the `s1` runs set
      aside whose folder holds a full record. A run that broke the timing rule is not a
      run of the design: this describes what the estimator returned on them and is not a
      test.

    PYTHONPATH=src uv run python scripts/study5_posthoc3.py

It reads `artifacts/study5/confirm.json`, the pools, the power log and, for B, the run
folders under `artifacts/rig/runs`, which are an asset of release `v1.0.0rc1` and whose
hashes are in `artifacts/study5/runs-sha256.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402
import study5_runs  # noqa: E402
from study5_posthoc import wilson_lower  # noqa: E402

from lacuna.envs.registry import ENVS  # noqa: E402
from lacuna.rig.sessions import draw  # noqa: E402

CONFIRM = ROOT / "artifacts" / "study5" / "confirm.json"
POSTHOC = ROOT / "artifacts" / "study5" / "posthoc.json"
SEAL = ROOT / "artifacts" / "study5" / "runs-sha256.json"
HOST_LOG = ROOT / "artifacts" / "study5" / "host" / "pmset-2026-11-16.txt"
RUNS_DIR = ROOT / "artifacts" / "rig" / "runs"
OUT = ROOT / "artifacts" / "study5" / "posthoc3.json"

PRIMARY = "ll_sa_hom"
CONDITION = "s1"
PERMUTATIONS = 10_000
SEED = 20261123
Z_ONE_SIDED_95 = 1.6448536269514722
# D93's line between the two kinds of lateness: a call this late, or a clock this far off.
LONG_S = 20.0
# A sleep that ended this long before a run started still counts as beside it: the
# containers' clock stays behind for a while after the machine wakes.
BEFORE_S = 60.0
NOTE = (
    "Post hoc (docs/DECISIONS.md D111, D112): computed after the verdicts of study 5 were "
    "read, with no registered criterion. No verdict depends on it."
)
MODEL = (
    "Assumes a run's outcome is its pool's mean, plus a term for each session it replays, "
    "plus noise of its own. The upper limit is rho plus 1.645 standard deviations of the "
    "permuted slopes, an approximation that holds only where the dependence is small."
)
NOT_A_TEST = (
    "A run set aside broke the registration's timing rule and is not a run of the design. "
    "This is what the registered estimator returned on those runs, not a test, and it "
    "replaces no registered figure."
)


# --- A. dependence between runs ------------------------------------------------------


def shared_sessions(runs: list[dict], pools: dict[str, list]) -> np.ndarray:
    """k[r, s]: the sessions runs r and s both replay; 0 across pools and on the diagonal."""
    drawn = [
        {s.session for s in draw(pools[r["pool"]], study5_runs.SESSIONS, r["seed"])} for r in runs
    ]
    k = np.zeros((len(runs), len(runs)))
    for i, a in enumerate(runs):
        for j in range(i + 1, len(runs)):
            if runs[j]["pool"] == a["pool"]:
                k[i, j] = k[j, i] = len(drawn[i] & drawn[j])
    return k


def _centred(y: np.ndarray, pool: np.ndarray) -> np.ndarray:
    out = y.astype(float).copy()
    for p in np.unique(pool):
        out[pool == p] -= out[pool == p].mean()
    return out


def _design(k: np.ndarray, pool: np.ndarray, per_run: int) -> tuple[np.ndarray, float]:
    """The regressor k / per_run for the pairs of one pool, centred within the pool, as a
    matrix that is zero elsewhere, and its sum of squares over pairs."""
    x = np.zeros_like(k, dtype=float)
    for p in np.unique(pool):
        idx = np.flatnonzero(pool == p)
        block = k[np.ix_(idx, idx)] / per_run
        off = ~np.eye(len(idx), dtype=bool)
        block = np.where(off, block - block[off].mean(), 0.0)
        x[np.ix_(idx, idx)] = block
    return x, float((x**2).sum() / 2)


def slope(c: np.ndarray, x: np.ndarray, ss: float) -> float:
    """OLS slope of the products c_r c_s on the centred regressor, over pairs r < s."""
    return float(c @ x @ c / 2 / ss)


def dependence(
    y: np.ndarray,
    pool: np.ndarray,
    k: np.ndarray,
    per_run: int = study5_runs.SESSIONS,
    permutations: int = PERMUTATIONS,
    seed: int = SEED,
) -> dict:
    """rho, the design effect it implies on the mean of y, and a permutation test."""
    c = _centred(np.asarray(y, dtype=float), pool)
    x, ss = _design(k, pool, per_run)
    pools = np.unique(pool)
    variance = float((c**2).sum() / sum((pool == p).sum() - 1 for p in pools))
    tau2 = slope(c, x, ss)
    rho = tau2 / variance if variance > 0 else 0.0
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(pool == p) for p in pools]
    permuted = np.empty(permutations)
    for b in range(permutations):
        d = c.copy()
        for i in idx:
            d[i] = c[rng.permutation(i)]
        permuted[b] = slope(d, x, ss) / variance if variance > 0 else 0.0
    sd = float(permuted.std(ddof=1))
    weight = float((k / per_run).sum(axis=1).mean())
    upper = rho + Z_ONE_SIDED_95 * sd
    return {
        "runs": len(c),
        "variance_within_pool": round(variance, 6),
        "tau2": round(tau2, 6),
        "rho": round(rho, 4),
        "permutations": permutations,
        "seed": seed,
        "p_one_sided": round(float((1 + (permuted >= rho).sum()) / (permutations + 1)), 4),
        "sd_of_permuted_rho": round(sd, 4),
        "rho_upper_95": round(upper, 4),
        "mean_shared_weight": round(weight, 3),
        "design_effect": round(1 + max(rho, 0.0) * weight, 2),
        "design_effect_upper_95": round(1 + max(upper, 0.0) * weight, 2),
    }


def outcomes(runs: list[dict], name: str = PRIMARY) -> dict[str, list[float | None]]:
    """Per run: whether the interval missed; the point's signed error; the true count's
    place in the interval, 0 to 1 when it is inside (None where the interval has no width
    or no upper end)."""
    miss, error, place = [], [], []
    for r in runs:
        e = r["estimates"][name]
        miss.append(0.0 if e["covers"] else 1.0)
        error.append(float(e["error"]))
        lo, hi = e["lo"], e["hi"]
        wide = isinstance(hi, (int, float)) and math.isfinite(hi) and hi > lo
        place.append((r["actions"] - lo) / (hi - lo) if wide else None)
    return {"miss": miss, "error": error, "place": place}


def dependence_between_runs(confirm: dict, pools: dict[str, list], bearable: float) -> dict:
    plan = {r["run"]: r for r in study5_runs.runs()}
    runs = [r for r in confirm["runs"] if r["condition"] == CONDITION]
    rows = [plan[r["run"]] for r in runs]
    assert all(row["pool"] == r["pool"] for row, r in zip(rows, runs, strict=True))
    k = shared_sessions(rows, pools)
    pool = np.array([r["pool"] for r in rows])
    same = pool[:, None] == pool[None, :]
    np.fill_diagonal(same, False)
    by_outcome = {}
    for label, y in outcomes(runs).items():
        keep = np.array([v is not None for v in y])
        by_outcome[label] = dependence(
            np.array([v for v in y if v is not None]), pool[keep], k[np.ix_(keep, keep)]
        )
    covered = sum(r["estimates"][PRIMARY]["covers"] for r in runs)
    miss = by_outcome["miss"]
    # The Wilson lower end of P1's coverage at the design effects estimated for misses.
    bound = {
        key: round(wilson_lower(covered / len(runs), len(runs) / miss[key]), 4)
        for key in ("design_effect", "design_effect_upper_95")
    }
    return {
        "condition": CONDITION,
        "primary": PRIMARY,
        "model": MODEL,
        "sessions_a_run": study5_runs.SESSIONS,
        "sessions_a_pool": study5_runs.POOL_SESSIONS,
        "runs": len(runs),
        "pairs_of_one_pool": int(same.sum() // 2),
        "shared_sessions": {
            "mean": round(float(k[same].mean()), 3),
            "least": int(k[same].min()),
            "most": int(k[same].max()),
        },
        "by_outcome": by_outcome,
        "design_effect_the_bound_bears": bearable,
        "covered": covered,
        "bar": lib.COVERAGE_FLOOR,
        "wilson_lo_at_the_design_effect_for_misses": bound["design_effect"],
        "wilson_lo_at_its_upper_limit": bound["design_effect_upper_95"],
        "upper_limit_on_misses_above_what_the_bound_bears": by_outcome["miss"][
            "design_effect_upper_95"
        ]
        > bearable,
    }


# --- B1. the runs set aside, beside the machine's power log --------------------------

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([+-]\d{4}) (\S+)\s+(.*)$")
SECS = re.compile(r"(\d+) secs\s*$")
POWER = re.compile(r"Using (AC|BATT|Batt)\s*\(Charge:\s*(\d+)%\)")


def _utc(stamp: str, offset: str) -> datetime:
    sign = 1 if offset[0] == "+" else -1
    tz = timezone(sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[3:5])))
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).astimezone(UTC)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def power_log(text: str) -> dict:
    """Sleeps (start, end, reason), readings of the power source, battery warnings and
    the time external power was plugged in, from the lines of `pmset -g log`."""
    sleeps, source, warnings, plugged = [], [], [], []
    for raw in text.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        at = _utc(m.group(1), m.group(2)).timestamp()
        kind, rest = m.group(3), m.group(4)
        p = POWER.search(rest)
        if p:
            source.append(
                {"at": _iso(at), "source": "ac" if p.group(1) == "AC" else "battery",
                 "charge": int(p.group(2))}
            )  # fmt: skip
        if kind == "Sleep" and "Entering Sleep state" in rest:
            secs = SECS.search(rest)
            reason = re.search(r"due to '([^']+)'", rest)
            if secs:
                sleeps.append(
                    {"start": at, "end": at + int(secs.group(1)), "secs": int(secs.group(1)),
                     "reason": reason.group(1) if reason else None}
                )  # fmt: skip
        if kind == "BatteryHealth":
            cap = re.search(r"cap: (\d+)", rest)
            warnings.append({"at": _iso(at), "capacity": int(cap.group(1)) if cap else None})
        if "USB-C_plug" in rest:
            plugged.append(at)
    return {"sleeps": sleeps, "source": source, "battery_warnings": warnings, "plugged": plugged}


def run_times(runs_dir: Path) -> list[dict]:
    """Every planned run in the order they were made: its start, length, worst lateness
    and clock offset, from its own `events.json` (None where the folder has none)."""
    out = []
    for i, r in enumerate(study5_runs.runs(), start=1):
        row = {"order": i, "run": r["run"], "condition": r["condition"]}
        path = runs_dir / r["run"] / "events.json"
        events = json.loads(path.read_text()) if path.exists() else {}
        lags = [
            (t or {}).get("max_lag_s")
            for t in (events.get("timing") or {}).values()
            if (t or {}).get("max_lag_s") is not None
        ]
        row |= {
            "started_unix": events.get("started_unix"),
            "duration_s": events.get("duration_s"),
            "max_lag_s": max(lags) if lags else None,
            "clock_s": events.get("containers_clock_minus_driver_s"),
        }
        out.append(row)
    return out


def _kind(why: list[str], row: dict) -> str:
    """One kind per run set aside, the first that applies."""
    if any("driver stopped" in w for w in why):
        return "no record"
    if any("delivered no line" in w for w in why):
        return "no access line"
    lag, clock = row.get("max_lag_s") or 0.0, abs(row.get("clock_s") or 0.0)
    return "long" if max(lag, clock) >= LONG_S else "short"


def beside_the_power_log(times: list[dict], aside: dict[str, list[str]], log: dict,
                         log_begins: float) -> dict:  # fmt: skip
    """Counts of runs and times only: which runs set aside sat beside a sleep of the
    machine, and how the rest fall around the time external power was plugged in."""
    plugged = min(log["plugged"]) if log["plugged"] else None
    last_on_battery = max(
        (
            datetime.strptime(s["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            .timestamp()
            for s in log["source"]
            if s["source"] == "battery"
            and plugged is not None
            and datetime.strptime(s["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            .timestamp() < plugged - 600
        ),
        default=None,
    )  # fmt: skip
    rows = []
    for t in times:
        if t["run"] not in aside:
            continue
        start = t["started_unix"]
        kind = _kind(aside[t["run"]], t)
        asleep = 0.0
        just_before = False
        if start is not None:
            end = start + (t["duration_s"] or 0.0)
            for s in log["sleeps"]:
                asleep += max(0.0, min(end, s["end"]) - max(start, s["start"]))
                just_before |= 0.0 <= start - s["end"] <= BEFORE_S
        rows.append(
            {
                "order": t["order"],
                "run": t["run"],
                "condition": t["condition"],
                "kind": kind,
                "started": _iso(start) if start is not None else None,
                "max_lag_s": t["max_lag_s"],
                "clock_s": t["clock_s"],
                "in_the_log": start is not None and start >= log_begins,
                "asleep_s_during_the_run": round(asleep, 1),
                "a_sleep_ended_just_before": just_before,
            }
        )
    long_runs = [r for r in rows if r["kind"] == "long"]
    logged = [r for r in long_runs if r["in_the_log"]]
    beside = [
        r for r in logged if r["asleep_s_during_the_run"] > 0 or r["a_sleep_ended_just_before"]
    ]
    # Every sleep the log records while runs were being made, and whether a run set
    # aside sat beside it.
    made = [t for t in times if t["started_unix"] is not None]
    first, last = made[0]["started_unix"], made[-1]["started_unix"] + made[-1]["duration_s"]
    sleeps = [s for s in log["sleeps"] if first <= s["start"] <= last]

    def beside_a_run(s: dict) -> bool:
        for r in long_runs:
            t = next(x for x in times if x["run"] == r["run"])
            if t["started_unix"] is None:
                continue
            end = t["started_unix"] + (t["duration_s"] or 0.0)
            if s["start"] <= end and t["started_unix"] - s["end"] <= BEFORE_S:
                return True
        return False

    def window(lo: float | None, hi: float | None) -> dict:
        inside = [
            t
            for t in made
            if (lo is None or t["started_unix"] >= lo) and (hi is None or t["started_unix"] < hi)
        ]
        names = {t["run"] for t in inside}
        hit = [r for r in rows if r["run"] in names]
        by = {
            c: {
                "runs": sum(t["condition"] == c for t in inside),
                "set_aside_short": sum(r["condition"] == c and r["kind"] == "short" for r in hit),
            }
            for c in ("s0", "s1", "s2")
        }
        return {
            "from": _iso(lo) if lo is not None else None,
            "to": _iso(hi) if hi is not None else None,
            "runs": len(inside),
            "set_aside": len(hit),
            "set_aside_short": sum(r["kind"] == "short" for r in hit),
            "set_aside_long": sum(r["kind"] == "long" for r in hit),
            "by_condition": by,
        }

    out = {
        "note": "Times and counts of runs. No estimate is read here.",
        "long_means": f"a call {LONG_S:.0f} s or more late, or a clock {LONG_S:.0f} s or more off",
        "log_begins": _iso(log_begins),
        "runs_made_from": _iso(first),
        "runs_made_to": _iso(last),
        "set_aside": len(rows),
        "by_kind": {
            k: sum(r["kind"] == k for r in rows)
            for k in ("long", "short", "no record", "no access line")
        },  # fmt: skip
        "long": {
            "runs": len(long_runs),
            "before_the_log_begins": len(long_runs) - len(logged),
            "in_the_log": len(logged),
            "beside_a_sleep": len(beside),
        },
        "sleeps_while_runs_were_made": {
            "sleeps": len(sleeps),
            "beside_a_run_set_aside": sum(beside_a_run(s) for s in sleeps),
            "list": [
                {"start": _iso(s["start"]), "secs": s["secs"], "reason": s["reason"]}
                for s in sleeps
            ],
        },
        "power_readings": log["source"],
        "battery_warnings": log["battery_warnings"],
        "external_power_plugged_in": _iso(plugged) if plugged is not None else None,
        "last_reading_on_battery_before_it": _iso(last_on_battery)
        if last_on_battery is not None
        else None,
        "runs": rows,
    }
    if plugged is not None and last_on_battery is not None:
        out["windows"] = {
            "before_the_last_battery_reading": window(None, last_on_battery),
            "on_battery_until_external_power": window(last_on_battery, plugged),
            "after_external_power": window(plugged, None),
        }
    return out


# --- B2. the registered estimator on the runs set aside ------------------------------


def _folder_sha256(folder: Path) -> str:
    """One hash for a run folder, as `scripts/study5_runs.py --seal` makes it."""
    return study5_runs.tree_hash(folder, [f for f in folder.rglob("*") if f.is_file()])


def estimator_on_set_aside(confirm: dict, runs_dir: Path, seal: dict) -> dict:
    planned = confirm["planned_by_condition"][CONDITION]
    in_test = [r for r in confirm["runs"] if r["condition"] == CONDITION]
    covered = sum(r["estimates"][PRIMARY]["covers"] for r in in_test)
    aside = [r["run"] for r in confirm["runs_set_aside"] if r["condition"] == CONDITION]
    rows, no_record = [], []
    for name in sorted(aside):
        folder = runs_dir / name
        if not (folder / "summary.json").exists():
            no_record.append(name)
            continue
        if _folder_sha256(folder) != seal["runs"][name]:
            raise SystemExit(f"{name}: the folder is not the one that was sealed")
        manifest = json.loads((folder / "manifest.json").read_text())
        env = ENVS[manifest["params"].get("env", "backoffice")]
        rec = lib.run_record(folder, env, boot_seed=int(manifest["params"]["stress_seed"]))
        e = rec["estimates"][PRIMARY]
        rows.append(
            {
                "run": name,
                "actions": rec["actions"],
                "union_count": rec["union_count"],
                "point": e["point"],
                "lo": e["lo"],
                "hi": e["hi"],
                "bounded": e["bounded"],
                "covers": e["covers"],
                "error": e["error"],
            }
        )
    k = sum(r["covers"] for r in rows)
    errors = [abs(r["error"]) for r in rows if isinstance(r["error"], (int, float))]

    def figure(c: int, n: int) -> dict:
        lo, hi = lib.wilson(c, n)
        return {"covered": c, "runs": n, "wilson": [round(lo, 4), round(hi, 4)]}

    return {
        "reading": NOT_A_TEST,
        "estimator": PRIMARY,
        "set_aside": len(aside),
        "with_a_full_record": len(rows),
        "with_no_record": no_record,
        "covered": k,
        "unbounded": sum(not r["bounded"] for r in rows),
        "median_abs_error": round(statistics.median(errors), 3) if errors else None,
        "runs": rows,
        "registered": {
            "in_the_test": figure(covered, len(in_test)),
            "not_in_the_test_as_uncovered": figure(covered, planned),
        },
        "with_the_runs_set_aside_as_scored": {
            "runs_with_a_record": figure(covered + k, len(in_test) + len(rows)),
            "no_record_as_uncovered": figure(covered + k, planned),
            "bar": lib.COVERAGE_FLOOR,
        },
    }


# --- main ----------------------------------------------------------------------------


def summarise(confirm: dict, pools: dict[str, list], posthoc: dict, host_log: str,
              runs_dir: Path, seal: dict, log_begins: float) -> dict:  # fmt: skip
    aside = {r["run"]: r["why"] for r in confirm["runs_set_aside"]}
    bearable = posthoc["dependence_between_runs"]["largest_design_effect_at_or_above_the_bar"]
    return {
        "note": NOTE,
        "dependence_between_runs": dependence_between_runs(confirm, pools, bearable),
        "runs_set_aside": {
            "beside_the_power_log": beside_the_power_log(
                run_times(runs_dir), aside, power_log(host_log), log_begins
            ),
            "estimator": estimator_on_set_aside(confirm, runs_dir, seal),
        },
        "host_log": {
            "file": str(HOST_LOG.relative_to(ROOT)),
            "sha256": hashlib.sha256(host_log.encode()).hexdigest(),
        },
    }


# The retained log began here when it was read (the file's own header says so).
LOG_BEGINS = datetime(2026, 11, 16, 5, 48, 20, tzinfo=UTC).timestamp()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--analysis", type=Path, default=CONFIRM)
    p.add_argument("--runs", type=Path, default=RUNS_DIR)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    if not args.runs.exists():
        sys.exit(f"no run folders at {args.runs}: they are an asset of release v1.0.0rc1")
    made = summarise(
        json.loads(args.analysis.read_text()),
        study5_runs.load_pools(study5_runs.POOLS_DIR),
        json.loads(POSTHOC.read_text()),
        HOST_LOG.read_text(),
        args.runs,
        json.loads(SEAL.read_text()),
        LOG_BEGINS,
    )
    args.out.write_text(json.dumps(made, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
