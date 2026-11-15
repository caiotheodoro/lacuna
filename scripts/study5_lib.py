"""Study 5: from a rig run's files to estimates, baselines, scores and verdicts.

`scripts/study5_analysis.py` and `scripts/study5_verdicts.py` import this module.
Nothing here starts a container or calls a model.

What an estimator is given. A run's witness files only: the per-plane tables
(`lacuna.rig.witnesses`), the same tables per session, and for the sequence
baseline the JSON-RPC ids on the telemetry records that were delivered. The
ledger reaches no estimator and no baseline (AGENTS.md rule 3). It is read to score
(`truth`) and to describe a run's loss (`lacuna.rig.witnesses.summarise`, which feeds
the gates).

The sequence baseline. JSON-RPC ids number the requests of one session on one
plane, at the source. A tool call's id is in a log or it is not, so the ids
between the first call and the last one seen that no record carries are requests
some pipeline lost. Its best fair case is granted: it knows a session's first
tool call has id `FIRST_CALL_ID` (a lost head is therefore a visible gap), it
reads ids from both telemetry logs of a plane together, and it turns missing
requests into actions with the share of seen tool calls that were successful
actions, pooled over the run. It cannot see a lost tail (`seq_gap_tail` adds the
expected one, as in the earlier studies), a session no telemetry record names,
or anything the end state alone shows.

The estimators are candidates until the registration names one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lacuna.core.tables import build
from lacuna.envs.base import Env
from lacuna.estimators import Estimate, baselines, loglinear, selection, stratified
from lacuna.rig import witnesses as rw

# `lacuna.rig.replay` initialises (request 0), lists the tools (1), then calls.
FIRST_CALL_ID = 2
Z = 1.959963984540054
BASELINES = ("seq_gap_tail", "union_count")
# The design's criteria (`docs/DESIGN-STUDY5.md`). The registration fixes them.
COVERAGE_FLOOR = 0.90  # P1: Wilson lower end of coverage at or above this
UNBOUNDED_MAX = 0.05  # P1: share of unbounded intervals below this
WIN_RATE_FLOOR = 0.5  # P2: exact lower bound on the win rate above this
EXERCISED_MIN = 0.02  # gate: mean share of actions in no log at or above this
TELEMETRY_LOSS_MIN = 0.01  # gate: some telemetry log loses at least this on average
FLOORS = ("union_count", "union_bound", "trust_trace")


# --- A run's data ---------------------------------------------------------------


@dataclass(frozen=True)
class Stream:
    """One session on one plane as the delivered telemetry shows it."""

    plane: str
    session: str
    ids: frozenset[int]  # JSON-RPC ids of the tool calls any record carries
    actions: int  # of those, successful action calls


def streams(loaded: dict, env: Env) -> list[Stream]:
    """`loaded` is `lacuna.rig.witnesses.load`'s result. Reads delivered records only."""
    seen: dict[tuple[str, str], dict[int, bool]] = {}

    def add(plane: str, c: dict) -> None:
        if not c["session"] or not isinstance(c["rpc_id"], int) or isinstance(c["rpc_id"], bool):
            return
        ids = seen.setdefault((plane, str(c["session"])), {})
        action = bool(c["ok"]) and c["tool"] in env.action_tools
        ids[c["rpc_id"]] = ids.get(c["rpc_id"], False) or action

    for plane, c in loaded["span_calls"]:
        add(plane, c)
    for name, calls in loaded["access_calls"].items():
        for c in calls:
            add(rw.LOG_PLANE[name], c)
    return [
        Stream(plane, session, frozenset(ids), sum(ids.values()))
        for (plane, session), ids in sorted(seen.items())
    ]


def plane_tables(obs: Sequence) -> dict[str, np.ndarray]:
    return {plane: table for plane, (_, table) in build(obs, rw.SPECS, rw.key).items()}


def session_tables(obs: Sequence) -> list[np.ndarray]:
    """One table per session and plane that any witness saw, in a fixed order."""
    by: dict[str, list] = {}
    for o in obs:
        by.setdefault(str(o.session), []).append(o)
    out = []
    for session in sorted(by):
        for _, table in sorted(plane_tables(by[session]).items()):
            if table.sum():
                out.append(table)
    return out


def truth(loaded: dict) -> dict[str, int]:
    """Actions per plane, from the ledger. For scoring only."""
    out: dict[str, int] = {p: 0 for p in sorted(rw.PLANES)}
    for e in loaded["ledger"]:
        out[str(e["principal"]).removeprefix("plane-")] += 1
    return out


# --- Estimators -------------------------------------------------------------------


def collapse_telemetry(table: np.ndarray) -> np.ndarray:
    """A three-witness table (spans, access, env) as two lists: any telemetry, env."""
    out = np.zeros(4, dtype=np.int64)
    for h in range(1, 8):
        out[(1 if h & 3 else 0) | (2 if h & 4 else 0)] += table[h]
    return out


def chapman_te(pooled: np.ndarray) -> Estimate:
    """Lincoln-Petersen between the telemetry, taken as one list, and the end state.

    The two telemetry logs lose records to the same outages, so their overlap says
    little about what both missed. Taken together they are one list whose misses are
    a matter of when; the end state's misses are a matter of what was overwritten.
    """
    return baselines.chapman(collapse_telemetry(pooled), 0, 1, name="chapman_te")


# Spans and access may depend on each other; the end state depends on neither.
TELEMETRY_PAIR: loglinear.Pairs = ((0, 1),)


def sa_hom(pooled: np.ndarray, per_session: Sequence[np.ndarray]) -> Estimate:
    """One model named in advance, its interval widened by how much sessions disagree.

    The model has the one interaction the rig's layout implies: the two telemetry logs
    lose records to the same outages. No model is selected, so a sparse table cannot
    tip the choice to a model whose missing cell is unstable. The profile cutoff is
    scaled by the between-session dispersion (`loglinear.homogeneity_dispersion`, D51),
    because an outage takes a block out of some sessions and nothing out of others.
    """
    phi = loglinear.homogeneity_dispersion(list(per_session)) if len(per_session) else 1.0
    return loglinear._single(pooled, TELEMETRY_PAIR, "ll_sa_hom", {"phi": phi}, phi=phi)


def seq_estimates(strs: Sequence[Stream]) -> dict[str, Estimate]:
    """`seq_gap` and `seq_gap_tail` on JSON-RPC ids, in actions."""
    live = [s for s in strs if s.ids]
    seen = sum(len(s.ids) for s in live)
    actions = sum(s.actions for s in live)
    share = actions / seen if seen else 0.0
    obs = [
        baselines.SeqObs(max(max(s.ids) - FIRST_CALL_ID + 1, len(s.ids)), len(s.ids)) for s in live
    ]

    def to_actions(requests: float) -> float:
        return actions + (requests - seen) * share

    gap = baselines.seq_gap(obs, seen)
    tail = baselines.seq_gap_tail(obs, seen)
    detail = {"requests_seen": seen, "action_share": round(share, 6), "streams": len(live)}
    return {
        "seq_gap": Estimate("seq_gap", actions, *(to_actions(x) for x in (gap.point,) * 3), detail),
        "seq_gap_tail": Estimate(
            "seq_gap_tail",
            actions,
            *(to_actions(x) for x in (tail.point, tail.lo, tail.hi)),
            detail | tail.detail,
        ),
    }


def _safe(name: str, n: int, fn: Callable[[], Estimate]) -> Estimate:
    """An estimator that raises has no answer: unbounded, with the reason kept."""
    try:
        e = fn()
    except (FloatingPointError, ValueError, ZeroDivisionError, np.linalg.LinAlgError) as err:
        return Estimate(
            name, n, math.inf, float(n), math.inf, {"error": f"{type(err).__name__}: {err}"}
        )
    return Estimate(name, e.n_observed, e.point, e.lo, e.hi, e.detail)


def estimates(
    per_plane: dict[str, np.ndarray],
    per_session: Sequence[np.ndarray],
    strs: Sequence[Stream],
    boot_seed: int = 0,
) -> dict[str, Estimate]:
    """Every candidate, floor and baseline for one run."""
    pooled = np.sum(list(per_plane.values()), axis=0).astype(np.int64)
    pooled[0] = 0
    n = int(pooled.sum())
    live = {p: (0, 1, 2) for p in per_plane}
    todo: dict[str, Callable[[], Estimate]] = {
        "strat_bic": lambda: stratified.strat_bic(per_plane, live=live),
        "ll_bic": lambda: loglinear.best_bic(pooled),
        "ll_average": lambda: loglinear.averaged(pooled),
        "ll_bic_boot": lambda: selection.bic_boot(pooled, seed=boot_seed),
        "ll_bic_hom": lambda: loglinear.bic_episodes(list(per_session)),
        "chapman_te": lambda: chapman_te(pooled),
        "ll_sa_hom": lambda: sa_hom(pooled, per_session),
        # The two telemetry logs alone, for the question the registration predicts fails.
        "chapman_sa": lambda: baselines.chapman(pooled, 0, 1, name="chapman_sa"),
        "union_count": lambda: baselines.union_count(pooled),
        "union_bound": lambda: baselines.union_bound(pooled),
        "trust_trace": lambda: baselines.trust_trace(pooled),
    }
    out = {name: _safe(name, n, fn) for name, fn in todo.items()}
    out |= seq_estimates(strs)
    return out


def _num(v: float) -> float | str:
    return v if math.isfinite(v) else "inf"


def score(e: Estimate, n_true: int) -> dict:
    return {
        "point": _num(round(e.point, 3) if math.isfinite(e.point) else e.point),
        "lo": _num(round(e.lo, 3) if math.isfinite(e.lo) else e.lo),
        "hi": _num(round(e.hi, 3) if math.isfinite(e.hi) else e.hi),
        "covers": bool(e.covers(n_true)),
        "bounded": bool(e.bounded and math.isfinite(e.point)),
        "abs_error": _num(round(abs(e.point - n_true), 3) if math.isfinite(e.point) else math.inf),
        "detail": {k: v for k, v in e.detail.items() if isinstance(v, (int, float, str))},
    }


def run_record(run_dir: Path, env: Env, boot_seed: int = 0) -> dict:
    """One run, scored. The caller decides whether a run that failed integrity is used."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = json.loads((run_dir / "events.json").read_text())
    loaded = rw.load(run_dir, env)
    per_plane = plane_tables(loaded["obs"])
    n_plane = truth(loaded)
    n_true = sum(n_plane.values())
    strs = streams(loaded, env)
    est = estimates(per_plane, session_tables(loaded["obs"]), strs, boot_seed)
    union = int(sum(t.sum() for t in per_plane.values()))
    summary = rw.summarise(run_dir, env)
    return {
        "run": run_dir.name,
        # What made the run, as its own manifest says.
        "commit": manifest.get("commit"),
        "dirty": manifest.get("dirty"),
        "timing": events.get("timing"),
        "condition": manifest["params"]["condition"],
        "stress_seed": manifest["params"]["stress_seed"],
        "schedule": manifest["schedule"],
        "sessions_sha256": manifest["sessions_sha256"],
        "integrity": events.get("integrity", {"ok": False, "problems": ["no integrity result"]}),
        "duration_s": events.get("duration_s"),
        "actions": n_true,
        "actions_by_plane": n_plane,
        "tables": {p: [int(x) for x in t] for p, t in per_plane.items()},
        "union_count": union,
        "union_missed_share": round(1 - union / n_true, 6) if n_true else None,
        "missed_share": {
            p: {w: v["witnesses"][w]["missed_share"] for w in (rw.SPANS, rw.ACCESS, rw.ENV)}
            for p, v in summary["planes"].items()
        },
        "estimates": {name: score(e, n_true) for name, e in est.items()},
    }


# --- Over runs ----------------------------------------------------------------------


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def exact_lower(k: int, n: int, alpha: float = 0.025) -> float:
    """Clopper-Pearson one-sided lower bound: the p at which P(X >= k | n, p) = alpha."""
    if k == 0 or n == 0:
        return 0.0

    def upper_tail(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return float(p >= 1.0)
        # In logs: comb(n, i) * p**i overflows a float past n of about 1000.
        logs = [
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p)
            for i in range(k, n + 1)
        ]  # fmt: skip
        return math.fsum(math.exp(x) for x in logs)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if upper_tail(mid) < alpha else (lo, mid)
    return lo


def _err(rec: dict, name: str) -> float:
    e = rec["estimates"][name]
    return math.inf if not e["bounded"] or e["abs_error"] == "inf" else float(e["abs_error"])


def coverage(records: Sequence[dict], name: str) -> dict:
    n = len(records)
    k = sum(r["estimates"][name]["covers"] for r in records)
    unbounded = sum(not r["estimates"][name]["bounded"] for r in records)
    lo, hi = wilson(k, n)
    # Both error figures are over the runs with a finite point, whatever the interval.
    signed = [
        float(r["estimates"][name]["point"]) - r["actions"]
        for r in records
        if r["estimates"][name]["point"] != "inf"
    ]
    errs = [abs(e) for e in signed]
    return {
        "runs": n,
        "covered": k,
        "coverage": round(k / n, 4) if n else None,
        "wilson_lo": round(lo, 4),
        "wilson_hi": round(hi, 4),
        "unbounded": unbounded,
        "unbounded_rate": round(unbounded / n, 4) if n else None,
        "runs_with_a_point": len(signed),
        "median_abs_error": round(float(np.median(errs)), 3) if errs else None,
        "mean_error": round(float(np.mean(signed)), 3) if signed else None,
    }


def paired(records: Sequence[dict], name: str, baseline: str) -> dict:
    """Sign test on absolute error. A tie or an unbounded estimate is a loss for `name`."""
    n = len(records)
    wins = sum(_err(r, name) < _err(r, baseline) for r in records)
    ties = sum(_err(r, name) == _err(r, baseline) for r in records)
    lower = exact_lower(wins, n)
    return {
        "runs": n,
        "wins": wins,
        "ties": ties,
        "win_rate": round(wins / n, 4) if n else None,
        "lower_97_5": round(lower, 4),
        "passes": bool(n and lower > WIN_RATE_FLOOR),
    }


def gates(records: Sequence[dict], all_runs: int, min_runs: int) -> dict:
    """The registration's gates on the runs that passed integrity."""
    shares = [r["union_missed_share"] for r in records if r["union_missed_share"] is not None]
    mean_missed = float(np.mean(shares)) if shares else 0.0
    # Mean share each telemetry witness missed, over runs and planes.
    tele = {}
    for w in (rw.SPANS, rw.ACCESS):
        vals = [v[w] for r in records for v in r["missed_share"].values() if v[w] is not None]
        tele[w] = round(float(np.mean(vals)), 4) if vals else 0.0
    return {
        "runs_made": all_runs,
        "runs_passing_integrity": len(records),
        "underpowered": len(records) < min_runs,
        "mean_union_missed_share": round(mean_missed, 4),
        "mean_telemetry_missed_share": tele,
        "not_exercised": mean_missed < EXERCISED_MIN
        or all(v < TELEMETRY_LOSS_MIN for v in tele.values()),
    }


def false_alarms(records: Sequence[dict], name: str) -> dict:
    """Runs where the interval lies wholly above the truth: actions claimed that were not."""
    above = sum(
        r["estimates"][name]["lo"] != "inf" and float(r["estimates"][name]["lo"]) > r["actions"]
        for r in records
    )
    return {"runs": len(records), "interval_above_truth": above}


def verdict(
    records: Sequence[dict],
    primary: str,
    all_runs: int,
    min_runs: int = 270,
    coverage_floor: float = COVERAGE_FLOOR,
    unbounded_max: float = UNBOUNDED_MAX,
) -> dict:
    """P1 and P2 for one condition's runs, with the gates applied."""
    g = gates(records, all_runs, min_runs)
    cov = coverage(records, primary) if records else None
    pairs = {b: paired(records, primary, b) for b in BASELINES} if records else {}
    blocked = "underpowered" if g["underpowered"] else "not exercised" if g["not_exercised"] else ""
    # On the numbers themselves: the rounded ones in `cov` are for reading.
    n = len(records)
    p1 = bool(
        cov
        and wilson(cov["covered"], n)[0] >= coverage_floor
        and cov["unbounded"] / n < unbounded_max
    )
    p2 = bool(pairs and all(p["passes"] for p in pairs.values()))
    return {
        "primary": primary,
        "gates": g,
        "coverage": cov,
        "paired": pairs,
        "P1": blocked or ("pass" if p1 else "fail"),
        "P2": blocked or ("pass" if p2 else "fail"),
    }
