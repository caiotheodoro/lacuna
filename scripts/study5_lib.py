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

from lacuna.core.records import WitnessSpec
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


# The same three witnesses with no id trusted: every record attaches by what the
# action was. The design promises this table beside the id-linked one (AGENTS.md
# rule 4), so the cost of not trusting the client library's ids can be read off.
SPECS_BY_SIGNATURE = tuple(WitnessSpec(s.name, s.sees, by_id=False) for s in rw.SPECS)


def plane_tables(obs: Sequence, by_id: bool = True) -> dict[str, np.ndarray]:
    specs = rw.SPECS if by_id else SPECS_BY_SIGNATURE
    return {plane: table for plane, (_, table) in build(obs, specs, rw.key).items()}


def session_tables(obs: Sequence, by_id: bool = True) -> list[np.ndarray]:
    """One table per session and plane that any witness saw, in a fixed order.

    The two planes of a session are two rows. They share the session's outage, so the
    rows are not independent; the dispersion computed from them says so only in part.
    """
    by: dict[str, list] = {}
    for o in obs:
        by.setdefault(str(o.session), []).append(o)
    out = []
    for session in sorted(by):
        for _, table in sorted(plane_tables(by[session], by_id).items()):
            if table.sum():
                out.append(table)
    return out


def _written_actions(loaded: dict, env: Env):
    """Each action a receiver wrote to its own log: the call, and whether the end state shows it.

    An action counts as shown if the end state has a row for what it was; two like
    actions with one row both count.
    """
    shown = {rw.key(o) for o in loaded["obs"] if o.witness == rw.ENV}
    for calls in loaded["written_calls"].values():
        for c in calls:
            if not c["ok"] or c["tool"] not in env.action_tools or not isinstance(c["args"], dict):
                continue
            norm = env.normalise(c["tool"], c["args"])
            if norm is not None:
                yield c, (c["session"], *norm) in shown


def lost_by_both(loaded: dict, env: Env) -> dict[str, int]:
    """The actions neither telemetry log delivered, and how many the end state shows.

    Description, for the breakdowns the registration labels exploratory. It reads the
    receivers' own complete logs, which are not a witness, so nothing here reaches an
    estimate. `rest` and `rest_shown` are the same two counts for every other action:
    an estimate that takes the end state's view of the delivered actions to hold for
    the lost ones is right when the two shares agree.
    """
    spans = _links(c for _, c in loaded["span_calls"])
    access = _links(c for calls in loaded["access_calls"].values() for c in calls)
    out = dict.fromkeys(LOST_KEYS, 0)
    for c, seen in _written_actions(loaded, env):
        link = (c["trace_id"], c["rpc_id"])
        dark = link not in spans and link not in access
        out["actions" if dark else "rest"] += 1
        out["shown_by_end_state" if dark else "rest_shown"] += seen
    return out


LOST_KEYS = ("actions", "shown_by_end_state", "rest", "rest_shown")
POSITION_PARTS = 5


def shown_by_position(loaded: dict, env: Env, parts: int = POSITION_PARTS) -> list[list[int]]:
    """Per part of a session, first to last: [actions, actions the end state shows].

    An action's place is its rank among its session's actions by the time its receiver
    wrote it, over both planes. Description, from the receivers' own logs, like
    `lost_by_both`.
    """
    sessions: dict[str, list[tuple[float, bool]]] = {}
    for c, seen in _written_actions(loaded, env):
        sessions.setdefault(c["session"], []).append((c.get("ts") or 0.0, seen))
    out = [[0, 0] for _ in range(parts)]
    for rows in sessions.values():
        rows.sort(key=lambda row: row[0])
        for i, (_, seen) in enumerate(rows):
            part = out[i * parts // len(rows)]
            part[0] += 1
            part[1] += seen
    return out


def _links(calls) -> set[tuple]:
    return {(c["trace_id"], c["rpc_id"]) for c in calls}


def outage_s(schedule: Sequence[dict], service: str = "otelcol-gw") -> float | None:
    """How long `service` was down by a run's schedule; None if it was not, or for good."""
    at = {e["action"]: e["at_s"] for e in schedule if e["service"] == service}
    return round(at["start"] - at["kill"], 3) if {"kill", "start"} <= set(at) else None


# Bands for the exploratory breakdowns, fixed here. Seconds the span relay was down, and
# actions lost by both telemetry logs. They were drawn after the pilots' three misses
# were seen (outages of 17.9, 19.3 and 20.1 s; 11, 4 and 5 actions lost by both logs),
# which is why a reading by band is exploratory.
OUTAGE_BANDS = ((0.0, 20.0), (20.0, 28.0), (28.0, math.inf))
LOST_BANDS = ((0, 12), (12, math.inf))


def band(value: float | None, bands: Sequence[tuple[float, float]]) -> str | None:
    """The band `value` is in, as "lo to hi"; a band holds its lower end and not its upper."""
    if value is None:
        return None
    for lo, hi in bands:
        if lo <= value < hi:
            return f"{lo:g} to {hi:g}" if math.isfinite(hi) else f"{lo:g} or more"
    return None


def breakdown(records: Sequence[dict], names: Sequence[str]) -> dict:
    """Coverage of each named estimate by outage length and by actions lost by both logs.

    Exploratory: the bands were drawn after the pilot's misses were seen.
    """
    out: dict[str, dict] = {"by_outage_s": {}, "by_lost_by_both": {}}
    out["not_covered"] = {
        name: outside(records, name)
        for name in names
        if name not in FLOORS and name not in BASELINES and not name.startswith("seq_gap")
    }
    for key, value, bands in (
        ("by_outage_s", lambda r: r.get("span_relay_outage_s"), OUTAGE_BANDS),
        ("by_lost_by_both", lambda r: (r.get("lost_by_both") or {}).get("actions"), LOST_BANDS),
    ):
        groups: dict[str, list[dict]] = {}
        for r in records:
            label = band(value(r), bands)
            if label is not None:
                groups.setdefault(label, []).append(r)
        for lo, _ in bands:
            label = band(lo, bands)
            rows = groups.get(label, [])
            lost = [r["lost_by_both"] for r in rows if r.get("lost_by_both")]
            out[key][label] = {
                "runs": len(rows),
                "covered": {
                    name: sum(r["estimates"][name]["covers"] for r in rows) for name in names
                },
                # Pooled over the band's runs: the lost actions and the rest, and how
                # many of each the end state shows.
                "lost_by_both": {k: sum(x[k] for x in lost) for k in LOST_KEYS},
            }
    return out


def outside(records: Sequence[dict], name: str) -> list[dict]:
    """The runs whose interval for `name` misses the truth, and by how many actions."""
    out = []
    for r in records:
        e = r["estimates"][name]
        if e["covers"]:
            continue
        lo, hi, n = float(e["lo"]), float(e["hi"]), r["actions"]
        by = lo - n if n < lo else n - hi
        out.append({"run": r.get("run"), "actions": n, "lo": e["lo"], "hi": e["hi"],
                    "outside_by": round(by, 3)})  # fmt: skip
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


def sa_hom(
    pooled: np.ndarray, per_session: Sequence[np.ndarray], name: str = "ll_sa_hom"
) -> Estimate:
    """One model named in advance, its interval widened by how much sessions disagree.

    The model has the one interaction the rig's layout implies: the two telemetry logs
    lose records to the same outages. No model is selected, so a sparse table cannot
    tip the choice to a model whose missing cell is unstable. The profile cutoff is
    scaled by the between-session dispersion (`loglinear.homogeneity_dispersion`, D51),
    because an outage takes a block out of some sessions and nothing out of others.
    """
    phi = loglinear.homogeneity_dispersion(list(per_session)) if len(per_session) else 1.0
    n = int(pooled[1:].sum())
    # The model's missing cell has a closed form: what env alone saw, times the odds
    # that an action telemetry saw is not in the end state. The iterative fit returns
    # NaN on a table with no loss at all (two non-zero cells); this does not.
    env_only = int(pooled[4])
    tele_only = int(pooled[1] + pooled[2] + pooled[3])
    both = int(pooled[5] + pooled[6] + pooled[7])
    detail = {"pairs": [list(p) for p in TELEMETRY_PAIR], "phi": phi}
    unbounded = Estimate("ll_sa_hom", n, math.inf, float(n), math.inf, detail | {"unbounded": 1})
    if both == 0:
        # No action is in both lists (or there are no actions): nothing ties the end
        # state to the telemetry, and the missing cell is 0/0 or infinite.
        return unbounded
    z0 = env_only * tele_only / both
    detail["z0"] = z0
    if n + z0 > loglinear.CAP_RATIO * n:
        # Past the cap every log-linear estimate here is reported as "no idea".
        return unbounded
    prof = loglinear.Profile(pooled, TELEMETRY_PAIR)
    try:
        mle, mle_ll = prof.maximise(n + z0)
        lo, hi = prof.interval(mle, mle_ll, chi2=loglinear.CHI2_1_95 * phi)
    except FloatingPointError:  # ConvergenceError is one
        return Estimate("ll_sa_hom", n, math.inf, float(n), math.inf, detail | {"nonconverged": 1})
    return Estimate("ll_sa_hom", n, n + z0, lo, hi, detail)


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
    by_signature: tuple[dict[str, np.ndarray], Sequence[np.ndarray]] | None = None,
) -> dict[str, Estimate]:
    """Every candidate, floor and baseline for one run.

    `by_signature` is the per-plane and per-session tables built with no id trusted;
    given, the primary candidate is also computed on them, as `ll_sa_hom_sig`.
    """
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
    if by_signature is not None:
        sig = np.sum(list(by_signature[0].values()), axis=0).astype(np.int64)
        sig[0] = 0
        todo["ll_sa_hom_sig"] = lambda: sa_hom(sig, by_signature[1], "ll_sa_hom_sig")
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
        # Unrounded, for comparing two estimates of one run. None when there is no point.
        "error": float(e.point - n_true) if math.isfinite(e.point) else None,
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
    by_signature = (plane_tables(loaded["obs"], False), session_tables(loaded["obs"], False))
    est = estimates(per_plane, session_tables(loaded["obs"]), strs, boot_seed, by_signature)
    pools = [Path(str(p.get("path"))).parent.name for p in manifest.get("pools") or []]
    union = int(sum(t.sum() for t in per_plane.values()))
    summary = rw.summarise(run_dir, env)
    return {
        "run": run_dir.name,
        # What made the run, as its own manifest says.
        "commit": manifest.get("commit"),
        "dirty": manifest.get("dirty"),
        "timing": events.get("timing"),
        # The pool the run drew its sessions from, if it drew from exactly one.
        "pool": pools[0] if len(pools) == 1 else None,
        # Whether a relay was still down when the last call was due (None: not recorded).
        "outage_past_last_call": bool(manifest["restarts_after_planned_end"])
        if "restarts_after_planned_end" in manifest
        else None,
        "containers_clock_minus_driver_s": events.get("containers_clock_minus_driver_s"),
        "condition": manifest["params"]["condition"],
        "stress_seed": manifest["params"]["stress_seed"],
        "schedule": manifest["schedule"],
        "span_relay_outage_s": outage_s(manifest["schedule"]),
        # Described from the receivers' complete logs; no estimate reads it.
        "lost_by_both": lost_by_both(loaded, env),
        "shown_by_position": shown_by_position(loaded, env),
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


def exact_upper(k: int, n: int, alpha: float = 0.025) -> float:
    """Clopper-Pearson one-sided upper bound on a proportion k/n."""
    return 1.0 - exact_lower(n - k, n, alpha)


def _err(rec: dict, name: str) -> float:
    e = rec["estimates"][name]
    return math.inf if not e["bounded"] or e["error"] is None else abs(float(e["error"]))


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
        # Coverage with the unbounded intervals left out, which cover by having no end.
        "bounded": n - unbounded,
        "covered_bounded": sum(
            r["estimates"][name]["covers"] and r["estimates"][name]["bounded"] for r in records
        ),
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
        # Runs where every action is in some log: there the union is exact and no
        # estimate can be strictly closer.
        "runs_with_nothing_in_no_log": sum(s == 0 for s in shares),
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
    # Underpowered is said first: with too few runs nothing else is read.
    blocked = "underpowered" if g["underpowered"] else "not exercised" if g["not_exercised"] else ""
    # On the numbers themselves: the rounded ones in `cov` are for reading.
    n = len(records)
    p1 = bool(
        cov
        and wilson(cov["covered"], n)[0] >= coverage_floor
        and cov["unbounded"] / n < unbounded_max
    )
    p2 = bool(pairs and all(p["passes"] for p in pairs.values()))
    # A criterion that is not met is "missed" only when the interval excludes meeting
    # it: coverage's upper end under the floor, or a win rate whose upper bound is
    # under a half against some baseline. Otherwise it is "not shown".
    p1_reading = "pass" if p1 else "not shown"
    if cov and (wilson(cov["covered"], n)[1] < coverage_floor
                or exact_lower(cov["unbounded"], n) >= unbounded_max):  # fmt: skip
        p1_reading = "missed"
    p2_reading = "pass" if p2 else "not shown"
    if any(exact_upper(p["wins"], p["runs"]) < WIN_RATE_FLOOR for p in pairs.values()):
        p2_reading = "missed"
    return {
        "primary": primary,
        "gates": g,
        "coverage": cov,
        "paired": pairs,
        "P1": blocked or ("pass" if p1 else "fail"),
        "P2": blocked or ("pass" if p2 else "fail"),
        # How to read a "fail": the criterion was shown to be missed, or merely not met.
        "P1_reading": blocked or p1_reading,
        "P2_reading": blocked or p2_reading,
    }
