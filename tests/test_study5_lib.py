"""scripts/study5_lib.py: streams, the sequence baseline, candidates, scores, verdicts."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402

from lacuna.envs.registry import BACKOFFICE  # noqa: E402
from lacuna.estimators import Estimate  # noqa: E402
from lacuna.rig import witnesses as rw  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "rig" / "run"


def call(session: str, rpc_id, tool: str = "cancel_order", ok: bool = True) -> dict:
    base = {"tool": tool, "args": {}, "ok": ok, "session": session}
    return base | {"trace_id": "t", "rpc_id": rpc_id, "ts": 0.0}


def loaded(spans: list[tuple[str, dict]], gateway: list[dict] = (), front: list[dict] = ()) -> dict:
    return {
        "span_calls": spans,
        "access_calls": {"gateway.jsonl": list(gateway), "front-b.jsonl": list(front)},
    }


# Streams -------------------------------------------------------------------------


def test_a_stream_is_the_ids_either_telemetry_log_of_a_plane_carries():
    d = loaded(
        [("a", call("s1", 2)), ("a", call("s1", 3, "get_order")), ("b", call("s1", 2))],
        gateway=[call("s1", 3, "get_order"), call("s1", 5)],
        front=[call("s2", 4, ok=False)],
    )
    got = {(s.plane, s.session): s for s in lib.streams(d, BACKOFFICE)}
    assert set(got) == {("a", "s1"), ("b", "s1"), ("b", "s2")}
    assert got[("a", "s1")].ids == {2, 3, 5} and got[("a", "s1")].actions == 2
    assert got[("b", "s1")].ids == {2} and got[("b", "s1")].actions == 1
    # A failed call has an id like any other request and is not an action.
    assert got[("b", "s2")].ids == {4} and got[("b", "s2")].actions == 0


def test_a_call_one_log_calls_failed_and_the_other_a_success_is_an_action_once():
    # A span that ended before the reply came says "not ok"; the receiver knows better.
    d = loaded([("a", call("s1", 2, ok=False))], gateway=[call("s1", 2, ok=True)])
    (s,) = lib.streams(d, BACKOFFICE)
    assert s.ids == {2} and s.actions == 1


def test_records_without_a_session_or_a_whole_number_id_are_in_no_stream():
    d = loaded([("a", call("", 2)), ("a", call("s1", None)), ("a", call("s1", "7")),
                ("a", call("s1", True))])  # fmt: skip
    assert lib.streams(d, BACKOFFICE) == []


# The sequence baseline ---------------------------------------------------------


def S(ids, actions=None, plane="a", session="s"):
    ids = frozenset(ids)
    return lib.Stream(plane, session, ids, len(ids) if actions is None else actions)


def test_nothing_lost_is_counted_exactly_by_both_sequence_estimates():
    est = lib.seq_estimates([S(range(2, 12)), S(range(2, 8), session="t")])
    for name in ("seq_gap", "seq_gap_tail"):
        assert est[name].point == 16 and est[name].lo == 16
    assert est["seq_gap_tail"].hi == 16


def test_an_interior_gap_and_a_lost_head_are_seen_and_a_lost_tail_is_not():
    whole = list(range(2, 22))  # 20 calls, all actions
    interior = lib.seq_estimates([S([i for i in whole if not 8 <= i <= 12])])
    assert interior["seq_gap"].point == 20
    head = lib.seq_estimates([S([i for i in whole if i >= 9])])
    assert head["seq_gap"].point == 20
    tail = lib.seq_estimates([S([i for i in whole if i <= 14])])
    assert tail["seq_gap"].point == 13  # the seven after id 14 left no gap
    # With no gap anywhere the tail term has no loss rate to work from either.
    assert tail["seq_gap_tail"].point == 13


def test_the_tail_term_adds_the_expected_tail_from_the_gap_rate():
    strs = [S([2, 3, 5, 6, 8, 9, 11], session=f"s{i}") for i in range(10)]
    est = lib.seq_estimates(strs)
    assert est["seq_gap"].point == 100  # ten slots a stream, all actions
    p = 3 / 9  # three gaps in the nine slots before each last survivor
    assert est["seq_gap_tail"].detail["loss_rate"] == pytest.approx(p)
    assert est["seq_gap_tail"].point == pytest.approx(100 + 10 * p / (1 - p))
    assert est["seq_gap_tail"].lo >= 100 and est["seq_gap_tail"].hi > est["seq_gap_tail"].point


def test_missing_requests_become_actions_at_the_share_of_seen_calls_that_were_actions():
    # 8 of 10 slots seen, 4 of the 8 are actions: each missing request is half an action.
    est = lib.seq_estimates([S([2, 3, 4, 5, 8, 9, 10, 11], actions=4)])
    assert est["seq_gap"].point == pytest.approx(4 + 2 * 0.5)
    assert est["seq_gap"].detail["action_share"] == 0.5
    assert est["seq_gap"].n_observed == 4


def test_a_session_no_telemetry_names_is_invisible_to_the_baseline():
    assert lib.seq_estimates([])["seq_gap_tail"].point == 0
    one = lib.seq_estimates([S(range(2, 12))])
    two = lib.seq_estimates([S(range(2, 12)), S([], session="gone")])
    assert one["seq_gap"].point == two["seq_gap"].point == 10


# Candidates ------------------------------------------------------------------------


def T(**cells: int) -> np.ndarray:
    """A (spans, access, env) table from cells named by who saw it, e.g. sae=50, e=3."""
    t = np.zeros(8, dtype=np.int64)
    for name, n in cells.items():
        t[sum({"s": 1, "a": 2, "e": 4}[c] for c in name)] = n
    return t


def test_telemetry_is_collapsed_to_one_list_against_the_end_state():
    t = T(s=1, a=2, sa=4, e=8, se=16, ae=32, sae=64)
    assert list(lib.collapse_telemetry(t)) == [0, 7, 8, 112]
    e = lib.chapman_te(T(sa=40, sae=60, e=6))
    # n_T = 100, n_E = 66, both = 60: (101 * 67) / 61 - 1
    assert e.name == "chapman_te" and e.point == pytest.approx(101 * 67 / 61 - 1)
    assert e.n_observed == 106


def test_the_named_model_is_telemetry_against_env_widened_by_session_spread():
    pooled = T(sa=40, sae=60, s=6, a=8, se=9, ae=12, e=10)
    even = [T(sa=4, sae=6, s=1, a=1, se=1, ae=1, e=1)] * 10
    e = lib.sa_hom(pooled, even)
    # Its point is Lincoln-Petersen between any-telemetry and env, uncorrected.
    n_t, n_e, both = 40 + 60 + 6 + 8 + 9 + 12, 60 + 9 + 12 + 10, 60 + 9 + 12
    assert e.point == pytest.approx(n_t * n_e / both, rel=1e-6)
    assert e.name == "ll_sa_hom" and e.detail["phi"] == 1.0 and e.lo <= e.point <= e.hi
    assert e.detail["pairs"] == [[0, 1]]
    # Sessions that disagree (an outage in half of them) widen the interval, not the point.
    dark, light = T(e=6, sae=1, sa=1), T(sa=7, sae=11, s=1, a=2, se=2, ae=2)
    wide = lib.sa_hom(pooled, [dark, light] * 5)
    assert wide.detail["phi"] > 1.5 and wide.point == e.point
    assert wide.lo <= e.lo and wide.hi > e.hi
    # A table the model cannot bound is unbounded, not a crash.
    none = lib.sa_hom(T(sa=5, e=3), [T(sa=5, e=3)])
    assert not none.bounded
    assert not lib.sa_hom(T(), []).bounded
    # A run that lost nothing has two non-zero cells. The iterative fit returns NaN on
    # some such tables (this one, from a pilot run); the closed form gives the count.
    quiet = T(sa=173, sae=200)
    e = lib.sa_hom(quiet, [quiet])
    assert (e.point, e.lo) == (373, 373) and 373 <= e.hi <= 375
    # Env saw nothing the telemetry missed: the missing cell is estimated at zero.
    zero = lib.sa_hom(T(s=5, sa=10, sae=30), [])
    assert (zero.point, zero.lo) == (45, 45) and 45 <= zero.hi <= 47
    # No action is in both lists: the end state says nothing about what telemetry
    # missed, whether or not env saw anything on its own.
    for table in (T(s=10, a=10, sa=10), T(s=10, sa=10, e=4)):
        blind = lib.sa_hom(table, [])
        assert not blind.bounded and blind.point == math.inf and blind.lo == table.sum()
    # An estimate past twenty times what was seen is "no idea", as for every other
    # log-linear estimate here, not a point.
    far = lib.sa_hom(T(s=100, a=100, sa=100, e=100, sae=1), [])
    assert far.point == math.inf and not far.bounded
    near = lib.sa_hom(T(s=100, a=100, sa=100, e=100, sae=20), [])
    assert near.point == pytest.approx(420 + 100 * 300 / 20) and near.bounded


def test_every_candidate_floor_and_baseline_is_returned_for_a_run():
    planes = {
        "a": T(sa=30, sae=50, s=4, a=3, se=6, ae=5, e=5),
        "b": T(sa=25, sae=45, s=3, a=6, se=4, ae=7, e=4),
    }
    est = lib.estimates(planes, list(planes.values()), [S(range(2, 40))])
    assert set(est) == {
        "strat_bic", "ll_bic", "ll_average", "ll_bic_boot", "ll_bic_hom", "chapman_te",
        "chapman_sa", "ll_sa_hom", "union_count", "union_bound", "trust_trace", "seq_gap",
        "seq_gap_tail",
    }  # fmt: skip
    sig = lib.estimates(planes, list(planes.values()), [], by_signature=(planes, []))
    assert sig["ll_sa_hom_sig"].name == "ll_sa_hom_sig"
    assert sig["ll_sa_hom_sig"].point == sig["ll_sa_hom"].point
    assert all(isinstance(e, Estimate) and e.name == name for name, e in est.items())
    union = int(sum(t.sum() for t in planes.values()))
    assert est["union_count"].point == union and est["union_bound"].lo == union
    assert est["trust_trace"].point == 30 + 50 + 4 + 6 + 25 + 45 + 3 + 4
    for name in ("strat_bic", "ll_bic", "chapman_te"):
        assert est[name].lo >= union and est[name].point >= union and est[name].bounded


def test_an_estimator_that_cannot_answer_is_unbounded_and_says_why():
    e = lib._safe("x", 7, lambda: (_ for _ in ()).throw(ValueError("no data")))
    assert (e.point, e.lo, e.hi) == (math.inf, 7.0, math.inf)
    assert e.detail == {"error": "ValueError: no data"}
    s = lib.score(e, 9)
    assert s["covers"] is True and s["bounded"] is False and s["abs_error"] == "inf"
    assert s["error"] is None
    json.dumps(s)


def test_a_score_is_json_and_an_interval_that_misses_does_not_cover():
    s = lib.score(Estimate("x", 10, 12.3456, 11.0, 14.0, {"phi": 1.5, "skip": [1]}), 15)
    assert s == {"point": 12.346, "lo": 11.0, "hi": 14.0, "covers": False, "bounded": True,
                 "abs_error": 2.654, "error": 12.3456 - 15, "detail": {"phi": 1.5}}  # fmt: skip


# A run ----------------------------------------------------------------------------


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "fixture"
    shutil.copytree(FIXTURE, run)
    manifest = {"params": {"condition": "none", "stress_seed": 0}, "schedule": [],
                "sessions_sha256": "x"}  # fmt: skip
    (run / "manifest.json").write_text(json.dumps(manifest))
    events = {"duration_s": 5.0, "integrity": {"ok": True, "problems": []}}
    (run / "events.json").write_text(json.dumps(events))
    return run


def test_an_unstressed_run_is_counted_exactly_by_the_union_and_the_sequence_baseline(tmp_path):
    rec = lib.run_record(_run(tmp_path), BACKOFFICE)
    n = rec["actions"]
    assert n == sum(rec["actions_by_plane"].values()) > 0
    assert rec["union_count"] == n and rec["union_missed_share"] == 0
    assert rec["estimates"]["union_count"]["abs_error"] == 0
    assert rec["estimates"]["seq_gap"]["abs_error"] == 0
    assert rec["estimates"]["seq_gap_tail"]["covers"] is True
    assert set(rec["tables"]) == {"a", "b"}
    assert sum(sum(t) for t in rec["tables"].values()) == n
    assert rec["missed_share"]["a"][rw.SPANS] == 0 and rec["missed_share"]["b"][rw.ACCESS] == 0
    assert rec["condition"] == "none" and rec["integrity"]["ok"] is True
    # A run made before the driver recorded these says so with None, not with a guess.
    assert rec["commit"] is None and rec["dirty"] is None and rec["timing"] is None
    assert rec["containers_clock_minus_driver_s"] is None
    json.dumps(rec)


def test_losing_a_tail_of_both_telemetry_logs_hides_it_from_the_baseline(tmp_path):
    run = _run(tmp_path)
    full = lib.run_record(run, BACKOFFICE)
    d = rw.load(run, BACKOFFICE)
    # Drop the last three calls of every plane-b session from both delivered logs.
    last: dict[str, int] = {}
    for plane, c in d["span_calls"]:
        if plane == "b":
            last[c["session"]] = max(last.get(c["session"], 0), c["rpc_id"])
    assert last
    gone = {(s, i) for s, top in last.items() for i in range(top - 2, top + 1)}

    def kept(c: dict) -> bool:
        return (c["session"], c["rpc_id"]) not in gone

    cut = {
        **d,
        "span_calls": [(p, c) for p, c in d["span_calls"] if p != "b" or kept(c)],
        "access_calls": {
            **d["access_calls"],
            "front-b.jsonl": [c for c in d["access_calls"]["front-b.jsonl"] if kept(c)],
        },
    }
    before = lib.seq_estimates(lib.streams(d, BACKOFFICE))["seq_gap"].point
    after = lib.seq_estimates(lib.streams(cut, BACKOFFICE))["seq_gap"].point
    assert before == full["actions"]
    assert after < before  # a lost tail leaves no gap


# Over runs --------------------------------------------------------------------------


def rec(n: int, missed: float = 0.05, tele: float = 0.2, **est) -> dict:
    """A run record with only what the aggregates read. est: name=(point, lo, hi)."""
    out = {
        "actions": n,
        "union_missed_share": missed,
        "missed_share": {p: {rw.SPANS: tele, rw.ACCESS: tele / 2, rw.ENV: 0.4} for p in "ab"},
        "estimates": {},
    }
    for name, (point, lo, hi) in est.items():
        out["estimates"][name] = lib.score(Estimate(name, 0, point, lo, hi), n)
    return out


def test_wilson_and_the_exact_lower_bound():
    lo, hi = lib.wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-4) and hi == pytest.approx(0.9785, abs=1e-4)
    assert lib.wilson(0, 0) == (0.0, 1.0)
    assert lib.wilson(10, 10)[1] == pytest.approx(1.0) and lib.wilson(0, 10)[0] == 0.0
    # Clopper-Pearson, one-sided 97.5%: 60 of 100 gives 0.4972, 61 of 100 gives 0.5073.
    assert lib.exact_lower(60, 100) == pytest.approx(0.4972, abs=1e-4)
    assert lib.exact_lower(61, 100) == pytest.approx(0.5073, abs=1e-4)
    assert lib.exact_lower(0, 10) == 0.0
    assert lib.exact_lower(10, 10) == pytest.approx(0.025 ** (1 / 10), abs=1e-6)


def test_coverage_counts_runs_and_an_unbounded_interval_covers_but_is_counted():
    records = (
        [rec(100, x=(100, 95, 105))] * 8
        + [rec(100, x=(90, 85, 95))]
        + [rec(100, x=(math.inf, 90, math.inf))]
    )
    c = lib.coverage(records, "x")
    assert (c["runs"], c["covered"], c["unbounded"]) == (10, 9, 1)
    assert c["coverage"] == 0.9 and c["unbounded_rate"] == 0.1
    assert c["median_abs_error"] == 0.0 and c["mean_error"] == pytest.approx(-10 / 9, abs=1e-3)
    assert (c["wilson_lo"], c["wilson_hi"]) == tuple(round(v, 4) for v in lib.wilson(9, 10))


def test_a_tie_or_an_unbounded_estimate_is_a_loss_in_the_paired_test():
    win = rec(100, x=(99, 90, 110), b=(95, 95, 95))
    tie = rec(100, x=(98, 90, 110), b=(102, 102, 102))
    lose = rec(100, x=(90, 80, 110), b=(99, 99, 99))
    unbounded = rec(100, x=(math.inf, 90, math.inf), b=(50, 50, 50))
    p = lib.paired([win] * 6 + [tie, lose, unbounded], "x", "b")
    assert (p["runs"], p["wins"], p["ties"]) == (9, 6, 1)
    assert p["win_rate"] == round(6 / 9, 4)
    assert p["passes"] is False  # six of nine does not put the lower bound above a half
    assert lib.paired([win] * 9, "x", "b")["passes"] is True
    # Both unbounded is a tie, and a tie is a loss.
    both = rec(100, x=(math.inf, 90, math.inf), b=(math.inf, 90, math.inf))
    assert lib.paired([both], "x", "b")["wins"] == 0


def test_gates_underpowered_and_not_exercised():
    good = [rec(100, missed=0.05, tele=0.2)] * 5
    g = lib.gates(good, all_runs=7, min_runs=5)
    assert g["runs_made"] == 7 and g["runs_passing_integrity"] == 5
    assert g["underpowered"] is False and g["not_exercised"] is False
    assert g["mean_union_missed_share"] == 0.05
    assert g["mean_telemetry_missed_share"] == {rw.SPANS: 0.2, rw.ACCESS: 0.1}
    assert lib.gates(good, 7, min_runs=6)["underpowered"] is True
    # The union misses too little, or neither telemetry log loses a hundredth.
    assert lib.gates([rec(100, missed=0.019)] * 5, 5, 5)["not_exercised"] is True
    assert lib.gates([rec(100, missed=0.05, tele=0.009)] * 5, 5, 5)["not_exercised"] is True
    # One log under a hundredth and the other over is exercised.
    assert lib.gates([rec(100, missed=0.05, tele=0.015)] * 5, 5, 5)["not_exercised"] is False
    assert lib.gates([], 3, 5)["not_exercised"] is True


def test_the_verdict_applies_the_gates_before_the_criteria():
    hit = rec(100, x=(99, 90, 110), seq_gap_tail=(90, 90, 90), union_count=(92, 92, 92))
    miss = rec(100, x=(80, 70, 90), seq_gap_tail=(99, 99, 99), union_count=(99, 99, 99))
    v = lib.verdict([hit] * 60, "x", all_runs=60, min_runs=50)
    assert v["P1"] == "pass" and v["P2"] == "pass" and v["primary"] == "x"
    assert set(v["paired"]) == set(lib.BASELINES)
    # 54 of 60 covered: the Wilson lower end is under 0.90.
    v = lib.verdict([hit] * 54 + [miss] * 6, "x", 60, min_runs=50)
    assert v["coverage"]["coverage"] == 0.9 and v["P1"] == "fail" and v["P2"] == "pass"
    # Beating one baseline is not enough.
    one = rec(100, x=(99, 90, 110), seq_gap_tail=(90, 90, 90), union_count=(100, 100, 100))
    assert lib.verdict([one] * 60, "x", 60, min_runs=50)["P2"] == "fail"
    assert lib.verdict([hit] * 60, "x", 60, min_runs=61)["P1"] == "underpowered"
    quiet = rec(100, missed=0.001, x=(99, 90, 110), seq_gap_tail=(9, 9, 9), union_count=(9, 9, 9))
    v = lib.verdict([quiet] * 60, "x", 60, min_runs=50)
    assert v["P1"] == v["P2"] == "not exercised"
    # Too many unbounded intervals fail P1 even when every interval covers.
    wide = rec(100, x=(math.inf, 90, math.inf), seq_gap_tail=(9, 9, 9), union_count=(9, 9, 9))
    assert lib.verdict([hit] * 56 + [wide] * 4, "x", 60, min_runs=50)["P1"] == "fail"


def test_a_false_alarm_is_an_interval_wholly_above_the_truth():
    ok = rec(100, x=(101, 99, 110))
    high = rec(100, x=(108, 103, 115))
    low = rec(100, x=(90, 85, 95))
    wide = rec(100, x=(math.inf, 90, math.inf))
    got = lib.false_alarms([ok, high, high, low, wide], "x")
    assert got == {"runs": 5, "interval_above_truth": 2}


def test_the_tail_interval_is_in_actions_not_requests():
    # Half the seen calls are actions, so every figure of the tail estimate is the
    # request figure with the unseen part halved.
    strs = [S([2, 3, 5, 6, 8, 9, 11], actions=4, session=f"s{i}") for i in range(10)]
    est = lib.seq_estimates(strs)["seq_gap_tail"]
    share = 40 / 70
    p = 3 / 9
    requests = 100 + 10 * p / (1 - p)
    assert est.point == pytest.approx(40 + (requests - 70) * share)
    full = lib.seq_estimates([S(s.ids, session=s.session) for s in strs])["seq_gap_tail"]
    assert est.hi - est.point == pytest.approx((full.hi - full.point) * share)
    assert est.lo == pytest.approx(40 + (full.lo - 70) * share)


def test_truth_is_counted_per_plane_from_the_ledger(tmp_path):
    run = _run(tmp_path)
    by_plane = {"a": 0, "b": 0}
    for line in (run / "truth" / "ledger.jsonl").read_text().splitlines():
        by_plane[json.loads(line)["principal"].removeprefix("plane-")] += 1
    assert by_plane["a"] != by_plane["b"]  # or a swap would not show
    assert lib.truth(rw.load(run, BACKOFFICE)) == by_plane
    assert lib.run_record(run, BACKOFFICE)["actions_by_plane"] == by_plane


def test_the_exact_lower_bound_does_not_overflow_on_many_runs():
    assert lib.exact_lower(1100, 2000) == pytest.approx(0.5279, abs=2e-3)
    assert lib.exact_lower(1100, 2000) < 0.55 < lib.exact_lower(1200, 2000)
    # The log-space sum agrees with the direct one where both work.
    direct = sum(math.comb(100, i) * 0.5**100 for i in range(61, 101))
    lo = lib.exact_lower(61, 100, alpha=direct)
    assert lo == pytest.approx(0.5, abs=1e-6)


def test_error_figures_are_over_the_runs_with_a_point_whatever_the_interval():
    floor = [rec(100, x=(92, 92, math.inf))] * 3 + [rec(100, x=(math.inf, 90, math.inf))]
    c = lib.coverage(floor, "x")
    assert c["unbounded"] == 4 and c["runs_with_a_point"] == 3
    assert c["median_abs_error"] == 8.0 and c["mean_error"] == -8.0


def test_the_gates_sit_at_the_designs_numbers():
    assert (lib.EXERCISED_MIN, lib.TELEMETRY_LOSS_MIN) == (0.02, 0.01)
    assert lib.gates([rec(100, missed=0.02)] * 5, 5, 5)["not_exercised"] is False
    assert lib.gates([rec(100, missed=0.0199)] * 5, 5, 5)["not_exercised"] is True
    assert lib.gates([rec(100, missed=0.05, tele=0.01)] * 5, 5, 5)["not_exercised"] is False
    assert lib.gates([rec(100, missed=0.05, tele=0.0099)] * 5, 5, 5)["not_exercised"] is True


def test_the_criteria_sit_at_the_designs_numbers_and_are_not_tested_on_rounded_ones():
    assert (lib.COVERAGE_FLOOR, lib.UNBOUNDED_MAX, lib.WIN_RATE_FLOOR) == (0.90, 0.05, 0.5)
    hit = rec(100, x=(99, 90, 110), seq_gap_tail=(90, 90, 90), union_count=(92, 92, 92))
    miss = rec(100, x=(80, 70, 90), seq_gap_tail=(90, 90, 90), union_count=(92, 92, 92))
    # 260 of 278: the Wilson lower end is 0.89998, which rounds to 0.9000.
    assert round(lib.wilson(260, 278)[0], 4) == 0.9 > lib.wilson(260, 278)[0]
    v = lib.verdict([hit] * 260 + [miss] * 18, "x", 278, min_runs=270)
    assert v["coverage"]["wilson_lo"] == 0.9 and v["P1"] == "fail"
    assert lib.verdict([hit] * 261 + [miss] * 17, "x", 278, min_runs=270)["P1"] == "pass"
    # Unbounded intervals: under one in twenty, so three of sixty is one too many.
    wide = rec(100, x=(math.inf, 90, math.inf), seq_gap_tail=(9, 9, 9), union_count=(9, 9, 9))
    assert lib.verdict([hit] * 58 + [wide] * 2, "x", 60, min_runs=50)["P1"] == "pass"
    assert lib.verdict([hit] * 57 + [wide] * 3, "x", 60, min_runs=50)["P1"] == "fail"
    # The paired test: 61 wins of 100 clears a half, 60 does not.
    lose = rec(100, x=(80, 70, 130), seq_gap_tail=(99, 99, 99), union_count=(99, 99, 99))
    assert lib.verdict([hit] * 61 + [lose] * 39, "x", 100, min_runs=50)["P2"] == "pass"
    assert lib.verdict([hit] * 60 + [lose] * 40, "x", 100, min_runs=50)["P2"] == "fail"


def test_a_run_made_by_the_current_driver_carries_what_made_it_and_its_timing(tmp_path):
    run = _run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    (run / "manifest.json").write_text(json.dumps(manifest | {"commit": "abc", "dirty": True}))
    events = json.loads((run / "events.json").read_text())
    timing = {"agent-a": {"start_lag_s": 0.0, "max_lag_s": 0.01}, "agent-b": None}
    events |= {"timing": timing, "containers_clock_minus_driver_s": 0.05}
    (run / "events.json").write_text(json.dumps(events))
    rec = lib.run_record(run, BACKOFFICE)
    assert (rec["commit"], rec["dirty"]) == ("abc", True)
    assert rec["timing"] == timing and rec["containers_clock_minus_driver_s"] == 0.05


def test_the_paired_test_compares_errors_as_they_are_not_as_they_are_printed():
    # 0.0004 apart: the printed errors are both 5.0, the estimates are not tied.
    near = rec(100, x=(104.9998, 90, 110), b=(105.0002, 105.0002, 105.0002))
    assert near["estimates"]["x"]["abs_error"] == near["estimates"]["b"]["abs_error"] == 5.0
    p = lib.paired([near], "x", "b")
    assert (p["wins"], p["ties"]) == (1, 0)


def test_coverage_also_counts_the_bounded_intervals_alone():
    records = [rec(100, x=(100, 95, 105))] * 6 + [rec(100, x=(80, 70, 90))] * 2
    records += [rec(100, x=(100, 95, math.inf))] * 2  # a point, and no upper end
    c = lib.coverage(records, "x")
    assert (c["covered"], c["unbounded"]) == (8, 2)
    assert (c["bounded"], c["covered_bounded"]) == (8, 6)


def test_the_gates_count_the_runs_where_the_union_is_already_exact():
    runs = [rec(100, missed=0.0)] * 2 + [rec(100, missed=0.06)] * 3
    assert lib.gates(runs, 5, 5)["runs_with_nothing_in_no_log"] == 2


def test_a_criterion_not_met_is_read_as_missed_only_when_the_interval_excludes_it():
    hit = rec(100, x=(99, 90, 110), seq_gap_tail=(90, 90, 90), union_count=(92, 92, 92))
    miss = rec(100, x=(80, 70, 90), seq_gap_tail=(99, 99, 99), union_count=(99, 99, 99))
    passed = lib.verdict([hit] * 60, "x", 60, min_runs=50)
    assert (passed["P1_reading"], passed["P2_reading"]) == ("pass", "pass")
    # 54 of 60 covered: not shown to be at 0.90, and not shown to be under it either.
    v = lib.verdict([hit] * 54 + [miss] * 6, "x", 60, min_runs=50)
    assert (v["P1"], v["P1_reading"]) == ("fail", "not shown")
    assert (v["P2"], v["P2_reading"]) == ("pass", "pass")
    # 40 of 60: the whole interval is under 0.90, and the win rate's is under a half
    # against neither baseline (40 wins of 60).
    v = lib.verdict([hit] * 40 + [miss] * 20, "x", 60, min_runs=50)
    assert lib.wilson(40, 60)[1] < 0.90
    assert (v["P1"], v["P1_reading"]) == ("fail", "missed")
    assert v["P2_reading"] == "pass"
    v = lib.verdict([hit] * 20 + [miss] * 40, "x", 60, min_runs=50)
    assert lib.exact_upper(20, 60) < 0.5
    assert (v["P2"], v["P2_reading"]) == ("fail", "missed")
    # Half and half: neither shown.
    v = lib.verdict([hit] * 30 + [miss] * 30, "x", 60, min_runs=50)
    assert (v["P2"], v["P2_reading"]) == ("fail", "not shown")
    # A gate speaks before any reading.
    assert lib.verdict([hit] * 60, "x", 60, min_runs=61)["P1_reading"] == "underpowered"
    assert lib.exact_upper(60, 60) == 1.0 and lib.exact_upper(0, 60) == pytest.approx(
        1 - 0.025 ** (1 / 60)
    )


def test_the_primary_candidate_is_also_computed_with_no_id_trusted(tmp_path):
    run = _run(tmp_path)
    d = rw.load(run, BACKOFFICE)
    by_id, by_sig = lib.plane_tables(d["obs"]), lib.plane_tables(d["obs"], by_id=False)
    # With nothing lost and no two actions alike, the two linkings give one table.
    assert {p: list(t) for p, t in by_id.items()} == {p: list(t) for p, t in by_sig.items()}
    rec_ = lib.run_record(run, BACKOFFICE)
    assert rec_["estimates"]["ll_sa_hom_sig"]["point"] == rec_["estimates"]["ll_sa_hom"]["point"]
    # Two calls alike in one session, one log losing the first and the other the
    # second: ids keep them apart, signatures merge them into one action seen by both.
    from lacuna.core.records import Obs

    def o(w, link):
        return Obs(w, "send_email", "a@b", "v", stratum="a", link_id=link, session="s")

    obs = [o(rw.SPANS, ("t", 2)), o(rw.ACCESS, ("t", 3))]
    assert int(lib.plane_tables(obs)["a"].sum()) == 2
    assert int(lib.plane_tables(obs, by_id=False)["a"].sum()) == 1


def test_a_run_names_its_pool_and_whether_an_outage_outlasted_its_calls(tmp_path):
    run = _run(tmp_path)
    assert lib.run_record(run, BACKOFFICE)["pool"] is None
    assert lib.run_record(run, BACKOFFICE)["outage_past_last_call"] is None
    manifest = json.loads((run / "manifest.json").read_text())
    late = [{"at_s": 130.0, "action": "start", "service": "otelcol-gw"}]
    manifest |= {"pools": [{"path": "/x/pools/qw/sessions.json", "sha256": "h"}],
                 "restarts_after_planned_end": late}  # fmt: skip
    (run / "manifest.json").write_text(json.dumps(manifest))
    rec_ = lib.run_record(run, BACKOFFICE)
    assert rec_["pool"] == "qw" and rec_["outage_past_last_call"] is True
    manifest |= {"restarts_after_planned_end": [], "pools": manifest["pools"] * 2}
    (run / "manifest.json").write_text(json.dumps(manifest))
    rec_ = lib.run_record(run, BACKOFFICE)
    assert rec_["pool"] is None and rec_["outage_past_last_call"] is False


# What was lost, described from the receivers' own logs ---------------------------------


def _cut_tail_of_plane_b(d: dict) -> tuple[dict, set]:
    """`d` with the last three calls of every plane-b session gone from both telemetry logs."""
    last: dict[str, int] = {}
    for plane, c in d["span_calls"]:
        if plane == "b":
            last[c["session"]] = max(last.get(c["session"], 0), c["rpc_id"])
    gone = {(s, i) for s, top in last.items() for i in range(top - 2, top + 1)}
    kept = lambda c: (c["session"], c["rpc_id"]) not in gone  # noqa: E731
    front = [c for c in d["access_calls"]["front-b.jsonl"] if kept(c)]
    cut = {
        **d,
        "span_calls": [(p, c) for p, c in d["span_calls"] if p != "b" or kept(c)],
        "access_calls": {**d["access_calls"], "front-b.jsonl": front},
    }
    return cut, gone


def test_the_actions_neither_log_delivered_are_counted_with_what_the_end_state_shows(tmp_path):
    d = rw.load(_run(tmp_path), BACKOFFICE)
    n = sum(lib.truth(d).values())
    whole = lib.lost_by_both(d, BACKOFFICE)
    assert whole["actions"] == 0 == whole["shown_by_end_state"] and whole["rest"] == n
    assert 0 < whole["rest_shown"] <= n
    cut, gone = _cut_tail_of_plane_b(d)
    lost = lib.lost_by_both(cut, BACKOFFICE)
    written = d["written_calls"]["front-b.jsonl"]
    expected = sum(
        rw._obs(rw.ACCESS, BACKOFFICE, "b", c) is not None
        for c in written
        if (c["session"], c["rpc_id"]) in gone
    )
    assert lost["actions"] == expected > 0
    assert lost["actions"] + lost["rest"] == n
    assert lost["shown_by_end_state"] + lost["rest_shown"] == whole["rest_shown"]
    # Lost by one log only is not lost by both.
    spans_only = {**cut, "access_calls": d["access_calls"]}
    assert lib.lost_by_both(spans_only, BACKOFFICE)["actions"] == 0


def test_what_the_end_state_shows_is_split_by_where_in_its_session_an_action_fell():
    def cancel(order: str, ts: float, session: str = "s") -> dict:
        return call(session, int(ts) + 2) | {"args": {"order_id": order}, "ts": ts}

    calls = [cancel(f"o{i}", float(i)) for i in range(10)]
    read = call("s", 99, tool="get_order") | {"ts": 0.5}
    failed = cancel("o99", 3.5) | {"ok": False}
    env = [
        rw.Obs(rw.ENV, "cancel_order", f"o{i}", "cancelled", stratum="a", session="s")
        for i in (7, 8, 9)
    ]
    d = {"obs": env, "written_calls": {"gateway.jsonl": [*reversed(calls), read, failed]}}
    # Ten actions in five parts of two, in the order they were written, not the file's.
    assert lib.shown_by_position(d, BACKOFFICE) == [[2, 0], [2, 0], [2, 0], [2, 1], [2, 2]]
    # A session's parts are its own: a one-action session falls in the first.
    d["written_calls"]["front-b.jsonl"] = [cancel("o7", 50.0, session="other")]
    assert lib.shown_by_position(d, BACKOFFICE)[0] == [3, 0]
    d["obs"].append(rw.Obs(rw.ENV, "cancel_order", "o7", "cancelled", session="other"))
    assert lib.shown_by_position(d, BACKOFFICE)[0] == [3, 1]


def test_an_outage_length_is_read_off_the_schedule():
    schedule = [
        {"at_s": 10.0, "action": "kill", "service": "otelcol-gw"},
        {"at_s": 10.0, "action": "kill", "service": "otelcol-node"},
        {"at_s": 31.25, "action": "start", "service": "otelcol-gw"},
        {"at_s": 40.0, "action": "kill", "service": "agent-a"},
    ]
    assert lib.outage_s(schedule) == 21.25
    assert lib.outage_s(schedule, "otelcol-node") is None  # never restarted
    assert lib.outage_s(schedule, "agent-a") is None and lib.outage_s([]) is None


def test_a_band_holds_its_lower_end_and_not_its_upper():
    assert lib.band(0.0, lib.OUTAGE_BANDS) == "0 to 20"
    assert lib.band(19.999, lib.OUTAGE_BANDS) == "0 to 20"
    assert lib.band(20.0, lib.OUTAGE_BANDS) == "20 to 28"
    assert lib.band(28.0, lib.OUTAGE_BANDS) == "28 or more"
    assert (
        lib.band(11, lib.LOST_BANDS) == "0 to 12" and lib.band(12, lib.LOST_BANDS) == "12 or more"
    )
    assert lib.band(None, lib.OUTAGE_BANDS) is None and lib.band(-1.0, lib.OUTAGE_BANDS) is None


def test_the_exploratory_breakdown_counts_coverage_per_band_and_pools_what_was_lost():
    def one(outage, lost, shown, covers: bool) -> dict:
        r = rec(100, x=(100, 90, 110) if covers else (120, 110, 130))
        return r | {
            "span_relay_outage_s": outage,
            "lost_by_both": {"actions": lost, "shown_by_end_state": shown, "rest": 100 - lost,
                             "rest_shown": 50},
        }  # fmt: skip

    runs = [one(15.0, 4, 0, False), one(19.0, 20, 10, True), one(30.0, 30, 12, True),
            one(None, 0, 0, True)]  # fmt: skip
    out = lib.breakdown(runs, ["x"])
    assert list(out["by_outage_s"]) == ["0 to 20", "20 to 28", "28 or more"]
    short = out["by_outage_s"]["0 to 20"]
    assert short["runs"] == 2 and short["covered"] == {"x": 1}
    assert short["lost_by_both"] == {"actions": 24, "shown_by_end_state": 10, "rest": 176,
                                     "rest_shown": 100}  # fmt: skip
    assert out["by_outage_s"]["20 to 28"]["runs"] == 0
    assert out["by_outage_s"]["20 to 28"]["covered"] == {"x": 0}
    assert out["not_covered"] == {
        "x": [{"run": None, "actions": 100, "lo": 110.0, "hi": 130.0, "outside_by": 10.0}]
    }
    # A run with no outage is in no outage band, and is in a band by what it lost.
    assert sum(b["runs"] for b in out["by_outage_s"].values()) == 3
    few, many = out["by_lost_by_both"]["0 to 12"], out["by_lost_by_both"]["12 or more"]
    assert (few["runs"], few["covered"]) == (2, {"x": 1})
    assert (many["runs"], many["covered"]) == (2, {"x": 2})
    # A record made before these fields existed is in no band.
    assert lib.breakdown([rec(100, x=(100, 90, 110))], ["x"])["by_lost_by_both"]["0 to 12"] == {
        "runs": 0, "covered": {"x": 0}, "lost_by_both": dict.fromkeys(lib.LOST_KEYS, 0),
    }  # fmt: skip


def test_a_run_not_covered_is_listed_with_how_far_outside_the_truth_fell():
    runs = [
        rec(100, x=(100, 90, 110)) | {"run": "in"},
        rec(100, x=(120, 104.5, 130)) | {"run": "above"},
        rec(100, x=(90, 88, 97)) | {"run": "below"},
        rec(100, x=(100, 100, float("inf"))) | {"run": "open"},
    ]
    assert lib.outside(runs, "x") == [
        {"run": "above", "actions": 100, "lo": 104.5, "hi": 130.0, "outside_by": 4.5},
        {"run": "below", "actions": 100, "lo": 88.0, "hi": 97.0, "outside_by": 3.0},
    ]
    # Floors and baselines are not listed: the breakdown lists the candidates.
    out = lib.breakdown([r | {"estimates": r["estimates"] | {"union_count": r["estimates"]["x"]}}
                         for r in runs], ["x", "union_count"])  # fmt: skip
    assert set(out["not_covered"]) == {"x"}


def test_a_run_record_carries_the_outage_length_and_the_description_of_what_was_lost(tmp_path):
    run = _run(tmp_path)
    rec_ = lib.run_record(run, BACKOFFICE)
    assert rec_["span_relay_outage_s"] is None
    assert rec_["lost_by_both"]["actions"] == 0
    assert rec_["lost_by_both"]["rest"] == rec_["actions"]
    assert sum(n for n, _ in rec_["shown_by_position"]) == rec_["actions"]
    assert len(rec_["shown_by_position"]) == lib.POSITION_PARTS
    shown = sum(k for _, k in rec_["shown_by_position"])
    assert shown == rec_["lost_by_both"]["rest_shown"]
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["schedule"] = [
        {"at_s": 8.0, "action": "kill", "service": "otelcol-gw"},
        {"at_s": 30.5, "action": "start", "service": "otelcol-gw"},
    ]
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert lib.run_record(run, BACKOFFICE)["span_relay_outage_s"] == 22.5
