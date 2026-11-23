"""scripts/study5_posthoc3.py: the dependence between runs estimated, and the runs set
aside described. Post hoc (D111, D112). On small inputs, then against the artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402
import study5_posthoc3 as post  # noqa: E402
import study5_runs  # noqa: E402

ARTIFACT = ROOT / "artifacts" / "study5" / "posthoc3.json"
RUNS = ROOT / "artifacts" / "rig" / "runs"


def _art(rel: str) -> dict:
    return json.loads((ROOT / "artifacts" / rel).read_text())


def _draws(rng, pools: int, runs: int, sessions: int, per_run: int):
    """Runs that each replay `per_run` of their pool's `sessions`: who replays what, the
    pool of each run, and the count of sessions each pair of one pool shares."""
    member = np.zeros((pools * runs, pools * sessions))
    pool = np.repeat(np.arange(pools), runs)
    for r in range(pools * runs):
        chosen = rng.choice(sessions, per_run, replace=False) + pool[r] * sessions
        member[r, chosen] = 1
    k = member @ member.T
    k[pool[:, None] != pool[None, :]] = 0
    np.fill_diagonal(k, 0)
    return member, pool, k


def test_the_slope_is_the_least_squares_slope_over_pairs():
    rng = np.random.default_rng(1)
    _, pool, k = _draws(rng, pools=2, runs=9, sessions=12, per_run=4)
    y = rng.normal(size=len(pool))
    c = post._centred(y, pool)
    x, ss = post._design(k, pool, per_run=4)
    # By hand: the products of pairs of one pool on k / 4, an intercept for each pool.
    rows = []
    for p in (0, 1):
        idx = np.flatnonzero(pool == p)
        pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1 :]]
        xs = np.array([k[i, j] / 4 for i, j in pairs])
        zs = np.array([c[i] * c[j] for i, j in pairs])
        rows.append((xs - xs.mean(), zs - zs.mean()))
    num = sum(float(a @ b) for a, b in rows)
    den = sum(float(a @ a) for a, _ in rows)
    assert post.slope(c, x, ss) == pytest.approx(num / den)
    assert ss == pytest.approx(den)


def test_it_finds_the_dependence_that_was_put_in_and_none_where_there_is_none():
    rng = np.random.default_rng(7)
    per_run, found, null = 24, [], []
    for _ in range(40):
        member, pool, k = _draws(rng, pools=3, runs=92, sessions=72, per_run=per_run)
        effect = rng.normal(size=member.shape[1])
        noise = rng.normal(size=len(pool))
        for rho, out in ((0.3, found), (0.0, null)):
            # A session's term has variance rho / per_run, so a run's outcome has variance 1
            # and the share that comes from its sessions is rho.
            y = np.sqrt(rho / per_run) * (member @ effect) + np.sqrt(1 - rho) * noise
            out.append(post.dependence(y, pool, k, per_run, permutations=50, seed=1)["rho"])
    assert np.mean(found) == pytest.approx(0.3, abs=0.06)
    assert np.mean(null) == pytest.approx(0.0, abs=0.06)


def test_the_design_effect_is_one_plus_rho_times_the_shared_weight():
    rng = np.random.default_rng(3)
    member, pool, k = _draws(rng, pools=1, runs=30, sessions=12, per_run=4)
    y = member @ rng.normal(size=12) + 0.1 * rng.normal(size=30)
    d = post.dependence(y, pool, k, per_run=4, permutations=200, seed=2)
    weight = (k / 4).sum(axis=1).mean()
    assert d["mean_shared_weight"] == pytest.approx(weight, abs=1e-3)
    assert d["design_effect"] == pytest.approx(1 + max(d["rho"], 0) * weight, abs=0.02)
    assert d["rho_upper_95"] == pytest.approx(
        d["rho"] + post.Z_ONE_SIDED_95 * d["sd_of_permuted_rho"], abs=2e-4
    )
    assert 0 < d["p_one_sided"] <= 1


def test_the_outcomes_are_read_from_the_registered_estimate():
    run = {
        "actions": 100,
        "estimates": {"ll_sa_hom": {"covers": False, "error": -4.0, "lo": 80.0, "hi": 96.0}},
    }
    assert post.outcomes([run]) == {"miss": [1.0], "error": [-4.0], "place": [1.25]}
    run["estimates"]["ll_sa_hom"] |= {"covers": True, "hi": 80.0}
    assert post.outcomes([run])["place"] == [None]


LOG = "\n".join(
    (
        "# a header line that is not an event",
        "2026-11-16 11:21:41 -0300 Sleep               \tEntering Sleep state due to "
        "'Clamshell Sleep':TCPKeepAlive=active Using Batt (Charge:28%) 63 secs",
        "2026-11-16 11:22:44 -0300 DarkWake            \tDarkWake from Deep Idle [CDNP] : "
        "due to x Using BATT (Charge:28%) 20 secs",
        "2026-11-16 11:45:56 -0300 BatteryHealth       \tWarning level: 2 time: 5 cap: 8",
        "2026-11-16 14:33:17 -0300 DarkWake            \tDarkWake from Deep Idle [CDNP] : "
        "due to smc USB-C_plug x/ Using AC (Charge:8%) 0 secs",
    )
)


def test_the_power_log_is_read_in_utc():
    log = post.power_log(LOG)
    (sleep,) = log["sleeps"]
    assert post._iso(sleep["start"]) == "2026-11-16T14:21:41Z"
    assert (sleep["secs"], sleep["reason"]) == (63, "Clamshell Sleep")
    assert sleep["end"] - sleep["start"] == 63
    assert [s["source"] for s in log["source"]] == ["battery", "battery", "ac"]
    assert log["battery_warnings"] == [{"at": "2026-11-16T14:45:56Z", "capacity": 8}]
    assert [post._iso(t) for t in log["plugged"]] == ["2026-11-16T17:33:17Z"]


def test_a_run_set_aside_has_one_kind():
    late = ["agent-a made a call 0.4 s late (limit 0.25 s)"]
    assert post._kind(late, {"max_lag_s": 0.4, "clock_s": 0.0}) == "short"
    assert post._kind(late, {"max_lag_s": 60.1, "clock_s": 0.0}) == "long"
    assert post._kind(["the containers' clock is off"], {"max_lag_s": 0.0, "clock_s": -33.8}) == (
        "long"
    )
    assert post._kind(["the run driver stopped: x"], {"max_lag_s": None, "clock_s": None}) == (
        "no record"
    )
    assert post._kind(["plane b: the log relay delivered no line"], {"max_lag_s": 0.0}) == (
        "no access line"
    )


# --- the artifact --------------------------------------------------------------------


def test_the_artifact_says_it_is_post_hoc():
    made = json.loads(ARTIFACT.read_text())
    assert made["note"].startswith("Post hoc") and "D111" in made["note"]
    assert "no registered criterion" in made["note"]
    assert "not a test" in made["runs_set_aside"]["estimator"]["reading"]
    assert "assum" in made["dependence_between_runs"]["model"].lower()
    log = ROOT / made["host_log"]["file"]
    assert hashlib.sha256(log.read_text().encode()).hexdigest() == made["host_log"]["sha256"]


def test_the_dependence_in_the_artifact_is_what_the_repository_gives():
    """Needs no run folder: the registered analysis, the pools and the registered draw."""
    confirm = _art("study5/confirm.json")
    bearable = _art("study5/posthoc.json")["dependence_between_runs"][
        "largest_design_effect_at_or_above_the_bar"
    ]
    pools = study5_runs.load_pools(study5_runs.POOLS_DIR)
    again = post.dependence_between_runs(confirm, pools, bearable)
    assert again == json.loads(ARTIFACT.read_text())["dependence_between_runs"]
    assert again["runs"] == 277 and again["sessions_a_run"] == 24
    # Two runs of a pool share about a third of their sessions, as the plan says.
    assert again["shared_sessions"]["mean"] == pytest.approx(8, abs=0.1)
    miss = again["by_outcome"]["miss"]
    assert again["upper_limit_on_misses_above_what_the_bound_bears"] == (
        miss["design_effect_upper_95"] > bearable
    )


def test_the_figures_beside_the_registered_ones_are_the_counts_added():
    v = _art("robustness/study5-verdicts.json")
    e = json.loads(ARTIFACT.read_text())["runs_set_aside"]["estimator"]
    cov, beside = v["primary"]["coverage"], v["P1_not_in_test_as_uncovered"]
    assert e["registered"]["in_the_test"]["covered"] == cov["covered"]
    assert e["registered"]["in_the_test"]["runs"] == cov["runs"]
    assert e["registered"]["not_in_the_test_as_uncovered"]["runs"] == beside["runs"]
    assert e["set_aside"] == beside["not_in_test"]
    assert e["with_a_full_record"] + len(e["with_no_record"]) == e["set_aside"]
    assert len(e["runs"]) == e["with_a_full_record"]
    assert e["covered"] == sum(r["covers"] for r in e["runs"])
    # A run counts as covered exactly when its interval holds its true count.
    assert all(r["covers"] == (r["lo"] <= r["actions"] <= r["hi"]) for r in e["runs"])
    scored = e["with_the_runs_set_aside_as_scored"]
    k = cov["covered"] + e["covered"]
    for key, n in (
        ("runs_with_a_record", cov["runs"] + e["with_a_full_record"]),
        ("no_record_as_uncovered", beside["runs"]),
    ):
        assert (scored[key]["covered"], scored[key]["runs"]) == (k, n)
        assert scored[key]["wilson"] == [round(x, 4) for x in lib.wilson(k, n)]


def test_the_runs_beside_the_power_log_are_the_runs_set_aside():
    confirm = _art("study5/confirm.json")
    b = json.loads(ARTIFACT.read_text())["runs_set_aside"]["beside_the_power_log"]
    aside = {r["run"]: r["condition"] for r in confirm["runs_set_aside"]}
    assert {r["run"]: r["condition"] for r in b["runs"]} == aside
    assert b["set_aside"] == len(aside) == sum(b["by_kind"].values())
    assert b["long"]["runs"] == b["by_kind"]["long"]
    assert b["long"]["in_the_log"] + b["long"]["before_the_log_begins"] == b["long"]["runs"]
    w = b["windows"]
    assert (
        sum(x["runs"] for x in w.values()) + b["by_kind"]["no record"] == (confirm["planned_runs"])
    )
    for x in w.values():
        assert sum(c["runs"] for c in x["by_condition"].values()) == x["runs"]
        assert (
            sum(c["set_aside_short"] for c in x["by_condition"].values()) == (x["set_aside_short"])
        )
    # No estimate, interval or error is in the part that sets runs beside the log.
    assert not {"point", "lo", "hi", "covers", "error"} & {k for r in b["runs"] for k in r}


@pytest.mark.skipif(not (RUNS / "s5-s1-219").exists(), reason="the raw runs are a release asset")
def test_the_artifact_is_what_the_script_writes(tmp_path):
    post.main(["--out", str(tmp_path / "p.json")])
    assert json.loads((tmp_path / "p.json").read_text()) == json.loads(ARTIFACT.read_text())
