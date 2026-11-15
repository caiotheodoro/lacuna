"""scripts/study5_runs.py: the plan of the study's runs, and what makes a run registered."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_analysis as sa  # noqa: E402
import study5_runs as sr  # noqa: E402

from lacuna.envs.registry import BACKOFFICE_CHURN  # noqa: E402
from lacuna.rig.manifest import (  # noqa: E402
    REGISTERED_HASHES,
    REGISTERED_PARAMS,
    registered,
    registered_problems,
)
from lacuna.rig.sessions import Session, draw, dump, planned_end, scripted_sessions  # noqa: E402
from lacuna.rig.stress import plan  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "rig" / "run"


# The plan ---------------------------------------------------------------------------


def test_the_plan_is_480_runs_with_names_and_seeds_of_their_own():
    runs = sr.runs()
    assert len(runs) == 480 and runs == sr.runs()
    by = {c: [r for r in runs if r["condition"] == c] for c in ("s0", "s1", "s2", "s3")}
    assert {c: len(v) for c, v in by.items()} == {"s0": 60, "s1": 300, "s2": 60, "s3": 60}
    assert len({r["run"] for r in runs}) == 480 and len({r["seed"] for r in runs}) == 480
    for c, first in (("s0", 70000), ("s1", 71000), ("s2", 72000), ("s3", 73000)):
        for r in by[c]:
            assert r["run"] == f"s5-{c}-{r['index']:03d}" and r["seed"] == first + r["index"]
            # Run i of a condition replays pool i mod 3, so no pool is a stretch of time.
            assert r["pool"] == ("ds", "glm", "mix")[r["index"] % 3]
    assert set(runs[0]) == {"run", "condition", "index", "seed", "pool"}
    assert sr.SAMPLE["run"] not in {r["run"] for r in runs}
    assert sr.SAMPLE["seed"] not in {r["seed"] for r in runs}


def test_the_order_spreads_every_condition_over_the_whole_sequence():
    runs = sr.runs()
    # Each run sits at the middle of its share of the sequence: five s1 runs to one of
    # each other condition, the rarer ones in the middle of their five.
    assert [r["run"] for r in runs[:9]] == [
        "s5-s1-000", "s5-s1-001", "s5-s0-000", "s5-s1-002", "s5-s2-000", "s5-s3-000",
        "s5-s1-003", "s5-s1-004", "s5-s1-005",
    ]  # fmt: skip
    assert runs[-1]["run"] == "s5-s1-299"
    for start in range(0, 480, 8):
        chunk = [r["condition"] for r in runs[start : start + 8]]
        assert sorted(chunk) == ["s0", "s1", "s1", "s1", "s1", "s1", "s2", "s3"]
    for c in ("s0", "s1", "s2", "s3"):
        idx = [r["index"] for r in runs if r["condition"] == c]
        assert idx == sorted(idx)


def test_a_runs_arguments_are_its_pool_its_seed_and_its_condition(tmp_path):
    cases = (("s5-s2-004", "glm", 72004, "s2"), ("s5-s0-002", "mix", 70002, "s0"))
    for name, pool, seed, cond in cases:
        run = next(r for r in sr.runs() if r["run"] == name)
        args = sr.argv(run, tmp_path / "pools", tmp_path / "runs")
        assert args[1].endswith("scripts/rig_run.py")
        assert dict(zip(args[2::2], args[3::2], strict=True)) == {
            "--run": name,
            "--out": str(tmp_path / "runs"),
            "--env": "backoffice_churn",
            "--sessions-file": str(tmp_path / "pools" / pool / "sessions.json"),
            "--sessions": "24",
            "--draw-seed": str(seed),
            "--stagger": "100.0",
            "--condition": cond,
            "--stress-seed": str(seed),
        }


def _pool(first_seed: int, n: int = sr.POOL_SESSIONS, tickets: int = 10) -> list[Session]:
    return [
        s
        for seed in range(first_seed, first_seed + n)
        for s in scripted_sessions(
            seed, tickets, ("a", "b"), pace_s=1.0, env=BACKOFFICE_CHURN, timeline=True
        )
    ]


@pytest.fixture(scope="module")
def pools() -> dict[str, list[Session]]:
    return {name: _pool(1000 * (k + 1)) for k, name in enumerate(sr.POOLS)}


def test_every_planned_run_can_be_made_from_pools_of_the_studys_shape(pools):
    assert sr.check(pools) == {}


def test_a_run_is_refused_for_the_sessions_its_own_seed_draws(pools):
    # One session of the ds pool has two shares for plane a. Only the runs whose seed
    # draws that session are refused, and they are exactly the ones a draw says.
    bad_name = pools["ds"][0].session
    spoiled = {**pools, "ds": [*pools["ds"], pools["ds"][0]]}
    bad = sr.check(spoiled)
    want = {
        r["run"]
        for r in sr.runs()
        if r["pool"] == "ds" and bad_name in {s.session for s in draw(pools["ds"], 24, r["seed"])}
    }
    assert set(bad) == want and 0 < len(want) < 140
    # About a third of a pool's 140 runs draw any one of its 72 sessions.
    assert 30 <= len(want) <= 65
    assert all(f"session {bad_name} has two shares for plane a" in why for why in bad.values())


def test_a_pool_too_small_or_a_kill_after_the_last_call_is_a_refusal():
    tiny = {name: _pool(1000 * (k + 1), n=6, tickets=1) for k, name in enumerate(sr.POOLS)}
    bad = sr.check(tiny)
    assert set(bad) == {r["run"] for r in sr.runs()} | {f"pool {p}" for p in sr.POOLS}
    assert "the pool gives 6 sessions, the run needs 24" in bad["s5-s0-000"]
    killed = [name for name, why in bad.items() if any("is killed at" in w for w in why)]
    assert killed and all(not name.startswith("s5-s0-") for name in killed)
    run = next(r for r in sr.runs() if r["run"] == killed[0])
    end = planned_end(draw(tiny[run["pool"]], 24, run["seed"]), sr.STAGGER)
    late = [e for e in plan(run["condition"], run["seed"]) if e.action == "kill" and e.at_s >= end]
    assert late and f"{late[0].service} is killed at {late[0].at_s} s" in " ".join(bad[killed[0]])


def test_the_plan_names_the_sessions_file_each_run_will_replay(pools):
    planned = sr.planned(pools)
    assert set(planned) == {r["run"] for r in sr.runs()}
    row = planned["s5-s1-004"]
    assert (row["condition"], row["seed"], row["pool"]) == ("s1", 71004, "glm")
    # The driver writes dump(draw(pool, 24, seed)) and hashes the file.
    text = dump(draw(pools["glm"], 24, 71004))
    assert row["sessions_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert len({r["sessions_sha256"] for r in planned.values()}) == 480


# What makes a run a registered run ------------------------------------------------

POOL_HASHES = {"ds": "p-ds", "glm": "p-glm", "mix": "p-mix"}
PLAN = {
    "s5-s1-000": {"condition": "s1", "seed": 71000, "pool": "ds", "sessions_sha256": "x0"},
    "s5-s1-001": {"condition": "s1", "seed": 71001, "pool": "glm", "sessions_sha256": "x1"},
}
MANIFEST = {
    "run": "s5-s1-000",
    "commit": "abc",
    "dirty": False,
    "source_sha256": "s" * 64,
    "driver_sha256": "d" * 64,
    "lock_sha256": "l" * 64,
    "configs": {"compose.yaml": "c1", "config/backend.yaml": "c2"},
    "images": {"kernel": "i1", "backend": "i2"},
    "params": {"env": "backoffice_churn", "sessions": 24, "stagger": 100.0, "lead": 6.0,
               "warmup_max": 5.0, "max_lag": 0.25, "settle": 8.0, "condition": "s1",
               "stress_seed": 71000, "draw_seed": 71000},
    "pools": [{"path": "/x/ds/sessions.json", "sha256": "p-ds"}],
    "sessions_sha256": "x0",
}  # fmt: skip
REG = registered(MANIFEST, POOL_HASHES, PLAN)


def differs(**over) -> list[str]:
    return registered_problems(MANIFEST | over, REG)


def params(**over) -> dict:
    return {"params": MANIFEST["params"] | over}


def test_a_registration_keeps_what_every_run_shares_and_the_plan():
    shared = {"commit", *REGISTERED_HASHES, "configs", "images", "params"}
    assert set(REG) == shared | {"pools", "runs"}
    assert set(REG["params"]) == set(REGISTERED_PARAMS)
    assert "condition" not in REG["params"] and "stress_seed" not in REG["params"]
    assert REG["runs"] == PLAN and REG["pools"] == POOL_HASHES
    assert registered_problems(MANIFEST, REG) == []
    # The registration survives being written and read.
    assert registered_problems(MANIFEST, json.loads(json.dumps(REG))) == []
    # Another run of the plan, with its own seed, pool and sessions, is one too. The
    # commit is not compared: committing the registration itself moves it.
    second = MANIFEST | {"run": "s5-s1-001", "commit": "later", "sessions_sha256": "x1"}
    second |= {"pools": [{"path": "/y/glm/sessions.json", "sha256": "p-glm"}]}
    second |= {"params": MANIFEST["params"] | {"stress_seed": 71001, "draw_seed": 71001}}
    assert registered_problems(second, REG) == []


def test_a_run_not_made_by_what_was_registered_is_named():
    assert "clean tree" in differs(dirty=True)[0]
    no_flag = {k: v for k, v in MANIFEST.items() if k != "dirty"}
    assert "clean tree" in registered_problems(no_flag, REG)[0]  # absent is not clean
    assert differs(source_sha256="t" * 64) == [
        "the source the image was built from is not the registered one"
    ]
    assert differs(driver_sha256="t" * 64) == ["the run driver is not the registered one"]
    assert differs(lock_sha256="t" * 64) == ["the lock file is not the registered one"]
    assert differs(configs=MANIFEST["configs"] | {"compose.yaml": "zz"}) == [
        "configs: compose.yaml is not the registered one"
    ]
    assert differs(configs={"compose.yaml": "c1"}) == [
        "configs: config/backend.yaml is not the registered one"
    ]
    assert "images: kernel" in differs(images=MANIFEST["images"] | {"kernel": "i9"})[0]
    assert "images: extra" in differs(images=MANIFEST["images"] | {"extra": "i3"})[0]
    assert differs(**params(max_lag=5.0)) == ["parameter max_lag is 5.0, registered 0.25"]
    assert "sessions" in differs(**params(sessions=4))[0]


def test_a_run_that_is_not_the_run_its_name_says_is_named():
    assert differs(run="s5-s1-999") == ["'s5-s1-999' is not a run of the plan"]
    assert differs(run="pt2-node-restart-0")[0].endswith("is not a run of the plan")
    assert differs(**params(condition="s0")) == ["condition is 's0', planned 's1'"]
    assert differs(**params(stress_seed=5)) == ["stress_seed is 5, planned 71000"]
    assert differs(**params(draw_seed=5)) == ["draw_seed is 5, planned 71000"]
    wrong_pool = "the run did not replay pool 'ds' and nothing else"
    glm = {"path": "/x/glm/sessions.json", "sha256": "p-glm"}
    assert differs(pools=[glm]) == [wrong_pool]  # a registered pool, but not this run's
    assert differs(pools=[*MANIFEST["pools"], glm]) == [wrong_pool]  # two pools drawn together
    assert differs(pools=MANIFEST["pools"] * 2) == [wrong_pool]
    assert differs(pools=[]) == [wrong_pool] and differs(pools=[None, "x"]) == [wrong_pool]
    assert differs(sessions_sha256="other") == [
        "the sessions replayed are not the ones the plan draws"
    ]
    # The name of one run with the seeds of another is neither.
    assert len(differs(run="s5-s1-001")) == 4


def test_a_sample_that_is_not_what_the_plans_runs_will_be_cannot_be_registered():
    sample = MANIFEST | {"run": "reg-sample"}
    ok = {"ok": True, "problems": []}
    assert sr.sample_problems(sample, ok, POOL_HASHES) == []
    assert "clean tree" in sr.sample_problems(sample | {"dirty": True}, ok, POOL_HASHES)[0]
    assert "integrity" in sr.sample_problems(sample, {"ok": False}, POOL_HASHES)[0]
    assert "integrity" in sr.sample_problems(sample, None, POOL_HASHES)[0]
    smoke = sample | {"params": sample["params"] | {"stagger": 3.0, "env": "backoffice"}}
    assert sr.sample_problems(smoke, ok, POOL_HASHES) == [
        "its env is 'backoffice', the plan's is 'backoffice_churn'",
        "its stagger is 3.0, the plan's is 100.0",
    ]
    assert "sessions" in sr.sample_problems(sample | params(sessions=4), ok, POOL_HASHES)[0]
    for pools in ([], [{"sha256": "nope"}], sample["pools"] * 2):
        assert sr.sample_problems(sample | {"pools": pools}, ok, POOL_HASHES) == [
            "it did not replay exactly one of the study's pools"
        ]
    assert "no images" in sr.sample_problems(sample | {"images": {}}, ok, POOL_HASHES)[0]
    assert "no configs" in sr.sample_problems(sample | {"configs": {}}, ok, POOL_HASHES)[0]


# Making the runs ----------------------------------------------------------------------


def _maker(out: Path, made: list[str], behave: dict[str, str]):
    """Stands in for the driver: writes a manifest (or not), returns an exit status."""

    def make(run: dict) -> int:
        made.append(run["run"])
        how = behave.get(run["run"], "ok")
        if how == "refused":
            return 1
        manifest = MANIFEST | {"run": run["run"], "pools": [{"sha256": POOL_HASHES[run["pool"]]}]}
        manifest["sessions_sha256"] = PLAN[run["run"]]["sessions_sha256"]
        manifest |= params(stress_seed=run["seed"], draw_seed=run["seed"])
        if how == "dirty":
            manifest["dirty"] = True
        (out / run["run"]).mkdir(parents=True)
        (out / run["run"] / "manifest.json").write_text(json.dumps(manifest))
        return 1 if how == "integrity" else 0

    return make


TWO = [r for r in sr.runs() if r["run"] in PLAN]


def test_go_makes_the_runs_that_do_not_exist_and_never_remakes_one(tmp_path):
    made: list[str] = []
    said: list[str] = []
    assert sr.go(TWO, tmp_path, REG, _maker(tmp_path, made, {}), say=said.append) == 0
    assert made == ["s5-s1-000", "s5-s1-001"] and said == ["s5-s1-000 exit=0", "s5-s1-001 exit=0"]
    assert sr.go(TWO, tmp_path, REG, _maker(tmp_path, made, {}), say=said.append) == 0
    assert len(made) == 2  # both folders exist: nothing is made again
    # A folder that exists is left alone whatever it holds, even nothing.
    shutil.rmtree(tmp_path / "s5-s1-001")
    (tmp_path / "s5-s1-001").mkdir()
    assert sr.go(TWO, tmp_path, REG, _maker(tmp_path, made, {}), say=said.append) == 0
    assert len(made) == 2


def test_go_stops_at_a_run_the_driver_refused_or_that_is_not_registered(tmp_path):
    for how, word in (("refused", "made no manifest"), ("dirty", "not a registered run")):
        out = tmp_path / how
        made: list[str] = []
        said: list[str] = []
        status = sr.go(TWO, out, REG, _maker(out, made, {"s5-s1-000": how}), say=said.append)
        assert status == 1 and made == ["s5-s1-000"]  # the second run is not spent
        assert said[-1].startswith("stopped:") and word in said[-1]


def test_a_run_that_fails_its_integrity_check_is_a_run_and_the_sequence_goes_on(tmp_path):
    made: list[str] = []
    said: list[str] = []
    make = _maker(tmp_path, made, {"s5-s1-000": "integrity"})
    assert sr.go(TWO, tmp_path, REG, make, say=said.append) == 0
    assert made == ["s5-s1-000", "s5-s1-001"] and said[0] == "s5-s1-000 exit=1"


def test_go_makes_at_most_the_limit(tmp_path):
    made: list[str] = []
    assert sr.go(TWO, tmp_path, REG, _maker(tmp_path, made, {}), limit=1, say=lambda _: None) == 0
    assert made == ["s5-s1-000"]


def test_go_does_not_start_without_a_registration(tmp_path):
    (why,) = sr.preflight(tmp_path / "registered.json", tmp_path)
    assert "nothing is registered" in why


def test_go_does_not_start_with_a_driver_lock_file_or_pool_that_is_not_registered(tmp_path):
    reg = dict(REG) | {"driver_sha256": "x", "lock_sha256": "y"}
    path = tmp_path / "registered.json"
    path.write_text(json.dumps(reg))
    why = " | ".join(sr.preflight(path, tmp_path))
    assert "scripts/rig_run.py is not the registered driver" in why
    assert "uv.lock is not the registered lock file" in why
    for pool in sr.POOLS:
        assert f"pool {pool} is missing or is not the registered pool" in why


# The analysis ---------------------------------------------------------------------------


def _run(tmp: Path, name: str, **over) -> Path:
    run = tmp / name
    shutil.copytree(FIXTURE, run)
    p = MANIFEST["params"] | {"env": "backoffice"}
    manifest = MANIFEST | {"schedule": [], "params": p} | over
    (run / "manifest.json").write_text(json.dumps(manifest))
    events = {"duration_s": 5.0, "integrity": {"ok": True, "problems": []}}
    (run / "events.json").write_text(json.dumps(events))
    return run


def test_the_analysis_sets_aside_a_run_whose_manifest_is_not_the_registered_one(tmp_path):
    reg = registered(MANIFEST | params(env="backoffice"), POOL_HASHES, PLAN)
    good, bad = _run(tmp_path, "s5-s1-000"), _run(tmp_path, "bad", source_sha256="t" * 64)
    runs, aside = sa.collect([good, bad], reg)
    assert [r["run"] for r in runs] == ["s5-s1-000"]
    assert aside == [
        {
            "run": "bad",
            "condition": "s1",
            "kind": "not registered",
            "why": [
                "the source the image was built from is not the registered one",
                "the folder is not named as its manifest says ('s5-s1-000')",
            ],
        }
    ]
    # A copy of a good run under another name is not a second run.
    copy = _run(tmp_path, "s5-s1-000-copy")
    runs, aside = sa.collect([good, copy], reg)
    assert [r["run"] for r in runs] == ["s5-s1-000"]
    assert aside[0]["why"] == ["the folder is not named as its manifest says ('s5-s1-000')"]
    assert len(sa.collect([good, bad])[0]) == 2  # without a registration both are scored


def test_the_studys_runs_are_never_scored_without_the_registration(tmp_path, monkeypatch):
    _run(tmp_path, "s5-s1-000")
    out = tmp_path / "a.json"
    argv = ["study5_analysis.py", "--runs", str(tmp_path), "--prefix", "s5-", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="--registered is required"):
        sa.main()
    assert not out.exists()
    reg = tmp_path / "registered.json"
    reg.write_text(json.dumps(registered(MANIFEST | params(env="backoffice"), POOL_HASHES, PLAN)))
    monkeypatch.setattr(sys, "argv", [*argv, "--registered", str(reg), "--min-runs", "1"])
    sa.main()
    artifact = json.loads(out.read_text())
    assert artifact["registered"] == "registered.json" and len(artifact["runs"]) == 1
    assert artifact["runs_set_aside"] == []


def test_a_pool_short_of_sessions_is_named_by_the_check(pools):
    assert (sr.POOL_SESSIONS, sr.MIN_POOL) == (72, 60)
    short = {**pools, "glm": _pool(2000, n=59)}
    bad = sr.check(short)
    assert bad == {"pool glm": ["it has 59 sessions, fewer than 60"]}
    assert sr.check({**pools, "glm": _pool(2000, n=60)}) == {}


def test_sealing_hashes_every_run_that_exists_and_names_the_ones_that_do_not(tmp_path):
    for name, text in (("s5-s1-000", "a"), ("s5-s0-000", "b")):
        (tmp_path / name / "witnesses").mkdir(parents=True)
        (tmp_path / name / "witnesses" / "env.jsonl").write_text(text)
    (tmp_path / "pt-none-0").mkdir()  # not a run of the plan: not sealed
    sealed = sr.seal(tmp_path)
    assert set(sealed["runs"]) == {"s5-s1-000", "s5-s0-000"}
    assert len(sealed["planned_and_not_made"]) == 478
    before = sealed["runs"]["s5-s1-000"]
    (tmp_path / "s5-s1-000" / "witnesses" / "env.jsonl").write_text("changed")
    after = sr.seal(tmp_path)["runs"]
    assert after["s5-s1-000"] != before and after["s5-s0-000"] == sealed["runs"]["s5-s0-000"]
