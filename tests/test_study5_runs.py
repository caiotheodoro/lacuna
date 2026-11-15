"""scripts/study5_runs.py: the plan of the study's runs, and what makes a run registered."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_analysis as sa  # noqa: E402
import study5_runs as sr  # noqa: E402

from lacuna.envs.registry import BACKOFFICE_CHURN  # noqa: E402
from lacuna.rig.manifest import REGISTERED_PARAMS, registered, registered_problems  # noqa: E402
from lacuna.rig.sessions import dump, planned_end, scripted_sessions  # noqa: E402
from lacuna.rig.stress import plan  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "rig" / "run"


def test_the_plan_is_420_runs_with_names_and_seeds_of_their_own():
    runs = sr.runs()
    assert len(runs) == 420 and runs == sr.runs()
    by = {c: [r for r in runs if r["condition"] == c] for c in ("s0", "s1", "s2")}
    assert {c: len(v) for c, v in by.items()} == {"s0": 60, "s1": 300, "s2": 60}
    assert len({r["run"] for r in runs}) == 420 and len({r["seed"] for r in runs}) == 420
    assert by["s1"][0]["run"] == "s5-s1-000" and {r["seed"] for r in by["s1"]} == set(
        range(71000, 71300)
    )
    # Every pool is replayed by a third of each condition's runs.
    for rows in by.values():
        counts = {p: sum(r["pool"] == p for r in rows) for p in sr.POOLS}
        assert len(set(counts.values())) == 1 and set(counts) == {"ds", "qw", "mix"}
    assert set(runs[0]) == {"run", "condition", "index", "seed", "pool"}


def test_the_order_spreads_every_condition_over_the_whole_sequence():
    runs = sr.runs()
    for start in range(0, 420, 42):  # ten stretches of the sequence
        chunk = [r["condition"] for r in runs[start : start + 42]]
        assert chunk.count("s1") == 30 and chunk.count("s0") == 6 and chunk.count("s2") == 6
    # Inside a condition the runs are made in index order.
    for c in ("s0", "s1", "s2"):
        idx = [r["index"] for r in runs if r["condition"] == c]
        assert idx == sorted(idx)


def test_a_runs_arguments_are_its_pool_its_seed_and_its_condition(tmp_path):
    run = next(r for r in sr.runs() if r["run"] == "s5-s2-004")
    args = sr.argv(run, tmp_path / "pools", tmp_path / "runs")
    assert args[1].endswith("scripts/rig_run.py")
    opts = dict(zip(args[2::2], args[3::2], strict=True))
    assert opts == {
        "--run": "s5-s2-004",
        "--out": str(tmp_path / "runs"),
        "--env": "backoffice_churn",
        "--sessions-file": str(tmp_path / "pools" / "qw" / "sessions.json"),
        "--sessions": "24",
        "--draw-seed": "72004",
        "--stagger": "100.0",
        "--condition": "s2",
        "--stress-seed": "72004",
    }


def _pools(tmp: Path, n: int = 36, tickets: int = 10) -> Path:
    for k, name in enumerate(sr.POOLS):
        (tmp / name).mkdir(parents=True)
        sessions = [
            s
            for seed in range(1000 * (k + 1), 1000 * (k + 1) + n)
            for s in scripted_sessions(
                seed, tickets, ("a", "b"), pace_s=1.0, env=BACKOFFICE_CHURN, timeline=True
            )
        ]
        (tmp / name / "sessions.json").write_text(dump(sessions))
    return tmp


def test_every_planned_run_can_be_made_from_pools_of_the_studys_shape(tmp_path):
    assert sr.check(_pools(tmp_path)) == {}


def test_a_run_the_driver_would_refuse_is_named_with_the_reason(tmp_path):
    # Pools too small to give 24 sessions, of sessions so short that kills come after.
    bad = sr.check(_pools(tmp_path, n=6, tickets=1))
    assert set(bad) == {r["run"] for r in sr.runs()}
    assert "the pool gives 6 sessions, the run needs 24" in bad["s5-s0-000"]
    killed = [name for name, why in bad.items() if any("is killed at" in w for w in why)]
    assert killed and all(not name.startswith("s5-s0-") for name in killed)
    # The reason quotes the schedule and the sessions the run would have.
    run = next(r for r in sr.runs() if r["run"] == killed[0])
    pool = sr.load(sr.pool_file(tmp_path, run["pool"]).read_text())
    end = planned_end(sr.draw(pool, 24, run["seed"]), sr.STAGGER)
    late = [e for e in plan(run["condition"], run["seed"]) if e.action == "kill" and e.at_s >= end]
    assert late and f"{late[0].service} is killed at {late[0].at_s} s" in " ".join(bad[killed[0]])


# What makes a run a registered run ------------------------------------------------

MANIFEST = {
    "commit": "abc",
    "dirty": False,
    "source_sha256": "s" * 64,
    "configs": {"compose.yaml": "c1", "config/backend.yaml": "c2"},
    "images": {"kernel": "i1", "backend": "i2"},
    "params": {"env": "backoffice_churn", "sessions": 24, "stagger": 100.0, "lead": 6.0,
               "warmup_max": 5.0, "max_lag": 0.25, "settle": 8.0, "condition": "s1",
               "stress_seed": 71000, "draw_seed": 71000},
    "pools": [{"path": "/x/ds/sessions.json", "sha256": "p-ds"}],
}  # fmt: skip
POOL_HASHES = {"ds": "p-ds", "qw": "p-qw", "mix": "p-mix"}


def test_a_registration_keeps_what_every_run_shares_and_not_what_differs_by_run():
    reg = registered(MANIFEST, POOL_HASHES)
    assert set(reg) == {"commit", "source_sha256", "configs", "images", "params", "pools"}
    assert set(reg["params"]) == set(REGISTERED_PARAMS)
    assert "condition" not in reg["params"] and "stress_seed" not in reg["params"]
    assert registered_problems(MANIFEST, reg) == []
    other = MANIFEST | {"params": MANIFEST["params"] | {"condition": "s2", "stress_seed": 9},
                        "pools": [{"path": "/y/qw/sessions.json", "sha256": "p-qw"}]}  # fmt: skip
    assert registered_problems(other, reg) == []


def test_each_way_a_run_can_differ_from_the_registration_is_named():
    reg = registered(MANIFEST, POOL_HASHES)

    def differs(**over) -> list[str]:
        return registered_problems(MANIFEST | over, reg)

    assert "uncommitted" in differs(dirty=True)[0]
    assert "source" in differs(source_sha256="t" * 64)[0]
    assert differs(configs=MANIFEST["configs"] | {"compose.yaml": "zz"}) == [
        "configs: compose.yaml is not the registered one"
    ]
    assert differs(configs={"compose.yaml": "c1"}) == [
        "configs: config/backend.yaml is not the registered one"
    ]
    assert "images: kernel" in differs(images=MANIFEST["images"] | {"kernel": "i9"})[0]
    assert "images: extra" in differs(images=MANIFEST["images"] | {"extra": "i3"})[0]
    slow = differs(params=MANIFEST["params"] | {"max_lag": 5.0})
    assert slow == ["parameter max_lag is 5.0, registered 0.25"]
    assert "sessions" in differs(params=MANIFEST["params"] | {"sessions": 4})[0]
    assert "no registered pool" in differs(pools=[])[0]
    stray = differs(pools=[{"path": "/x/other.json", "sha256": "nope"}])
    assert stray == ["pool /x/other.json is not a registered pool"]


def test_the_analysis_sets_aside_a_run_whose_manifest_is_not_the_registered_one(tmp_path):
    def make(name: str, **over) -> Path:
        run = tmp_path / name
        shutil.copytree(FIXTURE, run)
        params = MANIFEST["params"] | {"env": "backoffice", "condition": "s0", "stress_seed": 0}
        manifest = MANIFEST | {"schedule": [], "sessions_sha256": "x", "params": params} | over
        (run / "manifest.json").write_text(json.dumps(manifest))
        events = {"duration_s": 5.0, "integrity": {"ok": True, "problems": []}}
        (run / "events.json").write_text(json.dumps(events))
        return run

    reg = registered(MANIFEST | {"params": MANIFEST["params"] | {"env": "backoffice"}}, POOL_HASHES)
    good, bad = make("good"), make("bad", source_sha256="t" * 64)
    runs, aside = sa.collect([good, bad], reg)
    assert [r["run"] for r in runs] == ["good"]
    assert aside == [
        {"run": "bad", "why": ["the source the image was built from is not the registered source"]}
    ]
    # Without a registration both are scored.
    assert len(sa.collect([good, bad])[0]) == 2
