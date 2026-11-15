"""scripts/study5_analysis.py: which runs are scored, which are set aside, and the counts."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_analysis as sa  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "rig" / "run"


def make(tmp: Path, name: str, condition: str, ok: bool = True, seed: int = 0) -> Path:
    run = tmp / name
    shutil.copytree(FIXTURE, run)
    params = {"condition": condition, "stress_seed": seed, "env": "backoffice"}
    manifest = {"params": params, "schedule": [], "sessions_sha256": "x"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    problems = [] if ok else ["agent-a exited 1, expected 0"]
    events = {"duration_s": 5.0, "integrity": {"ok": ok, "problems": problems}}
    (run / "events.json").write_text(json.dumps(events))
    return run


def test_a_run_that_failed_integrity_or_cannot_be_read_is_set_aside_with_its_reason(tmp_path):
    good = make(tmp_path, "p-none-0", "none")
    bad = make(tmp_path, "p-none-1", "none", ok=False)
    crashed = make(tmp_path, "p-none-2", "none")
    events = json.loads((crashed / "events.json").read_text())
    del events["integrity"]
    (crashed / "events.json").write_text(json.dumps(events))
    empty = tmp_path / "p-none-3"
    empty.mkdir()
    runs, aside = sa.collect([good, bad, crashed, empty])
    assert [r["run"] for r in runs] == ["p-none-0"]
    assert aside == [
        {"run": "p-none-1", "condition": "none", "kind": "integrity",
         "why": ["agent-a exited 1, expected 0"]},
        {"run": "p-none-2", "condition": "none", "kind": "integrity",
         "why": ["no integrity result"]},
        {"run": "p-none-3", "condition": None, "kind": "unreadable",
         "why": "unreadable: FileNotFoundError"},
    ]  # fmt: skip


def test_conditions_are_scored_apart_and_runs_made_counts_the_ones_set_aside(tmp_path):
    dirs = [
        make(tmp_path, "p-none-0", "none"),
        make(tmp_path, "p-none-1", "none", seed=1),
        make(tmp_path, "p-blip-0", "node-blip"),
    ]
    runs, _ = sa.collect(dirs)
    out = sa.by_condition(runs, {"none": 5, "node-blip": 1}, min_runs=2)
    assert set(out) == {"none", "node-blip"}
    assert out["none"]["gates"]["runs_made"] == 5
    assert out["none"]["gates"]["runs_passing_integrity"] == 2
    assert out["none"]["gates"]["underpowered"] is False
    assert out["node-blip"]["gates"]["underpowered"] is True
    assert out["none"]["estimators"]["union_count"]["covered"] == 2
    # Baselines and floors are not compared with themselves or each other.
    assert "union_count" not in out["none"]["paired"]
    assert "seq_gap_tail" not in out["none"]["paired"]
    assert set(out["none"]["paired"]["strat_bic"]) == {"seq_gap_tail", "union_count"}
    # The exploratory breakdown is per condition, over every estimate.
    by_lost = out["none"]["exploratory"]["by_lost_by_both"]
    assert by_lost["0 to 12"]["runs"] == 2 and by_lost["12 or more"]["runs"] == 0
    assert by_lost["0 to 12"]["covered"]["union_count"] == 2
    assert all(b["runs"] == 0 for b in out["none"]["exploratory"]["by_outage_s"].values())
    # The bootstrap is seeded by the run's stress seed, so a run scores the same twice.
    again, _ = sa.collect(dirs)
    assert again == runs


def test_runs_of_one_pool_are_read_as_a_block_beside_the_condition(tmp_path):
    def pooled(name: str, pool: str, seed: int) -> Path:
        run = make(tmp_path, name, "s1", seed=seed)
        manifest = json.loads((run / "manifest.json").read_text())
        late = [{"at_s": 1.0}] if pool == "qw" else []
        pools = [{"path": f"/p/{pool}/sessions.json", "sha256": pool}]
        manifest |= {"pools": pools, "restarts_after_planned_end": late}
        (run / "manifest.json").write_text(json.dumps(manifest))
        return run

    dirs = [pooled("a", "ds", 0), pooled("b", "ds", 1), pooled("c", "qw", 2)]
    runs, _ = sa.collect(dirs)
    block = sa.by_condition(runs, {"s1": 3}, min_runs=1)["s1"]
    assert set(block["by_pool"]) == {"ds", "qw"}
    assert block["by_pool"]["ds"]["ll_sa_hom"]["runs"] == 2
    assert block["by_pool"]["qw"]["union_count"]["runs"] == 1
    assert block["runs_with_outage_past_last_call"] == 1
    # Scripted runs draw from no pool and have no block.
    plain, _ = sa.collect([make(tmp_path, "d", "none")])
    assert sa.by_condition(plain, {}, 1)["none"]["by_pool"] == {}
