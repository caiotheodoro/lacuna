"""scripts/study5_record.py: a pool on disk, from a scripted stand-in for the model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import study5_record as rec  # noqa: E402
from test_rig_record import PlanFollower  # noqa: E402

from lacuna.envs.registry import BACKOFFICE_CHURN as ENV  # noqa: E402
from lacuna.rig.record import RECORD_SYSTEM  # noqa: E402
from lacuna.rig.sessions import load  # noqa: E402

META = {"pool": "t", "models": {"a": "m1", "b": "m2"}}


def _models(seed: int) -> dict:
    ep = ENV.make_episode(seed, 10)
    return {"a": PlanFollower(ep).complete, "b": PlanFollower(ep).complete}


def test_a_pool_is_sessions_transcripts_and_a_description(tmp_path):
    out = tmp_path / "t"
    pool = rec.record_pool(out, [3, 4, 5], 10, ENV, _models, 1.0, META)
    assert json.loads((out / "pool.json").read_text()) == pool
    sessions = load((out / "sessions.json").read_text())
    assert [(s.session, s.plane) for s in sessions] == [
        (f"s{seed}", plane) for seed in (3, 4, 5) for plane in ("a", "b")
    ]
    lines = [json.loads(x) for x in (out / "transcripts.jsonl").read_text().splitlines()]
    assert [x["seed"] for x in lines] == [3, 4, 5]
    assert set(lines[0]["transcripts"]) == {"a", "b"}
    assert pool["sessions_recorded"] == 3 and pool["failed_sessions"] == []
    assert pool["env"] == "backoffice_churn" and pool["models"] == META["models"]
    assert pool["prompt_sha256"] == hashlib.sha256(RECORD_SYSTEM.encode()).hexdigest()
    digest = hashlib.sha256((out / "sessions.json").read_bytes()).hexdigest()
    assert pool["sessions_sha256"] == digest
    for row, seed in zip(pool["sessions"], (3, 4, 5), strict=True):
        ep = ENV.make_episode(seed, 10)
        planned = sum(len(t.plan) for t in ep.tickets)
        assert row["seed"] == seed and sum(row["calls"].values()) == planned
        assert row["failed_calls"] == 0 and row["unfinished_tickets"] == []
        assert row["attempts"] == 1
        assert sum(row["actions"].values()) + row["reads"] == planned
        assert 0 < row["hidden_share"] < 1
    total = sum(sum(r["actions"].values()) for r in pool["sessions"])
    rows = sum(r["rows_in_end_state"] for r in pool["sessions"])
    assert pool["actions_recorded"] == total
    assert pool["hidden_share"] == round(1 - rows / total, 4)


def test_a_pool_is_never_overwritten(tmp_path):
    out = tmp_path / "t"
    rec.record_pool(out, [3], 10, ENV, _models, 1.0, META)
    before = (out / "sessions.json").read_bytes()
    with pytest.raises(FileExistsError):
        rec.record_pool(out, [4], 10, ENV, _models, 1.0, META)
    assert (out / "sessions.json").read_bytes() == before


def test_a_session_that_raises_is_tried_again_and_one_that_fails_twice_is_left_out(tmp_path):
    tries: dict[int, int] = {}

    def flaky(seed: int) -> dict:
        tries[seed] = tries.get(seed, 0) + 1
        if seed == 4 or (seed == 3 and tries[seed] == 1):

            def boom(messages, tools=None, tag=""):
                raise RuntimeError(f"provider said no to {seed}")

            return {"a": boom, "b": boom}
        return _models(seed)

    pool = rec.record_pool(tmp_path / "t", [3, 4, 5], 10, ENV, flaky, 1.0, META)
    assert tries == {3: 2, 4: 2, 5: 1}
    assert [r["seed"] for r in pool["sessions"]] == [3, 5]
    assert [r["attempts"] for r in pool["sessions"]] == [2, 1]
    assert pool["failed_sessions"] == [
        {"seed": 4, "errors": ["RuntimeError: provider said no to 4"] * 2}
    ]
    names = {s.session for s in load((tmp_path / "t" / "sessions.json").read_text())}
    assert names == {"s3", "s5"}


def test_recording_stops_before_the_next_session_when_told_to(tmp_path):
    done: list[int] = []

    def models(seed: int) -> dict:
        done.append(seed)
        return _models(seed)

    pool = rec.record_pool(
        tmp_path / "t", [3, 4, 5], 10, ENV, models, 1.0, META,
        stop=lambda: "budget spent" if len(done) >= 2 else "",
    )  # fmt: skip
    assert done == [3, 4]
    assert pool["stopped"] == "budget spent" and pool["sessions_recorded"] == 2
    assert pool["seeds_asked"] == [3, 4, 5]


def test_only_the_provider_that_needs_them_is_sent_a_session_header_and_the_projects_name():
    sent = rec.session_headers("opencode.ai", "pilot-ds", 50000)
    assert sent == {"User-Agent": "lacuna/0.0.0", "x-opencode-session": "lacuna-pilot-ds-50000"}
    assert rec.session_headers("api.opencode.ai", "p", 1)["x-opencode-session"] == "lacuna-p-1"
    for host in ("api.mistral.ai", "example.com", "notopencode.ai", ""):
        assert rec.session_headers(host, "pilot-ds", 50000) == {}
