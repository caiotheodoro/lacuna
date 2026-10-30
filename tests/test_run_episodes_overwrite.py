"""scripts/run_episodes.py refuses to replace an existing run unless --overwrite is given."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_episodes  # noqa: E402

ARGS = ["--episodes", "1", "--tickets", "3", "--seed0", "7", "--run", "tmp-run"]


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_episodes, "ROOT", tmp_path)
    return tmp_path / "artifacts" / "episodes" / "tmp-run"


def test_first_run_writes_and_second_refuses(tmp_root, capsys):
    run_episodes.main(ARGS)
    eps = tmp_root / "episodes.jsonl"
    before = eps.read_bytes()
    assert before
    with pytest.raises(SystemExit) as e:
        run_episodes.main(ARGS)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "already exists" in err and "--overwrite" in err
    assert eps.read_bytes() == before


def test_refusal_touches_nothing(tmp_root):
    tmp_root.mkdir(parents=True)
    (tmp_root / "episodes.jsonl").write_text("kept\n")
    with pytest.raises(SystemExit):
        run_episodes.main(ARGS)
    assert sorted(p.name for p in tmp_root.iterdir()) == ["episodes.jsonl"]
    assert (tmp_root / "episodes.jsonl").read_text() == "kept\n"


def test_overwrite_replaces(tmp_root):
    tmp_root.mkdir(parents=True)
    (tmp_root / "episodes.jsonl").write_text("old\n")
    run_episodes.main([*ARGS, "--overwrite"])
    assert (tmp_root / "episodes.jsonl").read_text() != "old\n"
    assert (tmp_root / "summary.json").exists()


def test_a_new_run_name_is_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr(run_episodes, "ROOT", tmp_path)
    (tmp_path / "artifacts" / "episodes" / "other").mkdir(parents=True)
    (tmp_path / "artifacts" / "episodes" / "other" / "episodes.jsonl").write_text("x\n")
    run_episodes.main(ARGS)
    assert (tmp_path / "artifacts" / "episodes" / "tmp-run" / "episodes.jsonl").exists()
