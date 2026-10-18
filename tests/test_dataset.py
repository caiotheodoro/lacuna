"""The dataset builder's run mapping, the publication privacy gate, and the dry-run default.

Planted secrets are assembled at runtime so this file never holds a
key-shaped string itself (`make privacy-gate` scans it).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build = _load("build_dataset")
gate = _load("validate_dataset")
publish = _load("publish_dataset")


# --- run mapping -------------------------------------------------------------


def test_unknown_run_raises():
    with pytest.raises(build.UnknownRun, match="transfer-infra"):
        build.run_meta("transfer-infra")


def test_unknown_run_on_disk_fails_before_anything_is_written(tmp_path, monkeypatch):
    (tmp_path / "artifacts" / "episodes" / "transfer-infra").mkdir(parents=True)
    monkeypatch.setattr(build, "ARTIFACTS", tmp_path / "artifacts")
    out = tmp_path / "dataset"
    with pytest.raises(build.UnknownRun):
        build.build(out)
    assert not (out / "episodes").exists()


def test_every_recorded_run_is_mapped():
    for run in sorted(p.name for p in (ROOT / "artifacts" / "episodes").iterdir() if p.is_dir()):
        meta = build.run_meta(run)
        assert meta["phase"] in build.PHASES and meta["env"] in build.ENVS


def test_fill_cannot_contradict_the_summary():
    with pytest.raises(build.BuildError, match="tickets"):
        build.resolve("pilot-l0", build.RUNS["pilot-l0"], {"tickets": 30, "auditor_view": None})


def test_a_field_nobody_recorded_fails():
    with pytest.raises(build.BuildError, match="tickets"):
        build.resolve("long-120", build.RUNS["long-120"], {"tickets": None, "auditor_view": "full"})


def test_caveats_carry_the_decisions_that_govern_each_run():
    for run in ("pilot-l0", "long-30", "long-60", "long-120"):
        caveat = build.RUNS[run]["caveat"]
        assert "D14" in caveat and "D20" in caveat, run
    assert "narration" in build.RUNS["narr-30"]["caveat"].lower()
    assert build.RUNS["confirm"]["phase"] == "confirmatory"
    assert "D30" in build.RUNS["confirm"]["caveat"] and "D32" in build.RUNS["confirm"]["caveat"]
    assert build.RUNS["scripted-l0"]["phase"] == "scripted"


# --- privacy gate ------------------------------------------------------------

PLANTED_KEYS = {
    "sk- key": "sk-" + "A1b2C3d4" * 3,
    "oc_sk_ key": "oc_" + "sk_" + "Zx9" * 6,
    "Bearer token": "Bearer " + "tok3n" * 5,
    "AWS access key": "AKIA" + "ABCDEFGH12345678",
}


@pytest.mark.parametrize("label", sorted(PLANTED_KEYS))
def test_privacy_gate_catches_a_planted_key(label):
    key = PLANTED_KEYS[label]
    for text in (
        json.dumps({"content": f"use {key} here"}),
        # After a JSON newline escape the key is glued to the letter n.
        json.dumps({"content": "line one\n" + key}),
        # And doubly escaped, as inside the JSON-encoded transcript field.
        json.dumps({"transcript": json.dumps([{"content": "x\n" + key}])}),
    ):
        found, _ = gate.privacy_findings(text, "planted.jsonl")
        assert any(label in f for f in found), (label, text)


@pytest.mark.parametrize(
    "address", ["alice@gmail.com", "c1@example.com.evil.io", "c3@example.orgxx", "ops@corp.io"]
)
def test_privacy_gate_catches_a_non_example_email(address):
    found, _ = gate.privacy_findings(json.dumps({"to": address}), "planted.jsonl")
    assert found and "outside the example domains" in found[0]


def test_example_addresses_pass_and_the_scripted_redo_suffix_is_counted():
    text = json.dumps(
        ["c1@example.com", "c2.new@example.org", "a@mail.example.net", "c3.new@example.orgx"]
    )
    found, redo = gate.privacy_findings(text, "ok.jsonl")
    assert found == [] and redo == 1


def test_privacy_gate_catches_a_local_path():
    found, _ = gate.privacy_findings("written to " + "/Us" + "ers/someone/x.json", "card")
    assert any("local filesystem path" in f for f in found)


def test_check_privacy_fails_on_a_planted_file_and_on_no_files(tmp_path):
    assert gate.check_privacy(tmp_path)[1] is False
    (tmp_path / "episodes").mkdir()
    (tmp_path / "episodes" / "x.jsonl").write_text(json.dumps({"to": "bob@gmail.com"}) + "\n")
    name, ok, detail = gate.check_privacy(tmp_path)
    assert not ok and "gmail.com" in detail


# --- publish -----------------------------------------------------------------


def test_publish_defaults_to_a_private_dry_run():
    args = publish.parse_args([])
    assert args.dry_run and not args.yes_upload and args.private
    assert args.repo == "caiotheodoro/lacuna-capture-histories"


def test_dry_run_and_upload_cannot_both_be_asked_for():
    with pytest.raises(SystemExit):
        publish.parse_args(["--dry-run", "--yes-upload"])
