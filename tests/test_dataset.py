"""The dataset builder's run mapping, the publication privacy gate, and the publish gates.

Planted secrets are assembled at runtime so this file never holds a
credential-shaped string itself (`make privacy-gate` scans it).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EPISODES = ROOT / "artifacts" / "episodes"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build = _load("build_dataset")
gate = _load("validate_dataset")
publish = _load("publish_dataset")


def _rows(run: str) -> list[dict]:
    return [json.loads(line) for line in (EPISODES / run / "episodes.jsonl").open()]


# --- run mapping -------------------------------------------------------------

# Pinned here, apart from the builder: a run consistently relabelled in RUNS
# would otherwise pass every check that compares the build with itself.
PHASES = {
    "confirm": "confirmatory",
    "confirm-supp": "excluded",
    "diag-7009": "excluded",
    "diag-batch": "excluded",
    "confirm-plus-supp": "excluded",
    "scripted-l0": "scripted harness check",
    "pilot-l0": "exploratory",
    "long-30": "exploratory",
    "long-60": "exploratory",
    "long-120": "exploratory",
    "narr-30": "exploratory",
    "transfer": "confirmatory (transfer, separately pre-registered)",
    "transfer-scripted": "scripted harness check",
    "transfer-scripted-24": "scripted harness check",
}


def test_phase_is_pinned_for_every_run():
    assert {run: meta["phase"] for run, meta in build.RUNS.items()} == PHASES


def test_the_validator_pins_the_same_phases_without_importing_them():
    assert {run: phase for run, phase in gate.EXPECTED_PHASES.items()} == PHASES
    import ast

    tree = ast.parse((SCRIPTS / "validate_dataset.py").read_text())
    from_builder = [
        a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "build_dataset"
        for a in node.names
    ]
    plain = [
        a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for a in node.names
        if a.name == "build_dataset"
    ]
    # Only the card renderer comes from the builder; every decision is pinned here.
    assert from_builder == ["card"] and not plain


def test_unknown_run_raises():
    with pytest.raises(build.UnknownRun, match="transfer-infra"):
        build.run_meta("transfer-infra")


def test_unknown_run_leaves_the_previous_build_intact(tmp_path, monkeypatch):
    (tmp_path / "artifacts" / "episodes" / "transfer-infra").mkdir(parents=True)
    monkeypatch.setattr(build, "ARTIFACTS", tmp_path / "artifacts")
    out = tmp_path / "dataset"
    out.mkdir()
    (out / "README.md").write_text(f"previous build for {build.REPO}\n")
    (out / "runs.jsonl").write_text("{}\n")
    with pytest.raises(build.UnknownRun):
        build.build(out)
    assert (out / "runs.jsonl").read_text() == "{}\n"
    assert not list(tmp_path.glob(".dataset-build-*"))


def test_every_recorded_run_is_mapped():
    for run in sorted(p.name for p in EPISODES.iterdir() if p.is_dir()):
        meta = build.run_meta(run)
        assert meta["phase"] in build.PHASES and meta["env"] in build.ENVS


def test_fill_cannot_contradict_the_summary():
    with pytest.raises(build.BuildError, match="tickets"):
        build.resolve("pilot-l0", build.RUNS["pilot-l0"], {"tickets": 30, "auditor_view": None})


def test_a_field_nobody_recorded_fails():
    with pytest.raises(build.BuildError, match="tickets"):
        build.resolve("long-120", build.RUNS["long-120"], {"tickets": None, "auditor_view": "full"})


@pytest.mark.parametrize("run", ["pilot-l0", "scripted-l0"])
def test_filled_ticket_count_is_what_the_ledgers_say(run):
    episodes = [(e["seed"], e["ledger"]) for e in _rows(run)]
    assert gate.infer_tickets(episodes, 12) == build.RUNS[run]["fill"]["tickets"] == 7


def test_caveats_carry_the_decisions_that_govern_each_run():
    for run in ("pilot-l0", "long-30", "long-60", "long-120"):
        caveat = build.RUNS[run]["caveat"]
        assert "D14" in caveat and "D20" in caveat, run
    assert "narration" in build.RUNS["narr-30"]["caveat"].lower()
    assert all(d in build.RUNS["confirm"]["caveat"] for d in ("D32", "D34", "D36"))


def test_supplement_is_the_pools_recorded_composition():
    pool = json.loads((EPISODES / "confirm-plus-supp" / "summary.json").read_text())
    want = tuple((c["run"], c["seed"]) for c in pool["composition"]["supplement"])
    assert build.SUPPLEMENT["picks"] == want
    assert build.SUPPLEMENT_PHASE == gate.SUPPLEMENT_PHASE
    assert all(build.RUNS[run]["phase"] == "excluded" for run, _ in want)


def test_an_unplaced_artifact_fails_the_build(tmp_path):
    for name in build.SHIPPED_RESULTS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}")
    build.check_results_accounted(tmp_path)
    (tmp_path / "thinning" / "new-pool.json").write_text("{}")
    with pytest.raises(build.BuildError, match="new-pool"):
        build.check_results_accounted(tmp_path)


def test_builder_and_validator_place_results_the_same_way():
    assert set(build.SHIPPED_RESULTS) == gate.SHIPPED_RESULTS
    assert set(build.NOT_SHIPPED) == gate.NOT_SHIPPED
    assert not gate.SHIPPED_RESULTS & gate.NOT_SHIPPED
    assert build.DIAGNOSTIC_RUNS == gate.DIAGNOSTIC_RUNS


def test_missing_receipt_is_unknown_spend_not_zero():
    runs = [{"run": "a", "spend_usd": 1.5}, {"run": "b", "spend_usd": None}]
    assert build.spend_summary(runs) == (1.5, ["b"])
    assert build.fmt_usd(None) == "unknown"


def test_request_ids_are_redacted_and_the_gate_catches_one_left_in():
    err = 'HTTPStatusError: 400 {"message":"thinking mode (request_id: ad634575-0538-42c2'
    clean = build.redact(err)
    assert clean.endswith("(request_id: <redacted>") and "thinking mode" in clean
    assert gate.privacy_findings(clean, "x")[0] == []
    found, _ = gate.privacy_findings(json.dumps({"error": err}), "runs.jsonl")
    assert any("request id" in f for f in found) and any("UUID" in f for f in found)


def test_the_auditor_is_split_out_of_records():
    row = _rows("long-30")[0]
    plan = build.plan_run("long-30", EPISODES / "long-30")
    rows, _ = build.build_run("long-30", EPISODES / "long-30", plan, with_transcripts=False)
    built = next(r for r in rows if r["seed"] == row["seed"])
    assert {r["witness"] for r in built["records"]} == {0, 1, 2}
    assert {r["witness"] for r in built["auditor_records"]} == {3}
    assert len(built["records"]) + len(built["auditor_records"]) == len(row["records"])
    assert "transcript" not in built


# --- privacy gate ------------------------------------------------------------

PLANTED = {
    "sk- key": "sk-" + "A1b2C3d4" * 3,
    "oc_sk_ key": "oc_" + "sk_" + "Zx9" * 6,
    "hf_ token": "hf" + "_" + "AbCdEf123456" * 3,
    "GitHub token": "gh" + "p_" + "a1B2c3D4" * 4,
    "Slack token": "xo" + "xb-" + "1234-5678-abcd",
    "AWS access key": "AS" + "IA" + "ABCDEFGH12345678",
    "Bearer token": "Bear" + "er " + "tok3n" * 5,
    "PEM block": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "JWT": "ey" + "JhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxIn0.sig",
    "provider request id": "request" + "_id: " + "db9d948e",
    "UUID": "6f1c2a7e" + "-1b2c-4d5e-9f00-112233445566",
}


@pytest.mark.parametrize("label", sorted(PLANTED))
def test_privacy_gate_catches_a_planted_credential(label):
    key = PLANTED[label]
    for text in (
        json.dumps({"content": f"use {key} here"}),
        # After a JSON newline escape the key is glued to the letter n.
        json.dumps({"content": "line one\n" + key}),
        # And doubly escaped, as inside a JSON-encoded string field.
        json.dumps({"transcript": json.dumps([{"content": "x\n" + key}])}),
    ):
        found, _ = gate.privacy_findings(text, "planted.jsonl")
        assert any(label in f for f in found), (label, text)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("IPv4 address", "upstream 10.20.30.40 refused"),
        ("local filesystem path", "/Us" + "ers/someone/x.json"),
        ("local filesystem path", "/ho" + "me/someone/x.json"),
        ("local filesystem path", "/ro" + "ot/.cache/x"),
        ("local filesystem path", "/pri" + "vate/tmp/x"),
        ("local filesystem path", "~" + "/.config/x"),
        # A JSON-escaped Windows path whose \\b used to be blanked before matching.
        ("Windows path", json.dumps({"p": "C:" + "\\Users\\bob\\notes.txt"})),
        ("Windows path", "C:" + "\\Users\\bob"),
        ("owner name", "written by " + "Ca" + "io"),
        ("owner name", "THEO" + "DORO lab"),
        ("owner name", "ma" + "theus@example.com"),
    ],
)
def test_privacy_gate_catches_paths_ips_and_names(label, text):
    found, _ = gate.privacy_findings(text, "planted")
    assert any(label in f for f in found), (label, text, found)


def test_the_publish_target_is_the_one_name_allowed():
    assert gate.privacy_findings(f"Hugging Face dataset {build.REPO}", "README.md")[0] == []


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


def test_check_privacy_fails_on_a_planted_file_and_on_no_files(tmp_path):
    assert gate.check_privacy(tmp_path)[1] is False
    (tmp_path / "episodes").mkdir()
    (tmp_path / "episodes" / "x.jsonl").write_text(json.dumps({"to": "bob@gmail.com"}) + "\n")
    name, ok, detail = gate.check_privacy(tmp_path)
    assert not ok and "gmail.com" in detail


# --- validation must not take the upload offline -----------------------------


def test_load_check_leaves_this_process_online(tmp_path):
    pytest.importorskip("datasets")
    hub = pytest.importorskip("huggingface_hub")
    (tmp_path / "episodes").mkdir()
    (tmp_path / "episodes" / "confirm.jsonl").write_text(json.dumps({"seed": 1}) + "\n")
    (tmp_path / "README.md").write_text(
        "---\nconfigs:\n- config_name: confirm\n  data_files: episodes/confirm.jsonl\n"
        "  default: true\n---\n\nbody\n"
    )
    before = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    name, ok, detail = gate.check_loads(tmp_path)
    # The toy config does not match the pinned features, so the check fails;
    # what matters is that it ran in a child and this process stayed online.
    assert "child failed" not in detail and "features differ" in detail
    assert {k: os.environ.get(k) for k in before} == before
    constants = importlib.reload(hub.constants)
    assert constants.HF_HUB_OFFLINE is False


# --- publish -----------------------------------------------------------------


def test_publish_defaults_to_a_private_dry_run():
    args = publish.parse_args([])
    assert args.dry_run and not args.yes_upload and args.private
    assert args.repo == "caiotheodoro/lacuna-capture-histories"


def test_dry_run_and_upload_cannot_both_be_asked_for():
    with pytest.raises(SystemExit):
        publish.parse_args(["--dry-run", "--yes-upload"])


@pytest.mark.parametrize(
    ("card", "head", "allowed"),
    [
        ("abc1234", "abc1234", True),
        ("abc1234", "def5678", False),
        ("abc1234-dirty", "abc1234-dirty", False),
        ("abc1234", "abc1234-dirty", False),
        ("unknown", "abc1234", False),
    ],
)
def test_upload_needs_a_card_built_from_the_clean_head(card, head, allowed):
    assert (publish.upload_refusal(card, head) is None) is allowed
