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
    "bypass-llm": "confirmatory (bypass, separately pre-registered)",
    "bypass-l1": "pilot (bypass L1 organism, pre-registered, no verdict)",
    "bypass-scripted-10": "scripted harness check",
    "bypass-scripted-30": "scripted harness check",
    "s4-neutral": "confirmatory (study 4, separately pre-registered)",
    "s4-l2": "confirmatory (study 4, separately pre-registered)",
    "s4-qwen-backoffice": "confirmatory (study 4, separately pre-registered)",
    "s4-qwen-bypass": "confirmatory (study 4, separately pre-registered)",
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
    err = 'HTTPStatusError: 400 {"message":"thinking mode (request_id: 0a1b2c3d-4e5f-4a6b'
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
    "provider request id": "request" + "_id: " + "5e6f7a8b",
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


# --- scripts/loss_sim.py, shipped in the dataset ------------------------------

loss_sim = _load("loss_sim")


def _toy_rows() -> list[dict]:
    def rec(witness, oracle, op="send_email"):
        return {"witness": witness, "op": op, "resource": "r", "value": "v", "oracle": oracle}

    ledger = [{"id": i, "op": "x", "resource": "r", "value": "v", "step": i} for i in range(4)]
    records = [
        rec(0, 0),
        rec(1, 0),
        rec(2, 0),  # id 0: all three
        rec(0, 1),
        rec(1, 1),  # id 1: trace and gateway
        rec(2, 2, op="update_customer"),  # id 2: env only, outside the financial view
        rec(0, 99),  # an id that is not in the ledger
    ]
    return [{"ledger": ledger, "records": records, "n_true": 4}]


def test_loss_sim_without_loss_counts_the_complete_histories():
    out = loss_sim.thin(_toy_rows())
    assert out["n_true"] == 4 and out["unseen"] == 1
    assert out["cells"] == {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1}


def test_loss_sim_env_view_filters_only_the_env_witness():
    out = loss_sim.thin(_toy_rows(), env_view="financial")
    assert out["cells"] == {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1} and out["unseen"] == 2
    named = loss_sim.thin(_toy_rows(), env_view=["update_customer"])
    assert named["cells"][4] == 1 and named["cells"][7] == 0 and named["cells"][3] == 2


def test_loss_sim_total_loss_leaves_only_the_env_witness():
    out = loss_sim.thin(_toy_rows(), drop={0: 1.0, 1: 1.0})
    assert out["cells"] == {1: 0, 2: 0, 3: 0, 4: 2, 5: 0, 6: 0, 7: 0}
    assert sum(out["cells"].values()) + out["unseen"] == out["n_true"]


def test_loss_sim_is_reproducible_from_its_seed():
    rows = _toy_rows() * 50
    a = loss_sim.thin(rows, drop={0: 0.5, 1: 0.5}, seed=7)
    b = loss_sim.thin(rows, drop={0: 0.5, 1: 0.5}, seed=7)
    c = loss_sim.thin(rows, drop={0: 0.5, 1: 0.5}, seed=8)
    assert a == b and a != c


def test_loss_sim_stands_alone():
    import ast

    tree = ast.parse((SCRIPTS / "loss_sim.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) and "lacuna" not in imported


# --- the bypass runs (D61, D63) --------------------------------------------------

BYPASS_RUNS = ("bypass-llm", "bypass-l1", "bypass-scripted-10", "bypass-scripted-30")


def test_the_bypass_neutral_pool_ships_as_bypass_on_both_sides():
    assert build.SHIPPED_AS == gate.SHIPPED_AS == {"bypass-llm": "bypass"}
    assert build.ship_name("bypass-llm") == "bypass" and build.ship_name("bypass-l1") == "bypass-l1"
    assert gate.source_run("bypass") == "bypass-llm" and gate.source_run("confirm") == "confirm"


def test_the_bypass_registration_is_pinned_with_its_freeze_and_tag():
    assert "PREREG-BYPASS.md" in build.SHIPPED_DOCS and "PREREG-BYPASS.md" in gate.SHIPPED_DOCS
    assert gate.PREREGISTRATIONS["PREREG-BYPASS.md"][0] == "c451bc1"
    assert (
        build.PREREG_TAGS
        == gate.PREREG_TAGS
        == {
            "PREREG-BYPASS.md": "prereg-bypass",
            "PREREG-STUDY4.md": "prereg-study4",
        }
    )


@pytest.mark.parametrize("run", BYPASS_RUNS)
def test_the_builder_and_the_validator_derive_the_same_bypassed_actions(run):
    for e in _rows(run):
        by_transcript = build.bypassed_ids(run, "backoffice_bypass", e)
        flags = gate.bypassed_flags(e, "backoffice_bypass")
        assert by_transcript == {x["id"] for x, b in zip(e["ledger"], flags, strict=True) if b}


@pytest.mark.parametrize("run", ["confirm", "transfer"])
def test_nothing_is_bypassed_outside_the_bypass_env(run):
    e = _rows(run)[0]
    assert build.bypassed_ids(run, "backoffice", e) == set()
    assert not any(gate.bypassed_flags(e, "backoffice"))


def test_a_bypass_episode_without_its_transcript_fails_the_build():
    e = {**_rows("bypass-llm")[0], "transcript": None}
    with pytest.raises(build.BuildError, match="without its transcript"):
        build.bypassed_ids("bypass-llm", "backoffice_bypass", e)


def test_a_ledger_step_that_is_not_its_call_fails_the_build():
    e = _rows("bypass-scripted-10")[0]
    e["ledger"] = [{**x, "step": x["step"] + 1} for x in e["ledger"]]
    with pytest.raises(build.BuildError, match="step"):
        build.bypassed_ids("bypass-scripted-10", "backoffice_bypass", e)


def test_the_bypass_results_are_placed_to_ship_and_the_scripted_ones_are_not():
    shipped = {n for n in build.SHIPPED_RESULTS if "bypass" in n}
    assert shipped == {
        "robustness/bypass-verdicts.json",
        "robustness/bypass-analysis-bypass-llm.json",
        "robustness/bypass-analysis-bypass-l1.json",
    }
    assert "robustness/bypass-analysis-confirm.json" in build.NOT_SHIPPED
    assert "robustness/bypass-verdicts-scripted.json" in build.NOT_SHIPPED


def test_the_bypass_caveats_cite_their_decisions():
    for config, refs in (("bypass", ("D61", "D63")), ("bypass-l1", ("D63",))):
        run = gate.source_run(config)
        caveat = build.RUNS[run]["caveat"]
        assert all(ref in caveat for ref in refs), config
        assert set(gate.CAVEAT_REFS[config]) >= set(refs)


# --- study 5 (D89 to D101): placement, the docs the gate refuses, the rows -------------

ARTIFACTS = ROOT / "artifacts"
S5_SHIPPED = (
    "robustness/study5-verdicts.json",
    "study5/registered.json",
    "study5/runs-sha256.json",
    "study5/pilot-summary.json",
    "study5/pilot-recorded-summary.json",
    "study5/posthoc.json",
)
S5_KINDS = (
    "late_call",
    "originator_not_ready",
    "clock_offset",
    "driver_stopped",
    "no_access_line_delivered",
)
S5_ESTIMATORS = (
    "strat_bic",
    "ll_bic",
    "ll_average",
    "ll_bic_boot",
    "ll_bic_hom",
    "chapman_te",
    "ll_sa_hom",
    "chapman_sa",
    "union_count",
    "union_bound",
    "trust_trace",
    "ll_sa_hom_sig",
    "seq_gap",
    "seq_gap_tail",
)


def _json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())


def test_gitignored_local_files_are_not_results(tmp_path):
    for name in build.SHIPPED_RESULTS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}")
    raw = tmp_path / "rig" / "runs" / "s5-s1-000"
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text("{}")
    (tmp_path / "run.log").write_text("x")
    build.check_results_accounted(tmp_path)
    assert not [n for n in build.result_artifacts(tmp_path) if n.startswith("rig/runs/")]
    # Only what .gitignore names: a file beside the raw runs is a result and must be placed.
    (tmp_path / "rig" / "runs.json").write_text("{}")
    with pytest.raises(build.BuildError, match="rig/runs.json"):
        build.check_results_accounted(tmp_path)


def test_what_the_builder_skips_is_what_gitignore_ignores():
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert build.LOCAL_ONLY_DIRS == gate.LOCAL_ONLY_DIRS == ("rig/runs",)
    assert all(f"artifacts/{d}/" in ignored for d in build.LOCAL_ONLY_DIRS)
    assert "artifacts/*.log" in ignored
    assert build.is_local_only("rig/runs/s5-s1-000/events.json") and build.is_local_only("a.log")
    assert not build.is_local_only("rig/checkpoint.json")
    assert "gitignore" in build.is_local_only.__doc__ and "rig/runs" in build.is_local_only.__doc__


def test_every_study5_artifact_is_placed_and_six_ship():
    names = [n for n in build.result_artifacts(ARTIFACTS) if n.split("/")[0] != "thinning"]
    study5 = [n for n in names if n.startswith(("study5/", "rig/", "rig-spikes/")) or "study5" in n]
    # 48 until D112 added the later post hoc artifact and the machine's power log.
    assert len(study5) == 50
    assert {n for n in study5 if n in build.SHIPPED_RESULTS} == set(S5_SHIPPED)
    assert all(
        n in build.NOT_SHIPPED and build.NOT_SHIPPED[n].strip()
        for n in set(study5) - set(S5_SHIPPED)
    )
    assert set(S5_SHIPPED) <= gate.SHIPPED_RESULTS
    # What ships passes the gates as it is: nothing redacted, no gate loosened.
    for name in S5_SHIPPED:
        text = (ARTIFACTS / name).read_text()
        assert gate.privacy_findings(text, name)[0] == [], name
        assert gate.reasoning_findings(text) == [], name


def test_a_study5_file_said_to_be_refused_by_the_gate_is_refused():
    said = [n for n, why in build.NOT_SHIPPED.items() if "study5" in n or n.startswith("rig")]
    refused = [n for n in said if "privacy gate refuses" in build.NOT_SHIPPED[n]]
    assert "study5/confirm.json" in refused and len(refused) == 12
    for name in refused:
        assert gate.privacy_findings((ARTIFACTS / name).read_text(), name)[0], name
    found = " ".join(gate.privacy_findings(_json_text("study5/confirm.json"), "x")[0])
    assert all(k in found for k in ("local filesystem path", "owner name", "URL"))
    for pool in build.STUDY5_POOLS:
        name = f"study5/pools/{pool}/transcripts.jsonl"
        assert "transcript" in build.NOT_SHIPPED[name] and "D101" in build.NOT_SHIPPED[name]
        assert "D101" in build.NOT_SHIPPED[f"study5/pools/{pool}/sessions.json"]
    assert tuple(build.STUDY5_POOLS) == tuple(gate.STUDY5_POOLS)


def _json_text(name: str) -> str:
    return (ARTIFACTS / name).read_text()


def test_the_docs_the_gate_refuses_are_not_shipped():
    assert set(build.NOT_SHIPPED_DOCS) == set(gate.NOT_SHIPPED_DOCS)
    assert set(gate.NOT_SHIPPED_DOCS) == {"DECISIONS.md", "PREREG-STUDY5.md"}
    assert not set(gate.NOT_SHIPPED_DOCS) & (set(build.SHIPPED_DOCS) | set(gate.SHIPPED_DOCS))
    for doc in gate.NOT_SHIPPED_DOCS:
        found = gate.privacy_findings((ROOT / "docs" / doc).read_text(), f"docs/{doc}")[0]
        assert any("hostname" in f for f in found), doc
        assert "privacy gate" in build.NOT_SHIPPED_DOCS[doc]
    for doc in gate.SHIPPED_DOCS:
        assert gate.privacy_findings((ROOT / "docs" / doc).read_text(), f"docs/{doc}")[0] == []


def test_the_study5_registration_is_pinned_where_it_stays():
    import hashlib

    sha = hashlib.sha256((ROOT / "docs" / "PREREG-STUDY5.md").read_bytes()).hexdigest()
    assert gate.STUDY5_PREREG_SHA256 == sha
    assert (gate.STUDY5_PREREG, gate.STUDY5_FREEZE) == ("PREREG-STUDY5.md", "a77f6ec")
    assert build.STUDY5_PREREG_TAG == gate.STUDY5_PREREG_TAG == "prereg-study5"
    assert gate.FREEZES["PREREG-STUDY5.md"] == "a77f6ec"
    if not _shallow():  # a shallow clone, as CI's is, has no history to read the freeze from
        assert build.freeze_commit("PREREG-STUDY5.md") == "a77f6ec"
    assert _json("study5/registered.json")["commit"].startswith(gate.STUDY5_FREEZE)


def test_set_aside_reasons_become_a_fixed_vocabulary():
    assert tuple(k for k, _ in build.S5_SET_ASIDE) == gate.S5_SET_ASIDE_KINDS == S5_KINDS
    late = "agent-a made a call 0.401 s late (limit 0.25 s)"
    ready = "agent-b was ready 2.782 s after the run's start"
    # vocabulary order, each kind once, whatever the order and number of the texts
    assert build.set_aside_kinds([ready, late, late]) == ["late_call", "originator_not_ready"]
    stopped = "the run driver stopped: RuntimeError: dock" + "er wait\n/Us" + "ers/someone/x.sock"
    assert build.set_aside_kinds([stopped]) == ["driver_stopped"]
    with pytest.raises(build.BuildError, match="no kind"):
        build.set_aside_kinds(["a witness file is missing"])
    with pytest.raises(build.BuildError, match="no reason"):
        build.set_aside_kinds([])
    counts = dict.fromkeys(S5_KINDS, 0)
    for entry in _json("study5/confirm.json")["runs_set_aside"]:
        kinds = build.set_aside_kinds(entry["why"])
        assert kinds == gate.set_aside_kinds(entry["why"])
        for kind in kinds:
            counts[kind] += 1
    assert list(counts.values()) == [26, 6, 4, 3, 2]  # D92


def test_an_infinite_value_ships_as_null():
    assert build.finite_or_none("inf") is None and gate.finite_or_none("inf") is None
    for fn in (build.finite_or_none, gate.finite_or_none):
        five = fn(5)
        assert five == 5.0 and type(five) is float and fn(3.25) == 3.25
        for bad in ("-inf", "nan", "5", None, float("inf"), float("nan"), True):
            with pytest.raises((build.BuildError, ValueError)):
                fn(bad)


@pytest.fixture(scope="module")
def s5_rows() -> list[dict]:
    return build.study5_rows(ARTIFACTS)


def test_study5_rows_are_one_per_planned_run_in_the_plans_order(s5_rows):
    plan = _json("study5/registered.json")["runs"]
    assert [r["run"] for r in s5_rows] == list(plan) and len(s5_rows) == 420
    assert {r["phase"] for r in s5_rows} == {build.STUDY5_PHASE} == {gate.STUDY5_PHASE}
    assert build.STUDY5_PHASE == "confirmatory (study 5, separately pre-registered)"
    by = {c: [r for r in s5_rows if r["condition"] == c] for c in ("s0", "s1", "s2")}
    assert [len(by[c]) for c in by] == [60, 300, 60]
    assert [sum(r["in_test"] for r in by[c]) for c in by] == [58, 277, 51]
    for r in s5_rows:
        assert (r["condition"], r["pool"], r["seed"], r["sessions_sha256"]) == tuple(
            plan[r["run"]][k] for k in ("condition", "pool", "seed", "sessions_sha256")
        )


def test_a_run_set_aside_has_its_kinds_of_reason_and_no_estimate(s5_rows):
    aside = [r for r in s5_rows if not r["in_test"]]
    assert len(aside) == 34 and {r["set_aside_kind"] for r in aside} == {"integrity"}
    measured = (
        "actions",
        "actions_by_plane",
        "tables",
        "union_count",
        "share_in_no_log",
        "missed_share",
        "schedule",
        "span_relay_outage_s",
        "estimates",
    )
    for r in aside:
        assert r["set_aside_reasons"] and set(r["set_aside_reasons"]) <= set(S5_KINDS)
        assert all(r[k] is None for k in measured), r["run"]
    for r in s5_rows:
        if r["in_test"]:
            # null, as the kind is: a run in the test was not set aside, which is not the
            # same as set aside for no reason. The loader takes a null list.
            assert r["set_aside_kind"] is None and r["set_aside_reasons"] is None
            assert all(r[k] is not None for k in measured if k != "span_relay_outage_s")
    # The reason texts stay behind: one holds a local path. What ships passes the gate.
    text = "".join(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in s5_rows)
    assert gate.privacy_findings(text, "s5-runs.jsonl")[0] == []
    assert "late (limit" not in text and "dock" + "er" not in text


def test_a_run_in_the_test_has_what_the_analysis_scored(s5_rows):
    source = {r["run"]: r for r in _json("study5/confirm.json")["runs"]}
    row = next(r for r in s5_rows if r["run"] == "s5-s1-000")
    src = source["s5-s1-000"]
    assert (row["actions"], row["actions_by_plane"]) == (src["actions"], src["actions_by_plane"])
    assert row["tables"] == src["tables"] and row["union_count"] == src["union_count"]
    assert row["share_in_no_log"] == src["union_missed_share"]
    assert row["schedule"] == src["schedule"]
    assert row["span_relay_outage_s"] == src["span_relay_outage_s"]
    assert tuple(row["estimates"]) == S5_ESTIMATORS == gate.S5_ESTIMATORS
    e, s = row["estimates"]["ll_sa_hom"], src["estimates"]["ll_sa_hom"]
    scored = {k: v for k, v in e.items() if k != "nonconverged"}
    assert scored == {k: (float(s[k]) if k in ("point", "lo", "hi") else s[k]) for k in scored}
    assert list(e) == ["point", "lo", "hi", "covers", "bounded", "error", "nonconverged"]
    # Cell order: bit 0 spans, bit 1 access, bit 2 the end state. The share each log
    # missed, which the analysis computes another way, says so for every plane of every run.
    for r in s5_rows:
        if not r["in_test"]:
            continue
        assert sum(sum(t) for t in r["tables"].values()) == r["union_count"]
        for plane, table in r["tables"].items():
            assert len(table) == 8 and table[0] == 0
            for bit, log in enumerate(("spans", "access", "env")):
                seen = sum(n for i, n in enumerate(table) if i >> bit & 1)
                assert (
                    round(1 - seen / r["actions_by_plane"][plane], 4)
                    == (r["missed_share"][plane][log])
                ), (r["run"], plane, log)


def test_no_infinity_reaches_the_rows(s5_rows):
    source = {r["run"]: r for r in _json("study5/confirm.json")["runs"]}
    nulls = 0
    for r in s5_rows:
        for name, e in (r["estimates"] or {}).items():
            src = source[r["run"]]["estimates"][name]
            for key in ("point", "hi"):
                assert (e[key] is None) == (src[key] == "inf"), (r["run"], name, key)
                nulls += e[key] is None
            assert type(e["lo"]) is float and (e["error"] is None) == (e["point"] is None)
            assert e["bounded"] == (e["point"] is not None and e["hi"] is not None)
    assert nulls == 22 + 452
    odd = next(r for r in s5_rows if r["run"] == "s5-s0-021")["estimates"]["ll_sa_hom"]
    assert odd["point"] is None and odd["hi"] is None and odd["covers"] and not odd["bounded"]
    # A null point is an infinite estimate or a fit that gave no answer: the flag says which.
    failed = [
        (r["run"], name)
        for r in s5_rows
        for name, e in (r["estimates"] or {}).items()
        if e["nonconverged"]
    ]
    assert failed == [("s5-s0-021", "ll_sa_hom"), ("s5-s0-021", "ll_sa_hom_sig")]
    assert all(
        type(e["nonconverged"]) is bool and not (e["nonconverged"] and e["point"] is not None)
        for r in s5_rows
        for e in (r["estimates"] or {}).values()
    )


def test_builder_and_validator_derive_the_same_study5_rows(s5_rows):
    assert gate.study5_expected_rows(ARTIFACTS) == s5_rows
    assert [list(r) for r in s5_rows] == [[f["name"] for f in gate.S5_FEATURES]] * len(s5_rows)
    for r in s5_rows:
        assert gate.row_type_problems(r, gate.S5_FEATURES, nullable=gate.S5_NULLABLE) == []
    # the episode and run schemas keep their own, narrower, list of nullable fields
    assert not gate.S5_NULLABLE & gate.NULLABLE


def _s5_artifacts(tmp_path, edit) -> Path:
    (tmp_path / "study5").mkdir()
    for name in ("confirm.json", "registered.json"):
        data = _json(f"study5/{name}")
        edit(name, data)
        (tmp_path / "study5" / name).write_text(json.dumps(data))
    return tmp_path


def _edit_plan(run, key, value):
    def edit(name, data):
        if name == "registered.json":
            data["runs"][run][key] = value

    return edit


def _edit_analysis(fn):
    def edit(name, data):
        if name == "confirm.json":
            fn(data)

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_edit_plan("s5-s1-000", "pool", "glm"), "pool"),
        (_edit_plan("s5-s1-000", "sessions_sha256", "0" * 64), "sessions_sha256"),
        (_edit_plan("s5-s0-000", "condition", "s2"), "condition"),
        (_edit_analysis(lambda d: d["runs"].pop()), "neither"),
        (
            _edit_analysis(
                lambda d: d["runs_set_aside"].append(dict(d["runs_set_aside"][0], run="s5-s1-000"))
            ),
            "both",
        ),
        (_edit_analysis(lambda d: d["runs"][0].update(run="s5-s9-000")), "not in the plan"),
        (_edit_analysis(lambda d: d["runs_set_aside"][0].update(why=["something new"])), "no kind"),
        (_edit_analysis(lambda d: d["runs"][0]["estimates"].pop("seq_gap")), "estimators"),
        (_edit_analysis(lambda d: d["runs"][0]["estimates"]["ll_bic"].update(hi="-inf")), "inf"),
        (_edit_analysis(lambda d: d["runs"][0]["tables"]["a"].append(1)), "cells"),
    ],
    ids=[
        "pool",
        "sessions",
        "condition",
        "missing",
        "twice",
        "unplanned",
        "reason",
        "estimator",
        "infinity",
        "cells",
    ],
)
def test_study5_rows_stop_when_the_analysis_and_the_plan_disagree(tmp_path, edit, message):
    artifacts = _s5_artifacts(tmp_path, edit)
    with pytest.raises(build.BuildError, match=message):
        build.study5_rows(artifacts)


def test_the_validators_wilson_is_the_registered_one():
    v = _json("robustness/study5-verdicts.json")
    cov, aside = v["primary"]["coverage"], v["P1_not_in_test_as_uncovered"]
    lo, hi = gate.wilson(cov["covered"], cov["runs"])
    assert (round(lo, 4), round(hi, 4)) == (cov["wilson_lo"], cov["wilson_hi"])
    lo, hi = gate.wilson(aside["covered"], aside["runs"])
    assert [round(lo, 4), round(hi, 4)] == aside["wilson"]


def test_no_tracked_artifact_is_skipped_as_local_only(monkeypatch):
    """The skip rule is wider than .gitignore (`*.log` at any depth), so git has the last word."""
    assert build.tracked_artifacts() and not [
        n for n in build.tracked_artifacts() if build.is_local_only(n)
    ]
    build.check_local_only_untracked()
    assert gate.tracked_local_only() == []
    monkeypatch.setattr(build, "tracked_artifacts", lambda: ["thinning/kept.log", "derived.json"])
    with pytest.raises(build.BuildError, match="thinning/kept.log"):
        build.check_local_only_untracked()


def _shallow() -> bool:
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"],
        capture_output=True, text=True,
    )  # fmt: skip
    return out.stdout.strip() != "false"


def test_a_freeze_is_compared_by_full_hash():
    if _shallow():
        pytest.skip("a shallow clone has no history to read the freeze commit from")
    full = gate.last_commit("docs/PREREG-STUDY5.md")
    assert len(full) == 40 and full.startswith(gate.STUDY5_FREEZE)
    assert build.freeze_commit("PREREG-STUDY5.md") == full[:7]


def test_the_validator_holds_the_analysis_to_the_plan(tmp_path):
    artifacts = _s5_artifacts(tmp_path, _edit_analysis(lambda d: d["runs"][0].update(pool="glm")))
    with pytest.raises(ValueError, match="plan"):
        gate.study5_expected_rows(artifacts)


def test_the_validator_recomputes_the_design_effect():
    post = _json("study5/posthoc.json")["dependence_between_runs"]
    assert gate.largest_design_effect(post["covered"], post["runs"]) == 4.05
    assert post["largest_design_effect_at_or_above_the_bar"] == 4.05
