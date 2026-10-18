"""Mutations a release must not survive. Each one is applied to a clean build and
the validator (or the publisher) has to fail on it.

These were written against the revision-3 validator first; every one of them
passed there (the mutation went undetected), which is why they exist.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build = _load("build_dataset")
gate = _load("validate_dataset")
publish = _load("publish_dataset")


def _validate(d: Path) -> list:
    kw = {"heavy": False} if "heavy" in inspect.signature(gate.validate).parameters else {}
    return gate.validate(d, **kw)


def _failures(d: Path) -> list:
    return [c for c in _validate(d) if not c[1]]


@pytest.fixture(scope="session")
def clean(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("clean") / "dataset"
    build.build(out)
    return out


@pytest.fixture
def copy(clean, tmp_path) -> Path:
    dst = tmp_path / "dataset"
    shutil.copytree(clean, dst)
    return dst


def _replace(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new, count))


def _rewrite_rows(path: Path, edit) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    edit(rows)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def test_the_clean_build_passes(clean):
    assert _failures(clean) == []


# --- blocker 1: transcripts --------------------------------------------------


def test_a_build_with_transcripts_fails_validation(tmp_path):
    out = tmp_path / "dataset"
    build.build(out, with_transcripts=True)
    assert _failures(out)


def test_publish_refuses_transcripts_without_the_validator(tmp_path, monkeypatch):
    out = tmp_path / "dataset"
    build.build(out, with_transcripts=True)
    assert publish.release_blockers(out)
    monkeypatch.setattr(publish, "validate", lambda *a, **k: [("stub", True, "")])
    monkeypatch.setattr(publish, "run_tests", lambda: (True, "stub"))
    assert publish.main(["--dataset-dir", str(out)]) == 1


def test_a_stray_reasoning_key_fails(copy):
    _rewrite_rows(
        copy / "episodes" / "confirm.jsonl",
        lambda rows: rows[0].update({"reasoning_content": "provider text"}),
    )
    assert _failures(copy)


# --- blocker 2: the validator pins what the builder decides ------------------


def test_a_relabelled_excluded_run_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(build.RUNS["diag-batch"], "phase", "exploratory")
    out = tmp_path / "dataset"
    build.build(out)
    assert _failures(out)


def test_a_relabelled_phase_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(build.RUNS["long-60"], "phase", "confirmatory")
    out = tmp_path / "dataset"
    build.build(out)
    assert _failures(out)


def test_shipping_an_unshipped_result_fails(tmp_path, monkeypatch):
    name = "thinning/mnar-diagnostics.json"
    monkeypatch.setattr(build, "SHIPPED_RESULTS", (*build.SHIPPED_RESULTS, name))
    monkeypatch.setattr(
        build, "NOT_SHIPPED", {k: v for k, v in build.NOT_SHIPPED.items() if k != name}
    )
    out = tmp_path / "dataset"
    build.build(out)
    assert _failures(out)


def test_publish_runs_the_tests_itself(copy, monkeypatch):
    monkeypatch.setattr(publish, "run_tests", lambda: (False, "a test failed"))
    assert publish.main(["--dataset-dir", str(copy)]) == 1


# --- high: the card is checked against the data -----------------------------

CARD_EDITS = [
    ("H1 fails", "H1 passes"),
    ("0.877", "0.977"),
    ("| H1 recovery | FAIL |", "| H1 recovery | PASS |"),
    ("828", "900"),
    ("1.0000 in `pilot-l0`", "0.5000 in `pilot-l0`"),
    ("| 1.07 |", "| 9.07 |"),
    ("17 of 21", "2 of 21"),
    ("Transcripts are not included.", "Transcripts are included."),
    ("Apache-2.0, for everything", "CC-BY-4.0, for everything"),
    ("7009 (HTTPStatusError)", "7010 (HTTPStatusError)"),
    ("63.15", "13.15"),
]


@pytest.mark.parametrize(("old", "new"), CARD_EDITS, ids=[e[0] for e in CARD_EDITS])
def test_a_hand_edited_card_fails(copy, old, new):
    _replace(copy / "README.md", old, new)
    assert _failures(copy)


def test_a_broken_snippet_fails(copy):
    _replace(copy / "README.md", "/ (m12 + 1) - 1", "/ (m12 + 1) + 100")
    assert _failures(copy)


# --- medium -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("run", "key", "value"),
    [
        ("long-30", "w4_recall", 0.5),
        ("long-30", "caveat", "No caveat."),
        ("confirm", "stopped", "BudgetExceeded"),
        ("supplement", "seed0", 5),
        ("supplement", "agent", "scripted"),
        ("supplement", "llm_calls", 3),
    ],
)
def test_an_unchecked_runs_field_fails(copy, run, key, value):
    _rewrite_rows(
        copy / "runs.jsonl",
        lambda rows: next(r for r in rows if r["run"] == run).update({key: value}),
    )
    assert _failures(copy)


def test_a_duplicate_run_row_fails(copy):
    _rewrite_rows(copy / "runs.jsonl", lambda rows: rows.append(dict(rows[2])))
    assert _failures(copy)


@pytest.mark.parametrize(("a", "b"), [("long-30", "long-60"), ("supplement", "scripted-l0")])
def test_swapped_config_files_fail(copy, a, b):
    readme = copy / "README.md"
    text = readme.read_text()
    text = text.replace(f"episodes/{a}.jsonl", "@@A@@").replace(
        f"episodes/{b}.jsonl", f"episodes/{a}.jsonl"
    )
    readme.write_text(text.replace("@@A@@", f"episodes/{b}.jsonl"))
    assert _failures(copy)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("- name: seed\n    dtype: int64", "- name: seed\n    dtype: float64"),
        ("- n<1K", "- 1K<n<10K"),
        ("- en\n", "- fr\n"),
    ],
)
def test_wrong_hub_metadata_fails(copy, old, new):
    _replace(copy / "README.md", old, new)
    assert _failures(copy)


def test_a_changed_gitattributes_fails(copy):
    (copy / ".gitattributes").write_text("*.json filter=lfs diff=lfs merge=lfs -text\n")
    assert _failures(copy)


@pytest.mark.parametrize("name", ["notes.txt", "x.bin", "episodes/extra.jsonl"])
def test_an_extra_file_fails(copy, name):
    (copy / name).write_text("{}\n")
    assert _failures(copy)


@pytest.mark.parametrize(
    ("key", "value"), [("seed", 7000.0), ("agent_read_trace", 1), ("n_true", 41.0)]
)
def test_type_confusion_fails(copy, key, value):
    def edit(rows):
        rows[0][key] = value if key != "seed" else float(rows[0]["seed"])

    _rewrite_rows(copy / "episodes" / "confirm.jsonl", edit)
    assert _failures(copy)


# --- privacy rules ------------------------------------------------------------

PRIVACY = {
    "URL": "see https://wiki.corp-internal.io/page",
    "hostname": "connect to db01.prod.acmecorp.com now",
    "IPv6": "bound to fe80::1ff:fe23:4567:890a",
    "phone number": "call (212) 867-5309 today",
    "Stripe key": "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc",
    "Google API key": "AI" + "za" + "SyD-9tSrke72PouQMnMX-a7eZSW0jkF",
    "GitLab token": "glp" + "at-" + "xxxxXXXXxxxxXXXXxxxx",
    "local filesystem path /tmp": "/tm" + "p/build/output.json",
    "local filesystem path /var/folders": "/var/fold" + "ers/ab/cd1234/T/x",
    "Windows path": json.dumps({"p": "D:" + "\\data\\notes.txt"}),
    "UNC path": "\\\\" + "fileserver\\share\\x",
    "email without a domain": "mail ops" + "@" + "buildbox please",
}


@pytest.mark.parametrize("label", sorted(PRIVACY))
def test_privacy_gate_catches_each_new_shape(label):
    found, _ = gate.privacy_findings(PRIVACY[label], "planted")
    assert found, label


def test_privacy_gate_fails_on_a_planted_file(copy):
    (copy / "docs" / "DECISIONS.md").write_text(
        (copy / "docs" / "DECISIONS.md").read_text() + "\nsee https://intranet.acmecorp.com\n"
    )
    assert _failures(copy)


# --- card wording (items 11-17 of the revision-4 review) ----------------------


def _prose(path: Path) -> str:
    """The card with blockquote markers and line breaks collapsed, for phrase checks."""
    text = path.read_text().replace("\n>", " ")
    return " ".join(text.split())


def _receipts_total() -> float:
    return sum(
        json.loads(p.read_text())["total_usd"]
        for p in (ROOT / "artifacts" / "episodes").glob("*/receipt.json")
    )


@pytest.mark.parametrize(
    "phrase",
    [
        # 11: transcripts are not irrelevant to every number
        "No estimate depends on them; `agent_read_trace` was computed from them.",
        # 12: banner hedging
        "appear to depend on the agent's trajectory; the trigger is still open (D34)",
        "replaces each lost seed with its first successful rerun, "
        "on a client changed after the D33 and D34 key-presence change",
        "| H1 recovery | PASS (marginal) |",
        "The MCSE used here is the four-seed one",
        # 13: usability
        'load_dataset(repo, "confirm", split="train")',
        "Each config has one split, `train`.",
        "| 0 | W1 | trace |",
        "| 3 | W4 | auditor |",
        "**oracle linkage**",
        "**signature linkage**",
        "**`fin_mid`**",
        "**`ll_bic`**",
        "**`union_count`**",
        "**`step`**",
        # 14: spend, and the crashed attempt
        "The first confirmatory attempt, which crashed (D30), left no receipt",
        # 15: D43
        "W2 each hold every `confirm` action (828/828 and 828/828)",
        # 16: docs cite unshipped files; the freeze
        "**The docs cite files that are not here.**",
        "local git commit `6ea27e2`",
        # 17: inferred fields
        "Inferred, not recorded (`runs.inferred_fields`): "
        "`scripted-l0` (`tickets`, `auditor_view`); `pilot-l0` (`tickets`, `auditor_view`); "
        "`long-30` (`auditor_view`); `long-60` (`auditor_view`)",
    ],
)
def test_the_card_says_it(clean, phrase):
    assert phrase in _prose(clean / "README.md")


def test_the_card_states_spend_from_every_receipt(clean):
    assert f"Spend over every receipt on disk is {_receipts_total():.2f} USD" in _prose(
        clean / "README.md"
    )


def test_the_card_carries_the_preregistration_hash(clean):
    import hashlib

    sha = hashlib.sha256((ROOT / "docs" / "PRE-REGISTRATION.md").read_bytes()).hexdigest()
    assert f"sha256 `{sha}`" in _prose(clean / "README.md")


def test_the_card_prints_the_snippet_line_and_explains_the_63(clean):
    prose = _prose(clean / "README.md")
    assert "Expected output (one loss draw under oracle linkage):" in prose
    assert "under oracle linkage that mean is 60.62" in prose


# --- revision 5a: robustness artifacts (D44) ----------------------------------


@pytest.fixture
def artifacts_copy(tmp_path) -> Path:
    dst = tmp_path / "artifacts"
    shutil.copytree(ROOT / "artifacts", dst)
    return dst


def test_an_unplaced_robustness_file_fails_build_and_validator(clean, artifacts_copy):
    (artifacts_copy / "robustness" / "new-battery.json").write_text("{}\n")
    with pytest.raises(build.BuildError, match="new-battery"):
        build.check_results_accounted(artifacts_copy)
    assert gate.check_results(clean, artifacts_copy)[1] is False


def test_spend_that_disagrees_with_derived_json_fails(clean, artifacts_copy):
    derived = json.loads((artifacts_copy / "derived.json").read_text())
    derived["spend"]["total_usd"] = round(derived["spend"]["total_usd"] + 1, 2)
    (artifacts_copy / "derived.json").write_text(json.dumps(derived))
    assert [c for c in gate.validate(clean, artifacts_copy, heavy=False) if not c[1]]


def test_an_edited_banner_number_fails(copy):
    _replace(copy / "README.md", "0.893", "0.993")
    assert _failures(copy)


@pytest.mark.parametrize(
    "name", ["precision.json", "mnar-dose.json", "falsegap-battery.json", "collisions.json"]
)
def test_the_robustness_results_ship(clean, name):
    shipped = clean / "results" / "robustness" / name
    assert shipped.read_bytes() == (ROOT / "artifacts" / "robustness" / name).read_bytes()


@pytest.mark.parametrize(
    "phrase",
    [
        "`ll_bic` covers 0.893 ± 0.006 under signature linkage and 0.948 ± 0.004 under oracle",
        "restored pool, 0.862 against 0.945",
        "in 0.358 of replicates",
        "0.223 of `confirm` actions share a signature",
        "falls to 0.647 and 0.657 at 1.1x, below the 0.90 bar under both linkages",
        "recovers at 1.8x (0.907 oracle, 0.920 signature) only by widening",
        "already 0.887 under signature",
        "only when records are matched by signature and the env view is financial",
    ],
)
def test_the_card_states_the_robustness_findings(clean, phrase):
    assert phrase in _prose(clean / "README.md")


# --- revision 5b: the transfer study (D45-D48) ----------------------------------

TRANSFER_PHASE = "confirmatory (transfer, separately pre-registered)"


def test_a_swapped_transfer_config_fails(copy):
    readme = copy / "README.md"
    text = readme.read_text()
    assert "episodes/transfer.jsonl" in text
    text = text.replace("episodes/transfer.jsonl", "@@T@@").replace(
        "episodes/long-30.jsonl", "episodes/transfer.jsonl"
    )
    readme.write_text(text.replace("@@T@@", "episodes/long-30.jsonl"))
    assert _failures(copy)


def test_an_edited_transfer_preregistration_fails(copy):
    prereg = copy / "docs" / "PREREG-TRANSFER.md"
    assert prereg.exists()
    prereg.write_text(prereg.read_text().replace("T1", "T1 (amended)", 1))
    assert _failures(copy)


def test_an_edited_transfer_banner_number_fails(copy):
    _replace(copy / "README.md", "0.401", "0.501")
    assert _failures(copy)


def test_the_transfer_run_ships_as_its_own_confirmatory_config(clean):
    rows = [json.loads(x) for x in (clean / "episodes" / "transfer.jsonl").read_text().splitlines()]
    assert len(rows) == 23 and {r["phase"] for r in rows} == {TRANSFER_PHASE}
    assert {r["env"] for r in rows} == {"infra"}


@pytest.mark.parametrize("env", ["backoffice", "infra"])
def test_each_environment_ships_its_own_prompt_and_tools(clean, env):
    from lacuna.envs.registry import ENVS

    folder = clean / "environment" / env
    assert (folder / "system_prompt.txt").read_text() == ENVS[env].agent_system + "\n"
    assert json.loads((folder / "tools.json").read_text()) == ENVS[env].tools


def test_every_transfer_artifact_is_placed():
    names = build.result_artifacts(ROOT / "artifacts")
    transfer = [n for n in names if "transfer" in n]
    assert transfer and all(n in build.SHIPPED_RESULTS for n in transfer)


@pytest.mark.parametrize(
    "phrase",
    [
        "0.401 under signature linkage against 0.946 under oracle",
        "`ll_bic` covers 0.300 at the frozen loss seed",
        "T2 and T4 say nothing under signature linkage",
        "T3 passes only because the interval widened",
        "restores `ll_bic` to 0.930",
        "The verdict depends on density (post hoc, D46)",
        "1035 of 1036 `transfer` actions are in the scripted plans",
        "local commit `6774b24`",
    ],
)
def test_the_card_states_the_transfer_findings(clean, phrase):
    assert phrase in _prose(clean / "README.md")


def test_the_transfer_behaviour_result_ships_byte_identical(clean):
    # Redacted at the source on main (2b05580), so no special path remains.
    raw = (ROOT / "artifacts" / "robustness" / "transfer-behaviour.json").read_bytes()
    assert (clean / "results" / "robustness" / "transfer-behaviour.json").read_bytes() == raw
    assert not hasattr(build, "REDACTED_RESULTS") and not hasattr(gate, "REDACTED_RESULTS")


def test_builder_and_validator_agree_on_the_transfer_manifest():
    assert set(build.PREREGISTRATIONS) == set(gate.PREREGISTRATIONS)
    assert set(build.SHIPPED_DOCS) == set(gate.SHIPPED_DOCS)
    envs = {run: meta["env"] for run, meta in build.RUNS.items() if meta["env"] != "backoffice"}
    assert envs == gate.EXPECTED_ENV


# --- revision 6: holes a refuter found in the a5de0e9 validator -----------------


@pytest.mark.parametrize(
    ("run", "old", "new"),
    [
        ("transfer", "T1 fails", "T1 passes"),
        ("confirm", "deepseek-v4.1-flash", "v4-flash"),
        ("confirm", "Two of the three losses were provider 400s", "None of the three losses were"),
        ("long-30", "No transcript was stored", "Transcripts are included"),
    ],
)
def test_a_flipped_caveat_fails(copy, run, old, new):
    def edit(rows):
        row = next(r for r in rows if r["run"] == run)
        assert old in row["caveat"]
        row["caveat"] = row["caveat"].replace(old, new)

    _rewrite_rows(copy / "runs.jsonl", edit)
    assert _failures(copy)


def _swapped(original, *pairs):
    def patched(*args, **kwargs):
        out = dict(original(*args, **kwargs))
        for a, b in pairs:
            out[a], out[b] = out[b], out[a]
        return out

    return patched


@pytest.mark.parametrize(
    "pairs",
    [
        (("gap_sig", "gap_orc"),),
        (("mid_cov", "rho_cov"),),
        (("best", "worst"),),
    ],
    ids=["T2-T4 rates", "T3 coverage", "density range"],
)
def test_a_builder_that_swaps_transfer_figures_fails(tmp_path, monkeypatch, pairs):
    monkeypatch.setattr(build, "_transfer", _swapped(build._transfer, *pairs))
    out = tmp_path / "dataset"
    build.build(out)
    assert _failures(out)


def test_a_builder_that_swaps_precision_cells_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "_robustness", _swapped(build._robustness, ("reg_sig", "reg_orc")))
    out = tmp_path / "dataset"
    build.build(out)
    assert _failures(out)


def test_a_builder_that_misreads_a_verdict_fails(tmp_path, monkeypatch):
    original = build._transfer

    def patched(*args, **kwargs):
        out = dict(original(*args, **kwargs))
        # The builder's own reading of T1, flipped; the shipped JSON still says FAIL.
        if "t1" in out:  # the a5de0e9 builder
            out["t1"] = {**out["t1"], "passed": True}
        else:
            out["t"] = {**out["t"], "T1": {**out["t"]["T1"], "passed": True}}
        return out

    monkeypatch.setattr(build, "_transfer", patched)
    out = tmp_path / "dataset"
    build.build(out)
    assert _failures(out)


def test_a_forged_commit_stamp_fails(copy):
    import re

    readme = copy / "README.md"
    text = readme.read_text()
    stamp = re.search(r"lacuna commit `([^`]+)`", text).group(1)
    readme.write_text(text.replace(stamp, "deadbee"))  # every occurrence: card and citation
    assert _failures(copy)


def test_the_dry_run_exits_non_zero_when_the_gate_would_refuse(copy, monkeypatch):
    monkeypatch.setattr(publish, "run_tests", lambda: (True, "stub"))
    monkeypatch.setattr(publish, "validate", lambda *a, **k: [("stub", True, "")])
    monkeypatch.setattr(publish, "git_commit", lambda: "abc1234-dirty")
    assert publish.main(["--dataset-dir", str(copy)]) == 1


def _patched_docs(tmp_path, monkeypatch, copy, name, edit):
    docs = tmp_path / "repo-docs"
    shutil.copytree(ROOT / "docs", docs)
    for folder in (docs, copy / "docs"):
        (folder / name).write_text(edit((folder / name).read_text()))
    monkeypatch.setattr(gate, "DOCS", docs)


def test_reasoning_prose_in_a_doc_fails(copy, tmp_path, monkeypatch):
    _patched_docs(
        tmp_path,
        monkeypatch,
        copy,
        "DECISIONS.md",
        lambda t: t + "\nreasoning_content: The user wants the refunds done first.\n",
    )
    assert _failures(copy)


def test_a_doc_edited_after_the_build_commit_fails(copy, tmp_path, monkeypatch):
    _patched_docs(
        tmp_path, monkeypatch, copy, "TRANSFER.md", lambda t: t.replace("T1", "T1 (edited)", 1)
    )
    assert _failures(copy)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("hostname .me", "see status.acme-corp.me for details"),
        ("hostname .sh", "curl get.acme-corp.sh"),
        ("hostname .ly", "short link acmecorp.ly/x"),
        ("npm token", "npm" + "_" + "A1b2C3d4E5f6G7h8I9j0"),
        ("chatcmpl id", "chatcmpl" + "-" + "9aBcDeFgHiJkLmNoP"),
        ("req_ id", "req" + "_" + "011CRqzFq8uGa7xY3wZ1"),
    ],
)
def test_privacy_gate_catches_revision_6_shapes(label, text):
    found, _ = gate.privacy_findings(text, "planted")
    assert found, label


# --- revision 6: card prose (review and refuter items 5-12) --------------------


@pytest.mark.parametrize(
    "phrase",
    [
        # 5: labels and post-hoc tags
        "**Confirmatory: `confirm` (H1-H4) and `transfer` (T1-T4).**",
        "**Exploratory: every other LLM run, and every post-hoc result.**",
        "the transfer study's precision, mechanism, false-gap and density results (D46-D48)",
        "At 3000 replicates the shortfall is linkage, not the estimator (post hoc, D44).",
        "over 20 seeds (post hoc)",
        "`confirmatory (transfer, separately pre-registered)`",
        # 6: transfer facts
        "episode seeds from 77000) that are not shipped",
        "it does not test LLM behaviour beyond that",
        "**T1's failure was predicted before the freeze.** Prediction 3 in "
        "`docs/PREREG-TRANSFER.md`",
        "**The mechanism is linkage (post hoc, D47).** Over 10 loss seeds",
        "This cell has 10 seeds and the T1 precision above 20",
        # 7: stale text
        "One agent family, two environments.",
        "deepseek-v4.1-flash: `confirm`, `narr-30`, `transfer`",
        "In `confirm`, `supplement`, `scripted-l0`, `transfer`, `transfer-scripted`, "
        "`transfer-scripted-24`, witness 3 is a scripted stand-in",
        "| run | phase | model | auditor | tickets |",
        "for the scripted runs `transfer-scripted`, `transfer-scripted-24`, where they hold none",
        "331 episodes across 10 runs (104 LLM-agent, 227 scripted)",
        # 9: two environments in the glossary and fields
        "**`transfer_mid`**",
        "**`_rho100`**",
        "**`anti_*`**",
        "**`strat_bic`**",
        "`send_page` and `write_runbook` values in `infra`",
        "`backoffice` adds one closing report task",
        "`agent_read_trace` is always false in `infra`, which has no `read_file` tool",
        "name diagnostic reruns that are not `runs` rows or configs",
        "Result labels call W2 `proxy`, as in `ll_bic@trace+proxy+env`",
        # 10: W4 recall linkage
        "matched to the ledger by signature, since auditor records carry no ledger id",
        # 11: confirm-supp is study spend
        "which includes `confirm-supp`, the supplement's rerun of the lost seeds (D34)",
        # 12: pre-registration notes
        'Its text still reads "Status: ready to freeze"',
        "`docs/PREREG-TRANSFER.md` cites D34, which postdates its freeze",
    ],
)
def test_the_card_carries_the_revision_6_prose(clean, phrase):
    assert phrase in _prose(clean / "README.md")


def test_the_card_no_longer_says_stale_things(clean):
    prose = _prose(clean / "README.md")
    for stale in (
        "One agent family, one environment",
        "**Confirmatory: `confirm` only.**",
        "by oracle linkage.",
        "The mechanism is linkage (D48)",
    ):
        assert stale not in prose, stale


def test_the_dry_run_exits_zero_only_when_uploadable(copy, monkeypatch):
    monkeypatch.setattr(publish, "run_tests", lambda: (True, "stub"))
    monkeypatch.setattr(publish, "validate", lambda *a, **k: [("stub", True, "")])
    monkeypatch.setattr(publish, "card_commit", lambda folder: "abc1234")
    monkeypatch.setattr(publish, "git_commit", lambda: "abc1234")
    assert publish.main(["--dataset-dir", str(copy)]) == 0
    monkeypatch.setattr(publish, "git_commit", lambda: "def5678")
    assert publish.main(["--dataset-dir", str(copy)]) == 1
