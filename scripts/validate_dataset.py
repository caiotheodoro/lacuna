#!/usr/bin/env python3
"""Check dataset/ before anything is published. Exits 1 on any failure.

  uv run --group dataset python scripts/validate_dataset.py

The validator does not trust the builder. What ships, under which phase, in
which config, with which schema, is pinned here, apart from
scripts/build_dataset.py, so a decision changed in one place fails until it is
made in both. Rows and runs are checked against the sources the builder read:
each run's summary.json, receipt.json and episodes.jsonl, hypotheses.json, and
the ledgers themselves for any ticket count a summary did not record.

The card is regenerated from the shipped files with the builder's `card()`
and must match README.md exactly; its snippet is run and must print the lines
the card says it prints, and agree with an independent hand-rolled draw.
`datasets` loads every config in a child process (so this process never goes
offline); the loaded schema must match the pinned one, `null` standing for a
column with nothing in it. The privacy gate scans every file. No network.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
DATASET = ROOT / "dataset"

Check = tuple[str, bool, str]

# ---------------------------------------------------------------------------
# the pinned release manifest
# ---------------------------------------------------------------------------

SUPPLEMENT = "supplement"
SUPPLEMENT_PHASE = "post-hoc supplement (not the registered verdict)"
BYPASS_PHASE = "confirmatory (bypass, separately pre-registered)"
BYPASS_PILOT_PHASE = "pilot (bypass L1 organism, pre-registered, no verdict)"
STUDY4_PHASE = "confirmatory (study 4, separately pre-registered)"
# Runs whose shipped config is not their directory's name. The bypass study's neutral pool
# is `bypass-llm` in the code repository and in its result files.
SHIPPED_AS = {"bypass-llm": "bypass"}
# Every run directory under artifacts/episodes, and what the release does with it.
EXPECTED_PHASES = {
    "confirm": "confirmatory",
    "scripted-l0": "scripted harness check",
    "pilot-l0": "exploratory",
    "long-30": "exploratory",
    "long-60": "exploratory",
    "long-120": "exploratory",
    "narr-30": "exploratory",
    "transfer": "confirmatory (transfer, separately pre-registered)",
    "transfer-scripted": "scripted harness check",
    "transfer-scripted-24": "scripted harness check",
    "confirm-supp": "excluded",
    "diag-7009": "excluded",
    "diag-batch": "excluded",
    "confirm-plus-supp": "excluded",
    "bypass-llm": BYPASS_PHASE,
    "bypass-l1": BYPASS_PILOT_PHASE,
    "bypass-scripted-10": "scripted harness check",
    "bypass-scripted-30": "scripted harness check",
    "s4-neutral": STUDY4_PHASE,
    "s4-l2": STUDY4_PHASE,
    "s4-qwen-backoffice": STUDY4_PHASE,
    "s4-qwen-bypass": STUDY4_PHASE,
}
# Each run's environment; every other run is backoffice.
EXPECTED_ENV = {
    "transfer": "infra",
    "transfer-scripted": "infra",
    "transfer-scripted-24": "infra",
    "bypass-llm": "backoffice_bypass",
    "bypass-l1": "backoffice_bypass",
    "bypass-scripted-10": "backoffice_bypass",
    "bypass-scripted-30": "backoffice_bypass",
    "s4-neutral": "backoffice_bypass",
    "s4-l2": "backoffice_bypass",
    "s4-qwen-bypass": "backoffice_bypass",
}
# D67 to D70. The model, the organism and the prompt hashes of every LLM run in the bypass env
# and of study 4's back-office run, written here from the registrations (docs/PREREG-BYPASS.md,
# docs/PREREG-STUDY4.md) and checked three ways: against each run's summary.json, against
# runs.jsonl, and against the prompt strings in the code itself (`check_study4`).
DEEPSEEK = "deepseek-v4.1-flash"
QWEN = "qwen3.8-flash"
NEUTRAL_SYSTEM_SHA256 = "bcdea22452c60c956c03b6daa556be80ca4f38c49be517d2aafce8a96a8f56ba"
L1_PARAGRAPH_SHA256 = "7c53c35faef33d7ffef9ead072880ea8d0ffb8b366af723938f1ff7548852b53"
L1_SYSTEM_SHA256 = "46e4262929c041645f32cfae1557fce1cdcaba7ef5327dea1ec0c8612a24d6b1"
L2_PARAGRAPH_SHA256 = "9df375c4f52af320f32e4b3fd48a2ff774fee4a16375c0bcf812d598e5feb6d4"
L2_SYSTEM_SHA256 = "fd19833be304651360e7663df660c5d8cd2a59360047d60a8d722b28ca810fad"
# run -> (model, organism, organism paragraph sha256, full system prompt sha256); None where
# the run records none (the back-office run has no organism).
RUN_PINS = {
    "bypass-llm": (DEEPSEEK, "neutral", None, NEUTRAL_SYSTEM_SHA256),
    "bypass-l1": (DEEPSEEK, "l1", L1_PARAGRAPH_SHA256, L1_SYSTEM_SHA256),
    "s4-neutral": (DEEPSEEK, "neutral", None, NEUTRAL_SYSTEM_SHA256),
    "s4-l2": (DEEPSEEK, "l2", L2_PARAGRAPH_SHA256, L2_SYSTEM_SHA256),
    "s4-qwen-backoffice": (QWEN, None, None, None),
    "s4-qwen-bypass": (QWEN, "neutral", None, NEUTRAL_SYSTEM_SHA256),
}
STUDY4_RUNS = ("s4-neutral", "s4-l2", "s4-qwen-backoffice", "s4-qwen-bypass")
# Study 4's receipts at the runner's deepseek rates (D70): the sum over the four runs.
STUDY4_SPEND_USD = 4.17
# Diagnostic reruns whose spend the card reports separately from the study.
DIAGNOSTIC_RUNS = ("diag-7009", "diag-batch")
# The confirmatory run is the one hypotheses.json names; the pool and its
# composition come from confirm-plus-supp.
POOL = "confirm-plus-supp"
DEFAULT_CONFIG = "confirm"
# Each pre-registered study: its verdict file, and the key path in it that names its run.
CONFIRMATORY = {
    "confirm": ("confirmatory", "hypotheses.json", ("run",)),
    "transfer": (
        "confirmatory (transfer, separately pre-registered)",
        "transfer_hypotheses.json",
        ("run",),
    ),
    "bypass-llm": (BYPASS_PHASE, "robustness/bypass-verdicts.json", ("runs", "neutral")),
    # Study 4: one verdict file names its four runs.
    "s4-qwen-backoffice": (
        STUDY4_PHASE,
        "robustness/study4-verdicts.json",
        ("inputs", "replication", "run"),
    ),
    "s4-qwen-bypass": (
        STUDY4_PHASE,
        "robustness/study4-verdicts.json",
        ("inputs", "qwen_bypass", "run"),
    ),
    "s4-l2": (STUDY4_PHASE, "robustness/study4-verdicts.json", ("inputs", "adversary", "run")),
    "s4-neutral": (STUDY4_PHASE, "robustness/study4-verdicts.json", ("inputs", "neutral", "run")),
}
# Decisions each run's caveat must cite.
CAVEAT_REFS = {
    "confirm": ("D20", "D32", "D34", "D36", "D44", "D49", "D50"),
    "scripted-l0": (),
    "pilot-l0": ("D13", "D14", "D20"),
    "long-30": ("D13", "D14", "D18", "D20"),
    "long-60": ("D13", "D14", "D18", "D20"),
    "long-120": ("D13", "D14", "D18", "D20"),
    "narr-30": ("D14", "D20"),
    "transfer": ("D20", "D45", "D46", "D47", "D48", "D49", "D52"),
    "transfer-scripted": ("D45",),
    "transfer-scripted-24": ("D45", "D46"),
    "bypass": ("D20", "D61", "D63"),
    "bypass-l1": ("D20", "D63"),
    "bypass-scripted-10": ("D61", "D63"),
    "bypass-scripted-30": ("D61", "D63"),
    "s4-neutral": ("D20", "D67", "D69", "D70"),
    "s4-l2": ("D20", "D67", "D68", "D70", "D73"),
    "s4-qwen-backoffice": ("D20", "D67", "D70"),
    "s4-qwen-bypass": ("D20", "D67", "D70", "D73"),
    SUPPLEMENT: ("D34", "D35", "D36"),
}
# The exact text of every shipped caveat, by sha256. A caveat is the sentence
# that governs how a run may be read; changing one is a decision, made here
# as well as in the builder.
CAVEAT_SHA256 = {
    "confirm": "04ae06f7d1635f49a25a136421b604f1fb430a637f032c6fb98d3eeb0a2f9c3b",
    SUPPLEMENT: "2e6cc49247bb2d905e124ca03e1650a0b698212c907b7613472c1055435c6f53",
    "scripted-l0": "90c846eab5109653bdfef6cea134d0f5b3bf251be43456d4a5c56b6cbf60f8a3",
    "pilot-l0": "c8538b27c0dd6b349af36623caf267303254374c803ea3b9c267a35f38c8f57e",
    "long-30": "702329be3277d83eb48813e7e7f5e02d2e693433af9363802fdb8b4f41af8f29",
    "long-60": "d26929a46b46735b414a7786368ce41b0e33951f84cce2e0d8cc53f3d8ee6e0c",
    "long-120": "3387789c2ebc5619449c5d6828a80b32d67f4af6281c73d39cd50e8d31817a79",
    "narr-30": "391c41f59f071980e756d5672c6daf853d86138eb1055ebb33fe8265b891fe21",
    "transfer": "6a8293f5dea506c21605ed1cf6feb647f860f265decdf512ce39008399872e4e",
    "transfer-scripted": "cec399345f4accae019f3d442bdff81c11ce862e79381c46d134c6f1748a2816",
    "transfer-scripted-24": "172ec47327d2c9fecee48fc64350dd02408852341a58b5381e56aa565544b529",
    "bypass": "a58db0d4a49280b23ac0a4b1d16ab1a30078b54ee26c71b3d3a6da592f131266",
    "bypass-l1": "835ac4b3aab0d6f3961038e28b25386e030894c55afb924b66b9e44c73e7e3ea",
    "bypass-scripted-10": "09cca85f0d81310f6d664745b86bb31a50b259519e4ffe3e36b890965f1841a5",
    "bypass-scripted-30": "ed0d7b44ce90edb087ddb18baf290e5a255b4b1d199260d49e785548dc351ecb",
    "s4-neutral": "358dae4f06ed131fc5b801d057f67695c5486b225f5dfa564e2f042709275fd0",
    "s4-l2": "5cf57ab2500a298b90cdabcb6545de9387b63684b2349004ae082f006f4ffc38",
    "s4-qwen-backoffice": "36387797663a6a0592f9988237895ea4f10639429ce2ea5468ce15a2efb0078a",
    "s4-qwen-bypass": "c6f314a118754b11cb26e52b6f7a66fa601a11c081a5bf52564979951d304b5c",
}
SHIPPED_RESULTS = {
    "hypotheses.json",
    "hypotheses-supplement.json",
    "supplement-variance.json",
    "sim_coverage.json",
    "thinning/confirm.json",
    "thinning/confirm-mnar.json",
    "thinning/confirm-mnar-strat.json",
    "thinning/confirm-mnar-hidden.json",
    "thinning/confirm-falsepos.json",
    "thinning/long-30.bootstrap.json",
    "thinning/long-30.json",
    "robustness/precision.json",
    "robustness/mnar-dose.json",
    "robustness/falsegap-battery.json",
    "robustness/collisions.json",
    "robustness/linkage-arms.json",
    "robustness/burst-loss.json",
    "robustness/overdispersion.json",
    "robustness/episode-dispersion.json",
    "robustness/interval-comparison.json",
    "robustness/session-and-phase.json",
    "robustness/op-position.json",
    "robustness/final-controls.json",
    "transfer_hypotheses.json",
    "thinning/transfer.json",
    "thinning/transfer-falsepos.json",
    "thinning/transfer-scripted.json",
    "thinning/transfer-scripted-24.json",
    "robustness/transfer-precision.json",
    "robustness/transfer-mechanism.json",
    "robustness/transfer-density.json",
    "robustness/transfer-falsegap.json",
    "robustness/transfer-behaviour.json",
    "robustness/bypass-verdicts.json",
    "robustness/bypass-analysis-bypass-llm.json",
    "robustness/bypass-analysis-bypass-l1.json",
}
# Study 4's results stay in the code repository (D72): each records the provider's host, which the
# privacy gate refuses, and the three adversary files quote the provider's reasoning text. The
# card's study-4 figures are read from them here (`expected_figures`, `check_study4`).
STUDY4_RESULTS = {
    "robustness/study4-verdicts.json",
    "robustness/study4-replication-s4-qwen-backoffice.json",
    "robustness/study4-adversary-s4-neutral.json",
    "robustness/study4-adversary-s4-l2.json",
    "robustness/study4-adversary-s4-qwen-bypass.json",
}
NOT_SHIPPED = {
    *STUDY4_RESULTS,
    "derived.json",
    "robustness/bypass-counterfactuals.json",
    "robustness/bypass-analysis-confirm.json",
    "robustness/bypass-verdicts-scripted.json",
    "robustness/bypass-analysis-bypass-scripted-10.json",
    "robustness/bypass-analysis-bypass-scripted-30.json",
    "robustness/baselines.json",
    "robustness/baselines-bypass.json",
    "robustness/new-loss.json",
    "robustness/r-crosscheck.json",
    "robustness/selection-coverage.json",
    "thinning/confirm-heldout-fixed-episodes.json",
    "mnar-coverage-split.json",
    "thinning/long-30.fixed-episodes.json",
    "thinning/confirm-plus-supp.json",
    "thinning/var-confirm-s1.json",
    "thinning/var-confirm-s2.json",
    "thinning/var-confirm-s3.json",
    "thinning/var-supp-s1.json",
    "thinning/var-supp-s2.json",
    "thinning/var-supp-s3.json",
    "thinning/confirm-heldout-strat.json",
    "thinning/long-30-strat.json",
    "thinning/mnar-diagnostics.json",
}
SHIPPED_DOCS = (
    "DECISIONS.md",
    "PRE-REGISTRATION.md",
    "PREREG-TRANSFER.md",
    "PREREG-BYPASS.md",
    "PREREG-STUDY4.md",
    "TRANSFER.md",
)
SHIPPED_SCRIPTS = ("loss_sim.py",)
SCRIPTS_DIR = ROOT / "scripts"
# Each pre-registration: the commit that froze it and the sha256 of its text.
PREREGISTRATIONS = {
    "PRE-REGISTRATION.md": (
        "6ea27e2",
        "4ec73dff205358554a8b3f11b81f2507e7895ab2de416fd93e8d31ff6b8fdf86",
    ),
    "PREREG-TRANSFER.md": (
        "6774b24",
        "9954251b42542cfe28906e3d804df19c0636d7024a9e161b9a9afe2a50781b00",
    ),
    "PREREG-BYPASS.md": (
        "c451bc1",
        "38b56dd26edec66bf6d165c837a594c7935ba12c2da76daeaaa2a108c325a098",
    ),
    "PREREG-STUDY4.md": (
        "687ff87",
        "1c36299aae27cd0126c76a6c2e214b49cd4ac2d5f868e4e02a8cc6c3f0c544cf",
    ),
}
# The tag each of those was pushed under, where it has one: it must name the freeze commit.
PREREG_TAGS = {"PREREG-BYPASS.md": "prereg-bypass", "PREREG-STUDY4.md": "prereg-study4"}
FREEZES = {doc: commit for doc, (commit, _) in PREREGISTRATIONS.items()}
GITATTRIBUTES = "*.jsonl filter=lfs diff=lfs merge=lfs -text\n"
REQUIRED_TAGS = [
    "agent-auditing",
    "capture-recapture",
    "ai-governance",
    "logging",
    "llm-agents",
    "multiple-systems-estimation",
]
REQUIRED_SECTIONS = (
    "## What holds",
    "## Configs",
    "## Load it",
    "## What the data is",
    "## Field reference",
    "## The three witnesses, and why the LLM auditor is not one",
    "## Glossary",
    "## Simulate your own log loss",
    "## Exploratory vs confirmatory",
    "## Transfer study",
    "## Bypass study",
    "## Study 4",
    "## Known caveats",
    "## Synthetic data",
    "## Citation",
)
BANNER = "> **Read before any result below"
# D66: the card's front door. What holds is a table of at most this many rows, the caveat
# banner follows it, then the config table and the loading snippet, then one <details> block.
HOLDS_HEADER = "| claim | evidence | status |"
MAX_HOLDS_ROWS = 5
DETAILS = "<details>"
# Phrases the banner must keep: a caveat that matters cannot be trimmed out of the front door.
BANNER_TERMS = (
    "Synthetic data",
    "simulated loss",
    "post hoc",
    "no figure here measures real log loss",
    "registered",
    "both pre-registered",
    "stipulated channel",
    "not a witness (D20)",
    "withheld",
)
# Half of the 7553-word body of the card this layout replaced, plus slack; and the words
# that may sit before the <details> block. Study 4 (D72) raised them from 3800 and 800: four
# config rows, two What holds rows and a banner bullet up front, a Study 4 section below.
CARD_WORD_BUDGET = 4300
CARD_FRONT_BUDGET = 900
HARD_WITNESSES = {0, 1, 2}
AUDITOR = 3
BYPASS_ENV = "backoffice_bypass"


def episode_configs(artifacts: Path) -> list[str]:
    """Shipped episode configs, in card order: the default first, then the supplement."""
    on_disk = {p.name for p in (artifacts / "episodes").iterdir() if p.is_dir()}
    runs = [
        SHIPPED_AS.get(r, r)
        for r, ph in EXPECTED_PHASES.items()
        if ph != "excluded" and r in on_disk
    ]
    rest = [r for r in runs if r != DEFAULT_CONFIG]
    return [DEFAULT_CONFIG, SUPPLEMENT, *rest]


SOURCE_OF = {config: run for run, config in SHIPPED_AS.items()}


def source_run(config: str) -> str:
    """The run directory under artifacts/episodes/ that a shipped config was built from."""
    return SOURCE_OF.get(config, config)


def allowed_files(artifacts: Path) -> set[str]:
    files = {".gitattributes", "README.md", "runs.jsonl"}
    files |= {f"episodes/{c}.jsonl" for c in episode_configs(artifacts)}
    files |= {f"results/{n}" for n in SHIPPED_RESULTS}
    files |= {f"docs/{n}" for n in SHIPPED_DOCS}
    files |= {f"scripts/{n}" for n in SHIPPED_SCRIPTS}
    for env in shipped_envs(artifacts):
        files |= {f"environment/{env}/system_prompt.txt", f"environment/{env}/tools.json"}
    return files


def shipped_envs(artifacts: Path) -> list[str]:
    runs = [c for c in episode_configs(artifacts) if c != SUPPLEMENT]
    return sorted({EXPECTED_ENV.get(source_run(r), "backoffice") for r in runs})


# Schemas, written here independently of the builder's dataset_info.
def _f(name: str, spec: dict) -> dict:
    return {"name": name, **spec}


_S, _I, _F, _B = ({"dtype": t} for t in ("string", "int64", "float64", "bool"))
_RECORD = [_f("witness", _I), _f("op", _S), _f("resource", _S), _f("value", _S), _f("oracle", _I)]
_LEDGER = [
    _f("id", _I),
    _f("op", _S),
    _f("resource", _S),
    _f("value", _S),
    _f("step", _I),
    _f("bypassed", _B),
]
EPISODE_FEATURES = [
    _f("run", _S),
    _f("seed", _I),
    _f("model", _S),
    _f("env", _S),
    _f("tickets", _I),
    _f("phase", _S),
    _f("auditor_view", _S),
    _f("n_true", _I),
    _f("ledger", {"list": _LEDGER}),
    _f("records", {"list": _RECORD}),
    _f("auditor_records", {"list": _RECORD}),
    _f("agent_read_trace", _B),
    _f("n_calls", _I),
    _f("n_tool_errors", _I),
]
RUN_FEATURES = [
    _f("run", _S),
    _f("phase", _S),
    _f("env", _S),
    _f("agent", _S),
    _f("auditor", _S),
    _f("model", _S),
    _f("organism", _S),
    _f("organism_prompt_sha256", _S),
    _f("auditor_view", _S),
    _f("tickets", _I),
    _f("episodes", _I),
    _f("episodes_attempted", _I),
    _f("failed_episodes", {"list": [_f("seed", _I), _f("error", _S)]}),
    _f("stopped", _S),
    _f("seed0", _I),
    _f("n_true", _I),
    _f("recorded_fault_rates", {"struct": [_f("p_gateway_drop", _F), _f("p_trace_drop", _F)]}),
    _f("spend_usd", _F),
    _f("llm_calls", _I),
    _f("transcripts_stored", _B),
    _f("transcripts_included", _B),
    _f("w4_recall", _F),
    _f("oracle_table", {"sequence": "int64"}),
    _f("signature_table", {"sequence": "int64"}),
    _f("caveat", _S),
    _f("composition", {"list": [_f("run", _S), _f("seed", _I)]}),
    _f("inferred_fields", {"sequence": "string"}),
]
NULLABLE = {
    "model",
    "organism",
    "organism_prompt_sha256",
    "agent_read_trace",
    "records[].oracle",
    "auditor_records[].oracle",
    "episodes_attempted",
    "stopped",
    "seed0",
    "spend_usd",
    "llm_calls",
    "composition",
}

# ---------------------------------------------------------------------------
# privacy gate
# ---------------------------------------------------------------------------

# The owner's first and last names, built from fragments so the pattern does
# not spell them out in this source.
_OWNER_NAME_FRAGMENTS = ("ca" + "io", "theo" + "doro", "ma" + "theus")

# A credential prefix followed by anything fails, because a truncated key is
# still a leaked key. False positives fail closed and a human looks.
KEY_SHAPES = {
    "sk- key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]"),
    "Stripe key": re.compile(r"(?<![A-Za-z0-9])[sr]k_(?:live|test)_[A-Za-z0-9]"),
    "oc_sk_ key": re.compile(r"oc_sk_"),
    "hf_ token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]"),
    "GitHub token": re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]"),
    "GitLab token": re.compile(r"glpat-[A-Za-z0-9_-]"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{10,}"),
    "Slack token": re.compile(r"xox[abp]-"),
    "AWS access key": re.compile(r"A(?:KIA|SIA)[0-9A-Z]{16}|AKIA[0-9A-Z]{8,}"),
    "Bearer token": re.compile(r"Bearer\s+\S"),
    "PEM block": re.compile(r"-----BEGIN [A-Z0-9 ]*-----"),
    "JWT": re.compile(r"eyJ[A-Za-z0-9_-]+\."),
    "provider request id": re.compile(r"(?i)request[_ -]?id\W{0,3}[0-9a-f]{4,}"),
    # Eight-four-four is enough: stored errors are truncated mid-id.
    "UUID": re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}\b"),
    "URL": re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://\S|\bwww\.[a-z0-9-]+\.|\bmailto:"),
    "npm token": re.compile(r"(?<![A-Za-z0-9])npm_[A-Za-z0-9]{8,}"),
    "chatcmpl id": re.compile(r"chatcmpl-[A-Za-z0-9]"),
    "req_ id": re.compile(r"(?<![A-Za-z0-9])req_[A-Za-z0-9]{8,}"),
}
RAW_SHAPES = {
    "IPv4 address": re.compile(
        r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
    ),
    "IPv6": re.compile(
        r"(?<![0-9A-Za-z:])(?:(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
        r"|(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}:){0,5}[0-9A-Fa-f]{1,4}"
        r"|::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4})(?![0-9A-Za-z:])"
    ),
    "phone number": re.compile(
        r"(?<![\d.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\d.])"
        r"|\+\d{1,3}[\s.-]\d{1,4}[\s.-]\d{3,4}[\s.-]\d{3,4}(?!\d)"
    ),
    "local filesystem path": re.compile(r"/Users/|/home/|/root/|/private/|/tmp/|/var/folders/|~/"),
    "Windows path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\+[A-Za-z0-9_. -]"),
    "UNC path": re.compile(r"\\{2,}[A-Za-z0-9_.-]+\\+[A-Za-z0-9_.$-]"),
    "owner name": re.compile("(?i)" + "|".join(_OWNER_NAME_FRAGMENTS)),
    "hostname": re.compile(
        r"(?i)\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ai|dev|app|co|us|uk|de|fr|cn|ru|jp|"
        r"edu|gov|info|biz|xyz|cloud|tech|site|online|tv|cc|me|sh|ly|"
        r"internal|local|corp|lan)\b"
    ),
}
# The publish target and the code repository necessarily name the account,
# and the citation names its author; nothing else may. Longest first, so the
# shorter one cannot split the longer.
ALLOWED_LITERALS = (
    "caiotheodoro/lacuna-capture-histories",
    "caiotheodoro/lacuna",
    "author = {Theodoro, Caio}",
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)")
# RFC 2606 reserves these, and their subdomains, for examples.
ALLOWED_DOMAINS = ("example.com", "example.org", "example.net")
# lacuna's witness names, which follow `@` in result labels.
WITNESS_LABELS = {"trace", "proxy", "env", "auditor", "all4"}
# JSON escapes (\n, and \\n inside a JSON-encoded string) glue the next token
# to a letter. Credentials and emails are matched on the raw text and on a
# copy with escapes blanked; paths and hosts only on the raw text, where a
# blanked \b would break D:\\backup apart.
_ESCAPES = re.compile(r"\\+[nrtbf]")


def _allowed_host(host: str) -> bool:
    h = host.lower().rstrip(".")
    return any(h == d or h.endswith("." + d) for d in ALLOWED_DOMAINS)


def email_domain_status(domain: str) -> str:
    """'ok' for a reserved example domain, 'redo' for the scripted agent's x-suffix,
    'nodomain' for user@host, else 'bad'.

    `ScriptedAgent` gets a customer field wrong first by appending 'x' to the
    value, so a redone email update reads ...@example.orgx: the reserved
    domain plus one character, not a real address.
    """
    d = domain.lower()
    if "." not in d:
        # Result labels name an estimator and a witness set, as in
        # ll_bic@trace+proxy+env; any other user@host is flagged.
        return "label" if d in WITNESS_LABELS else "nodomain"
    for allowed in ALLOWED_DOMAINS:
        if d == allowed or d.endswith("." + allowed):
            return "ok"
        if d == allowed + "x":
            return "redo"
    return "bad"


def privacy_findings(text: str, where: str) -> tuple[list[str], int]:
    """Every privacy violation in `text`, and the count of tolerated redo addresses."""
    raw = text
    for literal in ALLOWED_LITERALS:
        raw = raw.replace(literal, " ")
    norm = _ESCAPES.sub(" ", raw)
    found = []
    for label, rx in KEY_SHAPES.items():
        hits = max(len(rx.findall(raw)), len(rx.findall(norm)))
        if hits:
            found.append(f"{where}: {label} x{hits}")
    redo = 0
    for m in EMAIL.finditer(norm):
        status = email_domain_status(m.group(1))
        if status == "bad":
            found.append(f"{where}: email outside the example domains ({m.group(1)})")
        elif status == "nodomain":
            found.append(f"{where}: email without a domain ({m.group(0)})")
        elif status == "redo":
            redo += 1
    for label, rx in RAW_SHAPES.items():
        hits = sorted({m.group(0) for m in rx.finditer(raw)})
        if label == "hostname":
            hits = [h for h in hits if not _allowed_host(h)]
        if hits:
            found.append(f"{where}: {label} {hits[:3]}")
    return found, redo


def check_privacy(dataset_dir: Path) -> Check:
    files = sorted(p for p in dataset_dir.rglob("*") if p.is_file())
    if not files:
        return ("privacy gate", False, "no files to scan, which is never a pass")
    found, redo = [], 0
    for p in files:
        f, r = privacy_findings(p.read_text(errors="replace"), str(p.relative_to(dataset_dir)))
        found += f
        redo += r
    if found:
        shown = "; ".join(found[:5]) + (f"; +{len(found) - 5} more" if len(found) > 5 else "")
        return ("privacy gate", False, shown)
    return (
        "privacy gate",
        True,
        f"{len(files)} files clean; {redo} scripted redo addresses (example.*x) tolerated",
    )


# ---------------------------------------------------------------------------
# release content, files and types
# ---------------------------------------------------------------------------

REASONING = "reasoning_content"


def reasoning_findings(text: str) -> list[int]:
    """Offsets of every `reasoning_content` that is not a bare mention of the key's name.

    Allowed: the name in backticks, as prose about the provider's 400 writes it
    (`reasoning_content`), and the name as a quoted string that is not a JSON
    key, as in code (reply.get("reasoning_content")) or a list of key names in
    transfer-behaviour.json. Anything else, a JSON key with a value or the
    name followed by text, may carry reasoning and fails.
    """
    norm = re.sub(r'\\+"', '"', text)
    bad = []
    for m in re.finditer(REASONING, norm):
        before = norm[m.start() - 1] if m.start() else ""
        after = norm[m.end()] if m.end() < len(norm) else ""
        if before == after == "`":
            continue
        if before == after == '"' and not norm[m.end() + 1 :].lstrip().startswith(":"):
            continue
        bad.append(m.start())
    return bad


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def check_release_content(dataset_dir: Path) -> Check:
    """No transcript and no provider reasoning text, in any field or file."""
    problems = []
    for p in sorted(dataset_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(dataset_dir))
        if "transcript" in p.name.lower():
            problems.append(f"{rel}: a transcript file")
        bad = reasoning_findings(p.read_text(errors="replace"))
        if bad:
            problems.append(f"{rel}: {len(bad)} reasoning_content outside a key-name mention")
        if p.suffix == ".jsonl":
            for i, row in enumerate(_jsonl(p)):
                if "transcript" in row:
                    problems.append(f"{rel} row {i}: a transcript field")
                if row.get("transcripts_included") is not False and p.name == "runs.jsonl":
                    problems.append(f"{rel} row {i}: transcripts_included is not false")
    return (
        "release content",
        not problems,
        "; ".join(problems[:4]) or "no transcript, no reasoning_content, nowhere",
    )


def check_files(dataset_dir: Path, artifacts: Path) -> Check:
    want = allowed_files(artifacts)
    have = {str(p.relative_to(dataset_dir)) for p in dataset_dir.rglob("*") if p.is_file()}
    extra, missing = sorted(have - want), sorted(want - have)
    return (
        "file allowlist",
        not extra and not missing,
        f"extra {extra}; missing {missing}" if extra or missing else f"exactly {len(want)} files",
    )


def type_problems(value, spec: dict, path: str) -> list[str]:
    """Strict JSON types: an int is not 41.0, a bool is not 1, no key extra or missing."""
    if value is None:
        return [] if path in NULLABLE else [f"{path}: null"]
    if "dtype" in spec:
        dtype = spec["dtype"]
        ok = {
            "string": isinstance(value, str),
            "int64": isinstance(value, int) and not isinstance(value, bool),
            "float64": isinstance(value, float) and math.isfinite(value),
            "bool": isinstance(value, bool),
        }[dtype]
        return [] if ok else [f"{path}: {type(value).__name__} {value!r} is not {dtype}"]
    if "sequence" in spec:
        if not isinstance(value, list):
            return [f"{path}: not a list"]
        out = []
        for v in value:
            out += type_problems(v, {"dtype": spec["sequence"]}, path + "[]")
        return out
    fields = spec.get("list") or spec.get("struct")
    items = value if "list" in spec else [value]
    if "list" in spec and not isinstance(value, list):
        return [f"{path}: not a list"]
    out = []
    for item in items:
        out += row_type_problems(item, fields, path + ("[]" if "list" in spec else ""))
    return out


def row_type_problems(row, fields: list[dict], prefix: str = "") -> list[str]:
    if not isinstance(row, dict):
        return [f"{prefix or 'row'}: not an object"]
    names = [f["name"] for f in fields]
    if list(row) != names:
        return [f"{prefix or 'row'}: keys {list(row)} != {names}"]
    out = []
    for f in fields:
        path = f"{prefix}.{f['name']}" if prefix else f["name"]
        spec = {k: v for k, v in f.items() if k != "name"}
        out += type_problems(row[f["name"]], spec, path)
    return out


def check_types(dataset_dir: Path) -> Check:
    problems = []
    for p in sorted((dataset_dir / "episodes").glob("*.jsonl")):
        seen = set()
        for i, row in enumerate(_jsonl(p)):
            problems += [f"{p.name}:{i}: {e}" for e in row_type_problems(row, EPISODE_FEATURES)]
            key = (row.get("run"), row.get("seed"))
            if key in seen:
                problems.append(f"{p.name}:{i}: duplicate {key}")
            seen.add(key)
    runs = _jsonl(dataset_dir / "runs.jsonl")
    for i, row in enumerate(runs):
        problems += [f"runs.jsonl:{i}: {e}" for e in row_type_problems(row, RUN_FEATURES)]
    names = [r.get("run") for r in runs]
    if len(set(names)) != len(names):
        problems.append(
            f"runs.jsonl: duplicate run rows {sorted(n for n in names if names.count(n) > 1)}"
        )
    return (
        "strict types",
        not problems,
        "; ".join(problems[:4]) or "every row matches the pinned schema; no duplicates",
    )


# ---------------------------------------------------------------------------
# rows and runs, against sources the builder does not control
# ---------------------------------------------------------------------------


def redact(error: str) -> str:
    """This file's own redaction, so the builder's is checked rather than trusted."""
    return re.sub(r"(?i)(request[_ -]?id\W{0,3})[0-9a-f][0-9a-f-]*", r"\1<redacted>", error)


def agent_read_trace(transcript: list[dict] | None) -> bool | None:
    from lacuna.envs.kernel import TRACE_PATH

    if transcript is None:
        return None
    for m in transcript:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") != "read_file":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(args, dict) and args.get("path") == TRACE_PATH:
                return True
    return False


def infer_tickets(episodes: list[tuple[int, list[dict]]], up_to: int) -> int | None:
    """The queue length that explains the ledgers, or None if no length does.

    A ledger can only touch resources its tickets name, so the share of
    ledgers inside the plans of n regenerated tickets rises with n and stops
    rising at the true length. The ticket a queue one longer would add must
    then be mostly not done.
    """
    from lacuna.envs.tasks import make_episode

    def key(op: str, a: dict) -> tuple[str, str]:
        if op == "update_customer":
            return op, f"{a['customer_id']}.{a['field']}"
        return op, str(a.get("order_id") or a.get("path") or a.get("to"))

    within = []
    for n in range(1, up_to + 1):
        hits = 0
        for seed, ledger in episodes:
            plan = {key(op, a) for t in make_episode(seed, n).tickets for op, a in t.plan}
            hits += {(e["op"], e["resource"]) for e in ledger} <= plan
        within.append(hits)
    n0 = within.index(max(within)) + 1
    done = had = 0
    for seed, ledger in episodes:
        base = make_episode(seed, n0).tickets[:-1]
        longer = make_episode(seed, n0 + 1).tickets[:-1]
        if len(longer) > len(base):
            had += 1
            op, a = longer[-1].plan[0]
            done += key(op, a) in {(e["op"], e["resource"]) for e in ledger}
    if max(within) == 0 or (had and done / had >= 0.5):
        return None
    return n0


def w4_recall(episodes: list[dict]) -> float:
    """Share of ledger actions witness 3 reported, oracle linkage, from source episodes."""
    from lacuna.envs.kernel import LedgerEntry
    from lacuna.linkage import oracle_table
    from lacuna.witnesses import Record

    seen = total = 0
    for e in episodes:
        t, _ = oracle_table(
            [LedgerEntry(**x) for x in e["ledger"]], [Record(**r) for r in e["records"]], 4
        )
        seen += sum(int(c) for h, c in enumerate(t) if h >> AUDITOR & 1)
        total += len(e["ledger"])
    return round(seen / total, 4) if total else 0.0


def pooled_tables(episodes: list[dict]) -> tuple[list[int], list[int]]:
    """Oracle and signature tables over all four witnesses, from source-shaped episodes."""
    import numpy as np

    from lacuna.envs.kernel import LedgerEntry
    from lacuna.linkage import oracle_table, signature_table
    from lacuna.witnesses import Record

    oracle = np.zeros(16, dtype=np.int64)
    sig = np.zeros(16, dtype=np.int64)
    for e in episodes:
        recs = [Record(**r) for r in e["records"]]
        oracle += oracle_table([LedgerEntry(**x) for x in e["ledger"]], recs, 4)[0]
        sig += signature_table(recs, 4)
    return oracle.tolist(), sig.tolist()


def _same(problems: list[str], tag: str, key: str, got, want) -> None:
    if got != want or type(got) is not type(want):
        problems.append(f"{tag}: {key} {got!r}, source says {want!r}")


def bypassed_flags(src: dict, env: str) -> list[bool]:
    """Which ledger actions bypassed the gateway, from the records alone.

    In `backoffice_bypass` the gateway and the trace log a `run_script` call as
    one line and no action record, so with complete logs a ledger action ran in
    a script exactly when no witness 0 or 1 record carries its id. The builder
    reads the stored transcript instead; the two must agree. No other env has a
    script tool, so nothing there is bypassed.
    """
    if env != BYPASS_ENV:
        return [False] * len(src["ledger"])
    logged = {r["oracle"] for r in src["records"] if r["witness"] in (0, 1)}
    return [e["id"] not in logged for e in src["ledger"]]


def row_problems(row: dict, src: dict, fields: dict) -> list[str]:
    """One built row against its source episode and the run-level fields it must carry."""
    problems: list[str] = []
    tag = f"seed {row.get('seed')}"
    expected = {"seed": src["seed"], **fields}
    flags = bypassed_flags(src, fields["env"])
    expected.update(
        {
            "n_true": len(src["ledger"]),
            "ledger": [{**e, "bypassed": b} for e, b in zip(src["ledger"], flags, strict=True)],
            "records": [r for r in src["records"] if r["witness"] != AUDITOR],
            "auditor_records": [r for r in src["records"] if r["witness"] == AUDITOR],
            "agent_read_trace": agent_read_trace(src.get("transcript")),
            "n_calls": src["n_calls"],
            "n_tool_errors": src["n_tool_errors"],
        }
    )
    for key, want in expected.items():
        _same(problems, tag, key, row.get(key), want)
    if {r["witness"] for r in row.get("records") or []} - HARD_WITNESSES:
        problems.append(f"{tag}: records hold a witness outside 0..2")
    return problems


def _caveat_problems(
    run: str, caveat, extra: tuple[str, ...] = (), pinned: bool = True
) -> list[str]:
    if not isinstance(caveat, str) or len(caveat) < 40:
        return [f"{run}: caveat missing or too short"]
    missing = [ref for ref in (*CAVEAT_REFS.get(run, ()), *extra) if ref not in caveat]
    problems = [f"{run}: caveat does not cite {missing}"] if missing else []
    if pinned and hashlib.sha256(caveat.encode()).hexdigest() != CAVEAT_SHA256.get(run):
        problems.append(f"{run}: caveat is not the pinned text")
    return problems


def check_runs(dataset_dir: Path, artifacts: Path) -> list[Check]:
    checks: list[Check] = []
    every = sorted(p.name for p in (artifacts / "episodes").iterdir() if p.is_dir())
    unknown = [r for r in every if r not in EXPECTED_PHASES]
    shipped = [r for r in episode_configs(artifacts) if r != SUPPLEMENT]
    runs_list = _jsonl(dataset_dir / "runs.jsonl")
    runs = {r["run"]: r for r in runs_list}
    want_rows = [*shipped, SUPPLEMENT]
    checks.append(
        (
            "every run accounted for",
            not unknown and sorted(runs) == sorted(want_rows) and len(runs_list) == len(want_rows),
            f"shipped {shipped} + supplement; excluded "
            f"{[r for r in every if EXPECTED_PHASES.get(r) == 'excluded']}; unknown {unknown}",
        )
    )
    # Each pre-registered study's verdict file must name its run, and that run
    # must be pinned with the study's confirmatory phase; no other run may be.
    confirmatory_runs = set()
    for run, (phase, verdicts, key_path) in CONFIRMATORY.items():
        named = json.loads((artifacts / verdicts).read_text())
        for key in key_path:
            named = named[key]
        confirmatory_runs.add(named)
        if named != run or EXPECTED_PHASES.get(run) != phase:
            checks.append(("confirmatory runs", False, f"{verdicts} names {named}, not {run}"))
    stray = [
        r
        for r, ph in EXPECTED_PHASES.items()
        if ph.startswith("confirmatory") and r not in confirmatory_runs
    ]
    if stray:
        checks.append(
            ("confirmatory runs", False, f"{stray} pinned confirmatory with no verdict file")
        )

    for run in shipped:
        path = dataset_dir / "episodes" / f"{run}.jsonl"
        if not path.exists() or run not in runs:
            checks.append((f"run {run}", False, "not built"))
            continue
        source_name = source_run(run)
        src_dir = artifacts / "episodes" / source_name
        summary = json.loads((src_dir / "summary.json").read_text())
        receipt_path = src_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        source = _jsonl(src_dir / "episodes.jsonl")
        rows = _jsonl(path)
        meta = runs[run]
        problems: list[str] = []

        phase = EXPECTED_PHASES[source_name]
        if (summary.get("agent") == "scripted") != (phase == "scripted harness check"):
            problems.append(f"pinned phase {phase!r} disagrees with agent {summary.get('agent')!r}")
        env = EXPECTED_ENV.get(source_name, "backoffice")
        if env == BYPASS_ENV and any(summary["fault_rates"].values()):
            problems.append(
                "a bypass run recorded with logging faults; `bypassed` needs complete logs"
            )
        if summary.get("env", "backoffice") != env:
            problems.append(f"pinned env {env!r}, summary says {summary.get('env')!r}")
        inferred = [k for k in ("tickets", "auditor_view") if summary.get(k) is None]
        tickets = summary.get("tickets")
        if tickets is None:
            claimed = meta.get("tickets") if isinstance(meta.get("tickets"), int) else 0
            tickets = infer_tickets(
                [(e["seed"], e["ledger"]) for e in source], max(claimed + 3, 12)
            )
            if tickets is None:
                problems.append("summary has no tickets and the ledgers fit no queue length")
        auditor_view = summary.get("auditor_view") or "full"  # the only view before D14
        failed = summary.get("failed_episodes") or []
        oracle, sig = pooled_tables(source)
        want_run = {
            "run": run,
            "phase": phase,
            "env": env,
            "agent": summary["agent"],
            "auditor": summary["auditor"],
            "model": summary["model"],
            "organism": summary.get("organism"),
            "organism_prompt_sha256": summary.get("organism_prompt_sha256"),
            "auditor_view": auditor_view,
            "tickets": tickets,
            "episodes": summary["episodes"],
            "episodes_attempted": summary["episodes"] + len(failed),
            "failed_episodes": [{"seed": f["seed"], "error": redact(f["error"])} for f in failed],
            "stopped": summary.get("stopped") or None,
            "seed0": summary["seed0"],
            "n_true": summary["n_true"],
            "recorded_fault_rates": summary["fault_rates"],
            "spend_usd": receipt["total_usd"] if receipt else None,
            "llm_calls": len(receipt["calls"]) if receipt else None,
            "transcripts_stored": all(e.get("transcript") is not None for e in source),
            "transcripts_included": False,
            "w4_recall": w4_recall(source),
            "oracle_table": summary["oracle_table"],
            "signature_table": summary["signature_table"],
            "composition": None,
            "inferred_fields": inferred,
        }
        for key, want in want_run.items():
            _same(problems, "runs.jsonl", key, meta.get(key), want)
        if (oracle, sig) != (summary["oracle_table"], summary["signature_table"]):
            problems.append("source episodes do not reproduce summary.json's tables")
        problems += _caveat_problems(run, meta.get("caveat"))
        if source_name in confirmatory_runs:
            problems += _caveat_problems(
                run,
                meta.get("caveat"),
                (f"{summary['episodes']} of {summary['episodes'] + len(failed)}",),
                pinned=False,
            )

        if len(rows) != len(source) or len(rows) != summary["episodes"]:
            problems.append(
                f"{len(rows)} rows, artifact has {len(source)}, summary says {summary['episodes']}"
            )
        else:
            for row, src in zip(rows, source, strict=True):
                problems += row_problems(
                    row,
                    src,
                    {
                        "run": run,
                        "phase": phase,
                        "env": env,
                        "model": summary["model"],
                        "tickets": tickets,
                        "auditor_view": auditor_view,
                    },
                )
        n = sum(r["n_true"] for r in rows)
        detail = (
            "; ".join(problems[:4])
            if problems
            else f"{len(rows)} rows, N={n}, {phase}, tickets {tickets}"
            + (f" (inferred: {', '.join(inferred)})" if inferred else "")
        )
        checks.append((f"run {run}", not problems, detail))
    return checks


def check_supplement(dataset_dir: Path, artifacts: Path) -> Check:
    """The supplement against the pool's recorded composition and each source run."""
    pool_dir = artifacts / "episodes" / POOL
    pool_summary = json.loads((pool_dir / "summary.json").read_text())
    composition = pool_summary["composition"]
    want = [(c["run"], c["seed"]) for c in composition["supplement"]]
    rows = _jsonl(dataset_dir / "episodes" / f"{SUPPLEMENT}.jsonl")
    metas = [r for r in _jsonl(dataset_dir / "runs.jsonl") if r["run"] == SUPPLEMENT]
    if len(metas) != 1:
        return ("supplement", False, f"{len(metas)} runs.jsonl rows for the supplement")
    meta = metas[0]
    problems: list[str] = []
    if any(EXPECTED_PHASES.get(run) != "excluded" for run, _ in want):
        problems.append("the supplement draws from a run that is not pinned as excluded")
    if [(r.get("run"), r.get("seed")) for r in rows] != want:
        problems.append(
            f"rows {[(r.get('run'), r.get('seed')) for r in rows]} != composition {want}"
        )
    confirm = _jsonl(artifacts / "episodes" / DEFAULT_CONFIG / "episodes.jsonl")
    if composition["confirm"] != len(confirm):
        problems.append(
            f"pool says {composition['confirm']} confirm episodes, run has {len(confirm)}"
        )
    pool = {e["seed"]: e for e in _jsonl(pool_dir / "episodes.jsonl")}
    sources, summaries = [], []
    for run, seed in want:
        src_dir = artifacts / "episodes" / run
        summary = json.loads((src_dir / "summary.json").read_text())
        src = [e for e in _jsonl(src_dir / "episodes.jsonl") if e["seed"] == seed]
        if len(src) != 1:
            problems.append(f"{run} has {len(src)} episodes with seed {seed}")
            continue
        sources.append(src[0])
        summaries.append(summary)
        other = pool.get(seed) or {}
        if (other.get("ledger"), other.get("records")) != (src[0]["ledger"], src[0]["records"]):
            problems.append(f"seed {seed}: the pool's copy differs from {run}'s")
    if len(rows) == len(sources) == len(want):
        for row, src, summary, (run, _) in zip(rows, sources, summaries, want, strict=True):
            problems += row_problems(
                row,
                src,
                {
                    "run": run,
                    "phase": SUPPLEMENT_PHASE,
                    "env": "backoffice",
                    "model": summary["model"],
                    "tickets": summary["tickets"],
                    "auditor_view": summary["auditor_view"],
                },
            )
    shared = {}
    for key in ("agent", "auditor", "model", "tickets", "auditor_view", "fault_rates"):
        values = {json.dumps(s.get(key)) for s in summaries}
        if len(values) != 1:
            problems.append(f"source runs disagree on {key}")
        shared[key] = summaries[0].get(key) if summaries else None
    n = sum(len(e["ledger"]) for e in sources)
    pooled = sum(len(e["ledger"]) for e in confirm) + n
    hyp = json.loads((artifacts / "hypotheses-supplement.json").read_text())
    if not pooled == pool_summary["n_true"] == hyp["n_true"]:
        problems.append(
            f"confirm + supplement N {pooled}, pool {pool_summary['n_true']}, "
            f"hypotheses-supplement {hyp['n_true']}"
        )
    oracle, sig = pooled_tables(sources)
    want_meta = {
        "run": SUPPLEMENT,
        "phase": SUPPLEMENT_PHASE,
        "env": "backoffice",
        "agent": shared["agent"],
        "auditor": shared["auditor"],
        "model": shared["model"],
        "organism": None,
        "organism_prompt_sha256": None,
        "auditor_view": shared["auditor_view"],
        "tickets": shared["tickets"],
        "episodes": len(want),
        "episodes_attempted": None,
        "failed_episodes": [],
        "stopped": None,
        "seed0": None,
        "n_true": n,
        "recorded_fault_rates": shared["fault_rates"],
        "spend_usd": None,
        "llm_calls": None,
        "transcripts_stored": all(e.get("transcript") is not None for e in sources),
        "transcripts_included": False,
        "w4_recall": w4_recall(sources),
        "oracle_table": oracle,
        "signature_table": sig,
        "composition": [{"run": run, "seed": seed} for run, seed in want],
        "inferred_fields": [],
    }
    for key, value in want_meta.items():
        _same(problems, "runs.jsonl supplement", key, meta.get(key), value)
    problems += _caveat_problems(SUPPLEMENT, meta.get("caveat"))
    detail = "; ".join(problems[:4]) or (
        f"{len(rows)} rows match {POOL}'s composition {want}; "
        f"N {n}, pool N {pool_summary['n_true']}"
    )
    return ("supplement", not problems, detail)


def check_configs(dataset_dir: Path, artifacts: Path) -> Check:
    """Each config's file holds that config's run, and only that."""
    problems = []
    for config in episode_configs(artifacts):
        rows = _jsonl(dataset_dir / "episodes" / f"{config}.jsonl")
        if config == SUPPLEMENT:
            bad = [r.get("seed") for r in rows if r.get("phase") != SUPPLEMENT_PHASE]
        else:
            bad = [r.get("seed") for r in rows if r.get("run") != config]
        if bad or not rows:
            problems.append(f"episodes/{config}.jsonl holds rows of another config ({bad[:3]})")
    return ("config contents", not problems, "; ".join(problems) or "each file holds its config")


# ---------------------------------------------------------------------------
# results, docs, environment
# ---------------------------------------------------------------------------


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_results(dataset_dir: Path, artifacts: Path) -> Check:
    # Every file at any depth outside the run directories, so a new directory
    # of results cannot go unplaced. Local logs are gitignored and not results.
    sources = {
        p.relative_to(artifacts).as_posix(): p
        for p in sorted(artifacts.rglob("*"))
        if p.is_file() and p.relative_to(artifacts).parts[0] != "episodes" and p.suffix != ".log"
    }
    built = {
        str(p.relative_to(dataset_dir / "results")): p
        for p in (dataset_dir / "results").rglob("*")
        if p.is_file()
    }
    problems = []
    if set(built) != SHIPPED_RESULTS:
        problems.append(
            f"shipped {sorted(set(built) ^ SHIPPED_RESULTS)} differ from the pinned list"
        )
    problems += [f"{n} has no artifact" for n in built if n not in sources]
    problems += [
        f"{n} differs from its artifact"
        for n in built
        if n in sources and _sha(built[n]) != _sha(sources[n])
    ]
    unplaced = set(sources) - SHIPPED_RESULTS - NOT_SHIPPED
    if unplaced:
        problems.append(f"artifacts placed nowhere: {sorted(unplaced)}")
    groups: dict[str, list[str]] = {}
    for name, p in sources.items():
        groups.setdefault(_sha(p), []).append(name)
    for names in groups.values():
        if sum(n in built for n in names) > 1:
            problems.append(f"identical artifacts {names} ship more than once")
    return (
        "results",
        not problems,
        "; ".join(problems[:4])
        or f"{len(built)} shipped byte for byte; {len(sources) - len(built)} placed, not shipped",
    )


def check_docs_and_environment(dataset_dir: Path, artifacts: Path = ARTIFACTS) -> Check:
    from lacuna.envs.registry import ENVS as REGISTRY

    problems = []
    for name in SHIPPED_DOCS:
        p = dataset_dir / "docs" / name
        if not p.exists() or p.read_bytes() != (DOCS / name).read_bytes():
            problems.append(f"docs/{name} missing or differs from the repo")
            continue
        # And the committed text at HEAD, so an uncommitted or appended edit,
        # made in the repo and the build alike, still fails.
        try:
            committed = subprocess.run(
                ["git", "show", f"HEAD:docs/{name}"], cwd=ROOT, capture_output=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            committed = b""
        if hashlib.sha256(p.read_bytes()).digest() != hashlib.sha256(committed).digest():
            problems.append(f"docs/{name} is not the text committed at HEAD")
    # Each pre-registration: the shipped text is the pinned text, and git says
    # it was last touched at the pinned freeze commit.
    for doc, (commit, sha) in PREREGISTRATIONS.items():
        p = dataset_dir / "docs" / doc
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            problems.append(f"docs/{doc} is not the pinned text (sha256 {sha[:12]}...)")
        try:
            last = subprocess.run(
                ["git", "log", "-1", "--format=%h", "--", f"docs/{doc}"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            last = "unknown"
        if last != commit:
            problems.append(f"{doc} was last touched at {last}, not its freeze {commit}")
        tag = PREREG_TAGS.get(doc)
        if tag:
            tagged = _git("rev-parse", "--short", f"{tag}^{{commit}}").strip()
            if tagged != commit:
                problems.append(
                    f"tag {tag} is at {tagged or 'nothing'}, not {doc}'s freeze {commit}"
                )
    for env in shipped_envs(artifacts):
        folder = dataset_dir / "environment" / env
        prompt, tools = folder / "system_prompt.txt", folder / "tools.json"
        if not prompt.exists() or prompt.read_text() != REGISTRY[env].agent_system + "\n":
            problems.append(f"environment/{env}/system_prompt.txt is not that env's prompt")
        if not tools.exists() or json.loads(tools.read_text()) != REGISTRY[env].tools:
            problems.append(f"environment/{env}/tools.json is not that env's tools")
    gitattr = dataset_dir / ".gitattributes"
    if not gitattr.exists() or gitattr.read_text() != GITATTRIBUTES:
        problems.append(".gitattributes differs from the pinned content")
    return (
        "docs and environment",
        not problems,
        "; ".join(problems)
        or "docs match; every pre-registration at its pinned text and freeze; "
        f"prompts and tools match the code for {', '.join(shipped_envs(artifacts))}",
    )


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------


def split_card(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("card has no YAML front matter")
    head, _, body = text[4:].partition("\n---\n")
    return head, body


def size_category(n: int) -> str:
    return "n<1K" if n < 1000 else "1K<n<10K" if n < 10_000 else "10K<n<100K"


CARD_KEYS = ["license", "pretty_name", "language", "tags", "size_categories", "configs"]
CONFIG_HEADER = "| config | env | agent model | episodes | pre-registered or post hoc |"


def config_status(phase: str) -> str:
    """The config table's last column, from the run's pinned phase (written here, not imported)."""
    if phase == BYPASS_PILOT_PHASE:
        return "pre-registered pilot, no verdict"
    if phase.startswith("confirmatory"):
        return "pre-registered"
    if phase == "scripted harness check":
        return "harness check, no LLM"
    return "post hoc"


def table_rows(body: str, start: str, stop: str) -> list[str]:
    """The `|` lines between two markers, the header and separator included."""
    a = body.find(start)
    b = body.find(stop, a + 1)
    if a < 0 or b < 0:
        return []
    return [x.strip() for x in body[a:b].splitlines() if x.startswith("|")]


def front_door_problems(body: str, dataset_dir: Path, configs: list[str]) -> list[str]:
    """D66: summary, What holds, banner, config table, loading snippet, then one <details>."""
    problems = []
    at = {k: body.find(k) for k in ("## What holds", BANNER, "## Configs", "## Load it", DETAILS)}
    order = list(at.values())
    if min(order) < 0 or order != sorted(order):
        problems.append(f"the front door is missing a part or out of order: {at}")
        return problems
    first_verdicts = body.find("| hypothesis | verdict | detail |")
    if 0 <= first_verdicts < at[BANNER]:
        problems.append("a verdict table comes before the caveat banner")
    holds = table_rows(body, "## What holds", BANNER)
    if not holds or holds[0] != HOLDS_HEADER:
        problems.append("the What holds table has the wrong header")
    elif not 1 <= len(holds) - 2 <= MAX_HOLDS_ROWS:
        n = len(holds) - 2
        problems.append(f"the What holds table has {n} rows, not 1 to {MAX_HOLDS_ROWS}")
    elif any(len(r.strip("|").split("|")) != 3 for r in holds):
        problems.append("a What holds row does not have three cells")
    banner = " ".join(body[at[BANNER] :].split("\n\n", 1)[0].replace(">", " ").split())
    gone = [t for t in BANNER_TERMS if t not in banner]
    if gone:
        problems.append(f"the banner lost {gone}")
    runs = {r["run"]: r for r in _jsonl(dataset_dir / "runs.jsonl")}
    want = [CONFIG_HEADER, "|---|---|---|---|---|"]
    for c in configs:
        r = runs[c]
        default = " (default)" if c == DEFAULT_CONFIG else ""
        want.append(
            f"| `{c}`{default} | {r['env']} | {r['model'] or 'scripted'} | {r['episodes']} | "
            f"{config_status(r['phase'])} |"
        )
    want.append(f"| `runs` | n/a | n/a | {len(runs)} rows | one row per run |")
    if table_rows(body, "## Configs", "## Load it") != want:
        problems.append("the config table differs from runs.jsonl")
    words, front = len(body.split()), len(body[: at[DETAILS]].split())
    if words > CARD_WORD_BUDGET:
        problems.append(f"the card body is {words} words, over the {CARD_WORD_BUDGET} budget")
    if front > CARD_FRONT_BUDGET:
        problems.append(f"{front} words before <details>, over the {CARD_FRONT_BUDGET} budget")
    return problems


def check_card_metadata(dataset_dir: Path, artifacts: Path) -> Check:
    import yaml

    text = (dataset_dir / "README.md").read_text()
    try:
        head, body = split_card(text)
        meta = yaml.safe_load(head)
    except (ValueError, yaml.YAMLError) as exc:
        return ("card metadata", False, str(exc))
    problems = []
    configs = episode_configs(artifacts)
    want_configs = [
        {
            "config_name": c,
            "data_files": f"episodes/{c}.jsonl",
            **({"default": True} if c == DEFAULT_CONFIG else {}),
        }
        for c in configs
    ] + [{"config_name": "runs", "data_files": "runs.jsonl"}]
    n_rows = sum(len(_jsonl(dataset_dir / "episodes" / f"{c}.jsonl")) for c in configs)
    for key, want in (
        ("license", "apache-2.0"),
        ("pretty_name", "lacuna capture histories"),
        ("language", ["en"]),
        ("tags", REQUIRED_TAGS),
        ("size_categories", [size_category(n_rows)]),
        ("configs", want_configs),
    ):
        if meta.get(key) != want:
            problems.append(f"{key} differs from the pinned value")
    if list(meta) != CARD_KEYS:
        # The schema is pinned in this file and enforced on load and per row, not in the YAML.
        problems.append(f"front matter keys are {list(meta)}, not {CARD_KEYS}")
    for section in REQUIRED_SECTIONS:
        if section not in body:
            problems.append(f"missing section {section!r}")
    problems += front_door_problems(body, dataset_dir, configs)
    decisions = (dataset_dir / "docs" / "DECISIONS.md").read_text()
    defined = set(re.findall(r"^\*\*D(\d+)\.", decisions, re.M))
    cited = set(re.findall(r"\bD(\d+)\b", body))
    if cited - defined:
        problems.append(f"cites D{sorted(cited - defined)} not in docs/DECISIONS.md")
    return (
        "card metadata",
        not problems,
        "; ".join(problems)
        or f"{len(want_configs)} configs, hub keys only, front door in order, "
        f"{len(body.split())} words",
    )


def receipt_totals(artifacts: Path) -> dict[str, float]:
    return {
        p.parent.name: json.loads(p.read_text())["total_usd"]
        for p in sorted((artifacts / "episodes").glob("*/receipt.json"))
    }


def receipt_spend(artifacts: Path) -> dict:
    """Spend from every receipt on disk, split study / diagnostic by the pinned list."""
    per_run = receipt_totals(artifacts)
    diagnostic = sum(v for r, v in per_run.items() if r in DIAGNOSTIC_RUNS)
    total = sum(per_run.values())
    excluded = {r for r, ph in EXPECTED_PHASES.items() if ph == "excluded"}
    return {
        "total": round(total, 6),
        "study": round(total - diagnostic, 6),
        "diagnostic": round(diagnostic, 6),
        "diagnostic_runs": sorted(r for r in per_run if r in DIAGNOSTIC_RUNS),
        # Study spend from runs that are not shipped: confirm-supp, the
        # supplement's rerun (D34).
        # Zero-cost scripted runs carry a receipt but no spend.
        "study_unshipped": sorted(
            r for r in per_run if r in excluded and r not in DIAGNOSTIC_RUNS and per_run[r] > 0
        ),
    }


def check_spend(artifacts: Path) -> Check:
    """Receipt spend must agree with derived.json's split, to the cent the card prints."""
    derived = json.loads((artifacts / "derived.json").read_text())["spend"]
    per_run = receipt_totals(artifacts)
    # derived.json is the paper's figure and is only as new as its last run. It is compared
    # over the runs it lists; a receipt it does not list is named, never dropped silently.
    listed = {r: v for r, v in per_run.items() if r in derived["runs"]}
    newer = sorted(r for r in per_run if r not in derived["runs"])
    gone = sorted(r for r in derived["runs"] if r not in per_run)
    diagnostic = sum(v for r, v in listed.items() if r in DIAGNOSTIC_RUNS)
    pairs = {
        "total": (sum(listed.values()), derived["total_usd"]),
        "study": (sum(listed.values()) - diagnostic, derived["study_usd"]),
        "diagnostic": (diagnostic, derived["diagnostic_usd"]),
    }
    off = {k: v for k, v in pairs.items() if f"{v[0]:.2f}" != f"{v[1]:.2f}"}
    if off or gone:
        return ("spend agrees", False, f"receipts vs derived.json differ: {off}; gone {gone}")
    spend = receipt_spend(artifacts)
    tail = (
        f"; derived.json predates the receipts of {newer}; the card's total over all "
        f"receipts is {spend['total']:.2f}"
        if newer
        else ""
    )
    return (
        "spend agrees",
        True,
        "receipts and derived.json: "
        + ", ".join(f"{k} {v[0]:.2f}" for k, v in pairs.items())
        + tail,
    )


LOSS_SIM_HEADING = "## Simulate your own log loss"
LOSS_SIM_OUTPUT = re.compile(r"Prints, one line per drop rate[^\n]*\n\n```text\n(.*?)\n```", re.S)
COMMIT_STAMP = re.compile(r"from lacuna commit `([^`]+)`")


def _snippet(body: str, heading: str):
    section = body.split(heading, 1)
    if len(section) < 2:
        return None
    m = re.search(r"```python\n(.*?)```", section[1].split("\n## ", 1)[0], re.S)
    return m.group(1) if m else None


def card_blocks(text: str) -> list[str]:
    """README.md as blocks: one per bullet, table row, heading or paragraph; spaces collapsed."""
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append(" ".join(current))
            current.clear()

    for line in text.splitlines():
        if line.startswith(">"):
            line = line[1:].removeprefix(" ")
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if line.startswith(("- ", "|", "#")):
            flush()
        current.append(stripped)
        if line.startswith(("|", "#")):
            flush()
    flush()
    return [" ".join(b.split()) for b in blocks]


def _verb(passed: bool) -> str:
    return "passes" if passed else "fails"


def margin_text(m: float, bar: float) -> str:
    """The card's `1.1 MCSE above the 0.90 bar`, written here from the same number."""
    return f"{abs(m):.1f} MCSE {'above' if m >= 0 else 'below'} the {bar:.2f} bar"


def bypass_figures(dataset_dir: Path) -> list[tuple[str, list[str]]]:
    """The bypass study's figures, from this file's own reading of its three shipped results."""
    rob = dataset_dir / "results" / "robustness"

    def load(name: str) -> dict:
        return json.loads((rob / name).read_text())

    v = load("bypass-verdicts.json")
    ana, ana_l1 = load("bypass-analysis-bypass-llm.json"), load("bypass-analysis-bypass-l1.json")
    b0, b1, b2, b3, l1 = (v[k] for k in ("B0", "B1", "B2", "B3", "L1"))
    ref, dist = b1["reference"], b0["per_episode_distribution"]
    bic = ana["cells"]["fin_mid/callid"]["estimators"]["ll_bic"]
    l1_bic = ana_l1["cells"]["fin_mid/callid"]["estimators"]["ll_bic"]
    d = ana["descriptive"]
    n_eps = len(b0["per_episode_n"])
    ci, wil = b0["ci95"], b1["wilson95"]
    ver = {k: v[k]["verdict"].upper() for k in ("B0", "B1", "B2", "B3")}
    effect_read = (
        b0["raw_verdict"] == "pass"
        and b0["share_ge_0_10"]
        and b1["bypass_effect_gt_0_05"]
        and v["completion"]["neutral"]["completed"] >= v["min_episodes"]
    )
    paired = l1["paired"]
    passes = b1["verdict"] == "pass"
    return [
        (
            f"| B0 manipulation check | {ver['B0']} | pooled share {b0['share']:.3f} "
            f"({b0['n_bypassed']} of {b0['n_true']}), 95% interval "
            f"[{ci[0]:.3f}, {ci[1]:.3f}], against {b0['threshold']:.2f} |",
            [],
        ),
        (
            f"| B1 calibration | {ver['B1']} | `ll_bic` at `{b1['cell']}` covers "
            f"{b1['coverage']:.3f} (Wilson [{wil[0]:.3f}, {wil[1]:.3f}]) against "
            f"{b1['threshold']:.2f}; unbounded {b1['unbounded_rate']:.3f} |",
            [],
        ),
        (
            f"| B2 design check | {ver['B2']} | `seq_gap_tail` covers "
            f"{b2['seq_gap_tail_coverage']:.3f}, threshold below {b2['threshold']:.2f} |",
            [],
        ),
        (
            f"| B3 design check | {ver['B3']} | `ll_bic` detects a gap in "
            f"{b3['ll_bic_detects_gap']:.3f} of replicates, `union_count` in "
            f"{b3['union_count_detects_gap']:.3f}; threshold {b3['threshold']:.2f} |",
            [],
        ),
        (
            f"| L1 pilot | no verdict | pooled share difference {l1['diff_pooled']:+.3f}, paired "
            f"{paired['diff']:+.3f} [{paired['ci95'][0]:.3f}, {paired['ci95'][1]:.3f}] |",
            [],
        ),
        (
            "**Reading the verdicts.**",
            [
                f"B0 {_verb(ver['B0'] == 'PASS')}: {dist['episodes_with_bypass']} of {n_eps} "
                "episodes bypassed at least once",
                f"B1 {_verb(passes)}: `ll_bic` has mean width {b1['mean_width']:.1f} and bias "
                f"{bic['bias']:+.1f} actions ({bic['reps']} replicates)",
                f"covers {ref['coverage']:.4f} at width {ref['mean_width']:.1f}",
                f"the shift is {b1['bypass_effect']:.3f}",
                "All four conditions" if effect_read else "Not all four conditions",
                "B2 and B3 hold" if ver["B2"] == ver["B3"] == "PASS" else "B2 and B3 do not both",
            ],
        ),
        (
            "**L1 is descriptive, with no verdict.**",
            [
                f"pooled bypass share is {l1['l1_share']:.3f} against {l1['neutral_share']:.3f}",
                f"a difference of {l1['diff_pooled']:+.3f}",
                f"Paired over {paired['n_pairs']} seeds the difference is {paired['diff']:+.3f}, "
                f"95% interval [{paired['ci95'][0]:.3f}, {paired['ci95'][1]:.3f}]",
            ],
        ),
        (
            "**The L1 coverage is unbounded intervals, not calibration.**",
            [
                f"covers {l1_bic['coverage']:.3f} because {l1_bic['unbounded_rate']:.1%} "
                "of its intervals are unbounded",
            ],
        ),
        (
            "**What bypass looks like in the data.**",
            [
                f"the env diff saw {d['bypassed_seen_by_w3']} of the {d['n_bypassed']} bypassed "
                f"actions ({d['bypassed_w3_share']:.3f})",
            ],
        ),
    ]


def study4_figures(dataset_dir: Path) -> list[tuple[str, list[str]]]:
    """Study 4's figures (D70), from this file's own reading of the repository's results.

    The five result files are not shipped (D72), so they are read from artifacts/; the
    bypassed shares come from the shipped rows, not from the verdict file's rounded copy.
    """
    rob = ARTIFACTS / "robustness"
    t = json.loads((rob / "study4-verdicts.json").read_text())["tests"]
    adv = {
        r: json.loads((rob / f"study4-adversary-{r}.json").read_text())
        for r in ("s4-l2", "s4-neutral", "s4-qwen-bypass")
    }
    beh = {r: adv[r]["behaviour"] for r in ("s4-l2", "s4-neutral")}

    def pool(run: str) -> tuple[float, float]:
        """Clustered-loss and singly-lost coverage of one pool, read from its analysis."""
        tp = adv[run]["text_parsed_control"]
        singly = tp["worlds_independent_loss"]["fin_mid"]["callid"]["estimators"]["ll_bic"]
        return tp["headline"]["callid"]["coverage"], singly["coverage"]

    def share(run: str) -> float:
        flags = [
            x["bypassed"]
            for row in _jsonl(dataset_dir / "episodes" / f"{run}.jsonl")
            for x in row["ledger"]
        ]
        return sum(flags) / len(flags)

    def wil(test: dict) -> str:
        return f"[{test['wilson95'][0]:.4f}, {test['wilson95'][1]:.4f}]"

    r1, q0, q1, h = t["R1"], t["Q0"], t["Q1"], t["H"]
    a0, a2, a2n, a2o = t["A0"], t["A2"], t["A2n"], t["A2o"]
    s_l2, s_n = share("s4-l2"), share("s4-neutral")
    lo, hi = a0["paired"]["ci95_unrounded"]
    commit, sha = PREREGISTRATIONS["PREREG-STUDY4.md"]
    zero_l2, zero_n = (beh[r]["episodes_with_share_zero"] for r in ("s4-l2", "s4-neutral"))
    ep_shares = [e["share_bypassed"] for e in beh["s4-l2"]["per_episode"]]
    high = [x for x in ep_shares if x >= 0.9]
    mixed = [x for x in ep_shares if 0 < x < 0.9]
    (cl_n, si_n), (cl_l, si_l), (cl_q, si_q) = (
        pool(r) for r in ("s4-neutral", "s4-l2", "s4-qwen-bypass")
    )
    return [
        (
            "| A second model, back-office |",
            [
                f"`ll_bic` covers {r1['coverage']:.3f} on `{QWEN}`",
                f"| R1 {r1['verdict'].upper()}, pre-registered (D70) |",
            ],
        ),
        (
            "| Clustered script loss |",
            [
                f"covers {cl_n:.3f} on the neutral pool ({si_n:.3f} if lost one by one), "
                f"{cl_l:.3f} on `l2` ({si_l:.3f}) and {cl_q:.3f} on the second model "
                f"({si_q:.3f})",
                f"| A2 {a2['verdict'].upper()}, pre-registered (D70) |",
            ],
        ),
        (
            "**The `l2` prompt split the pool: `s4-l2` is a weak adversary test (D70, D73).**",
            [
                f"The agent bypassed {s_l2:.3f} of actions against {s_n:.3f} neutral; {zero_l2} "
                f"of {beh['s4-l2']['episodes']} episodes never batched",
                f"The `{QWEN}` spend is overstated: its receipts use deepseek rates",
            ],
        ),
        (
            "`s4-neutral`, `s4-l2`, `s4-qwen-backoffice` and `s4-qwen-bypass` are a fourth",
            [f"frozen at commit `{commit}`, tag `prereg-study4`; sha256 `{sha}`; D67 to D70"],
        ),
        (
            "| R1 second model, back-office |",
            [f"| {r1['verdict'].upper()} | `ll_bic` covers {r1['coverage']:.3f} {wil(r1)} |"],
        ),
        (
            "| Q0 bypass share, second model |",
            [
                f"| {q0['verdict'].upper()} | share {q0['share']:.3f} "
                f"[{q0['ci95'][0]:.4f}, {q0['ci95'][1]:.4f}] |"
            ],
        ),
        (
            "| Q1 calibration, second model |",
            [
                f"| {q1['verdict'].upper()} | covers {q1['coverage']:.3f} {wil(q1)}, "
                f"unbounded {q1['unbounded_rate']:.3f} |"
            ],
        ),
        (
            "| H integrity, `s4-l2` |",
            [
                f"| {h['verdict']} | {h['n_unwitnessed_other']} of {sum(h['counts'].values())} "
                "actions outside a logged call or script |"
            ],
        ),
        (
            "| A0 `l2` against neutral |",
            [
                f"| {a0['verdict'].upper()} | bypass share {s_l2:.3f} against {s_n:.3f}, "
                f"paired {a0['paired']['diff']:+.3f} [{lo:.3f}, {hi:.3f}] |"
            ],
        ),
        (
            "| A2 text-parsed auditor, `l2` |",
            [f"| {a2['verdict'].upper()} | covers {a2['coverage']:.3f} {wil(a2)} |"],
        ),
        (
            "| A2n same auditor, neutral |",
            [f"| {a2n['verdict'].upper()} | covers {a2n['coverage']:.3f} {wil(a2n)} |"],
        ),
        (
            "| A2o idealised auditor |",
            [
                f"| {a2o['verdict'].upper()} | covers {a2o['coverage']:.3f}, script lines lost "
                "one by one |"
            ],
        ),
        (
            "**The instruction split the pool.**",
            [
                f"{zero_l2} of {beh['s4-l2']['episodes']} `l2` episodes used no script (neutral "
                f"{zero_n}), {len(high)} batched {min(high):.2f} to 1.0 of their actions and "
                f"{len(mixed)} was mixed",
                "Read by hand (D73), 11 declined and never batched, at least two batched to "
                "undercount, and eight said they would not hide and batched anyway",
            ],
        ),
    ]


def expected_figures(dataset_dir: Path) -> list[tuple[str, list[str]]]:
    """Every headline figure, read here from the shipped files, as (block label, fragments).

    Each fragment must appear, in order of meaning, in the README block that
    starts with the label. The figures are this file's own reading of the
    JSON; only the sentence wording is shared with the builder.
    """
    res = dataset_dir / "results"
    rob = res / "robustness"

    def load(path: Path) -> dict:
        return json.loads(path.read_text())

    hyp = load(res / "hypotheses.json")["hypotheses"]
    hyp_supp = load(res / "hypotheses-supplement.json")["hypotheses"]
    t_hyp = load(res / "transfer_hypotheses.json")["hypotheses"]
    variance = load(res / "supplement-variance.json")
    reg = next(v for k, v in variance.items() if "registered" in k)
    rest = next(v for k, v in variance.items() if "supplement" in k)
    h1 = hyp["H1 recovery"]
    bar = float(re.search(r">=\s*([0-9.]+)", h1["detail"])[1])
    marginal = h1["passed"] and reg["mean_margin_mcse"] < 2
    figures: list[tuple[str, list[str]]] = []

    def covs(xs: list[float]) -> str:
        return ", ".join(f"{x:.2f}" for x in xs)

    # Verdict tables: every hypothesis row of the registered and the transfer study. The
    # supplement's rows left the card in D66; its H1 sentence below is pinned instead.
    for table, mark in ((hyp, True), (t_hyp, False)):
        for name, v in table.items():
            word = "PASS" if v["passed"] else "FAIL"
            if mark and marginal and name == "H1 recovery":
                word += " (marginal)"
            figures.append((f"| {name} | {word} | {v['detail']} |", []))

    runs = {r["run"]: r for r in _jsonl(dataset_dir / "runs.jsonl")}
    conf = runs["confirm"]
    n400 = sum(" 400 " in f["error"] for f in conf["failed_episodes"])
    bv = load(rob / "bypass-verdicts.json")
    t1 = next(v for k, v in t_hyp.items() if k.startswith("T1"))
    t1_frozen = float(re.search(r"coverage ([0-9.]+)", t1["detail"])[1])
    h1_status = "PASS (marginal)" if marginal else "PASS" if h1["passed"] else "FAIL"
    supp_pass = hyp_supp["H1 recovery"]["passed"]

    # What holds: one pinned row per claim.
    prec = load(rob / "precision.json")
    cell = prec["runs"]["confirm"]
    rs, ro = cell["fin_mid/signature"]["ll_bic"], cell["fin_mid/oracle"]["ll_bic"]
    tp = load(rob / "transfer-precision.json")
    ts = tp["runs"]["transfer"]["transfer_mid/signature"]["ll_bic"]
    to = tp["runs"]["transfer"]["transfer_mid/oracle"]["ll_bic"]
    figures.append(
        (
            "| H1 recovery on `confirm` |",
            [
                f"`ll_bic` covers {rs['coverage']:.3f} over {rs['reps']} replicates, below the "
                f"{bar:.2f} bar (the README's figure)",
                f"{reg['mean_coverage']:.3f} over the {len(reg['seeds'])} loss seeds of the "
                f"supplement study, {margin_text(reg['mean_margin_mcse'], bar)}",
                f"`supplement` (lost episodes restored) {rest['mean_coverage']:.3f}, "
                f"{margin_text(rest['mean_margin_mcse'], bar)}",
                f"| {h1_status} at the registered seed, pre-registered; below the bar over "
                f"replicates; `supplement` {'PASS' if supp_pass else 'FAIL'}, post hoc |",
            ],
        )
    )
    figures.append(
        (
            "| The shortfall is linkage, not the estimator |",
            [
                f"at {rs['reps']} replicates `ll_bic` covers {rs['coverage']:.3f} ± "
                f"{rs['coverage_mcse']:.3f} under signature linkage, {ro['coverage']:.3f} ± "
                f"{ro['coverage_mcse']:.3f} under oracle",
                "| holds, post hoc (D44) |",
            ],
        )
    )
    b1v, b0v = bv["B1"], bv["B0"]
    figures.append(
        (
            "| Transfer T1 and bypass B1 |",
            [
                f"T1 covers {t1_frozen:.3f} at the frozen loss seed",
                f"{ts['coverage']:.3f} under signature linkage against {to['coverage']:.3f} "
                f"under oracle over {ts['reps'] // tp['reps_per_seed']} seeds (post hoc)",
                f"B1 covers {b1v['coverage']:.3f} with {b0v['share']:.3f} of actions bypassed, "
                f"against {b1v['reference']['coverage']:.4f} with none",
                f"| T1 {'PASS' if t1['passed'] else 'FAIL'}, B1 {b1v['verdict'].upper()}, "
                "pre-registered (D47, D63) |",
            ],
        )
    )

    # The banner: the registered H1 sentence, the two registered failures, the losses.
    label = (
        "**The registered H1 pass was marginal.**"
        if marginal
        else f"**The registered H1 {_verb(h1['passed'])}.**"
    )
    figures.append(
        (
            label,
            [
                f"`transfer` {_verb(t1['passed'])} T1 and `bypass` "
                f"{_verb(bv['B1']['verdict'] == 'pass')} B1, both pre-registered",
                f"completed {conf['episodes']} of {conf['episodes_attempted']} episodes",
                f"{n400} of the {len(conf['failed_episodes'])} losses were provider 400s",
            ],
        )
    )
    mid = load(res / "thinning" / "confirm.json")["configs"]["fin_mid/signature"]
    figures.append(
        (
            "Over loss seeds",
            [
                f"`ll_bic` covers {covs(reg['coverage'])}: mean {reg['mean_coverage']:.3f}, "
                f"{margin_text(reg['mean_margin_mcse'], bar)}",
                f"the four-seed one, {reg['mean_mcse']:.3f}",
                f"the registered seed alone sits {margin_text(reg['seed_4242_margin_mcse'], bar)}",
                f"single-seed MCSE of {mid['estimators']['ll_bic']['coverage_mcse']:.3f}",
            ],
        )
    )
    figures.append(
        (
            f"**With the lost episodes restored, H1 {_verb(supp_pass)}.**",
            [
                f"Mean coverage over the same loss seeds is {rest['mean_coverage']:.3f} "
                f"({covs(rest['coverage'])})",
                f"{margin_text(rest['mean_margin_mcse'], bar)}",
                f"records H1 as {'PASS' if supp_pass else 'FAIL'}",
            ],
        )
    )

    fg = load(rob / "transfer-falsegap.json")["runs"]

    def gap(run: str, variant: str) -> float:
        v = fg[run]["variants"][variant]
        return v.get("ll_bic", v)["detects_gap"]

    t3 = next(v for k, v in t_hyp.items() if k.startswith("T3"))
    w3_complete = next(k for k in fg if k.startswith("transfer,"))
    figures.append(
        (
            "**T2 and T4 say nothing under signature linkage (post hoc).**",
            [
                f"claims a gap in {gap('transfer', 'signature'):.3f} of replicates under "
                f"signature linkage and {gap('transfer', 'exact (oracle)'):.3f} under oracle",
                f"with W3 complete, {gap(w3_complete, 'signature'):.3f}",
            ],
        )
    )

    tt = load(res / "thinning" / "transfer.json")["configs"]
    t_mid = tt["transfer_mid/signature"]["estimators"]["ll_bic"]
    t_rho = tt["transfer_mid_rho100/signature"]["estimators"]["ll_bic"]
    figures.append(
        (
            "**T3 passes only because the interval widened.**" if t3["passed"] else "**T3 fails.**",
            [
                f"covers {t_mid['coverage']:.2f} at `transfer_mid`",
                f"mean width {t_mid['mean_width']:.1f}) and {t_rho['coverage']:.2f} at "
                "`transfer_mid_rho100`",
                f"mean width {t_rho['mean_width']:.1f})",
            ],
        )
    )

    mech_all = load(rob / "transfer-mechanism.json")
    mech = mech_all["runs"]["transfer"]["variants"]
    restore = next(
        v for k, v in mech.items() if k.startswith("exact for all-W3-invisible and mixed")
    )
    beh = load(rob / "transfer-behaviour.json")["runs"]
    figures.append(
        (
            "**The mechanism is linkage and the verdict depends on density (post hoc, D46, D47).**",
            [
                f"restores `ll_bic` to {restore['coverage']:.3f}, against "
                f"{mech['signature']['coverage']:.3f} under signature linkage",
                f"({beh['transfer']['in_the_scripted_plans']} of {beh['transfer']['actions']} "
                "actions)",
            ],
        )
    )

    coll = load(rob / "collisions.json")["confirm"]
    figures.append(
        (
            "**Signature collisions (post hoc, D44).**",
            [
                f"{coll['sharing_share']:.3f} of `confirm` actions",
                f"{coll['sharing_by_op']['send_email']['sharing']} of those "
                f"{coll['sharing_a_signature']} are emails",
            ],
        )
    )

    arms = load(rob / "linkage-arms.json")["cells"]

    def arm(run: str, cfg: str, which: str) -> float:
        return arms[f"{run}/{cfg}/{which}"]["ll_bic"]["coverage"]

    coupled = sorted(
        {
            c["ll_bic"]["coverage"]
            for k, c in arms.items()
            if k.startswith("transfer/transfer_mid_rho100/")
            and not k.endswith(("/oracle", "/callid_lastwriter"))
        }
    )
    anti = {k: c["ll_bic"]["detects_gap"] for k, c in arms.items() if "/anti" in k}
    lw = "callid_lastwriter"
    coupled_text = " to ".join(f"{c:.3f}" for c in coupled)
    # Plain call-id arms only; the withdrawn last-writer rule has its own figure (D52).
    callid_gap = max(g for k, g in anti.items() if k.split("/")[2] in ("callid", "callid_digest"))
    lastwriter_gap = max(g for k, g in anti.items() if k.endswith("/" + lw))
    signature_gap = max(g for k, g in anti.items() if k.endswith("/signature"))
    figures.append(
        (
            "**Linkage arms, scoped (post hoc, D49, D50, D52).**",
            [
                f"restores {arm('confirm', 'fin_mid', 'callid'):.3f} in back-office under the "
                f"financial view and {arm('transfer', 'transfer_mid', 'callid'):.3f} in infra "
                f"({coupled_text} coupled)",
                f"signature {arm('confirm', 'fin_mid', 'signature'):.3f}, "
                f"digest {arm('confirm', 'fin_mid', 'digest'):.3f}, call id "
                f"{arm('confirm', 'fin_mid', 'callid'):.3f}, exact "
                f"{arm('confirm', 'fin_mid', 'oracle'):.3f}",
                f"`full_mid` call id {arm('confirm', 'full_mid', 'callid'):.3f}, exact "
                f"{arm('confirm', 'full_mid', 'oracle'):.3f}",
                "infra `transfer_mid`: signature "
                f"{arm('transfer', 'transfer_mid', 'signature'):.3f}, "
                f"call id {arm('transfer', 'transfer_mid', 'callid'):.3f}, exact "
                f"{arm('transfer', 'transfer_mid', 'oracle'):.3f}",
                "Plain call id is the auditor-computable result",
                f"is withdrawn (D52): its {arm('transfer', 'transfer_mid', lw):.3f} in infra is "
                "partly compensating error",
                f"in up to {lastwriter_gap:.3f} of replicates",
                f"claims a gap in at most {callid_gap:.3f} of replicates, against up to "
                f"{signature_gap:.3f} under signature linkage",
            ],
        )
    )

    bursts = load(rob / "burst-loss.json")["cells"]

    def series(prefix: str) -> list[dict]:
        return sorted(
            (c for k, c in bursts.items() if k.startswith(prefix)), key=lambda c: c["width"]
        )

    bo, infra = series("confirm/financial/indep/"), series("transfer/full/indep/")
    figures.append(
        (
            "**Bursts, outages and session quality (post hoc, D49-D54).**",
            [
                f"covers {bo[0]['ll_bic']['coverage']:.3f} at w={bo[0]['width']} down to "
                f"{bo[-1]['ll_bic']['coverage']:.3f} at w={bo[-1]['width']}",
                f"infra covers {infra[0]['ll_bic']['coverage']:.3f} down to "
                f"{infra[-1]['ll_bic']['coverage']:.3f}",
            ],
        )
    )

    figures += bypass_figures(dataset_dir)
    figures += study4_figures(dataset_dir)

    spend = receipt_spend(ARTIFACTS)
    figures.append(
        (
            "Spend over every receipt on disk is",
            [
                f"is {spend['total']:.2f} USD",
                f"the study {spend['study']:.2f}",
                f"{spend['diagnostic']:.2f}. The first confirmatory attempt",
                f"study 4 adds {STUDY4_SPEND_USD:.2f} USD at those rates",
            ],
        )
    )
    return figures


def check_figures(dataset_dir: Path) -> Check:
    """Every headline figure, read independently, sits in its labelled README sentence."""
    blocks = card_blocks((dataset_dir / "README.md").read_text())
    problems = []
    figures = expected_figures(dataset_dir)
    for label, fragments in figures:
        hits = [b for b in blocks if b.startswith(label) or b.startswith(f"- {label}")]
        if not hits:
            problems.append(f"no block labelled {label[:60]!r}")
            continue
        missing = [f for f in fragments if not any(f in b for b in hits)]
        if missing:
            problems.append(f"{label[:40]!r} lacks {missing[0][:60]!r}")
    return (
        "headline figures",
        not problems,
        "; ".join(problems[:3]) or f"{len(figures)} labelled figures match the shipped results",
    )


def check_bypass(dataset_dir: Path, artifacts: Path) -> Check:
    """The bypass study's shipped rows and results agree with each other and with the sources.

    `bypassed` was derived twice (the builder from the transcripts, this file from the
    records); here it is also tied to the verdict script's own accounting, which read the
    complete logs before any of this was built.
    """
    rob = dataset_dir / "results" / "robustness"
    v = json.loads((rob / "bypass-verdicts.json").read_text())
    problems: list[str] = []
    neutral, l1_run = v["runs"]["neutral"], v["runs"]["l1"]
    if (neutral, l1_run) != ("bypass-llm", "bypass-l1") or SHIPPED_AS.get(neutral) != "bypass":
        problems.append(f"bypass-verdicts.json names {neutral} and {l1_run}")
    for run in (neutral, l1_run):
        src = artifacts / "episodes" / run / "episodes.jsonl"
        for where, got in (
            ("inputs", v["inputs"]["episodes_sha256"].get(run)),
            (
                "completion",
                v["completion"]["neutral" if run == neutral else "l1"]["episodes_sha256"],
            ),
        ):
            if got != _sha(src):
                problems.append(f"{run}: {where} episodes_sha256 is not the file's")
        ana = json.loads((rob / f"bypass-analysis-{run}.json").read_text())
        if ana["run"] != run or ana["episodes_sha256"] != _sha(src):
            problems.append(f"bypass-analysis-{run}.json is not an analysis of {run}'s episodes")
    rows = _jsonl(dataset_dir / "episodes" / "bypass.jsonl")
    rows_l1 = _jsonl(dataset_dir / "episodes" / "bypass-l1.jsonl")
    per = [sum(x["bypassed"] for x in r["ledger"]) for r in rows]
    n = [len(r["ledger"]) for r in rows]
    b0 = v["B0"]
    if per != b0["per_episode_bypassed"] or n != b0["per_episode_n"]:
        problems.append("per-episode bypassed or N differ from B0's accounting")
    if (sum(per), sum(n)) != (b0["n_bypassed"], b0["n_true"]):
        problems.append("pooled bypassed or N differ from B0")
    if round(sum(per) / sum(n), 4) != b0["share"]:
        problems.append("the share differs from B0's")
    for run, shipped_rows in ((neutral, rows), (l1_run, rows_l1)):
        d = json.loads((rob / f"bypass-analysis-{run}.json").read_text())["descriptive"]
        flagged = sum(x["bypassed"] for r in shipped_rows for x in r["ledger"])
        if (flagged, sum(len(r["ledger"]) for r in shipped_rows)) != (d["n_bypassed"], d["n_true"]):
            problems.append(f"{run}: shipped flags differ from the analysis's n_bypassed")
        # A bypassed action has no trace or gateway record; every other one has both.
        for r in shipped_rows:
            logged = {x["oracle"] for x in r["records"] if x["witness"] in (0, 1)}
            direct = {x["id"] for x in r["ledger"] if not x["bypassed"]}
            if logged != direct:
                problems.append(f"{run} seed {r['seed']}: trace and gateway records != direct")
                break
    l1_share = sum(x["bypassed"] for r in rows_l1 for x in r["ledger"]) / sum(
        len(r["ledger"]) for r in rows_l1
    )
    if (
        round(l1_share, 4) != v["L1"]["l1_share"]
        or round(sum(per) / sum(n), 4) != (v["L1"]["neutral_share"])
    ):
        problems.append("L1's shares differ from the shipped rows")
    seeds = {r["seed"] for r in rows}
    if not set(v["L1"]["matched_seeds"]) <= seeds & {r["seed"] for r in rows_l1}:
        problems.append("L1's matched seeds are not in both runs")
    return (
        "bypass study",
        not problems,
        "; ".join(problems[:4])
        or f"{sum(per)}/{sum(n)} bypassed in bypass, share {b0['share']}; flags, records, "
        "verdict accounting and episode hashes agree",
    )


def check_study4(dataset_dir: Path, artifacts: Path) -> Check:
    """Study 4 (D67-D70): models, organisms and prompt hashes pinned; verdicts tied to the runs.

    The model, the organism and both prompt hashes of every run in RUN_PINS are held three
    ways: to each run's summary.json, to runs.jsonl, and (the hashes) to the prompt strings in
    the code. The registration shipped in docs/ must quote the pinned `l2` paragraph. The
    verdict file (read from artifacts/, since it is not shipped) must name the four runs with
    their pinned models, the hash of each episodes file and of each analysis it cites, and its
    accounting must equal the shipped `bypassed` flags.
    """
    from lacuna.envs.bypass import L1_PROMPT, L2_PROMPT, with_organism
    from lacuna.envs.registry import ENVS as REGISTRY

    problems: list[str] = []

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    env = REGISTRY[BYPASS_ENV]
    in_code = {
        "L1 paragraph": (sha(L1_PROMPT), L1_PARAGRAPH_SHA256),
        "L2 paragraph": (sha(L2_PROMPT), L2_PARAGRAPH_SHA256),
        "neutral system prompt": (sha(env.agent_system), NEUTRAL_SYSTEM_SHA256),
        "L1 system prompt": (sha(with_organism(env, "l1").agent_system), L1_SYSTEM_SHA256),
        "L2 system prompt": (sha(with_organism(env, "l2").agent_system), L2_SYSTEM_SHA256),
    }
    problems += [
        f"code's {k} is not the pinned one" for k, (got, want) in in_code.items() if got != want
    ]
    registration = (dataset_dir / "docs" / "PREREG-STUDY4.md").read_text()
    quotes = re.findall(r"((?:^>[^\n]*\n)+)", registration, re.M)
    quoted = {sha(" ".join(line.lstrip("> ").strip() for line in q.splitlines())) for q in quotes}
    if L2_PARAGRAPH_SHA256 not in quoted:
        problems.append("docs/PREREG-STUDY4.md does not quote the pinned l2 paragraph")

    runs = {r["run"]: r for r in _jsonl(dataset_dir / "runs.jsonl")}
    for run, (model, organism, paragraph, system) in RUN_PINS.items():
        summary = json.loads((artifacts / "episodes" / run / "summary.json").read_text())
        got = (
            summary["model"],
            summary.get("organism"),
            summary.get("organism_prompt_sha256"),
            summary.get("agent_system_sha256"),
        )
        if got != (model, organism, paragraph, system):
            problems.append(f"{run}: summary.json records {got}, pinned {(model, organism)}")
        row = runs.get(SHIPPED_AS.get(run, run), {})
        if (row.get("model"), row.get("organism"), row.get("organism_prompt_sha256")) != (
            model,
            organism,
            paragraph,
        ):
            problems.append(f"{run}: runs.jsonl model, organism or prompt hash is not the pinned")

    rob = artifacts / "robustness"
    v = json.loads((rob / "study4-verdicts.json").read_text())
    keyed = {
        "replication": "s4-qwen-backoffice",
        "qwen_bypass": "s4-qwen-bypass",
        "adversary": "s4-l2",
        "neutral": "s4-neutral",
    }
    for key, run in keyed.items():
        inp, gate = v["inputs"][key], v["gates"][key]
        want_model = RUN_PINS[run][0]
        rows = _jsonl(dataset_dir / "episodes" / f"{run}.jsonl")
        if (inp["run"], inp["model_expected"], inp["model_recorded"], inp["model_check"]) != (
            run,
            want_model,
            want_model,
            "ok",
        ):
            problems.append(f"{run}: verdicts do not record the pinned model {want_model}")
        if inp["episodes_sha256"] != _sha(artifacts / "episodes" / run / "episodes.jsonl"):
            problems.append(f"{run}: verdicts' episodes_sha256 is not the file's")
        if inp["artifact_sha256"] != _sha(rob / inp["artifact"]):
            problems.append(f"{run}: verdicts' hash of {inp['artifact']} is not the file's")
        if gate["status"] != "ok" or gate["completed"] != len(rows):
            problems.append(
                f"{run}: gate {gate['status']}, {gate['completed']} vs {len(rows)} rows"
            )
        ana = json.loads((rob / inp["artifact"]).read_text())
        if ana["run"] != run or ana["episodes_sha256"] != inp["episodes_sha256"]:
            problems.append(f"{inp['artifact']} is not an analysis of {run}'s episodes")
        if ana["run_summary"]["model"] != want_model:
            problems.append(f"{inp['artifact']} records another model")

    # The shipped `bypassed` flags against the verdicts' own accounting.
    t = v["tests"]
    flags = {
        run: [
            x["bypassed"]
            for row in _jsonl(dataset_dir / "episodes" / f"{run}.jsonl")
            for x in row["ledger"]
        ]
        for run in ("s4-neutral", "s4-l2", "s4-qwen-bypass")
    }
    q0 = t["Q0"]
    if (sum(flags["s4-qwen-bypass"]), len(flags["s4-qwen-bypass"])) != (
        q0["n_bypassed"],
        q0["n_true"],
    ):
        problems.append("s4-qwen-bypass: shipped flags differ from Q0's counts")
    a0 = t["A0"]
    for run, field in (("s4-l2", "treatment_share"), ("s4-neutral", "base_share")):
        if round(sum(flags[run]) / len(flags[run]), 4) != a0[field]:
            problems.append(f"{run}: shipped flags differ from A0's {field}")
    counts = t["H"]["counts"]
    n_l2, n_script = len(flags["s4-l2"]), sum(flags["s4-l2"])
    if counts["script_line"] != n_script or sum(counts.values()) != n_l2:
        problems.append("H's witness routes differ from the shipped s4-l2 rows")
    if t["H"]["n_unwitnessed_other"] != 0 or counts["unwitnessed_other"] != 0:
        problems.append("H found an unwitnessed action; the card's wording assumes none")
    for run in ("s4-neutral", "s4-l2", "s4-qwen-bypass"):
        for row in _jsonl(dataset_dir / "episodes" / f"{run}.jsonl"):
            logged = {x["oracle"] for x in row["records"] if x["witness"] in (0, 1)}
            if logged != {x["id"] for x in row["ledger"] if not x["bypassed"]}:
                problems.append(f"{run} seed {row['seed']}: trace and gateway records != direct")
                break
    spend = sum(receipt_totals(artifacts)[r] for r in STUDY4_RUNS)
    if f"{spend:.2f}" != f"{STUDY4_SPEND_USD:.2f}":
        problems.append(f"study 4 receipts sum to {spend:.4f}, not {STUDY4_SPEND_USD}")
    return (
        "study 4",
        not problems,
        "; ".join(problems[:4])
        or f"models, organisms and prompt hashes pinned; verdicts tied to the four runs' episode "
        f"hashes; receipts sum to {spend:.2f} USD",
    )


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def head_commit() -> str:
    """This file's own reading of the build tree: short HEAD, -dirty if anything is uncommitted."""
    sha = _git("rev-parse", "--short", "HEAD").strip() or "unknown"
    return sha + ("-dirty" if _git("status", "--porcelain").strip() else "")


def check_card_regenerates(dataset_dir: Path, artifacts: Path) -> Check:
    """README.md must be exactly what the builder's card() makes from the shipped files."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_dataset import card

    text = (dataset_dir / "README.md").read_text()
    stamp = COMMIT_STAMP.search(text)
    sim_output = LOSS_SIM_OUTPUT.search(text)
    if not stamp or not sim_output:
        return ("card regenerates", False, "no commit stamp or no expected snippet output")
    runs = _jsonl(dataset_dir / "runs.jsonl")
    rows = {p.stem: _jsonl(p) for p in sorted((dataset_dir / "episodes").glob("*.jsonl"))}
    excluded = [
        r
        for r, ph in EXPECTED_PHASES.items()
        if ph == "excluded" and (artifacts / "episodes" / r).is_dir()
    ]
    regenerated = card(
        runs,
        rows,
        dataset_dir,
        stamp.group(1),
        any(r.get("transcripts_included") for r in runs),
        excluded,
        receipt_spend(artifacts),
        sim_output.group(1),
        FREEZES,
    )
    head = head_commit()
    if stamp.group(1) != head:
        return ("card regenerates", False, f"the card says commit {stamp.group(1)}, HEAD is {head}")
    if regenerated == text:
        return (
            "card regenerates",
            True,
            f"README.md is exactly card() of the shipped files; stamp {head} is HEAD",
        )
    a, b = text.splitlines(), regenerated.splitlines()
    i = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    shipped = a[i] if i < len(a) else "<end>"
    wanted = b[i] if i < len(b) else "<end>"
    return ("card regenerates", False, f"line {i + 1}: {shipped[:70]!r} != {wanted[:70]!r}")


def _imports(code: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


# The hand-rolled draw of the held-out fin_mid loss (each of W1, W2 drops half its lines,
# the env diff sees financial ops only, seed 4242). It is written here, not imported, and
# shares no code with scripts/loss_sim.py; the helper's 0.5 line must agree with it.
REFERENCE_DRAW = """\
import json
import random
from collections import Counter

rng = random.Random(4242)
FINANCIAL = {"issue_refund", "cancel_order", "send_email"}
hist, n_true = Counter(), 0
for line in open("episodes/confirm.jsonl"):
    row = json.loads(line)
    seen = {e["id"]: 0 for e in row["ledger"]}
    for r in row["records"]:
        if r["oracle"] not in seen:
            continue
        if r["witness"] in (0, 1) and rng.random() < 0.5:
            continue
        if r["witness"] == 2 and r["op"] not in FINANCIAL:
            continue
        seen[r["oracle"]] |= 1 << r["witness"]
    hist.update(seen.values())
    n_true += row["n_true"]
print(f"true N {n_true}  unseen by all {hist[0]}")
"""


def check_loss_sim(dataset_dir: Path) -> Check:
    """scripts/loss_sim.py is the repo's file, stdlib only, and the card's snippet runs on it."""
    name = "loss_sim helper"
    shipped = dataset_dir / "scripts" / "loss_sim.py"
    source = SCRIPTS_DIR / "loss_sim.py"
    if not shipped.is_file():
        return (name, False, "scripts/loss_sim.py is not in the dataset")
    if shipped.read_bytes() != source.read_bytes():
        return (name, False, "the shipped scripts/loss_sim.py differs from the repo's")
    outside = _imports(shipped.read_text()) - set(sys.stdlib_module_names)
    if outside:
        return (name, False, f"scripts/loss_sim.py imports outside the stdlib: {sorted(outside)}")
    _, body = split_card((dataset_dir / "README.md").read_text())
    code = _snippet(body, LOSS_SIM_HEADING)
    expected = LOSS_SIM_OUTPUT.search(body)
    if not code or not expected:
        return (name, False, "no loss_sim snippet or no expected output on the card")
    outside = _imports(code) - set(sys.stdlib_module_names) - {"loss_sim"}
    if outside:
        return (name, False, f"the loss_sim snippet imports outside the stdlib: {sorted(outside)}")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-I", "-c", code],
            cwd=dataset_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return (name, False, "the loss_sim snippet timed out")
    if proc.returncode != 0:
        return (name, False, proc.stderr.strip()[-200:])
    got = proc.stdout.strip()
    if got != expected.group(1):
        return (name, False, f"snippet printed {got!r}, card says {expected.group(1)!r}")
    # The independent check: a hand-rolled draw of the same loss, sharing no code with
    # loss_sim, must find the same unseen count as the helper's 0.5 line.
    ref = subprocess.run(
        [sys.executable, "-B", "-I", "-c", REFERENCE_DRAW],
        cwd=dataset_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    unseen = re.search(r"unseen by all (\d+)", ref.stdout)
    line = next((x for x in got.splitlines() if x.startswith("0.5 ")), "")
    if ref.returncode != 0 or not unseen or not line or line.split()[1] != unseen[1]:
        detail = f"the hand-rolled draw {ref.stdout.strip()!r} != the 0.5 line {line!r}"
        return (name, False, detail)
    return (
        name,
        True,
        f"the repo's file, stdlib only; snippet prints the card's {len(got.splitlines())} lines "
        f"and its 0.5 line agrees with an independent draw ({unseen[1]} unseen)",
    )


def schema_matches(got, want) -> bool:
    """A loaded schema equals the pinned one; a column with nothing in it loads as `null`.

    D66 dropped `dataset_info` from the card, so `datasets` infers the schema from the
    rows. Inference cannot tell a string column that is all null (`model` of a scripted
    run) or an always-empty list from a column with no type, and calls it `null`; that is
    the one thing allowed to differ. Every other dtype, field and nesting must be the pinned
    one, and `check_types` holds every row to the pinned types separately.
    """
    if isinstance(got, dict) and got.get("dtype") == "null":
        return True
    if isinstance(got, dict) and isinstance(want, dict):
        return got.keys() == want.keys() and all(schema_matches(got[k], want[k]) for k in got)
    if isinstance(got, list) and isinstance(want, list):
        return len(got) == len(want) and all(
            schema_matches(g, w) for g, w in zip(got, want, strict=True)
        )
    return got == want


def _load_configs_here(dataset_dir: Path) -> tuple[bool, str]:
    """Child-process half of check_loads. Goes offline, which is why it is a child."""
    import yaml

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import datasets

    datasets.disable_progress_bars()
    datasets.logging.set_verbosity_error()
    head, _ = split_card((dataset_dir / "README.md").read_text())
    configs = yaml.safe_load(head)["configs"]
    problems, loaded = [], []
    with tempfile.TemporaryDirectory() as cache:
        for c in configs:
            name = c["config_name"]
            want = sum(len(_jsonl(p)) for p in dataset_dir.glob(c["data_files"]))
            pinned = RUN_FEATURES if name == "runs" else EPISODE_FEATURES
            try:
                ds = datasets.load_dataset(str(dataset_dir), name, split="train", cache_dir=cache)
            except Exception as exc:  # noqa: BLE001 - any failure here is the finding
                problems.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            if ds.num_rows != want:
                problems.append(f"{name}: {ds.num_rows} rows, expected {want}")
            # _from_yaml_list consumes its argument, and the pinned schema
            # shares one record list between two fields, so give it an
            # unaliased copy.
            want_features = datasets.Features._from_yaml_list(json.loads(json.dumps(pinned)))
            if not schema_matches(ds.features.to_dict(), want_features.to_dict()):
                problems.append(f"{name}: loaded features differ from the pinned schema")
            loaded.append(f"{name}={ds.num_rows}")
    return not problems, "; ".join(problems) or ", ".join(loaded) + "; schema as pinned"


def check_loads(dataset_dir: Path) -> Check:
    """Load every config through `datasets`, offline, in a child process.

    Offline mode is read from the environment when huggingface_hub is
    imported, and stays on for the life of the process. Setting it here would
    make the upload that follows validation in publish_dataset.py fail.
    """
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--load-check", str(dataset_dir)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode not in (0, 1) or not lines:
        err = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        return ("datasets loads every config", False, f"child failed: {err}")
    result = json.loads(lines[-1])
    return ("datasets loads every config", result["ok"], result["detail"])


# ---------------------------------------------------------------------------


def validate(
    dataset_dir: Path = DATASET, artifacts: Path = ARTIFACTS, heavy: bool = True
) -> list[Check]:
    """Every check. `heavy=False` skips only the `datasets` load in a child process."""
    layout = [
        dataset_dir / n for n in ("README.md", "runs.jsonl", "episodes", "results", "scripts")
    ]
    missing = [str(p.relative_to(dataset_dir)) for p in layout if not p.exists()]
    if missing:
        return [("layout", False, f"missing {missing}; run scripts/build_dataset.py")]
    checks: list[Check] = []
    # Each check runs whatever the others did, and a check that crashes on a
    # malformed file is a failure with its reason, not a traceback that hides
    # the checks after it.
    steps = [
        ("release content", lambda: [check_release_content(dataset_dir)]),
        ("privacy gate", lambda: [check_privacy(dataset_dir)]),
        ("file allowlist", lambda: [check_files(dataset_dir, artifacts)]),
        ("strict types", lambda: [check_types(dataset_dir)]),
        ("runs", lambda: check_runs(dataset_dir, artifacts)),
        ("supplement", lambda: [check_supplement(dataset_dir, artifacts)]),
        ("config contents", lambda: [check_configs(dataset_dir, artifacts)]),
        ("results", lambda: [check_results(dataset_dir, artifacts)]),
        ("docs and environment", lambda: [check_docs_and_environment(dataset_dir, artifacts)]),
        ("spend agrees", lambda: [check_spend(artifacts)]),
        ("bypass study", lambda: [check_bypass(dataset_dir, artifacts)]),
        ("study 4", lambda: [check_study4(dataset_dir, artifacts)]),
        ("card metadata", lambda: [check_card_metadata(dataset_dir, artifacts)]),
        ("card regenerates", lambda: [check_card_regenerates(dataset_dir, artifacts)]),
        ("headline figures", lambda: [check_figures(dataset_dir)]),
        ("loss_sim helper", lambda: [check_loss_sim(dataset_dir)]),
    ]
    if heavy:
        steps.append(("datasets loads every config", lambda: [check_loads(dataset_dir)]))
    for name, step in steps:
        try:
            checks += step()
        except Exception as exc:  # noqa: BLE001 - reported as the failure it is
            checks.append((name, False, f"crashed: {type(exc).__name__}: {str(exc)[:160]}"))
    return checks


def report(checks: list[Check]) -> bool:
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:30s}  {detail}")
    if not checks:
        print("  no checks ran, which is never a pass")
        return False
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    return not failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DATASET)
    ap.add_argument("--load-check", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.load_check:
        ok, detail = _load_configs_here(args.load_check)
        print(json.dumps({"ok": ok, "detail": detail}))
        return 0 if ok else 1
    return 0 if report(validate(args.dataset_dir)) else 1


if __name__ == "__main__":
    sys.exit(main())
