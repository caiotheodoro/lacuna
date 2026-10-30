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
and must match README.md exactly; its snippet is run and must print the line
the card says it prints. `datasets` loads every config in a child process (so
this process never goes offline) and must produce the pinned features. The
privacy gate scans every file. No network.
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
    "bypass-scripted-10": "excluded",
    "bypass-scripted-30": "excluded",
}
# Each run's environment; every other run is backoffice.
EXPECTED_ENV = {
    "transfer": "infra",
    "transfer-scripted": "infra",
    "transfer-scripted-24": "infra",
    "bypass-scripted-10": "backoffice_bypass",
    "bypass-scripted-30": "backoffice_bypass",
}
# Diagnostic reruns whose spend the card reports separately from the study.
DIAGNOSTIC_RUNS = ("diag-7009", "diag-batch")
# The confirmatory run is the one hypotheses.json names; the pool and its
# composition come from confirm-plus-supp.
POOL = "confirm-plus-supp"
DEFAULT_CONFIG = "confirm"
# Each pre-registered study: its verdict file, which names its run.
CONFIRMATORY = {
    "confirm": ("confirmatory", "hypotheses.json"),
    "transfer": ("confirmatory (transfer, separately pre-registered)", "transfer_hypotheses.json"),
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
}
NOT_SHIPPED = {
    "derived.json",
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
SHIPPED_DOCS = ("DECISIONS.md", "PRE-REGISTRATION.md", "PREREG-TRANSFER.md", "TRANSFER.md")
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
}
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
    "## What the data is",
    "## Load it",
    "## The three witnesses, and why the LLM auditor is not one",
    "## Field reference",
    "## Glossary",
    "## Rebuild a capture-history table and rerun an estimator",
    "## Exploratory vs confirmatory",
    "## Transfer study",
    "## Known caveats",
    "## Synthetic data",
    "## Citation",
)
BANNER = "> **Read before any result below"
HARD_WITNESSES = {0, 1, 2}
AUDITOR = 3


def episode_configs(artifacts: Path) -> list[str]:
    """Shipped episode configs, in card order: the default first, then the supplement."""
    on_disk = {p.name for p in (artifacts / "episodes").iterdir() if p.is_dir()}
    runs = [r for r, ph in EXPECTED_PHASES.items() if ph != "excluded" and r in on_disk]
    rest = [r for r in runs if r != DEFAULT_CONFIG]
    return [DEFAULT_CONFIG, SUPPLEMENT, *rest]


def allowed_files(artifacts: Path) -> set[str]:
    files = {".gitattributes", "README.md", "runs.jsonl"}
    files |= {f"episodes/{c}.jsonl" for c in episode_configs(artifacts)}
    files |= {f"results/{n}" for n in SHIPPED_RESULTS}
    files |= {f"docs/{n}" for n in SHIPPED_DOCS}
    for env in shipped_envs(artifacts):
        files |= {f"environment/{env}/system_prompt.txt", f"environment/{env}/tools.json"}
    return files


def shipped_envs(artifacts: Path) -> list[str]:
    runs = [c for c in episode_configs(artifacts) if c != SUPPLEMENT]
    return sorted({EXPECTED_ENV.get(r, "backoffice") for r in runs})


# Schemas, written here independently of the builder's dataset_info.
def _f(name: str, spec: dict) -> dict:
    return {"name": name, **spec}


_S, _I, _F, _B = ({"dtype": t} for t in ("string", "int64", "float64", "bool"))
_RECORD = [_f("witness", _I), _f("op", _S), _f("resource", _S), _f("value", _S), _f("oracle", _I)]
_LEDGER = [_f("id", _I), _f("op", _S), _f("resource", _S), _f("value", _S), _f("step", _I)]
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
    "owner name": re.compile(r"(?i)caio|theodoro|matheus"),
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


def row_problems(row: dict, src: dict, fields: dict) -> list[str]:
    """One built row against its source episode and the run-level fields it must carry."""
    problems: list[str] = []
    tag = f"seed {row.get('seed')}"
    expected = {"seed": src["seed"], **fields}
    expected.update(
        {
            "n_true": len(src["ledger"]),
            "ledger": src["ledger"],
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
    for run, (phase, verdicts) in CONFIRMATORY.items():
        named = json.loads((artifacts / verdicts).read_text())["run"]
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
        src_dir = artifacts / "episodes" / run
        summary = json.loads((src_dir / "summary.json").read_text())
        receipt_path = src_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        source = _jsonl(src_dir / "episodes.jsonl")
        rows = _jsonl(path)
        meta = runs[run]
        problems: list[str] = []

        phase = EXPECTED_PHASES[run]
        if (summary.get("agent") == "scripted") != (phase == "scripted harness check"):
            problems.append(f"pinned phase {phase!r} disagrees with agent {summary.get('agent')!r}")
        env = EXPECTED_ENV.get(run, "backoffice")
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
        if run in confirmatory_runs:
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
        or "docs match; both pre-registrations at their pinned text and freeze; "
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
    want_info = [
        {
            "config_name": c["config_name"],
            "features": RUN_FEATURES if c["config_name"] == "runs" else EPISODE_FEATURES,
        }
        for c in want_configs
    ]
    n_rows = sum(len(_jsonl(dataset_dir / "episodes" / f"{c}.jsonl")) for c in configs)
    for key, want in (
        ("license", "apache-2.0"),
        ("pretty_name", "lacuna capture histories"),
        ("language", ["en"]),
        ("tags", REQUIRED_TAGS),
        ("size_categories", [size_category(n_rows)]),
        ("configs", want_configs),
        ("dataset_info", want_info),
    ):
        if meta.get(key) != want:
            problems.append(f"{key} differs from the pinned value")
    for section in REQUIRED_SECTIONS:
        if section not in body:
            problems.append(f"missing section {section!r}")
    if BANNER not in body or ("| PASS" in body and body.index(BANNER) > body.index("| PASS")):
        problems.append("the banner is missing or comes after a PASS row")
    decisions = (dataset_dir / "docs" / "DECISIONS.md").read_text()
    defined = set(re.findall(r"^\*\*D(\d+)\.", decisions, re.M))
    cited = set(re.findall(r"\bD(\d+)\b", body))
    if cited - defined:
        problems.append(f"cites D{sorted(cited - defined)} not in docs/DECISIONS.md")
    return (
        "card metadata",
        not problems,
        "; ".join(problems) or f"{len(want_configs)} configs, pinned features, banner first",
    )


def receipt_spend(artifacts: Path) -> dict:
    """Spend from every receipt on disk, split study / diagnostic by the pinned list."""
    per_run = {
        p.parent.name: json.loads(p.read_text())["total_usd"]
        for p in sorted((artifacts / "episodes").glob("*/receipt.json"))
    }
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
        "study_unshipped": sorted(r for r in per_run if r in excluded and r not in DIAGNOSTIC_RUNS),
    }


def check_spend(artifacts: Path) -> Check:
    """Receipt spend must agree with derived.json's split, to the cent the card prints."""
    spend = receipt_spend(artifacts)
    derived = json.loads((artifacts / "derived.json").read_text())["spend"]
    pairs = {
        "total": (spend["total"], derived["total_usd"]),
        "study": (spend["study"], derived["study_usd"]),
        "diagnostic": (spend["diagnostic"], derived["diagnostic_usd"]),
    }
    off = {k: v for k, v in pairs.items() if f"{v[0]:.2f}" != f"{v[1]:.2f}"}
    return (
        "spend agrees",
        not off,
        f"receipts vs derived.json differ: {off}"
        if off
        else "receipts and derived.json: " + ", ".join(f"{k} {v[0]:.2f}" for k, v in pairs.items()),
    )


SNIPPET_OUTPUT = re.compile(r"Expected output[^\n]*\n\n```text\n(.*?)\n```", re.S)
COMMIT_STAMP = re.compile(r"from lacuna commit `([^`]+)`")


def _snippet(body: str) -> str | None:
    section = body.split("## Rebuild a capture-history table and rerun an estimator", 1)
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

    # Verdict tables: every hypothesis row, in each study's table.
    for table, mark in ((hyp, True), (hyp_supp, False), (t_hyp, False)):
        for name, v in table.items():
            word = "PASS" if v["passed"] else "FAIL"
            if mark and marginal and name == "H1 recovery":
                word += " (marginal)"
            figures.append((f"| {name} | {word} | {v['detail']} |", []))

    label = (
        "**The registered H1 pass was marginal.**"
        if marginal
        else f"**The registered H1 {_verb(h1['passed'])}.**"
    )
    figures.append((label, [f"covers {covs(reg['coverage'])}", f"mean {reg['mean_coverage']:.3f}"]))
    figures.append(
        (
            f"**With the lost episodes restored, H1 {_verb(hyp_supp['H1 recovery']['passed'])}.**",
            [
                f"mean coverage over the same loss seeds is {rest['mean_coverage']:.3f} "
                f"({covs(rest['coverage'])})",
                f"records H1 as {'PASS' if hyp_supp['H1 recovery']['passed'] else 'FAIL'}",
            ],
        )
    )

    prec = load(rob / "precision.json")
    cell = prec["runs"]["confirm"]
    rs, ro = cell["fin_mid/signature"]["ll_bic"], cell["fin_mid/oracle"]["ll_bic"]
    ps = prec["runs"]["confirm-plus-supp"]
    figures.append(
        (
            f"**At {rs['reps']} replicates the shortfall is linkage, not the estimator "
            "(post hoc, D44).**",
            [
                f"`ll_bic` covers {rs['coverage']:.3f} ± {rs['coverage_mcse']:.3f} under "
                f"signature linkage and {ro['coverage']:.3f} ± {ro['coverage_mcse']:.3f} "
                "under oracle",
                "on the restored pool, "
                f"{ps['fin_mid/signature']['ll_bic']['coverage']:.3f} against "
                f"{ps['fin_mid/oracle']['ll_bic']['coverage']:.3f}",
                f"in {cell['fin_mid/signature']['ll_bic_selected']['W1W2']['share']:.3f} "
                "of replicates "
                f"({cell['fin_mid/oracle']['ll_bic_selected']['W1W2']['share']:.3f} under oracle",
            ],
        )
    )

    tp = load(rob / "transfer-precision.json")
    ts = tp["runs"]["transfer"]["transfer_mid/signature"]["ll_bic"]
    to = tp["runs"]["transfer"]["transfer_mid/oracle"]["ll_bic"]
    t1 = next(v for k, v in t_hyp.items() if k.startswith("T1"))
    t1_frozen = float(re.search(r"coverage ([0-9.]+)", t1["detail"])[1])
    t3 = next(v for k, v in t_hyp.items() if k.startswith("T3"))
    figures.append(
        (
            f"**The transfer study {_verb(t1['passed'])} T1 (D47).**",
            [
                f"covers {t1_frozen:.3f} at the frozen loss seed",
                f"{ts['coverage']:.3f} under signature linkage against "
                f"{to['coverage']:.3f} under oracle",
            ],
        )
    )
    figures.append(
        (
            f"**T1 {_verb(t1['passed'])}.**",
            [
                f"covers {t1_frozen:.3f} at the frozen loss seed",
                f"over {ts['reps'] // tp['reps_per_seed']} loss seeds ({ts['reps']} replicates)",
                f"{ts['coverage']:.3f} ± {ts['coverage_mcse']:.3f} under signature linkage and "
                f"{to['coverage']:.3f} ± {to['coverage_mcse']:.3f} under oracle",
            ],
        )
    )

    fg = load(rob / "transfer-falsegap.json")["runs"]

    def gap(run: str, variant: str) -> float:
        v = fg[run]["variants"][variant]
        return v.get("ll_bic", v)["detects_gap"]

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
    mid = tt["transfer_mid/signature"]["estimators"]["ll_bic"]
    rho = tt["transfer_mid_rho100/signature"]["estimators"]["ll_bic"]
    figures.append(
        (
            "**T3 passes only because the interval widened.**" if t3["passed"] else "**T3 fails.**",
            [
                f"covers {mid['coverage']:.2f} at `transfer_mid`",
                f"mean width {mid['mean_width']:.1f}) and {rho['coverage']:.2f} at "
                "`transfer_mid_rho100`",
                f"mean width {rho['mean_width']:.1f})",
            ],
        )
    )

    mech_all = load(rob / "transfer-mechanism.json")
    mech = mech_all["runs"]["transfer"]["variants"]
    restore = next(
        v for k, v in mech.items() if k.startswith("exact for all-W3-invisible and mixed")
    )
    figures.append(
        (
            "**The mechanism is linkage (post hoc, D47).**",
            [
                f"Over {mech['signature']['reps'] // mech_all['reps_per_seed']} loss seeds",
                f"restores `ll_bic` to {restore['coverage']:.3f}, against "
                f"{mech['signature']['coverage']:.3f} under signature linkage and "
                f"{mech['exact (oracle)']['coverage']:.3f} with exact matching everywhere",
            ],
        )
    )

    dens = load(rob / "transfer-density.json")
    cells = list(dens["cells"].values())
    best = max(cells, key=lambda c: c["signature"]["coverage"])
    worst = min(cells, key=lambda c: c["signature"]["coverage"])
    figures.append(
        (
            "**The verdict depends on density (post hoc, D46).**",
            [
                f"from {best['signature']['coverage']:.3f} ({best['services']} services, "
                f"{best['tickets']} tickets) to {worst['signature']['coverage']:.3f} "
                f"({worst['services']} services, {worst['tickets']} tickets)",
                f"stays {min(c['oracle']['coverage'] for c in cells):.2f} to "
                f"{max(c['oracle']['coverage'] for c in cells):.2f}",
                f"episode seeds from {dens['episode_seed0']}",
            ],
        )
    )

    beh = load(rob / "transfer-behaviour.json")["runs"]
    figures.append(
        (
            "**The paid run reproduces the scripted plans.**",
            [
                f"{beh['transfer']['in_the_scripted_plans']} of {beh['transfer']['actions']} "
                "`transfer` actions",
                f"({beh['transfer-scripted-24']['in_the_scripted_plans']} of "
                f"{beh['transfer-scripted-24']['actions']} for `transfer-scripted-24`)",
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
                f"({coll['unthinned_ids_seen']['trace']}/{coll['n']} and "
                f"{coll['unthinned_ids_seen']['gateway']}/{coll['n']})",
            ],
        )
    )

    dose = load(rob / "mnar-dose.json")["op_keyed"]
    levels = sorted({k.split("/")[0] for k in dose}, key=float)

    def cov(level: str, linkage: str) -> float:
        return dose[f"{level}/{linkage}"]["ll_bic"]["coverage"]

    first = next(
        lv
        for lv in levels
        if float(lv) > 1 and cov(lv, "oracle") < bar and cov(lv, "signature") < bar
    )
    recover = next(
        lv
        for lv in levels
        if float(lv) > float(first) and cov(lv, "oracle") >= bar and cov(lv, "signature") >= bar
    )
    figures.append(
        (
            "**MNAR dose response (post hoc, D44).**",
            [
                f"covers {cov('1', 'oracle'):.3f} under oracle linkage and already "
                f"{cov('1', 'signature'):.3f} under signature",
                f"falls to {cov(first, 'oracle'):.3f} and {cov(first, 'signature'):.3f} "
                f"at {first}x",
                f"recovers at {recover}x ({cov(recover, 'oracle'):.3f} oracle, "
                f"{cov(recover, 'signature'):.3f} signature) only by widening",
            ],
        )
    )

    battery = load(rob / "falsegap-battery.json")["pools"]
    inside = [
        e["ll_bic"]["detects_gap"]
        for p in battery.values()
        for k, e in p["configs"].items()
        if k.startswith("anti") and k.endswith("_financial/signature")
    ]
    outside = [
        e["ll_bic"]["detects_gap"]
        for p in battery.values()
        for k, e in p["configs"].items()
        if not (k.startswith("anti") and k.endswith("_financial/signature"))
    ]
    figures.append(
        (
            "**False gaps (post hoc, D44).**",
            [
                f"claims a gap in {min(inside):.2f} to {max(inside):.2f} of replicates only when "
                "records are matched by signature",
                f"elsewhere in at most {max(outside):.2f}",
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
    figures.append(
        (
            "**The back-office shortfall is a linkage choice (post hoc, D49, D50, D52).**",
            [
                f"id restores {arm('confirm', 'fin_mid', 'callid'):.3f} in back-office under "
                "the financial view",
                f"it reaches {arm('transfer', 'transfer_mid', 'callid'):.3f} "
                f"({coupled_text} coupled)",
            ],
        )
    )
    # Plain call-id arms only; the withdrawn last-writer rule has its own figure (D52).
    callid_gap = max(g for k, g in anti.items() if k.split("/")[2] in ("callid", "callid_digest"))
    lastwriter_gap = max(g for k, g in anti.items() if k.endswith("/" + lw))
    signature_gap = max(g for k, g in anti.items() if k.endswith("/signature"))
    figures.append(
        (
            "**Linkage arms, scoped (post hoc, D49, D50, D52).**",
            [
                f"Back-office `fin_mid`: signature {arm('confirm', 'fin_mid', 'signature'):.3f}, "
                f"digest {arm('confirm', 'fin_mid', 'digest'):.3f}, call id "
                f"{arm('confirm', 'fin_mid', 'callid'):.3f}, exact "
                f"{arm('confirm', 'fin_mid', 'oracle'):.3f}",
                f"`full_mid` call id {arm('confirm', 'full_mid', 'callid'):.3f}, exact "
                f"{arm('confirm', 'full_mid', 'oracle'):.3f}",
                "Infra `transfer_mid`: signature "
                f"{arm('transfer', 'transfer_mid', 'signature'):.3f}, "
                f"call id {arm('transfer', 'transfer_mid', 'callid'):.3f}, exact "
                f"{arm('transfer', 'transfer_mid', 'oracle'):.3f}",
                f"signature, digest and call id {coupled_text}, exact "
                f"{arm('transfer', 'transfer_mid_rho100', 'oracle'):.3f}",
                "Plain call id is the auditor-computable result",
                f"is withdrawn (D52): its {arm('transfer', 'transfer_mid', lw):.3f} in infra is "
                "partly compensating error",
                f"in up to {lastwriter_gap:.3f} of replicates",
                f"claims a gap in at most {callid_gap:.3f} of replicates, against up to "
                f"{signature_gap:.3f} under signature linkage",
                "holds by construction when trace and gateway together see every action",
            ],
        )
    )
    bursts = load(rob / "burst-loss.json")["cells"]

    def series(prefix: str) -> list[dict]:
        return sorted(
            (c for k, c in bursts.items() if k.startswith(prefix)), key=lambda c: c["width"]
        )

    bo, infra = series("confirm/financial/indep/"), series("transfer/full/indep/")
    od = load(rob / "overdispersion.json")["cells"]
    w12 = [
        od[f"burst/{run}/{view}/indep/w12"]
        for run, view in (("confirm", "financial"), ("long-30", "financial"), ("transfer", "full"))
    ]

    def trio(est: str) -> str:
        return ", ".join(f"{c[est]['coverage']:.3f}" for c in w12)

    coupled_od = od["heldout/confirm/fin_mid_rho100/oracle"]
    joint = od["burst/confirm/financial/joint/w12"]
    infra_od = od["heldout/transfer/transfer_mid_rho100/oracle"]
    phi_infra, bic_qp_infra = infra_od["mean_phi"], infra_od["ll_bic_qp"]["coverage"]
    ep = load(rob / "episode-dispersion.json")["cells"]
    sig_qp = ep["burst/transfer/full/indep/w12/signature"]["ll_independence_qp"]["coverage"]
    figures.append(
        (
            "**Bursts are an interval problem (post hoc, D49, D50).**",
            [
                f"covers {bo[0]['ll_bic']['coverage']:.3f} at w={bo[0]['width']} down to "
                f"{bo[-1]['ll_bic']['coverage']:.3f} at w={bo[-1]['width']}",
                f"bias at most {max(abs(c['ll_bic']['bias']) for c in bo):.1f} actions",
                f"infra covers {infra[0]['ll_bic']['coverage']:.3f} down to "
                f"{infra[-1]['ll_bic']['coverage']:.3f}",
                "only `ll_independence_qp`, and only under exact matching",
                f"at w=12 it covers {trio('ll_independence_qp')}",
                f"against {trio('ll_independence')} plain",
                f"and {sig_qp:.3f} in infra under signature linkage",
                f"it reads {phi_infra:.2f} in coupled infra, where `ll_bic_qp` covers "
                f"{bic_qp_infra:.3f}, and {joint['mean_phi']:.2f} under joint bursts",
                f"its interval is about {coupled_od['ll_independence_qp']['mean_width']:.0f} wide",
                f"safe under coupling ({coupled_od['ll_bic_qp']['coverage']:.3f}) but fixes "
                f"bursts only partly ({trio('ll_bic_qp')})",
                f"`ll_bic_qp` stays at {joint['ll_bic_qp']['coverage']:.3f}",
            ],
        )
    )

    ic = load(rob / "interval-comparison.json")["cells"]

    def icov(key: str, est: str) -> str:
        return f"{ic[key][est]['coverage']:.3f}"

    def widen(key: str) -> int:
        hom, plain = ic[key]["ll_bic_hom"]["mean_width"], ic[key]["ll_bic"]["mean_width"]
        return round(100 * (hom / plain - 1))

    shared = ("session/shared/confirm/full", "session/shared/transfer/full")
    sessions = ("session/gateway/confirm", "session/both/confirm")
    per = ic["perrecord/confirm"]
    figures.append(
        (
            "**`ll_bic_hom` keeps BIC's model and widens only (post hoc, D51, D52).**",
            [
                "collapses when session quality varies and hits all three logs at once: "
                + ", ".join(icov(k, "ll_qbic_hom") for k in shared),
                "against `ll_bic` " + ", ".join(icov(k, "ll_bic") for k in shared),
                "so it never covers less than `ll_bic`",
                "it covers "
                + ", ".join(
                    icov(k, "ll_bic_hom")
                    for k in ("burst/confirm/w12", "burst/long-60/w12", "burst/transfer/w12")
                )
                + " under bursts at w=12",
                f"{icov('joint/confirm/w12', 'll_bic_hom')} under a joint outage",
                f"{per['ll_bic_hom']['coverage']:.3f} under per-record loss at the plain width "
                f"({per['ll_bic_hom']['mean_width']:.1f} against "
                f"{per['ll_bic']['mean_width']:.1f})",
                f"{icov('coupled/confirm/rho100', 'll_bic_hom')} under coupling",
                "Shared session quality defeats the pooled intervals",
                "it still fails there, at " + ", ".join(icov(k, "ll_bic_hom") for k in shared),
                "widen it by "
                + " to ".join(f"{w}%" for w in sorted(widen(k) for k in sessions))
                + " with coverage "
                + " to ".join(sorted(icov(k, "ll_bic_hom") for k in sessions)),
                f"pooling the two environments by {widen('perrecord/backoffice+infra/full')}%",
                "gap detection falls to "
                f"{ic['burst/confirm/full/w12']['ll_bic_hom']['detects_gap']:.3f}",
                "It needs visible episode boundaries",
            ],
        )
    )

    sp = load(rob / "session-and-phase.json")["cells"]
    position = load(rob / "op-position.json")

    def first_quarter(pool: str) -> str:
        return f"{position[pool]['w3_visible_share_by_quarter'][0]:.3f}"

    burst_hom = [
        c["ll_bic_hom"]["coverage"] for k, c in ic.items() if k.split("/")[0] in ("burst", "joint")
    ] + [c["ll_bic_hom"]["coverage"] for k, c in sp.items() if k.endswith("/actual")]
    fc = load(rob / "final-controls.json")["cells"]
    burst_hom += [
        c["ll_bic_hom"]["coverage"]
        for k, c in fc.items()
        if k.startswith("burst/") and k.endswith("/actual")
    ]
    outages = [c for k, c in sp.items() if k.startswith("outage/")]
    ratios = [c["ll_bic_hom"]["mean_width"] / c["ll_bic"]["mean_width"] for c in outages]

    def spc(key: str, est: str) -> str:
        return f"{sp[key][est]['coverage']:.3f}"

    figures.append(
        (
            "**Where the dispersion correction costs or fails (post hoc, D53).**",
            [
                f"`ll_bic_hom` covers {min(burst_hom):.2f} to {max(burst_hom):.2f}",
                f"it covers {spc('phase/long-60/actual', 'll_bic_hom')} (long-60) and "
                f"{spc('phase/long-120/actual', 'll_bic_hom')} (long-120) under bursts at w=12",
                f"The env diff sees {first_quarter('long-60')} of the first quarter of a long-60 "
                f"episode and {first_quarter('long-120')} of a long-120 one, against "
                f"{first_quarter('confirm')} in confirm",
                f"nominal: {spc('phase/long-60/actual', 'll_bic')} to "
                f"{spc('phase/long-60/shuffled', 'll_bic')} (long-60) and "
                f"{spc('phase/long-120/actual', 'll_bic')} to "
                f"{spc('phase/long-120/shuffled', 'll_bic')} (long-120)",
                f"already nominal ({min(c['ll_bic']['coverage'] for c in outages):.3f} to "
                f"{max(c['ll_bic']['coverage'] for c in outages):.3f})",
                f"`ll_bic_hom` is {min(ratios):.1f} to {max(ratios):.1f} times wider",
                "gap detection falls to "
                f"{sp['outage/long-60/indep']['ll_bic_hom']['detects_gap']:.2f} (long-60, "
                "independent outages)",
            ],
        )
    )
    shared_sp = ("shared/confirm", "shared/transfer")

    def pair(est: str) -> str:
        return " and ".join(spc(k, est) for k in shared_sp)

    figures.append(
        (
            "**Shared session quality is partly recoverable (post hoc, D53).**",
            [
                f"the pooled intervals fail: `ll_bic` {pair('ll_bic')}, `ll_bic_hom` "
                f"{pair('ll_bic_hom')}",
                f"recovers {pair('tier_bic')}, and `lc_gated` {pair('lc_gated')}",
                "Neither reaches 0.95",
            ],
        )
    )

    def fcov(key: str, est: str = "ll_bic") -> str:
        return f"{fc[key][est]['coverage']:.3f}"

    three = ("confirm", "long-60", "long-120")
    longs = ("long-60", "long-120")
    figures.append(
        (
            "**Bursts mostly land where the env diff is blind (post hoc, D54).**",
            [
                "plain `ll_bic` at w=12 covers "
                + ", ".join(fcov(f"burst/{p}/visperm") for p in three),
                "against " + ", ".join(fcov(f"burst/{p}/actual") for p in three) + " actual",
                "("
                + " and ".join(f"{fc[f'burst/{p}/actual']['ll_bic']['bias']:+.1f}" for p in longs)
                + " actions)",
                "("
                + " and ".join(fcov(f"burst/{p}/actual", "ll_bic_hom") for p in longs)
                + ", widths "
                + " and ".join(
                    f"{fc[f'burst/{p}/actual']['ll_bic_hom']['mean_width']:.1f}" for p in longs
                )
                + ")",
                "It is a width-for-coverage trade, to be reported beside the plain interval",
            ],
        )
    )
    down = [
        fc[f"outage/{env}/{kind}"]
        for env in ("confirm", "transfer")
        for kind in ("all_three_independent", "gateway_with_diff", "bad_session")
    ]
    worst = [
        fc[f"outage/{env}/{kind}"]
        for env in ("confirm", "transfer")
        for kind in ("cascade", "all_three_together")
    ]
    down_ratio = [c["ll_bic_hom"]["mean_width"] / c["ll_bic"]["mean_width"] for c in down]

    def span(cs: list[dict], est: str) -> str:
        values = [c[est]["coverage"] for c in cs]
        return f"{min(values):.3f} to {max(values):.3f}"

    figures.append(
        (
            "**Outages that take the env diff down too (post hoc, D54).**",
            [
                "nominal ("
                + " / ".join(fcov(f"outage/{e}/gateway_trace") for e in ("confirm", "transfer"))
                + ") only while the env diff stays up",
                f"`ll_bic` covers {span(down, 'll_bic')} and `ll_bic_hom` "
                f"{span(down, 'll_bic_hom')}, at {min(down_ratio):.0f} to {max(down_ratio):.0f} "
                "times the width",
                f"covers at most {max(c['ll_bic_hom']['coverage'] for c in worst):.3f} there",
            ],
        )
    )
    figures.append(
        (
            "**`tier_bic` is not a default (post hoc, D54).**",
            [
                f"It covers {fcov('tier/confirm/fin_mid', 'tier_bic')} under per-record loss, "
                f"{fcov('tier/confirm/fin_mid_rho100', 'tier_bic')} coupled, "
                f"{fcov('tier/confirm/burst', 'tier_bic')} and "
                f"{fcov('tier/transfer/burst', 'tier_bic')} under bursts",
                f"{fcov('tier/confirm/fin_mid_mnar', 'tier_bic')} under action-dependent loss",
                f"it covers {fcov('tiercount/confirm', 'tier_bic_4')} (confirm) with 4 tiers "
                f"against {fcov('tiercount/confirm', 'tier_bic')} at the default tier count",
            ],
        )
    )

    spend = receipt_spend(ARTIFACTS)
    figures.append(
        (
            "Spend over every receipt on disk is",
            [
                f"is {spend['total']:.2f} USD",
                f"the study {spend['study']:.2f}",
                f"{spend['diagnostic']:.2f}. The first confirmatory attempt",
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
    expected_output = SNIPPET_OUTPUT.search(text)
    if not stamp or not expected_output:
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
        expected_output.group(1),
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


def check_snippet(dataset_dir: Path) -> Check:
    """The card's snippet, run as written in an isolated interpreter, prints the card's line."""
    text = (dataset_dir / "README.md").read_text()
    _, body = split_card(text)
    code = _snippet(body)
    expected = SNIPPET_OUTPUT.search(body)
    if not code or not expected:
        return ("card snippet runs", False, "no snippet or no expected output on the card")
    imported = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    outside = imported - set(sys.stdlib_module_names)
    if outside:
        return ("card snippet runs", False, f"imports outside the stdlib: {sorted(outside)}")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=dataset_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return ("card snippet runs", False, "timed out")
    got = proc.stdout.strip()
    ok = proc.returncode == 0 and got == expected.group(1)
    detail = (
        f"prints the card's line: {got}"
        if ok
        else f"printed {got!r}, card says {expected.group(1)!r}"
    )
    return ("card snippet runs", ok, detail if proc.returncode == 0 else proc.stderr.strip()[-200:])


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
            if ds.features != datasets.Features._from_yaml_list(json.loads(json.dumps(pinned))):
                problems.append(f"{name}: loaded features differ from the pinned schema")
            loaded.append(f"{name}={ds.num_rows}")
    return not problems, "; ".join(problems) or ", ".join(loaded) + "; features as pinned"


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
    layout = [dataset_dir / n for n in ("README.md", "runs.jsonl", "episodes", "results")]
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
        ("card metadata", lambda: [check_card_metadata(dataset_dir, artifacts)]),
        ("card regenerates", lambda: [check_card_regenerates(dataset_dir, artifacts)]),
        ("headline figures", lambda: [check_figures(dataset_dir)]),
        ("card snippet runs", lambda: [check_snippet(dataset_dir)]),
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
