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
    "confirm-supp": "excluded",
    "diag-7009": "excluded",
    "diag-batch": "excluded",
    "confirm-plus-supp": "excluded",
}
# Diagnostic reruns whose spend the card reports separately from the study.
DIAGNOSTIC_RUNS = ("diag-7009", "diag-batch")
# The confirmatory run is the one hypotheses.json names; the pool and its
# composition come from confirm-plus-supp.
POOL = "confirm-plus-supp"
DEFAULT_CONFIG = "confirm"
# Decisions each run's caveat must cite.
CAVEAT_REFS = {
    "confirm": ("D20", "D32", "D34", "D36"),
    "scripted-l0": (),
    "pilot-l0": ("D13", "D14", "D20"),
    "long-30": ("D13", "D14", "D18", "D20"),
    "long-60": ("D13", "D14", "D18", "D20"),
    "long-120": ("D13", "D14", "D18", "D20"),
    "narr-30": ("D14", "D20"),
    SUPPLEMENT: ("D34", "D35", "D36"),
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
SHIPPED_DOCS = ("DECISIONS.md", "PRE-REGISTRATION.md")
FREEZE_COMMIT = "6ea27e2"
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
    files |= {"environment/system_prompt.txt", "environment/tools.json"}
    return files


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
        r"edu|gov|info|biz|xyz|cloud|tech|site|online|tv|cc|internal|local|corp|lan)\b"
    ),
}
# The publish target necessarily names the account; nothing else may.
ALLOWED_LITERALS = ("caiotheodoro/lacuna-capture-histories",)
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

REASONING_KEY = re.compile(r'\\*"reasoning_content\\*"\s*:')


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
        if REASONING_KEY.search(p.read_text(errors="replace")):
            problems.append(f"{rel}: a reasoning_content key")
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


def _caveat_problems(run: str, caveat, extra: tuple[str, ...] = ()) -> list[str]:
    if not isinstance(caveat, str) or len(caveat) < 40:
        return [f"{run}: caveat missing or too short"]
    missing = [ref for ref in (*CAVEAT_REFS.get(run, ()), *extra) if ref not in caveat]
    return [f"{run}: caveat does not cite {missing}"] if missing else []


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
    confirmatory_run = json.loads((artifacts / "hypotheses.json").read_text())["run"]
    if (
        confirmatory_run != DEFAULT_CONFIG
        or EXPECTED_PHASES.get(confirmatory_run) != "confirmatory"
    ):
        checks.append(("confirmatory run", False, f"hypotheses.json names {confirmatory_run}"))

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
            "env": "backoffice",
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
        if run == confirmatory_run:
            problems += _caveat_problems(
                run,
                meta.get("caveat"),
                (f"{summary['episodes']} of {summary['episodes'] + len(failed)}",),
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
                        "env": "backoffice",
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
    sources = {f"thinning/{p.name}": p for p in sorted((artifacts / "thinning").glob("*.json"))}
    sources |= {p.name: p for p in sorted(artifacts.glob("*.json"))}
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


def check_docs_and_environment(dataset_dir: Path) -> Check:
    from lacuna.agents.llm import AGENT_SYSTEM
    from lacuna.envs.tools import TOOLS

    problems = []
    for name in SHIPPED_DOCS:
        p = dataset_dir / "docs" / name
        if not p.exists() or p.read_bytes() != (DOCS / name).read_bytes():
            problems.append(f"docs/{name} missing or differs from the repo")
    env = dataset_dir / "environment"
    prompt, tools = env / "system_prompt.txt", env / "tools.json"
    if not prompt.exists() or prompt.read_text() != AGENT_SYSTEM + "\n":
        problems.append("environment/system_prompt.txt is not AGENT_SYSTEM")
    if not tools.exists() or json.loads(tools.read_text()) != TOOLS:
        problems.append("environment/tools.json is not TOOLS")
    gitattr = dataset_dir / ".gitattributes"
    if not gitattr.exists() or gitattr.read_text() != GITATTRIBUTES:
        problems.append(".gitattributes differs from the pinned content")
    try:
        freeze = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", "docs/PRE-REGISTRATION.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        freeze = "unknown"
    if freeze != FREEZE_COMMIT:
        problems.append(f"PRE-REGISTRATION.md was last touched at {freeze}, not {FREEZE_COMMIT}")
    return (
        "docs and environment",
        not problems,
        "; ".join(problems) or "docs, prompt, tools and .gitattributes match; freeze intact",
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
    return {
        "total": round(total, 6),
        "study": round(total - diagnostic, 6),
        "diagnostic": round(diagnostic, 6),
        "diagnostic_runs": sorted(r for r in per_run if r in DIAGNOSTIC_RUNS),
    }


SNIPPET_OUTPUT = re.compile(r"Expected output[^\n]*\n\n```text\n(.*?)\n```", re.S)
COMMIT_STAMP = re.compile(r"from lacuna commit `([^`]+)`")


def _snippet(body: str) -> str | None:
    section = body.split("## Rebuild a capture-history table and rerun an estimator", 1)
    if len(section) < 2:
        return None
    m = re.search(r"```python\n(.*?)```", section[1].split("\n## ", 1)[0], re.S)
    return m.group(1) if m else None


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
        FREEZE_COMMIT,
    )
    if regenerated == text:
        return ("card regenerates", True, "README.md is exactly card() of the shipped files")
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
        ("docs and environment", lambda: [check_docs_and_environment(dataset_dir)]),
        ("card metadata", lambda: [check_card_metadata(dataset_dir, artifacts)]),
        ("card regenerates", lambda: [check_card_regenerates(dataset_dir, artifacts)]),
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
