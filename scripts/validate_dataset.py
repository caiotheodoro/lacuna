#!/usr/bin/env python3
"""Check dataset/ before anything is published. Exits 1 on any failure.

  uv run --group dataset python scripts/validate_dataset.py

Every row and run is checked against sources the builder does not control:
each run's summary.json, receipt.json and episodes.jsonl, hypotheses.json for
which run is confirmatory, and the ledgers themselves for any ticket count a
summary did not record. Then the results and docs byte for byte, the
environment files against the code, the card's metadata and banner, the
card's snippet run in an isolated interpreter, every config loaded through
`datasets` in a child process (so this process never goes offline), and the
privacy gate over every file. No network.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
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

HARD_WITNESSES = {0, 1, 2}
AUDITOR = 3
REQUIRED_TAGS = {
    "agent-auditing",
    "capture-recapture",
    "ai-governance",
    "logging",
    "llm-agents",
    "multiple-systems-estimation",
}
REQUIRED_SECTIONS = (
    "## What the data is",
    "## The three witnesses, and why the LLM auditor is not one",
    "## Field reference",
    "## Rebuild a capture-history table and rerun an estimator",
    "## Exploratory vs confirmatory",
    "## Known caveats",
    "## Synthetic data",
    "## Citation",
)
BANNER = "> **Read before any result below.**"
SHIPPED_DOCS = ("DECISIONS.md", "PRE-REGISTRATION.md")

# ---------------------------------------------------------------------------
# privacy gate
# ---------------------------------------------------------------------------

# Stricter than `make privacy-gate`: a credential prefix followed by anything
# fails, because a truncated key is still a leaked key. False positives fail
# closed and a human looks.
KEY_SHAPES = {
    "sk- key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]"),
    "oc_sk_ key": re.compile(r"oc_sk_"),
    "hf_ token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]"),
    "GitHub token": re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]"),
    "Slack token": re.compile(r"xox[abp]-"),
    "AWS access key": re.compile(r"A(?:KIA|SIA)[0-9A-Z]{16}|AKIA[0-9A-Z]{8,}"),
    "Bearer token": re.compile(r"Bearer\s+\S"),
    "PEM block": re.compile(r"-----BEGIN [A-Z0-9 ]*-----"),
    "JWT": re.compile(r"eyJ[A-Za-z0-9_-]+\."),
    "provider request id": re.compile(r"(?i)request[_ -]?id\W{0,3}[0-9a-f]{4,}"),
    # Eight-four-four is enough: stored errors are truncated mid-id.
    "UUID": re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}\b"),
}
IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
)
LOCAL_PATH = re.compile(
    r"/Users/|/home/|/root/|/private/|~/|[A-Za-z]:\\+(?:Users|Documents and Settings)\\+"
)
OWNER_NAMES = re.compile(r"(?i)caio|theodoro|matheus")
# The publish target necessarily names the account; nothing else may.
ALLOWED_LITERALS = ("caiotheodoro/lacuna-capture-histories",)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z0-9-]+)")
# RFC 2606 reserves these, and their subdomains, for examples.
ALLOWED_DOMAINS = ("example.com", "example.org", "example.net")
# JSON escapes (\n, and \\n inside a JSON-encoded string) glue the next token
# to a letter. Credentials and emails are matched on both the raw text and a
# copy with escapes blanked; paths only on the raw text, where a blanked \b
# would break C:\\Users\\bob apart.
_ESCAPES = re.compile(r"\\+[nrtbf]")


def email_domain_status(domain: str) -> str:
    """'ok' for a reserved example domain, 'redo' for the scripted agent's x-suffix, else 'bad'.

    `ScriptedAgent` gets a customer field wrong first by appending 'x' to the
    value, so a redone email update reads ...@example.orgx: the reserved
    domain plus one character, not a real address.
    """
    d = domain.lower()
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
        elif status == "redo":
            redo += 1
    for rx, label in (
        (IPV4, "IPv4 address"),
        (LOCAL_PATH, "local filesystem path"),
        (OWNER_NAMES, "owner name"),
    ):
        hits = sorted({m.group(0) for m in rx.finditer(raw)})
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
# rows and runs, against sources the builder does not control
# ---------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def redact(error: str) -> str:
    """This file's own redaction, so the builder's is checked rather than trusted."""
    return re.sub(r"(?i)(request[_ -]?id\W{0,3})[0-9a-f][0-9a-f-]*", r"\1<redacted>", error)


def expected_phases(run: str, summary: dict, confirmatory_run: str) -> set[str]:
    """Which phases a run may carry, from its summary and hypotheses.json alone."""
    if summary.get("agent") == "scripted":
        return {"scripted harness check"}
    if run == confirmatory_run:
        return {"confirmatory"}
    return {"exploratory", "post-hoc supplement"}


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


def _same(problems: list[str], tag: str, key: str, got, want) -> None:
    if got != want:
        problems.append(f"{tag}: {key} {got!r}, source says {want!r}")


def check_runs(dataset_dir: Path, artifacts: Path) -> list[Check]:
    checks: list[Check] = []
    on_disk = sorted(p.name for p in (artifacts / "episodes").iterdir() if p.is_dir())
    built = sorted(p.stem for p in (dataset_dir / "episodes").glob("*.jsonl"))
    runs = {r["run"]: r for r in _jsonl(dataset_dir / "runs.jsonl")}
    checks.append(
        (
            "every run built",
            on_disk == built == sorted(runs),
            f"artifacts {on_disk}; episodes/ {built}; runs.jsonl {sorted(runs)}",
        )
    )
    confirmatory_run = json.loads((artifacts / "hypotheses.json").read_text())["run"]

    from lacuna.envs.kernel import LedgerEntry
    from lacuna.linkage import oracle_table, signature_table
    from lacuna.witnesses import Record

    for run in on_disk:
        path = dataset_dir / "episodes" / f"{run}.jsonl"
        if not path.exists() or run not in runs:
            continue
        src_dir = artifacts / "episodes" / run
        summary = json.loads((src_dir / "summary.json").read_text())
        receipt_path = src_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        source = _jsonl(src_dir / "episodes.jsonl")
        rows = _jsonl(path)
        meta = runs[run]
        problems: list[str] = []

        tickets = summary.get("tickets")
        if tickets is None:
            claimed = meta.get("tickets") or 0
            tickets = infer_tickets(
                [(e["seed"], e["ledger"]) for e in source], max(claimed + 3, 12)
            )
            if tickets is None:
                problems.append("summary has no tickets and the ledgers fit no queue length")
        auditor_view = summary.get("auditor_view") or "full"  # the only view before D14
        failed = summary.get("failed_episodes") or []
        want_run = {
            "model": summary["model"],
            "agent": summary["agent"],
            "auditor": summary["auditor"],
            "episodes": summary["episodes"],
            "episodes_attempted": summary["episodes"] + len(failed),
            "n_true": summary["n_true"],
            "seed0": summary["seed0"],
            "tickets": tickets,
            "auditor_view": auditor_view,
            "recorded_fault_rates": summary["fault_rates"],
            "failed_episodes": [{"seed": f["seed"], "error": redact(f["error"])} for f in failed],
            "spend_usd": receipt["total_usd"] if receipt else None,
            "llm_calls": len(receipt["calls"]) if receipt else None,
            "oracle_table": summary["oracle_table"],
            "signature_table": summary["signature_table"],
            "transcripts_stored": all(e.get("transcript") is not None for e in source),
        }
        for key, want in want_run.items():
            _same(problems, "runs.jsonl", key, meta.get(key), want)
        if meta.get("phase") not in expected_phases(run, summary, confirmatory_run):
            problems.append(f"phase {meta.get('phase')!r} not allowed for this run")
        if "source" in meta:
            problems.append("runs.jsonl still carries a local source path")

        if len(rows) != len(source) or len(rows) != summary["episodes"]:
            problems.append(
                f"{len(rows)} rows, artifact has {len(source)}, summary says {summary['episodes']}"
            )
        else:
            included = meta.get("transcripts_included")
            oracle = sig = None
            for row, src in zip(rows, source, strict=True):
                tag = f"seed {row.get('seed')}"
                _same(problems, tag, "seed", row["seed"], src["seed"])
                for key, want in (
                    ("run", run),
                    ("phase", meta["phase"]),
                    ("env", meta["env"]),
                    ("model", summary["model"]),
                    ("tickets", tickets),
                    ("auditor_view", auditor_view),
                    ("n_true", len(src["ledger"])),
                    ("ledger", src["ledger"]),
                    ("records", [r for r in src["records"] if r["witness"] != AUDITOR]),
                    ("auditor_records", [r for r in src["records"] if r["witness"] == AUDITOR]),
                    ("agent_read_trace", agent_read_trace(src.get("transcript"))),
                    ("n_calls", src["n_calls"]),
                    ("n_tool_errors", src["n_tool_errors"]),
                ):
                    _same(problems, tag, key, row.get(key), want)
                if {r["witness"] for r in row["records"]} - HARD_WITNESSES:
                    problems.append(f"{tag}: records hold a witness outside 0..2")
                if included:
                    t = row.get("transcript")
                    if (None if t is None else json.loads(t)) != src.get("transcript"):
                        problems.append(f"{tag}: transcript differs from the artifact")
                elif "transcript" in row:
                    problems.append(f"{tag}: transcript present in a build that withholds it")
                ids = {e["id"] for e in row["ledger"]}
                if any(r["oracle"] is not None and r["oracle"] not in ids for r in row["records"]):
                    problems.append(f"{tag}: an oracle id is not in the ledger")
                recs = [Record(**r) for r in row["records"] + row["auditor_records"]]
                o, _ = oracle_table([LedgerEntry(**x) for x in row["ledger"]], recs, 4)
                s = signature_table(recs, 4)
                oracle = o if oracle is None else oracle + o
                sig = s if sig is None else sig + s
            if oracle is not None and oracle.tolist() != summary["oracle_table"]:
                problems.append("recomputed oracle table differs from summary.json")
            if sig is not None and sig.tolist() != summary["signature_table"]:
                problems.append("recomputed signature table differs from summary.json")
        n = sum(r["n_true"] for r in rows)
        detail = (
            "; ".join(problems[:4])
            if problems
            else f"{len(rows)} rows, N={n}, phase {meta['phase']}, tickets {tickets}, "
            "matches summary, receipt and episodes"
        )
        checks.append((f"run {run}", not problems, detail))
    return checks


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_results(dataset_dir: Path, artifacts: Path) -> Check:
    """Every distinct result ships once, byte for byte; identical copies ship one name."""
    sources = {f"thinning/{p.name}": p for p in sorted((artifacts / "thinning").glob("*.json"))}
    sources |= {name: artifacts / name for name in ("hypotheses.json", "sim_coverage.json")}
    built = {
        str(p.relative_to(dataset_dir / "results")): p
        for p in (dataset_dir / "results").rglob("*")
        if p.is_file()
    }
    problems = [f"{name} has no source" for name in built if name not in sources]
    groups: dict[str, list[str]] = {}
    for name, p in sources.items():
        groups.setdefault(_sha(p), []).append(name)
    for digest, names in groups.items():
        shipped = [n for n in names if n in built]
        if len(shipped) != 1:
            problems.append(f"{names}: {len(shipped)} shipped, want exactly one")
        problems += [f"{n} differs from its artifact" for n in shipped if _sha(built[n]) != digest]
    return (
        "results byte-identical",
        not problems,
        "; ".join(problems) or f"{len(built)} files, one per distinct artifact",
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
    if not prompt.exists() or prompt.read_text().rstrip("\n") != AGENT_SYSTEM:
        problems.append("environment/system_prompt.txt is not AGENT_SYSTEM")
    if not tools.exists() or json.loads(tools.read_text()) != TOOLS:
        problems.append("environment/tools.json is not TOOLS")
    return (
        "docs and environment",
        not problems,
        "; ".join(problems) or "docs match the repo; prompt and tools match the code",
    )


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------


def split_card(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("card has no YAML front matter")
    head, _, body = text[4:].partition("\n---\n")
    return head, body


def check_card(dataset_dir: Path) -> Check:
    import yaml

    text = (dataset_dir / "README.md").read_text()
    try:
        head, body = split_card(text)
        meta = yaml.safe_load(head)
    except (ValueError, yaml.YAMLError) as exc:
        return ("card metadata", False, str(exc))
    problems = []
    if meta.get("license") != "apache-2.0":
        problems.append(f"license {meta.get('license')!r}")
    if not meta.get("pretty_name"):
        problems.append("no pretty_name")
    missing_tags = REQUIRED_TAGS - set(meta.get("tags") or [])
    if missing_tags:
        problems.append(f"missing tags {sorted(missing_tags)}")
    configs = meta.get("configs") or []
    covered = set()
    for c in configs:
        hits = sorted(dataset_dir.glob(c["data_files"]))
        if not hits:
            problems.append(f"config {c['config_name']} matches no file")
        covered |= {str(p.relative_to(dataset_dir)) for p in hits}
    data = {"runs.jsonl"} | {
        str(p.relative_to(dataset_dir)) for p in (dataset_dir / "episodes").glob("*.jsonl")
    }
    if data - covered:
        problems.append(f"files in no config: {sorted(data - covered)}")
    names = [c["config_name"] for c in configs]
    if sorted(names) != sorted(i["config_name"] for i in meta.get("dataset_info") or []):
        problems.append("dataset_info and configs name different configs")
    defaults = [c for c in configs if c.get("default")]
    if [c["data_files"] for c in defaults] != ["episodes/confirm.jsonl"]:
        problems.append(f"default config is {[c['config_name'] for c in defaults]}, want confirm")
    for section in REQUIRED_SECTIONS:
        if section not in body:
            problems.append(f"missing section {section!r}")
    if BANNER not in body or ("| PASS |" in body and body.index(BANNER) > body.index("| PASS |")):
        problems.append("the banner is missing or comes after a PASS row")
    decisions = (dataset_dir / "docs" / "DECISIONS.md").read_text()
    defined = set(re.findall(r"^\*\*D(\d+)\.", decisions, re.M))
    cited = set(re.findall(r"\bD(\d+)\b", body))
    if cited - defined:
        problems.append(f"cites D{sorted(cited - defined)} not in docs/DECISIONS.md")
    runs = _jsonl(dataset_dir / "runs.jsonl")
    if not any(r["transcripts_included"] for r in runs) and "withheld" not in body:
        problems.append("transcripts are withheld but the card does not say so")
    if "synthetic" not in body.lower() or "not production-validated" not in body:
        problems.append("missing the synthetic / not production-validated disclaimer")
    return (
        "card metadata",
        not problems,
        "; ".join(problems)
        or f"{len(configs)} configs, default confirm, banner first, D-refs resolve",
    )


def check_snippet(dataset_dir: Path) -> Check:
    """The card's snippet, run as written in an isolated interpreter: stdlib only."""
    _, body = split_card((dataset_dir / "README.md").read_text())
    m = re.search(r"```python\n(.*?)```", body, re.S)
    if not m:
        return ("card snippet runs", False, "no python block on the card")
    code = m.group(1)
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
    out = proc.stdout.strip().splitlines()
    ok = proc.returncode == 0 and any("Chapman" in line for line in out)
    detail = out[-1] if ok else (proc.stderr.strip().splitlines() or ["?"])[-1]
    return ("card snippet runs", ok, f"stdlib only, isolated: {detail}" if ok else detail)


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
            try:
                ds = datasets.load_dataset(str(dataset_dir), name, split="train", cache_dir=cache)
            except Exception as exc:  # noqa: BLE001 - any failure here is the finding
                problems.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            if ds.num_rows != want:
                problems.append(f"{name}: {ds.num_rows} rows, expected {want}")
            loaded.append(f"{name}={ds.num_rows}")
    return not problems, "; ".join(problems) or ", ".join(loaded)


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


def validate(dataset_dir: Path = DATASET, artifacts: Path = ARTIFACTS) -> list[Check]:
    layout = [
        dataset_dir / n
        for n in ("README.md", "runs.jsonl", "episodes", "results", "environment", "docs")
    ]
    missing = [str(p.relative_to(dataset_dir)) for p in layout if not p.exists()]
    if missing:
        return [("layout", False, f"missing {missing}; run scripts/build_dataset.py")]
    checks: list[Check] = [("layout", True, "README, runs, episodes, results, environment, docs")]
    # Each check runs whatever the others did, and a check that crashes on a
    # malformed file is a failure with its reason, not a traceback that hides
    # the checks after it.
    steps = (
        ("privacy gate", lambda: [check_privacy(dataset_dir)]),
        ("runs", lambda: check_runs(dataset_dir, artifacts)),
        ("results byte-identical", lambda: [check_results(dataset_dir, artifacts)]),
        ("docs and environment", lambda: [check_docs_and_environment(dataset_dir)]),
        ("card metadata", lambda: [check_card(dataset_dir)]),
        ("card snippet runs", lambda: [check_snippet(dataset_dir)]),
        ("datasets loads every config", lambda: [check_loads(dataset_dir)]),
    )
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
