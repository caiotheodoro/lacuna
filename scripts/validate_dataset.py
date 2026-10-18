#!/usr/bin/env python3
"""Check dataset/ before anything is published. Exits 1 on any failure.

  uv run --group dataset python scripts/validate_dataset.py

Counts against each run's summary.json, ledger and witness integrity, the
pooled capture-history tables recomputed from the rows, results copied byte
for byte, the card's metadata, the card's own snippet run as written, every
config loaded through `datasets` the way the Hub viewer would, and the
privacy gate over every file. No network.
"""

from __future__ import annotations

import argparse
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
DATASET = ROOT / "dataset"

Check = tuple[str, bool, str]

WITNESS_IDS = {0, 1, 2, 3}
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
REQUIRED_REFS = ("D14", "D20", "D30", "D32")

# ---------------------------------------------------------------------------
# privacy gate
# ---------------------------------------------------------------------------

# Stricter than `make privacy-gate`: a key prefix followed by anything at all
# fails, because a truncated key is still a leaked key. False positives fail
# closed and a human looks.
KEY_SHAPES = {
    "sk- key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]"),
    "oc_sk_ key": re.compile(r"oc_sk_"),
    "Bearer token": re.compile(r"Bearer\s+\S"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{8,}"),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z0-9-]+)")
# RFC 2606 reserves these, and their subdomains, for examples.
ALLOWED_DOMAINS = ("example.com", "example.org", "example.net")
LOCAL_PATH = re.compile(r"/Users/|/home/[A-Za-z]|[A-Za-z]:\\+Users\\+")
# JSON escapes (\n, and \\n inside a JSON-encoded string) glue the next token
# to a letter; a key after a newline must still sit at a word boundary.
_ESCAPES = re.compile(r"\\+[nrtbf]")


def email_domain_status(domain: str) -> str:
    """'ok' for a reserved example domain, 'redo' for the scripted agent's x-suffix, else 'bad'.

    `ScriptedAgent` gets a customer field wrong first by appending 'x' to the
    value, so a redone email update reads ...@example.orgx. That is the
    reserved domain plus one character, not a real address.
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
    norm = _ESCAPES.sub(" ", text)
    found = []
    for label, rx in KEY_SHAPES.items():
        for m in rx.finditer(norm):
            found.append(f"{where}: {label} at offset {m.start()}")
    redo = 0
    for m in EMAIL.finditer(norm):
        status = email_domain_status(m.group(1))
        if status == "bad":
            found.append(f"{where}: email outside the example domains ({m.group(1)})")
        elif status == "redo":
            redo += 1
    for m in LOCAL_PATH.finditer(norm):
        found.append(f"{where}: local filesystem path at offset {m.start()}")
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
# counts and integrity
# ---------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


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

    from lacuna.envs.kernel import LedgerEntry
    from lacuna.linkage import oracle_table, signature_table
    from lacuna.witnesses import Record

    for run in on_disk:
        path = dataset_dir / "episodes" / f"{run}.jsonl"
        if not path.exists() or run not in runs:
            continue
        summary = json.loads((artifacts / "episodes" / run / "summary.json").read_text())
        source = _jsonl(artifacts / "episodes" / run / "episodes.jsonl")
        rows = _jsonl(path)
        meta = runs[run]
        problems: list[str] = []

        if len(rows) != summary["episodes"]:
            problems.append(f"{len(rows)} rows, summary says {summary['episodes']}")
        if meta["episodes"] != summary["episodes"] or meta["n_true"] != summary["n_true"]:
            problems.append("runs.jsonl episodes/n_true differ from summary")
        total = sum(r["n_true"] for r in rows)
        if total != summary["n_true"]:
            problems.append(f"sum n_true {total}, summary says {summary['n_true']}")
        failed = {f["seed"] for f in summary.get("failed_episodes") or []}
        seeds = [r["seed"] for r in rows]
        if len(set(seeds)) != len(seeds):
            problems.append("duplicate seeds")
        if failed & set(seeds):
            problems.append(f"failed seeds present as rows: {sorted(failed & set(seeds))}")

        witnesses: dict[int, int] = {}
        oracle = sum_sig = None
        for row, src in zip(rows, source, strict=False):
            tag = f"seed {row['seed']}"
            if row["run"] != run:
                problems.append(f"{tag}: run field {row['run']!r}")
            for key in ("phase", "env", "model", "tickets", "auditor_view"):
                if row[key] != meta[key]:
                    problems.append(f"{tag}: {key} {row[key]!r} != runs.jsonl {meta[key]!r}")
            if row["n_true"] != len(row["ledger"]):
                problems.append(f"{tag}: n_true {row['n_true']} != ledger {len(row['ledger'])}")
            if (row["seed"], row["ledger"], row["records"]) != (
                src["seed"],
                src["ledger"],
                src["records"],
            ):
                problems.append(f"{tag}: ledger or records differ from the artifact")
            t = row["transcript"]
            if (None if t is None else json.loads(t)) != src.get("transcript"):
                problems.append(f"{tag}: transcript differs from the artifact")
            ids = [e["id"] for e in row["ledger"]]
            id_set = set(ids)
            if len(id_set) != len(ids):
                problems.append(f"{tag}: duplicate ledger ids")
            for r in row["records"]:
                w = r["witness"]
                if not isinstance(w, int) or isinstance(w, bool) or w not in WITNESS_IDS:
                    problems.append(f"{tag}: witness id {w!r} outside 0..3")
                    continue
                witnesses[w] = witnesses.get(w, 0) + 1
                if r["oracle"] is not None and r["oracle"] not in id_set:
                    problems.append(f"{tag}: oracle id {r['oracle']} not in the ledger")
                if w == AUDITOR and r["oracle"] is not None:
                    problems.append(f"{tag}: an auditor record carries an oracle id")
            o, _ = oracle_table(
                [LedgerEntry(**x) for x in row["ledger"]], [Record(**r) for r in row["records"]], 4
            )
            s = signature_table([Record(**r) for r in row["records"]], 4)
            oracle = o if oracle is None else oracle + o
            sum_sig = s if sum_sig is None else sum_sig + s

        if oracle is not None:
            want = summary["oracle_table"]
            if oracle.tolist() != want or meta["oracle_table"] != want:
                problems.append("recomputed oracle table differs from summary.json")
            if sum_sig.tolist() != summary["signature_table"]:
                problems.append("recomputed signature table differs from summary.json")
        detail = (
            "; ".join(problems[:4])
            if problems
            else (
                f"{len(rows)} rows, N={total}, witnesses {dict(sorted(witnesses.items()))}, "
                "tables match summary"
            )
        )
        checks.append((f"run {run}", not problems, detail))
    return checks


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_results(dataset_dir: Path, artifacts: Path) -> Check:
    expected = {
        f"thinning/{p.name}": p for p in sorted((artifacts / "thinning").glob("*.json"))
    } | {name: artifacts / name for name in ("hypotheses.json", "sim_coverage.json")}
    built = {
        str(p.relative_to(dataset_dir / "results")): p
        for p in (dataset_dir / "results").rglob("*")
        if p.is_file()
    }
    problems = []
    if sorted(built) != sorted(expected):
        problems.append(f"files {sorted(built)} != sources {sorted(expected)}")
    for name, src in expected.items():
        if name in built and _sha(built[name]) != _sha(src):
            problems.append(f"{name} differs from its artifact")
    return (
        "results byte-identical",
        not problems,
        "; ".join(problems) or f"{len(expected)} files match their artifacts",
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
    if sum(bool(c.get("default")) for c in configs) != 1:
        problems.append("not exactly one default config")
    for section in REQUIRED_SECTIONS:
        if section not in body:
            problems.append(f"missing section {section!r}")
    for ref in REQUIRED_REFS:
        if ref not in body:
            problems.append(f"does not cite {ref}")
    if "synthetic" not in body.lower() or "not production-validated" not in body:
        problems.append("missing the synthetic / not production-validated disclaimer")
    return (
        "card metadata",
        not problems,
        "; ".join(problems) or f"{len(configs)} configs, every data file in one",
    )


def check_snippet(dataset_dir: Path) -> Check:
    _, body = split_card((dataset_dir / "README.md").read_text())
    m = re.search(r"```python\n(.*?)```", body, re.S)
    if not m:
        return ("card snippet runs", False, "no python block on the card")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", m.group(1)],
            cwd=dataset_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return ("card snippet runs", False, "timed out")
    out = proc.stdout.strip().splitlines()
    ok = proc.returncode == 0 and any(line.startswith("ll_bic") for line in out)
    detail = " | ".join(out[-3:]) if ok else (proc.stderr.strip().splitlines() or ["?"])[-1]
    return ("card snippet runs", ok, detail)


def check_loads(dataset_dir: Path) -> Check:
    """Load every config through `datasets`, offline, as the Hub viewer would."""
    import yaml

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import datasets

    datasets.disable_progress_bars()
    datasets.logging.set_verbosity_error()
    head, _ = split_card((dataset_dir / "README.md").read_text())
    configs = yaml.safe_load(head)["configs"]
    counts = {"episodes": sum(len(_jsonl(p)) for p in (dataset_dir / "episodes").glob("*.jsonl"))}
    counts["runs"] = len(_jsonl(dataset_dir / "runs.jsonl"))
    for p in (dataset_dir / "episodes").glob("*.jsonl"):
        counts[p.stem] = len(_jsonl(p))
    problems, loaded = [], []
    with tempfile.TemporaryDirectory() as cache:
        for c in configs:
            name = c["config_name"]
            try:
                ds = datasets.load_dataset(str(dataset_dir), name, split="train", cache_dir=cache)
            except Exception as exc:  # noqa: BLE001 - any failure here is the finding
                problems.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            if ds.num_rows != counts.get(name):
                problems.append(f"{name}: {ds.num_rows} rows, expected {counts.get(name)}")
            loaded.append(f"{name}={ds.num_rows}")
    return ("datasets loads every config", not problems, "; ".join(problems) or ", ".join(loaded))


# ---------------------------------------------------------------------------


def validate(dataset_dir: Path = DATASET, artifacts: Path = ARTIFACTS) -> list[Check]:
    layout = [dataset_dir / n for n in ("README.md", "runs.jsonl", "episodes", "results")]
    missing = [str(p.relative_to(dataset_dir)) for p in layout if not p.exists()]
    if missing:
        return [("layout", False, f"missing {missing}; run scripts/build_dataset.py")]
    checks: list[Check] = [("layout", True, "README.md, runs.jsonl, episodes/, results/")]
    # Each check runs whatever the others did, and a check that crashes on a
    # malformed file is a failure with its reason, not a traceback that hides
    # the checks after it.
    steps = (
        ("privacy gate", lambda: [check_privacy(dataset_dir)]),
        ("runs", lambda: check_runs(dataset_dir, artifacts)),
        ("results byte-identical", lambda: [check_results(dataset_dir, artifacts)]),
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
    args = ap.parse_args(argv)
    return 0 if report(validate(args.dataset_dir)) else 1


if __name__ == "__main__":
    sys.exit(main())
