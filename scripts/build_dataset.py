#!/usr/bin/env python3
"""Build the Hugging Face dataset of lacuna's capture histories into dataset/.

  uv run --group dataset python scripts/build_dataset.py
  uv run --group dataset python scripts/build_dataset.py --with-transcripts   # not for release

Reads artifacts/episodes/<run>/{episodes.jsonl,summary.json,receipt.json}, the
derived results, the agent's prompt and tools, and the decision log, and
writes dataset/ (gitignored). Uploads nothing: that is
scripts/publish_dataset.py, after scripts/validate_dataset.py.

Every run on disk needs an entry in RUNS. An unknown run fails the build
before anything is written, and the previous build survives. Every result on
the card is read from a file in the build (AGENTS.md rule 2).

Transcripts are withheld by default: they hold provider reasoning text that
Apache-2.0 cannot license, and no result depends on them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
OUT = ROOT / "dataset"
REPO = "caiotheodoro/lacuna-capture-histories"

PHASES = ("confirmatory", "exploratory", "post-hoc supplement", "scripted harness check")
ENVS = ("backoffice", "infra")
AUDITOR = 3

_FULL_W4 = (
    "W4 is the LLM auditor reading the full transcript. It reads every call "
    "verbatim, so its recall is 1 and it is a copy of the trace, not a witness "
    "(D14, D20). Its records are in auditor_records, not records. Every @4 "
    "number is superseded."
)
_V4 = "Agent deepseek-v4-flash, before the switch to v4.1-flash (D13)."
_POSTHOC = "Recorded with complete logs; loss is applied afterwards by lacuna.thinning."

# One entry per run. `fill` supplies a field an older summary.json did not
# record; a summary that records a different value fails the build, and
# scripts/validate_dataset.py checks the fill against the ledgers. `caveat`
# may use {episodes} and {attempted}, filled from that run's summary.
RUNS: dict[str, dict] = {
    "confirm": {
        "phase": "confirmatory",
        "env": "backoffice",
        "caveat": (
            "The pre-registered confirmatory run. {episodes} of {attempted} "
            "episodes completed; the failed seeds are in failed_episodes and were "
            "not replaced or re-seeded. Two of the three losses were a harness "
            "defect, not transient errors (D33, correcting D30). Witness 3 is the "
            "scripted stand-in, not an LLM, and is excluded from every estimate "
            f"(D20). Agent deepseek-v4.1-flash. {_POSTHOC} Read with D32: under "
            "loss that depends on the action, the method detects incompleteness "
            "but cannot measure it."
        ),
    },
    "confirm-supp": {
        "phase": "post-hoc supplement",
        "env": "backoffice",
        "caveat": (
            "Labelled post-hoc supplement (D33): the confirmatory run's lost seeds "
            "rerun on the fixed harness. Kept out of the registered verdict. "
            f"{_POSTHOC}"
        ),
    },
    "scripted-l0": {
        "phase": "scripted harness check",
        "env": "backoffice",
        # Predates the summary field; validate_dataset.py infers 7 from the ledgers.
        "fill": {"tickets": 7, "auditor_view": "full"},
        "caveat": (
            "Scripted agent and scripted auditor, no LLM: a harness check, not "
            "agent behaviour. The agent follows each ticket's plan with deliberate "
            "redo and redraft mistakes; a redone customer field first gets a "
            "trailing 'x', so some addresses end in example.orgx on purpose. "
            "Witness 3 is a scripted noisy copy of the transcript. Logging faults "
            "were applied at record time (recorded_fault_rates), so records are "
            "already thinned."
        ),
    },
    "pilot-l0": {
        "phase": "exploratory",
        "env": "backoffice",
        # 7 tickets per D18's table; validate_dataset.py infers it from the ledgers.
        "fill": {"tickets": 7, "auditor_view": "full"},
        "caveat": (
            f"First live run. {_V4} {_FULL_W4} Logging faults were applied at "
            "record time (recorded_fault_rates), so records are already thinned. "
            "No transcript was stored."
        ),
    },
    "long-30": {
        "phase": "exploratory",
        "env": "backoffice",
        "fill": {"auditor_view": "full"},
        "caveat": (
            f"Session-length run (D12, D18). {_V4} {_FULL_W4} {_POSTHOC} No "
            "transcript was stored. Source of the exploratory thinning results "
            "and of the estimator choice (D27)."
        ),
    },
    "long-60": {
        "phase": "exploratory",
        "env": "backoffice",
        "fill": {"auditor_view": "full"},
        "caveat": (
            f"Session-length run (D12, D18). {_V4} {_FULL_W4} {_POSTHOC} No "
            "transcript was stored. The agent skips a growing share of a long "
            "queue, so tickets overstate actions (D18)."
        ),
    },
    "long-120": {
        "phase": "exploratory",
        "env": "backoffice",
        "caveat": (
            f"Session-length run (D12, D18). {_V4} {_FULL_W4} {_POSTHOC} The "
            "agent skips more than half of a 120-ticket queue, so tickets "
            "overstate actions (D18)."
        ),
    },
    "narr-30": {
        "phase": "exploratory",
        "env": "backoffice",
        "caveat": (
            "Narration-only W4 (D14 fix, D20): the LLM auditor saw only what the "
            "agent said and reported nothing, so auditor_records is empty. The "
            "agent's narration is aggregate, not per action. Agent "
            f"deepseek-v4.1-flash. {_POSTHOC}"
        ),
    },
}

# Thinning summaries that are byte-identical to one kept under another name.
DUPLICATES = {"long-30.fixed-episodes.json": "long-30.json"}
SHIPPED_DOCS = ("DECISIONS.md", "PRE-REGISTRATION.md")
EPISODE_KEYS = {"seed", "ledger", "records", "n_calls", "n_tool_errors", "transcript"}
TRACE_PATH = "/var/log/agent_trace.jsonl"  # lacuna.envs.kernel.TRACE_PATH; checked below
_REQUEST_ID = re.compile(r"(request[_ -]?id\W{0,3})[0-9A-Fa-f][0-9A-Fa-f-]*", re.I)

SNIPPET = """\
import json
import random
from collections import Counter

rows = [json.loads(line) for line in open("episodes/confirm.jsonl")]

# Capture histories by oracle linkage: bit w is set if witness w saw the
# action (0 trace, 1 gateway, 2 env). Loss is injected here, as in results/,
# at the held-out fin_mid setting: each pipe drops half its lines and the env
# diff sees financial ops only. Witness 3 is in auditor_records, not records.
rng = random.Random(4242)
FINANCIAL = {"issue_refund", "cancel_order", "send_email"}
hist, n_true = Counter(), 0
for row in rows:
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

# The floor, and the simplest capture-recapture estimator (Chapman, W1 x W2).
union = n_true - hist[0]
n1 = sum(c for h, c in hist.items() if h & 1)
n2 = sum(c for h, c in hist.items() if h & 2)
m12 = sum(c for h, c in hist.items() if (h & 3) == 3)
chapman = (n1 + 1) * (n2 + 1) / (m12 + 1) - 1
print(f"true N {n_true}  unseen by all {hist[0]}  union {union}  Chapman {chapman:.1f}")
"""


class UnknownRun(ValueError):
    """A run on disk has no entry in RUNS. Add one; never guess its phase."""


class BuildError(RuntimeError):
    """The artifacts disagree with RUNS or with themselves."""


def run_meta(run: str) -> dict:
    if run not in RUNS:
        raise UnknownRun(
            f"run {run!r} has no entry in RUNS (scripts/build_dataset.py). Add its "
            f"phase, env and caveat; known runs: {sorted(RUNS)}"
        )
    meta = RUNS[run]
    if meta["phase"] not in PHASES:
        raise BuildError(f"{run}: phase {meta['phase']!r} not in {PHASES}")
    if meta["env"] not in ENVS:
        raise BuildError(f"{run}: env {meta['env']!r} not in {ENVS}")
    if not meta.get("caveat"):
        raise BuildError(f"{run}: no caveat")
    return meta


def resolve(run: str, meta: dict, summary: dict) -> dict:
    """tickets and auditor_view: the summary's value, else RUNS' fill, never both disagreeing."""
    out = {}
    for key in ("tickets", "auditor_view"):
        recorded, filled = summary.get(key), meta.get("fill", {}).get(key)
        if recorded is not None and filled is not None and recorded != filled:
            raise BuildError(f"{run}: summary says {key}={recorded!r}, RUNS fills {filled!r}")
        value = recorded if recorded is not None else filled
        if value is None:
            raise BuildError(f"{run}: no {key} in summary.json and none in RUNS fill")
        out[key] = value
    return out


def redact(error: str) -> str:
    """Provider request ids out of a stored error; the message stays."""
    return _REQUEST_ID.sub(r"\1<redacted>", error)


def read_trace(transcript: list[dict] | None) -> bool | None:
    """Did the agent read W1's trace file? None when no transcript was stored."""
    if transcript is None:
        return None
    for m in transcript:
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] != "read_file":
                continue
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(args, dict) and args.get("path") == TRACE_PATH:
                return True
    return False


def spend_summary(runs: list[dict]) -> tuple[float, list[str]]:
    """Spend over runs with a receipt, and the runs without one. Never sums unknown as 0."""
    known = [r["spend_usd"] for r in runs if r["spend_usd"] is not None]
    return round(sum(known), 6), [r["run"] for r in runs if r["spend_usd"] is None]


def fmt_usd(v: float | None) -> str:
    return "unknown" if v is None else f"{v:.2f}"


def _w4_recall(episodes: list[dict]) -> float:
    """Share of ledger actions witness 3 reported, matched by signature (oracle linkage)."""
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


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def freeze_commit() -> str:
    """The commit that last touched the pre-registration, which is its freeze."""
    try:
        return subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", "docs/PRE-REGISTRATION.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def plan_run(run: str, src: Path) -> dict:
    """Everything a run needs, read and checked before any output is written."""
    meta = run_meta(run)
    summary = json.loads((src / "summary.json").read_text())
    fields = resolve(run, meta, summary)
    receipt_path = src / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    return {"meta": meta, "summary": summary, "fields": fields, "receipt": receipt}


def build_run(run: str, src: Path, plan: dict, with_transcripts: bool) -> tuple[list[dict], dict]:
    meta, summary, fields, receipt = plan["meta"], plan["summary"], plan["fields"], plan["receipt"]
    raw = [json.loads(line) for line in (src / "episodes.jsonl").open() if line.strip()]
    rows = []
    for e in raw:
        extra = set(e) - EPISODE_KEYS
        if extra:
            raise BuildError(f"{run} seed {e.get('seed')}: unknown episode keys {sorted(extra)}")
        t = e.get("transcript")
        row = {
            "run": run,
            "seed": e["seed"],
            "model": summary["model"],
            "env": meta["env"],
            "tickets": fields["tickets"],
            "phase": meta["phase"],
            "auditor_view": fields["auditor_view"],
            "n_true": len(e["ledger"]),
            "ledger": e["ledger"],
            # The three hard witnesses only, so a naive union over records is
            # the floor. The auditor equals the ledger when it reads the full
            # transcript (D20), and would make that union N by construction.
            "records": [r for r in e["records"] if r["witness"] != AUDITOR],
            "auditor_records": [r for r in e["records"] if r["witness"] == AUDITOR],
            "agent_read_trace": read_trace(t),
            "n_calls": e["n_calls"],
            "n_tool_errors": e["n_tool_errors"],
        }
        if with_transcripts:
            # A JSON string, so every config has one fixed schema.
            row["transcript"] = None if t is None else json.dumps(t, separators=(",", ":"))
        rows.append(row)

    failed = summary.get("failed_episodes") or []
    attempted = summary["episodes"] + len(failed)
    run_row = {
        "run": run,
        "phase": meta["phase"],
        "env": meta["env"],
        "agent": summary["agent"],
        "auditor": summary["auditor"],
        "model": summary["model"],
        "auditor_view": fields["auditor_view"],
        "tickets": fields["tickets"],
        "episodes": summary["episodes"],
        "episodes_attempted": attempted,
        "failed_episodes": [{"seed": f["seed"], "error": redact(f["error"])} for f in failed],
        "stopped": summary.get("stopped") or None,
        "seed0": summary["seed0"],
        "n_true": summary["n_true"],
        "recorded_fault_rates": summary["fault_rates"],
        # No receipt means the spend is unknown, never zero.
        "spend_usd": receipt["total_usd"] if receipt else None,
        "llm_calls": len(receipt["calls"]) if receipt else None,
        "transcripts_stored": all(e.get("transcript") is not None for e in raw),
        "transcripts_included": with_transcripts,
        "w4_recall": _w4_recall(raw),
        "oracle_table": summary["oracle_table"],
        "signature_table": summary["signature_table"],
        "caveat": meta["caveat"].format(episodes=summary["episodes"], attempted=attempted),
    }
    return rows, run_row


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def _copy_results(out: Path) -> list[str]:
    copied = []
    for src in sorted((ARTIFACTS / "thinning").glob("*.json")):
        if src.name in DUPLICATES:
            twin = src.parent / DUPLICATES[src.name]
            if src.read_bytes() != twin.read_bytes():
                raise BuildError(f"{src.name} is no longer identical to {twin.name}; ship both")
            continue
        dst = out / "results" / "thinning" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied.append(f"results/thinning/{src.name}")
    for name in ("hypotheses.json", "sim_coverage.json"):
        shutil.copyfile(ARTIFACTS / name, out / "results" / name)
        copied.append(f"results/{name}")
    return copied


def _write_environment(out: Path) -> None:
    from lacuna.agents.llm import AGENT_SYSTEM
    from lacuna.envs.kernel import TRACE_PATH as KERNEL_TRACE
    from lacuna.envs.tools import TOOLS

    if KERNEL_TRACE != TRACE_PATH:
        raise BuildError(f"trace path moved to {KERNEL_TRACE}; update TRACE_PATH")
    env = out / "environment"
    env.mkdir(parents=True)
    (env / "system_prompt.txt").write_text(AGENT_SYSTEM + "\n")
    (env / "tools.json").write_text(json.dumps(TOOLS, indent=2) + "\n")


def _is_previous_build(out: Path) -> bool:
    readme = out / "README.md"
    return not any(out.iterdir()) or (readme.exists() and REPO in readme.read_text())


def build(out: Path = OUT, with_transcripts: bool = False) -> dict:
    """Plan every run, build into a temp dir beside `out`, then swap it in."""
    if out.exists() and not _is_previous_build(out):
        raise BuildError(f"{out} exists and is not a previous build; not touching it")
    episodes_dir = ARTIFACTS / "episodes"
    on_disk = sorted(p.name for p in episodes_dir.iterdir() if p.is_dir())
    plans = {run: plan_run(run, episodes_dir / run) for run in on_disk}  # raises first
    order = [r for r in RUNS if r in plans]

    tmp = Path(tempfile.mkdtemp(prefix=".dataset-build-", dir=out.parent))
    try:
        runs, all_rows = [], {}
        for run in order:
            rows, run_row = build_run(run, episodes_dir / run, plans[run], with_transcripts)
            _dump_jsonl(tmp / "episodes" / f"{run}.jsonl", rows)
            runs.append(run_row)
            all_rows[run] = rows
        _dump_jsonl(tmp / "runs.jsonl", runs)
        results = _copy_results(tmp)
        _write_environment(tmp)
        (tmp / "docs").mkdir()
        for name in SHIPPED_DOCS:
            shutil.copyfile(DOCS / name, tmp / "docs" / name)
        (tmp / ".gitattributes").write_text("*.jsonl filter=lfs diff=lfs merge=lfs -text\n")
        (tmp / "README.md").write_text(card(runs, all_rows, tmp, git_commit(), with_transcripts))
        if out.exists():
            old = Path(tempfile.mkdtemp(prefix=".dataset-old-", dir=out.parent))
            out.rename(old / "dataset")
            tmp.rename(out)
            shutil.rmtree(old)
        else:
            tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "runs": order,
        "declared_not_on_disk": [r for r in RUNS if r not in plans],
        "episodes": {r["run"]: r["episodes"] for r in runs},
        "results": results,
    }


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------

_STR = {"dtype": "string"}
_INT = {"dtype": "int64"}
_F64 = {"dtype": "float64"}
_BOOL = {"dtype": "bool"}


def _features(cols: list[tuple[str, dict]]) -> list[dict]:
    return [{"name": n, **t} for n, t in cols]


_RECORD = {
    "list": _features(
        [("witness", _INT), ("op", _STR), ("resource", _STR), ("value", _STR), ("oracle", _INT)]
    )
}


def episode_features(with_transcripts: bool) -> list[dict]:
    cols = [
        ("run", _STR),
        ("seed", _INT),
        ("model", _STR),
        ("env", _STR),
        ("tickets", _INT),
        ("phase", _STR),
        ("auditor_view", _STR),
        ("n_true", _INT),
        (
            "ledger",
            {
                "list": _features(
                    [
                        ("id", _INT),
                        ("op", _STR),
                        ("resource", _STR),
                        ("value", _STR),
                        ("step", _INT),
                    ]
                )
            },
        ),
        ("records", _RECORD),
        ("auditor_records", _RECORD),
        ("agent_read_trace", _BOOL),
        ("n_calls", _INT),
        ("n_tool_errors", _INT),
    ]
    if with_transcripts:
        cols.append(("transcript", _STR))
    return _features(cols)


RUN_FEATURES = _features(
    [
        ("run", _STR),
        ("phase", _STR),
        ("env", _STR),
        ("agent", _STR),
        ("auditor", _STR),
        ("model", _STR),
        ("auditor_view", _STR),
        ("tickets", _INT),
        ("episodes", _INT),
        ("episodes_attempted", _INT),
        ("failed_episodes", {"list": _features([("seed", _INT), ("error", _STR)])}),
        ("stopped", _STR),
        ("seed0", _INT),
        ("n_true", _INT),
        (
            "recorded_fault_rates",
            {"struct": _features([("p_gateway_drop", _F64), ("p_trace_drop", _F64)])},
        ),
        ("spend_usd", _F64),
        ("llm_calls", _INT),
        ("transcripts_stored", _BOOL),
        ("transcripts_included", _BOOL),
        ("w4_recall", _F64),
        ("oracle_table", {"sequence": "int64"}),
        ("signature_table", {"sequence": "int64"}),
        ("caveat", _STR),
    ]
)


def front_matter(runs: list[dict], with_transcripts: bool) -> str:
    import yaml

    n = sum(r["episodes"] for r in runs)
    size = "n<1K" if n < 1000 else "1K<n<10K"
    configs = []
    for r in runs:
        c = {"config_name": r["run"], "data_files": f"episodes/{r['run']}.jsonl"}
        if r["run"] == "confirm":
            c["default"] = True
        configs.append(c)
    configs.append({"config_name": "runs", "data_files": "runs.jsonl"})
    configs.sort(key=lambda c: not c.get("default"))
    info = [
        {
            "config_name": c["config_name"],
            "features": RUN_FEATURES
            if c["config_name"] == "runs"
            else episode_features(with_transcripts),
        }
        for c in configs
    ]
    meta = {
        "license": "apache-2.0",
        "pretty_name": "lacuna capture histories",
        "language": ["en"],
        "tags": [
            "agent-auditing",
            "capture-recapture",
            "ai-governance",
            "logging",
            "llm-agents",
            "multiple-systems-estimation",
        ],
        "size_categories": [size],
        "configs": configs,
        "dataset_info": info,
    }

    class _NoAlias(yaml.SafeDumper):
        # The Hub parses card metadata itself; spell every block out rather
        # than rely on it resolving &anchors.
        def ignore_aliases(self, data):
            return True

    return "---\n" + yaml.dump(meta, Dumper=_NoAlias, sort_keys=False, width=1000) + "---\n"


def _est(path: Path, key: str) -> tuple[dict, float]:
    d = json.loads(path.read_text())["configs"][key]
    return d["estimators"], d["mean_unseen_by_w1w2w3"]


def card(
    runs: list[dict], rows: dict[str, list[dict]], out: Path, commit: str, with_transcripts: bool
) -> str:
    by = {r["run"]: r for r in runs}
    conf = by["confirm"]
    hyp = json.loads((out / "results" / "hypotheses.json").read_text())
    thin_dir = out / "results" / "thinning"
    reps = json.loads((thin_dir / "confirm.json").read_text())["reps"]
    mid, mid_hidden = _est(thin_dir / "confirm.json", "fin_mid/signature")
    rho, rho_hidden = _est(thin_dir / "confirm.json", "fin_mid_rho100/signature")
    mnar, mnar_hidden = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar/signature")
    mnar_rho, _ = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar_rho100/signature")
    n_eps = sum(r["episodes"] for r in runs)
    n_actions = sum(r["n_true"] for r in runs)
    spend, no_receipt = spend_summary(runs)
    full_w4 = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "full"]
    narr = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "narration"]
    lost = conf["failed_episodes"]
    defect = [f for f in lost if "reasoning_content" in f["error"]]
    read = [r["agent_read_trace"] for r in rows["confirm"]]
    with_tx = [r["run"] for r in runs if r["transcripts_stored"] and not r["transcripts_included"]]
    supplement = (
        "A labelled post-hoc supplement, `confirm-supp`, reruns the lost seeds and is kept "
        "out of the verdict."
        if "confirm-supp" in by
        else "A labelled post-hoc supplement reruns the lost seeds; it is not in this build."
    )
    freeze = freeze_commit()

    def fmt(v: float | None, p: int = 2) -> str:
        return "unbounded" if v is None else f"{v:.{p}f}"

    def recall(rs: list[dict]) -> str:
        return ", ".join(f"{r['w4_recall']:.4f} in `{r['run']}`" for r in rs)

    run_table = "\n".join(
        f"| `{r['run']}` | {r['phase']} | {r['model'] or 'scripted'} | {r['tickets']} | "
        f"{r['episodes']} | {r['n_true']} | {r['w4_recall']:.4f} | {fmt_usd(r['spend_usd'])} |"
        for r in runs
    )
    hyp_table = "\n".join(
        f"| {name} | {'PASS' if h['passed'] else 'FAIL'} | {h['detail']} |"
        for name, h in hyp["hypotheses"].items()
    )
    unknown_spend = (
        f" Spend is unknown for {', '.join(f'`{r}`' for r in no_receipt)}, which has no "
        "receipt, and is not counted as zero."
        if no_receipt
        else ""
    )
    if with_transcripts:
        transcript_note = (
            "This build includes `transcript`. It holds provider reasoning text and "
            "is not for release under this license."
        )
    else:
        transcript_note = (
            f"Transcripts exist for {', '.join(f'`{r}`' for r in with_tx)} and are "
            "withheld from this release pending a check of the model provider's terms: "
            "they hold the provider's reasoning text, which this license cannot cover. "
            "No result here depends on them."
        )

    body = f"""
# lacuna capture histories

> **Read before any result below.**
> - **The confirmatory run is incomplete.** It completed {conf["episodes"]} of
>   {conf["episodes_attempted"]} episodes. {len(defect)} of the {len(lost)} losses were a
>   harness defect (an empty `reasoning_content` was dropped, so the provider returned
>   a 400; D33), not transient errors. The verdict is on the {conf["episodes"]} episodes as
>   registered. {supplement}
> - **Every coverage figure is under simulated loss.** Loss was injected post hoc into
>   complete logs. No figure here measures real log loss.
> - **The agent can read W1's trace file, though not write it**, and did so in
>   {sum(bool(x) for x in read)} of {len(read)} `confirm` episodes (`agent_read_trace`).

How many consequential actions did an agent take that no log line shows? This
dataset holds the witness records lacuna uses to answer that by
capture-recapture: {n_eps} episodes across {len(runs)} runs, {n_actions}
ground-truth actions, and for each action what every witness recorded.

**Synthetic data, not production-validated.** The environment, its customers
and its tickets are generated from a seed; the agent's actions are real model
output.

Built by `scripts/build_dataset.py` from lacuna commit `{commit}`. The lacuna
code repository is not public yet, so `docs/` ships the decision log and the
pre-registration that the D-numbers on this card refer to. Every result on
this card is read from a file in this dataset at build time.

## What the data is

An agent works a queue of support tickets in a synthetic back office:
refunds, cancellations, customer updates, emails and one report file. The
environment kernel keeps a hidden ledger of every consequential action. That
ledger is the ground truth, and no witness reads it. Each witness writes its
own record of what happened, and capture-recapture turns the overlaps between
those records into an estimate of the actions no witness saw.

- One config per run (`confirm` is the default): one row per episode, with the
  ledger and every witness record.
- `runs`: one row per run, with model, spend, failed episodes, the pooled
  capture-history tables and the caveat that governs the run.
- `results/`: derived results, copied byte for byte: the thinning summaries
  (`results/thinning/*.json`), the confirmatory verdicts
  (`results/hypotheses.json`) and the simulator coverage sweep
  (`results/sim_coverage.json`).
- `environment/`: the agent's system prompt and its tool schemas, verbatim.
- `docs/`: `DECISIONS.md` and `PRE-REGISTRATION.md`, as of the build commit.

| run | phase | model | tickets | episodes | N | W4 recall | spend (USD) |
|---|---|---|---|---|---|---|---|
{run_table}

LLM spend {spend:.2f} USD over the runs with a receipt.{unknown_spend} W4 recall
is the share of ledger actions witness 3 reported, by oracle linkage.
{transcript_note}

## The three witnesses, and why the LLM auditor is not one

| id | witness | what it sees | blind spots |
|---|---|---|---|
| 0 | trace | the agent harness's own log file | lines lost to logging faults |
| 1 | gateway | every tool call through the router | bypassable through code execution |
| 2 | env | end-of-shift state diff plus outbound mail | repeats, reverts, ops outside its view |
| 3 | auditor | an LLM rebuilding actions from the transcript | **not a witness** |

`records` holds witnesses 0-2 only, so a union over it is the floor. Witness 3
is in `auditor_records`, because the runs recorded it, and is excluded from
every estimate in `results/` (D20). An auditor reading the agent's own
transcript is either a copy of the trace or empty. With the full transcript it
reads every call verbatim: W4 recall {recall(full_w4)}. With narration only it
reports nothing: {recall(narr)}. The agent narrates in aggregate ("nine
refunds issued"), and a count cannot be overlapped with another witness's
record. In `confirm` and `scripted-l0`, witness 3 is a scripted stand-in that
copies the transcript with misses.

## Field reference

Episode configs:

| field | type | meaning |
|---|---|---|
| `run` | string | run name; see `runs` |
| `seed` | int | episode seed; the generator regenerates world and tickets from it |
| `model` | string or null | agent model; null for the scripted agent |
| `env` | string | environment; `backoffice` for every run here |
| `tickets` | int | tickets in the queue, plus one closing report task |
| `phase` | string | confirmatory, exploratory, post-hoc supplement, scripted harness check |
| `auditor_view` | string | what witness 3 read: `full` transcript or `narration` only |
| `n_true` | int | ground-truth actions, the ledger length |
| `ledger` | list | ground truth: `id`, `op`, `resource`, `value`, `step` |
| `records` | list | claims by witnesses 0-2: `witness`, `op`, `resource`, `value`, `oracle` |
| `auditor_records` | list | claims by witness 3, same shape, `oracle` always null |
| `agent_read_trace` | bool or null | the agent read W1's trace file; null if no transcript stored |
| `n_calls` | int | tool calls the agent made |
| `n_tool_errors` | int | tool results that were errors |

`record.oracle` is the ledger id when the record came from infrastructure that
can know it. It is for oracle linkage only and is never shown to an
estimator. `send_email` and `write_file` values are content digests.

`runs`: `run`, `phase`, `env`, `agent`, `auditor`, `model`, `auditor_view`,
`tickets`, `episodes`, `episodes_attempted`, `failed_episodes` (seed and the
recorded error, provider request ids redacted), `stopped`, `seed0`, `n_true`,
`recorded_fault_rates`, `spend_usd` (null when no receipt exists),
`llm_calls`, `transcripts_stored`, `transcripts_included`, `w4_recall`,
`oracle_table`, `signature_table` (16 cells, bit w for witness w; cell 0 of
`oracle_table` is the true unseen count), `caveat`.

## Rebuild a capture-history table and rerun an estimator

Run from the dataset root. Standard library only.

```python
{SNIPPET}```

Every run except `pilot-l0` and `scripted-l0` was recorded with complete logs
(`recorded_fault_rates` 0), so the raw histories have nothing hidden and loss
is injected afterwards, as above. `pilot-l0` and `scripted-l0` were thinned
at record time. The log-linear, latent-class and bound estimators behind
`results/` are in the lacuna code, which is not public yet; `results/thinning/`
holds their output over {reps} replicates per configuration.

## Exploratory vs confirmatory

**Confirmatory: `confirm` only.** Pre-registered in `docs/PRE-REGISTRATION.md`
and tested only on loss configurations held out of every exploratory run
(`fin_mid`, `fin_mid_rho100`). The evidence that the registration was frozen
before the run is local git commit `{freeze}`, which cannot be checked from
outside until the code repository is public. {conf["episodes"]} episodes,
N = {conf["n_true"]}, {hyp["linkage"]} linkage. Source: `results/hypotheses.json`.

| hypothesis | verdict | detail |
|---|---|---|
{hyp_table}

At `fin_mid`, {mid_hidden} actions reach no witness on average over {reps}
replicates. `union_count` covers N in {fmt(mid["union_count"]["coverage"])} of
replicates and `ll_bic` in {fmt(mid["ll_bic"]["coverage"])}, mean width
{fmt(mid["ll_bic"]["mean_width"], 1)}. At `fin_mid_rho100` ({rho_hidden} hidden),
with gateway and trace failing together, `ll_independence` covers
{fmt(rho["ll_independence"]["coverage"])} and `ll_bic` {fmt(rho["ll_bic"]["coverage"])}.
Source: `results/thinning/confirm.json`.

**Exploratory: every other LLM run and result.** That includes the `long-*`
thinning results, from which the estimator was chosen (D27), and
`results/thinning/confirm-mnar.json`, built after the confirmatory run (D31).

**Scripted harness check: `scripted-l0`.** No LLM; a check that the harness
and the witnesses work. Its own config only.

## Known caveats

- **Degenerate W4 (D14, D20).** `pilot-l0` and `long-*` used the
  full-transcript auditor, whose recall is 1. Every four-witness number from
  those runs is superseded and supports no claim.
- **Attrition in the confirmatory run (D30, D33).** Failed seeds:
  {"; ".join(f"{f['seed']} ({f['error'].split(':')[0]})" for f in lost)}. The
  defect losses may depend on how the agent reasoned, so which episodes were
  lost is not shown to be random.
- **The MNAR limit (D32).** Every configuration above deletes log lines
  independently of the action. When loss concentrates on the actions no
  witness sees (`fin_mid_mnar`, {mnar_hidden} hidden), `ll_bic` still detects
  the gap in {fmt(mnar["ll_bic"]["detects_gap"])} of replicates and
  `union_count` in {fmt(mnar["union_count"]["detects_gap"])}, but `ll_bic`
  covers ({fmt(mnar["ll_bic"]["coverage"])}) only with a mean width of
  {fmt(mnar["ll_bic"]["mean_width"], 1)} against N = {conf["n_true"]}. With a
  correlated outage as well it covers {fmt(mnar_rho["ll_bic"]["coverage"])}.
  Detection survives; quantification does not.
- **Two agent models (D13).** `pilot-l0` and `long-*` ran on
  deepseek-v4-flash, `narr-30` and `confirm` on deepseek-v4.1-flash.
  Comparisons across them are cross-model.
- **Long queues are abandoned (D18).** Actions per ticket fall as the queue
  grows, so `tickets` overstates the work in `long-60` and `long-120`.
- **Benign loss only (D10, D11).** No adversarial agent was run live; every
  adversarial number is from the simulator (`results/sim_coverage.json`). One
  agent family, one environment.
- **Linkage error is real.** Signature linkage leaves content digests out, so
  two different emails to one address in one episode look like one action seen
  twice. It is measured against oracle linkage, not hidden.

Each run's own caveat is in `runs.caveat`.

## Synthetic data

Every customer, order and ticket is generated from the episode seed: names
are "Customer i", email addresses are at example.com and example.org, phone
numbers start 555, and street names come from a fixed list of six. The
scripted agent deliberately plants wrong-first values with a trailing "x", so
some `scripted-l0` addresses end in example.orgx. No real person, customer,
tenant or work data is in any file. The agent's actions are real output of the
model named in each row, acting on those synthetic tickets. Before any upload,
`scripts/validate_dataset.py` scans every file for credential shapes, email
addresses outside the reserved example domains, IP addresses, local
filesystem paths, provider request ids and the owner's names, and fails if it
finds one.

## Citation

A placeholder until a paper or tagged release exists.

```bibtex
@misc{{lacuna_capture_histories,
  title  = {{lacuna capture histories: witness records for counting unlogged agent actions}},
  note   = {{Hugging Face dataset {REPO}, built from lacuna commit {commit}}},
  year   = {{2026}}
}}
```

Apache-2.0, for everything in this dataset. Transcripts are not included.
"""
    if with_transcripts:
        body = body.replace("Transcripts are not included.", "This build includes transcripts.")
    return front_matter(runs, with_transcripts) + body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--with-transcripts",
        action="store_true",
        help="include transcripts (provider reasoning text); not for release",
    )
    args = ap.parse_args()
    summary = build(with_transcripts=args.with_transcripts)
    for run, n in summary["episodes"].items():
        print(f"  {run:12} {n:4d} episodes")
    if summary["declared_not_on_disk"]:
        print(f"  declared in RUNS, not on disk: {summary['declared_not_on_disk']}")
    print(f"  results: {len(summary['results'])} files")
    print(f"wrote {OUT.relative_to(ROOT)}/{' (with transcripts)' if args.with_transcripts else ''}")


if __name__ == "__main__":
    main()
