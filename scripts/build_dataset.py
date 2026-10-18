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

SUPPLEMENT_PHASE = "post-hoc supplement (not the registered verdict)"
# `excluded` runs are on disk and accounted for, but not shipped as runs.
PHASES = ("confirmatory", "exploratory", SUPPLEMENT_PHASE, "scripted harness check", "excluded")
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
            "not replaced or re-seeded. Two of the three losses were provider 400s "
            "that depend on the agent's trajectory, trigger still open (D34; D33's "
            "mechanism is withdrawn). The registered H1 pass was marginal, and H1 "
            "fails with the lost episodes restored (D35, D36; see the supplement "
            "config). Witness 3 is the scripted stand-in, not an LLM, and is "
            f"excluded from every estimate (D20). Agent deepseek-v4.1-flash. "
            f"{_POSTHOC} Read with D32: under loss that depends on the action, the "
            "method detects incompleteness but cannot measure it."
        ),
    },
    "confirm-supp": {
        "phase": "excluded",
        "env": "backoffice",
        "caveat": (
            "Diagnostic rerun of the three lost confirmatory seeds (D33, D34). Not "
            "shipped as a run; its seed 7022 ships in the supplement config."
        ),
    },
    "diag-7009": {
        "phase": "excluded",
        "env": "backoffice",
        "caveat": (
            "Diagnostic rerun of seed 7009 (D34). Not shipped as a run; its episode "
            "ships in the supplement config."
        ),
    },
    "diag-batch": {
        "phase": "excluded",
        "env": "backoffice",
        "caveat": (
            "Diagnostic batch rerunning the lost and some completed seeds (D34). Not "
            "shipped as a run; its seed 7012 ships in the supplement config."
        ),
    },
    "confirm-plus-supp": {
        "phase": "excluded",
        "env": "backoffice",
        "caveat": (
            "A derived pool assembled from confirm and three rerun episodes, not new "
            "episodes (D35). Its composition is what the supplement config is checked "
            "against."
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

# The supplement config: each lost confirmatory seed's first successful
# completion on the changed client, D34's rule. validate_dataset.py checks it
# against confirm-plus-supp/summary.json's composition.
SUPPLEMENT = {
    "name": "supplement",
    "picks": (("confirm-supp", 7022), ("diag-7009", 7009), ("diag-batch", 7012)),
    "caveat": (
        "The three confirmatory seeds lost to attrition, each rerun until it first "
        "completed (D34's rule), drawn from the diagnostic runs named in "
        "composition. Not part of the registered verdict: it exists to show what "
        "the verdict would be without attrition (D35, D36). Spend is in the "
        "diagnostic runs' receipts and is not split per episode."
    ),
}

# Every artifact under artifacts/ (top level and thinning/) is either shipped
# under results/ or listed here with the reason it is not. A new artifact
# fails the build until it is placed, like an unknown run.
SHIPPED_RESULTS = (
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
)
NOT_SHIPPED = {
    "thinning/long-30.fixed-episodes.json": "byte-identical to thinning/long-30.json",
    "derived.json": "figures derived for the paper, not a measurement",
    "mnar-coverage-split.json": "splits strat_bic's coverage in "
    "thinning/confirm-mnar-hidden.json into refusals and finite intervals, for the paper",
    "thinning/confirm-plus-supp.json": "the 24-episode pool's thinning, summarised in "
    "hypotheses-supplement.json",
    "thinning/var-confirm-s1.json": "a loss-seed rerun summarised in supplement-variance.json",
    "thinning/var-confirm-s2.json": "a loss-seed rerun summarised in supplement-variance.json",
    "thinning/var-confirm-s3.json": "a loss-seed rerun summarised in supplement-variance.json",
    "thinning/var-supp-s1.json": "a loss-seed rerun summarised in supplement-variance.json",
    "thinning/var-supp-s2.json": "a loss-seed rerun summarised in supplement-variance.json",
    "thinning/var-supp-s3.json": "a loss-seed rerun summarised in supplement-variance.json",
    "thinning/confirm-heldout-strat.json": "not in this release",
    "thinning/long-30-strat.json": "not in this release",
    "thinning/mnar-diagnostics.json": "not in this release",
}
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


def _raw_episodes(run: str, src: Path) -> list[dict]:
    raw = [json.loads(line) for line in (src / "episodes.jsonl").open() if line.strip()]
    for e in raw:
        extra = set(e) - EPISODE_KEYS
        if extra:
            raise BuildError(f"{run} seed {e.get('seed')}: unknown episode keys {sorted(extra)}")
    return raw


def episode_row(run: str, e: dict, plan: dict, phase: str, with_transcripts: bool) -> dict:
    t = e.get("transcript")
    row = {
        "run": run,
        "seed": e["seed"],
        "model": plan["summary"]["model"],
        "env": plan["meta"]["env"],
        "tickets": plan["fields"]["tickets"],
        "phase": phase,
        "auditor_view": plan["fields"]["auditor_view"],
        "n_true": len(e["ledger"]),
        "ledger": e["ledger"],
        # The three hard witnesses only, so a naive union over records is the
        # floor. The auditor equals the ledger when it reads the full
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
    return row


def _tables(episodes: list[dict]) -> tuple[list[int], list[int]]:
    """Pooled oracle and signature tables over all four witnesses, as run_episodes.py does."""
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


def build_run(run: str, src: Path, plan: dict, with_transcripts: bool) -> tuple[list[dict], dict]:
    meta, summary, fields, receipt = plan["meta"], plan["summary"], plan["fields"], plan["receipt"]
    raw = _raw_episodes(run, src)
    rows = [episode_row(run, e, plan, meta["phase"], with_transcripts) for e in raw]

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
        "composition": None,
    }
    return rows, run_row


def build_supplement(plans: dict[str, dict], with_transcripts: bool) -> tuple[list[dict], dict]:
    """The supplement config: one picked episode from each named diagnostic run."""
    raw, rows = [], []
    for run, seed in SUPPLEMENT["picks"]:
        if run not in plans:
            raise BuildError(f"supplement needs {run}, which is not on disk")
        matches = [e for e in _raw_episodes(run, ARTIFACTS / "episodes" / run) if e["seed"] == seed]
        if len(matches) != 1:
            raise BuildError(f"supplement: {run} has {len(matches)} episodes with seed {seed}")
        raw.append(matches[0])
        rows.append(episode_row(run, matches[0], plans[run], SUPPLEMENT_PHASE, with_transcripts))
    shared = {}
    for key in ("model", "agent", "auditor", "fault_rates"):
        values = {json.dumps(plans[run]["summary"][key]) for run, _ in SUPPLEMENT["picks"]}
        if len(values) != 1:
            raise BuildError(f"supplement: source runs disagree on {key}: {sorted(values)}")
        shared[key] = plans[SUPPLEMENT["picks"][0][0]]["summary"][key]
    for key in ("tickets", "auditor_view", "env"):
        values = {rows[i][key] for i in range(len(rows))}
        if len(values) != 1:
            raise BuildError(f"supplement: source runs disagree on {key}: {sorted(values)}")
    oracle, sig = _tables(raw)
    run_row = {
        "run": SUPPLEMENT["name"],
        "phase": SUPPLEMENT_PHASE,
        "env": rows[0]["env"],
        "agent": shared["agent"],
        "auditor": shared["auditor"],
        "model": shared["model"],
        "auditor_view": rows[0]["auditor_view"],
        "tickets": rows[0]["tickets"],
        "episodes": len(rows),
        "episodes_attempted": None,
        "failed_episodes": [],
        "stopped": None,
        "seed0": None,
        "n_true": sum(r["n_true"] for r in rows),
        "recorded_fault_rates": shared["fault_rates"],
        "spend_usd": None,
        "llm_calls": None,
        "transcripts_stored": all(e.get("transcript") is not None for e in raw),
        "transcripts_included": with_transcripts,
        "w4_recall": _w4_recall(raw),
        "oracle_table": oracle,
        "signature_table": sig,
        "caveat": SUPPLEMENT["caveat"],
        "composition": [{"run": run, "seed": seed} for run, seed in SUPPLEMENT["picks"]],
    }
    return rows, run_row


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def result_artifacts(artifacts: Path) -> list[str]:
    """Every result artifact, as a path relative to artifacts/."""
    top = [p.name for p in sorted(artifacts.glob("*.json"))]
    return top + [f"thinning/{p.name}" for p in sorted((artifacts / "thinning").glob("*.json"))]


def check_results_accounted(artifacts: Path) -> None:
    """Every artifact is shipped or excluded with a reason; a new one fails the build."""
    found = set(result_artifacts(artifacts))
    unplaced = found - set(SHIPPED_RESULTS) - set(NOT_SHIPPED)
    if unplaced:
        raise BuildError(
            f"unplaced artifacts {sorted(unplaced)}: add to SHIPPED_RESULTS or NOT_SHIPPED"
        )
    missing = set(SHIPPED_RESULTS) - found
    if missing:
        raise BuildError(f"SHIPPED_RESULTS names missing artifacts {sorted(missing)}")
    dup, twin = "thinning/long-30.fixed-episodes.json", "thinning/long-30.json"
    if dup in found and (artifacts / dup).read_bytes() != (artifacts / twin).read_bytes():
        raise BuildError(f"{dup} is no longer identical to {twin}; ship it")


def _copy_results(out: Path) -> list[str]:
    for name in SHIPPED_RESULTS:
        dst = out / "results" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ARTIFACTS / name, dst)
    return [f"results/{name}" for name in SHIPPED_RESULTS]


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
    for run in on_disk:
        run_meta(run)  # an unknown run fails before anything is read or written
    # Excluded runs are planned only when the supplement draws from them; a
    # derived pool such as confirm-plus-supp has no run fields to plan.
    sources = {run for run, _ in SUPPLEMENT["picks"]}
    plans = {
        run: plan_run(run, episodes_dir / run)
        for run in on_disk
        if RUNS[run]["phase"] != "excluded" or run in sources
    }
    check_results_accounted(ARTIFACTS)
    order = [r for r in RUNS if r in on_disk and RUNS[r]["phase"] != "excluded"]
    excluded = [r for r in RUNS if r in on_disk and RUNS[r]["phase"] == "excluded"]

    tmp = Path(tempfile.mkdtemp(prefix=".dataset-build-", dir=out.parent))
    try:
        runs, all_rows = [], {}
        for run in order:
            rows, run_row = build_run(run, episodes_dir / run, plans[run], with_transcripts)
            _dump_jsonl(tmp / "episodes" / f"{run}.jsonl", rows)
            runs.append(run_row)
            all_rows[run] = rows
            if run == "confirm":
                rows, run_row = build_supplement(plans, with_transcripts)
                _dump_jsonl(tmp / "episodes" / f"{SUPPLEMENT['name']}.jsonl", rows)
                runs.append(run_row)
                all_rows[SUPPLEMENT["name"]] = rows
        _dump_jsonl(tmp / "runs.jsonl", runs)
        results = _copy_results(tmp)
        _write_environment(tmp)
        (tmp / "docs").mkdir()
        for name in SHIPPED_DOCS:
            shutil.copyfile(DOCS / name, tmp / "docs" / name)
        (tmp / ".gitattributes").write_text("*.jsonl filter=lfs diff=lfs merge=lfs -text\n")
        (tmp / "README.md").write_text(
            card(runs, all_rows, tmp, git_commit(), with_transcripts, excluded)
        )
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
        "excluded": excluded,
        "declared_not_on_disk": [r for r in RUNS if r not in on_disk],
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
        ("composition", {"list": _features([("run", _STR), ("seed", _INT)])}),
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


def _keyed(d: dict, word: str) -> dict:
    """The one entry of `d` whose key contains `word`."""
    hits = [v for k, v in d.items() if word in k]
    if len(hits) != 1:
        raise BuildError(f"expected one key containing {word!r}, found {len(hits)}")
    return hits[0]


def card(
    runs: list[dict],
    rows: dict[str, list[dict]],
    out: Path,
    commit: str,
    with_transcripts: bool,
    excluded: list[str] = (),
) -> str:
    by = {r["run"]: r for r in runs}
    conf = by["confirm"]
    supp = by[SUPPLEMENT["name"]]
    hyp = json.loads((out / "results" / "hypotheses.json").read_text())
    hyp_supp = json.loads((out / "results" / "hypotheses-supplement.json").read_text())
    variance = json.loads((out / "results" / "supplement-variance.json").read_text())
    registered, restored = _keyed(variance, "registered"), _keyed(variance, "supplement")
    h1_supp = hyp_supp["hypotheses"]["H1 recovery"]
    thin_dir = out / "results" / "thinning"
    strat, _ = _est(thin_dir / "confirm-mnar-strat.json", "fin_mid_mnar/signature")
    hidden, _ = _est(thin_dir / "confirm-mnar-hidden.json", "full_mid_hidden/signature")
    fp_sig, _ = _est(thin_dir / "confirm-falsepos.json", "anti_fin/signature")
    fp_orc, _ = _est(thin_dir / "confirm-falsepos.json", "anti_fin/oracle")
    reps = json.loads((thin_dir / "confirm.json").read_text())["reps"]
    mid, mid_hidden = _est(thin_dir / "confirm.json", "fin_mid/signature")
    rho, rho_hidden = _est(thin_dir / "confirm.json", "fin_mid_rho100/signature")
    mnar, mnar_hidden = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar/signature")
    mnar_rho, _ = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar_rho100/signature")
    real_runs = [r for r in runs if r["run"] != SUPPLEMENT["name"]]
    n_eps = sum(r["episodes"] for r in real_runs)
    n_actions = sum(r["n_true"] for r in real_runs)
    spend, no_receipt = spend_summary(real_runs)
    full_w4 = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "full"]
    narr = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "narration"]
    lost = conf["failed_episodes"]
    n400 = sum(" 400 " in f["error"] for f in lost)
    others = sorted({f["error"].split(":")[0] for f in lost if " 400 " not in f["error"]})
    read = [r["agent_read_trace"] for r in rows["confirm"]]
    with_tx = [r["run"] for r in runs if r["transcripts_stored"] and not r["transcripts_included"]]
    seeds = ", ".join(str(s) for s in registered["seeds"])
    freeze = freeze_commit()
    picks = ", ".join(f"{c['seed']} from `{c['run']}`" for c in supp["composition"])

    def covs(entry: dict) -> str:
        return ", ".join(f"{c:.2f}" for c in entry["coverage"])

    def fmt(v: float | None, p: int = 2) -> str:
        return "unbounded" if v is None else f"{v:.{p}f}"

    def recall(rs: list[dict]) -> str:
        return ", ".join(f"{r['w4_recall']:.4f} in `{r['run']}`" for r in rs)

    run_table = "\n".join(
        f"| `{r['run']}` | {r['phase']} | {r['model'] or 'scripted'} | {r['tickets']} | "
        f"{r['episodes']} | {r['n_true']} | {r['w4_recall']:.4f} | {fmt_usd(r['spend_usd'])} |"
        for r in runs
    )

    def verdicts(h: dict) -> str:
        return "\n".join(
            f"| {name} | {'PASS' if v['passed'] else 'FAIL'} | {v['detail']} |"
            for name, v in h["hypotheses"].items()
        )

    hyp_table, supp_table = verdicts(hyp), verdicts(hyp_supp)
    unknown_spend = (
        f" Spend is unknown for {', '.join(f'`{r}`' for r in no_receipt)}, which has no "
        "receipt, and is not counted as zero."
        if no_receipt
        else ""
    )
    excluded_note = (
        f"Not shipped as runs: {', '.join(f'`{r}`' for r in excluded)} (diagnostic reruns "
        "and a derived pool; each reason is in `RUNS` in the builder). The three episodes "
        "the supplement needs are drawn from them."
        if excluded
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

> **Read before any result below (D36).**
> - **The registered H1 pass was marginal.** Over loss seeds {seeds}, `ll_bic`
>   covers {covs(registered)}, mean {registered["mean_coverage"]:.3f}. Source:
>   `results/supplement-variance.json`.
> - **With the lost episodes restored, H1 {"fails" if not h1_supp["passed"] else "passes"}.**
>   The confirmatory run completed {conf["episodes"]} of {conf["episodes_attempted"]}
>   episodes. Restoring the {supp["episodes"]} lost ones (`supplement` config), mean
>   coverage over the same loss seeds is {restored["mean_coverage"]:.3f}
>   ({covs(restored)}). At the registered loss seed the artifact records H1 as
>   {"PASS" if h1_supp["passed"] else "FAIL"}, testing "{h1_supp["detail"]}". Source:
>   `results/hypotheses-supplement.json`. The registered verdict stands on its
>   {conf["episodes"]} episodes; the supplement is not part of it.
> - **The losses are not explained.** {n400} of the {len(lost)} were provider 400s
>   that depend on the agent's trajectory, trigger still open (D34); the rest:
>   {", ".join(others) or "none"}. D33's empty-string mechanism is withdrawn.
> - **Every coverage figure is under simulated loss.** Loss was injected post hoc into
>   complete logs. No figure here measures real log loss.
> - **The agent can read W1's trace file, though not write it**, and did so in
>   {sum(bool(x) for x in read)} of {len(read)} `confirm` episodes (`agent_read_trace`).

How many consequential actions did an agent take that no log line shows? This
dataset holds the witness records lacuna uses to answer that by
capture-recapture: {n_eps} episodes across {len(real_runs)} runs, {n_actions}
ground-truth actions, and for each action what every witness recorded, plus
a {supp["episodes"]}-episode `supplement` of rerun confirmatory seeds.

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
- `supplement`: the {supp["episodes"]} lost confirmatory seeds, rerun ({picks}).
  A post-hoc supplement, not the registered verdict.
- `runs`: one row per run and one for `supplement`, with model, spend, failed
  episodes, the pooled capture-history tables and the caveat that governs it.
- `results/`: derived results, copied byte for byte: the thinning summaries
  (`results/thinning/*.json`), the registered and supplement verdicts
  (`results/hypotheses.json`, `results/hypotheses-supplement.json`), the
  loss-seed variance (`results/supplement-variance.json`) and the simulator
  coverage sweep (`results/sim_coverage.json`).
- `environment/`: the agent's system prompt and its tool schemas, verbatim.
- `docs/`: `DECISIONS.md` and `PRE-REGISTRATION.md`, as of the build commit.

| run | phase | model | tickets | episodes | N | W4 recall | spend (USD) |
|---|---|---|---|---|---|---|---|
{run_table}

LLM spend {spend:.2f} USD over the runs with a receipt.{unknown_spend} The
supplement's spend sits in the diagnostic runs' receipts and is not split per
episode. W4 recall is the share of ledger actions witness 3 reported, by
oracle linkage. {transcript_note} {excluded_note}

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

**Post-hoc supplement: `supplement`.** The same analysis on the
{hyp_supp["episodes"]}-episode pool with the lost seeds restored, N =
{hyp_supp["n_true"]} (D35, D36). Not part of the registered verdict. Source:
`results/hypotheses-supplement.json`.

| hypothesis | verdict | detail |
|---|---|---|
{supp_table}

**Exploratory: every other LLM run and result.** That includes the `long-*`
thinning results, from which the estimator was chosen (D27), and the MNAR and
false-positive results built after the confirmatory run (D31, D37-D42).

**Scripted harness check: `scripted-l0`.** No LLM; a check that the harness
and the witnesses work. Its own config only.

## Known caveats

- **Degenerate W4 (D14, D20).** `pilot-l0` and `long-*` used the
  full-transcript auditor, whose recall is 1. Every four-witness number from
  those runs is superseded and supports no claim.
- **Attrition in the confirmatory run (D30, D34-D36).** Failed seeds:
  {"; ".join(f"{f['seed']} ({f['error'].split(':')[0]})" for f in lost)}. The
  400s depend on the agent's trajectory, so which episodes were lost is not
  shown to be random, and restoring them changes H1.
- **The MNAR limit (D32).** Every configuration above deletes log lines
  independently of the action. When loss concentrates on the actions no
  witness sees (`fin_mid_mnar`, {mnar_hidden} hidden), `ll_bic` still detects
  the gap in {fmt(mnar["ll_bic"]["detects_gap"])} of replicates and
  `union_count` in {fmt(mnar["union_count"]["detects_gap"])}, but `ll_bic`
  covers ({fmt(mnar["ll_bic"]["coverage"])}) only with a mean width of
  {fmt(mnar["ll_bic"]["mean_width"], 1)} against N = {conf["n_true"]}. With a
  correlated outage as well it covers {fmt(mnar_rho["ll_bic"]["coverage"])}.
  Detection survives; quantification does not.
- **Stratifying refuses rather than answers wrongly, until the key is hidden
  (D37-D40).** At `fin_mid_mnar`, where the loss depends on the operation,
  `strat_bic` is unbounded in {fmt(strat["strat_bic"]["unbounded_rate"])} of
  replicates. When the loss is keyed on something no record shows
  (`full_mid_hidden`) it covers {fmt(hidden["strat_bic"]["coverage"])} and
  `ll_bic` {fmt(hidden["ll_bic"]["coverage"])}. Sources:
  `results/thinning/confirm-mnar-strat.json`, `confirm-mnar-hidden.json`.
- **Gap detection can be falsely positive (D41, D42).** When the logs
  specialise (`anti_fin`, true hidden count 0) and records are matched by
  signature, `ll_bic` claims a gap in {fmt(fp_sig["ll_bic"]["detects_gap"])} of
  replicates; under oracle linkage, {fmt(fp_orc["ll_bic"]["detects_gap"])}.
  Source: `results/thinning/confirm-falsepos.json`.
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
    if summary["excluded"]:
        print(f"  excluded (accounted, not shipped as runs): {summary['excluded']}")
    if summary["declared_not_on_disk"]:
        print(f"  declared in RUNS, not on disk: {summary['declared_not_on_disk']}")
    print(f"  results: {len(summary['results'])} files")
    print(f"wrote {OUT.relative_to(ROOT)}/{' (with transcripts)' if args.with_transcripts else ''}")


if __name__ == "__main__":
    main()
