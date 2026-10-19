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
Apache-2.0 cannot license, and no estimate depends on them. A
--with-transcripts build exists for a later release; the validator and the
publisher both refuse it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
OUT = ROOT / "dataset"
REPO = "caiotheodoro/lacuna-capture-histories"

SUPPLEMENT_PHASE = "post-hoc supplement (not the registered verdict)"
# `excluded` runs are on disk and accounted for, but not shipped as runs.
TRANSFER_PHASE = "confirmatory (transfer, separately pre-registered)"
PHASES = (
    "confirmatory",
    TRANSFER_PHASE,
    "exploratory",
    SUPPLEMENT_PHASE,
    "scripted harness check",
    "excluded",
)
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
            "config). At 3000 replicates the shortfall is signature linkage, not the "
            "estimator (D44); linking by the shared tool-call id restores nominal "
            "coverage under the financial view, where the env diff is trivial to attach "
            "(D49, D50). Witness 3 is the scripted stand-in, not an LLM, and is "
            f"excluded from every estimate (D20). Agent deepseek-v4.1-flash. "
            f"{_POSTHOC} Read with D32: under loss that depends on the action, the "
            "method detects incompleteness but cannot measure it."
        ),
    },
    "confirm-supp": {
        "phase": "excluded",
        "env": "backoffice",
        "caveat": (
            "The supplement's rerun of the three lost confirmatory seeds on the changed "
            "client (D33, D34); its spend counts as study spend. Not shipped as a run; "
            "its seed 7022 ships in the supplement config."
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
    "transfer": {
        "phase": TRANSFER_PHASE,
        "env": "infra",
        "caveat": (
            "The transfer study: an on-call operator in the infra environment (D45), "
            "pre-registered separately in docs/PREREG-TRANSFER.md. {episodes} of "
            "{attempted} episodes completed; the failed seed is in failed_episodes and "
            "was not replaced. T1 fails, and the failure is the linkage (D47); a false "
            "gap needs a W3 blind spot (D48); the verdict depends on density (D46). "
            "Linking by the tool-call id narrows the shortfall but cannot close it, "
            "because the env diff has no id (D49); plain call-id linkage is the "
            "auditor-computable result, and a last-writer rule was withdrawn (D52). "
            "Witness 3 is the scripted stand-in and "
            f"is excluded from every estimate (D20). Agent deepseek-v4.1-flash. {_POSTHOC}"
        ),
    },
    "transfer-scripted": {
        "phase": "scripted harness check",
        "env": "infra",
        "caveat": (
            "Scripted agent and auditor in the infra environment, no LLM: the "
            "three-episode sanity check the transfer pre-registration names, run before "
            f"any paid episode (D45). {_POSTHOC}"
        ),
    },
    "transfer-scripted-24": {
        "phase": "scripted harness check",
        "env": "infra",
        "caveat": (
            "Scripted agent and auditor in the infra environment, no LLM: the "
            "24-episode pipeline check that predicted the transfer failure before the "
            f"paid run (D45, D46). {_POSTHOC}"
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
        "completed (D34's rule), drawn from the reruns named in composition. Not "
        "part of the registered verdict: it exists to show what the verdict would "
        "be without attrition (D35, D36). Spend is in those reruns' receipts and is "
        "not split per episode."
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
    # D44: the review round's checks, which change the headline.
    "robustness/precision.json",
    "robustness/mnar-dose.json",
    "robustness/falsegap-battery.json",
    "robustness/collisions.json",
    # D49: linkage arms and burst loss.
    "robustness/linkage-arms.json",
    "robustness/burst-loss.json",
    # D50: the quasi-Poisson interval under bursts, coupling and MNAR.
    "robustness/overdispersion.json",
    # D51: dispersion from replication across episodes.
    "robustness/episode-dispersion.json",
    # D52: keep BIC's model and widen only.
    "robustness/interval-comparison.json",
    # D53: phase structure, whole-session outages, shared session quality.
    "robustness/session-and-phase.json",
    "robustness/op-position.json",
    # D54: the final controls on bursts, outages and tier_bic.
    "robustness/final-controls.json",
    # The transfer study (D45-D48): its verdicts, thinning and robustness checks.
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
SHIPPED_DOCS = ("DECISIONS.md", "PRE-REGISTRATION.md", "PREREG-TRANSFER.md", "TRANSFER.md")
# Each pre-registration, by the doc that holds it.
PREREGISTRATIONS = ("PRE-REGISTRATION.md", "PREREG-TRANSFER.md")
# Diagnostic reruns, whose spend the card reports apart from the study's.
DIAGNOSTIC_RUNS = ("diag-7009", "diag-batch")
EPISODE_KEYS = {"seed", "ledger", "records", "n_calls", "n_tool_errors", "transcript"}
TRACE_PATH = "/var/log/agent_trace.jsonl"  # lacuna.envs.kernel.TRACE_PATH; checked below
_REQUEST_ID = re.compile(r"(request[_ -]?id\W{0,3})[0-9A-Fa-f][0-9A-Fa-f-]*", re.I)

SNIPPET = """\
import json
import random
from collections import Counter

rows = [json.loads(line) for line in open("episodes/confirm.jsonl")]

# Capture histories by oracle linkage: bit w is set if witness w saw the
# action (0 = W1 trace, 1 = W2 gateway, 2 = W3 env). One draw of the held-out
# fin_mid loss: each pipe drops half its lines and the env diff sees
# financial ops only. Witness 3 (W4) is in auditor_records, not records.
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


def receipt_spend(artifacts: Path, excluded: list[str]) -> dict:
    """Spend from every receipt on disk, shipped or not, split study / diagnostic.

    `confirm-supp` is the supplement's rerun of the lost seeds (D34), so it is
    study spend even though the run is not shipped; `study_unshipped` names
    every such run so the card can say so.
    """
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
        "study_unshipped": sorted(r for r in per_run if r in excluded and r not in DIAGNOSTIC_RUNS),
    }


def run_snippet(folder: Path) -> str:
    """What the card's snippet prints on this build, isolated from the repo's code."""
    proc = subprocess.run(
        [sys.executable, "-I", "-c", SNIPPET],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise BuildError(f"the card's snippet fails on this build: {proc.stderr.strip()[-200:]}")
    return proc.stdout.strip()


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


def freeze_commit(doc: str = "PRE-REGISTRATION.md") -> str:
    """The commit that last touched a pre-registration, which is its freeze."""
    try:
        return subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", f"docs/{doc}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def freeze_commits() -> dict[str, str]:
    return {doc: freeze_commit(doc) for doc in PREREGISTRATIONS}


def plan_run(run: str, src: Path) -> dict:
    """Everything a run needs, read and checked before any output is written."""
    meta = run_meta(run)
    summary = json.loads((src / "summary.json").read_text())
    recorded_env = summary.get("env", "backoffice")
    if recorded_env != meta["env"]:
        raise BuildError(f"{run}: summary says env {recorded_env!r}, RUNS says {meta['env']!r}")
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
        # Fields the summary did not record, filled from RUNS and checked
        # against the ledgers by validate_dataset.py.
        "inferred_fields": [k for k in ("tickets", "auditor_view") if summary.get(k) is None],
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
        "inferred_fields": [],
    }
    return rows, run_row


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def result_artifacts(artifacts: Path) -> list[str]:
    """Every file under artifacts/, at any depth, outside the run directories.

    Runs are placed by RUNS; everything else is a result and must be placed
    here, whatever directory it lands in. Local logs (artifacts/*.log) are
    gitignored and not results.
    """
    return sorted(
        p.relative_to(artifacts).as_posix()
        for p in artifacts.rglob("*")
        if p.is_file() and p.relative_to(artifacts).parts[0] != "episodes" and p.suffix != ".log"
    )


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


def _write_environment(out: Path, envs: list[str]) -> None:
    """Each shipped environment's agent prompt and tool schemas, from the env registry."""
    from lacuna.envs.kernel import TRACE_PATH as KERNEL_TRACE
    from lacuna.envs.registry import ENVS as REGISTRY

    if KERNEL_TRACE != TRACE_PATH:
        raise BuildError(f"trace path moved to {KERNEL_TRACE}; update TRACE_PATH")
    for name in envs:
        folder = out / "environment" / name
        folder.mkdir(parents=True)
        (folder / "system_prompt.txt").write_text(REGISTRY[name].agent_system + "\n")
        (folder / "tools.json").write_text(json.dumps(REGISTRY[name].tools, indent=2) + "\n")


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
        _write_environment(tmp, sorted({r["env"] for r in runs}))
        (tmp / "docs").mkdir()
        for name in SHIPPED_DOCS:
            shutil.copyfile(DOCS / name, tmp / "docs" / name)
        (tmp / ".gitattributes").write_text("*.jsonl filter=lfs diff=lfs merge=lfs -text\n")
        (tmp / "README.md").write_text(
            card(
                runs,
                all_rows,
                tmp,
                git_commit(),
                with_transcripts,
                excluded,
                receipt_spend(ARTIFACTS, excluded),
                run_snippet(tmp),
                freeze_commits(),
            )
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
        ("inferred_fields", {"sequence": "string"}),
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


def _robustness(folder: Path, bar: float) -> dict:
    """The D44 figures the card quotes, each read from results/robustness/."""
    prec = json.loads((folder / "precision.json").read_text())

    def ll_bic(pool: str, key: str) -> dict:
        return prec["runs"][pool][key]["ll_bic"]

    reg_sig, reg_orc = ll_bic("confirm", "fin_mid/signature"), ll_bic("confirm", "fin_mid/oracle")
    res_sig = ll_bic("confirm-plus-supp", "fin_mid/signature")
    res_orc = ll_bic("confirm-plus-supp", "fin_mid/oracle")
    selected = prec["runs"]["confirm"]

    coll = json.loads((folder / "collisions.json").read_text())["confirm"]

    dose = json.loads((folder / "mnar-dose.json").read_text())["op_keyed"]
    by_dose: dict[str, dict] = {}
    for key, entry in dose.items():
        level, linkage = key.split("/")
        by_dose.setdefault(level, {})[linkage] = entry
    levels = sorted(by_dose, key=float)

    def below(level: str) -> bool:
        return all(e["ll_bic"]["coverage"] < bar for e in by_dose[level].values())

    def above(level: str) -> bool:
        return all(e["ll_bic"]["coverage"] >= bar for e in by_dose[level].values())

    first = next(lv for lv in levels if float(lv) > 1 and below(lv))
    recover = next(lv for lv in levels if float(lv) > float(first) and above(lv))
    strat_cov = [e["strat_bic"]["coverage"] for lv in levels for e in by_dose[lv].values()]

    battery = json.loads((folder / "falsegap-battery.json").read_text())
    inside, outside, rates = [], [], set()
    for pool in battery["pools"].values():
        for key, entry in pool["configs"].items():
            gap = entry["ll_bic"]["detects_gap"]
            if key.startswith("anti"):
                rates.add(key[len("anti") :].split("_")[0])
            if key.startswith("anti") and key.endswith("_financial/signature"):
                inside.append(gap)
            else:
                outside.append(gap)
    return {
        "reps": reg_sig["reps"],
        "seeds": reg_sig["reps"] // prec["reps_per_seed"],
        "reg_sig": reg_sig,
        "reg_orc": reg_orc,
        "res_sig": res_sig,
        "res_orc": res_orc,
        "w1w2_sig": selected["fin_mid/signature"]["ll_bic_selected"]["W1W2"]["share"],
        "w1w2_orc": selected["fin_mid/oracle"]["ll_bic_selected"]["W1W2"]["share"],
        "coll": coll,
        "dose_first": first,
        "dose_recover": recover,
        "dose_mcar": by_dose["1"],
        "dose_at_first": by_dose[first],
        "dose_at_recover": by_dose[recover],
        "strat_min": min(strat_cov),
        "strat_max": max(strat_cov),
        "strat_refuse_low": by_dose[levels[0]]["oracle"]["strat_bic"]["unbounded_rate"],
        "strat_refuse_high": by_dose[levels[-1]]["oracle"]["strat_bic"]["unbounded_rate"],
        "fg_pools": len(battery["pools"]),
        "fg_rates": sorted(rates, key=float),
        "fg_inside_min": min(inside),
        "fg_inside_max": max(inside),
        "fg_outside_max": max(outside),
    }


def _transfer(out: Path, run: dict) -> dict:
    """The transfer study's figures, each read from the shipped results (D45-D48)."""
    results = out / "results"
    rob = results / "robustness"
    hyp = json.loads((results / "transfer_hypotheses.json").read_text())
    prec_all = json.loads((rob / "transfer-precision.json").read_text())
    prec = prec_all["runs"]["transfer"]
    sig, orc = prec["transfer_mid/signature"]["ll_bic"], prec["transfer_mid/oracle"]["ll_bic"]
    gap = json.loads((rob / "transfer-falsegap.json").read_text())["runs"]

    def gap_rate(run_key: str, variant: str) -> float:
        v = gap[run_key]["variants"][variant]
        return v.get("ll_bic", v)["detects_gap"]

    counterfactual = next(k for k in gap if k.startswith("transfer,"))
    mech_all = json.loads((rob / "transfer-mechanism.json").read_text())
    mech = mech_all["runs"]["transfer"]["variants"]
    restore = next(k for k in mech if k.startswith("exact for all-W3-invisible and mixed"))
    density = json.loads((rob / "transfer-density.json").read_text())
    cells = density["cells"]
    best = max(cells.values(), key=lambda c: c["signature"]["coverage"])
    worst = min(cells.values(), key=lambda c: c["signature"]["coverage"])
    behaviour = json.loads((rob / "transfer-behaviour.json").read_text())["runs"]
    thin = json.loads((results / "thinning" / "transfer.json").read_text())["configs"]
    mid, rho = thin["transfer_mid/signature"], thin["transfer_mid_rho100/signature"]
    mid_cfg = mid["config"]
    hyps = hyp["hypotheses"]
    prereg = (out / "docs" / "PREREG-TRANSFER.md").read_text()
    predicted = re.search(r"^(\d+)\. \*\*T1 fails\.\*\*", prereg, re.M)
    return {
        "hyp": hyp,
        "t": {k.split()[0]: v for k, v in hyps.items()},
        "t1_frozen": float(re.search(r"coverage ([0-9.]+)", hyps[next(iter(hyps))]["detail"])[1]),
        "sig": sig,
        "orc": orc,
        "seeds": sig["reps"] // prec_all["reps_per_seed"],
        "gap_sig": gap_rate("transfer", "signature"),
        "gap_orc": gap_rate("transfer", "exact (oracle)"),
        "gap_w3_complete": gap_rate(counterfactual, "signature"),
        "mid_cov": mid["estimators"]["ll_bic"]["coverage"],
        "mid_width": mid["estimators"]["ll_bic"]["mean_width"],
        "mid_hidden": mid["mean_unseen_by_w1w2w3"],
        "mid_cfg": mid_cfg,
        "rho_cov": rho["estimators"]["ll_bic"]["coverage"],
        "rho_width": rho["estimators"]["ll_bic"]["mean_width"],
        "rho_hidden": rho["mean_unseen_by_w1w2w3"],
        "mech_sig": mech["signature"]["coverage"],
        "mech_restore": mech[restore]["coverage"],
        "mech_exact": mech["exact (oracle)"]["coverage"],
        "mech_seeds": mech["signature"]["reps"] // mech_all["reps_per_seed"],
        "cells": len(cells),
        "cell_eps": density["episodes_per_cell"],
        "cell_seed0": density["episode_seed0"],
        "best": best,
        "worst": worst,
        "orc_min": min(c["oracle"]["coverage"] for c in cells.values()),
        "orc_max": max(c["oracle"]["coverage"] for c in cells.values()),
        "paid": behaviour["transfer"],
        "free": behaviour["transfer-scripted-24"],
        "predicted": predicted[1] if predicted else None,
        "prereg_cites_d34": "D34" in prereg,
        "run": run,
    }


def _linkage(folder: Path) -> dict:
    """The D49 figures: linkage arms and burst loss, read from results/robustness/."""
    arms_all = json.loads((folder / "linkage-arms.json").read_text())
    cells = arms_all["cells"]

    def cov(run: str, cfg: str, arm: str) -> float:
        return cells[f"{run}/{cfg}/{arm}"]["ll_bic"]["coverage"]

    # The matching arms; the last-writer rule (D50) is reported on its own.
    auditor_arms = [a for a in arms_all["arms"] if a not in ("oracle", "callid_lastwriter")]
    coupled = sorted({cov("transfer", "transfer_mid_rho100", a) for a in auditor_arms})
    anti = [(k.split("/"), c["ll_bic"]["detects_gap"]) for k, c in cells.items() if "/anti" in k]
    # Plain call-id arms; the withdrawn last-writer rule is reported on its own (D52).
    callid_gaps = [g for (run, cfg, arm), g in anti if arm in ("callid", "callid_digest")]
    lastwriter_gaps = [g for (run, cfg, arm), g in anti if arm == "callid_lastwriter"]
    signature_gaps = [g for (run, cfg, arm), g in anti if arm == "signature"]
    anti_envs = sorted({run for (run, cfg, arm), _ in anti})

    burst_all = json.loads((folder / "burst-loss.json").read_text())

    def burst(prefix: str) -> list[dict]:
        hits = [c for k, c in burst_all["cells"].items() if k.startswith(prefix)]
        return sorted(hits, key=lambda c: c["width"])

    bo, infra = burst("confirm/financial/indep/"), burst("transfer/full/indep/")

    od = json.loads((folder / "overdispersion.json").read_text())["cells"]
    pools = (("confirm", "financial"), ("long-30", "financial"), ("transfer", "full"))
    w12 = [od[f"burst/{run}/{view}/indep/w12"] for run, view in pools]
    coupled_od = od["heldout/confirm/fin_mid_rho100/oracle"]
    coupled_infra_od = od["heldout/transfer/transfer_mid_rho100/oracle"]
    joint = od["burst/confirm/financial/joint/w12"]

    ep = json.loads((folder / "episode-dispersion.json").read_text())["cells"]

    return {
        "od_ind_qp_signature": ep["burst/transfer/full/indep/w12/signature"]["ll_independence_qp"][
            "coverage"
        ],
        "od_phi_coupled_infra": coupled_infra_od["mean_phi"],
        "od_coupled_infra_bic_qp": coupled_infra_od["ll_bic_qp"]["coverage"],
        "od_phi_joint": joint["mean_phi"],
        "coupled_auditor_max": max(
            cov("transfer", "transfer_mid_rho100", a) for a in arms_all["arms"] if a != "oracle"
        ),
        "od_pools": [run for run, _ in pools],
        "od_reps": w12[0]["ll_bic"]["reps"],
        "od_ind_qp": [c["ll_independence_qp"]["coverage"] for c in w12],
        "od_ind": [c["ll_independence"]["coverage"] for c in w12],
        "od_bic_qp": [c["ll_bic_qp"]["coverage"] for c in w12],
        "od_coupled_ind_qp_width": coupled_od["ll_independence_qp"]["mean_width"],
        "od_coupled_bic_qp": coupled_od["ll_bic_qp"]["coverage"],
        "od_joint_bic_qp": joint["ll_bic_qp"]["coverage"],
        "od_joint_ind_qp_width": joint["ll_independence_qp"]["mean_width"],
        "bo_full": {
            a: cov("confirm", "full_mid", a) for a in ("callid", "callid_lastwriter", "oracle")
        },
        "infra_lastwriter": cov("transfer", "transfer_mid", "callid_lastwriter"),
        "coupled_lastwriter": cov("transfer", "transfer_mid_rho100", "callid_lastwriter"),
        "reps": cells["confirm/fin_mid/signature"]["ll_bic"]["reps"],
        "bo": {
            a: cov("confirm", "fin_mid", a) for a in ("signature", "digest", "callid", "oracle")
        },
        "infra": {a: cov("transfer", "transfer_mid", a) for a in ("signature", "callid", "oracle")},
        "coupled": coupled,
        "coupled_exact": cov("transfer", "transfer_mid_rho100", "oracle"),
        "callid_gap_max": max(callid_gaps),
        "lastwriter_gap_max": max(lastwriter_gaps),
        "signature_gap_max": max(signature_gaps),
        "anti_envs": anti_envs,
        "burst_p": burst_all["p_drop"],
        "burst_reps": bo[0]["ll_bic"]["reps"],
        "bo_first": bo[0],
        "bo_last": bo[-1],
        "bo_bias": max(abs(c["ll_bic"]["bias"]) for c in bo),
        "infra_first": infra[0],
        "infra_last": infra[-1],
    }


def _interval(folder: Path) -> dict:
    """The D52 figures: `ll_bic_hom` against `ll_bic` and `ll_qbic_hom`."""
    cells = json.loads((folder / "interval-comparison.json").read_text())["cells"]

    def cov(key: str, est: str) -> float:
        return cells[key][est]["coverage"]

    def widen(key: str) -> int:
        return round(
            100 * (cells[key]["ll_bic_hom"]["mean_width"] / cells[key]["ll_bic"]["mean_width"] - 1)
        )

    def trio(keys: list[str], est: str) -> str:
        return ", ".join(f"{cov(k, est):.3f}" for k in keys)

    shared = ["session/shared/confirm/full", "session/shared/transfer/full"]
    bursts = ["burst/confirm/w12", "burst/long-60/w12", "burst/transfer/w12"]
    sessions = ["session/gateway/confirm", "session/both/confirm"]
    return {
        "reps": cells["perrecord/confirm"]["ll_bic_hom"]["reps"],
        "shared_pools": ["confirm", "transfer"],
        "shared_qbic": trio(shared, "ll_qbic_hom"),
        "shared_bic": trio(shared, "ll_bic"),
        "shared_hom": trio(shared, "ll_bic_hom"),
        "burst_pools": [k.split("/")[1] for k in bursts],
        "burst": trio(bursts, "ll_bic_hom"),
        "joint": cov("joint/confirm/w12", "ll_bic_hom"),
        "per": cells["perrecord/confirm"]["ll_bic_hom"],
        "per_plain": cells["perrecord/confirm"]["ll_bic"],
        "coupled": cov("coupled/confirm/rho100", "ll_bic_hom"),
        "session_widen": " to ".join(f"{w}%" for w in sorted(widen(k) for k in sessions)),
        "session_cov": " to ".join(
            f"{c:.3f}" for c in sorted(cov(k, "ll_bic_hom") for k in sessions)
        ),
        "pooled_widen": f"{widen('perrecord/backoffice+infra/full')}%",
        "full_burst_gap": cells["burst/confirm/full/w12"]["ll_bic_hom"]["detects_gap"],
    }


def _session(folder: Path) -> dict:
    """The D53 figures, from session-and-phase.json and the burst cells of D52."""
    cells = json.loads((folder / "session-and-phase.json").read_text())["cells"]
    ic = json.loads((folder / "interval-comparison.json").read_text())["cells"]
    phase = {
        pool: {
            "hom": cells[f"phase/{pool}/actual"]["ll_bic_hom"]["coverage"],
            "actual": cells[f"phase/{pool}/actual"]["ll_bic"]["coverage"],
            "shuffled": cells[f"phase/{pool}/shuffled"]["ll_bic"]["coverage"],
        }
        for pool in ("long-60", "long-120")
    }
    bursts = [
        c["ll_bic_hom"]["coverage"] for k, c in ic.items() if k.startswith(("burst/", "joint/"))
    ] + [c["ll_bic_hom"]["coverage"] for k, c in cells.items() if k.endswith("/actual")]
    final = json.loads((folder / "final-controls.json").read_text())["cells"]
    bursts += [
        c["ll_bic_hom"]["coverage"]
        for k, c in final.items()
        if k.startswith("burst/") and k.endswith("/actual")
    ]
    outages = [c for k, c in cells.items() if k.startswith("outage/")]
    ratios = [c["ll_bic_hom"]["mean_width"] / c["ll_bic"]["mean_width"] for c in outages]
    shared_keys = ("shared/confirm", "shared/transfer")
    position = json.loads((folder / "op-position.json").read_text())
    return {
        "first_quarter": {
            pool: position[pool]["w3_visible_share_by_quarter"][0]
            for pool in ("confirm", "long-60", "long-120")
        },
        "phase": phase,
        "burst_min": min(bursts),
        "burst_max": max(bursts),
        "outage_bic_min": min(c["ll_bic"]["coverage"] for c in outages),
        "outage_bic_max": max(c["ll_bic"]["coverage"] for c in outages),
        "outage_ratio_min": min(ratios),
        "outage_ratio_max": max(ratios),
        "outage_gap": cells["outage/long-60/indep"]["ll_bic_hom"]["detects_gap"],
        "shared_pools": [k.split("/")[1] for k in shared_keys],
        "shared_reps": cells["shared/confirm"]["tier_bic"]["reps"],
        "shared": {
            est: " and ".join(f"{cells[k][est]['coverage']:.3f}" for k in shared_keys)
            for est in ("ll_bic", "ll_bic_hom", "tier_bic", "lc_gated")
        },
    }


def _controls(folder: Path) -> dict:
    """The D54 figures, from final-controls.json."""
    cells = json.loads((folder / "final-controls.json").read_text())["cells"]
    pools = ("confirm", "long-60", "long-120")
    longs = ("long-60", "long-120")

    def cov(key: str, est: str = "ll_bic") -> float:
        return cells[key][est]["coverage"]

    def listed(xs: list[float], p: int = 3, signed: bool = False) -> str:
        return " and ".join(f"{x:+.{p}f}" if signed else f"{x:.{p}f}" for x in xs)

    up = [cov(f"outage/{env}/gateway_trace") for env in ("confirm", "transfer")]
    down = [
        cells[f"outage/{env}/{kind}"]
        for env in ("confirm", "transfer")
        for kind in ("all_three_independent", "gateway_with_diff", "bad_session")
    ]
    worst = [
        cells[f"outage/{env}/{kind}"]
        for env in ("confirm", "transfer")
        for kind in ("cascade", "all_three_together")
    ]
    ratios = [c["ll_bic_hom"]["mean_width"] / c["ll_bic"]["mean_width"] for c in down]
    return {
        "pools": pools,
        "visperm": ", ".join(f"{cov(f'burst/{p}/visperm'):.3f}" for p in pools),
        "actual": ", ".join(f"{cov(f'burst/{p}/actual'):.3f}" for p in pools),
        "bias_long": listed(
            [cells[f"burst/{p}/actual"]["ll_bic"]["bias"] for p in longs], 1, signed=True
        ),
        "hom_long": listed([cov(f"burst/{p}/actual", "ll_bic_hom") for p in longs]),
        "hom_long_width": listed(
            [cells[f"burst/{p}/actual"]["ll_bic_hom"]["mean_width"] for p in longs], 1
        ),
        "outage_up": " / ".join(f"{x:.3f}" for x in up),
        "down_bic_min": min(c["ll_bic"]["coverage"] for c in down),
        "down_bic_max": max(c["ll_bic"]["coverage"] for c in down),
        "down_hom_min": min(c["ll_bic_hom"]["coverage"] for c in down),
        "down_hom_max": max(c["ll_bic_hom"]["coverage"] for c in down),
        "down_ratio_min": min(ratios),
        "down_ratio_max": max(ratios),
        "worst_hom": max(c["ll_bic_hom"]["coverage"] for c in worst),
        "tier": {
            k: cov(f"tier/confirm/{k}", "tier_bic")
            for k in ("fin_mid", "fin_mid_rho100", "fin_mid_mnar")
        },
        "tier_bursts": listed(
            [cov(f"tier/{env}/burst", "tier_bic") for env in ("confirm", "transfer")]
        ),
        "tier_4": cov("tiercount/confirm", "tier_bic_4"),
        "tier_default": cov("tiercount/confirm", "tier_bic"),
    }


def _keyed(d: dict, word: str) -> dict:
    """The one entry of `d` whose key contains `word`."""
    hits = [v for k, v in d.items() if word in k]
    if len(hits) != 1:
        raise BuildError(f"expected one key containing {word!r}, found {len(hits)}")
    return hits[0]


def verb(passed: bool) -> str:
    return "passes" if passed else "fails"


def card(
    runs: list[dict],
    rows: dict[str, list[dict]],
    out: Path,
    commit: str,
    with_transcripts: bool,
    excluded: list[str],
    spend: dict,
    snippet_output: str,
    freezes: dict[str, str],
) -> str:
    """The card, as a function of the shipped files plus four inputs from outside them.

    `spend` comes from every receipt on disk, `snippet_output` from running the
    snippet on this build, `freezes` from git, and `excluded` from RUNS.
    scripts/validate_dataset.py calls this with its own copies of those four
    and diffs the result against README.md, and separately checks every
    headline figure against its own reading of the shipped files.
    """
    from lacuna.thinning import ENV_VIEWS

    by = {r["run"]: r for r in runs}
    conf = by["confirm"]
    supp = by[SUPPLEMENT["name"]]
    results = out / "results"
    hyp = json.loads((results / "hypotheses.json").read_text())
    hyp_supp = json.loads((results / "hypotheses-supplement.json").read_text())
    variance = json.loads((results / "supplement-variance.json").read_text())
    registered, restored = _keyed(variance, "registered"), _keyed(variance, "supplement")
    h1, h1_supp = hyp["hypotheses"]["H1 recovery"], hyp_supp["hypotheses"]["H1 recovery"]
    bar = float(re.search(r">=\s*([0-9.]+)", h1["detail"]).group(1))
    thin_dir = results / "thinning"
    confirm_json = json.loads((thin_dir / "confirm.json").read_text())
    reps = confirm_json["reps"]
    mid, mid_hidden = _est(thin_dir / "confirm.json", "fin_mid/signature")
    _, mid_hidden_oracle = _est(thin_dir / "confirm.json", "fin_mid/oracle")
    mid_cfg = confirm_json["configs"]["fin_mid/signature"]["config"]
    rho, rho_hidden = _est(thin_dir / "confirm.json", "fin_mid_rho100/signature")
    mnar, mnar_hidden = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar/signature")
    mnar_rho, _ = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar_rho100/signature")
    strat, _ = _est(thin_dir / "confirm-mnar-strat.json", "fin_mid_mnar/signature")
    hidden, _ = _est(thin_dir / "confirm-mnar-hidden.json", "full_mid_hidden/signature")
    fp_sig, _ = _est(thin_dir / "confirm-falsepos.json", "anti_fin/signature")
    fp_orc, _ = _est(thin_dir / "confirm-falsepos.json", "anti_fin/oracle")
    rob = _robustness(results / "robustness", bar)
    lk = _linkage(results / "robustness")
    coupled = " to ".join(f"{c:.3f}" for c in lk["coupled"])

    def triple(xs: list[float]) -> str:
        return ", ".join(f"{x:.3f}" for x in xs)

    od_ind_qp, od_ind, od_bic_qp = (
        triple(lk["od_ind_qp"]),
        triple(lk["od_ind"]),
        triple(lk["od_bic_qp"]),
    )
    od_pools = ", ".join(lk["od_pools"])
    ic = _interval(results / "robustness")
    ic_pools = ", ".join(ic["shared_pools"])
    ic_burst_pools = ", ".join(ic["burst_pools"])
    sp = _session(results / "robustness")
    sp_pools = ", ".join(sp["shared_pools"])
    fc = _controls(results / "robustness")
    fc_pools = ", ".join(fc["pools"])
    financial = ", ".join(f"`{op}`" for op in ENV_VIEWS[mid_cfg["env_view"]])
    single_mcse = mid["ll_bic"]["coverage_mcse"]
    real_runs = [r for r in runs if r["run"] != SUPPLEMENT["name"]]
    llm_runs = [r for r in real_runs if r["agent"] == "llm"]
    scripted_runs = [r for r in real_runs if r["agent"] == "scripted"]
    n_eps = sum(r["episodes"] for r in real_runs)
    n_actions = sum(r["n_true"] for r in real_runs)
    _, no_receipt = spend_summary(real_runs)
    full_w4 = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "full"]
    narr = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "narration"]
    stand_in = [r["run"] for r in runs if r["auditor"] == "scripted"]
    lost = conf["failed_episodes"]
    n400 = sum(" 400 " in f["error"] for f in lost)
    others = sorted({f["error"].split(":")[0] for f in lost if " 400 " not in f["error"]})
    read = [r["agent_read_trace"] for r in rows["confirm"]]
    tx_llm = [
        r["run"]
        for r in runs
        if r["transcripts_stored"] and not r["transcripts_included"] and r["agent"] == "llm"
    ]
    tx_scripted = [
        r["run"]
        for r in runs
        if r["transcripts_stored"] and not r["transcripts_included"] and r["agent"] == "scripted"
    ]
    models = {}
    for r in llm_runs:
        models.setdefault(r["model"], []).append(r["run"])
    seeds = ", ".join(str(s) for s in registered["seeds"])
    picks = ", ".join(f"{c['seed']} from `{c['run']}`" for c in supp["composition"])
    docs_text = {doc: (out / "docs" / doc).read_text() for doc in PREREGISTRATIONS}
    shas = {
        doc: hashlib.sha256((out / "docs" / doc).read_bytes()).hexdigest()
        for doc in PREREGISTRATIONS
    }
    prereg_sha, freeze = shas["PRE-REGISTRATION.md"], freezes["PRE-REGISTRATION.md"]
    unfrozen_status = "ready to freeze" in docs_text["PRE-REGISTRATION.md"]
    tr = _transfer(out, by["transfer"])
    inferred = [r for r in runs if r["inferred_fields"]]

    def seen_by(witness: int) -> int:
        """Ledger actions of `confirm` that this witness recorded, before any thinning."""
        total = 0
        for row in rows["confirm"]:
            ids = {e["id"] for e in row["ledger"]}
            total += len({r["oracle"] for r in row["records"] if r["witness"] == witness} & ids)
        return total

    n_confirm = sum(len(r["ledger"]) for r in rows["confirm"])

    def covs(entry: dict) -> str:
        return ", ".join(f"{c:.2f}" for c in entry["coverage"])

    def margin(m: float) -> str:
        return f"{abs(m):.1f} MCSE {'above' if m >= 0 else 'below'} the {bar:.2f} bar"

    def fmt(v: float | None, p: int = 2) -> str:
        return "unbounded" if v is None else f"{v:.{p}f}"

    def recall(rs: list[dict]) -> str:
        return ", ".join(f"{r['w4_recall']:.4f} in `{r['run']}`" for r in rs)

    def names(rs: list[str]) -> str:
        return ", ".join(f"`{r}`" for r in rs)

    run_table = "\n".join(
        f"| `{r['run']}` | {r['phase']} | {r['model'] or 'scripted'} | {r['auditor']} | "
        f"{r['tickets']} | {r['episodes']} | {r['n_true']} | {r['w4_recall']:.4f} | "
        f"{fmt_usd(r['spend_usd'])} |"
        for r in runs
    )
    marginal = h1["passed"] and registered["mean_margin_mcse"] < 2

    def verdicts(h: dict, mark_marginal: bool) -> str:
        out_rows = []
        for name, v in h["hypotheses"].items():
            verdict = "PASS" if v["passed"] else "FAIL"
            if mark_marginal and name == "H1 recovery" and v["passed"]:
                verdict += " (marginal)"
            out_rows.append(f"| {name} | {verdict} | {v['detail']} |")
        return "\n".join(out_rows)

    hyp_table, supp_table = verdicts(hyp, marginal), verdicts(hyp_supp, False)
    registered_label = (
        "The registered H1 pass was marginal."
        if marginal
        else f"The registered H1 {verb(h1['passed'])}."
    )
    unknown_spend = (
        f" {names(no_receipt)} {'has' if len(no_receipt) == 1 else 'have'} no receipt "
        "(no LLM was called), so the spend is unknown, not zero."
        if no_receipt
        else ""
    )
    study_unshipped = (
        f", which includes {names(spend['study_unshipped'])}, the supplement's rerun of "
        "the lost seeds (D34)"
        if spend["study_unshipped"]
        else ""
    )
    excluded_note = (
        f"Not shipped as runs: {names(sorted(excluded))} (the supplement's rerun, "
        "diagnostic reruns and a derived pool). The three episodes the supplement needs "
        "are drawn from them."
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
            f"Transcripts exist for the LLM runs {names(tx_llm)}, where they hold the "
            "provider's reasoning text, and for the scripted runs "
            f"{names(tx_scripted)}, where they hold none. All are withheld from this "
            "release pending a check of the model provider's terms, which this license "
            "cannot cover. No estimate depends on them; `agent_read_trace` was computed "
            "from them."
        )
    inferred_note = "; ".join(
        f"`{r['run']}` ({', '.join(f'`{f}`' for f in r['inferred_fields'])})" for r in inferred
    )
    model_note = "; ".join(f"{m}: {names(rs)}" for m, rs in sorted(models.items()))
    t = tr["t"]
    t3_label = "T3 passes only because the interval widened." if t["T3"]["passed"] else "T3 fails."
    predicted = (
        f"- **T1's failure was predicted before the freeze.** Prediction {tr['predicted']} in "
        "`docs/PREREG-TRANSFER.md` says T1 fails under signature linkage, from the "
        "free scripted pipeline check (D46)."
        if tr["predicted"]
        else ""
    )
    d34_note = (
        " `docs/PREREG-TRANSFER.md` cites D34, which postdates its freeze; "
        "`docs/TRANSFER.md` discloses this."
        if tr["prereg_cites_d34"]
        else ""
    )
    status_note = (
        ' Its text still reads "Status: ready to freeze"; the commit above is what froze it.'
        if unfrozen_status
        else ""
    )
    dose = rob

    body = f"""
# lacuna capture histories

> **Read before any result below (D36, D44, D47).**
> - **{registered_label}** Over loss seeds {seeds}, `ll_bic`
>   covers {covs(registered)}: mean {registered["mean_coverage"]:.3f},
>   {margin(registered["mean_margin_mcse"])}. The MCSE used here is the four-seed one,
>   {registered["mean_mcse"]:.3f} over {len(registered["seeds"])} × {reps} replicates. One
>   seed alone has MCSE {single_mcse:.3f}, and the registered seed sits
>   {margin(registered["seed_4242_margin_mcse"])} on that scale. Source:
>   `results/supplement-variance.json`.
> - **At {rob["reps"]} replicates the shortfall is linkage, not the estimator (post hoc,
>   D44).** Over {rob["seeds"]} fresh loss seeds on the registered pool at `fin_mid`,
>   `ll_bic` covers {rob["reg_sig"]["coverage"]:.3f} ± {rob["reg_sig"]["coverage_mcse"]:.3f}
>   under signature linkage and {rob["reg_orc"]["coverage"]:.3f} ±
>   {rob["reg_orc"]["coverage_mcse"]:.3f} under oracle; on the restored pool,
>   {rob["res_sig"]["coverage"]:.3f} against {rob["res_orc"]["coverage"]:.3f}. BIC fits a
>   trace–gateway interaction, faked by signature collisions, in {rob["w1w2_sig"]:.3f} of
>   replicates ({rob["w1w2_orc"]:.3f} under oracle linkage). The four-seed mean above was
>   a high draw. Source: `results/robustness/precision.json`.
> - **The back-office shortfall is a linkage choice (post hoc, D49, D50, D52).** The
>   signature dropped content digests the logs carry. Linking by the shared tool-call
>   id restores {lk["bo"]["callid"]:.3f} in back-office under the financial view, where the
>   env diff is trivial to attach; in infra, whose env diff has no id, it reaches
>   {lk["infra"]["callid"]:.3f} ({coupled} coupled). It is not a general fix for linkage.
>   Source: `results/robustness/linkage-arms.json`.
> - **With the lost episodes restored, H1 {verb(h1_supp["passed"])}.** The confirmatory run
>   completed {conf["episodes"]} of {conf["episodes_attempted"]} episodes. The
>   `supplement` config replaces each lost seed with its first successful rerun, on a
>   client changed after the D33 and D34 key-presence change (reasoning is echoed back
>   whenever the key is present). With them, mean coverage over the same loss seeds is
>   {restored["mean_coverage"]:.3f} ({covs(restored)}), {margin(restored["mean_margin_mcse"])}
>   (four-seed MCSE {restored["mean_mcse"]:.3f}). At the registered loss seed the artifact
>   records H1 as {"PASS" if h1_supp["passed"] else "FAIL"}, testing "{h1_supp["detail"]}".
>   Source: `results/hypotheses-supplement.json`. The registered verdict stands on its
>   {conf["episodes"]} episodes; the supplement is not part of it.
> - **The losses are not explained.** {n400} of the {len(lost)} were provider 400s
>   that appear to depend on the agent's trajectory; the trigger is still open (D34).
>   The rest: {", ".join(others) or "none"}. D33's empty-string mechanism is withdrawn.
> - **Every coverage figure is under simulated loss.** Loss was injected post hoc into
>   complete logs, and no figure here measures real log loss. Before thinning, W1 and
>   W2 each hold every `confirm` action ({seen_by(0)}/{n_confirm} and
>   {seen_by(1)}/{n_confirm}), so all hidden mass comes from the post-hoc thinning plus
>   the injected financial env view (D43).
> - **The agent can read W1's trace file, though not write it**, and did so in
>   {sum(bool(x) for x in read)} of {len(read)} `confirm` episodes (`agent_read_trace`).
> - **The transfer study {verb(t["T1"]["passed"])} T1 (D47).** In a second environment,
>   pre-registered separately, `ll_bic` covers {tr["t1_frozen"]:.3f} at the frozen loss
>   seed; over {tr["seeds"]} seeds (post hoc), {tr["sig"]["coverage"]:.3f} under signature
>   linkage against {tr["orc"]["coverage"]:.3f} under oracle. See the transfer study section.

How many consequential actions did an agent take that no log line shows? This
dataset holds the witness records lacuna uses to answer that by
capture-recapture: {n_eps} episodes across {len(real_runs)} runs
({sum(r["episodes"] for r in llm_runs)} LLM-agent, {sum(r["episodes"] for r in scripted_runs)}
scripted), {n_actions} ground-truth actions ({sum(r["n_true"] for r in llm_runs)} LLM-agent,
{sum(r["n_true"] for r in scripted_runs)} scripted), and for each action what every witness
recorded, plus a {supp["episodes"]}-episode `supplement` of rerun confirmatory seeds.

**Synthetic data, not production-validated.** Both environments, their worlds
and their tickets are generated from a seed; the LLM agent's actions are real
model output.

Built by `scripts/build_dataset.py` from lacuna commit `{commit}`. The lacuna
code repository is not public yet, so `docs/` ships the decision log, both
pre-registrations and the transfer write-up that the D-numbers on this card
refer to. Every result on this card is computed at build time from the files
in this dataset, except the total spend, which also counts the receipts of
runs not shipped here.

## What the data is

An agent works a queue of support tickets in a synthetic back office:
refunds, cancellations, customer updates, emails and one report file. The
environment kernel keeps a hidden ledger of every consequential action. That
ledger is the ground truth, and no witness reads it. Each witness writes its
own record of what happened, and capture-recapture turns the overlaps between
those records into an estimate of the actions no witness saw. The `transfer`
runs use a second environment, `infra`, where an on-call operator scales
services, deploys, rolls back, changes config, opens incidents, pages teams and
writes runbooks.

- One config per run (`confirm` is the default): one row per episode, with the
  ledger and every witness record.
- `supplement`: the {supp["episodes"]} lost confirmatory seeds, rerun ({picks}).
  A post-hoc supplement, not the registered verdict.
- `runs`: one row per run and one for `supplement`, with model, spend, failed
  episodes, the pooled capture-history tables and the caveat that governs it.
- `results/`: derived results, copied byte for byte: the thinning summaries
  (`results/thinning/*.json`), the registered, supplement and transfer verdicts
  (`results/hypotheses.json`, `results/hypotheses-supplement.json`,
  `results/transfer_hypotheses.json`), the loss-seed variance
  (`results/supplement-variance.json`) and the simulator coverage sweep
  (`results/sim_coverage.json`).
- `results/robustness/`: post-hoc checks. The D44 round (precision at
  {rob["reps"]} replicates under both linkages, the MNAR dose response, the false-gap
  battery, the signature-collision count) and the transfer study's
  (`transfer-*.json`), the D49 round (linkage arms and burst loss), the D50
  quasi-Poisson check (`overdispersion.json`), the D51 episode-dispersion check
  (`episode-dispersion.json`), the D52 interval comparison
  (`interval-comparison.json`) and the D53 session and phase checks
  (`session-and-phase.json`, `op-position.json`), and the D54 final controls
  (`final-controls.json`).
- `environment/<env>/`: each environment's agent system prompt and tool schemas,
  verbatim ({names(sorted({r["env"] for r in runs}))}).
- `docs/`: `DECISIONS.md`, `PRE-REGISTRATION.md`, `PREREG-TRANSFER.md` and
  `TRANSFER.md`, as of the build commit.

| run | phase | model | auditor | tickets | episodes | N | W4 recall | spend (USD) |
|---|---|---|---|---|---|---|---|---|
{run_table}

Spend over every receipt on disk is {spend["total"]:.2f} USD: the study
{spend["study"]:.2f}{study_unshipped}, and the diagnostic reruns
({names(spend["diagnostic_runs"])}) {spend["diagnostic"]:.2f}. The first confirmatory
attempt, which crashed (D30), left no receipt and is not counted anywhere.{unknown_spend}
The supplement's spend sits in those reruns' receipts and is not split per
episode. W4 recall is the share of ledger actions witness 3 reported, matched
to the ledger by signature, since auditor records carry no ledger id; the
`auditor` column says whether that witness was an LLM or the scripted stand-in.
{transcript_note} {excluded_note}

## Load it

The repository is private: run `hf auth login` with an account that can read
it, or pass `token=` to `load_dataset`. Each config has one split, `train`.

```py
from datasets import load_dataset

repo = "{REPO}"
confirm = load_dataset(repo, "confirm", split="train")  # the default config
runs = load_dataset(repo, "runs", split="train")
print(confirm.num_rows, runs.num_rows)
```

## The three witnesses, and why the LLM auditor is not one

| id | in the docs | witness | what it sees | blind spots |
|---|---|---|---|---|
| 0 | W1 | trace | the agent harness's own log file | lines lost to logging faults |
| 1 | W2 | gateway | every tool call through the router | bypassable through code execution |
| 2 | W3 | env | end-of-shift state diff plus outbound mail | repeats, reverts, unviewed ops |
| 3 | W4 | auditor | an LLM rebuilding actions from the transcript | **not a witness** |

The docs name witnesses W1 to W4; the data numbers them 0 to 3. Result labels
call W2 `proxy`, as in `ll_bic@trace+proxy+env`. `records` holds witnesses 0-2
only, so a union over it is the floor. Witness 3 is in `auditor_records`,
because the runs recorded it, and is excluded from every estimate in
`results/` (D20). An auditor reading the agent's own transcript is either a
copy of the trace or empty. With the full transcript it reads every call
verbatim: W4 recall {recall(full_w4)}. With narration only it reports nothing:
{recall(narr)}. The agent narrates in aggregate ("nine refunds issued"), and a
count cannot be overlapped with another witness's record. In {names(stand_in)},
witness 3 is a scripted stand-in that copies the transcript with misses.

## Field reference

Episode configs:

| field | type | meaning |
|---|---|---|
| `run` | string | run name; in `supplement`, the rerun the episode came from |
| `seed` | int | episode seed; the generator regenerates world and tickets from it |
| `model` | string or null | agent model; null for the scripted agent |
| `env` | string | environment: `backoffice`, or `infra` for the transfer runs |
| `tickets` | int | tickets in the queue; `backoffice` adds one closing report task |
| `phase` | string | see below |
| `auditor_view` | string | what witness 3 read: `full` transcript or `narration` only |
| `n_true` | int | ground-truth actions, the ledger length |
| `ledger` | list | ground truth: `id`, `op`, `resource`, `value`, `step` |
| `records` | list | claims by witnesses 0-2: `witness`, `op`, `resource`, `value`, `oracle` |
| `auditor_records` | list | claims by witness 3, same shape, `oracle` always null |
| `agent_read_trace` | bool or null | the agent read W1's trace file; null if no transcript stored |
| `n_calls` | int | tool calls the agent made |
| `n_tool_errors` | int | tool results that were errors |

`phase` is one of `confirmatory`, `confirmatory (transfer, separately
pre-registered)`, `exploratory`, `scripted harness check`, or, in `supplement`,
`{SUPPLEMENT_PHASE}`. `agent_read_trace` is always false in `infra`, which has
no `read_file` tool. The `run` values in `supplement` rows name diagnostic
reruns that are not `runs` rows or configs.

`record.oracle` is the ledger id when the record came from infrastructure that
can know it. It is for oracle linkage only and is never shown to an
estimator. Content digests stand in for text: `send_email` and `write_file`
values in `backoffice`, `send_page` and `write_runbook` values in `infra`.

`runs`: `run`, `phase`, `env`, `agent`, `auditor`, `model`, `auditor_view`,
`tickets`, `episodes`, `episodes_attempted`, `failed_episodes` (seed and the
recorded error, provider request ids redacted), `stopped`, `seed0`, `n_true`,
`recorded_fault_rates`, `spend_usd` (null when no receipt exists),
`llm_calls`, `transcripts_stored`, `transcripts_included`, `w4_recall`,
`oracle_table`, `signature_table` (16 cells, bit w for witness w; cell 0 of
`oracle_table` is the true unseen count), `caveat`, `composition` (the
supplement's sources), `inferred_fields`.

Inferred, not recorded (`runs.inferred_fields`): {inferred_note}. The older
summaries did not record these. `tickets` is the queue length that explains
every ledger, checked by regenerating the tickets; `auditor_view` is `full`,
the only view before narration was added (D14).

## Glossary

- **oracle linkage**: matching each record to its ledger action by the ledger
  id infrastructure attaches (`record.oracle`). An upper bound; an auditor
  cannot do it.
- **signature linkage**: matching records by operation, resource and value,
  with content digests left out, pairing in order within a signature. What an
  auditor can compute, and both confirmatory studies' headline linkage.
- **`fin_mid`**: the held-out `backoffice` loss configuration. The gateway drops
  {mid_cfg["p_gateway_drop"]:.2f} of its lines and the trace
  {mid_cfg["p_trace_drop"]:.2f}, and the env view sees only {financial}.
- **`transfer_mid`**: the `infra` counterpart. The gateway drops
  {tr["mid_cfg"]["p_gateway_drop"]:.2f} and the trace {tr["mid_cfg"]["p_trace_drop"]:.2f}, and
  the env view is `{tr["mid_cfg"]["env_view"]}`.
- **`_rho100`**: the same drop rates with the gateway and trace losses coupled
  (rho = 1): an action whose trace line is lost loses its gateway line too, as
  one outage takes both down.
- **`anti_*`**: mutually exclusive loss. An action loses its trace line or its
  gateway line, never both, so nothing is hidden by construction and any gap
  an estimator claims is false.
- **`ll_bic`**: a log-linear capture-recapture model over witnesses 0-2 whose
  interaction terms are chosen by BIC, with a profile-likelihood 95% interval.
  The primary estimator (D27).
- **`strat_bic`**: `ll_bic` fitted per operation and combined. A stratum it
  cannot identify makes the interval unbounded, so it refuses rather than
  answers (D37).
- **`union_count`**: actions seen by at least one witness, the floor.
  `union_bound` is the interval from it to infinity.
- **`step`**: the kernel's call counter when the ledger action happened; every
  tool call, reads included, advances it.
- **detects gap**: the share of replicates whose interval's lower end is above
  `union_count`.
- **MCSE**: the Monte Carlo standard error of a coverage figure.

## Rebuild a capture-history table and rerun an estimator

Run from the dataset root. Standard library only.

```python
{SNIPPET}```

Expected output (one loss draw under oracle linkage):

```text
{snippet_output}
```

That is a single draw, matched by ledger id. The {mid_hidden} actions hidden
at `fin_mid` quoted on this card are the mean over {reps} draws under signature
linkage; under oracle linkage that mean is {mid_hidden_oracle}
(`results/thinning/confirm.json`).

Every run except `pilot-l0` and `scripted-l0` was recorded with complete logs
(`recorded_fault_rates` 0), so the raw histories have nothing hidden and loss
is injected afterwards, as above. `pilot-l0` and `scripted-l0` were thinned
at record time. The log-linear, latent-class and bound estimators behind
`results/` are in the lacuna code, which is not public yet; `results/thinning/`
holds their output over {reps} replicates per configuration.

## Exploratory vs confirmatory

**Confirmatory: `confirm` (H1-H4) and `transfer` (T1-T4).** Each was
pre-registered on its own, and their verdicts are the only pass/fail claims in
this dataset. `transfer` is covered in the transfer study section.

`confirm` was pre-registered in `docs/PRE-REGISTRATION.md` and tested only on
loss configurations held out of every exploratory run (`fin_mid`,
`fin_mid_rho100`). The evidence that the registration was frozen before the
run is local git commit `{freeze}`, which cannot be checked from outside until
the code repository is public.{status_note} The shipped file has sha256
`{prereg_sha}`. {conf["episodes"]} episodes, N = {conf["n_true"]},
{hyp["linkage"]} linkage. Source: `results/hypotheses.json`.

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

**Exploratory: every other LLM run, and every post-hoc result.** That includes
the `long-*` thinning results, from which the estimator was chosen (D27); the
MNAR and false-positive results (D31, D37-D42); the D44 precision, dose,
false-gap and collision checks; and the transfer study's precision, mechanism,
false-gap and density results (D46-D48); the D49 linkage arms and burst-loss
checks; and the D50 overdispersion, D51 episode-dispersion, D52
interval-comparison, D53 session-and-phase and D54 final-control checks. All were
built after the
confirmatory runs and cannot change their verdicts.

**Scripted harness checks: `scripted-l0`, `transfer-scripted`,
`transfer-scripted-24`.** No LLM; checks that the harness and the witnesses
work, each in its own config.

## Transfer study

`transfer` is a second confirmatory study, pre-registered separately in
`docs/PREREG-TRANSFER.md` (frozen at local commit
`{freezes["PREREG-TRANSFER.md"]}`, which cannot be checked from outside until the code
repository is public; sha256 `{shas["PREREG-TRANSFER.md"]}`) and written up in
`docs/TRANSFER.md` (D45-D48).{d34_note} {tr["run"]["episodes"]} of
{tr["run"]["episodes_attempted"]} episodes completed (failed:
{"; ".join(f"{f['seed']} ({f['error'].split(':')[0]})" for f in tr["run"]["failed_episodes"])}),
N = {tr["run"]["n_true"]}, {tr["hyp"]["linkage"]} linkage. Source:
`results/transfer_hypotheses.json`.

| hypothesis | verdict | detail |
|---|---|---|
{verdicts(tr["hyp"], False)}

- **T1 {verb(t["T1"]["passed"])}.** `ll_bic` covers {tr["t1_frozen"]:.3f} at the frozen loss
  seed. Post hoc, over {tr["seeds"]} loss seeds ({tr["sig"]["reps"]} replicates), it covers
  {tr["sig"]["coverage"]:.3f} ± {tr["sig"]["coverage_mcse"]:.3f} under signature
  linkage and {tr["orc"]["coverage"]:.3f} ± {tr["orc"]["coverage_mcse"]:.3f} under oracle.
  Source: `results/robustness/transfer-precision.json`.
{predicted}
- **T2 and T4 say nothing under signature linkage (post hoc).** With mutually
  exclusive loss, where nothing is hidden, `ll_bic` claims a gap in
  {tr["gap_sig"]:.3f} of replicates under signature linkage and {tr["gap_orc"]:.3f}
  under oracle; with W3 complete, {tr["gap_w3_complete"]:.3f}. Source:
  `results/robustness/transfer-falsegap.json`.
- **{t3_label}** `ll_bic` covers
  {tr["mid_cov"]:.2f} at `transfer_mid` ({tr["mid_hidden"]} hidden, mean width
  {tr["mid_width"]:.1f}) and {tr["rho_cov"]:.2f} at `transfer_mid_rho100`
  ({tr["rho_hidden"]} hidden, mean width {tr["rho_width"]:.1f}). Source:
  `results/thinning/transfer.json`.
- **The mechanism is linkage (post hoc, D47).** Over {tr["mech_seeds"]} loss seeds,
  matching exactly only the signature groups W3 misses (all-W3-invisible and
  mixed groups) restores `ll_bic` to {tr["mech_restore"]:.3f}, against
  {tr["mech_sig"]:.3f} under signature linkage and {tr["mech_exact"]:.3f} with exact
  matching everywhere. This cell has {tr["mech_seeds"]} seeds and the T1 precision
  above {tr["seeds"]}, which is why its signature coverage differs. Source:
  `results/robustness/transfer-mechanism.json`.
- **The verdict depends on density (post hoc, D46).** Across {tr["cells"]} cells of
  services × tickets ({tr["cell_eps"]} episodes each), signature-linkage coverage runs
  from {tr["best"]["signature"]["coverage"]:.3f} ({tr["best"]["services"]} services,
  {tr["best"]["tickets"]} tickets) to {tr["worst"]["signature"]["coverage"]:.3f}
  ({tr["worst"]["services"]} services, {tr["worst"]["tickets"]} tickets), as the share of
  W3-invisible actions that share a signature rises from
  {tr["best"]["invisible_sharing_a_signature_share"]:.3f} to
  {tr["worst"]["invisible_sharing_a_signature_share"]:.3f}; under oracle linkage it stays
  {tr["orc_min"]:.2f} to {tr["orc_max"]:.2f}. The grid's episodes are free scripted runs
  (episode seeds from {tr["cell_seed0"]}) that are not shipped, so this result cannot be
  reproduced from the dataset alone. Source: `results/robustness/transfer-density.json`.
- **The paid run reproduces the scripted plans.** {tr["paid"]["in_the_scripted_plans"]} of
  {tr["paid"]["actions"]} `transfer` actions are in the scripted plans
  ({tr["free"]["in_the_scripted_plans"]} of {tr["free"]["actions"]} for
  `transfer-scripted-24`), so the transfer run tests the method on an agent that
  follows the plans; it does not test LLM behaviour beyond that. Source:
  `results/robustness/transfer-behaviour.json`.

## Known caveats

- **Degenerate W4 (D14, D20).** `pilot-l0` and `long-*` used the
  full-transcript auditor, whose recall is 1. Every four-witness number from
  those runs is superseded and supports no claim.
- **Attrition in the confirmatory run (D30, D34-D36).** Failed seeds:
  {"; ".join(f"{f['seed']} ({f['error'].split(':')[0]})" for f in lost)}. The
  400s appear to depend on the agent's trajectory, so which episodes were lost
  is not shown to be random, and restoring them changes H1.
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
- **Signature collisions (post hoc, D44).** {rob["coll"]["sharing_share"]:.3f} of
  `confirm` actions share a signature with another action in the same episode,
  {rob["coll"]["sharing_by_op"]["send_email"]["sharing"]} of those
  {rob["coll"]["sharing_a_signature"]} are emails. Before thinning, trace and gateway each
  held every action ({rob["coll"]["unthinned_ids_seen"]["trace"]}/{rob["coll"]["n"]} and
  {rob["coll"]["unthinned_ids_seen"]["gateway"]}/{rob["coll"]["n"]}), the env diff
  {rob["coll"]["unthinned_ids_seen"]["env"]}. Source: `results/robustness/collisions.json`.
- **MNAR dose response (post hoc, D44).** Without the bias, `ll_bic` covers
  {dose["dose_mcar"]["oracle"]["ll_bic"]["coverage"]:.3f} under oracle linkage and already
  {dose["dose_mcar"]["signature"]["ll_bic"]["coverage"]:.3f} under signature. With loss keyed
  on the operation it falls to {dose["dose_at_first"]["oracle"]["ll_bic"]["coverage"]:.3f}
  and {dose["dose_at_first"]["signature"]["ll_bic"]["coverage"]:.3f} at {dose["dose_first"]}x,
  below the {bar:.2f} bar under both linkages, and recovers at {dose["dose_recover"]}x
  ({dose["dose_at_recover"]["oracle"]["ll_bic"]["coverage"]:.3f} oracle,
  {dose["dose_at_recover"]["signature"]["ll_bic"]["coverage"]:.3f} signature) only by
  widening: mean width {dose["dose_mcar"]["oracle"]["ll_bic"]["mean_width"]:.1f} without the
  bias against {dose["dose_at_recover"]["oracle"]["ll_bic"]["mean_width"]:.1f} there.
  `strat_bic` stays calibrated, {dose["strat_min"]:.2f} to {dose["strat_max"]:.2f} at every
  dose, by refusing more often: its unbounded share rises from
  {dose["strat_refuse_low"]:.2f} to {dose["strat_refuse_high"]:.2f}. Source:
  `results/robustness/mnar-dose.json`.
- **False gaps (post hoc, D44).** With mutually exclusive loss, where nothing is
  hidden, across {rob["fg_pools"]} pools and rates {rob["fg_rates"][0]} to
  {rob["fg_rates"][-1]}, `ll_bic` claims a gap in {rob["fg_inside_min"]:.2f} to
  {rob["fg_inside_max"]:.2f} of replicates only when records are matched by signature
  and the env view is financial, at every rate and in every pool; elsewhere in at
  most {rob["fg_outside_max"]:.2f}. Source: `results/robustness/falsegap-battery.json`.
- **Linkage arms, scoped (post hoc, D49, D50, D52).** `ll_bic` coverage over {lk["reps"]}
  replicates per cell. Back-office `fin_mid`: signature {lk["bo"]["signature"]:.3f},
  digest {lk["bo"]["digest"]:.3f}, call id {lk["bo"]["callid"]:.3f}, exact
  {lk["bo"]["oracle"]:.3f}. Under the financial view the call id equals exact matching,
  because the env diff sees every email, refund and cancellation and is trivial to
  attach. With the full view it falls short: `full_mid` call id
  {lk["bo_full"]["callid"]:.3f}, exact {lk["bo_full"]["oracle"]:.3f}. Infra `transfer_mid`:
  signature {lk["infra"]["signature"]:.3f}, call id {lk["infra"]["callid"]:.3f}, exact
  {lk["infra"]["oracle"]:.3f}. Infra `transfer_mid_rho100`: signature, digest and call id
  {coupled}, exact {lk["coupled_exact"]:.3f}. Plain call id is the auditor-computable
  result. A last-writer rule that attached the env diff's end-state row to the latest
  logged call (D50, D51) is withdrawn (D52): its {lk["infra_lastwriter"]:.3f} in infra is
  partly compensating error, it claims false gaps under exclusive loss in up to
  {lk["lastwriter_gap_max"]:.3f} of replicates, and it assumes each log lists calls in
  execution order. The plain call id removes every false gap measured, in both environments
  (the {" and ".join(f"`{r}`" for r in lk["anti_envs"])} pools): with exclusive loss it
  claims a gap in at most {lk["callid_gap_max"]:.3f} of replicates, against up to
  {lk["signature_gap_max"]:.3f} under signature linkage. That holds by construction when
  trace and gateway together see every action. "Exact" is oracle linkage, which no
  auditor has. The call id is the provider's tool-call id, which the harness writes
  into both logs; an agent that writes its own trace could reuse or invent ids,
  which is untested. Source: `results/robustness/linkage-arms.json`.
- **Bursts are an interval problem (post hoc, D49, D50).** When each block of w
  consecutive actions is dropped together (p = {lk["burst_p"]}, exact matching,
  {lk["burst_reps"]} replicates), plain `ll_bic` in back-office with the financial
  view and independent loss covers {lk["bo_first"]["ll_bic"]["coverage"]:.3f} at
  w={lk["bo_first"]["width"]} down to {lk["bo_last"]["ll_bic"]["coverage"]:.3f} at
  w={lk["bo_last"]["width"]}, with bias at most {lk["bo_bias"]:.1f} actions against about
  {lk["bo_first"]["ll_bic"]["hidden"]:.0f} hidden; infra covers
  {lk["infra_first"]["ll_bic"]["coverage"]:.3f} down to
  {lk["infra_last"]["ll_bic"]["coverage"]:.3f}. Of the D50 quasi-Poisson intervals, which
  scale the cutoff by a measured dispersion, only `ll_independence_qp`, and only under
  exact matching, recovers about 0.9: at w=12 it covers {od_ind_qp} ({od_pools}) against
  {od_ind} plain ({lk["od_reps"]} replicates), and {lk["od_ind_qp_signature"]:.3f} in infra
  under signature linkage. Under coupling it explodes: at
  `fin_mid_rho100` its interval is about {lk["od_coupled_ind_qp_width"]:.0f} wide.
  `ll_bic_qp`, which takes the dispersion from the model BIC selected, is safe under
  coupling ({lk["od_coupled_bic_qp"]:.3f}) but fixes bursts only partly ({od_bic_qp}).
  Neither helps joint bursts, where one outage takes both logs: `ll_bic_qp` stays at
  {lk["od_joint_bic_qp"]:.3f}, and `ll_independence_qp` covers only by widening to about
  {lk["od_joint_ind_qp_width"]:.0f}. Their dispersion factor is not an alarm: it reads
  {lk["od_phi_coupled_infra"]:.2f} in coupled infra, where `ll_bic_qp` covers
  {lk["od_coupled_infra_bic_qp"]:.3f}, and {lk["od_phi_joint"]:.2f} under joint bursts. Sources:
  `results/robustness/burst-loss.json`, `overdispersion.json`.
- **`ll_bic_hom` keeps BIC's model and widens only (post hoc, D51, D52).** The episode dispersion
  is the Pearson X²/df of the episodes × cells table, floored at 1. `ll_qbic_hom`
  (D51) chooses the model by QBIC on that scale, and collapses when session quality
  varies and hits all three logs at once: {ic["shared_qbic"]} ({ic_pools}), against
  `ll_bic` {ic["shared_bic"]}. `ll_bic_hom` keeps `ll_bic`'s model and point and scales
  only the profile cutoff by the dispersion, so it never covers less than `ll_bic`.
  With exact matching ({ic["reps"]} replicates per cell) it covers {ic["burst"]} under
  bursts at w=12 ({ic_burst_pools}); {ic["joint"]:.3f} under a joint outage;
  {ic["per"]["coverage"]:.3f} under per-record loss at the plain width
  ({ic["per"]["mean_width"]:.1f} against {ic["per_plain"]["mean_width"]:.1f}); and
  {ic["coupled"]:.3f} under coupling. Shared session quality defeats the pooled
  intervals: it still fails there, at {ic["shared_hom"]} (D53 below). The cost is width
  whenever sessions differ at all: session-varying
  drop rates widen it by {ic["session_widen"]} with coverage {ic["session_cov"]}, pooling
  the two environments by {ic["pooled_widen"]}, and under low-loss full-view bursts its
  gap detection falls to {ic["full_burst_gap"]:.3f}. It needs visible episode boundaries,
  and it trades width for coverage (D54 below).
  Source: `results/robustness/interval-comparison.json`.
- **Where the dispersion correction costs or fails (post hoc, D53).** Across every
  burst cell measured, `ll_bic_hom` covers {sp["burst_min"]:.2f} to {sp["burst_max"]:.2f}.
  On the long-episode pools it covers {sp["phase"]["long-60"]["hom"]:.3f} (long-60) and
  {sp["phase"]["long-120"]["hom"]:.3f} (long-120) under bursts at w=12, because env-blind
  actions come in runs and a block of lost lines lands on them: action-dependent loss
  arriving by time. The env diff sees {sp["first_quarter"]["long-60"]:.3f} of the first
  quarter of a long-60 episode and {sp["first_quarter"]["long-120"]:.3f} of a long-120 one,
  against {sp["first_quarter"]["confirm"]:.3f} in confirm (`op-position.json`). Shuffling action
  order inside episodes before the blocks fall makes plain `ll_bic` nominal:
  {sp["phase"]["long-60"]["actual"]:.3f} to
  {sp["phase"]["long-60"]["shuffled"]:.3f} (long-60) and {sp["phase"]["long-120"]["actual"]:.3f}
  to {sp["phase"]["long-120"]["shuffled"]:.3f} (long-120). Under whole-session outages, where
  a log is down for a whole episode, plain `ll_bic` is already nominal
  ({sp["outage_bic_min"]:.3f} to {sp["outage_bic_max"]:.3f}) while the env diff stays up (D54
  below); `ll_bic_hom` is
  {sp["outage_ratio_min"]:.1f} to {sp["outage_ratio_max"]:.1f} times wider and its gap detection
  falls to {sp["outage_gap"]:.2f} (long-60, independent outages). The correction costs most
  where it is not needed. Source: `results/robustness/session-and-phase.json`.
- **Shared session quality is partly recoverable (post hoc, D53).** When session
  quality hits all three logs at once, the pooled intervals fail: `ll_bic`
  {sp["shared"]["ll_bic"]}, `ll_bic_hom` {sp["shared"]["ll_bic_hom"]} ({sp_pools}). Episodes are
  observed, so `tier_bic`, which ranks episodes by the distinct actions the logs saw
  and runs `strat_bic` over tiers, recovers {sp["shared"]["tier_bic"]}, and `lc_gated`
  {sp["shared"]["lc_gated"]} ({sp["shared_reps"]} replicates). Neither reaches 0.95, and
  `tier_bic` is not a default (D54 below). Source:
  `results/robustness/session-and-phase.json`.
- **Bursts mostly land where the env diff is blind (post hoc, D54).** With the env
  diff's view permuted, plain `ll_bic` at w=12 covers {fc["visperm"]} ({fc_pools}),
  against {fc["actual"]} actual: most of the burst failure is a lost block landing where
  the diff is blind, not dispersion. The point is biased on the long pools
  ({fc["bias_long"]} actions), and `ll_bic_hom` covers there by widening around that
  biased point ({fc["hom_long"]}, widths {fc["hom_long_width"]}). It is a width-for-coverage
  trade, to be reported beside the plain interval. Source:
  `results/robustness/final-controls.json`.
- **Outages that take the env diff down too (post hoc, D54).** Whole-session outages
  of gateway and trace leave plain `ll_bic` nominal ({fc["outage_up"]}) only while the
  env diff stays up. When the diff can fail too, `ll_bic` covers
  {fc["down_bic_min"]:.3f} to {fc["down_bic_max"]:.3f} and `ll_bic_hom` {fc["down_hom_min"]:.3f} to
  {fc["down_hom_max"]:.3f}, at {fc["down_ratio_min"]:.0f} to {fc["down_ratio_max"]:.0f} times the
  width. A cascade, or all three logs failing together, defeats both: `ll_bic_hom`
  covers at most {fc["worst_hom"]:.3f} there. Source: `results/robustness/final-controls.json`.
- **`tier_bic` is not a default (post hoc, D54).** It covers {fc["tier"]["fin_mid"]:.3f}
  under per-record loss, {fc["tier"]["fin_mid_rho100"]:.3f} coupled, {fc["tier_bursts"]}
  under bursts (confirm, transfer) and {fc["tier"]["fin_mid_mnar"]:.3f} under
  action-dependent loss. Under shared session quality it covers {fc["tier_4"]:.3f}
  (confirm) with 4 tiers against {fc["tier_default"]:.3f} at the default tier count: the
  count is a free parameter. Source: `results/robustness/final-controls.json`.
- **Agent models (D13).** {model_note}. Comparisons across models are
  cross-model.
- **Long queues are abandoned (D18).** Actions per ticket fall as the queue
  grows, so `tickets` overstates the work in `long-60` and `long-120`.
- **Benign loss only (D10, D11).** No adversarial agent was run live; every
  adversarial number is from the simulator (`results/sim_coverage.json`). One
  agent family, two environments.
- **Linkage error is real.** Signature linkage leaves content digests out, so
  two different emails to one address in one episode look like one action seen
  twice. It is measured against oracle linkage, not hidden.
- **The docs cite files that are not here.** The four shipped docs refer to
  code, scripts, tests, artifacts and the paper in the lacuna repository, which
  is not public yet and not part of this dataset.

Each run's own caveat is in `runs.caveat`.

## Synthetic data

Every customer, order and ticket is generated from the episode seed: names
are "Customer i", email addresses are at example.com and example.org, phone
numbers start 555, and street names come from a fixed list of six. The
scripted agent deliberately plants wrong-first values with a trailing "x", so
some `scripted-l0` addresses end in example.orgx. In `infra`, ten services with
fixed generic names (api, auth, billing, payments, cart, checkout, catalog,
search, inventory, notify) belong to five teams, and versions, replica counts
and configs come from the seed. No real person, customer, tenant or work data
is in any file. The LLM agent's actions are real output of the model named in
each row, acting on those synthetic tickets. Before any upload,
`scripts/validate_dataset.py` scans every file for credential shapes, URLs,
host names, email addresses outside the reserved example domains, IP
addresses, phone numbers, local filesystem paths, provider request ids and the
owner's names, and fails if it finds one.

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
