#!/usr/bin/env python3
"""Build the Hugging Face dataset of lacuna's capture histories into dataset/.

  uv run --group dataset python scripts/build_dataset.py

Reads artifacts/episodes/<run>/{episodes.jsonl,summary.json,receipt.json} and
the derived results, and writes dataset/ (gitignored). Uploads nothing: that
is scripts/publish_dataset.py, after scripts/validate_dataset.py.

Every run on disk needs an entry in RUNS. An unknown run fails the build
rather than being guessed into a phase. Every number on the card is read from
an artifact at build time (AGENTS.md rule 2).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "dataset"
REPO = "caiotheodoro/lacuna-capture-histories"

PHASES = ("scripted", "exploratory", "confirmatory")
ENVS = ("backoffice", "infra")

_FULL_W4 = (
    "W4 is the LLM auditor reading the full transcript. It reads every call "
    "verbatim, so its recall is 1 and it is a copy of the trace, not a witness "
    "(D14, D20): drop witness 3 before estimating. Every @4 number is superseded."
)
_V4 = "Agent deepseek-v4-flash, before the switch to v4.1-flash (D13)."
_POSTHOC = "Recorded with complete logs; loss is applied afterwards by lacuna.thinning."

# One entry per run. `fill` supplies a field an older summary.json did not
# record; a summary that records a different value fails the build. `caveat`
# may use {episodes} and {attempted}, filled from that run's summary.
RUNS: dict[str, dict] = {
    "scripted-l0": {
        "phase": "scripted",
        "env": "backoffice",
        # Predates the summary field. 7 is make_episode's default, and every
        # seed's ledger matches the plans of exactly 7 regenerated tickets.
        "fill": {"tickets": 7, "auditor_view": "full"},
        "caveat": (
            "Scripted agent and scripted auditor, no LLM: a harness check, not "
            "agent behaviour. The agent follows each ticket's plan with deliberate "
            "redo and redraft mistakes; a redone customer field first gets a "
            "trailing 'x', so some addresses end in example.orgx. Witness 3 is a "
            "scripted noisy copy of the transcript. Logging faults were applied at "
            "record time (see recorded_fault_rates), so records are already thinned."
        ),
    },
    "pilot-l0": {
        "phase": "exploratory",
        "env": "backoffice",
        # 7 tickets per D18's table; the summary predates the field.
        "fill": {"tickets": 7, "auditor_view": "full"},
        "caveat": (
            f"First live run. {_V4} {_FULL_W4} Logging faults were applied at "
            "record time (see recorded_fault_rates), so records are already "
            "thinned. Transcripts were not stored."
        ),
    },
    "long-30": {
        "phase": "exploratory",
        "env": "backoffice",
        "fill": {"auditor_view": "full"},
        "caveat": (
            f"Session-length run (D12, D18). {_V4} {_FULL_W4} {_POSTHOC} "
            "Transcripts were not stored. Source of the exploratory thinning "
            "results and of the estimator choice (D27)."
        ),
    },
    "long-60": {
        "phase": "exploratory",
        "env": "backoffice",
        "fill": {"auditor_view": "full"},
        "caveat": (
            f"Session-length run (D12, D18). {_V4} {_FULL_W4} {_POSTHOC} "
            "Transcripts were not stored. The agent skips a growing share of a "
            "long queue, so tickets overstate actions (D18)."
        ),
    },
    "long-120": {
        "phase": "exploratory",
        "env": "backoffice",
        "caveat": (
            f"Session-length run (D12, D18). {_V4} {_FULL_W4} {_POSTHOC} "
            "The agent skips more than half of a 120-ticket queue, so tickets "
            "overstate actions (D18)."
        ),
    },
    "narr-30": {
        "phase": "exploratory",
        "env": "backoffice",
        "caveat": (
            "Narration-only W4 (D14 fix, D20): the LLM auditor saw only what the "
            "agent said and reported nothing, so witness 3 has no records. The "
            "agent's narration is aggregate, not per action. Agent "
            f"deepseek-v4.1-flash. {_POSTHOC}"
        ),
    },
    "confirm": {
        "phase": "confirmatory",
        "env": "backoffice",
        "caveat": (
            "The pre-registered confirmatory run (docs/PRE-REGISTRATION.md, frozen "
            "before it ran). {episodes} of {attempted} episodes completed; the "
            "failed seeds are in failed_episodes and were not replaced or "
            "re-seeded (D30). Witness 3 is the scripted stand-in, not an LLM, and "
            f"is excluded from every estimate (D20). Agent deepseek-v4.1-flash. "
            f"{_POSTHOC} Read with D32: under loss that depends on the action, "
            "the method detects incompleteness but cannot measure it."
        ),
    },
}

EPISODE_KEYS = {"seed", "ledger", "records", "n_calls", "n_tool_errors", "transcript"}
SNIPPET = """\
import json
from collections import Counter

rows = [json.loads(line) for line in open("episodes/confirm.jsonl")]

# Oracle capture histories, standard library only: bit w is set if witness w
# saw the action. Cell 0 is the true count no witness saw.
hist = Counter()
for row in rows:
    seen = {e["id"]: 0 for e in row["ledger"]}
    for r in row["records"]:
        if r["witness"] < 3 and r["oracle"] in seen:  # witness 3 is not a witness
            seen[r["oracle"]] |= 1 << r["witness"]
    hist.update(seen.values())
print("histories", dict(sorted(hist.items())))

# One loss replicate of the held-out fin_mid configuration, signature linkage
# (what an auditor without ledger ids can compute), then the estimators.
import numpy as np
from lacuna.envs.kernel import LedgerEntry
from lacuna.thinning import HELDOUT, estimate_all, tables, thin
from lacuna.witnesses import Record

rng = np.random.default_rng(4242)
table, per_op, n_true = 0, {}, 0
for row in rows:
    ledger = [LedgerEntry(**e) for e in row["ledger"]]
    records = [Record(**r) for r in row["records"]]
    t, p = tables(ledger, thin(records, HELDOUT[0], rng), "signature")
    table = table + t
    for op, tt in p.items():
        per_op[op] = per_op.get(op, 0) + tt
    n_true += row["n_true"]
est = estimate_all(table, per_op)  # uses witnesses 0-2 only
for name in ("union_count", "ll_bic"):
    e = est[name]
    print(f"{name:12} {e.point:7.1f} [{e.lo:.1f}, {e.hi:.1f}]  true N={n_true}")
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
        seen += sum(int(c) for h, c in enumerate(t) if h >> 3 & 1)
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


def build_run(run: str, src: Path) -> tuple[list[dict], dict]:
    meta = run_meta(run)
    summary = json.loads((src / "summary.json").read_text())
    receipt_path = src / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    fields = resolve(run, meta, summary)

    raw = [json.loads(line) for line in (src / "episodes.jsonl").open() if line.strip()]
    rows = []
    for e in raw:
        extra = set(e) - EPISODE_KEYS
        if extra:
            raise BuildError(f"{run} seed {e.get('seed')}: unknown episode keys {sorted(extra)}")
        t = e.get("transcript")
        rows.append(
            {
                "run": run,
                "seed": e["seed"],
                "model": summary["model"],
                "env": meta["env"],
                "tickets": fields["tickets"],
                "phase": meta["phase"],
                "auditor_view": fields["auditor_view"],
                "n_true": len(e["ledger"]),
                "ledger": e["ledger"],
                "records": e["records"],
                # A JSON string, so every config has one fixed schema whatever
                # keys a provider put in its messages.
                "transcript": None if t is None else json.dumps(t, separators=(",", ":")),
                "n_calls": e["n_calls"],
                "n_tool_errors": e["n_tool_errors"],
            }
        )

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
        "failed_episodes": [{"seed": f["seed"], "error": f["error"]} for f in failed],
        "stopped": summary.get("stopped") or None,
        "seed0": summary["seed0"],
        "n_true": summary["n_true"],
        "recorded_fault_rates": summary["fault_rates"],
        "spend_usd": receipt["total_usd"] if receipt else None,
        "llm_calls": len(receipt["calls"]) if receipt else None,
        "transcripts_stored": all(r["transcript"] is not None for r in rows),
        "w4_recall": _w4_recall(raw),
        "oracle_table": summary["oracle_table"],
        "signature_table": summary["signature_table"],
        "caveat": meta["caveat"].format(episodes=summary["episodes"], attempted=attempted),
        "source": f"artifacts/episodes/{run}/",
    }
    return rows, run_row


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def _copy_results(out: Path) -> list[str]:
    copied = []
    for src in sorted((ARTIFACTS / "thinning").glob("*.json")):
        dst = out / "results" / "thinning" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied.append(f"results/thinning/{src.name}")
    for name in ("hypotheses.json", "sim_coverage.json"):
        shutil.copyfile(ARTIFACTS / name, out / "results" / name)
        copied.append(f"results/{name}")
    return copied


def _clear(out: Path) -> None:
    """Remove a previous build, refusing anything that is not one."""
    if not out.exists():
        return
    readme = out / "README.md"
    if any(out.iterdir()) and not (readme.exists() and REPO in readme.read_text()):
        raise BuildError(f"{out} exists and is not a previous build; not deleting it")
    shutil.rmtree(out)


def build(out: Path = OUT) -> dict:
    _clear(out)
    out.mkdir(parents=True)
    on_disk = sorted(p.name for p in (ARTIFACTS / "episodes").iterdir() if p.is_dir())
    for run in on_disk:
        run_meta(run)  # every unknown run fails before anything is written
    order = [r for r in RUNS if r in on_disk]

    runs = []
    for run in order:
        rows, run_row = build_run(run, ARTIFACTS / "episodes" / run)
        _dump_jsonl(out / "episodes" / f"{run}.jsonl", rows)
        runs.append(run_row)
    _dump_jsonl(out / "runs.jsonl", runs)
    results = _copy_results(out)
    (out / ".gitattributes").write_text("*.jsonl filter=lfs diff=lfs merge=lfs -text\n")
    (out / "README.md").write_text(card(runs, out, git_commit()))
    return {
        "runs": order,
        "declared_not_on_disk": [r for r in RUNS if r not in on_disk],
        "episodes": {r["run"]: r["episodes"] for r in runs},
        "results": results,
    }


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------

_STR = {"dtype": "string"}
_INT = {"dtype": "int64"}


def _features(cols: list[tuple[str, dict]]) -> list[dict]:
    return [{"name": n, **t} for n, t in cols]


EPISODE_FEATURES = _features(
    [
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
        (
            "records",
            {
                "list": _features(
                    [
                        ("witness", _INT),
                        ("op", _STR),
                        ("resource", _STR),
                        ("value", _STR),
                        ("oracle", _INT),
                    ]
                )
            },
        ),
        ("transcript", _STR),
        ("n_calls", _INT),
        ("n_tool_errors", _INT),
    ]
)

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
            {
                "struct": _features(
                    [
                        ("p_gateway_drop", {"dtype": "float64"}),
                        ("p_trace_drop", {"dtype": "float64"}),
                    ]
                )
            },
        ),
        ("spend_usd", {"dtype": "float64"}),
        ("llm_calls", _INT),
        ("transcripts_stored", {"dtype": "bool"}),
        ("w4_recall", {"dtype": "float64"}),
        ("oracle_table", {"sequence": "int64"}),
        ("signature_table", {"sequence": "int64"}),
        ("caveat", _STR),
        ("source", _STR),
    ]
)


def front_matter(runs: list[dict]) -> str:
    import yaml

    n = sum(r["episodes"] for r in runs)
    size = "n<1K" if n < 1000 else "1K<n<10K"
    configs = [
        {"config_name": "episodes", "data_files": "episodes/*.jsonl", "default": True},
        {"config_name": "runs", "data_files": "runs.jsonl"},
    ] + [{"config_name": r["run"], "data_files": f"episodes/{r['run']}.jsonl"} for r in runs]
    info = [
        {"config_name": c["config_name"], "features": RUN_FEATURES}
        if c["config_name"] == "runs"
        else {"config_name": c["config_name"], "features": EPISODE_FEATURES}
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


def card(runs: list[dict], out: Path, commit: str) -> str:
    by = {r["run"]: r for r in runs}
    conf = by["confirm"]
    hyp = json.loads((out / "results" / "hypotheses.json").read_text())
    thin_dir = out / "results" / "thinning"
    mid, mid_hidden = _est(thin_dir / "confirm.json", "fin_mid/signature")
    rho, rho_hidden = _est(thin_dir / "confirm.json", "fin_mid_rho100/signature")
    mnar, mnar_hidden = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar/signature")
    mnar_rho, _ = _est(thin_dir / "confirm-mnar.json", "fin_mid_mnar_rho100/signature")
    n_eps = sum(r["episodes"] for r in runs)
    n_actions = sum(r["n_true"] for r in runs)
    spend = sum(r["spend_usd"] or 0 for r in runs)
    full_w4 = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "full"]
    narr = [r for r in runs if r["auditor"] == "llm" and r["auditor_view"] == "narration"]
    failures = "; ".join(
        f"seed {f['seed']}: {f['error'].split(':')[0]}" for f in conf["failed_episodes"]
    )

    reps = json.loads((thin_dir / "confirm.json").read_text())["reps"]

    def fmt(v: float | None, p: int = 2) -> str:
        return "unbounded" if v is None else f"{v:.{p}f}"

    def usd(v: float | None) -> str:
        return "—" if v is None else f"{v:.2f}"

    def recall(rs: list[dict]) -> str:
        return ", ".join(f"{r['w4_recall']:.4f} in `{r['run']}`" for r in rs)

    run_table = "\n".join(
        f"| `{r['run']}` | {r['phase']} | {r['model'] or 'scripted'} | {r['tickets']} | "
        f"{r['episodes']} | {r['n_true']} | {r['w4_recall']:.4f} | "
        f"{'yes' if r['transcripts_stored'] else 'no'} | {usd(r['spend_usd'])} |"
        for r in runs
    )
    hyp_table = "\n".join(
        f"| {name} | {'PASS' if h['passed'] else 'FAIL'} | {h['detail']} |"
        for name, h in hyp["hypotheses"].items()
    )

    body = f"""
# lacuna capture histories

How many consequential actions did an agent take that no log line shows? This
dataset holds the witness records [lacuna](#citation) uses to answer that by
capture-recapture: {n_eps} episodes across {len(runs)} runs, {n_actions}
ground-truth actions, and for each action what every witness recorded.

**Synthetic data, not production-validated.** The environment, its customers
and its tickets are generated from a seed; the agent text is real model output.

Built by `scripts/build_dataset.py` from lacuna commit `{commit}`. Every
result on this card is read from a file in this dataset at build time.

## What the data is

An agent works a queue of support tickets in a synthetic back office:
refunds, cancellations, customer updates, emails and one report file. The
environment kernel keeps a hidden ledger of every consequential action. That
ledger is the ground truth, and no witness reads it. Each witness writes its
own record of what happened, and capture-recapture turns the overlaps between
those records into an estimate of the actions no witness saw.

- `episodes` (and one config per run): one row per episode, with the ledger,
  every witness record and, where stored, the transcript.
- `runs`: one row per run, with model, spend, failed episodes, the pooled
  capture-history tables and the caveat that governs the run.
- `results/`: derived results, copied byte for byte. The thinning summaries
  (`results/thinning/*.json`), the confirmatory verdicts
  (`results/hypotheses.json`) and the simulator coverage sweep
  (`results/sim_coverage.json`).

| run | phase | model | tickets | episodes | N | W4 recall | transcripts | spend (USD) |
|---|---|---|---|---|---|---|---|---|
{run_table}

Total LLM spend {spend:.2f} USD, from each run's receipt. W4 recall is the
share of ledger actions witness 3 reported, by oracle linkage.

## The three witnesses, and why the LLM auditor is not one

| id | witness | what it sees | blind spots |
|---|---|---|---|
| 0 | trace | the agent harness's own log file | lines lost to logging faults |
| 1 | gateway | every tool call through the router | bypassable through code execution |
| 2 | env | end-of-shift state diff plus outbound mail | repeats, reverts, ops outside its view |
| 3 | auditor | an LLM rebuilding actions from the transcript | **not a witness: drop it** |

Witness 3 is in `records` because the runs recorded it, and it is excluded
from every estimate in `results/` (D20). An auditor reading the agent's own
transcript is either a copy of the trace or empty. With the full transcript it
reads every call verbatim: W4 recall {recall(full_w4)}. With narration only it
reports nothing: {recall(narr)}.
The agent narrates in aggregate ("nine refunds issued"), and a count cannot be
overlapped with another witness's record. In `confirm` and `scripted-l0`,
witness 3 is a scripted stand-in that copies the transcript with misses.

## Field reference

`episodes`:

| field | type | meaning |
|---|---|---|
| `run` | string | run name; see `runs` |
| `seed` | int | episode seed; `lacuna.envs.tasks.make_episode(seed, tickets)` regenerates it |
| `model` | string or null | agent model; null for the scripted agent |
| `env` | string | environment; `backoffice` for every run here |
| `tickets` | int | tickets in the queue, plus one closing report task |
| `phase` | string | `scripted`, `exploratory` or `confirmatory` |
| `auditor_view` | string | what witness 3 read: `full` transcript or `narration` only |
| `n_true` | int | ground-truth actions, the ledger length |
| `ledger` | list | ground truth: `id`, `op`, `resource`, `value`, `step` |
| `records` | list | witness claims: `witness` (0-3), `op`, `resource`, `value`, `oracle` |
| `transcript` | string or null | JSON-encoded OpenAI-format messages; null where not stored |
| `n_calls` | int | tool calls the agent made |
| `n_tool_errors` | int | tool results that were errors |

`record.oracle` is the ledger id when the record came from infrastructure that
can know it. It is for oracle linkage only and is never shown to an
estimator. Witness 3 records never carry one. `send_email` and `write_file`
values are content digests. `transcript` includes the provider's
`reasoning_content` where it returned one; read it with
`json.loads(row["transcript"])`.

`runs`: `run`, `phase`, `env`, `agent`, `auditor`, `model`, `auditor_view`,
`tickets`, `episodes`, `episodes_attempted`, `failed_episodes` (seed and the
recorded error), `stopped`, `seed0`, `n_true`, `recorded_fault_rates`,
`spend_usd`, `llm_calls`, `transcripts_stored`, `w4_recall`, `oracle_table`,
`signature_table` (16 cells, bit w for witness w; cell 0 of `oracle_table` is
the true unseen count), `caveat`, `source`.

## Rebuild a capture-history table and rerun an estimator

Run from the dataset root. The first half needs only the standard library;
the second needs the lacuna package on the path (`PYTHONPATH=src` in a lacuna
checkout).

```python
{SNIPPET}```

Every run except `pilot-l0` and `scripted-l0` was recorded with complete logs
(`recorded_fault_rates` 0), so the raw histories have nothing hidden, and
`thin` applies logging loss afterwards. `pilot-l0` and `scripted-l0` were
thinned at record time. `scripts/thin_episodes.py` in lacuna repeats the loop
above over 100 replicates per configuration to produce `results/thinning/`.

## Exploratory vs confirmatory

**Confirmatory: `confirm` only.** Pre-registered in `docs/PRE-REGISTRATION.md`,
frozen before the run, and tested only on loss configurations held out of
every exploratory run (`fin_mid`, `fin_mid_rho100`). {conf["episodes"]}
episodes, N = {conf["n_true"]}, {hyp["linkage"]} linkage. Source:
`results/hypotheses.json`.

| hypothesis | verdict | detail |
|---|---|---|
{hyp_table}

At `fin_mid`, {mid_hidden} actions reach no witness on average over
{reps} replicates. `union_count` covers N in
{fmt(mid["union_count"]["coverage"])} of replicates and `ll_bic` in
{fmt(mid["ll_bic"]["coverage"])}, mean width {fmt(mid["ll_bic"]["mean_width"], 1)}.
At `fin_mid_rho100` ({rho_hidden} hidden), with gateway and trace failing
together, `ll_independence` covers {fmt(rho["ll_independence"]["coverage"])}
and `ll_bic` {fmt(rho["ll_bic"]["coverage"])}. Source:
`results/thinning/confirm.json`.

**Exploratory: every other run and result.** That includes the `long-*`
thinning results, from which the estimator was chosen (D27), and
`results/thinning/confirm-mnar.json`, built after the confirmatory run (D31).

**Scripted: `scripted-l0`.** No LLM; a check that the harness and the
witnesses work.

## Known caveats

- **Degenerate W4 (D14, D20).** `pilot-l0` and `long-*` used the
  full-transcript auditor, whose recall is 1. Every four-witness number from
  those runs is superseded and supports no claim.
- **Incomplete confirmatory run (D30).** {conf["episodes"]} of
  {conf["episodes_attempted"]} episodes completed ({failures}). The failed
  seeds were not replaced or re-seeded, so N is {conf["n_true"]} rather than
  the size the power calculation targeted.
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
  deepseek-v4-flash, `narr-30` and `confirm` on deepseek-v4.1-flash. Comparisons
  across them are cross-model.
- **Long queues are abandoned (D18).** Actions per ticket fall as the queue
  grows, so `tickets` overstates the work in `long-60` and `long-120`.
- **Benign loss only (D10, D11).** No adversarial agent was run live; every
  adversarial number in lacuna is from the simulator
  (`results/sim_coverage.json`). One agent family, one environment.
- **Linkage error is real.** Signature linkage leaves content digests out, so
  two different emails to one address in one episode look like one action seen
  twice. It is measured against oracle linkage, not hidden.
- `results/thinning/long-30.json` and `long-30.fixed-episodes.json` are
  byte-identical: the same fixed-episode result under two names. The headline
  exploratory numbers use `long-30.bootstrap.json` (D23, D26).

Each run's own caveat is in `runs.caveat`.

## Synthetic data

Every customer, order and ticket is generated from the episode seed by
`lacuna.envs.tasks`: names are "Customer i", email addresses are at
example.com and example.org, phone numbers start 555, and street names come
from a fixed list of six. No real person, customer, tenant or work data is in
any file. The agent and auditor text is real output of the model named in
each row, acting on those synthetic tickets. Before any upload,
`scripts/validate_dataset.py` scans every file for key-shaped strings, email
addresses outside the reserved example domains, and local filesystem paths,
and fails if it finds one.

## Citation

Citation details are a placeholder until a paper or tagged release exists.

```bibtex
@misc{{lacuna_capture_histories,
  title  = {{lacuna capture histories: witness records for counting unlogged agent actions}},
  author = {{Theodoro, Caio}},
  year   = {{2026}},
  note   = {{Hugging Face dataset {REPO}, built from lacuna commit {commit}}}
}}
```

Apache-2.0.
"""
    return front_matter(runs) + body


def main() -> None:
    summary = build()
    for run, n in summary["episodes"].items():
        print(f"  {run:12} {n:4d} episodes")
    if summary["declared_not_on_disk"]:
        print(f"  declared in RUNS, not on disk: {summary['declared_not_on_disk']}")
    print(f"  results: {len(summary['results'])} files")
    print(f"wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
