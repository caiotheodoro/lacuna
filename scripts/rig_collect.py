"""Collect rig runs into one committed artifact.

    PYTHONPATH=src uv run python scripts/rig_collect.py

Raw runs (`artifacts/rig/runs/`) are not committed: they are large and hold container
addresses. This writes what a document may quote (AGENTS.md rule 2): for each run its
parameters, the stress events as planned and as done, the config hashes and image ids,
and the summary `lacuna.rig.witnesses.summarise` computes from the run's files now.
Nothing is estimated here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lacuna.envs.registry import ENVS
from lacuna.rig.integrity import file_problems, killed_for_good
from lacuna.rig.stress import Event
from lacuna.rig.witnesses import SERVICE_PLANE, summarise

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return out.stdout.strip()


def one(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = json.loads((run_dir / "events.json").read_text())
    gave_up, messages = {}, {}
    summary = summarise(run_dir, ENVS[manifest["params"].get("env", "backoffice")])
    killed = killed_for_good([Event(**e) for e in manifest["schedule"]])
    dead = {SERVICE_PLANE[s] for s in killed if s in SERVICE_PLANE}
    # Batches the SDK gave up on, as each originator's stderr says.
    for p in sorted((run_dir / "agents").glob("*.err")):
        err = p.read_text()
        gave_up[p.stem] = err.count("Failed to export spans batch")
        # What kind of failure the SDK reported for each refused export, by count.
        messages[p.stem] = {
            "transient_errors": err.count("Transient error"),
            "name_resolution_errors": sum(
                1 for ln in err.splitlines() if "Transient error" in ln and "NameResolution" in ln
            ),
            "connection_refused": sum(
                1
                for ln in err.splitlines()
                if "Transient error" in ln and "Connection refused" in ln
            ),
        }
    return {
        "run": manifest["run"],
        "commit": manifest["commit"],
        "dirty": manifest["dirty"],
        "source_sha256": manifest.get("source_sha256"),  # absent before the integrity gate
        "params": manifest["params"],
        "configs": manifest["configs"],
        "images": manifest["images"],
        "sessions_sha256": manifest["sessions_sha256"],
        "schedule": manifest["schedule"],
        "events_done": events["events"],
        "duration_s": events["duration_s"],
        "agent_exit_codes": events["agent_exit_codes"],
        "integrity": events.get("integrity"),  # likewise
        "span_batches_given_up": gave_up,
        "sdk_export_messages": messages,
        # The checks that need only the run's files, applied again with today's code.
        "integrity_from_files": file_problems(summary, dead),
        **summary,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "rig" / "runs")
    p.add_argument("--prefix", nargs="+", default=["cp-"], help="run name prefixes")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "rig" / "checkpoint.json")
    p.add_argument("--note", default="", help="a sentence about these runs, kept in the artifact")
    a = p.parse_args()
    dirs = sorted(d for pre in a.prefix for d in a.runs.glob(f"{pre}*") if d.is_dir())
    runs, set_aside = [], []
    for d in dirs:
        events = d / "events.json"
        record = json.loads(events.read_text()) if events.exists() else {}
        verdict = record.get("integrity")
        if "started_unix" not in record and "duration_s" not in record:
            # The driver never got as far as a run: nothing to summarise.
            why = verdict["problems"] if verdict else ["no events.json: the driver did not finish"]
            set_aside.append({"run": d.name, "problems": why})
            continue
        run = one(d)
        wrong = (verdict["problems"] if verdict else []) + run["integrity_from_files"]
        if wrong:
            # Kept and named, not analysed: its summary stays out of the artifact.
            set_aside.append({"run": d.name, "problems": wrong})
        else:
            runs.append(run)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "what": "runs of the study-5 rig (rig/compose.yaml) on one laptop; scripted "
                "sessions, no model; feasibility only, nothing registered",
                "generated_by": "scripts/rig_collect.py from artifacts/rig/runs/ (not committed)",
                "note": a.note,
                # The code that computed the summaries below, which is later than the
                # code that made the runs (each run names its own commit).
                "summarised_at": {
                    "commit": _git("rev-parse", "HEAD"),
                    "dirty": bool(_git("status", "--porcelain", "--", "src", "scripts")),
                },
                "runs_set_aside": set_aside,
                "runs": runs,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {a.out}: {len(runs)} runs, {len(set_aside)} set aside")
    for bad in set_aside:
        print(f"  set aside {bad['run']}: {'; '.join(bad['problems'])}")


if __name__ == "__main__":
    main()
