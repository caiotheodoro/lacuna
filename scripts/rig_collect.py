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
from pathlib import Path

from lacuna.rig.witnesses import summarise

ROOT = Path(__file__).resolve().parents[1]


def one(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = json.loads((run_dir / "events.json").read_text())
    gave_up = {}
    # Batches the SDK gave up on, as each originator's stderr says.
    for p in sorted((run_dir / "agents").glob("*.err")):
        gave_up[p.stem] = p.read_text().count("Failed to export spans batch")
    return {
        "run": manifest["run"],
        "commit": manifest["commit"],
        "dirty": manifest["dirty"],
        "source_sha256": manifest["source_sha256"],
        "params": manifest["params"],
        "configs": manifest["configs"],
        "images": manifest["images"],
        "sessions_sha256": manifest["sessions_sha256"],
        "schedule": manifest["schedule"],
        "events_done": events["events"],
        "duration_s": events["duration_s"],
        "agent_exit_codes": events["agent_exit_codes"],
        "integrity": events["integrity"],
        "span_batches_given_up": gave_up,
        **summarise(run_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "rig" / "runs")
    p.add_argument("--prefix", nargs="+", default=["cp-"], help="run name prefixes")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "rig" / "checkpoint.json")
    a = p.parse_args()
    dirs = sorted(
        d for pre in a.prefix for d in a.runs.glob(f"{pre}*") if (d / "events.json").exists()
    )
    runs = [one(d) for d in dirs]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "what": "runs of the study-5 rig (rig/compose.yaml) on one laptop; scripted "
                "sessions, no model; feasibility only, nothing registered",
                "generated_by": "scripts/rig_collect.py from artifacts/rig/runs/ (not committed)",
                "runs": runs,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {a.out.relative_to(ROOT)}: {len(runs)} runs")


if __name__ == "__main__":
    main()
