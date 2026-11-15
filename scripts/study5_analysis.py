#!/usr/bin/env python3
"""Score a set of study-5 rig runs and write one artifact. Calls no model, starts no container.

    PYTHONPATH=src uv run python scripts/study5_analysis.py --prefix pilot- \\
        --out artifacts/study5/pilot.json

Every run folder under `--runs` whose name starts with a `--prefix` is read. A run that
failed its integrity check, that the driver did not finish, or (with `--registered`)
whose manifest differs from the registered one, is listed under
`runs_set_aside` with the reason and is in no figure. For the rest the artifact holds
the run's tables, every candidate estimate, the floors and the baselines, scored against
that run's ledger, and per condition the coverage of each and the paired comparisons
with the baselines (`scripts/study5_lib.py`).

Nothing here is a verdict. `scripts/study5_verdicts.py` applies the registered criteria.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import study5_lib as lib  # noqa: E402

from lacuna.envs.registry import ENVS  # noqa: E402
from lacuna.rig.manifest import registered_problems  # noqa: E402

# The study's own runs (`scripts/study5_runs.py`). They are never scored without the
# registration that says what makes one.
STUDY_PREFIX = "s5-"


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return out.stdout.strip()


def collect(run_dirs: list[Path], reg: dict | None = None) -> tuple[list[dict], list[dict]]:
    """(scored runs, runs set aside with the reason).

    With `reg`, the registration's record of what makes a run
    (`lacuna.rig.manifest.registered`), a run whose manifest differs is set aside too.
    """
    runs, aside = [], []
    for d in run_dirs:
        try:
            events = json.loads((d / "events.json").read_text())
            manifest = json.loads((d / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            aside.append({"run": d.name, "why": f"unreadable: {type(e).__name__}"})
            continue
        integrity = events.get("integrity") or {"ok": False, "problems": ["no integrity result"]}
        if not integrity["ok"]:
            aside.append({"run": d.name, "why": integrity["problems"]})
            continue
        differs = registered_problems(manifest, reg) if reg is not None else []
        if differs:
            aside.append({"run": d.name, "why": differs})
            continue
        env = ENVS[manifest["params"].get("env", "backoffice")]
        runs.append(lib.run_record(d, env, boot_seed=int(manifest["params"]["stress_seed"])))
    return runs, aside


def by_condition(runs: list[dict], made: dict[str, int], min_runs: int) -> dict:
    out = {}
    for condition in sorted({r["condition"] for r in runs}):
        mine = [r for r in runs if r["condition"] == condition]
        names = sorted(mine[0]["estimates"])
        out[condition] = {
            "gates": lib.gates(mine, made.get(condition, len(mine)), min_runs),
            "estimators": {name: lib.coverage(mine, name) for name in names},
            "paired": {
                name: {b: lib.paired(mine, name, b) for b in lib.BASELINES if b != name}
                for name in names
                if name not in lib.FLOORS and not name.startswith("seq_gap")
            },
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "rig" / "runs")
    p.add_argument("--prefix", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-runs", type=int, default=270, help="the underpowered gate, per condition")
    p.add_argument(
        "--registered", type=Path, default=None,
        help="the registration's record of what makes a run; a run that differs is set aside",
    )  # fmt: skip
    p.add_argument("--note", default="")
    a = p.parse_args()

    dirs = sorted(
        d for d in a.runs.iterdir() if d.is_dir() and any(d.name.startswith(x) for x in a.prefix)
    )
    if not dirs:
        sys.exit(f"no run under {a.runs} starts with {a.prefix}")
    if a.registered is None and any(x.startswith(STUDY_PREFIX) for x in a.prefix):
        sys.exit(f"runs named {STUDY_PREFIX}* are the study's: --registered is required")
    reg = json.loads(a.registered.read_text()) if a.registered else None
    runs, aside = collect(dirs, reg)
    made: dict[str, int] = {}
    for d in dirs:
        try:
            c = json.loads((d / "manifest.json").read_text())["params"]["condition"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
        made[c] = made.get(c, 0) + 1
    artifact = {
        "note": a.note,
        "summarised_at": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--", "src", "rig", "scripts")),
        },
        "prefixes": a.prefix,
        "registered": a.registered.name if a.registered else None,
        "runs_read": len(dirs),
        "runs_set_aside": aside,
        "by_condition": by_condition(runs, made, a.min_runs),
        "runs": runs,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(artifact, indent=1) + "\n")
    for condition, block in artifact["by_condition"].items():
        g = block["gates"]
        print(
            f"{condition}: {g['runs_passing_integrity']} runs, union missed "
            f"{g['mean_union_missed_share']}, telemetry {g['mean_telemetry_missed_share']}"
        )
        for name, c in block["estimators"].items():
            print(
                f"  {name:14s} coverage {c['covered']}/{c['runs']}  unbounded {c['unbounded']}"
                f"  median |err| {c['median_abs_error']}  mean err {c['mean_error']}"
            )
    print(f"{len(aside)} runs set aside; wrote {a.out}")


if __name__ == "__main__":
    main()
