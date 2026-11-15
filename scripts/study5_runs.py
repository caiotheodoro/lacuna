#!/usr/bin/env python3
"""Study 5's runs: which ones there are, whether they can be made, and making them.

    PYTHONPATH=src uv run python scripts/study5_runs.py --list
    PYTHONPATH=src uv run python scripts/study5_runs.py --check
    PYTHONPATH=src uv run python scripts/study5_runs.py --register RUN_DIR/manifest.json
    PYTHONPATH=src uv run python scripts/study5_runs.py --go [--limit N]

The plan is a function of nothing: 420 runs, each with a name, a condition, a pool and
its seeds. A run replays 24 sessions drawn from one pool by the run's seed, under the
stress schedule its condition and seed give (`lacuna.rig.stress`).

`--check` makes no run. It draws every run's sessions and schedule and says which runs
the driver would refuse (sessions that clash, a kill drawn for after the last call). It
must print nothing before the registration is frozen.

`--register` writes `artifacts/study5/registered.json` from the manifest of one run
made at the frozen commit and the pools' hashes. `scripts/study5_analysis.py
--registered` then sets aside any run whose manifest differs.

`--go` makes the runs that do not exist yet, one at a time, in an order that spreads
every condition over the whole sequence. A run is never made twice: a folder that
exists is left alone, whatever it holds. Needs docker.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lacuna.rig.manifest import registered, sha256_file
from lacuna.rig.sessions import Session, clashes, draw, load, planned_end
from lacuna.rig.stress import misfit, plan

ROOT = Path(__file__).resolve().parents[1]
POOLS_DIR = ROOT / "artifacts" / "study5" / "pools"
RUNS_DIR = ROOT / "artifacts" / "rig" / "runs"
REGISTERED = ROOT / "artifacts" / "study5" / "registered.json"

# condition -> (runs, first seed). A run's stress seed and draw seed are both
# first seed + its index, so no two runs of the study share a seed.
CONDITIONS: dict[str, tuple[int, int]] = {"s0": (60, 70000), "s1": (300, 71000), "s2": (60, 72000)}
POOLS = ("ds", "qw", "mix")
ENV = "backoffice_churn"
SESSIONS = 24
STAGGER = 100.0


def runs() -> list[dict]:
    """Every run of the study, in the order they are made.

    The order interleaves the conditions in proportion, so a slow drift of the machine
    over the hours the runs take is not a difference between conditions.
    """
    out = []
    for condition, (n, seed0) in CONDITIONS.items():
        for i in range(n):
            out.append(
                {
                    "run": f"s5-{condition}-{i:03d}",
                    "condition": condition,
                    "index": i,
                    "seed": seed0 + i,
                    "pool": POOLS[i % len(POOLS)],
                    "_at": (i + 0.5) / n,
                }
            )
    out.sort(key=lambda r: (r["_at"], r["condition"]))
    return [{k: v for k, v in r.items() if k != "_at"} for r in out]


def pool_file(pools_dir: Path, pool: str) -> Path:
    return pools_dir / pool / "sessions.json"


def argv(run: dict, pools_dir: Path, out: Path) -> list[str]:
    """The driver's arguments for one run. Everything else is the driver's default."""
    return [
        sys.executable,
        str(ROOT / "scripts" / "rig_run.py"),
        "--run", run["run"],
        "--out", str(out),
        "--env", ENV,
        "--sessions-file", str(pool_file(pools_dir, run["pool"])),
        "--sessions", str(SESSIONS),
        "--draw-seed", str(run["seed"]),
        "--stagger", str(STAGGER),
        "--condition", run["condition"],
        "--stress-seed", str(run["seed"]),
    ]  # fmt: skip


def refusals(run: dict, pool: list[Session]) -> list[str]:
    """Why the driver would refuse this run, from its sessions and schedule alone."""
    sessions = draw(pool, SESSIONS, run["seed"])
    out = list(clashes(sessions))
    names = {s.session for s in sessions}
    if len(names) != SESSIONS:
        out.append(f"the pool gives {len(names)} sessions, the run needs {SESSIONS}")
    end = planned_end(sessions, STAGGER)
    late = misfit(plan(run["condition"], run["seed"]), end)
    out += [
        f"{e.service} is killed at {e.at_s} s, the last call is due at {end:.1f} s" for e in late
    ]
    return out


def check(pools_dir: Path) -> dict[str, list[str]]:
    """Run name -> why it cannot be made, for every run that cannot."""
    pools = {p: load(pool_file(pools_dir, p).read_text()) for p in POOLS}
    bad = {}
    for run in runs():
        why = refusals(run, pools[run["pool"]])
        if why:
            bad[run["run"]] = why
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--list", action="store_true", help="print the plan as JSON lines")
    what.add_argument("--check", action="store_true", help="say which runs would be refused")
    what.add_argument("--register", type=Path, metavar="MANIFEST")
    what.add_argument("--go", action="store_true", help="make the runs that do not exist")
    p.add_argument("--pools", type=Path, default=POOLS_DIR)
    p.add_argument("--out", type=Path, default=RUNS_DIR)
    p.add_argument("--registered", type=Path, default=REGISTERED)
    p.add_argument("--limit", type=int, default=None, help="with --go: make at most this many")
    a = p.parse_args()

    if a.list:
        for run in runs():
            print(json.dumps(run))
        return
    if a.check:
        bad = check(a.pools)
        for name, why in bad.items():
            print(f"{name}: {'; '.join(why)}")
        sys.exit(1 if bad else 0)
    if a.register:
        if a.registered.exists():
            sys.exit(f"{a.registered} exists; a registration is written once")
        pools = {q: sha256_file(pool_file(a.pools, q)) for q in POOLS}
        reg = registered(json.loads(a.register.read_text()), pools)
        a.registered.write_text(json.dumps(reg, indent=1) + "\n")
        print(f"wrote {a.registered}")
        return
    made = 0
    for run in runs():
        if (a.out / run["run"]).exists():
            continue
        if a.limit is not None and made >= a.limit:
            break
        code = subprocess.run(argv(run, a.pools, a.out), stdout=subprocess.DEVNULL).returncode
        made += 1
        print(f"{run['run']} exit={code}", flush=True)


if __name__ == "__main__":
    main()
