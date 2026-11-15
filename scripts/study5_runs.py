#!/usr/bin/env python3
"""Study 5's runs: which ones there are, whether they can be made, and making them.

    PYTHONPATH=src uv run python scripts/study5_runs.py --list
    PYTHONPATH=src uv run python scripts/study5_runs.py --check
    PYTHONPATH=src uv run python scripts/study5_runs.py --sample
    PYTHONPATH=src uv run python scripts/study5_runs.py --register
    PYTHONPATH=src uv run python scripts/study5_runs.py --go [--limit N]
    PYTHONPATH=src uv run python scripts/study5_runs.py --seal

The plan is a function of nothing: 480 runs, each with a name, a condition, a pool and
its seeds. A run replays 24 sessions drawn from one pool by the run's seed, under the
stress schedule its condition and seed give (`lacuna.rig.stress`).

`--check` makes no run. It draws every run's sessions and schedule and says which runs
the driver would refuse (sessions that clash, a kill drawn for after the last call). It
must print nothing before the registration is frozen.

`--sample` makes one unstressed run that is not in the plan, `reg-sample`, the way
`--go` makes a run. `--register` writes `artifacts/study5/registered.json` from that
run's manifest, the pools' hashes and the plan with the hash of the sessions each run
will replay. It refuses a sample that is not what the plan's runs will be.
`scripts/study5_analysis.py --registered` then sets aside any run whose manifest
differs, or that is not the run of the plan its name says.

`--go` makes the runs that do not exist yet, one at a time, in an order that spreads
every condition over the whole sequence. A run is never made twice: a folder that
exists is left alone, whatever it holds. It starts only from a clean tree with the
registered driver, lock file and pools, and it stops at the first run that the driver
refuses or whose manifest is not the registered one, so a wrong setup costs one run
and not the study. Needs docker.

`--seal` writes `artifacts/study5/runs-sha256.json`, one hash per run folder, once. It
is committed before `scripts/study5_analysis.py` is run on the study's runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from lacuna.rig.manifest import registered, registered_problems, sha256_file, tree_hash
from lacuna.rig.sessions import Session, clashes, draw, dump, load, planned_end
from lacuna.rig.stress import misfit, plan

ROOT = Path(__file__).resolve().parents[1]
POOLS_DIR = ROOT / "artifacts" / "study5" / "pools"
RUNS_DIR = ROOT / "artifacts" / "rig" / "runs"
REGISTERED = ROOT / "artifacts" / "study5" / "registered.json"
SEAL = ROOT / "artifacts" / "study5" / "runs-sha256.json"

# condition -> (runs, first seed). A run's stress seed and draw seed are both
# first seed + its index, so no two runs of the study share a seed.
CONDITIONS: dict[str, tuple[int, int]] = {
    "s0": (60, 70000),
    "s1": (300, 71000),
    "s2": (60, 72000),
    "s3": (60, 73000),
}
POOLS = ("ds", "glm", "mix")
ENV = "backoffice_churn"
SESSIONS = 24
STAGGER = 100.0
# Sessions asked of each pool, and the fewest a pool may have and still be used. With
# 72, two runs of a pool share about a third of their sessions.
POOL_SESSIONS = 72
MIN_POOL = 60
# Made like a run of the plan, and not one: the registration reads its manifest.
SAMPLE = {"run": "reg-sample", "condition": "s0", "index": -1, "seed": 69999, "pool": "ds"}


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


def load_pools(pools_dir: Path) -> dict[str, list[Session]]:
    missing = [str(pool_file(pools_dir, p)) for p in POOLS if not pool_file(pools_dir, p).exists()]
    if missing:
        sys.exit("no such pool file: " + ", ".join(missing))
    return {p: load(pool_file(pools_dir, p).read_text()) for p in POOLS}


def folder_problems(pools_dir: Path) -> dict[str, list[str]]:
    """Pool name -> why its folder is not a finished recording. Reads only what the
    recorder wrote about the recording itself, never a session."""
    bad = {}
    for name in POOLS:
        folder = pools_dir / name
        why = []
        try:
            meta = json.loads((folder / "pool.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            meta = None
            why.append("it has no readable pool.json")
        if meta is not None:
            if meta.get("stopped"):
                why.append(f"its recording ended early ({meta['stopped']})")
            if "spent_usd" not in meta:
                why.append("its pool.json has no spent_usd: the recorder did not finish")
        if not (folder / "receipt.json").exists():
            why.append("it has no receipt.json")
        if why:
            bad[f"pool {name}"] = why
    return bad


def check(pools: dict[str, list[Session]]) -> dict[str, list[str]]:
    """Run name -> why it cannot be made, for every run that cannot.

    A pool with fewer than `MIN_POOL` sessions is listed under its own name.
    """
    bad = {}
    for name, pool in pools.items():
        n = len({s.session for s in pool})
        if n < MIN_POOL:
            bad[f"pool {name}"] = [f"it has {n} sessions, fewer than {MIN_POOL}"]
    for run in runs():
        why = refusals(run, pools[run["pool"]])
        if why:
            bad[run["run"]] = why
    return bad


def planned(pools: dict[str, list[Session]]) -> dict[str, dict]:
    """Every run of the plan with the hash of the sessions file the driver will write."""
    out = {}
    for run in runs():
        text = dump(draw(pools[run["pool"]], SESSIONS, run["seed"]))
        out[run["run"]] = {
            "condition": run["condition"],
            "seed": run["seed"],
            "pool": run["pool"],
            "sessions_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    return out


def sample_problems(
    manifest: dict, integrity: dict | None, pool_hashes: dict[str, str]
) -> list[str]:
    """Why a run's manifest cannot stand for the plan's runs. Empty means it can."""
    out = []
    if manifest.get("dirty") is not False:
        out.append("it was not made from a clean tree")
    if not (integrity or {}).get("ok"):
        out.append("it did not pass the integrity check")
    params = manifest.get("params") or {}
    for key, want in (("env", ENV), ("sessions", SESSIONS), ("stagger", STAGGER)):
        if params.get(key) != want:
            out.append(f"its {key} is {params.get(key)!r}, the plan's is {want!r}")
    hashes = [p.get("sha256") for p in manifest.get("pools") or [] if isinstance(p, dict)]
    if len(hashes) != 1 or hashes[0] not in pool_hashes.values():
        out.append("it did not replay exactly one of the study's pools")
    for what in ("configs", "images"):
        if not manifest.get(what):
            out.append(f"it names no {what}")
    return out


def preflight(reg_path: Path, pools_dir: Path) -> list[str]:
    """Why `--go` must not start. Empty means it may."""
    if not reg_path.exists():
        return [f"{reg_path} does not exist: nothing is registered"]
    reg = json.loads(reg_path.read_text())
    out = []
    if shutil.which("docker") is None:
        out.append("docker is not installed")
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "rig", "scripts", "uv.lock"],
        capture_output=True, text=True,
    )  # fmt: skip
    if status.returncode != 0 or status.stdout.strip():
        out.append("the tree has uncommitted changes under src, rig, scripts or uv.lock")
    if sha256_file(ROOT / "scripts" / "rig_run.py") != reg["driver_sha256"]:
        out.append("scripts/rig_run.py is not the registered driver")
    if sha256_file(ROOT / "uv.lock") != reg["lock_sha256"]:
        out.append("uv.lock is not the registered lock file")
    for pool, want in reg["pools"].items():
        f = pool_file(pools_dir, pool)
        if not f.exists() or sha256_file(f) != want:
            out.append(f"pool {pool} is missing or is not the registered pool")
    return out


def seal(out: Path) -> dict:
    """One hash per run folder that exists, over every file in it.

    Written and committed before any run is analysed, so what was analysed can be told
    from what was made: the raw runs themselves are too large to commit.
    """
    hashes = {}
    for run in runs():
        folder = out / run["run"]
        if folder.is_dir():
            hashes[run["run"]] = tree_hash(folder, [f for f in folder.rglob("*") if f.is_file()])
    missing = sorted(r["run"] for r in runs() if r["run"] not in hashes)
    return {"runs": hashes, "planned_and_not_made": missing}


def go(
    plan_runs: list[dict],
    out: Path,
    reg: dict,
    make: Callable[[dict], int],
    limit: int | None = None,
    say: Callable[[str], None] = print,
) -> int:
    """Make the runs that do not exist. Returns the exit status: 0 only if none stopped it.

    `make(run)` makes one run and returns the driver's exit status. A run that fails
    its integrity check is a run: it is kept and the sequence goes on. A run the driver
    left no folder for, or whose manifest is not the registered one, stops everything:
    the setup is wrong, and going on would spend the study's runs on it.
    """
    made = 0
    for run in plan_runs:
        folder = out / run["run"]
        if folder.exists():
            continue
        if limit is not None and made >= limit:
            break
        code = make(run)
        made += 1
        say(f"{run['run']} exit={code}")
        manifest = folder / "manifest.json"
        if not manifest.exists():
            say(f"stopped: the driver made no manifest for {run['run']}")
            return 1
        differs = registered_problems(json.loads(manifest.read_text()), reg)
        if differs:
            say(f"stopped: {run['run']} is not a registered run: " + "; ".join(differs))
            return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--list", action="store_true", help="print the plan as JSON lines")
    what.add_argument("--check", action="store_true", help="say which runs would be refused")
    what.add_argument("--sample", action="store_true", help="make the run a registration reads")
    what.add_argument("--register", action="store_true", help="write the registration's record")
    what.add_argument("--go", action="store_true", help="make the runs that do not exist")
    what.add_argument("--seal", action="store_true", help="hash every run's files, once")
    p.add_argument("--pools", type=Path, default=POOLS_DIR)
    p.add_argument("--out", type=Path, default=RUNS_DIR)
    p.add_argument("--registered", type=Path, default=REGISTERED)
    p.add_argument("--limit", type=int, default=None, help="with --go: make at most this many")
    a = p.parse_args()

    def make(run: dict) -> int:
        proc = subprocess.Popen(argv(run, a.pools, a.out), stdout=subprocess.DEVNULL)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # The driver got the same signal and is taking the rig down. Let it finish.
            proc.wait()
            raise

    if a.list:
        for run in runs():
            print(json.dumps(run))
        return
    if a.check:
        bad = folder_problems(a.pools) | check(load_pools(a.pools))
        for name, why in bad.items():
            print(f"{name}: {'; '.join(why)}")
        sys.exit(1 if bad else 0)
    if a.sample:
        if (a.out / SAMPLE["run"]).exists():
            sys.exit(f"{a.out / SAMPLE['run']} exists; the sample is made once")
        load_pools(a.pools)
        sys.exit(make(SAMPLE))
    if a.register:
        if a.registered.exists():
            sys.exit(f"{a.registered} exists; a registration is written once")
        pools = load_pools(a.pools)
        bad = folder_problems(a.pools) | check(pools)
        if bad:
            sys.exit(f"{len(bad)} runs of the plan would be refused; see --check")
        hashes = {q: sha256_file(pool_file(a.pools, q)) for q in POOLS}
        folder = a.out / SAMPLE["run"]
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
            integrity = json.loads((folder / "events.json").read_text()).get("integrity")
        except (FileNotFoundError, json.JSONDecodeError):
            sys.exit(f"no finished sample run in {folder}; make it with --sample")
        wrong = sample_problems(manifest, integrity, hashes)
        if wrong:
            sys.exit("the sample cannot be registered: " + "; ".join(wrong))
        reg = registered(manifest, hashes, planned(pools))
        a.registered.write_text(json.dumps(reg, indent=1) + "\n")
        print(f"wrote {a.registered}: {len(reg['runs'])} runs, {len(hashes)} pools")
        return
    if a.seal:
        if SEAL.exists():
            sys.exit(f"{SEAL} exists; the runs are sealed once")
        sealed = seal(a.out)
        SEAL.write_text(json.dumps(sealed, indent=1) + "\n")
        print(f"wrote {SEAL}: {len(sealed['runs'])} of {len(runs())} runs exist")
        return
    stop = preflight(a.registered, a.pools)
    if stop:
        sys.exit("not started: " + "; ".join(stop))
    try:
        sys.exit(go(runs(), a.out, json.loads(a.registered.read_text()), make, a.limit))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
