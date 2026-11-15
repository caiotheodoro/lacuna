"""What a run used, by hash: the part of a run's manifest that needs no docker.

`scripts/rig_run.py` adds the image digests, the commit and the run's parameters.
A run whose manifest differs from the registered one is set aside by the analysis
(`registered_problems`); that is the last part of the registration's integrity gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hashes(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """sha256 of each file, keyed by its path relative to `root`, in path order."""
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}


def tree_hash(root: Path, paths: Iterable[Path]) -> str:
    """One sha256 over a set of files: their relative paths and contents, in path order.

    The run driver takes it over the source the rig image is built from, so a run made
    from an uncommitted tree still says exactly which source it ran.
    """
    lines = [f"{rel} {digest}" for rel, digest in config_hashes(root, paths).items()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# What a registration pins about every run, besides the hashes.
REGISTERED_PARAMS = ("env", "sessions", "stagger", "lead", "warmup_max", "max_lag", "settle")
# The hashes every run of a registration shares.
REGISTERED_HASHES = ("source_sha256", "driver_sha256", "lock_sha256")


def registered(manifest: dict, pools: dict[str, str], runs: dict[str, dict]) -> dict:
    """What a registration records: one sample run's manifest, the pools, and the plan.

    `pools` maps each pool's name to the sha256 of its sessions file. `runs` maps each
    planned run's name to its `condition`, `seed`, `pool` and the `sessions_sha256` of
    the sessions it will replay. The commit is kept for the record and is not compared:
    committing this very record moves it. The hashes are what a run is held to.
    """
    return {
        "commit": manifest["commit"],
        **{k: manifest[k] for k in REGISTERED_HASHES},
        "configs": dict(manifest["configs"]),
        "images": dict(manifest["images"]),
        "params": {k: manifest["params"][k] for k in REGISTERED_PARAMS},
        "pools": dict(pools),
        "runs": {name: dict(r) for name, r in runs.items()},
    }


def registered_problems(manifest: dict, reg: dict) -> list[str]:
    """How a run's manifest differs from the registration. Empty means it does not.

    Two questions: was the run made by what was registered (source, driver, lock file,
    configs, images, parameters, a clean tree), and is it the run of the plan its name
    says it is (condition, seeds, pool, the very sessions). A run that differs is kept
    and not analysed.
    """
    out: list[str] = []
    if manifest.get("dirty") is not False:
        out.append("the run was not made from a clean tree")
    names = {"source_sha256": "the source the image was built from",
             "driver_sha256": "the run driver", "lock_sha256": "the lock file"}  # fmt: skip
    for key in REGISTERED_HASHES:
        if manifest.get(key) != reg[key]:
            out.append(f"{names[key]} is not the registered one")
    for what in ("configs", "images"):
        mine, want = manifest.get(what) or {}, reg[what]
        for name in sorted(set(mine) | set(want)):
            if mine.get(name) != want.get(name):
                out.append(f"{what}: {name} is not the registered one")
    params = manifest.get("params") or {}
    for k, v in reg["params"].items():
        if params.get(k) != v:
            out.append(f"parameter {k} is {params.get(k)!r}, registered {v!r}")

    planned = reg["runs"].get(manifest.get("run"))
    if planned is None:
        out.append(f"{manifest.get('run')!r} is not a run of the plan")
        return out
    if params.get("condition") != planned["condition"]:
        out.append(f"condition is {params.get('condition')!r}, planned {planned['condition']!r}")
    for k in ("stress_seed", "draw_seed"):
        if params.get(k) != planned["seed"]:
            out.append(f"{k} is {params.get(k)!r}, planned {planned['seed']!r}")
    pools = [p.get("sha256") if isinstance(p, dict) else None for p in manifest.get("pools") or []]
    if pools != [reg["pools"][planned["pool"]]]:
        out.append(f"the run did not replay pool {planned['pool']!r} and nothing else")
    if manifest.get("sessions_sha256") != planned["sessions_sha256"]:
        out.append("the sessions replayed are not the ones the plan draws")
    return out
